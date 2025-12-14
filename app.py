import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from model import AgeGenderModel
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/best_model.pth"
model = None
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
@app.on_event("startup")
async def load_model():
    global model
    model = AgeGenderModel(pretrained=False)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    print("✅ Model Loaded!")
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    print("\n" + "="*20)
    print("📸 استقبلت صورة من الموقع...")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    input_tensor = val_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred_age, pred_gender = model(input_tensor)
        age_val = round(pred_age.item(), 1)
        probs = torch.softmax(pred_gender, dim=1)
        gender_idx = torch.argmax(probs, dim=1).item()
        gender_prob = probs[0][gender_idx].item()
        gender_label = "Male" if gender_idx == 0 else "Female"
    print(f"🧠 حكم الموديل: {gender_label} - العمر: {age_val}")
    print("="*20 + "\n")
    return {
        "age": age_val,
        "gender": gender_label,
        "confidence": round(gender_prob * 100, 2)
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)