import torch
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix
from model import AgeGenderModel
from preprocessing import AgeGenderDataset, get_val_transforms
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
df = pd.read_csv("processed_data/metadata.csv")
test_df = df[df["split"] == "test"].reset_index(drop=True)
dataset = AgeGenderDataset(
    image_paths=test_df["image_path"].tolist(),
    ages=test_df["age"].tolist(),
    genders=test_df["gender"].tolist(),
    transform=get_val_transforms()
)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
model = AgeGenderModel(pretrained=False).to(DEVICE)
state = torch.load("checkpoints/best_model.pth", map_location=DEVICE)
model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
model.eval()
pred_ages, pred_genders, true_ages, true_genders = [], [], [], []
print(f"🚀 Start Testing on {len(test_df)} images (Please Wait)...")
with torch.no_grad():
    for batch in loader:
        imgs = batch["image"].to(DEVICE)
        out_age, out_gender = model(imgs)
        pred_ages.extend(out_age.squeeze(1).cpu().tolist())
        pred_genders.extend(out_gender.argmax(1).cpu().tolist())
        true_ages.extend(batch["age"].tolist())
        true_genders.extend(batch["gender"].tolist())
mae = mean_absolute_error(true_ages, pred_ages)
acc = accuracy_score(true_genders, pred_genders)
print("\n" + "="*30)
print(f"🔥 RESULTS:\n📉 MAE (Age Error): {mae:.2f} years\n🎯 Accuracy (Gender): {acc:.2%}")
print("Confusion Matrix:\n", confusion_matrix(true_genders, pred_genders))
print("="*30)
pd.DataFrame({
    "path": test_df["image_path"],
    "real_age": true_ages, "pred_age": [round(x, 1) for x in pred_ages],
    "real_gender": true_genders, "pred_gender": pred_genders
}).to_csv("results/test_predictions.csv", index=False)
print("✔ Results saved to: results/test_predictions.csv")