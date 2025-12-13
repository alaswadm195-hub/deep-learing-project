# app.py
import io
import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# استدعاء كلاسات مشروعنا
from model import AgeGenderModel
from preprocessing import ImagePreprocessor, FaceDetector

# ===========================
# إعدادات السيرفر والموديل
# ===========================
app = FastAPI(title="Age & Gender AI API")

# السماح للموقع (الفرونت إند) إنه يكلم السيرفر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# تأكد من مسار الموديل بتاعك هنا 👇
MODEL_PATH = Path("checkpoints/best_model.pth")

# متغيرات عامة هنحملها مرة واحدة
model = None
preprocessor = None
face_detector = None

# ===========================
# دوال مساعدة
# ===========================
def ensure_tensor_bchw(t):
    """دالة التأكد من أبعاد التنسور (زي اللي في predict_one.py)"""
    if not torch.is_tensor(t): t = torch.from_numpy(t)
    if t.dim() == 3:
        if t.shape[2] == 3: t = t.permute(2, 0, 1)
        elif t.shape[0] == 3: pass
        else: t = t.permute(2, 0, 1) if t.shape[-1] == 3 else t
    elif t.dim() == 4:
        if t.shape[1] == 3: pass
        elif t.shape[-1] == 3: t = t.permute(0, 3, 1, 2)
    if t.dim() == 3: t = t.unsqueeze(0)
    t = t.float()
    return t

@app.on_event("startup")
async def load_ai_models():
    """هنا بنحمل الموديل مرة واحدة أول ما السيرفر يشتغل عشان السرعة"""
    global model, preprocessor, face_detector
    print("⏳ جاري تحميل الموديل والأدوات...")
    
    # 1. تحميل الموديل
    model = AgeGenderModel(pretrained=False)
    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        print("✅ تم تحميل أوزان الموديل بنجاح.")
    except Exception as e:
        print(f"❌ خطأ في تحميل الموديل: {e}")
        print("تأكد من مسار ملف best_model.pth")

    # 2. تحميل المعالجات
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    # استخدم 'mtcnn' للدقة أو None للسرعة لو صورك مقصوصة جاهزة
    face_detector = FaceDetector(method='mtcnn', device=str(DEVICE)) 
    print("🚀 السيرفر جاهز لاستقبال الصور!")

# ===========================
# نقطة الاتصال (API Endpoint)
# ===========================
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """بيستقبل الصورة من الموقع ويرجع النتيجة JSON"""
    
    # 1. قراءة ملف الصورة من الطلب
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        return {"error": "الملف المرفق ليس صورة صالحة"}

    # 2. كشف الوجه (اختياري)
    face_t = face_detector.detect_and_crop_face(img_bgr)
    warning = None
    if face_t is None:
        warning = "لم يتم اكتشاف وجه، سيتم تحليل الصورة كاملة."
        # تحويل الصورة كاملة لتنسور
        face_input = torch.from_numpy(img_bgr)
    else:
        face_input = face_t # ده تنسور جاهز من الكاشف

    # 3. المعالجة الأولية
    processed = preprocessor.preprocess(face_input)
    input_tensor = ensure_tensor_bchw(processed).to(DEVICE)

    # 4. التوقع (Inference)
    with torch.no_grad():
        pred_age, pred_gender = model(input_tensor)
        
        # استخراج القيم
        age_val = round(pred_age.item(), 1)
        
        probs = torch.softmax(pred_gender, dim=1)
        gender_idx = torch.argmax(probs, dim=1).item()
        gender_prob = probs[0][gender_idx].item()
        
        gender_label = "Male" if gender_idx == 0 else "Female"

    # 5. إرسال الرد
    return {
        "age": age_val,
        "gender": gender_label,
        "confidence": round(gender_prob * 100, 2),
        "warning": warning
    }