# test.py — تقييم الموديل النهائي
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm  # <--- إضافة شريط التحميل

from preprocessing import AgeGenderDataset, get_val_transforms
from model import AgeGenderModel

# ===== إعداد المسارات =====
META_PATH = Path("processed_data/metadata.csv")
CHECKPOINT = Path("checkpoints/best_model.pth")
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Running on: {DEVICE}")

# ===== تحميل metadata =====
if not META_PATH.exists():
    raise FileNotFoundError("ملف metadata.csv مش موجود! تأكد إنك شغلت preprocessing الأول.")

df = pd.read_csv(META_PATH)
test_df = df[df["split"] == "test"].reset_index(drop=True)

print(f"📊 Testing on {len(test_df)} images...")

test_paths = test_df["image_path"].tolist()
test_ages = test_df["age"].tolist()
test_genders = test_df["gender"].tolist()

# ===== DataLoader =====
transform = get_val_transforms()
dataset = AgeGenderDataset(
    test_paths, test_ages, test_genders,
    transform=transform,
    face_detector=None # الصور مقصوصة جاهزة
)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== تحميل الموديل =====
# تأكد إن إعدادات الموديل هنا نفس اللي دربت بيها (نفس الـ dropout والـ hidden_dim)
model = AgeGenderModel(pretrained=False) 
try:
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading checkpoint: {e}")
    exit()

model.to(DEVICE)
model.eval()

# ===== Lists =====
pred_age_list = []
pred_gender_list = []
true_age_list = []
true_gender_list = []
paths_list = []

# ===== Testing Loop =====
print("⏳ Starting Inference...")
with torch.no_grad():
    # هنا ضفنا tqdm عشان الشكل الجمالي
    for batch_idx, batch in enumerate(tqdm(loader, desc="Testing")):
        images = batch["image"].to(DEVICE)
        ages = batch["age"].to(DEVICE)
        genders = batch["gender"].to(DEVICE)

        pred_age, pred_gender = model(images)

        # العمر: تحويل الناتج لرقم
        pred_age = pred_age.squeeze(1)

        # الجنس: اختيار أعلى logit
        pred_gender_class = torch.argmax(pred_gender, dim=1)

        pred_age_list.extend(pred_age.cpu().tolist())
        pred_gender_list.extend(pred_gender_class.cpu().tolist())
        true_age_list.extend(ages.cpu().tolist())
        true_gender_list.extend(genders.cpu().tolist())

        # حفظ المسارات (عشان نعرف كل صورة نتيجتها إيه)
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + len(images)
        paths_list.extend(test_paths[start_idx:end_idx])

# ===== حساب MAE =====
mae_tensor = torch.tensor(
    [abs(a - b) for a, b in zip(pred_age_list, true_age_list)]
)
mae = mae_tensor.mean().item()

# ===== تقارير الجنس =====
gender_acc = accuracy_score(true_gender_list, pred_gender_list)
cm = confusion_matrix(true_gender_list, pred_gender_list)
report = classification_report(true_gender_list, pred_gender_list, target_names=["Male", "Female"])

# عرض النتائج بشكل منسق
print("\n" + "="*30)
print(f"🔥 FINAL RESULTS")
print("="*30)
print(f"📉 Test MAE (Age): {mae:.4f} years")
print(f"🎯 Test Gender Accuracy: {gender_acc:.2%}")
print("-" * 30)
print("Confusion Matrix:")
print(cm)
print("-" * 30)
print("Classification Report:")
print(report)
print("="*30)

# ===== حفظ النتائج =====
out_df = pd.DataFrame({
    "image_path": paths_list,
    "age_true": true_age_list,
    "age_pred": [round(x, 1) for x in pred_age_list], # تقريب العمر لرقم عشري واحد
    "gender_true": true_gender_list,
    "gender_pred": pred_gender_list
})
Path("results").mkdir(exist_ok=True)
out_df.to_csv("results/test_predictions.csv", index=False)
print("\n✔ Saved detailed results → results/test_predictions.csv")