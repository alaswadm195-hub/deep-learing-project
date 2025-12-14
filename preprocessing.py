"""preprocessing.py
Preprocessing utilities for Age & Gender project.
Corrected dunder methods (__init__, __len__, __getitem__) and updated augmentations.
"""
import os
import re
import json
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from facenet_pytorch import MTCNN
from tqdm import tqdm
from sklearn.model_selection import train_test_split
class ImageValidator:
    def __init__(self):
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.corrupted_images: List[Tuple[str, str]] = []
    def is_valid_image(self, image_path: str) -> bool:
        try:
            if Path(image_path).suffix.lower() not in self.valid_extensions:
                return False
            img = Image.open(image_path)
            img.verify()
            if getattr(img, "size", None) is None:
                return False
            return True
        except Exception as e:
            self.corrupted_images.append((image_path, str(e)))
            return False
    def clean_dataset(self, dataset_path: str, remove_corrupted: bool = True) -> dict:
        results = {'total': 0, 'valid': 0, 'corrupted': 0, 'removed': 0}
        for root, _dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = os.path.join(root, file)
                results['total'] += 1
                if self.is_valid_image(file_path):
                    results['valid'] += 1
                else:
                    results['corrupted'] += 1
                    if remove_corrupted:
                        try:
                            os.remove(file_path)
                            results['removed'] += 1
                        except Exception:
                            pass
        return results
class FaceDetector:
    def __init__(self, method: str = 'mtcnn', device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.method = method
        self.device = device
        if method == 'mtcnn':
            self.detector = MTCNN(keep_all=False, device=self.device, post_process=False)
        elif method == 'haar':
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
        else:
            raise ValueError(f"Unknown face detection method: {method}")
    def detect_and_crop_face(self, image, margin: int = 20, align: bool = False) -> Optional[torch.Tensor]:
        try:
            if torch.is_tensor(image):
                image_np = image.cpu().numpy().astype('uint8')
            else:
                image_np = image
            if image_np is None or getattr(image_np, 'ndim', None) != 3 or image_np.shape[2] != 3:
                return None
            if self.method == 'mtcnn':
                image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                boxes, probs = self.detector.detect(image_rgb)
                if boxes is None or len(boxes) == 0:
                    return None
                x1, y1, x2, y2 = boxes[0].astype(int)
                h, w = image_np.shape[:2]
                x1 = max(0, x1 - margin); y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin); y2 = min(h, y2 + margin)
                face_np = image_np[y1:y2, x1:x2]
                if face_np is None or getattr(face_np, "size", 0) == 0:
                    return None
                return torch.as_tensor(face_np, dtype=torch.uint8)
            else:  
                gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                if len(faces) == 0:
                    return None
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                x1 = max(0, x - margin); y1 = max(0, y - margin)
                x2 = min(image_np.shape[1], x + w + margin); y2 = min(image_np.shape[0], y + h + margin)
                face_np = image_np[y1:y2, x1:x2]
                if face_np is None or getattr(face_np, "size", 0) == 0:
                    return None
                return torch.as_tensor(face_np, dtype=torch.uint8)
        except Exception:
            return None
class ImagePreprocessor:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    def preprocess(self, image, normalize: bool = True) -> torch.Tensor:
        if torch.is_tensor(image):
            image_np = image.cpu().numpy().astype('uint8')
        else:
            image_np = image
        if image_np is None or getattr(image_np, 'ndim', None) != 3 or image_np.shape[2] != 3:
            c, h, w = 3, self.target_size[1], self.target_size[0]
            return torch.zeros((c, h, w), dtype=torch.float32)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, self.target_size, interpolation=cv2.INTER_AREA)
        arr = resized.astype('float32') / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()
        if normalize:
            mean = self.mean.view(3, 1, 1)
            std = self.std.view(3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor
class LabelExtractor:
    def __init__(self):
        self.gender_mapping = {'male': 0, 'm': 0, 'female': 1, 'f': 1}
    def extract_from_filename(self, filename: str) -> Tuple[Optional[int], Optional[int]]:
        try:
            name = Path(filename).stem
            if name.endswith('.chip'):
                name = name.rsplit('.', 1)[0]
            parts = re.split(r'[_\-]+', name)
            age = None
            gender = None
            if len(parts) >= 2:
                try:
                    age_candidate = int(parts[0])
                    if 0 <= age_candidate <= 120:
                        age = age_candidate
                    gender_candidate = int(parts[1])
                    if gender_candidate in [0, 1]:
                        gender = gender_candidate
                except Exception:
                    pass
            if age is None or gender is None:
                for part in parts:
                    if part.isdigit():
                        potential_age = int(part)
                        if 0 <= potential_age <= 120 and age is None:
                            age = potential_age
                    pl = part.lower()
                    if pl in self.gender_mapping and gender is None:
                        gender = self.gender_mapping[pl]
            return age, gender
        except Exception:
            return None, None
class AgeGenderDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        ages: List[int],
        genders: List[int],
        transform=None,
        face_detector: Optional[FaceDetector] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
    ):
        self.image_paths = image_paths
        self.ages = torch.tensor(ages, dtype=torch.float32)
        self.genders = torch.tensor(genders, dtype=torch.long)
        self.transform = transform
        self.face_detector = face_detector
        self.preprocessor = preprocessor or ImagePreprocessor()
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_np = cv2.imread(str(image_path))
        if image_np is None:
            image_tensor = torch.zeros((3, self.preprocessor.target_size[1], self.preprocessor.target_size[0]), dtype=torch.float32)
        else:
            if self.face_detector is not None:
                face_t = self.face_detector.detect_and_crop_face(image_np)
                if face_t is not None:
                    proc_np = face_t.cpu().numpy().astype('uint8')
                else:
                    proc_np = image_np
            else:
                proc_np = image_np
            if self.transform:
                proc_np = cv2.cvtColor(proc_np, cv2.COLOR_BGR2RGB)
                augmented = self.transform(image=proc_np)
                image_tensor = augmented['image']
            else:
                image_tensor = self.preprocessor.preprocess(proc_np, normalize=True)
        age = self.ages[idx]
        gender = self.genders[idx]
        return {'image': image_tensor, 'age': age, 'gender': gender}
def get_train_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
def get_val_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
def create_dataloaders(
    train_paths, train_ages, train_genders,
    val_paths, val_ages, val_genders,
    test_paths, test_ages, test_genders,
    batch_size=32, num_workers=4, use_face_detection=False
):
    face_detector = FaceDetector() if use_face_detection else None
    train_dataset = AgeGenderDataset(train_paths, train_ages, train_genders, transform=get_train_transforms(), face_detector=face_detector)
    val_dataset = AgeGenderDataset(val_paths, val_ages, val_genders, transform=get_val_transforms(), face_detector=face_detector)
    test_dataset = AgeGenderDataset(test_paths, test_ages, test_genders, transform=get_val_transforms(), face_detector=face_detector)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
class DatasetPreparer:
    def __init__(self, dataset_path: str, output_path: str = './processed_data'):
        self.dataset_path = str(Path(dataset_path).resolve())
        self.output_path = str(Path(output_path).resolve())
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.validator = ImageValidator()
        self.face_detector = FaceDetector(method='mtcnn')
        self.preprocessor = ImagePreprocessor()
        self.label_extractor = LabelExtractor()
        self.metadata: List[dict] = []
    def prepare(self, clean_corrupted=True, detect_faces=True, save_processed_images=True):
        print("🚀 بدء تحضير الداتاست...")
        print("\n📋 الخطوة 1: التحقق من سلامة الصور...")
        results = self.validator.clean_dataset(self.dataset_path, remove_corrupted=clean_corrupted)
        print(f"✅ إجمالي الصور: {results['total']}")
        print(f"✅ صور سليمة: {results['valid']}")
        print("\n🔄 الخطوة 2: معالجة الصور...")
        self._process_images(detect_faces, save_processed_images)
        print("\n💾 الخطوة 3: حفظ البيانات...")
        self._save_metadata()
        print("\n✂ الخطوة 4: تقسيم البيانات...")
        self._split_data()
        print("\n✨ تم الانتهاء من تحضير الداتاست!")
        return self.metadata
    def _process_images(self, detect_faces: bool, save_processed: bool):
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(Path(self.dataset_path).rglob(f'*{ext}'))
        processed_dir = Path(self.output_path) / 'processed_images'
        if save_processed:
            processed_dir.mkdir(exist_ok=True)
        for img_path in tqdm(image_files, desc="معالجة الصور"):
            try:
                image = cv2.imread(str(img_path))
                if image is None: continue
                if detect_faces:
                    face_t = self.face_detector.detect_and_crop_face(image)
                    if face_t is not None:
                        proc = face_t.cpu().numpy().astype('uint8')
                    else:
                        continue
                else:
                    proc = image
                age, gender = self.label_extractor.extract_from_filename(img_path.name)
                if age is None or gender is None: continue
                processed_path = str(img_path)
                if save_processed:
                    processed_filename = f"{age}_{gender}_{img_path.stem}.jpg"
                    processed_path = str(processed_dir / processed_filename)
                    processed_img = cv2.resize(proc, (224, 224), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(processed_path, processed_img)
                self.metadata.append({
                    'image_path': processed_path,
                    'original_path': str(img_path),
                    'age': age,
                    'gender': gender,
                    'face_detected': bool(detect_faces)
                })
            except Exception:
                continue
    def _save_metadata(self):
        df = pd.DataFrame(self.metadata)
        csv_path = Path(self.output_path) / 'metadata.csv'
        df.to_csv(csv_path, index=False)
    def _split_data(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        df = pd.DataFrame(self.metadata)
        if len(df) == 0: return
        try:
            train_df, temp_df = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=42, stratify=df['gender'])
            val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (val_ratio + test_ratio), random_state=42, stratify=temp_df['gender'])
        except Exception:
            train_df, temp_df = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)
        train_df['split'] = 'train'
        val_df['split'] = 'val'
        test_df['split'] = 'test'
        final = pd.concat([train_df, val_df, test_df], ignore_index=True)
        self.metadata = final.to_dict(orient='records')
        csv_path = Path(self.output_path) / 'metadata.csv'
        pd.DataFrame(self.metadata).to_csv(csv_path, index=False)