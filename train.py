# src/train.py
import os
import time
import json
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pandas as pd
from tqdm import tqdm

# استدعاء الدوال من preprocessing.py و model.py
from preprocessing import create_dataloaders, DatasetPreparer, get_train_transforms, get_val_transforms
from model import AgeGenderModel

# -------------------------
# Settings
# -------------------------
DATA_META = Path("processed_data/metadata.csv")
OUTPUT_DIR = Path("checkpoints")
BATCH_SIZE = 8            # مناسب للـ CPU، لو GPU قوي خليه 32
NUM_WORKERS = 0           # Windows fix
LR = 1e-4                 # Learning Rate
WEIGHT_DECAY = 1e-4
EPOCHS = 25               # عدد الـ Epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_SUBSET = False        
SUBSET_TRAIN = 500
SUBSET_VAL = 200
SUBSET_TEST = 200
SEED = 42
LAMBDA_AGE = 1.0
LAMBDA_GENDER = 1.0       
PIN_MEMORY = False

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_metadata(csv_path: Path):
    df = pd.read_csv(csv_path)
    required_cols = {'image_path', 'age', 'gender', 'split'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"metadata.csv missing columns: {required_cols - set(df.columns)}")
    
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    return train_df, val_df, test_df

def build_dataloaders_from_df(train_df, val_df, test_df):
    def df_to_lists(df):
        return df['image_path'].tolist(), df['age'].astype(float).tolist(), df['gender'].astype(int).tolist()

    train_paths, train_ages, train_genders = df_to_lists(train_df)
    val_paths, val_ages, val_genders = df_to_lists(val_df)
    test_paths, test_ages, test_genders = df_to_lists(test_df)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_paths, train_ages, train_genders,
        val_paths, val_ages, val_genders,
        test_paths, test_ages, test_genders,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, use_face_detection=False
    )
    return train_loader, val_loader, test_loader

# -------------------------
# Train Step
# -------------------------
def train_one_epoch(model, loader, optimizer, scaler, device, use_amp: bool):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    age_loss_fn = nn.L1Loss(reduction='sum')
    gender_loss_fn = nn.CrossEntropyLoss(reduction='sum')
    
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        imgs = batch['image'].to(device)
        ages = batch['age'].to(device).float()
        genders = batch['gender'].to(device).long()

        optimizer.zero_grad()
        
        if use_amp:
            from torch.cuda.amp import autocast
            with autocast():
                pred_age, pred_gender = model(imgs)
                pred_age = pred_age.squeeze(1)
                
                la = age_loss_fn(pred_age, ages)
                lg = gender_loss_fn(pred_gender, genders)
                loss = LAMBDA_AGE * la + LAMBDA_GENDER * lg
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_age, pred_gender = model(imgs)
            pred_age = pred_age.squeeze(1)
            
            la = age_loss_fn(pred_age, ages)
            lg = gender_loss_fn(pred_gender, genders)
            loss = LAMBDA_AGE * la + LAMBDA_GENDER * lg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bsz = imgs.size(0)
        total_loss += loss.item()
        total_samples += bsz
        pbar.set_postfix({'loss': f"{total_loss / max(1, total_samples):.4f}"})
        
    avg_loss = total_loss / total_samples
    return avg_loss

# -------------------------
# Validation Step
# -------------------------
def validate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    age_abs_sum = 0.0
    gender_correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            imgs = batch['image'].to(device)
            ages = batch['age'].to(device).float()
            genders = batch['gender'].to(device).long()

            pred_age, pred_gender = model(imgs)
            pred_age = pred_age.squeeze(1)
            
            age_abs_sum += torch.sum(torch.abs(pred_age - ages)).item()
            preds = pred_gender.argmax(dim=1)
            gender_correct += (preds == genders).sum().item()
            total += imgs.size(0)

    mae = age_abs_sum / total if total > 0 else float('nan')
    acc = gender_correct / total if total > 0 else float('nan')
    return mae, acc

# -------------------------
# Main
# -------------------------
def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_META.exists():
        raise FileNotFoundError(f"metadata file not found at {DATA_META}. Run preprocessing first.")

    train_df, val_df, test_df = load_metadata(DATA_META)

    if USE_SUBSET:
        print(f"⚠ WARN: Using subset of data (Train={SUBSET_TRAIN})")
        train_df = train_df.sample(n=min(SUBSET_TRAIN, len(train_df)), random_state=SEED)
        val_df = val_df.sample(n=min(SUBSET_VAL, len(val_df)), random_state=SEED)
        test_df = test_df.sample(n=min(SUBSET_TEST, len(test_df)), random_state=SEED)

    train_loader, val_loader, test_loader = build_dataloaders_from_df(train_df, val_df, test_df)

    print(f"Dataset: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 1. Initialize Model
    model = AgeGenderModel(pretrained=True, freeze_backbone=False).to(DEVICE)

    # 2. Setup AMP
    use_amp = torch.cuda.is_available()
    scaler = None
    if use_amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("🚀 Mixed Precision (AMP) Enabled")
    else:
        class DummyScaler:
            def scale(self, x): return x
            def unscale_(self, opt): return None
            def step(self, opt): return None
            def update(self): return None
        scaler = DummyScaler()
        print("⚠ Running on CPU or without AMP")

    # 3. Setup Optimizer & Scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_mae = float('inf')
    start_epoch = 1

    # ==========================================
    # 🔄 Resume Logic (Start)
    # ==========================================
    checkpoint_path = OUTPUT_DIR / "last_checkpoint.pth"
    if checkpoint_path.exists():
        print(f"🔄 Found checkpoint at {checkpoint_path}! Resuming training...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            
            # Load states
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore best metric if available
            if 'val_mae' in checkpoint:
                best_val_mae = checkpoint['val_mae'] # نستخدم آخر قيمة محفوظة كبداية للمقارنة
            
            # Set start epoch
            start_epoch = checkpoint['epoch'] + 1
            print(f"⏩ Resuming from Epoch {start_epoch}")
            
        except Exception as e:
            print(f"⚠ Failed to resume from checkpoint: {e}. Starting from scratch.")
            start_epoch = 1
    # ==========================================
    # 🔄 Resume Logic (End)
    # ==========================================

    print(f"🔥 Starting Training Loop from Epoch {start_epoch} to {EPOCHS}...")
    
    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()
        
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, DEVICE, use_amp)
        
        # Validate
        val_mae, val_acc = validate(model, val_loader, DEVICE)
        
        # Scheduler Step
        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_mae)
        after_lr = optimizer.param_groups[0]["lr"]
        
        t1 = time.time()

        # Print Stats
        lr_status = f" | LR: {after_lr:.1e}" if after_lr != before_lr else ""
        print(f"Epoch {epoch:02d}/{EPOCHS} [{t1-t0:.0f}s] "
              f"Loss: {train_loss:.4f} | "
              f"Val MAE: {val_mae:.4f} | "
              f"Val Acc: {val_acc:.2%} {lr_status}")

        # Save Best Model
        if val_mae < best_val_mae:
            print(f"  ⭐ New Best MAE! ({best_val_mae:.4f} -> {val_mae:.4f}). Saving model...")
            best_val_mae = val_mae
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pth")
            
            with open(OUTPUT_DIR / "best_metrics.json", "w") as f:
                json.dump({'epoch': epoch, 'val_mae': val_mae, 'val_gender_acc': val_acc}, f)

        # Save Last Checkpoint (For Resuming)
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_mae': best_val_mae # نحفظ أفضل قيمة وصلنا لها
        }
        torch.save(ckpt, OUTPUT_DIR / "last_checkpoint.pth")

    print("\n🏁 Training Finished!")
    
    # Final Test
    if (OUTPUT_DIR / "best_model.pth").exists():
        print("Running Final Test Set Evaluation with Best Model...")
        best_state = torch.load(OUTPUT_DIR / "best_model.pth", map_location=DEVICE)
        model.load_state_dict(best_state)
        test_mae, test_acc = validate(model, test_loader, DEVICE)
        print(f"🏆 Final Test Results -> MAE: {test_mae:.4f} years | Gender Acc: {test_acc:.2%}")
    else:
        print("⚠ No best_model.pth found to test.")

if __name__ == "__main__":
    main()