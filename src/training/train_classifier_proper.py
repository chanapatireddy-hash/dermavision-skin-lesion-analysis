"""
Proper training script for skin lesion classifier.
Trains on high-quality synthetic dermoscopy data (3 classes).
"""
import os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CLASSES     = ["Common Nevus", "Atypical Nevus", "Melanoma"]
NUM_CLASSES = 3
IMG_SIZE    = 224
BATCH_SIZE  = 16
EPOCHS      = 25
LR          = 1e-4
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─── Dataset ─────────────────────────────────────────────────────────────────
class SkinDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.paths  = df['filepath'].tolist()
        self.labels = df['label'].tolist()
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert('RGB')
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── Model ───────────────────────────────────────────────────────────────────
def build_model():
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # Freeze all but last 3 layers for speed
    for i, (name, param) in enumerate(model.features.named_parameters()):
        if i < 130:
            param.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    return model

def train():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[TRAIN] Device: {device}")

    DATA_ROOT = os.path.join(PROJECT_ROOT, 'datasets', 'synthetic')
    train_csv = os.path.join(DATA_ROOT, 'train', 'labels.csv')
    val_csv   = os.path.join(DATA_ROOT, 'val',   'labels.csv')

    # Generate data if missing
    if not os.path.exists(train_csv):
        print("[TRAIN] Generating synthetic dataset first...")
        from src.data.generate_synthetic import generate_split, N_PER_CLASS_TRAIN, N_PER_CLASS_VAL
        generate_split(os.path.join(DATA_ROOT, 'train'), N_PER_CLASS_TRAIN, 0)
        generate_split(os.path.join(DATA_ROOT, 'val'),   N_PER_CLASS_VAL, 999999)

    train_ds = SkinDataset(train_csv, train_transform)
    val_ds   = SkinDataset(val_csv,   val_transform)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"[TRAIN] Train: {len(train_ds)}  Val: {len(val_ds)}")

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    start_time   = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
        scheduler.step()
        train_acc  = correct / total
        train_loss = train_loss / total

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(device), labels.to(device)
                out  = model(imgs)
                loss = criterion(out, labels)
                val_loss  += loss.item() * imgs.size(0)
                _, pred    = torch.max(out, 1)
                val_correct += (pred == labels).sum().item()
                val_total   += labels.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_acc  = val_correct / val_total
        val_loss = val_loss / val_total

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            ckpt = os.path.join(CHECKPOINT_DIR, 'best_classifier.pth')
            torch.save(model.state_dict(), ckpt)
            print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")

    elapsed = time.time() - start_time
    print(f"\n[TRAIN] Finished in {elapsed:.1f}s. Best Val Acc = {best_val_acc:.3f}")

    # Final report on validation set
    print("\n[EVAL] Per-class metrics:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))
    print("[EVAL] Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    # First generate data
    DATA_ROOT = os.path.join(PROJECT_ROOT, 'datasets', 'synthetic')
    train_csv = os.path.join(DATA_ROOT, 'train', 'labels.csv')
    if not os.path.exists(train_csv):
        print("Generating synthetic dataset...")
        sys.path.insert(0, PROJECT_ROOT)
        from src.data.generate_synthetic import generate_split, N_PER_CLASS_TRAIN, N_PER_CLASS_VAL
        generate_split(os.path.join(DATA_ROOT, 'train'), N_PER_CLASS_TRAIN, 0)
        generate_split(os.path.join(DATA_ROOT, 'val'),   N_PER_CLASS_VAL,   999999)

    train()
