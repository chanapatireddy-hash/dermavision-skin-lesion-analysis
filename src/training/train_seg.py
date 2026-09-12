import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset import get_dataloaders
from src.models.unet import UNet
from src.models.self_attention_unet import SelfAttentionUNet
from src.evaluation.metrics import DiceLoss, dice_coeff, iou_coeff
from tqdm import tqdm
import time

def train_segmentation(model_type='unet', data_dir='data/PH2Dataset', epochs=3, batch_size=4, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == 'unet':
        model = UNet(n_channels=3, n_classes=1).to(device)
    else:
        model = SelfAttentionUNet(n_channels=3, n_classes=1).to(device)
        
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size, img_size=256)
    
    criterion_bce = nn.BCELoss()
    criterion_dice = DiceLoss()
    
    # Combined Loss
    def combined_loss(pred, target):
        return criterion_bce(pred, target) + criterion_dice(pred, target)
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_dice = 0.0
    
    print(f"Starting training {model_type} for {epochs} epochs on {device}...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        # In a sanity check with missing data, train_loader might be empty or dummy
        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train"):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            preds = model(images)
            loss = combined_loss(preds, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset) if len(train_loader.dataset) > 0 else 1
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                preds = model(images)
                loss = combined_loss(preds, masks)
                
                val_loss += loss.item() * images.size(0)
                
                # Binarize predictions for metrics
                preds_bin = (preds > 0.5).float()
                val_dice += dice_coeff(preds_bin, masks).item() * images.size(0)
                val_iou += iou_coeff(preds_bin, masks).item() * images.size(0)
                
        val_loss /= len(val_loader.dataset) if len(val_loader.dataset) > 0 else 1
        val_dice /= len(val_loader.dataset) if len(val_loader.dataset) > 0 else 1
        val_iou /= len(val_loader.dataset) if len(val_loader.dataset) > 0 else 1
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")
        
        # Save best model
        os.makedirs('checkpoints', exist_ok=True)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), f"checkpoints/best_{model_type}.pth")
            print("Saved new best model!")
            
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s. Best Val Dice: {best_val_dice:.4f}")

if __name__ == '__main__':
    # Sanity check run (3 epochs) for both models
    print("--- Training Baseline U-Net ---")
    train_segmentation('unet', epochs=3)
    
    print("\n--- Training Self-Attention U-Net ---")
    train_segmentation('sa_unet', epochs=3)
