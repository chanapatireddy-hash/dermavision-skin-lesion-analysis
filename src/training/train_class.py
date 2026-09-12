import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset import get_dataloaders
from src.models.classifier import LesionClassifier
from tqdm import tqdm
import time

def train_classification(data_dir='data/PH2Dataset', epochs=3, batch_size=4, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LesionClassifier(num_classes=3).to(device)
    
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size, img_size=224)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    print(f"Starting classification training for {epochs} epochs on {device}...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} Train"):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            # Apply mask to image (Crop/Mask strategy)
            # Expand mask to 3 channels to multiply with image
            masks_expanded = masks.repeat(1, 3, 1, 1)
            masked_images = images * masks_expanded
            
            optimizer.zero_grad()
            outputs = model(masked_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_loss /= len(train_loader.dataset) if len(train_loader.dataset) > 0 else 1
        train_acc = correct / total if total > 0 else 0
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                
                masks_expanded = masks.repeat(1, 3, 1, 1)
                masked_images = images * masks_expanded
                
                outputs = model(masked_images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss /= len(val_loader.dataset) if len(val_loader.dataset) > 0 else 1
        val_acc = correct / total if total > 0 else 0
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        os.makedirs('checkpoints', exist_ok=True)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"checkpoints/best_classifier.pth")
            
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s. Best Val Acc: {best_val_acc:.4f}")

if __name__ == '__main__':
    train_classification(epochs=3)
