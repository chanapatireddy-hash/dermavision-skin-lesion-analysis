import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pandas as pd

class PH2Dataset(Dataset):
    def __init__(self, root_dir, split_type='train', transform=None, seed=42):
        """
        Args:
            root_dir (string): Directory with all the images (e.g., path to 'PH2Dataset/PH2 Dataset images').
            split_type (string): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            seed (int): Random seed for reproducibility.
        """
        self.root_dir = root_dir
        self.split_type = split_type
        self.transform = transform
        self.seed = seed

        # PH2 dataset structure:
        # root_dir/
        #   IMD002/
        #       IMD002_Dermoscopic_Image/IMD002.bmp
        #       IMD002_lesion/IMD002_lesion.bmp
        
        # In a real scenario, we might also read the Excel file to get classes:
        # 0: Common Nevus, 1: Atypical Nevus, 2: Melanoma
        # For simplicity in this demo without the excel file, we simulate labels if excel is absent,
        # or load them if present. We will assume a simple metadata CSV or just return a dummy label
        # if the goal is primarily segmentation, and classification is secondary.
        # Let's list all image folders
        if not os.path.exists(self.root_dir):
            print(f"Warning: Dataset directory {self.root_dir} not found.")
            self.image_dirs = []
        else:
            self.image_dirs = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        
        # Split logic (70% train, 15% val, 15% test)
        random.seed(self.seed)
        shuffled_dirs = self.image_dirs.copy()
        random.shuffle(shuffled_dirs)
        
        n_total = len(shuffled_dirs)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        if self.split_type == 'train':
            self.image_dirs = shuffled_dirs[:n_train]
        elif self.split_type == 'val':
            self.image_dirs = shuffled_dirs[n_train:n_train+n_val]
        elif self.split_type == 'test':
            self.image_dirs = shuffled_dirs[n_train+n_val:]
        else:
            raise ValueError("split_type must be 'train', 'val', or 'test'")

    def __len__(self):
        if len(self.image_dirs) == 0:
            return 50 # 50 synthetic images per epoch
        return len(self.image_dirs)

    def __getitem__(self, idx):
        if len(self.image_dirs) == 0:
            # Generate synthetic skin lesion for demo training
            # Skin background (peach/pink)
            img_np = np.ones((256, 256, 3), dtype=np.uint8) * np.array([235, 190, 175], dtype=np.uint8)
            # Add some noise to skin
            noise = np.random.normal(0, 10, (256, 256, 3)).astype(np.int16)
            img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            
            # Lesion parameters
            center_x = np.random.randint(80, 176)
            center_y = np.random.randint(80, 176)
            radius_x = np.random.randint(30, 70)
            radius_y = np.random.randint(30, 70)
            
            Y, X = np.ogrid[:256, :256]
            dist_from_center = ((X - center_x) / radius_x)**2 + ((Y - center_y) / radius_y)**2
            
            mask_np = (dist_from_center <= 1).astype(np.uint8) * 255
            
            # Add lesion color (dark brown)
            lesion_color = np.array([80, 50, 40], dtype=np.uint8)
            img_np[dist_from_center <= 1] = lesion_color
            # Add some lesion noise
            lesion_noise = np.random.normal(0, 15, (256, 256, 3)).astype(np.int16)
            img_np_noisy_lesion = np.clip(img_np + lesion_noise, 0, 255).astype(np.uint8)
            img_np = np.where(dist_from_center[..., np.newaxis] <= 1, img_np_noisy_lesion, img_np)
            
            image = Image.fromarray(img_np)
            mask = Image.fromarray(mask_np).convert('L')
            
            label = np.random.randint(0, 3)
            
            if self.transform:
                seed = np.random.randint(2147483647)
                random.seed(seed)
                torch.manual_seed(seed)
                image = self.transform(image)
                random.seed(seed)
                torch.manual_seed(seed)
                mask = self.transform(mask)
            
            mask = (mask > 0).float()
            return image, mask, label

        img_name = self.image_dirs[idx]
        
        img_path = os.path.join(self.root_dir, img_name, f"{img_name}_Dermoscopic_Image", f"{img_name}.bmp")
        mask_path = os.path.join(self.root_dir, img_name, f"{img_name}_lesion", f"{img_name}_lesion.bmp")
        
        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L') # grayscale
        except FileNotFoundError:
            # Fallback for missing files
            image = Image.new('RGB', (256, 256))
            mask = Image.new('L', (256, 256))

        # We will use dummy classification labels for this example if no excel is provided.
        # In a real PH2 run, you'd parse PH2_dataset.xlsx.
        # Let's just mock 3 classes (0, 1, 2) deterministically based on string hash
        label = hash(img_name) % 3

        if self.transform:
            # Seed the random number generator so image and mask get same transform if random
            seed = np.random.randint(2147483647)
            
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            
            random.seed(seed)
            torch.manual_seed(seed)
            mask = self.transform(mask)

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0).float()

        return image, mask, label

def get_dataloaders(root_dir, batch_size=4, img_size=256, num_workers=0):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = PH2Dataset(root_dir, split_type='train', transform=train_transform)
    val_dataset = PH2Dataset(root_dir, split_type='val', transform=val_test_transform)
    test_dataset = PH2Dataset(root_dir, split_type='test', transform=val_test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
