import os
import time
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from model import get_model


# Config
class Config:
    DATA_DIR = "data"
    TRAIN_CSV = "boneage-training-dataset.csv"
    TRAIN_IMG_DIR = "boneage-training-dataset"
    
    PRETRAINED = True
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    IMG_SIZE = 224
    VAL_SPLIT = 0.15
    TRAIN_FRACTION = 1  # Verinin ne kadarı kullanılacak (0.5 = %50)
    SEED = 42  # None = rastgele, sayı = sabit
    NUM_WORKERS = 0
    SAVE_PATH = "best_model.pth"


# Dataset
class BoneAgeDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.mean_age = self.df['boneage'].mean()
        self.std_age = self.df['boneage'].std()
        
        print(f"Dataset: {len(self.df)} samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id']}.png")
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        gender = torch.tensor([1.0 if row['male'] else 0.0])
        target = torch.tensor([row['boneage']], dtype=torch.float32)
        
        return image, gender, target


# Transforms
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# Training
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for images, genders, targets in tqdm(dataloader, desc="Training"):
        images, genders, targets = images.to(device), genders.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, genders)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    
    return total_loss / len(dataloader.dataset)


# Validation
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, genders, targets in tqdm(dataloader, desc="Validating"):
            images, genders, targets = images.to(device), genders.to(device), targets.to(device)
            
            outputs = model(images, genders)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    return total_loss / len(dataloader.dataset), mae


# Main
def main():
    print("=" * 50)
    print("Bone Age Prediction - Training")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Seed
    if Config.SEED is not None:
        torch.manual_seed(Config.SEED)
        print(f"Seed: {Config.SEED}")
    
    # Dataset
    csv_path = os.path.join(Config.DATA_DIR, Config.TRAIN_CSV)
    img_dir = os.path.join(Config.DATA_DIR, Config.TRAIN_IMG_DIR)
    full_dataset = BoneAgeDataset(csv_path, img_dir, transform=get_transforms(train=True))
    
    # Use fraction of data
    total_size = int(len(full_dataset) * Config.TRAIN_FRACTION)
    unused_size = len(full_dataset) - total_size
    used_dataset, _ = random_split(full_dataset, [total_size, unused_size])
    
    # Train/Val split
    val_size = int(total_size * Config.VAL_SPLIT)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(used_dataset, [train_size, val_size])
    val_dataset.dataset.transform = get_transforms(train=False)
    
    print(f"Using {Config.TRAIN_FRACTION*100:.0f}% of data: Train={train_size}, Val={val_size}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Model
    model = get_model(Config.PRETRAINED).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ResNet50 ({total_params:,} params)")
    
    # Loss, Optimizer, Scheduler
    criterion = nn.L1Loss()  # MAE
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    best_mae = float('inf')
    start_time = time.time()
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} months")
        
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_mae': val_mae,
                'config': {'img_size': Config.IMG_SIZE}
            }, Config.SAVE_PATH)
            print(f"✓ Best model saved! MAE: {val_mae:.2f}")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Best MAE: {best_mae:.2f} months")
    print(f"Time: {int(total_time // 60)}m {int(total_time % 60)}s")


if __name__ == "__main__":
    main()
