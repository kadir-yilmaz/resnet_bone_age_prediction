import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from model import get_model


# Config
class Config:
    DATA_DIR = "data"
    TEST_CSV = "boneage-test-dataset.csv"
    TEST_IMG_DIR = "boneage-test-dataset"
    MODEL_PATH = "best_model.pth"
    IMG_SIZE = 224
    BATCH_SIZE = 16
    NUM_WORKERS = 0


# Dataset
class TestDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        print(f"Test dataset: {len(self.df)} samples")
    
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
        
        return image, gender, target, row['id']


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_targets, all_genders, all_ids = [], [], [], []
    
    with torch.no_grad():
        for images, genders, targets, ids in tqdm(dataloader, desc="Evaluating"):
            images, genders = images.to(device), genders.to(device)
            outputs = model(images, genders)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
            all_genders.extend(genders.cpu().numpy().flatten())
            all_ids.extend(ids)
    
    return np.array(all_preds), np.array(all_targets), np.array(all_genders), all_ids


def main():
    print("=" * 50)
    print("Bone Age Prediction - Test")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if not os.path.exists(Config.MODEL_PATH):
        print(f"Error: Model not found: {Config.MODEL_PATH}")
        return
    
    # Load model
    checkpoint = torch.load(Config.MODEL_PATH, map_location=device, weights_only=False)
    model = get_model(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("Model: ResNet50")
    
    # Dataset
    csv_path = os.path.join(Config.DATA_DIR, Config.TEST_CSV)
    img_dir = os.path.join(Config.DATA_DIR, Config.TEST_IMG_DIR)
    test_dataset = TestDataset(csv_path, img_dir, transform=get_test_transform())
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    # Evaluate
    predictions, targets, genders, ids = evaluate(model, test_loader, device)
    
    # Metrics
    errors = predictions - targets
    abs_errors = np.abs(errors)
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    print(f"\n📊 Results:")
    print(f"   MAE:  {mae:.2f} months ({mae/12:.2f} years)")
    print(f"   RMSE: {rmse:.2f} months ({rmse/12:.2f} years)")
    
    within_6mo = np.mean(abs_errors <= 6) * 100
    within_12mo = np.mean(abs_errors <= 12) * 100
    print(f"\n🎯 Accuracy:")
    print(f"   Within 6 months:  {within_6mo:.1f}%")
    print(f"   Within 12 months: {within_12mo:.1f}%")
    
    # Save results
    results_df = pd.DataFrame({
        'ID': [int(i) for i in ids],
        'Ground Truth (months)': targets,
        'Prediction (months)': np.round(predictions, 1),
        'Error (months)': np.round(errors, 1),
        'Gender': ['M' if g == 1 else 'F' for g in genders]
    })
    results_df.to_csv("test_results.csv", index=False)
    print(f"\n✓ Results saved to: test_results.csv")


if __name__ == "__main__":
    main()
