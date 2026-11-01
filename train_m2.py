import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# --- 1. NOVÝ DATASET ČTOUCÍ Z CSV ---
class WikiArtCSVDataset(Dataset):
    def __init__(self, csv_file, split, transform=None):
        # Načtení celého CSV
        self.df = pd.read_csv(csv_file)
        # Filtrace pouze pro požadovaný split (train, val, nebo test)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.transform = transform
        
        # Dynamické vytvoření mapování textových labelů na čísla pro zachovaných 14 tříd
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        label_str = self.df.loc[idx, 'label']
        label = self.class_to_idx[label_str]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- 2. DYNAMICKÁ AUGMENTACE (DATA ENHANCEMENT) ---
# Aplikuje se pouze na trénovací data pro zamezení overfittingu
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet standard
    transforms.RandomErasing(p=0.2) # Strukturální regularizace požadovaná v zadání[cite: 2]
])

# Pro validaci a testování nesmí probíhat náhodné deformace
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. INICIALIZACE DATALOADERŮ ---
csv_path = "dataset_split.csv"

train_dataset = WikiArtCSVDataset(csv_file=csv_path, split='train', transform=train_transform)
val_dataset = WikiArtCSVDataset(csv_file=csv_path, split='val', transform=val_test_transform)
test_dataset = WikiArtCSVDataset(csv_file=csv_path, split='test', transform=val_test_transform)

print(f"Počet trénovacích vzorků: {len(train_dataset)}")
print(f"Počet validačních vzorků: {len(val_dataset)}")
print(f"Počet testovacích vzorků: {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)