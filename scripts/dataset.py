import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DishDataset(Dataset):
    def __init__(self, df, images_dir="data/images", transform=None,
                 tokenizer=None, max_length=64,
                 target_mean=0.0, target_std=1.0,
                 mass_mean=0.0, mass_std=1.0):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_mean = target_mean
        self.target_std = target_std
        self.mass_mean = mass_mean
        self.mass_std = mass_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dish_id = row['dish_id']

        img_path = os.path.join(self.images_dir, str(dish_id), "rgb.png")
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                image = np.array(img.convert("RGB"))
        else:
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        if self.transform:
            image = self.transform(image=image)['image']

        ingredients = row['ingredients_clean']
        text_input = str(ingredients) if pd.notna(ingredients) else ""

        if self.tokenizer:
            text_encoded = self.tokenizer(
                text_input, padding='max_length', truncation=True,
                max_length=self.max_length, return_tensors="pt"
            )
            text_features = {k: v.squeeze(0).clone() for k, v in text_encoded.items()}
        else:
            text_features = text_input

        mass_val = row['total_mass']
        total_cal = row['total_calories']
        mass_val = mass_val if pd.notna(mass_val) else 0.0
        total_cal = total_cal if pd.notna(total_cal) else 0.0

        # Считаем плотность (ккал на 100 г) вместо чистых калорий
        density_val = total_cal / (mass_val + 1e-6) * 100

        mass_norm = (mass_val - self.mass_mean) / (self.mass_std + 1e-8)
        target_norm = (density_val - self.target_mean) / (self.target_std + 1e-8)

        mass = np.array([mass_norm], dtype=np.float32)
        target = np.array([target_norm], dtype=np.float32)

        return {'image': image, 'text': text_features, 'mass': mass, 'target': target}


def get_train_transforms(img_size=224):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_valid_transforms(img_size=224):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
