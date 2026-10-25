"""
Датасет и загрузчики для мультимодальной модели калорийности.

DishDataset возвращает:
    - image:       torch.Tensor (3, H, W) после albumentations + ToTensorV2
    - text:        dict с input_ids / attention_mask от HuggingFace tokenizer
    - mass:        np.float32 [1] — нормализованная масса
    - target:      np.float32 [1] — нормализованная ПЛОТНОСТЬ (ккал/100г)

Почему плотность, а не абсолютные калории:
    - модель иначе тривиально выучивает «калории ≈ масса»,
    - плотность — более информативная цель, калории восстанавливаются
      как pred_density * true_mass / 100.
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Допустимый диапазон плотности (ккал/100г) — защита от выбросов и нулевой массы
DENSITY_MIN = 0.0
DENSITY_MAX = 1000.0


class DishDataset(Dataset):
    """
    Параметры
    ----------
    df : pd.DataFrame
        Ожидаемые колонки: dish_id, ingredients_clean, total_mass, total_calories.
    images_dir : str
        Корневая папка, внутри — подпапки {dish_id}/rgb.png
    transform : albumentations.Compose | None
    tokenizer : HuggingFace tokenizer | None
    max_length : int
        Максимальная длина токенов.
    target_mean, target_std, mass_mean, mass_std : float
        Статистики нормализации (считаются на train_part в ноутбуке).
    validate_paths : bool
        Проверять ли существование всех картинок при создании датасета.
        True для train/val/test (мы всегда знаем, что данные полные).
    """

    def __init__(self, df, images_dir="data/images", transform=None,
                 tokenizer=None, max_length=64,
                 target_mean=0.0, target_std=1.0,
                 mass_mean=0.0, mass_std=1.0,
                 validate_paths=True):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_mean = target_mean
        self.target_std = target_std
        self.mass_mean = mass_mean
        self.mass_std = mass_std

        # Валидация путей: упадём сразу, если чего-то нет
        if validate_paths:
            missing = []
            for dish_id in self.df["dish_id"].values:
                p = os.path.join(self.images_dir, str(dish_id), "rgb.png")
                if not os.path.exists(p):
                    missing.append(p)
            if missing:
                raise FileNotFoundError(
                    f"Не найдено {len(missing)} изображений. Примеры:\n"
                    + "\n".join(missing[:10])
                )

        # Логирование (кратко)
        n = len(self.df)
        print(f"[DishDataset] size={n}, images_dir={self.images_dir}, "
              f"transform={'yes' if transform else 'no'}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        dish_id = row["dish_id"]

        # ---------------- Изображение ----------------
        img_path = os.path.join(self.images_dir, str(dish_id), "rgb.png")
        with Image.open(img_path) as img:
            image = np.array(img.convert("RGB"))

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        # ---------------- Текст ----------------
        ingredients = row.get("ingredients_clean", "")
        text_input = str(ingredients) if pd.notna(ingredients) else ""

        if self.tokenizer is not None:
            encoded = self.tokenizer(
                text_input,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            text_features = {k: v.squeeze(0) for k, v in encoded.items()}
        else:
            text_features = text_input

        # ---------------- Масса и целевая (плотность) ----------------
        mass_val = float(row.get("total_mass", 0.0) or 0.0)
        cal_val  = float(row.get("total_calories", 0.0) or 0.0)

        # Плотность: ккал на 100 г. Clamp — защита от деления на ~0 и выбросов.
        density_val = cal_val / (mass_val + 1e-6) * 100.0
        density_val = float(np.clip(density_val, DENSITY_MIN, DENSITY_MAX))

        mass_norm   = (mass_val   - self.mass_mean)   / (self.mass_std   + 1e-8)
        target_norm = (density_val - self.target_mean) / (self.target_std + 1e-8)

        mass   = np.array([mass_norm],   dtype=np.float32)
        target = np.array([target_norm], dtype=np.float32)

        return {
            "image":  image,
            "text":   text_features,
            "mass":   mass,
            "target": target,
        }


def get_train_transforms(img_size: int = 224,
                         scale_min: float = 0.8,
                         flip_p: float = 0.5,
                         color_p: float = 0.3):
    """Аугментации для тренировки. Параметры совпадают с Config."""
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size),
                            scale=(scale_min, 1.0), p=1.0),
        A.HorizontalFlip(p=flip_p),
        A.ColorJitter(brightness=0.1, contrast=0.1,
                      saturation=0.1, hue=0.05, p=color_p),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_valid_transforms(img_size: int = 224):
    """Валидация/тест — только resize + normalize, без аугментаций."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def build_dataloaders(train_ds, valid_ds, test_ds, batch_size=4,
                      num_workers=2):
    """Удобная обёртка для DataLoader-ов."""
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, valid_loader, test_loader
