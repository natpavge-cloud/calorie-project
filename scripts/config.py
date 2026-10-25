"""
Конфигурация проекта предсказания калорийности.

Один класс Config со всеми параметрами: пути, модели, гиперпараметры,
аугментации, устройство. Заполняется статически, статистики нормализации
(TARGET_MEAN/STD, MASS_MEAN/STD) выставляются в ноутбуке на train_part.
"""
import os
import torch


class Config:
    def __init__(self):
        # =====================================================================
        # 1. Воспроизводимость и устройство
        # =====================================================================
        self.SEED = 42
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # =====================================================================
        # 2. Пути (относительно корня репозитория)
        # =====================================================================
        # CSV
        self.DISH_CSV = os.path.join("data", "dish.csv")
        self.INGREDIENTS_CSV = os.path.join("data", "ingredients.csv")
        # Изображения: каждая подпапка = dish_id, внутри rgb.png
        self.IMAGES_DIR = os.path.join("data", "images")
        # Куда сохранять чекпоинт
        self.MODEL_SAVE_PATH = "best_calorie_model.pth"

        # =====================================================================
        # 3. Модели (энкодеры)
        # =====================================================================
        # Текстовый энкодер: ингредиенты на английском -> английский BERT
        self.TEXT_MODEL = "distilbert-base-uncased"
        # Картиночный энкодер: EfficientNet-B0 — компромисс скорость/качество
        self.IMAGE_MODEL = "efficientnet_b0"
        # Размер эмбеддингов после проекционных слоёв
        self.HIDDEN_DIM = 256
        # Размер входного изображения
        self.IMG_SIZE = 224
        # Максимальная длина токенов текста
        self.MAX_LENGTH = 64

        # =====================================================================
        # 4. Обучение
        # =====================================================================
        # BATCH_SIZE = 4 — рекомендация ТЗ, чтобы не было OOM
        self.BATCH_SIZE = 4
        self.EPOCHS = 20
        # Доля валидации, выделяемая из train
        self.VAL_SPLIT = 0.15
        # Weight decay для AdamW
        self.WEIGHT_DECAY = 1e-2

        # Раздельные learning rate
        self.TEXT_LR = 1e-5
        self.IMAGE_LR = 1e-4
        self.CLASSIFIER_LR = 1e-3

        # Mixed precision (AMP)
        self.USE_AMP = True

        # Количество воркеров DataLoader
        self.NUM_WORKERS = 2

        # Клиппинг градиентов
        self.GRAD_CLIP = 1.0

        # =====================================================================
        # 5. Аугментации
        # =====================================================================
        self.AUG_SCALE_MIN = 0.8
        self.AUG_FLIP_P = 0.5
        self.AUG_COLOR_P = 0.3

        # =====================================================================
        # 6. Статистики нормализации (заполняются в ноутбуке)
        # =====================================================================
        self.TARGET_MEAN = None
        self.TARGET_STD = None
        self.MASS_MEAN = None
        self.MASS_STD = None

    def summary(self) -> str:
        return (
            f"=== Config ===\n"
            f"  DEVICE:         {self.DEVICE}\n"
            f"  TEXT_MODEL:     {self.TEXT_MODEL}\n"
            f"  IMAGE_MODEL:    {self.IMAGE_MODEL}\n"
            f"  IMG_SIZE:       {self.IMG_SIZE}\n"
            f"  MAX_LENGTH:     {self.MAX_LENGTH}\n"
            f"  HIDDEN_DIM:     {self.HIDDEN_DIM}\n"
            f"  BATCH_SIZE:     {self.BATCH_SIZE}\n"
            f"  EPOCHS:         {self.EPOCHS}\n"
            f"  TEXT_LR:        {self.TEXT_LR}\n"
            f"  IMAGE_LR:       {self.IMAGE_LR}\n"
            f"  CLASSIFIER_LR:  {self.CLASSIFIER_LR}\n"
            f"  WEIGHT_DECAY:   {self.WEIGHT_DECAY}\n"
            f"  USE_AMP:        {self.USE_AMP}\n"
            f"  NUM_WORKERS:    {self.NUM_WORKERS}\n"
            f"  VAL_SPLIT:      {self.VAL_SPLIT}\n"
            f"  SAVE_PATH:      {self.MODEL_SAVE_PATH}\n"
            f"=============="
        )
