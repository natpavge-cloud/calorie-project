import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchmetrics
import timm
from transformers import AutoModel
from tqdm.auto import tqdm


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiModalCalorieModel(nn.Module):
    def __init__(self, image_model_name='efficientnet_b0',
                 text_model_name='distilbert-base-uncased',
                 hidden_dim=256):
        super().__init__()
        self.image_encoder = timm.create_model(image_model_name, pretrained=True, num_classes=0)
        num_image_features = self.image_encoder.num_features

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        num_text_features = self.text_encoder.config.hidden_size

        self.image_projection = nn.Sequential(
            nn.Linear(num_image_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(num_text_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3)
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, image, text_inputs, mass):
        image_feats = self.image_encoder(image)
        image_embeddings = self.image_projection(image_feats)

        text_outputs = self.text_encoder(**text_inputs)
        last_hidden = text_outputs.last_hidden_state
        attention_mask = text_inputs['attention_mask'].unsqueeze(-1).float()
        text_feats = (last_hidden * attention_mask).sum(1) / attention_mask.sum(1).clamp(min=1e-9)
        text_embeddings = self.text_projection(text_feats)

        combined = torch.cat([image_embeddings, text_embeddings, mass], dim=1)
        x = self.fusion_layer(combined)
        return self.head(x)


def train(cfg, train_dataset, valid_dataset):
    seed_everything(cfg.SEED)
    device = torch.device(cfg.DEVICE)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    model = MultiModalCalorieModel(
        image_model_name=cfg.IMAGE_MODEL,
        text_model_name=cfg.TEXT_MODEL,
        hidden_dim=cfg.HIDDEN_DIM
    ).to(device)

    criterion = nn.L1Loss()

    density_mae  = torchmetrics.MeanAbsoluteError().to(device)
    calories_mae = torchmetrics.MeanAbsoluteError().to(device)

    optimizer = AdamW([
        {"params": model.text_encoder.parameters(),   "lr": cfg.TEXT_LR},
        {"params": model.image_encoder.parameters(),  "lr": cfg.IMAGE_LR},
        {"params": model.text_projection.parameters(),"lr": cfg.CLASSIFIER_LR},
        {"params": model.image_projection.parameters(),"lr": cfg.CLASSIFIER_LR},
        {"params": model.fusion_layer.parameters(),   "lr": cfg.CLASSIFIER_LR},
        {"params": model.head.parameters(),           "lr": cfg.CLASSIFIER_LR},
    ], weight_decay=cfg.WEIGHT_DECAY)

    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=1e-6)

    t_mean, t_std = cfg.TARGET_MEAN, cfg.TARGET_STD
    m_mean, m_std = cfg.MASS_MEAN,   cfg.MASS_STD

    best_val_cal_mae = float('inf')
    best_epoch = 0

    print(f"Старт обучения на {device}. Эпох: {cfg.EPOCHS}, Batch: {cfg.BATCH_SIZE}")
    print(f"Модель: {cfg.IMAGE_MODEL} + {cfg.TEXT_MODEL}")

    for epoch in range(cfg.EPOCHS):
        model.train()
        density_mae.reset()
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        for batch in bar:
            images = batch['image'].to(device, non_blocking=True)
            mass   = batch['mass'].to(device, non_blocking=True)
            target = batch['target'].to(device, non_blocking=True)
            text   = {k: v.to(device, non_blocking=True) for k, v in batch['text'].items()}

            optimizer.zero_grad()
            out = model(images, text, mass)
            loss = criterion(out, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            density_mae.update(out, target)
            bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_density_mae = density_mae.compute().item() * t_std

        model.eval()
        density_mae.reset()
        calories_mae.reset()
        with torch.no_grad():
            for batch in valid_loader:
                images = batch['image'].to(device, non_blocking=True)
                mass   = batch['mass'].to(device, non_blocking=True)
                target = batch['target'].to(device, non_blocking=True)
                text   = {k: v.to(device, non_blocking=True) for k, v in batch['text'].items()}

                out = model(images, text, mass)
                density_mae.update(out, target)

                pred_density = out    * t_std + t_mean
                true_density = target * t_std + t_mean
                true_mass    = mass   * m_std + m_mean
                calories_mae.update(pred_density * true_mass / 100,
                                    true_density * true_mass / 100)

        val_density_mae  = density_mae.compute().item()
        val_calories_mae = calories_mae.compute().item()
        current_lr = optimizer.param_groups[-1]['lr']

        print(f"--> Epoch {epoch+1} | Train density MAE: {train_density_mae:.2f} | "
              f"Val density MAE: {val_density_mae:.2f} | Val CALORIES MAE: {val_calories_mae:.2f} | "
              f"LR_head: {current_lr:.2e}")

        if val_calories_mae < best_val_cal_mae:
            best_val_cal_mae = val_calories_mae
            best_epoch = epoch + 1
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': best_epoch,
                'best_val_calories_mae': best_val_cal_mae,
                'target_mean': t_mean, 'target_std': t_std,
                'mass_mean': m_mean,   'mass_std': m_std,
                'config': {
                    'TEXT_MODEL':  cfg.TEXT_MODEL,
                    'IMAGE_MODEL': cfg.IMAGE_MODEL,
                    'HIDDEN_DIM':  cfg.HIDDEN_DIM,
                    'MAX_LENGTH':  cfg.MAX_LENGTH,
                    'IMG_SIZE':    cfg.IMG_SIZE,
                }
            }, cfg.MODEL_SAVE_PATH)
            print(f"    [!] Новый лучший результат (val cal MAE={best_val_cal_mae:.2f}). Сохранён.")

        scheduler.step()

    print(f"\nГотово. Лучшая эпоха: {best_epoch} | Best val cal MAE: {best_val_cal_mae:.2f} ккал")
    return model


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)
