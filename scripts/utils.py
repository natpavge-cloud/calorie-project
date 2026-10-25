"""
Обучение, валидация и инференс мультимодальной модели калорийности.
"""
import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import torchmetrics
import timm
from transformers import AutoModel
from tqdm.auto import tqdm


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiModalCalorieModel(nn.Module):
    def __init__(self, image_model_name="efficientnet_b0",
                 text_model_name="distilbert-base-uncased",
                 hidden_dim=256):
        super().__init__()
        self.image_encoder = timm.create_model(
            image_model_name, pretrained=True, num_classes=0
        )
        num_image_features = self.image_encoder.num_features
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        num_text_features = self.text_encoder.config.hidden_size

        self.image_projection = nn.Sequential(
            nn.Linear(num_image_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(num_text_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3),
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, image, text_inputs, mass):
        image_feats = self.image_encoder(image)
        image_emb = self.image_projection(image_feats)

        text_out = self.text_encoder(**text_inputs)
        last_hidden = text_out.last_hidden_state
        mask = text_inputs["attention_mask"].unsqueeze(-1).float()
        text_feats = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        text_emb = self.text_projection(text_feats)

        combined = torch.cat([image_emb, text_emb, mass], dim=1)
        x = self.fusion_layer(combined)
        return self.head(x)


def build_optimizer(model, cfg):
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")

    def split_params(module, lr):
        decay, no_decay_params = [], []
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                no_decay_params.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay,           "lr": lr, "weight_decay": cfg.WEIGHT_DECAY},
            {"params": no_decay_params, "lr": lr, "weight_decay": 0.0},
        ]

    param_groups = []
    param_groups += split_params(model.text_encoder,     cfg.TEXT_LR)
    param_groups += split_params(model.image_encoder,    cfg.IMAGE_LR)
    param_groups += split_params(model.text_projection,  cfg.CLASSIFIER_LR)
    param_groups += split_params(model.image_projection, cfg.CLASSIFIER_LR)
    param_groups += split_params(model.fusion_layer,     cfg.CLASSIFIER_LR)
    param_groups += split_params(model.head,             cfg.CLASSIFIER_LR)
    return AdamW(param_groups)


def train(cfg, train_dataset, valid_dataset):
    seed_everything(cfg.SEED)
    device = torch.device(cfg.DEVICE)
    print(cfg.summary())

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )

    model = MultiModalCalorieModel(
        image_model_name=cfg.IMAGE_MODEL,
        text_model_name=cfg.TEXT_MODEL,
        hidden_dim=cfg.HIDDEN_DIM,
    ).to(device)

    criterion = nn.L1Loss()
    density_mae  = torchmetrics.MeanAbsoluteError().to(device)
    calories_mae = torchmetrics.MeanAbsoluteError().to(device)

    optimizer = build_optimizer(model, cfg)
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=0.5, patience=2, min_lr=1e-6)

    use_amp = cfg.USE_AMP and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    t_mean, t_std = cfg.TARGET_MEAN, cfg.TARGET_STD
    m_mean, m_std = cfg.MASS_MEAN,   cfg.MASS_STD

    best_val_cal_mae = float("inf")
    best_epoch = 0
    history = []

    print(f"\nСтарт обучения на {device}. AMP={use_amp}. Эпох: {cfg.EPOCHS}\n")

    for epoch in range(cfg.EPOCHS):
        t0 = time.time()

        model.train()
        density_mae.reset()
        train_loss_sum = 0.0
        n_batches = 0

        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.EPOCHS} [Train]")
        for batch in bar:
            images = batch["image"].to(device, non_blocking=True)
            mass   = batch["mass"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            text   = {k: v.to(device, non_blocking=True)
                      for k, v in batch["text"].items()}

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=use_amp):
                out  = model(images, text, mass)
                loss = criterion(out, target)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            density_mae.update(out.detach(), target)
            train_loss_sum += loss.item()
            n_batches += 1
            bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss_avg      = train_loss_sum / max(n_batches, 1)
        train_density_mae_c = density_mae.compute().item() * t_std

        model.eval()
        density_mae.reset()
        calories_mae.reset()
        with torch.no_grad():
            for batch in valid_loader:
                images = batch["image"].to(device, non_blocking=True)
                mass   = batch["mass"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                text   = {k: v.to(device, non_blocking=True)
                          for k, v in batch["text"].items()}

                with autocast("cuda", enabled=use_amp):
                    out = model(images, text, mass)

                density_mae.update(out, target)
                pred_density = out    * t_std + t_mean
                true_density = target * t_std + t_mean
                true_mass    = mass   * m_std + m_mean
                calories_mae.update(
                    pred_density * true_mass / 100,
                    true_density * true_mass / 100,
                )

        val_density_mae  = density_mae.compute().item()
        val_calories_mae = calories_mae.compute().item()
        dt = time.time() - t0

        print(f"--> Epoch {epoch+1:02d} | {dt:5.1f}s | "
              f"train_loss={train_loss_avg:.4f} | "
              f"train_densMAE={train_density_mae_c:.2f} | "
              f"val_densMAE={val_density_mae:.2f} | "
              f"val_CAL_MAE={val_calories_mae:.2f} | "
              f"lr_head={optimizer.param_groups[-1]['lr']:.2e}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss_avg,
            "val_density_mae": val_density_mae,
            "val_calories_mae": val_calories_mae,
            "time_sec": dt,
        })

        if val_calories_mae < best_val_cal_mae:
            best_val_cal_mae = val_calories_mae
            best_epoch = epoch + 1
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": best_epoch,
                "best_val_calories_mae": best_val_cal_mae,
                "target_mean": t_mean, "target_std": t_std,
                "mass_mean":   m_mean, "mass_std":   m_std,
                "config": {
                    "TEXT_MODEL":  cfg.TEXT_MODEL,
                    "IMAGE_MODEL": cfg.IMAGE_MODEL,
                    "HIDDEN_DIM":  cfg.HIDDEN_DIM,
                    "MAX_LENGTH":  cfg.MAX_LENGTH,
                    "IMG_SIZE":    cfg.IMG_SIZE,
                },
            }, cfg.MODEL_SAVE_PATH)
            print(f"    [+] Новый best (val_cal_MAE={best_val_cal_mae:.2f}). Сохранено.")

        scheduler.step(val_calories_mae)

    print(f"\n=== Обучение завершено ===")
    print(f"Лучшая эпоха: {best_epoch}, best val_cal_MAE: {best_val_cal_mae:.2f} ккал")
    return model, pd.DataFrame(history)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def evaluate_on_test(cfg, test_dataset, checkpoint_path=None):
    device = torch.device(cfg.DEVICE)
    ckpt = load_checkpoint(checkpoint_path or cfg.MODEL_SAVE_PATH, device)

    t_mean, t_std = ckpt["target_mean"], ckpt["target_std"]
    m_mean, m_std = ckpt["mass_mean"],   ckpt["mass_std"]

    model = MultiModalCalorieModel(
        image_model_name=ckpt["config"]["IMAGE_MODEL"],
        text_model_name=ckpt["config"]["TEXT_MODEL"],
        hidden_dim=ckpt["config"]["HIDDEN_DIM"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = DataLoader(test_dataset, batch_size=1,
                        shuffle=False, num_workers=cfg.NUM_WORKERS)

    preds, targets, masses = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test inference"):
            images = batch["image"].to(device)
            mass   = batch["mass"].to(device)
            target = batch["target"].to(device)
            text   = {k: v.to(device) for k, v in batch["text"].items()}
            out = model(images, text, mass).item()

            pred_density = out          * t_std + t_mean
            true_density = target.item()* t_std + t_mean
            true_mass    = mass.item()  * m_std + m_mean

            preds.append(pred_density * true_mass / 100)
            targets.append(true_density * true_mass / 100)
            masses.append(true_mass)

    df = pd.DataFrame({
        "dish_id":       test_dataset.df["dish_id"].values,
        "pred_calories": preds,
        "true_calories": targets,
        "true_mass":     masses,
    })
    df["error"] = (df["true_calories"] - df["pred_calories"]).abs()

    mae = df["error"].mean()
    print(f"\nTest MAE: {mae:.2f} ккал")
    print(f"Медиана ошибки: {df['error'].median():.2f}")
    print(f"90-й перцентиль: {df['error'].quantile(0.9):.2f}")

    return df, model, ckpt
