# 🍎 Calorie Prediction — нейросеть для оценки калорийности блюд

Мультимодальная нейросеть, которая по **фотографии блюда** и **списку ингредиентов**
предсказывает его калорийность (ккал). Разработана в рамках проекта по Deep Learning.

## 📊 Результаты

| Метрика | Значение |
|---|---|
| **Test MAE (целевая)** | **32.34 ккал** ✅ (< 50) |
| **Медиана ошибки** | **9.97 ккал** |
| **90-й перцентиль** | **84.63 ккал** |
| **Best val MAE** | 23.90 ккал (эпоха 20) |
| **Время обучения** | 33.7 мин на Tesla T4 |

### Baseline'ы для сравнения

| Baseline | Test MAE |
|---|---|
| Медиана train | 167.34 ккал |
| Линейная регрессия по массе | 104.57 ккал |
| TF-IDF + Ridge (только текст) | 106.09 ккал |
| **Масса + текст (Ridge)** | **67.45 ккал** |
| **Наша модель (EfficientNet + DistilBERT + масса)** | **32.34 ккал** |

Наша нейросеть **в 2 раза** лучше лучшего классического baseline.

## 🏗️ Архитектура
Изображение (224×224×3) ──► EfficientNet-B0 ──► [Linear+LN+Drop] ──┐
│
Текст (ингредиенты) ──► DistilBERT ──► [mean pool] ──► [Linear+LN+Drop] ──┼──► concat ──► [Linear+ReLU] ──► Linear → 1 (плотность)
│
Масса (нормализованная) ──────────────────────────────────────────┘
**Ключевое решение:** модель предсказывает не абсолютные калории, а **плотность** (ккал/100г). Это защищает от тривиального решения «калории ≈ масса». Итоговая калорийность восстанавливается как `pred_density * mass / 100`.

## 📁 Структура репозитория
calorie-project/
├── data/
│ ├── dish.csv # метаданные блюд (calories, mass, ingredients, split)
│ ├── ingredients.csv # id → название ингредиента
│ └── images/ # [не в git] папки {dish_id}/rgb.png (1.4 GB)
├── scripts/
│ ├── init.py
│ ├── config.py # все гиперпараметры, пути, seed
│ ├── dataset.py # DishDataset, аугментации, DataLoader
│ └── utils.py # модель, train, evaluate_on_test
├── notebook.ipynb # EDA + обучение + инференс + анализ ошибок
├── best_calorie_model.pth # чекпоинт (Git LFS, ~271 MB)
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
## 🚀 Как воспроизвести

### 1. Установка

```bash
pip install -r requirements.txt2. Данные
Скачать датасет и распаковать:

text
data/
├── dish.csv
├── ingredients.csv
└── images/
    ├── dish_1558720236/rgb.png
    └── ...
3. Обучение
python
from scripts.config import Config
from scripts.dataset import DishDataset, get_train_transforms, get_valid_transforms
from scripts.utils import seed_everything, train
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# --- подготовка данных ---
dish = pd.read_csv("data/dish.csv")
ingr = pd.read_csv("data/ingredients.csv")
ingr_mapping = dict(zip(ingr["id"], ingr["ingr"]))

def map_ingredients(s):
    if pd.isna(s): return ""
    return ", ".join(
        ingr_mapping[int(x.replace("ingr_", ""))]
        for x in s.split(";")
        if x.strip() and x.replace("ingr_", "").isdigit()
        and int(x.replace("ingr_", "")) in ingr_mapping
    )

dish["ingredients_clean"] = dish["ingredients"].apply(map_ingredients)

# --- сплит и статистики ---
cfg = Config()
seed_everything(cfg.SEED)

train_full = dish[dish["split"] == "train"].reset_index(drop=True)
test       = dish[dish["split"] == "test"].reset_index(drop=True)
train_part, val_part = train_test_split(train_full, test_size=cfg.VAL_SPLIT, random_state=cfg.SEED)

train_part["density"] = train_part["total_calories"] / (train_part["total_mass"] + 1e-6) * 100
cfg.TARGET_MEAN = train_part["density"].mean()
cfg.TARGET_STD  = train_part["density"].std()
cfg.MASS_MEAN   = train_part["total_mass"].mean()
cfg.MASS_STD    = train_part["total_mass"].std()

# --- датасеты ---
tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL)

def make_ds(df, tf):
    return DishDataset(df=df, images_dir=cfg.IMAGES_DIR, transform=tf,
                       tokenizer=tokenizer, max_length=cfg.MAX_LENGTH,
                       target_mean=cfg.TARGET_MEAN, target_std=cfg.TARGET_STD,
                       mass_mean=cfg.MASS_MEAN, mass_std=cfg.MASS_STD)

train_ds = make_ds(train_part, get_train_transforms(cfg.IMG_SIZE))
val_ds   = make_ds(val_part,   get_valid_transforms(cfg.IMG_SIZE))

# --- обучение ---
model, history = train(cfg, train_ds, val_ds)
4. Инференс
python
from scripts.utils import evaluate_on_test
df_test, model, ckpt = evaluate_on_test(cfg, test_ds)
print(f"Test MAE: {df_test['error'].mean():.2f} ккал")
📓 Jupyter Notebook
Полный пайплайн в notebook.ipynb:

EDA — распределения, baseline'ы.

Обучение — конфиг, датасеты, train().

Валидация — MAE на тесте, гистограмма ошибок.

Анализ ошибок — топ-5 худших предсказаний + причины.

⚠️ Ограничения и что можно улучшить
Недооценка высокоплотных блюд. Миндаль (580 ккал/100г), масла, орехи, сыры — редкие в train, поэтому модель их недооценивает. Пример: на фото миндаль занимает 60% тарелки, но модель предсказывает 460 ккал вместо 888.

Фото не передаёт вес. EfficientNet-B0 не может оценить глубину блюда или размер порции по 2D-изображению.

Текст слабо «привязан» к калорийности. DistilBERT обучен на общих текстах, для него almonds и apple — равнозначные токены.

Гипотезы для улучшения:

Feature engineering: density_prior из названий ингредиентов (если есть almonds → +N к плотности).

Weighted MAE: увеличить вес ошибок на высококалорийных примерах.

Больше эпох: на 20-й val_MAE ещё падала (23.90). Можно 30–40.

Убрать мусорные токены (deprecated) из текста.

🛠️ Технологии
PyTorch · timm (EfficientNet-B0) · HuggingFace transformers (DistilBERT) · albumentations · torchmetrics

👤 Автор
Проект выполнен в рамках курса по Deep Learning.

⚠️ Модель — исследовательский прототип. Не использовать для медицинских или диетологических решений без валидации специалистом.
