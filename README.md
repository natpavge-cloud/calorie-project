# Calorie Prediction - Multimodal Deep Learning

Мультимодальная нейросеть для предсказания калорийности блюд по фотографии и текстовому описанию ингредиентов.

## Проект выполнила:
Павлова Наталья Геннадьевна, группа М25-555

## Результаты

| Метрика | Значение |
|---|---|
| Test MAE | 31.71 ккал (цель < 50) |
| Медиана ошибки | 11.60 ккал |
| 90-й перцентиль | 81.09 ккал |
| Best val MAE | 24.92 ккал |

## Задача

На вход подаётся фотография блюда (`rgb.png`) и список ингредиентов на английском языке. На выходе — калорийность блюда в ккал.

Ключевая идея: модель предсказывает не абсолютную калорийность, а плотность (ккал на 100 г), после чего калорийность восстанавливается как `calories = density * mass / 100`. Это защищает модель от тривиального выучивания зависимости «калории ≈ масса» и заставляет учитывать визуальную и текстовую модальности.

## Архитектура

Image -> EfficientNet-B0 -> Linear(->256) ┐
├-> concat(mass) -> MLP -> head(1)
Text -> DistilBERT -> Linear(->256) ┘


- Image encoder: `efficientnet_b0` (timm, pretrained)
- Text encoder: `distilbert-base-uncased`
- Fusion: конкатенация эмбеддингов изображения, текста и нормализованной массы
- Head: `Linear(256, 1)` без функции активации (задача регрессии)
- Loss: `L1Loss` (напрямую оптимизирует MAE)
- Оптимизатор: AdamW с раздельными learning rate для энкодеров и классификатора
- Mixed precision (AMP)

## Обучение

- 20 эпох, batch_size = 4, GPU Tesla T4
- Optimizer: AdamW
  - text_encoder: 1e-5
  - image_encoder: 1e-4
  - classifier / projection layers: 1e-3
- Scheduler: ReduceLROnPlateau
- Аугментации: RandomResizedCrop, HorizontalFlip, ColorJitter
- Seed = 42 для воспроизводимости

## Воспроизведение

Установка зависимостей:

```bash
pip install -r requirements.txt

Данные должны быть размещены в папке data/ в следующей структуре:

data/dish.csv
data/ingredients.csv
data/images/{dish_id}/rgb.png

Запуск обучения выполняется в notebook.ipynb:

from scripts.config import Config
from scripts.dataset import DishDataset, get_train_transforms, get_valid_transforms
from scripts.utils import train

cfg = Config()
# подготовка датасетов
model, history = train(cfg, train_ds, valid_ds)

from scripts.config import Config
from scripts.dataset import DishDataset, get_train_transforms, get_valid_transforms
from scripts.utils import train

cfg = Config()

Чекпоинт лучшей эпохи сохраняется по пути из cfg.MODEL_SAVE_PATH.
Инференс на тестовой выборке выполняется функцией evaluate_on_test.

Baseline-модели
Модель	MAE, ккал
Константа (медиана train)	~167
Масса (LinearRegression)	~104
Текст (TF-IDF + Ridge)	~106
Масса + текст	~67
Мультимодальная DL-модель	31.71

Анализ ошибок
Для пяти объектов с максимальной абсолютной ошибкой в ноутбуке приведены фотографии, списки ингредиентов, истинные и предсказанные значения калорийности. Основные причины крупных ошибок:

- нетипичный размер порции при схожем визуальном виде блюда;

- редкие или смешанные ингредиенты, отсутствовавшие в train;

- высокая доля скрытых калорий (соусы, масла, орехи), которые плохо различимы на фото.

Подробный разбор и гипотезы по улучшению приведены в ноутбуке (раздел 6).

## Структура репозитория

├── data/                # датасет (не коммитится, см. .gitignore)
├── scripts/
│   ├── __init__.py
│   ├── config.py        # гиперпараметры и пути
│   ├── dataset.py       # DishDataset, аугментации, DataLoader
│   └── utils.py         # модель, train(), evaluate_on_test()
├── notebook.ipynb       # EDA, обучение, анализ ошибок
├── requirements.txt
├── .gitignore
└── README.md
Окружение
Python 3.10+

PyTorch 2.x, CUDA (для обучения)

timm, transformers, albumentations, torchmetrics
