# Calorie Prediction - Multimodal Neural Network

Предсказание калорийности блюда по фотографии и списку ингредиентов.

**Целевая метрика:** MAE < 50 кКал на тестовой выборке.

## Структура репозитория

- `scripts/` - Python-модули (config, dataset, model, train)
- `notebook.ipynb` - EDA, обучение, инференс (с сохранёнными выводами)
- `data/` - датасет (не входит в репозиторий, см. `data/README.md`)
- `requirements.txt` - зависимости

## Данные

Датасет не хранится в репозитории из-за размера (1.3 ГБ).
Скачайте его из LMS и разместите в папке `data/` согласно `data/README.md`.

## Воспроизведение

1. Откройте `notebook.ipynb` в Google Colab.
2. Смонтируйте Google Drive и разместите проект в `/content/drive/MyDrive/calorie-project/`.
3. Установите зависимости: `pip install -r requirements.txt`.
4. Запустите все ячейки по порядку.

## Результаты

_Будут заполнены после обучения._

- Финальный MAE на тесте: _TBD_
- Лучшая эпоха: _TBD_