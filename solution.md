# Tracking Homework 04

## Что сделано

В проекте реализованы:
- обучение ResNet18 на `Train_1`;
- тестирование на `Test_1`;
- логирование метрик в TensorBoard;
- сохранение моделей и метрик;
- загрузка артефактов в MinIO;
- дообучение модели на `Train_2`;
- сравнение базовой и дообученной моделей;
- DVC pipeline.

## Запуск

Для установки зависимостей:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Для запуска MinIO:

```bash
docker compose up
```

Для запуска обучения:

```bash
python -m src.train
```

Для запуска DVC pipeline:

```bash
PYTHONPATH=. dvc repro
```

## Результаты

| Model | Accuracy | F1-score |
|---|---|---|
| Base | 0.8902 | 0.8899 |
| Finetuned | 0.9135 | 0.9134 |

После дообучения качество модели улучшилось.