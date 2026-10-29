# Рекомендательная система фильмов

Проект разработан в рамках учебной дисциплины «Машинное Обучение». Включает в себя программный интерфейс (FastAPI REST API), хранилище данных на MySQL, графический веб-интерфейс (Streamlit) и аналитический модуль с диагностическими графиками.

---

## Содержание
1. [Структура проекта](#структура-проекта)
2. [Реализация требований](#реализация-требований)
3. [Установка и запуск](#установка-и-запуск)

---

## Структура проекта

```text
movie_recommender/
│
├── run.py                          # Лаунчер (seed, api, app, test, all)
├── requirements.txt                # Зависимости Python
├── .env                            # Параметры подключения к MySQL
├── README.md                       # Документация
│
├── database/                       # Слой данных (MySQL + SQLAlchemy)
│   ├── models.py                   # Модели User, Movie, Rating
│   ├── connection.py               # Сессии SQLAlchemy
│   ├── seed_data.py                # Генератор данных (30 фильмов, пользователи, оценки)
│   └── schema.sql                  # DDL-скрипт для MySQL
│
├── ml_engine/                      # Алгоритмы рекомендаций
│   ├── popularity.py               # 1. Популярные фильмы (взвешенный рейтинг)
│   ├── collaborative.py            # 2. Коллаборативная фильтрация (User-Based и Item-Based)
│   ├── content_based.py            # 3. Фильтрация по контенту (TF-IDF)
│   ├── personal_hybrid.py          # 4. Персональные рекомендации (SVD + контент)
│   └── evaluation.py               # Оценка моделей (RMSE, MAE, Scree Plot, PCA, Sparsity)
│
├── api/                            # Backend на FastAPI
│   ├── main.py                     # Входная точка FastAPI, Swagger, CORS
│   ├── schemas.py                  # Pydantic схемы
│   └── routers/
│       ├── recommendations.py      # Эндпоинты рекомендаций
│       ├── movies.py               # Каталог, поиск и фильтрация фильмов
│       ├── ratings.py              # Оценка фильмов (сохранение в MySQL)
│       └── analytics.py            # Данные для графиков
│
├── app/                            # Frontend на Streamlit
│   ├── streamlit_app.py            # Интерфейс приложения
│   ├── ml_visualizations.py        # Графики и диаграммы
│   └── components/
│       └── movie_card.py           # Карточки фильмов
│
└── tests/
    ├── test_recommenders.py        # Модульные тесты алгоритмов
    └── test_api_endpoints.py       # Тесты эндпоинтов API
```

---

## Реализация требований

### 4.1 Разработка API
- 1. Топ-10 популярных фильмов: `GET /api/recommendations/popular?limit=10&genre=...`
- 2. Коллаборативная фильтрация:
  - По пользователям (User-Based): `GET /api/recommendations/collaborative/user-based?user_id=1&limit=10`
  - По фильмам (Item-Based): `GET /api/recommendations/collaborative/item-based?movie_id=1&limit=10`
- 3. Рекомендации по контенту: `GET /api/recommendations/content-based?movie_id=1&limit=10`
- 4. Персональные рекомендации для пользователя: `GET /api/recommendations/personal?user_id=1&limit=10&alpha=0.65`

### 4.2 Разработка Приложения (Streamlit + MySQL)
- Графический интерфейс на Streamlit.
- Хранение данных в MySQL:
  - Таблицы `users`, `movies`, `ratings`.
  - Возможность выбрать фильм и выставить оценку (от 1.0 до 5.0).
  - Результаты сохраняются в MySQL.
  - Отображение истории оценок выбранного пользователя.
- Справка по командам и параметрам.
- Графики и аналитика:
  - 1. Разреженность матрицы взаимодействий и сводка.
  - 2. Распределение оценок.
  - 3. Фильмы по жанрам.
  - 4. Сравнение метрик моделей (RMSE, MAE, Precision, Recall).
  - 5. График объясненной дисперсии (Scree Plot).
  - 6. 2D проекция фильмов (PCA).
  - 7. Матрица сходства фильмов.

---

## Установка и запуск

### 1. Настройка окружения
В файле `.env` укажите параметры доступа к серверу MySQL:
```ini
DB_USER=root
DB_PASSWORD=1234
DB_HOST=localhost
DB_PORT=3306
DB_NAME=movie_recommender_db
```

### 2. Запуск приложения
```bash
python run.py all
```

Или по отдельности:
```bash
# Инициализация БД
python run.py seed

# Запуск FastAPI (порт 8000)
python run.py api

# Запуск Streamlit (порт 8501)
python run.py app

# Запуск тестов
python run.py test
```

- Веб-интерфейс Streamlit: http://localhost:8501
- Документация Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
