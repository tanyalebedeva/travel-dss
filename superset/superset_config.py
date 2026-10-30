# Конфигурация Superset для СППР (используется в docker-compose)
import os

SECRET_KEY = os.environ.get("SUPERSET_SECRET_KEY", "tdss-local-dev-key-change-me")
# Витрины СППР лежат в SQLite-файле /app/tdss_data/tdss.db - разрешаем такое подключение
PREVENT_UNSAFE_DB_CONNECTIONS = False
BABEL_DEFAULT_LOCALE = "ru"
LANGUAGES = {"ru": {"flag": "ru", "name": "Русский"}, "en": {"flag": "us", "name": "English"}}
FEATURE_FLAGS = {"DASHBOARD_NATIVE_FILTERS": True, "DASHBOARD_CROSS_FILTERS": True}
ROW_LIMIT = 10000
# Русский формат чисел: пробел - разделитель тысяч, запятая - десятичный
D3_FORMAT = {"decimal": ",", "thousands": " ", "grouping": [3], "currency": ["", " ₽"]}
