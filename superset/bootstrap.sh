#!/bin/bash
# Первичная настройка Superset и импорт дашборда СППР (выполняется при каждом старте контейнера)
set -e
superset db upgrade
superset fab create-admin --username admin --firstname Admin --lastname TDSS \
  --email admin@example.com --password admin || true
superset init
superset import-dashboards -p /app/dashboard_export.zip -u admin || echo "Дашборд уже импортирован"
exec /usr/bin/run-server.sh
