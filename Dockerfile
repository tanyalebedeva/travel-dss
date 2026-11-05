# Образ прототипа СППР: интерфейс ЛПР на Streamlit и все модули tdss/.
#
#   docker build -t travel-dss .
#   docker run --rm -p 8501:8501 travel-dss          # http://localhost:8501
#
# Java для ризонера Pellet не ставится по умолчанию: без неё база знаний работает
# на эквивалентном выводе на Python с тем же результатом, а образ меньше примерно на 200 МБ.
# Сборка с Pellet:  docker build --build-arg WITH_JAVA=1 -t travel-dss .

FROM python:3.11-slim

ARG WITH_JAVA=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=300 \
    MPLBACKEND=Agg \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN if [ "$WITH_JAVA" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

# Зависимости ставятся отдельным слоем: при правке кода они не перекачиваются
COPY requirements.txt .
RUN pip install --retries 10 -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 dss && chown -R dss:dss /app
USER dss

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
