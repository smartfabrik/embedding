FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "openai>=1.55,<3" \
    "qdrant-client>=1.13" \
    "pymysql>=1.1" \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "pydantic>=2"

COPY embedding_indexer.py /app/embedding_indexer.py
COPY embedding_search_api.py /app/embedding_search_api.py
COPY cross_sell_aggregator.py /app/cross_sell_aggregator.py

ENV PYTHONUNBUFFERED=1

# Default: run search API; CronJob overrides command for indexer
CMD ["uvicorn", "embedding_search_api:app", "--host", "0.0.0.0", "--port", "8080"]
