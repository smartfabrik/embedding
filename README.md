# smartkram Embedding Service

OpenAI-Embeddings-based Hybrid-Search backend for [smartkram.de](https://smartkram.de).

## Components

- **`embedding_indexer.py`** — Reads products from shop_live MariaDB,
  builds composite-text (Title + Brand + Categories + Description),
  generates OpenAI `text-embedding-3-large` vectors (3072d),
  upserts to Qdrant with hash-based skip for unchanged docs.
  - Modes: `--full` (initial reindex), `--diff` (default, last 25h), `--post-id N`
- **`embedding_search_api.py`** — FastAPI service with endpoints:
  - `GET /search?q=<text>&limit=20&brand=&min_price=&max_price=&in_stock=`
  - `GET /similar/{post_id}?limit=10`  — used by Cross-Sell / Verwandte Artikel
  - `GET /health`

## Deployment

K8s manifests in [smartfabrik/infrastructure](https://github.com/smartfabrik/infrastructure)
under `flux/smartkram-plesk-2/99-apps/vector-search.yaml`.

Image is published to `ghcr.io/smartfabrik/embedding:<YYYYMMDD.NN>` on every
push to `main`. Flux ImagePolicy picks up new tags automatically.

## Build locally
```bash
docker build -t smartkram-embedding:dev .
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -e QDRANT_API_KEY=... \
  smartkram-embedding:dev
```
