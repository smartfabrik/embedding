#!/usr/bin/env python3
"""
smartkram Embedding-Search-API

FastAPI Service. Endpoint:
  GET /search?q=<query>&limit=20&brand=<slug>&min_price=&max_price=&in_stock=true

Antwort:
  {
    "hits": [
      {"post_id": ..., "score": 0.87, "title": "...", "brand": "...", ...},
      ...
    ],
    "took_ms": 32
  }

Hybrid mit BM25 (Relevanssi auf Plesk) wird im WP-Plugin via RRF gemerged.
Diese API liefert nur Vector-Hits.
"""

import os
import time
from typing import List, Optional

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant.vector-search.svc.cluster.local:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
SERVICE_API_KEY = os.environ.get("SERVICE_API_KEY", "smartkram-vector-2026-secure-token-changeme")
EMBEDDING_MODEL = "text-embedding-3-large"
COLLECTION = "smartkram_products"

app = FastAPI(title="smartkram Embedding Search API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
oac = OpenAI(api_key=OPENAI_API_KEY)


class Hit(BaseModel):
    post_id: int
    score: float
    title: str
    brand: str
    brand_slug: str
    sku: str
    mpn: str
    gtin: str
    price: float
    stock_status: str
    categories: str


class SearchResponse(BaseModel):
    hits: List[Hit]
    took_ms: int
    query: str


def auth(api_key: str):
    if not SERVICE_API_KEY:
        return
    if api_key != SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")


@app.get("/health")
def health():
    try:
        info = qc.get_collection(COLLECTION)
        return {"ok": True, "points": info.points_count}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(20, ge=1, le=100),
    brand: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    in_stock: Optional[bool] = None,
    api_key: str = Query(..., alias="api_key"),
):
    auth(api_key)
    t0 = time.time()

    # Embed query
    embed_resp = oac.embeddings.create(model=EMBEDDING_MODEL, input=[q])
    query_vec = embed_resp.data[0].embedding

    # Build filter
    must = []
    if brand:
        must.append(qm.FieldCondition(key="brand_slug", match=qm.MatchValue(value=brand)))
    if in_stock:
        must.append(qm.FieldCondition(key="stock_status", match=qm.MatchValue(value="instock")))
    if min_price is not None or max_price is not None:
        must.append(qm.FieldCondition(
            key="price",
            range=qm.Range(
                gte=min_price if min_price is not None else None,
                lte=max_price if max_price is not None else None,
            ),
        ))
    qfilter = qm.Filter(must=must) if must else None

    res = qc.query_points(
        collection_name=COLLECTION,
        query=query_vec,
        limit=limit,
        with_payload=True,
        query_filter=qfilter,
    ).points

    hits = []
    for p in res:
        pl = p.payload or {}
        hits.append(Hit(
            post_id=int(p.id),
            score=float(p.score),
            title=pl.get("title", ""),
            brand=pl.get("brand", ""),
            brand_slug=pl.get("brand_slug", ""),
            sku=pl.get("sku", ""),
            mpn=pl.get("mpn", ""),
            gtin=pl.get("gtin", ""),
            price=float(pl.get("price", 0)),
            stock_status=pl.get("stock_status", ""),
            categories=pl.get("categories", ""),
        ))

    return SearchResponse(
        hits=hits,
        took_ms=int((time.time() - t0) * 1000),
        query=q,
    )


@app.get("/similar/{post_id}", response_model=SearchResponse)
def similar(post_id: int, limit: int = Query(10, ge=1, le=50), api_key: str = Query(...)):
    """Get similar products (used by Cross-Sell / Verwandte-Artikel)."""
    auth(api_key)
    t0 = time.time()
    # Universal approach: retrieve source vector, then nearest-neighbor search
    source = qc.retrieve(
        collection_name=COLLECTION,
        ids=[post_id],
        with_vectors=True,
        with_payload=False,
    )
    if not source:
        return SearchResponse(hits=[], took_ms=int((time.time() - t0) * 1000), query=f"similar:{post_id}")
    source_vec = source[0].vector
    if isinstance(source_vec, dict):
        # Named vectors — pick first
        source_vec = next(iter(source_vec.values()))
    res = qc.query_points(
        collection_name=COLLECTION,
        query=source_vec,
        limit=limit + 1,
        with_payload=True,
    ).points
    hits = []
    for p in res:
        if int(p.id) == post_id:
            continue
        pl = p.payload or {}
        hits.append(Hit(
            post_id=int(p.id),
            score=float(p.score),
            title=pl.get("title", ""),
            brand=pl.get("brand", ""),
            brand_slug=pl.get("brand_slug", ""),
            sku=pl.get("sku", ""),
            mpn=pl.get("mpn", ""),
            gtin=pl.get("gtin", ""),
            price=float(pl.get("price", 0)),
            stock_status=pl.get("stock_status", ""),
            categories=pl.get("categories", ""),
        ))
        if len(hits) >= limit:
            break
    return SearchResponse(hits=hits, took_ms=int((time.time() - t0) * 1000), query=f"similar:{post_id}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
