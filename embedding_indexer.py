#!/usr/bin/env python3
"""
smartkram Embedding-Indexer

Liest Produkte aus shop_live, baut Composite-Text auf (Title + Brand + Categories
+ Description-Excerpt), generiert OpenAI-Embedding (text-embedding-3-large, 3072d),
schreibt in Qdrant. Hash-basierter Skip für unveränderte Produkte.

Modi:
  --full       Komplett-Re-Index (alle Produkte)
  --diff       Nur geänderte Produkte (default, Daily-Cron)
  --post-id N  Einzelnes Produkt (für save_post Webhook)
"""

import argparse
import hashlib
import os
import re
import sys
import time
from typing import List, Optional, Tuple

import pymysql
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# ---- Config ----
DB_HOST = os.environ.get("DB_HOST", "138.201.19.46")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))
DB_USER = os.environ.get("DB_USER", "shop_live")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "shop_live")
DB_PREFIX = os.environ.get("DB_PREFIX", "VYkNzBJ_")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant.vector-search.svc.cluster.local:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

COLLECTION_NAME = "smartkram_products"
BATCH_SIZE = 100  # OpenAI accepts batch input
EMBED_BATCH = 100  # Embeddings per OpenAI call

HTML_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def db_connect() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME, charset="utf8mb4",
    )


def qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)


def ensure_collection(qc: QdrantClient) -> None:
    cols = [c.name for c in qc.get_collections().collections]
    if COLLECTION_NAME in cols:
        return
    qc.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qm.VectorParams(
            size=EMBEDDING_DIM,
            distance=qm.Distance.COSINE,
            on_disk=True,
        ),
        hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=100, on_disk=True),
        optimizers_config=qm.OptimizersConfigDiff(memmap_threshold=20000),
        # On-disk payload to keep RAM low
        on_disk_payload=True,
    )
    # Index keyword fields for filter performance
    for field, schema in [
        ("brand_slug", qm.PayloadSchemaType.KEYWORD),
        ("stock_status", qm.PayloadSchemaType.KEYWORD),
        ("price", qm.PayloadSchemaType.FLOAT),
    ]:
        try:
            qc.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)
        except Exception:
            pass
    print(f"[init] collection {COLLECTION_NAME} created (dim={EMBEDDING_DIM}, cosine)", file=sys.stderr)


def html_to_text(s: Optional[str], max_len: int = 600) -> str:
    if not s:
        return ""
    s = HTML_RE.sub(" ", s)
    s = WS_RE.sub(" ", s).strip()
    return s[:max_len]


def fetch_products(conn, only_post_id: Optional[int] = None, since_modified: Optional[str] = None,
                   limit: int = 0) -> List[Tuple]:
    """Fetch products with brand, categories, attributes."""
    posts_t = f"{DB_PREFIX}posts"
    pm_t = f"{DB_PREFIX}postmeta"
    tr_t = f"{DB_PREFIX}term_relationships"
    tt_t = f"{DB_PREFIX}term_taxonomy"
    terms_t = f"{DB_PREFIX}terms"

    where = ["p.post_type='product'", "p.post_status='publish'"]
    args = []
    if only_post_id:
        where.append("p.ID = %s")
        args.append(only_post_id)
    if since_modified:
        where.append("p.post_modified >= %s")
        args.append(since_modified)

    sql = f"""
        SELECT p.ID, p.post_title, p.post_excerpt, p.post_content, p.post_modified,
               (SELECT meta_value FROM {pm_t} WHERE post_id=p.ID AND meta_key='_sku' LIMIT 1) AS sku,
               (SELECT meta_value FROM {pm_t} WHERE post_id=p.ID AND meta_key='_ts_mpn' LIMIT 1) AS mpn,
               (SELECT meta_value FROM {pm_t} WHERE post_id=p.ID AND meta_key='_ts_gtin' LIMIT 1) AS gtin,
               (SELECT meta_value FROM {pm_t} WHERE post_id=p.ID AND meta_key='_stock_status' LIMIT 1) AS stock_status,
               (SELECT CAST(meta_value AS DECIMAL(10,2)) FROM {pm_t} WHERE post_id=p.ID AND meta_key='_price' LIMIT 1) AS price,
               (SELECT GROUP_CONCAT(t.name ORDER BY t.name SEPARATOR ', ')
                FROM {tr_t} tr JOIN {tt_t} tt ON tt.term_taxonomy_id=tr.term_taxonomy_id JOIN {terms_t} t ON t.term_id=tt.term_id
                WHERE tr.object_id=p.ID AND tt.taxonomy='product_brand') AS brand_names,
               (SELECT t.slug FROM {tr_t} tr JOIN {tt_t} tt ON tt.term_taxonomy_id=tr.term_taxonomy_id JOIN {terms_t} t ON t.term_id=tt.term_id
                WHERE tr.object_id=p.ID AND tt.taxonomy='product_brand' LIMIT 1) AS brand_slug,
               (SELECT GROUP_CONCAT(t.name ORDER BY t.name SEPARATOR ', ')
                FROM {tr_t} tr JOIN {tt_t} tt ON tt.term_taxonomy_id=tr.term_taxonomy_id JOIN {terms_t} t ON t.term_id=tt.term_id
                WHERE tr.object_id=p.ID AND tt.taxonomy='product_cat') AS cat_names
        FROM {posts_t} p
        WHERE {' AND '.join(where)}
        ORDER BY p.ID
    """
    if limit:
        sql += f" LIMIT {int(limit)}"

    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall()


def build_text(row) -> Tuple[str, dict]:
    """Build composite text for embedding + payload metadata."""
    (pid, title, excerpt, content, modified, sku, mpn, gtin,
     stock_status, price, brand_names, brand_slug, cat_names) = row

    title = title or ""
    excerpt = html_to_text(excerpt, 400)
    content_excerpt = html_to_text(content, 400)
    desc = excerpt or content_excerpt or ""

    parts = [
        f"Titel: {title}",
        f"Marke: {brand_names}" if brand_names else "",
        f"Kategorien: {cat_names}" if cat_names else "",
        f"Artikelnummer: {mpn}" if mpn else "",
        f"GTIN: {gtin}" if gtin else "",
        f"SKU: {sku}" if sku else "",
        f"Beschreibung: {desc}" if desc else "",
    ]
    text = "\n".join(p for p in parts if p)

    payload = {
        "post_id": int(pid),
        "title": title,
        "brand": brand_names or "",
        "brand_slug": brand_slug or "",
        "categories": cat_names or "",
        "sku": sku or "",
        "mpn": mpn or "",
        "gtin": gtin or "",
        "stock_status": stock_status or "",
        "price": float(price) if price is not None else 0.0,
        "modified": modified.isoformat() if modified else "",
        "content_hash": hashlib.sha1(text.encode("utf-8")).hexdigest(),
    }
    return text, payload


def existing_hashes(qc: QdrantClient, post_ids: List[int]) -> dict:
    """Return {post_id: content_hash} for products already in Qdrant."""
    if not post_ids:
        return {}
    out = {}
    for chunk in [post_ids[i:i + 200] for i in range(0, len(post_ids), 200)]:
        res = qc.retrieve(
            collection_name=COLLECTION_NAME,
            ids=chunk,
            with_payload=["content_hash"],
            with_vectors=False,
        )
        for p in res:
            out[int(p.id)] = (p.payload or {}).get("content_hash", "")
    return out


def embed_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def index_products(conn, qc: QdrantClient, oac: OpenAI, only_post_id=None,
                   since_modified=None, full=False, limit=0):
    products = fetch_products(conn, only_post_id, since_modified, limit)
    print(f"[fetch] {len(products)} products", file=sys.stderr)
    if not products:
        return 0

    # Build texts + payloads
    items = []  # list of (post_id, text, payload)
    for row in products:
        text, payload = build_text(row)
        items.append((int(row[0]), text, payload))

    # Skip-by-hash if not full
    if not full and not only_post_id:
        existing = existing_hashes(qc, [pid for pid, _, _ in items])
        skipped = 0
        filtered = []
        for pid, text, payload in items:
            if existing.get(pid) == payload["content_hash"]:
                skipped += 1
            else:
                filtered.append((pid, text, payload))
        items = filtered
        print(f"[skip] {skipped} unchanged (hash match)", file=sys.stderr)

    if not items:
        print("[done] nothing to do", file=sys.stderr)
        return 0

    # Embed in batches
    t0 = time.time()
    n_done = 0
    for i in range(0, len(items), EMBED_BATCH):
        chunk = items[i:i + EMBED_BATCH]
        texts = [t for _, t, _ in chunk]
        try:
            embeddings = embed_batch(oac, texts)
        except Exception as e:
            print(f"[err] embedding batch failed: {e}", file=sys.stderr)
            time.sleep(2)
            continue

        # Upsert to Qdrant
        points = [
            qm.PointStruct(id=pid, vector=vec, payload=payload)
            for (pid, _, payload), vec in zip(chunk, embeddings)
        ]
        qc.upsert(collection_name=COLLECTION_NAME, points=points, wait=False)
        n_done += len(chunk)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        eta = (len(items) - n_done) / rate if rate > 0 else 0
        print(f"[batch] {n_done}/{len(items)} ({rate:.1f}/s, eta {eta:.0f}s)", file=sys.stderr)

    print(f"[done] indexed {n_done} in {time.time()-t0:.1f}s", file=sys.stderr)
    return n_done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true", help="Re-index everything (ignore hash)")
    p.add_argument("--diff", action="store_true", help="Only changed since last day (default for cron)")
    p.add_argument("--post-id", type=int, help="Single product")
    p.add_argument("--since", help="ISO date (YYYY-MM-DD HH:MM:SS) for diff mode")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY env not set", file=sys.stderr)
        sys.exit(1)
    if not QDRANT_API_KEY:
        print("ERROR: QDRANT_API_KEY env not set", file=sys.stderr)
        sys.exit(1)

    conn = db_connect()
    qc = qdrant_client()
    oac = OpenAI(api_key=OPENAI_API_KEY)
    ensure_collection(qc)

    since = None
    if args.diff and not args.since:
        since = "DATE_SUB(NOW(), INTERVAL 25 HOUR)"  # not used; see fetch_products
        from datetime import datetime, timedelta
        since = (datetime.utcnow() - timedelta(hours=25)).strftime("%Y-%m-%d %H:%M:%S")
    elif args.since:
        since = args.since

    n = index_products(conn, qc, oac,
                       only_post_id=args.post_id,
                       since_modified=since,
                       full=args.full,
                       limit=args.limit)

    info = qc.get_collection(COLLECTION_NAME)
    print(f"[stats] collection size: {info.points_count}", file=sys.stderr)
    return 0 if n >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
