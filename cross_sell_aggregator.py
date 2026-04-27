#!/usr/bin/env python3
"""
smartkram Cross-Sell Aggregator (Apriori-basiert)

Liest alle abgeschlossenen Bestellungen der letzten 24 Monate aus shop_live,
berechnet Co-Occurrence-Pairs (A,B) → Support, Confidence, Lift,
schreibt Top-Empfehlungen pro Produkt in wp_smartkram_cross_sell.

Tabelle:
  CREATE TABLE wp_smartkram_cross_sell (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    product_id BIGINT NOT NULL,
    related_id BIGINT NOT NULL,
    support INT,         -- # Bestellungen wo BEIDE drin
    confidence FLOAT,    -- P(B|A) = co_count / count_a
    lift FLOAT,          -- confidence / P(B)
    score FLOAT,         -- combined ranking score
    last_updated DATETIME,
    UNIQUE KEY pid_rid (product_id, related_id),
    KEY pid_score (product_id, score DESC)
  );

Schwellwerte (Standard):
  - min_support = 3   (mind. 3 gemeinsame Bestellungen)
  - min_confidence = 0.05  (5% derer die A kauften, kauften auch B)
  - min_lift = 1.5     (mind. 1.5x häufiger als Zufall)

Ranking-Score = log(support) * confidence * lift
  → priorisiert: starke Korrelationen mit ausreichend Daten

Speichert pro Produkt die TOP_N=20 besten Empfehlungen.
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
from math import log

import pymysql

def _build_db_config():
    cfg = {
        "user": os.environ.get("DB_USER", "shop_live"),
        "password": os.environ.get("DB_PASSWORD", ""),
        "database": os.environ.get("DB_NAME", "shop_live"),
        "charset": "utf8mb4",
    }
    host = os.environ.get("DB_HOST", "")
    if host.startswith("/"):
        cfg["unix_socket"] = host
    elif host:
        cfg["host"] = host
        cfg["port"] = int(os.environ.get("DB_PORT", "3306"))
    else:
        cfg["unix_socket"] = "/run/mysqld/mysqld.sock"
    return cfg

DB = _build_db_config()
PREFIX = "VYkNzBJ_"
TOP_N_PER_PRODUCT = 20
MIN_SUPPORT = 3
MIN_CONFIDENCE = 0.05
MIN_LIFT = 1.5


def ensure_table(conn):
    sql = f"""
        CREATE TABLE IF NOT EXISTS {PREFIX}smartkram_cross_sell (
            id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            product_id BIGINT UNSIGNED NOT NULL,
            related_id BIGINT UNSIGNED NOT NULL,
            support INT UNSIGNED NOT NULL,
            confidence FLOAT NOT NULL,
            lift FLOAT NOT NULL,
            score FLOAT NOT NULL,
            last_updated DATETIME NOT NULL,
            UNIQUE KEY pid_rid (product_id, related_id),
            KEY pid_score (product_id, score DESC),
            KEY rid (related_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def fetch_order_baskets(conn, since_date: datetime, limit_orders: int = 0):
    """Yield list of product_ids per completed order in time window."""
    # WooCommerce uses CPT for orders (legacy) or wc_order_stats (HPOS).
    # Try posts-based first.
    # HPOS (High-Performance Order Storage)
    sql = f"""
        SELECT o.id AS order_id, GROUP_CONCAT(oim.meta_value) AS product_ids
        FROM {PREFIX}wc_orders o
        INNER JOIN {PREFIX}woocommerce_order_items oi
            ON oi.order_id = o.id AND oi.order_item_type = 'line_item'
        INNER JOIN {PREFIX}woocommerce_order_itemmeta oim
            ON oim.order_item_id = oi.order_item_id AND oim.meta_key = '_product_id'
        WHERE o.type = 'shop_order'
          AND o.status IN ('wc-completed','wc-processing','wc-on-hold','completed','processing','on-hold')
          AND o.date_created_gmt >= %s
        GROUP BY o.id
    """
    if limit_orders:
        sql += f" LIMIT {int(limit_orders)}"
    args = (since_date.strftime("%Y-%m-%d %H:%M:%S"),)
    with conn.cursor() as cur:
        cur.execute(sql, args)
        for order_id, product_ids in cur.fetchall():
            if not product_ids:
                continue
            ids = [int(p) for p in product_ids.split(",") if p.isdigit() and int(p) > 0]
            ids = list(set(ids))  # dedupe within basket
            if len(ids) >= 2:  # Need at least 2 for pairs
                yield order_id, ids


def aggregate(conn, since_date: datetime, limit_orders: int = 0):
    """Compute pair counts + single-product order counts."""
    pair_counts = defaultdict(int)  # (a,b) sorted → count
    single_counts = defaultdict(int)  # product_id → count
    total_orders = 0

    print(f"[scan] orders since {since_date.isoformat()}", file=sys.stderr)
    t0 = time.time()
    for order_id, ids in fetch_order_baskets(conn, since_date, limit_orders):
        total_orders += 1
        for pid in ids:
            single_counts[pid] += 1
        for a, b in combinations(sorted(ids), 2):
            pair_counts[(a, b)] += 1
        if total_orders % 5000 == 0:
            print(f"[scan] {total_orders} orders, {len(pair_counts)} pairs ({time.time()-t0:.0f}s)", file=sys.stderr)

    print(f"[scan] DONE: {total_orders} orders, {len(single_counts)} products, {len(pair_counts)} pairs", file=sys.stderr)
    return total_orders, single_counts, pair_counts


def compute_recommendations(total_orders, single_counts, pair_counts):
    """For each product, compute top-N recommendations (related products)."""
    # Pre-compute P(B) for lift
    p_b = {pid: cnt / total_orders for pid, cnt in single_counts.items()}

    # Build per-product recommendation list (asymmetric, so each (A,B) generates two recs)
    recs = defaultdict(list)
    skipped = {"low_support": 0, "low_confidence": 0, "low_lift": 0, "missing_b": 0}

    for (a, b), pair_cnt in pair_counts.items():
        if pair_cnt < MIN_SUPPORT:
            skipped["low_support"] += 1
            continue

        # A → B
        cnt_a = single_counts[a]
        cnt_b = single_counts[b]
        if cnt_b == 0 or cnt_a == 0:
            skipped["missing_b"] += 1
            continue

        conf_a_to_b = pair_cnt / cnt_a
        lift_a_to_b = conf_a_to_b / p_b[b]
        if lift_a_to_b < MIN_LIFT or conf_a_to_b < MIN_CONFIDENCE:
            skipped["low_lift" if lift_a_to_b < MIN_LIFT else "low_confidence"] += 1
        else:
            score = log(pair_cnt + 1) * conf_a_to_b * lift_a_to_b
            recs[a].append((b, pair_cnt, conf_a_to_b, lift_a_to_b, score))

        # B → A (symmetric pair, asymmetric recommendation)
        conf_b_to_a = pair_cnt / cnt_b
        lift_b_to_a = conf_b_to_a / p_b[a]
        if lift_b_to_a < MIN_LIFT or conf_b_to_a < MIN_CONFIDENCE:
            continue
        score = log(pair_cnt + 1) * conf_b_to_a * lift_b_to_a
        recs[b].append((a, pair_cnt, conf_b_to_a, lift_b_to_a, score))

    # Trim to top N per product
    trimmed = {}
    for pid, lst in recs.items():
        lst.sort(key=lambda x: x[4], reverse=True)
        trimmed[pid] = lst[:TOP_N_PER_PRODUCT]

    print(f"[rec] {len(trimmed)} products with recommendations | skipped: {skipped}", file=sys.stderr)
    return trimmed


def write_recommendations(conn, recs):
    """Truncate + bulk insert."""
    table = f"{PREFIX}smartkram_cross_sell"
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE {table}")
    conn.commit()

    rows = []
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    for pid, lst in recs.items():
        for rid, support, conf, lift, score in lst:
            rows.append((int(pid), int(rid), int(support), float(conf), float(lift), float(score), now))

    print(f"[write] inserting {len(rows)} rows", file=sys.stderr)
    BATCH = 1000
    sql = f"INSERT INTO {table} (product_id, related_id, support, confidence, lift, score, last_updated) VALUES (%s,%s,%s,%s,%s,%s,%s)"
    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH):
            cur.executemany(sql, rows[i:i + BATCH])
            conn.commit()
            if (i + BATCH) % 10000 == 0:
                print(f"[write] {i+BATCH}/{len(rows)}", file=sys.stderr)
    print(f"[write] DONE: {len(rows)} rows", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--months", type=int, default=24, help="History window")
    p.add_argument("--limit-orders", type=int, default=0, help="Debug: limit orders")
    args = p.parse_args()

    conn = pymysql.connect(**DB)
    ensure_table(conn)

    since = datetime.utcnow() - timedelta(days=args.months * 30)
    total, singles, pairs = aggregate(conn, since, args.limit_orders)
    if total == 0:
        print("No orders found", file=sys.stderr)
        return 1

    recs = compute_recommendations(total, singles, pairs)
    write_recommendations(conn, recs)
    print("[done]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
