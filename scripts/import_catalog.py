#!/usr/bin/env python3
"""
Import organization catalog data into the chatbot's products table.

Supports:
- CSV files
- JSON files (array of objects)
- SQLite DB files (read from a source table)

This script normalizes common column names from different org schemas and
loads them into the `products` table so the chatbot can answer product/service
questions immediately.

Usage examples:
    # CSV
    python scripts/import_catalog.py --source ./acme_catalog.csv --mode upsert

    # JSON
    python scripts/import_catalog.py --source ./catalog.json --mode replace

    # SQLite
    python scripts/import_catalog.py --source ./org.db --source-table catalog_items --mode upsert
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

# Ensure project root is importable when running as: python scripts/import_catalog.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot.database import supabase


NAME_KEYS = ["name", "product_name", "title", "item_name", "service_name"]
DESC_KEYS = ["description", "details", "summary", "about"]
PRICE_KEYS = ["price", "unit_price", "cost", "amount", "rate"]
CATEGORY_KEYS = ["category", "group", "type", "department"]
SKU_KEYS = ["sku", "code", "item_code", "product_code"]
STOCK_KEYS = ["stock", "inventory", "qty", "quantity", "available_units"]
IMAGE_KEYS = ["image_url", "image", "image_link", "thumbnail", "photo_url"]
SERVICE_KEYS = ["is_service", "service", "kind", "record_type"]


def _pick_value(record: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def _normalize_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_price(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    text = str(value).strip()
    text = re.sub(r"[^0-9.\-]", "", text)
    try:
        return float(text)
    except Exception:
        return 0.0


def _normalize_stock(value: Any, default: int) -> int:
    if value in (None, ""):
        return default
    try:
        return max(0, int(float(str(value).strip())))
    except Exception:
        return default


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", value.upper()).strip("-")
    return text[:40] if text else "ITEM"


def _looks_like_service(record: Dict[str, Any], category: str) -> bool:
    service_raw = _pick_value(record, SERVICE_KEYS)
    if service_raw is not None:
        flag = str(service_raw).strip().lower()
        return flag in {"1", "true", "yes", "service", "services"}

    category_lower = category.lower()
    return any(word in category_lower for word in ("service", "plan", "subscription", "support"))


def _normalize_record(raw: Dict[str, Any], index: int) -> Dict[str, Any] | None:
    # Lower-case keys for robust cross-schema mapping.
    record = {str(k).strip().lower(): v for k, v in raw.items()}

    name = _normalize_str(_pick_value(record, NAME_KEYS))
    if not name:
        return None

    description = _normalize_str(_pick_value(record, DESC_KEYS))
    category = _normalize_str(_pick_value(record, CATEGORY_KEYS)) or "General"
    is_service = _looks_like_service(record, category)

    if is_service and "service" not in category.lower():
        category = "Services"

    price = _normalize_price(_pick_value(record, PRICE_KEYS))
    stock_default = 9999 if is_service else 0
    stock = _normalize_stock(_pick_value(record, STOCK_KEYS), stock_default)

    sku = _normalize_str(_pick_value(record, SKU_KEYS))
    if not sku:
        sku = f"AUTO-{_slug(name)}-{index}"

    image_url = _normalize_str(_pick_value(record, IMAGE_KEYS)) or None

    return {
        "name": name,
        "description": description or None,
        "price": price,
        "category": category,
        "sku": sku,
        "stock": stock,
        "image_url": image_url,
    }


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    if isinstance(payload, dict):
        items = payload.get("items") or payload.get("products") or payload.get("data")
        if isinstance(items, list):
            return [p for p in items if isinstance(p, dict)]
    raise ValueError("JSON source must be a list of objects or include items/products/data list")


def _read_sqlite(path: Path, source_table: str) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {source_table}")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def _load_records(path: Path, source_table: str | None) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".json":
        return _read_json(path)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        if not source_table:
            raise ValueError("--source-table is required for SQLite sources")
        return _read_sqlite(path, source_table)
    raise ValueError(f"Unsupported source type: {suffix}. Use .csv, .json, .db, .sqlite, or .sqlite3")


def _chunked(items: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _import_to_supabase(rows: List[Dict[str, Any]], mode: str) -> None:
    if not supabase:
        raise RuntimeError("Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY first.")

    if mode == "replace":
        # Delete all existing products before import.
        supabase.table("products").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()

    if mode in {"replace", "append"}:
        for chunk in _chunked(rows, 200):
            supabase.table("products").insert(chunk).execute()
        return

    # upsert mode: keep existing and update by SKU
    for chunk in _chunked(rows, 200):
        supabase.table("products").upsert(chunk, on_conflict="sku").execute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Import organization catalog data for chatbot auto-answering")
    parser.add_argument("--source", required=True, help="Path to source file (.csv, .json, .db/.sqlite)")
    parser.add_argument("--source-table", default=None, help="SQLite table name when --source points to a DB file")
    parser.add_argument("--mode", choices=["upsert", "append", "replace"], default="upsert", help="Import mode (default: upsert)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and normalize only; do not write to DB")
    args = parser.parse_args()

    source_path = Path(args.source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    raw_records = _load_records(source_path, args.source_table)
    if not raw_records:
        raise ValueError("No records found in source")

    normalized: List[Dict[str, Any]] = []
    skipped = 0
    for idx, raw in enumerate(raw_records, start=1):
        item = _normalize_record(raw, idx)
        if item is None:
            skipped += 1
            continue
        normalized.append(item)

    if not normalized:
        raise ValueError("All records were skipped; no valid name/title field found")

    print(f"Loaded {len(raw_records)} record(s) from {source_path.name}")
    print(f"Normalized {len(normalized)} record(s), skipped {skipped}")
    print(f"Mode: {args.mode}")

    if args.dry_run:
        print("Dry run complete. No DB changes were made.")
        print("Sample normalized row:")
        print(json.dumps(normalized[0], indent=2))
        return

    _import_to_supabase(normalized, args.mode)
    print("Import completed successfully.")
    print("The chatbot can now answer catalog/service questions from this uploaded data.")


if __name__ == "__main__":
    main()
