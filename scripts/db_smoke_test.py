#!/usr/bin/env python3
"""
DB smoke test — hit /api/products and /api/orders and print results.

Usage:
    BASE_URL=https://seyo009-ai-customer-chatbot.hf.space python scripts/db_smoke_test.py

    # With an admin API key (needed for /api/admin/db_status):
    BASE_URL=https://... ADMIN_API_KEY=your-key python scripts/db_smoke_test.py
"""
import json
import os
import sys
try:
    import urllib.request as urlrequest
    import urllib.error as urlerror
except ImportError:
    print("ERROR: urllib not available (Python 3.x required)")
    sys.exit(1)

BASE_URL = os.environ.get("BASE_URL", "http://localhost:7860").rstrip("/")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")


def _get(path, headers=None):
    url = BASE_URL + path
    req = urlrequest.Request(url, headers=headers or {})
    try:
        with urlrequest.urlopen(req, timeout=10) as resp:
            body = resp.read().decode()
            return resp.status, json.loads(body)
    except urlerror.HTTPError as e:
        body = e.read().decode()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"raw": body}
    except Exception as exc:
        return None, {"error": str(exc)}


def _print_result(label, status, data):
    ok = status == 200
    symbol = "✅" if ok else "❌"
    print(f"\n{symbol}  {label}  (HTTP {status})")
    if isinstance(data, dict):
        for k, v in list(data.items())[:8]:
            print(f"   {k}: {v}")
    elif isinstance(data, list):
        print(f"   {len(data)} item(s) returned")
        if data:
            print(f"   first item keys: {list(data[0].keys()) if isinstance(data[0], dict) else data[0]}")
    else:
        print(f"   {str(data)[:200]}")


def main():
    print(f"DB Smoke Test — target: {BASE_URL}")
    print("=" * 60)

    # 1. Health check
    status, data = _get("/health")
    _print_result("GET /health", status, data)

    # 2. Products
    status, data = _get("/api/products")
    products = data.get("products", data) if isinstance(data, dict) else data
    _print_result("GET /api/products", status, data)

    # 3. Products search
    status, data = _get("/api/products?search=iphone")
    _print_result("GET /api/products?search=iphone", status, data)

    # 4. Orders
    status, data = _get("/api/orders")
    _print_result("GET /api/orders", status, data)

    # 5. DB status (admin endpoint — skip if no key)
    if ADMIN_API_KEY:
        status, data = _get("/api/admin/db_status", headers={"X-API-Key": ADMIN_API_KEY})
        _print_result("GET /api/admin/db_status", status, data)
    else:
        print("\n⚠️   Skipping GET /api/admin/db_status — set ADMIN_API_KEY env var to include it")

    # 6. Chat: order lookup
    payload = json.dumps({"message": "where is order ORD-001", "session_id": "smoke-test"}).encode()
    req = urlrequest.Request(
        BASE_URL + "/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=15) as resp:
            status, chat_data = resp.status, json.loads(resp.read().decode())
    except urlerror.HTTPError as e:
        status, chat_data = e.code, {"error": e.read().decode()}
    except Exception as exc:
        status, chat_data = None, {"error": str(exc)}
    _print_result("POST /api/chat (order lookup ORD-001)", status, chat_data)

    print("\n" + "=" * 60)
    print("Smoke test complete.")


if __name__ == "__main__":
    main()
