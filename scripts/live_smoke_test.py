#!/usr/bin/env python3
"""
Live smoke test for the AI Customer Chatbot deployed on Hugging Face Spaces.

Usage:
    python scripts/live_smoke_test.py

Environment variables:
    BASE_URL  - Base URL of the deployed Space
                (default: https://seyo009-ai-customer-chatbot.hf.space)

Exit codes:
    0  - All checks passed
    1  - One or more checks failed
    2  - BASE_URL is completely unreachable
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_URL = os.environ.get(
    "BASE_URL", "https://seyo009-ai-customer-chatbot.hf.space"
).rstrip("/")

MAX_RETRIES = 3
RETRY_BACKOFF = 2  # seconds between retries (doubles each attempt)
CONNECT_TIMEOUT = 20  # seconds

# Strings that indicate the Supabase project is paused or the DB is down
_SUPABASE_PAUSED_HINTS = (
    "project paused",
    "connection failure",
    "connection refused",
    "supabase project is paused",
    "db_error",
)

# ---------------------------------------------------------------------------
# Counters / state
# ---------------------------------------------------------------------------
_passed = 0
_failed = 0
_errors: list[str] = []


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _do_request(req: urllib.request.Request) -> tuple[int, dict]:
    """Execute *req* and return ``(status_code, parsed_json)``."""
    try:
        with urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read())
        except Exception:
            body = {"error": exc.reason}
        return exc.code, body
    except urllib.error.URLError as exc:
        return 0, {"error": str(exc.reason)}
    except Exception as exc:
        return 0, {"error": str(exc)}


def _request_with_retry(
    req: urllib.request.Request,
) -> tuple[int, dict]:
    """Retry *req* up to MAX_RETRIES times with exponential backoff."""
    delay = RETRY_BACKOFF
    last_code, last_data = 0, {}
    for attempt in range(1, MAX_RETRIES + 1):
        last_code, last_data = _do_request(req)
        if last_code != 0:
            return last_code, last_data
        if attempt < MAX_RETRIES:
            print(
                f"    [retry {attempt}/{MAX_RETRIES - 1}] unreachable, "
                f"waiting {delay}s …"
            )
            time.sleep(delay)
            delay *= 2
    return last_code, last_data


def get(path: str) -> tuple[int, dict]:
    url = BASE_URL + path
    req = urllib.request.Request(url, method="GET")
    return _request_with_retry(req)


def post(path: str, payload: dict) -> tuple[int, dict]:
    url = BASE_URL + path
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    return _request_with_retry(req)


# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------
def check(name: str, condition: bool, detail: str = "") -> None:
    global _passed, _failed
    if condition:
        _passed += 1
        print(f"  [PASS] {name}")
    else:
        _failed += 1
        msg = f"  [FAIL] {name}" + (f" — {detail}" if detail else "")
        print(msg)
        _errors.append(msg)


# ---------------------------------------------------------------------------
# Supabase-paused detection
# ---------------------------------------------------------------------------
def _is_supabase_paused(code: int, data: dict) -> bool:
    """Return True if the response looks like a paused-Supabase error."""
    if data.get("code", "").upper() == "DB_ERROR":
        return True
    error_text = str(data.get("error", "") or data.get("message", "")).lower()
    return any(hint in error_text for hint in _SUPABASE_PAUSED_HINTS)


def _warn_if_supabase_paused(endpoint: str, code: int, data: dict) -> None:
    if _is_supabase_paused(code, data):
        print()
        print("  ⚠️  HINT: Supabase may be paused.")
        print(
            "  ➜  Unpause the project in the Supabase dashboard: "
            "https://supabase.com/dashboard"
        )
        print(
            f"  ➜  Endpoint {endpoint} returned "
            f"code={data.get('code')!r}, error={data.get('error')!r}"
        )
        print()


# ---------------------------------------------------------------------------
# Connectivity pre-flight
# ---------------------------------------------------------------------------
def _assert_reachable() -> None:
    """Exit with code 2 if BASE_URL is not reachable at all."""
    print(f"Checking connectivity to {BASE_URL} …")
    code, data = get("/health")
    if code == 0:
        print(f"\n✗ ERROR: {BASE_URL} is unreachable.")
        print(
            "  Make sure the Space is running and BASE_URL is set correctly."
        )
        sys.exit(2)
    print(f"  Connected (HTTP {code})\n")


# ---------------------------------------------------------------------------
# Test sections
# ---------------------------------------------------------------------------
def test_health() -> None:
    print("=" * 60)
    print("1. HEALTH CHECK")
    print("=" * 60)
    code, data = get("/health")
    check("GET /health returns 200", code == 200, f"got {code}")
    check(
        "status == 'healthy'",
        data.get("status") == "healthy",
        f"got {data.get('status')!r}",
    )


def test_chat_intent() -> None:
    print("\n" + "=" * 60)
    print("2. INTENT MATCHING (via /api/chat)")
    print("=" * 60)

    cases = [
        ("greeting", "hello there", "greeting"),
        ("shipping", "how long does shipping take", "shipping"),
        ("returns", "I want to return an item", "returns"),
    ]
    for label, msg, expected in cases:
        code, data = post("/api/chat", {"message": msg})
        actual_intent = data.get("intent", "NONE")
        check(
            f"{label}: '{msg}'",
            code == 200 and actual_intent == expected,
            f"expected intent={expected!r}, got intent={actual_intent!r}, code={code}",
        )


def test_chat_fallback() -> None:
    print("\n" + "=" * 60)
    print("3. AI FALLBACK (via /api/chat)")
    print("=" * 60)

    code, data = post(
        "/api/chat", {"message": "quantum entanglement effects on supply chains"}
    )
    check(
        "unknown topic -> 200",
        code == 200 and data.get("success"),
        f"code={code}",
    )
    check(
        "fallback returns non-empty response",
        isinstance(data.get("response"), str)
        and len(data.get("response", "")) > 0,
        f"response={data.get('response', '')[:80]!r}",
    )


def test_products() -> None:
    print("\n" + "=" * 60)
    print("4. PRODUCTS ENDPOINT (/api/products)")
    print("=" * 60)

    code, data = get("/api/products")
    _warn_if_supabase_paused("/api/products", code, data)

    check(
        "GET /api/products returns 200",
        code == 200,
        f"got {code}, body={json.dumps(data)[:120]}",
    )
    if code == 200:
        check(
            "/api/products returns a list",
            isinstance(data.get("products"), list),
            f"'products' key type={type(data.get('products')).__name__}",
        )


def test_orders() -> None:
    print("\n" + "=" * 60)
    print("5. ORDERS ENDPOINT (/api/orders)")
    print("=" * 60)

    code, data = get("/api/orders")
    _warn_if_supabase_paused("/api/orders", code, data)

    check(
        "GET /api/orders returns 200",
        code == 200,
        f"got {code}, body={json.dumps(data)[:120]}",
    )
    if code == 200:
        check(
            "/api/orders returns a list",
            isinstance(data.get("orders"), list),
            f"'orders' key type={type(data.get('orders')).__name__}",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"\nSmoke test target: {BASE_URL}")
    print(f"Retries per request: {MAX_RETRIES}, backoff: {RETRY_BACKOFF}s\n")

    _assert_reachable()

    test_health()
    test_chat_intent()
    test_chat_fallback()
    test_products()
    test_orders()

    total = _passed + _failed
    print("\n" + "=" * 60)
    print(f"RESULTS: {_passed}/{total} passed, {_failed} failed")
    print("=" * 60)

    if _errors:
        print("\nFailed checks:")
        for err in _errors:
            print(err)
        sys.exit(1)

    print("\nAll checks passed! ✓")
    sys.exit(0)


if __name__ == "__main__":
    main()
