#!/usr/bin/env python3
"""Comprehensive test suite for the live HF Space backend."""
import json
import time
import urllib.request
import urllib.error
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://seyo009-ai-customer-chatbot.hf.space"
CHAT = f"{BASE}/api/chat"
FEEDBACK = f"{BASE}/api/feedback"

passed = 0
failed = 0
errors = []


def post(url, data, timeout=20):
    """POST JSON, return (status_code, parsed_json)."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return 0, {"error": str(e)}


def get(url, timeout=15):
    """GET request, return (status_code, parsed_json)."""
    try:
        resp = urllib.request.urlopen(url, timeout=timeout)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return 0, {"error": str(e)}


def check(name, condition, detail=""):
    global passed, failed, errors
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
        errors.append(f"  {name}: {detail}")
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))


def chat(msg, **kwargs):
    payload = {"message": msg}
    payload.update(kwargs)
    return post(CHAT, payload)


# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("1. HEALTH CHECK")
print("=" * 60)
code, data = get(f"{BASE}/health")
check("health returns 200", code == 200)
check("status is healthy", data.get("status") == "healthy")
check("intents_count >= 29", data.get("intents_count", 0) >= 29, f"got {data.get('intents_count')}")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("2. INPUT VALIDATION")
print("=" * 60)

code, data = post(CHAT, "not json string")
check("invalid JSON -> 400", code == 400, f"got {code}")

code, data = post(CHAT, {})
check("empty body -> 400", code == 400, f"got {code}")

code, data = chat("")
check("empty message -> 400 EMPTY_MESSAGE", code == 400 and data.get("code") == "EMPTY_MESSAGE", f"{code} {data.get('code')}")

code, data = chat("   ")
check("whitespace-only -> 400", code == 400, f"got {code}")

code, data = chat("a" * 2001)
check("2001 chars -> 400 MESSAGE_TOO_LONG", code == 400 and data.get("code") == "MESSAGE_TOO_LONG", f"{code} {data.get('code')}")

code, data = chat("a" * 2000)
check("2000 chars -> 200 (at limit)", code == 200, f"got {code}")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("3. INPUT SANITIZATION")
print("=" * 60)

code, data = chat("<script>alert('xss')</script>hello")
check("HTML tags stripped", "<script>" not in data.get("message", ""), f"message={data.get('message','')[:60]}")
check("greeting intent still matched", data.get("intent") == "greeting", f"intent={data.get('intent')}")

code, data = chat('<img onerror="alert(1)" src=x>test')
check("img tag stripped", "<img" not in data.get("message", ""), f"message={data.get('message','')[:60]}")

code, data = chat("normal message & special <chars>")
check("sanitized normal msg succeeds", code == 200 and data.get("success"), f"code={code}")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("4. INTENT MATCHING (17 intents)")
print("=" * 60)

intent_tests = [
    ("greeting",        "hello there",                          "greeting"),
    ("goodbye",         "goodbye see you later",                "goodbye"),
    ("thanks",          "thanks a lot",                         "thanks"),
    ("help",            "help me please",                       "help"),
    ("shipping",        "how long does shipping take",          "shipping"),
    ("intl shipping",   "do you ship internationally",          "shipping"),
    ("returns",         "I want to return an item",             "returns"),
    ("pricing",         "how much does this product cost",      ["pricing", "product_info"]),
    ("order_tracking",  "track my order",                       ["order_tracking", "order_status"]),
    ("order_status",    "where is my order",                    "order_status"),
    ("product_info",    "what products do you sell",            "product_info"),
    ("stock",           "is this product in stock",             "stock_availability"),
    ("payment",         "what payment methods do you accept",   ["payment", "payment_methods"]),
    ("complaint",       "I want to file a complaint",           "complaint"),
    ("live_agent",      "can I speak to a human",               "live_agent"),
    ("hours_location",  "what are your business hours",         "hours_location"),
    ("warranty",        "what is the warranty policy",          "warranty"),
]

for label, msg, expected in intent_tests:
    code, data = chat(msg)
    actual = data.get("intent", "NONE")
    # Accept list of valid intents
    if isinstance(expected, list):
        match = actual in expected
        exp_str = "|".join(expected)
    else:
        match = actual == expected
        exp_str = expected
    check(f"{label}: '{msg[:35]}'", match, f"expected={exp_str}, got={actual}")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("5. PRODUCT & ORDER LOOKUPS")
print("=" * 60)

code, data = chat("products")
check("bare 'products' -> product list", code == 200 and data.get("success"), f"type={data.get('type')}")

code, data = chat("electronics")
check("category 'electronics' search", code == 200 and data.get("success"), f"type={data.get('type')}")

code, data = chat("show me trending products")
check("trending products", code == 200 and data.get("success"), f"intent={data.get('intent')}")

code, data = chat("where is order ORD-001")
check("order ORD-001 lookup", code == 200, f"type={data.get('type')}")

code, data = chat("order ORD-9999-999")
check("nonexistent order", code == 200 and "couldn't find" in data.get("response", "").lower(), f"resp={data.get('response','')[:60]}")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("6. FEEDBACK ENDPOINT")
print("=" * 60)

code, data = post(FEEDBACK, {"rating": "up", "session_id": "test-123"})
check("feedback up -> 200", code == 200 and data.get("success"), f"code={code}, data={data}")

code, data = post(FEEDBACK, {"rating": "down", "session_id": "test-123"})
check("feedback down -> 200", code == 200 and data.get("success"), f"code={code}, data={data}")

code, data = post(FEEDBACK, {"rating": "invalid"})
check("invalid rating -> 400", code == 400, f"code={code}")

code, data = post(FEEDBACK, {})
check("no rating -> 400", code == 400, f"code={code}")

code, data = post(FEEDBACK, "not json")
check("bad JSON -> 400", code == 400, f"code={code}")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("7. SESSION & CONTEXT")
print("=" * 60)

sid = "test-session-" + str(int(time.time()))

code, data = chat("hi", session_id=sid)
check("session msg 1 accepted", code == 200 and data.get("success"), f"code={code}")

code, data = chat("tell me about shipping", session_id=sid)
check("session msg 2 (shipping)", code == 200 and data.get("intent") == "shipping", f"intent={data.get('intent')}")

code, data = chat("and what about returns?", session_id=sid)
check("session msg 3 (follow-up)", code == 200 and data.get("success"), f"intent={data.get('intent')}")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("8. RESPONSE FORMAT VALIDATION")
print("=" * 60)

code, data = chat("hello")
check("has 'success' field", "success" in data)
check("has 'type' field", "type" in data)
check("has 'intent' field", "intent" in data)
check("has 'message' field", "message" in data)
check("has 'response' field", "response" in data)
check("has 'model' field", "model" in data)
check("response is non-empty string", isinstance(data.get("response"), str) and len(data.get("response", "")) > 0)

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("9. AI FALLBACK")
print("=" * 60)

code, data = chat("quantum entanglement effects on supply chains")
check("unknown topic -> 200", code == 200 and data.get("success"), f"code={code}")
check("fallback responds", len(data.get("response", "")) > 10, f"response={data.get('response','')[:60]}")

# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("10. REDIRECT TESTS")
print("=" * 60)

try:
    req = urllib.request.Request(BASE, method="GET")
    # Don't follow redirects
    import http.client
    opener = urllib.request.build_opener(urllib.request.HTTPHandler)
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise urllib.error.HTTPError(newurl, code, msg, headers, fp)
    opener = urllib.request.build_opener(NoRedirect)
    try:
        resp = opener.open(BASE)
        content_type = resp.headers.get("Content-Type", "")
        check("/ serves UI (200 HTML)", resp.status == 200 and "text/html" in content_type.lower(), f"status={resp.status}, content_type={content_type}")
    except urllib.error.HTTPError as e:
        check("/ redirects (allowed)", e.code in (301, 302, 307, 308), f"code={e.code}")
except Exception as e:
    check("/ redirect test", False, str(e))


# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
total = passed + failed
print("\n" + "=" * 60)
print(f"RESULTS: {passed}/{total} passed, {failed} failed")
print("=" * 60)
if errors:
    print("\nFailed tests:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
