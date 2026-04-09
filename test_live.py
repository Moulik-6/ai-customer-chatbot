#!/usr/bin/env python3
"""Comprehensive live smoke test for the deployed HF Space backend."""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.getenv("HF_SPACE_BASE", "https://seyo009-ai-customer-chatbot.hf.space")
CHAT = f"{BASE}/api/chat"
FEEDBACK = f"{BASE}/api/feedback"


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


def chat(msg, **kwargs):
    payload = {"message": msg}
    payload.update(kwargs)
    return post(CHAT, payload)


def run_check(report, name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    if condition:
        report["passed"] += 1
    else:
        report["failed"] += 1
        report["errors"].append(f"  {name}: {detail}")
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not condition else ""))


def run_smoke_test(base=BASE):
    report = {"passed": 0, "failed": 0, "errors": []}
    chat_url = f"{base}/api/chat"
    feedback_url = f"{base}/api/feedback"

    def chat_with_base(msg, **kwargs):
        payload = {"message": msg}
        payload.update(kwargs)
        return post(chat_url, payload)

    print("\n" + "=" * 60)
    print("1. HEALTH CHECK")
    print("=" * 60)
    code, data = get(f"{base}/health")
    run_check(report, "health returns 200", code == 200)
    run_check(report, "status is healthy", data.get("status") == "healthy")
    run_check(report, "intents_count >= 29", data.get("intents_count", 0) >= 29, f"got {data.get('intents_count')}")

    print("\n" + "=" * 60)
    print("2. INPUT VALIDATION")
    print("=" * 60)
    code, data = post(chat_url, "not json string")
    run_check(report, "invalid JSON -> 400", code == 400, f"got {code}")
    code, data = post(chat_url, {})
    run_check(report, "empty body -> 400", code == 400, f"got {code}")
    code, data = chat_with_base("")
    run_check(report, "empty message -> 400 EMPTY_MESSAGE", code == 400 and data.get("code") == "EMPTY_MESSAGE", f"{code} {data.get('code')}")
    code, data = chat_with_base("   ")
    run_check(report, "whitespace-only -> 400", code == 400, f"got {code}")
    code, data = chat_with_base("a" * 2001)
    run_check(report, "2001 chars -> 400 MESSAGE_TOO_LONG", code == 400 and data.get("code") == "MESSAGE_TOO_LONG", f"{code} {data.get('code')}")
    code, data = chat_with_base("a" * 2000)
    run_check(report, "2000 chars -> 200 (at limit)", code == 200, f"got {code}")

    print("\n" + "=" * 60)
    print("3. INPUT SANITIZATION")
    print("=" * 60)
    code, data = chat_with_base("<script>alert('xss')</script>hello")
    run_check(report, "HTML tags stripped", "<script>" not in data.get("message", ""), f"message={data.get('message','')[:60]}")
    run_check(report, "greeting intent still matched", data.get("intent") == "greeting", f"intent={data.get('intent')}")
    code, data = chat_with_base('<img onerror="alert(1)" src=x>test')
    run_check(report, "img tag stripped", "<img" not in data.get("message", ""), f"message={data.get('message','')[:60]}")
    code, data = chat_with_base("normal message & special <chars>")
    run_check(report, "sanitized normal msg succeeds", code == 200 and data.get("success"), f"code={code}")

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
        code, data = chat_with_base(msg)
        actual = data.get("intent", "NONE")
        if isinstance(expected, list):
            match = actual in expected
            exp_str = "|".join(expected)
        else:
            match = actual == expected
            exp_str = expected
        run_check(report, f"{label}: '{msg[:35]}'", match, f"expected={exp_str}, got={actual}")

    print("\n" + "=" * 60)
    print("5. PRODUCT & ORDER LOOKUPS")
    print("=" * 60)
    code, data = chat_with_base("products")
    run_check(report, "bare 'products' -> product list", code == 200 and data.get("success"), f"type={data.get('type')}")
    code, data = chat_with_base("electronics")
    run_check(report, "category 'electronics' search", code == 200 and data.get("success"), f"type={data.get('type')}")
    code, data = chat_with_base("show me trending products")
    run_check(report, "trending products", code == 200 and data.get("success"), f"intent={data.get('intent')}")
    code, data = chat_with_base("where is order ORD-001")
    run_check(report, "order ORD-001 lookup", code == 200, f"type={data.get('type')}")
    code, data = chat_with_base("order ORD-9999-999")
    run_check(report, "nonexistent order", code == 200 and "couldn't find" in data.get("response", "").lower(), f"resp={data.get('response','')[:60]}")

    print("\n" + "=" * 60)
    print("6. FEEDBACK ENDPOINT")
    print("=" * 60)
    code, data = post(feedback_url, {"rating": "up", "session_id": "test-123"})
    run_check(report, "feedback up -> 200", code == 200 and data.get("success"), f"code={code}, data={data}")
    code, data = post(feedback_url, {"rating": "down", "session_id": "test-123"})
    run_check(report, "feedback down -> 200", code == 200 and data.get("success"), f"code={code}, data={data}")
    code, data = post(feedback_url, {"rating": "invalid"})
    run_check(report, "invalid rating -> 400", code == 400, f"code={code}")
    code, data = post(feedback_url, {})
    run_check(report, "no rating -> 400", code == 400, f"code={code}")
    code, data = post(feedback_url, "not json")
    run_check(report, "bad JSON -> 400", code == 400, f"code={code}")

    print("\n" + "=" * 60)
    print("7. SESSION & CONTEXT")
    print("=" * 60)
    sid = "test-session-" + str(int(time.time()))
    code, data = chat_with_base("hi", session_id=sid)
    run_check(report, "session msg 1 accepted", code == 200 and data.get("success"), f"code={code}")
    code, data = chat_with_base("tell me about shipping", session_id=sid)
    run_check(report, "session msg 2 (shipping)", code == 200 and data.get("intent") == "shipping", f"intent={data.get('intent')}")
    code, data = chat_with_base("and what about returns?", session_id=sid)
    run_check(report, "session msg 3 (follow-up)", code == 200 and data.get("success"), f"intent={data.get('intent')}")

    print("\n" + "=" * 60)
    print("8. RESPONSE FORMAT VALIDATION")
    print("=" * 60)
    code, data = chat_with_base("hello")
    run_check(report, "has 'success' field", "success" in data)
    run_check(report, "has 'type' field", "type" in data)
    run_check(report, "has 'intent' field", "intent" in data)
    run_check(report, "has 'message' field", "message" in data)
    run_check(report, "has 'response' field", "response" in data)
    run_check(report, "has 'model' field", "model" in data)
    run_check(report, "response is non-empty string", isinstance(data.get("response"), str) and len(data.get("response", "")) > 0)

    print("\n" + "=" * 60)
    print("9. AI FALLBACK")
    print("=" * 60)
    code, data = chat_with_base("quantum entanglement effects on supply chains")
    run_check(report, "unknown topic -> 200", code == 200 and data.get("success"), f"code={code}")
    run_check(report, "fallback responds", len(data.get("response", "")) > 10, f"response={data.get('response','')[:60]}")

    print("\n" + "=" * 60)
    print("10. REDIRECT TESTS")
    print("=" * 60)
    try:
        req = urllib.request.Request(base, method="GET")
        opener = urllib.request.build_opener(urllib.request.HTTPHandler)

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise urllib.error.HTTPError(newurl, code, msg, headers, fp)

        opener = urllib.request.build_opener(NoRedirect)
        try:
            resp = opener.open(base)
            content_type = resp.headers.get("Content-Type", "")
            run_check(report, "/ serves UI (200 HTML)", resp.status == 200 and "text/html" in content_type.lower(), f"status={resp.status}, content_type={content_type}")
        except urllib.error.HTTPError as e:
            run_check(report, "/ redirects (allowed)", e.code in (301, 302, 307, 308), f"code={e.code}")
    except Exception as e:
        run_check(report, "/ redirect test", False, str(e))

    total = report["passed"] + report["failed"]
    print("\n" + "=" * 60)
    print(f"RESULTS: {report['passed']}/{total} passed, {report['failed']} failed")
    print("=" * 60)
    if report["errors"]:
        print("\nFailed tests:")
        for e in report["errors"]:
            print(e)
        return report

    print("\nAll tests passed!")
    return report


def main():
    report = run_smoke_test()
    sys.exit(1 if report["failed"] else 0)


if __name__ == "__main__":
    main()
