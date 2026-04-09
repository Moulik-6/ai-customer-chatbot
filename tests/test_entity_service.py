from chatbot.services.entity_service import extract_order_number


def test_extract_order_number_split_date_variant():
    assert extract_order_number("track ord-2026-0409-a80939") == "ORD-20260409-A80939"


def test_extract_order_number_colon_variant():
    assert extract_order_number("track order ORD:20260409:A80939") == "ORD-20260409-A80939"


def test_extract_order_number_compact_variant():
    assert extract_order_number("track ORD20260409A80939") == "ORD-20260409-A80939"


def test_extract_order_number_loose_numeric_variant():
    assert extract_order_number("where is order 001") == "ORD-001"
