"""
Entity extraction — pull order numbers, emails, SKUs, product names from messages.
"""
import re

# Precompiled patterns
_RE_ORDER_NUMBER_CANONICAL = re.compile(
    r'\bORD(?:[-\s:][A-Z0-9]{2,})+\b',
    re.IGNORECASE,
)
_RE_ORDER_NUMBER_LOOSE = re.compile(
    r'\b(?:ord|order)\s*(?:number\s*)?[-:#]?\s*(\d{3,})\b',
    re.IGNORECASE,
)
_RE_ORDER_NUMBER_COMPACT = re.compile(
    r'\bORD([0-9]{8})([A-Z0-9]{4,})\b',
    re.IGNORECASE,
)
_RE_EMAIL = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
_RE_SKU = re.compile(r'\b[A-Z]{2,}[-][A-Z0-9][-A-Z0-9]{2,}\b')
_RE_PHONE = re.compile(r'(?:\+?1[-\.\s]?)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}')


def extract_order_number(message):
    """Extract order number from either canonical or loose phrasing."""
    match = _RE_ORDER_NUMBER_CANONICAL.search(message)
    if match:
        raw = re.sub(r'[-\s:]+', '-', match.group(0).strip()).upper()
        parts = [p for p in raw.split('-') if p]

        # Normalize user-typed split date variants:
        # ORD-2026-0409-ABC123 -> ORD-20260409-ABC123
        if (
            len(parts) >= 4
            and parts[0] == 'ORD'
            and parts[1].isdigit() and len(parts[1]) == 4
            and parts[2].isdigit() and len(parts[2]) == 4
        ):
            parts = ['ORD', parts[1] + parts[2], *parts[3:]]

        return '-'.join(parts)

    compact = _RE_ORDER_NUMBER_COMPACT.search(message)
    if compact:
        date_part = compact.group(1)
        suffix_part = compact.group(2).upper()
        return f"ORD-{date_part}-{suffix_part}"

    loose = _RE_ORDER_NUMBER_LOOSE.search(message)
    return f"ORD-{loose.group(1)}" if loose else None


def extract_email(message):
    """Extract email address from message."""
    match = _RE_EMAIL.search(message)
    return match.group(0).lower() if match else None


def extract_sku(message):
    """Extract product SKU (e.g. IPHONE-15-PRO) from message."""
    match = _RE_SKU.search(message.upper())
    return match.group(0) if match else None


def extract_product_name(message):
    """Extract a product name using keyword triggers."""
    triggers = [
        r'(?:about|for|on|called|named)\s+["\']?(.{3,40}?)["\']?\s*(?:\?|$|\.)',
        r'(?:price of|cost of|details on|info on|stock of)\s+["\']?(.{3,40}?)["\']?\s*(?:\?|$|\.)',
        r'(?:do you (?:have|sell|carry))\s+["\']?(.{3,40}?)["\']?\s*(?:\?|$|\.)',
    ]
    for pattern in triggers:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('"\'')
    return None
