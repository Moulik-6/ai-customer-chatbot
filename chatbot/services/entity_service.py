"""
Entity extraction — pull order numbers, emails, SKUs, product names from messages.
"""
import re

# Precompiled patterns
_RE_ORDER_NUMBER = re.compile(r'ORD[-\s]?\d{4}[-\s]?\d{3,4}', re.IGNORECASE)
_RE_EMAIL = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
_RE_SKU = re.compile(r'\b[A-Z]{2,}[-][A-Z0-9][-A-Z0-9]{2,}\b')
_RE_PHONE = re.compile(r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')


def extract_order_number(message):
    """Extract order number (ORD-XXXX-XXX) from message."""
    match = _RE_ORDER_NUMBER.search(message)
    return match.group(0).replace(' ', '-').upper() if match else None


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
