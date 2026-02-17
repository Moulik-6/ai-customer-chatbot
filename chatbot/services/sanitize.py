"""
Input sanitization utilities for PostgREST and other query backends.
"""
import html
import re


def sanitize_search(value: str) -> str:
    """
    Sanitize a search string for safe use in PostgREST filter expressions.

    PostgREST uses commas, dots, parentheses, and other punctuation as filter
    operators.  Strip everything except alphanumeric characters, spaces, and
    hyphens to prevent filter injection.
    """
    return re.sub(r'[^a-zA-Z0-9\s\-]', '', value).strip()


def sanitize_chat_input(value: str) -> str:
    """
    Sanitize user chat input before processing.

    - Strips leading/trailing whitespace
    - Removes HTML tags to prevent XSS in stored/echoed messages
    - Escapes remaining HTML entities
    - Collapses excessive whitespace
    """
    # Strip HTML tags
    cleaned = re.sub(r'<[^>]+>', '', value)
    # Escape remaining HTML entities
    cleaned = html.escape(cleaned, quote=True)
    # Collapse multiple spaces / newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned
