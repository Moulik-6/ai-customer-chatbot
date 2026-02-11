"""
Input sanitization utilities for PostgREST and other query backends.
"""
import re


def sanitize_search(value: str) -> str:
    """
    Sanitize a search string for safe use in PostgREST filter expressions.

    PostgREST uses commas, dots, parentheses, and other punctuation as filter
    operators.  Strip everything except alphanumeric characters, spaces, and
    hyphens to prevent filter injection.
    """
    return re.sub(r'[^a-zA-Z0-9\s\-]', '', value).strip()
