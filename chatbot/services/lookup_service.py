"""
Database lookups — query orders, products, and customers from Supabase.
"""
import logging
import re
from ..database import supabase
from .sanitize import sanitize_search
from .entity_service import extract_sku

logger = logging.getLogger(__name__)

_NOISE_TERMS = {
    'show', 'me', 'the', 'a', 'an', 'products', 'product', 'item', 'items',
    'about', 'details', 'info', 'information', 'stock', 'in', 'available',
    'what', 'is', 'are', 'for', 'of', 'please', 'can', 'you'
}


def _candidate_product_queries(query):
    """Generate progressively simpler search candidates from natural language input."""
    cleaned = sanitize_search(query)
    if not cleaned:
        return []

    tokens = [t for t in re.split(r'\s+', cleaned.lower()) if t]
    keyword_tokens = [t for t in tokens if t not in _NOISE_TERMS]

    candidates = [cleaned]
    if keyword_tokens:
        candidates.append(' '.join(keyword_tokens))
        # Also try individual keywords from most specific to broadest.
        for token in keyword_tokens:
            if len(token) >= 3:
                candidates.append(token)

    # Preserve insertion order while removing duplicates.
    return list(dict.fromkeys([c.strip() for c in candidates if c.strip()]))


def lookup_order_status(order_number):
    """Lookup order by order number (includes items)."""
    try:
        if not supabase:
            return None
        result = supabase.table('orders').select('*,order_items(*)').eq('order_number', order_number).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Error looking up order {order_number!r}: {e}", exc_info=True)
        return None


def lookup_orders_by_email(email):
    """Lookup all orders for a customer by email."""
    try:
        if not supabase:
            return None
        result = (supabase.table('orders')
                  .select('*,order_items(*)')
                  .eq('customer_email', email)
                  .order('order_date', desc=True)
                  .limit(5)
                  .execute())
        return result.data if result.data else None
    except Exception as e:
        logger.error(f"Error looking up orders by email {email!r}: {e}", exc_info=True)
        return None


def lookup_product(query):
    """Lookup product by SKU (exact) or name/description (fuzzy)."""
    try:
        if not supabase:
            return None

        # Try exact SKU match first
        sku = extract_sku(query) if query == query.upper() else extract_sku(query.upper())
        if sku:
            result = supabase.table('products').select('*').eq('sku', sku).execute()
            if result.data:
                return result.data

        # Fuzzy name/description/category search with natural-language fallback candidates
        for candidate in _candidate_product_queries(query):
            result = (supabase.table('products')
                      .select('*')
                      .or_(f"name.ilike.%{candidate}%,description.ilike.%{candidate}%,sku.ilike.%{candidate}%,category.ilike.%{candidate}%")
                      .limit(5)
                      .execute())
            if result.data:
                return result.data

        return None
    except Exception as e:
        logger.error(f"Error looking up product {query!r}: {e}", exc_info=True)
        return None


def lookup_customer_by_email(email):
    """Lookup customer info by aggregating their orders."""
    try:
        if not supabase:
            return None
        result = (supabase.table('orders')
                  .select('customer_name, customer_email, customer_phone, shipping_address, status, order_number, total_amount, order_date')
                  .eq('customer_email', email)
                  .order('order_date', desc=True)
                  .limit(10)
                  .execute())
        if not result.data:
            return None
        return {
            'name': result.data[0].get('customer_name'),
            'email': email,
            'phone': result.data[0].get('customer_phone'),
            'address': result.data[0].get('shipping_address'),
            'total_orders': len(result.data),
            'orders': result.data,
        }
    except Exception as e:
        logger.error(f"Error looking up customer {email!r}: {e}", exc_info=True)
        return None


def list_products(limit=10, in_stock_only=False):
    """List recent products (for generic 'what products?' queries)."""
    try:
        if not supabase:
            return None
        # Primary path: sort by recency when created_at exists.
        try:
            query = (supabase.table('products')
                     .select('name,price,category,sku,stock')
                     .limit(limit)
                     .order('created_at', desc=True))
            if in_stock_only:
                query = query.gt('stock', 0)

            result = query.execute()
            if result.data:
                return result.data
        except Exception as e:
            logger.warning(f"Primary list_products query failed, retrying without created_at sort: {e}")

        # Fallback path: list without relying on created_at column.
        query = supabase.table('products').select('name,price,category,sku,stock').limit(limit)
        if in_stock_only:
            query = query.gt('stock', 0)
        result = query.execute()
        return result.data if result.data else None
    except Exception as e:
        logger.error(f"Error listing products: {e}", exc_info=True)
        return None
