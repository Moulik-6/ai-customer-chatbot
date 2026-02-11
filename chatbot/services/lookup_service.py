"""
Database lookups — query orders, products, and customers from Supabase.
"""
import logging
from ..database import supabase
from .sanitize import sanitize_search
from .entity_service import extract_sku

logger = logging.getLogger(__name__)


def lookup_order_status(order_number):
    """Lookup order by order number (includes items)."""
    try:
        if not supabase:
            return None
        result = supabase.table('orders').select('*,order_items(*)').eq('order_number', order_number).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logger.error(f"Error looking up order: {e}")
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
        logger.error(f"Error looking up orders by email: {e}")
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

        # Fuzzy name/description/category search
        safe_query = sanitize_search(query)
        result = (supabase.table('products')
                  .select('*')
                  .or_(f"name.ilike.%{safe_query}%,description.ilike.%{safe_query}%,sku.ilike.%{safe_query}%,category.ilike.%{safe_query}%")
                  .limit(5)
                  .execute())
        return result.data if result.data else None
    except Exception as e:
        logger.error(f"Error looking up product: {e}")
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
        logger.error(f"Error looking up customer: {e}")
        return None


def list_products(limit=10):
    """List recent products (for generic 'what products?' queries)."""
    try:
        if not supabase:
            return None
        result = (supabase.table('products')
                  .select('name,price,category,sku')
                  .limit(limit)
                  .order('created_at', desc=True)
                  .execute())
        return result.data if result.data else None
    except Exception as e:
        logger.error(f"Error listing products: {e}")
