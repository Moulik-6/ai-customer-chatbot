"""
Database lookups — query orders, products, and customers from Supabase.
"""
import logging
import re
import requests
import time
from ..database import supabase
from ..config import AFTERSHIP_API_KEY
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


# ── Live Tracking Cache (TTL: 1 hour) ──────────────────────
_tracking_cache = {}  # Format: {tracking_number: {'data': {...}, 'timestamp': time.time()}}
_TRACKING_CACHE_TTL = 3600  # 1 hour


def _parse_tracking_response(tracking_data):
    """Parse AfterShip tracking response into readable format."""
    if not tracking_data:
        return None
    
    tag = tracking_data.get('tag', 'unknown')  # delivered, in_transit, exception, etc
    status_map = {
        'delivered': ('🎉 Delivered', 'Your order has been delivered'),
        'in_transit': ('📦 In Transit', 'Your order is on the way'),
        'out_for_delivery': ('🚚 Out for Delivery', 'Your order is being delivered today'),
        'pending': ('⏳ Pending', 'Your order is being prepared'),
        'exception': ('⚠️ Issue Detected', 'There\'s an issue with your shipment'),
        'returned': ('🔄 Returned', 'Your order has been returned'),
    }
    
    status, description = status_map.get(tag, ('📋 Unknown', 'Tracking status unavailable'))
    
    checkpoints = tracking_data.get('checkpoints', [])
    latest_checkpoint = checkpoints[0] if checkpoints else None
    
    return {
        'tag': tag,
        'status': status,
        'description': description,
        'timestamp': tracking_data.get('updated_at'),
        'location': latest_checkpoint.get('location', {}).get('name') if latest_checkpoint else None,
        'message': latest_checkpoint.get('message') if latest_checkpoint else None,
        'latest_event': {
            'time': latest_checkpoint.get('checkpoint_time') if latest_checkpoint else None,
            'location': latest_checkpoint.get('location', {}).get('name') if latest_checkpoint else None,
            'message': latest_checkpoint.get('message') if latest_checkpoint else None,
        },
        'estimated_delivery': tracking_data.get('expected_delivery'),
        'checkpoints': checkpoints[:3],  # Last 3 events
    }


def get_live_tracking(tracking_number, carrier='auto'):
    """
    Fetch live tracking from AfterShip (supports 500+ carriers).
    Caches result for 1 hour to avoid rate limiting.
    Falls back gracefully if API unavailable.
    """
    if not tracking_number or not AFTERSHIP_API_KEY:
        return None
    
    # Check cache first
    if tracking_number in _tracking_cache:
        cached = _tracking_cache[tracking_number]
        if time.time() - cached['timestamp'] < _TRACKING_CACHE_TTL:
            logger.info(f"Returning cached tracking for {tracking_number}")
            return cached['data']
        else:
            del _tracking_cache[tracking_number]  # Expired
    
    try:
        # AfterShip API endpoint
        url = f'https://api.aftership.com/v4/trackings/{carrier}/{tracking_number}'
        
        headers = {
            'aftership-api-key': AFTERSHIP_API_KEY,
            'Content-Type': 'application/json'
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            tracking_data = response.json().get('data', {}).get('tracking', {})
            parsed = _parse_tracking_response(tracking_data)
            
            # Cache the result
            _tracking_cache[tracking_number] = {
                'data': parsed,
                'timestamp': time.time()
            }
            
            logger.info(f"Live tracking fetched for {tracking_number}: {tracking_data.get('tag')}")
            return parsed
        
        elif response.status_code == 404:
            logger.warning(f"Tracking number {tracking_number} not found in AfterShip")
            return None
        
        else:
            logger.warning(f"AfterShip API error {response.status_code}: {response.text}")
            return None
    
    except requests.Timeout:
        logger.warning(f"AfterShip API timeout for {tracking_number}")
        return None
    except Exception as e:
        logger.error(f"Error fetching live tracking for {tracking_number}: {e}", exc_info=True)
        return None
