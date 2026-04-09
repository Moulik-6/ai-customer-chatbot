"""
Database lookups — query orders, products, and customers from Supabase.
"""
import logging
import re
import requests
import time
import uuid
from datetime import datetime, timedelta, timezone
from ..database import supabase
from ..config import AFTERSHIP_API_KEY, TRACKING_MOCK_MODE, TRACKING_MOCK_CARRIER
from .sanitize import sanitize_search
from .entity_service import extract_sku
from .email_service import send_order_confirmation_email

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


def _generate_order_number():
    """Generate an order number with low collision risk."""
    return f"ORD-{datetime.now(timezone.utc):%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"


def _pick_order_product(product_query):
    """Pick the best in-stock product match for order placement."""
    matches = lookup_product(product_query)
    if not matches:
        return None

    # Prefer in-stock items and keep deterministic selection by highest stock.
    in_stock = [p for p in matches if int(p.get('stock') or 0) > 0]
    if in_stock:
        return sorted(in_stock, key=lambda p: int(p.get('stock') or 0), reverse=True)[0]

    return matches[0]


def create_order_from_chat(customer_name, customer_email, product_query, quantity=1):
    """
    Create a pending order from chat context and reserve stock.

    Returns dict:
      {"success": True, "order": {...}} on success
      {"success": False, "code": "...", "error": "..."} on failure
    """
    try:
        if not supabase:
            return {
                "success": False,
                "code": "DB_NOT_CONFIGURED",
                "error": "Database is not configured right now.",
            }

        qty = max(1, int(quantity or 1))
        product = _pick_order_product(product_query)
        if not product:
            return {
                "success": False,
                "code": "PRODUCT_NOT_FOUND",
                "error": f"I couldn't find a product matching '{product_query}'.",
            }

        stock = int(product.get('stock') or 0)
        if stock < qty:
            return {
                "success": False,
                "code": "OUT_OF_STOCK",
                "error": (
                    f"Not enough stock for {product.get('name', 'that product')}. "
                    f"Requested {qty}, available {stock}."
                ),
            }

        unit_price = float(product.get('price') or 0)
        subtotal = round(unit_price * qty, 2)

        order_payload = {
            'order_number': _generate_order_number(),
            'customer_name': customer_name,
            'customer_email': customer_email,
            'status': 'pending',
            'total_amount': subtotal,
            'notes': 'Order created via chatbot',
        }

        created_order = supabase.table('orders').insert(order_payload).execute().data[0]

        item_payload = {
            'order_id': created_order['id'],
            'product_id': product.get('id'),
            'product_name': product.get('name'),
            'product_sku': product.get('sku'),
            'quantity': qty,
            'unit_price': unit_price,
            'subtotal': subtotal,
        }
        supabase.table('order_items').insert(item_payload).execute()

        # Reserve stock for this order.
        supabase.table('products').update({'stock': stock - qty}).eq('id', product['id']).execute()

        order = lookup_order_status(created_order['order_number'])
        final_order = order or created_order

        email_sent = send_order_confirmation_email(customer_email, final_order)
        return {'success': True, 'order': final_order, 'email_sent': email_sent}

    except Exception as e:
        logger.error(f"Error creating order from chat: {e}", exc_info=True)
        return {
            "success": False,
            "code": "DB_ERROR",
            "error": "Failed to create order. Please try again.",
        }


# ── Live Tracking Cache (TTL: 1 hour) ──────────────────────
_tracking_cache = {}  # Format: {tracking_number: {'data': {...}, 'timestamp': time.time()}}
_TRACKING_CACHE_TTL = 3600  # 1 hour


def _build_mock_tracking(tracking_number, carrier='mock', expected_status=None):
    """Generate deterministic mock tracking events for local testing."""
    if not tracking_number:
        return None

    now = datetime.now(timezone.utc)
    status_cycle = ['pending', 'in_transit', 'out_for_delivery', 'delivered']
    idx = sum(ord(ch) for ch in tracking_number) % len(status_cycle)
    tag = status_cycle[idx]

    status_aliases = {
        'pending': 'pending',
        'processing': 'pending',
        'shipped': 'in_transit',
        'in_transit': 'in_transit',
        'out_for_delivery': 'out_for_delivery',
        'delivered': 'delivered',
        'cancelled': 'cancelled',
        'returned': 'returned',
    }
    if expected_status:
        normalized = status_aliases.get(str(expected_status).strip().lower())
        if normalized:
            tag = normalized

    checkpoint_templates = {
        'pending': [
            (now - timedelta(hours=16), 'Warehouse A', 'Shipment information received'),
            (now - timedelta(hours=10), 'Sorting Center', 'Package prepared for dispatch'),
        ],
        'in_transit': [
            (now - timedelta(hours=20), 'Origin Facility', 'Package accepted by carrier'),
            (now - timedelta(hours=9), 'Regional Hub', 'Package in transit'),
            (now - timedelta(hours=2), 'Destination Hub', 'Arrived at destination facility'),
        ],
        'out_for_delivery': [
            (now - timedelta(hours=18), 'Destination Hub', 'Arrived at destination facility'),
            (now - timedelta(hours=6), 'Local Depot', 'Loaded on delivery vehicle'),
            (now - timedelta(minutes=45), 'Local Route', 'Out for delivery'),
        ],
        'delivered': [
            (now - timedelta(hours=22), 'Destination Hub', 'Arrived at destination facility'),
            (now - timedelta(hours=7), 'Local Route', 'Out for delivery'),
            (now - timedelta(hours=1), 'Recipient Address', 'Delivered successfully'),
        ],
        'cancelled': [
            (now - timedelta(hours=8), 'Order Management', 'Shipment cancelled by merchant'),
        ],
        'returned': [
            (now - timedelta(days=3), 'Recipient Address', 'Delivery attempt failed'),
            (now - timedelta(days=1), 'Return Hub', 'Package returned to sender'),
        ],
    }

    checkpoints = []
    for ts, location, message in checkpoint_templates[tag]:
        checkpoints.append({
            'checkpoint_time': ts.isoformat(),
            'message': message,
            'location': {'name': location},
        })

    expected_delivery = None
    if tag in ('pending', 'in_transit'):
        expected_delivery = (now + timedelta(days=2)).date().isoformat()
    elif tag == 'out_for_delivery':
        expected_delivery = now.date().isoformat()

    return {
        'tag': tag,
        'status': {
            'pending': '⏳ Pending',
            'in_transit': '📦 In Transit',
            'out_for_delivery': '🚚 Out for Delivery',
            'delivered': '🎉 Delivered',
            'cancelled': '❌ Cancelled',
            'returned': '🔄 Returned',
        }[tag],
        'description': f"Mock tracking from {carrier.upper()} for testing",
        'timestamp': now.isoformat(),
        'location': checkpoints[-1]['location']['name'] if checkpoints else None,
        'message': checkpoints[-1]['message'] if checkpoints else None,
        'latest_event': {
            'time': checkpoints[-1]['checkpoint_time'] if checkpoints else None,
            'location': checkpoints[-1]['location']['name'] if checkpoints else None,
            'message': checkpoints[-1]['message'] if checkpoints else None,
        },
        'estimated_delivery': expected_delivery,
        'checkpoints': list(reversed(checkpoints[-3:])),
    }


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


def get_live_tracking(tracking_number, carrier='auto', expected_status=None):
    """
    Fetch live tracking from AfterShip (supports 500+ carriers).
    Caches result for 1 hour to avoid rate limiting.
    Falls back gracefully if API unavailable.
    """
    if not tracking_number:
        return None

    if TRACKING_MOCK_MODE:
        mock_data = _build_mock_tracking(
            tracking_number,
            carrier=TRACKING_MOCK_CARRIER,
            expected_status=expected_status,
        )
        if mock_data:
            logger.info(f"Using mock tracking for {tracking_number}")
        return mock_data

    if not AFTERSHIP_API_KEY:
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
