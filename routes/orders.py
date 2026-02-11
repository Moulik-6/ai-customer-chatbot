"""
Order routes — CRUD for the orders + order_items tables.
"""
import logging

from flask import Blueprint, request, jsonify

from extensions import limiter
from auth import require_admin_key
from database import supabase

logger = logging.getLogger(__name__)

orders_bp = Blueprint('orders', __name__)


def _require_supabase():
    if not supabase:
        return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
    return None


@orders_bp.route('/api/orders', methods=['GET'])
def get_orders():
    """Get all orders or filter by customer/status."""
    err = _require_supabase()
    if err:
        return err

    try:
        customer_email = request.args.get('customer_email')
        status = request.args.get('status')
        limit = min(int(request.args.get('limit', 50)), 200)
        offset = int(request.args.get('offset', 0))

        query = supabase.table('orders').select('*')
        if customer_email:
            query = query.eq('customer_email', customer_email)
        if status:
            query = query.eq('status', status)

        query = query.range(offset, offset + limit - 1).order('order_date', desc=True)
        result = query.execute()

        return jsonify({"success": True, "count": len(result.data), "orders": result.data}), 200

    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        return jsonify({"error": "Failed to fetch orders", "code": "DB_ERROR"}), 500


@orders_bp.route('/api/orders/<order_id>', methods=['GET'])
def get_order(order_id):
    """Get a specific order with items."""
    err = _require_supabase()
    if err:
        return err

    try:
        order_result = supabase.table('orders').select('*').eq('id', order_id).execute()
        if not order_result.data:
            return jsonify({"error": "Order not found", "code": "NOT_FOUND"}), 404

        items_result = supabase.table('order_items').select('*').eq('order_id', order_id).execute()
        order = order_result.data[0]
        order['items'] = items_result.data

        return jsonify({"success": True, "order": order}), 200

    except Exception as e:
        logger.error(f"Error fetching order: {e}")
        return jsonify({"error": "Failed to fetch order", "code": "DB_ERROR"}), 500


@orders_bp.route('/api/orders/number/<order_number>', methods=['GET'])
def get_order_by_number(order_number):
    """Get order by order number."""
    err = _require_supabase()
    if err:
        return err

    try:
        order_result = supabase.table('orders').select('*').eq('order_number', order_number).execute()
        if not order_result.data:
            return jsonify({"error": "Order not found", "code": "NOT_FOUND"}), 404

        order = order_result.data[0]
        items_result = supabase.table('order_items').select('*').eq('order_id', order['id']).execute()
        order['items'] = items_result.data

        return jsonify({"success": True, "order": order}), 200

    except Exception as e:
        logger.error(f"Error fetching order: {e}")
        return jsonify({"error": "Failed to fetch order", "code": "DB_ERROR"}), 500


@orders_bp.route('/api/orders', methods=['POST'])
@limiter.limit("20 per minute")
@require_admin_key
def create_order():
    """Create a new order with items."""
    err = _require_supabase()
    if err:
        return err

    try:
        data = request.get_json()
        for field in ('order_number', 'customer_name', 'customer_email', 'total_amount', 'items'):
            if not data.get(field):
                return jsonify({"error": f"Missing required field: {field}", "code": "MISSING_FIELD"}), 400

        if not isinstance(data['items'], list) or len(data['items']) == 0:
            return jsonify({"error": "Order must contain at least one item", "code": "INVALID_ITEMS"}), 400

        order_data = {
            "order_number": data['order_number'],
            "customer_name": data['customer_name'],
            "customer_email": data['customer_email'],
            "customer_phone": data.get('customer_phone'),
            "shipping_address": data.get('shipping_address'),
            "status": data.get('status', 'pending'),
            "total_amount": float(data['total_amount']),
            "tracking_number": data.get('tracking_number'),
            "notes": data.get('notes'),
        }

        order_result = supabase.table('orders').insert(order_data).execute()
        order = order_result.data[0]
        order_id = order['id']

        items_data = []
        for item in data['items']:
            items_data.append({
                "order_id": order_id,
                "product_id": item.get('product_id'),
                "product_name": item['product_name'],
                "product_sku": item.get('product_sku'),
                "quantity": int(item['quantity']),
                "unit_price": float(item['unit_price']),
                "subtotal": float(item['quantity']) * float(item['unit_price']),
            })

        items_result = supabase.table('order_items').insert(items_data).execute()
        order['items'] = items_result.data

        return jsonify({"success": True, "order": order}), 201

    except Exception as e:
        logger.error(f"Error creating order: {e}")
        return jsonify({"error": "Failed to create order", "code": "DB_ERROR"}), 500


@orders_bp.route('/api/orders/<order_id>', methods=['PUT'])
@require_admin_key
def update_order(order_id):
    """Update order status or details."""
    err = _require_supabase()
    if err:
        return err

    try:
        data = request.get_json()
        data.pop('id', None)
        data.pop('created_at', None)
        data.pop('order_number', None)
        data.pop('items', None)

        result = supabase.table('orders').update(data).eq('id', order_id).execute()
        if not result.data:
            return jsonify({"error": "Order not found", "code": "NOT_FOUND"}), 404
        return jsonify({"success": True, "order": result.data[0]}), 200

    except Exception as e:
        logger.error(f"Error updating order: {e}")
        return jsonify({"error": "Failed to update order", "code": "DB_ERROR"}), 500


@orders_bp.route('/api/orders/<order_id>/status', methods=['PATCH'])
@require_admin_key
def update_order_status(order_id):
    """Update order status and tracking."""
    err = _require_supabase()
    if err:
        return err

    try:
        data = request.get_json()
        update_data = {}
        if 'status' in data:
            update_data['status'] = data['status']
        if 'tracking_number' in data:
            update_data['tracking_number'] = data['tracking_number']
        if 'notes' in data:
            update_data['notes'] = data['notes']

        if not update_data:
            return jsonify({"error": "No valid fields to update", "code": "INVALID_DATA"}), 400

        result = supabase.table('orders').update(update_data).eq('id', order_id).execute()
        if not result.data:
            return jsonify({"error": "Order not found", "code": "NOT_FOUND"}), 404
        return jsonify({"success": True, "order": result.data[0]}), 200

    except Exception as e:
        logger.error(f"Error updating order status: {e}")
        return jsonify({"error": "Failed to update order status", "code": "DB_ERROR"}), 500
