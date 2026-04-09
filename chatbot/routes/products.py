"""
Product routes — CRUD for the products table.
"""
import base64
import logging

import requests
from flask import Blueprint, request, jsonify, Response

from ..extensions import limiter
from ..auth import require_admin_key
from ..database import supabase
from ..services.sanitize import sanitize_search

logger = logging.getLogger(__name__)

products_bp = Blueprint('products', __name__)


def _require_supabase():
    """Return an error tuple if Supabase is not configured, else None."""
    if not supabase:
        return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
    return None


@products_bp.route('/api/products', methods=['GET'])
def get_products():
    """Get all products or search/filter products."""
    err = _require_supabase()
    if err:
        return err

    try:
        search = request.args.get('search', '').strip()
        search = sanitize_search(search)
        category = request.args.get('category')
        limit = min(int(request.args.get('limit', 100)), 500)
        offset = int(request.args.get('offset', 0))

        query = supabase.table('products').select('*')
        if search:
            query = query.or_(f"name.ilike.%{search}%,description.ilike.%{search}%,sku.ilike.%{search}%")
        if category:
            query = query.eq('category', category)

        query = query.range(offset, offset + limit - 1).order('created_at', desc=True)
        result = query.execute()

        return jsonify({"success": True, "count": len(result.data), "products": result.data}), 200

    except Exception as e:
        logger.error(f"Error fetching products: {e}")
        return jsonify({"error": "Failed to fetch products", "code": "DB_ERROR"}), 500


@products_bp.route('/api/products/<product_id>', methods=['GET'])
def get_product(product_id):
    """Get a specific product by ID."""
    err = _require_supabase()
    if err:
        return err

    try:
        result = supabase.table('products').select('*').eq('id', product_id).execute()
        if not result.data:
            return jsonify({"error": "Product not found", "code": "NOT_FOUND"}), 404
        return jsonify({"success": True, "product": result.data[0]}), 200

    except Exception as e:
        logger.error(f"Error fetching product: {e}")
        return jsonify({"error": "Failed to fetch product", "code": "DB_ERROR"}), 500


@products_bp.route('/api/products/<product_id>/image', methods=['GET'])
def get_product_image(product_id):
    """Proxy a product image through the app so the frontend can load it reliably."""
    err = _require_supabase()
    if err:
        return err

    try:
        result = supabase.table('products').select('id,name,image_url').eq('id', product_id).execute()
        if not result.data:
            return jsonify({"error": "Product not found", "code": "NOT_FOUND"}), 404

        image_url = (result.data[0].get('image_url') or '').strip()
        if not image_url:
            return jsonify({"error": "Image not found", "code": "NOT_FOUND"}), 404

        if image_url.startswith('data:image/'):
            header, encoded = image_url.split(',', 1)
            mime_type = header.split(';', 1)[0].split(':', 1)[1]
            payload = base64.b64decode(encoded)
            return Response(payload, mimetype=mime_type, headers={'Cache-Control': 'public, max-age=86400'})

        upstream = requests.get(
            image_url,
            timeout=20,
            headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36'},
        )
        upstream.raise_for_status()
        content_type = upstream.headers.get('content-type', 'image/jpeg').split(';', 1)[0].strip()
        return Response(upstream.content, mimetype=content_type, headers={'Cache-Control': 'public, max-age=86400'})

    except Exception as e:
        logger.warning(f"Error proxying product image {product_id}: {e}")
        return jsonify({"error": "Failed to fetch product image", "code": "IMAGE_FETCH_FAILED"}), 502


@products_bp.route('/api/products', methods=['POST'])
@limiter.limit("20 per minute")
@require_admin_key
def create_product():
    """Create a new product."""
    err = _require_supabase()
    if err:
        return err

    try:
        data = request.get_json()
        for field in ('name', 'price'):
            if not data.get(field):
                return jsonify({"error": f"Missing required field: {field}", "code": "MISSING_FIELD"}), 400

        product_data = {
            "name": data['name'],
            "description": data.get('description'),
            "price": float(data['price']),
            "category": data.get('category'),
            "sku": data.get('sku'),
            "stock": int(data.get('stock', 0)),
            "image_url": data.get('image_url'),
            "is_duplicate": data.get('is_duplicate', False),
            "duplicate_of": data.get('duplicate_of'),
        }

        result = supabase.table('products').insert(product_data).execute()
        return jsonify({"success": True, "product": result.data[0]}), 201

    except Exception as e:
        logger.error(f"Error creating product: {e}")
        return jsonify({"error": "Failed to create product", "code": "DB_ERROR"}), 500


@products_bp.route('/api/products/<product_id>', methods=['PUT'])
@require_admin_key
def update_product(product_id):
    """Update an existing product."""
    err = _require_supabase()
    if err:
        return err

    try:
        data = request.get_json()
        data.pop('id', None)
        data.pop('created_at', None)

        result = supabase.table('products').update(data).eq('id', product_id).execute()
        if not result.data:
            return jsonify({"error": "Product not found", "code": "NOT_FOUND"}), 404
        return jsonify({"success": True, "product": result.data[0]}), 200

    except Exception as e:
        logger.error(f"Error updating product: {e}")
        return jsonify({"error": "Failed to update product", "code": "DB_ERROR"}), 500


@products_bp.route('/api/products/<product_id>', methods=['DELETE'])
@require_admin_key
def delete_product(product_id):
    """Delete a product."""
    err = _require_supabase()
    if err:
        return err

    try:
        result = supabase.table('products').delete().eq('id', product_id).execute()
        if not result.data:
            return jsonify({"error": "Product not found", "code": "NOT_FOUND"}), 404
        return jsonify({"success": True, "message": "Product deleted successfully"}), 200

    except Exception as e:
        logger.error(f"Error deleting product: {e}")
        return jsonify({"error": "Failed to delete product", "code": "DB_ERROR"}), 500


@products_bp.route('/api/products/duplicates', methods=['GET'])
def get_duplicate_products():
    """Get all products marked as duplicates."""
    err = _require_supabase()
    if err:
        return err

    try:
        result = supabase.table('products').select('*').eq('is_duplicate', True).execute()
        return jsonify({"success": True, "count": len(result.data), "duplicates": result.data}), 200

    except Exception as e:
        logger.error(f"Error fetching duplicates: {e}")
        return jsonify({"error": "Failed to fetch duplicates", "code": "DB_ERROR"}), 500
