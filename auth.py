"""
Auth utilities — admin API-key decorator.
"""
from functools import wraps
from flask import request, jsonify
from config import ADMIN_API_KEY


def require_admin_key(f):
    """Decorator to protect admin/write endpoints with an API key."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not ADMIN_API_KEY:
            # No key configured — allow access (dev mode)
            return f(*args, **kwargs)
        key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if key != ADMIN_API_KEY:
            return jsonify({
                "error": "Unauthorized. Provide a valid X-API-Key header.",
                "code": "UNAUTHORIZED"
            }), 401
        return f(*args, **kwargs)
    return decorated
