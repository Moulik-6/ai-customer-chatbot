"""
AI Customer Service Chatbot — Flask application entry point.

This is a slim orchestrator that wires together:
  config     — environment variables & constants
  database   — Supabase + SQLite clients
  auth       — admin API-key decorator
  models     — AI model loading & inference
  services   — intent matching, entity extraction, lookups, formatters
  routes     — Flask Blueprints for chat, admin, products, orders
"""
import os
import secrets
import logging
from flask import Flask, jsonify
from flask_cors import CORS

from config import PORT
from extensions import limiter

logger = logging.getLogger(__name__)


def create_app():
    """Application factory — build and return the Flask app."""
    app = Flask(__name__)

    # Secret key for session signing (generate if not set)
    app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))

    # CORS — restrict to known origins in production
    allowed_origins = os.getenv('CORS_ORIGINS', 'http://localhost:7860').split(',')
    CORS(app, origins=allowed_origins)

    # Security headers
    @app.after_request
    def set_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self'; "
            "connect-src 'self'"
        )
        return response

    # Attach rate limiter to app
    limiter.init_app(app)

    # Register blueprints
    from routes.chat import chat_bp
    from routes.admin import admin_bp
    from routes.products import products_bp
    from routes.orders import orders_bp

    app.register_blueprint(chat_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(products_bp)
    app.register_blueprint(orders_bp)

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Endpoint not found", "code": "NOT_FOUND"}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({"error": "Method not allowed", "code": "METHOD_NOT_ALLOWED"}), 405

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}")
        return jsonify({"error": "Internal server error", "code": "INTERNAL_ERROR"}), 500

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
