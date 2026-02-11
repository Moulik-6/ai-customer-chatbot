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
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import PORT

logger = logging.getLogger(__name__)


def create_app():
    """Application factory — build and return the Flask app."""
    app = Flask(__name__)
    CORS(app)

    # Rate limiting
    Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per hour"],
        storage_uri="memory://",
    )

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
