"""
Chat routes — main chat endpoint, index page, health check.
"""
import logging

import requests
from flask import Blueprint, request, jsonify, send_from_directory, current_app

from extensions import limiter
from config import HUGGINGFACE_MODEL, MODEL_TYPE, USE_LOCAL_MODEL, MOCK_MODE, BASE_DIR
from database import supabase, log_conversation
from services.intent_service import match_intent, INTENTS
from services.entity_service import (
    extract_order_number, extract_email, extract_sku, extract_product_name,
)
from services.lookup_service import (
    lookup_order_status, lookup_orders_by_email,
    lookup_product, lookup_customer_by_email,
)
from services.formatter_service import (
    format_order, format_orders_list, format_product,
    format_product_list, format_customer,
)
from models.ai_model import query_model

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)


# ── Static pages ──────────────────────────────────────────

@chat_bp.route('/', methods=['GET'])
def index():
    return send_from_directory(BASE_DIR, 'index.html')


@chat_bp.route('/index.html', methods=['GET'])
def index_html():
    return send_from_directory(BASE_DIR, 'index.html')


# ── Health check ──────────────────────────────────────────

@chat_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "ai-customer-chatbot",
        "model": HUGGINGFACE_MODEL,
        "use_local_model": USE_LOCAL_MODEL,
        "mock_mode": MOCK_MODE,
        "intents_count": len(INTENTS),
    }), 200


# ── Main chat endpoint ───────────────────────────────────

@chat_bp.route('/api/chat', methods=['POST'])
@limiter.limit("30 per minute")
def chat():
    """
    Main chat endpoint for customer service chatbot.

    Expected JSON payload: { "message": "hello" }
    """
    # Access the limiter from the app
    limiter = current_app.extensions.get('limiter')

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be valid JSON", "code": "INVALID_REQUEST"}), 400

        message = data.get('message', '').strip()
        if not message:
            return jsonify({"error": "Message field is required and cannot be empty", "code": "EMPTY_MESSAGE"}), 400
        if len(message) > 2000:
            return jsonify({"error": "Message must not exceed 2000 characters", "code": "MESSAGE_TOO_LONG"}), 400

        logger.info(f"Processing {MODEL_TYPE} request: {message[:100]}...")

        session_id = data.get('session_id', request.headers.get('X-Session-ID', 'unknown'))
        ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)

        # ── Entity extraction ─────────────────────────────
        order_number = extract_order_number(message)
        email = extract_email(message)
        sku = extract_sku(message)
        product_name = extract_product_name(message)

        intent_match = match_intent(message)
        intent_tag = intent_match['tag'] if intent_match else None

        # Helper — build & return a DB-backed response
        def _db_response(bot_response, intent, response_type, extra=None):
            resp = {
                "success": True, "type": response_type,
                "intent": intent, "message": message,
                "response": bot_response, "model": "database",
            }
            if extra:
                resp.update(extra)
            log_conversation(
                session_id=session_id, user_message=message,
                bot_response=bot_response, intent=intent,
                model_used="database", response_type=response_type,
                ip_address=ip_address,
            )
            return jsonify(resp), 200

        # ========== 1. ORDER LOOKUP (by order number) ==========
        if order_number:
            order = lookup_order_status(order_number)
            if order:
                bot_response = format_order(order)
                logger.info(f"Order lookup: {order_number}")
                return _db_response(bot_response, "order_tracking", "order_lookup", {"order": order})
            bot_response = (
                f"❌ Sorry, I couldn't find order **{order_number}** in our system. "
                "Please check the order number and try again. Or contact support@company.com for assistance."
            )
            return _db_response(bot_response, "order_tracking", "order_not_found")

        # ========== 2. CUSTOMER LOOKUP (by email) ==========
        if email:
            if intent_tag in ('order_tracking', 'order_status', 'shipping'):
                orders = lookup_orders_by_email(email)
                if orders:
                    bot_response = format_orders_list(orders, email)
                    logger.info(f"Orders lookup by email: {email} ({len(orders)} found)")
                    return _db_response(bot_response, "order_tracking", "orders_by_email")
                bot_response = f"I couldn't find any orders associated with **{email}**. Please check the email address or provide an order number."
                return _db_response(bot_response, "order_tracking", "customer_not_found")

            customer = lookup_customer_by_email(email)
            if customer:
                bot_response = format_customer(customer)
                logger.info(f"Customer lookup: {email}")
                return _db_response(bot_response, "account", "customer_lookup")
            bot_response = f"I couldn't find an account associated with **{email}**. Would you like help creating one?"
            return _db_response(bot_response, "account", "customer_not_found")

        # ========== 3. PRODUCT LOOKUP (by SKU or name) ==========
        if intent_tag in ('product_info', 'pricing', 'stock_availability', 'size_fitting'):
            search_term = sku or product_name
            if search_term:
                products = lookup_product(search_term)
                if products:
                    bot_response = format_product(products)
                    logger.info(f"Product lookup: {search_term} ({len(products)} found)")
                    return _db_response(bot_response, intent_tag, "product_lookup")
            else:
                # No specific product — list available products
                if supabase:
                    try:
                        result = (supabase.table('products')
                                  .select('name,price,category,sku')
                                  .limit(10)
                                  .order('created_at', desc=True)
                                  .execute())
                        if result.data:
                            bot_response = format_product_list(result.data)
                            return _db_response(bot_response, intent_tag, "product_list")
                    except Exception as e:
                        logger.error(f"Error listing products: {e}")
            # Fall through to intent response

        # ========== 4. ORDER TRACKING (no order number) ==========
        if intent_tag == 'order_tracking':
            bot_response = intent_match['response']
            log_conversation(
                session_id=session_id, user_message=message,
                bot_response=bot_response, intent=intent_tag,
                model_used="intents", response_type="intent",
                ip_address=ip_address,
            )
            return jsonify({
                "success": True, "type": "intent", "intent": intent_tag,
                "message": message, "response": bot_response, "model": "intents",
            }), 200

        # ========== 5. OTHER INTENT MATCHES ==========
        if intent_match:
            logger.info(f"Intent matched: {intent_match['tag']}")
            log_conversation(
                session_id=session_id, user_message=message,
                bot_response=intent_match['response'],
                intent=intent_match['tag'],
                model_used="intents", response_type="intent",
                ip_address=ip_address,
            )
            return jsonify({
                "success": True, "type": "intent",
                "intent": intent_match['tag'], "message": message,
                "response": intent_match['response'], "model": "intents",
            }), 200

        # ========== 6. FALLBACK: AI MODEL ==========
        api_response = query_model(message)

        if api_response['type'] == 'generation':
            response_data = {
                "success": True, "type": "generation",
                "message": message, "response": api_response['result'],
                "model": api_response['model'],
            }
            log_conversation(
                session_id=session_id, user_message=message,
                bot_response=api_response['result'], intent=None,
                model_used=api_response['model'], response_type="generation",
                ip_address=ip_address,
            )
        else:
            response_data = {
                "success": True, "type": "classification",
                "message": message,
                "classification": {
                    "top_label": api_response['top_label'],
                    "scores": api_response['result'],
                },
                "model": api_response['model'],
            }
            log_conversation(
                session_id=session_id, user_message=message,
                bot_response=f"Classification: {api_response['top_label']}",
                intent=None, model_used=api_response['model'],
                response_type="classification", ip_address=ip_address,
            )

        logger.info(f"Response generated successfully (type: {api_response['type']})")
        return jsonify(response_data), 200

    except TimeoutError:
        logger.error("API request timed out")
        return jsonify({"error": "Request to AI service timed out. Please try again.", "code": "TIMEOUT"}), 504

    except ValueError as e:
        error_msg = str(e)
        if "Invalid API key" in error_msg:
            return jsonify({"error": "Authentication failed", "code": "AUTH_ERROR"}), 401
        return jsonify({"error": "Invalid response from AI service", "code": "INVALID_RESPONSE", "details": error_msg}), 500

    except requests.RequestException as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return jsonify({"error": "API rate limit exceeded. Please try again later.", "code": "RATE_LIMITED"}), 429
        if "loading" in error_msg.lower():
            return jsonify({"error": "Model is loading. Please try again in a moment.", "code": "MODEL_LOADING"}), 503
        logger.error(f"API request failed: {error_msg}")
        return jsonify({"error": "Failed to communicate with AI service", "code": "SERVICE_ERROR"}), 503

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred", "code": "INTERNAL_ERROR"}), 500
