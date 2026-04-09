"""
Chat routes — main chat endpoint, index page, health check.
"""
import logging
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import requests
from flask import Blueprint, request, jsonify, redirect, send_from_directory

from ..extensions import limiter
from ..config import (
    HUGGINGFACE_MODEL, MODEL_TYPE, USE_LOCAL_MODEL, MOCK_MODE,
    CHAT_RATE_LIMIT, FRONTEND_URL, PROJECT_ROOT,
)
from ..database import log_conversation
from ..services.intent_service import match_intent, INTENTS
from ..services.entity_service import (
    extract_order_number, extract_email, extract_sku, extract_product_name,
)
from ..services.lookup_service import (
    lookup_order_status, lookup_orders_by_email,
    lookup_product, lookup_customer_by_email, list_products,
    get_live_tracking,
)
from ..services.formatter_service import (
    format_order, format_orders_list, format_product,
    format_product_list, format_customer,
)
from ..services.sanitize import sanitize_chat_input
from ..models.ai_model import query_model

logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__)


_LOCAL_FRONTEND_DIR = Path(PROJECT_ROOT) / 'frontend'
_RE_PRODUCT_HINT = re.compile(
    r'\b(product|products|catalog|inventory|stock|price|pricing|cost|electronics|phone|laptop|ipad|apple)\b',
    re.IGNORECASE,
)
_RE_PRODUCT_LIST_REQUEST = re.compile(
    r'\b(list all products|list products|show products|show me products|show me your products|what products do you have|product catalog)\b',
    re.IGNORECASE,
)
_RE_STOCK_LIST_REQUEST = re.compile(
    r'\b(what is in stock|what products are in stock|in stock|available now)\b',
    re.IGNORECASE,
)

# ── Conversation context (in-memory, per session) ────────
# Stores last MAX_CONTEXT_TURNS exchanges per session_id.
MAX_CONTEXT_TURNS = 5
_conversation_context = defaultdict(list)  # session_id -> [{user, bot}, ...]


@chat_bp.route('/', methods=['GET'])
def index():
    if FRONTEND_URL:
        return redirect(FRONTEND_URL)
    return send_from_directory(_LOCAL_FRONTEND_DIR, 'index.html')


@chat_bp.route('/index.html', methods=['GET'])
def index_html():
    if FRONTEND_URL:
        return redirect(FRONTEND_URL)
    return send_from_directory(_LOCAL_FRONTEND_DIR, 'index.html')


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


# ── Feedback endpoint ─────────────────────────────────────

@chat_bp.route('/api/feedback', methods=['POST'])
@limiter.limit("30 per minute")
def feedback():
    """
    Record user satisfaction rating for a bot response.

    Expected JSON: { "message_id": "...", "rating": "up"|"down" }
    """
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    rating = data.get('rating', '')
    if rating not in ('up', 'down'):
        return jsonify({"error": "rating must be 'up' or 'down'"}), 400

    message_text = data.get('message', '')[:200]
    session_id = data.get('session_id', request.headers.get('X-Session-ID', 'unknown'))

    logger.info(f"Feedback: {rating} | session={session_id} | msg={message_text[:60]}")

    # TODO: persist to DB when a feedback table is ready
    return jsonify({"success": True, "rating": rating}), 200


# ── Main chat endpoint ───────────────────────────────────

@chat_bp.route('/api/chat', methods=['POST'])
@limiter.limit(CHAT_RATE_LIMIT)
def chat():
    """
    Main chat endpoint for customer service chatbot.

    Expected JSON payload: { "message": "hello" }
    """
    start_time = time.monotonic()

    try:
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body must be valid JSON", "code": "INVALID_REQUEST"}), 400

        message = data.get('message', '').strip()
        if not message:
            return jsonify({"error": "Message field is required and cannot be empty", "code": "EMPTY_MESSAGE"}), 400
        if len(message) > 2000:
            return jsonify({"error": "Message must not exceed 2000 characters", "code": "MESSAGE_TOO_LONG"}), 400

        # Sanitize input (strip HTML tags, escape entities)
        message = sanitize_chat_input(message)

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

        def _elapsed_ms():
            return int((time.monotonic() - start_time) * 1000)

        def _store_context(bot_response):
            """Append the exchange to the session's conversation history."""
            ctx = _conversation_context[session_id]
            ctx.append({'user': message, 'bot': bot_response})
            if len(ctx) > MAX_CONTEXT_TURNS:
                _conversation_context[session_id] = ctx[-MAX_CONTEXT_TURNS:]

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
                ip_address=ip_address, response_time_ms=_elapsed_ms(),
            )
            _store_context(bot_response)
            return jsonify(resp), 200

        def _enhance_intent_response(base_response, intent):
            """Polish intent responses with FLAN when available; otherwise keep base text."""
            prompt = (
                "Rewrite this customer support response to sound natural and helpful. "
                "Keep the meaning the same, avoid inventing new policy/details, and keep it concise (1-2 sentences).\n\n"
                f"Intent: {intent}\n"
                f"Customer message: {message}\n"
                f"Base response: {base_response}\n"
                "Rewritten response:"
            )
            try:
                enhanced = query_model(prompt, context=context)
                if (
                    enhanced.get('type') == 'generation'
                    and enhanced.get('model') != 'fallback'
                    and (enhanced.get('result') or '').strip()
                ):
                    return enhanced['result'].strip(), enhanced['model']
            except Exception as exc:
                logger.warning(f"Intent enhancement skipped: {exc}")
            return base_response, 'intents'

        def _parse_model_plan(raw_text):
            """Parse planner JSON from model output, tolerating extra wrapper text."""
            if not raw_text:
                return None
            text = raw_text.strip()

            # Try full text first, then first JSON object block.
            candidates = [text]
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and end > start:
                candidates.append(text[start:end + 1])

            for candidate in candidates:
                try:
                    parsed = json.loads(candidate)
                    action = parsed.get('action', '').strip().lower()
                    if action in {
                        'none', 'list_products', 'list_stock_products',
                        'search_products', 'lookup_order',
                        'lookup_orders_by_email', 'lookup_customer_by_email',
                    }:
                        return parsed
                except Exception:
                    continue
            return None

        # ========== 0. PRIMARY: AI MODEL GENERATION ==========
        # Let the model choose whether to call a DB lookup tool first.
        # Intent and DB routes are fallback when model generation is unavailable.
        context = _conversation_context.get(session_id, [])

        planner_prompt = (
            "You are a routing planner for customer support tools. "
            "Choose one DB action for the user's message. "
            "Return ONLY valid JSON with keys: action, query, order_number, email. "
            "Allowed actions: none, list_products, list_stock_products, search_products, "
            "lookup_order, lookup_orders_by_email, lookup_customer_by_email.\n\n"
            f"User message: {message}\n"
            f"Extracted order_number: {order_number or ''}\n"
            f"Extracted email: {email or ''}\n"
            f"Extracted sku: {sku or ''}\n"
            f"Extracted product_name: {product_name or ''}\n"
            "JSON:"
        )

        planner_response = query_model(planner_prompt, context=context, use_support_prompt=False)
        plan = _parse_model_plan(planner_response.get('result') if planner_response else None)
        def _fallback_db_action():
            if order_number:
                return 'lookup_order'
            if email:
                if intent_tag in ('order_tracking', 'order_status', 'shipping'):
                    return 'lookup_orders_by_email'
                return 'lookup_customer_by_email'
            if _RE_STOCK_LIST_REQUEST.search(message):
                return 'list_stock_products'
            if _RE_PRODUCT_LIST_REQUEST.search(message):
                return 'list_products'
            if sku or product_name or _RE_PRODUCT_HINT.search(message):
                return 'search_products'
            return 'none'

        action = (plan.get('action') if plan else '') or _fallback_db_action()
        action = action.strip().lower()
        plan_query = (plan.get('query') if plan else '') or ''
        plan_order = (plan.get('order_number') if plan else '') or ''
        plan_email = (plan.get('email') if plan else '') or ''
        plan_query = plan_query.strip()
        plan_order = plan_order.strip()
        plan_email = plan_email.strip().lower()

        if action == 'list_stock_products':
            stock_products = list_products(in_stock_only=True)
            if stock_products:
                return _db_response(format_product_list(stock_products), "stock_availability", "product_list")

        if action == 'list_products':
            all_products = list_products()
            if all_products:
                return _db_response(format_product_list(all_products), "product_info", "product_list")

        if action == 'search_products':
            search_term = plan_query or sku or product_name or message
            products = lookup_product(search_term)
            if products:
                return _db_response(format_product(products), "product_info", "product_lookup")

        if action == 'lookup_order':
            lookup_num = plan_order or order_number
            if lookup_num:
                order = lookup_order_status(lookup_num)
                if order:
                    # Fetch live tracking if tracking number available
                    live_tracking = None
                    if order.get('tracking_number'):
                        live_tracking = get_live_tracking(
                            order['tracking_number'],
                            expected_status=order.get('status'),
                        )
                    return _db_response(format_order(order, live_tracking=live_tracking), "order_tracking", "order_lookup", {"order": order})
                bot_response = (
                    f"❌ Sorry, I couldn't find order **{lookup_num}** in our system. "
                    "Please check the order number and try again. Or contact support@company.com for assistance."
                )
                return _db_response(bot_response, "order_tracking", "order_not_found")

        if action == 'lookup_orders_by_email':
            lookup_email = plan_email or email
            if lookup_email:
                orders = lookup_orders_by_email(lookup_email)
                if orders:
                    return _db_response(format_orders_list(orders, lookup_email), "order_tracking", "orders_by_email")
                bot_response = (
                    f"I couldn't find any orders associated with **{lookup_email}**. "
                    "Please check the email address or provide an order number."
                )
                return _db_response(bot_response, "order_tracking", "customer_not_found")

        if action == 'lookup_customer_by_email':
            lookup_email = plan_email or email
            if lookup_email:
                customer = lookup_customer_by_email(lookup_email)
                if customer:
                    return _db_response(format_customer(customer), "account", "customer_lookup")
                bot_response = f"I couldn't find an account associated with **{lookup_email}**. Would you like help creating one?"
                return _db_response(bot_response, "account", "customer_not_found")

        api_response = query_model(message, context=context)
        model_generation_ready = (
            api_response.get('type') == 'generation'
            and bool((api_response.get('result') or '').strip())
            and api_response.get('model') != 'fallback'
        )

        if model_generation_ready:
            bot_response = api_response['result']
            log_conversation(
                session_id=session_id, user_message=message,
                bot_response=bot_response, intent=None,
                model_used=api_response['model'], response_type="generation",
                ip_address=ip_address, response_time_ms=_elapsed_ms(),
            )
            _store_context(bot_response)
            return jsonify({
                "success": True, "type": "generation",
                "message": message, "response": bot_response,
                "model": api_response['model'],
            }), 200

        # ========== 0. EXPLICIT PRODUCT LIST REQUESTS (DB-first) ==========
        if _RE_STOCK_LIST_REQUEST.search(message):
            stock_products = list_products(in_stock_only=True)
            if stock_products:
                bot_response = format_product_list(stock_products)
                return _db_response(bot_response, "stock_availability", "product_list")
            bot_response = (
                "I couldn't fetch in-stock products right now. "
                "Please try again in a moment, or ask for a specific product name or SKU."
            )
            return _db_response(bot_response, "stock_availability", "product_catalog_unavailable")

        if _RE_PRODUCT_LIST_REQUEST.search(message):
            all_products = list_products()
            if all_products:
                bot_response = format_product_list(all_products)
                return _db_response(bot_response, "product_info", "product_list")
            bot_response = (
                "I couldn't fetch the product catalog right now. "
                "Please try again in a moment, or ask for a specific product name or SKU."
            )
            return _db_response(bot_response, "product_info", "product_catalog_unavailable")

        # ========== 1. ORDER LOOKUP (by order number) ==========
        if order_number:
            order = lookup_order_status(order_number)
            if order:
                # Fetch live tracking if tracking number available
                live_tracking = None
                if order.get('tracking_number'):
                    live_tracking = get_live_tracking(
                        order['tracking_number'],
                        expected_status=order.get('status'),
                    )
                bot_response = format_order(order, live_tracking=live_tracking)
                logger.info(f"Order lookup: {order_number}")
                return _db_response(bot_response, "order_tracking", "order_lookup", {"order": order})
            logger.debug(f"[DB_LOOKUP_MISS] order_number={order_number!r} — no match in DB (check RLS / empty table / wrong format)")
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
                logger.debug(f"[DB_LOOKUP_MISS] email={email!r} (order intent) — no orders found in DB")
                bot_response = f"I couldn't find any orders associated with **{email}**. Please check the email address or provide an order number."
                return _db_response(bot_response, "order_tracking", "customer_not_found")

            customer = lookup_customer_by_email(email)
            if customer:
                bot_response = format_customer(customer)
                logger.info(f"Customer lookup: {email}")
                return _db_response(bot_response, "account", "customer_lookup")
            logger.debug(f"[DB_LOOKUP_MISS] email={email!r} — no customer found in DB")
            bot_response = f"I couldn't find an account associated with **{email}**. Would you like help creating one?"
            return _db_response(bot_response, "account", "customer_not_found")

        # ========== 3. PRODUCT LOOKUP (by SKU or name) ==========
        if intent_tag in ('product_info', 'pricing', 'stock_availability', 'size_fitting'):
            search_term = sku or product_name
            products = None

            if search_term:
                products = lookup_product(search_term)
                if products:
                    bot_response = format_product(products)
                    logger.info(f"Product lookup: {search_term} ({len(products)} found)")
                    return _db_response(bot_response, intent_tag, "product_lookup")
                logger.debug(f"[DB_LOOKUP_MISS] sku/product_name={search_term!r} — no product match in DB")

            # Always try the full message even if an entity was extracted but returned no matches.
            products = lookup_product(message)
            if products:
                bot_response = format_product(products)
                logger.info(f"Product search (raw message): {message[:50]} ({len(products)} found)")
                return _db_response(bot_response, intent_tag, "product_lookup")

            # For stock intent, surface in-stock catalog when no direct match was found.
            if intent_tag == 'stock_availability':
                all_products = list_products(in_stock_only=True)
                if all_products:
                    bot_response = format_product_list(all_products)
                    return _db_response(bot_response, intent_tag, "product_list")

            # For other product intents, surface general catalog before generic intent fallback.
            all_products = list_products()
            if all_products:
                bot_response = format_product_list(all_products)
                return _db_response(bot_response, intent_tag, "product_list")

            bot_response = (
                "I couldn't fetch the product catalog right now. "
                "Please try again in a moment, or ask for a specific product name or SKU."
            )
            return _db_response(bot_response, intent_tag, "product_catalog_unavailable")

        # ========== 3.5 PRODUCT LOOKUP (DB-first for product-like free text) ==========
        # If text smells like a product/pricing request, try DB before model fallback.
        if not intent_match and _RE_PRODUCT_HINT.search(message):
            products = lookup_product(message)
            if products:
                bot_response = format_product(products)
                logger.info(f"Product DB-first fallback: {message[:50]} ({len(products)} found)")
                return _db_response(bot_response, "product_info", "product_lookup")

            all_products = list_products()
            if all_products:
                bot_response = format_product_list(all_products)
                return _db_response(bot_response, "product_info", "product_list")

        # ========== 4. ORDER TRACKING (no order number) ==========
        # Intent responses may be enhanced by the AI model when available.
        if intent_tag == 'order_tracking':
            bot_response, model_used = _enhance_intent_response(intent_match['response'], intent_tag)
            log_conversation(
                session_id=session_id, user_message=message,
                bot_response=bot_response, intent=intent_tag,
                model_used=model_used, response_type="intent",
                ip_address=ip_address, response_time_ms=_elapsed_ms(),
            )
            _store_context(bot_response)
            return jsonify({
                "success": True, "type": "intent", "intent": intent_tag,
                "message": message, "response": bot_response, "model": model_used,
            }), 200

        # ========== 5. OTHER INTENT MATCHES ==========
        # Intent responses may be enhanced by the AI model when available.
        if intent_match:
            logger.info(f"Intent matched: {intent_match['tag']}")
            bot_response, model_used = _enhance_intent_response(intent_match['response'], intent_match['tag'])
            log_conversation(
                session_id=session_id, user_message=message,
                bot_response=bot_response,
                intent=intent_match['tag'],
                model_used=model_used, response_type="intent",
                ip_address=ip_address, response_time_ms=_elapsed_ms(),
            )
            _store_context(bot_response)
            return jsonify({
                "success": True, "type": "intent",
                "intent": intent_match['tag'], "message": message,
                "response": bot_response, "model": model_used,
            }), 200

        # ========== 5.5 PRODUCT SEARCH FALLBACK ==========
        # Short messages without intent might be product/category names
        if len(message.split()) <= 4 and not intent_match:
            products = lookup_product(message)
            if products:
                bot_response = format_product(products)
                logger.info(f"Product fallback search: {message[:50]} ({len(products)} found)")
                return _db_response(bot_response, "product_info", "product_lookup")

        # ========== 6. FINAL FALLBACK ==========
        # If model wasn't available for primary generation, return its fallback text.
        if api_response.get('type') == 'generation':
            response_data = {
                "success": True, "type": "generation",
                "message": message, "response": api_response['result'],
                "model": api_response['model'],
            }
            log_conversation(
                session_id=session_id, user_message=message,
                bot_response=api_response['result'], intent=None,
                model_used=api_response['model'], response_type="generation",
                ip_address=ip_address, response_time_ms=_elapsed_ms(),
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
                response_time_ms=_elapsed_ms(),
            )

        logger.info(f"Response generated successfully (type: {api_response['type']})")
        _store_context(response_data.get('response', ''))
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
