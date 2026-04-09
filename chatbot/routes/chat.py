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
from ..database import supabase
from ..services.intent_service import match_intent, INTENTS
from ..services.entity_service import (
    extract_order_number, extract_email, extract_sku, extract_product_name,
)
from ..services.lookup_service import (
    lookup_order_status, lookup_orders_by_email,
    lookup_product, lookup_customer_by_email, list_products,
    get_live_tracking, create_order_from_chat,
)
from ..services.formatter_service import (
    format_order, format_orders_list, format_product,
    format_product_list, format_customer, format_order_created,
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
_RE_ORDER_TRACK_REQUEST = re.compile(
    r'\b(track|where|status|check|show|view|list)\b[^\n]{0,40}\borders?\b|\border\s*(?:status|tracking)\b',
    re.IGNORECASE,
)
_RE_RETURN_REQUEST = re.compile(r'\b(return|refund|send\s+back)\b', re.IGNORECASE)
_RE_RETURN_POLICY_REQUEST = re.compile(r'\b(return|refund)\s+policy\b', re.IGNORECASE)
_RE_CREATE_ORDER_REQUEST = re.compile(
    r'\b(buy|purchase|place\s+an?\s+order|order\s+now|i\s+want\s+to\s+order)\b',
    re.IGNORECASE,
)
_VAGUE_TERMS = {
    'help', 'something', 'anything', 'stuff', 'thing', 'things', 'details', 'info',
    'information', 'more', 'that', 'this', 'it', 'there', 'whatever', 'issue', 'problem',
    'question', 'questions', 'show me', 'tell me more', 'what about', 'can you', 'assist',
}
_RE_QUANTITY = re.compile(r'\b(?:qty|quantity|x)\s*[:=]?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:units?|pcs|pieces)\b', re.IGNORECASE)
_RE_NAME = re.compile(r'\b(?:name\s+is|i\s+am|this\s+is)\s+([A-Za-z][A-Za-z\s]{1,50})\b', re.IGNORECASE)

# ── Conversation context (in-memory, per session) ────────
# Stores last MAX_CONTEXT_TURNS exchanges per session_id.
MAX_CONTEXT_TURNS = 5
_conversation_context = defaultdict(list)  # session_id -> [{user, bot}, ...]
_pending_order_drafts = {}  # session_id -> {product_query, quantity, customer_name, created_at}
_PENDING_ORDER_TTL_SECONDS = 20 * 60


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

        def _normalize_product_cards(products):
            cards = []
            for p in products or []:
                if not isinstance(p, dict):
                    continue
                cards.append({
                    'id': p.get('id'),
                    'name': p.get('name') or p.get('product_name'),
                    'sku': p.get('sku') or p.get('product_sku'),
                    'price': p.get('price') if p.get('price') is not None else p.get('unit_price'),
                    'stock': p.get('stock'),
                    'image_url': p.get('image_url'),
                })
            return cards

        def _fetch_product_image_map(order):
            image_map = {}
            if not supabase or not order:
                return image_map

            items = order.get('order_items') or []
            product_ids = [item.get('product_id') for item in items if item.get('product_id')]
            product_ids = [pid for pid in product_ids if pid]

            if product_ids:
                try:
                    result = supabase.table('products').select('id,name,sku,image_url').in_('id', product_ids).execute()
                    for p in result.data or []:
                        image_map[p.get('id')] = p
                except Exception as exc:
                    logger.debug(f"Failed to fetch product images by id: {exc}")

            # Fallback by SKU if product_id is missing.
            for item in items:
                if item.get('product_id') in image_map:
                    continue
                sku_value = item.get('product_sku')
                if not sku_value:
                    continue
                try:
                    result = supabase.table('products').select('id,name,sku,image_url').eq('sku', sku_value).limit(1).execute()
                    if result.data:
                        image_map[sku_value] = result.data[0]
                except Exception as exc:
                    logger.debug(f"Failed to fetch product image by sku: {exc}")

            return image_map

        def _order_products_for_ui(order):
            items = (order or {}).get('order_items') or []
            image_map = _fetch_product_image_map(order)
            cards = []
            for item in items:
                source = image_map.get(item.get('product_id')) or image_map.get(item.get('product_sku')) or {}
                cards.append({
                    'id': item.get('product_id') or source.get('id'),
                    'name': item.get('product_name') or source.get('name'),
                    'sku': item.get('product_sku') or source.get('sku'),
                    'price': item.get('unit_price'),
                    'stock': None,
                    'image_url': source.get('image_url'),
                })
            return cards

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
                        'create_order',
                    }:
                        return parsed
                except Exception:
                    continue
            return None

        def _extract_quantity(raw_message):
            match = _RE_QUANTITY.search(raw_message)
            if not match:
                return 1
            qty = match.group(1) or match.group(2)
            try:
                return max(1, min(int(qty), 50))
            except Exception:
                return 1

        def _extract_customer_name(raw_message, customer_email):
            match = _RE_NAME.search(raw_message)
            if match:
                return match.group(1).strip().title()
            if customer_email and '@' in customer_email:
                local = customer_email.split('@', 1)[0].replace('.', ' ').replace('_', ' ').strip()
                if local:
                    return local.title()
            return 'Customer'

        def _extract_order_product_query(raw_message):
            text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', '', raw_message)
            text = re.sub(r'\b(?:buy|purchase|place\s+an?\s+order|order\s+now|i\s+want\s+to\s+order)\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(?:qty|quantity|x)\s*[:=]?\s*\d{1,3}\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\b\d{1,3}\s*(?:units?|pcs|pieces)\b', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\b(?:for|email|to|please|name\s+is|i\s+am|this\s+is)\b', ' ', text, flags=re.IGNORECASE)
            text = re.sub(r'\s+', ' ', text).strip(' .,:;')
            return text[:80] if text else None

        def _token_in_message(token, raw_message):
            if not token:
                return False
            token_norm = re.sub(r'[^A-Za-z0-9]', '', token).lower()
            msg_norm = re.sub(r'[^A-Za-z0-9]', '', raw_message or '').lower()
            return bool(token_norm and token_norm in msg_norm)

        def _needs_clarification(raw_message):
            text = (raw_message or '').strip().lower()
            if not text:
                return True

            words = [w for w in re.split(r'\s+', text) if w]
            compact = re.sub(r'[^a-z0-9]+', ' ', text)

            if len(words) <= 2:
                return True

            if any(term in text for term in _VAGUE_TERMS):
                return True

            if len(words) <= 4 and not (
                order_number
                or email
                or sku
                or product_name
                or _RE_PRODUCT_HINT.search(message)
            ):
                return True

            return False

        def _clarify_message():
            if order_number or email or explicit_track_request or explicit_return_request:
                return (
                    "I can help with that, but I need your order number or the email used for the order. "
                    "For example: **ORD-2026-001**."
                ), 'order_tracking', 'clarification'

            if sku or product_name or _RE_PRODUCT_HINT.search(message):
                return (
                    "Which product do you mean? Share the product name or SKU and I’ll pull up the details."
                ), 'product_info', 'clarification'

            return (
                "Can you tell me a bit more about what you need help with? I can help with orders, returns, shipping, and products."
            ), None, 'clarification'

        def _get_pending_order_draft():
            draft = _pending_order_drafts.get(session_id)
            if not draft:
                return None
            age = time.time() - draft.get('created_at', 0)
            if age > _PENDING_ORDER_TTL_SECONDS:
                _pending_order_drafts.pop(session_id, None)
                return None
            return draft

        def _set_pending_order_draft(product_query, quantity, customer_name):
            _pending_order_drafts[session_id] = {
                'product_query': product_query,
                'quantity': quantity,
                'customer_name': customer_name,
                'created_at': time.time(),
            }

        def _clear_pending_order_draft():
            _pending_order_drafts.pop(session_id, None)

        # Complete a pending order draft when the user provides email in a follow-up message.
        pending_draft = _get_pending_order_draft()
        if pending_draft and email:
            creation_result = create_order_from_chat(
                customer_name=pending_draft.get('customer_name') or _extract_customer_name(message, email),
                customer_email=email,
                product_query=pending_draft.get('product_query'),
                quantity=pending_draft.get('quantity') or 1,
            )

            if creation_result.get('success'):
                _clear_pending_order_draft()
                created_order = creation_result.get('order')
                email_sent = creation_result.get('email_sent')
                bot_response = format_order_created(created_order, email_sent=email_sent)
                return _db_response(
                    bot_response,
                    "order_create",
                    "order_created",
                    {"order": created_order, "email_sent": email_sent},
                )

            bot_response = creation_result.get('error') or (
                "I couldn't place your order right now. Please try again in a moment."
            )
            return _db_response(bot_response, "order_create", "order_create_failed")

        # Ask for required details before planner routing to avoid accidental lookups.
        explicit_track_request = bool(_RE_ORDER_TRACK_REQUEST.search(message))
        explicit_return_request = bool(_RE_RETURN_REQUEST.search(message)) and not bool(_RE_RETURN_POLICY_REQUEST.search(message))

        if (intent_tag in ('order_tracking', 'order_status') or explicit_track_request) and not order_number and not email:
            bot_response = (
                "I can track that for you. Please share your order number "
                "(example: **ORD-2026-001**) or the email used for the order."
            )
            return _db_response(bot_response, 'order_tracking', 'order_tracking_missing_details')

        if (intent_tag == 'returns' or explicit_return_request) and not order_number and not email and not _RE_RETURN_POLICY_REQUEST.search(message):
            bot_response = (
                "I can help with a return. Please share your order number "
                "(example: **ORD-2026-001**) or the email used when ordering."
            )
            return _db_response(bot_response, 'returns', 'returns_missing_details')

        # ========== 0. PRIMARY: AI MODEL GENERATION ==========
        # Let the model choose whether to call a DB lookup tool first.
        # Intent and DB routes are fallback when model generation is unavailable.
        context = _conversation_context.get(session_id, [])

        planner_prompt = (
            "You are a routing planner for customer support tools. "
            "Choose one DB action for the user's message. "
            "Return ONLY valid JSON with keys: action, query, order_number, email. "
            "Allowed actions: none, list_products, list_stock_products, search_products, "
            "lookup_order, lookup_orders_by_email, lookup_customer_by_email, create_order.\n\n"
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
            if _RE_CREATE_ORDER_REQUEST.search(message):
                return 'create_order'
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

        vague_request = _needs_clarification(message)
        if vague_request and not (
            order_number
            or email
            or sku
            or product_name
            or _RE_PRODUCT_HINT.search(message)
            or _RE_PRODUCT_LIST_REQUEST.search(message)
            or _RE_STOCK_LIST_REQUEST.search(message)
            or _RE_CREATE_ORDER_REQUEST.search(message)
            or explicit_track_request
            or explicit_return_request
        ):
            if intent_tag == 'help':
                bot_response = (
                    "Tell me what you need help with, and I’ll point you to the right answer. "
                    "I can help with orders, returns, shipping, products, and account questions."
                )
                return _db_response(bot_response, 'help', 'clarification')

            bot_response, clarification_intent, response_type = _clarify_message()
            return _db_response(bot_response, clarification_intent, response_type)

        if action == 'create_order':
            create_email = plan_email or email
            create_product_query = plan_query or sku or product_name or _extract_order_product_query(message)
            create_quantity = _extract_quantity(message)
            create_name = _extract_customer_name(message, create_email or email)

            if not create_product_query:
                bot_response = (
                    "I can place the order for you. Please tell me which product you want, "
                    "for example: `buy iPhone 15 qty 1`."
                )
                return _db_response(bot_response, "order_create", "order_create_missing_details")

            if not create_email:
                _set_pending_order_draft(
                    product_query=create_product_query,
                    quantity=create_quantity,
                    customer_name=create_name,
                )
                bot_response = (
                    f"Great choice. I can place **{create_quantity} x {create_product_query}**. "
                    "Please share your email to complete the order."
                )
                return _db_response(bot_response, "order_create", "order_create_waiting_email")

            creation_result = create_order_from_chat(
                customer_name=create_name,
                customer_email=create_email,
                product_query=create_product_query,
                quantity=create_quantity,
            )

            if creation_result.get('success'):
                created_order = creation_result.get('order')
                email_sent = creation_result.get('email_sent')
                bot_response = format_order_created(created_order, email_sent=email_sent)
                return _db_response(
                    bot_response,
                    "order_create",
                    "order_created",
                    {
                        "order": created_order,
                        "email_sent": email_sent,
                        "products": _order_products_for_ui(created_order),
                    },
                )

            bot_response = creation_result.get('error') or (
                "I couldn't place your order right now. Please try again in a moment."
            )
            return _db_response(bot_response, "order_create", "order_create_failed")

        if action == 'list_stock_products':
            stock_products = list_products(in_stock_only=True)
            if stock_products:
                return _db_response(
                    format_product_list(stock_products),
                    "stock_availability",
                    "product_list",
                    {"products": _normalize_product_cards(stock_products)},
                )

        if action == 'list_products':
            all_products = list_products()
            if all_products:
                return _db_response(
                    format_product_list(all_products),
                    "product_info",
                    "product_list",
                    {"products": _normalize_product_cards(all_products)},
                )

        if action == 'search_products':
            search_term = plan_query or sku or product_name or message
            products = lookup_product(search_term)
            if products:
                return _db_response(
                    format_product(products),
                    "product_info",
                    "product_lookup",
                    {"products": _normalize_product_cards(products)},
                )

        if action == 'lookup_order':
            lookup_num = order_number
            if not lookup_num and _token_in_message(plan_order, message):
                lookup_num = plan_order

            if not lookup_num:
                bot_response = (
                    "I can track that for you. Please share your order number "
                    "(example: **ORD-2026-001**)."
                )
                return _db_response(bot_response, "order_tracking", "order_tracking_missing_number")

            order = lookup_order_status(lookup_num)
            if order:
                # Fetch live tracking if tracking number available
                live_tracking = None
                if order.get('tracking_number'):
                    live_tracking = get_live_tracking(
                        order['tracking_number'],
                        expected_status=order.get('status'),
                    )
                return _db_response(
                    format_order(order, live_tracking=live_tracking),
                    "order_tracking",
                    "order_lookup",
                    {
                        "order": order,
                        "products": _order_products_for_ui(order),
                    },
                )
            bot_response = (
                f"❌ Sorry, I couldn't find order **{lookup_num}** in our system. "
                "Please check the order number and try again. Or contact support@company.com for assistance."
            )
            return _db_response(bot_response, "order_tracking", "order_not_found")

        if action == 'lookup_orders_by_email':
            lookup_email = email
            if not lookup_email and _token_in_message(plan_email, message):
                lookup_email = plan_email
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
            lookup_email = email
            if not lookup_email and _token_in_message(plan_email, message):
                lookup_email = plan_email
            if lookup_email:
                customer = lookup_customer_by_email(lookup_email)
                if customer:
                    return _db_response(format_customer(customer), "account", "customer_lookup")
                bot_response = f"I couldn't find an account associated with **{lookup_email}**. Would you like help creating one?"
                return _db_response(bot_response, "account", "customer_not_found")

        # If planner returns none for a product-like short prompt, force a DB lookup.
        if action == 'none' and not order_number and not email:
            looks_like_product = (
                bool(sku)
                or bool(product_name)
                or bool(_RE_PRODUCT_HINT.search(message))
                or len(message.split()) <= 4
            )
            if looks_like_product:
                products = lookup_product(message)
                if products:
                    bot_response = format_product(products)
                    logger.info(f"Product override lookup (planner=none): {message[:50]} ({len(products)} found)")
                    return _db_response(
                        bot_response,
                        "product_info",
                        "product_lookup",
                        {"products": _normalize_product_cards(products)},
                    )

        api_response = query_model(message, context=context)
        model_generation_ready = (
            api_response.get('type') == 'generation'
            and bool((api_response.get('result') or '').strip())
            and api_response.get('model') != 'fallback'
        )

        if model_generation_ready and not intent_match:
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
                return _db_response(
                    bot_response,
                    "stock_availability",
                    "product_list",
                    {"products": _normalize_product_cards(stock_products)},
                )
            bot_response = (
                "I couldn't fetch in-stock products right now. "
                "Please try again in a moment, or ask for a specific product name or SKU."
            )
            return _db_response(bot_response, "stock_availability", "product_catalog_unavailable")

        if _RE_PRODUCT_LIST_REQUEST.search(message):
            all_products = list_products()
            if all_products:
                bot_response = format_product_list(all_products)
                return _db_response(
                    bot_response,
                    "product_info",
                    "product_list",
                    {"products": _normalize_product_cards(all_products)},
                )
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
                return _db_response(
                    bot_response,
                    "order_tracking",
                    "order_lookup",
                    {
                        "order": order,
                        "products": _order_products_for_ui(order),
                    },
                )
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

        # ========== 2.5 PRODUCT LOOKUP OVERRIDE (short/product-like prompts) ==========
        # Catch messages like "iphone 15" that may otherwise map to a generic intent.
        if not order_number and not email:
            looks_like_product = (
                bool(sku)
                or bool(product_name)
                or bool(_RE_PRODUCT_HINT.search(message))
                or len(message.split()) <= 4
            )
            if looks_like_product:
                products = lookup_product(message)
                if products:
                    bot_response = format_product(products)
                    logger.info(f"Product override lookup: {message[:50]} ({len(products)} found)")
                    return _db_response(
                        bot_response,
                        "product_info",
                        "product_lookup",
                        {"products": _normalize_product_cards(products)},
                    )

        # ========== 3. PRODUCT LOOKUP (by SKU or name) ==========
        if intent_tag in ('product_info', 'pricing', 'stock_availability', 'size_fitting'):
            search_term = sku or product_name
            products = None

            if search_term:
                products = lookup_product(search_term)
                if products:
                    bot_response = format_product(products)
                    logger.info(f"Product lookup: {search_term} ({len(products)} found)")
                    return _db_response(
                        bot_response,
                        intent_tag,
                        "product_lookup",
                        {"products": _normalize_product_cards(products)},
                    )
                logger.debug(f"[DB_LOOKUP_MISS] sku/product_name={search_term!r} — no product match in DB")

            # Always try the full message even if an entity was extracted but returned no matches.
            products = lookup_product(message)
            if products:
                bot_response = format_product(products)
                logger.info(f"Product search (raw message): {message[:50]} ({len(products)} found)")
                return _db_response(
                    bot_response,
                    intent_tag,
                    "product_lookup",
                    {"products": _normalize_product_cards(products)},
                )

            # For stock intent, surface in-stock catalog when no direct match was found.
            if intent_tag == 'stock_availability':
                all_products = list_products(in_stock_only=True)
                if all_products:
                    bot_response = format_product_list(all_products)
                    return _db_response(
                        bot_response,
                        intent_tag,
                        "product_list",
                        {"products": _normalize_product_cards(all_products)},
                    )

            # For other product intents, surface general catalog before generic intent fallback.
            all_products = list_products()
            if all_products:
                bot_response = format_product_list(all_products)
                return _db_response(
                    bot_response,
                    intent_tag,
                    "product_list",
                    {"products": _normalize_product_cards(all_products)},
                )

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
                return _db_response(
                    bot_response,
                    "product_info",
                    "product_lookup",
                    {"products": _normalize_product_cards(products)},
                )

            all_products = list_products()
            if all_products:
                bot_response = format_product_list(all_products)
                return _db_response(
                    bot_response,
                    "product_info",
                    "product_list",
                    {"products": _normalize_product_cards(all_products)},
                )

        # ========== 4. ORDER TRACKING (no order number) ==========
        # Intent responses may be enhanced by the AI model when available.
        if intent_tag == 'order_tracking':
            if not order_number and not email:
                bot_response = (
                    "I can track that for you. Please share your order number "
                    "(example: **ORD-2026-001**) or the email used for the order."
                )
                return _db_response(bot_response, intent_tag, "order_tracking_missing_details")

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

        if intent_tag == 'returns' and not order_number and not email and not _RE_RETURN_POLICY_REQUEST.search(message):
            bot_response = (
                "I can help with a return. Please share your order number "
                "(example: **ORD-2026-001**) or the email used when ordering."
            )
            return _db_response(bot_response, intent_tag, "returns_missing_details")

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
                return _db_response(
                    bot_response,
                    "product_info",
                    "product_lookup",
                    {"products": _normalize_product_cards(products)},
                )

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
