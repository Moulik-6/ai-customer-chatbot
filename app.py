import os
import json
import random
import re
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app)

# Supabase setup
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
supabase: Client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")
else:
    logger.warning("Supabase credentials not found - database features disabled")

# SQLite fallback for local development
DB_PATH = os.path.join(os.path.dirname(__file__), 'chatbot.db')


def _get_db():
    """Get a SQLite connection with row factory enabled. Uses check_same_thread=False for Flask."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
    return conn

def init_database():
    """Initialize SQLite database for conversation logging"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            intent TEXT,
            model_used TEXT,
            response_type TEXT,
            ip_address TEXT
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id)
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

def log_conversation(session_id, user_message, bot_response, intent=None, model_used=None, response_type=None, ip_address=None):
    """Log a conversation to the database (Supabase or SQLite fallback)"""
    try:
        logger.info(f"Attempting to log conversation - session: {session_id}, type: {response_type}")
        
        # Try Supabase first
        if supabase:
            data = {
                "session_id": session_id,
                "user_message": user_message,
                "bot_response": bot_response,
                "intent": intent,
                "model_used": model_used,
                "response_type": response_type,
                "ip_address": ip_address
            }
            result = supabase.table('conversations').insert(data).execute()
            logger.info(f"Successfully logged conversation to Supabase")
            return
        
        # SQLite fallback
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (session_id, user_message, bot_response, intent, model_used, response_type, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, user_message, bot_response, intent, model_used, response_type, ip_address))
        
        conn.commit()
        logger.info(f"Successfully logged conversation to SQLite - ID: {cursor.lastrowid}")
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log conversation: {e}", exc_info=True)

# Initialize database on startup
init_database()

# Configuration
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
MOCK_MODE = os.getenv('MOCK_MODE', 'false').strip().lower() in ('1', 'true', 'yes')
MODEL_TYPE = 'generation'
HUGGINGFACE_MODEL = 'google/flan-t5-xl'  # 3B params for high-quality customer service responses
USE_LOCAL_MODEL = os.getenv('USE_LOCAL_MODEL', 'true').strip().lower() in ('1', 'true', 'yes')

# Model configurations
MODEL_CONFIGS = {
    'generation': {
        'default': 'gpt2',
        'alternatives': ['distilgpt2', 'mistralai/Mistral-7B-Instruct-v0.1'],
        'params': {
            'max_length': 150,
            'temperature': 0.7,
            'top_p': 0.9,
        }
    },
    'classification': {
        'default': 'distilbert-base-uncased-finetuned-sst-2-english',
        'alternatives': ['bert-base-uncased'],
        'params': {
            'top_k': 2,
        }
    }
}

HUGGINGFACE_API_BASE = os.getenv(
    'HUGGINGFACE_API_BASE',
    'https://router.huggingface.co/hf-inference/models'
)
HUGGINGFACE_API_URL = f"{HUGGINGFACE_API_BASE}/{HUGGINGFACE_MODEL}"

# Validate API key on startup (required only for remote API usage)
if not HUGGINGFACE_API_KEY and not MOCK_MODE and not USE_LOCAL_MODEL:
    logger.error("HUGGINGFACE_API_KEY not found in environment variables")
    raise ValueError("HUGGINGFACE_API_KEY environment variable is required")

if os.getenv('HUGGINGFACE_MODEL') and os.getenv('HUGGINGFACE_MODEL') != HUGGINGFACE_MODEL:
    logger.warning("Overriding HUGGINGFACE_MODEL to google/flan-t5-base")
if os.getenv('MODEL_TYPE') and os.getenv('MODEL_TYPE') != MODEL_TYPE:
    logger.warning("Overriding MODEL_TYPE to generation")

logger.info(
    f"Initialized with model: {HUGGINGFACE_MODEL} (Type: {MODEL_TYPE}, Mock: {MOCK_MODE})"
)

INTENTS_PATH = os.path.join(os.path.dirname(__file__), 'intents.json')


def _load_intents():
    try:
        with open(INTENTS_PATH, 'r', encoding='utf-8') as handle:
            data = json.load(handle)
        return data.get('intents', [])
    except FileNotFoundError:
        logger.warning("intents.json not found; intent matching disabled")
        return []
    except json.JSONDecodeError as exc:
        logger.error(f"Failed to parse intents.json: {exc}")
        return []


INTENTS = _load_intents()
logger.info(f"Loaded intents: {len(INTENTS)}")

# Precompile normalization regex patterns (avoid recompiling per message)
_RE_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_RE_MULTI_SPACE = re.compile(r"\s+")


def _normalize_text(value):
    normalized = value.lower()
    normalized = _RE_NON_ALNUM.sub(" ", normalized)
    return _RE_MULTI_SPACE.sub(" ", normalized).strip()


# Precompile intent patterns at startup for O(1) regex matching per pattern
_COMPILED_INTENTS = []
for _intent in INTENTS:
    _patterns = _intent.get('patterns', [])
    _responses = _intent.get('responses', [])
    if not _patterns or not _responses:
        continue
    _compiled_patterns = []
    for _p in _patterns:
        _norm = _normalize_text(_p)
        if _norm:
            _compiled_patterns.append(re.compile(r"\b" + re.escape(_norm) + r"\b"))
    if _compiled_patterns:
        _COMPILED_INTENTS.append({
            'tag': _intent.get('tag', 'unknown'),
            'responses': _responses,
            'patterns': _compiled_patterns
        })
logger.info(f"Precompiled {len(_COMPILED_INTENTS)} intent patterns")


def _match_intent_response(message):
    if not _COMPILED_INTENTS:
        return None
    normalized = _normalize_text(message)
    for intent in _COMPILED_INTENTS:
        for pattern_re in intent['patterns']:
            if pattern_re.search(normalized):
                return {
                    'tag': intent['tag'],
                    'response': random.choice(intent['responses'])
                }
    return None


# Precompile order number regex
_RE_ORDER_NUMBER = re.compile(r'ORD[-\s]?\d{4}[-\s]?\d{3,4}', re.IGNORECASE)
_RE_EMAIL = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
_RE_SKU = re.compile(r'\b[A-Z]{2,}[-][A-Z0-9][-A-Z0-9]{2,}\b')
_RE_PHONE = re.compile(r'(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')


def _extract_order_number(message):
    """Extract order number from message (format: ORD-XXXX-XXX or similar)"""
    match = _RE_ORDER_NUMBER.search(message)
    if match:
        return match.group(0).replace(' ', '-').upper()
    return None


def _extract_email(message):
    """Extract email address from message"""
    match = _RE_EMAIL.search(message)
    return match.group(0).lower() if match else None


def _extract_sku(message):
    """Extract product SKU from message (e.g., IPHONE-15-PRO)"""
    match = _RE_SKU.search(message.upper())
    return match.group(0) if match else None


def _extract_product_name(message):
    """Extract potential product name from message using keyword hints"""
    # Look for phrases after common product-inquiry triggers
    triggers = [
        r'(?:about|for|on|called|named)\s+["\']?(.{3,40}?)["\']?\s*(?:\?|$|\.)',
        r'(?:price of|cost of|details on|info on|stock of)\s+["\']?(.{3,40}?)["\']?\s*(?:\?|$|\.)',
        r'(?:do you (?:have|sell|carry))\s+["\']?(.{3,40}?)["\']?\s*(?:\?|$|\.)',
    ]
    for pattern in triggers:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('"\'')
    return None


# ========== DATABASE LOOKUP FUNCTIONS ==========

def _lookup_order_status(order_number):
    """Lookup order status from Supabase by order number"""
    try:
        if not supabase:
            return None
        
        result = supabase.table('orders').select('*,order_items(*)').eq('order_number', order_number).execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]
        
        return None
    except Exception as e:
        logger.error(f"Error looking up order: {e}")
        return None


def _lookup_orders_by_email(email):
    """Lookup all orders for a customer by email"""
    try:
        if not supabase:
            return None
        
        result = (supabase.table('orders')
                  .select('*,order_items(*)')
                  .eq('customer_email', email)
                  .order('order_date', desc=True)
                  .limit(5)
                  .execute())
        
        if result.data and len(result.data) > 0:
            return result.data
        
        return None
    except Exception as e:
        logger.error(f"Error looking up orders by email: {e}")
        return None


def _lookup_product(query):
    """Lookup product by name (fuzzy) or SKU (exact)"""
    try:
        if not supabase:
            return None
        
        # Try exact SKU match first
        sku = _extract_sku(query) if query == query.upper() else _extract_sku(query.upper())
        if sku:
            result = supabase.table('products').select('*').eq('sku', sku).execute()
            if result.data and len(result.data) > 0:
                return result.data
        
        # Fuzzy name search
        result = (supabase.table('products')
                  .select('*')
                  .or_(f"name.ilike.%{query}%,description.ilike.%{query}%,sku.ilike.%{query}%")
                  .limit(3)
                  .execute())
        
        if result.data and len(result.data) > 0:
            return result.data
        
        return None
    except Exception as e:
        logger.error(f"Error looking up product: {e}")
        return None


def _lookup_customer_by_email(email):
    """Lookup customer info by finding their orders"""
    try:
        if not supabase:
            return None
        
        result = (supabase.table('orders')
                  .select('customer_name, customer_email, customer_phone, shipping_address, status, order_number, total_amount, order_date')
                  .eq('customer_email', email)
                  .order('order_date', desc=True)
                  .limit(10)
                  .execute())
        
        if result.data and len(result.data) > 0:
            customer = {
                'name': result.data[0].get('customer_name'),
                'email': email,
                'phone': result.data[0].get('customer_phone'),
                'address': result.data[0].get('shipping_address'),
                'total_orders': len(result.data),
                'orders': result.data
            }
            return customer
        
        return None
    except Exception as e:
        logger.error(f"Error looking up customer: {e}")
        return None


# ========== RESPONSE FORMATTERS ==========

def _format_order_response(order):
    """Format order data into a customer-friendly response"""
    if not order:
        return None
    
    status = order.get('status', 'unknown').upper()
    customer = order.get('customer_name', 'Customer')
    total = order.get('total_amount', 0)
    tracking = order.get('tracking_number')
    order_date = order.get('order_date', '')
    items = order.get('order_items', [])
    
    # Status emoji
    status_emoji = {
        'PENDING': '⏳',
        'PROCESSING': '🔄',
        'SHIPPED': '📦',
        'DELIVERED': '✅',
        'CANCELLED': '❌'
    }.get(status, '📋')
    
    response = f"{status_emoji} **Order Status: {status}**\n"
    response += f"Order #: {order['order_number']}\n"
    response += f"Total: ${total:.2f}\n"
    
    if tracking:
        response += f"Tracking: {tracking}\n"
    
    response += f"\n**Items ({len(items)}):**\n"
    for item in items:
        response += f"• {item['product_name']} x{item['quantity']} @ ${item['unit_price']:.2f}\n"
    
    if status == 'SHIPPED':
        response += "\n📬 Your order is on the way! Use your tracking number to get delivery updates."
    elif status == 'DELIVERED':
        response += "\n🎉 Your order has been delivered!"
    elif status == 'PROCESSING':
        response += "\n⚙️ We're preparing your order for shipment. You'll receive tracking info soon."
    elif status == 'PENDING':
        response += "\n👀 Your order is confirmed and being prepared."
    elif status == 'CANCELLED':
        response += "\n✋ This order has been cancelled."
    
    return response


def _format_orders_list_response(orders, email):
    """Format multiple orders for a customer"""
    if not orders:
        return None
    
    status_emoji = {
        'pending': '⏳', 'processing': '🔄', 'shipped': '📦',
        'delivered': '✅', 'cancelled': '❌'
    }
    
    response = f"📋 **Orders for {email}** ({len(orders)} found):\n\n"
    for order in orders:
        status = order.get('status', 'unknown')
        emoji = status_emoji.get(status, '📋')
        total = order.get('total_amount', 0)
        date = order.get('order_date', '')[:10]
        response += f"{emoji} **{order['order_number']}** — {status.upper()} — ${total:.2f} ({date})\n"
    
    response += "\nTo see details for a specific order, provide the order number (e.g., ORD-2026-001)."
    return response


def _format_product_response(products):
    """Format product data into a customer-friendly response"""
    if not products:
        return None
    
    if len(products) == 1:
        p = products[0]
        stock_status = "✅ In Stock" if p.get('stock', 0) > 0 else "❌ Out of Stock"
        response = f"🛍️ **{p['name']}**\n"
        if p.get('description'):
            response += f"{p['description']}\n"
        response += f"💰 Price: ${p['price']:.2f}\n"
        response += f"📦 {stock_status}"
        if p.get('stock', 0) > 0:
            response += f" ({p['stock']} available)"
        response += "\n"
        if p.get('sku'):
            response += f"SKU: {p['sku']}\n"
        if p.get('category'):
            response += f"Category: {p['category']}\n"
        return response
    
    # Multiple products
    response = f"🔍 **Found {len(products)} products:**\n\n"
    for p in products:
        stock_status = "In Stock" if p.get('stock', 0) > 0 else "Out of Stock"
        response += f"• **{p['name']}** — ${p['price']:.2f} ({stock_status})\n"
    
    response += "\nWould you like more details about any of these products?"
    return response


def _format_customer_response(customer):
    """Format customer data into a response"""
    if not customer:
        return None
    
    response = f"👤 **Customer: {customer['name']}**\n"
    response += f"📧 Email: {customer['email']}\n"
    if customer.get('phone'):
        response += f"📱 Phone: {customer['phone']}\n"
    if customer.get('address'):
        response += f"📍 Address: {customer['address']}\n"
    response += f"🛒 Total Orders: {customer['total_orders']}\n"
    
    if customer['orders']:
        response += "\n**Recent Orders:**\n"
        status_emoji = {
            'pending': '⏳', 'processing': '🔄', 'shipped': '📦',
            'delivered': '✅', 'cancelled': '❌'
        }
        for order in customer['orders'][:3]:
            status = order.get('status', 'unknown')
            emoji = status_emoji.get(status, '📋')
            response += f"{emoji} {order['order_number']} — {status.upper()} — ${order['total_amount']:.2f}\n"
    
    return response


def _build_flan_prompt(message):
    """Build a structured few-shot prompt for FLAN-T5 to produce professional responses."""
    return (
        "You are a professional, friendly customer support assistant for an online store. "
        "Rules: Be concise (2-3 sentences max). Be helpful and empathetic. "
        "Never invent policies, prices, or order details. "
        "If you need more information, ask the customer politely.\n\n"
        "Example 1:\n"
        "Customer: I want to return my purchase\n"
        "Assistant: I'd be happy to help with your return! Our return policy allows returns within 30 days of purchase. "
        "Could you please provide your order number so I can look into this for you?\n\n"
        "Example 2:\n"
        "Customer: My package hasn't arrived yet\n"
        "Assistant: I'm sorry to hear about the delay. To check the status of your delivery, "
        "could you share your order number? I'll look into it right away.\n\n"
        "Example 3:\n"
        "Customer: Do you have any discounts?\n"
        "Assistant: We regularly offer promotions and seasonal discounts! "
        "I'd recommend checking our website or subscribing to our newsletter for the latest deals.\n\n"
        f"Customer: {message}\n"
        "Assistant:"
    )

# Load local model if not in mock mode and enabled
LOCAL_MODEL = None
if not MOCK_MODE and USE_LOCAL_MODEL:
    try:
        logger.info(f"Loading local model: {HUGGINGFACE_MODEL}")
        device = 0 if torch.cuda.is_available() else -1
        
        # Check if using a seq2seq model (like FLAN-T5)
        is_seq2seq = 'flan' in HUGGINGFACE_MODEL.lower() or 't5' in HUGGINGFACE_MODEL.lower()
        
        if is_seq2seq:
            # For seq2seq models like FLAN-T5, load directly with model class
            logger.info(f"Loading seq2seq model: {HUGGINGFACE_MODEL}")
            tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(HUGGINGFACE_MODEL)
            if device >= 0:
                model = model.cuda()
            LOCAL_MODEL = {'tokenizer': tokenizer, 'model': model, 'type': 'seq2seq'}
        elif MODEL_TYPE == 'classification':
            LOCAL_MODEL = pipeline('text-classification', model=HUGGINGFACE_MODEL, device=device)
        else:  # text-generation
            LOCAL_MODEL = pipeline('text-generation', model=HUGGINGFACE_MODEL, device=device)
        
        logger.info(f"Local model loaded successfully on {'GPU' if device >= 0 else 'CPU'}")
    except Exception as e:
        logger.error(f"Failed to load local model: {str(e)}")
        raise ValueError(f"Could not load local model: {str(e)}")


def query_huggingface(prompt):
    """
    Query Hugging Face API with the given prompt.
    Supports both text generation (GPT2) and classification (DistilBERT) models.
    
    Args:
        prompt (str): The input text for the model
        
    Returns:
        dict: Response containing:
            - 'type': 'generation' or 'classification'
            - 'result': Generated text OR classification scores
            - 'model': Model used
            
    Raises:
        requests.RequestException: If API request fails
        ValueError: If response format is invalid
        TimeoutError: If request times out
    """
    if MOCK_MODE:
        return _mock_response(prompt)

    if LOCAL_MODEL:
        return _local_model_response(prompt)

    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    
    # Build request payload based on model type
    if MODEL_TYPE == 'generation':
        payload = {
            "inputs": prompt,
            "parameters": MODEL_CONFIGS['generation']['params']
        }
    else:  # classification
        payload = {
            "inputs": prompt,
            "parameters": MODEL_CONFIGS['classification']['params']
        }
    
    try:
        logger.debug(f"Sending request to {HUGGINGFACE_API_URL}")
        
        response = requests.post(
            HUGGINGFACE_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Handle HTTP errors
        if response.status_code == 401:
            logger.error("Authentication failed - invalid API key")
            raise ValueError("Invalid Hugging Face API key")
        elif response.status_code == 429:
            logger.error("Rate limit exceeded")
            raise requests.RequestException("API rate limit exceeded. Please try again later.")
        elif response.status_code == 503:
            logger.error("Model is loading or temporarily unavailable")
            raise requests.RequestException("Model is loading. Please try again in a moment.")
        
        response.raise_for_status()
        
        result = response.json()
        logger.debug(f"Received response: {str(result)[:200]}")
        
        # Parse response based on model type
        if MODEL_TYPE == 'generation':
            return _parse_generation_response(result, prompt)
        else:
            return _parse_classification_response(result)
        
    except requests.Timeout as e:
        logger.error(f"Hugging Face API request timed out after 30 seconds")
        raise TimeoutError("Request to AI service timed out") from e
    except requests.ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        raise requests.RequestException("Failed to connect to AI service") from e
    except requests.RequestException as e:
        logger.error(f"Hugging Face API error: {str(e)}")
        raise
    except (ValueError, KeyError) as e:
        logger.error(f"Response parsing error: {str(e)}")
        raise ValueError(f"Invalid response format from Hugging Face API: {str(e)}") from e


def _parse_generation_response(response, original_prompt):
    """
    Parse response from text generation model (GPT2, Mistral, etc).
    
    Args:
        response: API response
        original_prompt: Original input prompt
        
    Returns:
        dict: Parsed response with generated text
    """
    try:
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], dict) and 'generated_text' in response[0]:
                generated_text = response[0]['generated_text']
                # Remove the original prompt from the generated text
                if generated_text.startswith(original_prompt):
                    generated_text = generated_text[len(original_prompt):].strip()
                
                return {
                    'type': 'generation',
                    'result': generated_text,
                    'model': HUGGINGFACE_MODEL
                }
        
        raise ValueError(f"Unexpected generation response format: {response}")
        
    except Exception as e:
        logger.error(f"Error parsing generation response: {str(e)}")
        raise


def _parse_classification_response(response):
    """
    Parse response from classification model (DistilBERT, etc).
    
    Args:
        response: API response
        
    Returns:
        dict: Parsed response with classification scores
    """
    try:
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], list):
                # Response is list of scores per label
                scores = response[0]
                # Sort by score descending
                scores_sorted = sorted(scores, key=lambda x: x.get('score', 0), reverse=True)
                
                return {
                    'type': 'classification',
                    'result': scores_sorted,
                    'top_label': scores_sorted[0].get('label') if scores_sorted else 'unknown',
                    'model': HUGGINGFACE_MODEL
                }
        
        raise ValueError(f"Unexpected classification response format: {response}")
        
    except Exception as e:
        logger.error(f"Error parsing classification response: {str(e)}")
        raise


def _mock_response(prompt):
    """
    Return a deterministic local response when MOCK_MODE is enabled.
    """
    if MODEL_TYPE == 'generation':
        return {
            'type': 'generation',
            'result': f"(mock) You said: {prompt}",
            'model': 'mock'
        }

    return {
        'type': 'classification',
        'result': [
            {'label': 'POSITIVE', 'score': 0.75},
            {'label': 'NEGATIVE', 'score': 0.25}
        ],
        'top_label': 'POSITIVE',
        'model': 'mock'
    }


def _local_model_response(prompt):
    """
    Run inference using a locally loaded Hugging Face model.
    Uses inference_mode for faster execution and tuned params for coherent output.
    """
    try:
        # Check if using a seq2seq model (like FLAN-T5)
        if isinstance(LOCAL_MODEL, dict) and LOCAL_MODEL.get('type') == 'seq2seq':
            # For seq2seq models like FLAN-T5
            tokenizer = LOCAL_MODEL['tokenizer']
            model = LOCAL_MODEL['model']
            device = next(model.parameters()).device
            
            prompt_text = _build_flan_prompt(prompt)
            # Tokenize input
            inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)
            
            # Generate response — inference_mode is faster than no_grad
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3,
                    top_p=0.85,
                    top_k=40,
                    do_sample=True,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    num_beams=2,
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'type': 'generation',
                'result': generated_text,
                'model': HUGGINGFACE_MODEL
            }
        elif isinstance(LOCAL_MODEL, dict) and LOCAL_MODEL.get('type') == 'classification':
            # For classification models
            result = LOCAL_MODEL(prompt)
            scores = [{'label': r['label'], 'score': round(r['score'], 4)} for r in result]
            scores_sorted = sorted(scores, key=lambda x: x.get('score', 0), reverse=True)
            
            return {
                'type': 'classification',
                'result': scores_sorted,
                'top_label': scores_sorted[0].get('label') if scores_sorted else 'unknown',
                'model': HUGGINGFACE_MODEL
            }
        elif MODEL_TYPE == 'generation':
            # For causal language models like GPT-2
            result = LOCAL_MODEL(prompt, max_length=150, temperature=0.7, top_p=0.9, do_sample=True)
            generated_text = result[0]['generated_text']
            # Remove the original prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return {
                'type': 'generation',
                'result': generated_text,
                'model': HUGGINGFACE_MODEL
            }
        else:  # classification
            result = LOCAL_MODEL(prompt)
            scores = [{'label': r['label'], 'score': round(r['score'], 4)} for r in result]
            scores_sorted = sorted(scores, key=lambda x: x.get('score', 0), reverse=True)
            
            return {
                'type': 'classification',
                'result': scores_sorted,
                'top_label': scores_sorted[0].get('label') if scores_sorted else 'unknown',
                'model': HUGGINGFACE_MODEL
            }
    except Exception as e:
        logger.error(f"Local model inference error: {str(e)}")
        raise

@app.route('/', methods=['GET'])
def index():
    return send_from_directory(os.path.dirname(__file__), 'index.html')


@app.route('/index.html', methods=['GET'])
def index_html():
    return send_from_directory(os.path.dirname(__file__), 'index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint for customer service chatbot.
    
    Expected JSON payload:
    {
        "message": "hello"
    }
    
    Returns:
        JSON response with the chatbot's reply or classification results
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            logger.warning("Empty request body received")
            return jsonify({
                "error": "Request body must be valid JSON",
                "code": "INVALID_REQUEST"
            }), 400
        
        # Validate message field
        message = data.get('message', '').strip()
        if not message:
            logger.warning("Empty message received")
            return jsonify({
                "error": "Message field is required and cannot be empty",
                "code": "EMPTY_MESSAGE"
            }), 400
        
        if len(message) > 2000:
            logger.warning(f"Message too long: {len(message)} characters")
            return jsonify({
                "error": "Message must not exceed 2000 characters",
                "code": "MESSAGE_TOO_LONG"
            }), 400
        
        logger.info(f"Processing {MODEL_TYPE} request: {message[:100]}...")
        
        # Get session ID and IP address
        session_id = data.get('session_id', request.headers.get('X-Session-ID', 'unknown'))
        ip_address = request.headers.get('X-Forwarded-For', request.remote_addr)

        # ========== SMART ENTITY EXTRACTION ==========
        # Extract all identifiers from the message upfront
        order_number = _extract_order_number(message)
        email = _extract_email(message)
        sku = _extract_sku(message)
        product_name = _extract_product_name(message)
        
        intent_match = _match_intent_response(message)
        intent_tag = intent_match['tag'] if intent_match else None
        
        # Helper to build and return a database-backed response
        def _db_response(bot_response, intent, response_type, extra=None):
            response_data = {
                "success": True,
                "type": response_type,
                "intent": intent,
                "message": message,
                "response": bot_response,
                "model": "database"
            }
            if extra:
                response_data.update(extra)
            log_conversation(
                session_id=session_id, user_message=message,
                bot_response=bot_response, intent=intent,
                model_used="database", response_type=response_type,
                ip_address=ip_address
            )
            return jsonify(response_data), 200

        # ========== 1. ORDER LOOKUP (by order number) ==========
        if order_number:
            order = _lookup_order_status(order_number)
            if order:
                bot_response = _format_order_response(order)
                logger.info(f"Order lookup: {order_number}")
                return _db_response(bot_response, "order_tracking", "order_lookup", {"order": order})
            else:
                bot_response = (
                    f"❌ Sorry, I couldn't find order **{order_number}** in our system. "
                    "Please check the order number and try again. Or contact support@company.com for assistance."
                )
                return _db_response(bot_response, "order_tracking", "order_not_found")

        # ========== 2. CUSTOMER LOOKUP (by email) ==========
        if email:
            # Determine what the user is asking about
            if intent_tag in ('order_tracking', 'order_status', 'shipping'):
                # Looking up orders by email
                orders = _lookup_orders_by_email(email)
                if orders:
                    bot_response = _format_orders_list_response(orders, email)
                    logger.info(f"Orders lookup by email: {email} ({len(orders)} found)")
                    return _db_response(bot_response, "order_tracking", "orders_by_email")
                else:
                    bot_response = f"I couldn't find any orders associated with **{email}**. Please check the email address or provide an order number."
                    return _db_response(bot_response, "order_tracking", "customer_not_found")
            else:
                # General customer lookup
                customer = _lookup_customer_by_email(email)
                if customer:
                    bot_response = _format_customer_response(customer)
                    logger.info(f"Customer lookup: {email}")
                    return _db_response(bot_response, "account", "customer_lookup")
                else:
                    bot_response = f"I couldn't find an account associated with **{email}**. Would you like help creating one?"
                    return _db_response(bot_response, "account", "customer_not_found")

        # ========== 3. PRODUCT LOOKUP (by SKU or name) ==========
        if intent_tag in ('product_info', 'pricing', 'stock_availability', 'size_fitting'):
            search_term = sku or product_name
            if search_term:
                products = _lookup_product(search_term)
                if products:
                    bot_response = _format_product_response(products)
                    logger.info(f"Product lookup: {search_term} ({len(products)} found)")
                    return _db_response(bot_response, intent_tag, "product_lookup")
            # No product found or no search term — fall through to intent response

        # ========== 4. ORDER TRACKING (no order number provided) ==========
        if intent_tag == 'order_tracking':
            bot_response = intent_match['response']
            response_data = {
                "success": True, "type": "intent", "intent": intent_tag,
                "message": message, "response": bot_response, "model": "intents"
            }
            log_conversation(
                session_id=session_id, user_message=message,
                bot_response=bot_response, intent=intent_tag,
                model_used="intents", response_type="intent",
                ip_address=ip_address
            )
            return jsonify(response_data), 200

        # ========== 5. OTHER INTENT MATCHES ==========
        if intent_match:
            response_data = {
                "success": True,
                "type": "intent",
                "intent": intent_match['tag'],
                "message": message,
                "response": intent_match['response'],
                "model": "intents"
            }
            logger.info(f"Intent matched: {intent_match['tag']}")
            
            log_conversation(
                session_id=session_id,
                user_message=message,
                bot_response=intent_match['response'],
                intent=intent_match['tag'],
                model_used="intents",
                response_type="intent",
                ip_address=ip_address
            )
            
            return jsonify(response_data), 200
        
        # ========== 6. FALLBACK: AI MODEL ==========
        api_response = query_huggingface(message)
        
        # Format response based on model type
        if api_response['type'] == 'generation':
            response_data = {
                "success": True,
                "type": "generation",
                "message": message,
                "response": api_response['result'],
                "model": api_response['model']
            }
            
            # Log to database
            log_conversation(
                session_id=session_id,
                user_message=message,
                bot_response=api_response['result'],
                intent=None,
                model_used=api_response['model'],
                response_type="generation",
                ip_address=ip_address
            )
        else:  # classification
            response_data = {
                "success": True,
                "type": "classification",
                "message": message,
                "classification": {
                    "top_label": api_response['top_label'],
                    "scores": api_response['result']
                },
                "model": api_response['model']
            }
            
            # Log to database
            log_conversation(
                session_id=session_id,
                user_message=message,
                bot_response=f"Classification: {api_response['top_label']}",
                intent=None,
                model_used=api_response['model'],
                response_type="classification",
                ip_address=ip_address
            )
        
        logger.info(f"Response generated successfully (type: {api_response['type']})")
        return jsonify(response_data), 200
        
    except TimeoutError:
        logger.error("API request timed out")
        return jsonify({
            "error": "Request to AI service timed out. Please try again.",
            "code": "TIMEOUT"
        }), 504
        
    except ValueError as e:
        error_msg = str(e)
        if "Invalid API key" in error_msg:
            return jsonify({
                "error": "Authentication failed",
                "code": "AUTH_ERROR"
            }), 401
        return jsonify({
            "error": "Invalid response from AI service",
            "code": "INVALID_RESPONSE",
            "details": error_msg
        }), 500
        
    except requests.RequestException as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return jsonify({
                "error": "API rate limit exceeded. Please try again later.",
                "code": "RATE_LIMITED"
            }), 429
        elif "loading" in error_msg.lower():
            return jsonify({
                "error": "Model is loading. Please try again in a moment.",
                "code": "MODEL_LOADING"
            }), 503
        
        logger.error(f"API request failed: {str(e)}")
        return jsonify({
            "error": "Failed to communicate with AI service",
            "code": "SERVICE_ERROR"
        }), 503
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR"
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "ai-customer-chatbot",
        "model": HUGGINGFACE_MODEL,
        "use_local_model": USE_LOCAL_MODEL,
        "mock_mode": MOCK_MODE,
        "intents_count": len(INTENTS)
    }), 200


@app.route('/api/admin/logs', methods=['GET'])
def get_logs():
    """
    Admin endpoint to retrieve conversation logs
    
    Query parameters:
    - limit: Number of records to return (default: 50, max: 500)
    - offset: Number of records to skip (default: 0)
    - session_id: Filter by specific session
    - since: Filter by date (YYYY-MM-DD format)
    """
    try:
        # Parse query parameters
        limit = min(int(request.args.get('limit', 50)), 500)
        offset = int(request.args.get('offset', 0))
        session_id = request.args.get('session_id')
        since_date = request.args.get('since')
        
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        # Build query
        query = "SELECT * FROM conversations WHERE 1=1"
        params = []
        
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        
        if since_date:
            query += " AND DATE(timestamp) >= ?"
            params.append(since_date)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Get total count
        count_query = "SELECT COUNT(*) as total FROM conversations WHERE 1=1"
        count_params = []
        if session_id:
            count_query += " AND session_id = ?"
            count_params.append(session_id)
        if since_date:
            count_query += " AND DATE(timestamp) >= ?"
            count_params.append(since_date)
        
        cursor.execute(count_query, count_params)
        total = cursor.fetchone()['total']
        
        # Convert rows to dict
        logs = [dict(row) for row in rows]
        
        conn.close()
        
        return jsonify({
            "success": True,
            "total": total,
            "limit": limit,
            "offset": offset,
            "logs": logs
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return jsonify({
            "error": "Failed to fetch logs",
            "code": "DB_ERROR"
        }), 500


@app.route('/api/admin/stats', methods=['GET'])
def get_stats():
    """Get conversation statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total conversations
        cursor.execute("SELECT COUNT(*) as total FROM conversations")
        total = cursor.fetchone()[0]
        
        # By intent
        cursor.execute("""
            SELECT intent, COUNT(*) as count 
            FROM conversations 
            WHERE intent IS NOT NULL 
            GROUP BY intent 
            ORDER BY count DESC
        """)
        by_intent = [{"intent": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        # By model
        cursor.execute("""
            SELECT model_used, COUNT(*) as count 
            FROM conversations 
            WHERE model_used IS NOT NULL 
            GROUP BY model_used 
            ORDER BY count DESC
        """)
        by_model = [{"model": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        # By date (last 7 days)
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count 
            FROM conversations 
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY DATE(timestamp) 
            ORDER BY date DESC
        """)
        by_date = [{"date": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        # Unique sessions
        cursor.execute("SELECT COUNT(DISTINCT session_id) as count FROM conversations")
        unique_sessions = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            "success": True,
            "stats": {
                "total_conversations": total,
                "unique_sessions": unique_sessions,
                "by_intent": by_intent,
                "by_model": by_model,
                "last_7_days": by_date
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return jsonify({
            "error": "Failed to fetch statistics",
            "code": "DB_ERROR"
        }), 500


@app.route('/api/admin/debug', methods=['GET'])
def debug_db():
    """Debug endpoint to check database status"""
    import os.path
    try:
        db_exists = os.path.isfile(DB_PATH)
        db_readable = os.access(DB_PATH, os.R_OK) if db_exists else False
        db_writable = os.access(DB_PATH, os.W_OK) if db_exists else False
        
        if not db_exists:
            return jsonify({
                "db_path": DB_PATH,
                "db_exists": False,
                "message": "Database file doesn't exist"
            }), 200
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversations")
        count = cursor.fetchone()[0]
        
        cursor.execute("SELECT * FROM conversations ORDER BY timestamp DESC LIMIT 5")
        recent = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            "db_path": DB_PATH,
            "db_exists": db_exists,
            "db_readable": db_readable,
            "db_writable": db_writable,
            "record_count": count,
            "recent_ids": [r[0] for r in recent] if recent else []
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "db_path": DB_PATH
        }), 500


# ========== PRODUCT MANAGEMENT ENDPOINTS ==========

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all products or search/filter products"""
    try:
        if not supabase:
            return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
        
        # Parse query parameters
        search = request.args.get('search', '').strip()
        category = request.args.get('category')
        limit = min(int(request.args.get('limit', 100)), 500)
        offset = int(request.args.get('offset', 0))
        
        # Build query
        query = supabase.table('products').select('*')
        
        if search:
            query = query.or_(f"name.ilike.%{search}%,description.ilike.%{search}%,sku.ilike.%{search}%")
        
        if category:
            query = query.eq('category', category)
        
        query = query.range(offset, offset + limit - 1).order('created_at', desc=True)
        
        result = query.execute()
        
        return jsonify({
            "success": True,
            "count": len(result.data),
            "products": result.data
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching products: {e}")
        return jsonify({
            "error": "Failed to fetch products",
            "code": "DB_ERROR"
        }), 500


@app.route('/api/products/<product_id>', methods=['GET'])
def get_product(product_id):
    """Get a specific product by ID"""
    try:
        if not supabase:
            return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
        
        result = supabase.table('products').select('*').eq('id', product_id).execute()
        
        if not result.data:
            return jsonify({
                "error": "Product not found",
                "code": "NOT_FOUND"
            }), 404
        
        return jsonify({
            "success": True,
            "product": result.data[0]
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching product: {e}")
        return jsonify({
            "error": "Failed to fetch product",
            "code": "DB_ERROR"
        }), 500


@app.route('/api/products', methods=['POST'])
def create_product():
    """Create a new product"""
    try:
        if not supabase:
            return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'price']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    "error": f"Missing required field: {field}",
                    "code": "MISSING_FIELD"
                }), 400
        
        # Prepare product data
        product_data = {
            "name": data['name'],
            "description": data.get('description'),
            "price": float(data['price']),
            "category": data.get('category'),
            "sku": data.get('sku'),
            "stock": int(data.get('stock', 0)),
            "image_url": data.get('image_url'),
            "is_duplicate": data.get('is_duplicate', False),
            "duplicate_of": data.get('duplicate_of')
        }
        
        result = supabase.table('products').insert(product_data).execute()
        
        return jsonify({
            "success": True,
            "product": result.data[0]
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating product: {e}")
        return jsonify({
            "error": "Failed to create product",
            "code": "DB_ERROR"
        }), 500


@app.route('/api/products/<product_id>', methods=['PUT'])
def update_product(product_id):
    """Update an existing product"""
    try:
        if not supabase:
            return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
        
        data = request.get_json()
        
        # Remove id from update data if present
        data.pop('id', None)
        data.pop('created_at', None)
        
        result = supabase.table('products').update(data).eq('id', product_id).execute()
        
        if not result.data:
            return jsonify({
                "error": "Product not found",
                "code": "NOT_FOUND"
            }), 404
        
        return jsonify({
            "success": True,
            "product": result.data[0]
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating product: {e}")
        return jsonify({
            "error": "Failed to update product",
            "code": "DB_ERROR"
        }), 500


@app.route('/api/products/<product_id>', methods=['DELETE'])
def delete_product(product_id):
    """Delete a product"""
    try:
        if not supabase:
            return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
        
        result = supabase.table('products').delete().eq('id', product_id).execute()
        
        if not result.data:
            return jsonify({
                "error": "Product not found",
                "code": "NOT_FOUND"
            }), 404
        
        return jsonify({
            "success": True,
            "message": "Product deleted successfully"
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting product: {e}")
        return jsonify({
            "error": "Failed to delete product",
            "code": "DB_ERROR"
        }), 500


@app.route('/api/products/duplicates', methods=['GET'])
def get_duplicate_products():
    """Get all products marked as duplicates"""
    try:
        if not supabase:
            return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
        
        result = supabase.table('products').select('*').eq('is_duplicate', True).execute()
        
        return jsonify({
            "success": True,
            "count": len(result.data),
            "duplicates": result.data
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching duplicates: {e}")
        return jsonify({
            "error": "Failed to fetch duplicates",
            "code": "DB_ERROR"
        }), 500


# ========== ORDER MANAGEMENT ENDPOINTS ==========

@app.route('/api/orders', methods=['GET'])
def get_orders():
    """Get all orders or filter by customer/status"""
    try:
        if not supabase:
            return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
        
        # Parse query parameters
        customer_email = request.args.get('customer_email')
        status = request.args.get('status')
        limit = min(int(request.args.get('limit', 50)), 200)
        offset = int(request.args.get('offset', 0))
        
        # Build query
        query = supabase.table('orders').select('*')
        
        if customer_email:
            query = query.eq('customer_email', customer_email)
        
        if status:
            query = query.eq('status', status)
        
        query = query.range(offset, offset + limit - 1).order('order_date', desc=True)
        
        result = query.execute()
        
        return jsonify({
            "success": True,
            "count": len(result.data),
            "orders": result.data
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        return jsonify({
            "error": "Failed to fetch orders",
            "code": "DB_ERROR"
        }), 500


@app.route('/api/orders/<order_id>', methods=['GET'])
def get_order(order_id):
    """Get a specific order with items"""
    try:
        if not supabase:
            return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
        
        # Get order
        order_result = supabase.table('orders').select('*').eq('id', order_id).execute()
        
        if not order_result.data:
            return jsonify({
                "error": "Order not found",
                "code": "NOT_FOUND"
            }), 404
        
        # Get order items
        items_result = supabase.table('order_items').select('*').eq('order_id', order_id).execute()
        
        order = order_result.data[0]
        order['items'] = items_result.data
        
        return jsonify({
            "success": True,
            "order": order
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching order: {e}")
        return jsonify({
            "error": "Failed to fetch order",
            "code": "DB_ERROR"
        }), 500


@app.route('/api/orders/number/<order_number>', methods=['GET'])
def get_order_by_number(order_number):
    """Get order by order number"""
    try:
        if not supabase:
            return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
        
        # Get order
        order_result = supabase.table('orders').select('*').eq('order_number', order_number).execute()
        
        if not order_result.data:
            return jsonify({
                "error": "Order not found",
                "code": "NOT_FOUND"
            }), 404
        
        order = order_result.data[0]
        
        # Get order items
        items_result = supabase.table('order_items').select('*').eq('order_id', order['id']).execute()
        order['items'] = items_result.data
        
        return jsonify({
            "success": True,
            "order": order
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching order: {e}")
        return jsonify({
            "error": "Failed to fetch order",
            "code": "DB_ERROR"
        }), 500


@app.route('/api/orders', methods=['POST'])
def create_order():
    """Create a new order with items"""
    try:
        if not supabase:
            return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['order_number', 'customer_name', 'customer_email', 'total_amount', 'items']
        for field in required_fields:
            if not data.get(field):
                return jsonify({
                    "error": f"Missing required field: {field}",
                    "code": "MISSING_FIELD"
                }), 400
        
        if not isinstance(data['items'], list) or len(data['items']) == 0:
            return jsonify({
                "error": "Order must contain at least one item",
                "code": "INVALID_ITEMS"
            }), 400
        
        # Prepare order data
        order_data = {
            "order_number": data['order_number'],
            "customer_name": data['customer_name'],
            "customer_email": data['customer_email'],
            "customer_phone": data.get('customer_phone'),
            "shipping_address": data.get('shipping_address'),
            "status": data.get('status', 'pending'),
            "total_amount": float(data['total_amount']),
            "tracking_number": data.get('tracking_number'),
            "notes": data.get('notes')
        }
        
        # Create order
        order_result = supabase.table('orders').insert(order_data).execute()
        order = order_result.data[0]
        order_id = order['id']
        
        # Create order items
        items_data = []
        for item in data['items']:
            items_data.append({
                "order_id": order_id,
                "product_id": item.get('product_id'),
                "product_name": item['product_name'],
                "product_sku": item.get('product_sku'),
                "quantity": int(item['quantity']),
                "unit_price": float(item['unit_price']),
                "subtotal": float(item['quantity']) * float(item['unit_price'])
            })
        
        items_result = supabase.table('order_items').insert(items_data).execute()
        order['items'] = items_result.data
        
        return jsonify({
            "success": True,
            "order": order
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating order: {e}")
        return jsonify({
            "error": "Failed to create order",
            "code": "DB_ERROR"
        }), 500


@app.route('/api/orders/<order_id>', methods=['PUT'])
def update_order(order_id):
    """Update order status or details"""
    try:
        if not supabase:
            return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
        
        data = request.get_json()
        
        # Remove fields that shouldn't be updated
        data.pop('id', None)
        data.pop('created_at', None)
        data.pop('order_number', None)  # Order number shouldn't change
        data.pop('items', None)  # Items updated separately
        
        result = supabase.table('orders').update(data).eq('id', order_id).execute()
        
        if not result.data:
            return jsonify({
                "error": "Order not found",
                "code": "NOT_FOUND"
            }), 404
        
        return jsonify({
            "success": True,
            "order": result.data[0]
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating order: {e}")
        return jsonify({
            "error": "Failed to update order",
            "code": "DB_ERROR"
        }), 500


@app.route('/api/orders/<order_id>/status', methods=['PATCH'])
def update_order_status(order_id):
    """Update order status and tracking"""
    try:
        if not supabase:
            return jsonify({"error": "Database not configured", "code": "DB_NOT_CONFIGURED"}), 500
        
        data = request.get_json()
        
        update_data = {}
        if 'status' in data:
            update_data['status'] = data['status']
        if 'tracking_number' in data:
            update_data['tracking_number'] = data['tracking_number']
        if 'notes' in data:
            update_data['notes'] = data['notes']
        
        if not update_data:
            return jsonify({
                "error": "No valid fields to update",
                "code": "INVALID_DATA"
            }), 400
        
        result = supabase.table('orders').update(update_data).eq('id', order_id).execute()
        
        if not result.data:
            return jsonify({
                "error": "Order not found",
                "code": "NOT_FOUND"
            }), 404
        
        return jsonify({
            "success": True,
            "order": result.data[0]
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating order status: {e}")
        return jsonify({
            "error": "Failed to update order status",
            "code": "DB_ERROR"
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "code": "NOT_FOUND"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        "error": "Method not allowed",
        "code": "METHOD_NOT_ALLOWED"
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "error": "Internal server error",
        "code": "INTERNAL_ERROR"
    }), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=False)  # Disable debug mode to avoid memory issues