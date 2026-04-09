"""
Centralized configuration — loads environment variables and defines constants.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────
# PACKAGE_DIR  → .../chatbot/
# PROJECT_ROOT → .../ai-customer-chatbot/
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DATA_DIR = PACKAGE_DIR / 'data'
DB_PATH = str(PROJECT_ROOT / 'chatbot.db')
INTENTS_PATH = str(DATA_DIR / 'intents.json')
FINETUNED_MODEL_PATH = os.getenv('FINETUNED_MODEL_PATH')  # Optional: path to fine-tuned model

# ── Supabase ──────────────────────────────────────────────
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# ── Live Tracking ─────────────────────────────────────────
AFTERSHIP_API_KEY = os.getenv('AFTERSHIP_API_KEY')
TRACKING_MOCK_MODE = os.getenv('TRACKING_MOCK_MODE', 'false').strip().lower() in ('1', 'true', 'yes')
TRACKING_MOCK_CARRIER = os.getenv('TRACKING_MOCK_CARRIER', 'mock').strip()

# ── Admin Auth ────────────────────────────────────────────
ADMIN_API_KEY = os.getenv('ADMIN_API_KEY')
FRONTEND_URL = os.getenv('FRONTEND_URL', '').strip()

# ── Email (Order Confirmations) ───────────────────────────
SMTP_HOST = os.getenv('SMTP_HOST', '').strip()
SMTP_USERNAME = os.getenv('SMTP_USERNAME', '').strip()
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD', '').strip()
SMTP_FROM_EMAIL = os.getenv('SMTP_FROM_EMAIL', '').strip()
SMTP_USE_TLS = os.getenv('SMTP_USE_TLS', 'true').strip().lower() in ('1', 'true', 'yes')
ORDER_EMAIL_ENABLED = os.getenv('ORDER_EMAIL_ENABLED', 'true').strip().lower() in ('1', 'true', 'yes')


def _parse_int_env(name, default):
    raw_value = os.getenv(name, str(default)).strip()
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc


SMTP_PORT = _parse_int_env('SMTP_PORT', 587)

# ── Hugging Face / Model ─────────────────────────────────
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
HUGGINGFACE_MODEL = os.getenv('HUGGINGFACE_MODEL', 'google/flan-t5-small').strip()
MODEL_TYPE = 'generation'
MOCK_MODE = os.getenv('MOCK_MODE', 'false').strip().lower() in ('1', 'true', 'yes')
USE_LOCAL_MODEL = os.getenv('USE_LOCAL_MODEL', 'true').strip().lower() in ('1', 'true', 'yes')

HUGGINGFACE_API_BASE = os.getenv(
    'HUGGINGFACE_API_BASE',
    'https://router.huggingface.co/hf-inference/models'
)
HUGGINGFACE_API_URL = f"{HUGGINGFACE_API_BASE}/{HUGGINGFACE_MODEL}"

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

# ── Flask ─────────────────────────────────────────────────
PORT = int(os.getenv('PORT', 7860))
CHAT_RATE_LIMIT = os.getenv('CHAT_RATE_LIMIT', '30 per minute')

# ── Startup validation ────────────────────────────────────
if not HUGGINGFACE_API_KEY and not MOCK_MODE and not USE_LOCAL_MODEL:
    logger.warning("HUGGINGFACE_API_KEY not set — AI responses will fail unless MOCK_MODE is enabled")

logger.info(
    f"Config loaded — model: {HUGGINGFACE_MODEL}, type: {MODEL_TYPE}, "
    f"mock: {MOCK_MODE}, local: {USE_LOCAL_MODEL}"
)
