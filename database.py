"""
Database layer — Supabase client + SQLite fallback + conversation logging.
"""
import sqlite3
import logging
from config import SUPABASE_URL, SUPABASE_KEY, DB_PATH

logger = logging.getLogger(__name__)

# ── Supabase client ───────────────────────────────────────
supabase = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase: {e}")
else:
    logger.warning("Supabase credentials not found — database features disabled")


# ── SQLite helpers ────────────────────────────────────────
def _get_db():
    """Return a SQLite connection with row factory and WAL mode."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_database():
    """Create the conversations table + indexes if they don't exist."""
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

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id)')

    conn.commit()
    conn.close()
    logger.info("SQLite database initialized")


def log_conversation(session_id, user_message, bot_response,
                     intent=None, model_used=None, response_type=None, ip_address=None):
    """Log a conversation to Supabase (preferred) or SQLite fallback."""
    try:
        if supabase:
            supabase.table('conversations').insert({
                "session_id": session_id,
                "user_message": user_message,
                "bot_response": bot_response,
                "intent": intent,
                "model_used": model_used,
                "response_type": response_type,
                "ip_address": ip_address
            }).execute()
            logger.info("Conversation logged to Supabase")
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversations
            (session_id, user_message, bot_response, intent, model_used, response_type, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (session_id, user_message, bot_response, intent, model_used, response_type, ip_address))
        conn.commit()
        conn.close()
        logger.info(f"Conversation logged to SQLite — ID: {cursor.lastrowid}")
    except Exception as e:
        logger.error(f"Failed to log conversation: {e}", exc_info=True)


# Initialize on import
init_database()
