"""
Database layer — Supabase client + SQLite fallback + conversation logging.

Logging strategy:
  • If Supabase is available → write there too (cloud backup)
  • Always write to SQLite (admin reads from here)
  • Writes happen in a background thread so the user never waits

Admin endpoints always read SQLite so they work even when Supabase is down.
"""
import sqlite3
import logging
import threading
from contextlib import contextmanager

from .config import SUPABASE_URL, SUPABASE_KEY, DB_PATH

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
@contextmanager
def get_db():
    """Context-managed SQLite connection with WAL mode + row factory."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """Create the conversations table + indexes if they don't exist."""
    with get_db() as conn:
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
                ip_address TEXT,
                response_time_ms INTEGER
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON conversations(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_session ON conversations(session_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_intent ON conversations(intent)')
        conn.commit()

    # Migrate: add response_time_ms column if missing (existing DBs)
    try:
        with get_db() as conn:
            conn.execute("SELECT response_time_ms FROM conversations LIMIT 1")
    except sqlite3.OperationalError:
        with get_db() as conn:
            conn.execute("ALTER TABLE conversations ADD COLUMN response_time_ms INTEGER")
            conn.commit()
            logger.info("Migrated: added response_time_ms column")

    logger.info("SQLite database initialized")


# ── Background logging ────────────────────────────────────
def _log_in_background(session_id, user_message, bot_response,
                       intent, model_used, response_type, ip_address,
                       response_time_ms):
    """Write to both SQLite (always) and Supabase (if available) in a daemon thread."""

    def _write():
        # 1. Always write to SQLite (admin reads from here)
        try:
            with get_db() as conn:
                conn.execute('''
                    INSERT INTO conversations
                    (session_id, user_message, bot_response, intent,
                     model_used, response_type, ip_address, response_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, user_message, bot_response, intent,
                      model_used, response_type, ip_address, response_time_ms))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log conversation to SQLite: {e}")

        # 2. Also write to Supabase if available
        if supabase:
            try:
                supabase.table('conversations').insert({
                    "session_id": session_id,
                    "user_message": user_message,
                    "bot_response": bot_response,
                    "intent": intent,
                    "model_used": model_used,
                    "response_type": response_type,
                    "ip_address": ip_address,
                    "response_time_ms": response_time_ms,
                }).execute()
            except Exception as e:
                logger.error(f"Failed to log conversation to Supabase: {e}")

    thread = threading.Thread(target=_write, daemon=True)
    thread.start()


def log_conversation(session_id, user_message, bot_response,
                     intent=None, model_used=None, response_type=None,
                     ip_address=None, response_time_ms=None):
    """
    Log a conversation (non-blocking).

    Writes to SQLite always + Supabase if configured, in a background thread.
    """
    _log_in_background(
        session_id, user_message, bot_response,
        intent, model_used, response_type, ip_address, response_time_ms,
    )


def purge_old_logs(days=90):
    """Delete conversation logs older than N days. Returns count deleted."""
    with get_db() as conn:
        cursor = conn.execute(
            "DELETE FROM conversations WHERE timestamp < datetime('now', ?)",
            (f'-{days} days',),
        )
        conn.commit()
        return cursor.rowcount


# Initialize on import
init_database()
