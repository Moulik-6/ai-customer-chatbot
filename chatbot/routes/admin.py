"""
Admin routes — conversation logs, stats, debug, log management.
"""
import logging
import os

from flask import Blueprint, request, jsonify

from ..auth import require_admin_key
from ..config import DB_PATH
from ..database import get_db, purge_old_logs

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__)


@admin_bp.route('/api/admin/logs', methods=['GET'])
@require_admin_key
def get_logs():
    """
    Retrieve conversation logs.

    Query params: limit, offset, session_id, since (YYYY-MM-DD).
    """
    try:
        limit = min(int(request.args.get('limit', 50)), 500)
        offset = int(request.args.get('offset', 0))
        session_id = request.args.get('session_id')
        since_date = request.args.get('since')

        with get_db() as conn:
            cursor = conn.cursor()

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

            logs = [dict(row) for row in rows]

        return jsonify({
            "success": True, "total": total,
            "limit": limit, "offset": offset, "logs": logs,
        }), 200

    except Exception as e:
        logger.error(f"Error fetching logs: {e}")
        return jsonify({"error": "Failed to fetch logs", "code": "DB_ERROR"}), 500


@admin_bp.route('/api/admin/stats', methods=['GET'])
@require_admin_key
def get_stats():
    """Get conversation statistics."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM conversations")
            total = cursor.fetchone()[0]

            cursor.execute("""
                SELECT intent, COUNT(*) as count
                FROM conversations WHERE intent IS NOT NULL
                GROUP BY intent ORDER BY count DESC
            """)
            by_intent = [{"intent": r[0], "count": r[1]} for r in cursor.fetchall()]

            cursor.execute("""
                SELECT model_used, COUNT(*) as count
                FROM conversations WHERE model_used IS NOT NULL
                GROUP BY model_used ORDER BY count DESC
            """)
            by_model = [{"model": r[0], "count": r[1]} for r in cursor.fetchall()]

            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM conversations
                WHERE timestamp >= datetime('now', '-7 days')
                GROUP BY DATE(timestamp) ORDER BY date DESC
            """)
            by_date = [{"date": r[0], "count": r[1]} for r in cursor.fetchall()]

            cursor.execute("SELECT COUNT(DISTINCT session_id) FROM conversations")
            unique_sessions = cursor.fetchone()[0]

            cursor.execute("""
                SELECT ROUND(AVG(response_time_ms)) as avg_ms,
                       MIN(response_time_ms) as min_ms,
                       MAX(response_time_ms) as max_ms
                FROM conversations
                WHERE response_time_ms IS NOT NULL
                  AND timestamp >= datetime('now', '-7 days')
            """)
            perf_row = cursor.fetchone()
            performance = {
                "avg_response_ms": perf_row[0],
                "min_response_ms": perf_row[1],
                "max_response_ms": perf_row[2],
            } if perf_row[0] is not None else None

        return jsonify({
            "success": True,
            "stats": {
                "total_conversations": total,
                "unique_sessions": unique_sessions,
                "by_intent": by_intent,
                "by_model": by_model,
                "last_7_days": by_date,
                "performance": performance,
            },
        }), 200

    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return jsonify({"error": "Failed to fetch statistics", "code": "DB_ERROR"}), 500


@admin_bp.route('/api/admin/debug', methods=['GET'])
@require_admin_key
def debug_db():
    """Debug endpoint to check database status."""
    try:
        db_exists = os.path.isfile(DB_PATH)
        db_readable = os.access(DB_PATH, os.R_OK) if db_exists else False
        db_writable = os.access(DB_PATH, os.W_OK) if db_exists else False

        if not db_exists:
            return jsonify({
                "db_exists": False,
                "message": "Database file doesn't exist",
            }), 200

        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations")
            count = cursor.fetchone()[0]
            cursor.execute("SELECT id FROM conversations ORDER BY timestamp DESC LIMIT 5")
            recent = cursor.fetchall()

        return jsonify({
            "db_exists": db_exists,
            "db_readable": db_readable, "db_writable": db_writable,
            "record_count": count,
            "recent_ids": [r[0] for r in recent] if recent else [],
        }), 200

    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        return jsonify({"error": "Internal error checking database status", "code": "INTERNAL_ERROR"}), 500


@admin_bp.route('/api/admin/logs/purge', methods=['POST'])
@require_admin_key
def purge_logs():
    """
    Delete conversation logs older than N days.

    JSON body: { "days": 90 }   (default 90)
    """
    try:
        data = request.get_json() or {}
        days = int(data.get('days', 90))
        if days < 1:
            return jsonify({"error": "days must be >= 1", "code": "INVALID_PARAM"}), 400

        deleted = purge_old_logs(days)
        logger.info(f"Purged {deleted} log entries older than {days} days")
        return jsonify({"success": True, "deleted": deleted, "older_than_days": days}), 200

    except Exception as e:
        logger.error(f"Error purging logs: {e}")
        return jsonify({"error": "Failed to purge logs", "code": "DB_ERROR"}), 500
