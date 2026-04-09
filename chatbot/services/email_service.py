"""
Email service — send order confirmation emails.
"""
import logging
import smtplib
from email.message import EmailMessage

from ..config import (
    ORDER_EMAIL_ENABLED,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USERNAME,
    SMTP_PASSWORD,
    SMTP_FROM_EMAIL,
    SMTP_USE_TLS,
)

logger = logging.getLogger(__name__)


def _email_configured():
    return all([SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_FROM_EMAIL])


def _build_order_email(order):
    order_number = order.get('order_number', 'N/A')
    status = str(order.get('status', 'pending')).upper()
    total = float(order.get('total_amount') or 0)
    items = order.get('order_items', [])

    lines = [
        f"Thanks for your order!",
        "",
        f"Order Number: {order_number}",
        f"Status: {status}",
        f"Total: ${total:.2f}",
        "",
        "Items:",
    ]

    if items:
        for item in items:
            name = item.get('product_name', 'Product')
            qty = int(item.get('quantity') or 1)
            unit_price = float(item.get('unit_price') or 0)
            lines.append(f"- {name} x{qty} @ ${unit_price:.2f}")
    else:
        lines.append("- Item details unavailable")

    lines.extend([
        "",
        "You can track your order anytime by asking:",
        f"track order {order_number}",
        "",
        "Thank you for shopping with us!",
    ])

    return '\n'.join(lines)


def send_order_confirmation_email(customer_email, order):
    """
    Send order confirmation email.

    Returns a dict with keys:
      sent: bool
      error: optional human-readable error message
      code: optional machine-readable failure code
    """
    if not ORDER_EMAIL_ENABLED:
        logger.info("Order confirmation email skipped: ORDER_EMAIL_ENABLED is false")
        return {"sent": False, "code": "EMAIL_DISABLED", "error": "ORDER_EMAIL_ENABLED is false"}

    if not customer_email:
        logger.warning("Order confirmation email skipped: missing customer email")
        return {"sent": False, "code": "MISSING_CUSTOMER_EMAIL", "error": "Missing customer email"}

    if not _email_configured():
        logger.warning("Order confirmation email skipped: SMTP settings incomplete")
        return {"sent": False, "code": "SMTP_INCOMPLETE", "error": "SMTP settings incomplete"}

    try:
        msg = EmailMessage()
        msg['Subject'] = f"Order Confirmation - {order.get('order_number', 'N/A')}"
        msg['From'] = SMTP_FROM_EMAIL
        msg['To'] = customer_email
        msg.set_content(_build_order_email(order))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            if SMTP_USE_TLS:
                server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"Order confirmation email sent to {customer_email} for {order.get('order_number')}")
        return {"sent": True}

    except smtplib.SMTPAuthenticationError as e:
        logger.error(
            "Failed to send order confirmation email: SMTP authentication failed (code=%s, response=%s)",
            getattr(e, "smtp_code", None),
            getattr(e, "smtp_error", None),
            exc_info=True,
        )
        return {
            "sent": False,
            "code": "SMTP_AUTHENTICATION_FAILED",
            "error": "SMTP authentication failed. Check SMTP_USERNAME and SMTP_PASSWORD.",
        }

    except smtplib.SMTPRecipientsRefused as e:
        logger.error("Failed to send order confirmation email: recipient refused: %s", e, exc_info=True)
        return {"sent": False, "code": "SMTP_RECIPIENT_REFUSED", "error": "Recipient address was refused by the SMTP server."}

    except smtplib.SMTPSenderRefused as e:
        logger.error("Failed to send order confirmation email: sender refused: %s", e, exc_info=True)
        return {"sent": False, "code": "SMTP_SENDER_REFUSED", "error": "Sender address was refused by the SMTP server. Verify SMTP_FROM_EMAIL."}

    except smtplib.SMTPException as e:
        logger.error(f"Failed to send order confirmation email: SMTP error: {e}", exc_info=True)
        return {"sent": False, "code": "SMTP_ERROR", "error": str(e)}

    except Exception as e:
        logger.error(f"Failed to send order confirmation email: {e}", exc_info=True)
        return {"sent": False, "code": "UNKNOWN_ERROR", "error": str(e)}
