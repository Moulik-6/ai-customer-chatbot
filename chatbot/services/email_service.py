"""
Email service — send order confirmation emails.
"""
import logging
import smtplib
import socket
from email.message import EmailMessage

import requests

from ..config import (
    ORDER_EMAIL_ENABLED,
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USERNAME,
    SMTP_PASSWORD,
    SMTP_FROM_EMAIL,
    SMTP_USE_TLS,
    BREVO_API_KEY,
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


def _send_via_brevo_api(customer_email, order):
    subject = f"Order Confirmation - {order.get('order_number', 'N/A')}"
    body = _build_order_email(order)
    payload = {
        "sender": {"email": SMTP_FROM_EMAIL},
        "to": [{"email": customer_email}],
        "subject": subject,
        "textContent": body,
    }
    headers = {
        "accept": "application/json",
        "api-key": BREVO_API_KEY,
        "content-type": "application/json",
    }

    try:
        response = requests.post(
            "https://api.brevo.com/v3/smtp/email",
            json=payload,
            headers=headers,
            timeout=15,
        )
        if response.status_code in (200, 201, 202):
            logger.info(f"Order confirmation email sent via Brevo API to {customer_email} for {order.get('order_number')}")
            return {"sent": True}

        err_msg = response.text[:500]
        logger.error("Brevo API send failed: status=%s body=%s", response.status_code, err_msg)
        return {
            "sent": False,
            "code": "BREVO_API_ERROR",
            "error": f"Brevo API send failed ({response.status_code}).",
        }
    except requests.Timeout:
        logger.error("Brevo API send timed out", exc_info=True)
        return {"sent": False, "code": "BREVO_API_TIMEOUT", "error": "Brevo API request timed out."}
    except requests.RequestException as e:
        logger.error(f"Brevo API send failed: {e}", exc_info=True)
        return {"sent": False, "code": "BREVO_API_REQUEST_FAILED", "error": str(e)}


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

    # Prefer HTTPS delivery from restricted hosting environments (like HF Spaces).
    if BREVO_API_KEY:
        api_result = _send_via_brevo_api(customer_email, order)
        if api_result.get("sent"):
            return api_result
        logger.warning(
            "Brevo API delivery failed; falling back to SMTP. code=%s error=%s",
            api_result.get("code"),
            api_result.get("error"),
        )

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

    except (socket.timeout, TimeoutError):
        logger.error("Failed to send order confirmation email: SMTP connection timed out", exc_info=True)
        return {"sent": False, "code": "SMTP_TIMEOUT", "error": "SMTP connection timed out."}

    except Exception as e:
        logger.error(f"Failed to send order confirmation email: {e}", exc_info=True)
        return {"sent": False, "code": "UNKNOWN_ERROR", "error": str(e)}
