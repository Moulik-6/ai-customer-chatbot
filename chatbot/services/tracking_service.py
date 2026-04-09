"""
Tracking service — live carrier tracking stub.

This module provides a placeholder interface for integrating a third-party
carrier tracking provider (e.g. AfterShip, EasyPost, Shippo).

TODO: Choose a provider and implement _call_provider() below.
TODO: Add TRACKING_API_KEY to .env.example and config.py when ready.
TODO: Store carrier + tracking_last_checked_at in the orders table for caching.
"""
import logging

logger = logging.getLogger(__name__)

# TODO: replace with a real provider SDK / HTTP call
_PROVIDER = None  # e.g. aftership, easypost, shippo


def get_tracking_status(tracking_number: str, carrier: str | None = None) -> dict | None:
    """
    Return live tracking status for a shipment.

    Parameters
    ----------
    tracking_number : str
        The carrier-assigned tracking number stored in orders.tracking_number.
    carrier : str | None
        Optional carrier code (e.g. 'ups', 'fedex', 'usps', 'dhl').
        When None the provider may auto-detect the carrier.

    Returns
    -------
    dict | None
        On success, a dict with at least:
            {
                "status": str,          # e.g. "In Transit"
                "last_update": str,     # ISO timestamp or human-readable
                "location": str | None, # last known location
                "events": list[dict],   # tracking events (newest first)
            }
        Returns None when the provider is not configured or lookup fails.

    Notes
    -----
    - This is a *stub* — it always returns None until a provider is wired in.
    - Cache results in the DB (orders.tracking_status, orders.tracking_last_checked_at)
      and only refresh when stale (e.g. > 15 minutes) to avoid rate limits and cost.
    - To expose this via REST, add GET /api/orders/number/<order_number>/tracking
      in chatbot/routes/orders.py.
    """
    if _PROVIDER is None:
        logger.debug(
            "[tracking_service] No provider configured — returning None. "
            "Set _PROVIDER and implement _call_provider() to enable live tracking."
        )
        return None

    try:
        return _call_provider(tracking_number, carrier)
    except Exception as e:
        logger.error(f"[tracking_service] Failed to fetch tracking for {tracking_number!r}: {e}", exc_info=True)
        return None


def _call_provider(tracking_number: str, carrier: str | None) -> dict | None:
    """
    Internal: call the configured third-party provider.

    TODO: implement this function.

    Example skeleton for AfterShip:

        import aftership
        client = aftership.APIv4Client(api_key=TRACKING_API_KEY)
        result = client.trackings.get_tracking(
            slug=carrier or "auto",
            tracking_number=tracking_number,
        )
        events = [
            {
                "timestamp": e["checkpoint_time"],
                "status": e["message"],
                "location": e.get("location"),
            }
            for e in result.get("checkpoints", [])
        ]
        return {
            "status": result.get("tag", "Unknown"),
            "last_update": events[0]["timestamp"] if events else None,
            "location": events[0]["location"] if events else None,
            "events": events,
        }
    """
    raise NotImplementedError("Tracking provider not yet implemented.")
