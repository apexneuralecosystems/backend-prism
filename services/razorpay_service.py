"""
Razorpay payment service for creating orders and verifying payments.
Uses RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET from environment.
Amounts are in paise (INR smallest unit).
"""
import os
from typing import Any, Dict, Optional

_razorpay_client = None


def get_razorpay_client():
    """Get or create Razorpay client from environment."""
    global _razorpay_client
    if _razorpay_client is not None:
        return _razorpay_client
    key_id = os.getenv("RAZORPAY_KEY_ID")
    key_secret = os.getenv("RAZORPAY_KEY_SECRET")
    if not key_id or not key_secret:
        raise ValueError("RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET must be set")
    import razorpay
    _razorpay_client = razorpay.Client(auth=(key_id, key_secret))
    return _razorpay_client


def create_razorpay_order(
    amount_paise: int,
    currency: str = "INR",
    receipt: Optional[str] = None,
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a Razorpay order.
    :param amount_paise: Amount in paise (INR smallest unit).
    :param currency: Currency code (default INR).
    :param receipt: Optional receipt id.
    :param notes: Optional notes (e.g. num_credits, org_email).
    :return: Razorpay order response with id, amount, currency, etc.
    """
    client = get_razorpay_client()
    payload: Dict[str, Any] = {"amount": amount_paise, "currency": currency}
    if receipt:
        payload["receipt"] = receipt
    if notes:
        payload["notes"] = notes
    return client.order.create(payload)


def verify_razorpay_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """
    Verify Razorpay payment signature after checkout.
    :return: True if valid, False otherwise.
    """
    try:
        import razorpay
        client = get_razorpay_client()
        client.utility.verify_payment_signature({
            "razorpay_order_id": order_id,
            "razorpay_payment_id": payment_id,
            "razorpay_signature": signature,
        })
        return True
    except Exception:
        return False
