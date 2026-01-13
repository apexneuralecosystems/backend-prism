import asyncio
import concurrent.futures
from apex.payments import create_order, capture_order, get_order

# Create a thread pool executor for running sync Apex functions
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)

# Global client reference - will be set from main.py
_payment_client = None

def set_payment_client(client):
    """Set the payment client to use for PayPal operations"""
    global _payment_client
    _payment_client = client

async def create_payment_order(amount, currency="USD", description="Payment", return_url=None, cancel_url=None):
    """Create PayPal order using apex.payments.create_order"""
    loop = asyncio.get_running_loop()
    # Try with client if available, otherwise let apex use environment variables
    client_to_use = _payment_client if _payment_client is not None else None
    result = await loop.run_in_executor(
        _executor,
        lambda: create_order(
            amount=amount,
            currency=currency,
            description=description,
            return_url=return_url,
            cancel_url=cancel_url,
            save_to_db=False,  # We handle DB saving manually
            client=client_to_use  # Pass client if available, else None (apex will use env vars)
        )
    )
    
    # Extract order_id and approval_url from response
    if isinstance(result, dict):
        if 'order' in result:
            order_data = result['order']
            order_id = result.get('order_id') or order_data.get('id')
            payment_id = result.get('payment_id')
        else:
            order_data = result
            order_id = result.get('id') or result.get('order_id')
            payment_id = result.get('payment_id')
        
        # Extract approval URL from links
        approval_url = None
        links = order_data.get("links", [])
        for link in links:
            if link.get("rel") == "approve":
                approval_url = link.get("href")
                break
        
        if not approval_url:
            approval_url = order_data.get("approval_url") or result.get("approval_url")
        
        if not approval_url:
            raise Exception(f"No approval URL in response")
        
        return {
            "order_id": order_id,
            "approval_url": approval_url,
            "status": order_data.get("status", "created"),
            "payment_id": payment_id,
            "paypal_order_id": order_id
        }
    else:
        raise Exception(f"Unexpected response type from create_order: {type(result)}")

async def capture_payment_order(order_id):
    """Capture PayPal order using apex.payments.capture_order"""
    loop = asyncio.get_running_loop()
    # Try with client if available, otherwise let apex use environment variables
    client_to_use = _payment_client if _payment_client is not None else None
    capture = await loop.run_in_executor(
        _executor,
        lambda: capture_order(
            order_id=order_id,
            update_db=False,  # We handle DB updates manually
            client=client_to_use  # Pass client if available, else None (apex will use env vars)
        )
    )
    
    # Extract capture details
    capture_id = None
    amount = None
    currency = None
    
    if isinstance(capture, dict):
        capture_id = capture.get("id") or capture.get("capture_id")
        if "amount" in capture:
            amount_info = capture["amount"]
            if isinstance(amount_info, dict):
                amount = amount_info.get("value")
                currency = amount_info.get("currency_code")
        elif "purchase_units" in capture:
            for unit in capture["purchase_units"]:
                if "payments" in unit and "captures" in unit["payments"]:
                    for cap in unit["payments"]["captures"]:
                        capture_id = cap.get("id")
                        if "amount" in cap:
                            amount = cap["amount"].get("value")
                            currency = cap["amount"].get("currency_code")
                        break
    
    return {
        "status": capture.get("status", "completed"),
        "capture_id": capture_id,
        "amount": amount,
        "currency": currency,
        "order_id": order_id,
        "paypal_order_id": capture.get("id") or order_id
    }

async def get_payment_order(order_id):
    """Get PayPal order status using apex.payments.get_order"""
    loop = asyncio.get_running_loop()
    # Try with client if available, otherwise let apex use environment variables
    client_to_use = _payment_client if _payment_client is not None else None
    order = await loop.run_in_executor(
        _executor,
        lambda: get_order(order_id=order_id, client=client_to_use)  # Pass client if available, else None
    )
    
    # Extract amount and currency from purchase_units
    amount = None
    currency = None
    if "purchase_units" in order:
        for unit in order["purchase_units"]:
            if "amount" in unit:
                amount = unit["amount"].get("value")
                currency = unit["amount"].get("currency_code")
                break
    
    return {
        "order_id": order.get("id") or order_id,
        "status": order.get("status"),
        "amount": amount,
        "currency": currency
    }
