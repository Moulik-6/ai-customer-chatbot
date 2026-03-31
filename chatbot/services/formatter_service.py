"""
Response formatters — turn raw DB rows into customer-friendly messages.
"""

_STATUS_EMOJI = {
    'pending': '⏳', 'processing': '🔄', 'shipped': '📦',
    'delivered': '✅', 'cancelled': '❌',
}


def format_order(order, live_tracking=None):
    """Format a single order with items and optional live tracking."""
    if not order:
        return None

    status = order.get('status', 'unknown').upper()
    total = order.get('total_amount', 0)
    tracking = order.get('tracking_number')
    items = order.get('order_items', [])
    emoji = _STATUS_EMOJI.get(status.lower(), '📋')

    lines = [
        f"{emoji} **Order Status: {status}**",
        f"Order #: {order['order_number']}",
        f"Total: ${total:.2f}",
    ]
    
    # Add live tracking info if available
    if tracking and live_tracking:
        lines.append(f"\n**Live Tracking:**")
        lines.append(f"{live_tracking['status']}")
        if live_tracking.get('description'):
            lines.append(f"{live_tracking['description']}")
        if live_tracking.get('location'):
            lines.append(f"📍 Last location: {live_tracking['location']}")
        if live_tracking.get('message'):
            lines.append(f"📝 {live_tracking['message']}")
        if live_tracking.get('estimated_delivery'):
            lines.append(f"📅 Est. Delivery: {live_tracking['estimated_delivery']}")
    elif tracking:
        lines.append(f"Tracking: {tracking}")

    lines.append(f"\n**Items ({len(items)}):**")
    for item in items:
        lines.append(f"• {item['product_name']} x{item['quantity']} @ ${item['unit_price']:.2f}")

    tips = {
        'SHIPPED': "\n📬 Your order is on the way! Use your tracking number to get delivery updates.",
        'DELIVERED': "\n🎉 Your order has been delivered!",
        'PROCESSING': "\n⚙️ We're preparing your order for shipment. You'll receive tracking info soon.",
        'PENDING': "\n👀 Your order is confirmed and being prepared.",
        'CANCELLED': "\n✋ This order has been cancelled.",
    }
    lines.append(tips.get(status, ''))
    return '\n'.join(lines)


def format_orders_list(orders, email):
    """Format multiple orders for a customer."""
    if not orders:
        return None

    lines = [f"📋 **Orders for {email}** ({len(orders)} found):\n"]
    for o in orders:
        status = o.get('status', 'unknown')
        emoji = _STATUS_EMOJI.get(status, '📋')
        total = o.get('total_amount', 0)
        date = o.get('order_date', '')[:10]
        lines.append(f"{emoji} **{o['order_number']}** — {status.upper()} — ${total:.2f} ({date})")

    lines.append("\nTo see details for a specific order, provide the order number (e.g., ORD-2026-001).")
    return '\n'.join(lines)


def format_product(products):
    """Format one or more products."""
    if not products:
        return None

    if len(products) == 1:
        p = products[0]
        stock = "✅ In Stock" if p.get('stock', 0) > 0 else "❌ Out of Stock"
        lines = [f"🛍️ **{p['name']}**"]
        if p.get('description'):
            lines.append(p['description'])
        lines.append(f"💰 Price: ${p['price']:.2f}")
        lines.append(f"📦 {stock}" + (f" ({p['stock']} available)" if p.get('stock', 0) > 0 else ''))
        if p.get('sku'):
            lines.append(f"SKU: {p['sku']}")
        if p.get('category'):
            lines.append(f"Category: {p['category']}")
        return '\n'.join(lines)

    # Multiple products
    lines = [f"🔍 **Found {len(products)} products:**\n"]
    for p in products:
        stock = "In Stock" if p.get('stock', 0) > 0 else "Out of Stock"
        lines.append(f"• **{p['name']}** — ${p['price']:.2f} ({stock})")
    lines.append("\nWould you like more details about any of these products?")
    return '\n'.join(lines)


def format_product_list(products):
    """Format a list of products (for generic 'what products?' queries)."""
    if not products:
        return None
    lines = ["Here are our available products:\n"]
    for p in products:
        price = f"${p['price']:.2f}" if p.get('price') else 'N/A'
        lines.append(f"• **{p['name']}** — {price}")
    if len(products) >= 10:
        lines.append("\n...and more! Ask about a specific product for details.")
    return '\n'.join(lines)


def format_customer(customer):
    """Format customer profile + recent orders."""
    if not customer:
        return None

    lines = [
        f"👤 **Customer: {customer['name']}**",
        f"📧 Email: {customer['email']}",
    ]
    if customer.get('phone'):
        lines.append(f"📱 Phone: {customer['phone']}")
    if customer.get('address'):
        lines.append(f"📍 Address: {customer['address']}")
    lines.append(f"🛒 Total Orders: {customer['total_orders']}")

    if customer.get('orders'):
        lines.append("\n**Recent Orders:**")
        for o in customer['orders'][:3]:
            status = o.get('status', 'unknown')
            emoji = _STATUS_EMOJI.get(status, '📋')
            lines.append(f"{emoji} {o['order_number']} — {status.upper()} — ${o['total_amount']:.2f}")

    return '\n'.join(lines)
