"""
Test script for Orders API endpoints
Run this after setting up Supabase to test the order management features
"""

import requests
import json
from datetime import datetime, timedelta

# Change this to your deployed URL or use localhost for local testing
BASE_URL = "http://localhost:5000"  # or "https://seyo009-ai-customer-chatbot.hf.space"


def populate_sample_orders():
    """Populate database with sample orders"""
    print("\n📦 Creating sample orders...")
    
    sample_orders = [
        {
            "order_number": "ORD-2026-001",
            "customer_name": "John Smith",
            "customer_email": "john.smith@email.com",
            "customer_phone": "+1-555-0101",
            "shipping_address": "123 Main St, New York, NY 10001",
            "status": "delivered",
            "total_amount": 1549.98,
            "tracking_number": "1Z999AA10123456784",
            "notes": "Delivered to front door",
            "items": [
                {
                    "product_name": "iPhone 15 Pro Max 256GB",
                    "product_sku": "IPHONE-15-PRO-MAX-256",
                    "quantity": 1,
                    "unit_price": 1199.99
                },
                {
                    "product_name": "AirPods Pro (2nd Gen)",
                    "product_sku": "AIRPODS-PRO-2",
                    "quantity": 1,
                    "unit_price": 249.99
                },
                {
                    "product_name": "iPhone 15 Clear Case",
                    "product_sku": "IPHONE-15-CASE-CLR",
                    "quantity": 1,
                    "unit_price": 49.99
                }
            ]
        },
        {
            "order_number": "ORD-2026-002",
            "customer_name": "Sarah Johnson",
            "customer_email": "sarah.j@email.com",
            "customer_phone": "+1-555-0102",
            "shipping_address": "456 Oak Ave, Los Angeles, CA 90001",
            "status": "shipped",
            "total_amount": 3499.99,
            "tracking_number": "1Z999AA10123456785",
            "notes": "Signature required",
            "items": [
                {
                    "product_name": "MacBook Pro 16\" M3 Max",
                    "product_sku": "MBP-16-M3MAX-1TB",
                    "quantity": 1,
                    "unit_price": 3499.99
                }
            ]
        },
        {
            "order_number": "ORD-2026-003",
            "customer_name": "Michael Chen",
            "customer_email": "m.chen@email.com",
            "customer_phone": "+1-555-0103",
            "shipping_address": "789 Pine Rd, Seattle, WA 98101",
            "status": "processing",
            "total_amount": 1849.97,
            "items": [
                {
                    "product_name": "iPad Pro 12.9\" M2 256GB",
                    "product_sku": "IPAD-PRO-129-M2-256",
                    "quantity": 1,
                    "unit_price": 1099.99
                },
                {
                    "product_name": "Apple Pencil (2nd Gen)",
                    "product_sku": "PENCIL-2",
                    "quantity": 1,
                    "unit_price": 129.99
                },
                {
                    "product_name": "Magic Keyboard for iPad Pro 12.9\"",
                    "product_sku": "MAGIC-KB-129",
                    "quantity": 1,
                    "unit_price": 349.99
                },
                {
                    "product_name": "USB-C to HDMI Adapter",
                    "product_sku": "USBC-HDMI-ADP",
                    "quantity": 2,
                    "unit_price": 69.99
                }
            ]
        },
        {
            "order_number": "ORD-2026-004",
            "customer_name": "Emily Davis",
            "customer_email": "emily.davis@email.com",
            "customer_phone": "+1-555-0104",
            "shipping_address": "321 Elm St, Chicago, IL 60601",
            "status": "pending",
            "total_amount": 799.98,
            "notes": "Gift wrapping requested",
            "items": [
                {
                    "product_name": "Sony WH-1000XM5",
                    "product_sku": "SONY-WH1000XM5",
                    "quantity": 2,
                    "unit_price": 399.99
                }
            ]
        },
        {
            "order_number": "ORD-2026-005",
            "customer_name": "Robert Wilson",
            "customer_email": "r.wilson@email.com",
            "customer_phone": "+1-555-0105",
            "shipping_address": "654 Maple Dr, Austin, TX 78701",
            "status": "cancelled",
            "total_amount": 699.97,
            "notes": "Customer requested cancellation - refund processed",
            "items": [
                {
                    "product_name": "Nintendo Switch OLED",
                    "product_sku": "SWITCH-OLED-WHT",
                    "quantity": 1,
                    "unit_price": 349.99
                },
                {
                    "product_name": "The Legend of Zelda: TOTK",
                    "product_sku": "ZELDA-TOTK",
                    "quantity": 1,
                    "unit_price": 59.99
                },
                {
                    "product_name": "Super Mario Bros Wonder",
                    "product_sku": "MARIO-WONDER",
                    "quantity": 1,
                    "unit_price": 59.99
                },
                {
                    "product_name": "Switch Pro Controller",
                    "product_sku": "SWITCH-PRO-CTRL",
                    "quantity": 1,
                    "unit_price": 69.99
                }
            ]
        },
        {
            "order_number": "ORD-2026-006",
            "customer_name": "Lisa Anderson",
            "customer_email": "lisa.a@email.com",
            "customer_phone": "+1-555-0106",
            "shipping_address": "987 Cedar Ln, Miami, FL 33101",
            "status": "shipped",
            "total_amount": 4299.98,
            "tracking_number": "1Z999AA10123456786",
            "items": [
                {
                    "product_name": "Canon EOS R5 Body",
                    "product_sku": "CANON-R5-BODY",
                    "quantity": 1,
                    "unit_price": 3899.99
                },
                {
                    "product_name": "SanDisk 128GB CFexpress Card",
                    "product_sku": "SD-CFE-128",
                    "quantity": 2,
                    "unit_price": 199.99
                }
            ]
        }
    ]
    
    created_orders = []
    for order in sample_orders:
        response = requests.post(f"{BASE_URL}/api/orders", json=order)
        if response.status_code == 201:
            result = response.json()
            print(f"✅ Created: {order['order_number']} - {order['customer_name']} - ${order['total_amount']:.2f} - {order['status']}")
            created_orders.append(result['order'])
        else:
            print(f"❌ Failed: {order['order_number']} - {response.text}")
    
    return created_orders


def test_get_all_orders():
    """Test getting all orders"""
    print("\n1. Getting all orders...")
    
    response = requests.get(f"{BASE_URL}/api/orders")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Found {data.get('count', 0)} orders")
    
    if data.get('orders'):
        for order in data['orders'][:3]:  # Show first 3
            print(f"  - {order['order_number']}: {order['customer_name']} - ${order['total_amount']} - {order['status']}")


def test_get_order_by_number(order_number):
    """Test getting order by order number"""
    print(f"\n2. Getting order by number: {order_number}...")
    
    response = requests.get(f"{BASE_URL}/api/orders/number/{order_number}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        order = data['order']
        print(f"Order: {order['order_number']}")
        print(f"Customer: {order['customer_name']} ({order['customer_email']})")
        print(f"Status: {order['status']}")
        print(f"Total: ${order['total_amount']}")
        print(f"Items ({len(order['items'])}):")
        for item in order['items']:
            print(f"  - {item['product_name']} x{item['quantity']} @ ${item['unit_price']} = ${item['subtotal']}")


def test_filter_by_customer(email):
    """Test filtering orders by customer email"""
    print(f"\n3. Getting orders for customer: {email}...")
    
    response = requests.get(f"{BASE_URL}/api/orders?customer_email={email}")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Found {data.get('count', 0)} orders for this customer")
    
    if data.get('orders'):
        for order in data['orders']:
            print(f"  - {order['order_number']}: ${order['total_amount']} - {order['status']}")


def test_filter_by_status(status):
    """Test filtering orders by status"""
    print(f"\n4. Getting orders with status: {status}...")
    
    response = requests.get(f"{BASE_URL}/api/orders?status={status}")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Found {data.get('count', 0)} {status} orders")
    
    if data.get('orders'):
        for order in data['orders']:
            print(f"  - {order['order_number']}: {order['customer_name']} - ${order['total_amount']}")


def test_update_order_status(order_id, new_status, tracking_number=None):
    """Test updating order status"""
    print(f"\n5. Updating order status to: {new_status}...")
    
    update_data = {"status": new_status}
    if tracking_number:
        update_data["tracking_number"] = tracking_number
    
    response = requests.patch(f"{BASE_URL}/api/orders/{order_id}/status", json=update_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        order = data['order']
        print(f"✅ Updated: {order['order_number']} -> {order['status']}")
        if order.get('tracking_number'):
            print(f"   Tracking: {order['tracking_number']}")


def test_create_new_order():
    """Test creating a new order"""
    print("\n6. Creating a new order...")
    
    new_order = {
        "order_number": "ORD-2026-999",
        "customer_name": "Test Customer",
        "customer_email": "test@example.com",
        "customer_phone": "+1-555-9999",
        "shipping_address": "999 Test St, Test City, TC 99999",
        "status": "pending",
        "total_amount": 299.98,
        "items": [
            {
                "product_name": "Test Product A",
                "product_sku": "TEST-A",
                "quantity": 2,
                "unit_price": 99.99
            },
            {
                "product_name": "Test Product B",
                "product_sku": "TEST-B",
                "quantity": 1,
                "unit_price": 99.99
            }
        ]
    }
    
    response = requests.post(f"{BASE_URL}/api/orders", json=new_order)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 201:
        data = response.json()
        order = data['order']
        print(f"✅ Created: {order['order_number']}")
        print(f"   Order ID: {order['id']}")
        print(f"   Items: {len(order['items'])}")
        return order['id']
    
    return None


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ORDER API TEST SUITE")
    print("=" * 60)
    print(f"\nTesting endpoint: {BASE_URL}")
    print("\nMake sure:")
    print("1. Supabase is configured (see SUPABASE_SETUP.md)")
    print("2. Orders and order_items tables are created")
    print("3. App is running (python app.py)")
    print("4. SUPABASE_URL and SUPABASE_KEY are in .env")
    
    choice = input("\n👉 Ready to start? (y/n): ").lower()
    
    if choice != 'y':
        print("Exiting...")
        exit()
    
    # Run tests
    try:
        # Populate sample orders
        print("\n" + "=" * 60)
        print("STEP 1: Creating Sample Orders")
        print("=" * 60)
        created_orders = populate_sample_orders()
        
        if not created_orders:
            print("\n❌ No orders created. Check your setup.")
            exit()
        
        # Run query tests
        print("\n" + "=" * 60)
        print("STEP 2: Testing Order Queries")
        print("=" * 60)
        
        test_get_all_orders()
        test_get_order_by_number("ORD-2026-001")
        test_filter_by_customer("john.smith@email.com")
        test_filter_by_status("shipped")
        
        # Test updates
        print("\n" + "=" * 60)
        print("STEP 3: Testing Order Updates")
        print("=" * 60)
        
        if created_orders:
            first_order = created_orders[0]
            test_update_order_status(
                first_order['id'], 
                "shipped", 
                "1Z999AA10999999999"
            )
        
        # Create new order
        print("\n" + "=" * 60)
        print("STEP 4: Testing Order Creation")
        print("=" * 60)
        test_create_new_order()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        print("\n💡 View your data at: https://app.supabase.com")
        print("   Go to Table Editor > orders and order_items")
        print("\n📊 Order Statuses:")
        print("   - pending: Order received, not yet processed")
        print("   - processing: Being prepared for shipment")
        print("   - shipped: On the way to customer")
        print("   - delivered: Successfully delivered")
        print("   - cancelled: Order cancelled")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Check if app is running")
        print("- Verify Supabase credentials in .env")
        print("- Make sure tables are created (see SUPABASE_SETUP.md)")
        print("- Check if products exist for product_id references")
