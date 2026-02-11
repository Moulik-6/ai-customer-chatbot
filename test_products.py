"""
Test script for Product API endpoints
Run this after setting up Supabase to test the product management features
"""

import requests
import json

# Change this to your deployed URL or use localhost for local testing
BASE_URL = "http://localhost:5000"  # or "https://seyo009-ai-customer-chatbot.hf.space"

def test_create_product():
    """Test creating a product"""
    print("\n1. Creating a product...")
    
    product = {
        "name": "iPhone 15 Pro Max 256GB",
        "description": "Latest iPhone with A17 Pro chip, titanium design, and advanced camera system",
        "price": 1199.99,
        "category": "Electronics",
        "sku": "IPHONE-15-PRO-MAX-256",
        "stock": 45,
        "image_url": "https://example.com/iphone15.jpg"
    }
    
    response = requests.post(f"{BASE_URL}/api/products", json=product)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 201:
        return response.json()['product']['id']
    return None


def test_create_duplicate():
    """Test creating a duplicate product"""
    print("\n2. Creating a duplicate product...")
    
    # First, create an original product
    original = {
        "name": "Samsung Galaxy S24 Ultra",
        "price": 1299.99,
        "category": "Electronics",
        "sku": "SAMSUNG-S24-ULTRA-512"
    }
    
    response = requests.post(f"{BASE_URL}/api/products", json=original)
    original_id = response.json()['product']['id'] if response.status_code == 201 else None
    
    # Create a duplicate
    duplicate = {
        "name": "Samsung Galaxy S24 Ultra (Duplicate Listing)",
        "price": 1299.99,
        "category": "Electronics",
        "sku": "SAMSUNG-S24-ULTRA-512-DUP",
        "is_duplicate": True,
        "duplicate_of": original_id
    }
    
    response = requests.post(f"{BASE_URL}/api/products", json=duplicate)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_get_all_products():
    """Test getting all products"""
    print("\n3. Getting all products...")
    
    response = requests.get(f"{BASE_URL}/api/products")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Found {data.get('count', 0)} products")
    print(f"Response: {json.dumps(data, indent=2)}")


def test_search_products():
    """Test searching products"""
    print("\n4. Searching for 'iPhone'...")
    
    response = requests.get(f"{BASE_URL}/api/products?search=iPhone")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Found {data.get('count', 0)} products")
    print(f"Response: {json.dumps(data, indent=2)}")


def test_get_duplicates():
    """Test getting duplicate products"""
    print("\n5. Getting duplicate products...")
    
    response = requests.get(f"{BASE_URL}/api/products/duplicates")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Found {data.get('count', 0)} duplicates")
    print(f"Response: {json.dumps(data, indent=2)}")


def test_update_product(product_id):
    """Test updating a product"""
    print(f"\n6. Updating product {product_id}...")
    
    update_data = {
        "stock": 100,
        "price": 1099.99
    }
    
    response = requests.put(f"{BASE_URL}/api/products/{product_id}", json=update_data)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_delete_product(product_id):
    """Test deleting a product"""
    print(f"\n7. Deleting product {product_id}...")
    
    response = requests.delete(f"{BASE_URL}/api/products/{product_id}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def populate_sample_products():
    """Populate database with sample products"""
    print("\n📦 Populating sample products...")
    
    sample_products = [
        {
            "name": "MacBook Pro 16\" M3 Max",
            "description": "Professional laptop with M3 Max chip, 36GB RAM, 1TB SSD",
            "price": 3499.99,
            "category": "Computers",
            "sku": "MBP-16-M3MAX-1TB",
            "stock": 12
        },
        {
            "name": "Sony WH-1000XM5",
            "description": "Premium noise-cancelling wireless headphones",
            "price": 399.99,
            "category": "Audio",
            "sku": "SONY-WH1000XM5",
            "stock": 67
        },
        {
            "name": "Nintendo Switch OLED",
            "description": "Gaming console with OLED screen",
            "price": 349.99,
            "category": "Gaming",
            "sku": "SWITCH-OLED-WHT",
            "stock": 85
        },
        {
            "name": "iPad Pro 12.9\" M2",
            "description": "Professional tablet with M2 chip and Liquid Retina XDR display",
            "price": 1099.99,
            "category": "Tablets",
            "sku": "IPAD-PRO-129-M2-256",
            "stock": 34
        },
        {
            "name": "Canon EOS R5",
            "description": "Mirrorless camera with 45MP full-frame sensor",
            "price": 3899.99,
            "category": "Cameras",
            "sku": "CANON-R5-BODY",
            "stock": 8
        }
    ]
    
    for product in sample_products:
        response = requests.post(f"{BASE_URL}/api/products", json=product)
        if response.status_code == 201:
            print(f"✅ Created: {product['name']}")
        else:
            print(f"❌ Failed: {product['name']} - {response.text}")


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 PRODUCT API TEST SUITE")
    print("=" * 60)
    print(f"\nTesting endpoint: {BASE_URL}")
    print("\nMake sure:")
    print("1. Supabase is configured (see SUPABASE_SETUP.md)")
    print("2. App is running (python app.py)")
    print("3. SUPABASE_URL and SUPABASE_KEY are in .env")
    
    choice = input("\n👉 Ready to start? (y/n): ").lower()
    
    if choice != 'y':
        print("Exiting...")
        exit()
    
    # Run tests
    try:
        # Populate sample data
        populate_sample_products()
        
        # Run CRUD tests
        product_id = test_create_product()
        test_create_duplicate()
        test_get_all_products()
        test_search_products()
        test_get_duplicates()
        
        if product_id:
            test_update_product(product_id)
            # Uncomment to test delete
            # test_delete_product(product_id)
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        print("\n💡 View your data at: https://app.supabase.com")
        print("   Go to Table Editor > products")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Check if app is running")
        print("- Verify Supabase credentials in .env")
        print("- Make sure tables are created (see SUPABASE_SETUP.md)")
