---
title: AI Customer Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# AI Customer Chatbot

A professional AI-powered customer service chatbot built with Flask, powered by Google's FLAN-T5-large model, with Supabase database integration.

## Features

- 🤖 **AI Responses**: Google FLAN-T5-large (780M parameters) for intelligent customer service
- 💬 **Intent Matching**: Fast keyword-based responses for common queries
- 📊 **Database Logging**: All conversations logged to Supabase for analytics
- 🛍️ **Product Management**: Full CRUD API for product catalog with duplicate detection
- 🎨 **Premium UI**: ChatGPT/Claude-inspired dark theme interface
- 🐳 **Containerized**: Docker deployment ready
- 🌐 **CORS-enabled**: Easy web integration

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Supabase (Free Database)

Follow [SUPABASE_SETUP.md](SUPABASE_SETUP.md) to:

- Create free Supabase account
- Create `conversations` and `products` tables
- Get your credentials

### 3. Configure Environment

Create `.env` file:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here
HUGGINGFACE_API_KEY=optional
```

### 4. Run Locally

```bash
python app.py
```

## API Endpoints

### Chat API

```bash
# Send message
POST /api/chat
{
  "message": "I need help with my order",
  "session_id": "user-123"
}

# Health check
GET /health
```

### Product Management

```bash
# Get all products
GET /api/products?search=iphone&category=electronics

# Get product by ID
GET /api/products/<id>

# Create product
POST /api/products
{
  "name": "iPhone 15 Pro",
  "price": 999.99,
  "category": "Electronics",
  "sku": "IPHONE-15-PRO",
  "stock": 50,
  "is_duplicate": false
}

# Update product
PUT /api/products/<id>

# Delete product
DELETE /api/products/<id>

# Get duplicates
GET /api/products/duplicates
```

### Order Management

```bash
# Get all orders
GET /api/orders?customer_email=user@example.com&status=shipped

# Get order by ID
GET /api/orders/<id>

# Get order by order number
GET /api/orders/number/ORD-2026-001

# Create order
POST /api/orders
{
  "order_number": "ORD-2026-001",
  "customer_name": "John Smith",
  "customer_email": "john@example.com",
  "customer_phone": "+1-555-0101",
  "shipping_address": "123 Main St, New York, NY 10001",
  "status": "pending",
  "total_amount": 1299.98,
  "items": [
    {
      "product_name": "iPhone 15 Pro",
      "product_sku": "IPHONE-15-PRO",
      "quantity": 1,
      "unit_price": 999.99
    },
    {
      "product_name": "AirPods Pro",
      "product_sku": "AIRPODS-PRO-2",
      "quantity": 1,
      "unit_price": 249.99
    }
  ]
}

# Update order
PUT /api/orders/<id>

# Update order status
PATCH /api/orders/<id>/status
{
  "status": "shipped",
  "tracking_number": "1Z999AA10123456784"
}
```

### Analytics

```bash
# View conversation logs
GET /api/admin/logs?limit=50&session_id=user-123

# View statistics
GET /api/admin/stats
```

## Deployment

Deployed on Hugging Face Spaces with automatic Docker build and deployment.

Visit: https://huggingface.co/spaces/Seyo009/ai-customer-chatbot
