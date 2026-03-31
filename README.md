---
title: AI Customer Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: run.py
pinned: false
---

# AI Customer Chatbot

A Flask-based customer-support chatbot that combines database lookups, intent matching, and a FLAN-T5 AI fallback to answer questions about orders, products, shipping, and more.

**Live demo**: https://huggingface.co/spaces/Seyo009/ai-customer-chatbot

---

## How Chat Routing Works

Every incoming message is processed through a **three-layer priority system** in `chatbot/routes/chat.py`:

1. **Database lookups (highest priority)**
   - Explicit product/stock list requests → queries `products` table
   - Order number detected (e.g. `ORD-123`) → queries `orders` table
   - Email address detected → queries orders or customer info
   - Product/SKU/name detected with a product intent → queries `products` table
   - Product-like keywords in free text → queries `products` table

2. **Intent matching (fast path)**
   - Matches 26 precompiled patterns from `chatbot/data/intents.json`
   - Returns a deterministic canned response immediately
   - **Intent responses never call the AI model**

3. **FLAN-T5 fallback (last resort)**
   - Only reached when no DB result and no intent match
   - Calls `query_model()` in `chatbot/models/ai_model.py`
   - Defaults to `google/flan-t5-small` (set via `HUGGINGFACE_MODEL`)
   - Supports local inference (`USE_LOCAL_MODEL=true`) or HF Inference API (`HUGGINGFACE_API_KEY`)

---

## API Endpoints

### Chat

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/api/chat` | — | Send a message and receive a response |
| `GET` | `/health` | — | Service health check |
| `POST` | `/api/feedback` | — | Submit thumbs-up/down on a response |

**Request** (`/api/chat`):
```json
{ "message": "Where is my order ORD-123?", "session_id": "optional-uuid" }
```

**Response**:
```json
{
  "success": true,
  "type": "order_lookup",
  "intent": "order_tracking",
  "message": "Where is my order ORD-123?",
  "response": "📦 Order ORD-123 — Status: shipped ...",
  "model": "database"
}
```

### Products

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/api/products` | — | List products (`?search=` & `?category=`) |
| `GET` | `/api/products/<id>` | — | Get product by ID |
| `POST` | `/api/products` | `X-API-Key` | Create a product |
| `PUT` | `/api/products/<id>` | `X-API-Key` | Update a product |
| `DELETE` | `/api/products/<id>` | `X-API-Key` | Delete a product |
| `GET` | `/api/products/duplicates` | `X-API-Key` | List duplicate products |

### Orders

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/api/orders` | — | List orders (`?customer_email=` & `?status=`) |
| `GET` | `/api/orders/<id>` | — | Get order by ID |
| `GET` | `/api/orders/number/<num>` | — | Get order by order number |
| `POST` | `/api/orders` | `X-API-Key` | Create an order with items |
| `PUT` | `/api/orders/<id>` | `X-API-Key` | Update an order |
| `PATCH` | `/api/orders/<id>/status` | `X-API-Key` | Update order status + tracking number |

### Admin

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/api/admin/logs` | `X-API-Key` | Conversation logs (`?limit=` & `?session_id=`) |
| `GET` | `/api/admin/stats` | `X-API-Key` | Usage statistics |

> `X-API-Key` is required on all write and admin endpoints when `ADMIN_API_KEY` is set.

---

## Database Schemas

### Products

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID / serial | Primary key |
| `name` | text | Product display name |
| `description` | text | Short product description |
| `price` | numeric | Price in USD |
| `category` | text | Product category (e.g. `Electronics`) |
| `sku` | text | Unique stock-keeping unit identifier |
| `stock` | integer | Units in stock |
| `image_url` | text | URL to product image (optional) |
| `created_at` | timestamp | Row creation time |

### Orders

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID / serial | Primary key |
| `order_number` | text | Human-readable ID (e.g. `ORD-001`) |
| `customer_name` | text | Full name of the customer |
| `customer_email` | text | Customer email address |
| `status` | text | `pending` / `processing` / `shipped` / `delivered` / `cancelled` / `returned` |
| `total_amount` | numeric | Total order value in USD |
| `tracking_number` | text | Carrier tracking number (optional) |
| `order_date` | timestamp | When the order was placed |

### Order Items

| Field | Type | Description |
|-------|------|-------------|
| `id` | UUID / serial | Primary key |
| `order_id` | UUID / serial | Foreign key → `orders.id` |
| `product_name` | text | Name of the product ordered |
| `product_sku` | text | SKU of the product ordered |
| `quantity` | integer | Number of units |
| `unit_price` | numeric | Price per unit at time of order |
| `subtotal` | numeric | `quantity × unit_price` |

---

## Sample Product Catalog

| SKU | Name | Category | Price | Stock |
|-----|------|----------|-------|-------|
| `IPHONE-15-PRO` | iPhone 15 Pro | Electronics | $999.99 | 50 |
| `MACBOOK-AIR-M2` | MacBook Air M2 | Electronics | $1,099.99 | 30 |
| `IPAD-PRO-12` | iPad Pro 12.9" | Electronics | $1,299.99 | 25 |
| `AIRPODS-PRO-2` | AirPods Pro 2nd Gen | Electronics | $249.99 | 100 |
| `APPLE-WATCH-S9` | Apple Watch Series 9 | Electronics | $399.99 | 60 |
| `SAMSUNG-S24-ULTRA` | Samsung Galaxy S24 Ultra | Electronics | $1,199.99 | 40 |
| `SONY-WH1000XM5` | Sony WH-1000XM5 Headphones | Electronics | $349.99 | 75 |
| `LOGITECH-MX-MASTER` | Logitech MX Master 3S Mouse | Accessories | $99.99 | 120 |
| `ANKER-CHARGER-65W` | Anker 65W USB-C Charger | Accessories | $35.99 | 200 |
| `SAMSUNG-T7-SSD` | Samsung T7 Portable SSD 1TB | Storage | $89.99 | 85 |

---

## Sample Orders

| Order # | Customer | Email | Status | Total | Tracking |
|---------|----------|-------|--------|-------|---------|
| `ORD-001` | Alice Johnson | alice@example.com | `shipped` | $1,249.98 | `1Z999AA10123456784` |
| `ORD-002` | Bob Smith | bob@example.com | `delivered` | $249.99 | `1Z999AA10987654321` |
| `ORD-003` | Carol White | carol@example.com | `processing` | $1,099.99 | — |
| `ORD-004` | David Brown | david@example.com | `pending` | $399.99 | — |
| `ORD-005` | Eve Davis | eve@example.com | `cancelled` | $35.99 | — |

**Example order payload** (POST `/api/orders`):
```json
{
  "order_number": "ORD-006",
  "customer_name": "Frank Miller",
  "customer_email": "frank@example.com",
  "status": "pending",
  "total_amount": 1299.99,
  "items": [
    {
      "product_name": "iPad Pro 12.9\"",
      "product_sku": "IPAD-PRO-12",
      "quantity": 1,
      "unit_price": 1299.99,
      "subtotal": 1299.99
    }
  ]
}
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Database

Follow [SUPABASE_SETUP.md](SUPABASE_SETUP.md) to create the `orders`, `order_items`, `products`, and `conversations` tables in Supabase (a SQLite file is used automatically as fallback).

### 3. Configure Environment

```bash
cp .env.example .env
# Fill in SUPABASE_URL, SUPABASE_KEY, and optionally ADMIN_API_KEY
```

### 4. Run

```bash
python run.py
# Starts on http://localhost:7860
```

---

## Order Number Format

Order numbers follow the pattern `ORD-XXX` (minimum 3 digits, case-insensitive):

| Format | Example |
|--------|---------|
| Standard | `ORD-001`, `ORD-1234` |
| With spaces | `ORD 001` |
| Longer | `ORD-0001-234` |

---

## Fine-Tuning FLAN-T5 (Optional)

You can fine-tune the default model on your own dataset:

```bash
python training/prepare_data.py   # Prepare training data
python training/finetune.py       # Fine-tune (15–120 min depending on hardware)
python training/upload_model_to_hf.py  # Upload to HF Hub
```

Then point the app at your fine-tuned model:

```bash
HUGGINGFACE_MODEL=your-org/your-model
USE_LOCAL_MODEL=false
```

See [training/README.md](training/README.md) for full instructions.

---

## Security & Rate Limiting

- **Chat rate limit**: 30 requests/minute per IP
- **Global default**: 200 requests/hour
- **Write/admin auth**: `X-API-Key` header required when `ADMIN_API_KEY` is set
- **XSS protection**: All bot responses are HTML-escaped before rendering

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Flask 2.3+, Python 3.11 |
| AI Fallback | FLAN-T5-small (default) via Hugging Face Transformers or Inference API |
| Database | Supabase (PostgreSQL) with SQLite fallback |
| Frontend | Vanilla HTML/CSS/JS, dark theme |
| Deployment | Docker on Hugging Face Spaces (port 7860) |

