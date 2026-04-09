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
| `GET` | `/api/admin/db_status` | `X-API-Key` | Supabase connectivity & table access health check |
| `GET` | `/api/admin/debug` | `X-API-Key` | SQLite database status |
| `POST` | `/api/admin/logs/purge` | `X-API-Key` | Delete logs older than N days |

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
| `carrier` | text | Carrier code for live tracking, e.g. `ups`, `fedex`, `usps`, `dhl` (optional) |
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

| Product ID | SKU | Name | Category | Price | Stock |
|------------|-----|------|----------|-------|-------|
| `f7a237a6-8fe8-4dc6-a853-2394b235d4e9` | `IPHONE-15-PRO-MAX` | iPhone 15 Pro Max 256GB | Electronics | $1,199.99 | 45 |
| `64720218-1786-4d24-bd97-f85f1f15c058` | `IPHONE-15-128` | iPhone 15 128GB | Electronics | $799.99 | 120 |
| `c659eb46-43de-4c4e-b148-60bd121de699` | `GALAXY-S24-ULTRA` | Samsung Galaxy S24 Ultra | Electronics | $1,299.99 | 60 |
| `378b6c9d-f4cd-45ee-9e91-d23cd8430df0` | `PIXEL-8-PRO` | Google Pixel 8 Pro | Electronics | $999.99 | 35 |
| `7520fad7-022b-4922-aa24-7a8d419a72eb` | `MBP-14-M3PRO` | MacBook Pro 14" M3 Pro | Electronics | $1,999.99 | 25 |
| `36e32000-b185-4072-98b7-016727aee46a` | `MBA-13-M3` | MacBook Air 13" M3 | Electronics | $1,099.99 | 80 |
| `0d4ca37e-c909-4425-ab5c-b0dad791bc11` | `IPAD-PRO-129` | iPad Pro 12.9" M2 | Electronics | $1,099.99 | 40 |
| `706a748f-9f5b-4a17-ba70-9ae08079f930` | `AIRPODS-PRO-2` | AirPods Pro 2nd Gen | Electronics | $249.99 | 200 |
| `5e6fca07-d35d-4d2c-abd2-123e5d709921` | `SONY-WH1000XM5` | Sony WH-1000XM5 | Electronics | $349.99 | 75 |
| `e4a88690-d8de-41b5-bdd2-36a0c591dc5b` | `AWATCH-S9` | Apple Watch Series 9 | Electronics | $399.99 | 90 |

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

## Troubleshooting: DB Lookups Return Empty

If `/api/products` returns `[]` or order lookups say "not found" despite having data in Supabase, the most likely cause is **Row Level Security (RLS)**.

Supabase enables RLS by default. Without SELECT policies, **the anon key cannot read any rows**.

### Quick check

```bash
curl -H "X-API-Key: your-admin-key" https://your-host/api/admin/db_status
```

If `supabase_configured` is `true` but `can_select_products` / `can_select_orders` are `false`, add SELECT policies (see [SUPABASE_SETUP.md](SUPABASE_SETUP.md) for example SQL).

### Other common causes

| Symptom | Likely cause |
|---------|-------------|
| `DB_NOT_CONFIGURED` error | `SUPABASE_URL` or `SUPABASE_KEY` env vars not set |
| Empty arrays despite rows existing | RLS enabled without SELECT policies |
| Order items missing | FK `order_items.order_id → orders.id` not created |
| Product lookup misses | SKU case mismatch or column name differs from schema |

---

## DB Smoke Test

A lightweight script is included to verify DB connectivity from the command line:

```bash
BASE_URL=https://seyo009-ai-customer-chatbot.hf.space python scripts/db_smoke_test.py

# Include DB status check (requires admin key):
BASE_URL=https://... ADMIN_API_KEY=your-key python scripts/db_smoke_test.py
```

---

## Live Tracking (Stub)

A stub service is available at `chatbot/services/tracking_service.py`. It provides the interface for integrating a third-party carrier tracking provider (AfterShip, EasyPost, Shippo, etc.) without requiring an external dependency today.

To enable live tracking:
1. Choose a provider and add its SDK to `requirements.txt`.
2. Add `TRACKING_API_KEY` to `.env.example` and `chatbot/config.py`.
3. Implement `_call_provider()` in `tracking_service.py`.
4. Add a `carrier` column (e.g. `ups`, `fedex`) to your `orders` table.
5. Optionally expose `GET /api/orders/number/<order_number>/tracking` in `chatbot/routes/orders.py`.

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

## Smoke Testing

`scripts/live_smoke_test.py` is a stdlib-only smoke test that hits the deployed
Space and verifies the core endpoints are healthy.

### Run locally

```bash
# Against the default Space URL
python scripts/live_smoke_test.py

# Against a custom URL
BASE_URL=https://your-space.hf.space python scripts/live_smoke_test.py
```

The script checks:

| # | Endpoint | What is verified |
|---|----------|-----------------|
| 1 | `GET /health` | Returns 200 and `status == "healthy"` |
| 2 | `POST /api/chat` | Intent matching works (greeting / shipping / returns) |
| 3 | `POST /api/chat` | AI fallback returns a non-empty response |
| 4 | `GET /api/products` | Returns 200 with a `products` list |
| 5 | `GET /api/orders` | Returns 200 with an `orders` list |

Each request is retried up to 3 times with exponential backoff.
Exit code `0` means all checks passed; `1` means at least one failed;
`2` means the Space was completely unreachable.

### CI (GitHub Actions)

The workflow `.github/workflows/smoke-test.yml` runs the smoke test
automatically on every pull request and can be triggered manually via
"Run workflow".

Set a **repository variable** (Settings → Variables → Actions) called
`BASE_URL` to point the CI job at a different deployment; if the variable
is absent the default URL is used.

### What `DB_ERROR` likely means

If `/api/products` or `/api/orders` returns:

```json
{"code": "DB_ERROR", "error": "Failed to fetch products"}
```

the most common cause is that the **Supabase project is paused**.
Free-tier Supabase projects are paused automatically after a period of
inactivity.

The smoke test will print an explicit hint in that case:

```
⚠️  HINT: Supabase may be paused.
➜  Unpause the project in the Supabase dashboard: https://supabase.com/dashboard
```

**Fix**: log in to [Supabase dashboard](https://supabase.com/dashboard),
open your project, and click **Restore project** (or **Unpause**).
Once the project is active, re-deploy the Space (or restart it) so the
new connection can be established.

Other possible causes of `DB_ERROR`:

| Cause | How to confirm | Fix |
|-------|---------------|-----|
| `SUPABASE_URL` / `SUPABASE_KEY` not set in Space secrets | `/health` shows no DB fields | Add secrets in HF Space settings |
| Row Level Security blocks reads | Supabase logs show "permission denied" | Add SELECT policies for `products`, `orders`, `order_items` |
| Tables not created | `/api/products` returns DB_ERROR even after unpause | Run the SQL in `SUPABASE_SETUP.md` |

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

