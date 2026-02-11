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

A professional AI-powered customer service chatbot built with Flask, powered by Google's FLAN-T5-XL model (3B parameters), with Supabase database integration and smart multi-table lookups.

## Features

- 🤖 **AI Responses** — Google FLAN-T5-XL (3B params) with few-shot prompting & beam search
- 🔍 **Smart DB Lookups** — Automatically queries orders, customers, and products based on user input
- 💬 **Intent Matching** — 26 precompiled intent patterns for instant responses
- 🛍️ **Product Management** — Full CRUD API with search, category filter, duplicate detection
- 📦 **Order Management** — Order CRUD, status tracking, tracking numbers
- 📊 **Conversation Logging** — All chats logged to Supabase (SQLite fallback)
- 🎨 **Premium UI** — ChatGPT/Claude-inspired dark theme with session persistence
- � **API Key Auth** — Admin/write endpoints protected with `X-API-Key` header
- ⚡ **Rate Limiting** — 30 req/min on chat, 200 req/hr global default
- 🛡️ **XSS Protection** — HTML-escaped bot responses with safe markdown rendering
- 🐳 **Docker Deployment** — Ready for Hugging Face Spaces (model weights cached in image)

## Project Structure

```
ai-customer-chatbot/
├── app.py                          # Slim entry point — Flask factory + blueprint registration
├── config.py                       # Centralized env vars & constants
├── database.py                     # Supabase client + SQLite fallback + conversation logging
├── auth.py                         # Admin API-key decorator
├── models/
│   ├── __init__.py
│   └── ai_model.py                 # FLAN-T5 loading, prompt building, inference
├── services/
│   ├── __init__.py
│   ├── intent_service.py           # Load & match intents from intents.json
│   ├── entity_service.py           # Regex extraction (order #, email, SKU, product name)
│   ├── lookup_service.py           # Supabase queries (orders, products, customers)
│   └── formatter_service.py        # Format DB rows into customer-friendly responses
├── routes/
│   ├── __init__.py
│   ├── chat.py                     # /api/chat, /, /health — main chat + smart lookups
│   ├── admin.py                    # /api/admin/* — logs, stats, debug
│   ├── products.py                 # /api/products — CRUD
│   └── orders.py                   # /api/orders — CRUD + status tracking
├── index.html                      # Chat frontend — dark theme UI
├── intents.json                    # 26 customer service intent categories
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker config for HF Spaces (port 7860)
├── SUPABASE_SETUP.md               # Full database schema & setup guide
├── .env.example                    # Environment variable template
├── .gitignore                      # Git ignore rules
├── .gitattributes                  # HF Spaces LFS config
└── .dockerignore                   # Docker build exclusions
```

## How the Chat Works

When a user sends a message, the chatbot follows a **6-level priority system**:

1. **Order by number** — Detects `ORD-XXXX` patterns → queries `orders` table
2. **Customer/Orders by email** — Detects email addresses → queries orders or customer info based on intent
3. **Product by SKU/name** — For product/pricing/stock intents → queries `products` table
4. **Order tracking prompt** — Order intent but no order number → asks user for it
5. **Intent match** — Matches against 26 keyword patterns → returns canned response
6. **AI fallback** — Sends to FLAN-T5-XL for a generated response

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Supabase

Follow [SUPABASE_SETUP.md](SUPABASE_SETUP.md) to create the `orders`, `order_items`, `products`, and `conversations` tables.

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Supabase URL + key
# Optionally set ADMIN_API_KEY to protect write/admin endpoints
```

### 4. Run

```bash
python app.py
# Runs on http://localhost:7860
```

## API Endpoints

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send a message and get a response |
| `GET` | `/health` | Health check |

### Products

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/products` | List products (`?search=` & `?category=`) |
| `GET` | `/api/products/<id>` | Get product by ID |
| `POST` | `/api/products` | Create product |
| `PUT` | `/api/products/<id>` | Update product |
| `DELETE` | `/api/products/<id>` | Delete product |
| `GET` | `/api/products/duplicates` | List duplicate products |

### Orders

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/orders` | List orders (`?customer_email=` & `?status=`) |
| `GET` | `/api/orders/<id>` | Get order by ID |
| `GET` | `/api/orders/number/<num>` | Get order by order number |
| `POST` | `/api/orders` | Create order with items |
| `PUT` | `/api/orders/<id>` | Update order |
| `PATCH` | `/api/orders/<id>/status` | Update order status + tracking |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/admin/logs` | Conversation logs (`?limit=` & `?session_id=`) |
| `GET` | `/api/admin/stats` | Usage statistics |

> **Note**: POST/PUT/DELETE/PATCH endpoints and all `/api/admin/*` routes require an `X-API-Key` header when `ADMIN_API_KEY` is set in the environment.

## Security & Rate Limiting

- **Rate limiting**: `/api/chat` is limited to 30 requests/minute per IP. Write endpoints are limited to 20/min. Global default is 200/hr.
- **Admin auth**: Set `ADMIN_API_KEY` in your `.env` to protect all write and admin endpoints. Pass the key via `X-API-Key` header.
- **XSS protection**: Bot responses are HTML-escaped before rendering. Only safe markdown (`**bold**`, newlines) is rendered.
- **Auto-detect API URL**: The frontend auto-detects the backend URL from `window.location`, so it works in local dev and production without changes.

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Flask 2.3, Python 3.11 |
| AI Model | Google FLAN-T5-XL (3B params, local) |
| Database | Supabase (PostgreSQL) / SQLite fallback |
| Frontend | Vanilla HTML/CSS/JS, dark theme |
| Deployment | Docker on Hugging Face Spaces |

## Deployment

Deployed on Hugging Face Spaces with Docker.

**Live**: https://huggingface.co/spaces/Seyo009/ai-customer-chatbot
