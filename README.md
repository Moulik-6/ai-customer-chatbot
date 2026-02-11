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
