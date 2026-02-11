# Supabase Database Setup Guide

This guide will help you set up Supabase for your chatbot's database (products + conversation logging).

## 📋 Step 1: Create Supabase Account

1. Go to [supabase.com](https://supabase.com)
2. Click "Start your project"
3. Sign up with GitHub/Google/Email
4. Create a new project:
   - **Project name**: `ai-chatbot`
   - **Database password**: (save this!)
   - **Region**: Choose closest to you
   - **Pricing plan**: Free

## 🗄️ Step 2: Create Database Tables

Once your project is created, go to **SQL Editor** and run these commands:

### **Conversations Table** (Chat Logs)

```sql
CREATE TABLE conversations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    intent TEXT,
    model_used TEXT,
    response_type TEXT,
    ip_address TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_conversations_timestamp ON conversations(timestamp);
CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE INDEX idx_conversations_intent ON conversations(intent);
```

### **Products Table** (Duplicate Products)

```sql
CREATE TABLE products (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    category TEXT,
    sku TEXT,
    stock INTEGER DEFAULT 0,
    image_url TEXT,
    is_duplicate BOOLEAN DEFAULT FALSE,
    duplicate_of UUID REFERENCES products(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_sku ON products(sku);
CREATE INDEX idx_products_is_duplicate ON products(is_duplicate);
CREATE INDEX idx_products_name ON products(name);

-- Full text search
CREATE INDEX idx_products_search ON products USING GIN(to_tsvector('english', name || ' ' || COALESCE(description, '')));

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_products_updated_at BEFORE UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

## 🔑 Step 3: Get Your Credentials

1. Go to **Project Settings** (gear icon)
2. Click **API** in the sidebar
3. Copy these values:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **anon public key**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` (long string)

## ⚙️ Step 4: Configure Environment Variables

### **Local Development** (.env file)

Create/update your `.env` file:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here
HUGGINGFACE_API_KEY=your-key-here
```

### **Hugging Face Spaces**

1. Go to your Space settings
2. Click **Repository secrets**
3. Add these secrets:
   - `SUPABASE_URL` = `https://your-project.supabase.co`
   - `SUPABASE_KEY` = `your-anon-key`

## 🧪 Step 5: Test the Integration

After deploying, test the endpoints:

### **Test Conversation Logging**

```bash
# Send a chat message - it will auto-log
curl -X POST https://your-space.hf.space/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "hello", "session_id": "test-123"}'

# Check logs in Supabase dashboard > Table Editor > conversations
```

### **Test Product Creation**

```bash
# Create a product
curl -X POST https://your-space.hf.space/api/products \
  -H "Content-Type: application/json" \
  -d '{
    "name": "iPhone 15 Pro",
    "price": 999.99,
    "category": "Electronics",
    "sku": "IPHONE-15-PRO",
    "stock": 50,
    "description": "Latest iPhone model"
  }'

# Get all products
curl https://your-space.hf.space/api/products

# Search products
curl "https://your-space.hf.space/api/products?search=iphone"

# Get duplicates
curl https://your-space.hf.space/api/products/duplicates
```

## 📊 Step 6: View Data in Supabase

1. Go to **Table Editor** in Supabase dashboard
2. Select `conversations` or `products` table
3. View all your data in real-time!

## 🎯 API Endpoints Available

### **Conversations** (Auto-logged)

- Automatically logs every chat interaction
- View logs: `GET /api/admin/logs`
- View stats: `GET /api/admin/stats`

### **Products** (CRUD)

- `GET /api/products` - Get all products (with search/filter)
- `GET /api/products/<id>` - Get product by ID
- `POST /api/products` - Create new product
- `PUT /api/products/<id>` - Update product
- `DELETE /api/products/<id>` - Delete product
- `GET /api/products/duplicates` - Get duplicate products

## 🔒 Security Notes

- The **anon key** is safe to use in frontend (has Row Level Security)
- For production, enable RLS policies in Supabase
- Never commit `.env` to git (already in `.gitignore`)

## 📈 Free Tier Limits

- **Database**: 500MB (plenty for most apps)
- **API calls**: Unlimited
- **Bandwidth**: 2GB/month
- **Rows**: Unlimited

## ✅ You're Done!

Your chatbot now has:

- ✅ Cloud database (Supabase PostgreSQL)
- ✅ Conversation logging
- ✅ Product management with duplicate tracking
- ✅ Real-time dashboard to view data
- ✅ Automatic backups (by Supabase)
