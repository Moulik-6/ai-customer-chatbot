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

### **Orders Table** (Customer Orders)

```sql
CREATE TABLE orders (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    order_number TEXT UNIQUE NOT NULL,
    customer_name TEXT NOT NULL,
    customer_email TEXT NOT NULL,
    customer_phone TEXT,
    shipping_address TEXT,
    order_date TIMESTAMPTZ DEFAULT NOW(),
    status TEXT DEFAULT 'pending',
    total_amount DECIMAL(10, 2) NOT NULL,
    tracking_number TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_orders_customer_email ON orders(customer_email);
CREATE INDEX idx_orders_order_number ON orders(order_number);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_order_date ON orders(order_date);

-- Auto-update updated_at timestamp
CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
```

### **Order Items Table** (Products in Orders)

```sql
CREATE TABLE order_items (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    order_id UUID NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id UUID REFERENCES products(id),
    product_name TEXT NOT NULL,
    product_sku TEXT,
    quantity INTEGER NOT NULL DEFAULT 1,
    unit_price DECIMAL(10, 2) NOT NULL,
    subtotal DECIMAL(10, 2) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
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

## 📦 Order Number Format

When creating orders, use the flexible order number format for automatic chatbot detection:

**Format**: `ORD-XXX` or longer (minimum 3 digits)

**Examples**: `ORD-234`, `ORD-0001-100`, `ORD-1234-567`, `ORD-9999-9999`

**Regex Pattern**: `ORD[-\s]?\d{3,}`

The chatbot automatically extracts and looks up orders matching this pattern in chat messages.

### Sample Order Insert (SQL)

```sql
INSERT INTO orders (
    order_number,
    customer_name,
    customer_email,
    customer_phone,
    shipping_address,
    status,
    total_amount,
    tracking_number,
    notes
) VALUES (
    'ORD-0001-100',
    'John Doe',
    'john@example.com',
    '+1 (555) 123-4567',
    '123 Main St, Springfield, IL 62701',
    'shipped',
    2499.99,
    'TRACK-123456789',
    'Express shipping'
);
```

### Sample Order Items Insert (SQL)

```sql
INSERT INTO order_items (
    order_id,
    product_name,
    product_sku,
    quantity,
    unit_price,
    subtotal
) VALUES (
    (SELECT id FROM orders WHERE order_number='ORD-0001-100'),
    'iPhone 15 Pro',
    'IPHONE-15-PRO',
    1,
    999.99,
    999.99
);
```

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

### **Orders** (CRUD)

- `GET /api/orders` - Get all orders (`?customer_email=` & `?status=`)
- `GET /api/orders/<id>` - Get order by ID
- `GET /api/orders/number/<order_number>` - Get order by order number (e.g., `/api/orders/number/ORD-0001-100`)
- `POST /api/orders` - Create order with items
- `PUT /api/orders/<id>` - Update order
- `PATCH /api/orders/<id>/status` - Update status + tracking number

## 🔒 Security Notes

- The **anon key** is safe to use in frontend (has Row Level Security)
- For production, enable RLS policies in Supabase
- Never commit `.env` to git (already in `.gitignore`)

---

## ⚠️ Row Level Security (RLS) — Why DB Lookups May Appear Broken

Supabase enables **Row Level Security (RLS)** by default on new tables.  
When RLS is active and no policies are defined, **all reads via the anon key return empty results** — even if rows exist in the table.  

This is the most common reason you see:
- `/api/products` returning `[]`
- Order lookups returning "order not found"
- The chatbot falling back to FLAN / intents despite having data in Supabase

### How to fix: add SELECT policies

Go to **Supabase → Authentication → Policies** (or use the SQL Editor) and run:

```sql
-- ⚠️ DEMO ONLY — allows any anonymous visitor to read all rows.
-- In production, restrict to authenticated users or add row-level conditions.

-- Allow public read access on products
CREATE POLICY "Allow public read on products"
ON products FOR SELECT
USING (true);

-- Allow public read access on orders
CREATE POLICY "Allow public read on orders"
ON orders FOR SELECT
USING (true);

-- Allow public read access on order_items
CREATE POLICY "Allow public read on order_items"
ON order_items FOR SELECT
USING (true);
```

> **Production warning**: the policies above allow *any* visitor to read *all* rows.  
> For a real deployment you should restrict order reads to the authenticated owner, for example:  
> `USING (customer_email = auth.jwt() ->> 'email')`

### Alternative: disable RLS (development only)

```sql
ALTER TABLE products DISABLE ROW LEVEL SECURITY;
ALTER TABLE orders DISABLE ROW LEVEL SECURITY;
ALTER TABLE order_items DISABLE ROW LEVEL SECURITY;
```

> Only do this for local development or a private sandbox — never on a production database.

### Verify RLS is the problem

Use the built-in debug endpoint (requires `ADMIN_API_KEY`):

```bash
curl -H "X-API-Key: your-admin-key" https://your-host/api/admin/db_status
```

If `supabase_configured` is `true` but `can_select_products` / `can_select_orders` are `false`, it is almost certainly an RLS / missing-policy issue.

---

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
