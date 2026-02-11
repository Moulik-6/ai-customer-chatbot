# 🚀 Quick Reference - Supabase Integration

## What Was Added

### ✅ Database Integration

- **Supabase PostgreSQL** (free tier - 500MB storage)
- Replaces local SQLite with cloud database
- Automatic backups and scaling

### ✅ Two Main Tables

#### 1. **conversations** - Auto-logged chat data

- Every chat interaction is automatically saved
- Includes: session_id, timestamps, intent, model used, IP addresses
- Great for analytics and improving the bot

#### 2. **products** - Product catalog with duplicate tracking

- Full CRUD operations (Create, Read, Update, Delete)
- Search by name/description/SKU
- Category filtering
- Duplicate product detection
- Stock management

## Setup in 3 Steps

### 1️⃣ Create Supabase Account (5 min)

```
1. Go to supabase.com
2. Sign up (free)
3. Create new project
4. Copy URL + API key
```

### 2️⃣ Create Tables (2 min)

```
1. Open SQL Editor in Supabase
2. Copy/paste SQL from SUPABASE_SETUP.md
3. Click Run
```

### 3️⃣ Add Credentials (1 min)

```
HF Spaces > Settings > Secrets:
- SUPABASE_URL = your project URL
- SUPABASE_KEY = your anon key
```

**Done! Push to HF Spaces and it will auto-deploy.**

---

## API Examples

### Chat (auto-logs to DB)

```bash
POST /api/chat
{"message": "hello", "session_id": "user-123"}
```

### Create Product

```bash
POST /api/products
{
  "name": "iPhone 15",
  "price": 999,
  "category": "Electronics",
  "stock": 50
}
```

### Search Products

```bash
GET /api/products?search=iphone&category=Electronics
```

### Mark as Duplicate

```bash
POST /api/products
{
  "name": "iPhone 15 (duplicate)",
  "price": 999,
  "is_duplicate": true,
  "duplicate_of": "original-product-id"
}
```

### Get All Duplicates

```bash
GET /api/products/duplicates
```

### View Chat Logs

```bash
GET /api/admin/logs?limit=50
GET /api/admin/stats
```

---

## Testing Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file
cp .env.example .env
# Edit .env with your Supabase credentials

# 3. Run app
python app.py

# 4. Test products API
python test_products.py
```

---

## Files Changed

- ✅ `app.py` - Added Supabase client + product endpoints
- ✅ `requirements.txt` - Added supabase==2.3.4
- ✅ `.env.example` - Added SUPABASE_URL, SUPABASE_KEY
- ✅ `README.md` - Updated with new features
- 📄 `SUPABASE_SETUP.md` - Complete setup guide
- 📄 `test_products.py` - Test script with sample data

---

## Next Steps

1. **Follow SUPABASE_SETUP.md** for detailed setup
2. **Add secrets to HF Spaces** (Settings > Repository secrets)
3. **Push to deploy**: `git push hf-space main`
4. **Test with**: `python test_products.py` (locally)
5. **View data** at https://app.supabase.com (Table Editor)

---

## Why Supabase?

✅ **Free forever** (500MB storage)  
✅ **No credit card** required  
✅ **PostgreSQL** (powerful & reliable)  
✅ **Automatic backups** included  
✅ **Real-time dashboard** to view data  
✅ **REST API** auto-generated  
✅ **Easy Python integration** (supabase-py)

---

## Fallback

If Supabase credentials are missing, app automatically falls back to local SQLite for conversation logging. Products API will return error `DB_NOT_CONFIGURED`.
