# Live Order Tracking Setup Guide

This guide will help you set up live order tracking integration with AfterShip, which supports **500+ shipping carriers worldwide**.

## 📋 Step 1: Create AfterShip Account

1. Go to [aftership.com](https://www.aftership.com)
2. Sign up for a free account
3. Go to **Settings → API Keys**
4. Create a new API key (or use the existing one)
5. Copy your API key

## 🔧 Step 2: Add API Key to Environment

Add your AfterShip API key to your `.env` file:

```bash
# .env file
AFTERSHIP_API_KEY=your_api_key_here
```

**For Hugging Face Spaces deployment:**
1. Go to your Space settings
2. Add a secret: `AFTERSHIP_API_KEY` = your API key
3. Click "Save"

## 🚚 Step 3: Update Order Tracking Numbers

Make sure your orders in Supabase have `tracking_number` values:

```sql
-- Example: Update an order with tracking number
UPDATE orders 
SET tracking_number = 'YOUR_TRACKING_NUMBER'
WHERE order_number = 'ORD-001';
```

The tracking number format varies by carrier:
- **FedEx**: 12-14 digits (e.g., `7493847539`)
- **UPS**: 1Z followed by 16 characters (e.g., `1Z999AA10123456784`)
- **DHL**: 10-11 digits (e.g., `1234567890`)
- **USPS**: 20-30 digits (e.g., `9400111899223410000000`)

## 🎯 Step 4: Test Live Tracking

Test the integration by asking your chatbot:

```
"What's the status of order ORD-001?"
"Can you track my order with tracking number 1Z999AA10123456784?"
"Where's my package for order ORD-234?"
```

The chatbot will now return:
- ✅ Current delivery status (In Transit, Delivered, Out for Delivery, etc.)
- 📍 Last known location
- 📅 Estimated delivery date
- 📝 Latest tracking event

## 📊 Supported Carriers

AfterShip supports 500+ carriers including:

### **Major Carriers**
- FedEx, UPS, DHL, USPS
- Amazon (Fulfillment), Amazon Logistics
- Royal Mail, Canada Post, Australia Post
- DPD, GLS, Hermes
- Asendia, Borderguru, CouriersPlease
- and many more...

### **Regional Carriers**
- Asia: SF Express, Yunda, ZTO, S.F.
- Europe: GLS, Hermes, DPD, Chronopost
- Americas: Estafeta, SolidTruck

**Full list**: [aftership.com/couriers](https://www.aftership.com/couriers)

## 🔄 How Live Tracking Works

1. **User asks about order status**
2. **Chatbot extracts order number** from message
3. **Looks up order in local DB** (gets tracking number)
4. **Fetches live status from AfterShip** (real-time carrier data)
5. **Caches result for 1 hour** (avoids API rate limits)
6. **Returns formatted response** with:
   - Current status (Delivered, In Transit, etc.)
   - Last location and time
   - Estimated delivery date
   - Recent tracking events

## ⚙️ API Limits & Caching

- **Free tier**: 100 requests/month
- **Paid tier**: 1,000+ requests/month
- **Caching**: Results cached for 1 hour per tracking number
- **Timeout**: 5 seconds max per API call

If AfterShip API is unavailable, the chatbot gracefully falls back to showing the local tracking number.

## 🛠️ Troubleshooting

**Q: Tracking number not found**
- A: Check the tracking number format and carrier. Try the AfterShip website directly to verify.

**Q: API key showing errors**
- A: Verify the key in `.env` is correct. Restart your Flask app: `python run.py`

**Q: Slow responses**
- A: First request fetches from carrier (2-5 sec). Subsequent requests use 1-hour cache (instant).

**Q: "Tracking not available yet"**
- A: New shipments take 24-48 hours to appear in carrier systems.

## 📚 Advanced Options

### Use Alternative Carrier APIs (instead of AfterShip)

If you prefer direct carrier integration:

**FedEx API:**
```python
from chatbot.services.lookup_service import get_fedex_tracking
tracking = get_fedex_tracking('7493847539', api_key='YOUR_KEY')
```

**UPS API:**
```python
from chatbot.services.lookup_service import get_ups_tracking
tracking = get_ups_tracking('1Z999AA10123456784', api_key='YOUR_KEY')
```

These can be added in future updates based on your carrier needs.

## 📞 Support

- **AfterShip Docs**: [documentation.aftership.com](https://documentation.aftership.com)
- **Chatbot Issues**: Check [GitHub issues](https://github.com/Moulik-6/ai-customer-chatbot)

