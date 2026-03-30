#!/usr/bin/env python3
"""
Prepare fine-tuning data: Banking77 + synthetic customer service conversations.
Output: JSONL files for training (instruction-following format).
"""
import json
import logging
from datasets import load_dataset
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create output directory
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

# ── LOAD BANKING77 ──────────────────────────────────────────
logger.info("Loading Banking77 dataset...")
try:
    dataset = load_dataset("banking77", split="train")
    logger.info(f"Banking77 loaded: {len(dataset)} examples")
except Exception as e:
    logger.error(f"Failed to load Banking77: {e}")
    logger.info("Make sure you have: pip install datasets")
    raise

# ── SYNTHETIC DATA GENERATION ──────────────────────────────
# Generated customer service conversations for e-commerce
SYNTHETIC_DATA = [
    # Order Tracking
    {"input": "Where is my order?", "output": "I'd be happy to help you track your order! Could you please provide your order number (e.g., ORD-1234)? Once I have that, I can check the current status and shipping details."},
    {"input": "How do I track my shipment?", "output": "To track your shipment, please provide your order number. I can then check the delivery status, estimated arrival date, and tracking information for you."},
    {"input": "What's the status of order ORD-5678?", "output": "Let me look up order ORD-5678 for you. Once retrieved, I'll show you the current status, tracking number, and estimated delivery date."},
    {"input": "When will my package arrive?", "output": "To provide you with delivery details, I'll need your order number. Please share it, and I can tell you the exact arrival date and current location of your package."},
    
    # Returns & Refunds
    {"input": "How do I return an item?", "output": "I'd be happy to help with your return! Our return policy allows returns within 30 days of purchase for most items. Please provide your order number, and I can guide you through the process."},
    {"input": "What is your return policy?", "output": "Our return policy allows customers to return items within 30 days of purchase. Items must be in original condition. To initiate a return, please provide your order number."},
    {"input": "Can I get a refund?", "output": "Yes, we offer refunds for eligible returns. To process your refund, I'll need your order number. Once verified, refunds are typically processed within 5-7 business days."},
    {"input": "My item is damaged. What should I do?", "output": "I'm sorry to hear your item arrived damaged. Please provide your order number and a photo of the damage. We'll arrange a replacement or full refund immediately."},
    
    # Product Information
    {"input": "Do you have this in stock?", "output": "I can check stock availability for you! Could you please tell me which product you're interested in? You can provide the product name or SKU, and I'll check if it's currently in stock."},
    {"input": "What is the price of iPhone 15?", "output": "Let me check the price of iPhone 15 for you. I can provide current pricing, available colors, and stock status."},
    {"input": "Do you have size L available?", "output": "To check size availability, please let me know which product you're interested in. I'll then check if size L is in stock and provide alternatives if needed."},
    {"input": "What are the product specifications?", "output": "I'd be happy to provide product specifications. Please tell me which product or SKU you'd like to know more about."},
    
    # Payments & Shipping
    {"input": "Do you offer free shipping?", "output": "We offer free shipping on orders over $50. For orders below that, standard shipping is $5.99. Express options are also available at checkout."},
    {"input": "What payment methods do you accept?", "output": "We accept all major credit cards, PayPal, Apple Pay, Google Pay, and more. All payments are secure and encrypted."},
    {"input": "How much does shipping cost?", "output": "Shipping costs depend on your location and delivery speed. Standard shipping is $5.99, and free shipping applies to orders over $50."},
    
    # Account & Loyalty
    {"input": "How do I create an account?", "output": "Creating an account is easy! Click the sign-up button on our website, provide your email and password, and you're all set. You'll earn loyalty points on every purchase."},
    {"input": "Do you have a loyalty program?", "output": "Yes! We have a loyalty program where you earn points on every purchase. Points can be redeemed for discounts on future orders."},
    {"input": "How do I reset my password?", "output": "Click the 'Forgot Password' link on the login page, enter your email, and we'll send you a reset link. Follow the instructions to create a new password."},
]

# ── FORMAT BANKING77 FOR INSTRUCTION TUNING ──────────────
logger.info("Processing Banking77 for instruction tuning...")
banking_formatted = []
for example in dataset:
    banking_formatted.append({
        "input": example['text'],
        "output": f"Classified as: {example['label']}"
    })

# ── COMBINE DATASETS ──────────────────────────────────────
all_data = SYNTHETIC_DATA + banking_formatted
logger.info(f"Combined dataset size: {len(all_data)} examples")
logger.info(f"  - Synthetic: {len(SYNTHETIC_DATA)}")
logger.info(f"  - Banking77: {len(banking_formatted)}")

# ── SPLIT & SAVE ──────────────────────────────────────────
import random
random.seed(42)
random.shuffle(all_data)

# 80/10/10 split
train_size = int(0.8 * len(all_data))
val_size = int(0.1 * len(all_data))

train_data = all_data[:train_size]
val_data = all_data[train_size:train_size + val_size]
test_data = all_data[train_size + val_size:]

logger.info(f"Split sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

# Save as JSONL
for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
    path = data_dir / f"{split}.jsonl"
    with open(path, 'w') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')
    logger.info(f"Saved {split}: {path}")

logger.info("\n✅ Data preparation complete!")
logger.info(f"Next: Run fine-tuning with `python training/finetune.py`")
