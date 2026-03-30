#!/usr/bin/env python3
"""
Inference script for fine-tuned FLAN-T5 model.
Can also be used as evaluation on held-out test set.
"""
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
model_path = Path(__file__).parent / "finetuned_model"

if not model_path.exists():
    logger.error(f"Model not found at {model_path}")
    logger.info("Run: python training/finetune.py")
    exit(1)

logger.info(f"Loading fine-tuned model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    model = model.cuda()

logger.info(f"Model loaded on {device}")

# ── INFERENCE FUNCTION ──────────────────────────────────
def inference(prompt: str) -> str:
    """Run inference on the fine-tuned model."""
    input_text = f"Customer: {prompt}"
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_beams=2,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# ── TEST ON EXAMPLES ──────────────────────────────────
if __name__ == "__main__":
    test_prompts = [
        "Where is my order ORD-1234?",
        "How do I return an item?",
        "What's the price of iPhone 15?",
        "Do you offer free shipping?",
    ]
    
    logger.info("\n" + "=" * 60)
    logger.info("INFERENCE TEST")
    logger.info("=" * 60)
    
    for prompt in test_prompts:
        response = inference(prompt)
        print(f"\n📝 Customer: {prompt}")
        print(f"🤖 Assistant: {response}")
    
    logger.info("\n✅ Inference test complete!")
