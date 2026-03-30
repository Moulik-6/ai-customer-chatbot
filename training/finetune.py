#!/usr/bin/env python3
"""
Fine-tune FLAN-T5-small on customer service data.
Optimized for CPU (but uses GPU if available).

Usage:
    python training/finetune.py
"""
import json
import logging
import torch
import os
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Trainer, TrainingArguments, DataCollatorForSeq2Seq,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── CONFIG ──────────────────────────────────────────────────
MODEL_NAME = "google/flan-t5-small"  # Small = ~80M params, fast on CPU
OUTPUT_DIR = Path(__file__).parent / "finetuned_model"
DATA_DIR = Path(__file__).parent / "data"

# Training hyperparameters (CPU-friendly)
BATCH_SIZE = 4  # Small batch for CPU memory
GRAD_ACCUM_STEPS = 2  # Effective batch = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_STEPS = 500

# Detect device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# ── LOAD DATA ──────────────────────────────────────────────
logger.info("Loading training data...")
train_data = []
val_data = []

for line in open(DATA_DIR / "train.jsonl"):
    train_data.append(json.loads(line))

for line in open(DATA_DIR / "val.jsonl"):
    val_data.append(json.loads(line))

train_dataset = Dataset.from_dict({
    "input": [ex["input"] for ex in train_data],
    "output": [ex["output"] for ex in train_data],
})

val_dataset = Dataset.from_dict({
    "input": [ex["input"] for ex in val_data],
    "output": [ex["output"] for ex in val_data],
})

logger.info(f"Loaded: {len(train_dataset)} train, {len(val_dataset)} val examples")

# ── LOAD MODEL & TOKENIZER ──────────────────────────────────
logger.info(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

if device == "cuda":
    model = model.cuda()

logger.info(f"Model loaded: {MODEL_NAME}")
logger.info(f"Model size: {model.num_parameters() / 1e6:.1f}M parameters")

# ── PREPROCESSING ──────────────────────────────────────────
def preprocess_function(examples):
    """Tokenize input and target."""
    inputs = [f"Customer: {ex}" for ex in examples["input"]]
    targets = examples["output"]
    
    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding=True
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=256, truncation=True, padding=True
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(
    preprocess_function, batched=True, batch_size=32,
    desc="Processing train dataset"
)
val_dataset = val_dataset.map(
    preprocess_function, batched=True, batch_size=32,
    desc="Processing val dataset"
)

# ── TRAINING ARGUMENTS ──────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    logging_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=42,
    # CPU optimizations
    dataloader_pin_memory=False,
    optim="adafactor",  # More memory efficient
    fp16=device == "cuda",  # Mixed precision if GPU
    no_cuda=device == "cpu",
)

# ── DATA COLLATOR ──────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(
    tokenizer, model=model, padding=True
)

# ── TRAINER ──────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# ── TRAIN ──────────────────────────────────────────────────
logger.info("Starting training...")
logger.info(f"  Epochs: {EPOCHS}")
logger.info(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
logger.info(f"  Learning rate: {LEARNING_RATE}")
logger.info(f"  Steps per epoch: ~{len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS)}")

trainer.train()

# ── SAVE FINAL MODEL ──────────────────────────────────────
logger.info(f"Saving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info("\n✅ Fine-tuning complete!")
logger.info(f"Model saved to: {OUTPUT_DIR}")
logger.info(f"Add to .env: FINETUNED_MODEL_PATH={OUTPUT_DIR}")
