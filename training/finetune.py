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
import inspect
from pathlib import Path
from torch.utils.data import Dataset
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


class Seq2SeqChatDataset(Dataset):
    """Lightweight torch dataset to avoid datasets/dill issues on Python 3.14."""

    def __init__(self, rows, tokenizer, max_input_len=512, max_target_len=256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        input_text = f"Customer: {row['input']}"
        target_text = row["output"]

        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
        )
        label_inputs = self.tokenizer(
            text_target=target_text,
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = label_inputs["input_ids"]
        return model_inputs

# ── LOAD DATA ──────────────────────────────────────────────
logger.info("Loading training data...")
train_data = []
val_data = []

for line in open(DATA_DIR / "train.jsonl"):
    train_data.append(json.loads(line))

for line in open(DATA_DIR / "val.jsonl"):
    val_data.append(json.loads(line))
logger.info(f"Loaded JSONL rows: {len(train_data)} train, {len(val_data)} val examples")

# ── LOAD MODEL & TOKENIZER ──────────────────────────────────
logger.info(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

if device == "cuda":
    model = model.cuda()

logger.info(f"Model loaded: {MODEL_NAME}")
logger.info(f"Model size: {model.num_parameters() / 1e6:.1f}M parameters")

# ── BUILD TORCH DATASETS ──────────────────────────────────
train_dataset = Seq2SeqChatDataset(train_data, tokenizer)
val_dataset = Seq2SeqChatDataset(val_data, tokenizer)
logger.info(f"Built torch datasets: {len(train_dataset)} train, {len(val_dataset)} val")

# ── TRAINING ARGUMENTS ──────────────────────────────────────
_ta_params = inspect.signature(TrainingArguments.__init__).parameters
train_args_kwargs = {
    "output_dir": str(OUTPUT_DIR),
    "learning_rate": LEARNING_RATE,
    "per_device_train_batch_size": BATCH_SIZE,
    "per_device_eval_batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
    "num_train_epochs": EPOCHS,
    "warmup_steps": WARMUP_STEPS,
    "weight_decay": 0.01,
    "logging_steps": 50,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "seed": 42,
    # CPU optimizations
    "dataloader_pin_memory": False,
    "optim": "adafactor",  # More memory efficient
    "fp16": device == "cuda",  # Mixed precision if GPU
}

# Transformers changed parameter names across versions
if "evaluation_strategy" in _ta_params:
    train_args_kwargs["evaluation_strategy"] = "epoch"
elif "eval_strategy" in _ta_params:
    train_args_kwargs["eval_strategy"] = "epoch"

if "save_strategy" in _ta_params:
    train_args_kwargs["save_strategy"] = "epoch"

if "no_cuda" in _ta_params:
    train_args_kwargs["no_cuda"] = device == "cpu"
elif "use_cpu" in _ta_params:
    train_args_kwargs["use_cpu"] = device == "cpu"

# Drop unsupported keys to stay version-compatible
train_args_kwargs = {k: v for k, v in train_args_kwargs.items() if k in _ta_params}
training_args = TrainingArguments(**train_args_kwargs)

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
