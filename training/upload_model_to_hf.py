#!/usr/bin/env python3
"""Upload fine-tuned model directory to a Hugging Face model repository.

Usage:
  HF_TOKEN=hf_xxx HF_MODEL_REPO=seyo009/ai-customer-chatbot-flan-small-ft \
  python training/upload_model_to_hf.py
"""
from pathlib import Path
import os
import sys

from huggingface_hub import create_repo, upload_folder

MODEL_DIR = Path(__file__).parent / "finetuned_model"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "seyo009/ai-customer-chatbot-flan-small-ft")


def main():
    if not HF_TOKEN:
        print("ERROR: Missing HF_TOKEN (or HUGGINGFACE_API_KEY) environment variable")
        sys.exit(1)

    if not MODEL_DIR.exists():
        print(f"ERROR: Model directory not found: {MODEL_DIR}")
        print("Run training first: python training/finetune.py")
        sys.exit(1)

    print(f"Creating/checking model repo: {HF_MODEL_REPO}")
    create_repo(repo_id=HF_MODEL_REPO, repo_type="model", token=HF_TOKEN, exist_ok=True)

    print(f"Uploading model folder: {MODEL_DIR}")
    upload_folder(
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        folder_path=str(MODEL_DIR),
        token=HF_TOKEN,
        commit_message="Upload fine-tuned FLAN-T5 model",
    )

    print("Upload complete")
    print(f"Model repo: https://huggingface.co/{HF_MODEL_REPO}")


if __name__ == "__main__":
    main()
