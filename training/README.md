# Fine-Tuning FLAN-T5 for Customer Service

This guide helps you fine-tune FLAN-T5 using Banking77 + synthetic customer service data, with deployment in 24 hours.

## If You Feel Stuck: Do Exactly This

If you do not have persistent storage in Hugging Face Space, use this flow:

1. Push this repo to a trainer Space (or run locally).
2. Run training:
```bash
python training/prepare_data.py
python training/finetune.py
python training/inference.py
```
3. Immediately upload the trained model to a Hugging Face model repo:
```bash
HF_TOKEN=your_write_token \
HF_MODEL_REPO=seyo009/ai-customer-chatbot-flan-small-ft \
python training/upload_model_to_hf.py
```
4. In your production Space secrets set:
```text
HUGGINGFACE_MODEL=seyo009/ai-customer-chatbot-flan-small-ft
USE_LOCAL_MODEL=false
```
5. Restart the production Space.

This avoids losing the model when Space restarts.

## Overview

- **Model**: FLAN-T5-small (80M params, CPU-friendly)
- **Data**: Banking77 (13K+ examples) + synthetic customer service conversations (20+ examples)
- **Training time**: ~1-2 hours on i5 CPU, ~15-30 min on GPU
- **Total data points**: 13,000+

## Quick Start (4 Steps)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
python training/prepare_data.py
```

**Output**: `training/data/{train,val,test}.jsonl`

This combines:
- ✅ Banking77 dataset (intent classification examples)
- ✅ Synthetic customer service conversations (e-commerce examples)

Note: The data loader uses HTTP/parquet fallback and is compatible with Python 3.14.

### 3. Fine-Tune Model

**Option A: Local CPU (1-2 hours)**
```bash
python training/finetune.py
```

**Option B: GPU (Fast) - Run on Google Colab**
```python
# Upload repo to Colab, then:
!python training/finetune.py  # ~15-30 min on Tesla T4
```

**Option C: HF Spaces Training (Recommended)**
- Use HF Spaces' free GPU
- Push repo → Go to Space settings → Training env

### 4. Test & Deploy

```bash
# Test inference
python training/inference.py

# Upload model to HF model repo
HF_TOKEN=your_write_token \
HF_MODEL_REPO=seyo009/ai-customer-chatbot-flan-small-ft \
python training/upload_model_to_hf.py

# In production Space secrets set:
# HUGGINGFACE_MODEL=seyo009/ai-customer-chatbot-flan-small-ft
# USE_LOCAL_MODEL=false

# Restart app
python run.py
```

## File Structure

```
training/
├── prepare_data.py       # Download Banking77 + generate synthetic data
├── finetune.py           # Fine-tuning script (CPU/GPU optimized)
├── inference.py          # Test inference on fine-tuned model
├── upload_model_to_hf.py # Upload trained model to HF model repo
├── data/                 # (generated after prepare_data.py)
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── finetuned_model/      # (generated after finetune.py)
    └── (model files)
```

## Performance Expectations

### On i5 CPU
- **Batch size**: 4
- **Time per epoch**: ~20-30 min
- **Total training**: ~1-2 hours for 3 epochs

### On GPU (e.g., T4)
- **Batch size**: 16-32
- **Time per epoch**: 2-5 min
- **Total training**: 15-30 min

### Results
- **Banking77 accuracy**: ~75-80% (depends on dataset)
- **Customer service responses**: Natural, contextual, helpful

## Integration

Once trained, deploy via model repo and point production Space to it:

1. Upload local folder with `training/upload_model_to_hf.py`.
2. Set Space secret `HUGGINGFACE_MODEL` to your model repo id.
3. Keep `USE_LOCAL_MODEL=false` in Space.
4. Restart the Flask app:
   ```bash
   python run.py
   ```

The chatbot will now use the fine-tuned model repo for generation.

## Data Details

### Banking77 Examples
- 13,083 customer service queries
- 77 different intent labels
- Real banking customer interactions

### Synthetic Data (e-commerce)
- 20 customer service conversations
- Order tracking, returns, products, shipping
- Matches your chatbot domain

**Combined**: 13,000+ training examples

## Troubleshooting

### Out of Memory (CPU)
```python
# Reduce in finetune.py
BATCH_SIZE = 2  # Instead of 4
GRAD_ACCUM_STEPS = 1
```

### Model Not Loading
```bash
# Check if path exists
ls training/finetuned_model/

# Re-run fine-tuning
python training/finetune.py
```

### Slow Training
- Use GPU (Colab/HF Spaces): ~10x faster
- Or run fewer epochs:
  ```python
  EPOCHS = 1  # Instead of 3
  ```

## Deployment to HF Spaces

1. **Upload fine-tuned model to model repo**:
   ```bash
   HF_TOKEN=your_write_token \
   HF_MODEL_REPO=seyo009/ai-customer-chatbot-flan-small-ft \
   python training/upload_model_to_hf.py
   ```

2. **Update HF Space secrets**:
   ```
   HUGGINGFACE_MODEL=seyo009/ai-customer-chatbot-flan-small-ft
   USE_LOCAL_MODEL=false
   ```

3. **Restart Space** (goes live immediately)

## Next Steps

- [ ] Run `prepare_data.py`
- [ ] Run `finetune.py` (locally or GPU)
- [ ] Test with `inference.py`
- [ ] Update `.env` with model path
- [ ] Restart Flask app
- [ ] Push to repo & deploy

## Support

For issues:
- Check logs: `python training/finetune.py` (verbose output)
- Try Colab for faster iteration
- Monitor HF Space logs after deployment
