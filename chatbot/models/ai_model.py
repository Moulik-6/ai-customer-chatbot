"""
AI model — load FLAN-T5 (or other HF models), build prompts, run inference.
"""
import logging
import requests

from ..config import (
    HUGGINGFACE_MODEL, HUGGINGFACE_API_KEY, HUGGINGFACE_API_URL,
    MODEL_TYPE, MODEL_CONFIGS, MOCK_MODE, USE_LOCAL_MODEL,
)

logger = logging.getLogger(__name__)

# ── Prompt builder ────────────────────────────────────────

def _build_flan_prompt(message, context=None):
    """Build a structured few-shot prompt for FLAN-T5 with optional conversation history."""
    history_block = ""
    if context:
        history_block = "Recent conversation:\n"
        for turn in context[-5:]:  # last 5 exchanges
            history_block += f"Customer: {turn['user']}\nAssistant: {turn['bot']}\n"
        history_block += "\n"

    return (
        "You are a professional, friendly customer support assistant for an online store. "
        "Rules: Be concise (2-3 sentences max). Be helpful and empathetic. "
        "Never invent policies, prices, or order details. "
        "If you need more information, ask the customer politely.\n\n"
        "Example 1:\n"
        "Customer: I want to return my purchase\n"
        "Assistant: I'd be happy to help with your return! Our return policy allows returns within 30 days of purchase. "
        "Could you please provide your order number so I can look into this for you?\n\n"
        "Example 2:\n"
        "Customer: My package hasn't arrived yet\n"
        "Assistant: I'm sorry to hear about the delay. To check the status of your delivery, "
        "could you share your order number? I'll look into it right away.\n\n"
        "Example 3:\n"
        "Customer: Do you have any discounts?\n"
        "Assistant: We regularly offer promotions and seasonal discounts! "
        "I'd recommend checking our website or subscribing to our newsletter for the latest deals.\n\n"
        f"{history_block}"
        f"Customer: {message}\n"
        "Assistant:"
    )


# ── Load local model at import time ──────────────────────
LOCAL_MODEL = None

if not MOCK_MODE and USE_LOCAL_MODEL:
    try:
        import torch
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

        logger.info(f"Loading local model: {HUGGINGFACE_MODEL}")
        device = 0 if torch.cuda.is_available() else -1

        is_seq2seq = 'flan' in HUGGINGFACE_MODEL.lower() or 't5' in HUGGINGFACE_MODEL.lower()

        if is_seq2seq:
            tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(HUGGINGFACE_MODEL)
            if device >= 0:
                model = model.cuda()
            LOCAL_MODEL = {'tokenizer': tokenizer, 'model': model, 'type': 'seq2seq'}
        elif MODEL_TYPE == 'classification':
            LOCAL_MODEL = pipeline('text-classification', model=HUGGINGFACE_MODEL, device=device)
        else:
            LOCAL_MODEL = pipeline('text-generation', model=HUGGINGFACE_MODEL, device=device)

        logger.info(f"Local model loaded on {'GPU' if device >= 0 else 'CPU'}")
    except ImportError:
        logger.warning("torch/transformers not installed — falling back to remote inference")
        USE_LOCAL_MODEL = False
    except Exception as e:
        logger.error(f"Failed to load local model: {e}")
        raise ValueError(f"Could not load local model: {e}")


# ── Public API ────────────────────────────────────────────

def query_model(prompt, context=None):
    """
    Route to the right inference backend and return a standardized dict:
      {'type': 'generation'|'classification', 'result': ..., 'model': ...}

    *context* is an optional list of {'user': ..., 'bot': ...} dicts
    representing recent conversation history.
    """
    if MOCK_MODE:
        return _mock_response(prompt)
    if LOCAL_MODEL:
        return _local_inference(prompt, context=context)

    # Check if remote inference is actually available
    if not HUGGINGFACE_API_KEY:
        logger.warning("No HUGGINGFACE_API_KEY set — returning helpful fallback")
        return _unavailable_fallback(prompt)

    try:
        return _remote_inference(prompt, context=context)
    except (ValueError, requests.RequestException, TimeoutError) as e:
        logger.warning(f"Remote inference failed ({e}) — returning fallback")
        return _unavailable_fallback(prompt)


# ── Private helpers ───────────────────────────────────────

def _local_inference(prompt, context=None):
    """Run inference on the locally loaded model."""
    import torch

    try:
        if isinstance(LOCAL_MODEL, dict) and LOCAL_MODEL.get('type') == 'seq2seq':
            tokenizer = LOCAL_MODEL['tokenizer']
            model = LOCAL_MODEL['model']
            device = next(model.parameters()).device

            prompt_text = _build_flan_prompt(prompt, context=context)
            inputs = tokenizer(prompt_text, return_tensors="pt", padding=True,
                               max_length=512, truncation=True).to(device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.3, top_p=0.85, top_k=40,
                    do_sample=True,
                    repetition_penalty=1.3, no_repeat_ngram_size=3,
                    early_stopping=True, num_beams=2,
                )
            return {
                'type': 'generation',
                'result': tokenizer.decode(outputs[0], skip_special_tokens=True),
                'model': HUGGINGFACE_MODEL,
            }

        if MODEL_TYPE == 'generation':
            result = LOCAL_MODEL(prompt, max_length=150, temperature=0.7, top_p=0.9, do_sample=True)
            text = result[0]['generated_text']
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            return {'type': 'generation', 'result': text, 'model': HUGGINGFACE_MODEL}

        # classification
        result = LOCAL_MODEL(prompt)
        scores = sorted(
            [{'label': r['label'], 'score': round(r['score'], 4)} for r in result],
            key=lambda x: x['score'], reverse=True,
        )
        return {
            'type': 'classification', 'result': scores,
            'top_label': scores[0]['label'] if scores else 'unknown',
            'model': HUGGINGFACE_MODEL,
        }
    except Exception as e:
        logger.error(f"Local inference error: {e}")
        raise


def _remote_inference(prompt, context=None):
    """Call the Hugging Face Inference API."""
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    params_key = 'generation' if MODEL_TYPE == 'generation' else 'classification'

    # For generation models, use the full prompt with context
    input_text = prompt
    if MODEL_TYPE == 'generation':
        input_text = _build_flan_prompt(prompt, context=context)

    payload = {"inputs": input_text, "parameters": MODEL_CONFIGS[params_key]['params']}

    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 401:
            raise ValueError("Invalid Hugging Face API key")
        if response.status_code == 429:
            raise requests.RequestException("API rate limit exceeded. Please try again later.")
        if response.status_code == 503:
            raise requests.RequestException("Model is loading. Please try again in a moment.")
        response.raise_for_status()

        result = response.json()

        if MODEL_TYPE == 'generation':
            text = result[0]['generated_text'] if isinstance(result, list) else ''
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            return {'type': 'generation', 'result': text, 'model': HUGGINGFACE_MODEL}

        scores = sorted(result[0], key=lambda x: x.get('score', 0), reverse=True) if isinstance(result, list) and isinstance(result[0], list) else []
        return {
            'type': 'classification', 'result': scores,
            'top_label': scores[0]['label'] if scores else 'unknown',
            'model': HUGGINGFACE_MODEL,
        }
    except requests.Timeout:
        raise TimeoutError("Request to AI service timed out")
    except requests.ConnectionError:
        raise requests.RequestException("Failed to connect to AI service")
    except requests.RequestException:
        raise
    except (ValueError, KeyError) as e:
        raise ValueError(f"Invalid response from Hugging Face API: {e}")


def _unavailable_fallback(prompt):
    """Friendly response when the AI backend is not available."""
    return {
        'type': 'generation',
        'result': (
            "I'm not sure I fully understood that. Here's what I can help with:\n"
            "• **Order tracking** — just provide your order number or email\n"
            "• **Product search** — ask about our products or categories\n"
            "• **Shipping & returns** — policies and timelines\n"
            "• **Account help** — questions about your account\n\n"
            "Try rephrasing your question, or type **help** for more options!"
        ),
        'model': 'fallback',
    }


def _mock_response(prompt):
    """Deterministic mock response for testing."""
    if MODEL_TYPE == 'generation':
        return {'type': 'generation', 'result': f"(mock) You said: {prompt}", 'model': 'mock'}
    return {
        'type': 'classification',
        'result': [{'label': 'POSITIVE', 'score': 0.75}, {'label': 'NEGATIVE', 'score': 0.25}],
        'top_label': 'POSITIVE', 'model': 'mock',
    }
