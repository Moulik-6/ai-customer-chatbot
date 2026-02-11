import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from dotenv import load_dotenv
import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
CORS(app)

# Configuration
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
MOCK_MODE = os.getenv('MOCK_MODE', 'false').strip().lower() in ('1', 'true', 'yes')
MODEL_TYPE = os.getenv('MODEL_TYPE', 'generation')  # 'generation' or 'classification'
HUGGINGFACE_MODEL = os.getenv('HUGGINGFACE_MODEL', 'gpt2')

# Model configurations
MODEL_CONFIGS = {
    'generation': {
        'default': 'gpt2',
        'alternatives': ['distilgpt2', 'mistralai/Mistral-7B-Instruct-v0.1'],
        'params': {
            'max_length': 150,
            'temperature': 0.7,
            'top_p': 0.9,
        }
    },
    'classification': {
        'default': 'distilbert-base-uncased-finetuned-sst-2-english',
        'alternatives': ['bert-base-uncased'],
        'params': {
            'top_k': 2,
        }
    }
}

HUGGINGFACE_API_BASE = os.getenv(
    'HUGGINGFACE_API_BASE',
    'https://router.huggingface.co/hf-inference/models'
)
HUGGINGFACE_API_URL = f"{HUGGINGFACE_API_BASE}/{HUGGINGFACE_MODEL}"

# Validate API key on startup
if not HUGGINGFACE_API_KEY and not MOCK_MODE:
    logger.error("HUGGINGFACE_API_KEY not found in environment variables")
    raise ValueError("HUGGINGFACE_API_KEY environment variable is required")

logger.info(
    f"Initialized with model: {HUGGINGFACE_MODEL} (Type: {MODEL_TYPE}, Mock: {MOCK_MODE})"
)

# Load local model if not in mock mode
LOCAL_MODEL = None
if not MOCK_MODE:
    try:
        logger.info(f"Loading local model: {HUGGINGFACE_MODEL}")
        device = 0 if torch.cuda.is_available() else -1
        
        # Check if using a seq2seq model (like FLAN-T5)
        is_seq2seq = 'flan' in HUGGINGFACE_MODEL.lower() or 't5' in HUGGINGFACE_MODEL.lower()
        
        if is_seq2seq:
            # For seq2seq models like FLAN-T5, load directly with model class
            logger.info(f"Loading seq2seq model: {HUGGINGFACE_MODEL}")
            tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(HUGGINGFACE_MODEL)
            if device >= 0:
                model = model.cuda()
            LOCAL_MODEL = {'tokenizer': tokenizer, 'model': model, 'type': 'seq2seq'}
        elif MODEL_TYPE == 'classification':
            LOCAL_MODEL = pipeline('text-classification', model=HUGGINGFACE_MODEL, device=device)
        else:  # text-generation
            LOCAL_MODEL = pipeline('text-generation', model=HUGGINGFACE_MODEL, device=device)
        
        logger.info(f"Local model loaded successfully on {'GPU' if device >= 0 else 'CPU'}")
    except Exception as e:
        logger.error(f"Failed to load local model: {str(e)}")
        raise ValueError(f"Could not load local model: {str(e)}")


def query_huggingface(prompt):
    """
    Query Hugging Face API with the given prompt.
    Supports both text generation (GPT2) and classification (DistilBERT) models.
    
    Args:
        prompt (str): The input text for the model
        
    Returns:
        dict: Response containing:
            - 'type': 'generation' or 'classification'
            - 'result': Generated text OR classification scores
            - 'model': Model used
            
    Raises:
        requests.RequestException: If API request fails
        ValueError: If response format is invalid
        TimeoutError: If request times out
    """
    if MOCK_MODE:
        return _mock_response(prompt)

    if LOCAL_MODEL:
        return _local_model_response(prompt)

    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    
    # Build request payload based on model type
    if MODEL_TYPE == 'generation':
        payload = {
            "inputs": prompt,
            "parameters": MODEL_CONFIGS['generation']['params']
        }
    else:  # classification
        payload = {
            "inputs": prompt,
            "parameters": MODEL_CONFIGS['classification']['params']
        }
    
    try:
        logger.debug(f"Sending request to {HUGGINGFACE_API_URL}")
        
        response = requests.post(
            HUGGINGFACE_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Handle HTTP errors
        if response.status_code == 401:
            logger.error("Authentication failed - invalid API key")
            raise ValueError("Invalid Hugging Face API key")
        elif response.status_code == 429:
            logger.error("Rate limit exceeded")
            raise requests.RequestException("API rate limit exceeded. Please try again later.")
        elif response.status_code == 503:
            logger.error("Model is loading or temporarily unavailable")
            raise requests.RequestException("Model is loading. Please try again in a moment.")
        
        response.raise_for_status()
        
        result = response.json()
        logger.debug(f"Received response: {str(result)[:200]}")
        
        # Parse response based on model type
        if MODEL_TYPE == 'generation':
            return _parse_generation_response(result, prompt)
        else:
            return _parse_classification_response(result)
        
    except requests.Timeout as e:
        logger.error(f"Hugging Face API request timed out after 30 seconds")
        raise TimeoutError("Request to AI service timed out") from e
    except requests.ConnectionError as e:
        logger.error(f"Connection error: {str(e)}")
        raise requests.RequestException("Failed to connect to AI service") from e
    except requests.RequestException as e:
        logger.error(f"Hugging Face API error: {str(e)}")
        raise
    except (ValueError, KeyError) as e:
        logger.error(f"Response parsing error: {str(e)}")
        raise ValueError(f"Invalid response format from Hugging Face API: {str(e)}") from e


def _parse_generation_response(response, original_prompt):
    """
    Parse response from text generation model (GPT2, Mistral, etc).
    
    Args:
        response: API response
        original_prompt: Original input prompt
        
    Returns:
        dict: Parsed response with generated text
    """
    try:
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], dict) and 'generated_text' in response[0]:
                generated_text = response[0]['generated_text']
                # Remove the original prompt from the generated text
                if generated_text.startswith(original_prompt):
                    generated_text = generated_text[len(original_prompt):].strip()
                
                return {
                    'type': 'generation',
                    'result': generated_text,
                    'model': HUGGINGFACE_MODEL
                }
        
        raise ValueError(f"Unexpected generation response format: {response}")
        
    except Exception as e:
        logger.error(f"Error parsing generation response: {str(e)}")
        raise


def _parse_classification_response(response):
    """
    Parse response from classification model (DistilBERT, etc).
    
    Args:
        response: API response
        
    Returns:
        dict: Parsed response with classification scores
    """
    try:
        if isinstance(response, list) and len(response) > 0:
            if isinstance(response[0], list):
                # Response is list of scores per label
                scores = response[0]
                # Sort by score descending
                scores_sorted = sorted(scores, key=lambda x: x.get('score', 0), reverse=True)
                
                return {
                    'type': 'classification',
                    'result': scores_sorted,
                    'top_label': scores_sorted[0].get('label') if scores_sorted else 'unknown',
                    'model': HUGGINGFACE_MODEL
                }
        
        raise ValueError(f"Unexpected classification response format: {response}")
        
    except Exception as e:
        logger.error(f"Error parsing classification response: {str(e)}")
        raise


def _mock_response(prompt):
    """
    Return a deterministic local response when MOCK_MODE is enabled.
    """
    if MODEL_TYPE == 'generation':
        return {
            'type': 'generation',
            'result': f"(mock) You said: {prompt}",
            'model': 'mock'
        }

    return {
        'type': 'classification',
        'result': [
            {'label': 'POSITIVE', 'score': 0.75},
            {'label': 'NEGATIVE', 'score': 0.25}
        ],
        'top_label': 'POSITIVE',
        'model': 'mock'
    }


def _local_model_response(prompt):
    """
    Run inference using a locally loaded Hugging Face model.
    """
    try:
        # Check if using a seq2seq model (like FLAN-T5)
        if isinstance(LOCAL_MODEL, dict) and LOCAL_MODEL.get('type') == 'seq2seq':
            # For seq2seq models like FLAN-T5
            tokenizer = LOCAL_MODEL['tokenizer']
            model = LOCAL_MODEL['model']
            device = next(model.parameters()).device
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                'type': 'generation',
                'result': generated_text,
                'model': HUGGINGFACE_MODEL
            }
        elif isinstance(LOCAL_MODEL, dict) and LOCAL_MODEL.get('type') == 'classification':
            # For classification models
            result = LOCAL_MODEL(prompt)
            scores = [{'label': r['label'], 'score': round(r['score'], 4)} for r in result]
            scores_sorted = sorted(scores, key=lambda x: x.get('score', 0), reverse=True)
            
            return {
                'type': 'classification',
                'result': scores_sorted,
                'top_label': scores_sorted[0].get('label') if scores_sorted else 'unknown',
                'model': HUGGINGFACE_MODEL
            }
        elif MODEL_TYPE == 'generation':
            # For causal language models like GPT-2
            result = LOCAL_MODEL(prompt, max_length=150, temperature=0.7, top_p=0.9, do_sample=True)
            generated_text = result[0]['generated_text']
            # Remove the original prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return {
                'type': 'generation',
                'result': generated_text,
                'model': HUGGINGFACE_MODEL
            }
        else:  # classification
            result = LOCAL_MODEL(prompt)
            scores = [{'label': r['label'], 'score': round(r['score'], 4)} for r in result]
            scores_sorted = sorted(scores, key=lambda x: x.get('score', 0), reverse=True)
            
            return {
                'type': 'classification',
                'result': scores_sorted,
                'top_label': scores_sorted[0].get('label') if scores_sorted else 'unknown',
                'model': HUGGINGFACE_MODEL
            }
    except Exception as e:
        logger.error(f"Local model inference error: {str(e)}")
        raise


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint for customer service chatbot.
    
    Expected JSON payload:
    {
        "message": "hello"
    }
    
    Returns:
        JSON response with the chatbot's reply or classification results
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            logger.warning("Empty request body received")
            return jsonify({
                "error": "Request body must be valid JSON",
                "code": "INVALID_REQUEST"
            }), 400
        
        # Validate message field
        message = data.get('message', '').strip()
        if not message:
            logger.warning("Empty message received")
            return jsonify({
                "error": "Message field is required and cannot be empty",
                "code": "EMPTY_MESSAGE"
            }), 400
        
        if len(message) > 2000:
            logger.warning(f"Message too long: {len(message)} characters")
            return jsonify({
                "error": "Message must not exceed 2000 characters",
                "code": "MESSAGE_TOO_LONG"
            }), 400
        
        logger.info(f"Processing {MODEL_TYPE} request: {message[:100]}...")
        
        # Query Hugging Face
        api_response = query_huggingface(message)
        
        # Format response based on model type
        if api_response['type'] == 'generation':
            response_data = {
                "success": True,
                "type": "generation",
                "message": message,
                "response": api_response['result'],
                "model": api_response['model']
            }
        else:  # classification
            response_data = {
                "success": True,
                "type": "classification",
                "message": message,
                "classification": {
                    "top_label": api_response['top_label'],
                    "scores": api_response['result']
                },
                "model": api_response['model']
            }
        
        logger.info(f"Response generated successfully (type: {api_response['type']})")
        return jsonify(response_data), 200
        
    except TimeoutError:
        logger.error("API request timed out")
        return jsonify({
            "error": "Request to AI service timed out. Please try again.",
            "code": "TIMEOUT"
        }), 504
        
    except ValueError as e:
        error_msg = str(e)
        if "Invalid API key" in error_msg:
            return jsonify({
                "error": "Authentication failed",
                "code": "AUTH_ERROR"
            }), 401
        return jsonify({
            "error": "Invalid response from AI service",
            "code": "INVALID_RESPONSE",
            "details": error_msg
        }), 500
        
    except requests.RequestException as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower():
            return jsonify({
                "error": "API rate limit exceeded. Please try again later.",
                "code": "RATE_LIMITED"
            }), 429
        elif "loading" in error_msg.lower():
            return jsonify({
                "error": "Model is loading. Please try again in a moment.",
                "code": "MODEL_LOADING"
            }), 503
        
        logger.error(f"API request failed: {str(e)}")
        return jsonify({
            "error": "Failed to communicate with AI service",
            "code": "SERVICE_ERROR"
        }), 503
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "An unexpected error occurred",
            "code": "INTERNAL_ERROR"
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "ai-customer-chatbot"
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        "error": "Endpoint not found",
        "code": "NOT_FOUND"
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        "error": "Method not allowed",
        "code": "METHOD_NOT_ALLOWED"
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "error": "Internal server error",
        "code": "INTERNAL_ERROR"
    }), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 7860))
    debug = os.getenv('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=False)  # Disable debug mode to avoid memory issues