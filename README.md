---
title: AI Customer Chatbot
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# AI Customer Chatbot

A professional AI-powered customer service chatbot built with Flask and powered by Google's FLAN-T5 model.

## Features

- Real-time customer service responses using FLAN-T5 language model
- Flask REST API with `/api/chat` endpoint
- Containerized with Docker for easy deployment
- CPU-optimized (no GPU required)
- CORS-enabled for web integration

## Local Testing

```bash
python app.py
```

Test the API:

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"I need help with my order"}'
```

## Deployment

Deployed on Hugging Face Spaces with automatic Docker build and deployment.

Visit: https://huggingface.co/spaces/Seyo009/ai-customer-chatbot
