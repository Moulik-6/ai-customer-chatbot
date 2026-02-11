FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV FLASK_ENV=production
ENV MOCK_MODE=false
ENV USE_LOCAL_MODEL=false
ENV CORS_ORIGINS=https://ai-customer-chatbot-tau.vercel.app,https://seyo009-ai-customer-chatbot.hf.space,http://localhost:7860

EXPOSE 7860

CMD ["python", "run.py"]
