FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model weights so they're cached in the Docker layer
# This avoids re-downloading ~6GB on every container restart
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    AutoTokenizer.from_pretrained('google/flan-t5-xl'); \
    AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-xl')"

COPY . .

ENV FLASK_ENV=production
ENV MOCK_MODE=false

EXPOSE 7860

CMD ["python", "run.py"]
