FROM python:3.12-slim

WORKDIR /app

# Install system deps (none needed currently, but placeholder for future OCR/etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY batch.py .
COPY static/ ./static/

# Model weights are auto-downloaded from HuggingFace Hub at startup.
# To use local weights instead, mount: -v ./model:/app/model

# HuggingFace Spaces expects port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
