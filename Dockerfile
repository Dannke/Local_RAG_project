# syntax=docker/dockerfile:1

FROM python:3.11-slim

# System deps: build tools for faiss/numpy wheels if needed
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/.cache/huggingface

WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source and install the package
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install --no-deps -e .

# Copy app files
COPY app.py .
COPY assets/ ./assets/
COPY .streamlit/ ./.streamlit/
COPY data/ ./data/
COPY .env.example ./

# Create writable data dir + keep a gitkeep for the persisted chat volume
RUN mkdir -p /app/data/chats && touch /app/data/chats/.gitkeep

# Expose Streamlit default port
EXPOSE 8501

# Non-root user for safety
RUN useradd --create-home --uid 1000 raguser && \
    chown -R raguser:raguser /app /data
USER raguser

# Streamlit entrypoint, headless, no telemetry
ENTRYPOINT ["streamlit", "run", "app.py", "--server.headless=true", "--server.port=8501", "--server.address=0.0.0.0"]
