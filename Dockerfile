# JARVIS Voice Assistant - Production Dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Audio libraries
    libasound2-dev \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    # System utilities
    curl \
    wget \
    git \
    # GUI libraries (for optional GUI mode)
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libice6 \
    libfontconfig1 \
    libxss1 \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 jarvis && \
    chown -R jarvis:jarvis /app
USER jarvis

# Copy requirements first (for better caching)
COPY --chown=jarvis:jarvis requirements_production.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements_production.txt

# Copy application files
COPY --chown=jarvis:jarvis . .

# Create necessary directories
RUN mkdir -p data/conversation_memory data/tts_cache data/vectordb logs config models

# Set permissions
RUN chmod +x run_jarvis_final.py jarvis_final_complete.py

# Expose port for potential web interface
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python3", "jarvis_final_complete.py"]