# Smart Clip MCP - Dockerfile
# Multi-stage build: slim runtime image

# ---- Builder ----
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# ---- Runtime ----
FROM python:3.12-slim AS runtime

LABEL maintainer="Ambrose"
LABEL description="Smart Clip MCP - AI-powered video clipping server"

# FFmpeg + system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Working directory for input videos and output clips
WORKDIR /workspace

# Copy source for editable install reference
COPY --from=builder /build /app
WORKDIR /app

# Environment defaults
ENV SMART_CLIP_TRANSPORT=sse
ENV SMART_CLIP_HOST=0.0.0.0
ENV SMART_CLIP_PORT=8000
ENV SMART_CLIP_WHISPER_MODEL=base
ENV SMART_CLIP_LANGUAGE=zh
# OPENAI_API_KEY must be set at runtime

# Volume for video files
VOLUME /workspace

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

ENTRYPOINT ["smart-clip-mcp"]
CMD ["--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
