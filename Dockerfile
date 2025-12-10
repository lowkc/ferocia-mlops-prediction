# Build stage
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.4.24 /uv /bin/uv

WORKDIR /app
COPY . .
RUN uv build

# Runtime stage
FROM python:3.13-slim

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.4.24 /uv /bin/uv
COPY --from=builder /app/dist/*.whl ./

# Copy application files
COPY confs/ ./confs/
COPY models/ ./models/

# Install package and clean up
RUN uv pip install --system --no-cache $(ls *.whl) && rm -f *.whl /bin/uv

# Create non-root user to run
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Define environment variables
ENV PORT=8000
EXPOSE 8000

# Simple, straightforward command
CMD ["sh", "-c", "exec uvicorn src.serving.api:app --host 127.0.0.1 --port ${PORT}"]
