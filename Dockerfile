FROM python:3.11-slim

WORKDIR /app

# Default to no auto-migrations in container runtime
ENV AUTO_MIGRATE_ON_STARTUP=false

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt requirements-postgres.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-postgres.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --home-dir /app --shell /usr/sbin/nologin app \
    && chown -R app:app /app \
    && chmod +x /app/entrypoint.sh

USER app

# Expose port
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]
