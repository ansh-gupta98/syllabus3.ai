# ── Stage 1: builder ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy app source
COPY main.py .

# Create a non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

# SQLite DB will be stored here (mount a volume in production)
VOLUME ["/app/data"]

EXPOSE 8000

ENV DATABASE_URL="sqlite:////app/data/edu_platform.db"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]