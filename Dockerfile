# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .

# ── Expose port (HF Spaces default) ─────────────────────────────────────────
EXPOSE 7860

# ── Create DB tables on startup, then serve ───────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
