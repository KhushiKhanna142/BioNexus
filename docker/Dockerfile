# Multi-stage build: heavy build deps never ship in the final image.
FROM python:3.11-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir uv

COPY pyproject.toml .
RUN uv pip install --system --no-cache .

# ---- runtime stage ----
FROM python:3.11-slim AS runtime

# Non-root user — never run training/inference containers as root in prod.
RUN useradd --create-home --shell /bin/bash mosaic
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ ./src/
COPY configs/ ./configs/
ENV PYTHONPATH=/app/src

USER mosaic

ENTRYPOINT ["python", "-m", "mosaic.train"]
CMD ["--config", "configs/training/default.yaml"]
