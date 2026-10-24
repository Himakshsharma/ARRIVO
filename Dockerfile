# ---------------------------------------------------------------------------
# Arrivo – one image, two roles (API or dashboard), chosen by the start command.
#   docker build -t arrivo .
#   docker run -p 8000:8000 arrivo                                  # API  -> http://localhost:8000/docs
#   docker run -p 8501:8501 -e PORT=8501 arrivo scripts/start_dashboard.sh
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    ARRIVO_OFFLINE=1 PORT=8000
# libgomp1 = OpenMP runtime needed by LightGBM/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Train inside the image if no models were copied in -> the pickled models always
# match the library versions installed in this image.
RUN if [ ! -f models/eta_model.joblib ]; then python -m src.pipeline --trials 15; fi \
    && chmod +x scripts/*.sh

# Run as a non-root user
RUN useradd -m arrivo && chown -R arrivo /app
USER arrivo

EXPOSE 8000 8501 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fs http://localhost:${PORT}/health || curl -fs http://localhost:${PORT}/_stcore/health || exit 1
CMD ["scripts/start_api.sh"]
