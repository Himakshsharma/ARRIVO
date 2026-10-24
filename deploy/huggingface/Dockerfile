# Hugging Face Spaces (Docker SDK): copy this file to the Space root as `Dockerfile`
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 ARRIVO_OFFLINE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt
COPY --chown=user . .
USER user
RUN if [ ! -f models/eta_model.joblib ]; then python -m src.pipeline --trials 15; fi
EXPOSE 7860
CMD ["sh", "scripts/start_all.sh"]
