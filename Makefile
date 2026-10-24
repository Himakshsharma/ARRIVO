# Shortcuts (Linux/macOS/WSL/Git Bash). Windows PowerShell users: run the commands after the colon directly.
.PHONY: setup data train api dashboard test docker lint
setup:      ; python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
train:      ; python -m src.pipeline --trials 30
eda:        ; python -m src.eda
api:        ; uvicorn api.main:app --reload --port 8000
dashboard:  ; streamlit run dashboard/app.py
test:       ; ARRIVO_OFFLINE=1 pytest -q
mlflow:     ; mlflow ui --backend-store-uri ./mlruns
docker:     ; docker compose up --build
lint:       ; ruff check src api dashboard tests
