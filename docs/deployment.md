# Deployment guide

You need: a GitHub account, and a free Render **or** Hugging Face account.

## 0. Push the project to GitHub

```bash
cd arrivo
git init && git add . && git commit -m "Arrivo: ETA & delay risk engine"
git branch -M main
git remote add origin https://github.com/<you>/arrivo.git
git push -u origin main
```

`data/raw/` is git-ignored. The Docker build regenerates the synthetic dataset and trains the models (~3–8 minutes) so the saved models always match the library versions inside the image. To deploy models trained on the **Kaggle** file instead, remove the two `models/...` lines from `.dockerignore` and commit `models/` after training locally with the same library versions as the image.

## Option A — Render (API + dashboard as two services)

1. Render dashboard → **New + → Blueprint** → select your repo. Render reads `render.yaml` and proposes `arrivo-api` and `arrivo-dashboard`.
2. Click **Apply**. Leave `ARRIVO_API_URL` empty for now.
3. When `arrivo-api` is live, open `https://arrivo-api-XXXX.onrender.com/docs` to check Swagger.
4. Open `arrivo-dashboard` → **Environment** → set `ARRIVO_API_URL` to the API's URL (no trailing slash) → **Save** (it redeploys).
5. Optional: set `ARRIVO_LLM_API_KEY` on `arrivo-api` for LLM-written customer alerts.

Notes: free services sleep after ~15 minutes idle; the first request then takes ~30–60 s. Open both URLs a minute before your demo. If the dashboard can't reach the API it still works in embedded mode.

## Option B — Hugging Face Spaces (one container, simplest)

1. huggingface.co → **New Space** → SDK **Docker** → blank template.
2. Clone the Space, copy the whole project into it, then:
   ```bash
   cp deploy/huggingface/Dockerfile ./Dockerfile
   cp deploy/huggingface/README.md ./README.md   # Space config header (port 7860)
   git add . && git commit -m "Deploy Arrivo" && git push
   ```
3. The Space builds and starts `scripts/start_all.sh` (API on 127.0.0.1:8000 inside the container, dashboard on port 7860).
4. Optional: Space **Settings → Variables and secrets** → add `ARRIVO_LLM_API_KEY`.

Keep your GitHub README as the project README; the Space README is only the config header.

## Option C — your laptop with Docker

```bash
docker compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

## Smoke test after any deployment

```bash
curl https://<api-url>/health
curl -X POST https://<api-url>/predict -H "Content-Type: application/json" \
  -d '{"origin_city":"Delhi","destination_city":"Jaipur","dispatch_time":"2025-10-18T10:00:00"}'
```
