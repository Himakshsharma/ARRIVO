#!/bin/sh
# Hugging Face Spaces: one container -> API in the background + dashboard on the public port 7860
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
export ARRIVO_API_URL=http://127.0.0.1:8000
sleep 4
exec streamlit run dashboard/app.py --server.port 7860 --server.address 0.0.0.0 \
     --server.headless true --browser.gatherUsageStats false
