#!/bin/sh
# Start the Streamlit dashboard (PORT is set by Render/Docker; defaults to 8501)
exec streamlit run dashboard/app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0 \
     --server.headless true --browser.gatherUsageStats false
