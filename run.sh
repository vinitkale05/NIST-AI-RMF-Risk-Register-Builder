#!/bin/bash
set -e

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "🛡️ Starting NIST AI RMF Risk Register Builder on Streamlit..."
streamlit run app.py
