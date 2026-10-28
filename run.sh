#!/usr/bin/env bash
# Start the VoxParsi UI. Run from Windows with:
#   wsl.exe -e bash -lc "cd /home/mmatinm/Code/VoxParsi && ./run.sh"
# then open http://localhost:8501
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/streamlit run app.py "$@"
