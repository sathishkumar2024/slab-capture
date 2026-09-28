#!/usr/bin/env bash
# Run the stitch server on this computer (Linux / macOS).  Usage:  ./run_local.sh
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
echo "Stitch server on http://localhost:${PORT:-8000}  (Ctrl+C to stop)"
exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
