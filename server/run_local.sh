#!/usr/bin/env bash
# Run the stitch server on this computer (Linux / macOS).  Usage:  ./run_local.sh
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
# (re)install when the environment is missing, half-built, or requirements changed
if [ ! -x .venv/bin/uvicorn ] || [ requirements.txt -nt .venv/.installed ]; then
  if [ ! -x "$PY" ] || ! "$PY" -m pip --version >/dev/null 2>&1; then
    rm -rf .venv
    if ! python3 -m venv .venv; then
      rm -rf .venv
      echo
      echo "Could not create a Python environment. On Ubuntu / Debian run:"
      echo "    sudo apt install python3-venv python3-pip"
      echo "then run ./run_local.sh again."
      exit 1
    fi
  fi
  echo "Installing packages (first run takes ~2 min)..."
  "$PY" -m pip install -q --upgrade pip
  if ! "$PY" -m pip install -q -r requirements.txt; then
    echo; echo "Package install failed (check your internet connection), then run ./run_local.sh again."
    exit 1
  fi
  touch .venv/.installed
fi
echo "Stitch server on http://localhost:${PORT:-8000}  (Ctrl+C to stop)"
exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
