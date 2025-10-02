#!/usr/bin/env bash
set -e
python -m venv .venv
# shellcheck disable=SC1091
if [ -d ".venv/bin" ]; then source .venv/bin/activate; else source .venv/Scripts/activate; fi
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
echo "✅ Environment ready. Activate with: source .venv/bin/activate"
