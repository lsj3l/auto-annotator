#!/usr/bin/env bash
set -eu
AUTOANNOTATOR_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$AUTOANNOTATOR_ROOT"
if command -v python3.12 >/dev/null 2>&1; then
  AUTOANNOTATOR_PYTHON=python3.12
else
  AUTOANNOTATOR_PYTHON=python3
fi
"$AUTOANNOTATOR_PYTHON" -c 'import sys; assert sys.version_info >= (3, 9), "Python 3.9 or newer is required; Python 3.12 is recommended."'
if [ ! -x .venv/bin/python ]; then
  "$AUTOANNOTATOR_PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
printf '%s\n' 'Auto Annotator is installed. Start it with bash scripts/run.sh.'
