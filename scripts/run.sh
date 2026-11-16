#!/usr/bin/env bash
set -eu
AUTOANNOTATOR_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$AUTOANNOTATOR_ROOT"
if [ ! -x .venv/bin/python ]; then
  bash scripts/setup.sh
fi
# .env is a local shell-format configuration file owned by the person running the app.
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
exec .venv/bin/python app.py
