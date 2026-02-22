#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$REPO_DIR/.venv/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "Error: virtualenv not found at $REPO_DIR/.venv" >&2
  echo "Create it first, e.g.: python3 -m venv .venv && $REPO_DIR/.venv/bin/python -m pip install -r requirements.txt" >&2
  exit 1
fi

"$VENV_PY" "$REPO_DIR/main.py" "$@"
