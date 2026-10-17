#!/usr/bin/env bash
# start.sh - run llmbench.py in its own virtualenv.
# Any arguments are passed straight through, e.g.:
#   ./start.sh --exclude MiniMax-M3
#   ./start.sh --suites speed,embed
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

VENV=".venv"
PY="${PYTHON:-python3}"
# simultaneous users for the parallel suite: 1 = off, 2 or "2,4" = test them
# (override per run: PARALLEL=2 ./start.sh, or pass --parallel yourself)
PARALLEL="${PARALLEL:-1}"

# --- .env -------------------------------------------------------------------
if [[ ! -f .env ]]; then
    for tpl in env.example .env.example; do
        if [[ -f "$tpl" ]]; then
            cp "$tpl" .env
            chmod 600 .env
            echo "Created .env from $tpl - set LITELLM_API_KEY in it, then run ./start.sh again."
            exit 1
        fi
    done
    echo "No .env found (and no env.example to copy). See README.md." >&2
    exit 1
fi
if grep -qE '^[[:space:]]*LITELLM_API_KEY=sk-change-me' .env; then
    echo ".env still contains the placeholder key - edit LITELLM_API_KEY first." >&2
    exit 1
fi

# --- virtualenv -------------------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
    command -v "$PY" >/dev/null || { echo "$PY not found - install Python 3.10+." >&2; exit 1; }
    "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
        || { echo "Python 3.10+ required ($("$PY" -V))." >&2; exit 1; }
    echo "Creating virtualenv in $VENV ..."
    "$PY" -m venv "$VENV" || {
        echo "venv creation failed - on Debian/Ubuntu: sudo apt install python3-venv" >&2; exit 1; }
    "$VENV/bin/pip" install --quiet --upgrade pip
fi
"$VENV/bin/python" -c 'import requests' 2>/dev/null || "$VENV/bin/pip" install --quiet requests

# --- run --------------------------------------------------------------------
ARGS=("$@")
if [[ "$PARALLEL" != "1" && " $* " != *" --parallel"* ]]; then
    ARGS+=(--parallel "$PARALLEL")
fi
exec "$VENV/bin/python" llmbench.py "${ARGS[@]}"
