#!/usr/bin/env bash
# monitor-m4.sh - Strata's monitor for m4serve: the venv's Python (with textual), the key and port from m4.env.
#   ~/strata-m4/monitor-m4.sh [--once | weitere Optionen von strata monitor]
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$H/m4.env"; set +a
SRC="${M4_SRC:-$HOME/strata-m4-src}"
KEY="${M4_API_KEY:-${STRATA_API_KEY:-}}"
ARGS=(--url "http://127.0.0.1:${M4_PORT:-8095}")
if [[ -n "$KEY" && "$KEY" != none ]]; then export M4_MONITOR_KEY="$KEY"; ARGS+=(--credential-env M4_MONITOR_KEY); fi
exec "$H/venv/bin/python" "$SRC/terminal/strata.py" monitor "${ARGS[@]}" "$@"
