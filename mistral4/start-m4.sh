#!/usr/bin/env bash
# start-m4.sh - m4serve mit den Einstellungen aus ~/strata-m4/m4.env (Port, Host, API-Key, Engine-Optionen).
#   ~/strata-m4/start-m4.sh            im Vordergrund (Strg+C beendet)
#   nohup ~/strata-m4/start-m4.sh > ~/strata-m4/logs/server.log 2>&1 &
H="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a; source "$H/m4.env"; set +a        # exportiert auch M4_API_KEY (m4serve liest ihn aus der Umgebung)
exec "$H/venv/bin/python" "$H/m4serve.py" --engine "$H/bin/strata-m4" --model "$M4_MODEL" --tokenizer "$M4_TOKENIZER" \
  --engine-args "$M4_ENGINE_ARGS --heat $H/mistral4.m4heat" \
  --host "${M4_HOST:-127.0.0.1}" --port "${M4_PORT:-8095}" ${M4_TEMPERATURE:+--temperature "$M4_TEMPERATURE"} \
  ${M4_PARALLEL:+--parallel "$M4_PARALLEL"} "$@"
