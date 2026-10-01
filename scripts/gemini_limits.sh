#!/bin/sh
# Gemini free-tier limits live only in AI Studio (behind a Google login), so CI
# cannot read them. This drives autobrowse's explore browser on the default
# `google` profile, reads the rate-limit table, and feeds it to sync_models.py.
# Then commit + push: the sync-models workflow releases it.
#
#   GEMINI_API_KEY=... scripts/gemini_limits.sh
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AB="$ROOT/../autobrowse"
PORT=9090
TOKEN_FILE="${TMPDIR:-/tmp/}autobrowse/explore-$PORT.token"
ROWS="$(mktemp)"

(cd "$AB" && pnpm -s autobrowse explore google --url 'https://aistudio.google.com/rate-limit' >/dev/null 2>&1) &
trap 'call "{\"cmd\":\"close\"}" >/dev/null 2>&1 || true; rm -f "$ROWS"' EXIT
call() { curl -sf -X POST -H "Authorization: Bearer $(cat "$TOKEN_FILE")" "http://127.0.0.1:$PORT/" -d "$1"; }

i=0; until [ -f "$TOKEN_FILE" ] && call '{"cmd":"eval","js":"1"}' >/dev/null 2>&1; do
  i=$((i+1)); [ $i -gt 60 ] && { echo "explore browser did not start" >&2; exit 1; }; sleep 1
done
sleep 8  # the table renders after the page loads
call '{"cmd":"eval","js":"[...document.querySelectorAll(\"button\")].find(b=>/See more/.test(b.innerText))?.click()"}' >/dev/null
sleep 2
call '{"cmd":"eval","js":"JSON.stringify([...document.querySelectorAll(\"table tr\")].map(tr=>[...tr.cells].map(td=>td.innerText.trim().replace(/\\s+/g,\" \"))))"}' \
  | python3 -c 'import json,sys; r=json.loads(json.load(sys.stdin)["result"]); assert len(r) > 5, "no table: signed out?"; print(json.dumps(r))' > "$ROWS"

"$ROOT/keycycle/.venv/bin/python" "$ROOT/scripts/sync_models.py" --gemini-rows "$ROWS"
