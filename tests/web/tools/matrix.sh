#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
ws=${PIE_MATRIX_WS:-ws://127.0.0.1:28417/v1/ws}
out=${PIE_MATRIX_OUT:-benches/web/results}
mkdir -p "$out"
./tests/web/tools/inferlets.sh
echo "== browser"
: > "$out/matrix-browser.raw"
for id in $(python3 -c 'import json; print(" ".join(e["id"] for e in json.load(open("tests/web/matrix.json"))))'); do
  node tests/web/tools/headless.mjs . "tests/web/matrix.html?only=$id&entry_timeout=${PIE_MATRIX_ENTRY_TIMEOUT_MS:-240000}${PIE_MATRIX_QUERY:-}" --timeout 300000 >> "$out/matrix-browser.raw" 2>&1 || true
done
grep -E "^MATRIX" "$out/matrix-browser.raw" > "$out/matrix-browser.log" || true
echo "== native"
export PIE_MATRIX_ENTRY_TIMEOUT_MS=${PIE_MATRIX_ENTRY_TIMEOUT_MS:-240000}
serve_pid=""
start_server() {
  [ -z "${PIE_MATRIX_SERVE:-}" ] && return 0
  if [ -n "$serve_pid" ]; then kill "$serve_pid" 2>/dev/null || true; wait "$serve_pid" 2>/dev/null || true; fi
  bash -c "$PIE_MATRIX_SERVE" > "$out/matrix-serve.raw" 2>&1 &
  serve_pid=$!
  for _ in $(seq 1 180); do grep -q "Server ready" "$out/matrix-serve.raw" 2>/dev/null && return 0; sleep 1; done
  echo "the native server did not come up"; return 1
}
trap '[ -n "$serve_pid" ] && kill "$serve_pid" 2>/dev/null || true' EXIT
start_server
: > "$out/matrix-native.raw"
for id in $(python3 -c 'import json; print(" ".join(e["id"] for e in json.load(open("tests/web/matrix.json"))))'); do
  node tests/web/tools/matrix-native.mjs "$ws" "$id" >> "$out/matrix-native.raw" 2>&1 && status=0 || status=$?
  if [ "$status" = "3" ]; then start_server; fi
done
grep -E "^MATRIX" "$out/matrix-native.raw" > "$out/matrix-native.log" || true
python3 - "$out" <<'PY'
import sys, json
out = sys.argv[1]
def load(p):
    rows = {}
    for line in open(p):
        if line.startswith("MATRIX "):
            id_, status, output, ms = line[7:].rstrip("\n").split("\t")
            rows[id_] = (status, json.loads(output), ms)
    return rows
b, n = load(f"{out}/matrix-browser.log"), load(f"{out}/matrix-native.log")
entries = {e["id"]: e for e in json.load(open("tests/web/matrix.json"))}
def sampled(id_):
    e = entries.get(id_, {})
    return e.get("sampled", False) or float(e.get("input", {}).get("temperature", 0)) > 0.1
same = ran = 0
print(f"{'inferlet':34s} {'browser':10s} {'native':10s} answer")
for id_ in sorted(set(b) | set(n)):
    bs, bo, bm = b.get(id_, ("missing", "", ""))
    ns, no, nm = n.get(id_, ("missing", "", ""))
    ok = bs == "ok" and ns == "ok"
    ran += ok
    if not ok:
        if bs != "ok" and ns != "ok":
            verdict = "both fail"
        elif sampled(id_):
            verdict = "differs (sampled; a self-check failed on one host)"
        else:
            verdict = "DIVERGE"
    elif bo == no:
        verdict = "same"
    else:
        verdict = "differs (sampled)" if sampled(id_) else "DIFFERS"
    same += verdict == "same"
    print(f"{id_:34s} {bs[:10]:10s} {ns[:10]:10s} {verdict}")
    if bs != "ok": print(f"    browser: {bs}")
    if ns != "ok": print(f"    native:  {ns}")
    if verdict.startswith("DIFFERS"):
        print(f"    browser: {json.dumps(bo)[:160]}")
        print(f"    native:  {json.dumps(no)[:160]}")
print(f"\n{len(set(b)|set(n))} inferlets: {ran} ran on both hosts, {same} answered the same")
PY
