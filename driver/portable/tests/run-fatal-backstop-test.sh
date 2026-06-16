#!/usr/bin/env bash
# Hermetic test for the embedded-driver fatal-signal backstop.
#
# Compiles fatal_backstop_test.cpp (which includes the real
# driver/portable/src/fatal_backstop.hpp) and asserts the contract across
# all five fatal signals:
#   * marked driver thread        -> diagnostic emitted + dies by the signal
#   * unmarked thread             -> NO diagnostic + chains to prev (recovers)
#   * marked + recovering prev    -> diagnostic AND chains to prev (recovers)
#
# No pie/ggml/Metal build needed; runs on macOS and Linux.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$here/fatal_backstop_test.cpp"
bin="$(mktemp -t fatal_backstop_test.XXXXXX)"
trap 'rm -f "$bin"' EXIT

cxx="${CXX:-c++}"
echo "== compiling fatal_backstop_test ($cxx, c++20) =="
"$cxx" -std=c++20 -Wall -Wextra -O0 -g "$src" -o "$bin" -lpthread

pass=0
fail=0
note() { printf '  %s\n' "$*"; }
ok()   { pass=$((pass + 1)); printf 'PASS: %s\n' "$*"; }
bad()  { fail=$((fail + 1)); printf 'FAIL: %s\n' "$*"; }

DIAG_RE='\[pie-driver-portable\] fatal: caught'

# Existing cases run under Linux semantics (wasm traps reach the handler):
# that is the regime where the thread-local gate matters.
export PIE_BACKSTOP_FORCE_WASM_TRAPS=1

# marked: diagnostic present + terminated by the raised signal.
test_marked() {
    local signame="$1"
    local signum; signum="$("$bin" signum "$signame")"
    local err; err="$(mktemp)"
    set +e
    "$bin" marked "$signame" 2>"$err"
    local rc=$?
    set -e
    local body; body="$(cat "$err")"; rm -f "$err"

    if [[ $rc -le 128 ]]; then
        bad "marked $signame: expected death-by-signal (rc>128), got rc=$rc"; return
    fi
    if [[ $((rc - 128)) -ne $signum ]]; then
        bad "marked $signame: died by $((rc - 128)), expected $signum"; return
    fi
    if ! grep -Eq "$DIAG_RE $signame \(model=/test/model.gguf\)" <<<"$body"; then
        bad "marked $signame: diagnostic line missing/wrong: [$body]"; return
    fi
    ok "marked $signame: diagnostic emitted + died by $signame (rc=$rc)"
}

# unmarked: NO diagnostic + recovered via chain to the prev handler.
test_unmarked() {
    local signame="$1"
    local out err; out="$(mktemp)"; err="$(mktemp)"
    set +e
    "$bin" unmarked "$signame" >"$out" 2>"$err"
    local rc=$?
    set -e
    local sout serr; sout="$(cat "$out")"; serr="$(cat "$err")"; rm -f "$out" "$err"

    if [[ $rc -ne 0 ]]; then
        bad "unmarked $signame: expected recovery (rc=0), got rc=$rc [$serr]"; return
    fi
    if grep -Eq "$DIAG_RE" <<<"$serr"; then
        bad "unmarked $signame: diagnostic leaked on a non-driver thread: [$serr]"; return
    fi
    if ! grep -q "RECOVERED prev_ran=1" <<<"$sout"; then
        bad "unmarked $signame: prev handler not chained / no recovery: [$sout]"; return
    fi
    ok "unmarked $signame: no diagnostic + chained to prev (recovered)"
}

# marked + recovering prev: diagnostic AND chain runs (recovers).
test_marked_recover() {
    local signame="$1"
    local out err; out="$(mktemp)"; err="$(mktemp)"
    set +e
    "$bin" marked_recover "$signame" >"$out" 2>"$err"
    local rc=$?
    set -e
    local sout serr; sout="$(cat "$out")"; serr="$(cat "$err")"; rm -f "$out" "$err"

    if [[ $rc -ne 0 ]]; then
        bad "marked_recover $signame: expected recovery (rc=0), got rc=$rc [$serr]"; return
    fi
    if ! grep -Eq "$DIAG_RE $signame" <<<"$serr"; then
        bad "marked_recover $signame: diagnostic missing: [$serr]"; return
    fi
    if ! grep -q "RECOVERED-MARKED prev_ran=1" <<<"$sout"; then
        bad "marked_recover $signame: prev handler not chained: [$sout]"; return
    fi
    ok "marked_recover $signame: diagnostic emitted AND chained to prev"
}

# worker: an unmarked compute-worker thread under Apple/Mach semantics
# (wasm traps cannot reach the handler) -> diagnostic still emitted +
# terminated by the signal. This is the inference-time ggml-worker fault
# F2 is about. Runs in a subshell so the FORCE override is local.
test_worker() {
    local signame="$1"
    local signum; signum="$("$bin" signum "$signame")"
    local err; err="$(mktemp)"
    set +e
    ( PIE_BACKSTOP_FORCE_WASM_TRAPS=0 "$bin" worker "$signame" ) 2>"$err"
    local rc=$?
    set -e
    local body; body="$(cat "$err")"; rm -f "$err"

    if [[ $rc -le 128 ]]; then
        bad "worker $signame: expected death-by-signal (rc>128), got rc=$rc"; return
    fi
    if [[ $((rc - 128)) -ne $signum ]]; then
        bad "worker $signame: died by $((rc - 128)), expected $signum"; return
    fi
    if ! grep -Eq "$DIAG_RE $signame \(model=/test/model.gguf\)" <<<"$body"; then
        bad "worker $signame: diagnostic missing on unmarked worker (Mach): [$body]"; return
    fi
    ok "worker $signame: unmarked worker fault surfaced + died by $signame (rc=$rc)"
}

for s in SIGSEGV SIGBUS SIGILL SIGFPE SIGABRT; do
    test_marked "$s"
done
# Chain/gate behavior is signal-independent; SIGSEGV/SIGBUS are the wasm
# guard-page signals that matter most on Linux.
for s in SIGSEGV SIGBUS; do
    test_unmarked "$s"
    test_marked_recover "$s"
done
# Inference-time worker-thread coverage on Mach-port platforms (#691 F2).
for s in SIGSEGV SIGBUS; do
    test_worker "$s"
done

echo
echo "== fatal_backstop: $pass passed, $fail failed =="
[[ $fail -eq 0 ]]
