#!/bin/bash
# Mutation test for tests/memory_planner_parity.rs.
#
# A golden hash proves the transcript did not move. It does NOT prove the
# transcript would have moved had the port been wrong -- a sweep that happens
# to miss every branch it exercises passes just as cleanly. So: break the port
# in ways a careless transcription plausibly would, and require each break to
# be caught.
#
# The mutations are applied to the SOURCE, not to the transcript, so what is
# measured is whether the sweep reaches the code -- which is the actual
# question. A no-op control is included and MUST be missed; if it is caught,
# the harness is reporting noise and none of the other results mean anything.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE="$(cd "$HERE/../../.." && pwd)"
SRC="$CRATE/src/store/memory_planner.rs"
BACKUP="$(mktemp)"
cp "$SRC" "$BACKUP"
trap 'cp "$BACKUP" "$SRC"; rm -f "$BACKUP"' EXIT

FEATURES="${MP_FEATURES:-cuda-13}"
pass=0
fail=0

# $1 = label, $2 = expectation (catch|miss), $3 = from, $4 = to
mutate() {
  local label="$1" expect="$2" from="$3" to="$4"
  cp "$BACKUP" "$SRC"
  python3 - "$SRC" "$from" "$to" <<'PY'
import sys
path, frm, to = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(path, encoding="utf-8").read()
if s.count(frm) != 1:
    sys.exit(f"MUTATION NOT APPLICABLE: {frm!r} occurs {s.count(frm)} times")
open(path, "w", encoding="utf-8").write(s.replace(frm, to))
PY
  if [[ $? -ne 0 ]]; then
    echo "  SKIP  $label (pattern did not match uniquely)"
    fail=$((fail + 1))
    return
  fi
  local out
  out="$(cd "$CRATE" && cargo test --features "$FEATURES" \
        --test memory_planner_parity 2>&1)"
  local caught="miss"
  if ! grep -q "test result: ok" <<<"$out"; then
    caught="catch"
  fi
  if [[ "$caught" == "$expect" ]]; then
    echo "  ok    $label ($caught)"
    pass=$((pass + 1))
  else
    echo "  FAIL  $label (expected $expect, got $caught)"
    fail=$((fail + 1))
  fi
}

echo "mutation testing tests/memory_planner_parity.rs (features=$FEATURES)"

# --- The budget arithmetic -------------------------------------------------
mutate "safety reserve floor 512 MiB -> 256 MiB" catch \
  'let graph_runtime_reserve = (512 * 1024 * 1024).max(one_percent);' \
  'let graph_runtime_reserve = (256 * 1024 * 1024).max(one_percent);'

mutate "safety reserve cap min -> max" catch \
  '(1024 * 1024 * 1024).min(graph_runtime_reserve)' \
  '(1024 * 1024 * 1024).max(graph_runtime_reserve)'

mutate "budget guard <= -> <" catch \
  'if usable <= current_used + safety {' \
  'if usable < current_used + safety {'

# --- The lattice -----------------------------------------------------------
mutate "R0 > N skip dropped" catch \
  '                    if r0 > n {
                        continue;
                    }' \
  '                    if false {
                        continue;
                    }'

mutate "capacity ladder extra removed" catch \
  '    if profile == "capacity" {
        ns.push(1.max(prefill_target / 4));
    }' \
  '    if profile == "capacity" {
    }'

mutate "calibration token sweep upper bound 131072 -> 65536" catch \
  '        while n <= 131_072 {' \
  '        while n <= 65_536 {'

mutate "request ladder loses its 32 rung" catch \
  '        128,
        64,
        32,
    ];' \
  '        128,
        64,
    ];'

mutate "narrow_latency_auto threshold 100 -> 64" catch \
  'let narrow_latency_auto = auto_profile && prop.sm_count < 100 && hf.hidden_size <= 2048;' \
  'let narrow_latency_auto = auto_profile && prop.sm_count < 64 && hf.hidden_size <= 2048;'

mutate "score_as_auto stops tracking the narrow collapse" catch \
  'let score_as_auto = auto_profile && !narrow_latency_auto;' \
  'let score_as_auto = auto_profile;'

# --- Feasibility -----------------------------------------------------------
mutate "arena alignment 2 MiB -> 1 MiB" catch \
  'arena = policy::align_up(arena, 2 * 1024 * 1024);' \
  'arena = policy::align_up(arena, 1024 * 1024);'

mutate "attention int section 8 MiB dropped" catch \
  '                    arena += 8 * 1024 * 1024;' \
  '                    arena += 0;'

mutate "kv_pages sized from the remainder, not the budget" catch \
  'let kv_pages = i32::try_from((budget / per_page_bytes).min(' \
  'let kv_pages = i32::try_from((remaining / per_page_bytes).min('

mutate "min_kv_tokens floor no longer clamped to the budget" catch \
  'let min_kv_tokens = MIN_KV_TOKENS_FLOOR.min(kv_tokens).max(horizon_floor);' \
  'let min_kv_tokens = MIN_KV_TOKENS_FLOOR.max(horizon_floor);'

mutate "minimum wave check dropped" catch \
  '                    if state_bytes > remaining
                        || minimum_wave_kv_bytes > remaining - state_bytes
                    {' \
  '                    if state_bytes > remaining {'

# --- The score -------------------------------------------------------------
mutate "auto cohort weight 6.0 -> 5.0" catch \
  '        cohort_score * 6.0' \
  '        cohort_score * 5.0'

mutate "auto prefill weight loses its tp>1 case" catch \
  '            if tp_size > 1 { 4.0 } else { 3.0 }' \
  '            3.0'

mutate "capacity kv weight 9.0 -> 8.0" catch \
  '                self.kv_score * 9.0' \
  '                self.kv_score * 8.0'

mutate "latency drops its R/N term" catch \
  '                    - f64::from(self.r) / f64::from(1.max(self.n))' \
  '                    - 0.0'

mutate "page_score tp=1 auto preference flips" catch \
  '            return if kv_page_size == 16 { 0.20 } else { -0.05 };
        }
        let latency_shaped' \
  '            return if kv_page_size == 32 { 0.20 } else { -0.05 };
        }
        let latency_shaped'

mutate "qwen3.5-moe TP2 knee 2048 -> 4096" catch \
  '                        score += if n >= 2048 { 1.5 } else { -1.5 };' \
  '                        score += if n >= 4096 { 1.5 } else { -1.5 };'

mutate "nemotron TP2 adjustment removed" catch \
  '                    if prefs.nemotron_h_tp2_ada {
                        score += if n >= 8192 { 1.5 } else { -1.5 };' \
  '                    if prefs.nemotron_h_tp2_ada && false {
                        score += if n >= 8192 { 1.5 } else { -1.5 };'

# --- Selection -------------------------------------------------------------
mutate "profile cache no longer disabled while calibrating" catch \
  'let use_profile_cache = auto_profile && FORCED_PREFILL == 0 && !cfg.calibrating;' \
  'let use_profile_cache = auto_profile && FORCED_PREFILL == 0;'

mutate "budget drift tolerance 0.05 -> 0.50" catch \
  'pub const BUDGET_TOLERANCE: f64 = 0.05;' \
  'pub const BUDGET_TOLERANCE: f64 = 0.50;'

mutate "drift comparison > -> >=" catch \
  '            if drift > BUDGET_TOLERANCE {' \
  '            if drift >= BUDGET_TOLERANCE {'

mutate "calibration picks max score instead of max area" catch \
  '        let pick = max_by_key_first(&candidates, area);' \
  '        let pick = max_by_score(&candidates);'

mutate "max_element tie rule < -> <=" catch \
  '        if candidates[best].score < candidates[i].score {' \
  '        if candidates[best].score <= candidates[i].score {'

mutate "cache pins ignore the page size" catch \
  '                if m.kv_page_size > 0 && c.plan.kv_page_size != m.kv_page_size {
                    continue;
                }' \
  '                if false {
                    continue;
                }'

mutate "profile cache selects the FIRST match rather than the best" catch \
  '                if best.is_none_or(|b| candidates[b].score < c.score) {' \
  '                if best.is_none() {'

mutate "pins honoured during calibration too" catch \
  '        if !cfg.calibrating && cfg.max_forward_tokens > 0 {' \
  '        if cfg.max_forward_tokens > 0 {'

# --- The control -----------------------------------------------------------
# A rename changes nothing observable. If this is CAUGHT the harness is
# reporting noise and every result above is meaningless.
mutate "CONTROL: rename a local (no-op)" miss \
  '    let current_used = mem.total_bytes - mem.free_bytes;
    let safety = reserves(mem.total_bytes);' \
  '    let resident = mem.total_bytes - mem.free_bytes;
    let current_used = resident;
    let safety = reserves(mem.total_bytes);'

echo
echo "mutations: $pass as expected, $fail unexpected"
[[ $fail -eq 0 ]]
