#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

cargo test --locked -p pie-plex -p pie-policy -p pie-plex-py --all-targets
cargo test --manifest-path sdk/rust/plex/Cargo.toml --locked
wasm-tools component wit interface/plex/wit-0.6 >/dev/null
scripts/check-plex-layering.sh
scripts/build-plex-policies.sh
cargo run --quiet --release --locked -p pie-policy --example bench_v0_6 -- \
    --check tests/policies/performance-budgets.json >/tmp/plex-v0.6-benchmark.json
python3 -m json.tool /tmp/plex-v0.6-benchmark.json >/dev/null
python3 -m json.tool tests/policies/performance-targets.json >/dev/null
python3 -m json.tool tests/policies/fidelity-audit.json >/dev/null
python3 -m json.tool tests/policies/reproducibility-gap-taxonomy.json >/dev/null
python3 -m json.tool tests/policies/reproducibility-gap-matrix.json >/dev/null
python3 -m json.tool plex_policy_performance_report.json >/dev/null
python3 -m json.tool plex_policy_reproducibility_roadmap.json >/dev/null
python3 -m py_compile \
    scripts/benchmark-plex-policy-performance.py \
    scripts/benchmark-vllm-plex-policy.py \
    scripts/run-vllm-plex-performance-matrix.py \
    scripts/generate-plex-policy-performance-report.py \
    scripts/generate-plex-current-model-presentation.py \
    scripts/merge-plex-reproducibility-gaps.py \
    scripts/generate-plex-reproducibility-roadmap.py

python3 scripts/generate-plex-replication-report.py
python3 scripts/update-plex-paper-replication-status.py
python3 scripts/generate-plex-reproducibility-roadmap.py \
    --taxonomy tests/policies/reproducibility-gap-taxonomy.json \
    --matrix tests/policies/reproducibility-gap-matrix.json \
    --output-json plex_policy_reproducibility_roadmap.json \
    --output-md plex_policy_reproducibility_roadmap.md
python3 scripts/generate-plex-current-model-presentation.py >/dev/null
python3 - <<'PY'
import json

with open("tests/policies/replication-report.json") as handle:
    replication = json.load(handle)
with open("plex_policy_reproducibility_roadmap.json") as handle:
    roadmap = json.load(handle)
if replication["evidence_counts"].get("policy-kernel-reproduction") != 13:
    raise SystemExit("expected 13 policy-kernel reproductions")
if replication["evidence_counts"].get(
    "decision-trace-parity-with-deferred-mechanics"
) != 4:
    raise SystemExit("expected 4 decision-trace parity reproductions")
if roadmap.get("current_model_closed_count") != 17:
    raise SystemExit("expected all 17 current-model policies to be closed")
PY
git --no-pager diff --exit-code -- \
    plex_replication_report.md \
    tests/policies/replication-report.json \
    plex-serving-policy-wiki/catalog.json \
    plex-serving-policy-wiki/papers \
    plex_policy_reproducibility_roadmap.json \
    plex_policy_reproducibility_roadmap.md \
    plex_current_model_presentation.html

if [[ -n "${PLEX_PYTHON:-}" ]]; then
    PLEX_TEST_POLICY="$root/tests/policies/target/packages/plex_coordinated.plexpkg" \
        "$PLEX_PYTHON" -m pytest -q sdk/python-plex/tests
    "$PLEX_PYTHON" scripts/benchmark-plex-policy-performance.py \
        --trials 4 \
        --output /tmp/plex-policy-performance-smoke.json >/dev/null
    "$PLEX_PYTHON" - <<'PY'
import json

with open("/tmp/plex-policy-performance-smoke.json") as handle:
    report = json.load(handle)
if report["policy_count"] != 31:
    raise SystemExit("performance smoke did not cover all 31 policies")
if report["trend_reproduced_count"] != 31:
    raise SystemExit("performance smoke did not reproduce all 31 kernel trends")
PY
else
    echo "PLEX_PYTHON is unset; Python tests require an environment with pie-plex and pytest" >&2
fi
