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

python3 scripts/generate-plex-replication-report.py
python3 scripts/update-plex-paper-replication-status.py
git --no-pager diff --exit-code -- \
    plex_replication_report.md \
    tests/policies/replication-report.json \
    plex-serving-policy-wiki/catalog.json \
    plex-serving-policy-wiki/papers

if [[ -n "${PLEX_PYTHON:-}" ]]; then
    PLEX_TEST_POLICY="$root/tests/policies/target/packages/plex_coordinated.plexpkg" \
        "$PLEX_PYTHON" -m pytest -q sdk/python-plex/tests
else
    echo "PLEX_PYTHON is unset; Python tests require an environment with pie-plex and pytest" >&2
fi
