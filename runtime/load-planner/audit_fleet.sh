#!/usr/bin/env bash
set -euo pipefail

backend="${1:?usage: audit_fleet.sh BACKEND SNAPSHOT...}"
shift
if [[ "$#" -eq 0 ]]; then
  echo "audit_fleet.sh: at least one snapshot is required" >&2
  exit 2
fi

for snapshot in "$@"; do
  echo "planning ${backend}: ${snapshot}" >&2
  cargo run -q -p pie-load-planner --example plan_dump -- \
    "${snapshot}" "${backend}" >/dev/null
done
