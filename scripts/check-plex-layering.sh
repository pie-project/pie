#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

forbidden='pie-(engine|gateway|worker)|runtime/engine|gateway/|worker/'

if grep -ERn "$forbidden" interface/plex runtime/policy/Cargo.toml; then
    echo "PLEX contract or policy host depends on Pie mechanics" >&2
    exit 1
fi

if cargo tree --locked -p pie-policy --depth 1 -e normal --prefix none \
    | grep -Eq '^pie-(engine|gateway|worker)( |$)'; then
    echo "pie-policy dependency graph reaches Pie mechanics" >&2
    exit 1
fi

if grep -ERn 'crate::(scheduler|store|inferlet)|pie_engine|pie_gateway|pie_worker' \
    runtime/policy/src; then
    echo "pie-policy source imports a forbidden runtime layer" >&2
    exit 1
fi
