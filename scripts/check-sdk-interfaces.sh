#!/usr/bin/env bash
#
# Fail if an SDK references a WIT interface that `interface/inferlet/` no
# longer defines.
#
# Why this exists: the Python and JavaScript SDKs are not in the
# `scripts/sync-wit.sh` vendored set, and neither appears in ci.yml — the only
# workflows that touch them are the manually triggered release publishes. When
# `pie:core/inference` was replaced by `forward` / `forward-recurrent` /
# `forward-hybrid`, both SDKs kept importing the interface that had been
# deleted, and nothing said so. `release-pypi.yml`'s own smoke test comments
# that it CANNOT verify `import inferlet`, because `wit_world` only exists
# inside a componentize-py build.
#
# A full build would be the real gate. This is the cheap one that would have
# caught the actual drift: it does not need componentize-py, jco, or a
# toolchain — just the names.
#
# Usage:
#   scripts/check-sdk-interfaces.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/interface/inferlet"

# Interfaces the world actually offers: one file per interface, plus whatever
# `world.wit` imports from the `pie:inferlet` package itself.
known=$(
  find "$SRC" -maxdepth 1 -name '*.wit' ! -name 'world.wit' -exec basename {} .wit \; \
    | sort -u
)

fail=0
note() { printf '%s\n' "$*" >&2; }

# Python: `from wit_world.imports import <name>` / `from wit_world.imports.<name> import`
# The generated `bindings/` tree is itself an artifact of the stale world, so
# it is excluded — regenerating it is the fix, not editing it.
py_refs=$(
  grep -rhoE 'wit_world\.imports(\.[a-z0-9_]+| import [a-z0-9_]+)' \
    "$ROOT/sdk/python/src/inferlet" \
    --exclude-dir=bindings 2>/dev/null \
    | sed -E 's/.*(\.| )([a-z0-9_]+)$/\2/' | sort -u || true
)

# JavaScript: `bindings/interfaces/<ns>-<package>-<interface>.d.ts`. Here the
# PACKAGE is the sharper signal — every pie binding is `pie-core-*`,
# `pie-instruct-*` or `pie-zo-*`, and the only pie package that exists is
# `pie:inferlet`. Interfaces whose names happen to have survived the move
# (model, session, chat, ...) are just as unreachable as `inference`.
pkg=$(sed -nE 's/^package ([a-z0-9]+):([a-z0-9-]+).*/\1-\2/p' "$SRC/world.wit" | head -1)
js_stale=$(
  find "$ROOT/sdk/javascript/src/bindings/interfaces" -name 'pie-*.d.ts' \
    -exec basename {} .d.ts \; 2>/dev/null \
    | grep -v "^$pkg-" | sort -u || true
)
if [ -n "$js_stale" ]; then
  note "javascript bindings target packages other than '${pkg/-/:}':"
  for f in $js_stale; do note "  $f.d.ts"; done
  fail=1
fi

js_refs=$(
  find "$ROOT/sdk/javascript/src/bindings/interfaces" -name "$pkg-*.d.ts" \
    -exec basename {} .d.ts \; 2>/dev/null \
    | sed -E "s/^$pkg-//" | sort -u || true
)

check() {
  local lang="$1" refs="$2"
  for ref in $refs; do
    # WIT interface files are kebab-case; Python identifiers are snake_case.
    local wit="${ref//_/-}"
    if ! grep -qx -- "$wit" <<<"$known"; then
      note "$lang references interface '$wit', which interface/inferlet/ does not define"
      fail=1
    fi
  done
}

check python "$py_refs"
check javascript "$js_refs"

if [ "$fail" -ne 0 ]; then
  note ""
  note "Known interfaces:"
  note "$(tr '\n' ' ' <<<"$known")"
  note ""
  note "These SDKs target a removed surface and cannot work as published."
  note "See forward_refactor.md 10.4."
  exit 1
fi

echo "SDK interface references are all defined by interface/inferlet/."
