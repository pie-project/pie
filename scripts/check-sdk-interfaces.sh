#!/usr/bin/env bash
#
# Gate the Python and JavaScript inferlet SDKs against the WIT world they claim
# to target. Two questions, in order:
#
#   1. Does either SDK reference an interface `interface/inferlet/` no longer
#      defines?  (drift downward: dead code that still looks alive)
#   2. Does either SDK reach the forward-pass surface at all?
#      (drift upward: a live-looking package that cannot run a model)
#
# Why this exists: neither SDK is in `scripts/sync-wit.sh`'s vendored set and
# neither appears in ci.yml -- the only workflows that touch them are the
# manually triggered release publishes. When `pie:core/inference` was replaced
# by `forward` / `forward-recurrent` / `forward-hybrid`, both SDKs kept
# importing the interface that had been deleted, and nothing said so.
# `release-pypi.yml`'s own smoke test comments that it CANNOT verify
# `import inferlet`, because `wit_world` only exists inside a componentize-py
# build.
#
# Check 1 alone is not enough, and getting it to pass is the reason check 2
# exists. Deleting the modules built on the removed interface makes check 1
# green while leaving both SDKs unable to run a forward pass -- a green gate
# over an unpublishable package. Check 2 is what keeps that honest: it fails
# until the hand-written layer actually binds the forward-pass interfaces, and
# it goes green on its own the moment that lands. It is EXPECTED TO FAIL today.
#
# Neither check needs componentize-py, jco, or a toolchain -- just the names.
# A full build is the real gate; this is the cheap one that would have caught
# the actual drift.
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

# The forward-pass surface. An SDK that binds none of these can do everything
# except the one thing an inferlet exists to do, so publishing it is worse than
# publishing nothing: it looks like a working SDK. `forward` / `-recurrent` /
# `-hybrid` are alternatives -- a model needs exactly one -- so any ONE of them
# satisfies the requirement; the supporting resources are all mandatory,
# because a pass cannot be built without a channel to bind, a working set to
# bind it against, or a pipeline to submit on.
forward_any="forward forward-recurrent forward-hybrid"
forward_all="channel working-set pipeline"

fail=0
note() { printf '%s\n' "$*" >&2; }

# Python: `from wit_world.imports import <name>[, <name> as <alias>]...` and
# `from wit_world.imports.<name> import ...`. The comma list matters: a single
# `import a, b, c` is idiomatic, and reading only the first name would let the
# rest go unchecked -- which for check 2 means reporting a surface as unbound
# when it is bound on one line.
#
# The generated `bindings/` tree is itself an artifact of whatever world it was
# last generated from, so it is excluded -- regenerating it is the fix, not
# editing it, and its presence says nothing about what the SDK actually binds.
py_dotted=$(
  grep -rhoE 'wit_world\.imports\.[a-z0-9_]+' \
    "$ROOT/sdk/python/src/inferlet" \
    --exclude-dir=bindings 2>/dev/null \
    | sed -E 's/.*\.//' || true
)
py_listed=$(
  grep -rhoE 'wit_world\.imports import [a-z0-9_, ]+' \
    "$ROOT/sdk/python/src/inferlet" \
    --exclude-dir=bindings 2>/dev/null \
    | sed -E 's/.*imports import //' | tr ',' '\n' \
    | sed -E 's/ +as +.*//; s/^ +//; s/ +$//' | grep -v '^$' || true
)
py_refs=$(printf '%s\n%s\n' "$py_dotted" "$py_listed" | grep -v '^$' | sort -u || true)

# JavaScript: hand-written modules import the WIT specifier directly
# (`import * as _chat from 'pie:inferlet/chat'`), which tsconfig's generated
# `paths` map resolves. Read the specifiers, not the bindings directory -- same
# reason as Python.
pkg_ns=$(sed -nE 's/^package ([a-z0-9]+):([a-z0-9-]+).*/\1:\2/p' "$SRC/world.wit" | head -1)
js_refs=$(
  grep -rhoE "['\"]${pkg_ns}/[a-z0-9-]+" "$ROOT/sdk/javascript/src" \
    --include='*.ts' --exclude-dir=bindings 2>/dev/null \
    | sed -E "s|.*${pkg_ns}/||" | sort -u || true
)

# Anything spelled `pie:<something-else>/...` is a reference to a package that
# does not exist -- the WIT namespace was consolidated into one package.
js_foreign=$(
  grep -rhoE "['\"]pie:[a-z0-9-]+/[a-z0-9-]+" "$ROOT/sdk/javascript/src" \
    --include='*.ts' --exclude-dir=bindings 2>/dev/null \
    | sed -E "s|.*(pie:[a-z0-9-]+/[a-z0-9-]+)|\1|" | grep -v "^$pkg_ns/" | sort -u || true
)
if [ -n "$js_foreign" ]; then
  note "javascript imports packages other than '$pkg_ns':"
  for f in $js_foreign; do note "  $f"; done
  fail=1
fi

# -- Check 1: every referenced interface is defined ---------------------------
check_defined() {
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

check_defined python "$py_refs"
check_defined javascript "$js_refs"

if [ "$fail" -ne 0 ]; then
  note ""
  note "Known interfaces:"
  note "$(tr '\n' ' ' <<<"$known")"
  note ""
  note "These SDKs target a removed surface and cannot work as published."
  note "See forward_refactor.md 10.4."
  exit 1
fi

# -- Check 2: the forward-pass surface is actually bound ----------------------
check_forward() {
  local lang="$1" refs="$2"
  local normalized missing="" got_variant=0

  normalized=$(tr '_' '-' <<<"$refs")

  for want in $forward_any; do
    if grep -qx -- "$want" <<<"$normalized"; then got_variant=1; fi
  done
  if [ "$got_variant" -eq 0 ]; then
    missing="$missing one-of(${forward_any// /|})"
  fi

  for want in $forward_all; do
    grep -qx -- "$want" <<<"$normalized" || missing="$missing $want"
  done

  if [ -n "$missing" ]; then
    note "$lang binds no forward-pass surface; missing:$missing"
    return 1
  fi
  return 0
}

check_forward python "$py_refs" || fail=1
check_forward javascript "$js_refs" || fail=1

if [ "$fail" -ne 0 ]; then
  note ""
  note "This is the KNOWN, EXPECTED state -- not a regression to bisect."
  note ""
  note "The guest forward-pass surface is no longer a fixed host-side sampler."
  note "The guest traces a program and ships canonical PTIR container bytes."
  note "The Rust SDK produces them with compiler/dsl (the tracing eDSL) and"
  note "compiler/ir (the container encoder). Neither has a Python or JavaScript"
  note "counterpart, and any port has to agree with the Rust encoder byte for"
  note "byte, so it is its own project -- see forward_refactor.md 10.4."
  note ""
  note "Until then these SDKs expose the non-forward interfaces only and cannot"
  note "run a model, so they must not be published. This gate goes green by"
  note "itself once the hand-written layer binds the interfaces above."
  exit 1
fi

echo "SDK interface references are all defined, and the forward-pass surface is bound."
