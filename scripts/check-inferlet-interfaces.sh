#!/usr/bin/env bash
#
# Gate the Python and JavaScript inferlet libraries against the WIT world they claim
# to target. Two questions, in order:
#
#   1. Does either library reference an interface `crates/inferlet/wit/` no longer
#      defines?  (drift downward: dead code that still looks alive)
#   2. Does either library reach the forward-pass surface at all?
#      (drift upward: a live-looking package that cannot run a model)
#
# Why this exists: when `pie:core/inference` was replaced by `forward` /
# `forward-recurrent` / `forward-hybrid` / `forward-diffusion`, both libraries kept
# importing the interface that had been deleted, and nothing said so.
# `release-pypi.yml`'s own smoke test comments that it CANNOT verify
# `import inferlet`, because `wit_world` only exists inside a componentize-py
# build. ci.yml's `packages` job and both release workflows run this.
#
# Check 1 alone is not enough, and getting it to pass is the reason check 2
# exists. Deleting the modules built on the removed interface makes check 1
# green while leaving both libraries unable to run a forward pass -- a green gate
# over an unpublishable package. Check 2 is what keeps that honest: it fails
# until the hand-written layer actually binds the forward-pass interfaces.
# Both libraries bind them through their `eta` module (the tracing eDSL + container
# encoder ported from `eta-dsl`/`eta-ir`, pinned byte-for-byte to the Rust
# encoder by `crates/eta-dsl/tests/inferlet_goldens.rs`), so this gate is green.
#
# Neither check needs componentize-py, jco, or a toolchain -- just the names.
# A full build is the real gate; this is the cheap one that would have caught
# the actual drift.
#
# Usage:
#   scripts/check-inferlet-interfaces.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/crates/inferlet/wit"

# Interfaces the world actually offers: one file per interface, plus whatever
# `world.wit` imports from the `pie:inferlet` package itself.
known=$(
  find "$SRC" -maxdepth 1 -name '*.wit' ! -name 'world.wit' -exec basename {} .wit \; \
    | sort -u
)

# The forward-pass surface. A library that binds none of these can do everything
# except the one thing an inferlet exists to do, so publishing it is worse than
# publishing nothing: it looks like a working library. `forward` / `-recurrent` /
# `-hybrid` are alternatives -- a model needs exactly one -- so any ONE of them
# satisfies the requirement; the supporting resources are all mandatory,
# because a pass cannot be built without a channel to bind, a working set to
# bind it against, or a pipeline to submit on.
forward_any="forward forward-recurrent forward-hybrid forward-diffusion"
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
# editing it, and its presence says nothing about what the inferlet library actually binds.
py_dotted=$(
  grep -rhoE 'wit_world\.imports\.[a-z0-9_]+' \
    "$ROOT/python/inferlet/src/inferlet" \
    --exclude-dir=bindings --exclude-dir=__pycache__ --exclude='*.pyc' 2>/dev/null \
    | sed -E 's/.*\.//' || true
)
py_listed=$(
  grep -rhoE 'wit_world\.imports import [a-z0-9_, ]+' \
    "$ROOT/python/inferlet/src/inferlet" \
    --exclude-dir=bindings --exclude-dir=__pycache__ --exclude='*.pyc' 2>/dev/null \
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
  grep -rhoE "['\"]${pkg_ns}/[a-z0-9-]+" "$ROOT/javascript/inferlet/src" \
    --include='*.ts' --exclude-dir=bindings --exclude-dir=__pycache__ --exclude='*.pyc' 2>/dev/null \
    | sed -E "s|.*${pkg_ns}/||" | sort -u || true
)

# Anything spelled `pie:<something-else>/...` is a reference to a package that
# does not exist -- the WIT namespace was consolidated into one package.
js_foreign=$(
  grep -rhoE "['\"]pie:[a-z0-9-]+/[a-z0-9-]+" "$ROOT/javascript/inferlet/src" \
    --include='*.ts' --exclude-dir=bindings --exclude-dir=__pycache__ --exclude='*.pyc' 2>/dev/null \
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
    # componentize-py names the `types` interface `pie_inferlet_types`, since
    # a bare `types` would shadow the stdlib module.
    [ "$wit" = "pie-inferlet-types" ] && wit="types"
    if ! grep -qx -- "$wit" <<<"$known"; then
      note "$lang references interface '$wit', which crates/inferlet/wit/ does not define"
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
  note "These libraries target a removed surface and cannot work as published."
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
  note "The guest forward-pass surface is no longer a fixed host-side sampler."
  note "The guest traces a program and ships canonical ETA container bytes."
  note "Each library's \`eta\` module (a port of crates/eta-dsl + crates/eta-ir) must"
  note "bind the interfaces above; a library that does not cannot run a model and"
  note "must not be published."
  exit 1
fi

echo "Inferlet library interface references are all defined, and the forward-pass surface is bound."
