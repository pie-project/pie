#!/usr/bin/env bash
# Linker wrapper for the Tier-3 `wasm32-wasip3` target.
#
# `wasm32-wasip3` ships no self-contained wasi-libc, so a plain link fails with
# `unable to find library -lc`. This wrapper injects a `-L` to a wasi-libc
# search directory and then execs the toolchain's real `wasm-component-ld`.
#
# libc source (in priority order):
#   1. `$WASI_SDK_PATH` — a pinned wasi-sdk (its `share/wasi-sysroot/lib/...`),
#      preferred when you want a wasi-0.3-matched libc.
#   2. otherwise the rustup `wasm32-wasip2` self-contained libc (a "donor"
#      installed by scripts/setup-wasip3.sh) — ABI-compatible for linking.
#
# Referenced from .cargo/config.toml as `linker = "wasip3-link.sh"`; installed
# onto PATH (~/.cargo/bin) by scripts/setup-wasip3.sh so it resolves for the
# whole crew + CI without per-machine sysroot edits.
set -euo pipefail

sysroot="$(rustc --print sysroot)"
host="$(rustc -vV | sed -n 's/^host: //p')"
lld="$sysroot/lib/rustlib/$host/bin/wasm-component-ld"

# macOS SIP strips DYLD_* when exec'ing this (bash) wrapper, losing the
# fallback path rustc set for rust-lld's @rpath/libLLVM.dylib. Re-export it
# so the real linker (a non-restricted binary) resolves libLLVM again.
export DYLD_FALLBACK_LIBRARY_PATH="$sysroot/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"

if [ -n "${WASI_SDK_PATH:-}" ] && [ -d "$WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasip2" ]; then
  libc_dir="$WASI_SDK_PATH/share/wasi-sysroot/lib/wasm32-wasip2"
else
  libc_dir="$sysroot/lib/rustlib/wasm32-wasip2/lib/self-contained"
fi

if [ ! -d "$libc_dir" ]; then
  echo "wasip3-link.sh: wasi-libc not found at $libc_dir" >&2
  echo "  run scripts/setup-wasip3.sh (installs the wasm32-wasip2 libc donor)" >&2
  exit 1
fi

exec "$lld" "$@" -L "$libc_dir"
