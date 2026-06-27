#!/usr/bin/env bash
# One-time (idempotent) setup for building `wasm32-wasip3` inferlet components.
#
# `wasm32-wasip3` is a Tier-3 target with two gaps that this script closes,
# reproducibly, with no per-machine sysroot edits:
#
#   1. No prebuilt std -> built from source via `-Zbuild-std` (configured in
#      the inferlet `.cargo/config.toml`). Needs the `rust-src` component, which
#      `rust-toolchain.toml` already pins; this script is a no-op for that.
#   2. No bundled wasi-libc -> the link step needs a `-L` to a wasi-libc. We:
#        a. install the `wasm32-wasip2` std as a libc DONOR (its self-contained
#           libc.a is ABI-compatible for linking wasip3 cdylibs), and
#        b. install the linker wrapper (scripts/wasip3-link.sh) onto PATH so
#           `.cargo/config.toml`'s `linker = "wasip3-link.sh"` resolves
#           everywhere.
#
# Optional: export `WASI_SDK_PATH=<pinned wasi-sdk>` to link against the
# wasi-sdk wasi-sysroot libc instead of the rustup donor (the wrapper prefers
# it when set).
#
# Run once after cloning / after a toolchain change:  ./scripts/setup-wasip3.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. wasi-libc donor (idempotent; targets the pinned toolchain via cwd's
#    rust-toolchain.toml).
echo "==> installing wasm32-wasip2 (wasi-libc donor)"
rustup target add wasm32-wasip2

# 2. linker wrapper onto PATH.
cargo_bin="${CARGO_HOME:-$HOME/.cargo}/bin"
mkdir -p "$cargo_bin"
install -m 0755 "$script_dir/wasip3-link.sh" "$cargo_bin/wasip3-link.sh"
echo "==> installed $cargo_bin/wasip3-link.sh"

echo "wasip3 setup complete."
