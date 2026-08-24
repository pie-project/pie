#!/usr/bin/env bash
#
# TYPE-CHECK AND LINT `driver-metal`'s APPLE HALF ON A MACHINE WITH NO APPLE IN IT.
#
# `driver-metal` splits behind a Cargo feature: without `metal-4` the crate is
# `baker/`, `layout/`, `model/` and `envelope/` -- the half that is a function
# of a plan's numbers -- and with it, six more modules that name Metal types.
# `src/lib.rs` turns the feature into a `compile_error!` off an Apple target,
# so `cargo check -p driver-metal --all-targets` on a Linux box compiles the
# portable half and CANNOT SEE the other one.
#
# That is not a theoretical gap. It hid a four-argument `model::produce`
# called with two for the whole of Q6 through Q11, plus twenty-nine more
# errors across `serve/`, `bind/` and `fire/`, plus nine device test targets
# still naming `driver_metal::lowering`, `model_compiler::lower` and
# `model::shared::llama_like` -- every one of them deleted at P5, in the same
# commit that wrote the gated half. The crate's own manifest argues for the
# feature over a `cfg` on the grounds that "a platform cfg cannot be tested. A
# feature can" -- true, and nothing was testing it.
#
# ── WHY THIS WORKS WITHOUT A MAC ────────────────────────────────────────────
#
# `cargo check` type-checks; it does not link. So the only thing it needs for
# `aarch64-apple-darwin` is that target's `rust-std`, which rustup ships
# prebuilt for every host -- no Xcode, no SDK, no code signing, no device.
# `objc2`, `objc2-metal` and `block2` are pure Rust bindings and compile for
# the target like any other crate; build scripts still run on the HOST, which
# is why none of this needs an Apple machine to execute anything.
#
# The target triple is what flips `target_vendor = "apple"`, which is what
# `lib.rs`'s `compile_error!` reads -- so this is the same compilation the
# people who work on this crate get, on the same source, with the same
# feature set.
#
# ── WHAT IT CANNOT CATCH ────────────────────────────────────────────────────
#
# Everything past type-checking, and the list is worth having in full:
#
#   * LINKING. No Apple SDK is present, so an undefined symbol, a missing
#     framework, or a `#[link]` naming something that is not there passes
#     here and fails on a Mac.
#   * CODEGEN AND MONOMORPHISATION ERRORS. Clippy stops after type checking,
#     so a post-mono error -- an over-large `const` evaluation, an `asm!` the
#     target rejects -- is not reached.
#   * BEHAVIOUR. Nothing runs. Every `device_*` test target is COMPILED here
#     and executed nowhere: whether Metal accepts a shader, whether an ICB
#     replays, whether a page lands where the pool says, are all questions
#     only a device answers. Those tests exist and are `required-features =
#     ["metal-4"]`; a runner with a GPU is what turns them from compiled into
#     measured, and this script deliberately claims nothing about them.
#   * ANYTHING BEHIND A `cfg` THIS TARGET DOES NOT SELECT -- an x86 Mac, an
#     iOS target, a different macOS version. One triple is one compilation.
#
# What it DOES catch is the entire class that actually broke this crate: a
# call, a field, a variant or an import in the gated half that stopped
# matching its definition. That class is caught in about a minute, on the
# machine the change is being written on, which is the argument for a script
# a developer can run over a CI job they cannot.

set -euo pipefail

TARGET=aarch64-apple-darwin

if ! rustc --print target-libdir --target "$TARGET" >/dev/null 2>&1; then
    echo "error: no \`rust-std\` for $TARGET." >&2
    echo "       rustup target add $TARGET" >&2
    exit 1
fi

# CLIPPY AND NOT `check`, because clippy type-checks everything `check` does
# and then lints, and the crate sets `missing_docs` and four clippy denials in
# `lib.rs` that no job was reading for the gated half. `-D warnings` is the
# spelling every other gate in this tree uses.
#
# `--all-targets` is load-bearing and not thoroughness: nine of this crate's
# thirty-three `metal-4` test targets had drifted out of compiling while the
# library still built, and a check of the library alone would report them
# green forever. It is also what compiles the `#[cfg(test)]` modules inside
# `src/`, which a plain `cargo check` does not.
exec cargo clippy --no-deps -p driver-metal \
    --features metal-4 \
    --target "$TARGET" \
    --all-targets \
    "$@" -- -D warnings
