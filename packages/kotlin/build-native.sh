#!/usr/bin/env bash
# Build the Rust core (core/, `pie-kotlin-core`) for Android into
# server/src/main/jniLibs/arm64-v8a, where the AAR picks it up.
#
#   ./build-native.sh
#
# Needs the NDK (ANDROID_NDK_HOME), cargo-ndk (`cargo install cargo-ndk`),
# the aarch64-linux-android Rust target, and slangc (or PIE_SLANGC) for the
# Vulkan kernels.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CARGO_PROFILE_RELEASE_STRIP=symbols cargo ndk -t arm64-v8a -P 29 -o "$here/server/src/main/jniLibs" \
  build --release --manifest-path "$here/core/Cargo.toml"
ls -la "$here/server/src/main/jniLibs/arm64-v8a"
