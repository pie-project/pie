#!/usr/bin/env bash
# Builds server/ (the Rust core) for every Apple slice and bundles them, with
# include/pie_server.h as the PieServerCore module, into
# build/PieServerCore.xcframework, which Package.swift links.
#
#   ./build-xcframework.sh [release|dev]
set -euo pipefail
cd "$(dirname "$0")"
root="$(cd ../.. && pwd)"

profile="${1:-release}"
case "$profile" in
  release) flag="--release"; dir=release ;;
  dev) flag=""; dir=debug ;;
  *) echo "usage: $0 [release|dev]" >&2; exit 2 ;;
esac

# Package.swift's platforms, for the C dependencies too.
export IPHONEOS_DEPLOYMENT_TARGET=26.0 IPHONESIMULATOR_DEPLOYMENT_TARGET=26.0 MACOSX_DEPLOYMENT_TARGET=26.0

slices=(aarch64-apple-ios aarch64-apple-ios-sim aarch64-apple-darwin)
for triple in "${slices[@]}"; do
  echo "== $triple ($profile)"
  rustup target add "$triple" >/dev/null
  (cd "$root" && CARGO_TARGET_DIR="$root/target-apple" cargo build --target "$triple" -p pie-server-swift $flag)
done

headers="build/headers"
rm -rf "$headers" build/PieServerCore.xcframework
mkdir -p "$headers"
cp server/include/pie_server.h "$headers/"
cat > "$headers/module.modulemap" <<'MAP'
module PieServerCore {
    header "pie_server.h"
    link "pie_server"
    export *
}
MAP

args=()
for triple in "${slices[@]}"; do
  args+=(-library "$root/target-apple/$triple/$dir/libpie_server.a" -headers "$headers")
done
xcodebuild -create-xcframework "${args[@]}" -output build/PieServerCore.xcframework >/dev/null
du -sh build/PieServerCore.xcframework/*/libpie_server.a
echo "== build/PieServerCore.xcframework"
