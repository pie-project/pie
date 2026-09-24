#!/usr/bin/env bash
# Build tests/inferlets to wasm and stage them as <name>/<version>.{wasm,toml},
# the layout tools/publish_inferlets.py in pie-project/registry consumes.
#
#   scripts/stage-inferlets.sh [OUT_DIR]     (default: target/inferlet-publish)

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/tests/inferlets"
out="${1:-$root/target/inferlet-publish}"

cargo build --manifest-path "$src/Cargo.toml" --workspace \
  --target wasm32-wasip2 --release

rm -rf "$out"
mkdir -p "$out"

staged=0
for wasm in "$src/target/wasm32-wasip2/release"/*.wasm; do
  [ -e "$wasm" ] || continue
  crate="$(basename "$wasm" .wasm)"
  dir="${crate//_/-}"
  manifest="$src/$dir/Pie.toml"
  if [ ! -f "$manifest" ]; then
    echo "no Pie.toml for $dir, skipping" >&2
    continue
  fi
  version="$(sed -n 's/^version *= *"\(.*\)".*/\1/p' "$manifest" | head -1)"
  if [ -z "$version" ]; then
    echo "no version in $manifest, skipping" >&2
    continue
  fi
  mkdir -p "$out/$dir"
  cp "$wasm" "$out/$dir/$version.wasm"
  cp "$manifest" "$out/$dir/$version.toml"
  staged=$((staged + 1))
done

{
  echo "# name<TAB>version<TAB>bytes<TAB>sha256"
  for wasm in "$out"/*/*.wasm; do
    [ -e "$wasm" ] || continue
    dir="$(basename "$(dirname "$wasm")")"
    version="$(basename "$wasm" .wasm)"
    size="$(wc -c < "$wasm" | tr -d ' ')"
    sum="$(shasum -a 256 "$wasm" | cut -d' ' -f1)"
    printf '%s\t%s\t%s\t%s\n' "$dir" "$version" "$size" "$sum"
  done
} > "$out/INDEX.tsv"

echo "staged $staged inferlets into $out"
