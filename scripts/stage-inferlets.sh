#!/usr/bin/env bash
# Build examples to wasm and stage them as <name>/<version>.{wasm,toml},
# the layout `pie inferlet install` takes. Script inferlets (`main.py`,
# `index.js`) are staged as their source: <name>/<version>.{py,js}.
#
#   scripts/stage-inferlets.sh [OUT_DIR]     (default: target/inferlet-publish)

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/examples"
out="${1:-$root/target/inferlet-publish}"

manifest_version() { sed -n 's/^version *= *"\(.*\)".*/\1/p' "$1" | head -1; }

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
  version="$(manifest_version "$manifest")"
  if [ -z "$version" ]; then
    echo "no version in $manifest, skipping" >&2
    continue
  fi
  mkdir -p "$out/$dir"
  cp "$wasm" "$out/$dir/$version.wasm"
  cp "$manifest" "$out/$dir/$version.toml"
  staged=$((staged + 1))
done

# Script inferlets: the source is the artifact.
for manifest in "$src"/*/Pie.toml; do
  dir="$(basename "$(dirname "$manifest")")"
  for script in main.py index.js; do
    source="$src/$dir/$script"
    [ -f "$source" ] || continue
    version="$(manifest_version "$manifest")"
    [ -n "$version" ] || continue
    mkdir -p "$out/$dir"
    cp "$source" "$out/$dir/$version.${script##*.}"
    cp "$manifest" "$out/$dir/$version.toml"
    staged=$((staged + 1))
    break
  done
done

{
  echo "# name<TAB>version<TAB>bytes<TAB>sha256"
  for artifact in "$out"/*/*.wasm "$out"/*/*.py "$out"/*/*.js; do
    [ -e "$artifact" ] || continue
    dir="$(basename "$(dirname "$artifact")")"
    version="$(basename "${artifact%.*}")"
    size="$(wc -c < "$artifact" | tr -d ' ')"
    sum="$(shasum -a 256 "$artifact" | cut -d' ' -f1)"
    printf '%s\t%s\t%s\t%s\n' "$dir" "$version" "$size" "$sum"
  done
} > "$out/INDEX.tsv"

echo "staged $staged inferlets into $out"
