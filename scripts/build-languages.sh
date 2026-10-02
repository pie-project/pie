#!/usr/bin/env bash
# Build the language support a pie installation needs for script inferlets:
# one language component per language, each its own release asset laid
# out like $PIE_HOME so it extracts straight into it.
#
#   pie-language-python.tar.gz          languages/python.wasm
#   pie-language-javascript.tar.gz      languages/javascript.wasm
#
# and the same wasm into the packages that ship it: javascript/language-*
# (npm) and python/language-*/src/pie_language_* (PyPI).
#
#   scripts/build-languages.sh [OUT_DIR] [LANGUAGE...]
#     OUT_DIR     where the archives go        (default: target/languages)
#     LANGUAGE    python | javascript          (default: both)
#
# Environment:
#   COMPONENTIZE_PY   passed through to the Python language component build

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/target/languages}"
case "$out" in /*) ;; *) out="$PWD/$out" ;; esac
shift || true
languages=("$@")
[ "${#languages[@]}" -gt 0 ] || languages=(python javascript)

mkdir -p "$out"
for language in "${languages[@]}"; do
  case "$language" in
    python|javascript) ;;
    *) echo "unknown language $language (python | javascript)" >&2; exit 1 ;;
  esac
  stage="$out/stage-$language"
  rm -rf "$stage"
  mkdir -p "$stage/languages"
  echo "== $language"
  "$root/$language/inferlet/language/build.sh" "$stage/languages/$language.wasm"
  tar -C "$stage" -czf "$out/pie-language-$language.tar.gz" languages
  cp "$stage/languages/$language.wasm" "$root/javascript/language-$language/$language.wasm"
  cp "$stage/languages/$language.wasm" "$root/python/language-$language/src/pie_language_$language/$language.wasm"
  rm -rf "$stage"
  echo "== $out/pie-language-$language.tar.gz ($(du -h "$out/pie-language-$language.tar.gz" | cut -f1))"
done
