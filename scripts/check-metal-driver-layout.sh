#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

fail=0
metal_root="driver/metal"
metal_src="$metal_root/src"

report_matches() {
  local label="$1"
  local matches="$2"
  if [ -n "$matches" ]; then
    echo "::error::${label}"
    echo "$matches"
    fail=1
  fi
}

glob_matches=$(grep -rn 'file(GLOB' "$metal_root" \
  --include='CMakeLists.txt' \
  --exclude-dir='build*' \
  2>/dev/null || true)
report_matches "Metal CMake files must use explicit source lists" "$glob_matches"

unwired_tests=""
while IFS= read -r test_source; do
  name=$(basename "$test_source")
  if ! grep -RqsF "$name" "$metal_root/CMakeLists.txt" "$metal_root/tests" \
      --include='CMakeLists.txt'; then
    unwired_tests+="${test_source}"$'\n'
  fi
done < <(find "$metal_root/tests" -type f \( -name '*.cpp' -o -name '*.mm' \) | sort)
report_matches "Metal test translation units missing from CMake" "${unwired_tests%$'\n'}"

kernel_upward=$(grep -rlnE \
  '#include "(model|batch|pipeline|loader|store)/' \
  "$metal_src/kernels" 2>/dev/null || true)
report_matches "Metal kernels must not include higher layers" "$kernel_upward"

loader_upward=$(grep -rlnE \
  '#include "(batch|pipeline)/' \
  "$metal_src/loader" 2>/dev/null || true)
report_matches "Metal loader must not include the fire path" "$loader_upward"

shipping_test_deps=$(grep -rlnE \
  '#include ".*(tools|tests)/' \
  "$metal_src" 2>/dev/null || true)
report_matches "Metal shipping sources must not include tools or tests" "$shipping_test_deps"

legacy_layout=""
for path in "$metal_src/raw_metal" "$metal_src/executor" "$metal_src/ptir"; do
  if [ -e "$path" ]; then
    legacy_layout+="${path}"$'\n'
  fi
done
report_matches "Legacy Metal source modules must not reappear" "${legacy_layout%$'\n'}"

legacy_refs=$(grep -rnE \
  'raw_metal|pie_metal_driver::(raw_metal|executor|ptir_host)' \
  "$metal_src" --include='*.cpp' --include='*.hpp' --include='*.mm' \
  2>/dev/null || true)
report_matches "Legacy Metal module references must not reappear" "$legacy_refs"

archive="${1:-}"
if [ -n "$archive" ]; then
  if [ ! -f "$archive" ]; then
    echo "::error::Metal archive not found: $archive"
    fail=1
  else
    main_symbols=$(nm -g "$archive" | grep -E '(^|[[:space:]])_?main$' || true)
    report_matches "Shipped Metal archive exports main" "$main_symbols"

    abi_count=$(nm -g "$archive" |
      grep -E '[[:space:]]T[[:space:]]_?pie_metal_' |
      sed -E 's/^.*[[:space:]](_?pie_metal_[^[:space:]]+)$/\1/' |
      sort -u |
      wc -l |
      tr -d ' ')
    if [ "$abi_count" -ne 11 ]; then
      echo "::error::Expected 11 unique pie_metal_* ABI exports, found $abi_count"
      fail=1
    fi
  fi
fi

if [ "$fail" -ne 0 ]; then
  echo "Metal driver layout gate FAILED."
  exit 1
fi

echo "Metal driver layout gate PASSED."
