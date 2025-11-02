#!/bin/bash

# Lint changed Python files in backend/backend-python (pylint_v1 workflow)
# This script is used by .github/workflows/pylint_v1.yml to check changed files in PRs
#
# Usage:
#   ./tests/run_pylint_v1.sh <base_ref> <head_ref>
#
# Example:
#   ./tests/run_pylint_v1.sh origin/main HEAD

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default refs if not provided
BASE_REF="${1:-origin/main}"
HEAD_REF="${2:-HEAD}"

echo "Checking changed files between $BASE_REF and $HEAD_REF"

# Directories to always check all Python files (matches backend-python-pylint.sh)
PREDEFINED_PATHS=(
  "backend/backend-python"              # Direct .py files only
  "backend/backend-python/config"       # All config files
  "backend/backend-python/model"        # All model files
  "backend/backend-python/metal_kernels"  # All metal_kernels files (recursive)
)

# Directories to exclude from linting (takes priority over everything)
EXCLUDED_DIRS=(
  "backend/backend-python/tests"
)

# Helper function to check if file is in excluded directory
is_excluded() {
  local file="$1"
  for excluded in "${EXCLUDED_DIRS[@]}"; do
    if [[ "$file" == "$excluded"* ]]; then
      return 0  # true - file is excluded
    fi
  done
  return 1  # false - file is not excluded
}

# Get changed Python files
echo "Finding changed Python files..."
CHANGED_FILES=$(git diff --name-only --diff-filter=ACMRT "$BASE_REF" "$HEAD_REF" -- \
  'backend/backend-python' \
  | grep -E '\.py$' || true)

# Collect predefined files
echo "Including predefined files..."
PREDEFINED_FILES=()

# For the first path (backend-python), only include direct children
if [ -d "$ROOT/backend/backend-python" ]; then
  for file in "$ROOT/backend/backend-python"/*.py; do
    if [ -f "$file" ]; then
      rel_path="${file#$ROOT/}"
      PREDEFINED_FILES+=("$rel_path")
    fi
  done
fi

# For other paths, include recursively
for path in "backend/backend-python/config" "backend/backend-python/model" "backend/backend-python/metal_kernels"; do
  if [ -d "$ROOT/$path" ]; then
    while IFS= read -r -d '' file; do
      rel_path="${file#$ROOT/}"
      PREDEFINED_FILES+=("$rel_path")
    done < <(find "$ROOT/$path" -type f -name "*.py" -print0)
  fi
done

# Combine all files, removing duplicates and applying exclusions
ALL_FILES=()
if [ -n "$CHANGED_FILES" ]; then
  while IFS= read -r file; do
    if [ -n "$file" ] && ! is_excluded "$file"; then
      ALL_FILES+=("$file")
    fi
  done <<< "$CHANGED_FILES"
fi
for file in "${PREDEFINED_FILES[@]}"; do
  # Skip if excluded or already in array
  if is_excluded "$file"; then
    continue
  fi
  skip=0
  for existing in "${ALL_FILES[@]}"; do
    if [ "$existing" = "$file" ]; then
      skip=1
      break
    fi
  done
  [ $skip -eq 0 ] && ALL_FILES+=("$file")
done

if [ "${#ALL_FILES[@]}" -eq 0 ]; then
  echo "No Python files to lint (no changes detected)"
  exit 0
fi

echo "Files to lint:"
printf '  %s\n' "${ALL_FILES[@]}"
echo ""

# Run Black
echo "Running Black..."
cd "$ROOT"
uvx black --check "${ALL_FILES[@]}"
echo "✓ Black passed"
echo ""

# Run Pylint
echo "Running Pylint..."
PYTHONPATH=$ROOT:$ROOT/backend/backend-python uv run \
  --project "$ROOT/backend/backend-python" \
  --with pylint \
  --with torch \
  pylint --disable=R "${ALL_FILES[@]}"
echo "✓ Pylint passed"
echo ""

# Run Pyright
echo "Running Pyright..."
PYTHONPATH=$ROOT:$ROOT/backend/backend-python uv run \
  --project "$ROOT/backend/backend-python" \
  --with pyright \
  --with torch \
  pyright "${ALL_FILES[@]}"
echo "✓ Pyright passed"
echo ""

echo "All checks passed! ✨"
