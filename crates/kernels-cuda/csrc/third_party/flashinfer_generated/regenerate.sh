#!/bin/bash
# Regenerate the vendored FlashInfer CUTLASS grouped-GEMM MoE launchers after
# bumping the vendored FlashInfer revision.
#
#   FLASHINFER_SRC=/path/to/flashinfer-src ./regenerate.sh
#
# Load only the CUTLASS generator package. Importing flashinfer itself also
# loads optional runtime backends whose Python dependencies are irrelevant to
# this source-only generation step.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FIS="${FLASHINFER_SRC:?set FLASHINFER_SRC to the flashinfer source dir}"
GENERATOR="$FIS/flashinfer/jit/gemm/cutlass/generate_kernels.py"
if [[ ! -f "$GENERATOR" ]]; then
  echo "FlashInfer CUTLASS generator not found: $GENERATOR" >&2
  exit 1
fi
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
python3 - "$FIS" "$OUT" <<'PY'
import importlib
import re
import subprocess
import sys
import types
from pathlib import Path


def add_namespace(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__package__ = name
    module.__path__ = [str(path)]
    sys.modules[name] = module


source_root = Path(sys.argv[1])
package_root = source_root / "flashinfer"
add_namespace("flashinfer", package_root)
add_namespace("flashinfer.jit", package_root / "jit")
add_namespace("flashinfer.jit.gemm", package_root / "jit" / "gemm")
add_namespace(
    "flashinfer.jit.gemm.cutlass",
    package_root / "jit" / "gemm" / "cutlass",
)


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def is_cuda_version_at_least(required: str) -> bool:
    output = subprocess.check_output(["nvcc", "--version"], text=True)
    match = re.search(r"release ([0-9]+(?:\.[0-9]+)+)", output)
    if match is None:
        raise RuntimeError("unable to parse CUDA version from nvcc --version")
    return version_tuple(match.group(1)) >= version_tuple(required)


cpp_ext = types.ModuleType("flashinfer.jit.cpp_ext")
cpp_ext.is_cuda_version_at_least = is_cuda_version_at_least
sys.modules[cpp_ext.__name__] = cpp_ext

generator = importlib.import_module(
    "flashinfer.jit.gemm.cutlass.generate_kernels"
)
generator.generate_gemm_operations(Path(sys.argv[2]), "89;89-real")
PY
GENERATED="$OUT/gemm_grouped"
FILE_COUNT="$(find "$GENERATED/80" -name '*.generated.cu' -type f | wc -l)"
INSTANTIATION_COUNT="$(
  grep -R -h '^ *template void' "$GENERATED/80"/*.generated.cu | wc -l
)"
if [[ "$FILE_COUNT" -ne 9 || "$INSTANTIATION_COUNT" -ne 60 ]]; then
  echo "unexpected generated MoE kernel set: $FILE_COUNT files, $INSTANTIATION_COUNT instantiations" >&2
  exit 1
fi
if [[ -d "$HERE/gemm_grouped" ]] &&
   diff -qr "$GENERATED" "$HERE/gemm_grouped" >/dev/null; then
  echo "verified $FILE_COUNT files ($INSTANTIATION_COUNT template instantiations); no changes"
  exit 0
fi
rm -rf "$HERE/gemm_grouped"
cp -r "$GENERATED" "$HERE/gemm_grouped"
echo "regenerated $FILE_COUNT files ($INSTANTIATION_COUNT template instantiations)"
