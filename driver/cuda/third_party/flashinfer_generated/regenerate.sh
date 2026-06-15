#!/bin/bash
# Regenerate the vendored FlashInfer CUTLASS grouped-GEMM MoE launchers after
# bumping the vendored FlashInfer revision.
#
#   FLASHINFER_SRC=/path/to/flashinfer-src ./regenerate.sh
#
# tvm_ffi is import-only in the codegen path (never called), so it is stubbed
# to avoid the heavy apache-tvm-ffi dependency. tqdm and pynvml ARE used by
# the generator and must be the real, lightweight packages:
#   pip install tqdm pynvml
# (Stubbing tqdm silently truncates the generated kernel set.)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FIS="${FLASHINFER_SRC:?set FLASHINFER_SRC to the flashinfer source dir}"
OUT="$(mktemp -d)"
FLASHINFER_DISABLE_VERSION_CHECK=1 PYTHONPATH="$FIS" python3 - "$OUT" <<'PY'
import sys, types
sys.modules["tvm_ffi"] = types.ModuleType("tvm_ffi")  # heavy + unused -> stub
from pathlib import Path
from flashinfer.jit.gemm.cutlass.generate_kernels import generate_gemm_operations
generate_gemm_operations(Path(sys.argv[1]), "89;89-real")
PY
rm -rf "$HERE/gemm_grouped"
cp -r "$OUT/gemm_grouped" "$HERE/gemm_grouped"
rm -rf "$OUT"
echo "regenerated $(find "$HERE/gemm_grouped/80" -name '*.generated.cu' | wc -l) launchers (expect: 60 across the files)"
