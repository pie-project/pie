#!/bin/bash
# Regenerate the vendored Hopper (sm90) TMA warp-specialized MoE grouped-GEMM
# instantiations in gemm_grouped/90/ after bumping the vendored FlashInfer
# revision or changing PIE_MOE_ACTIVATION in driver/cuda/src/kernels.def.
#
#   BUILD_DIR=/path/to/cuda/build ./regenerate_sm90.sh
#
# H100 is sm90: without these the CutlassMoeFCRunner has no compiled launcher
# for Hopper, getWorkspaceSize throws "Could not find valid config", and EVERY
# MoE model silently falls back to the unfused expert path. That fallback is
# the leading suspect for pie's MoE throughput gap on this box.
#
# Unlike gemm_grouped/80/ -- a straight copy of FlashInfer's generator output --
# this set is derived from the link requirements of pie's own dispatch. The
# compile-time dispatch in moe_gemm_template_dispatch_tma_ws.h names a launcher
# for every tile/cluster/epilogue/swap_ab combination the heuristic could pick,
# and that set is both larger and differently shaped than the subset NVIDIA
# ships (upstream emits ~1440 instantiations covering fp4/fp8/fp16 as well, of
# which pie needs neither the dtypes nor the same bf16 tiles). So: compile the
# dispatch TU, ask the linker which launchers it wants, and instantiate exactly
# those. See moe_gemm_tma_ws_instantiate.h for how the combinations that have no
# compilable CUTLASS specialisation are handled.
#
# BUILD_DIR must be a configured pie CUDA build tree -- the nvcc include set is
# taken from its pie_flashinfer_cutlass_moe response file.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD="${BUILD_DIR:?set BUILD_DIR to a configured driver/cuda build directory}"
RSP="$BUILD/CMakeFiles/pie_flashinfer_cutlass_moe.dir/includes_CUDA.rsp"
NVCC="${CUDACXX:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
FILES_PER_CHUNK=7

if [[ ! -f "$RSP" ]]; then
  echo "not a configured pie CUDA build tree: $RSP missing" >&2
  exit 1
fi

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# 1. Compile the dispatch TU on its own and collect the launcher symbols it
#    references but does not define.
"$NVCC" -O3 -DNDEBUG -std=c++20 --generate-code=arch=compute_90a,code=[sm_90a] \
  -Xcompiler=-fPIC --extended-lambda --expt-relaxed-constexpr \
  -DENABLE_BF16=1 -DENABLE_FP8=1 -DUSING_OSS_CUTLASS_MOE_GEMM=1 \
  -DCOMPILE_HOPPER_TMA_GROUPED_GEMMS=1 --options-file "$RSP" \
  -c "$HERE/../flashinfer_moe/moe_gemm_kernels_bf16_bf16.cu" -o "$OUT/dispatch.o"
nm -uC "$OUT/dispatch.o" |
  grep -oP 'tma_warp_specialized_generic_moe_gemm_kernelLauncher<\K.*' > "$OUT/wanted.txt"

# 2. Turn each demangled template-argument list back into a macro invocation.
python3 - "$OUT/wanted.txt" "$OUT/gen" "$FILES_PER_CHUNK" <<'PY'
import re, sys
from pathlib import Path

wanted, outdir, per_chunk = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
FUSION = {0: "NONE", 1: "ACTIVATION", 2: "GATED_ACTIVATION", 3: "FINALIZE"}


def split_top_level(text):
    parts, depth, cur = [], 0, ""
    for ch in text:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return parts


insts = set()
for line in wanted.read_text().splitlines():
    args = split_top_level(line.rsplit(">(", 1)[0])
    arch = args[0].rsplit("::", 1)[-1]
    dtypes = args[1:4]
    schedule = args[4].rsplit("::", 1)[-1]
    epilogue = args[5].rsplit("::", 1)[-1]
    fusion = FUSION[int(re.search(r"\)(\d+)$", args[6]).group(1))]
    cta = re.findall(r"C<(\d+)>", args[7])
    cga = re.findall(r"C<(\d+)>", args[8])
    flags = args[9:13]
    insts.add(
        "PIE_INSTANTIATE_TMA_WS_MOE_GEMM(%s);"
        % ", ".join([arch, *dtypes, schedule, epilogue, fusion, *cta, *cga, *flags])
    )

header = (
    '#include "moe_gemm_tma_ws_instantiate.h"\n\n'
    "namespace tensorrt_llm::kernels::cutlass_kernels_oss {\n\n"
)
outdir.mkdir(parents=True)
ordered = sorted(insts)
chunks = [ordered[i : i + per_chunk] for i in range(0, len(ordered), per_chunk)]
for index, chunk in enumerate(chunks):
    body = "\n".join(chunk)
    path = outdir / f"cutlass_kernel_file_gemm_grouped_sm90_group{index}.generated.cu"
    path.write_text(f"{header}{body}\n\n}}  // namespace tensorrt_llm::kernels::cutlass_kernels_oss\n")
print(f"{len(ordered)} instantiations in {len(chunks)} files")
PY

if [[ -d "$HERE/gemm_grouped/90" ]] &&
   diff -qr "$OUT/gen" "$HERE/gemm_grouped/90" >/dev/null; then
  echo "no changes"
  exit 0
fi
rm -rf "$HERE/gemm_grouped/90"
cp -r "$OUT/gen" "$HERE/gemm_grouped/90"
echo "regenerated $HERE/gemm_grouped/90"
