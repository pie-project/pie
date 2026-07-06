# driver/metal/kvattn — Metal KV/attention parity (M3)

Metal duals of the CUDA driver's paged-KV / attention path, validated against the
CUDA index math and (for the numeric attention kernels) a CPU reference. Standalone,
no MLX, reuses the ptir `MetalHarness`. The macOS/Apple-Silicon cross-backend track
for the KV + attention geometry.

## Kernels

| kernel | CUDA source | check |
|--------|-------------|-------|
| `write_kv`     | `kv_paged.cu` `write_kv_kernel` (NHD + HND layouts) | **bit-exact** (u16 movement) |
| `gather_rows`  | `gather_rows.cu` `gather_bf16_rows_kernel`          | **bit-exact** (u16 movement) |
| `paged_attention_decode` | paged SDPA (raw_metal `sdpa_paged` / FlashInfer decode) | **~f32-epsilon** vs CPU ref (`max_abs ≈ 9e-8`) |

K/V movement (`write_kv`/`gather_rows`) is carried as opaque 16-bit words (bf16
bits): a copy/gather is bit-exact regardless of numeric interpretation. The paged
index math (`find_request` → `abs_kv_pos` → `page/offset`, head-major vs
non-head-major dst) mirrors `write_kv_kernel` exactly.

`paged_attention_decode` is single-query causal attention over the paged NHD KV
cache (same layout as `write_kv` + raw_metal `sdpa_paged`): one thread per
`(row, q_head)`, online (flash) softmax over kv positions `[0, position_ids[row]]`,
GQA grouping. Softmax `exp` is not bit-exact GPU-vs-host, so it is validated within
a tight tolerance vs a CPU reference that uses the **same online-softmax order** —
observed `max_abs ≈ 9e-8` (f32 epsilon), i.e. only exp rounding differs.

## Build & run

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/kvattn_test        # -> KVATTN_TEST_OK
```

Direct:

```sh
clang++ -std=c++17 -fobjc-arc -O2 -I . -I ../ptir \
  -DKVATTN_KERNELS_DIR="\"$PWD/kernels\"" \
  -x objective-c++ ../ptir/metal_harness.mm kvattn_test.mm \
  -framework Metal -framework Foundation -o kvattn_test && ./kvattn_test
```
