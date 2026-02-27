# MoE FP4 GEMM Kernel Optimization History

Decode optimization for GPT-OSS-20B on M1 Max (32GB, 400 GB/s bandwidth).

## Summary

| Round | Latency | Key change |
|-------|---------|------------|
| Baseline | 1432ms | Initial implementation |
| Round 1-3 | 519ms | Eliminate syncs, cache params, fuse RoPE |
| Round 4-5 | 71ms | Weight compaction (CPU roundtrip) fixes TLB thrashing |
| Round 6 | 41ms | SIMD K-split decode MoE GEMM kernels (inspired by llama.cpp) |

## Weight dimensions (GPT-OSS-20B)

- Hidden: H=2880, H_PAD=3072 (padded to multiple of 32)
- Intermediate: I=2880
- Experts: E=32 total, K=4 active per token
- FP4 GEMM1 per expert: [2×2880, 3072/2] blocks + [2×2880, 3072/32] scales ≈ 9.2 MB
- FP4 GEMM2 per expert: [3072, 2880/2] blocks + [3072, 2880/32] scales ≈ 4.7 MB
- Total weight reads per decode step: ~3.75 GB → theoretical minimum at 400 GB/s ≈ 9.4ms

## Approaches tried (decode MoE GEMM)

### 1. One-thread-per-output (Kernels 5-6) — REMOVED

**Approach:** Each thread computes one output element, iterating over the full K
dimension (3072 elements) to accumulate the dot product.

**Grid:** (K_experts, intermediate_size, 1)

**Problem:** FP4 dequantization is 5+ ops per value (extract nibble, LUT lookup,
multiply by E8M0 scale). With 3072 elements per thread, the GPU is serialized on
compute — each thread does ~15K FP ops sequentially. GPU occupancy is high (many
threadgroups) but each thread is individually slow.

**Result:** ~2.7 ms/layer, achieving only ~29 GB/s out of 400 GB/s peak.
The bottleneck is compute, not memory bandwidth.

### 2. SIMD-cooperative with shared memory input (Kernels 7-8) — REMOVED

**Approach:** 8 simdgroups per threadgroup (256 threads), each simdgroup handles
one output column. Input activations loaded into shared (threadgroup) memory once,
reused by all 8 simdgroups. Each thread still iterates over K sequentially.

**Grid:** (K_experts, ceil(I/8), 1), Group: (1, 256, 1)

**Problem:** Same fundamental issue — each SIMD lane still processes the full K
dimension sequentially with expensive FP4 dequant. Shared memory helps with input
reuse but doesn't address the core compute bottleneck. The 8-way reuse saves
bandwidth but the dequant compute dominates.

**Result:** ~1.94 ms/layer. Modest improvement from input reuse, but still
compute-bound. All kernel variants in this family (with/without shared mem,
with/without SwiGLU fusion) give essentially identical timing.

### 3. Arithmetic dequant, no LUT (Kernels v3) — REMOVED

**Approach:** Replace LUT-based FP4 dequant with pure arithmetic bit manipulation:
extract sign/exponent/mantissa, construct IEEE 754 float32 directly. Hypothesis:
LUT random access is the bottleneck.

**Problem:** The arithmetic path has similar or slightly more ops than LUT
(conditional select, shifts, ORs). On Apple GPU, threadgroup memory (where the LUT
lives) has effectively L1 latency — the 16-entry LUT fits in a single cache line.
No improvement.

**Result:** Same ~1.94 ms/layer. The bottleneck is total FP ops, not LUT latency.

### 4. Row-major vectorized (Kernels v4) — REMOVED

**Approach:** Process weights in row-major order with float4 vectorized loads.
Read 4 weight bytes at once, dequant 8 FP4 values, dot-product with float4
activation vectors.

**Problem:** Still one-thread-per-output-column. Vectorization helps with memory
access patterns but doesn't reduce the fundamental compute per thread.

**Result:** Same ~1.94 ms/layer.

### 5. simdgroup_multiply_accumulate (SMA) — NOT IMPLEMENTED (analysis only)

**Approach:** Use Apple's matrix multiply intrinsic (`simdgroup_multiply_accumulate`)
for 8×8 tile multiply-accumulate.

**Problem:** For decode (M=1), we'd dispatch 8×8 tiles where only 1 row has real
data. The other 7 rows are padding. This wastes 7/8 of the compute throughput.
SMA is designed for batched GEMM where M >> 1.

Additionally, SMA requires dequantized weights in registers before the multiply,
so the FP4 dequant cost is not eliminated — it just happens before the SMA call
instead of during scalar accumulation.

**Result:** Not implemented. Theoretical analysis shows it can't help for M=1 decode.

### 6. bf16 torch.bmm with runtime gather — REJECTED (analysis only)

**Approach:** Pre-dequantize all expert weights to bf16 at init time, then use
`torch.index_select` to gather K=4 active experts and `torch.bmm` for the matmul.
bf16 torch.bmm achieves 307 GB/s on M1 Max.

**Problem:** Runtime `index_select` to gather 4 experts from 32 costs ~1.5ms
per dispatch (copies ~37 MB). The gather + bmm total is slower than the custom
FP4 kernel despite the faster matmul. Also doubles GPU memory usage.

**Result:** Not integrated. The gather overhead eliminates the bmm advantage.

### 7. SIMD K-split decode kernels (inspired by llama.cpp) — CURRENT PRODUCTION ✓

**Approach:** Inspired by llama.cpp's `ggml-metal.metal`. Instead of one thread per
output column (N-split), 32 SIMD lanes cooperatively split the K (reduction)
dimension. Each pair of lanes (ix=tiisg/2, it=tiisg%2) processes a different
32-element FP4 block. After the K loop, `simd_sum` reduces across all 32 lanes.

**Key techniques:**
- **Threadgroup LUT:** 16-float FP4 lookup table in threadgroup memory (faster
  than constant memory for repeated access)
- **SIMD K-split:** ix = tiisg/2 gives block index (0..15), it = tiisg%2 gives
  half-block (0 or 1). Each lane processes 16 bytes = 32 FP4 values.
- **float4 dot products:** Hardware `dot()` for 4-element multiply-accumulate
- **Deferred scale:** One E8M0 scale multiply per 32-element block, applied after
  the dot product (not per-element)
- **NR0=2:** Each simdgroup handles 2 output rows, reusing activation loads
- **N_SG=2:** 2 simdgroups per threadgroup → 4 output rows per TG

**Why it works:** The fundamental insight is that the FP4 dequant compute is
distributed across 32 threads instead of being serialized in one thread.
Each thread dequants only 16 FP4 values per block (instead of all 3072),
and the 32 partial sums are combined with a single `simd_sum` at the end.
This hides the dequant latency behind the SIMD parallelism.

**Grid:** (ceil(N/4), K_active, 1), ThreadsPerTG: (32, 2, 1)

**Result:** 0.71 ms/layer synced, 1.58 ms/layer pipelined (24-layer).
Full decode: **41.4 ms** (down from 71 ms with old kernels).

## Kernel correctness verification

Decode kernels accumulate in float32 but write bf16 output. When comparing against
a float32 reference, apparent "errors" of 1-2 appear due to bf16 output rounding.
Always compare against `reference.bfloat16().float()` for true correctness
assessment. With this comparison, all errors are exactly 0.000000.

The GEMM2 error amplification is even worse: if GEMM1 output is bf16-rounded
before being fed into GEMM2 verification, accumulated rounding across 2880
intermediate dimensions can produce apparent errors of 100+.

## Lessons learned

1. **FP4 GEMM is compute-bound on M1 Max for M=1.** The per-nibble dequant cost
   (~5 ops per 4-bit value) dominates over memory bandwidth at M=1. Any approach
   that serializes dequant in one thread will hit the same wall (~1.94 ms/layer).

2. **SIMD K-split is the right strategy for M=1 FP4.** Distributing the K dimension
   across 32 SIMD lanes reduces per-thread compute by 32×. The `simd_sum` reduction
   is effectively free (single cycle on Apple GPU).

3. **Threadgroup LUT > constant memory LUT > arithmetic dequant.** For a 16-entry
   table accessed by all threads in a simdgroup, threadgroup memory gives the best
   latency. Arithmetic dequant saves no memory traffic and uses more ALU ops.

4. **Don't use SMA for M=1.** `simdgroup_multiply_accumulate` wastes 7/8 compute
   when only 1 row of the 8×8 tile has real data.

5. **Runtime weight gathering eliminates bmm advantages.** Even if the matmul itself
   is faster with pre-dequantized bf16, the cost of `index_select` to gather K
   active experts at runtime negates the benefit.

6. **bf16 output quantization artifacts are NOT kernel bugs.** Always compare kernel
   output against a bf16-rounded reference, not float32.
