// The scalar layer and the fixed-width integer names, out of the
// prelude: NVRTC has no CUDA device headers, and this file is meant
// to compile under both it and nvcc.
#include "pie_device.cuh"
#include "gemm/gemv.hpp"


// `<type_traits>` was here for the sweep entry points' `std::integral_constant`
// dispatch. They are deleted and nothing left in the file names it.
#include <cstdint>

namespace pie_cuda_driver::kernels::gemm {

namespace {

// One warp per output row. Each lane walks the row in float4 strides (8
// bf16 = 16 B, so a warp step covers 512 B — four full cache lines) and
// accumulates in fp32; a single shuffle tree finishes the dot product.
//
// The walk is unrolled by `kUnroll` and the loads are hoisted above the math
// that consumes them. That is the whole reason for the unroll: written the
// obvious way each lane has exactly ONE load in flight, because the FMA on
// `w4[i]` is the next instruction after it, and a warp that is waiting on a
// single HBM round trip cannot cover the latency no matter how many warps the
// SM holds. Measured on an H100 at gpt-oss's o_proj shape (N=2880, K=4096,
// 23.6 MB per layer) the one-at-a-time version sustained about 963 GB/s.
template <int kWarps, int kUnrollP = 4>
__global__ void gemv_bf16_kernel(
    const device::bf16* __restrict__ weight,
    const device::bf16* __restrict__ act,
    const device::bf16* __restrict__ bias,
    device::bf16* __restrict__ out,
    int N, int K, float beta)
{
    const int row = blockIdx.x * kWarps + threadIdx.y;
    if (row >= N) return;
    const float4* w4 =
        reinterpret_cast<const float4*>(weight + (long long)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;
    constexpr int kUnroll = kUnrollP;
    float acc = 0.f;
    int i = threadIdx.x;
    for (; i + 32 * (kUnroll - 1) < vectors; i += 32 * kUnroll) {
        float4 wv[kUnroll];
        float4 xv[kUnroll];
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            wv[u] = w4[i + 32 * u];
            xv[u] = x4[i + 32 * u];
        }
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const device::bf16* wb =
                reinterpret_cast<const device::bf16*>(&wv[u]);
            const device::bf16* xb =
                reinterpret_cast<const device::bf16*>(&xv[u]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += device::bf16_to_f32(wb[j]) * device::bf16_to_f32(xb[j]);
            }
        }
    }
    for (; i < vectors; i += 32) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const device::bf16* wb = reinterpret_cast<const device::bf16*>(&wv);
        const device::bf16* xb = reinterpret_cast<const device::bf16*>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += device::bf16_to_f32(wb[j]) * device::bf16_to_f32(xb[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (threadIdx.x == 0) {
        if (beta != 0.f) acc += beta * device::bf16_to_f32(out[row]);
        // Round to bf16 *before* adding the bias, then round again. That is
        // a redundant-looking double rounding, and it is deliberate: it is
        // exactly what the separate `add_bias_bf16_kernel` did when it read
        // this kernel's bf16 output back. Folding the two launches into one
        // is a launch-count optimization, not an arithmetic change, so it
        // has to stay bit-identical or it stops being free to validate.
        device::bf16 v = device::f32_to_bf16(acc);
        if (bias != nullptr) {
            v = device::f32_to_bf16(device::bf16_to_f32(v) +
                                 device::bf16_to_f32(bias[row]));
        }
        out[row] = v;
    }
}

// How deep to unroll the row walk. Blackwell wants a SHALLOWER unroll than
// Hopper, which is the opposite of what the comment above predicts, so it is
// selected from the device rather than fixed.
//
// The unroll exists to keep several loads in flight; on an H100 one-at-a-time
// sustained only ~963 GB/s, and 4 was the tuned depth there. On B200 four is
// past the point of diminishing returns and costs more than it buys -- two is
// enough to cover the latency and leaves registers for occupancy. Measured
// cold, 8 rotating buffers, bf16, M=1 (driver/cuda/bench/gemv_bench.cu sweep):
//
//   shape                     unroll=4          unroll=2
//   qwen27 gate/up      34.2us 5.22TB/s   30.7us 5.80TB/s   -10.2%
//   qwen27 down         35.3us 5.06TB/s   32.2us 5.54TB/s    -8.8%
//   gemma31 down        43.3us 5.34TB/s   39.1us 5.91TB/s    -9.7%
//   gptoss lm_head     227.4us 5.09TB/s  194.5us 5.95TB/s   -14.5%
//
// This kernel is 78% of Qwen3.6-27B's decode step and 77% of gemma-4-31B's,
// so the depth is worth taking from a measurement. Hopper keeps 4: it was
// tuned there and nothing here re-measured it on that part.
//
// # This answered to `getenv("PIE_GEMV_B200_TUNING")`, and it must not
//
// The variable set the return to 4 -- "revert to the Hopper constants
// without a rebuild" -- and it reached THREE launchers, not one: this
// function's answer gates the row-per-warp unroll in `gemv_bf16`, the
// warps AND unroll of the split-K form in the same launcher, and the fused
// QKV kernel in `gemv3_bf16`. It is deleted, and unlike
// `PIE_QWEN35_GDN_SMEM_STEP` (§30) it is NOT deleted because the arms agree.
// They do not. Measured on an L40S, nine shapes, both arms fired against
// byte-identical inputs with a poisoned output buffer, a permutation control
// that moved ~89% of the weight bytes and a truncation control that left half
// the output poison, all firing at every shape (§36 has the run):
//
//   arms                                    benign data   wide-exponent data
//   gemv_bf16_kernel<4,4>  vs <4,2>            0 bytes       0 bytes
//   gemv_splitk_bf16_kernel<8,1> vs <4,2>      0 bytes       5 bytes / 9 shapes
//   gemv3_bf16_kernel<8,1> vs <2,2>            0 bytes       3 bytes / 9 shapes
//
// The unroll depth alone is safe by construction and measured so: at kUnroll
// 4 and 2 a lane visits the same vectors in the same order (i, i+32, i+64 …),
// so the fp32 accumulation is the same additions in the same sequence. The
// WARP count is not: eight warps partition K at stride 256 and four at
// stride 128, and the shared-memory tree then sums different partials. That
// is below bf16's last bit on model-shaped weights, which is why the benign
// column is zero, and it is NOT below it once the exponents spread.
//
// So the variable did what an env-var selector always does: it made the same
// trace on the same weights on the same GPU emit different bits, with nothing
// in the plan, the replay, or another backend able to say which arm ran. A
// bisect switch is not worth that, and nothing in the repository ever set it.
//
// What is left is a DEVICE FACT, which is a different thing with a different
// fix: `cudaDevAttrComputeCapabilityMajor` is a property of the machine, the
// same on every replay on that machine, and discoverable by any backend that
// asks. It is still answered here rather than carried from load -- §36 says
// what carrying it would take -- but it is answerable, and an environment
// variable is not.
inline int gemv_unroll_depth() {
    static const int depth = [] {
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 4;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   dev) != cudaSuccess) {
            return 4;
        }
        return major >= 10 ? 2 : 4;
    }();
    return depth;
}

// One BLOCK per output row, its warps splitting K between them and reducing
// through shared memory.
//
// The kernel above gives one warp to each row, so the grid is N/4 blocks and
// the machine is only busy if N is large. gpt-oss decodes through three shapes
// where it is not: k_proj and v_proj are N=512, which is 128 blocks on 132 SMs,
// and the MoE router is N=32, which is EIGHT. vLLM runs the same projections
// through cuBLAS's `splitK` nvjet kernels at 2.3-2.5 TB/s for exactly this
// reason. Splitting inside the block keeps it to one launch and needs no
// scratch, which the reduce-across-blocks form would.

// MEASURED DEAD END, recorded because the reasoning looks sound and someone
// will try it again: staging the activation in shared memory before the dot
// product is SLOWER. Every block reads the same activation vector, so the
// loop below issues two global loads per iteration -- one weight, one
// activation -- and repeats the activation read for all N blocks, which looks
// like an obvious thing to hoist. It is not: those re-reads are already
// served by L2, and paying a __syncthreads plus a shared-memory round trip to
// avoid them costs more than they do. Measured cold on B200 (bf16, one token):
// gpt-oss q_proj 10.3 -> 13.7 us, o_proj 8.3 -> 11.5, gemma-4-31B kv_proj
// 12.5 -> 16.4. Results were bit-identical, so this is purely a loss.
template <int kWarps, int kUnrollP = 1>
__global__ void gemv_splitk_bf16_kernel(
    const device::bf16* __restrict__ weight,
    const device::bf16* __restrict__ act,
    const device::bf16* __restrict__ bias,
    device::bf16* __restrict__ out,
    int N, int K, float beta)
{
    const int row = blockIdx.x;
    if (row >= N) return;
    const int warp = threadIdx.y;
    const float4* w4 =
        reinterpret_cast<const float4*>(weight + (long long)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;

    float acc = 0.f;
    // Warp `warp` walks a strided share of the row, so the whole block still
    // reads it in the same coalesced order the one-warp kernel does.
    // `kUnrollP` hoists loads above the math for the same reason the
    // row-per-warp kernel does it: written flat, each lane has ONE load in
    // flight. Default 1 keeps the shipping code byte-identical.
    constexpr int kU = kUnrollP;
    const int stride = kWarps * 32;
    int i = warp * 32 + threadIdx.x;
    for (; i + stride * (kU - 1) < vectors; i += stride * kU) {
        float4 wv[kU];
        float4 xv[kU];
        #pragma unroll
        for (int u = 0; u < kU; ++u) {
            wv[u] = w4[i + stride * u];
            xv[u] = x4[i + stride * u];
        }
        #pragma unroll
        for (int u = 0; u < kU; ++u) {
            const device::bf16* wb =
                reinterpret_cast<const device::bf16*>(&wv[u]);
            const device::bf16* xb =
                reinterpret_cast<const device::bf16*>(&xv[u]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += device::bf16_to_f32(wb[j]) * device::bf16_to_f32(xb[j]);
            }
        }
    }
    for (; i < vectors; i += stride) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const device::bf16* wb = reinterpret_cast<const device::bf16*>(&wv);
        const device::bf16* xb = reinterpret_cast<const device::bf16*>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += device::bf16_to_f32(wb[j]) * device::bf16_to_f32(xb[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    __shared__ float partial[kWarps];
    if (threadIdx.x == 0) partial[warp] = acc;
    __syncthreads();
    if (warp != 0 || threadIdx.x != 0) return;
    float total = 0.f;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) total += partial[w];
    if (beta != 0.f) total += beta * device::bf16_to_f32(out[row]);
    // Same double rounding as the kernel above, for the same reason: it is
    // what the separate bias kernel used to do, so the fold stays bit-exact.
    device::bf16 v = device::f32_to_bf16(total);
    if (bias != nullptr) {
        v = device::f32_to_bf16(device::bf16_to_f32(v) + device::bf16_to_f32(bias[row]));
    }
    out[row] = v;
}

bool aligned16(const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

}  // namespace

bool gemv_bf16(
    const void* weight,
    const void* act,
    const void* bias,
    void*       out,
    int N, int K,
    cudaStream_t stream,
    float beta)
{
    // The float4 loads need each row to start 16-byte aligned: that holds
    // iff the base is aligned and the row stride is a multiple of 8 bf16.
    if (N <= 0 || K <= 0 || (K % 8) != 0) return false;
    if (weight == nullptr || act == nullptr || out == nullptr) return false;
    if (!aligned16(weight) || !aligned16(act)) return false;
    constexpr int kWarps = 4;
    // Below this the row-per-warp grid cannot fill the device, and splitting K
    // inside the block is strictly more parallel for the same traffic. 2048
    // rows is 512 blocks, about four per SM, which is where the row-per-warp
    // form stops being the constraint.
    //
    // # 4096 is the default it always had, and it is a constant now
    //
    // This read `getenv("PIE_GEMV_SPLITK_MAX_ROWS")`, defaulting to 4096.
    // That is a threshold, not a toggle, and the two sides of it are two
    // DIFFERENT `__global__`s at different grids -- `N` blocks of `32 x 8`
    // against `N/4` blocks of `32 x 4` -- so the variable chose which kernel
    // ran at all. Measured on an L40S at nine shapes with wide-exponent
    // weights (§36), split-K and row-per-warp disagree at two of them, 1 byte
    // each: the split is over K, the reduction tree differs, and bf16 does
    // not always absorb it. Under graph replay they also differ in TIME, in
    // both directions and by more than the constant admits:
    //
    //   N        split-K<8,1>   row-per-warp<4,4>
    //   32          1.48 us        2.47 us     split-K 1.67x
    //   512         2.39           3.09        split-K 1.29x
    //   2048        3.47           3.32        row-per-warp 1.05x
    //   4096        5.32           4.73        row-per-warp 1.12x
    //   8192        9.10           7.59        row-per-warp 1.20x
    //
    // So on a 142-SM L40S the crossover is near 2048 -- which is the number
    // the paragraph above names -- and the shipping 4096 takes the slower
    // kernel over 2048 < N <= 4096. The value is NOT changed here: 4096 is
    // what every deployment runs today, the tables it came from were taken on
    // a 132-SM B200, and moving it is a separate claim that wants a B200 to
    // make. What changes is that it is a named constant with its provenance
    // written down instead of a string read from the environment, so a replay
    // sees the same kernel the plan does.
    constexpr int kSplitKMaxRows = 4096;
    if (N <= kSplitKMaxRows) {
        // Warps per block and unroll depth, measured under GRAPH REPLAY at the
        // shapes these five models actually decode through.
        //
        // Two earlier attempts got this wrong the same way: both swept EAGER,
        // where the launch floor on this box is ~4.1 us. Most of these shapes
        // run under that, so the sweep compared launch overhead rather than
        // kernels. The first shipped a blanket warps=2 and cost gemma-4-26B
        // 3.4% and Qwen3.6-27B 1.9%; the second papered over it with a size
        // threshold. Timed the way pie actually decodes -- inside a captured
        // graph -- w=8,u=1 is the WORST config on 11 of 12 shapes:
        //
        //   shape                MB   w8u1   w4u2   per-shape best
        //   qwen35 q_proj      16.8   5.45   4.00   3.79 (w2u1)
        //   qwen35 o_proj      16.8   4.32   3.45   3.45
        //   qwen35 lin qk       8.4   3.52   2.65   2.65
        //   qwen27 gdn qk      21.0   5.22   4.17   3.91 (w2u2)
        //   gptoss o_proj      23.6   6.25   4.86   4.80 (w2u1)
        //   gemma31 kv_proj    44.0  11.35   9.25   9.25
        //   gemma26 q_proj     23.1   7.34   5.48   4.85 (w2u2)
        //
        // Summed over all twelve, w=4,u=2 is 20% faster than w=8,u=1 and
        // within 3% of picking the best config per shape -- one config, no
        // threshold. Its only loss is gpt-oss's 0.2 MB router, 1.6% of 1.9 us.
        // Hopper keeps w=8,u=1, which is where it was tuned.
        if (gemv_unroll_depth() == 2) {
            constexpr int kSplitWarpsB = 4;
            gemv_splitk_bf16_kernel<kSplitWarpsB, /*kUnrollP=*/2>
                <<<dim3(static_cast<unsigned>(N)), dim3(32, kSplitWarpsB), 0,
                   stream>>>(
                    static_cast<const device::bf16*>(weight),
                    static_cast<const device::bf16*>(act),
                    static_cast<const device::bf16*>(bias),
                    static_cast<device::bf16*>(out),
                    N, K, beta);
            return true;
        }
        constexpr int kSplitWarps = 8;
        gemv_splitk_bf16_kernel<kSplitWarps>
            <<<dim3(static_cast<unsigned>(N)), dim3(32, kSplitWarps), 0,
               stream>>>(
                static_cast<const device::bf16*>(weight),
                static_cast<const device::bf16*>(act),
                static_cast<const device::bf16*>(bias),
                static_cast<device::bf16*>(out),
                N, K, beta);
        return true;
    }
    const long long blocks = (N + kWarps - 1) / kWarps;
    if (blocks > 2147483647LL) return false;
    // Everything below is unconditional, so the caller never has to
    // reason about a half-enqueued launch. In particular this must not
    // poll `cudaGetLastError`: that would consume an unrelated pending
    // error the driver's own checks are waiting to report.
    if (gemv_unroll_depth() == 2) {
        gemv_bf16_kernel<kWarps, 2>
            <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kWarps), 0,
               stream>>>(
                static_cast<const device::bf16*>(weight),
                static_cast<const device::bf16*>(act),
                static_cast<const device::bf16*>(bias),
                static_cast<device::bf16*>(out),
                N, K, beta);
        return true;
    }
    gemv_bf16_kernel<kWarps, 4>
        <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kWarps), 0, stream>>>(
            static_cast<const device::bf16*>(weight),
            static_cast<const device::bf16*>(act),
            static_cast<const device::bf16*>(bias),
            static_cast<device::bf16*>(out),
            N, K, beta);
    return true;
}

// ── DELETED: the fused QKV triple and the three sweep entry points ────
//
// `gemv3_bf16`, `gemv3_bf16_tuned`, `gemv_bf16_tuned` and `gemv_splitk_tuned`
// were here, with `gemv3_bf16_kernel` above them: five `<<<>>>` over one
// `__global__` nobody launched. §10.10's rule is that a launcher goes when its
// WHOLE consumer set has gone, so each was checked on its own rather than as a
// group, over `.cu/.cpp/.cuh/.hpp/.rs/.py/.txt/.cmake/.toml` in the whole
// worktree:
//
//   gemv3_bf16          declared in `gemv.hpp`, defined here, called nowhere.
//                       Its table row was deleted by §27 (no model text
//                       lowered `cuda::gemv3`); `gemm.cpp` calls only
//                       `gemv_bf16`.
//   gemv3_bf16_tuned    sweep entry. No harness: `benches/` is Python against
//                       the served engine and `driver/cuda/bench/gemv_bench.cu`
//                       -- named in the comment that stood here -- is in no
//                       source directory of this repository.
//   gemv_bf16_tuned     the same, for the row-per-warp form.
//   gemv_splitk_tuned   the same, for the split-K form.
//
// `scripts/csrc-reachability-audit.py` reports all four UNREACHABLE from every
// root, and it over-approximates reachability on purpose. Its one blind spot
// is a call through a function pointer; §37 inspected all 22 non-call
// name-mentions in the tree and found none, and none of these four names
// appears as anything but a declaration, a definition or prose.
//
// `gemv_splitk_bf16_kernel` STAYS: `gemv_bf16` launches it twice, at
// `<4,2>` and `<8,1>`. `gemv3_bf16_kernel` went with its two callers.


}  // namespace pie_cuda_driver::kernels::gemm