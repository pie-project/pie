//===-- vision/gemma4_vision.cuh - the gemma-4 vision tower's device text -===//
//
// The nine `__global__`s of the Gemma-4 vision encoder, as templates over the
// storage format, in a namespace NVRTC can name.
//
// # Why this exists
//
// `gemma4_vision.cu` used to be one file: nine kernels and three host entry
// points, the kernels inside `namespace pie_cuda_driver::model { namespace {
// ... } }`. That shape is fine for nvcc and impossible for the JIT. An
// anonymous namespace has no name to hand `nvrtcAddNameExpression`, so the
// runtime cannot resolve a `CUfunction` for any of them; and a non-template
// `__global__` in a header emits one strong definition per translation unit,
// so the moment the anonymous namespace goes the ahead-of-time link fails on
// duplicates. A NAMED template fixes both, which is why the two changes are
// one change.
//
// The `.cu` next door keeps the host half -- `run_gemma4_vision`,
// `scatter_gemma4_vision`, `encode_gemma4_vision`, the `DeviceScratch` arena
// and the cuBLAS handle -- and `#include`s this. It is a MOVE and not a copy:
// there is exactly one definition of each kernel below in the whole tree, and
// `tests/sources.rs::no_global_is_defined_twice` fails the build over the
// alternative. It has to, because `norm/altup_aux` shipped a release with two
// copies that had drifted and every test passing.
//
// # What is unchanged
//
// The arithmetic, line for line. `bf` became `T`; `F`/`Bf` became
// `Elem<T>::to_f32`/`from_f32` through `tower_naive_kernels.cuh`'s two
// helpers; the flat element counts became `usize` where they were `long`,
// which is the same width on this target and the only one a row can state.
// Nothing else. The tower is parity-checked against HF-bf16 dumps at rel_rms
// 1.07% / cosine 0.99994, and that number belongs to THIS arithmetic.
//
// # What is rowed, and what refused
//
// Three of the nine. The other six are the reason `LaunchRule` has a
// vocabulary rather than a constructor: five launch on a 2-D block
// (`B2 = dim3(16,16)`) and one on a 3-D grid, and every ported rule states
// `block = [n,1,1]` over a 1-D or 2-D grid. Each kernel below carries its
// `<<<>>>` and the reason, per `new-horizon.md` §17.9 -- a refusal that names
// the launch is worth more than a rule stretched to cover it.
//
//   k_scale          Rule::Elementwise
//   k_pool_finish    Rule::Elementwise
//   k_softmax        Rule::PerRow
//   k_addpos_grid2d  refused -- 2-D block
//   k_rope_axial2d   refused -- 3-D grid
//   k_qk             refused -- 2-D block
//   k_av             refused -- 2-D block
//   k_pool           refused -- 2-D block
//   k_gelu_mul       refused -- DEAD; no launch to cite
//
//===---------------------------------------------------------------------===//
#pragma once

#include "vision/gemma4_naive_kernels.cuh"

namespace pie_cuda_driver::kernels::vision::device {

/// `[-1, 1]`-rescale of the patch pixels: `o = 2*(p - 0.5)`.
///
/// `Rule::Elementwise`: `k_scale<<<((long)N*Hd+255)/256, 256, 0, S>>>` with
/// `t = (long)N*Hd`, and `elementwise` evaluates `rows * width` -- the row
/// states `rows = N`, `width = Hd` -- to the same `ceil(N*Hd/256)` blocks of
/// 256 with no dynamic shared memory.
template <class T>
__global__ void k_scale(const T* p, T* o, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < t) o[i] = Bf<T>(2.f * (F(p[i]) - 0.5f));
}

/// Add the two axial position-table rows for each patch's (x, y).
///
/// UNROWED: `k_addpos_grid2d<<<G2(Hd,N), B2, 0, S>>>` where
/// `B2 = dim3(16,16)` and `G2(X,Y) = dim3((X+15)/16, (Y+15)/16)`. Every ported
/// rule states `block = [n,1,1]`; a 16x16 block is outside the vocabulary, and
/// the kernel READS `threadIdx.y`, so it is not a 1-D block wearing a 2-D
/// spelling.
///
/// `pos` is float and not int because it is the same buffer `k_rope_axial2d`
/// consumes, where the values are trigonometric arguments. The `llrintf` and
/// the two clamps below are what a grid index costs for sharing it.
template <class T>
__global__ void k_addpos_grid2d(T* y, const T* tb, const float* pos, int N, int O, int P) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, o = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || o >= O) return;
    long x = (long)llrintf(pos[2L * n]), yy = (long)llrintf(pos[2L * n + 1]);
    if (x < 0) x = 0;
    if (yy < 0) yy = 0;
    y[(long)n * O + o] = Bf<T>(F(y[(long)n * O + o]) + F(tb[(0L * P + x) * O + o]) + F(tb[(1L * P + yy) * O + o]));
}

/// 2-D axial RoPE over a 64-wide head: the first 32 lanes rotate on x, the
/// second 32 on y.
///
/// UNROWED: `dim3 rg(1,NH,N); k_rope_axial2d<<<rg, 32, 0, S>>>`. A 3-D grid,
/// and the kernel reads all three of `blockIdx.{x,y,z}`. No ported rule emits
/// one -- `Rule::PerHead` is the closest at `[heads, rows, 1]`, and it is the
/// wrong two of the three axes as well as the wrong block width.
///
/// The head dimension is hard-coded 64 (and the half-width 16) because the
/// launcher checks `Hd == 768 && NH == 12` and throws otherwise; a row would
/// have to state that constant as a template argument to be honest about it.
template <class T>
__global__ void k_rope_axial2d(T* q, const float* pos, int N, int H, float theta) {
    int n = blockIdx.z, head = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || head >= H || c >= 16) return;
    T* v = q + (((long)n * H + head) * 64);
    float px = pos[2L * n], py = pos[2L * n + 1];
    float invf = powf(theta, -(float)c / 16.f);
    float cx = cosf(px * invf), sx = sinf(px * invf), cy = cosf(py * invf), sy = sinf(py * invf);
    float a = F(v[c]), b = F(v[c + 16]);
    v[c] = Bf<T>(a * cx - b * sx);
    v[c + 16] = Bf<T>(b * cx + a * sx);
    float e = F(v[32 + c]), f = F(v[48 + c]);
    v[32 + c] = Bf<T>(e * cy - f * sy);
    v[48 + c] = Bf<T>(f * cy + e * sy);
}

/// One head's `Q K^T * scale` into an `[N, N]` float score matrix.
///
/// UNROWED: `k_qk<<<G2(N,N), B2, 0, S>>>`, `B2 = dim3(16,16)`. 2-D block,
/// `threadIdx.y` read.
///
/// One head per launch, in a host loop over `NH` -- which is the other half of
/// why this is unrowable even ignoring the block: `head` is a launch-time
/// scalar the host varies across twelve fires, and no rule's `Dims` carries a
/// head INDEX.
template <class T>
__global__ void k_qk(const T* q, const T* k, float* s, int N, int H, int head, float scale) {
    int i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N) return;
    const T* qi = q + ((long)i * H + head) * 64;
    const T* kj = k + ((long)j * H + head) * 64;
    float a = 0;
    for (int d = 0; d < 64; d++) a += F(qi[d]) * F(kj[d]);
    s[(long)i * N + j] = a * scale;
}

/// In-place row softmax over an `[N, N]` float score matrix.
///
/// `Rule::PerRow`: `k_softmax<<<N, 256, 0, S>>>`, and `per_row` evaluates to
/// `grid[rows,1,1] block[256,1,1] smem 0` with the row stating `rows = N`.
/// The shared memory is STATIC (`__shared__ float wm[32], wsv[32], smx, ssum;`),
/// so zero dynamic bytes is the whole contract -- `Rule::Rms`'s 32 dynamic
/// bytes would allocate what nothing reads, which `per_row`'s own doc names as
/// the case it exists for.
///
/// The scores are FLOAT and stay float: this kernel neither reads nor writes
/// the storage format, so `T` never appears in its signature. It is a template
/// anyway, for the one reason every kernel in this header is one -- a
/// non-template `__global__` in a header included by three translation units
/// does not link.
template <class T>
__global__ void k_softmax(float* s, int N) {
    int i = blockIdx.x;
    if (i >= N) return;
    float* r = s + (long)i * N;
    float mx = -1e30f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) mx = fmaxf(mx, r[j]);
    for (int o = warpSize / 2; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
    __shared__ float wm[32], wsv[32], smx, ssum;
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = mx;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -1e30f;
        int nw = (blockDim.x + 31) / 32;
        for (int i2 = 0; i2 < nw; i2++) m = fmaxf(m, wm[i2]);
        smx = m;
    }
    __syncthreads();
    float sm = 0;
    for (int j = threadIdx.x; j < N; j += blockDim.x) { float e = __expf(r[j] - smx); r[j] = e; sm += e; }
    for (int o = warpSize / 2; o > 0; o >>= 1) sm += __shfl_down_sync(0xffffffff, sm, o);
    if ((threadIdx.x & 31) == 0) wsv[threadIdx.x >> 5] = sm;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        int nw = (blockDim.x + 31) / 32;
        for (int i2 = 0; i2 < nw; i2++) t += wsv[i2];
        ssum = t;
    }
    __syncthreads();
    float inv = 1.f / ssum;
    for (int j = threadIdx.x; j < N; j += blockDim.x) r[j] *= inv;
}

/// One head's `softmax(QK^T) V` back into the `[N, H, 64]` activation.
///
/// UNROWED: `k_av<<<G2(64,N), B2, 0, S>>>`, `B2 = dim3(16,16)`. 2-D block,
/// and the same per-head launch loop as `k_qk`.
template <class T>
__global__ void k_av(const float* s, const T* v, T* o, int N, int H, int head) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || d >= 64) return;
    const float* sr = s + (long)n * N;
    float a = 0;
    for (int j = 0; j < N; j++) a += sr[j] * F(v[((long)j * H + head) * 64 + d]);
    o[((long)n * H + head) * 64 + d] = Bf<T>(a);
}

/// Tanh-approximation GeGLU: `gelu(g) * u`.
///
/// UNROWED, and uniquely so: this kernel is DEAD. Nothing launches it, so
/// there is no `<<<>>>` to check a rule against, and Rule 3 of the migration
/// says a row with no cited launcher is a guess. `run_gemma4_vision` calls
/// `kernels::mlp::geglu_tanh_bf16` where this used to fire -- a fused
/// launcher with its own row -- and the definition was left behind.
///
/// Kept rather than deleted because deleting device text is a separate change
/// with a separate blast radius: the audio tower and the old driver's copy of
/// this file both still reference the shape, and the migration's rule is that
/// a `.cu` loses no kernel in the split. It is instantiated by the unit probe,
/// so NVRTC compiles it and it cannot rot silently.
template <class T>
__global__ void k_gelu_mul(const T* g, const T* u, T* o, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < t) {
        float x = F(g[i]);
        float gl = 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        o[i] = Bf<T>(gl * F(u[i]));
    }
}

/// Scatter-accumulate each patch's hidden into its pooling group, scaled by
/// `1/k2`.
///
/// UNROWED: `k_pool<<<G2(Hd,N), B2, 0, S>>>`, `B2 = dim3(16,16)`. 2-D block.
///
/// The accumulator is FLOAT and the add is atomic because the group map `grp`
/// is data -- several patches land on one output row and the order they land
/// in is the scheduler's. That is also why `k_pool_finish` exists: the scale
/// and the narrowing cannot happen inside the accumulation.
template <class T>
__global__ void k_pool(const T* h, const int* grp, float* o, int N, int D, float k2) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || d >= D) return;
    atomicAdd(&o[(long)grp[n] * D + d], F(h[(long)n * D + d]) / k2);
}

/// Scale the float pooling accumulator and narrow it to the storage format.
///
/// `Rule::Elementwise`: `k_pool_finish<<<((long)OUTL*Hd+255)/256, 256, 0, S>>>`
/// with `t = (long)OUTL*Hd`, and `elementwise` evaluates `rows * width` -- the
/// row states `rows = OUTL`, `width = Hd` -- to the same `ceil(OUTL*Hd/256)`
/// blocks of 256.
///
/// `s` is `sqrtf((float)Hd)` computed on the HOST and passed as a scalar. That
/// is an operand and not an extent: no rule recovers it, and a row that
/// omitted it would bind seven bytes short.
template <class T>
__global__ void k_pool_finish(const float* in, T* o, float s, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < t) o[i] = Bf<T>(in[i] * s);
}

}  // namespace pie_cuda_driver::kernels::vision::device
