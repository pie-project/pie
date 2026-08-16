//===-- vision/gemma4_audio.cuh - the gemma-4 audio tower's device text --===//
//
// The twelve `__global__`s of the Gemma-4 audio encoder, as templates over the
// storage format, in a namespace NVRTC can name.
//
// # Why this exists
//
// Same reason as `vision/gemma4_vision.cuh`, and the same two problems solved
// by the same one change. The kernels were non-template `__global__`s inside
// `namespace pie_cuda_driver::model { namespace { ... } }`: anonymous, so
// `nvrtcAddNameExpression` has no name to give NVRTC and the runtime cannot
// resolve a `CUfunction`; and non-template, so putting them in a header would
// emit one strong definition per translation unit and break the ahead-of-time
// link on the duplicates. A NAMED template answers both.
//
// `gemma4_audio.cu` keeps the host half -- `run_gemma4_audio`,
// `scatter_gemma4_audio`, `encode_gemma4_audio`, the checkpoint hook, the
// scratch arena, the conformer loop -- and `#include`s this. It is a MOVE and
// not a copy; `tests/sources.rs::no_global_is_defined_twice` fails the build
// over the alternative, because `norm/altup_aux` shipped a release with two
// copies that had drifted and every test passing.
//
// # What is unchanged
//
// The arithmetic, line for line: `bf` became `T`, `F`/`Bf` became
// `Elem<T>::to_f32`/`from_f32`, and the flat element counts became `usize`
// from `long` (same width here, and the only one a row can state). The tower
// is parity-checked against `gemma4_audio_parity_ref.py` at cosine 0.99997 /
// rel_rms 0.744% on 188 mel frames, and that number belongs to THIS
// arithmetic. `k_local_attn`'s two-pass online softmax in particular is a
// fold order, not an implementation detail.
//
// # What is rowed, and what refused
//
// Two of the twelve, and the audio tower is the clearest case in the family
// for why `LaunchRule` is a vocabulary and not a constructor. Nine of the ten
// refusals are the same shape: this tower indexes its rectangles with
// `dim3 B2(16,16)` and `G2(X,Y) = dim3((X+15)/16, (Y+15)/16)`, and three of
// those also carry the channel on `gridDim.z`. Every ported rule states
// `block = [n,1,1]`. The tenth and eleventh are each their own kind of
// host-side choice. Each kernel below carries its `<<<>>>` and its reason,
// per `new-horizon.md` §17.9.
//
//   k_silu             Rule::Elementwise
//   k_axpy             Rule::Elementwise
//   k_matmul_bias      refused -- 2-D block
//   k_glu              refused -- 2-D block
//   k_layernorm_relu   refused -- one block per row at 128, not 256
//   k_conv2d_s2        refused -- 3-D grid AND 2-D block
//   k_chlast           refused -- 3-D grid AND 2-D block
//   k_chfirst          refused -- 3-D grid AND 2-D block
//   k_sscp_flatten     refused -- 2-D block
//   k_qkv_scale        refused -- 2-D block
//   k_rel_pos_enc      refused -- 2-D block
//   k_local_attn       refused -- grid.x is a tile count, not a row count
//
// The thirteenth kernel this file used to hold is not refused -- it is gone.
// `k_depthwise_causal` was a local copy of
// `kernels::ssm::causal_conv1d_prefill_noact_bf16`, bit for bit the same
// accumulation in the same order, and the call site now says so.
//
//===---------------------------------------------------------------------===//
#pragma once

#include "vision/gemma4_naive_kernels.cuh"

namespace pie::vision {

/// Scalar row-major `y = x * W^T + b`, one thread per output element, `b`
/// OPTIONAL.
///
/// UNROWED: `k_matmul_bias<<<G2(OPD,N), B2, 0, S>>>` where `B2 = dim3(16,16)`.
/// 2-D block, and `threadIdx.y` is read.
///
/// The bias-free twin is `k_matmul` in `tower_naive_kernels.cuh` and this does
/// not call it: a `b ? F(b[o]) : 0.f` initialiser is the whole difference, and
/// merging them would put a branch in the inner loop of the tower's hottest
/// naive kernel to save four lines.
template <class T>
__global__ void k_matmul_bias(const T* x, const T* W, const T* b, T* y, int N, int K, int O) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, o = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || o >= O) return;
    const T* xr = x + (long)n * K;
    const T* wr = W + (long)o * K;
    float a = b ? F(b[o]) : 0.f;
    for (int k = 0; k < K; k++) a += F(xr[k]) * F(wr[k]);
    y[(long)n * O + o] = Bf<T>(a);
}

/// SiLU: `o = x * sigmoid(x)`.
///
/// `Rule::Elementwise`: fired twice, `k_silu<<<((long)N*IM+255)/256, 256, 0, S>>>`
/// in the feed-forward and `k_silu<<<((long)N*Hd+255)/256, 256, 0, S>>>` after
/// the conv module. `elementwise` evaluates `rows * width` to the same
/// `ceil(n/256)` blocks of 256 with no dynamic shared memory, so ONE row
/// covers both fires -- the two call sites differ only in the rectangle they
/// hand it, which is what a rule is for.
///
/// `__expf` and not `expf`: the fast intrinsic, which is what the parity run
/// measured.
template <class T>
__global__ void k_silu(const T* x, T* o, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < t) { float v = F(x[i]); o[i] = Bf<T>(v / (1.f + __expf(-v))); }
}

/// `a += scale * b` -- the macaron half-step residual.
///
/// `Rule::Elementwise`: `k_axpy<<<((long)N*Hd+255)/256, 256, 0, S>>>`, and
/// `elementwise` evaluates `rows * width` to the same `ceil(N*Hd/256)` blocks
/// of 256.
///
/// `scale` is `w.residual_weight` read off the checkpoint on the host. It is
/// an OPERAND, not an extent -- no rule recovers it, and a row that left it
/// out would bind one argument short of the kernel's arity.
template <class T>
__global__ void k_axpy(T* a, const T* b, float scale, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < t) a[i] = Bf<T>(F(a[i]) + scale * F(b[i]));
}

/// GLU over the last dim: `o[n,d] = x[n,d] * sigmoid(x[n, d+D])`.
///
/// UNROWED: `k_glu<<<G2(Hd,N), B2, 0, S>>>`, `B2 = dim3(16,16)`. 2-D block.
///
/// `Rule::SplitPacked` is the rule for a packed-in, split-out pointwise and it
/// is NOT this one: it states `in_width` on a 1-D block over a flat grid, and
/// this reads `threadIdx.y`.
template <class T>
__global__ void k_glu(const T* x, T* o, int N, int D) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || d >= D) return;
    float a = F(x[(long)n * 2 * D + d]), g = F(x[(long)n * 2 * D + D + d]);
    o[(long)n * D + d] = Bf<T>(a / (1.f + __expf(-g)));
}

/// LayerNorm over the channel axis (no bias, OPTIONAL learnable scale) then
/// ReLU, over a `[rows, C]` tensor whose row is one `(t, f)` cell's channel
/// vector.
///
/// UNROWED: `k_layernorm_relu<<<T1*F1, 128, 0, S>>>` at both SSCP call sites.
/// The GRID is `Rule::PerRow`'s and the BLOCK is not -- `per_row` fixes 256
/// and this launches 128 -- and the two have to agree for the row to be true.
/// Widening the launch to 256 would state the rule, and it is a numerics
/// change and not a spelling one: the fold below sums `(blockDim.x+31)/32`
/// per-warp partials serially in thread 0, so 128 threads and 256 threads add
/// the same values in a different order and answer with a different last bit.
/// That is the `rmsnorm_residual_add_scale_rmsnorm` precedent, and it needs
/// the audio parity harness rather than a row.
template <class T>
__global__ void k_layernorm_relu(const T* x, const T* w, T* o, int R, int C, float eps) {
    int r = blockIdx.x;
    if (r >= R) return;
    const T* xr = x + (long)r * C;
    T* orow = o + (long)r * C;
    float m = 0;
    for (int c = threadIdx.x; c < C; c += blockDim.x) m += F(xr[c]);
    for (int s = warpSize / 2; s > 0; s >>= 1) m += __shfl_down_sync(0xffffffff, m, s);
    __shared__ float wm[32], wv[32], mean, inv;
    if ((threadIdx.x & 31) == 0) wm[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; i++) t += wm[i];
        mean = t / C;
    }
    __syncthreads();
    float v = 0;
    for (int c = threadIdx.x; c < C; c += blockDim.x) { float d = F(xr[c]) - mean; v += d * d; }
    for (int s = warpSize / 2; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffff, v, s);
    if ((threadIdx.x & 31) == 0) wv[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; i++) t += wv[i];
        inv = rsqrtf(t / C + eps);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float y = (F(xr[c]) - mean) * inv * (w ? F(w[c]) : 1.f);
        orow[c] = Bf<T>(y > 0.f ? y : 0.f);
    }
}

/// `Conv2d(in_ch, out_ch, k=3, s=2, p=1)` over an `[IC, Tin, Fin]` feature map.
///
/// UNROWED: `dim3 g((F1+15)/16, (T1+15)/16, C0); k_conv2d_s2<<<g, B2, 0, S>>>`
/// at both SSCP call sites. A 3-D grid AND a 2-D block, and the kernel reads
/// `blockIdx.z` for the output channel. Nothing in the vocabulary emits either.
///
/// PARITY TODO carried over from the original: verify padding=1 + stride=2
/// indexing against torch `Conv2d`. The shape math (`To = (Tin-1)/2+1`) is
/// checked; the corner handling is not.
template <class T>
__global__ void k_conv2d_s2(const T* in, const T* W, T* out,
                            int IC, int Tin, int Fin, int OC, int To, int Fo) {
    int oc = blockIdx.z;
    int to = blockIdx.y * blockDim.y + threadIdx.y, fo = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= OC || to >= To || fo >= Fo) return;
    float acc = 0;
    for (int ic = 0; ic < IC; ic++) {
        const T* wk = W + (((long)oc * IC + ic) * 3) * 3;  // [3,3]
        for (int kt = 0; kt < 3; kt++) for (int kf = 0; kf < 3; kf++) {
            int ti = to * 2 + kt - 1, fi = fo * 2 + kf - 1;  // stride 2, pad 1
            if (ti < 0 || ti >= Tin || fi < 0 || fi >= Fin) continue;
            acc += F(in[((long)ic * Tin + ti) * Fin + fi]) * F(wk[kt * 3 + kf]);
        }
    }
    out[((long)oc * To + to) * Fo + fo] = Bf<T>(acc);
}

/// `[OC, To, Fo]` to `[To*Fo, OC]` -- channels-last, so the LayerNorm above
/// runs over the channel axis.
///
/// UNROWED: `dim3 g((F1+15)/16, (T1+15)/16, C0); k_chlast<<<g, B2, 0, S>>>` at
/// both call sites. 3-D grid, 2-D block.
///
/// A pure permutation: it never widens, so `T` appears only as the element
/// this moves and the arithmetic is address arithmetic.
template <class T>
__global__ void k_chlast(const T* in, T* out, int OC, int To, int Fo) {
    int oc = blockIdx.z;
    int to = blockIdx.y * blockDim.y + threadIdx.y, fo = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= OC || to >= To || fo >= Fo) return;
    out[(((long)to * Fo + fo) * OC) + oc] = in[((long)oc * To + to) * Fo + fo];
}

/// The inverse of [`k_chlast`]: `[To*Fo, OC]` back to `[OC, To, Fo]`.
///
/// UNROWED: `dim3 g((F1+15)/16, (T1+15)/16, C0); k_chfirst<<<g, B2, 0, S>>>`
/// at both call sites. 3-D grid, 2-D block.
template <class T>
__global__ void k_chfirst(const T* in, T* out, int OC, int To, int Fo) {
    int oc = blockIdx.z;
    int to = blockIdx.y * blockDim.y + threadIdx.y, fo = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= OC || to >= To || fo >= Fo) return;
    out[((long)oc * To + to) * Fo + fo] = in[(((long)to * Fo + fo) * OC) + oc];
}

/// Flatten the final SSCP map `[OC, To, Fo]` to `[To, Fo*OC]` for
/// `input_proj_linear` -- HF's `permute(0,2,3,1).reshape(B, To, Fo*OC)`.
///
/// UNROWED: `dim3 g((FLAT+15)/16, (N+15)/16); k_sscp_flatten<<<g, B2, 0, S>>>`.
/// 2-D block.
template <class T>
__global__ void k_sscp_flatten(const T* in, T* out, int OC, int To, int Fo) {
    int to = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    int FoOC = Fo * OC;
    if (to >= To || j >= FoOC) return;
    int fo = j / OC, oc = j % OC;
    out[(long)to * FoOC + j] = in[((long)oc * To + to) * Fo + fo];
}

/// In-place pre-scale of Q and K: `q *= q_scale * softplus(per_dim_scale)`,
/// `k *= k_scale`.
///
/// UNROWED: `k_qkv_scale<<<G2(Hd,N), B2, 0, S>>>`, `B2 = dim3(16,16)`. 2-D
/// block.
///
/// Both scales are host constants -- `q_scale = hd^-0.5 / ln2`,
/// `k_scale = ln(1+e) / ln2` -- computed once per encode and passed as
/// operands. They look like extents and are not: nothing about the rectangle
/// determines them.
template <class T>
__global__ void k_qkv_scale(T* q, T* k, const T* pds, int N, int H, int hd,
                            float q_scale, float k_scale) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, e = blockIdx.x * blockDim.x + threadIdx.x;
    int HD = H * hd;
    if (n >= N || e >= HD) return;
    int d = e % hd;
    float sp = logf(1.f + expf(F(pds[d])));  // softplus(per_dim_scale)
    q[(long)n * HD + e] = Bf<T>(F(q[(long)n * HD + e]) * q_scale * sp);
    k[(long)n * HD + e] = Bf<T>(F(k[(long)n * HD + e]) * k_scale);
}

/// The sinusoidal relative-position encoding `pe[P, hidden]`, `P = max_past+1`.
///
/// UNROWED: `dim3 g((Hd+15)/16, (P+15)/16); k_rel_pos_enc<<<g, B2, 0, S>>>`.
/// 2-D block.
///
/// `position_ids = arange(max_past, -1, -1)`, so row `r` holds position id
/// `(P-1) - r` and `pe[r] = concat(sin(scaled_time[r]), cos(scaled_time[r]))`.
/// The `log_inc` is written as `logf(10000.f/1.f)` rather than folded to a
/// constant because that is HF's expression and the fold is not exact.
template <class T>
__global__ void k_rel_pos_enc(T* pe, int P, int hidden) {
    int r = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= P || d >= hidden) return;
    int num_ts = hidden / 2;
    float log_inc = logf(10000.f / 1.f) / fmaxf((float)(num_ts - 1), 1.f);
    int m = d < num_ts ? d : (d - num_ts);
    float inv = expf((float)m * -log_inc);
    float pos = (float)((P - 1) - r);  // position_id = max_past - r
    float t = pos * inv;
    pe[(long)r * hidden + d] = Bf<T>(d < num_ts ? sinf(t) : cosf(t));
}

/// Exact O(N^2) causal-sliding-window attention with relative-position bias
/// and logit soft-cap. `q`, `k` arrive pre-scaled from [`k_qkv_scale`].
///
/// UNROWED: `dim3 g((N+127)/128, NH); k_local_attn<<<g, 128, 0, S>>>`. The
/// block IS `PAD_BLOCK`, and the grid is still not any rule's. `per_head`
/// emits `[heads, rows, 1]`; this is `[ceil(rows/128), heads, 1]` -- the axes
/// are the other way round AND `grid.x` is a TILE count where every ported
/// rule's leading axis is a count of things. `elementwise_rows` gets the
/// tiling right and the block width and the axis order wrong. A row here
/// would be a rule invented to fit one launcher, which §17.9 is explicit is
/// worse than the refusal.
///
/// `float acc[256]` is a per-thread local array -- 1 KiB of local memory a
/// thread, which is why the launch is 128 wide and not 256. That is a host
/// occupancy decision wearing a block width, and it is the second reason no
/// rule states this.
///
/// The masking: HF's blocked 5-D path (chunk 12 / past 12 / future 0) plus
/// `_rel_shift` collapses, for this config, to a plain causal sliding window
/// -- query `t` attends keys `j` with `0 <= t-j < max_past` -- and the
/// rel_shift gather collapses to `matrix_bd[t,j]` reading pe row
/// `(P-1)-(t-j)`. Verified flat-vs-blocked to <1e-6 abs.
template <class T>
__global__ void k_local_attn(const T* q, const T* k, const T* v,
                             const T* relk, T* out,
                             int N, int H, int hd, int P, float cap) {
    int head = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (head >= H || i >= N) return;
    float acc[256];  // hd <= 256 (gemma4: 128)
    for (int d = 0; d < hd; d++) acc[d] = 0.f;
    // mask: query t attends keys j with 0 <= (t-j) < max_past (= P-1, no
    // future). So distance is in [0, P-2]; lo = i-(P-2).
    int lo = i - (P - 2);
    if (lo < 0) lo = 0;
    const T* qr = q + ((long)i * H + head) * hd;
    float mx = -1e30f;
    for (int j = lo; j <= i; j++) {
        const T* kr = k + ((long)j * H + head) * hd;
        const T* rr = relk + ((long)((P - 1) - (i - j)) * H + head) * hd;
        float s = 0;
        for (int d = 0; d < hd; d++) s += F(qr[d]) * (F(kr[d]) + F(rr[d]));
        s = cap * tanhf(s / cap);  // logit soft-cap
        mx = fmaxf(mx, s);
    }
    float denom = 0;
    for (int j = lo; j <= i; j++) {
        const T* kr = k + ((long)j * H + head) * hd;
        const T* rr = relk + ((long)((P - 1) - (i - j)) * H + head) * hd;
        float s = 0;
        for (int d = 0; d < hd; d++) s += F(qr[d]) * (F(kr[d]) + F(rr[d]));
        s = cap * tanhf(s / cap);
        float w = __expf(s - mx);
        denom += w;
        const T* vr = v + ((long)j * H + head) * hd;
        for (int d = 0; d < hd; d++) acc[d] += w * F(vr[d]);
    }
    float inv = denom > 0.f ? 1.f / denom : 0.f;
    for (int d = 0; d < hd; d++) out[((long)i * H + head) * hd + d] = Bf<T>(acc[d] * inv);
}

}  // namespace pie::vision
