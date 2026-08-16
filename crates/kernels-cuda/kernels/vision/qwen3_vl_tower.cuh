//===-- vision/qwen3_vl_tower.cuh - the qwen3-vl tower's device text -----===//
//
// The eleven `__global__`s of the Qwen3-VL vision encoder, as templates over
// the storage format, in a namespace NVRTC can name.
//
// # Why this exists
//
// Same reason as the two gemma-4 headers next door, and the same one change
// solving the same two problems. The kernels were non-template `__global__`s
// inside `namespace pie_cuda_driver::model { namespace { ... } }`: anonymous,
// so `nvrtcAddNameExpression` has nothing to give NVRTC and the runtime cannot
// resolve a `CUfunction`; and non-template, so a header holding them would
// emit one strong definition per translation unit and break the link. A NAMED
// template answers both.
//
// `qwen3_vl_tower.cu` keeps the host half -- and this tower's host half is the
// biggest of the three, which is the other reason to split it. It holds a
// mutex-guarded bilinear pos-embed interpolation cache keyed on
// `(grid_h, grid_w)`, the `merge_reorder` permutation table, the flashinfer
// attention plan, the cuBLAS handle and the deepstack merger loop. None of it
// is device text and none of it can be carried into an NVRTC header set.
//
// It is a MOVE and not a copy; `tests/sources.rs::no_global_is_defined_twice`
// fails the build over the alternative, because `norm/altup_aux` shipped a
// release with two copies that had drifted and every test passing.
//
// # What is unchanged
//
// The arithmetic, line for line: `bf` became `T`, `F`/`Bf` became
// `Elem<T>::to_f32`/`from_f32`, and the flat element counts became `usize`
// from `long`. This tower is NOT yet parity-verified against
// `qwen3_vl_vision_parity_ref.py`, which makes preserving the arithmetic
// exactly more important rather than less: a first draft that changes under a
// migration cannot be checked against the dumps it was written for.
//
// # What is rowed, and what refused
//
// Four of the eleven, and the seven refusals split three ways -- which is why
// this file's accounting is worth reading and not just its rows:
//
//   k_bias            Rule::Elementwise
//   k_add_pe          Rule::Elementwise
//   k_gelu_tanh       Rule::Elementwise
//   k_gelu_bias       Rule::Elementwise
//   k_split_rope_qkv  Unstated -- grid is PerHead's, block is not; the
//                     tower's Rust states [NH,N,1] x [HEAD/2,1,1]
//   k_merge_gather    Rule::Tile16
//   k_add_inplace     refused -- DEAD; no launch to cite
//   k_split_qkv       refused -- DEAD; no launch to cite
//   k_split_qkv_bias  refused -- DEAD; no launch to cite
//   k_rope_vis        refused -- DEAD; no launch to cite
//   k_rope_qk         refused -- DEAD; no launch to cite
//
// FIVE of the eleven are dead, which is the largest single finding of this
// migration and belongs here rather than in a commit message. They were
// superseded in place: `k_split_rope_qkv` fused `k_split_qkv_bias` and
// `k_rope_qk` into one pass and both survivors were left behind;
// `k_split_qkv` and `k_rope_vis` are the un-fused ancestors of those two; and
// `k_add_inplace` is byte-for-byte `k_add` in `tower_naive_kernels.cuh` under
// a different name, which is exactly why `no_global_is_defined_twice` -- which
// compares NAMES -- never saw it. They are kept, not deleted: removing device
// text is its own change with its own blast radius, and the unit probe
// instantiates every one of them so NVRTC compiles them and none can rot
// silently.
//
//===---------------------------------------------------------------------===//
#pragma once

#include "vision/tower_naive_kernels.cuh"

namespace pie::vision {

/// Add `b[col]` to `y[m, col]` -- the GEMM epilogue the naive `k_matmul`
/// folded in and cuBLAS does not.
///
/// `Rule::Elementwise`: fired three times, all
/// `k_bias<<<((long)M*O+255)/256, 256, 0, S>>>` -- once in `gemm_bias` for
/// every projection with a bias, once after o_proj and once after fc2 --
/// and `elementwise` evaluates `rows * width` to the same `ceil(M*O/256)`
/// blocks of 256 with no dynamic shared memory. One row covers all three;
/// they differ only in the rectangle.
///
/// `N` stays `int` and `M` widens to `usize` because the modulo is
/// per-COLUMN: `i % N` on a 64-bit `i` against an `int` N is what the
/// original computed, and narrowing `i` first would wrap on a tower whose
/// token count times hidden crosses 2^31. `usize` rather than the `long` the
/// original wrote, for the reason every flat count in this family carries it:
/// `Ty::Usize` is `std::size_t` and `usize` exactly, where `Ty::I64`
/// is `long long` and mangles as a third type on LP64.
///
/// The widening also repairs the guard. The original opened
/// `long i = blockIdx.x * blockDim.x + threadIdx.x;` — an `unsigned * unsigned`
/// product that wraps at 2^32 BEFORE the assignment widens it, so the `long M`
/// it was guarding against never protected anything. Casting `blockDim.x`
/// first is what the other twelve flat kernels here do and what this now does.
template <class T>
__global__ void k_bias(T* y, const T* b, usize M, int N) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i >= M * (usize)N) return;
    y[i] = Bf<T>(F(y[i]) + F(b[i % (usize)N]));
}

/// In-place `h[i] += x[i]`.
///
/// UNROWED, and for the one reason that is not about geometry: this kernel is
/// DEAD. Nothing launches it, so there is no `<<<>>>` to check a rule against,
/// and Rule 3 says a row with no cited launcher is a guess.
///
/// It is byte-for-byte `k_add` in `vision/tower_naive_kernels.cuh` -- and
/// byte-for-byte `pie::norm::residual_add`, and byte-for-byte
/// [`k_add_pe`] below. Four copies of `a[i] += b[i]`, and
/// `no_global_is_defined_twice` sees none of them, because it compares
/// namespace-qualified NAMES and all four are spelled differently. That is the
/// limit of a name-based duplicate check, recorded here because the next
/// person to widen that test should know what it will find.
template <class T>
__global__ void k_add_inplace(T* h, const T* x, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < t) h[i] = Bf<T>(F(h[i]) + F(x[i]));
}

/// Add the precomputed interpolated absolute position embedding
/// `pe[n_patch, D]` into `h`.
///
/// `Rule::Elementwise`: `k_add_pe<<<((long)N*Hd+255)/256, 256, 0, S>>>`, and
/// `elementwise` evaluates `rows * width` to the same `ceil(N*Hd/256)` blocks
/// of 256.
///
/// The BILINEAR INTERPOLATION that produced `pe` is not here and cannot be: it
/// runs on the host, cached under a mutex keyed on `(grid_h, grid_w)`, because
/// the table is per-image-shape and every image of a given shape reuses it.
/// See `qwen3_vl_tower.cu`. This kernel is the four lines that are left.
template <class T>
__global__ void k_add_pe(T* h, const T* pe, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < t) h[i] = Bf<T>(F(h[i]) + F(pe[i]));
}

/// Plain non-gated `gelu_pytorch_tanh`:
/// `o = 0.5*x*(1+tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`.
///
/// `Rule::Elementwise`: `k_gelu_tanh<<<((long)N*Dmid+255)/256, 256, 0, S>>>`
/// in `mlp()`, and `elementwise` evaluates `rows * width` to the same
/// `ceil(N*Dmid/256)` blocks of 256.
///
/// The `erf_gelu` flag in `mlp()` picks between this and
/// `tower_naive_kernels.cuh`'s `k_gelu_erf` at the CALL, not inside a kernel:
/// the ViT blocks want the tanh approximation and the patch mergers want
/// `nn.GELU(approximate='none')`. Two kernels and two rows, because they are
/// two functions -- merging them by name is the mistake that header's comment
/// records.
template <class T>
__global__ void k_gelu_tanh(const T* x, T* o, usize t) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < t) { float v = F(x[i]); o[i] = Bf<T>(0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)))); }
}

/// Split fused QKV `[N, 3*hidden]` (row layout `q|k|v`) into three `[N, hidden]`.
///
/// UNROWED: DEAD. Nothing launches it. Its bias-adding successor
/// [`k_split_qkv_bias`] is dead too, and the successor to that,
/// [`k_split_rope_qkv`], is the one the tower fires. Had it a launch it would
/// still be refused -- `G2(H,N)` over `B2 = dim3(16,16)`, a 2-D block.
template <class T>
__global__ void k_split_qkv(const T* qkv, T* q, T* k, T* v, int N, int H) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || d >= H) return;
    const T* r = qkv + (long)n * 3 * H;
    q[(long)n * H + d] = r[d];
    k[(long)n * H + d] = r[H + d];
    v[(long)n * H + d] = r[2 * H + d];
}

/// Split fused QKV AND add the per-section bias in one pass.
///
/// UNROWED: DEAD, superseded by [`k_split_rope_qkv`], which folded the rope in
/// as well. Same 2-D block if it were fired.
template <class T>
__global__ void k_split_qkv_bias(const T* qkv, const T* b, T* q, T* k, T* v, int N, int H) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || d >= H) return;
    const T* r = qkv + (long)n * 3 * H;
    float bq = b ? F(b[d]) : 0.f, bk = b ? F(b[H + d]) : 0.f, bv = b ? F(b[2 * H + d]) : 0.f;
    q[(long)n * H + d] = Bf<T>(F(r[d]) + bq);
    k[(long)n * H + d] = Bf<T>(F(r[H + d]) + bk);
    v[(long)n * H + d] = Bf<T>(F(r[2 * H + d]) + bv);
}

/// `gelu_pytorch_tanh` with a fused per-column bias add -- fc1's bias kernel
/// folded into the activation.
///
/// `Rule::Elementwise`: `k_gelu_bias<<<((long)N*IM+255)/256, 256, 0, S>>>`,
/// and `elementwise` evaluates `rows * width` to the same `ceil(N*IM/256)`
/// blocks of 256.
///
/// It computes its own `t = (long)N*D` from two operands rather than taking a
/// count, which is what makes both `N` and `D` OPERANDS here where the flat
/// kernels take one `usize`. The rule recovers the same product from
/// `rows * width` and the kernel recovers it from its arguments; they agree,
/// and neither is the other's source.
template <class T>
__global__ void k_gelu_bias(T* x, const T* b, int N, int D) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long t = (long)N * D;
    if (i >= t) return;
    float v = F(x[i]) + (b ? F(b[i % D]) : 0.f);
    x[i] = Bf<T>(0.5f * v * (1.f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v))));
}

/// ViT 2-D RoPE over one tensor, transformers' `rotate_half` layout.
///
/// UNROWED: DEAD, superseded by [`k_rope_qk`] and then by
/// [`k_split_rope_qkv`]. Its launch, when it had one, was
/// `dim3(half/32, NH, N)` over 32 threads -- a 3-D grid, which no rule emits.
///
/// The layout, kept because [`k_split_rope_qkv`] implements it and this is
/// where it is written down: the rotary table is built from `position_ids`
/// `[N,2] = (row, col)`; rope dim is `head_dim/2 = 32`, `inv_freq` has
/// `head_dim/4 = 16` entries, and `rotary_pos_emb = (pos[...,None]*inv_freq)
/// .flatten(1)` gives per token `[row*invf(0..15), col*invf(0..15)]`. Then
/// `emb = cat(rope, rope)` makes cos/sin length 64, and attention applies
/// `q*cos + rotate_half(q)*sin` over the full `head_dim`, `rotate_half`
/// splitting at 32. So pair index `j` in `[0,16)` rotates `(q[j], q[j+32])` by
/// `row*invf[j]`, and `j` in `[16,32)` by `col*invf[j-16]`.
///
/// PARITY TODO carried over: confirm the `(j, j+half)` pairing and the
/// `[0:quarter)=row, [quarter:half)=col` split against transformers'
/// `apply_rotary_pos_emb_vision` and `Qwen3VLVisionRotaryEmbedding(head_dim//2)`.
template <class T>
__global__ void k_rope_vis(T* q, const float* pos, int N, int NH, int HEAD, float theta) {
    int n = blockIdx.z, head = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    int half = HEAD / 2, quarter = HEAD / 4;
    if (n >= N || head >= NH || j >= half) return;
    T* v = q + (((long)n * NH + head) * HEAD);
    float row = pos[2L * n], col = pos[2L * n + 1];
    int c = (j < quarter) ? j : (j - quarter);
    float coord = (j < quarter) ? row : col;
    float invf = powf(theta, -2.f * (float)c / (float)half);
    float ang = coord * invf, cs = cosf(ang), sn = sinf(ang);
    float a = F(v[j]), b = F(v[j + half]);
    v[j] = Bf<T>(a * cs - b * sn);
    v[j + half] = Bf<T>(b * cs + a * sn);
}

/// The same 2-D RoPE applied to BOTH q and k in one launch.
///
/// UNROWED: DEAD, superseded by [`k_split_rope_qkv`]. Same 3-D grid as
/// [`k_rope_vis`] if it were fired.
template <class T>
__global__ void k_rope_qk(T* q, T* k, const float* pos, int N, int NH, int HEAD, float theta) {
    int n = blockIdx.z, head = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    int half = HEAD / 2, quarter = HEAD / 4;
    if (n >= N || head >= NH || j >= half) return;
    float row = pos[2L * n], col = pos[2L * n + 1];
    int c = (j < quarter) ? j : (j - quarter);
    float coord = (j < quarter) ? row : col;
    float invf = powf(theta, -2.f * (float)c / (float)half);
    float ang = coord * invf, cs = cosf(ang), sn = sinf(ang);
    long base = ((long)n * NH + head) * HEAD;
    T* vq = q + base;
    float aq = F(vq[j]), bq = F(vq[j + half]);
    vq[j] = Bf<T>(aq * cs - bq * sn);
    vq[j + half] = Bf<T>(bq * cs + aq * sn);
    T* vk = k + base;
    float ak = F(vk[j]), bk = F(vk[j + half]);
    vk[j] = Bf<T>(ak * cs - bk * sn);
    vk[j + half] = Bf<T>(bk * cs + ak * sn);
}

/// Fused split-QKV + bias + 2-D RoPE: read fused `qkv[N, 3H]` once, add the
/// per-section bias, write `q,k,v[N,H]` with q and k already rotated.
///
/// ROWED `LaunchRule::Unstated`, and it was the closest call in the family.
/// The launch is
/// `k_split_rope_qkv<<<dim3(NH,N), HEAD/2, 0, S>>>`. The GRID is exactly
/// `Rule::PerHead`'s `[heads, rows, 1]` and the kernel reads `blockIdx.x` as
/// the head and `blockIdx.y` as the row, which is `per_head`'s own convention.
/// The BLOCK is not: `per_head` fixes `PAD_BLOCK = 128` and this launches
/// `HEAD/2`, which is 32 for qwen3-vl's `head_dim = 64`. Rule 3 requires the
/// evaluated launch and the `<<<>>>` to AGREE, and `[64,1,1] x [128,1,1]`
/// against `[64,1,1] x [32,1,1]` does not.
///
/// It is not a rounding difference either. The loop is
/// `for (j = threadIdx.x; j < half; j += blockDim.x)`, so 128 threads over a
/// 32-wide half means 96 idle lanes and the same result -- correct, and four
/// times the launch. Widening it is a performance decision for the tower's
/// owner and not something a migration may take silently.
///
/// So the row states NO rule and the tower's Rust states the geometry, which
/// leaves that decision exactly where it was. "No rule states this grid" and
/// "this cannot be a row" were always different claims; only the first was
/// ever measured here, and the second rested on an AOT dispatcher being the
/// only thing that could fire a row. A driver states its own grids --
/// `attn::attn_score_fold_heads` and `families::attn`'s `ATTN_SCORE_POST`
/// are the precedents -- so the paragraphs above are now the row's
/// rationale rather than its absence. See `families::vision::QWEN3_VL_ROWS`.
template <class T>
__global__ void k_split_rope_qkv(const T* qkv, const T* b, T* q, T* k, T* v,
                                 const float* pos, int N, int NH, int HEAD, float theta) {
    int n = blockIdx.y, head = blockIdx.x;
    if (n >= N || head >= NH) return;
    const int H = NH * HEAD, half = HEAD / 2, quarter = HEAD / 4;
    const T* qr = qkv + (long)n * 3 * H + head * HEAD;  // q section for this head
    const T* kr = qr + H;                                // k section
    const T* vr = kr + H;                                // v section
    const T* bq = b ? b + head * HEAD : nullptr;
    const T* bk = b ? b + H + head * HEAD : nullptr;
    const T* bv = b ? b + 2 * H + head * HEAD : nullptr;
    long o = ((long)n * NH + head) * HEAD;
    T* qo = q + o;
    T* ko = k + o;
    T* vo = v + o;
    float row = pos[2L * n], col = pos[2L * n + 1];
    for (int j = threadIdx.x; j < half; j += blockDim.x) {
        int c = (j < quarter) ? j : (j - quarter);
        float coord = (j < quarter) ? row : col;
        float invf = powf(theta, -2.f * (float)c / (float)half);
        float ang = coord * invf, cs = cosf(ang), sn = sinf(ang);
        float q0 = F(qr[j]) + (bq ? F(bq[j]) : 0.f), q1 = F(qr[j + half]) + (bq ? F(bq[j + half]) : 0.f);
        qo[j] = Bf<T>(q0 * cs - q1 * sn);
        qo[j + half] = Bf<T>(q1 * cs + q0 * sn);
        float k0 = F(kr[j]) + (bk ? F(bk[j]) : 0.f), k1 = F(kr[j + half]) + (bk ? F(bk[j + half]) : 0.f);
        ko[j] = Bf<T>(k0 * cs - k1 * sn);
        ko[j + half] = Bf<T>(k1 * cs + k0 * sn);
        vo[j] = Bf<T>(F(vr[j]) + (bv ? F(bv[j]) : 0.f));
        vo[j + half] = Bf<T>(F(vr[j + half]) + (bv ? F(bv[j + half]) : 0.f));
    }
}

/// The 2x2 spatial-merge gather: `g[tok, u*C + c] = h[tok*U + u, c]`.
///
/// UNROWED: `k_merge_gather<<<G2(W,n_token), B2, 0, S>>>` at both merger call
/// sites, `B2 = dim3(16,16)`. 2-D block, and `threadIdx.y` is read.
///
/// The input is already in spatial-merge order -- every `merge^2` consecutive
/// patch rows form one output token -- because the HOST reordered it. That is
/// `merge_reorder` in the `.cu`, and it is why a plain concatenation suffices
/// here where HF needs a five-way reshape.
///
/// PARITY TODO carried over: confirm the within-group order `u = 0..U-1`
/// matches HF's `(h//m, m, w//m, m, C) -> (h//m * w//m, m*m, C)`.
template <class T>
__global__ void k_merge_gather(const T* h, T* g, int n_token, int U, int C) {
    int tok = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    int W = U * C;
    if (tok >= n_token || d >= W) return;
    int u = d / C, c = d % C;
    g[(long)tok * W + d] = h[((long)tok * U + u) * C + c];
}

}  // namespace pie::vision
