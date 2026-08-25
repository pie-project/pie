// Kimi Delta Attention: the per-channel-gated delta recurrence, prologue and
// scan fused into one launch.
//
// Two entrypoints, one per point. `kda_step` walks ONE token per workgroup and
// `kda_chunked` walks a request's whole CSR window. Same recurrence written
// twice, kept apart for `ssm/gated_delta.wgsl`'s reason: the chunk's barriers
// are the loop's and the step has no loop to put them in.
//
// ── WHAT MAKES THIS NOT `gated_delta` ─────────────────────────────────────
//
// The decay. `ssm.gated_delta` has ONE scalar `g` per (token, head) and scales
// every cell of the head's state by it. KDA has a decay PER KEY CHANNEL --
// `sg[i]`, a whole `head_dim` vector -- so the state's rows fade at different
// rates. That is the whole architectural difference and it is why the two
// files are two files: every other line rhymes, and a shared body would carry
// a `vec-or-scalar` branch through the innermost loop of both for the sake of
// deleting thirty lines.
//
// The projections differ with it. GDN takes q, k and v out of one packed
// post-convolution row and its gates out of a second packed row; KDA takes
// q, k and v out of a row that is three equal thirds, its forget gate out of a
// full `[N, heads * head_dim]` projection, and its beta out of a `[N, heads]`
// column, and it applies `a_log` and `dt_bias` here rather than in a prep pass.
//
// ── THE ARITHMETIC ────────────────────────────────────────────────────────
//
// `kda_qkv_prep` + `kda_gate_beta` + `kda_recurrent_step_batched` in
// `kernels-cuda/kernels/ssm/kda.cuh`, which is where the numeric contract was
// measured. Per (token t, head h), with `W = heads * head_dim`:
//
//     q_inv   = rsqrt( sum over the WHOLE q third of q^2 + norm_eps )
//     k_inv   = rsqrt( sum over the WHOLE k third of k^2 + norm_eps )
//     sq[i]   = mixed[t, 0*W + h*D + i] * q_inv
//     sk[i]   = mixed[t, 1*W + h*D + i] * k_inv
//     z       = f[t, h*D + i] + dt_bias[h*D + i]
//     sg[i]   = exp( -exp(a_log[h]) * softplus(z) )
//     beta    = sigmoid( b[t, h] )
//
//     S[i, c] *= sg[i]                        (the decay is on the KEY axis)
//     mem[c]   = sum_i S[i, c] * sk[i]
//     delta[c] = (mixed[t, 2*W + h*D + c] - mem[c]) * beta
//     S[i, c] += sk[i] * delta[c]
//     y[t, h*D + c] = sum_i S[i, c] * sq[i]
//
// ── THE L2 NORM SPANS THE WHOLE PLANE, NOT THE HEAD ───────────────────────
//
// `kda_qkv_prep` reduces over `width`, and the claim body passes
// `heads * head_dim` for it -- so `q_inv` is ONE number for the whole token,
// shared by every head, not a per-head norm. It reads like a transcription
// slip and it is not: it is what the reference computes, what `kernels-metal`
// ported, and what the tolerance was measured against. A per-head norm would
// be a different model.
//
// The cost of having no scratch allocator lands exactly here. `kernels-cuda`
// folds the plane once per token in a prep launch and hands the scan a
// normalised row; this kernel has nowhere to put such a row, so each of the
// `heads` workgroups folds the same plane again. That is `heads` times the
// q/k read traffic and none of the state traffic, which is the smaller half of
// the kernel by some margin -- and it is the honest price rather than a
// per-head norm that would be cheaper and wrong.
//
// ── THE STATE'S LAYOUT IS `[slot, head, head_dim, head_dim]`, K-MAJOR ─────
//
// `state[((slot * heads + h) * D + i) * D + c]`, key index outer and value
// channel inner, so that the invocations of a workgroup touch consecutive
// words at each step of the inner loop. `kernels-cuda` and `kernels-metal`
// both write the transpose; the slab is private to this family on every plane,
// and `driver-wgpu` sizes it as a flat product either reading fills.
//
// ── SOFTPLUS AND ITS BRANCH ───────────────────────────────────────────────
//
// `z > 20 ? z : log(1 + exp(z))`, the same spelling `ssm/gdn_gates.wgsl` uses
// and for the same reason: `exp(z)` overflows f32 near 88 while `log(1+exp(z))`
// is within a ULP of `z` far below that, so the branch is what keeps a large
// forget gate finite. cuda writes `log1pf`, which WGSL has not; the two differ
// only where `exp(z)` is below the epsilon of 1.

//#include "common/bf16.inc.wgsl"

// One workgroup per (head, token-or-request).
const PIE_WIDTH = 128u;

// The widest head this scan stages. Three `f32` rows of it is 3 KiB of a
// 16 KiB workgroup budget; the claim body refuses anything wider.
const PIE_DMAX = 256u;

@group(0) @binding(0) var<storage, read_write> mixed: array<u32>;
@group(0) @binding(1) var<storage, read_write> f_gate: array<u32>;
@group(0) @binding(2) var<storage, read_write> b_beta: array<u32>;
@group(0) @binding(3) var<storage, read_write> dt_bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> a_log: array<f32>;
@group(0) @binding(5) var<storage, read_write> rstate: array<f32>;
@group(0) @binding(6) var<storage, read_write> slots: array<u32>;
@group(0) @binding(7) var<storage, read_write> y: array<f32>;
//#if defined(PIE_CHUNKED)
@group(0) @binding(8) var<storage, read_write> indptr: array<i32>;
//#endif

struct Params {
    heads: i32,
    head_dim: i32,
    norm_eps: f32,
}
@group(1) @binding(0) var<uniform> params: Params;

var<workgroup> sq: array<f32, PIE_DMAX>;
var<workgroup> sk: array<f32, PIE_DMAX>;
var<workgroup> sg: array<f32, PIE_DMAX>;
var<workgroup> fold_q: array<f32, PIE_WIDTH>;
var<workgroup> fold_k: array<f32, PIE_WIDTH>;

fn mixed_at(i: u32) -> f32 {
    return pie_bf16_at(mixed[i >> 1u], i);
}

@compute @workgroup_size(PIE_WIDTH)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let tid = lid.x;
    let h = wg.y;
    let heads = u32(params.heads);
    let d = u32(params.head_dim);
    let plane = heads * d;
    let head = h * d;
    let alpha = exp(a_log[h]);

//#if defined(PIE_CHUNKED)
    let r = wg.z;
    let begin = u32(indptr[r]);
    let end = u32(indptr[r + 1u]);
    // Workgroup-uniform: `r` is `workgroup_id.z`, so the whole workgroup
    // leaves together and no barrier below is reached by a subset.
    if (end <= begin) { return; }
    let seat = slots[begin];
//#else
    let n = wg.z;
    let begin = n;
    let end = n + 1u;
    let seat = slots[n];
//#endif

    let state = ((seat * heads + h) * d) * d;

    for (var t = begin; t < end; t = t + 1u) {
        let row = t * 3u * plane;

        // The whole q and k thirds, not this head's slice of them. See the
        // header: the norm is the token's, and the redundancy is what having
        // no scratch plane costs.
        var sum_q = 0.0;
        var sum_k = 0.0;
        for (var i = tid; i < plane; i = i + PIE_WIDTH) {
            let qv = mixed_at(row + i);
            let kv = mixed_at(row + plane + i);
            sum_q = sum_q + qv * qv;
            sum_k = sum_k + kv * kv;
        }
        fold_q[tid] = sum_q;
        fold_k[tid] = sum_k;
        workgroupBarrier();
        var stride = PIE_WIDTH >> 1u;
        loop {
            if (stride == 0u) { break; }
            if (tid < stride) {
                fold_q[tid] = fold_q[tid] + fold_q[tid + stride];
                fold_k[tid] = fold_k[tid] + fold_k[tid + stride];
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }
        let q_inv = inverseSqrt(fold_q[0] + params.norm_eps);
        let k_inv = inverseSqrt(fold_k[0] + params.norm_eps);

        // Stage this head's three rows. The decay folds `exp` in here rather
        // than at the point of use, so the inner loops read one number.
        for (var i = tid; i < d; i = i + PIE_WIDTH) {
            let at = head + i;
            sq[i] = mixed_at(row + at) * q_inv;
            sk[i] = mixed_at(row + plane + at) * k_inv;
            let z = pie_bf16_at(f_gate[(t * plane + at) >> 1u], t * plane + at)
                + dt_bias[at];
            var softplus = z;
            if (z <= 20.0) {
                softplus = log(1.0 + exp(z));
            }
            sg[i] = exp(-alpha * softplus);
        }
        // Holds `fold_q[0]` until every invocation has read it, and publishes
        // the three staged rows to the whole workgroup.
        workgroupBarrier();

        let at_b = t * heads + h;
        let beta = 1.0 / (1.0 + exp(-pie_bf16_at(b_beta[at_b >> 1u], at_b)));
        let out = t * plane + head;
        let vbase = row + 2u * plane + head;

        for (var c = tid; c < d; c = c + PIE_WIDTH) {
            var mem = 0.0;
            for (var i = 0u; i < d; i = i + 1u) {
                let cell = state + i * d + c;
                let s = rstate[cell] * sg[i];
                rstate[cell] = s;
                mem = mem + s * sk[i];
            }
            let delta = (mixed_at(vbase + c) - mem) * beta;
            var acc = 0.0;
            for (var i = 0u; i < d; i = i + 1u) {
                let cell = state + i * d + c;
                let s = rstate[cell] + sk[i] * delta;
                rstate[cell] = s;
                acc = acc + s * sq[i];
            }
            y[out + c] = acc;
        }
        // The next token restages `sq`/`sk`/`sg` and `fold_*` while this one's
        // readers may still be in the loop above.
        workgroupBarrier();
    }
}

// pie:instantiate kda_step_bfloat16
// pie:instantiate kda_chunked_bfloat16 PIE_CHUNKED=1
