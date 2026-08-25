// The gated delta-rule recurrence, prologue and scan fused into one launch.
//
// Two entrypoints, one per point: `gated_delta` walks ONE token per workgroup
// and `gated_delta_chunked` walks a request's whole CSR window. They are the
// same recurrence written twice, and they are kept apart deliberately -- the
// step's state cells are read once and the chunk's are carried across a loop
// whose barriers only the chunk needs, so folding them would put a token-axis
// branch around every barrier in both.
//
// ── WHY THE PROLOGUE IS IN HERE ───────────────────────────────────────────
//
// `kernels-cuda` stages this in three launches: `qwen_gdn_qk_norm` writes an
// L2-normalised q and k into scratch, `qwen_gdn_v_gates` widens v and cuts the
// gate row, and only then does a scan kernel read the five planes back. That
// shape needs a scratch allocator, and this plane has none -- `Encode` offers
// `fire` and `resolve` and nothing that hands a body a buffer. So the prologue
// is where a body can put it: inside the kernel, once per (head, token),
// computed by the same workgroup that is about to consume it.
//
// It costs nothing this plane was not already paying. The q/k rows a head
// needs are `k_dim` values, they are read once, and the workgroup that reads
// them is the only one that wants them.
//
// ── THE ARITHMETIC ────────────────────────────────────────────────────────
//
// `qwen_gdn_qk_norm` + `qwen_gdn_v_gates` +
// `chunk_gated_delta_prefill_batched` in `kernels-cuda/kernels/ssm/`, which is
// where the numeric contract was measured. Per (token t, value head hv), with
// `hk = hv / (v_heads / k_heads)` the key head it shares:
//
//     q       = qkv[t, hk-slice of the q third]
//     k       = qkv[t, hk-slice of the k third]
//     v       = qkv[t, hv-slice of the v third]
//     q_norm  = q * rsqrt(sum(q^2) + 1e-6) / sqrt(k_dim)
//     k_norm  = k * rsqrt(sum(k^2) + 1e-6)
//     g       = exp(gates[t, hv])          -- the FIRST half of the row
//     beta    =     gates[t, v_heads + hv] -- the SECOND half
//
//     S           *= g                        (every cell)
//     kv_mem[c]    = sum_i S[i, c] * k_norm[i]
//     delta[c]     = (v[c] - kv_mem[c]) * beta
//     S[i, c]     += k_norm[i] * delta[c]
//     y[t, hv, c]  = sum_i S[i, c] * q_norm[i]
//
// The `1e-6` inside both rsqrts is the kernel's, not the statement's:
// `ssm.gated_delta` declares no epsilon and cuda hard-codes this one.
// `/ sqrt(k_dim)` lands on q ALONE -- it is the attention scale folded into
// the query, and putting it on k as well would square it.
//
// THE TWO STATE PASSES ARE NOT ONE PASS. `kv_mem` is read out of the DECAYED
// state and `y` out of the UPDATED state, so the decay must be committed
// before the first sum and the rank-one update before the second. A body that
// fused them would compute `y` against a state one update behind, which is
// finite, plausible, and drifts a little further every token.
//
// ── THE PACKING IS THE KERNEL'S ───────────────────────────────────────────
//
// `gates` arrives as ONE rectangle of `2 * v_heads` floats, `[g_log | beta]`,
// exactly as `ssm/gdn_gates.wgsl` wrote it. The seam is `v_heads`, which this
// kernel is TOLD, and the executor hands over the whole rectangle. Two compact
// halves and a kernel indexing a packed row is the defect this family has
// already shipped once.
//
// ── THE STATE'S LAYOUT IS `[slot, hv, k_dim, v_dim]`, K-MAJOR ─────────────
//
// A cell is `state[((slot * v_heads + hv) * k_dim + i) * v_dim + c]`, so the
// VALUE channel is the fast axis. That is the choice that makes the inner loop
// coalesce: an invocation owns one channel `c` and walks `i`, so at a given
// `i` the invocations of a workgroup touch consecutive words. The transpose --
// value-major, `[.., v_dim, k_dim]`, which `kernels-cuda`'s `KLast` template
// parameter also offers -- gives each invocation a contiguous run and the
// workgroup a strided one, which is the wrong way round for a machine that
// coalesces per warp.
//
// It is a free choice because the slab is this family's alone: only these two
// entrypoints and nothing else in the tree ever addresses `recurrent_state`
// for a GDN layer, and `driver-wgpu` sizes it as a flat
// `v_heads * v_dim * k_dim` product per slot that either reading fills.
//
// ── THE STATE IS UPDATED IN PLACE, AND THAT IS SAFE HERE ──────────────────
//
// Unlike the conv window there is no second plane and none is wanted. A cell
// `(hv, i, c)` of a slot is written by exactly one invocation of exactly one
// workgroup -- the one that owns value channel `c` of head `hv` for that
// request -- so no other invocation can observe it half-updated. The conv
// window needs a ping-pong because its SHIFT moves a value between rows that
// different invocations own; nothing here moves between cells.

//#include "common/bf16.inc.wgsl"

// One workgroup per (value head, token-or-request). `PIE_WIDTH` invocations
// share the head's q/k rows and split its value channels.
const PIE_WIDTH = 128u;

// The widest key head this scan stages. 256 f32 twice is 2 KiB of a 16 KiB
// workgroup budget; the claim body refuses anything wider rather than letting
// the array run short.
const PIE_KMAX = 256u;

@group(0) @binding(0) var<storage, read_write> qkv: array<u32>;
@group(0) @binding(1) var<storage, read_write> gates: array<f32>;
@group(0) @binding(2) var<storage, read_write> rstate: array<f32>;
@group(0) @binding(3) var<storage, read_write> slots: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
//#if defined(PIE_CHUNKED)
@group(0) @binding(5) var<storage, read_write> indptr: array<i32>;
//#endif

struct Params {
    k_heads: i32,
    v_heads: i32,
    k_dim: i32,
    v_dim: i32,
}
@group(1) @binding(0) var<uniform> params: Params;

var<workgroup> sq: array<f32, PIE_KMAX>;
var<workgroup> sk: array<f32, PIE_KMAX>;
var<workgroup> fold_q: array<f32, PIE_WIDTH>;
var<workgroup> fold_k: array<f32, PIE_WIDTH>;

fn qkv_at(i: u32) -> f32 {
    return pie_bf16_at(qkv[i >> 1u], i);
}

@compute @workgroup_size(PIE_WIDTH)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let tid = lid.x;
    let hv = wg.y;
    let heads_k = u32(params.k_heads);
    let heads_v = u32(params.v_heads);
    let dk = u32(params.k_dim);
    let dv = u32(params.v_dim);
    let hk = hv / (heads_v / heads_k);

    let keys = heads_k * dk;
    let pitch = 2u * keys + heads_v * dv;
    let scale = inverseSqrt(f32(dk));

//#if defined(PIE_CHUNKED)
    let r = wg.z;
    let begin = u32(indptr[r]);
    let end = u32(indptr[r + 1u]);
    // Workgroup-uniform: `r` is `workgroup_id.z`, so every invocation here
    // reads the same two words and leaves together. No barrier below is ever
    // reached by a subset of the workgroup.
    if (end <= begin) { return; }
    // Every token of a request sits in the same seat, so the window's first
    // row names it.
    let seat = slots[begin];
//#else
    let n = wg.z;
    let begin = n;
    let end = n + 1u;
    let seat = slots[n];
//#endif

    let state = ((seat * heads_v + hv) * dk) * dv;

    for (var t = begin; t < end; t = t + 1u) {
        let row = t * pitch;
        let qbase = row + hk * dk;
        let kbase = qbase + keys;
        let vbase = row + 2u * keys + hv * dv;

        // Stage the head's q and k rows and fold their squares in one pass.
        var sum_q = 0.0;
        var sum_k = 0.0;
        for (var i = tid; i < dk; i = i + PIE_WIDTH) {
            let qv = qkv_at(qbase + i);
            let kv = qkv_at(kbase + i);
            sq[i] = qv;
            sk[i] = kv;
            sum_q = sum_q + qv * qv;
            sum_k = sum_k + kv * kv;
        }
        fold_q[tid] = sum_q;
        fold_k[tid] = sum_k;
        workgroupBarrier();
        // A power-of-two ladder: `PIE_WIDTH` is 128 and the two folds share
        // every level, so the barrier count is the same as for one of them.
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
        let q_inv = inverseSqrt(fold_q[0] + 1e-6) * scale;
        let k_inv = inverseSqrt(fold_k[0] + 1e-6);
        // The scale lands on the slots this invocation staged, so it needs no
        // barrier of its own -- but every invocation must see the WHOLE row
        // before the state loop reads it, which the next barrier is for. It
        // also holds `fold_q[0]` still until every invocation has read it.
        for (var i = tid; i < dk; i = i + PIE_WIDTH) {
            sq[i] = sq[i] * q_inv;
            sk[i] = sk[i] * k_inv;
        }
        workgroupBarrier();

        // `[g_log | beta]`, cut where the packing says and nowhere else.
        let fused = t * 2u * heads_v + hv;
        let decay = exp(gates[fused]);
        let beta = gates[fused + heads_v];
        let out = (t * heads_v + hv) * dv;

        for (var c = tid; c < dv; c = c + PIE_WIDTH) {
            var kv_mem = 0.0;
            for (var i = 0u; i < dk; i = i + 1u) {
                let at = state + i * dv + c;
                let s = rstate[at] * decay;
                rstate[at] = s;
                kv_mem = kv_mem + s * sk[i];
            }
            let delta = (qkv_at(vbase + c) - kv_mem) * beta;
            var acc = 0.0;
            for (var i = 0u; i < dk; i = i + 1u) {
                let at = state + i * dv + c;
                let s = rstate[at] + sk[i] * delta;
                rstate[at] = s;
                acc = acc + s * sq[i];
            }
            y[out + c] = acc;
        }
        // The next token restages `sq`/`sk` and `fold_*` while this one's
        // readers may still be in the loop above. A value channel is owned by
        // the same invocation at every token, so the state cells need no such
        // fence.
        workgroupBarrier();
    }
}

// pie:instantiate gated_delta_bfloat16
// pie:instantiate gated_delta_chunked_bfloat16 PIE_CHUNKED=1
