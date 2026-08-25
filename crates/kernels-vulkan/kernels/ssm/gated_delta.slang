// The gated delta-rule recurrence, prologue and scan fused into one launch.
//
// Two entrypoints, one per point: `gated_delta` walks ONE token per workgroup
// and `gated_delta_chunked` walks a request's whole CSR window. Same
// recurrence, written twice and kept apart: the step reads each state cell
// once and the chunk carries them across a loop whose barriers only the chunk
// needs, so folding the two would wrap every barrier in this file in a
// token-axis branch.
//
// ── WHY THE PROLOGUE IS IN HERE ───────────────────────────────────────────
//
// `kernels-cuda` stages this in three launches -- `qwen_gdn_qk_norm` writes an
// L2-normalised q and k into fp32 scratch, `qwen_gdn_v_gates` widens v and
// cuts the gate row, and only then does the scan read the five planes back --
// and `ssm/gdn_prep.slang` on this plane is that same staging ported. It needs
// a scratch allocator. This plane has none: `kernels_vulkan::plane::Encode` is
// `fire`, `resolve`, `staged` and `windowed`, and `staged` names the five
// tables `driver-vulkan`'s `binding::FireTable` translates -- there is no door
// through which a claim body is handed a buffer of its own.
//
// So the prologue goes where a body can put it: inside the kernel, once per
// (value head, token), computed by the very workgroup about to consume it.
// That costs nothing this plane was not already paying -- the q and k rows a
// head needs are `k_dim` values, read once, by the only workgroup that wants
// them -- and it is why the scratch door's refusal is not on the way of this
// point.
//
// ── THE ARITHMETIC ────────────────────────────────────────────────────────
//
// `qwen_gdn_qk_norm` and `qwen_gdn_v_gates` in
// `kernels-cuda/kernels/ssm/gated_delta_net_prep.cuh` and
// `chunk_gated_delta_prefill_batched` in
// `kernels-cuda/kernels/ssm/gated_delta_net.cuh`, which is where the numeric
// contract was measured. Per (token t, value head hv), with
// `hk = hv / (v_heads / k_heads)` the key head it shares:
//
//     q       = qkv[t, hk-slice of the q third]
//     k       = qkv[t, hk-slice of the k third]
//     v       = qkv[t, hv-slice of the v third]
//     q_norm  = q * rsqrt(sum(q^2) + 1e-6) / sqrt(k_dim)
//     k_norm  = k * rsqrt(sum(k^2) + 1e-6)
//     g       = exp(gates[t, hv])            -- the FIRST half of the row
//     beta    =     gates[t, v_heads + hv]   -- the SECOND half
//
//     S           *= g                          (every cell)
//     kv_mem[c]    = sum_i S[i, c] * k_norm[i]
//     delta[c]     = (v[c] - kv_mem[c]) * beta
//     S[i, c]     += k_norm[i] * delta[c]
//     y[t, hv, c]  = sum_i S[i, c] * q_norm[i]
//
// The `1e-6` inside both roots is the kernel's and not the statement's:
// `ssm.gated_delta` declares no epsilon and cuda hard-codes this one. The
// `/ sqrt(k_dim)` lands on q ALONE -- it is the attention scale folded into
// the query, and putting it on k as well would square it.
//
// THE TWO STATE PASSES ARE NOT ONE PASS. `kv_mem` is read out of the DECAYED
// state and `y` out of the UPDATED state, so the decay has to be committed
// before the first sum and the rank-one update before the second. A body that
// fused them computes `y` against a state one update behind: finite,
// plausible, and a little further wrong at every token.
//
// ── THE PACKING IS THE KERNEL'S ───────────────────────────────────────────
//
// `gates` arrives as ONE rectangle of `2 * v_heads` floats, `[g_log | beta]`,
// exactly as `ssm/gdn_gates.slang` wrote it. The seam is `v_heads`, which this
// kernel is TOLD, and the executor hands over the whole rectangle. Two compact
// halves against a kernel indexing a packed row is a defect this family has
// already shipped once, and it is invisible at a single token because there
// the two layouts coincide.
//
// ── THE STATE'S LAYOUT IS `[slot, hv, k_dim, v_dim]`, K-MAJOR ─────────────
//
// A cell is `rstate[((slot * v_heads + hv) * k_dim + i) * v_dim + c]`, so the
// VALUE channel is the fast axis. That is the reading that makes the inner
// loop coalesce: an invocation owns one channel `c` and walks `i`, so at a
// given `i` the invocations of a workgroup touch consecutive words. The
// transpose -- value-major, which `kernels-cuda`'s `KLast` template parameter
// also offers -- gives each invocation a contiguous run and the workgroup a
// strided one, which is the wrong way round for hardware that coalesces per
// subgroup.
//
// It is a free choice because the slab is this family's alone: nothing else in
// this tree addresses `Slab::State` for a GDN layer, and the pool sizes it as
// a flat `v_heads * k_dim * v_dim` product per slot that either reading fills.
//
// ── THE STATE IS UPDATED IN PLACE, AND THAT IS SAFE HERE ──────────────────
//
// Unlike the conv window there is no second plane and none is wanted. A cell
// `(hv, i, c)` of a slot is written by exactly one invocation of exactly one
// workgroup -- the one that owns value channel `c` of head `hv` for that
// request -- so no other invocation can observe it half-updated. The conv
// window needs its ping-pong because its SHIFT moves a value between rows that
// different invocations own; nothing here moves between cells.

#include "common/bf16.slang"

// One workgroup per (value head, token-or-request). `PIE_WIDTH` invocations
// share the head's q and k rows and split its value channels.
#define PIE_WIDTH 128

// The widest key head this scan stages. 256 floats twice is 2 KiB of the 16
// KiB of shared memory Vulkan guarantees; the claim body refuses anything
// wider rather than letting the array run short.
#define PIE_KMAX 256

PIE_BUFFER_RO(0, qkv);
[[vk::binding(1)]] StructuredBuffer<float> gates;
[[vk::binding(2)]] RWStructuredBuffer<float> rstate;
[[vk::binding(3)]] StructuredBuffer<uint> slots;
[[vk::binding(4)]] RWStructuredBuffer<float> y;
#if defined(PIE_CHUNKED)
[[vk::binding(5)]] StructuredBuffer<int> indptr;
#endif

// Sixteen bytes against the 128 `Device::max_push` guarantees. All four are
// stated by `ssm.gated_delta` itself and none is derived from an extent: the
// packed qkv row is `2 * k_heads * k_dim + v_heads * v_dim` wide and the claim
// body checks the operand against exactly that product before firing.
struct Push {
    int k_heads;
    int v_heads;
    int k_dim;
    int v_dim;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float sq[PIE_KMAX];
groupshared float sk[PIE_KMAX];
groupshared float fold_q[PIE_WIDTH];
groupshared float fold_k[PIE_WIDTH];

[numthreads(PIE_WIDTH, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    const int tid = int(local.x);
    const int hv = int(group.y);
    const int hk = hv / (pc.v_heads / pc.k_heads);
    const int dk = pc.k_dim;
    const int dv = pc.v_dim;
    const int keys = pc.k_heads * dk;
    const int pitch = 2 * keys + pc.v_heads * dv;
    const float scale = rsqrt(float(dk));

#if defined(PIE_CHUNKED)
    const int r = int(group.z);
    const int begin = indptr[r];
    const int end = indptr[r + 1];
    // Workgroup-uniform: `r` is the group id, so every invocation reads the
    // same two words and leaves together. No barrier below is ever reached by
    // a subset of the workgroup, which is what makes this a return rather than
    // a guard on the stores.
    if (end <= begin) return;
    // Every token of a request sits in the same seat, so the window's first
    // row names it.
    const int seat = int(slots[begin]);
#else
    const int n = int(group.z);
    const int begin = n;
    const int end = n + 1;
    const int seat = int(slots[n]);
#endif

    const int state = ((seat * pc.v_heads + hv) * dk) * dv;

    for (int t = begin; t < end; ++t) {
        const int row = t * pitch;
        const int qbase = row + hk * dk;
        const int kbase = qbase + keys;
        const int vbase = row + 2 * keys + hv * dv;

        // Stage the head's q and k rows and fold their squares in one pass.
        float sum_q = 0.0;
        float sum_k = 0.0;
        for (int i = tid; i < dk; i += PIE_WIDTH) {
            const float qv = PIE_LOAD(qkv[qbase + i]);
            const float kv = PIE_LOAD(qkv[kbase + i]);
            sq[i] = qv;
            sk[i] = kv;
            sum_q += qv * qv;
            sum_k += kv * kv;
        }
        fold_q[tid] = sum_q;
        fold_k[tid] = sum_k;
        GroupMemoryBarrierWithGroupSync();
        // A power-of-two ladder. The two folds share every level, so the
        // barrier count is the same as it would be for one of them.
        for (int stride = PIE_WIDTH / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                fold_q[tid] += fold_q[tid + stride];
                fold_k[tid] += fold_k[tid + stride];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        const float q_inv = rsqrt(fold_q[0] + 1e-6) * scale;
        const float k_inv = rsqrt(fold_k[0] + 1e-6);
        // The scale lands on the slots this invocation staged, so it needs no
        // barrier of its own -- but every invocation has to see the WHOLE row
        // before the state loop reads it, which the barrier after is for. It
        // also holds `fold_q[0]` and `fold_k[0]` still until every invocation
        // has read them.
        for (int i = tid; i < dk; i += PIE_WIDTH) {
            sq[i] *= q_inv;
            sk[i] *= k_inv;
        }
        GroupMemoryBarrierWithGroupSync();

        // `[g_log | beta]`, cut where the packing says and nowhere else.
        const int fused = t * 2 * pc.v_heads + hv;
        const float decay = exp(gates[fused]);
        const float beta = gates[fused + pc.v_heads];
        const int out = (t * pc.v_heads + hv) * dv;

        for (int c = tid; c < dv; c += PIE_WIDTH) {
            float kv_mem = 0.0;
            for (int i = 0; i < dk; ++i) {
                const int at = state + i * dv + c;
                const float s = rstate[at] * decay;
                rstate[at] = s;
                kv_mem += s * sk[i];
            }
            const float delta = (PIE_LOAD(qkv[vbase + c]) - kv_mem) * beta;
            float acc = 0.0;
            for (int i = 0; i < dk; ++i) {
                const int at = state + i * dv + c;
                const float s = rstate[at] + sk[i] * delta;
                rstate[at] = s;
                acc += s * sq[i];
            }
            y[out + c] = acc;
        }
        // The next token restages `sq`, `sk` and both folds while this one's
        // readers may still be inside the loop above. A value channel is owned
        // by the same invocation at every token, so the state cells need no
        // such fence.
        GroupMemoryBarrierWithGroupSync();
    }
}

// pie:instantiate gated_delta_bfloat16
// pie:instantiate gated_delta_chunked_bfloat16 PIE_CHUNKED=1
