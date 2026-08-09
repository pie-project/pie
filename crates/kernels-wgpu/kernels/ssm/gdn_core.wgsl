// Fused GDN core for decode.
//
// The conv/silu prologue, the q/k L2 norms, the gated recurrent update and the
// ping-pong conv-state writeback, in one dispatch: `gdn_core.metal` and
// `ssm/gdn_core.comp` have the same four phases in the same order, and the
// fusion is the point -- the split path in `gdn_prep.wgsl` exists for prefill,
// where the q/k work is worth staging, and costs an extra round trip through
// memory that a single decode token does not want to pay.
//
// 32 lanes on x walk the KEY dimension and 4 rows on y walk the VALUE
// dimension, so a workgroup covers 4 value channels and each reduction is over
// the 32 x-lanes of ONE row.
//
// ## Why the reduction is local and not `common/reduce.inc.wgsl`
//
// `pie_workgroup_sum(lane, lanes, v)` reduces the WHOLE workgroup. Here that
// would be 128 invocations spanning four independent value channels, and the
// answer would be the sum of four different dot products. `row_sum32` below
// reduces the 32 lanes of one y-row, which is a different recurrence, so it is
// stated here rather than bent into the shared one.
//
// ## The store guard, and why it is not an early return
//
// `dv_idx` is the global y and the grid rounds up to the workgroup's 4, so a
// `Dv` that is not a multiple of 4 launches lanes with no value channel to own.
// Every GDN checkpoint the tree has seen states `Dv = 128`, which is why this
// has never fired -- and is exactly the shape of thing that fires on the first
// checkpoint that does not.
//
// The guard is on the STORES. `row_sum32` barriers, and `workgroupBarrier()`
// must sit in control flow uniform across the workgroup: returning early would
// leave the surviving lanes waiting at a barrier their neighbours will never
// reach, which is a HANG rather than a wrong number. The extra lanes' loads are
// harmless -- they read a neighbouring head and the arithmetic is thrown away.
// The Vulkan sibling has this guard in `gdn_prep`'s recurrent arm and NOT here;
// carrying it across is a deliberate divergence, not an oversight.

//#include "common/bf16.inc.wgsl"
//#include "ssm/gdn_params.inc.wgsl"

@group(0) @binding(0) var<storage, read> mixed: array<u32>;
@group(0) @binding(1) var<storage, read> conv_state: array<f32>;
@group(0) @binding(2) var<storage, read_write> rstate: array<f32>;
// ATOMIC, because a bf16 tensor is `array<u32>` with two values per word and
// adjacent `dv_idx` are adjacent invocations -- inside this workgroup for the
// four y-rows, and across workgroups at every 4-channel boundary. WGSL has no
// sub-word atomic, so the read-modify-write `pie_store_bf16` performs would
// drop one of the two. See `store_core_out`.
@group(0) @binding(3) var<storage, read_write> core_out: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read> conv_w: array<u32>;
@group(0) @binding(5) var<storage, read> conv_b: array<u32>;
@group(0) @binding(6) var<storage, read> A_log: array<f32>;
@group(0) @binding(7) var<storage, read> dt_bias: array<u32>;
@group(0) @binding(8) var<storage, read> a_gate: array<u32>;
@group(0) @binding(9) var<storage, read> b_gate: array<u32>;
// The ping-pong half: the conv state is never updated in place, because a
// workgroup that shifted its own taps would race every other workgroup reading
// the same slot's earlier taps.
@group(0) @binding(10) var<storage, read_write> new_conv_state: array<f32>;
@group(0) @binding(11) var<storage, read> p: GdnCoreParams;
//#if defined(PIE_SLOTTED)
@group(0) @binding(12) var<storage, read> slot_ids: array<u32>;
//#endif

var<workgroup> sh_reduce: array<f32, 128>;

// The bf16 half-index split, per buffer. `pie_load_bf16` would say this once
// and cannot be called: naga 30 refuses a `ptr<storage, ...>` function
// parameter, so a module calling it parses and then fails
// `create_shader_module` on every device. Only the subscript is local; the
// widening is still the fragment's.
fn load_mixed(i: i32) -> f32 {
    let word = mixed[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1) == 1));
}

fn load_conv_w(i: i32) -> f32 {
    let word = conv_w[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1) == 1));
}

fn load_conv_b(i: i32) -> f32 {
    let word = conv_b[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1) == 1));
}

fn load_dt_bias(i: i32) -> f32 {
    let word = dt_bias[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1) == 1));
}

fn load_a_gate(i: i32) -> f32 {
    let word = a_gate[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1) == 1));
}

fn load_b_gate(i: i32) -> f32 {
    let word = b_gate[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1) == 1));
}

// AND then OR: each touches only this invocation's sixteen bits, so whichever
// order the two writers of a word interleave in, each half ends at its own
// writer's value.
fn store_core_out(i: i32, v: f32) {
    let at = u32(i) >> 1u;
    let b = pie_f32_to_bf16(v);
    if ((i & 1) == 1) {
        atomicAnd(&core_out[at], 0x0000ffffu);
        atomicOr(&core_out[at], b << 16u);
    } else {
        atomicAnd(&core_out[at], 0xffff0000u);
        atomicOr(&core_out[at], b);
    }
}

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

// The sum of `v` across the 32 x-lanes of ONE y-row, broadcast to all of them.
//
// Three barriers' worth of care: the tree needs one per level, and the read of
// the result needs one AFTER it, because callers reduce four times per launch
// and an invocation racing ahead to the next call would overwrite a partial its
// neighbour has not read yet. The guard is on the ADD, never on reaching the
// barrier.
fn row_sum32(lx: u32, ly: u32, v: f32) -> f32 {
    let at = ly * 32u + lx;
    sh_reduce[at] = v;
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride = stride >> 1u) {
        if (lx < stride) {
            sh_reduce[at] = sh_reduce[at] + sh_reduce[at + stride];
        }
        workgroupBarrier();
    }
    let outv = sh_reduce[ly * 32u];
    workgroupBarrier();
    return outv;
}

// One causal-conv output channel, silu'd: `Kc - 1` taps from the stored state
// and the last from this token's mixed projection.
fn convsilu(slot: i32, b_idx: i32, c: i32) -> f32 {
    var acc = load_conv_b(c);
    for (var j = 0; j < p.Kc - 1; j = j + 1) {
        // Tap `j` of the state is at `j + 1`: the state holds the last `Kc - 1`
        // tokens oldest-first, and slot 0 is the one about to fall off.
        acc = acc + conv_state[(slot * p.Kc + (j + 1)) * p.conv_dim + c]
                  * load_conv_w(c * p.Kc + j);
    }
    acc = acc + load_mixed(b_idx * p.conv_dim + c) * load_conv_w(c * p.Kc + (p.Kc - 1));
    return silu(acc);
}

// Shift this channel's window by one and append the current token.
fn write_conv(slot: i32, b_idx: i32, c: i32) {
    for (var j = 0; j < p.Kc - 1; j = j + 1) {
        new_conv_state[(slot * p.Kc + j) * p.conv_dim + c] =
            conv_state[(slot * p.Kc + (j + 1)) * p.conv_dim + c];
    }
    new_conv_state[(slot * p.Kc + (p.Kc - 1)) * p.conv_dim + c] =
        load_mixed(b_idx * p.conv_dim + c);
}

@compute @workgroup_size(32, 4)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let n = i32(wid.z);
    let b_idx = n / p.Hv;
    let hv_idx = n % p.Hv;
    // Group-query: `rep` value heads share one key head, and only the FIRST of
    // them may write the shared q/k conv state back -- otherwise `rep`
    // workgroups write the same channels and the last one wins by luck.
    let rep = p.Hv / p.Hk;
    let hk_idx = hv_idx / rep;
    let hk_first = (hv_idx % rep) == 0;
    let dk_idx = i32(lid.x);
    let dv_idx = i32(gid.y);
    let n_per_t = p.Dk / 32;
    // See the header: the store guard, not an early return.
    let mine = dv_idx < p.Dv;
//#if defined(PIE_SLOTTED)
    // The paged path: the row's state lives wherever the allocator put it, and
    // `b_idx` is only where the row is in THIS batch.
    let slot = i32(slot_ids[b_idx]);
//#else
    let slot = b_idx;
//#endif

    // `n_per_t` is `Dk / 32` and at most 8 for every checkpoint the tree
    // compiles for (`Dk <= 256`); the arrays are sized for that ceiling and the
    // loops run to `n_per_t`.
    var qraw: array<f32, 8>;
    var kraw: array<f32, 8>;
    var qsq = 0.0;
    var ksq = 0.0;
    for (var i = 0; i < n_per_t; i = i + 1) {
        let d = n_per_t * dk_idx + i;
        qraw[i] = convsilu(slot, b_idx, p.q_off + hk_idx * p.Dk + d);
        kraw[i] = convsilu(slot, b_idx, p.k_off + hk_idx * p.Dk + d);
        qsq = qsq + qraw[i] * qraw[i];
        ksq = ksq + kraw[i] * kraw[i];
    }
    // The q normalisation folds in `1/sqrt(Dk)`; the k one does not, because
    // the scale belongs to the query side of the product exactly once.
    let qinv = p.inv_sqrt_dk / sqrt(row_sum32(lid.x, lid.y, qsq) + p.eps);
    let kinv = 1.0 / sqrt(row_sum32(lid.x, lid.y, ksq) + p.eps);

    let vval = convsilu(slot, b_idx, p.v_off + hv_idx * p.Dv + dv_idx);
    let ad = load_a_gate(b_idx * p.Hv + hv_idx) + load_dt_bias(hv_idx);
    // softplus, in the form that does not overflow: `max(x,0) + log1p(e^-|x|)`
    // rather than `log(1 + e^x)`, whose exponential is infinite by x = 89.
    let sp = max(ad, 0.0) + log(1.0 + exp(-abs(ad)));
    let decay = exp(-exp(A_log[hv_idx]) * sp);
    let beta = 1.0 / (1.0 + exp(-load_b_gate(b_idx * p.Hv + hv_idx)));

    let state_base = ((slot * p.Hv + hv_idx) * p.Dv + dv_idx) * p.Dk;
    var st: array<f32, 8>;
    var kv = 0.0;
    for (var i = 0; i < n_per_t; i = i + 1) {
        let d = n_per_t * dk_idx + i;
        st[i] = rstate[state_base + d] * decay;
        kv = kv + st[i] * (kraw[i] * kinv);
    }
    kv = row_sum32(lid.x, lid.y, kv);
    // The delta rule: how far this key's read of the state is from the value it
    // should have produced, gated by beta.
    let delta = (vval - kv) * beta;
    var outv = 0.0;
    for (var i = 0; i < n_per_t; i = i + 1) {
        let d = n_per_t * dk_idx + i;
        let kk = kraw[i] * kinv;
        st[i] = st[i] + kk * delta;
        outv = outv + st[i] * (qraw[i] * qinv);
        if (mine) {
            rstate[state_base + d] = st[i];
        }
    }
    outv = row_sum32(lid.x, lid.y, outv);
    if (dk_idx == 0 && mine) {
        store_core_out((b_idx * p.Hv + hv_idx) * p.Dv + dv_idx, outv);
    }

    // The q/k conv writeback is one workgroup's job per key head: `dv_idx == 0`
    // picks one value row and `hk_first` one value head of the group. The v
    // writeback below is per value channel and every lane does its own.
    if (dv_idx == 0 && hk_first) {
        for (var i = 0; i < n_per_t; i = i + 1) {
            let d = n_per_t * dk_idx + i;
            write_conv(slot, b_idx, p.q_off + hk_idx * p.Dk + d);
            write_conv(slot, b_idx, p.k_off + hk_idx * p.Dk + d);
        }
    }
    if (mine) {
        write_conv(slot, b_idx, p.v_off + hv_idx * p.Dv + dv_idx);
    }
}

// pie:instantiate gdn_core_bfloat16
// pie:instantiate gdn_core_slotted_bfloat16 PIE_SLOTTED=1
