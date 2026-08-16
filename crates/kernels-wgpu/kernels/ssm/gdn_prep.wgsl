// Split GDN prep/recurrent path, including the prompt prefill scan.
//
// Three kernels in one file, the way `ssm/gdn_prep.comp` has them, because they
// share the fp32 scratch layout and the ping-pong conv-state rules and a split
// would let the two halves of the scratch drift apart:
//
//   * PREP (and its `_slotted` and `_prefill` variants) runs the conv/silu
//     prologue, the q/k L2 norms and the gate math, and stages the results as
//     fp32 in `pre_q`/`pre_k`/`pre_gate`. The point of the split is that
//     `gdn_core.wgsl` recomputes all of that per value channel; here it is done
//     once.
//   * RECURRENT consumes that scratch and does the gated state update -- the
//     same recurrence `gdn_core.wgsl` fuses, minus the prologue.
//   * SCAN is the prefill's sequential half: ONE workgroup per (head, value
//     block) walking `n_scan` tokens in order, because the recurrence is
//     sequential in t by definition and the only parallelism left is across
//     heads and value channels.
//
// Three things are worth reading before touching anything.
//
// **The reductions are local, not `common/reduce.inc.wgsl`.**
// `pie_workgroup_sum(lane, lanes, v)` reduces the WHOLE workgroup.
// `row_sum32` here reduces the 32 x-lanes of ONE y-row of a 32x4 workgroup, and
// `lane_sum` reduces `PIE_LANES`-wide GROUPS inside a single 32-lane workgroup
// -- `2` of them at `PIE_LANES = 16`, `8` at 4. Neither is the shared
// function's recurrence, and bending it into one would cost every caller a
// parameter for a case only this file has.
//
// **The `Dv` guard is on the STORES.** `local_size_y` is 4 in the recurrent arm
// and Metal sizes the grid at `Dv` exactly, so a `Dv` that is not a multiple of
// 4 launches lanes with no value channel to own. `workgroupBarrier()` must sit
// in control flow uniform across the workgroup, so returning early would leave
// the surviving lanes waiting at a barrier their neighbours never reach: a HANG
// rather than a wrong number. The extra lanes read a neighbouring head and
// their arithmetic is thrown away.
//
// **`core_out` is `array<atomic<u32>>`.** A bf16 tensor is `array<u32>`, two
// values to a word (`common/bf16.inc.wgsl`), and both arms that write it have
// adjacent value channels in different invocations -- different workgroups, at
// every block boundary. WGSL has no sub-word atomic, so a read-modify-write
// would drop one of the pair. See `store_core_out`.

//#include "common/bf16.inc.wgsl"
//#include "ssm/gdn_params.inc.wgsl"

//#if defined(PIE_SCAN)
@group(0) @binding(0) var<storage, read_write> rstate: array<f32>;
@group(0) @binding(1) var<storage, read_write> core_out: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> pre_q: array<f32>;
@group(0) @binding(3) var<storage, read_write> pre_k: array<f32>;
@group(0) @binding(4) var<storage, read_write> pre_gate: array<f32>;
@group(0) @binding(5) var<storage, read_write> p: GdnCoreParams;
@group(0) @binding(6) var<storage, read_write> slot_ids: array<u32>;

// The two SCALAR operands, which is why they are here and the eleven fields of
// `GdnCoreParams` are not: the row states these two as numbers and the rest as
// a pointer.
//
// `row_pitch` describes the IN-PROJECTION, `mixed`, and nothing else. Every
// fp32 scratch row is packed at its OWN width -- `Hv*Dk` for q and k,
// `2*Hv + Hv*Dv` for the gates and the staged v -- and the body computes those
// from `GdnCoreParams`. A shared pitch has to clear the widest of them, and
// `pre_gate` is wider than `mixed` on any stack whose value width reaches its
// key width.
struct Params {
    row_pitch: i32,
    n_scan: i32,
}
@group(1) @binding(0) var<uniform> params: Params;
//#else
@group(0) @binding(0) var<storage, read_write> mixed: array<u32>;
@group(0) @binding(1) var<storage, read_write> conv_state: array<f32>;
//#if defined(PIE_RECURRENT)
@group(0) @binding(2) var<storage, read_write> rstate: array<f32>;
@group(0) @binding(3) var<storage, read_write> core_out: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> conv_w: array<u32>;
@group(0) @binding(5) var<storage, read_write> conv_b: array<u32>;
@group(0) @binding(6) var<storage, read_write> pre_q: array<f32>;
@group(0) @binding(7) var<storage, read_write> pre_k: array<f32>;
@group(0) @binding(8) var<storage, read_write> pre_gate: array<f32>;
@group(0) @binding(9) var<storage, read_write> new_conv_state: array<f32>;
@group(0) @binding(10) var<storage, read_write> p: GdnCoreParams;
//#if defined(PIE_SLOTTED)
@group(0) @binding(11) var<storage, read_write> slot_ids: array<u32>;
//#endif
//#else
@group(0) @binding(2) var<storage, read_write> conv_w: array<u32>;
@group(0) @binding(3) var<storage, read_write> conv_b: array<u32>;
@group(0) @binding(4) var<storage, read_write> A_log: array<f32>;
@group(0) @binding(5) var<storage, read_write> dt_bias: array<u32>;
@group(0) @binding(6) var<storage, read_write> a_gate: array<u32>;
@group(0) @binding(7) var<storage, read_write> b_gate: array<u32>;
// The prep arm WRITES the scratch the recurrent arm reads: same three tensors,
// opposite access, which is the whole reason the split exists.
@group(0) @binding(8) var<storage, read_write> pre_q: array<f32>;
@group(0) @binding(9) var<storage, read_write> pre_k: array<f32>;
@group(0) @binding(10) var<storage, read_write> pre_gate: array<f32>;
@group(0) @binding(11) var<storage, read_write> new_conv_state: array<f32>;
@group(0) @binding(12) var<storage, read_write> p: GdnCoreParams;
//#if defined(PIE_SLOTTED) || defined(PIE_PREFILL)
// Prefill is always slotted -- it writes the conv state of the sequence it is
// filling -- so it binds `slot_ids` whether or not `PIE_SLOTTED` is set, and
// reads slot 0 because one prefill is one sequence.
@group(0) @binding(13) var<storage, read_write> slot_ids: array<u32>;
//#endif
//#if defined(PIE_PREFILL)
struct Params {
    row_pitch: i32,
    n_scan: i32,
}
@group(1) @binding(0) var<uniform> params: Params;
//#endif
//#endif
//#endif

// One f32 per lane of the widest workgroup here (32 x 4).
var<workgroup> sh_reduce: array<f32, 128>;

fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

// The bf16 half-index split, per buffer. `pie_load_bf16` takes a
// `ptr<storage, ...>`, which naga 30 refuses as a function parameter
// (`unrestricted_pointer_parameters` is unimplemented), so the subscript is
// local and only the widening is the fragment's.
//#if !defined(PIE_SCAN)
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
//#endif

//#if !defined(PIE_SCAN) && !defined(PIE_RECURRENT)
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
//#endif

//#if defined(PIE_SCAN) || defined(PIE_RECURRENT)
// AND then OR: each touches only this invocation's sixteen bits, so whichever
// order the two writers of a word interleave in, each half ends at its own
// writer's value. A read-modify-write here would silently drop one channel of
// every adjacent pair that straddles a workgroup.
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
//#endif

//#if defined(PIE_SCAN)
// The sum of `v` across one `PIE_LANES`-wide group of the 32-lane workgroup,
// broadcast to that group.
//
// `base` is the group's first lane, so the tree folds inside the group and the
// `2 to 8` groups of a workgroup never see each other's partials -- each owns a
// different value block. The trailing barrier is load-bearing: this is called
// `2 * PIE_VROWS` times per token, and without it a lane racing to the next
// call overwrites a partial its neighbour has not read.
fn lane_sum(lx: u32, v: f32) -> f32 {
    let base = (lx / u32(PIE_LANES)) * u32(PIE_LANES);
    sh_reduce[lx] = v;
    workgroupBarrier();
    for (var stride = u32(PIE_LANES) >> 1u; stride > 0u; stride = stride >> 1u) {
        let rel = lx - base;
        if (rel < stride) {
            sh_reduce[lx] = sh_reduce[lx] + sh_reduce[lx + stride];
        }
        workgroupBarrier();
    }
    let outv = sh_reduce[base];
    workgroupBarrier();
    return outv;
}

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let lanes = i32(PIE_LANES);
    let vrows = i32(PIE_VROWS);
    // How many value GROUPS a 32-lane workgroup holds: at `PIE_LANES = 8` the
    // workgroup covers four value blocks at once, each with its own 8-lane
    // reduction. `PIE_LANES` is the key dimension's parallelism and `PIE_VROWS`
    // the value dimension's, and the launch picks the pair that fits `Dk`/`Dv`.
    let rows = 32 / lanes;
    let hv_idx = i32(wid.z);
    let dv_base = (i32(wid.y) * rows + (i32(lid.x) / lanes)) * vrows;
    let dk_idx = i32(lid.x) % lanes;
    // ONE ROW PER SEAT, IN TURN, and not one seat for the whole rectangle.
    //
    // `slot_ids` is per ROW and a fire holds whatever the batcher packed, so a
    // prefill rectangle is one sequence per seat laid end to end. Reading
    // `slot_ids[0]` and keeping it walked every request in the fire into the
    // FIRST one's state -- each one continuing the last one's recurrence and
    // all of them writing back over one seat. `kernels-metal`'s scan records
    // the same defect and the same repair.
    //
    // It is invisible to a fire of one request, which is every probe in
    // `driver-wgpu/tests/hybrid_probe.rs`: with one seat the comparison below
    // is never true and this is exactly the code that ran before.
    var slot = i32(slot_ids[0]);
    let n_per_t = p.Dk / lanes;
    // See the prefill arm above: every scratch row is packed at its own width
    // and `row_pitch` reaches this entrypoint to describe a buffer it never
    // reads.
    let qk_pitch = p.Hv * p.Dk;
    let g_pitch = 2 * p.Hv + p.Hv * p.Dv;
    let o_pitch = p.Hv * p.Dv;
    let row_lead = dk_idx == 0;
    let active_group = dv_base < p.Dv;
    // The tail: a value block only partly inside `Dv`. `vn` is how many of this
    // group's `PIE_VROWS` rows are real, and every store below is guarded by
    // it -- the reductions are NOT, because a lane that skipped one would hang
    // its group at a barrier.
    var vn = 0;
    if (active_group) {
        vn = min(vrows, p.Dv - dv_base);
    }

    // The whole recurrent state this workgroup owns, in registers: `PIE_VROWS`
    // value rows by up to 32 key elements per lane. It is loaded once, walked
    // over every token, and written back once -- which is the entire reason
    // this arm exists instead of calling the recurrent kernel `n_scan` times.
    var st: array<array<f32, 32>, PIE_VROWS>;
    var state_base = ((slot * p.Hv + hv_idx) * p.Dv + dv_base) * p.Dk;
    for (var v = 0; v < vrows; v = v + 1) {
        for (var i = 0; i < 32; i = i + 1) {
            var val = 0.0;
            if (v < vn && i < n_per_t) {
                val = rstate[state_base + v * p.Dk + n_per_t * dk_idx + i];
            }
            st[v][i] = val;
        }
    }

    // Sequential in t by construction: token t's state is token t-1's. The
    // bound is a uniform, so every invocation runs the same number of
    // iterations and the barriers inside stay uniform.
    for (var t = 0; t < params.n_scan; t = t + 1) {
        // The seat this row belongs to. When it changes the previous request
        // has ended, so its state is written back and the new one's is loaded
        // -- which for a one-request fire is exactly the load above and the
        // store below and nothing else.
        //
        // UNIFORM, and that is what makes it legal here: every invocation
        // reads the same `slot_ids[t]` and carries the same `slot`, so the
        // branch is taken by all of the workgroup or none of it. There is no
        // barrier inside it either way.
        let seat = i32(slot_ids[t]);
        if (seat != slot) {
            for (var v = 0; v < vrows; v = v + 1) {
                for (var i = 0; i < 32; i = i + 1) {
                    if (v < vn && i < n_per_t) {
                        rstate[state_base + v * p.Dk + n_per_t * dk_idx + i] = st[v][i];
                    }
                }
            }
            slot = seat;
            state_base = ((slot * p.Hv + hv_idx) * p.Dv + dv_base) * p.Dk;
            for (var v = 0; v < vrows; v = v + 1) {
                for (var i = 0; i < 32; i = i + 1) {
                    var val = 0.0;
                    if (v < vn && i < n_per_t) {
                        val = rstate[state_base + v * p.Dk + n_per_t * dk_idx + i];
                    }
                    st[v][i] = val;
                }
            }
        }
        let row_t = t * o_pitch;
        let row_f = t * qk_pitch;
        let g_row = t * g_pitch;
        var q: array<f32, 32>;
        var k: array<f32, 32>;
        for (var i = 0; i < 32; i = i + 1) {
            let d = n_per_t * dk_idx + i;
            var qv = 0.0;
            var kv_ = 0.0;
            if (i < n_per_t) {
                qv = pre_q[row_f + hv_idx * p.Dk + d];
                kv_ = pre_k[row_f + hv_idx * p.Dk + d];
            }
            q[i] = qv;
            k[i] = kv_;
        }
        // The two gates the prep arm staged: `ga` is the decay and `gb` the
        // delta-rule beta. Two floats per head, before the per-channel v.
        let ga = pre_gate[g_row + 2 * hv_idx + 0];
        let gb = pre_gate[g_row + 2 * hv_idx + 1];
        var kv: array<f32, PIE_VROWS>;
        for (var v = 0; v < vrows; v = v + 1) {
            var acc = 0.0;
            for (var i = 0; i < 32; i = i + 1) {
                if (i < n_per_t) {
                    st[v][i] = st[v][i] * ga;
                    acc = acc + st[v][i] * k[i];
                }
            }
            // A tail row contributes the identity rather than skipping the
            // call: `lane_sum` barriers.
            var contrib = 0.0;
            if (v < vn) {
                contrib = acc;
            }
            kv[v] = lane_sum(lid.x, contrib);
        }
        for (var v = 0; v < vrows; v = v + 1) {
            var vv = 0.0;
            if (v < vn) {
                // The staged v channels sit AFTER the two per-head gates,
                // which is what the `2 * p.Hv` is: the scratch row is
                // `[gate pairs for every head][v for every head and channel]`.
                vv = pre_gate[g_row + 2 * p.Hv + hv_idx * p.Dv + dv_base + v];
            }
            let delta = (vv - kv[v]) * gb;
            var outv = 0.0;
            for (var i = 0; i < 32; i = i + 1) {
                if (i < n_per_t) {
                    st[v][i] = st[v][i] + k[i] * delta;
                    outv = outv + st[v][i] * q[i];
                }
            }
            var contrib = 0.0;
            if (v < vn) {
                contrib = outv;
            }
            let total = lane_sum(lid.x, contrib);
            if (v < vn && row_lead) {
                store_core_out(row_t + hv_idx * p.Dv + dv_base + v, total);
            }
        }
    }
    for (var v = 0; v < vrows; v = v + 1) {
        for (var i = 0; i < 32; i = i + 1) {
            if (v < vn && i < n_per_t) {
                rstate[state_base + v * p.Dk + n_per_t * dk_idx + i] = st[v][i];
            }
        }
    }
}

//#else

// The sum of `v` across the 32 x-lanes of ONE y-row, broadcast to them. See the
// file header for why this is not `pie_workgroup_sum`.
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

// One causal-conv output channel for a single token, silu'd: `Kc - 1` taps from
// the stored state and the last from this token's mixed projection.
fn convsilu_token(slot: i32, b_idx: i32, c: i32) -> f32 {
    var acc = load_conv_b(c);
    for (var j = 0; j < p.Kc - 1; j = j + 1) {
        // Tap `j` is at state index `j + 1`: the window is oldest-first and
        // slot 0 is the one about to fall off.
        acc = acc + conv_state[(slot * p.Kc + (j + 1)) * p.conv_dim + c]
                  * load_conv_w(c * p.Kc + j);
    }
    acc = acc + load_mixed(b_idx * p.conv_dim + c) * load_conv_w(c * p.Kc + (p.Kc - 1));
    return silu(acc);
}

// Shift this channel's window by one and append the current token. Written to
// the ping-pong half, never in place: a workgroup that shifted the state it is
// reading would race every other workgroup on the same slot.
fn write_conv_token(slot: i32, b_idx: i32, c: i32) {
    for (var j = 0; j < p.Kc - 1; j = j + 1) {
        new_conv_state[(slot * p.Kc + j) * p.conv_dim + c] =
            conv_state[(slot * p.Kc + (j + 1)) * p.conv_dim + c];
    }
    new_conv_state[(slot * p.Kc + (p.Kc - 1)) * p.conv_dim + c] =
        load_mixed(b_idx * p.conv_dim + c);
}

//#if defined(PIE_PREFILL)
// Tap `j` of token `t`'s window. A prefill has the earlier tokens in `mixed`
// itself, so only the taps before the prompt's start come from the conv state
// -- which is what the negative `idx` branch is, and why a prefill needs no
// per-token state update until the last token.
fn tap(slot: i32, t: i32, start: i32, j: i32, c: i32) -> f32 {
    let idx = t - (p.Kc - 1) + j;
    if (idx >= start) {
        return load_mixed(idx * params.row_pitch + c);
    }
    // `p.Kc + local` with `local` negative indexes from the END of the stored
    // window, which is where THIS REQUEST'S left context is. `local` counts
    // from the request's own first row and not the fire's, so a request that
    // is not first in the rectangle reads its own seat's history instead of
    // its neighbour's tokens.
    let local = idx - start;
    return conv_state[(slot * p.Kc + p.Kc + local) * p.conv_dim + c];
}

fn convsilu_prefill(slot: i32, t: i32, start: i32, c: i32) -> f32 {
    var acc = load_conv_b(c);
    for (var j = 0; j < p.Kc - 1; j = j + 1) {
        acc = acc + tap(slot, t, start, j, c) * load_conv_w(c * p.Kc + j);
    }
    acc = acc + load_mixed(t * params.row_pitch + c) * load_conv_w(c * p.Kc + (p.Kc - 1));
    return silu(acc);
}

// The state the NEXT dispatch will read: the last `Kc` tokens ending at `t`.
// Only the final token of the prompt calls this.
fn write_conv_prefill(slot: i32, t: i32, start: i32, c: i32) {
    for (var j = 0; j < p.Kc; j = j + 1) {
        let idx = t - (p.Kc - 1) + j;
        var v = 0.0;
        if (idx >= start) {
            v = load_mixed(idx * params.row_pitch + c);
        } else {
            v = conv_state[(slot * p.Kc + p.Kc + (idx - start)) * p.conv_dim + c];
        }
        new_conv_state[(slot * p.Kc + j) * p.conv_dim + c] = v;
    }
}
//#endif

//#if defined(PIE_RECURRENT)
@compute @workgroup_size(32, 4)
//#else
@compute @workgroup_size(32)
//#endif
// `gid` is DECLARED ONLY WHERE IT IS READ, which is the recurrent arm's
// `dv_idx`. Declaring it unconditionally cost nothing at run time -- naga keeps
// an unread builtin and the driver binds nothing for it -- but it cost a
// reflection fact: `Declared::grid_axes` reports a wholly unused
// `global_invocation_id` as reading ALL THREE axes, so the non-recurrent
// variants claimed to read `gid.y` while their grid states `1` there, and
// `driver-wgpu::geometry`'s flat-axis sweep read that as three kernels whose
// every index past the first is never written. The sweep was right to ask; the
// declaration was what lied.
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
//#if defined(PIE_RECURRENT)
        @builtin(global_invocation_id) gid: vec3<u32>,
//#endif
        @builtin(workgroup_id) wid: vec3<u32>) {
//#if defined(PIE_RECURRENT)
    let n = i32(wid.z);
    let b_idx = n / p.Hv;
    let hv_idx = n % p.Hv;
    let dk_idx = i32(lid.x);
    let dv_idx = i32(gid.y);
    let n_per_t = p.Dk / 32;
    // See the file header: the guard is on the stores, because `row_sum32`
    // barriers below and a return in front of a barrier is a hang.
    let mine = dv_idx < p.Dv;
//#if defined(PIE_SLOTTED)
    let slot = i32(slot_ids[b_idx]);
//#else
    let slot = b_idx;
//#endif
    // The v channel is the one thing the prep arm did NOT stage: it is per
    // value channel, so staging it would cost a `Dv`-wide scratch to save a
    // conv this kernel is already shaped to do.
    let vval = convsilu_token(slot, b_idx, p.v_off + hv_idx * p.Dv + dv_idx);
    let state_base = ((slot * p.Hv + hv_idx) * p.Dv + dv_idx) * p.Dk;
    let pk_base = n * p.Dk;
    var kv = 0.0;
    var st: array<f32, 8>;
    var q: array<f32, 8>;
    var k: array<f32, 8>;
    for (var i = 0; i < n_per_t; i = i + 1) {
        let d = n_per_t * dk_idx + i;
        // Already normalised by the prep arm; this arm never re-normalises,
        // which is exactly what the split buys.
        q[i] = pre_q[pk_base + d];
        k[i] = pre_k[pk_base + d];
        st[i] = rstate[state_base + d] * pre_gate[2 * n + 0];
        kv = kv + st[i] * k[i];
    }
    kv = row_sum32(lid.x, lid.y, kv);
    let delta = (vval - kv) * pre_gate[2 * n + 1];
    var outv = 0.0;
    for (var i = 0; i < n_per_t; i = i + 1) {
        let d = n_per_t * dk_idx + i;
        st[i] = st[i] + k[i] * delta;
        outv = outv + st[i] * q[i];
        if (mine) {
            rstate[state_base + d] = st[i];
        }
    }
    outv = row_sum32(lid.x, lid.y, outv);
    if (dk_idx == 0 && mine) {
        store_core_out((b_idx * p.Hv + hv_idx) * p.Dv + dv_idx, outv);
    }
    if (mine) {
        write_conv_token(slot, b_idx, p.v_off + hv_idx * p.Dv + dv_idx);
    }
//#elif defined(PIE_PREFILL)
    let n = i32(wid.z);
    let t = n / p.Hv;
    let hv_idx = n % p.Hv;
    let rep = p.Hv / p.Hk;
    let hk_idx = hv_idx / rep;
    let hk_first = (hv_idx % rep) == 0;
    // THIS ROW'S SEAT, and the row where it changes IS the request boundary.
    //
    // A fire holds whatever the batcher packed and `slot_ids` is per ROW, so
    // two requests in one prefill are two seats and no second table has to say
    // where one ends. Reading `slot_ids[0]` made the whole rectangle one
    // sequence: every request after the first convolved over its
    // predecessor's tokens and carried its recurrent state, and the first one
    // lost its history the moment only the LAST request's last token wrote
    // `new_conv_state`. `kernels-metal`'s prep records the same three defects.
    //
    // `start` is this request's first row, found by walking back while the
    // seat holds. Bounded by `Kc - 1` because that is as far as a convolution
    // window reaches, so nothing here is linear in the prompt -- and with one
    // seat it lands on `max(0, t - (Kc - 1))`, which makes every branch below
    // resolve exactly as it did when this read `slot_ids[0]`.
    let slot = i32(slot_ids[t]);
    var start = t;
    for (var back = 0; back < p.Kc - 1 && start > 0; back = back + 1) {
        if (i32(slot_ids[start - 1]) != slot) {
            break;
        }
        start = start - 1;
    }
    let dk_idx = i32(lid.x);
    let n_per_t = p.Dk / 32;
    // EVERY SCRATCH ROW IS PACKED AT ITS OWN WIDTH, and `row_pitch` describes
    // the in-projection alone. One shared pitch has to clear the widest of
    // them, and `pre_gate` is WIDER than `mixed` on any stack whose value width
    // reaches its key width: `kernels-metal`'s note gives qwen3-next asking
    // `2*Hv + Hv*Dv = 8320` floats against a `conv_dim` of 8192, so a shared
    // pitch writes each token's v channels over the next token's gates.
    //
    // This body divided `row_pitch` by two and strode all three by it, which
    // is the ABI `ssm/gdn_prep.metal` used to have and repaired. Its repair
    // records the symptom: *"a three-token prefill came back all-NaN, a
    // four-token one did not, and the difference was only which garbage the
    // arena happened to hold."*
    let qk_pitch = p.Hv * p.Dk;
    let g_pitch = 2 * p.Hv + p.Hv * p.Dv;
    var qraw: array<f32, 8>;
    var kraw: array<f32, 8>;
    var qsq = 0.0;
    var ksq = 0.0;
    for (var i = 0; i < n_per_t; i = i + 1) {
        let d = n_per_t * dk_idx + i;
        qraw[i] = convsilu_prefill(slot, t, start, p.q_off + hk_idx * p.Dk + d);
        kraw[i] = convsilu_prefill(slot, t, start, p.k_off + hk_idx * p.Dk + d);
        qsq = qsq + qraw[i] * qraw[i];
        ksq = ksq + kraw[i] * kraw[i];
    }
    // `inv_sqrt_dk` rides on the query side only: the product q.k must be
    // scaled once, not twice.
    let qinv = p.inv_sqrt_dk / sqrt(row_sum32(lid.x, lid.y, qsq) + p.eps);
    let kinv = 1.0 / sqrt(row_sum32(lid.x, lid.y, ksq) + p.eps);
    let qk_row = t * qk_pitch;
    let g_row = t * g_pitch;
    for (var i = 0; i < n_per_t; i = i + 1) {
        let d = n_per_t * dk_idx + i;
        // Staged by VALUE head, not key head: the scan reads `hv_idx * Dk`, so
        // a group-query model writes each key head's q/k once per value head
        // that shares it. That is the redundancy the split accepts in exchange
        // for the scan never touching the conv.
        pre_q[qk_row + hv_idx * p.Dk + d] = qraw[i] * qinv;
        pre_k[qk_row + hv_idx * p.Dk + d] = kraw[i] * kinv;
    }
    if (dk_idx == 0) {
        // `a` and `b` are the IN-PROJECTION's own outputs, one scalar per value
        // head, so their rows are `Hv` wide -- not `row_pitch`, which belongs
        // to `mixed` alone. `gdn_prep_slotted` and `gdn_core` both index
        // `b_idx * Hv + hv_idx`, and this arm read them at the prompt's pitch.
        let gate_at = t * p.Hv + hv_idx;
        let ad = load_a_gate(gate_at) + load_dt_bias(hv_idx);
        // softplus in the form that does not overflow: `log(1 + e^x)` is
        // infinite by x = 89 and this is not.
        let sp = max(ad, 0.0) + log(1.0 + exp(-abs(ad)));
        pre_gate[g_row + 2 * hv_idx + 0] = exp(-exp(A_log[hv_idx]) * sp);
        pre_gate[g_row + 2 * hv_idx + 1] = 1.0 / (1.0 + exp(-load_b_gate(gate_at)));
    }
    // `Dv` is not bounded by the workgroup width, so this strides: a body that
    // gave each of the 32 lanes one channel would stage a quarter of a
    // `Dv = 128` head and leave the rest holding the last prompt's values.
    for (var dv = dk_idx; dv < p.Dv; dv = dv + 32) {
        pre_gate[g_row + 2 * p.Hv + hv_idx * p.Dv + dv] =
            convsilu_prefill(slot, t, start, p.v_off + hv_idx * p.Dv + dv);
    }
    // Only each REQUEST'S last token carries that request's history forward.
    // Every earlier token's window is already in `mixed`. Returning here is
    // safe in a way it would not be twenty lines up: both `row_sum32` calls
    // are done, and the condition is uniform because one workgroup owns one
    // `(t, hv)` pair.
    if (t != params.n_scan - 1 && i32(slot_ids[t + 1]) == slot) {
        return;
    }
    if (hk_first) {
        for (var i = 0; i < n_per_t; i = i + 1) {
            let d = n_per_t * dk_idx + i;
            write_conv_prefill(slot, t, start, p.q_off + hk_idx * p.Dk + d);
            write_conv_prefill(slot, t, start, p.k_off + hk_idx * p.Dk + d);
        }
    }
    for (var dv = dk_idx; dv < p.Dv; dv = dv + 32) {
        write_conv_prefill(slot, t, start, p.v_off + hv_idx * p.Dv + dv);
    }
//#else
    let n = i32(wid.z);
    let b_idx = n / p.Hv;
    let hv_idx = n % p.Hv;
    let rep = p.Hv / p.Hk;
    let hk_idx = hv_idx / rep;
    // Only the first value head of a group writes the shared q/k conv state
    // back; the others would write the same channels and the last would win by
    // luck.
    let hk_first = (hv_idx % rep) == 0;
    let dk_idx = i32(lid.x);
    let n_per_t = p.Dk / 32;
//#if defined(PIE_SLOTTED)
    let slot = i32(slot_ids[b_idx]);
//#else
    let slot = b_idx;
//#endif
    var qraw: array<f32, 8>;
    var kraw: array<f32, 8>;
    var qsq = 0.0;
    var ksq = 0.0;
    for (var i = 0; i < n_per_t; i = i + 1) {
        let d = n_per_t * dk_idx + i;
        qraw[i] = convsilu_token(slot, b_idx, p.q_off + hk_idx * p.Dk + d);
        kraw[i] = convsilu_token(slot, b_idx, p.k_off + hk_idx * p.Dk + d);
        qsq = qsq + qraw[i] * qraw[i];
        ksq = ksq + kraw[i] * kraw[i];
    }
    let qinv = p.inv_sqrt_dk / sqrt(row_sum32(lid.x, lid.y, qsq) + p.eps);
    let kinv = 1.0 / sqrt(row_sum32(lid.x, lid.y, ksq) + p.eps);
    // The decode scratch is indexed by the flat (batch, head) pair rather than
    // by a token row: one token per launch, so there is no row pitch.
    let pk_base = n * p.Dk;
    for (var i = 0; i < n_per_t; i = i + 1) {
        let d = n_per_t * dk_idx + i;
        pre_q[pk_base + d] = qraw[i] * qinv;
        pre_k[pk_base + d] = kraw[i] * kinv;
    }
    if (dk_idx == 0) {
        let ad = load_a_gate(b_idx * p.Hv + hv_idx) + load_dt_bias(hv_idx);
        let sp = max(ad, 0.0) + log(1.0 + exp(-abs(ad)));
        pre_gate[2 * n + 0] = exp(-exp(A_log[hv_idx]) * sp);
        pre_gate[2 * n + 1] = 1.0 / (1.0 + exp(-load_b_gate(b_idx * p.Hv + hv_idx)));
    }
    // The v channels are NOT staged here: the recurrent arm does its own v
    // conv, per value channel, because staging them would cost a `Dv`-wide
    // scratch write this arm has no lanes for.
    if (hk_first) {
        for (var i = 0; i < n_per_t; i = i + 1) {
            let d = n_per_t * dk_idx + i;
            write_conv_token(slot, b_idx, p.q_off + hk_idx * p.Dk + d);
            write_conv_token(slot, b_idx, p.k_off + hk_idx * p.Dk + d);
        }
    }
//#endif
}
//#endif

// pie:instantiate gdn_core_recurrent_bfloat16 PIE_RECURRENT=1
// pie:instantiate gdn_core_recurrent_slotted_bfloat16 PIE_RECURRENT=1 PIE_SLOTTED=1
// pie:instantiate gdn_prep_bfloat16
// pie:instantiate gdn_prep_slotted_bfloat16 PIE_SLOTTED=1
// pie:instantiate gdn_prep_prefill_bfloat16 PIE_PREFILL=1
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_16_v_1 PIE_SCAN=1 PIE_LANES=16 PIE_VROWS=1
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_16_v_2 PIE_SCAN=1 PIE_LANES=16 PIE_VROWS=2
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_16_v_4 PIE_SCAN=1 PIE_LANES=16 PIE_VROWS=4
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_32_v_2 PIE_SCAN=1 PIE_LANES=32 PIE_VROWS=2
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_32_v_4 PIE_SCAN=1 PIE_LANES=32 PIE_VROWS=4
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_32_v_8 PIE_SCAN=1 PIE_LANES=32 PIE_VROWS=8
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_4_v_1 PIE_SCAN=1 PIE_LANES=4 PIE_VROWS=1
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_8_v_1 PIE_SCAN=1 PIE_LANES=8 PIE_VROWS=1
// pie:instantiate gdn_core_recurrent_prefill_bfloat16_l_8_v_2 PIE_SCAN=1 PIE_LANES=8 PIE_VROWS=2
