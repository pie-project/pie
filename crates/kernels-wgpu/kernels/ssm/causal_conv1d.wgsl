// The short causal depthwise convolution over the recurrent conv slab.
//
// Two entrypoints, one per point. `causal_conv1d` is the decode step -- one
// input row per request, the rest of the window in the slab -- and
// `causal_conv1d_chunked` is the prefill window, a CSR range of rows per
// request with only the leading taps off the slab. They are two kernels with
// the same arithmetic written twice, because the step's window is entirely in
// the slab and the chunk's is mostly in `x`; folding them would put a branch
// on the token axis inside the innermost loop of both.
//
// ── THE SLAB IS `[K, C]`, AND `C` IS THE FAST AXIS ─────────────────────────
//
// A slot's conv state is `conv_k * conv_dim` floats -- K ROWS of C channels,
// oldest row first, addressed `state[k * C + c]`.
// `driver-wgpu`'s `resources::Recurrent::conv_bytes_per_slot` is exactly that
// product, so the rectangle indexed here is the one the pool allocates.
//
// K rows is the RECTANGLE and not the live window. Between fires only `K - 1`
// of the rows carry taps the next token convolves over: the step reads rows
// `1 .. K-1` plus the incoming column, shifts every row down one, and lands
// the new column at row `K - 1`. Row 0 is where the shift's tail goes -- read
// once, by the token that arrives before it is overwritten, and never again. A
// declaration that said `K - 1` rows would be stating the live WINDOW, which
// is a different number from the rectangle, and the two differ by exactly the
// row this paragraph exists to keep.
//
// The arithmetic is `causal_conv1d_update_batched` and
// `causal_conv1d_prefill_batched` in `kernels-cuda/kernels/ssm/causal_conv1d.cuh`,
// which is where the numeric contract was measured:
//
//     y[t, c] = silu( sum_{k=0..K-1} W[c, k] * x[t - K + 1 + k, c] )
//
// with `x[t < 0, c]` read from the slab at row `K + t`. There is no bias:
// `ssm.causal_conv1d` declares none and the cuda claim body passes a null one.
//
// ── WHY THE SHIFTED WINDOW LANDS IN A SECOND PLANE ────────────────────────
//
// `conv_state` is read and `new_conv_state` is written, never the other way
// and never alternating. `driver-wgpu`'s `RecurrentPool` allocates two conv
// planes per layer and `carry_back` copies the written one over the read one
// after the fire retires; its own docs record what happened when nothing did.
// The cuda kernels shift in place because one BLOCK owns a channel for the
// whole launch, so the shift is over rows that block alone read. A WebGPU
// dispatch makes no such promise across workgroups, and a channel whose taps
// are being read by the token after it cannot be the channel a shift lands on.
//
// A request the fire does not name keeps whatever its `new_conv_state` rows
// held, which the previous carry-back made equal to its `conv_state` rows --
// so the copy is an identity for it. That invariant is why the chunked kernel
// may leave an empty window alone rather than copying it forward by hand.
//
// ── ONE INVOCATION OWNS A CHANNEL PAIR ─────────────────────────────────────
//
// `y` is bf16 and WGSL's smallest addressable storage element is four bytes,
// so a lane that owned ONE channel would have to read-modify-write a word its
// neighbour owns the other half of. Every invocation therefore owns the pair
// `(2i, 2i + 1)` and writes a whole word with `pie_pack_bf16`, which is
// `norm/residual_add.wgsl`'s rule and the reason the claim body refuses an odd
// channel count. The slab is f32 and needs none of this; only `x` and `y` do.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> conv_state: array<f32>;
@group(0) @binding(3) var<storage, read_write> new_conv_state: array<f32>;
@group(0) @binding(4) var<storage, read_write> slots: array<u32>;
//#if defined(PIE_CHUNKED)
@group(0) @binding(5) var<storage, read_write> indptr: array<i32>;
@group(0) @binding(6) var<storage, read_write> y: array<u32>;
//#else
@group(0) @binding(5) var<storage, read_write> y: array<u32>;
//#endif

struct Params {
    channels: i32,
    conv_width: i32,
}
@group(1) @binding(0) var<uniform> params: Params;

// SiLU in the one spelling every plane's conv shares: `pie::ssm::silu_f`.
fn silu(z: f32) -> f32 {
    return z / (1.0 + exp(-z));
}

// One bf16 out of a packed `array<u32>`, by its ELEMENT index.
fn tap_of(i: i32) -> f32 {
    let at = u32(i);
    return pie_bf16_at(x[at >> 1u], at);
}

fn weight_of(i: i32) -> f32 {
    let at = u32(i);
    return pie_bf16_at(weight[at >> 1u], at);
}

//#if defined(PIE_CHUNKED)

// Prefill window: rows `indptr[r] .. indptr[r + 1]` of `x` for request `r`,
// with the taps before the window read off the slab and the trailing `K` rows
// of `[slab window | x window]` persisted back into the second plane.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chans = params.channels;
    let taps = params.conv_width;
    let lo = 2 * i32(gid.x);
    if (lo + 1 >= chans) { return; }
    let r = i32(gid.y);

    let begin = indptr[r];
    let end = indptr[r + 1];
    // An empty window leaves BOTH planes alone; see the header.
    if (end <= begin) { return; }
    let span = end - begin;

    // Every token of a request sits in the same seat, so the window's first
    // row names it.
    let slab = i32(slots[u32(begin)]) * taps * chans;
    let w_lo = lo * taps;
    let w_hi = w_lo + taps;

    for (var t = 0; t < span; t = t + 1) {
        var acc_lo = 0.0;
        var acc_hi = 0.0;
        for (var k = 0; k < taps; k = k + 1) {
            let src = t - (taps - 1) + k;
            var v_lo: f32;
            var v_hi: f32;
            if (src < 0) {
                let at = slab + (taps + src) * chans + lo;
                v_lo = conv_state[u32(at)];
                v_hi = conv_state[u32(at + 1)];
            } else {
                let at = (begin + src) * chans + lo;
                v_lo = tap_of(at);
                v_hi = tap_of(at + 1);
            }
            acc_lo = acc_lo + v_lo * weight_of(w_lo + k);
            acc_hi = acc_hi + v_hi * weight_of(w_hi + k);
        }
        let out = (begin + t) * chans + lo;
        y[u32(out) >> 1u] = pie_pack_bf16(silu(acc_lo), silu(acc_hi));
    }

    // The trailing `K` rows of `[slab window | x window]`, oldest first, which
    // is where a follow-up step resumes from.
    for (var s = 0; s < taps; s = s + 1) {
        let src = span - taps + s;
        let dst = slab + s * chans + lo;
        if (src < 0) {
            let at = slab + (taps + src) * chans + lo;
            new_conv_state[u32(dst)] = conv_state[u32(at)];
            new_conv_state[u32(dst + 1)] = conv_state[u32(at + 1)];
        } else {
            let at = (begin + src) * chans + lo;
            new_conv_state[u32(dst)] = tap_of(at);
            new_conv_state[u32(dst + 1)] = tap_of(at + 1);
        }
    }
}

//#else

// Decode step: one input row per request, the other `K - 1` taps off the slab.
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chans = params.channels;
    let taps = params.conv_width;
    let lo = 2 * i32(gid.x);
    if (lo + 1 >= chans) { return; }
    let r = i32(gid.y);

    let slab = i32(slots[gid.y]) * taps * chans;
    let w_lo = lo * taps;
    let w_hi = w_lo + taps;
    let arriving = r * chans + lo;
    let fresh_lo = tap_of(arriving);
    let fresh_hi = tap_of(arriving + 1);

    // Rows `1 .. K-1` of the slab, then the arriving column at tap `K - 1`.
    var acc_lo = fresh_lo * weight_of(w_lo + taps - 1);
    var acc_hi = fresh_hi * weight_of(w_hi + taps - 1);
    for (var k = 0; k + 1 < taps; k = k + 1) {
        let at = slab + (k + 1) * chans + lo;
        acc_lo = acc_lo + conv_state[u32(at)] * weight_of(w_lo + k);
        acc_hi = acc_hi + conv_state[u32(at + 1)] * weight_of(w_hi + k);
    }
    y[u32(arriving) >> 1u] = pie_pack_bf16(silu(acc_lo), silu(acc_hi));

    // Shift every row down one and land the arriving column at row `K - 1`.
    for (var k = 0; k + 1 < taps; k = k + 1) {
        let dst = slab + k * chans + lo;
        let src = dst + chans;
        new_conv_state[u32(dst)] = conv_state[u32(src)];
        new_conv_state[u32(dst + 1)] = conv_state[u32(src + 1)];
    }
    let last = slab + (taps - 1) * chans + lo;
    new_conv_state[u32(last)] = fresh_lo;
    new_conv_state[u32(last + 1)] = fresh_hi;
}

//#endif

// pie:instantiate causal_conv1d_bfloat16
// pie:instantiate causal_conv1d_chunked_bfloat16 PIE_CHUNKED=1
