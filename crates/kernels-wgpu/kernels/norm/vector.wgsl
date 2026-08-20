// Weightless RMSNorm: `out = x / rms(x)`.
//
// gemma's value norm, and the absence of a GAIN is the whole difference from
// `norm/rms.wgsl` -- there is no weight buffer, so the two storage bindings are
// `x` and `out` and the norm weight's is simply not there. The arithmetic is
// `norm/rms.wgsl`'s with `gain_at` deleted: fp32 accumulate, one workgroup per
// row, one bf16 round on the store.
//
// ## The two scalars are MARKS, not a struct
//
// They arrived as `VNormParams { eps, axis_size }` on a `@group(0)` storage
// binding -- MLX's Metal layout, ported here through `norm/rms_params.h` and
// its Slang twin -- so `norm::vnorm_single_row` forwarded `ctx.params()` whole
// and no signature could name either number. It states `eps: Const<f32>` and
// `axis_size: Const<i32>` now, which is words 0 and 1 of the SAME statement
// run reached by index instead of by struct field, and
// `driver-wgpu::lowering::routine::bind` packs them into the `@group(1)` block
// below in the order the BODY passes them. The struct's field order is that
// order, because the struct's field order was the statement's.
//
// `axis_size` is stated as `u32` here and marked `Const<i32>` there, exactly
// as `norm/rms.wgsl`'s `RmsParams.axis_size` is against
// `rms_strided_head_row`'s `axis`. The run is a `Vec<u32>` and the BITS are
// the value; what the mark's Rust type decides is what the BODY may do with
// the number, and a body sizing a launch wants a signed extent it can refuse.
//
// `axis_size` is why the launch is one workgroup per AXIS and not one per row:
// this kernel gives workgroup `x` the span `x * axis_size`. A value norm's
// axis is the HEAD and its row is every head, so a launch that took the fire's
// width for the axis would reduce the whole row as one, which is not a coarser
// normalization but a different number in every channel.

//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

// One name for the attribute below and for the width `pie_inv_rms` folds: a
// body that declared 256 and reduced over 128 would norm by half a row.
const PIE_LANES = 256u;

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
// Atomic for the odd-`axis_size` edge alone; `store_half` says why, and the
// host binds the same read_write storage buffer either way.
@group(0) @binding(1) var<storage, read_write> out_: array<atomic<u32>>;

// The fields are the body's argument order, which is the statement's run
// order: `eps` at word 0 and `axis_size` at word 1, exactly where
// `VNormParams` had them. A block whose fields were transposed would divide
// every channel by a head dimension read as a float and span the row by an
// epsilon read as a count -- both finite, neither refused.
struct Params {
    eps: f32,
    axis_size: u32,
}
@group(1) @binding(0) var<uniform> params: Params;

// The half-index split. A word and an index, not a pointer and an index: core
// WGSL allows a pointer parameter only in the `function`, `private` and
// `workgroup` address spaces, so `common/bf16.inc.wgsl` cannot take the buffer
// and every binding gets its own two-line reader.
fn x_at(i: u32) -> f32 {
    return pie_bf16_at(x[i >> 1u], i);
}

// One bf16 of a word this invocation does not own outright.
//
// Only an odd `axis_size` reaches this -- a head dim of 128 does not -- and it
// puts one head's last channel and the next head's first channel in the SAME
// word, written by two different workgroups. A read-modify-write would keep
// whichever landed second; the device-scoped compare-exchange keeps both, and
// retries the spurious failure `...Weak` is permitted. Same pattern, and same
// reason, as `norm/rms.wgsl` and `kernels/quant/qmm_t.wgsl`.
fn store_half(i: u32, value: f32) {
    let at = i >> 1u;
    var old = atomicLoad(&out_[at]);
    loop {
        let res = atomicCompareExchangeWeak(&out_[at], old, pie_bf16_into(old, i, value));
        if (res.exchanged) { break; }
        old = res.old_value;
    }
}

@compute @workgroup_size(PIE_LANES)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    let axis = params.axis_size;
    let base = wg.x * axis;
    let span = PIE_LANES * u32(N_READS);

    var acc = 0.0;
    for (var start = lane * u32(N_READS); start < axis; start = start + span) {
        for (var i = 0u; i < u32(N_READS); i = i + 1u) {
            // A head dim that is not a multiple of `span` leaves the last chunk
            // ragged, so the tail is tested per element rather than assumed.
            if (start + i < axis) {
                let xi = x_at(base + start + i);
                acc = acc + xi * xi;
            }
        }
    }
    // Reached by every lane, including one whose chunk was entirely past the
    // end: `pie_inv_rms` barriers inside, and an early return in front of a
    // barrier is a hang rather than a wrong number.
    let inv = pie_inv_rms(lane, PIE_LANES, acc, axis, params.eps);

    // One lane per WORD, which is the smallest thing WGSL can store. A word
    // wholly inside the row has one writer and goes out whole; the two edge
    // cases can be shared with a neighbouring row and go through `store_half`.
    // The bounds are absolute word indices, so a row that does not begin on a
    // word boundary is still addressed correctly.
    let first = base >> 1u;
    let end = (base + axis + 1u) >> 1u;
    for (var word = first + lane; word < end; word = word + PIE_LANES) {
        let lo = word * 2u;
        let hi = lo + 1u;
        let has_lo = lo >= base && lo < base + axis;
        let has_hi = hi < base + axis;
        if (has_lo && has_hi) {
            // Both halves are this row's: one writer, one write, no hazard.
            atomicStore(&out_[word], pie_pack_bf16(
                x_at(lo) * inv,
                x_at(hi) * inv,
            ));
        } else if (has_hi) {
            // An odd `axis_size` is the only way a row starts in a word's upper
            // half; the previous row owns the lower one and is being written by
            // another workgroup right now, so this half goes through the CAS.
            store_half(hi, x_at(hi) * inv);
        } else if (has_lo) {
            // The mirror case, and the next row is the other writer. Tested
            // rather than left as the remaining case, so a word range computed
            // wrong writes nothing instead of a neighbour's element.
            store_half(lo, x_at(lo) * inv);
        }
    }
}

// pie:instantiate vnorm_single_row_bfloat16 N_READS=4
