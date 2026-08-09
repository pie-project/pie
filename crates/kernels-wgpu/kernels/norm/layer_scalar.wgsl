// Per-layer scalar multiply: `out = x * scalar[0]`.
//
// gemma4 scales each decoder layer's output by a learned per-layer scalar of
// shape `[1]`, broadcast over the hidden width. The scalar is read from a
// device buffer rather than stated, because which layer is running is the
// FIRE's business and a resident slot stays resident.
//
// ## The grid is the extent
//
// `LayerScalarParams` is bound and NOT read, and that is the Metal port's
// finding kept rather than tidied away: the field was the hidden width -- ONE
// ROW -- while `LaunchRule::Elementwise` dispatches `width * rows`, so reading
// it as a bound returned every row after the first immediately and left them
// holding whatever the arena had. The struct stays bound so the row's operand
// list, and therefore the bind group layout, is the same on all three backends.
//
// The real bound is `arrayLength(&out_)`: `dispatch_workgroups` counts
// WORKGROUPS, so the host rounds the group count up and the last group runs
// past the data. An overshoot has to be harmless; an UNDERSHOOT writes nothing,
// reads back as the zeros the pool was born with, and completes -- which is the
// host's problem and not this file's.

//#include "common/bf16.inc.wgsl"

struct LayerScalarParams {
    // Was the hidden width. See above: bound for the ABI, never read.
    hidden: u32,
}

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> scalar: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;
@group(0) @binding(3) var<storage, read> params: LayerScalarParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // One invocation per WORD, which is TWO bf16 values -- the launch's x
    // extent is half the element count, as it is in `norm/residual_add.wgsl`.
    // Writing a whole word is what keeps this store off a read-modify-write
    // that the neighbouring element's invocation would race.
    let i = gid.x;
    if (i >= arrayLength(&out_)) { return; }

    let s = pie_bf16_at(scalar[0], 0u);
    let v = x[i];
    out_[i] = pie_pack_bf16(
        pie_bf16_to_f32(v & 0xffffu) * s,
        pie_bf16_to_f32(v >> 16u) * s,
    );
}

// pie:instantiate layer_scalar_mul_bfloat16
