// Per-layer scalar multiply: `out = x * scalar[0]`.
//
// gemma4 scales each decoder layer's output by a learned per-layer scalar of
// shape `[1]`, broadcast over the hidden width. The scalar is read from a
// device buffer rather than stated, because which layer is running is the
// FIRE's business and a resident slot stays resident.
//
// ## The grid is the extent, and the block that said otherwise is gone
//
// `LayerScalarParams` was bound and NOT read, and that is the Metal port's
// finding kept rather than tidied away: the field was the hidden width -- ONE
// ROW -- while `LaunchRule::Elementwise` dispatches `width * rows`, so reading
// it as a bound returned every row after the first immediately and left them
// holding whatever the arena had. The struct then STAYED bound, one storage
// binding for one dead word, so the row's operand list -- and therefore the
// bind group layout -- was the same on all three backends.
//
// It is deleted now on all three, which is the same thing said a cheaper way.
// A block a routine forwards has to be staged, descriptor-ed and given a slot
// in the numbering on every plane that binds it; a block no shader reads earns
// none of that. `layer_scalar_mul` states no `Const` mark in its place, since
// the field it would carry is the one this section explains away. The
// statement may still carry the word -- `model-dsl` states it -- and nothing
// now reads it, which is the honest arrangement: an unread word in a run costs
// four bytes, an unread BINDING costs an entry in every layout that declares
// it.
//
// The real bound is `arrayLength(&out_)`: `dispatch_workgroups` counts
// WORKGROUPS, so the host rounds the group count up and the last group runs
// past the data. An overshoot has to be harmless; an UNDERSHOOT writes nothing,
// reads back as the zeros the pool was born with, and completes -- which is the
// host's problem and not this file's.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> scalar: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;

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
