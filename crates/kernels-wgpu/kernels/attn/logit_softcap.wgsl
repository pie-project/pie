// gemma's final logit softcap: `cap * tanh(x / cap)`, so no logit runs away.
//
// A statement and not a mode -- a deployment without one names nothing here,
// rather than passing an infinite cap and paying for a `tanh` that is the
// identity. The row's third operand is a `Buf`, so the cap arrives as a STRUCT
// a storage buffer binds and this file declares no `@group(1)`; the old
// per-row vocabulary bound stays unused, because the elementwise launch already
// is the whole extent.

//#include "common/bf16.inc.wgsl"
//#include "common/math.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> logits: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;

struct SoftcapParams { cap: f32, unused: u32 }
@group(0) @binding(2) var<storage, read_write> params: SoftcapParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    // The tail guard is the buffer's own length: `dispatch_workgroups` rounds
    // the group count up, so the last group runs past the data. An overshoot
    // has to be harmless; an undershoot is the host's problem and the silent
    // one.
    if (i >= arrayLength(&out_)) { return; }

    // A whole word at a time -- two logits -- because a half-word store is a
    // read-modify-write of a word the neighbouring invocation also owns.
    let x = logits[i];
    let cap = params.cap;
    out_[i] = pie_pack_bf16(
        cap * pie_tanh(pie_bf16_to_f32(x & 0xffffu) / cap),
        cap * pie_tanh(pie_bf16_to_f32(x >> 16u) / cap),
    );
}

// pie:instantiate logit_softcap_bfloat16
