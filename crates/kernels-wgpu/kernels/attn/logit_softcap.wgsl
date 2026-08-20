// gemma's final logit softcap: `cap * tanh(x / cap)`, so no logit runs away.
//
// A statement and not a mode -- a deployment without one names nothing here,
// rather than passing an infinite cap and paying for a `tanh` that is the
// identity.
//
// THE CAP IS A MARK, NOT A STRUCT. It used to arrive as `SoftcapParams { cap,
// unused }` on a `@group(0)` storage binding -- MLX's ABI, carried here
// through Metal and Vulkan -- with a second word nothing read, held only so
// the struct kept its size. The routine states `cap: Const<f32>` now and
// `driver-wgpu::lowering::routine::bind` packs it into the `@group(1)` block,
// which is the same word of the same `Lowered::params` run reached by its
// index instead of by a struct field. The old per-row vocabulary bound stays
// gone, because the elementwise launch already is the whole extent.

//#include "common/bf16.inc.wgsl"
//#include "common/math.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> logits: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;

struct Params { cap: f32 }
@group(1) @binding(0) var<uniform> params: Params;

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
