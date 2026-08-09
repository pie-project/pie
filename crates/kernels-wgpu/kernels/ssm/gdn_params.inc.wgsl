// The shared host/shader ABI for the GDN kernels.
//
// One struct, filled by the host once and read by all three shaders. The field
// ORDER is the contract: `gdn_params.h` on the Metal side, `gdn_params.glsl` on
// the Vulkan one and this file are the same eleven fields in the same order,
// and a field moved here reads the wrong number there with nothing to say so.
//
// It rides in `@group(0)` as a storage buffer rather than in the `@group(1)`
// uniform block, because every GDN row states it as a BUFFER operand -- a
// pointer to where the host already assembled the numbers -- and not as a list
// of scalars. WGSL's storage layout for a struct of 4-byte scalars is std430's,
// so this is byte-for-byte the block the Vulkan sibling reads.
//
// `Dk` and `Dv` are the key and value head dimensions and are NOT the same
// number in a GDN checkpoint (128 and 128 in the ones the tree has seen, which
// is exactly why swapping them would go unnoticed until one that differs
// arrives). `Hk` and `Hv` likewise: `Hv / Hk` is the group-query replication
// factor, and the conv-state writeback is done once per KEY head, by the first
// value head of each group.
struct GdnCoreParams {
    Dk: i32,
    Dv: i32,
    Hk: i32,
    Hv: i32,
    // The channel count of the fused qkv conv input: `Hk*Dk + Hk*Dk + Hv*Dv`
    // plus whatever the projection carries beside it, which is why the three
    // offsets below are stated rather than derived.
    conv_dim: i32,
    // The causal conv width. `Kc - 1` past taps come from the conv state and
    // the last one is the current token.
    Kc: i32,
    q_off: i32,
    k_off: i32,
    v_off: i32,
    eps: f32,
    inv_sqrt_dk: f32,
}
