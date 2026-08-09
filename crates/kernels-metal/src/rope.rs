//! Rotary embeddings.
//!
//! Four spellings of the schedule (`neox`, `freqs`, `prop`, and the strided
//! form), each in a decode and a multi-batch shape. The `freqs` pair reads a
//! host-computed table, which is what llama-3.1's wavelength ramp needs.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 1 in rope.metal
    // IN PLACE on ONE tensor: buffer 0 is both the input and the result, and
    // the template is `rope_neox_decode`.
    //
    // Which makes a TEXT gap visible that no row can close. `dsl::metal::rope`
    // states one launch carrying q and k — two inputs and two results — and
    // this kernel rotates one buffer. The statement should be two, one per
    // tensor, and until it is, the second tensor is not rotated at all.
    kernel!(neox_decode "neox_decode", file = Some("rope/neox.metal"), launch = kernels::LaunchRule::Rope,
        operands = kernels::operands![
            x: BufMut <- kernels::Source::Out(0),
            position: I32s <- kernels::Source::Positions,
            scale: F32 <- kernels::Source::ParamF32(0),
            base: F32 <- kernels::Source::ParamF32(1),
            head_dim: I32 <- kernels::Source::Param(2),
        ],
        axes = &[BF16]),
    // 1 in rope.metal
    // The rotation a deployment that RESCALES its ladder takes. Same body as
    // `neox_decode` with the frequencies read from a buffer instead of raised
    // from a base -- which is the only form that can express llama-3's
    // piecewise rescaling or YaRN's, because neither is a base.
    kernel!(neox_freqs_decode "neox_freqs_decode", file = Some("rope/neox.metal"),
        launch = kernels::LaunchRule::Rope,
        operands = kernels::operands![
            x: BufMut <- kernels::Source::Out(0),
            position: I32s <- kernels::Source::Positions,
            scale: F32 <- kernels::Source::ParamF32(0),
            inv_freq: Buf <- kernels::Source::RopeFrequencies,
            head_dim: I32 <- kernels::Source::Param(1),
            // YaRN's attention-temperature correction. One for a deployment
            // that has none, which is every llama-3 one -- its rescaling is in
            // the frequencies and not in a gain.
            mscale: F32 <- kernels::Source::ParamF32(2),
        ],
        axes = &[BF16]),
    // 1 in rope.metal
    kernel!(neox_freqs_mb "neox_freqs_mb", axes = &[BF16]),
    // 1 in rope.metal
    // The batched form, and the same shape: one tensor, per-token positions.
    kernel!(neox_mb "neox_mb", file = Some("rope/neox.metal"), launch = kernels::LaunchRule::Rope,
        operands = kernels::operands![
            x: BufMut <- kernels::Source::Out(0),
            position: I32s <- kernels::Source::Positions,
            scale: F32 <- kernels::Source::ParamF32(0),
            base: F32 <- kernels::Source::ParamF32(1),
            head_dim: I32 <- kernels::Source::Param(2),
        ],
        axes = &[BF16]),
    // 1 in rope.metal
    // gemma's rotation: the same neox body over a PROPORTIONAL slice of each
    // head rather than all of it. Same operands as `neox_decode`, and in
    // place like every rotation in this file.
    kernel!(neox_prop_decode "neox_prop_decode", file = Some("rope/neox.metal"),
        launch = kernels::LaunchRule::Rope,
        operands = kernels::operands![
            x: BufMut <- kernels::Source::Out(0),
            position: I32s <- kernels::Source::Positions,
            scale: F32 <- kernels::Source::ParamF32(0),
            base: F32 <- kernels::Source::ParamF32(1),
            head_dim: I32 <- kernels::Source::Param(2),
        ],
        axes = &[BF16]),
    // 1 in rope.metal
    kernel!(neox_prop_mb "neox_prop_mb", axes = &[BF16]),
    // 1 in rope.metal
    kernel!(neox_strided "neox_strided", axes = &[BF16]),
];
