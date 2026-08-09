//! Rotary embeddings.
//!
//! Four spellings of the schedule (`neox`, `freqs`, `prop`, and the strided
//! form), each in a decode and a multi-batch shape. The `freqs` pair reads a
//! host-computed table, which is what llama-3.1's wavelength ramp needs.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 1 in rope/neox.comp
    // IN PLACE on ONE tensor: buffer 0 is both the input and the result, and
    // the template is `rope_neox_decode`.
    //
    // Which makes a TEXT gap visible that no row can close. `dsl::metal::rope`
    // -- the sibling's, and the only spelling either backend has --
    // states one launch carrying q and k — two inputs and two results — and
    // this kernel rotates one buffer. The statement should be two, one per
    // tensor, and until it is, the second tensor is not rotated at all.
    kernel!(neox_decode "neox_decode", file = Some("rope/neox.comp"), launch = kernels::LaunchRule::Rope,
        operands = kernels::operands![
            x: BufMut <- kernels::Source::Out(0),
            position: I32s <- kernels::Source::Positions,
            scale: F32 <- kernels::Source::ParamF32(0),
            base: F32 <- kernels::Source::ParamF32(1),
            head_dim: I32 <- kernels::Source::Param(2),
        ],
        // The rotation's extent is the STATEMENT's, not the fire's: gemma-4
        // rotates a quarter of each full-attention head and all of each
        // sliding one. The kernel never reads param 3 -- its operand list
        // stops at 2 -- but `Rule::Rope`'s grid is half of it.
        grid_param = Some(3),
        // The heads are counted by the SAME width the kernel is told.
        head_param = Some(2),
        axes = &[BF16]),
    // 1 in rope/neox.comp
    // The rotation a deployment that RESCALES its ladder takes. Same body as
    // `neox_decode` with the frequencies read from a buffer instead of raised
    // from a base -- which is the only form that can express llama-3's
    // piecewise rescaling or YaRN's, because neither is a base.
    kernel!(neox_freqs_decode "neox_freqs_decode", file = Some("rope/neox.comp"),
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
        // See `neox_decode`: the extent is the statement's.
        grid_param = Some(3),
        // The heads are counted by the SAME width the kernel is told.
        head_param = Some(1),
        axes = &[BF16]),
    // 1 in rope/neox.comp
    // The batched form of the rescaled ladder, and the row a PREFILL on
    // llama-3.1, llama-3.2 or any YaRN deployment needs. It was bare, so the
    // statement had nothing to name and named the decode symbol instead — a
    // single-row kernel over a multi-row grid, which rotates row zero and
    // leaves every row after it untouched. Rope is the identity at position
    // zero, so row zero agreed with the reference either way and the failure
    // was silent.
    //
    // Same operands as `neox_freqs_decode`: the row stride the shader needs is
    // `grid.y * head_dim`, and `Rule::Rope` now takes its head axis from the
    // tensor being turned, so the grid says it.
    kernel!(neox_freqs_mb "neox_freqs_mb", file = Some("rope/neox.comp"),
        launch = kernels::LaunchRule::Rope,
        operands = kernels::operands![
            x: BufMut <- kernels::Source::Out(0),
            position: I32s <- kernels::Source::Positions,
            scale: F32 <- kernels::Source::ParamF32(0),
            inv_freq: Buf <- kernels::Source::RopeFrequencies,
            head_dim: I32 <- kernels::Source::Param(1),
            mscale: F32 <- kernels::Source::ParamF32(2),
        ],
        // See `neox_decode`: the extent is the statement's.
        grid_param = Some(3),
        // The heads are counted by the SAME width the kernel is told.
        head_param = Some(1),
        axes = &[BF16]),
    // 1 in rope/neox.comp
    // The batched form, and the same shape: one tensor, per-token positions.
    kernel!(neox_mb "neox_mb", file = Some("rope/neox.comp"), launch = kernels::LaunchRule::Rope,
        operands = kernels::operands![
            x: BufMut <- kernels::Source::Out(0),
            position: I32s <- kernels::Source::Positions,
            scale: F32 <- kernels::Source::ParamF32(0),
            base: F32 <- kernels::Source::ParamF32(1),
            head_dim: I32 <- kernels::Source::Param(2),
        ],
        // The rotation's extent is the STATEMENT's, not the fire's: gemma-4
        // rotates a quarter of each full-attention head and all of each
        // sliding one. The kernel never reads param 3 -- its operand list
        // stops at 2 -- but `Rule::Rope`'s grid is half of it.
        grid_param = Some(3),
        // The heads are counted by the SAME width the kernel is told.
        head_param = Some(2),
        axes = &[BF16]),
    // 1 in rope/neox.comp
    // gemma's rotation: the same neox body over a PROPORTIONAL slice of each
    // head rather than all of it. Same operands as `neox_decode`, and in
    // place like every rotation in this file.
    kernel!(neox_prop_decode "neox_prop_decode", file = Some("rope/neox.comp"),
        launch = kernels::LaunchRule::Rope,
        operands = kernels::operands![
            x: BufMut <- kernels::Source::Out(0),
            position: I32s <- kernels::Source::Positions,
            scale: F32 <- kernels::Source::ParamF32(0),
            base: F32 <- kernels::Source::ParamF32(1),
            head_dim: I32 <- kernels::Source::Param(2),
        ],
        // The rotation's extent is the STATEMENT's, not the fire's: gemma-4
        // rotates a quarter of each full-attention head and all of each
        // sliding one. The kernel never reads param 3 -- its operand list
        // stops at 2 -- but `Rule::Rope`'s grid is half of it.
        grid_param = Some(3),
        // The heads are counted by the SAME width the kernel is told.
        head_param = Some(2),
        axes = &[BF16]),
    // 1 in rope/neox.comp
    kernel!(neox_prop_mb "neox_prop_mb", axes = &[BF16]),
    // 1 in rope/neox.comp
    kernel!(neox_strided "neox_strided", axes = &[BF16]),
];
