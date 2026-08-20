//! The instantiation axes the family tables share.
//!
//! An axis is declared once here when more than one family spells the same
//! point set, and inline at the row when it does not. The split is not
//! aesthetic: a shared constant is a claim that two kernels are compiled for
//! the SAME set, and `scripts/metal-kernel-audit.py` fails the moment that
//! stops being true — so `GROUP` and `GROUP_64` are deliberately different
//! names rather than one relaxed one.
//!
//! Point order within an axis follows the number, not the string, because a
//! reader checking coverage against a checkpoint reads it as a range.

use kernels::Axis;

/// The activation dtype, and today it has exactly one point.
///
/// `driver-metal`'s `AffineFormat::kernel_suffix` states the fact this
/// encodes: bf16 is not an axis — it is the one activation dtype the driver
/// instantiates. Declared anyway, and this is the divergence from CUDA that
/// `.wiki/kernel-x/metal-refactor.md` §2.2 exists to justify: on CUDA the `_bf16`
/// suffix is the sole record of an operand type the `void*` ABI drops, so it
/// must stay authored. Here the macro pastes it, so it is a coordinate — and a
/// second activation dtype becomes one more point instead of 470 new names.
pub const BF16: Axis = Axis {
    what: "activation dtype",
    points: &["_bfloat16"],
};

/// The affine group size. Not inferable from the tensors — g64/b8 and g128/b4
/// pack to identical shapes — so the checkpoint's `config.json` is the only
/// source and a pipeline built for the wrong point returns fluent nonsense
/// rather than failing. See `AffineFormat`.
pub const GROUP: Axis = Axis {
    what: "affine group size",
    points: &["_gs_32", "_gs_64", "_gs_128"],
};

/// One group size, for the kernels compiled at g64 alone. That this differs
/// from [`GROUP`] is the coverage gap, stated.
pub const GROUP_64: Axis = Axis {
    what: "affine group size",
    points: &["_gs_64"],
};

/// g32 alone.
pub const GROUP_32: Axis = Axis {
    what: "affine group size",
    points: &["_gs_32"],
};

/// The affine bit width.
pub const BITS: Axis = Axis {
    what: "affine bit width",
    points: &["_b_4", "_b_8"],
};

/// 4-bit alone.
pub const BITS_4: Axis = Axis {
    what: "affine bit width",
    points: &["_b_4"],
};

/// 8-bit alone.
pub const BITS_8: Axis = Axis {
    what: "affine bit width",
    points: &["_b_8"],
};

/// The routed GEMM's row tile. `moe_tile_rows` picks one from the batch shape
/// and the sort pads to it, so the two must agree — which is why all three are
/// compiled rather than one chosen at build time.
pub const TILE_M: Axis = Axis {
    what: "GEMM row tile",
    points: &["_bm_16", "_bm_32", "_bm_64"],
};

/// The GEMM's column tile.
pub const TILE_N: Axis = Axis {
    what: "GEMM column tile",
    points: &["_bn_16", "_bn_32", "_bn_64"],
};

/// One column tile, for the kernels compiled at bn=32 alone.
pub const TILE_N_32: Axis = Axis {
    what: "GEMM column tile",
    points: &["_bn_32"],
};
