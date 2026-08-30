//! `Elementwise`: the per-element passes — norms and the residual folds,
//! rotary position, the sigmoid gates, and the hyper-connection mixers.
//! One submodule per member of the family; the entries inside keep one
//! entry per IR variant.

/// The clipped linears' clamp — its own member of the family because the
/// sites it serves are projections and not a fused activation.
pub mod clip;

pub mod gate;

pub mod hc;

/// The CENTRED norm: `norm`'s reductions plus the mean subtraction, and no
/// weight at all. Its own member of the family because it reduces something
/// else, not because it scales differently.
pub mod layernorm;

pub mod norm;

pub mod rope;

/// The multimodal rotary — `rope`'s partial arm over an `(t, h, w)` triple.
/// Its own member of the family because it reads its own position stream
/// under its own statute, not because the rotation differs.
pub mod rope_mrope;
