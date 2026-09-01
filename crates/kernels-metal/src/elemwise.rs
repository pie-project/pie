//! `Elementwise`: the family that walks a rectangle without mixing rows —
//! the norms and residual folds, the neox rotations, the sigmoid gate, and
//! the hyper-connection points. One entry per IR variant.

pub mod gate;
pub mod hc;
pub mod norm;
pub mod rope;

/// The multimodal rotary — `rope`'s partial arm over an `(t, h, w)` triple.
/// Its own member of the family because it reads its own position stream
/// under its own statute, not because the rotation differs.
pub mod rope_mrope;
