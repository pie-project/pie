//! `Elementwise`: the per-element passes — norms and the residual folds,
//! rotary position, the sigmoid gates, and the hyper-connection mixers.
//! One submodule per member of the family; the entries inside keep one
//! entry per IR variant.

pub mod gate;

pub mod hc;

pub mod norm;

pub mod rope;
