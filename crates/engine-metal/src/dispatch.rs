//! The six `Dispatch*` impls (one module per family), plus [`copy`], which
//! encodes the row-gather/scatter around a copied region rather than an op.

pub(crate) mod copy;

mod attn;
mod collective;
mod custom;
mod elemwise;
mod layout;
mod linear;
mod rs;
