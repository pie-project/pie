//! Every `Dispatch*` trait, answered by a kernels-cerebras entry; an op the
//! kernels lack is refused by name.

mod attn;
mod collective;
mod custom;
mod elemwise;
mod gather;
mod linear;
mod sampling;
mod ssm;
