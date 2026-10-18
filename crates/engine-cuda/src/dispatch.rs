mod attn;
mod collective;
pub(crate) mod copy;
#[cfg(not(feature = "rocm"))]
mod custom;
#[cfg(feature = "rocm")]
#[path = "dispatch/hip/custom.rs"]
mod custom;
mod elemwise;
mod layout;
#[cfg(not(feature = "rocm"))]
mod linear;
#[cfg(feature = "rocm")]
#[path = "dispatch/hip/linear.rs"]
mod linear;
mod probe;
mod spatial;
