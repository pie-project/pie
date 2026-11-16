//! kernels-cuda's integration tests, one binary: `<group>/<family>.rs`, each
//! test named for the sentence it checks.
#![cfg(feature = "cuda")]

mod attn;
mod collective;
mod common;
mod conv;
mod elemwise;
mod gather;
mod linear;
mod runtime;
mod sampling;
mod ssm;
