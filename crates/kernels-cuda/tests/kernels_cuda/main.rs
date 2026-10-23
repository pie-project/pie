//! kernels-cuda's integration tests, one binary: `<group>/<family>.rs`, each
//! test named for the sentence it checks.
#![cfg(feature = "cuda")]

mod attn;
mod common;
mod conv;
mod elemwise;
mod gather;
mod linear;
mod runtime;
mod sampling;
mod ssm;

// Strict kernels for the whole binary, before any test runs: an
// instantiation that will not compile fails its test instead of falling
// back unseen.
#[cfg(target_os = "linux")]
#[used]
#[unsafe(link_section = ".init_array")]
static STRICT_KERNELS: extern "C" fn() = {
    extern "C" fn on() {
        kernels_cuda::jit::strict_kernels();
    }
    on
};
