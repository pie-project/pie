//! engine-cuda's golden probes, a binary of their own: the golden-logits
//! fixtures and hashes (`golden_logits`, run by hand) and the boots that
//! read their fixtures.

// On ROCm the engine's kernels are kernels-rocm's, under kernels-cuda's name.
#[cfg(feature = "rocm")]
extern crate kernels_rocm as kernels_cuda;

#[cfg(feature = "device")]
#[path = "../engine_cuda/common/boot.rs"]
mod common;

mod golden_logits;
mod two_models_exported_boot_a_clean_home_measuring_nothing;
