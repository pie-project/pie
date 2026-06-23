//! `pie-worker` library — engine boot path + supporting modules.
//!
//! The `pie` CLI binary (`src/main.rs`) is one consumer; the upcoming
//! `pie-server` pyo3 wheel under `sdk/python-server/` is another.
//! Both link against this lib, so the engine boot logic
//! ([`serve::start_engine`]) has a single source of truth.
//!
//! Modules are `pub` so external callers (the pyo3 wheel) can reach
//! the surface they need — `serve::start_engine`, `config::Config`,
//! `embedded_driver::EmbeddedDriver`, etc.

pub mod bootstrap_translate;
pub mod cli;
pub mod config;
pub mod driver_ffi;
pub mod embedded_driver;
pub mod hf;
pub mod paths;
pub mod py_runtime;

pub mod serve;

#[cfg(any(feature = "driver-cuda", feature = "driver-portable"))]
#[used]
static PIE_WEIGHT_LOADER_LINK_ANCHOR: unsafe extern "C" fn(
    *const pie_weight_loader::PieLoaderCompileInput,
    *mut *mut pie_weight_loader::PieLoaderProgramHandle,
    *mut pie_weight_loader::PieLoaderError,
) -> pie_weight_loader::PieLoaderStatus = pie_weight_loader::pie_loader_compile;
