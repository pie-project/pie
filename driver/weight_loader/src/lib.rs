//! Pure Rust compiler boundary for Pie weight loading.
//!
//! The crate intentionally keeps CUDA, file IO, and `WeightStore` ownership on
//! the C++ side. Rust receives metadata/config/ABI data and returns a flat
//! executable storage program view.

pub mod abi;
pub mod config;
pub mod dump;
pub mod error;
pub mod ffi;
pub mod ffi_arena;
pub mod ffi_types;
pub mod frontend;
pub mod ir;
pub mod optimizer;
pub mod reference;
pub mod schema;
pub mod schemas;
pub mod semantic;
pub mod source;
pub mod storage;
pub mod storage_compiler;
pub mod typecheck;
pub mod types;

pub use ffi::{
    PieLoaderProgramHandle, pie_loader_compile, pie_loader_error_free, pie_loader_program_free,
    pie_loader_program_deserialize, pie_loader_program_serialize,
    pie_loader_program_serialized_len, pie_loader_program_view,
};
pub use ffi_types::*;
