//! `dtype::{f32, i32, u32, bool}` — the semantic scalar types (overview §1).
//!
//! Re-exports the canonical [`pie_ptir::DType`]; the lowercase module
//! consts let authors write `Channel::new([1], dtype::i32)` verbatim.

pub use pie_ptir::DType;

#[allow(non_upper_case_globals)]
pub const f32: DType = DType::F32;
#[allow(non_upper_case_globals)]
pub const i32: DType = DType::I32;
#[allow(non_upper_case_globals)]
pub const u32: DType = DType::U32;
#[allow(non_upper_case_globals)]
pub const bool: DType = DType::Bool;
