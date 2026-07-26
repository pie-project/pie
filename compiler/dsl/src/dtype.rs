//! `dtype::{f32, i32, u32, bool}` — the semantic scalar types.
//!
//! Re-exports the canonical [`pie_ir::DType`]; the lowercase module
//! consts let authors write `Channel::new([1], dtype::i32)` verbatim.

pub use pie_ir::DType;

/// The 32-bit IEEE-754 floating-point scalar type ([`DType::F32`]).
#[allow(non_upper_case_globals)]
pub const f32: DType = DType::F32;
/// The 32-bit signed two's-complement integer scalar type ([`DType::I32`]).
#[allow(non_upper_case_globals)]
pub const i32: DType = DType::I32;
/// The 32-bit unsigned integer scalar type ([`DType::U32`]); the wire's
/// dimension and index type.
#[allow(non_upper_case_globals)]
pub const u32: DType = DType::U32;
/// The one-byte boolean scalar type ([`DType::Bool`]); the dtype of
/// comparison and mask results.
#[allow(non_upper_case_globals)]
pub const bool: DType = DType::Bool;
