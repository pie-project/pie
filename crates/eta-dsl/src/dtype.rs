//! `dtype::{f32, i32, u32, bool}` — the semantic scalar types.
//!
//! Re-exports the canonical [`eta_ir::Dtype`]; the lowercase module
//! consts let authors write `Channel::new([1], dtype::i32)` verbatim.

pub use eta_ir::Dtype;

/// The 32-bit IEEE-754 floating-point scalar type ([`Dtype::F32`]).
#[allow(non_upper_case_globals)]
pub const f32: Dtype = Dtype::F32;
/// The 32-bit signed two's-complement integer scalar type ([`Dtype::I32`]).
#[allow(non_upper_case_globals)]
pub const i32: Dtype = Dtype::I32;
/// The 32-bit unsigned integer scalar type ([`Dtype::U32`]); the wire's
/// dimension and index type.
#[allow(non_upper_case_globals)]
pub const u32: Dtype = Dtype::U32;
/// The one-byte boolean scalar type ([`Dtype::Bool`]); comparisons and masks.
#[allow(non_upper_case_globals)]
pub const bool: Dtype = Dtype::Bool;
