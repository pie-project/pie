//! Registering an adapter bank's rows: a residency load of a few rows, not a
//! copy and not a capacity negotiation.
//!
//! `Budgets::max_adapters` is a budget, not an admission cap: capacity is
//! stated once at load, and registration either names an id inside it or
//! errors — no eviction, no LRU.
//!
//! Planes are submitted zero-padded to the bank's full rank; the shell zeroes
//! the slot before writing. `A`'s unused ranks are trailing rows, `B`'s are a
//! stride inside every row, so the caller (not the shell) must pad each
//! correctly.

use serde::{Deserialize, Serialize};

/// One plane of one adapter, as the caller hands it across.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterPlane {
    /// Bank name, the same `model_ir` spelling the model declared it under
    /// (named rather than indexed: a param index is only valid for one bake).
    pub bank: String,
    /// One whole slot of that bank, in the bank's declared dtype and layout.
    pub bytes: Vec<u8>,
}

/// Everything one registration states.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterRegistration {
    /// Row filled in every named bank. Same id as
    /// [`Lane::adapter`](crate::fire::Lane::adapter).
    pub id: u32,
    /// The planes, in any order; a bank omitted here keeps its current value.
    pub planes: Vec<AdapterPlane>,
}
