//! Registering an adapter bank's rows — the correction class's one verb
//! (design §8, decision 17).
//!
//! # Why this is a verb of its own and not `transfer::`
//!
//! [`transfer`](crate::transfer) moves state the engine ALREADY HOLDS between
//! places it already owns: kv pages between slots or domains, recurrent state
//! between banks, an elastic pool between sizes. Nothing crosses the boundary
//! but addresses. A registration is the other shape entirely — host bytes
//! arriving for the first time, landed into device residency under a name the
//! plan declared. That is what [`load`](crate::load) is, and this is a second,
//! smaller load of a few rows.
//!
//! Reading it as a copy would also put it on the wrong side of the one
//! property that matters: `copy_kv` is a per-request scheduling verb the
//! runtime calls on the fire path's shoulder, and this is a residency verb it
//! calls once per adapter and never again.
//!
//! # Why registering is not a capacity negotiation
//!
//! [`Budgets::max_adapters`](crate::load::Budgets::max_adapters) is a BUDGET,
//! not an admission cap (decision 17): the capacity is stated once, the load
//! is refused if the plan cannot seat it, and after that a registration either
//! names an id inside it or is a caller error with a number in it. There is no
//! eviction here, no LRU, no "adapter slot" lease — vLLM's cost in the axis
//! tart measured was admission and capacity, not kernels, and the way not to
//! pay it is not to build the machinery.
//!
//! # Why the planes are full-capacity
//!
//! An adapter trained at rank 4 in a bank declared at rank 16 is submitted
//! zero-padded, and the shell zeroes the slot before it writes. Padding is the
//! caller's because the two planes pad differently — `A`'s unused ranks are
//! trailing ROWS and `B`'s are a stride inside every row — so a shell that
//! wrote a short plane's prefix would be right for one and wrong for the
//! other. The padding is also exact rather than approximate: a zero row of `A`
//! contributes a zero to the waist, and a zero column of `B` contributes zero
//! to the sum.

use serde::{Deserialize, Serialize};

/// One plane of one adapter, as the caller hands it across.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterPlane {
    /// Which bank, by the name the plan's `Param` carries — the same
    /// `model_ir` spelling the model text declared it under. Named rather than
    /// indexed because a param index is a fact about one bake and a caller
    /// that held one across a re-trace would write into somebody else's plane.
    pub bank: String,
    /// One whole slot of that bank, in the bank's declared dtype and layout.
    pub bytes: Vec<u8>,
}

/// Everything one registration states.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterRegistration {
    /// Which row of every named bank this fills. The same id a
    /// [`Lane::adapter`](crate::fire::Lane::adapter) names.
    pub id: u32,
    /// The planes, in any order. A bank this list omits keeps what it held,
    /// which is what makes registering one site at a time expressible.
    pub planes: Vec<AdapterPlane>,
}
