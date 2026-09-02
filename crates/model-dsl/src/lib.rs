//! The forward-pass authoring eDSL over the typed IR. Model texts call the
//! per-family wrappers in [`ops`], which compute shapes in plain Rust and
//! push typed op variants onto a recorded [`Trace`]; `split`, `Value::merge`
//! and the [`Predicate`] algebra carry the guard tracking. This crate names
//! no backend and reaches no device.
//!
//! Where a weight's bytes come from is not said here: `checkpoint`'s
//! `contract::Expr` / `ModelContract` is the typed load-contract language,
//! and each family writes its provenance there directly.

#![allow(clippy::too_many_arguments)]

pub mod declare;
pub mod facts;
pub mod forward;
pub mod ops;
mod record;

pub use declare::*;
pub use facts::*;
pub use forward::*;
/// The only door to the IR, and the reason it is a list and not a glob: a
/// model text and the tools around it read plans, and everything they read
/// a plan with comes through here. `model` never names `model_ir` directly.
/// Names are added only when something needs them (e.g. `resolve_classes`
/// for the class sweep every merge must survive), and removed when nothing
/// does.
pub use model_ir::{
    Attention, CacheRow, Def, Dtype, GateActivation, Layout, Linear, MropeForm, Operands, Operation,
    Param, ParamSource, Trace, Platform, Shard, ValueId, resolve_classes,
};
pub use record::{Recorder, Refine, SplitSpec, Value};

/// What the catalog registers per model: trace me for this platform.
pub type TraceFn = fn(Platform) -> Trace;

/// What the catalog registers per model: sort this request into my facts and
/// pack them into the one `u64` a lane carries. Monomorphic on purpose: the
/// runtime's fire path holds only a SKU string and cannot name a family's
/// own `Classify::of` type, so this column closes over it once, per family.
pub use model_ir::ClassifyFn;

/// The catalog rows of one family: `(sku, tp, trace, model)` each, closing
/// the model expression into a [`TraceFn`] and a [`ClassifyFn`].
///
/// TP is a column (a fact about the row) rather than derived by tracing and
/// scanning for an `AllReduce`. The trace closure wraps `$m` in
/// [`placing_for`] so a `Dtype` placement can see which platform is asking
/// (macro hygiene means `$m` can't see that binding otherwise); the classify
/// closure does not wrap it, since a fact word never depends on platform.
#[macro_export]
macro_rules! catalog {
    ($( ($name:literal, $tp:literal, $trace:path, $m:expr $(,)?) ),+ $(,)?) => {
        &[ $( (
            $name,
            $tp,
            (|platform| {
                let model = $m;
                $trace($name, &model, platform)
            }) as _,
            (|request: &$crate::Request| $crate::word_of(|| $m, request)) as _,
        ) ),+ ]
    };
}

pub mod seam {
    //! The seam vocabulary and the statement that plants one, kept beside
    //! the surface that states it (the IR itself keeps only the `Seam` rows
    //! a plan carries).

    use crate::record::Value;

    pub struct Def {
        pub name: &'static str,
    }

    pub const ATTN_Q: Def = Def { name: "attn.q" };

    pub const ATTN_OUT: Def = Def { name: "attn.out" };

    pub const ATTN_QV: Def = Def { name: "attn.qv" };

    pub const RECURRENT: Def = Def { name: "recurrent" };

    pub const IN: Def = Def { name: "in" };

    pub const OUT: Def = Def { name: "out" };

    /// The draft readout: the MTP head's logits over the draft window, a
    /// second readout of the same fire materialized outside the graph
    /// exactly as [`OUT`] is (a seam rather than a return value, since
    /// `ForwardHybrid::forward` hands back only the trunk's logits).
    pub const MTP: Def = Def { name: "mtp" };

    /// The draft readout as TOKENS: the `[rows, depth]` i32 plane a draft
    /// head's argmax chain writes (`ops::layout::argmax` over every step's
    /// logits), one entry per readout row per step. What `mtp_drafts` is
    /// pointed at; a text with a draft head plants it once, and its width
    /// is the depth the shell advertises.
    pub const MTP_DRAFTS: Def = Def { name: "mtp.drafts" };

    /// The score readout: the attention's per-query log-sum-exp over the
    /// capture window, the mass the softmax normalized by. Observation runs
    /// in-graph (a declared column); whatever a guest does with the numbers
    /// happens before/after the fire.
    pub const SCORES: Def = Def {
        name: "attn.scores",
    };

    /// Plant a seam on the values it names. The first one carries the recorder
    /// — every value of one trace carries the same one — so the slice must not
    /// be empty; a seam over nothing has nothing to say.
    pub fn at(def: Def, values: &[&Value]) {
        let first = values
            .first()
            .unwrap_or_else(|| panic!("seam `{}` names no value", def.name));
        first.rec().seam(def.name, values);
    }
}
