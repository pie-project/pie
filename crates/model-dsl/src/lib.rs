//! The forward-pass authoring eDSL over the typed IR (design §4, §10). Model
//! texts call the per-family wrappers in [`ops`], which compute shapes in
//! plain Rust and push typed op variants onto a recorded [`Trace`]; `split`,
//! `Value::merge` and the [`Predicate`] algebra carry the guard tracking over
//! from the old surface unchanged. This crate names no backend and reaches no
//! device.
//!
//! WHERE A WEIGHT'S BYTES COME FROM IS NOT SAID HERE. It was, in a `load`
//! module of production tables with a source algebra of their own, and that
//! algebra was a second spelling of one `model-loader` already owns:
//! `contract::Expr` and `ModelContract` are the typed load-contract language,
//! its own doc names `model::contract` as the declarer, and `infer`/`compile`/
//! `executor` are what check and run it. Each family writes its provenance in
//! that language directly. A declaration surface that restated it would be two
//! vocabularies for one fact, free to disagree about a tensor neither side
//! could see.

#![allow(clippy::too_many_arguments)]

pub mod declare;
pub mod facts;
pub mod forward;
pub mod ops;
mod record;

pub use declare::*;
pub use facts::*;
pub use forward::*;
/// THE ONLY DOOR TO THE IR, and the reason it is a list and not a glob: a
/// model text and the tools around it read plans, and everything they read a
/// plan WITH comes through here. `model` never names `model_ir` — a crate
/// that authored against the IR directly would be a second surface with a
/// second set of rules about what a plan may say.
///
/// EVERY NAME ON IT IS ASKED FOR BY SOMEBODY, and the list narrows when that
/// stops being true. `GeomKind` was on it until M20 and is the proof: a
/// forward pass used to name a geometry kind to ask the runtime for a vector,
/// and now it names the kv row instead ([`Input::write_page`]), so the name
/// left the door. `Attention`, `CacheRow`, `Def`, `Linear`, `Trace` and
/// `ValueId` are what `model::deployment` reads a traced plan with;
/// `Dtype` is what a catalog row and a load contract
/// are written in;
/// `Platform` is what a trace is taken at; `Param` and `Shard` are the plan's
/// demand column and the axis a rank cut runs along, which
/// `model/tests/every_param_has_one_producer.rs` holds against the contract
/// that fills it.
///
/// `Operands` is how a tool asks a node what it reads and writes without
/// matching every op variant itself. `resolve_classes` is the newest name on
/// the list and the one a model text feels most directly: the class sweep
/// every merge a forward pass writes has to survive (palo design §1). The IR
/// owns it because the compiler's accept pass runs the same walk, and it
/// arrives here for the reason everything else does —
/// `model/tests/every_class_resolves_every_merge.rs` reads traced plans with
/// these two, and reading a plan goes through this door.
pub use model_ir::{
    Attention, CacheRow, Def, Dtype, Linear, Operands, Operation, Param, ParamSource, Trace,
    Platform, Shard, ValueId, resolve_classes,
};
pub use record::{Recorder, Refine, SplitSpec, Value};

/// What the catalog registers per model: trace me for this platform.
pub type TraceFn = fn(Platform) -> Trace;

/// What the catalog registers per model: sort this request into my facts and
/// pack them into the one `u64` a lane carries.
///
/// MONOMORPHIC ON PURPOSE. `Classify` is a trait over a family's own `Facts`
/// struct, and the party that needs a word — the engine's fire path — holds
/// a SKU string and nothing else: it cannot name `qwen_3::forward::Facts`,
/// and a plan cannot tell it either, because `Guard::Fact(bit)` numbers its
/// bits and a bit is a position and nothing else (see [`Predicate`]). The
/// column is what closes that: one pointer per row, wrapping the family's own
/// `Classify::of(r).word()`, so which bit `qo_one` is stays the model's own
/// business and the word still travels.
pub type ClassifyFn = fn(&Request) -> u64;

/// The catalog rows of one family: `(sku, tp, trace, model)` each, closing the
/// model expression into a [`TraceFn`] and a [`ClassifyFn`].
///
/// TP IS A COLUMN BECAUSE IT IS A FACT ABOUT THE ROW. It was reachable only by
/// reading the model expression's arguments — or, worse, by tracing the whole
/// forward pass and looking for an `AllReduce`, which is how `model::identify`
/// used to tell one rank of a world from a whole model. A name is a name; the
/// world size a row was built for is a number, and it belongs beside the name
/// that promises it.
///
/// THE CLASSIFY COLUMN IS THE SAME ARGUMENT ONE STEP FURTHER IN. A lane's word
/// is what the driver composes a fire from, and computing it means calling
/// `<M::Facts as Classify>::of` — a call only a party that can NAME the
/// family's `Facts` can write. The catalog is where the family's type is last
/// visible, so it is where the call is closed over; everywhere downstream
/// holds a SKU string and a [`ClassifyFn`].
#[macro_export]
macro_rules! catalog {
    ($( ($name:literal, $tp:literal, $trace:path, $m:expr $(,)?) ),+ $(,)?) => {
        &[ $( (
            $name,
            $tp,
            (|platform| $trace($name, &$m, platform)) as _,
            (|request: &$crate::Request| $crate::word_of(|| $m, request)) as _,
        ) ),+ ]
    };
}

pub mod seam {
    //! The seam vocabulary and the statement that plants one. The names lived
    //! in old `model-ir`; the new IR keeps only the `Seam` rows a plan
    //! carries, so the vocabulary lives here, beside the surface that states
    //! it.

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

    /// **THE DRAFT READOUT** (design §9, palo C3). The MTP head's logits over
    /// the draft window — a SECOND readout of the same fire, materialized
    /// outside the graph exactly as [`OUT`] is.
    ///
    /// A SEAM RATHER THAN A RETURN VALUE, because `ForwardHybrid::forward`
    /// hands back one value and the trunk's logits are it. Design §9 says an
    /// export is "a place a declared value materializes outside the graph",
    /// and `model_ir::check::classes`' roots are written for exactly this:
    /// "a model that exports a second value gets the same treatment without
    /// this file learning a new name". So the draft column is DEMANDED by
    /// every class that runs the head, and dead in every class that does not.
    ///
    /// WHAT STILL OWES IT A HOME: `model_compiler::arena` gives the delivery
    /// tail — liveness to fire end, read in every class — to the `"out"` seam
    /// by name, and to no other. A draft column is read after the graph by the
    /// same sampler that reads `"out"` (`driver::program`'s `MtpLogits` and
    /// `MtpDrafts` intrinsics index the readout at `mtp_draft_row`), so it
    /// wants the same tail. Until the compiler's export pass lands, the model
    /// text states the export truthfully and the shell owes the pin — the same
    /// order the masked axis went in.
    pub const MTP: Def = Def { name: "mtp" };

    /// **THE SCORE READOUT** (design §9's named archetype, palo C4). The
    /// attention's per-query log-sum-exp over the capture window — the mass
    /// the softmax normalized by, which is what a scores consumer actually
    /// needs and what the kernel already hands back beside `o`.
    ///
    /// OBSERVATION IN-GRAPH, COMPUTATION AT THE BOUNDARY: the arm runs inside
    /// the immutable graph and writes a declared column; whatever a guest does
    /// with the numbers happens where guest code is allowed to happen, which
    /// is before and after the fire.
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
