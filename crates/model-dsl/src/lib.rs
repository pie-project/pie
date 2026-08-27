//! The forward-pass authoring eDSL over the typed IR (design §4, §10). Model
//! texts call the per-family wrappers in [`ops`], which compute shapes in
//! plain Rust and push typed op variants onto a recorded [`Plan`]; `split` /
//! [`merge!`] / [`facts!`] carry the guard tracking over from the old surface
//! unchanged. This crate names no backend and reaches no device.
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
/// stops being true. `Attention`, `CacheRow`, `Def`, `Linear`, `Plan` and
/// `ValueId` are what `model::deployment` reads a traced plan with;
/// `Collective` and `Operation` are how `model::identify` tells one rank of a
/// world from a whole model; `Dtype` is what a catalog row and a load contract
/// are written in; `GeomKind` is what a forward pass asks the runtime for;
/// `Plane` is what a trace is taken at; `Param` and `Shard` are the plan's
/// demand column and the axis a rank cut runs along, which
/// `model/tests/every_param_has_one_producer.rs` holds against the contract
/// that fills it.
pub use model_ir::{
    Attention, CacheRow, Collective, Def, Dtype, GeomKind, Linear, Operation, Param, Plan, Plane,
    Shard, ValueId,
};
pub use record::{Recorder, SplitSpec, Value};

/// What the catalog registers per model: trace me for this plane.
pub type TraceFn = fn(Plane) -> Plan;

#[macro_export]
macro_rules! catalog {
    ($( ($name:literal, $trace:path, $m:expr $(,)?) ),+ $(,)?) => {
        &[ $( ($name, (|plane| $trace($name, &$m, plane)) as _) ),+ ]
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

    pub trait Sees {
        fn values(&self) -> Vec<&Value>;
    }

    impl Sees for (&Value,) {
        fn values(&self) -> Vec<&Value> {
            vec![self.0]
        }
    }

    impl Sees for (&Value, &Value) {
        fn values(&self) -> Vec<&Value> {
            vec![self.0, self.1]
        }
    }

    pub fn at<S: Sees>(def: Def, sees: S) {
        let values = sees.values();
        values[0].rec().seam(def.name, &values);
    }
}
