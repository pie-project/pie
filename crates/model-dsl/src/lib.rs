//! The forward-pass authoring eDSL over the typed IR (design §4, §10). Model
//! texts call the per-family wrappers in [`kernels`], which compute shapes in
//! plain Rust and push typed op variants onto a recorded [`Plan`]; `split` /
//! [`merge!`] / [`facts!`] carry the guard tracking over from the old surface
//! unchanged. This crate names no backend and reaches no device.
//!
//! Weight import rows (the old surface's `load.rs`) are deliberately absent —
//! they arrive with the loader port.

#![allow(clippy::too_many_arguments)]

pub mod declare;
pub mod facts;
pub mod forward;
pub mod kernels;
mod record;

pub use declare::*;
pub use facts::*;
pub use forward::*;
pub use model_ir::{Dtype, GeomKind, Plan, Plane, ValueId};
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
