//! The forward-pass authoring surface: model texts state backend-neutral
//! role points onto a [`Plan`]; planes answer them with claims. See
//! `.wiki/baker.md` for the design this crate is converging to — the
//! hand-written [`kernels`] surface is scheduled to be generated from the
//! kernel crates' own declarations, family by family.

#![allow(clippy::too_many_arguments)]

pub mod declare;
pub mod facts;
pub mod forward;
pub mod kernels;
pub mod load;
mod record;

pub use declare::*;
pub use facts::*;
pub use forward::*;
pub use model_dsl_macros::Facts;
pub use model_ir::kernels::Backend as Plane;
pub use model_ir::plan::Plan;
pub use record::{Value, Windows};

pub mod axes {
    /// One weight-bearing axis: what the checkpoint holds for a bank of
    /// projections. `NAME` doubles as the plan's repr column and joins
    /// into the catalogued SKU string.
    pub trait Dtype: 'static {
        const NAME: &'static str;
    }

    /// The KV cache axis: which scheme the pages hold — a load-time fact,
    /// stated where the SKU is named.
    pub trait KvDtype: 'static {
        const NATIVE_BF16: bool;
        const NAME: &'static str;
    }

    /// Plain bf16: the repr every dense row ships today.
    pub enum Bf16 {}
    impl Dtype for Bf16 {
        const NAME: &'static str = "bf16";
    }

    /// MXFP4 experts in Marlin layout (gpt-oss's shipped form).
    pub enum Mxfp4 {}
    impl Dtype for Mxfp4 {
        const NAME: &'static str = "mxfp4";
    }

    /// WNA16 int4 (glm's routed experts).
    pub enum Wna16 {}
    impl Dtype for Wna16 {
        const NAME: &'static str = "wna16";
    }

    /// Native bf16 KV pages.
    pub enum NativeKv {}
    impl KvDtype for NativeKv {
        const NATIVE_BF16: bool = true;
        const NAME: &'static str = "kv-bf16";
    }
}

/// Enumerate one shipping SKU: a name, the trace fn, and the monomorphized
/// model it instantiates. The row is a `(name, fn(Plane) -> Plan)` pair in
/// a plain table the family's cover file exposes.
#[macro_export]
macro_rules! catalog {
    ($( ($name:literal, $trace:path, $m:expr $(,)?) ),+ $(,)?) => {
        &[ $( ($name, (|plane| $trace($name, &$m, plane)) as _) ),+ ]
    };
}

pub mod seam {
    pub use model_ir::seam::*;

    use crate::record::Value;

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

    pub fn at<S: Sees>(def: Def, sees: S, layer: u32) {
        let values = sees.values();
        values[0].rec.seam(def.name, &values, Some(layer));
    }
}
