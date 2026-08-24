use model_dsl::axes::{Bf16, Mxfp4, NativeKv};
use model_dsl::load::SfBf16;
use model_dsl::{Plan, Plane};

pub mod forward;
pub mod import;
pub mod model;

use import::import_hf;
use model::Model;

pub type ShippedW1 = Bf16;
pub type ShippedW2 = Mxfp4;
pub type ShippedKv = NativeKv;

pub type TraceFn = fn(Plane) -> Plan;

pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalog![
    (
        "kimik3-bf16-mxfp4-kv-bf16",
        model_dsl::trace_hybrid, Model::<ShippedW1, ShippedW2, ShippedKv>::k3(),
    ),
    (
        "kimik3-bf16-mxfp4-kv-bf16-tp2",
        model_dsl::trace_hybrid, Model::<ShippedW1, ShippedW2, ShippedKv, 2>::k3(),
    ),
];

model_dsl::allow_import! {
    import_hf::<SfBf16, ShippedW1, ShippedW2, ShippedKv> => ("kimik3-bf16-mxfp4-kv-bf16", Model::<ShippedW1, ShippedW2, ShippedKv>::k3()),
}
