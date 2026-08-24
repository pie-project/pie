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
        "gptoss-20b-bf16-mxfp4-kv-bf16",
        model_dsl::trace, Model::<ShippedW1, ShippedW2, ShippedKv>::b20(),
    ),
    (
        "gptoss-120b-bf16-mxfp4-kv-bf16",
        model_dsl::trace, Model::<ShippedW1, ShippedW2, ShippedKv>::b120(),
    ),
    (
        "gptoss-120b-bf16-mxfp4-kv-bf16-tp2",
        model_dsl::trace, Model::<ShippedW1, ShippedW2, ShippedKv, 2>::b120(),
    ),
];

model_dsl::allow_import! {
    import_hf::<SfBf16, ShippedW1, ShippedW2, ShippedKv> => ("gptoss-20b-bf16-mxfp4-kv-bf16", Model::<ShippedW1, ShippedW2, ShippedKv>::b20()),
    import_hf::<SfBf16, ShippedW1, ShippedW2, ShippedKv> => ("gptoss-120b-bf16-mxfp4-kv-bf16", Model::<ShippedW1, ShippedW2, ShippedKv>::b120()),
}
