use model_dsl::axes::{Bf16, NativeKv};
use model_dsl::load::SfBf16;
use model_dsl::{Plan, Plane};

pub mod forward;
pub mod import;
pub mod model;

use import::import_hf;
use model::Model;

pub type ShippedW1 = Bf16;
pub type ShippedKv = NativeKv;

pub type TraceFn = fn(Plane) -> Plan;

pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalog![
    (
        "qwen35-a3b-bf16-kv-bf16",
        model_dsl::trace_hybrid, Model::<ShippedW1, ShippedKv>::a3b(),
    ),
    (
        "qwen35-d3b-bf16-kv-bf16",
        model_dsl::trace_hybrid, Model::<ShippedW1, ShippedKv>::d3b(),
    ),
    (
        "qwen35-d0.8b-bf16-kv-bf16",
        model_dsl::trace_hybrid, Model::<ShippedW1, ShippedKv>::d0_8b(),
    ),
    (
        "qwen35-a3b-bf16-kv-bf16-tp2",
        model_dsl::trace_hybrid, Model::<ShippedW1, ShippedKv, 2>::a3b(),
    ),
];

model_dsl::allow_import! {
    import_hf::<SfBf16, ShippedW1, ShippedKv> => ("qwen35-a3b-bf16-kv-bf16", Model::<ShippedW1, ShippedKv>::a3b()),
    import_hf::<SfBf16, ShippedW1, ShippedKv> => ("qwen35-d3b-bf16-kv-bf16", Model::<ShippedW1, ShippedKv>::d3b()),
    import_hf::<SfBf16, ShippedW1, ShippedKv> => ("qwen35-d0.8b-bf16-kv-bf16", Model::<ShippedW1, ShippedKv>::d0_8b()),
}
