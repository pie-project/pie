use model_dsl::axes::{Bf16, NativeKv};
use model_dsl::load::{GgufBf16, SfBf16};
use model_dsl::{Plan, Plane};

pub mod forward;
pub mod import;
pub mod model;
pub mod template;

use import::{import_gguf, import_hf};
use model::Model;

pub type ShippedW1 = Bf16;
pub type ShippedKv = NativeKv;

pub type TraceFn = fn(Plane) -> Plan;

pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalog![
    (
        "gemma4-e4b-bf16-kv-bf16",
        model_dsl::trace, Model::<ShippedW1, ShippedKv>::e4b(),
    ),
    (
        "gemma4-31b-bf16-kv-bf16",
        model_dsl::trace, Model::<ShippedW1, ShippedKv>::b31(),
    ),
    (
        "gemma4-31b-bf16-kv-bf16-tp2",
        model_dsl::trace, Model::<ShippedW1, ShippedKv, 2>::b31(),
    ),
];

model_dsl::allow_import! {
    import_hf::<SfBf16, ShippedW1, ShippedKv> => ("gemma4-e4b-bf16-kv-bf16", Model::<ShippedW1, ShippedKv>::e4b()),
    import_hf::<SfBf16, ShippedW1, ShippedKv> => ("gemma4-31b-bf16-kv-bf16", Model::<ShippedW1, ShippedKv>::b31()),
    import_gguf::<GgufBf16, ShippedW1, ShippedKv> => ("gemma4-e4b-bf16-kv-bf16", Model::<ShippedW1, ShippedKv>::e4b()),
    import_gguf::<GgufBf16, ShippedW1, ShippedKv> => ("gemma4-31b-bf16-kv-bf16", Model::<ShippedW1, ShippedKv>::b31()),
}
