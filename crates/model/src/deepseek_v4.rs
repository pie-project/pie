use model_dsl::axes::{Bf16, NativeKv};
use model_dsl::load::SfBf16;

pub mod forward;
pub mod import;
pub mod model;

use import::import_hf;
use model::Model;

pub type ShippedW1 = Bf16;
pub type ShippedKv = NativeKv;

pub const CATALOG: &[(&str, model_dsl::TraceFn)] = model_dsl::catalog![
    (
        "dsv4-base-bf16-kv-bf16",
        model_dsl::trace,
        Model::<ShippedW1, ShippedKv>::base(),
    ),
    (
        "dsv4-base-bf16-kv-bf16-tp2",
        model_dsl::trace,
        Model::<ShippedW1, ShippedKv, 2>::base(),
    ),
];

model_dsl::allow_import! {
    import_hf::<SfBf16, ShippedW1, ShippedKv> => ("dsv4-base-bf16-kv-bf16", Model::<ShippedW1, ShippedKv>::base()),
    import_hf::<SfBf16, ShippedW1, ShippedKv> => ("dsv4-base-bf16-kv-bf16-tp2", Model::<ShippedW1, ShippedKv, 2>::base()),
}
