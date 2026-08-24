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
        "gemma4-e4b-bf16-kv-bf16",
        model_dsl::trace,
        Model::<ShippedW1, ShippedKv>::e4b(),
    ),
    (
        "gemma4-31b-bf16-kv-bf16",
        model_dsl::trace,
        Model::<ShippedW1, ShippedKv>::b31(),
    ),
    (
        "gemma4-31b-bf16-kv-bf16-tp2",
        model_dsl::trace,
        Model::<ShippedW1, ShippedKv, 2>::b31(),
    ),
];

model_dsl::allow_import! {
    import_hf::<SfBf16, ShippedW1, ShippedKv> => ("gemma4-e4b-bf16-kv-bf16", Model::<ShippedW1, ShippedKv>::e4b()),
    import_hf::<SfBf16, ShippedW1, ShippedKv> => ("gemma4-31b-bf16-kv-bf16", Model::<ShippedW1, ShippedKv>::b31()),
    // A `-tp2` row reads its sibling's table VERBATIM -- the same fn over the
    // same structure, at the degree the text was traced at. Nothing about a
    // checkpoint changes when a deployment cuts it, so the flavors it can be
    // built from do not change either.
    import_hf::<SfBf16, ShippedW1, ShippedKv> => ("gemma4-31b-bf16-kv-bf16-tp2", Model::<ShippedW1, ShippedKv, 2>::b31()),
}
