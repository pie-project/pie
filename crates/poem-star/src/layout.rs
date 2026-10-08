//! What a layout may use: weights, dtypes and the arithmetic their shapes
//! are written in.

use poem_dsl::{Dtype, Weight};
use starlark::environment::GlobalsBuilder;
use starlark::values::float::UnpackFloat;
use starlark::values::list::UnpackList;
use starlark::values::none::NoneOr;
use starlark_derive::starlark_module;

use crate::values::{DtypeValue, WeightValue, word};

#[starlark_module]
pub(crate) fn layout(builder: &mut GlobalsBuilder) {
    /// A weight named `name`, of `shape`, stored as `dtype`.
    fn weight(
        #[starlark(require = pos)] name: String,
        #[starlark(require = pos)] shape: UnpackList<u64>,
        #[starlark(require = pos)] dtype: &DtypeValue,
    ) -> anyhow::Result<WeightValue> {
        Ok(WeightValue(Weight::sym(name, shape.items, dtype.0)))
    }

    /// The dtype a bank stored as `dtype` computes in.
    fn compute(#[starlark(require = pos)] dtype: &DtypeValue) -> anyhow::Result<DtypeValue> {
        poem_dsl::compute_dtype(dtype.0)
            .map(DtypeValue)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "`{}` is no representation a bank is stored in",
                    word(dtype.0)
                )
            })
    }

    /// The environment variable `name`, or `None`.
    fn env(#[starlark(require = pos)] name: String) -> anyhow::Result<NoneOr<String>> {
        Ok(match std::env::var(name) {
            Ok(value) => NoneOr::Other(value),
            Err(_) => NoneOr::None,
        })
    }
}

#[starlark_module]
pub(crate) fn numbers(builder: &mut GlobalsBuilder) {
    /// `x` rounded to the nearest f32, as a model's Rust spelling computes.
    fn f32(#[starlark(require = pos)] x: UnpackFloat) -> anyhow::Result<f64> {
        Ok(f64::from(x.0 as f32))
    }

    fn sqrt(#[starlark(require = pos)] x: UnpackFloat) -> anyhow::Result<f64> {
        Ok(x.0.sqrt())
    }
}

/// The `dtype` namespace: every dtype, by the name the catalog spells it.
pub(crate) fn dtypes(builder: &mut GlobalsBuilder) {
    builder.namespace("dtype", |ns| {
        for dtype in Dtype::ALL {
            ns.set(&word(dtype), DtypeValue(dtype));
        }
    });
}
