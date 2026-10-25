//! What a layout may use: weights, dtypes and the arithmetic their shapes
//! are written in.

use crate::{Dtype, Precision, Weight};
use std::cell::Cell;

use starlark::environment::GlobalsBuilder;
use starlark::values::float::UnpackFloat;
use starlark::values::list::ListRef;
use starlark::values::list::UnpackList;
use starlark::values::structs::AllocStruct;
use starlark::values::tuple::UnpackTuple;
use starlark::values::{Heap, Value};
use starlark_derive::starlark_module;

use crate::star::values::{
    DtypeValue, PrecisionValue, WeightValue, precision, precision_word, word,
};

thread_local! {
    /// The dtype the deployment whose layout runs computes in: its first
    /// stated dtype's (`compute`), a weight's where it states none.
    pub(crate) static COMPUTE: Cell<Option<Dtype>> = const { Cell::new(None) };
}

#[starlark_module]
pub(crate) fn layout(builder: &mut GlobalsBuilder) {
    /// A weight of `shape`, stored as `dtype` (a dtype, or a precision whose
    /// products quantize their activations), or, stating none, in the
    /// deployment's compute dtype. It is named by where the layout puts it.
    /// A weight stating its `values` (row-major, as many as its shape holds)
    /// holds them whatever the checkpoint does.
    fn weight<'v>(
        #[starlark(require = pos)] shape: UnpackList<u64>,
        #[starlark(require = pos)] dtype: Option<Value<'v>>,
        #[starlark(require = named)] values: Option<UnpackList<UnpackFloat>>,
    ) -> anyhow::Result<WeightValue> {
        let stated = match dtype.filter(|d| !d.is_none()) {
            Some(dtype) => precision(dtype)?,
            None => Precision::of(COMPUTE.with(Cell::get).ok_or_else(|| {
                anyhow::anyhow!("a weight states no dtype and its deployment computes in none")
            })?),
        };
        let mut w = Weight::at(String::new(), shape.items, stated);
        if let Some(values) = values {
            let held: u64 = w.shape.iter().product();
            if values.items.len() as u64 != held {
                anyhow::bail!(
                    "a weight of shape {:?} holds {held} values, and states {}",
                    w.shape,
                    values.items.len()
                );
            }
            w.values = Some(values.items.iter().map(|v| v.0 as f32).collect());
        }
        Ok(WeightValue(w))
    }

    /// The deployment's stated dtypes named as the layout reads them: one
    /// dtype fills every role, else each role takes one in order. With
    /// `exact`, each role must be stated.
    fn roles<'v>(
        #[starlark(require = pos)] deploy: Value<'v>,
        #[starlark(args)] names: UnpackTuple<String>,
        #[starlark(require = named, default = false)] exact: bool,
        heap: Heap<'v>,
    ) -> anyhow::Result<Value<'v>> {
        let names = names.items;
        let weights = deploy
            .get_attr("weights", heap)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .ok_or_else(|| anyhow::anyhow!("a deployment states its weights' dtypes"))?;
        let weights: Vec<Value<'v>> = ListRef::from_value(weights)
            .ok_or_else(|| anyhow::anyhow!("a deployment's weights are a list"))?
            .iter()
            .collect();
        let said = || {
            weights
                .iter()
                .map(|w| w.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let fields: Vec<(String, Value<'v>)> = if weights.len() == names.len() {
            names.into_iter().zip(weights.iter().copied()).collect()
        } else if weights.len() == 1 && !exact {
            names.into_iter().map(|name| (name, weights[0])).collect()
        } else {
            anyhow::bail!(
                "this deployment stores its weights at [{}], and its layout reads {}a dtype for each of: {}",
                said(),
                if exact { "" } else { "one dtype, or " },
                names.join(", "),
            )
        };
        Ok(heap.alloc(AllocStruct(fields)))
    }

    /// The dtype a bank stored as `dtype` computes in.
    fn compute<'v>(#[starlark(require = pos)] dtype: Value<'v>) -> anyhow::Result<DtypeValue> {
        let stated = precision(dtype)?;
        crate::compute_dtype(stated.dtype)
            .map(DtypeValue)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "`{}` is no representation a bank is stored in",
                    precision_word(stated)
                )
            })
    }
}

#[starlark_module]
pub(crate) fn numbers(builder: &mut GlobalsBuilder) {
    /// `x` rounded to the nearest f32.
    fn f32(#[starlark(require = pos)] x: UnpackFloat) -> anyhow::Result<f64> {
        Ok(f64::from(x.0 as f32))
    }

    /// `x` to the power `y`, both rounded to f32 and raised in f32.
    fn powf32(
        #[starlark(require = pos)] x: UnpackFloat,
        #[starlark(require = pos)] y: UnpackFloat,
    ) -> anyhow::Result<f64> {
        Ok(f64::from((x.0 as f32).powf(y.0 as f32)))
    }

    /// `numpy.random.default_rng(seed).integers(0, high, size)`, bit for bit
    /// (`high` past 2^32).
    fn numpy_integers(
        #[starlark(require = pos)] seed: i64,
        #[starlark(require = pos)] high: i64,
        #[starlark(require = pos)] size: u32,
    ) -> anyhow::Result<Vec<i64>> {
        let seed = u128::try_from(seed).map_err(|_| anyhow::anyhow!("a negative seed {seed}"))?;
        let high = u64::try_from(high).map_err(|_| anyhow::anyhow!("a negative bound {high}"))?;
        if high <= 1 << 32 {
            anyhow::bail!("a bound of {high} is drawn by NumPy's 32-bit sampler, not this one");
        }
        let draws = crate::numpy::Generator::seeded(seed).integers(high, size as usize);
        draws
            .into_iter()
            .map(|d| i64::try_from(d).map_err(|_| anyhow::anyhow!("a draw past i64")))
            .collect()
    }

    /// The least prime greater than `n`.
    fn prime_after(#[starlark(require = pos)] n: i64) -> anyhow::Result<i64> {
        let is_prime = |v: i64| {
            if v < 2 {
                return false;
            }
            if v % 2 == 0 {
                return v == 2;
            }
            let mut d = 3;
            while d * d <= v {
                if v % d == 0 {
                    return false;
                }
                d += 2;
            }
            true
        };
        let mut candidate = n + 1;
        while !is_prime(candidate) {
            candidate += 1;
        }
        Ok(candidate)
    }

    fn sqrt(#[starlark(require = pos)] x: UnpackFloat) -> anyhow::Result<f64> {
        Ok(x.0.sqrt())
    }

    fn exp(#[starlark(require = pos)] x: UnpackFloat) -> anyhow::Result<f64> {
        Ok(x.0.exp())
    }

    /// `x` rounded to an f32, its exponential as an f32 computes it.
    fn expf(#[starlark(require = pos)] x: UnpackFloat) -> anyhow::Result<f64> {
        Ok(f64::from((x.0 as f32).exp()))
    }
}

/// The `dtype` namespace: every dtype, by the name the catalog spells it.
pub(crate) fn dtypes(builder: &mut GlobalsBuilder) {
    builder.namespace("dtype", |ns| {
        for dtype in Dtype::ALL {
            ns.set(&word(dtype), DtypeValue(dtype));
        }
        for acted in Precision::ACTED {
            ns.set(&precision_word(acted), PrecisionValue(acted));
        }
    });
}
