//! What a layout may use: weights, dtypes and the arithmetic their shapes
//! are written in.

use crate::{Dtype, Weight};
use starlark::environment::GlobalsBuilder;
use starlark::values::float::UnpackFloat;
use starlark::values::list::UnpackList;
use starlark::values::none::NoneOr;
use starlark_derive::starlark_module;

use crate::star::values::{DtypeValue, WeightValue, word};

thread_local! {
    static ENV: std::cell::RefCell<Vec<(String, Option<String>)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// `f`, with the variables of `vars` (`None` unset) as a layout's `env`
/// reads them on this thread, whatever the process's environment holds.
pub fn with_env<R>(vars: &[(&str, Option<&str>)], f: impl FnOnce() -> R) -> R {
    let held: Vec<(String, Option<String>)> = vars
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.map(str::to_string)))
        .collect();
    let before = ENV.with(|env| std::mem::replace(&mut *env.borrow_mut(), held));
    let out = f();
    ENV.with(|env| *env.borrow_mut() = before);
    out
}

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
        crate::compute_dtype(dtype.0)
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
        if let Some(value) = ENV.with(|env| {
            env.borrow()
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.clone())
        }) {
            return Ok(match value {
                Some(value) => NoneOr::Other(value),
                None => NoneOr::None,
            });
        }
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

    /// `x` to the power `y`, both rounded to f32 and raised in f32, as a
    /// model's Rust spelling computes.
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
    });
}
