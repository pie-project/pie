//! The forms a `[rates]` value takes, each written and read in one place.
//! Which key holds which form is [`super::RATES`]'s table.

use std::fmt::Write as _;

use toml::Spanned;
use toml::de::DeValue;

use super::{Curve, Landing, ProductTime, WidthCurves, by_toml};
use crate::curves::{DenseSettled, FireTerm, KernelCurve};
use crate::registry::{Walk, found, int};

/// A `[rates]` value: how it is written under its key, how it is read (a
/// value that does not read is said at its span and left unstated), and how
/// a layer's lies over the one below where both state it.
pub(super) trait Form: Sized + Clone {
    fn write(&self, key: &str, out: &mut String);
    fn read(walk: &mut Walk<'_>, key: &str, v: &Spanned<DeValue<'_>>) -> Option<Self>;
    /// `self` over `below`, both stated: `self` whole, unless a list's
    /// entries join `below`'s.
    fn over(self, _below: Self) -> Self {
        self
    }
}

/// A field holding its key's value: stated or not, laid over the layer
/// below, written and read.
pub(super) trait Slot: Default + Clone {
    fn stated(&self) -> bool;
    fn lay(self, below: Self) -> Self;
    fn write(&self, key: &str, out: &mut String);
    fn read(walk: &mut Walk<'_>, key: &str, v: &Spanned<DeValue<'_>>) -> Self;
}

impl<T: Form> Slot for Option<T> {
    fn stated(&self) -> bool {
        self.is_some()
    }

    fn lay(self, below: Self) -> Self {
        match (self, below) {
            (Some(mine), Some(theirs)) => Some(mine.over(theirs)),
            (mine, theirs) => mine.or(theirs),
        }
    }

    fn write(&self, key: &str, out: &mut String) {
        if let Some(value) = self {
            value.write(key, out);
        }
    }

    fn read(walk: &mut Walk<'_>, key: &str, v: &Spanned<DeValue<'_>>) -> Self {
        T::read(walk, key, v)
    }
}

/// The settles, one entry per key and many a device: none stated is empty.
/// A load's settle joins the others (another model's), its own replacing
/// one of the same key.
impl Slot for Vec<DenseSettled> {
    fn stated(&self) -> bool {
        !self.is_empty()
    }

    fn lay(self, below: Self) -> Self {
        joined(self, below, |d| d.of)
    }

    fn write(&self, key: &str, out: &mut String) {
        if self.is_empty() {
            return;
        }
        let mut settles: Vec<&DenseSettled> = self.iter().collect();
        settles.sort_by_key(|d| d.of);
        let rows: Vec<String> = settles
            .iter()
            .map(|settled| {
                let ranges: Vec<String> = settled
                    .ranges
                    .iter()
                    .map(|r| {
                        format!(
                            "[{}, {}, {}, {}, {}, {}]",
                            r[0], r[1], r[2], r[3], r[4], r[5]
                        )
                    })
                    .collect();
                let planes: Vec<String> = settled
                    .planes
                    .iter()
                    .map(|(n, k)| format!("[{n}, {k}]"))
                    .collect();
                format!(
                    "  [{}, {}, {}, [{}], [{}]],\n",
                    settled.method,
                    settled.of,
                    settled.code,
                    ranges.join(", "),
                    planes.join(", ")
                )
            })
            .collect();
        let _ = writeln!(out, "{key} = [\n{}]", rows.concat());
    }

    fn read(walk: &mut Walk<'_>, _key: &str, v: &Spanned<DeValue<'_>>) -> Self {
        walk.dense_settled(v)
    }
}

/// `mine` joined to `theirs` entry by entry, by `id`: `mine` replacing an
/// entry both state, the whole sorted by `id`.
pub(super) fn joined<T, K: Ord>(mine: Vec<T>, theirs: Vec<T>, id: impl Fn(&T) -> K) -> Vec<T> {
    let mut all: Vec<T> = theirs
        .into_iter()
        .filter(|t| mine.iter().all(|m| id(m) != id(t)))
        .collect();
    all.extend(mine);
    all.sort_by_key(|t| id(t));
    all
}

/// A positive, finite number, written as a float or a whole number.
fn positive(v: &DeValue<'_>) -> Option<f64> {
    let f = match v {
        DeValue::Float(f) => f.as_str().replace('_', "").parse::<f64>().ok(),
        DeValue::Integer(_) => int(v).map(|i| i as f64),
        _ => None,
    };
    f.filter(|f| f.is_finite() && *f > 0.0)
}

/// A rate: a positive number.
impl Form for f64 {
    fn write(&self, key: &str, out: &mut String) {
        let _ = writeln!(out, "{key} = {self:?}");
    }

    fn read(walk: &mut Walk<'_>, key: &str, v: &Spanned<DeValue<'_>>) -> Option<f64> {
        let rate = positive(v.get_ref());
        if rate.is_none() {
            walk.err(
                v.span().start,
                format!(
                    "`rates.{key}` is a positive number, not {}",
                    found(v.get_ref())
                ),
            );
        }
        rate
    }
}

/// A time in nanoseconds: a whole number.
impl Form for u64 {
    fn write(&self, key: &str, out: &mut String) {
        let _ = writeln!(out, "{key} = {self}");
    }

    fn read(walk: &mut Walk<'_>, key: &str, v: &Spanned<DeValue<'_>>) -> Option<u64> {
        let ns = int(v.get_ref()).and_then(|i| u64::try_from(i).ok());
        if ns.is_none() {
            walk.err(
                v.span().start,
                format!(
                    "`rates.{key}` is a whole number of nanoseconds, not {}",
                    found(v.get_ref())
                ),
            );
        }
        ns
    }
}

/// A count: a whole number, at least one.
impl Form for u32 {
    fn write(&self, key: &str, out: &mut String) {
        let _ = writeln!(out, "{key} = {self}");
    }

    fn read(walk: &mut Walk<'_>, key: &str, v: &Spanned<DeValue<'_>>) -> Option<u32> {
        let count = int(v.get_ref())
            .and_then(|i| u32::try_from(i).ok())
            .filter(|c| *c > 0);
        if count.is_none() {
            walk.err(
                v.span().start,
                format!(
                    "`rates.{key}` is a whole number, at least one, not {}",
                    found(v.get_ref())
                ),
            );
        }
        count
    }
}

/// A curve: `[[x, rate], ...]`.
impl Form for Curve {
    fn write(&self, key: &str, out: &mut String) {
        let points: Vec<String> = self.iter().map(|(x, r)| format!("[{x}, {r:?}]")).collect();
        let _ = writeln!(out, "{key} = [{}]", points.join(", "));
    }

    fn read(walk: &mut Walk<'_>, key: &str, v: &Spanned<DeValue<'_>>) -> Option<Curve> {
        walk.curve(v, key)
    }
}

/// A curve at each width: `[[width, [[x, rate], ...]], ...]`.
impl Form for WidthCurves {
    fn write(&self, key: &str, out: &mut String) {
        let curves: Vec<String> = self
            .iter()
            .map(|(w, c)| {
                let points: Vec<String> = c.iter().map(|(x, r)| format!("[{x}, {r:?}]")).collect();
                format!("  [{w}, [{}]],\n", points.join(", "))
            })
            .collect();
        let _ = writeln!(out, "{key} = [\n{}]", curves.concat());
    }

    fn read(walk: &mut Walk<'_>, key: &str, v: &Spanned<DeValue<'_>>) -> Option<WidthCurves> {
        walk.width_curves(v, key)
    }
}

/// The products' times, entry by entry: a load's shapes join those already
/// measured, its own replacing a shape both state.
impl Form for Vec<ProductTime> {
    fn write(&self, key: &str, out: &mut String) {
        let rows: Vec<String> = self
            .iter()
            .map(|t| {
                format!(
                    "  [{}, {}, {}, {}, {}, {:?}, {:?}],\n",
                    t.kind, t.rows, t.k, t.m, t.kernel, t.ns, t.spread
                )
            })
            .collect();
        let _ = writeln!(out, "{key} = [\n{}]", rows.concat());
    }

    fn read(walk: &mut Walk<'_>, _key: &str, v: &Spanned<DeValue<'_>>) -> Option<Self> {
        walk.product_times(v)
    }

    fn over(self, below: Self) -> Self {
        joined(self, below, ProductTime::key)
    }
}

/// The dense kernels' curves, entry by entry as the products' times.
impl Form for Vec<KernelCurve> {
    fn write(&self, key: &str, out: &mut String) {
        let rows: Vec<String> = self
            .iter()
            .map(|c| {
                let points: Vec<String> = c
                    .curve
                    .iter()
                    .map(|(x, r)| format!("[{x}, {r:?}]"))
                    .collect();
                let (at, why) = c.stop.unwrap_or((0, 0));
                let spread: Vec<String> = c
                    .spread
                    .iter()
                    .map(|(x, f)| format!("[{x}, {f:?}]"))
                    .collect();
                format!(
                    "  [{}, {}, {}, {}, {at}, {why}, {}, [{}], [{}]],\n",
                    c.kind,
                    c.key,
                    c.n,
                    c.k,
                    by_toml(&c.by),
                    points.join(", "),
                    spread.join(", ")
                )
            })
            .collect();
        let _ = writeln!(out, "{key} = [\n{}]", rows.concat());
    }

    fn read(walk: &mut Walk<'_>, _key: &str, v: &Spanned<DeValue<'_>>) -> Option<Self> {
        walk.kernel_curves(v)
    }

    fn over(self, below: Self) -> Self {
        joined(self, below, KernelCurve::id)
    }
}

/// The fire's terms, one entry per kind and shape: a layer's entries join
/// the one below, a shape's own replacing it.
impl Form for Vec<FireTerm> {
    fn write(&self, key: &str, out: &mut String) {
        let rows: Vec<String> = self
            .iter()
            .map(|t| {
                let shape: Vec<String> = t.shape.iter().map(u32::to_string).collect();
                let points: Vec<String> = t
                    .curve
                    .iter()
                    .map(|(x, ns)| format!("[{x}, {ns:?}]"))
                    .collect();
                format!(
                    "  [{}, [{}], {}, [{}]],\n",
                    t.kind,
                    shape.join(", "),
                    by_toml(&t.by),
                    points.join(", ")
                )
            })
            .collect();
        let _ = writeln!(out, "{key} = [\n{}]", rows.concat());
    }

    fn read(walk: &mut Walk<'_>, _key: &str, v: &Spanned<DeValue<'_>>) -> Option<Self> {
        walk.fire_terms(v)
    }

    fn over(self, below: Self) -> Self {
        joined(self, below, FireTerm::id)
    }
}

/// How a machine lands weights: seven positive numbers.
impl Form for Landing {
    fn write(&self, key: &str, out: &mut String) {
        let values: Vec<String> = self.values().iter().map(|v| format!("{v:?}")).collect();
        let _ = writeln!(out, "{key} = [{}]", values.join(", "));
    }

    fn read(walk: &mut Walk<'_>, key: &str, v: &Spanned<DeValue<'_>>) -> Option<Landing> {
        let values: Option<Vec<f64>> = match v.get_ref() {
            DeValue::Array(items) => items.iter().map(|item| positive(item.get_ref())).collect(),
            _ => None,
        };
        let landing = values
            .and_then(|values| <[f64; 7]>::try_from(values).ok())
            .map(Landing::of);
        if landing.is_none() {
            walk.err(
                v.span().start,
                format!(
                    "`rates.{key}` is a list of seven positive numbers: [copy_latency_ns, \
                     copy_gbps, copy_full_bytes, fill_gbps, read_spawn_ns, read_thread_gbps, \
                     read_whole_gbps]"
                ),
            );
        }
        landing
    }
}
