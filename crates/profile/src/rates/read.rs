//! Reading `[rates]`: each entry of a profile's rates table checked and
//! taken, or refused (a checked-in file) or skipped (a user's) by name.

use toml::Spanned;
use toml::de::DeValue;

use super::{MEASURED_BY, MEASURED_ON};
use crate::registry::{Strict, Unread, Walk, found, int};

/// A stored key's `method, [[kernel, code], ...]` ([`crate::rates::MeasuredBy`]).
pub(crate) fn by_of(
    method: &Spanned<DeValue<'_>>,
    kernels: &Spanned<DeValue<'_>>,
) -> Option<crate::rates::MeasuredBy> {
    let whole = |v: &DeValue<'_>| int(v).and_then(|x| u32::try_from(x).ok());
    let method = whole(method.get_ref())?;
    let DeValue::Array(items) = kernels.get_ref() else {
        return None;
    };
    let kernels = items
        .iter()
        .map(|item| match item.get_ref() {
            DeValue::Array(p) if p.len() == 2 => match p[0].get_ref() {
                DeValue::String(kernel) if !kernel.is_empty() => {
                    whole(p[1].get_ref()).map(|code| (kernel.to_string(), code))
                }
                _ => None,
            },
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some(crate::rates::MeasuredBy { method, kernels })
}

impl Walk<'_> {
    pub(crate) fn rates(&mut self, value: &Spanned<DeValue<'_>>) -> crate::rates::Rates {
        let mut rates = crate::rates::Rates::default();
        let DeValue::Table(t) = value.get_ref() else {
            self.err(value.span().start, "`rates` is a table");
            return rates;
        };
        for (k, v) in t.iter() {
            // A user file is a load's own output: an entry of it that does
            // not read (written by another pie, in another form) is skipped
            // alone, by name, and measured again; the rest of the file is
            // read. No older form is read in its place.
            let before = self.diags.len();
            let (key, rates) = (k.get_ref().to_string(), &mut rates);
            self.rate_entry(k, v, rates);
            if matches!(self.strict, Strict::User | Strict::Measured) && self.diags.len() > before {
                let why: Vec<String> = self.diags.drain(before..).map(|d| d.message).collect();
                rates.unstate(&key);
                self.skipped_rates
                    .push(format!("{key} ({})", why.join("; ")));
            }
        }
        rates
    }

    /// One `[rates]` entry, `k = v`, into `rates`: a measurement by its
    /// key's form ([`super::RATES`]), or the keys those measurements were
    /// taken under.
    fn rate_entry(
        &mut self,
        k: &Spanned<std::borrow::Cow<'_, str>>,
        v: &Spanned<DeValue<'_>>,
        rates: &mut super::Rates,
    ) {
        match k.get_ref().as_ref() {
            key @ (MEASURED_ON | MEASURED_BY) if self.strict == Strict::CheckedIn => {
                self.err(
                    k.span().start,
                    format!(
                        "`rates.{key}` keys a user's file's measurements: a checked-in \
                         profile states its rates as decisions, unkeyed"
                    ),
                );
            }
            MEASURED_ON => match v.get_ref() {
                DeValue::String(on) if !on.trim().is_empty() => {
                    rates.measured_on = Some(on.to_string());
                }
                other => self.err(
                    v.span().start,
                    format!(
                        "`rates.{MEASURED_ON}` names the device measured on, not {}",
                        found(other)
                    ),
                ),
            },
            MEASURED_BY => rates.measured_by = self.measured_by(v),
            key if rates.read_key(self, key, v) => {}
            other => {
                self.stale_rates.push(other.to_string());
                self.unread
                    .push((format!("rates.{other}"), Unread::Entry(v.span())));
                // A user file is a load's own output, written by an older
                // pie: a rate this pie no longer reads is left unread, the
                // rest of the file read (a checked-in file stating one is
                // refused by name).
                if self.strict == Strict::User {
                    self.ignored_rates.push(other.to_string());
                    return;
                }
                let holds: Vec<String> =
                    super::RATES.iter().map(|key| format!("`{key}`")).collect();
                self.err(
                    k.span().start,
                    format!(
                        "`rates` holds {}, `{MEASURED_ON}` and `{MEASURED_BY}`, not `{other}`",
                        holds.join(", ")
                    ),
                );
            }
        }
    }

    /// `rates.product_ns`: a list of `[kind, rows, k, m, kernel, ns, spread]`,
    /// whole numbers then a positive time and a spread at least zero, no
    /// shape twice.
    pub(super) fn product_times(
        &mut self,
        value: &Spanned<DeValue<'_>>,
    ) -> Option<Vec<crate::rates::ProductTime>> {
        let what = "`rates.product_ns` is a list of [kind, rows, k, m, kernel, ns, spread], each \
                    shape once";
        let DeValue::Array(entries) = value.get_ref() else {
            self.err(
                value.span().start,
                format!("{what}, not {}", found(value.get_ref())),
            );
            return None;
        };
        let number = |v: &DeValue<'_>| match v {
            DeValue::Float(f) => f.as_str().replace('_', "").parse::<f64>().ok(),
            DeValue::Integer(_) => int(v).map(|i| i as f64),
            _ => None,
        };
        let mut out: Vec<crate::rates::ProductTime> = Vec::with_capacity(entries.len());
        for entry in entries.iter() {
            let parsed = match entry.get_ref() {
                DeValue::Array(f) if f.len() == 7 => {
                    let whole = |i: usize| int(f[i].get_ref()).and_then(|v| u32::try_from(v).ok());
                    match (
                        whole(0),
                        whole(1),
                        whole(2),
                        whole(3),
                        whole(4),
                        number(f[5].get_ref()),
                        number(f[6].get_ref()),
                    ) {
                        (
                            Some(kind),
                            Some(rows),
                            Some(k),
                            Some(m),
                            Some(kernel),
                            Some(ns),
                            Some(spread),
                        ) if ns.is_finite() && ns > 0.0 && spread.is_finite() && spread >= 0.0 => {
                            Some(crate::rates::ProductTime {
                                kind,
                                rows,
                                k,
                                m,
                                kernel,
                                ns,
                                spread,
                            })
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            match parsed {
                Some(t) if out.iter().all(|o| o.key() != t.key()) => out.push(t),
                _ => {
                    self.err(
                        entry.span().start,
                        format!("{what}, not {}", found(entry.get_ref())),
                    );
                    return None;
                }
            }
        }
        Some(out)
    }

    /// `rates.measured_by`: `[[rate, method, [[kernel, code], ...]], ...]`,
    /// each rate of [`crate::rates::KEYED`] once.
    fn measured_by(
        &mut self,
        v: &Spanned<DeValue<'_>>,
    ) -> Option<Vec<(String, crate::rates::MeasuredBy)>> {
        let what = "`rates.measured_by` is a list of [rate, method, [[kernel, code], ...]], each \
                    rate `[rates]` keys once";
        let DeValue::Array(entries) = v.get_ref() else {
            self.err(
                v.span().start,
                format!("{what}, not {}", found(v.get_ref())),
            );
            return None;
        };
        let mut out: Vec<(String, crate::rates::MeasuredBy)> = Vec::with_capacity(entries.len());
        for entry in entries.iter() {
            let parsed = match entry.get_ref() {
                DeValue::Array(f) if f.len() == 3 => match f[0].get_ref() {
                    DeValue::String(rate)
                        if crate::rates::KEYED.contains(&rate.as_ref())
                            && out.iter().all(|(r, _)| r != rate.as_ref()) =>
                    {
                        by_of(&f[1], &f[2]).map(|by| (rate.to_string(), by))
                    }
                    _ => None,
                },
                _ => None,
            };
            match parsed {
                Some(keyed) => out.push(keyed),
                None => {
                    self.err(
                        entry.span().start,
                        format!("{what}, not {}", found(entry.get_ref())),
                    );
                    return None;
                }
            }
        }
        Some(out)
    }

    /// `rates.dense_settled_keyed`: a list of `[method, of, code, [[bottom,
    /// top, replaced, vector, wide, tier], ...]]`, whole numbers, one per
    /// key (`of`), no key twice.
    pub(super) fn dense_settled(
        &mut self,
        v: &Spanned<DeValue<'_>>,
    ) -> Vec<crate::curves::DenseSettled> {
        let what = "`rates.dense_settled_keyed` is a list of [method, of, code, [[bottom, top, \
                    replaced, vector, wide, tier], ...], [[n, k], ...]], whole numbers, one per key";
        let whole = |v: &DeValue<'_>| int(v).and_then(|x| u32::try_from(x).ok());
        let one = |v: &DeValue<'_>| match v {
            DeValue::Array(f) if f.len() == 5 => {
                let head = whole(f[0].get_ref())
                    .zip(whole(f[1].get_ref()))
                    .zip(whole(f[2].get_ref()));
                let ranges = match f[3].get_ref() {
                    DeValue::Array(rs) => rs
                        .iter()
                        .map(|r| match r.get_ref() {
                            DeValue::Array(x) if x.len() == 6 => {
                                let mut out = [0u32; 6];
                                for (slot, x) in out.iter_mut().zip(x.iter()) {
                                    *slot = whole(x.get_ref())?;
                                }
                                Some(out)
                            }
                            _ => None,
                        })
                        .collect::<Option<Vec<[u32; 6]>>>(),
                    _ => None,
                };
                let planes = match f[4].get_ref() {
                    DeValue::Array(ps) => ps
                        .iter()
                        .map(|p| match p.get_ref() {
                            DeValue::Array(x) if x.len() == 2 => {
                                Some((whole(x[0].get_ref())?, whole(x[1].get_ref())?))
                            }
                            _ => None,
                        })
                        .collect::<Option<Vec<(u32, u32)>>>(),
                    _ => None,
                };
                head.zip(ranges)
                    .zip(planes)
                    .map(
                        |((((method, of), code), ranges), planes)| crate::curves::DenseSettled {
                            method,
                            of,
                            code,
                            ranges,
                            planes,
                        },
                    )
            }
            _ => None,
        };
        let DeValue::Array(all) = v.get_ref() else {
            self.err(
                v.span().start,
                format!("{what}, not {}", found(v.get_ref())),
            );
            return Vec::new();
        };
        // One entry in another form (another pie's) or of a key stated
        // before is skipped alone, by name, and that model settles again;
        // the list's others are read. No older form is read in its place.
        let mut out: Vec<crate::curves::DenseSettled> = Vec::with_capacity(all.len());
        for (at, entry) in all.iter().enumerate() {
            match one(entry.get_ref()) {
                Some(settled) if out.iter().all(|d| d.of != settled.of) => out.push(settled),
                Some(settled) => self.skipped_rates.push(format!(
                    "{}[{at}] (its key {:#010x} stated twice; settled again)",
                    crate::rates::DENSE_SETTLED,
                    settled.of
                )),
                None => self.skipped_rates.push(format!(
                    "{}[{at}] ({what}, not {}; settled again)",
                    crate::rates::DENSE_SETTLED,
                    found(entry.get_ref())
                )),
            }
        }
        out
    }

    /// `rates.kernel_curves`: a list of `[kind, key, n, k,
    /// stopped at rows, why, method, [[kernel, code], ...], curve, spreads]`,
    /// whole numbers, the curve's key, then a curve of `[rows, TFLOP/s]`
    /// ascending in rows and its points' spreads; no kernel and shape twice;
    /// a stop of `0, 0` is none.
    pub(super) fn kernel_curves(
        &mut self,
        value: &Spanned<DeValue<'_>>,
    ) -> Option<Vec<crate::curves::KernelCurve>> {
        let what = "`rates.kernel_curves` is a list of [kind, key, n, k, stopped \
                    at rows, why, method, [[kernel, code], ...], [[rows, TFLOP/s], ...], [[rows, \
                    spread], ...]], no kernel and shape twice";
        let DeValue::Array(entries) = value.get_ref() else {
            self.err(
                value.span().start,
                format!("{what}, not {}", found(value.get_ref())),
            );
            return None;
        };
        let mut out: Vec<crate::curves::KernelCurve> = Vec::with_capacity(entries.len());
        for (at, entry) in entries.iter().enumerate() {
            // One entry in another form (another pie's) is skipped alone, by
            // name, and measured again; the list's others are read.
            let before = self.diags.len();
            if let Some(curve) = self.kernel_curve(entry, &out, what) {
                out.push(curve);
            } else if self.diags.len() > before {
                let why: Vec<String> = self.diags.drain(before..).map(|d| d.message).collect();
                self.skipped_rates.push(format!(
                    "{}[{at}] ({}; measured again)",
                    crate::rates::KERNEL_CURVES,
                    why.join("; ")
                ));
            }
        }
        Some(out)
    }

    /// One entry of `rates.kernel_curves`, beside `out` (the
    /// list's entries read before it): `None` with its complaint where it
    /// does not read.
    fn kernel_curve(
        &mut self,
        entry: &Spanned<DeValue<'_>>,
        out: &[crate::curves::KernelCurve],
        what: &str,
    ) -> Option<crate::curves::KernelCurve> {
        {
            let fields = match entry.get_ref() {
                DeValue::Array(f) if f.len() == 10 => {
                    let whole = |i: usize| int(f[i].get_ref()).and_then(|v| u32::try_from(v).ok());
                    match (
                        whole(0),
                        whole(1),
                        whole(2),
                        whole(3),
                        whole(4),
                        whole(5),
                        by_of(&f[6], &f[7]),
                    ) {
                        (
                            Some(kind),
                            Some(key),
                            Some(n),
                            Some(k),
                            Some(at),
                            Some(why),
                            Some(by),
                        ) if n > 0 && k > 0 && (at == 0) == (why == 0) => Some((
                            kind,
                            key,
                            n,
                            k,
                            (at > 0).then_some((at, why)),
                            by,
                            &f[8],
                            Some(&f[9]),
                        )),
                        _ => None,
                    }
                }
                _ => None,
            };
            let Some((kind, key, n, k, stop, by, points, spread)) = fields
                .filter(|&(kind, key, n, k, ..)| out.iter().all(|c| c.id() != (kind, key, n, k)))
            else {
                self.err(
                    entry.span().start,
                    format!("{what}, not {}", found(entry.get_ref())),
                );
                return None;
            };
            // A curve stopped at its first rows holds no point.
            let empty = matches!(points.get_ref(), DeValue::Array(p) if p.is_empty());
            let curve = if empty && stop.is_some() {
                Vec::new()
            } else {
                self.curve(points, crate::rates::KERNEL_CURVES)?
            };
            // The points' spreads, `[rows, fraction at least zero]` each.
            let spread = match spread.map(|s| s.get_ref()) {
                None => Vec::new(),
                Some(DeValue::Array(items)) => {
                    let mut out = Vec::with_capacity(items.len());
                    for item in items.iter() {
                        let pair = match item.get_ref() {
                            DeValue::Array(p) if p.len() == 2 => {
                                let x = int(p[0].get_ref()).and_then(|v| u32::try_from(v).ok());
                                let f = match p[1].get_ref() {
                                    DeValue::Float(f) => f.as_str().replace('_', "").parse().ok(),
                                    DeValue::Integer(_) => int(p[1].get_ref()).map(|i| i as f64),
                                    _ => None,
                                }
                                .filter(|f: &f64| f.is_finite() && *f >= 0.0);
                                x.zip(f)
                            }
                            _ => None,
                        };
                        match pair {
                            Some(pair) => out.push(pair),
                            None => {
                                self.err(
                                    item.span().start,
                                    format!("{what}: a spread is [rows, fraction at least zero]"),
                                );
                                return None;
                            }
                        }
                    }
                    out
                }
                Some(_) => {
                    self.err(
                        entry.span().start,
                        format!("{what}: the spreads are a list"),
                    );
                    return None;
                }
            };
            Some(crate::curves::KernelCurve {
                kind,
                key,
                n,
                k,
                curve,
                stop,
                spread,
                by,
            })
        }
    }

    /// `rates.<key>` (`gemm_tflops_by_rows`, `attn_tflops_by_rows_keys`): `[outer, curve]`
    /// pairs, a whole positive outer key (a width, a row tile, rows) and a
    /// curve of `[x, TFLOP/s]` each, strictly ascending in the outer key.
    pub(super) fn width_curves(
        &mut self,
        value: &Spanned<DeValue<'_>>,
        key: &str,
    ) -> Option<super::WidthCurves> {
        let (outer, inner) = if key == crate::rates::ATTN_BY_ROWS_KEYS {
            ("rows", "keys")
        } else {
            ("width", "rows")
        };
        let what = format!(
            "`rates.{key}` is a non-empty list of [{outer}, [[{inner}, TFLOP/s], ...]] pairs, \
             ascending in {outer}"
        );
        let what = what.as_str();
        let DeValue::Array(widths) = value.get_ref() else {
            self.err(
                value.span().start,
                format!("{what}, not {}", found(value.get_ref())),
            );
            return None;
        };
        let mut out: Vec<(u32, Vec<(u32, f64)>)> = Vec::with_capacity(widths.len());
        for entry in widths.iter() {
            let pair = match entry.get_ref() {
                DeValue::Array(pair) if pair.len() == 2 => int(pair[0].get_ref())
                    .and_then(|i| u32::try_from(i).ok())
                    .filter(|&w| w > 0)
                    .map(|w| (w, &pair[1])),
                _ => None,
            };
            let Some((width, curve)) = pair.filter(|&(w, _)| out.last().is_none_or(|l| w > l.0))
            else {
                self.err(
                    entry.span().start,
                    format!("{what}, not {}", found(entry.get_ref())),
                );
                return None;
            };
            out.push((width, self.curve(curve, key)?));
        }
        if out.is_empty() {
            self.err(value.span().start, format!("{what}, not an empty list"));
            return None;
        }
        Some(out)
    }

    /// `rates.<key>`: `[x, rate]` pairs, a whole positive `x` and a positive
    /// rate each, non-empty and strictly ascending in `x`.
    pub(super) fn curve(
        &mut self,
        value: &Spanned<DeValue<'_>>,
        key: &str,
    ) -> Option<Vec<(u32, f64)>> {
        let what = format!(
            "`rates.{key}` is a non-empty list of [x, rate] pairs, ascending in x, each rate \
             positive"
        );
        let DeValue::Array(points) = value.get_ref() else {
            self.err(
                value.span().start,
                format!("{what}, not {}", found(value.get_ref())),
            );
            return None;
        };
        let mut curve: Vec<(u32, f64)> = Vec::with_capacity(points.len());
        for point in points.iter() {
            let pair = match point.get_ref() {
                DeValue::Array(pair) if pair.len() == 2 => {
                    let x = int(pair[0].get_ref())
                        .and_then(|i| u32::try_from(i).ok())
                        .filter(|&m| m > 0);
                    let rate = match pair[1].get_ref() {
                        DeValue::Float(f) => f.as_str().replace('_', "").parse::<f64>().ok(),
                        DeValue::Integer(_) => int(pair[1].get_ref()).map(|i| i as f64),
                        _ => None,
                    }
                    .filter(|f| f.is_finite() && *f > 0.0);
                    x.zip(rate)
                }
                _ => None,
            };
            match pair {
                Some((x, r)) if curve.last().is_none_or(|&(last, _)| x > last) => {
                    curve.push((x, r))
                }
                _ => {
                    self.err(
                        point.span().start,
                        format!("{what}, not {}", found(point.get_ref())),
                    );
                    return None;
                }
            }
        }
        if curve.is_empty() {
            self.err(value.span().start, format!("{what}, not an empty list"));
            return None;
        }
        Some(curve)
    }
}
