//! Weights and per-PE data onto the wafer: what goes where ([`Binding`]). A binding names a staged
//! array (or a generated one: rcfg, tables), the rectangle it covers and the PE symbol it lands
//! in; the engine's placer turns a program's bindings into device copies in a fixed order.
//!
//! The tile-making half (`prep.rs` and the Python preps) stays outside for now (ARCHITECTURE.md:
//! the [`Transform`]s are its future home).

use serde::Serialize;

use super::floor::Rect;
use super::npy::Npy;

/// Where a copy's words come from. The words are row-major over the binding's rectangle, one
/// PE's after another; the per-PE count is their number over the rectangle's PEs.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Source {
    /// A staged npy (`<prefix><name>.npy`), a block of it: its leading index range and the
    /// per-PE word count.
    Staged {
        name: String,
        index: (u32, u32),
        words: u32,
    },
    /// Any sub-array of a staged npy ([`Slice`]): every block copy the `place*.rs` made.
    Slice(Slice),
    /// Every PE's rcfg words over the rectangle ([`super::emit::rcfg`]).
    Rcfg,
}

/// A sub-array of a staged npy (`<prefix><name>.npy`) laid onto a rectangle.
///
/// The array is `[lead?][d0]..[dk-1][per-PE words..]`. `lead` fixes the leading index when the
/// array has that axis (more dims than `dims` plus the per-PE axis; a table shared by every unit
/// has none, and the `lead` is skipped). `dims` are ranges of the next axes; the axes after them
/// make one PE's words. The ranges' elements go row-major (the last range fastest), one PE each,
/// each PE's words zero-padded to `pad_to` and written `across` times in a row (one row's
/// gammas across a band's columns), the whole repeated `repeat` times down the rectangle's rows
/// (a row-invariant table: the head's `hg`). `optional`: no copy when the file is not staged
/// (the CY's optional tables).
///
/// The families' first forms map onto it (ARCHITECTURE.md, change log F):
/// B `Block { lead, rows, cols }` = `dims: [rows, cols]` (or `[rows]`), `Repeat { rows, times }`
/// = `dims: [rows], repeat: times`, `IfStaged(..)` = `optional: true`; C `Block { rows, cols }`
/// = `dims: [rows, cols]`, `RowsAcross { rows }` = `dims: [rows], across: rect.w`; D `Slice` with
/// `across: 1, optional: false`; E `Block { ranges, repeat_rows }` = `dims: ranges`,
/// `repeat: rect.h` when `repeat_rows`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Slice {
    pub name: String,
    pub lead: Option<u32>,
    pub dims: Vec<(u32, u32)>,
    pub pad_to: Option<u32>,
    pub across: u32,
    pub repeat: u32,
    pub optional: bool,
}

impl Slice {
    /// The `dims` ranges of `name`, one PE an element, nothing else.
    pub fn of(name: &str, lead: Option<u32>, dims: Vec<(u32, u32)>) -> Slice {
        Slice {
            name: name.to_string(),
            lead,
            dims,
            pad_to: None,
            across: 1,
            repeat: 1,
            optional: false,
        }
    }

    /// Repeated `n` times down the rectangle's rows.
    pub fn repeat(self, n: u32) -> Slice {
        Slice { repeat: n, ..self }
    }

    /// Each PE's words zero-padded to `n`.
    pub fn pad_to(self, n: u32) -> Slice {
        Slice {
            pad_to: Some(n),
            ..self
        }
    }

    /// No copy when the file is not staged.
    pub fn optional(self) -> Slice {
        Slice {
            optional: true,
            ..self
        }
    }

    /// The words of the slice of `a` (see [`Slice`]).
    pub fn words(&self, a: &Npy) -> Result<Vec<u32>, String> {
        let mut shape: &[usize] = &a.shape;
        let mut base = 0usize;
        if let Some(l) = self.lead
            && shape.len() > self.dims.len() + 1
        {
            base = l as usize * a.stride0();
            shape = &shape[1..];
        }
        let k = self.dims.len();
        if shape.len() < k {
            return Err(format!("{}: shape {:?}, {k} ranges", self.name, a.shape));
        }
        for (i, &(lo, hi)) in self.dims.iter().enumerate() {
            if lo > hi || hi as usize > shape[i] {
                return Err(format!(
                    "{}: range {lo}..{hi} of axis {i} (shape {:?})",
                    self.name, a.shape
                ));
            }
        }
        // stride[i]: elements an index of axis i
        let mut stride = vec![1usize; shape.len() + 1];
        for i in (0..shape.len()).rev() {
            stride[i] = stride[i + 1] * shape[i];
        }
        let per: usize = shape[k..].iter().product::<usize>().max(1);
        let mut flat = Vec::new();
        if k == 0 {
            a.words(base, stride[0], &mut flat)?;
        } else {
            // every index of the outer ranges (the last outer axis fastest); the last range is
            // one contiguous read
            let outer = &self.dims[..k - 1];
            let (l0, l1) = self.dims[k - 1];
            let count: usize = outer.iter().map(|r| (r.1 - r.0) as usize).product();
            for n in 0..count {
                let mut rem = n;
                let mut at = base + l0 as usize * stride[k];
                for (i, r) in outer.iter().enumerate().rev() {
                    let len = (r.1 - r.0) as usize;
                    at += (r.0 as usize + rem % len) * stride[i + 1];
                    rem /= len;
                }
                a.words(at, (l1 - l0) as usize * stride[k], &mut flat)?;
            }
        }
        let to = self.pad_to.map_or(per, |p| p as usize);
        if to < per {
            return Err(format!("{}: {per} words a PE, padded to {to}", self.name));
        }
        let across = self.across.max(1) as usize;
        let mut row = Vec::with_capacity(flat.len() / per * to * across);
        for pe in flat.chunks(per) {
            for _ in 0..across {
                row.extend_from_slice(pe);
                row.resize(row.len() + to - pe.len(), 0);
            }
        }
        let mut v = Vec::with_capacity(row.len() * self.repeat.max(1) as usize);
        for _ in 0..self.repeat.max(1) {
            v.extend_from_slice(&row);
        }
        Ok(v)
    }
}

/// The transforms a prep applies to a checkpoint tensor before it is staged (declared here so a
/// binding states them; the preps implement them today).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Transform {
    /// Zero-pad a dim to `to` (Qwen3-8B's hidden 4,096 -> 4,104).
    PadZero {
        dim: u32,
        to: u32,
    },
    /// Pad the head's vocabulary rows with copies of the last real row (never zeros: a padded id
    /// must not win the argmax).
    PadLastRow {
        to: u32,
    },
    /// gpt-oss's MXFP4 experts re-encoded losslessly as E5M2.
    Mxfp4ToE5m2,
    Bf16ToF16,
}

/// One copy onto the wafer.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Binding {
    /// The PE symbol it lands in (`w`, `rcfg`, `hw`, `cw`, ...).
    pub sym: &'static str,
    pub rect: Rect,
    pub source: Source,
    /// 16-bit words (`put16`).
    pub b16: bool,
    /// For staged sources: the checkpoint tensor(s) and transforms that made it (documentation
    /// today; the prep's input later).
    pub from: Vec<(String, Vec<Transform>)>,
}

/// The checkpoint tensors a binding's words come from, and how ([`Binding::from`]).
pub type From = Vec<(String, Vec<Transform>)>;

/// A family's bindings in placement order.
#[derive(Default)]
pub struct Copies(pub Vec<Binding>);

impl Copies {
    /// Every PE's rcfg words over `rect`.
    pub fn rcfg(&mut self, rect: Rect) {
        self.push("rcfg", rect, Source::Rcfg, false, Vec::new());
    }

    /// `slice` onto `rect`'s `sym`, 16-bit words when `b16`.
    pub fn slice(&mut self, sym: &'static str, rect: Rect, slice: Slice, b16: bool, from: From) {
        self.push(sym, rect, Source::Slice(slice), b16, from);
    }

    pub fn push(&mut self, sym: &'static str, rect: Rect, source: Source, b16: bool, from: From) {
        if !rect.is_empty() {
            self.0.push(Binding {
                sym,
                rect,
                source,
                b16,
                from,
            });
        }
    }
}

/// `[r0, r1)` in runs of at most `step`: the row blocks a copy goes in.
pub fn chunks(r0: u32, r1: u32, step: u32) -> impl Iterator<Item = (u32, u32)> {
    (r0..r1)
        .step_by(step as usize)
        .map(move |a| (a, (a + step).min(r1)))
}

/// One checkpoint tensor, untransformed.
pub fn from(t: &str) -> From {
    vec![(t.to_string(), Vec::new())]
}
