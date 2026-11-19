//! A region's ops, built a row's math at a time: what poem-compiler lowers
//! a chain's ops to, and what a kernel's test builds the same region with.
//! Each builder states its math in the order the hand kernels compute it
//! (the JIT compiles with `fmad=false`, so a product and a sum stay two
//! roundings), so a region that fuses them lands their bits.

use crate::region::{Binary, Elem, Literal, Op, Reduce, Unary, ValueId};

/// A region's ops under construction: values `0..params` are its
/// parameters, each op's result the next value.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ops {
    pub params: u32,
    pub ops: Vec<Op>,
}

/// Which of each head's columns a rotation turns, and with which mate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pairs {
    /// The first `rotary / 2` columns, column `i` with `i + span`, at
    /// `theta^(-i / span)` (a partial rotary, or Gemma 4's proportional
    /// one, `span` half the head).
    Halves { rotary: u32, span: u32 },
    /// The last `rotary` columns, pair `d` at `theta^(-2d / rotary)`: the
    /// columns `2d`, `2d + 1` of the tail where `interleaved`, else `d` and
    /// `d + rotary / 2`.
    Tail { rotary: u32, interleaved: bool },
}

/// YaRN's ramp over the pairs: pair `d`'s frequency scaled by `(1 - r) +
/// r / factor`, `r` the pair's place between `low` and `high`, clamped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ramp {
    pub factor: f32,
    pub low: f32,
    pub high: f32,
}

/// A rotary turn of each `head_dim`-wide head of a row at the row's
/// position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Turn {
    pub head_dim: u32,
    pub pairs: Pairs,
    pub theta: f32,
    /// The angle negated: the turn undone.
    pub inverse: bool,
    pub ramp: Option<Ramp>,
    /// A scale both the cos and the sin carry (YaRN's attention factor),
    /// each times it in f32 before the pair turns.
    pub mscale: Option<f32>,
}

impl Ops {
    #[must_use]
    pub fn new(params: u32) -> Self {
        Self {
            params,
            ops: Vec::new(),
        }
    }

    /// `op`, its value the next.
    pub fn push(&mut self, op: Op) -> ValueId {
        let id = self.params + u32::try_from(self.ops.len()).unwrap_or(u32::MAX);
        self.ops.push(op);
        id
    }

    pub fn f32(&mut self, x: f32) -> ValueId {
        self.push(Op::Const(Literal::F32(x)))
    }

    pub fn u32(&mut self, n: u32) -> ValueId {
        self.push(Op::Const(Literal::U32(n)))
    }

    fn bin(&mut self, op: Binary, a: ValueId, b: ValueId) -> ValueId {
        self.push(Op::Binary(op, a, b))
    }

    /// `x · rsqrt(mean(x²) + eps)` over a row of `width`, as the hand
    /// kernels compute it.
    #[allow(clippy::cast_precision_loss)]
    pub fn rms_normed(&mut self, x: ValueId, width: u32, eps: f32) -> ValueId {
        let sq = self.bin(Binary::Mul, x, x);
        let sum = self.push(Op::Reduce(Reduce::Sum, sq));
        let n = self.f32(width as f32);
        let ms = self.bin(Binary::Div, sum, n);
        let eps = self.f32(eps);
        let shifted = self.bin(Binary::Add, ms, eps);
        let r = self.push(Op::Unary(Unary::Rsqrt, shifted));
        self.bin(Binary::Mul, x, r)
    }

    /// The RMS norm of each `head_dim`-wide head of `x`: each head's `x ·
    /// rsqrt(mean(x²) + eps)` as the hand kernels compute it.
    #[allow(clippy::cast_precision_loss)]
    pub fn rms_normed_heads(&mut self, x: ValueId, head_dim: u32, eps: f32) -> ValueId {
        let sq = self.bin(Binary::Mul, x, x);
        let sum = self.push(Op::ReduceHeads(Reduce::Sum, sq, head_dim));
        let n = self.f32(head_dim as f32);
        let ms = self.bin(Binary::Div, sum, n);
        let eps = self.f32(eps);
        let shifted = self.bin(Binary::Add, ms, eps);
        let r = self.push(Op::Unary(Unary::Rsqrt, shifted));
        let r = self.push(Op::Broadcast(r));
        self.bin(Binary::Mul, x, r)
    }

    /// `x`, its `head_dim`-wide heads turned over position axes (a diffusion
    /// transformer's rotation): pair `(2w, 2w + 1)` of an axis of `dim` turns at
    /// `theta^(-2w / dim)` (`powf` in f32) times the axis's scalar position `pos`,
    /// each output one fused multiply-add, as the hand rotation lands it.
    #[allow(clippy::cast_precision_loss)]
    pub fn rotate_axes(
        &mut self,
        x: ValueId,
        head_dim: u32,
        axes: &[(u32, f32, ValueId)],
    ) -> ValueId {
        let col = self.push(Op::Column);
        let head = self.u32(head_dim);
        let i = self.bin(Binary::Rem, col, head);
        let two = self.u32(2);
        let odd = self.bin(Binary::Rem, i, two);
        let zero = self.u32(0);
        let first = self.bin(Binary::Eq, odd, zero);
        // each axis's angle, the column's pair counted from the axis's first column
        let mut start = 0u32;
        let mut angles = Vec::new();
        for &(dim, theta, pos) in axes {
            let from = self.u32(start);
            let off = self.bin(Binary::Sub, i, from);
            let pair2 = self.bin(Binary::Sub, off, odd);
            let e = self.push(Op::Cast(pair2, Elem::F32));
            let d = self.f32(dim as f32);
            let e = self.bin(Binary::Div, e, d);
            let e = self.push(Op::Unary(Unary::Neg, e));
            let t = self.f32(theta);
            let freq = self.bin(Binary::Pow, t, e);
            let angle = self.bin(Binary::Mul, pos, freq);
            start += dim;
            angles.push((start, angle));
        }
        // the angle of the axis the column falls in: the first whose end is past it
        let mut angle = angles.last().map_or(zero, |&(_, a)| a);
        for &(end, a) in angles.iter().rev().skip(1) {
            let end = self.u32(end);
            let within = self.bin(Binary::Lt, i, end);
            angle = self.push(Op::Select(within, a, angle));
        }
        let cos = self.push(Op::Unary(Unary::Cos, angle));
        let sin = self.push(Op::Unary(Unary::Sin, angle));
        let one = self.u32(1);
        let up = self.bin(Binary::Add, col, one);
        let down = self.bin(Binary::Sub, col, one);
        let mate = self.push(Op::Select(first, up, down));
        let b = self.push(Op::Peer(x, mate));
        let bs = self.bin(Binary::Mul, b, sin);
        let nbs = self.push(Op::Unary(Unary::Neg, bs));
        let add = self.push(Op::Select(first, nbs, bs));
        self.push(Op::Fma(x, cos, add))
    }

    /// [`Ops::rotate_axes`] with the pairs a head's halves make: `split`, each axis's `dim`
    /// columns its own halves; else NeoX's, the axes' angles in turn over the rotated half.
    /// The columns past `rotary` are `x`.
    #[allow(clippy::cast_precision_loss)]
    pub fn rotate_axes_halves(
        &mut self,
        x: ValueId,
        head_dim: u32,
        axes: &[(u32, f32, ValueId)],
        split: bool,
        rotary: u32,
    ) -> ValueId {
        let col = self.push(Op::Column);
        let head = self.u32(head_dim);
        let i = self.bin(Binary::Rem, col, head);
        let half = self.u32(rotary / 2);
        // NeoX: the column's angle index and whether it is the pair's first
        let neox_first = self.bin(Binary::Lt, i, half);
        let neox_j = {
            let back = self.bin(Binary::Sub, i, half);
            self.push(Op::Select(neox_first, i, back))
        };
        // each axis's (end, first, w, hop) where the column falls in it
        let mut runs: Vec<(u32, ValueId, ValueId, u32)> = Vec::new();
        let mut start = 0u32;
        for &(dim, _, _) in axes {
            let h = dim / 2;
            if split {
                let from = self.u32(start);
                let off = self.bin(Binary::Sub, i, from);
                let hh = self.u32(h);
                let first = self.bin(Binary::Lt, off, hh);
                let back = self.bin(Binary::Sub, off, hh);
                let w = self.push(Op::Select(first, off, back));
                runs.push((start + dim, first, w, h));
                start += dim;
            } else {
                let from = self.u32(start);
                let w = self.bin(Binary::Sub, neox_j, from);
                runs.push((start + h, neox_first, w, rotary / 2));
                start += h;
            }
        }
        // the key a column's axis is picked by: its channel (split) or its angle index
        let key = if split { i } else { neox_j };
        let pick = |me: &mut Self, of: &dyn Fn(usize) -> ValueId| {
            let mut v = of(runs.len() - 1);
            for k in (0..runs.len() - 1).rev() {
                let end = me.u32(runs[k].0);
                let within = me.bin(Binary::Lt, key, end);
                v = me.push(Op::Select(within, of(k), v));
            }
            v
        };
        let mut angles = Vec::new();
        for (k, &(dim, theta, pos)) in axes.iter().enumerate() {
            let e = self.push(Op::Cast(runs[k].2, Elem::F32));
            let two = self.f32(2.0 / dim as f32);
            let e = self.bin(Binary::Mul, e, two);
            let e = self.push(Op::Unary(Unary::Neg, e));
            let t = self.f32(theta);
            let freq = self.bin(Binary::Pow, t, e);
            angles.push(self.bin(Binary::Mul, pos, freq));
        }
        let angle = pick(self, &|k| angles[k]);
        let firsts: Vec<ValueId> = runs.iter().map(|r| r.1).collect();
        let first = pick(self, &|k| firsts[k]);
        let hops: Vec<ValueId> = runs.iter().map(|r| r.3).map(|h| self.u32(h)).collect();
        let hop = pick(self, &|k| hops[k]);
        let cos = self.push(Op::Unary(Unary::Cos, angle));
        let sin = self.push(Op::Unary(Unary::Sin, angle));
        let up = self.bin(Binary::Add, col, hop);
        let down = self.bin(Binary::Sub, col, hop);
        let mut mate = self.push(Op::Select(first, up, down));
        // past the rotated columns a column is its own mate (and `x` as it is)
        let within = (rotary < head_dim).then(|| {
            let r = self.u32(rotary);
            self.bin(Binary::Lt, i, r)
        });
        if let Some(w) = within {
            mate = self.push(Op::Select(w, mate, col));
        }
        let b = self.push(Op::Peer(x, mate));
        let bs = self.bin(Binary::Mul, b, sin);
        let nbs = self.push(Op::Unary(Unary::Neg, bs));
        let add = self.push(Op::Select(first, nbs, bs));
        let turned = self.push(Op::Fma(x, cos, add));
        match within {
            Some(w) => self.push(Op::Select(w, turned, x)),
            None => turned,
        }
    }

    /// `x`, a row of `width`, its heads turned at `pos` (a scalar, the row's
    /// position) as the hand rotations turn them: each turned column's
    /// pair index `d`, its frequency `theta^e` (`powf`, the exponent as the
    /// kernels write it), YaRN's ramp on it, the angle `pos · freq` (its
    /// sign flipped where `inverse`), and the column `a` with its mate `b`
    /// landing `a cos - b sin` where it is the pair's first and `a cos +
    /// b sin` where it is the second. The columns no pair turns are `x` as
    /// it is.
    #[allow(clippy::cast_precision_loss)]
    pub fn rotate(&mut self, x: ValueId, pos: ValueId, turn: &Turn) -> ValueId {
        self.rotate_with(x, pos, turn, None)
    }

    /// [`Ops::rotate`] at frequencies stated outright (a checkpoint's rope
    /// scaling, `table`: pair `d` turning at `table[d]`), where given, in
    /// place of `theta`'s and the ramp's: each column's frequency its pair's
    /// entry, a function of the column alone the schedule tables.
    pub fn rotate_with(
        &mut self,
        x: ValueId,
        pos: ValueId,
        turn: &Turn,
        table: Option<&[f32]>,
    ) -> ValueId {
        let col = self.push(Op::Column);
        let head = self.u32(turn.head_dim);
        let i = self.bin(Binary::Rem, col, head);
        // Each column: whether a pair turns it, whether it is the pair's
        // first, its pair index, and how far its mate sits.
        let (turned, first, d, hop) = match turn.pairs {
            Pairs::Halves { rotary, span } => {
                let (half, span) = (self.u32(rotary / 2), self.u32(span));
                let first = self.bin(Binary::Lt, i, half);
                let from = self.bin(Binary::Ge, i, span);
                let upto = self.bin(Binary::Add, span, half);
                let below = self.bin(Binary::Lt, i, upto);
                let second = self.bin(Binary::And, from, below);
                let turned = self.bin(Binary::Or, first, second);
                // a pair's index, `i` or `i - span` (its column within the
                // span), the same every `span` columns; where the pairs'
                // count divides the span, the same every that many (a
                // column no pair turns takes some index, never read), so
                // a table of the frequencies holds the pairs alone
                let d = self.bin(Binary::Rem, i, span);
                let d = if (rotary / 2) > 0 && turn_span(&turn.pairs).is_multiple_of(rotary / 2) {
                    self.bin(Binary::Rem, d, half)
                } else {
                    d
                };
                (turned, first, d, span)
            }
            Pairs::Tail {
                rotary,
                interleaved,
            } => {
                let offset = self.u32(turn.head_dim - rotary);
                let turned = self.bin(Binary::Ge, i, offset);
                let j = self.bin(Binary::Sub, i, offset);
                if interleaved {
                    let two = self.u32(2);
                    let zero = self.u32(0);
                    let odd = self.bin(Binary::Rem, j, two);
                    let first = self.bin(Binary::Eq, odd, zero);
                    let d = self.bin(Binary::Div, j, two);
                    let one = self.u32(1);
                    (turned, first, d, one)
                } else {
                    let half = self.u32(rotary / 2);
                    let first = self.bin(Binary::Lt, j, half);
                    let back = self.bin(Binary::Sub, j, half);
                    let d = self.push(Op::Select(first, j, back));
                    (turned, first, d, half)
                }
            }
        };
        let up = self.bin(Binary::Add, col, hop);
        let down = self.bin(Binary::Sub, col, hop);
        let mate = self.push(Op::Select(first, up, down));
        let mate = self.push(Op::Select(turned, mate, col));
        let pair = d;
        let d = self.push(Op::Cast(d, Elem::F32));
        // the exponent as the kernels write it: `-d / span` or
        // `-2 * d / rotary`
        let exponent = match turn.pairs {
            Pairs::Halves { span, .. } => {
                let negated = self.push(Op::Unary(Unary::Neg, d));
                let span = self.f32(span as f32);
                self.bin(Binary::Div, negated, span)
            }
            Pairs::Tail { rotary, .. } => {
                let two = self.f32(-2.0);
                let twice = self.bin(Binary::Mul, two, d);
                let rotary = self.f32(rotary as f32);
                self.bin(Binary::Div, twice, rotary)
            }
        };
        let theta = self.f32(turn.theta);
        let mut freq = match table {
            // each pair's entry: the first, then each later one where the
            // pair is its
            Some(table) => {
                let mut freq = self.f32(table.first().copied().unwrap_or(0.0));
                for (at, &entry) in (1u32..).zip(table.iter().skip(1)) {
                    let at = self.u32(at);
                    let here = self.bin(Binary::Eq, pair, at);
                    let entry = self.f32(entry);
                    freq = self.push(Op::Select(here, entry, freq));
                }
                freq
            }
            None => self.bin(Binary::Pow, theta, exponent),
        };
        if let Some(ramp) = turn.ramp.filter(|_| table.is_none()) {
            // YaRN's ramp as the kernels' `yarn_original_freq` takes it
            let denom = if ramp.high == ramp.low {
                (ramp.high + 1e-3) - ramp.low
            } else {
                ramp.high - ramp.low
            };
            let low = self.f32(ramp.low);
            let from = self.bin(Binary::Sub, d, low);
            let denom = self.f32(denom);
            let r = self.bin(Binary::Div, from, denom);
            let (zero, one) = (self.f32(0.0), self.f32(1.0));
            let r = self.bin(Binary::Max, r, zero);
            let r = self.bin(Binary::Min, r, one);
            let keep = self.bin(Binary::Sub, one, r);
            let factor = self.f32(ramp.factor);
            let scaled = self.bin(Binary::Div, r, factor);
            let blend = self.bin(Binary::Add, keep, scaled);
            freq = self.bin(Binary::Mul, freq, blend);
        }
        let pos = if turn.inverse {
            self.push(Op::Unary(Unary::Neg, pos))
        } else {
            pos
        };
        let angle = self.bin(Binary::Mul, pos, freq);
        let mut cos = self.push(Op::Unary(Unary::Cos, angle));
        let mut sin = self.push(Op::Unary(Unary::Sin, angle));
        if let Some(m) = turn.mscale {
            let m = self.f32(m);
            cos = self.bin(Binary::Mul, cos, m);
            sin = self.bin(Binary::Mul, sin, m);
        }
        let b = self.push(Op::Peer(x, mate));
        let ac = self.bin(Binary::Mul, x, cos);
        let bs = self.bin(Binary::Mul, b, sin);
        let lead = self.bin(Binary::Sub, ac, bs);
        let trail = self.bin(Binary::Add, ac, bs);
        let out = self.push(Op::Select(first, lead, trail));
        self.push(Op::Select(turned, out, x))
    }
}

/// The span a split turn's pairs sit apart, or 0.
const fn turn_span(pairs: &Pairs) -> u32 {
    match *pairs {
        Pairs::Halves { span, .. } => span,
        Pairs::Tail { .. } => 0,
    }
}
