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
                let d = if (rotary / 2) > 0 && turn_span(&turn.pairs) % (rotary / 2) == 0 {
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
        let mut freq = self.bin(Binary::Pow, theta, exponent);
        if let Some(ramp) = turn.ramp {
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
        let cos = self.push(Op::Unary(Unary::Cos, angle));
        let sin = self.push(Op::Unary(Unary::Sin, angle));
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
