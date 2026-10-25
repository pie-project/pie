//! The kernel form: a region scheduled into passes over the row.
//!
//! A kernel runs in phases. Each phase computes its uniforms (values one a
//! row, from constants and reductions' totals) and then, where it has any
//! row work, makes one pass over the row: one loop over the elements a
//! thread holds, computing that phase's row values, storing results, and
//! accumulating each reduction's partials, which are combined over the row
//! after the loop. A row value a later phase reads is kept in registers
//! across the passes; one only its own pass reads lives in that pass alone.
//!
//! A reduction takes its elements two at a time, in order: an even element
//! waits, and its odd neighbour joins it into the partial (`a = a ∘ (x₀ ∘
//! x₁)`), as the hand row kernels take a bf16 pair. Its total is the frame's
//! combine of the partials over the row's threads ([`crate::frame`]).

use dtype::Dtype;

use crate::region::{Binary, Elem, Literal, Reduce, Unary, ValueId};

/// How a kernel reads a value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand {
    /// Element `e` of row parameter `k` (the frame's load).
    Param(u32),
    /// Scalar parameter `k`, one value a row.
    One(u32),
    /// A row value kept in registers across passes, at element `e`.
    Kept(ValueId),
    /// A value of the current pass, or a uniform.
    Local(ValueId),
}

/// What a kernel computes of its operands.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Expr {
    Literal(Literal),
    /// The operand as it is (a scalar across the row).
    Copy(Operand),
    Unary(Unary, Operand),
    Binary(Binary, Operand, Operand),
    Select(Operand, Operand, Operand),
    Cast {
        value: Operand,
        from: Elem,
        to: Elem,
    },
}

/// A value one a row, computed before its phase's pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniform {
    pub value: ValueId,
    pub elem: Elem,
    pub expr: Expr,
    /// Rounded where it is defined.
    pub round: Option<Dtype>,
}

/// A row value computed in a pass.
#[derive(Clone, Debug, PartialEq)]
pub struct RowAssign {
    pub value: ValueId,
    pub elem: Elem,
    pub expr: Expr,
    /// Rounded where it is defined (a value that does not leave; a result
    /// rounds by its store).
    pub round: Option<Dtype>,
    /// The result it is stored as first: the stored value is what later
    /// reads see.
    pub store: Option<u32>,
    /// The other results it is stored as.
    pub also: Vec<u32>,
    /// Kept for a later pass.
    pub kept: bool,
    /// Read by nothing (a result alone).
    pub unread: bool,
}

/// A reduction's partials, accumulated in a pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Partial {
    pub value: ValueId,
    pub op: Reduce,
    pub input: Operand,
}

/// A reduction combined over the row after its pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Total {
    pub value: ValueId,
    pub op: Reduce,
    pub round: Option<Dtype>,
}

/// One pass over the row.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Loop {
    pub partials: Vec<Partial>,
    pub rows: Vec<RowAssign>,
    /// Results that are parameters, copied through: `(result, param)`.
    pub copies: Vec<(u32, u32)>,
    /// Results complete after the pass, flushed to memory.
    pub flushes: Vec<u32>,
    pub totals: Vec<Total>,
}

/// One phase: its uniforms, then its pass where it has row work.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Phase {
    pub uniforms: Vec<Uniform>,
    pub pass: Option<Loop>,
}

/// A region scheduled.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    /// The region's parameters and results.
    pub params: u32,
    pub results: u32,
    /// The row values kept across passes, in value order.
    pub kept: Vec<(ValueId, Elem)>,
    pub phases: Vec<Phase>,
}

impl Kernel {
    /// The passes it makes over the row.
    #[must_use]
    pub fn passes(&self) -> u32 {
        u32::try_from(self.phases.iter().filter(|p| p.pass.is_some()).count()).unwrap_or(u32::MAX)
    }
}
