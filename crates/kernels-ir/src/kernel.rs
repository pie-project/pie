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
//! A reduction accumulates each thread's partial over its elements, and its
//! total is the skeleton's combine of the partials over the row's threads
//! ([`crate::skeleton`]), in whatever order the skeleton's schedule takes: the
//! order is the kernel's to choose, the sum is the region's. Over heads, a
//! partial is a vector of 8's (each lies in one head), combined over the
//! head's vectors.
//!
//! A value one a head is held a vector of 8 at a time (`v[e / 8]`),
//! computed between passes. A gather within the row reads the row back
//! where its pass shared it, one phase on. What a column alone computes
//! dearly (a rotation's frequencies) is tabled once a block, before the
//! first pass, and read back by column.

use dtype::Dtype;

use crate::region::{Binary, Elem, Literal, Reduce, Unary, ValueId};

/// How a kernel reads a value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Operand {
    /// Element `e` of row parameter `k` (the skeleton's load).
    Param(u32),
    /// Scalar parameter `k`, one value a row.
    One(u32),
    /// A row value kept in registers across passes, at element `e`.
    Kept(ValueId),
    /// A value of the current pass, or a uniform.
    Local(ValueId),
    /// A value one a head, at element `e`'s vector (`p` between passes).
    Heads(ValueId),
    /// A tabled value at element `e`'s column: the table's entry `at +
    /// column % period`.
    Table { at: u32, period: u32 },
    /// A value of a table's entry, computed where the table is.
    Entry(ValueId),
    /// Small-vector parameter `k`, its values one a row (an element `i` of
    /// it where an expression picks one).
    Vector(u32),
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
    /// `a * b + c`, rounded once.
    Fma(Operand, Operand, Operand),
    /// Element `e`'s column (a table's entry's, in a table).
    Column,
    /// The row a pass shared, at the column the operand names.
    Peer(Operand),
    /// Element `i` of a small vector.
    Pick(Operand, u32),
    /// A small vector with element `i` the scalar.
    Insert(Operand, u32, Operand),
    /// A small vector, each value the scalar.
    Splat(Operand),
    /// A repeat's carried value, opened from its initial value: the
    /// uniforms after it, up to its [`Expr::Repeat`], run as its body.
    Carry(Operand),
    /// The repeat that `carry` opened closed: its body run `count` times,
    /// each pass's `next` the next pass's carried value.
    Repeat {
        count: u32,
        carry: ValueId,
        next: Operand,
    },
}

/// A value one a head, computed before its phase's pass a vector at a time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadAssign {
    pub value: ValueId,
    pub elem: Elem,
    pub expr: Expr,
    pub round: Option<Dtype>,
}

/// A value tabled once a block: `period` entries from `at` of the table,
/// each computed from its column by `entries` (the values it is computed
/// from, in order, the last the tabled value).
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    pub value: ValueId,
    pub at: u32,
    pub period: u32,
    pub entries: Vec<(ValueId, Elem, Expr)>,
}

/// A value one a row, computed before its phase's pass: a scalar, or a
/// small vector of `vector` values.
#[derive(Clone, Debug, PartialEq)]
pub struct Uniform {
    pub value: ValueId,
    pub elem: Elem,
    pub expr: Expr,
    /// Rounded where it is defined.
    pub round: Option<Dtype>,
    /// The values a small vector holds; `None` for a scalar.
    pub vector: Option<u32>,
    /// The results it is stored as (a small vector, its values one a row).
    pub stores: Vec<u32>,
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
    /// Over each head of this many vectors of 8, not the row.
    pub seg: Option<u32>,
}

/// A reduction combined over the row after its pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Total {
    pub value: ValueId,
    pub op: Reduce,
    pub round: Option<Dtype>,
    pub seg: Option<u32>,
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
    /// The rows the pass shares, for a gather a later pass makes.
    pub shares: Vec<Operand>,
    /// The pass overwrites a row an earlier pass shared, read back before
    /// it: every thread of the row past those reads first.
    pub sync_first: bool,
}

/// One phase: its uniforms and values a head, then its pass where it has
/// row work.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Phase {
    pub uniforms: Vec<Uniform>,
    pub heads: Vec<HeadAssign>,
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
    /// The values one a head, in value order.
    pub heads: Vec<(ValueId, Elem)>,
    /// What the block tables before its first pass, and the table's floats.
    pub tables: Vec<Table>,
    pub table: u32,
    /// Whether a pass shares its row (a gather).
    pub shares: bool,
    pub phases: Vec<Phase>,
}

impl Kernel {
    /// The passes it makes over the row.
    #[must_use]
    pub fn passes(&self) -> u32 {
        u32::try_from(self.phases.iter().filter(|p| p.pass.is_some()).count()).unwrap_or(u32::MAX)
    }
}
