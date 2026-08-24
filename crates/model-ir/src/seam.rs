//! THE SEAMS A TEXT MAY STAND AT, and the one column of each that anything
//! reads: its name.
//!
//! # What a `Def` used to carry, and why four fifths of it went
//!
//! `sees`, `caps`, `position` and `sink` stood beside `name`, plus the `Cap`
//! enum, a `Position` struct, an `ALL` table and a `by_name` lookup. Not one
//! of them was ever read: `model_dsl::seam::at` records `def.name`, and the
//! four executors find the logits by comparing a recorded seam against
//! [`OUT`]`.name`. R5 measured it and the columns went.
//!
//! ONE OF THEM STATED A REAL LAW and could not keep stating it, which is why
//! it is written out here rather than dropped. `ATTN_QV`'s `position` said
//! the transform seam sits AFTER the packed row is cut (`gemm.matmul`,
//! `layout.split_qkv`) and BEFORE anything consumes the cut (`norm.rmsnorm`,
//! `rope`, `attention.kv_append`), and `seam::check_plan` walked a
//! `ForwardPlan` checking it. That walk went with `ForwardPlan` at R4e, and
//! the form that replaced it cannot express the check: a legacy
//! `SeamStatement` carried the op index it stood after, and
//! [`crate::plan::Seam`] carries the values it sees and the layer it stands
//! at — no op index — so "after this statement, before that one" has nothing
//! to compare. The law still holds of every text that stands at `attn.qv`;
//! it is a rule an author keeps, not one a walk can check, until a plan
//! carries the order again.
//!
//! # The names are still six and are still closed
//!
//! A seam is a place a HOST may stand — observe the row, transform it, sink
//! the page mask, sample, emit — and a text that invented one would be
//! offering a door no host knows to knock on. So the set is spelled here
//! rather than left to the string a text passes.

/// One seam: a place a text stands and a host may reach.
pub struct Def {
    pub name: &'static str,
}

/// After the q projection, before attention: where a mask sink and an
/// observer of the query rows stand.
pub const ATTN_Q: Def = Def { name: "attn.q" };

/// After attention, before the output projection: the attended rows, and
/// where a scores reader stands.
pub const ATTN_OUT: Def = Def { name: "attn.out" };

/// After the packed qkv row is cut and before anything consumes the cut —
/// the ordering law this module's header states.
pub const ATTN_QV: Def = Def { name: "attn.qv" };

/// The mixed row of a recurrent layer.
pub const RECURRENT: Def = Def { name: "recurrent" };

/// The whole fire's entry: where a host puts rows in and emits.
pub const IN: Def = Def { name: "in" };

/// The whole fire's exit: the logits, and where sampling stands.
pub const OUT: Def = Def { name: "out" };
