//! The shared model-state geometry: where a lane's kv rows land, what the
//! arena's rectangles are, and what a load-time reading of the IR may
//! refuse. Backend-neutral — no device, buffer, pool, or dtype pinning here;
//! that stays with the shell.
//!
//! # The three questions the page arithmetic answers
//!
//! A paged kv space answers three, and confusing them is how a cache silently
//! reads somebody else's tokens:
//!
//! ```text
//! which pages does lane L own?     the PAGING — static, per slot, per load
//! how much of them is live?        kv_len / last_page_len — per fire, per lane
//! where does token T land?         write_page / write_offset — per fire, per row
//! ```
//!
//! The third is stated per token rather than derived from a position, since a
//! derivation cannot spell a fresh-page write that is not the page run's
//! tail; both backends' writers take `write_page`/`write_offset` as declared
//! buffers and derive nothing.
//!
//! # What is NOT here
//!
//! The IR probe that reads a plan's cache facts stayed in the shells: CUDA
//! and Metal genuinely disagree about where a schedule's reading has an
//! author, which is a behavior difference, not a spelling one. What the two
//! agree on — [`SpaceFacts`](kv::SpaceFacts), [`reads`](kv::reads),
//! [`row_of`](kv::row_of), [`space_of`](kv::space_of),
//! [`width_of`](kv::width_of) — is here, and both probes are written over it.

use std::fmt;

pub mod arena;
pub mod check;
pub mod kv;

pub use kv::{Geometry, Paging, Reader, Seat, SpaceFacts, geometry, geometry_with, indptr};

/// What the neutral store refuses: the three ways the arithmetic can be
/// wrong, each already a variant both shells carry under the same name and
/// fields. A shell converts on the way out and keeps its own wording for the
/// reader; the condition itself is stated once, here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fault {
    /// A count past a ceiling somebody reserved bytes for.
    Ceiling {
        /// What overflowed.
        what: &'static str,
        /// What was asked.
        need: u64,
        /// What was reserved.
        have: u64,
    },

    /// The plan names something no seat was bound for.
    Unbound {
        /// The seat, named as the IR names it.
        what: String,
    },

    /// A schedule value built over one class mask and read under another.
    Straddled {
        /// The plan value holding the schedule.
        value: u32,
        /// The node that reads it.
        node: u32,
        /// The classes the schedule was planned for.
        planned: String,
        /// The classes that consume it.
        consumed: String,
    },
}

/// The store's own result.
pub type Result<T> = std::result::Result<T, Fault>;

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ceiling { what, need, have } => {
                write!(f, "this fire wants {need} {what} and {have} was reserved")
            }
            Self::Unbound { what } => write!(f, "this plan names {what}, which nothing binds"),
            Self::Straddled {
                value,
                node,
                planned,
                consumed,
            } => write!(
                f,
                "the schedule in value {value} is planned over {planned} and read by node \
                 {node} over {consumed}"
            ),
        }
    }
}

impl std::error::Error for Fault {}
