//! The shared model-state geometry: where a lane's kv rows land, what the
//! arena's rectangles are, and what a load-time reading of the IR may refuse.
//!
//! **NOT ONE LINE OF THIS NAMES A DEVICE**, and that is the whole reason the
//! module exists. Design §3 gives `engine::store` the page arithmetic and
//! leaves only the BYTES to a shell — buffers, pools, dtype pinning, device
//! pointers — so what lands here is exactly what two shells were computing
//! twice: `Paging`/`Seat`/`Geometry`/`indptr` (the 27 self-accusing
//! `engine::store candidate` markers of survey §2 debt 3), the arena carve's
//! `slot × bucket → rectangle`, and the bake-time schedule/window check.
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
//! The first is a deployment's budget. The second is what the attention
//! schedule is planned against. The third is what the append kernel writes
//! through, and it is stated per token rather than derived from a position
//! because a derivation cannot spell a fresh-page write that is not the page
//! run's tail. Both backends' writers agree on that and neither could do
//! otherwise: CUDA's explicit-descriptor writer (`kernels/attn/kv.cuh`) and
//! Metal's `attn/kv_write.metal` `kv_append_paged_*` entries both take
//! `write_page` and `write_offset` as declared buffers and derive nothing,
//! and the ops themselves name them (`attention.kv_append`), so a shell that
//! wanted a derivation would have nowhere to put it.
//!
//! # What is NOT here
//!
//! The IR probe that reads a plan's cache facts (`probe`, `Facts`) stayed in
//! the shells: the two copies genuinely disagree about where a schedule's
//! reading has an author — CUDA reads it off the plan op that CARVES the
//! schedule (three passes, `ScheduleFacts`, the latent arms), Metal infers it
//! from the launches that consume it (two passes). That is a behaviour
//! difference, not a spelling one, and picking either would change a shell.
//! What the two do agree on — [`SpaceFacts`](kv::SpaceFacts), the launch
//! flattening [`reads`](kv::reads), [`row_of`](kv::row_of),
//! [`space_of`](kv::space_of), [`width_of`](kv::width_of) — is here, and both
//! probes are written over it.

use std::fmt;

pub mod arena;
pub mod check;
pub mod kv;

pub use kv::{Geometry, Paging, Reader, Seat, SpaceFacts, geometry, geometry_with, indptr};

/// What the neutral store refuses.
///
/// **THREE VARIANTS BECAUSE THE ARITHMETIC HAS THREE WAYS TO BE WRONG**, and
/// each one is a variant both shells already carry under the same name and the
/// same fields — `Ceiling`, `Unbound`, `Straddled`. A shell converts on the
/// way out (`impl From<store::Fault> for crate::error::Fault`) and keeps its
/// own sentence for the reader, which is why the wording of a refusal is still
/// the shell's and the CONDITION for it is no longer duplicated.
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
