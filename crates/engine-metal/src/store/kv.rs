//! What the plan says about its own caches: the rows' readings and the
//! readings the schedules are carved for.
//!
//! **THE PAGE ARITHMETIC IS NO LONGER HERE.** `Paging`, `Seat`, `Geometry`,
//! `geometry`/`geometry_with` and `indptr` — every item this file used to
//! carry a `// engine::store candidate` marker over — live in
//! [`engine::store::kv`]. This file used to be the CUDA shell's file line for
//! line, and that duplication was the evidence rather than the design: a page
//! id is not a pointer, a token offset is not a buffer handle, and nothing in
//! it was encoded, allocated or bound. What is re-exported below is that
//! module, spelled at the path this shell's callers already use.
//!
//! # Why this shell's `probe` stayed
//!
//! It is the one thing the two files did NOT agree about. This one reads a
//! schedule's reading off the LAUNCHES that consume it — two passes, refusing
//! when two launches read one plan value at two readings — while the CUDA
//! twin reads it off the plan op that CARVES it, keeps the value's
//! `StructKind` beside the reading, and walks the latent (mla) arms this
//! plane has no shaders for. That is a behaviour difference rather than a
//! spelling one, so it is stated twice on purpose until this plane grows the
//! same ops.
//!
//! # The three questions
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
//! schedule is planned against. The third is what the append shader writes
//! through, and it is stated per token rather than derived from a position
//! because a derivation cannot spell a fresh-page write that is not the page
//! run's tail. On this plane that is not even a choice the shell could make
//! differently: `attn/kv_write.metal`'s `kv_append_paged_*` entries take
//! `write_page` and `write_offset` as declared buffers and derive nothing,
//! and the ops themselves name them (`attention.kv_append`), so a shell that
//! wanted a derivation would have nowhere to put it.

use model_ir::{Trace, ValueId};

use crate::error::{Fault, Result};

pub use engine::store::kv::{
    Geometry, Paging, Reader, Seat, SpaceFacts, indptr, reads, row_of, space_of,
};

/// Compute one fire's geometry — [`engine::store::kv::geometry`], in this
/// shell's `Result`.
///
/// # Errors
///
/// [`Fault::Ceiling`] for a lane past the pool's slots or past its slot's
/// page block.
pub fn geometry(paging: &Paging, seats: &[Seat]) -> Result<Geometry> {
    Ok(engine::store::kv::geometry(paging, seats)?)
}

/// Compute one fire's geometry against a caller-supplied page table —
/// [`engine::store::kv::geometry_with`], in this shell's `Result`.
///
/// # Errors
///
/// [`Fault::Ceiling`] for a lane past the pool's slots (shell-owned only),
/// past its slot's page block (shell-owned only), or — caller-owned — one
/// whose stated pages do not cover the tokens it is about to hold.
pub fn geometry_with(paging: &Paging, seats: &[Seat], tables: &[&[u32]]) -> Result<Geometry> {
    Ok(engine::store::kv::geometry_with(paging, seats, tables)?)
}

/// A value's row width — [`engine::store::kv::width_of`], in this shell's
/// `Result`.
///
/// # Errors
///
/// [`Fault::Unbound`] for a value its own plan does not declare, for one that
/// declares a host struct rather than a rectangle, and for one whose width
/// carries a symbolic dim.
pub fn width_of(trace: &Trace, value: ValueId) -> Result<u64> {
    Ok(engine::store::kv::width_of(trace, value)?)
}

/// Everything the plan restates about its own caches, keyed two ways.
///
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Facts {
    /// Per `Trace::caches` ROW: what one page of it holds. `None` for a row
    /// no attention op reads.
    pub rows: Vec<Option<SpaceFacts>>,
    /// Per `Trace::values` ID: the reading one attention SCHEDULE is carved
    /// for. `None` for every value that is not a plan struct some launch
    /// consumes.
    pub plans: Vec<Option<SpaceFacts>>,
}

impl Facts {
    /// The facts a cache row's pool is sized at.
    ///
    /// # Errors
    ///
    /// [`Fault::Unbound`] for a row no attention op reads, so nothing states
    /// its heads.
    pub fn row(&self, at: usize, name: &str) -> Result<SpaceFacts> {
        self.rows
            .get(at)
            .copied()
            .flatten()
            .ok_or_else(|| Fault::Unbound {
                what: format!(
                    "cache `{name}`, which no attention op reads — so nothing states its heads"
                ),
            })
    }
}

/// Read every cache row's and every attention schedule's facts off the plan.
///
/// # Errors
///
/// [`Fault::Unbound`] for a cache row two ops give two widths, for a plan
/// value two launches read at two readings, or for a q rectangle that is not
/// a whole number of heads wide.
///
pub fn probe(trace: &Trace) -> Result<Facts> {
    let mut out = Facts {
        rows: vec![None; trace.caches.len()],
        plans: vec![None; trace.values.len()],
    };

    // Pass one: the rows. `kv_heads` is stated by the prefill arms alone, so
    // a row only a decode reads carries zero — the reading the sibling shell
    // has always had, kept.
    for node in &trace.nodes {
        let Some(read) = reads(&node.op) else {
            continue;
        };
        let Some(row) = row_of(trace, read.cache) else {
            continue;
        };
        let seat = out.rows.get_mut(row).ok_or_else(|| Fault::Unbound {
            what: format!("cache row {row}, which this plan does not declare"),
        })?;
        match seat {
            None => {
                *seat = Some(SpaceFacts {
                    head_dim: read.head_dim,
                    kv_heads: read.kv_heads.unwrap_or(0),
                    q_heads: 0,
                    window: None,
                });
            }
            Some(known) => {
                if known.head_dim != read.head_dim {
                    return Err(Fault::Unbound {
                        what: format!(
                            "cache row {row}, whose readers disagree about its shape: \
                             head_dim {} against head_dim {}",
                            known.head_dim, read.head_dim
                        ),
                    });
                }
                if let Some(heads) = read.kv_heads {
                    if known.kv_heads != 0 && known.kv_heads != heads {
                        return Err(Fault::Unbound {
                            what: format!(
                                "cache row {row}, whose readers state {} and {heads} kv heads",
                                known.kv_heads
                            ),
                        });
                    }
                    known.kv_heads = heads;
                }
            }
        }
    }

    // Pass two: the schedules. A plan value is carved for ONE reading —
    // head width, query heads and window — and its kv head count is the row's,
    // because `attention.masked` and `attention.decode` state none.
    for node in &trace.nodes {
        let Some(read) = reads(&node.op) else {
            continue;
        };
        let width = width_of(trace, read.q)?;
        if read.head_dim == 0 || width % u64::from(read.head_dim) != 0 {
            return Err(Fault::Unbound {
                what: format!(
                    "plan value {}, whose query rectangle is {width} wide and whose head \
                     width is {} — not a whole number of heads",
                    read.plan.0, read.head_dim
                ),
            });
        }
        let kv_heads = row_of(trace, read.cache)
            .and_then(|row| out.rows.get(row).copied().flatten())
            .map_or(0, |row| row.kv_heads);
        let stated = SpaceFacts {
            head_dim: read.head_dim,
            kv_heads,
            q_heads: u32::try_from(width / u64::from(read.head_dim)).unwrap_or(u32::MAX),
            window: read.window,
        };
        let seat = out
            .plans
            .get_mut(read.plan.0 as usize)
            .ok_or_else(|| Fault::Unbound {
                what: format!("plan value {}, which this plan does not declare", read.plan.0),
            })?;
        match seat {
            None => *seat = Some(stated),
            Some(known) if *known == stated => {}
            Some(known) => {
                return Err(Fault::Unbound {
                    what: format!(
                        "plan value {}, whose launches read it at two readings — {known:?} \
                         against {stated:?}. A schedule is carved for ONE of them (the \
                         window sizes its kv chunking, the head width its tile), so the \
                         model text mints a second plan for the second reading rather \
                         than sharing this one",
                        read.plan.0
                    ),
                });
            }
        }
    }
    Ok(out)
}
