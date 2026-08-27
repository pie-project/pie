//! Page arithmetic: where a lane's kv rows land, and what the geometry
//! vectors say about it.
//!
//! Not one line of this names a device. That is not tidiness — design §6
//! gives `driver::store` (`kv.rs`, `budget.rs`) to exactly this arithmetic
//! and leaves only the BYTES to a shell, so every item below is marked
//! `driver::store candidate` and is written to be lifted whole once that
//! module exists. Until then it lives here, private to the shell, and its
//! tests run on a laptop.
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
//! schedule is planned against. The third is what the append kernel writes
//! through, and it is stated per token rather than derived from a position
//! because a derivation cannot spell a fresh-page write that is not the page
//! run's tail (`kernels/attn/kv.cuh`, the explicit-descriptor writer).
//!
//! # Why the pages are a block per slot
//!
//! v1 hands each slot one contiguous run of pages, sized at the deployment's
//! context ceiling. It is the arithmetic a page ALLOCATOR would replace, and
//! the reason not to write one yet is that a free-list changes nothing above
//! this file: the geometry vectors are the interface, and a lane's pages are
//! whatever `pages_of` says they are.

use model_ir::{Attention, CacheRow, Def, Dim, Operation, Plan, Ty, ValueId};

use crate::error::{Fault, Result};

/// What one kv space's consumers restate about it.
///
/// **THE IR DOES NOT CARRY THESE, ON PURPOSE.** `CacheRow::Kv` is storage
/// only — a per-token row shape and an element — and heads, head width and
/// the sliding window live on the OPS that read the space (`Attention::Decode
/// { head_dim }`, `Attention::Prefill { head_dim, kv_heads }`). So the shell
/// reads them off the plan rather than off a config it would then have to
/// keep in step, and a disagreement between two consumers of one space is a
/// refusal here instead of a schedule carved at the wrong shape.
///
// driver::store candidate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpaceFacts {
    /// The kv head width both sides restate.
    pub head_dim: u32,
    /// How many kv heads one row carries.
    pub kv_heads: u32,
    /// How many query heads attend them — the q rectangle's width over
    /// `head_dim`, which is the only place this number is written down.
    pub q_heads: u32,
    /// The sliding window the schedules are carved for, if any.
    pub window: Option<u32>,
}

/// Read every kv space's facts off the plan that uses them.
///
/// The answer is indexed by `space`, with `None` for a space no attention op
/// names — a plan may declare a space its trace never reads, and refusing
/// that would refuse a legal model.
///
/// # Errors
///
/// [`Fault::Unbound`] for a space whose consumers cannot be read: two ops
/// disagreeing about the head width, or a q rectangle that is not a whole
/// number of heads wide.
///
// driver::store candidate
pub fn probe(plan: &Plan) -> Result<Vec<Option<SpaceFacts>>> {
    let spaces = plan
        .caches
        .iter()
        .filter_map(|row| match row {
            CacheRow::Kv { space, .. } => Some(*space as usize + 1),
            CacheRow::State { .. } => None,
        })
        .max()
        .unwrap_or(0);
    let mut found: Vec<Option<SpaceFacts>> = vec![None; spaces];

    for node in &plan.nodes {
        let Operation::Attention(op) = &node.op else {
            continue;
        };
        let (cache, q, head_dim, kv_heads, window) = match op {
            Attention::Decode {
                q,
                cache,
                window,
                head_dim,
                ..
            }
            | Attention::DecodeLse {
                q,
                cache,
                window,
                head_dim,
                ..
            }
            | Attention::Masked {
                q,
                cache,
                window,
                head_dim,
                ..
            } => (*cache, *q, *head_dim, None, *window),
            Attention::Prefill {
                q,
                cache,
                window,
                head_dim,
                kv_heads,
                ..
            }
            | Attention::PrefillLse {
                q,
                cache,
                window,
                head_dim,
                kv_heads,
                ..
            } => (*cache, *q, *head_dim, Some(*kv_heads), *window),
            _ => continue,
        };
        let Some(space) = space_of(plan, cache) else {
            continue;
        };
        let width = width_of(plan, q)?;
        if head_dim == 0 || width % u64::from(head_dim) != 0 {
            return Err(Fault::Unbound {
                what: format!(
                    "cache space {space}, whose query rectangle is {width} wide and \
                     whose head width is {head_dim} — not a whole number of heads"
                ),
            });
        }
        let q_heads = u32::try_from(width / u64::from(head_dim)).unwrap_or(u32::MAX);
        let seat = found.get_mut(space as usize).ok_or_else(|| Fault::Unbound {
            what: format!("cache space {space}, which its own cache row does not declare"),
        })?;
        match seat {
            None => {
                *seat = Some(SpaceFacts {
                    head_dim,
                    kv_heads: kv_heads.unwrap_or(0),
                    q_heads,
                    window,
                });
            }
            Some(known) => {
                if known.head_dim != head_dim || known.q_heads != q_heads || known.window != window
                {
                    return Err(Fault::Unbound {
                        what: format!(
                            "cache space {space}, whose consumers disagree about its \
                             shape: {known:?} against head_dim {head_dim}, q_heads \
                             {q_heads}, window {window:?}"
                        ),
                    });
                }
                if let Some(heads) = kv_heads {
                    if known.kv_heads != 0 && known.kv_heads != heads {
                        return Err(Fault::Unbound {
                            what: format!(
                                "cache space {space}, whose consumers state {} and \
                                 {heads} kv heads",
                                known.kv_heads
                            ),
                        });
                    }
                    known.kv_heads = heads;
                }
            }
        }
    }
    Ok(found)
}

/// The space a cache-id value names, or `None` for a recurrent one.
///
// driver::store candidate
#[must_use]
pub fn space_of(plan: &Plan, cache: ValueId) -> Option<u32> {
    let Def::Cache(row) = plan.values.get(cache.0 as usize)?.def else {
        return None;
    };
    match plan.caches.get(row as usize)? {
        CacheRow::Kv { space, .. } => Some(*space),
        CacheRow::State { .. } => None,
    }
}

/// A value's row width: everything in its declared shape past the leading
/// dim, which is the IR's own reading of `rows x width`.
///
// driver::store candidate
pub fn width_of(plan: &Plan, value: ValueId) -> Result<u64> {
    let decl = plan
        .values
        .get(value.0 as usize)
        .ok_or_else(|| Fault::Unbound {
            what: format!("value {}, which its own plan does not declare", value.0),
        })?;
    let Ty::Tensor { shape, .. } = &decl.ty else {
        return Err(Fault::Unbound {
            what: format!("value {}, which declares a host struct, as a rectangle", value.0),
        });
    };
    let mut width = 1u64;
    for dim in shape.iter().skip(1) {
        match dim {
            Dim::Const(n) => width = width.saturating_mul(*n),
            other => {
                return Err(Fault::Unbound {
                    what: format!(
                        "value {}, whose width carries the symbolic dim {other:?}",
                        value.0
                    ),
                });
            }
        }
    }
    Ok(width)
}

/// How a deployment hands its pages out: a fixed block per slot.
///
// driver::store candidate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Paging {
    /// Tokens per page.
    pub page_size: u32,
    /// Pages every slot owns, whether it uses them or not.
    pub pages_per_slot: u32,
    /// How many slots the pool holds.
    pub slots: u32,
}

impl Paging {
    /// The paging that gives `slots` sequences `context` tokens each.
    ///
    /// # Errors
    ///
    /// [`Fault::Ceiling`] for a page size of zero, which spells no geometry
    /// at all.
    pub fn of(page_size: u32, context: u32, slots: u32) -> Result<Paging> {
        if page_size == 0 {
            return Err(Fault::Ceiling {
                what: "tokens per page",
                need: 1,
                have: 0,
            });
        }
        Ok(Paging {
            page_size,
            pages_per_slot: context.div_ceil(page_size).max(1),
            slots,
        })
    }

    /// Every page the pool holds.
    #[must_use]
    pub fn pages(&self) -> u64 {
        u64::from(self.pages_per_slot) * u64::from(self.slots)
    }

    /// The most tokens one slot can hold.
    #[must_use]
    pub fn context(&self) -> u32 {
        self.pages_per_slot.saturating_mul(self.page_size)
    }

    /// The first page id of a slot's block.
    #[must_use]
    pub fn base(&self, slot: u32) -> u64 {
        u64::from(slot) * u64::from(self.pages_per_slot)
    }
}

/// One lane of a fire, as the page arithmetic needs it.
///
// driver::store candidate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Seat {
    /// Which pool slot this lane's sequence lives in.
    pub slot: u32,
    /// How many kv tokens the slot already holds.
    pub have: u32,
    /// How many token rows this fire adds.
    pub rows: u32,
}

/// One fire's geometry for one kv space, host side — the vectors the shell
/// uploads and the twins the plan builders walk.
///
/// The lanes are in FIRE order, which is the seriated order the composition
/// chose, not the order the engine submitted: every one of these is indexed
/// by the descriptor's lane offset, and reading them in submission order is
/// how a request attends another request's pages.
///
// driver::store candidate
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Geometry {
    /// `[lanes + 1]`: each lane's span of [`indices`](Geometry::indices).
    pub indptr: Vec<i32>,
    /// The flat page-id list those bounds cut.
    pub indices: Vec<i32>,
    /// `[lanes]`: how full each lane's last page is, after this fire.
    pub last_page_len: Vec<i32>,
    /// `[lanes]`: each lane's total kv length, after this fire.
    pub kv_len: Vec<i32>,
    /// `[rows]`: the page each token row is appended into.
    pub write_page: Vec<i32>,
    /// `[rows]`: its offset inside that page.
    pub write_offset: Vec<i32>,
}

/// Compute one fire's geometry.
///
/// **THE LENGTHS ARE AFTER THE APPEND, AND THAT IS NOT A CHOICE.** The
/// attention this fire runs must see the tokens this fire just wrote — a
/// prefill attends its own prompt — so `kv_len` and `last_page_len` describe
/// the space as it will be once `attention.kv_append` has run, and the append
/// is ordered before the attention by the plan.
///
/// # Errors
///
/// [`Fault::Ceiling`] for a lane past the pool's slots or past its slot's
/// page block.
///
// driver::store candidate
pub fn geometry(paging: &Paging, seats: &[Seat]) -> Result<Geometry> {
    geometry_with(paging, seats, &[])
}

/// Compute one fire's geometry against a CALLER-SUPPLIED page table.
///
/// **WHO OWNS THE PAGE TABLE IS A SUBMISSION-LEVEL FACT** (`KvDelta::pages`,
/// which says: empty means the shell owns it). v1's own paging hands each
/// slot one contiguous block, which is the arithmetic
/// [`geometry`] does; an engine with a real page allocator — copy-on-write
/// forks, a prefix cache, pages that move between sequences — keeps its own
/// table and states it per lane, and then the block formula is wrong for
/// every lane.
///
/// So `tables[i]`, when non-empty, is lane `i`'s pages IN SEQUENCE ORDER and
/// every page id here is the caller's. The rest is unchanged: the lengths
/// are still after the append, and a token still lands at
/// `pages[at / page_size]`, offset `at % page_size`.
///
/// # Errors
///
/// [`Fault::Ceiling`] for a lane past the pool's slots (shell-owned only),
/// past its slot's page block (shell-owned only), or — caller-owned — one
/// whose stated pages do not cover the tokens it is about to hold.
///
// driver::store candidate
pub fn geometry_with(paging: &Paging, seats: &[Seat], tables: &[&[u32]]) -> Result<Geometry> {
    let rows: u64 = seats.iter().map(|s| u64::from(s.rows)).sum();
    let mut out = Geometry {
        indptr: Vec::with_capacity(seats.len() + 1),
        indices: Vec::new(),
        last_page_len: Vec::with_capacity(seats.len()),
        kv_len: Vec::with_capacity(seats.len()),
        write_page: Vec::with_capacity(rows as usize),
        write_offset: Vec::with_capacity(rows as usize),
    };
    out.indptr.push(0);

    for (lane, seat) in seats.iter().enumerate() {
        let table = tables.get(lane).copied().unwrap_or(&[]);
        let after = u64::from(seat.have) + u64::from(seat.rows);
        // A lane with rows always holds at least one page; `after` is at
        // least one because a lane IS its rows (`fire::Fault::EmptyLane`).
        let pages = after.div_ceil(u64::from(paging.page_size)).max(1);

        if table.is_empty() {
            if seat.slot >= paging.slots {
                return Err(Fault::Ceiling {
                    what: "kv slots",
                    need: u64::from(seat.slot) + 1,
                    have: u64::from(paging.slots),
                });
            }
            if after > u64::from(paging.context()) {
                return Err(Fault::Ceiling {
                    what: "kv tokens in one slot",
                    need: after,
                    have: u64::from(paging.context()),
                });
            }
            let base = paging.base(seat.slot);
            for page in 0..pages {
                out.indices.push(narrow(base + page));
            }
            for token in 0..u64::from(seat.rows) {
                let at = u64::from(seat.have) + token;
                out.write_page
                    .push(narrow(base + at / u64::from(paging.page_size)));
                out.write_offset
                    .push(narrow(at % u64::from(paging.page_size)));
            }
        } else {
            if (table.len() as u64) < pages {
                return Err(Fault::Ceiling {
                    what: "kv pages this lane stated",
                    need: pages,
                    have: table.len() as u64,
                });
            }
            for &page in &table[..pages as usize] {
                out.indices.push(narrow(u64::from(page)));
            }
            for token in 0..u64::from(seat.rows) {
                let at = u64::from(seat.have) + token;
                let page = table[(at / u64::from(paging.page_size)) as usize];
                out.write_page.push(narrow(u64::from(page)));
                out.write_offset
                    .push(narrow(at % u64::from(paging.page_size)));
            }
        }

        out.indptr.push(narrow(out.indices.len() as u64));
        out.last_page_len
            .push(narrow(after - (pages - 1) * u64::from(paging.page_size)));
        out.kv_len.push(narrow(after));
    }
    Ok(out)
}

/// The fire's shared boundary vector: `[lanes + 1]` token-row bounds.
///
/// Ambient rather than declared (design §5 removed `qo_indptr` as a named
/// input), which is why it is computed beside the per-space geometry instead
/// of inside it — one fire has one of these however many cache spaces it
/// touches.
///
// driver::store candidate
#[must_use]
pub fn indptr(seats: &[Seat]) -> Vec<i32> {
    let mut out = Vec::with_capacity(seats.len() + 1);
    let mut at = 0u64;
    out.push(0);
    for seat in seats {
        at += u64::from(seat.rows);
        out.push(narrow(at));
    }
    out
}

/// A count, as the `i32` every geometry vector is spelled in.
///
/// Saturating rather than wrapping: these are page ids and token offsets, all
/// bounded by budgets checked above, and a count that reached `i32::MAX` here
/// would be a bound that was never enforced — clamping keeps it a wrong
/// number rather than a negative one, which is what a kernel would read as an
/// enormous unsigned index.
fn narrow(n: u64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spaces_heads_are_read_off_the_ops_that_restate_them() {
        // The IR carries a per-token row and an element, and nothing else:
        // heads and head width live on `Attention::Decode`/`Prefill`, and the
        // query head count only on the q rectangle's own declaration. Reading
        // them here is what keeps a pool's shape and a model's text from
        // being two opinions.
        let trace = model::trace_of("qwen35-d0.8b-bf16-kv-bf16")
            .expect("the catalog ships the smoke's SKU");
        let plan = trace(model_dsl::Plane::Cuda);
        let spaces = probe(&plan).expect("a hybrid SKU's one kv space reads");

        assert_eq!(spaces.len(), 1, "qwen3.5 declares one kv geometry space");
        assert_eq!(
            spaces[0],
            Some(SpaceFacts {
                head_dim: 256,
                kv_heads: 2,
                q_heads: 8,
                window: None,
            }),
        );

        // And the recurrent rows are not in it: `space_of` answers `None` for
        // a state cache, which is what keeps the 36 gdn banks out of the page
        // arithmetic entirely.
        let states = plan
            .caches
            .iter()
            .filter(|row| matches!(row, model_ir::CacheRow::State { .. }))
            .count();
        assert_eq!(states, 36, "18 gdn layers, a conv bank and a delta bank each");
    }

    fn paging() -> Paging {
        Paging::of(16, 64, 4).expect("a page size of 16 spells geometry")
    }

    #[test]
    fn a_slot_owns_a_contiguous_block_of_pages() {
        let p = paging();
        assert_eq!(p.pages_per_slot, 4);
        assert_eq!(p.context(), 64);
        assert_eq!(p.pages(), 16);
        assert_eq!(p.base(0), 0);
        assert_eq!(p.base(3), 12);
    }

    #[test]
    fn a_prefill_writes_its_own_prompt_and_then_attends_it() {
        // One lane, 20 tokens into an empty slot: two pages, the second one
        // four tokens deep, and every row addressed explicitly.
        let g = geometry(
            &paging(),
            &[Seat {
                slot: 1,
                have: 0,
                rows: 20,
            }],
        )
        .expect("one lane of 20 rows pages");

        assert_eq!(g.indptr, vec![0, 2]);
        assert_eq!(g.indices, vec![4, 5], "slot 1's block starts at page 4");
        assert_eq!(g.kv_len, vec![20], "the length is AFTER the append");
        assert_eq!(g.last_page_len, vec![4]);
        assert_eq!(g.write_page.len(), 20);
        assert_eq!(&g.write_page[..17], &[4; 16].iter().chain(&[5]).copied().collect::<Vec<_>>()[..]);
        assert_eq!(g.write_offset[0], 0);
        assert_eq!(g.write_offset[15], 15);
        assert_eq!(g.write_offset[16], 0, "row 16 opens the second page");
    }

    #[test]
    fn a_decode_step_appends_one_row_past_what_the_slot_holds() {
        let g = geometry(
            &paging(),
            &[Seat {
                slot: 0,
                have: 20,
                rows: 1,
            }],
        )
        .expect("one decode row pages");

        assert_eq!(g.kv_len, vec![21]);
        assert_eq!(g.indptr, vec![0, 2]);
        assert_eq!(g.last_page_len, vec![5]);
        assert_eq!(g.write_page, vec![1], "token 20 is page 1 of slot 0's block");
        assert_eq!(g.write_offset, vec![4]);
    }

    #[test]
    fn a_full_page_is_full_rather_than_one_page_short() {
        // 16 tokens at a page size of 16 is ONE page whose last page length
        // is 16 — not two pages with an empty tail. Getting this wrong is a
        // schedule that reads a page of poison.
        let g = geometry(
            &paging(),
            &[Seat {
                slot: 0,
                have: 0,
                rows: 16,
            }],
        )
        .expect("a full page pages");
        assert_eq!(g.indptr, vec![0, 1]);
        assert_eq!(g.last_page_len, vec![16]);
    }

    #[test]
    fn lanes_are_laid_out_in_fire_order() {
        let g = geometry(
            &paging(),
            &[
                Seat { slot: 2, have: 3, rows: 2 },
                Seat { slot: 0, have: 0, rows: 1 },
            ],
        )
        .expect("two lanes page");

        assert_eq!(g.indptr, vec![0, 1, 2]);
        assert_eq!(g.indices, vec![8, 0], "slot 2 first, because the fire says so");
        assert_eq!(g.kv_len, vec![5, 1]);
        assert_eq!(g.write_page, vec![8, 8, 0]);
        assert_eq!(g.write_offset, vec![3, 4, 0]);
        assert_eq!(indptr(&[
            Seat { slot: 2, have: 3, rows: 2 },
            Seat { slot: 0, have: 0, rows: 1 },
        ]), vec![0, 2, 3]);
    }

    #[test]
    fn a_sequence_past_its_slots_pages_is_refused_rather_than_wrapped() {
        let refusal = geometry(
            &paging(),
            &[Seat { slot: 0, have: 60, rows: 8 }],
        );
        assert!(
            matches!(refusal, Err(Fault::Ceiling { need: 68, have: 64, .. })),
            "a slot that overruns its block must name the numbers: {refusal:?}"
        );
    }

    #[test]
    fn a_lane_past_the_pools_slots_is_refused() {
        let refusal = geometry(&paging(), &[Seat { slot: 4, have: 0, rows: 1 }]);
        assert!(matches!(refusal, Err(Fault::Ceiling { need: 5, have: 4, .. })));
    }
}
