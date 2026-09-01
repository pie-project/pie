//! Page arithmetic: where a lane's kv rows land, and what the geometry
//! vectors say about it.
//!
//! **THIS WAS THE CUDA SHELL'S FILE AND THE METAL SHELL'S FILE, LINE FOR
//! LINE.** A page id is not a pointer, a token offset is not a buffer handle,
//! and nothing here is encoded, allocated or bound — so the two copies
//! diverged only where a sentence named a `.cuh` or a `.metal`. Two shells
//! that compute a lane's pages the same way was a property kept by making the
//! two files answerable against each other; it is structural now.
//!
//! # Why the pages are a block per slot
//!
//! v1 hands each slot one contiguous run of pages, sized at the deployment's
//! context ceiling. It is the arithmetic a page ALLOCATOR would replace, and
//! the reason not to write one yet is that a free-list changes nothing above
//! this file: the geometry vectors are the interface, and a lane's pages are
//! whatever [`geometry_with`]'s stated table says they are.

use model_ir::{Attention, CacheRow, Def, Dim, Operation, Trace, Ty, ValueId};

use crate::store::{Fault, Result};

/// A reading: query heads, kv heads, head width, sliding window — held by a
/// cache ROW (what one page of it carries) or by a plan value (what one
/// schedule is carved for).
///
/// **THE IR CARRIES THESE WHERE EACH HAS AN AUTHOR.** `CacheRow::Kv` is
/// storage only — the planes one entry is written as, and an element — so a
/// row's heads and head width are read off the OPS that walk the space
/// (`Attention::Decode { head_dim }`, `Attention::Prefill { head_dim,
/// kv_heads }`), not off a config the shell would then have to keep in step.
/// A SCHEDULE is the other way round: it is carved for ONE reading before any
/// launch touches it, so the op that carves it states that reading outright
/// (`Attention::PlanDecode { q_heads, kv_heads, head_dim, window }`, and
/// `Attention::MlaPlan { heads, kv_lora_rank }` for the absorbed latent one).
/// The shell reads it off that op, and every launch that consumes the
/// schedule restates its share for the shell to check.
///
/// **WHAT THEY ARE KEYED BY IS THE WHOLE OF BUILD LOG 20's FIRST BLOCKER.**
/// They used to be folded up to the geometry SPACE, one answer per space, and
/// two consumers that disagreed were a refusal. That reading cannot state
/// gemma, whose sliding layers are 2 heads of 256 under a 512-wide window
/// while its global layers are 2 heads of 512 with no window — one sequence,
/// one page-id space, two readings — and it cannot state gpt-oss, whose two
/// layer kinds share a row width and alternate a 128-wide window with none.
/// Neither model is lying: a space is the PAGE-ID space (one `page_size`, one
/// block per slot, one lane's pages), and nothing about a page id says how
/// wide the row it addresses is. The dev lineage says the same in its own
/// vocabulary — one `KvCache` with `per_layer_head_dim_`,
/// `per_layer_num_kv_heads_` and a `per_layer_window_left` on the weights —
/// and the IR already agrees, because `CacheRow::Kv { planes }` is declared
/// per ROW and a shell's `Pools::reserve` already allocates one slab per row
/// at that row's own planes.
///
/// So the facts are keyed by the thing that actually holds them:
///
/// ```text
/// the ROW   head_dim, kv_heads   what a paged launch restates about this cache
/// the PLAN  + q_heads, window    the reading one schedule is carved for
/// ```
///
/// and a disagreement is still a refusal — it names a cache row whose readers
/// give it two widths, or a plan value whose launch restates a reading its
/// schedule was not carved for, and the sentence for a plan tells the author
/// to state the second reading on a second plan op rather than to give up.
///
/// **A ROW'S BYTES ARE NOT IN HERE.** They are the declaration's:
/// `CacheRow::Kv` names the planes one entry is written as, at their own
/// widths, and the pool allocates what is declared. A row's facts are the
/// RESTATEMENT the paged launches make of that row, which a shell's pool
/// reservation checks the declaration against — and which is simply absent
/// for a row no paged launch reads (a latent page, an indexer's keys, a
/// pooled cache's entries: the latent, index and pool launches do not feed
/// the row pass).
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

/// What one paged-kv launch restates, flattened out of the variant. `q` is the
/// rectangle whose width names its query heads — the only place a paged launch
/// writes that number down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Reader {
    /// The query rectangle.
    pub q: ValueId,
    /// The schedule it reads.
    pub plan: ValueId,
    /// The cache-id value it walks.
    pub cache: ValueId,
    /// The kv head width it restates.
    pub head_dim: u32,
    /// The kv head count, for the arms that state one.
    pub kv_heads: Option<u32>,
    /// The sliding window it restates.
    pub window: Option<u32>,
}

/// The five paged-kv launches, as one shape.
#[must_use]
pub fn reads(op: &Operation) -> Option<Reader> {
    let Operation::Attention(op) = op else {
        return None;
    };
    match op {
        Attention::Decode {
            q,
            plan,
            cache,
            window,
            head_dim,
            ..
        }
        | Attention::DecodeLse {
            q,
            plan,
            cache,
            window,
            head_dim,
            ..
        }
        | Attention::Masked {
            q,
            plan,
            cache,
            window,
            head_dim,
            ..
        } => Some(Reader {
            q: *q,
            plan: *plan,
            cache: *cache,
            head_dim: *head_dim,
            kv_heads: None,
            window: *window,
        }),
        Attention::Prefill {
            q,
            plan,
            cache,
            window,
            head_dim,
            kv_heads,
            ..
        }
        | Attention::PrefillLse {
            q,
            plan,
            cache,
            window,
            head_dim,
            kv_heads,
            ..
        } => Some(Reader {
            q: *q,
            plan: *plan,
            cache: *cache,
            head_dim: *head_dim,
            kv_heads: Some(*kv_heads),
            window: *window,
        }),
        _ => None,
    }
}

/// The `Trace::caches` row a cache-id value names, or `None` for a recurrent
/// one.
#[must_use]
pub fn row_of(trace: &Trace, cache: ValueId) -> Option<usize> {
    let Def::Cache(row) = trace.values.get(cache.0 as usize)?.def else {
        return None;
    };
    match trace.caches.get(row as usize)? {
        CacheRow::Kv { .. } => Some(row as usize),
        CacheRow::State { .. } => None,
    }
}

/// The space a cache-id value names, or `None` for a recurrent one.
#[must_use]
pub fn space_of(trace: &Trace, cache: ValueId) -> Option<u32> {
    let Def::Cache(row) = trace.values.get(cache.0 as usize)?.def else {
        return None;
    };
    match trace.caches.get(row as usize)? {
        CacheRow::Kv { space, .. } => Some(*space),
        CacheRow::State { .. } => None,
    }
}

/// A value's row width: everything in its declared shape past the leading
/// dim, which is the IR's own reading of `rows x width`.
///
/// # Errors
///
/// [`Fault::Unbound`] for a value its own plan does not declare, for one
/// that declares a host struct rather than a rectangle, and for one whose
/// width carries a symbolic dim.
pub fn width_of(trace: &Trace, value: ValueId) -> Result<u64> {
    let decl = trace
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
/// hands the device and the twins the plan builders walk.
///
/// The lanes are in FIRE order, which is the seriated order the composition
/// chose, not the order the runtime submitted: every one of these is indexed
/// by the descriptor's lane offset, and reading them in submission order is
/// how a request attends another request's pages.
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

impl Geometry {
    /// **EXTEND THE LANE TABLES OUT TO `lanes`, AS GENUINELY EMPTY LANES.**
    ///
    /// A fire staged at a bucket's lane CEILING hands its kernels more lanes
    /// than it brought, and what those lanes hold decides whether the guards
    /// are belt-and-braces or load-bearing. Stale bytes from the last fire
    /// are a page run that still points at somebody's pages; this writes the
    /// one thing that is true of a lane nobody submitted — no pages, no
    /// tokens — so a reader that walks a padded lane finds emptiness rather
    /// than a neighbour's cache.
    ///
    /// * [`indptr`](Geometry::indptr) repeats its last bound, which is
    ///   `[begin, begin)`: zero pages, and flashinfer's `get_length` returns
    ///   `0` for exactly that test (`page.cuh`, `indptr[b + 1] ==
    ///   indptr[b]`) without reading anything else. It is also what makes
    ///   `indptr[batch_size]` — the `last_indptr`
    ///   `paged_kv_t::protective_get_kv_offset` clamps against — DEFINED and
    ///   still monotone at a baked `batch_size` past the live lanes; too
    ///   small is that clamp's silent failure
    ///   (`kernels_cuda::attn::fa2_abi::make_paged_kv`).
    /// * [`last_page_len`](Geometry::last_page_len) is zero: the honest
    ///   reading of "this lane's last page holds nothing", and unread anyway
    ///   for a lane whose bounds are equal.
    /// * [`kv_len`](Geometry::kv_len) is zero: the length AFTER the append,
    ///   for a lane that appends nothing to nothing. The host plan builders
    ///   walk this one directly.
    ///
    /// [`indices`](Geometry::indices) does not move, because empty lanes own
    /// no pages; neither do the two row tables, because an empty lane brings
    /// no rows.
    ///
    /// **IT ONLY EVER GROWS.** A `lanes` at or below what the fire brought
    /// leaves every vector exactly as it was, which is what makes a caller's
    /// clamp of a ceiling down to its carve safe to spell as a `min`.
    pub fn pad_to(&mut self, lanes: usize) {
        pad_indptr(&mut self.indptr, lanes);
        self.last_page_len.resize(self.last_page_len.len().max(lanes), 0);
        self.kv_len.resize(self.kv_len.len().max(lanes), 0);
    }
}

/// **EXTEND A `[n + 1]` BOUNDARY VECTOR TO `[lanes + 1]` BY REPEATING ITS
/// LAST BOUND**, which spells `lanes - n` empty lanes and nothing else.
///
/// The one operation both boundary vectors of a fire want — the per-space
/// page CSR ([`Geometry::pad_to`]) and the fire-wide row vector
/// [`indptr`] — so it is written once. Never shrinks, for
/// [`Geometry::pad_to`]'s reason.
///
/// An EMPTY vector is left empty, because empty is a caller's off switch —
/// the fire-wide qo vector stages nothing when it holds nothing
/// (`engine_cuda::inputs::Fire::qo_absolute`) — and padding it would turn
/// that switch on.
pub fn pad_indptr(indptr: &mut Vec<i32>, lanes: usize) {
    let Some(&last) = indptr.last() else {
        return;
    };
    indptr.resize(indptr.len().max(lanes + 1), last);
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
pub fn geometry(paging: &Paging, seats: &[Seat]) -> Result<Geometry> {
    geometry_with(paging, seats, &[])
}

/// Compute one fire's geometry against a CALLER-SUPPLIED page table.
///
/// **WHO OWNS THE PAGE TABLE IS A SUBMISSION-LEVEL FACT** (`KvDelta::pages`,
/// which says: empty means the shell owns it). v1's own paging hands each
/// slot one contiguous block, which is the arithmetic
/// [`geometry`] does; a runtime with a real page allocator — copy-on-write
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

    #[test]
    fn a_caller_owned_page_table_is_used_verbatim() {
        // The runtime's own allocator: lane 0's pages are 9 and 2, in that
        // sequence order, and nothing about slot blocks applies.
        let table: [u32; 2] = [9, 2];
        let tables: [&[u32]; 1] = [&table];
        let g = geometry_with(
            &paging(),
            &[Seat { slot: 0, have: 14, rows: 4 }],
            &tables,
        )
        .expect("a stated table that covers 18 tokens pages");

        assert_eq!(g.indices, vec![9, 2]);
        assert_eq!(g.kv_len, vec![18]);
        assert_eq!(g.write_page, vec![9, 9, 2, 2], "row 16 crosses into page 2");
        assert_eq!(g.write_offset, vec![14, 15, 0, 1]);
    }

    #[test]
    fn a_stated_table_too_short_for_its_tokens_is_refused() {
        let table: [u32; 1] = [9];
        let tables: [&[u32]; 1] = [&table];
        let refusal = geometry_with(
            &paging(),
            &[Seat { slot: 0, have: 14, rows: 4 }],
            &tables,
        );
        assert!(
            matches!(refusal, Err(Fault::Ceiling { need: 2, have: 1, .. })),
            "18 tokens need two pages and the caller stated one: {refusal:?}"
        );
    }

    #[test]
    fn the_recurrent_rows_are_not_the_page_arithmetics() {
        // `space_of`/`row_of` answer `None` for a state cache, which is what
        // keeps the 36 gdn banks out of the page arithmetic entirely.
        let trace = model::trace_of("qwen35-d0.8b-bf16-kv-bf16")
            .expect("the catalog ships the smoke's SKU");
        let trace = trace(model_dsl::Platform::Cuda);
        let states = trace
            .caches
            .iter()
            .filter(|row| matches!(row, model_ir::CacheRow::State { .. }))
            .count();
        assert_eq!(states, 36, "18 gdn layers, a conv bank and a delta bank each");

        for (at, decl) in trace.values.iter().enumerate() {
            let model_ir::Def::Cache(row) = decl.def else {
                continue;
            };
            let id = ValueId(at as u32);
            match trace.caches[row as usize] {
                model_ir::CacheRow::Kv { space, .. } => {
                    assert_eq!(row_of(&trace, id), Some(row as usize));
                    assert_eq!(space_of(&trace, id), Some(space));
                }
                model_ir::CacheRow::State { .. } => {
                    assert_eq!(row_of(&trace, id), None);
                    assert_eq!(space_of(&trace, id), None);
                }
            }
        }
    }

    #[test]
    fn a_padded_lane_owns_no_page_and_no_token() {
        // Two live lanes, staged out to a ceiling of five: the page CSR goes
        // flat and the two per-lane tables go to zero, which is the whole of
        // what "an empty lane" means to a reader.
        let mut g = geometry(
            &paging(),
            &[
                Seat { slot: 0, have: 0, rows: 20 },
                Seat { slot: 1, have: 3, rows: 1 },
            ],
        )
        .expect("two lanes page");
        let pages = g.indices.clone();
        let (write_page, write_offset) = (g.write_page.clone(), g.write_offset.clone());

        g.pad_to(5);

        assert_eq!(g.indptr, vec![0, 2, 3, 3, 3, 3], "the tail repeats the last bound");
        assert_eq!(g.last_page_len, vec![4, 4, 0, 0, 0]);
        assert_eq!(g.kv_len, vec![20, 4, 0, 0, 0]);
        assert_eq!(g.indices, pages, "an empty lane owns no page");
        assert_eq!(g.write_page, write_page, "an empty lane brings no row");
        assert_eq!(g.write_offset, write_offset);
    }

    #[test]
    fn a_padded_lane_reads_zero_length_the_way_the_device_computes_it() {
        // `paged_kv_t::get_length` in page.cuh: equal bounds is zero, before
        // `last_page_len` is read at all. Restated here because the pad
        // values are chosen for it.
        let mut g = geometry(&paging(), &[Seat { slot: 2, have: 0, rows: 5 }])
            .expect("one lane pages");
        g.pad_to(3);

        let length = |lane: usize| -> i32 {
            if g.indptr[lane + 1] == g.indptr[lane] {
                return 0;
            }
            (g.indptr[lane + 1] - g.indptr[lane] - 1) * 16 + g.last_page_len[lane]
        };
        assert_eq!(length(0), 5, "the live lane still reads its own tokens");
        assert_eq!(length(1), 0);
        assert_eq!(length(2), 0);
        // And the protective bound the fa2 kernels clamp against is defined
        // at the CEILING, which is the whole point of padding the vector.
        assert_eq!(g.indptr[3], g.indptr[1], "monotone, and no page past the live ones");
    }

    #[test]
    fn a_pad_below_the_live_lanes_moves_nothing() {
        // What lets a caller clamp a bucket's ceiling down to its carve with
        // a `min` and stage the result unconditionally.
        let mut g = geometry(
            &paging(),
            &[
                Seat { slot: 0, have: 0, rows: 2 },
                Seat { slot: 1, have: 0, rows: 2 },
                Seat { slot: 2, have: 0, rows: 2 },
            ],
        )
        .expect("three lanes page");
        let before = g.clone();

        g.pad_to(1);
        g.pad_to(0);

        assert_eq!(g, before);
    }

    #[test]
    fn the_fire_wide_qo_vector_pads_to_zero_row_lanes() {
        let mut bounds = indptr(&[
            Seat { slot: 0, have: 0, rows: 7 },
            Seat { slot: 1, have: 0, rows: 1 },
        ]);
        assert_eq!(bounds, vec![0, 7, 8]);

        pad_indptr(&mut bounds, 4);

        assert_eq!(bounds, vec![0, 7, 8, 8, 8], "every padded lane spans no row");
        // Never shrinks, for `Geometry::pad_to`'s reason.
        pad_indptr(&mut bounds, 1);
        assert_eq!(bounds, vec![0, 7, 8, 8, 8]);
    }
}
