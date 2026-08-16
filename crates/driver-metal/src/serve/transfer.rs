//! The three control ops — `copy_kv`, `copy_state`, `resize_pool` — decided
//! before anything moves.
//!
//! These are the driver's non-launch entry points: fork a conversation's KV
//! pages, branch its recurrent state, grow or shrink an elastic pool. Each
//! arrives as a descriptor from the engine and each either refuses or does
//! something the caller cannot undo.
//!
//! This module is the *deciding* half — refuse or produce a plan — and it needs
//! no device. The moving half is `gpu/`, which executes a plan this
//! module has already declared valid.
//!
//! # Why the split is the point
//!
//! The C++ decides and moves in one pass, and the seam between its two halves
//! is where its atomicity claim is lost. `copy_kv_pages` opens with
//!
//! > *"Bounds-check EVERY page first — never a partial copy on a late
//! > failure."*
//!
//! and it honours that — for pages. But `copy_kv_impl` calls it and *then*
//! validates and copies the cells. A request whose pages are in range and whose
//! cells are not therefore copies every page, fails on a cell, and returns
//! `PIE_STATUS_DRIVER_ERROR` over a pool it has already changed. The invariant
//! is stated inside one half and broken across the two, which is the failure
//! mode a stated invariant is supposed to prevent.
//!
//! `copy_state` has the same shape one level down: it validates every slot
//! range up front, then executes them one at a time, so a range that fails in
//! the executor leaves its predecessors applied.
//!
//! Here a control op is planned whole. [`plan_kv_copy`] validates the pages
//! *and* the cells before either can move, and returns one [`KvCopyWork`]; if
//! it refuses, nothing has happened, because nothing that could happen exists
//! yet.
//!
//! # Refusals are values
//!
//! The C++ writes a sentence to `std::cerr` and returns a bare status code, so
//! the caller gets a number and the operator gets a log line with nothing tying
//! the two together. [`Refusal`] carries the reason *and* the status the ABI
//! must report, so the two cannot disagree and the caller can act on the
//! distinction it already has to make: [`Refusal::status`] separates "this
//! build cannot ever do that" from "that request was malformed".

use driver_api::local::{
    DeviceDomain, PIE_ELASTIC_POOL_KV, PIE_ELASTIC_POOL_STATE, PIE_ELASTIC_POOL_WORKSPACE,
    PIE_MEMORY_DOMAIN_METAL_SHARED, PIE_STATUS_INVALID_ARGUMENT, PIE_STATUS_UNSUPPORTED,
};
use driver_api::plan::{KvCopyPlan, PoolResizePlan, StateCopyPlan};

use crate::layout::kv_move::{CellMovePlan, KvMoveCell, PoolGrid, plan_cell_moves};

/// What the driver can do, as far as a control op is concerned.
///
/// Passed in rather than read from a global, because the C++ read `facts_` and
/// `executor_` from the enclosing `Impl` and so could only be tested with a
/// device attached. Every refusal below is a function of these three numbers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// Whether the loaded checkpoint has GDN / linear-attention layers. The
    /// recurrent state only exists if it does.
    pub has_linear_attn: bool,
    /// Physical pages in the paged KV pool; zero when no pool was allocated.
    pub kv_total_pages: u32,
    /// Recurrent-state slots the executor holds.
    pub rs_slots: u32,
}

/// Why a control op will not run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The checkpoint has no linear-attention layers, so there is no recurrent
    /// state and no hybrid KV geometry to act on.
    NoLinearAttention,
    /// A domain other than `PIE_MEMORY_DOMAIN_METAL_SHARED` was named. There is
    /// no host-pinned swap pool in this build, so a cross-domain copy is
    /// refused rather than silently reinterpreted as a same-domain one.
    ForeignDomain {
        /// The source domain the request named.
        src: DeviceDomain,
        /// The destination domain the request named.
        dst: DeviceDomain,
    },
    /// No paged KV pool is allocated: the configured page count and page size
    /// produced a zero-sized pool, or its allocation failed at setup.
    NoKvPool,
    /// The page-id arrays disagree in length, so no pairing exists.
    PageCountMismatch {
        /// Source page ids supplied.
        src: usize,
        /// Destination page ids supplied.
        dst: usize,
    },
    /// A page id is outside the pool.
    PageOutOfRange {
        /// Index of the offending pair in the request.
        index: usize,
        /// The offending page id.
        page: u32,
        /// Pages the pool holds.
        total_pages: u32,
    },
    /// A cell names a page or row outside the pool.
    CellOutOfRange {
        /// Index of the offending cell in the request.
        index: usize,
    },
    /// A slot id is outside the executor's recurrent-state slots.
    SlotOutOfRange {
        /// Index of the offending range in the request.
        index: usize,
        /// The offending slot id.
        slot: u32,
        /// Slots the executor holds.
        rs_slots: u32,
    },
    /// The pool id names no elastic pool.
    UnknownPool {
        /// The id the request named.
        pool_id: u64,
    },
}

impl Refusal {
    /// The ABI status this refusal reports.
    ///
    /// The split is the one the engine acts on, so it is worth stating rather
    /// than deriving at each call site: `UNSUPPORTED` means this build will
    /// never do it and the caller should stop asking; `INVALID_ARGUMENT` means
    /// this particular request was malformed and a different one may succeed.
    /// The C++ chose between them inline at ten sites.
    #[must_use]
    pub const fn status(&self) -> i32 {
        match self {
            // Facts about the build and the loaded checkpoint.
            Refusal::NoLinearAttention
            | Refusal::ForeignDomain { .. }
            | Refusal::NoKvPool
            | Refusal::UnknownPool { .. } => PIE_STATUS_UNSUPPORTED,
            // Facts about this request.
            Refusal::PageCountMismatch { .. }
            | Refusal::PageOutOfRange { .. }
            | Refusal::CellOutOfRange { .. }
            | Refusal::SlotOutOfRange { .. } => PIE_STATUS_INVALID_ARGUMENT,
        }
    }
}

impl core::fmt::Display for Refusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Refusal::NoLinearAttention => {
                write!(f, "the checkpoint has no linear-attention layers")
            }
            Refusal::ForeignDomain { src, dst } => write!(
                f,
                "only same-domain Metal-shared copies are supported; got src {src}, dst {dst}"
            ),
            Refusal::NoKvPool => write!(f, "no paged KV pool is allocated"),
            Refusal::PageCountMismatch { src, dst } => {
                write!(f, "{src} source pages against {dst} destination pages")
            }
            Refusal::PageOutOfRange {
                index,
                page,
                total_pages,
            } => write!(
                f,
                "page {page} at index {index} is outside the pool's [0, {total_pages})"
            ),
            Refusal::CellOutOfRange { index } => {
                write!(f, "cell {index} names a page or row outside the pool")
            }
            Refusal::SlotOutOfRange {
                index,
                slot,
                rs_slots,
            } => write!(f, "slot {slot} at range {index} is outside [0, {rs_slots})"),
            Refusal::UnknownPool { pool_id } => write!(f, "no elastic pool has id {pool_id}"),
        }
    }
}

/// One planned KV copy: the page pairs and the cell moves, both already checked.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct KvCopyWork {
    /// `(src, dst)` page pairs, **in request order**.
    ///
    /// The order is load-bearing and the C++ says so in a comment: a chain like
    /// `{1->0, 2->1}` reads page 1 for the second pair *after* the first has
    /// already overwritten it. Each pair is independent and the caller
    /// sequences; a true swap needs a scratch page or separate calls. The
    /// contract is carried here rather than in a comment beside a loop.
    pub pages: Vec<(u32, u32)>,
    /// The cell moves, or `None` when the request had none.
    pub cells: Option<CellMovePlan>,
    /// One past the highest page the whole request touches, for the elastic
    /// ensure — the maximum over both halves, because the pool must be grown
    /// once for the operation rather than once per half.
    pub pages_touched: u32,
}

/// Decide a `copy_kv` request whole: refuse, or return work that will not fail
/// partway.
///
/// # Errors
///
/// [`Refusal`] with the reason; nothing has been planned and nothing moved.
pub fn plan_kv_copy(
    plan: &KvCopyPlan,
    caps: Capabilities,
    grid: PoolGrid,
) -> Result<KvCopyWork, Refusal> {
    if !caps.has_linear_attn {
        return Err(Refusal::NoLinearAttention);
    }
    if plan.src_domain != PIE_MEMORY_DOMAIN_METAL_SHARED
        || plan.dst_domain != PIE_MEMORY_DOMAIN_METAL_SHARED
    {
        return Err(Refusal::ForeignDomain {
            src: plan.src_domain,
            dst: plan.dst_domain,
        });
    }
    if caps.kv_total_pages == 0 {
        return Err(Refusal::NoKvPool);
    }
    if plan.src_page_ids.len() != plan.dst_page_ids.len() {
        return Err(Refusal::PageCountMismatch {
            src: plan.src_page_ids.len(),
            dst: plan.dst_page_ids.len(),
        });
    }

    // Both halves are checked before either is built. This is the whole
    // difference from the C++: there, the pages have already moved by the time
    // a bad cell is noticed.
    let mut pages_touched = 0u32;
    let mut pages = Vec::with_capacity(plan.src_page_ids.len());
    for (index, (&src, &dst)) in plan.src_page_ids.iter().zip(&plan.dst_page_ids).enumerate() {
        for page in [src, dst] {
            if page >= caps.kv_total_pages {
                return Err(Refusal::PageOutOfRange {
                    index,
                    page,
                    total_pages: caps.kv_total_pages,
                });
            }
            pages_touched = pages_touched.max(page + 1);
        }
        pages.push((src, dst));
    }

    let cells = if plan.cells.is_empty() {
        None
    } else {
        let wire: Vec<KvMoveCell> = plan
            .cells
            .iter()
            .map(|c| KvMoveCell {
                dst_page_id: c.dst_page_id,
                dst_token_offset: c.dst_token_offset,
                src_page_id: c.src_page_id,
                src_token_offset: c.src_token_offset,
            })
            .collect();
        let moved =
            plan_cell_moves(&wire, grid).map_err(|e| Refusal::CellOutOfRange { index: e.index })?;
        pages_touched = pages_touched.max(moved.pages_touched);
        Some(moved)
    };

    Ok(KvCopyWork {
        pages,
        cells,
        pages_touched,
    })
}

/// Decide a `copy_state` request whole: refuse, or return the slot pairs.
///
/// # Errors
///
/// [`Refusal`] with the reason; no slot has been copied.
pub fn plan_state_copy(
    plan: &StateCopyPlan,
    caps: Capabilities,
) -> Result<Vec<(u32, u32)>, Refusal> {
    if !caps.has_linear_attn {
        return Err(Refusal::NoLinearAttention);
    }
    let mut pairs = Vec::with_capacity(plan.slot_ranges.len());
    for (index, range) in plan.slot_ranges.iter().enumerate() {
        for slot in [range.src_slot_id, range.dst_slot_id] {
            if slot >= caps.rs_slots {
                return Err(Refusal::SlotOutOfRange {
                    index,
                    slot,
                    rs_slots: caps.rs_slots,
                });
            }
        }
        pairs.push((range.src_slot_id, range.dst_slot_id));
    }
    Ok(pairs)
}

/// Which pool a resize names, once the id has been recognised.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    /// The paged KV pool, whose resize has its own committed-page path.
    Kv,
    /// The recurrent-state pool.
    State,
    /// The workspace pool.
    Workspace,
}

/// A decided resize.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resize {
    /// The pool named.
    pub pool: Pool,
    /// The page count asked for.
    pub target_pages: u64,
}

/// Decide a `resize_pool` request.
///
/// # Errors
///
/// [`Refusal`] with the reason; nothing has been resized.
pub fn plan_pool_resize(plan: &PoolResizePlan, caps: Capabilities) -> Result<Resize, Refusal> {
    // A match over the named ids rather than the C++'s `pool_id >
    // PIE_ELASTIC_POOL_WORKSPACE`. That comparison is correct only while the
    // ids stay contiguous from zero, which is a property of the enum's current
    // spelling rather than of the ABI, and it is the kind of thing that stays
    // correct until someone adds an id out of order.
    let pool = match plan.pool_id {
        PIE_ELASTIC_POOL_KV => Pool::Kv,
        PIE_ELASTIC_POOL_STATE => Pool::State,
        PIE_ELASTIC_POOL_WORKSPACE => Pool::Workspace,
        pool_id => return Err(Refusal::UnknownPool { pool_id }),
    };
    if !caps.has_linear_attn {
        return Err(Refusal::NoLinearAttention);
    }
    if pool == Pool::Kv && caps.kv_total_pages == 0 {
        return Err(Refusal::NoKvPool);
    }
    Ok(Resize {
        pool,
        target_pages: plan.target_pages,
    })
}

#[cfg(test)]
mod tests {
    use driver_api::local::{
        KvMoveCell, PIE_MEMORY_DOMAIN_HOST_PINNED, PIE_MEMORY_DOMAIN_METAL_PRIVATE, StateCopyRange,
    };

    use super::*;

    fn caps() -> Capabilities {
        Capabilities {
            has_linear_attn: true,
            kv_total_pages: 8,
            rs_slots: 4,
        }
    }

    fn grid() -> PoolGrid {
        PoolGrid {
            total_pages: 8,
            page_size: 16,
            row_bytes: 32,
        }
    }

    fn kv(src: Vec<u32>, dst: Vec<u32>, cells: Vec<KvMoveCell>) -> KvCopyPlan {
        KvCopyPlan {
            src_domain: PIE_MEMORY_DOMAIN_METAL_SHARED,
            dst_domain: PIE_MEMORY_DOMAIN_METAL_SHARED,
            src_page_ids: src,
            dst_page_ids: dst,
            cells,
            ..KvCopyPlan::default()
        }
    }

    fn cell(dst_page: u32, dst_row: u32, src_page: u32, src_row: u32) -> KvMoveCell {
        KvMoveCell {
            dst_page_id: dst_page,
            dst_token_offset: dst_row,
            src_page_id: src_page,
            src_token_offset: src_row,
        }
    }

    #[test]
    fn a_bad_cell_refuses_the_whole_request_so_the_pages_never_move() {
        // The defect this module exists for. The pages are all in range and the
        // C++ would have copied every one of them before reaching the cell.
        let plan = kv(vec![0, 1], vec![2, 3], vec![cell(0, 0, 99, 0)]);
        let err = plan_kv_copy(&plan, caps(), grid()).expect_err("the cell is out of range");
        assert!(matches!(err, Refusal::CellOutOfRange { index: 0 }));
        assert_eq!(err.status(), PIE_STATUS_INVALID_ARGUMENT);
    }

    #[test]
    fn a_planned_copy_carries_the_high_water_of_both_halves_not_of_either() {
        // The elastic ensure runs once for the operation, so it must be told
        // the maximum over pages AND cells. Taking either alone under-grows the
        // pool for the other.
        let plan = kv(vec![0], vec![1], vec![cell(6, 0, 0, 0)]);
        let work = plan_kv_copy(&plan, caps(), grid()).expect("in range");
        assert_eq!(
            work.pages_touched, 7,
            "the cell reaches further than the pages"
        );

        let plan = kv(vec![0], vec![5], vec![cell(1, 0, 0, 0)]);
        let work = plan_kv_copy(&plan, caps(), grid()).expect("in range");
        assert_eq!(
            work.pages_touched, 6,
            "the pages reach further than the cell"
        );
    }

    #[test]
    fn the_page_pairs_keep_request_order_because_a_chain_reads_what_the_last_pair_wrote() {
        let plan = kv(vec![1, 2], vec![0, 1], vec![]);
        let work = plan_kv_copy(&plan, caps(), grid()).expect("in range");
        assert_eq!(work.pages, vec![(1, 0), (2, 1)]);
    }

    #[test]
    fn a_page_id_outside_the_pool_names_its_index_and_the_bound() {
        let plan = kv(vec![0, 8], vec![1, 2], vec![]);
        let err = plan_kv_copy(&plan, caps(), grid()).expect_err("page 8 is out of range");
        assert_eq!(
            err,
            Refusal::PageOutOfRange {
                index: 1,
                page: 8,
                total_pages: 8
            }
        );
    }

    #[test]
    fn mismatched_page_arrays_refuse_rather_than_pair_the_shorter_prefix() {
        let plan = kv(vec![0, 1, 2], vec![3], vec![]);
        let err = plan_kv_copy(&plan, caps(), grid()).expect_err("no pairing exists");
        assert_eq!(err, Refusal::PageCountMismatch { src: 3, dst: 1 });
    }

    #[test]
    fn a_cross_domain_copy_is_unsupported_rather_than_read_as_same_domain() {
        for (src, dst) in [
            (
                PIE_MEMORY_DOMAIN_HOST_PINNED,
                PIE_MEMORY_DOMAIN_METAL_SHARED,
            ),
            (
                PIE_MEMORY_DOMAIN_METAL_SHARED,
                PIE_MEMORY_DOMAIN_METAL_PRIVATE,
            ),
        ] {
            let plan = KvCopyPlan {
                src_domain: src,
                dst_domain: dst,
                ..KvCopyPlan::default()
            };
            let err = plan_kv_copy(&plan, caps(), grid()).expect_err("foreign domain");
            assert!(matches!(err, Refusal::ForeignDomain { .. }));
            assert_eq!(err.status(), PIE_STATUS_UNSUPPORTED);
        }
    }

    #[test]
    fn a_build_fact_is_unsupported_and_a_request_fact_is_invalid_argument() {
        // The distinction the engine acts on: stop asking, or ask differently.
        let no_gdn = Capabilities {
            has_linear_attn: false,
            ..caps()
        };
        assert_eq!(
            plan_kv_copy(&kv(vec![], vec![], vec![]), no_gdn, grid())
                .expect_err("no gdn")
                .status(),
            PIE_STATUS_UNSUPPORTED
        );
        let no_pool = Capabilities {
            kv_total_pages: 0,
            ..caps()
        };
        assert_eq!(
            plan_kv_copy(&kv(vec![], vec![], vec![]), no_pool, grid())
                .expect_err("no pool")
                .status(),
            PIE_STATUS_UNSUPPORTED
        );
        assert_eq!(
            plan_kv_copy(&kv(vec![0], vec![], vec![]), caps(), grid())
                .expect_err("mismatch")
                .status(),
            PIE_STATUS_INVALID_ARGUMENT
        );
    }

    #[test]
    fn an_empty_kv_request_is_allowed_and_plans_nothing() {
        let work = plan_kv_copy(&kv(vec![], vec![], vec![]), caps(), grid()).expect("legal");
        assert!(work.pages.is_empty());
        assert!(work.cells.is_none());
        assert_eq!(work.pages_touched, 0, "an empty request grows nothing");
    }

    fn range(src: u32, dst: u32) -> StateCopyRange {
        StateCopyRange {
            src_slot_id: src,
            dst_slot_id: dst,
            ..StateCopyRange::default()
        }
    }

    #[test]
    fn a_state_copy_refuses_before_any_slot_moves_when_a_later_range_is_wild() {
        // The C++ validates every range up front too — this pins that it stays
        // that way, and that the refusal names WHICH range rather than only
        // that one was bad.
        let plan = StateCopyPlan {
            slot_ranges: vec![range(0, 1), range(1, 2), range(2, 9)],
        };
        let err = plan_state_copy(&plan, caps()).expect_err("slot 9 is wild");
        assert_eq!(
            err,
            Refusal::SlotOutOfRange {
                index: 2,
                slot: 9,
                rs_slots: 4
            }
        );
    }

    #[test]
    fn a_state_copy_without_linear_attention_has_no_state_to_copy() {
        let no_gdn = Capabilities {
            has_linear_attn: false,
            ..caps()
        };
        let plan = StateCopyPlan {
            slot_ranges: vec![range(0, 1)],
        };
        assert_eq!(
            plan_state_copy(&plan, no_gdn),
            Err(Refusal::NoLinearAttention)
        );
    }

    #[test]
    fn every_named_pool_resolves_and_an_unnamed_id_refuses_by_name() {
        for (id, want) in [
            (PIE_ELASTIC_POOL_KV, Pool::Kv),
            (PIE_ELASTIC_POOL_STATE, Pool::State),
            (PIE_ELASTIC_POOL_WORKSPACE, Pool::Workspace),
        ] {
            let plan = PoolResizePlan {
                pool_id: id,
                target_pages: 32,
                ..PoolResizePlan::default()
            };
            let resize = plan_pool_resize(&plan, caps()).expect("named pool");
            assert_eq!(resize.pool, want);
            assert_eq!(resize.target_pages, 32);
        }
        let plan = PoolResizePlan {
            pool_id: 3,
            ..PoolResizePlan::default()
        };
        assert_eq!(
            plan_pool_resize(&plan, caps()),
            Err(Refusal::UnknownPool { pool_id: 3 })
        );
    }

    #[test]
    fn only_the_kv_pool_resize_needs_a_kv_pool() {
        let no_pool = Capabilities {
            kv_total_pages: 0,
            ..caps()
        };
        let kv_plan = PoolResizePlan {
            pool_id: PIE_ELASTIC_POOL_KV,
            ..PoolResizePlan::default()
        };
        assert_eq!(plan_pool_resize(&kv_plan, no_pool), Err(Refusal::NoKvPool));

        let ws_plan = PoolResizePlan {
            pool_id: PIE_ELASTIC_POOL_WORKSPACE,
            ..PoolResizePlan::default()
        };
        assert!(
            plan_pool_resize(&ws_plan, no_pool).is_ok(),
            "the workspace pool does not depend on the KV pool existing"
        );
    }
}
