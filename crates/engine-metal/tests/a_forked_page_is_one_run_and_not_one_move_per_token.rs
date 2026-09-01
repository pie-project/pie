//! **`copy_kv`'s plan, on the metal plane, without a device.**
//!
//! Every claim below is arithmetic over two integer lists, which is exactly
//! what `engine_metal::store::Move::plan` is — so the whole file runs on any
//! target and needs no Metal, no checkpoint and no weights. What it settles is
//! the half of a fork that decides WHICH cells move and in how many copies;
//! the half that moves bytes is `Pools::copy_kv`, needs a device, and is named
//! in the verify queue.
//!
//! # The submission these tests are about
//!
//! A prefix-tree fork. Two sequences share a prefix, the child is given fresh
//! page ids for the pages it will write, and the boundary page — the one the
//! parent has half filled — has its live tokens copied out so the child can
//! append past them without writing into the parent's cells:
//!
//! ```text
//! parent   [ p0 full ][ p1 full ][ p2 : 5 of 8 live ]
//! child    [ p0      ][ p1      ][ p9 : 5 copied    ] then appends at 5
//!            shared      shared    src_page_ids/dst_page_ids is neither of
//!                                  these — it is p2 -> p9's five cells
//! ```
//!
//! The runtime states that tail one `KvMove` PER TOKEN
//! (`runtime::pipeline::fire`'s `copy_into` builds exactly that vector), and
//! the thing this file pins is that five cells cost ONE copy per plane rather
//! than five: at eighteen layers and two planes that is 36 blits instead of
//! 180, for the same bytes, per fork.
//!
//! # VERIFY QUEUE — what this file does NOT settle
//!
//! **That the bytes arrive.** `Pools::copy_kv` walks `rows × planes` and
//! encodes one `copyFromBuffer:` per run per plane into a command buffer of
//! its own on the fire queue; whether a fork's continuation then reads its own
//! keys at every layer is a device claim, and it needs a checkpoint, a fire
//! against the parent, a `copy_kv`, and a fire against the child whose logits
//! are diffed against the parent's own continuation. It is banked as session G
//! of `.wiki/alto/metal-verify-queue.md`.

use engine::transfer::{KvCopy, KvMove};
use engine_metal::store::Move;

/// The page size every test below is written at — the one this shell's
/// smoke deployments page at, so the numbers read like a real fork's.
const PAGE: u32 = 8;

/// This shell's own domain, which is the only pair `copy_kv` serves. The
/// domains are not what `Move::plan` reads — the verb checks them before it
/// builds anything — so they are stated once, here, and never varied.
fn copy(src_pages: &[u32], dst_pages: &[u32], moves: Vec<KvMove>) -> KvCopy {
    KvCopy {
        src: engine::transfer::MemoryDomain::MetalShared,
        dst: engine::transfer::MemoryDomain::MetalShared,
        src_page_ids: src_pages.to_vec(),
        dst_page_ids: dst_pages.to_vec(),
        moves,
    }
}

/// One `KvMove` per token of a run — the vector the runtime actually builds.
fn tail(src_page: u32, dst_page: u32, from: u32, tokens: u32) -> Vec<KvMove> {
    (0..tokens)
        .map(|at| KvMove {
            src_page_id: src_page,
            src_token_offset: from + at,
            dst_page_id: dst_page,
            dst_token_offset: from + at,
        })
        .collect()
}

/// **A whole-page graft is one move per pair, at the page's own length.**
///
/// The contract hands the whole-page half as two parallel lists and says
/// nothing about how much of each page is live — that is the runtime's
/// bookkeeping — so the whole page moves, both ends at offset zero.
#[test]
fn a_page_pair_is_one_move_over_the_whole_page() {
    let plan = Move::plan(&copy(&[3, 4], &[9, 10], Vec::new()), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![
            Move { src_page: 3, src_token: 0, dst_page: 9, dst_token: 0, tokens: PAGE },
            Move { src_page: 4, src_token: 0, dst_page: 10, dst_token: 0, tokens: PAGE },
        ]
    );
}

/// **THE CLAIM THIS FILE EXISTS FOR.** Five consecutive cells stated one per
/// token are one run, and a run is one blit per plane.
#[test]
fn a_forked_tail_is_one_run_and_not_one_move_per_token() {
    let plan = Move::plan(&copy(&[], &[], tail(2, 9, 0, 5)), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![Move { src_page: 2, src_token: 0, dst_page: 9, dst_token: 0, tokens: 5 }]
    );
}

/// **The merge is exact, so cells that do not continue a run start one.**
///
/// A caller whose moves do not form runs gets the same bytes at more copies —
/// never different bytes, and never a run that spans the gap.
#[test]
fn a_gap_in_the_cells_cuts_the_run() {
    let cells = vec![
        KvMove { src_page_id: 2, src_token_offset: 0, dst_page_id: 9, dst_token_offset: 0 },
        KvMove { src_page_id: 2, src_token_offset: 1, dst_page_id: 9, dst_token_offset: 1 },
        // the gap: token 2 is not copied
        KvMove { src_page_id: 2, src_token_offset: 3, dst_page_id: 9, dst_token_offset: 3 },
    ];
    let plan = Move::plan(&copy(&[], &[], cells), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![
            Move { src_page: 2, src_token: 0, dst_page: 9, dst_token: 0, tokens: 2 },
            Move { src_page: 2, src_token: 3, dst_page: 9, dst_token: 3, tokens: 1 },
        ]
    );
}

/// **A run stops at the page's own edge**, because the offsets it merges are
/// offsets INSIDE a page: two pages' worth of cells are two runs however
/// consecutive the token numbers look.
#[test]
fn a_run_never_walks_off_the_end_of_its_page() {
    let mut cells = tail(2, 9, PAGE - 2, 2);
    cells.extend(tail(3, 10, 0, 2));
    let plan = Move::plan(&copy(&[], &[], cells), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![
            Move { src_page: 2, src_token: PAGE - 2, dst_page: 9, dst_token: PAGE - 2, tokens: 2 },
            Move { src_page: 3, src_token: 0, dst_page: 10, dst_token: 0, tokens: 2 },
        ]
    );
}

/// **A cell that names one place twice is not a move.** A caller listing a
/// fork's whole tail states the shared cells too, and it is asking for
/// nothing — dropped rather than refused, and dropped rather than encoded as
/// a self-copy the blit would call an overlap.
#[test]
fn a_cell_that_moves_nowhere_is_dropped() {
    let cells = vec![
        KvMove { src_page_id: 2, src_token_offset: 0, dst_page_id: 2, dst_token_offset: 0 },
        KvMove { src_page_id: 2, src_token_offset: 1, dst_page_id: 9, dst_token_offset: 1 },
    ];
    let plan = Move::plan(&copy(&[], &[], cells), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![Move { src_page: 2, src_token: 1, dst_page: 9, dst_token: 1, tokens: 1 }]
    );
}

/// **An offset past the page is refused, and the sentence names both
/// numbers** — the caller's statement is wrong, and no retry of it helps.
#[test]
fn an_offset_past_the_page_is_refused_by_name() {
    let cells = vec![KvMove {
        src_page_id: 2,
        src_token_offset: PAGE,
        dst_page_id: 9,
        dst_token_offset: 0,
    }];
    let why = Move::plan(&copy(&[], &[], cells), PAGE).expect_err("past the page");
    assert!(why.contains("token offsets"), "{why}");
    assert!(why.contains(&PAGE.to_string()), "{why}");
}

/// **A run whose two ends overlap is refused rather than shifted.**
///
/// Both ends live in one reservation, so a blit whose regions overlap is
/// undefined — silently. A caller that means "shift a page's tokens" states a
/// staging page and two moves.
#[test]
fn a_move_whose_ends_overlap_is_refused_by_name() {
    let why = Move::plan(&copy(&[], &[], tail(2, 2, 0, 3).into_iter().map(|cell| KvMove {
        dst_token_offset: cell.src_token_offset + 1,
        ..cell
    }).collect()), PAGE)
    .expect_err("overlapping ends");
    assert!(why.contains("overlap"), "{why}");
}

/// **Two ends of one page that do NOT overlap are a legal move**, which is
/// what keeps the refusal above a statement about regions rather than about
/// pages: a page's first three cells copied to its last three is one blit.
#[test]
fn one_page_is_not_by_itself_an_overlap() {
    let cells: Vec<KvMove> = (0..3)
        .map(|at| KvMove {
            src_page_id: 2,
            src_token_offset: at,
            dst_page_id: 2,
            dst_token_offset: PAGE - 3 + at,
        })
        .collect();
    let plan = Move::plan(&copy(&[], &[], cells), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![Move { src_page: 2, src_token: 0, dst_page: 2, dst_token: PAGE - 3, tokens: 3 }]
    );
}

/// **Page lists that are not parallel are refused** — the contract's own
/// clause (`KvCopy::validate`), restated where the arithmetic is so that a
/// `zip` never grafts a fork's pages half over.
#[test]
fn page_lists_that_are_not_parallel_are_refused() {
    let why = Move::plan(&copy(&[3, 4], &[9], Vec::new()), PAGE).expect_err("unequal lists");
    assert!(why.contains("src_page_ids"), "{why}");
}

/// **The two spellings compose**, because they flatten to one shape: a
/// fork that grafts two whole pages and copies a partial tail is three runs,
/// in the order the submission stated them.
#[test]
fn the_two_spellings_are_one_list_of_runs() {
    let plan = Move::plan(&copy(&[0, 1], &[7, 8], tail(2, 9, 0, 5)), PAGE).expect("the plan");
    assert_eq!(
        plan,
        vec![
            Move { src_page: 0, src_token: 0, dst_page: 7, dst_token: 0, tokens: PAGE },
            Move { src_page: 1, src_token: 0, dst_page: 8, dst_token: 0, tokens: PAGE },
            Move { src_page: 2, src_token: 0, dst_page: 9, dst_token: 0, tokens: 5 },
        ]
    );
}
