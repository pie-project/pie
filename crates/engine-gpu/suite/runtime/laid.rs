//! Every buffer a fire can need is laid at load: a fire at the top rung
//! whose recurrent lanes replay as many buffered rows as the rung's tokens
//! ahead of its own (the widest fire the runtime lets one carry,
//! `engine::fire::rung_rows`) reserves nothing past what the load did, and
//! one whose extended rows are past that is refused by name, not grown
//! for. qwen35-tiny (recurrent layers), seat-addressed buffering lanes, its
//! top rung cut to [`TOP`] tokens so a lane's buffer holds what each
//! lane replays and lands.
//!
//! The whole device ledger is read: the shell's buffers (the recurrent
//! scratch, the readout, the snapshot table, the inputs with the group
//! masks) and a backend's own for its frames (wgpu's uniforms and staging
//! scratch) are all laid at load.

use engine::fire::{FoldLen, RsReset, RsVerb};

use crate::backend::{Lane, Seated, Shell};
use crate::model_logits::{CONTEXT, SEATS, fixture, load_fires, word};

const MODEL: &str = "qwen35-tiny";
/// The top rung's tokens: a lane of `SEATS` replays and lands three
/// fires' rows of `TOP / SEATS` in its seat's buffer pages.
const TOP: u32 = 32;

/// The bytes the device's ledger holds, every owner's: a fire that
/// allocates anything moves it.
fn laid(shell: &Shell) -> u64 {
    shell.device().books().reserved()
}

/// The tokens of a lane's `rows` rows from `held` on.
fn tokens(held: u32, rows: u32) -> Vec<u32> {
    (held..held + rows).map(|at| (at * 37 + 11) % 256).collect()
}

/// A fire of `SEATS` buffering lanes of `rows` rows each, every lane's
/// rows from `held` on and its state holding none of them (its buffer and
/// its fold from slot 0, folding nothing): each replays its `held`
/// buffered rows ahead of its own.
fn buffering(
    shell: &mut Shell,
    trace: &poem_ir::Trace,
    held: u32,
    rows: u32,
) -> engine_gpu::Result<()> {
    let pages: Vec<u32> = vec![0; CONTEXT.div_ceil(16) as usize];
    let verb = RsVerb::Buffer {
        pages,
        base: 0,
        folded: 0,
        fold: FoldLen::Host(0),
    };
    let fed: Vec<Vec<u32>> = (0..SEATS).map(|_| tokens(held, rows)).collect();
    let seated: Vec<Seated<'_>> = (0..SEATS)
        .map(|lane| Seated {
            held: Some(held),
            rs: &verb,
            rs_reset: if held == 0 {
                RsReset::Fresh
            } else {
                RsReset::Held
            },
            ..Seated::of(Lane {
                slot: lane,
                word: word(trace, rows),
                tokens: &fed[lane as usize],
            })
        })
        .collect();
    shell.fire_seated(&seated).map(|_| ())
}

/// The top rung's tokens, as the pool states its rungs.
fn top(shell: &Shell) -> u32 {
    shell
        .pool_chunk_facts()
        .act_chunks
        .last()
        .map(|&(tokens, _)| tokens)
        .expect("the load states its rungs")
}

#[test]
fn a_top_rung_fire_replaying_its_rungs_tokens_reserves_nothing_past_the_load() {
    let fixture = fixture(MODEL, None);
    let Some(mut shell) = load_fires(&fixture, CONTEXT, TOP) else {
        return;
    };
    let top = top(&shell);
    let rows = top / SEATS;
    assert_eq!(top, TOP, "the load's top rung is the budget's");
    let reserved = laid(&shell);
    // The first fire buffers `rows` rows a lane; the second, at the top
    // rung, replays them ahead of its own: `2 * top` extended rows.
    buffering(&mut shell, &fixture.trace, 0, rows).expect("the first buffering fire lands");
    buffering(&mut shell, &fixture.trace, rows, rows)
        .expect("a top-rung fire replaying the rung's tokens lands in the scratch laid at load");
    assert_eq!(
        laid(&shell),
        reserved,
        "a fire grew a buffer the load laid: {}",
        shell.device().books().ledger()
    );
}

#[test]
fn a_fire_past_the_recurrent_scratch_laid_is_refused_by_name() {
    let fixture = fixture(MODEL, None);
    let Some(mut shell) = load_fires(&fixture, CONTEXT, TOP) else {
        return;
    };
    let top = top(&shell);
    let rows = top / SEATS;
    buffering(&mut shell, &fixture.trace, 0, rows).expect("the first buffering fire lands");
    buffering(&mut shell, &fixture.trace, rows, rows).expect("the widest fire lands");
    let reserved = laid(&shell);
    // Each lane now replays twice its rows: `3 * top` extended rows, past
    // the widest rung's.
    let why = buffering(&mut shell, &fixture.trace, 2 * rows, rows)
        .expect_err("a fire past the recurrent scratch laid at load is refused")
        .to_string();
    assert!(
        why.contains("a fire's recurrent scratch of") && why.contains("laid at load"),
        "refused for another reason: {why}"
    );
    assert_eq!(laid(&shell), reserved, "the refused fire grew a buffer");
}

/// A top-rung fire whose every lane lands its state at the most interior
/// rows a fire lands (`kernels_slang::ssm::RS_SNAPSHOTS`) writes its
/// snapshot table into the one laid at load: it reserves nothing.
#[test]
fn a_top_rung_fire_snapshotting_every_lane_reserves_nothing_past_the_load() {
    let fixture = fixture(MODEL, None);
    let Some(mut shell) = load_fires(&fixture, CONTEXT, TOP) else {
        return;
    };
    let rows = top(&shell) / SEATS;
    let most = kernels_slang::ssm::RS_SNAPSHOTS;
    assert!(rows > most, "a lane has interior rows to snapshot");
    let fed: Vec<Vec<u32>> = (0..SEATS).map(|_| tokens(0, rows)).collect();
    let snaps: Vec<Vec<engine::fire::RsSnapshot>> = (0..SEATS)
        .map(|lane| {
            (1..=most)
                .map(|n| engine::fire::RsSnapshot {
                    row: n,
                    slot: lane,
                    pieces: Vec::new(),
                    fixed: Vec::new(),
                })
                .collect()
        })
        .collect();
    let reserved = laid(&shell);
    let seated: Vec<Seated<'_>> = (0..SEATS)
        .map(|lane| Seated {
            held: Some(0),
            rs_reset: RsReset::Fresh,
            rs_snapshots: &snaps[lane as usize],
            ..Seated::of(Lane {
                slot: lane,
                word: word(&fixture.trace, rows),
                tokens: &fed[lane as usize],
            })
        })
        .collect();
    shell
        .fire_seated(&seated)
        .expect("a top-rung fire snapshotting every lane lands");
    assert_eq!(
        laid(&shell),
        reserved,
        "a fire grew a buffer the load laid: {}",
        shell.device().books().ledger()
    );
}
