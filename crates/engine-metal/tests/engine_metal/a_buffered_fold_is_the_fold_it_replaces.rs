#![cfg(target_vendor = "apple")]

use std::path::PathBuf;

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;

const WINDOW: usize = 6;

/// The discard rounds of claim 9.
const ROUNDS: u32 = 8;

const PAGE: u32 = 16;

fn fixture() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_QWEN4_FIXTURE") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4-micro");
    path.is_dir().then_some(path)
}

/// The micro model of the qwen_4 package: its widths are toys, its layers
/// the real model's kinds.
const MICRO: &str = "qwen38-flash-next-micro";

fn package() -> &'static poem::star::Package {
    poem_compiler::catalog::package_of(MICRO).expect("the qwen_4 package holds its micro model")
}

fn deploy() -> poem::star::Deploy {
    poem::star::Deploy {
        weights: vec![Dtype::Bf16.into()],
        kv: Dtype::Bf16,
        tp: 1,
        ep: 1,
        parts: vec![],
        drafter: None,
        yarn: None,
        tap: None,
        layers: None,
        experts: None,
    }
}

fn traced() -> poem::Trace {
    package()
        .trace(MICRO, &deploy(), "qwen4-micro", Platform::Metal)
        .unwrap_or_else(|why| panic!("the micro model traces: {why:#}"))
}

thread_local! {
    /// The trace the running test's shell serves, and its tokens a buffer
    /// page: each test states its own model.
    static TRACE: std::cell::RefCell<Option<poem::Trace>> = const { std::cell::RefCell::new(None) };
    static PER_PAGE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

fn word(query_len: u32) -> u64 {
    TRACE.with(|trace| {
        trace
            .borrow()
            .as_ref()
            .expect("the test states its trace")
            .facts
            .word(&Request::new(query_len, false))
    })
}

/// States `shell`'s model for the helpers below.
fn serving(shell: Shell, trace: poem::Trace) -> Shell {
    eprintln!(
        "recurrent slot: {} bytes; buffered page slab: {:.1} KiB; serves rs verbs: {}",
        shell.state_slot_bytes(),
        shell.buffer_bytes() as f64 / 1024.0,
        shell.serves_rs_verbs()
    );
    assert!(shell.state_slot_bytes() > 0, "a recurrent plan has recurrent state");
    assert!(
        shell.buffer_bytes() > 0,
        "the plan reserved no buffered page slab, so nothing below can be true"
    );
    assert!(shell.serves_rs_verbs());
    PER_PAGE.with(|per| per.set(shell.pool_layout().rs_tokens.max(1)));
    TRACE.with(|slot| *slot.borrow_mut() = Some(trace));
    shell
}

fn ready() -> Option<Shell> {
    if !engine_metal::device::present() {
        eprintln!("skipping the buffered fold gate: no Metal device");
        return None;
    }
    let Some(fixture) = fixture() else {
        eprintln!(
            "skipping the buffered fold gate: no qwen4 fixture (crates/engine-metal/tests/fixtures/qwen4-micro or PIE_QWEN4_FIXTURE)"
        );
        return None;
    };
    let trace = traced();
    let source =
        ztensor_compat::index(fixture.join("model.safetensors")).expect("the fixture opens");
    let contract = package()
        .import(MICRO, &deploy(), &source, Platform::Metal)
        .expect("the micro text's import fits the fixture it was generated for");
    drop(source);
    let shell = load(trace.clone(), &contract, &fixture);
    Some(serving(shell, trace))
}

/// A shell of `trace` over `checkpoint`: eight seats of 128 tokens.
fn load(trace: poem::Trace, contract: &checkpoint::contract::ModelContract, checkpoint: &std::path::Path) -> Shell {
    Shell::load(Boot {
        kv_budget: crate::full_pool(&trace, PAGE, 128, 8),
        voxels: None,
        trace,
        contract,
        checkpoint,
        budget: Budget::new(4, 64),
        patches: None,
        profile: None,
        page_size: PAGE,
        context: 128,
        slots: 8,
        runahead: engine::runahead::Runahead::F1,
        allowance: engine::Allowance::WHOLE,
        profiles: None,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .and_then(|mut shell| shell.raise_whole().map(|()| shell))
    .expect("the micro shell loads")
}

/// A Qwen3.5 shell (`qwen35-tiny`: gated-delta layers and attention, no
/// PLE) over its own synthetic checkpoint.
fn ready_gdn() -> Option<Shell> {
    if !engine_metal::device::present() {
        eprintln!("skipping the buffered fold gate: no Metal device");
        return None;
    }
    let fixture: &'static _ = Box::leak(Box::new(
        super::a_scattered_pool_serves_the_same_logits::fixture_of("qwen35-tiny", None),
    ));
    let shell = load(fixture.trace.clone(), &fixture.contract, &fixture.container);
    Some(serving(shell, fixture.trace.clone()))
}

fn window(seed: u32) -> Vec<u32> {
    (0..WINDOW as u32)
        .map(|at| (seed + at * 37) % 256)
        .collect()
}

/// The buffer pages a lane names for a verb holding `tokens` past its base.
/// A lane fired directly names them from its seat's run on (one page a
/// seat), so a list runs into the seats above it: as few as the verb needs.
fn pages(tokens: u32) -> Vec<u32> {
    (0..tokens.div_ceil(PER_PAGE.with(std::cell::Cell::get).max(1))).collect()
}

/// A buffering verb whose state holds the slots below `folded`: a fire whose
/// rows land at slot `first` replays the buffered slots `folded..first`, then
/// folds `fold` of them.
fn buffer(folded: u32, fold: u32) -> RsVerb {
    buffer_over(2 * WINDOW as u32, folded, fold)
}

/// [`buffer`] over pages holding `tokens` past the base.
fn buffer_over(tokens: u32, folded: u32, fold: u32) -> RsVerb {
    RsVerb::Buffer {
        pages: pages(tokens),
        base: 0,
        folded,
        fold: FoldLen::Host(fold),
    }
}

fn seated<'a>(slot: u32, tokens: &'a [u32], rs: &'a RsVerb, reset: RsReset) -> Seated<'a> {
    Seated {
        rs,
        rs_reset: reset,
        ..Seated::of(Lane {
            slot,
            word: word(tokens.len() as u32),
            tokens,
        })
    }
}

fn seated_at<'a>(
    slot: u32,
    tokens: &'a [u32],
    rs: &'a RsVerb,
    reset: RsReset,
    held: u32,
) -> Seated<'a> {
    Seated {
        held: Some(held),
        ..seated(slot, tokens, rs, reset)
    }
}

fn spread(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn argmax(logits: &[f32]) -> usize {
    (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap_or(0)
}

const LOGIT_FLOOR: f32 = 0.05;

/// Whether two states' bytes are the same state: every word equal, but a
/// normal f32 within rounding of the other's at the state's own magnitude
/// (a fold and the replay it stands for sum a scan in different chunks), a
/// few ulps of its largest element. A word below the normal range (an
/// integer row's small values, an n-gram id, read as a denormal) is compared
/// exactly, so a leaked row of ids is never let through; a leaked float row
/// moves its values by far more than rounding.
fn same_state(a: &[u8], b: &[u8]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("{} bytes against {}", a.len(), b.len()));
    }
    let words = |bytes: &[u8]| -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    };
    let (x, y) = (words(a), words(b));
    let normal = |w: u32| (w >> 23) & 0xff != 0 && (w >> 23) & 0xff != 0xff;
    let largest = x
        .iter()
        .filter(|&&w| normal(w))
        .map(|&w| f32::from_bits(w).abs())
        .fold(0f32, f32::max);
    let tolerance = largest * f32::EPSILON * 16.0;
    let mut apart = 0usize;
    let mut worst = 0f32;
    let mut first = None;
    for (at, (&p, &q)) in x.iter().zip(&y).enumerate() {
        if p == q {
            continue;
        }
        let gap = (f32::from_bits(p) - f32::from_bits(q)).abs();
        if normal(p) && normal(q) && gap <= tolerance {
            worst = worst.max(gap);
            continue;
        }
        apart += 1;
        first.get_or_insert(at);
    }
    if worst > 0.0 {
        eprintln!("states within rounding: {worst:e} at most, against {tolerance:e}");
    }
    match first {
        None => Ok(()),
        Some(at) => Err(format!(
            "{apart} of {} words apart past rounding ({tolerance:e}), the first at byte {}",
            x.len(),
            at * 4
        )),
    }
}

fn assert_same_state(got: &[u8], want: &[u8], what: &str) {
    if let Err(why) = same_state(got, want) {
        panic!("{what}: {why}");
    }
}

#[test]
fn a_buffered_fold_is_the_fold_it_replaces() {
    if let Some(shell) = ready() {
        holds_the_contract(shell);
    }
}

/// The same contract over a gated-delta model with no PLE (Qwen3.5): its
/// conv and scan rows replay as the fold they defer.
#[test]
fn a_gdn_buffered_fold_is_the_fold_it_replaces() {
    if let Some(shell) = ready_gdn() {
        holds_the_contract(shell);
    }
}

/// A buffering fire leaves the state as it was; a fold replaying it is the
/// fold it deferred, at any length; a folding lane beside a buffering one
/// answers as alone; and rounds of a two-row window with a rejected row
/// leave the banks a plain fold of the survivors leaves.
fn holds_the_contract(mut shell: Shell) {
    let fold = RsVerb::Fold;
    let tokens = window(11);
    let k = WINDOW as u32;

    shell.open(0).expect("slot 0 opens");
    let zeroed = shell.state_bytes(0).expect("slot 0 reads back");
    assert!(
        zeroed.iter().all(|byte| *byte == 0),
        "an opened slot's banks are zero"
    );
    let reference = shell
        .fire_seated(&[seated(0, &tokens, &fold, RsReset::Fresh)])
        .expect("the folding fire runs");
    let folded = shell.state_bytes(0).expect("slot 0 reads back");
    assert_ne!(
        folded, zeroed,
        "a fold over {WINDOW} tokens left the banks untouched"
    );

    shell.open(1).expect("slot 1 opens");
    let buffered = shell
        .fire_seated(&[seated(1, &tokens, &buffer(0, 0), RsReset::Fresh)])
        .expect("the buffering fire runs");
    assert_eq!(
        shell.state_bytes(1).expect("slot 1"),
        zeroed,
        "a `RsVerb::Buffer` fire perturbed the folded state"
    );
    let d = spread(&reference[0], &buffered[0]);
    eprintln!("claim 1: buffered window vs folding window, last-row logit spread {d:.4}");
    assert!(
        d <= LOGIT_FLOOR,
        "a buffered fire answers other logits than the fold it defers: {d}"
    );

    let second = window(101);
    let j = 3u32;
    shell.open(2).expect("slot 2 opens");
    shell
        .fire_seated(&[seated(2, &tokens[..j as usize], &fold, RsReset::Fresh)])
        .expect("the short folding fire runs");
    let folded_j = shell.state_bytes(2).expect("slot 2");
    assert_ne!(
        folded_j, folded,
        "folding {j} tokens and folding {k} leave the same banks, so claim 3 can't tell them apart"
    );
    shell.open(3).expect("slot 3 opens");
    shell
        .fire_seated(&[seated(3, &tokens, &buffer(0, 0), RsReset::Fresh)])
        .expect("the buffering fire runs");
    shell
        .fire_seated(&[seated_at(3, &second, &buffer(0, j), RsReset::Held, k)])
        .expect("window B replays A and folds part of it");
    assert_same_state(
        &shell.state_bytes(3).expect("slot 3"),
        &folded_j,
        &format!("a replay truncated at {j} did not leave the banks a fold of {j} leaves"),
    );

    shell.open(4).expect("slot 4 opens");
    let mixed = shell
        .fire_seated(&[seated(4, &tokens, &buffer(0, j), RsReset::Fresh)])
        .expect("the mixed fire runs");
    assert_same_state(
        &shell.state_bytes(4).expect("slot 4"),
        &folded_j,
        &format!("the mixed row did not land the fold at row {j}"),
    );
    let d = spread(&reference[0], &mixed[0]);
    eprintln!("claim 4: mixed row vs folding window, last-row logit spread {d:.4}");
    assert!(
        d <= LOGIT_FLOOR,
        "the mixed row answers other logits than the fold: {d}"
    );

    shell.open(5).expect("slot 5 opens");
    shell
        .fire_seated(&[seated(5, &tokens, &buffer(0, 0), RsReset::Fresh)])
        .expect("window A buffers");
    let round = shell
        .fire_seated(&[seated_at(5, &second, &buffer(0, k), RsReset::Held, k)])
        .expect("window B replays A and buffers itself");
    assert_same_state(
        &shell.state_bytes(5).expect("slot 5"),
        &folded,
        "replaying A ahead of B did not leave the banks Fold(A) leaves",
    );
    shell.open(6).expect("slot 6 opens");
    shell
        .fire_seated(&[seated(6, &tokens, &fold, RsReset::Fresh)])
        .expect("A folds plainly");
    let plain = shell
        .fire_seated(&[seated_at(6, &second, &fold, RsReset::Held, k)])
        .expect("B folds after it");
    let d = spread(&plain[0], &round[0]);
    eprintln!(
        "claim 5: B after replay(A) vs B after fold(A), last-row logit spread {d:.4}; argmax {} vs {}",
        argmax(&round[0]),
        argmax(&plain[0])
    );
    assert!(
        d <= LOGIT_FLOOR,
        "the read path answers other logits than a plain fold of the prefix: {d}"
    );

    let third = window(203);
    shell.open(7).expect("slot 7 opens");
    let alone = shell
        .fire_seated(&[seated(7, &third, &fold, RsReset::Fresh)])
        .expect("the lone fold runs");
    let alone_state = shell.state_bytes(7).expect("slot 7");
    shell.open(7).expect("slot 7 reopens");
    shell.open(1).expect("slot 1 reopens");
    let beside = shell
        .fire_seated(&[
            seated(7, &third, &fold, RsReset::Fresh),
            seated(1, &tokens, &buffer(0, 0), RsReset::Fresh),
        ])
        .expect("the mixed fire runs");
    assert_same_state(
        &shell.state_bytes(7).expect("slot 7"),
        &alone_state,
        "a folding lane's banks changed for having a buffering peer",
    );
    assert_eq!(
        shell.state_bytes(1).expect("slot 1"),
        zeroed,
        "the buffering peer folded"
    );
    let d = spread(&alone[0], &beside[0]);
    eprintln!("claim 6: folding lane beside a buffering peer, logit spread {d:.4}");
    assert!(
        d <= LOGIT_FLOOR,
        "a folding lane answers differently beside a buffering peer: {d}"
    );

    let steps: Vec<u32> = (0..12u32).map(|i| (503 + i * 37) % 256).collect();
    let junk = 7u32;
    // The buffering lane is slot 0: a lane fired here directly names its
    // buffer pages from its seat's run on (one page a seat), so the rounds'
    // pages, past the window and every survivor, run into the seats above it
    // (lanes that only fold) and must stay inside the load's seats.
    shell.open(0).expect("slot 0 reopens");
    shell
        .fire_seated(&[seated(0, &tokens, &fold, RsReset::Fresh)])
        .expect("A folds");
    shell.open(6).expect("slot 6 reopens");
    shell
        .fire_seated(&[seated(6, &tokens, &fold, RsReset::Fresh)])
        .expect("A folds for the reference");
    let rounds = ROUNDS;
    for r in 0..rounds {
        let rows = [steps[r as usize], junk];
        // A holds slots 0..k folded; round r's window lands at slot k + r,
        // past the survivor s_{r-1} the round before left at k + r - 1.
        // The window and every survivor: the last round reaches slot k + 8.
        let held = k + ROUNDS + 1;
        let verb = if r == 0 {
            buffer_over(held, k, 0)
        } else {
            buffer_over(held, k + r - 1, 1)
        };
        let win = shell
            .fire_seated(&[seated_at(0, &rows, &verb, RsReset::Held, k + r)])
            .expect("a two-row window with a rejected row fires");
        let mut prefix: Vec<u32> = tokens.clone();
        prefix.extend_from_slice(&steps[..=r as usize]);
        shell.open(7).expect("slot 7 reopens");
        shell
            .fire_seated(&[seated(7, &prefix, &fold, RsReset::Fresh)])
            .expect("the prefix folds");
        let ref1 = shell
            .fire_seated(&[seated_at(7, &[junk], &fold, RsReset::Held, prefix.len() as u32)])
            .expect("junk folds alone");
        shell.open(3).expect("slot 3 reopens");
        shell
            .fire_seated(&[seated(
                3,
                &prefix[..prefix.len() - 1],
                &fold,
                RsReset::Fresh,
            )])
            .expect("the shorter prefix folds");
        let ref2 = shell
            .fire_seated(&[seated_at(3, &rows, &fold, RsReset::Held, prefix.len() as u32 - 1)])
            .expect("[s_r, junk] folds");
        let d1 = spread(&win[0], &ref1[0]);
        let d2 = spread(&win[0], &ref2[0]);
        eprintln!(
            "claim 9: round {r} junk-row logits: window vs one-row fold {d1:.4} (argmax {} vs {}), vs two-row fold {d2:.4} (argmax {})",
            argmax(&win[0]),
            argmax(&ref1[0]),
            argmax(&ref2[0])
        );
        if r > 0 {
            shell
                .fire_seated(&[seated_at(
                    6,
                    &steps[r as usize - 1..r as usize],
                    &fold,
                    RsReset::Held,
                    k + r - 1,
                )])
                .expect("a one-row fold fires");
            let same = same_state(
                &shell.state_bytes(0).expect("slot 0"),
                &shell.state_bytes(6).expect("slot 6"),
            )
            .is_ok();
            eprintln!(
                "claim 9: round {r} — banks after folding s_{} through the discard path equal the plain fold: {same}",
                r - 1
            );
            assert!(
                same,
                "round {r}: replaying and folding one survivor of a two-row window left other banks than folding it plainly"
            );
        }
    }
}
