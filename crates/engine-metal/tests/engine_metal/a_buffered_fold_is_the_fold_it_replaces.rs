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
    assert!(
        shell.state_slot_bytes() > 0,
        "a recurrent plan has recurrent state"
    );
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
fn load(
    trace: poem::Trace,
    contract: &checkpoint::contract::ModelContract,
    checkpoint: &std::path::Path,
) -> Shell {
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
        top: engine_metal::Top::EveryRung,
        measure: engine::rates::Measuring::Time,
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
        lineage: engine::fire::Lineage::Set,
    }
}

fn seated<'a>(slot: u32, tokens: &'a [u32], rs: &'a RsVerb, reset: RsReset) -> Seated<'a> {
    Seated {
        replay_stated: None,
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
        replay_stated: None,
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

const LOGIT_FLOOR: f32 = 0.05;

/// Each state plane `(name, bytes, dtype)` the record lays end to end.
type Planes = Vec<(String, u64, poem_ir::Dtype)>;

fn planes_of(shell: &Shell) -> Planes {
    shell
        .state_planes()
        .into_iter()
        .map(|(cache, bytes, dtype)| {
            let name = match shell.trace().caches.get(cache as usize) {
                Some(
                    poem_ir::CacheRow::State { name, .. } | poem_ir::CacheRow::Kv { name, .. },
                ) => name.clone(),
                None => format!("cache {cache}"),
            };
            (name, bytes, dtype)
        })
        .collect()
}

/// Where `got` and `want` differ, plane by plane: each element read in its
/// plane's own dtype, how many differ, the largest difference, the plane's
/// largest value, and the worst difference in ulps of the element's own
/// value (an element's ulp is the gap to the next float above its
/// magnitude). Printed only when the states differ past the check.
fn tell_planes(got: &[u8], want: &[u8], planes: &Planes) {
    let mut at = 0usize;
    for (name, bytes, dtype) in planes {
        let len = *bytes as usize;
        let (g, w) = (&got[at..at + len], &want[at..at + len]);
        let read = |b: &[u8]| -> Vec<f32> {
            match dtype {
                poem_ir::Dtype::F32 => b
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                poem_ir::Dtype::Bf16 => b
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                    .collect(),
                _ => Vec::new(),
            }
        };
        let ulp = |v: f32| -> f32 {
            let m = v.abs().max(f32::MIN_POSITIVE);
            match dtype {
                poem_ir::Dtype::Bf16 => {
                    f32::from_bits(((m.to_bits() >> 16) + 1) << 16)
                        - f32::from_bits((m.to_bits() >> 16) << 16)
                }
                _ => f32::from_bits(m.to_bits() + 1) - m,
            }
        };
        let (gv, wv) = (read(g), read(w));
        let largest = wv.iter().fold(0f32, |a, v| a.max(v.abs()));
        let (mut apart, mut worst, mut worst_ulps, mut first) = (0usize, 0f32, 0f32, None);
        let mut worst_at = (0usize, 0f32, 0f32);
        for (k, (&x, &y)) in gv.iter().zip(&wv).enumerate() {
            if x.to_bits() == y.to_bits() {
                continue;
            }
            apart += 1;
            first.get_or_insert(k);
            let gap = (x - y).abs();
            if gap > worst {
                worst_at = (k, x, y);
            }
            worst = worst.max(gap);
            worst_ulps = worst_ulps.max(gap / ulp(y));
        }
        if apart > 0 || gv.is_empty() {
            eprintln!(
                "plane {name} ({dtype:?}, {len} bytes at byte {at}): {apart} of {} apart, \
                 max |diff| {worst:e}, max |value| {largest:e}, worst {worst_ulps:.1} ulps of its \
                 own value, first at element {first:?}; the largest at element {} is {:e} \
                 against {:e}",
                gv.len(),
                worst_at.0,
                worst_at.1,
                worst_at.2
            );
        }
        at += len;
    }
    if at != got.len() {
        eprintln!("the planes cover {at} of {} bytes", got.len());
    }
}

/// A float plane's elements, read in its dtype; `None` for an integer plane
/// (n-gram ids, which a fold carries exactly).
fn floats(bytes: &[u8], dtype: poem_ir::Dtype) -> Option<Vec<f64>> {
    match dtype {
        poem_ir::Dtype::F32 => Some(
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
                .collect(),
        ),
        poem_ir::Dtype::Bf16 => Some(
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| {
                    f64::from(f32::from_bits(
                        u32::from(u16::from_le_bytes([c[0], c[1]])) << 16,
                    ))
                })
                .collect(),
        ),
        _ => None,
    }
}

/// The distance between two states: the L2 norm of their difference over
/// every float plane. With `exact`, `Err` names an integer plane that
/// differs at all; without, integer planes are not counted (two different
/// folds carry different n-gram ids).
fn distance(a: &[u8], b: &[u8], planes: &Planes, exact: bool) -> Result<f64, String> {
    let mut at = 0usize;
    let mut sum = 0f64;
    for (name, bytes, dtype) in planes {
        let len = *bytes as usize;
        let (x, y) = (&a[at..at + len], &b[at..at + len]);
        match (floats(x, *dtype), floats(y, *dtype)) {
            (Some(x), Some(y)) => {
                sum += x
                    .iter()
                    .zip(&y)
                    .map(|(p, q)| (p - q) * (p - q))
                    .sum::<f64>();
            }
            _ if exact && x != y => {
                return Err(format!("integer plane {name} ({dtype:?}) differs"));
            }
            _ => {}
        }
        at += len;
    }
    assert_eq!(at, a.len(), "the planes cover the record");
    Ok(sum.sqrt())
}

/// `got` is the state `want` is, as the fold the test compares against: its
/// distance from `want` is under a tenth of `scale`, the distance between
/// two different folds of the same window (the fold of `j` tokens and of
/// `k`). A fold that took rows it should not have moves the state by about
/// `scale`. Rounding carried through the layers moves it by a small
/// fraction of it, since each layer takes the last one's slightly
/// different output. Integer planes are exact.
fn assert_same_state(got: &[u8], want: &[u8], what: &str, planes: &Planes, scale: f64) {
    let d = match distance(got, want, planes, true) {
        Ok(d) => d,
        Err(why) => {
            tell_planes(got, want, planes);
            panic!("{what}: {why}");
        }
    };
    eprintln!(
        "state {d:e} from the fold, against {scale:e} between two folds ({:.1e} of it)",
        d / scale
    );
    if d >= scale / 10.0 {
        tell_planes(got, want, planes);
        panic!(
            "{what}: the state is {d:e} from the fold, past a tenth of the {scale:e} between \
             the fold of {WINDOW} tokens and of 3"
        );
    }
}

#[test]
fn a_buffered_fold_is_the_fold_it_replaces() {
    let _device = crate::device_alone();
    if let Some(shell) = ready() {
        holds_the_contract(shell);
    }
}

/// The same contract over a gated-delta model with no PLE (Qwen3.5): its
/// conv and scan rows replay as the fold they defer.
#[test]
fn a_gdn_buffered_fold_is_the_fold_it_replaces() {
    let _device = crate::device_alone();
    if let Some(shell) = ready_gdn() {
        holds_the_contract(shell);
    }
}

/// A lane replaying rows it buffered, in a fire beside a folding lane (a
/// decode or a prompt piece of another sequence), answers as it does in a
/// fire of its own: its logits, and the state its replay folds; the folding
/// lane answers as alone too. A beam's lanes replay their lineage beside
/// other sequences' decodes so (as engine-gpu's suite and CUDA's
/// golden_logits state it).
#[test]
fn a_replaying_lane_beside_a_folding_lane_answers_as_alone() {
    let _device = crate::device_alone();
    if let Some(shell) = ready_gdn() {
        replays_beside_a_fold(shell);
    }
}

fn replays_beside_a_fold(mut shell: Shell) {
    let planes = planes_of(&shell);
    let fold = RsVerb::Fold;
    let tokens = window(11);
    let next = window(101);
    let third = window(203);
    let k = WINDOW as u32;
    let (fresh, held) = (RsReset::Fresh, RsReset::Held);
    let buffered = buffer(0, 0);
    let replay = buffer_over(k + 1, 0, 0);
    let replay_fold = buffer_over(k + 2, 0, k + 1);
    // One lane's fire, or a peer folding `third` fresh beside it.
    let run = |shell: &mut Shell, slot: u32, beside: Option<u32>| {
        shell.open(slot).expect("the slot opens");
        shell
            .fire_seated(&[seated(slot, &tokens, &buffered, fresh)])
            .expect("the buffering fire runs");
        let mut answers = Vec::new();
        for (rows, verb, at) in [(&next[..1], &replay, k), (&next[1..2], &replay_fold, k + 1)] {
            let out = match beside {
                None => shell
                    .fire_seated(&[seated_at(slot, rows, verb, held, at)])
                    .expect("the replaying fire runs")
                    .swap_remove(0),
                Some(peer) => {
                    shell.open(peer).expect("the peer opens");
                    shell
                        .fire_seated(&[
                            seated(peer, &third, &fold, fresh),
                            seated_at(slot, rows, verb, held, at),
                        ])
                        .expect("the mixed fire runs")
                        .swap_remove(1)
                }
            };
            answers.push(out);
        }
        (
            answers,
            shell.state_bytes(slot).expect("the slot reads back"),
        )
    };
    let (alone, alone_state) = run(&mut shell, 0, None);
    let (beside, beside_state) = run(&mut shell, 1, Some(2));
    shell.open(3).expect("slot 3 opens");
    let peer_alone = shell
        .fire_seated(&[seated(3, &third, &fold, fresh)])
        .expect("the peer alone")
        .swap_remove(0);
    shell.open(2).expect("slot 2 opens");
    shell.open(1).expect("slot 1 opens");
    shell
        .fire_seated(&[seated(1, &tokens, &buffered, fresh)])
        .expect("the buffering fire runs");
    let peer_beside = shell
        .fire_seated(&[
            seated(2, &third, &fold, fresh),
            seated_at(1, &next[..1], &replay, held, k),
        ])
        .expect("the mixed fire runs")
        .swap_remove(0);
    // The scale a state is measured against: two different folds' distance.
    shell.open(3).expect("slot 3 opens");
    shell
        .fire_seated(&[seated(3, &tokens[..3], &fold, fresh)])
        .expect("the short fold runs");
    let short = shell.state_bytes(3).expect("slot 3");
    let scale = distance(&short, &alone_state, &planes, false).expect("float planes only");
    for (at, (a, b)) in alone.iter().zip(&beside).enumerate() {
        let d = spread(a, b);
        assert!(
            d <= LOGIT_FLOOR,
            "replaying row {at} answers {d} apart beside a folding lane"
        );
    }
    assert_same_state(
        &beside_state,
        &alone_state,
        "the state a replay folds beside a folding lane",
        &planes,
        scale,
    );
    let d = spread(&peer_alone, &peer_beside);
    assert!(
        d <= LOGIT_FLOOR,
        "the folding lane answers {d} apart beside a replay"
    );
}

/// A buffering fire leaves the state as it was; a fold replaying it is the
/// fold it deferred, at any length; a folding lane beside a buffering one
/// answers as alone; and rounds of a two-row window with a rejected row
/// leave the banks a plain fold of the survivors leaves.
fn holds_the_contract(mut shell: Shell) {
    let planes = planes_of(&shell);
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
    let scale = distance(&folded_j, &folded, &planes, false).expect("float planes only");
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
        &planes,
        scale,
    );

    shell.open(4).expect("slot 4 opens");
    let mixed = shell
        .fire_seated(&[seated(4, &tokens, &buffer(0, j), RsReset::Fresh)])
        .expect("the mixed fire runs");
    assert_same_state(
        &shell.state_bytes(4).expect("slot 4"),
        &folded_j,
        &format!("the mixed row did not land the fold at row {j}"),
        &planes,
        scale,
    );
    let d = spread(&reference[0], &mixed[0]);
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
        &planes,
        scale,
    );
    shell.open(6).expect("slot 6 opens");
    shell
        .fire_seated(&[seated(6, &tokens, &fold, RsReset::Fresh)])
        .expect("A folds plainly");
    let plain = shell
        .fire_seated(&[seated_at(6, &second, &fold, RsReset::Held, k)])
        .expect("B folds after it");
    let d = spread(&plain[0], &round[0]);
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
        &planes,
        scale,
    );
    assert_eq!(
        shell.state_bytes(1).expect("slot 1"),
        zeroed,
        "the buffering peer folded"
    );
    let d = spread(&alone[0], &beside[0]);
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
        shell
            .fire_seated(&[seated_at(0, &rows, &verb, RsReset::Held, k + r)])
            .expect("a two-row window with a rejected row fires");
        let mut prefix: Vec<u32> = tokens.clone();
        prefix.extend_from_slice(&steps[..=r as usize]);
        shell.open(7).expect("slot 7 reopens");
        shell
            .fire_seated(&[seated(7, &prefix, &fold, RsReset::Fresh)])
            .expect("the prefix folds");
        shell
            .fire_seated(&[seated_at(
                7,
                &[junk],
                &fold,
                RsReset::Held,
                prefix.len() as u32,
            )])
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
        shell
            .fire_seated(&[seated_at(
                3,
                &rows,
                &fold,
                RsReset::Held,
                prefix.len() as u32 - 1,
            )])
            .expect("[s_r, junk] folds");
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
            assert_same_state(
                &shell.state_bytes(0).expect("slot 0"),
                &shell.state_bytes(6).expect("slot 6"),
                &format!(
                    "claim 9, round {r}: replaying and folding one survivor of a two-row window \
                     left other banks than folding it plainly"
                ),
                &planes,
                scale,
            );
        }
    }
}
