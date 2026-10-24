#![cfg(target_vendor = "apple")]

use std::path::PathBuf;

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;

const WINDOW: usize = 6;

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
    }
}

fn traced() -> poem::Trace {
    package()
        .trace(MICRO, &deploy(), "qwen4-micro", Platform::Metal)
        .unwrap_or_else(|why| panic!("the micro model traces: {why:#}"))
}

fn word(query_len: u32) -> u64 {
    traced().facts.word(&Request::new(query_len, false))
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
    let shell = Shell::load(Boot {
        kv_budget: crate::full_pool(&trace, PAGE, 128, 8),
        voxels: None,
        trace,
        contract: &contract,
        checkpoint: &fixture,
        budget: Budget::new(4, 64),
        patches: None,
        profile: None,
        page_size: PAGE,
        context: 128,
        slots: 8,
        runahead: engine::runahead::Runahead::F1,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        profiles: None,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the micro shell loads");
    eprintln!(
        "recurrent slot: {} bytes; buffered page slab: {:.1} KiB; serves rs verbs: {}",
        shell.state_slot_bytes(),
        shell.buffer_bytes() as f64 / 1024.0,
        shell.serves_rs_verbs()
    );
    assert!(
        shell.state_slot_bytes() > 0,
        "a GDN plan has recurrent state"
    );
    assert!(
        shell.buffer_bytes() > 0,
        "a GDN plan reserved no buffered page slab, so nothing below can be true"
    );
    assert!(shell.serves_rs_verbs());
    Some(shell)
}

fn window(seed: u32) -> Vec<u32> {
    (0..WINDOW as u32)
        .map(|at| (seed + at * 37) % 256)
        .collect()
}

fn pages() -> Vec<u32> {
    vec![0]
}

/// A buffering verb whose state holds the slots below `folded`: a fire whose
/// rows land at slot `first` replays the buffered slots `folded..first`, then
/// folds `fold` of them.
fn buffer(folded: u32, fold: u32) -> RsVerb {
    RsVerb::Buffer {
        pages: pages(),
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

#[test]
fn a_buffered_fold_is_the_fold_it_replaces() {
    let Some(mut shell) = ready() else {
        return;
    };
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
        .fire_seated(&[seated(3, &second, &buffer(0, j), RsReset::Held)])
        .expect("window B replays A and folds part of it");
    assert_eq!(
        shell.state_bytes(3).expect("slot 3"),
        folded_j,
        "a replay truncated at {j} did not leave the banks a fold of {j} leaves"
    );

    shell.open(4).expect("slot 4 opens");
    let mixed = shell
        .fire_seated(&[seated(4, &tokens, &buffer(0, j), RsReset::Fresh)])
        .expect("the mixed fire runs");
    assert_eq!(
        shell.state_bytes(4).expect("slot 4"),
        folded_j,
        "the mixed row did not land the fold at row {j}"
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
        .fire_seated(&[seated(5, &second, &buffer(0, k), RsReset::Held)])
        .expect("window B replays A and buffers itself");
    assert_eq!(
        shell.state_bytes(5).expect("slot 5"),
        folded,
        "replaying A ahead of B did not leave the banks Fold(A) leaves"
    );
    shell.open(6).expect("slot 6 opens");
    shell
        .fire_seated(&[seated(6, &tokens, &fold, RsReset::Fresh)])
        .expect("A folds plainly");
    let plain = shell
        .fire_seated(&[seated(6, &second, &fold, RsReset::Held)])
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
    assert_eq!(
        shell.state_bytes(7).expect("slot 7"),
        alone_state,
        "a folding lane's banks changed for having a buffering peer"
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
    shell.open(5).expect("slot 5 reopens");
    shell
        .fire_seated(&[seated(5, &tokens, &fold, RsReset::Fresh)])
        .expect("A folds");
    shell.open(6).expect("slot 6 reopens");
    shell
        .fire_seated(&[seated(6, &tokens, &fold, RsReset::Fresh)])
        .expect("A folds for the reference");
    let rounds = 8u32;
    for r in 0..rounds {
        let rows = [steps[r as usize], junk];
        // A holds slots 0..k folded; round r's window lands at slot k + r,
        // past the survivor s_{r-1} the round before left at k + r - 1.
        let verb = if r == 0 {
            buffer(k, 0)
        } else {
            buffer(k + r - 1, 1)
        };
        let win = shell
            .fire_seated(&[seated_at(5, &rows, &verb, RsReset::Held, k + r)])
            .expect("a two-row window with a rejected row fires");
        let mut prefix: Vec<u32> = tokens.clone();
        prefix.extend_from_slice(&steps[..=r as usize]);
        shell.open(7).expect("slot 7 reopens");
        shell
            .fire_seated(&[seated(7, &prefix, &fold, RsReset::Fresh)])
            .expect("the prefix folds");
        let ref1 = shell
            .fire_seated(&[seated(7, &[junk], &fold, RsReset::Held)])
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
            .fire_seated(&[seated(3, &rows, &fold, RsReset::Held)])
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
                .fire_seated(&[seated(
                    6,
                    &steps[r as usize - 1..r as usize],
                    &fold,
                    RsReset::Held,
                )])
                .expect("a one-row fold fires");
            let same =
                shell.state_bytes(5).expect("slot 5") == shell.state_bytes(6).expect("slot 6");
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
