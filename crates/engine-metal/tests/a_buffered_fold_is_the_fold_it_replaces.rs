#![cfg(target_vendor = "apple")]

use std::path::PathBuf;

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_metal::serve::Seated;
use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Dtype, Platform, Request};

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

fn word(query_len: u32) -> u64 {
    models::qwen_4::forward::Facts::of(&Request::new(query_len, false)).word()
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
    let micro = models::qwen_4::model::Model::flash_micro(Dtype::Bf16, Dtype::Bf16, 1);
    let trace = model_dsl::trace_hybrid("qwen4-micro", &micro, Platform::Metal);
    let source =
        ztensor_compat::index(fixture.join("model.safetensors")).expect("the fixture opens");
    let contract = micro
        .import(&source, Platform::Metal)
        .expect("the micro text's import fits the fixture it was generated for");
    drop(source);
    let shell = Shell::load(Boot {
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
        pages: (8) * (128) / (PAGE),
        runahead: engine::runahead::Runahead::F1,
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

fn buffer(at: u32, fold: u32, replay: u32) -> RsVerb {
    RsVerb::Buffer {
        pages: pages(),
        at,
        fold: FoldLen::Host(fold),
        replay,
    }
}

fn fold_buffered(bound: u32, len: u32) -> RsVerb {
    fold_buffered_at(0, bound, len)
}

fn fold_buffered_at(at: u32, bound: u32, len: u32) -> RsVerb {
    RsVerb::FoldBuffered {
        pages: pages(),
        at,
        bound,
        len: FoldLen::Host(len),
    }
}

fn window_verb(read: Vec<u32>, write: Vec<u32>, fold: u32) -> RsVerb {
    RsVerb::Window {
        read,
        write,
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
        .fire_seated(&[seated(1, &tokens, &buffer(0, 0, 0), RsReset::Fresh)])
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

    shell
        .fire_seated(&[seated(1, &tokens, &fold_buffered(k, k), RsReset::Held)])
        .expect("the fold-buffered fire runs");
    let replayed = shell.state_bytes(1).expect("slot 1");
    assert_eq!(replayed.len(), folded.len());
    assert_eq!(
        replayed, folded,
        "`Buffer` then `FoldBuffered(k)` did not leave the banks `Fold` leaves"
    );

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
        .fire_seated(&[seated(3, &tokens, &buffer(0, 0, 0), RsReset::Fresh)])
        .expect("the buffering fire runs");
    shell
        .fire_seated(&[seated(3, &tokens, &fold_buffered(k, j), RsReset::Held)])
        .expect("the truncated fold runs");
    assert_eq!(
        shell.state_bytes(3).expect("slot 3"),
        folded_j,
        "a replay truncated at {j} did not leave the banks a fold of {j} leaves"
    );

    shell.open(4).expect("slot 4 opens");
    let mixed = shell
        .fire_seated(&[seated(4, &tokens, &buffer(0, j, 0), RsReset::Fresh)])
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

    let second = window(101);
    shell.open(5).expect("slot 5 opens");
    shell
        .fire_seated(&[seated(5, &tokens, &buffer(0, 0, 0), RsReset::Fresh)])
        .expect("window A buffers");
    let round = shell
        .fire_seated(&[seated(5, &second, &buffer(k, k, k), RsReset::Held)])
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
    shell
        .fire_seated(&[seated(
            5,
            &second,
            &fold_buffered_at(k, k, k),
            RsReset::Held,
        )])
        .expect("B's buffer folds");
    shell.open(6).expect("slot 6 reopens");
    shell
        .fire_seated(&[seated(6, &tokens, &fold, RsReset::Fresh)])
        .expect("A folds plainly again");
    for row in 0..second.len() {
        shell
            .fire_seated(&[seated(6, &second[row..=row], &fold, RsReset::Held)])
            .expect("one row of B folds");
    }
    assert_eq!(
        shell.state_bytes(5).expect("slot 5"),
        shell.state_bytes(6).expect("slot 6"),
        "folding the round's buffer did not leave the banks Fold(A) then Fold(B), a row at a time, leaves"
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
            seated(1, &tokens, &buffer(0, 0, 0), RsReset::Fresh),
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

    let (run_a, run_b) = (vec![1u32], vec![2u32]);
    shell.open(1).expect("slot 1 reopens");
    shell
        .fire_seated(&[seated(
            1,
            &tokens,
            &window_verb(Vec::new(), run_a.clone(), 0),
            RsReset::Fresh,
        )])
        .expect("window round 1 runs");
    assert_eq!(
        shell.state_bytes(1).expect("slot 1"),
        zeroed,
        "round 1 folded something with fold 0"
    );
    let round2 = shell
        .fire_seated(&[seated_at(
            1,
            &second,
            &window_verb(run_a.clone(), run_b.clone(), j),
            RsReset::Held,
            j,
        )])
        .expect("window round 2 runs");
    assert_eq!(
        shell.state_bytes(1).expect("slot 1"),
        folded_j,
        "round 2 did not leave the banks Fold(A[..j]) leaves"
    );
    shell.open(2).expect("slot 2 reopens");
    shell
        .fire_seated(&[seated(2, &tokens[..j as usize], &fold, RsReset::Fresh)])
        .expect("A[..j] folds plainly");
    let plain = shell
        .fire_seated(&[seated(2, &second, &fold, RsReset::Held)])
        .expect("B folds after it");
    let d = spread(&plain[0], &round2[0]);
    eprintln!("claim 7: window round 2 vs B after fold(A[..j]), last-row logit spread {d:.4}");
    assert!(
        d <= LOGIT_FLOOR,
        "the window verb answers other logits than a plain fold of the prefix: {d}"
    );
    let third = window(307);
    shell
        .fire_seated(&[seated_at(
            1,
            &third,
            &window_verb(run_b, run_a, j),
            RsReset::Held,
            2 * j,
        )])
        .expect("window round 3 runs");
    shell.open(3).expect("slot 3 reopens");
    shell
        .fire_seated(&[seated(3, &tokens[..j as usize], &fold, RsReset::Fresh)])
        .expect("A[..j] folds plainly");
    shell
        .fire_seated(&[seated(3, &second[..j as usize], &fold, RsReset::Held)])
        .expect("B[..j] folds after it");
    assert_eq!(
        shell.state_bytes(1).expect("slot 1"),
        shell.state_bytes(3).expect("slot 3"),
        "round 3 did not leave the banks Fold(A[..j]) then Fold(B[..j]) leaves"
    );

    let steps = window(401);
    shell.open(4).expect("slot 4 reopens");
    shell
        .fire_seated(&[seated(4, &tokens, &fold, RsReset::Fresh)])
        .expect("A folds");
    shell.open(5).expect("slot 5 reopens");
    shell
        .fire_seated(&[seated(5, &tokens, &fold, RsReset::Fresh)])
        .expect("A folds for the reference");
    let held0 = k;
    let mut worst = 0f32;
    for step in 0..3usize {
        let row = &steps[step..step + 1];
        let (read, write) = if step % 2 == 0 {
            (vec![3u32], vec![4u32])
        } else {
            (vec![4u32], vec![3u32])
        };
        let fold_now = if step == 0 { 0 } else { 1 };
        let windowed = shell
            .fire_seated(&[seated_at(
                4,
                row,
                &window_verb(read, write, fold_now),
                RsReset::Held,
                held0 + step as u32,
            )])
            .expect("a one-row window fires");
        let plain = shell
            .fire_seated(&[seated(5, row, &fold, RsReset::Held)])
            .expect("a one-row fold fires");
        let d = spread(&plain[0], &windowed[0]);
        eprintln!(
            "claim 8: step {step} one-row window (fold {fold_now}) vs plain fold: logit spread {d:.4}; argmax {} vs {}",
            argmax(&windowed[0]),
            argmax(&plain[0])
        );
        worst = worst.max(d);
    }
    assert!(
        worst <= LOGIT_FLOOR,
        "one-row windows answer other logits than one-row folds: {worst}"
    );
    shell
        .fire_seated(&[seated_at(
            4,
            &steps[3..4],
            &window_verb(vec![4u32], vec![3u32], 1),
            RsReset::Held,
            held0 + 3,
        )])
        .expect("the closing window fires");
    assert_eq!(
        shell.state_bytes(4).expect("slot 4"),
        shell.state_bytes(5).expect("slot 5"),
        "three one-row window rounds did not leave the banks three one-row folds leave"
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
        let verb = if r == 0 {
            buffer(0, 0, 0)
        } else {
            buffer(r, 1, 1)
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
