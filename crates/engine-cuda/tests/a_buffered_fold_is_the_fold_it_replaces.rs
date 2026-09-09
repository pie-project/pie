use std::path::{Path, PathBuf};

use engine::fire::{FoldLen, RsReset, RsVerb};
use engine_cuda::{Boot, Lane, Seated, Shell};
use model_compiler::Budget;
use model_dsl::{Platform, Request};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

const WINDOW: usize = 20;

const PAGE: u32 = 16;

const PAGES: u64 = WINDOW.div_ceil(PAGE as usize) as u64;

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
}

fn container(snapshot: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .ok()?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

fn word(query_len: u32) -> u64 {
    (models::sku(SKU).expect("the catalog ships the SKU").classify)(&Request::new(query_len, false))
}

fn ready(what: &str) -> Option<Shell> {
    if !engine_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no Qwen3.5-0.8B snapshot in the hugging face cache \
             (set PIE_SMOKE_SNAPSHOT)"
        );
        return None;
    };
    let Some(container) = container(&checkpoint) else {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    };
    let sku = models::sku(SKU).expect("the catalog ships the SKU");
    let trace = (sku.trace)(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = sku
        .contract(&source, Platform::Cuda)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        voxels: None,
        deferred_tier: false,
        classify: sku.classify,
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: PAGE,
        context: 512,
        slots: 8,
        pages: 8 * 512 / PAGE,
        ordinal: 0,
        graphs: engine_cuda::Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the shell loads");
    eprintln!(
        "{what}: buffered-activation pool is {:.2} MiB",
        shell.buffer_bytes() as f64 / (1 << 20) as f64,
    );
    assert!(
        shell.buffer_bytes() > 0,
        "a GDN plan reserved no buffered-activation pool, so nothing below can be true"
    );
    Some(shell)
}

fn window() -> Vec<u32> {
    (0..WINDOW as u32).map(|at| 1000 + at * 37).collect()
}

fn buffer(at: u32) -> RsVerb {
    mixed(at, 0)
}

fn mixed(at: u32, fold: u32) -> RsVerb {
    replaying(at, fold, 0)
}

fn replaying(at: u32, fold: u32, replay: u32) -> RsVerb {
    RsVerb::Buffer {
        pages: (0..PAGES as u32).collect(),
        at,
        fold: FoldLen::Host(fold),
        replay,
    }
}

fn fold_buffered(len: u32) -> RsVerb {
    bounded_fold(WINDOW as u32, len)
}

fn bounded_fold(bound: u32, len: u32) -> RsVerb {
    fold_buffered_at(0, bound, len)
}

fn fold_buffered_at(at: u32, bound: u32, len: u32) -> RsVerb {
    RsVerb::FoldBuffered {
        pages: (0..PAGES as u32).collect(),
        at,
        bound,
        len: FoldLen::Host(len),
    }
}

fn fire(shell: &mut Shell, lanes: &[Seated<'_>]) -> engine_cuda::Result<Vec<Vec<f32>>> {
    shell.fire_media(lanes, &[], &[], &mut Vec::new())
}

fn seated<'a>(slot: u32, tokens: &'a [u32], rs: RsVerb, reset: RsReset) -> Seated<'a> {
    Seated {
        lane: Lane {
            slot,
            word: word(tokens.len() as u32),
            tokens,
        },
        rs,
        rs_reset: reset,
        ..Seated::of(Lane {
            slot,
            word: word(tokens.len() as u32),
            tokens,
        })
    }
}

#[test]
fn a_buffered_fold_is_the_fold_it_replaces_every_case() {
    a_deployments_first_fire_is_its_every_fire();
    a_buffered_fold_is_the_fold_it_replaces();
    one_fire_folds_the_lane_that_committed_and_not_the_lane_that_buffered();
    the_read_path_replays_the_buffer_it_folds();
}

fn a_deployments_first_fire_is_its_every_fire() {
    let Some(mut shell) = ready("first-fire determinism") else {
        return;
    };
    let tokens = window();
    let mut banks: Vec<Vec<u8>> = Vec::new();
    let mut rows: Vec<Vec<f32>> = Vec::new();
    for slot in 0..3u32 {
        shell.open(slot).expect("slot opens");
        let out = fire(&mut shell, &[seated(slot, &tokens, RsVerb::Fold, RsReset::Fresh)])
            .expect("the fold runs");
        banks.push(shell.state_bytes(slot).expect("banks"));
        rows.push(out.into_iter().next().expect("a row"));
    }
    for (at, what) in [(1usize, "the second fold"), (2, "the third fold")] {
        let (worst, coarse, any) = bank_gap(&banks[0], &banks[at]);
        let d = spread(&rows[0], &rows[at]);
        eprintln!("first fold vs {what}: banks worst {worst:.3e}, {coarse} coarse, {any} differ; logit spread {d:.3e}");
        assert_eq!(
            (coarse, any),
            (0, 0),
            "{what} left different banks than the first (worst {worst:.3e}, logit spread {d:.3e}) \
             — a kernel changed under the deployment, as the gemm tuner used to"
        );
    }
}

fn a_buffered_fold_is_the_fold_it_replaces() {
    let Some(mut shell) = ready("the buffered fold") else {
        return;
    };
    let tokens = window();

    shell.open(0).expect("slot 0 opens");
    let zeroed = shell.state_bytes(0).expect("slot 0 reads back");
    assert!(
        zeroed.iter().all(|byte| *byte == 0),
        "an opened slot's recurrent banks are not zero, so nothing below is a comparison"
    );
    let reference = fire(&mut shell, &[seated(0, &tokens, RsVerb::Fold, RsReset::Fresh)])
        .expect("the folding fire runs");
    let folded = shell.state_bytes(0).expect("slot 0 reads back");
    assert_ne!(
        folded, zeroed,
        "a fold over {WINDOW} tokens left the banks untouched, so the reference is empty"
    );

    shell.open(1).expect("slot 1 opens");
    fire(&mut shell, &[seated(1, &tokens, buffer(0), RsReset::Fresh)])
        .expect("the buffering fire runs");
    assert_eq!(
        shell.state_bytes(1).expect("slot 1 reads back"),
        zeroed,
        "a `RsVerb::Buffer` fire perturbed the folded state — the whole point of the \
         buffer is that a rejected draft is pure host bookkeeping"
    );

    fire(&mut shell, &[seated(1, &tokens, fold_buffered(WINDOW as u32), RsReset::Held)])
        .expect("the fold-buffered fire runs");
    let replayed = shell.state_bytes(1).expect("slot 1 reads back");
    assert_eq!(
        replayed.len(),
        folded.len(),
        "the two slots hold different amounts of state, which is a pool bug"
    );
    let differing = folded
        .iter()
        .zip(&replayed)
        .filter(|(a, b)| a != b)
        .count();
    let (worst, coarse, any) = bank_gap(&folded, &replayed);
    eprintln!("claim 2: {differing} bytes differ; as f32 cells: worst |d| {worst:.3e}, {coarse} cells past 1e-3 relative, {any} cells differ at all");
    if std::env::var_os("PIE_GATE_PROBE").is_none() {
        assert_eq!(
            differing, 0,
            "buffer-then-fold-buffered left {differing} of {} state bytes different from the \
             fold it replaces",
            folded.len(),
        );
    }

    shell.open(2).expect("slot 2 opens");
    fire(&mut shell, &[seated(2, &tokens, buffer(0), RsReset::Fresh)])
        .expect("the second buffering fire runs");
    fire(&mut shell, &[seated(2, &tokens, fold_buffered(4), RsReset::Held)])
        .expect("the truncated fold-buffered fire runs");
    let truncated = shell.state_bytes(2).expect("slot 2 reads back");
    assert_ne!(
        truncated, zeroed,
        "a fold truncated at 4 tokens folded nothing at all"
    );
    assert_ne!(
        truncated, replayed,
        "a fold truncated at 4 tokens left the same bytes as one over all {WINDOW}, so \
         `commit_len` reached the launch and was ignored"
    );

    shell.open(3).expect("slot 3 opens");
    fire(&mut shell, &[seated(
            3,
            &tokens[..4],
            bounded_fold(4, 4),
            RsReset::Fresh,
        )])
        .expect("the shorter-bounded fold-buffered fire runs");
    let shorter = shell.state_bytes(3).expect("slot 3 reads back");
    {
        let (w, c, a) = bank_gap(&truncated, &shorter);
        eprintln!("claim 4: truncated(4) vs bounded(4): worst {w:.3e}, {c} coarse, {a} differ");
        let (w, c, a) = bank_gap(&folded, &truncated);
        eprintln!("claim 3: fold(20) vs truncated(4): worst {w:.3e}, {c} coarse, {a} differ");
    }
    assert_ne!(
        shorter, zeroed,
        "a replay bounded at 4 tokens folded nothing at all"
    );
    let differing = truncated
        .iter()
        .zip(&shorter)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "a fold of {WINDOW} buffered tokens truncated at 4 left {differing} of {} state \
         bytes different from a fold of the same 4 over a buffer bounded there — the \
         length seat is still changing the arithmetic and not only the count",
        truncated.len(),
    );

    shell.open(4).expect("slot 4 opens");
    let mixed_out = fire(&mut shell, &[seated(4, &tokens, mixed(0, 4), RsReset::Fresh)])
        .expect("the mixed fire runs");
    let mixed_state = shell.state_bytes(4).expect("slot 4 reads back");
    {
        let (w, c, a) = bank_gap(&truncated, &mixed_state);
        eprintln!("claim 5: mixed(0,4) vs truncated(4): worst {w:.3e}, {c} coarse, {a} differ");
        shell.open(5).expect("slot 5 opens");
        fire(&mut shell, &[seated(5, &tokens[..4], RsVerb::Fold, RsReset::Fresh)]).expect("four fold plainly");
        let four = shell.state_bytes(5).expect("slot 5 reads back");
        let (w, c, a) = bank_gap(&four, &mixed_state);
        eprintln!("probe: fold(4) vs mixed(0,4): worst {w:.3e}, {c} coarse, {a} differ");
        let (w, c, a) = bank_gap(&four, &truncated);
        eprintln!("probe: fold(4) vs truncated replay(4): worst {w:.3e}, {c} coarse, {a} differ");
        shell.open(5).expect("slot 5 reopens");
        fire(&mut shell, &[seated(5, &tokens, RsVerb::Fold, RsReset::Fresh)]).expect("fold again");
        let again = shell.state_bytes(5).expect("slot 5 reads back");
        let (w, c, a) = bank_gap(&folded, &again);
        eprintln!("probe: fold(20) twice: worst {w:.3e}, {c} coarse, {a} differ");
    }
    let differing = truncated
        .iter()
        .zip(&mixed_state)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "one fire that buffers {WINDOW} tokens and folds 4 of them left {differing} of {} \
         state bytes different from buffering them and folding 4 in two fires",
        truncated.len(),
    );

    let mixed_row = mixed_out.first().expect("the mixed fire reads a row back");
    let plain_row = reference.first().expect("the folding fire reads a row back");
    assert_eq!(
        mixed_row.len(),
        plain_row.len(),
        "the two fires read back different row widths"
    );
    assert!(
        !plain_row.is_empty(),
        "the reference fire read no logits back, so the tail cannot be pinned"
    );
    let worst = mixed_row
        .iter()
        .zip(plain_row)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= 1e-3,
        "the mixed fire's own outputs differ from the plain fold's by {worst} — the tail \
         segment `[4, {WINDOW})` did not run, or did not continue from the boundary"
    );
}

fn one_fire_folds_the_lane_that_committed_and_not_the_lane_that_buffered() {
    let Some(mut shell) = ready("the per-lane fold predicate") else {
        return;
    };
    let tokens = window();

    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    let zeroed = shell.state_bytes(1).expect("slot 1 reads back");

    fire(&mut shell, &[
            seated(0, &tokens, RsVerb::Fold, RsReset::Fresh),
            seated(1, &tokens, buffer(0), RsReset::Fresh),
        ])
        .expect("the mixed fire runs");

    let predicate = shell.fold_predicate(2).expect("the predicate reads back");
    assert_eq!(
        predicate,
        vec![1, 0],
        "the fold predicate the device wrote is {predicate:?}, and this fire's lanes are \
         one folding lane followed by one buffering lane"
    );
    assert_ne!(
        shell.state_bytes(0).expect("slot 0 reads back"),
        zeroed,
        "the folding lane of a mixed fire did not fold"
    );
    assert_eq!(
        shell.state_bytes(1).expect("slot 1 reads back"),
        zeroed,
        "the buffering lane of a mixed fire folded anyway, so the predicate did not reach \
         the scan"
    );
}

const LOGIT_FLOOR: f32 = 2e-2;

fn bank_gap(a: &[u8], b: &[u8]) -> (f32, usize, usize) {
    let cells = |bytes: &[u8]| -> Vec<f32> {
        bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    };
    let (a, b) = (cells(a), cells(b));
    let mut worst = 0.0f32;
    let mut coarse = 0usize;
    let mut any = 0usize;
    for (x, y) in a.iter().zip(&b) {
        let d = (x - y).abs();
        if d > 0.0 {
            any += 1;
        }
        if d > 1e-3 * x.abs().max(y.abs()).max(1e-6) {
            coarse += 1;
        }
        worst = worst.max(d);
    }
    (worst, coarse, any)
}

fn spread(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (at, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = at;
        }
    }
    best
}

fn the_read_path_replays_the_buffer_it_folds() {
    let Some(mut shell) = ready("the read path") else {
        return;
    };
    let first = window();
    let k = WINDOW as u32;
    let second: Vec<u32> = (0..6u32).map(|at| 3000 + at * 41).collect();

    shell.open(6).expect("slot 6 opens");
    fire(&mut shell, &[seated(6, &first, RsVerb::Fold, RsReset::Fresh)]).expect("A folds plainly");
    let folded_a = shell.state_bytes(6).expect("slot 6 reads back");
    let plain = fire(&mut shell, &[seated(6, &second, RsVerb::Fold, RsReset::Held)]).expect("B folds after it");
    let folded_ab = shell.state_bytes(6).expect("slot 6 reads back");

    shell.open(7).expect("slot 7 opens");
    fire(&mut shell, &[seated(7, &first, buffer(0), RsReset::Fresh)]).expect("window A buffers");
    let round = fire(&mut shell, &[seated(7, &second, replaying(k, k, k), RsReset::Held)])
        .expect("window B replays A and buffers itself");
    let differing = folded_a
        .iter()
        .zip(&shell.state_bytes(7).expect("slot 7 reads back"))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "replaying A ahead of B left {differing} of {} state bytes different from the banks \
         Fold(A) leaves",
        folded_a.len()
    );
    let d = spread(&plain[0], &round[0]);
    eprintln!(
        "read path: B after replay(A) vs B after fold(A), last-row logit spread {d:.4}; argmax {} vs {}",
        argmax(&round[0]),
        argmax(&plain[0])
    );
    assert!(d <= LOGIT_FLOOR, "the read path answers other logits than a plain fold of the prefix: {d}");

    fire(&mut shell, &[seated(7, &second, fold_buffered_at(k, 6, 6), RsReset::Held)]).expect("B's buffer folds");
    let differing = folded_ab
        .iter()
        .zip(&shell.state_bytes(7).expect("slot 7 reads back"))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "folding the round's buffer left {differing} of {} state bytes different from the \
         banks Fold(A) then Fold(B) leaves",
        folded_ab.len()
    );

    shell.open(2).expect("slot 2 opens");
    fire(&mut shell, &[seated(2, &first[..4], RsVerb::Fold, RsReset::Fresh)]).expect("A[..4] folds plainly");
    let folded_a4 = shell.state_bytes(2).expect("slot 2 reads back");
    shell.open(3).expect("slot 3 opens");
    fire(&mut shell, &[seated(3, &first, buffer(0), RsReset::Fresh)]).expect("window A buffers again");
    let partial = fire(&mut shell, &[seated(3, &second, replaying(k, 4, k), RsReset::Held)])
        .expect("window B replays A and folds four of it");
    let truncated = shell.state_bytes(3).expect("slot 3 reads back");
    shell.open(4).expect("slot 4 opens");
    fire(&mut shell, &[seated(4, &first, buffer(0), RsReset::Fresh)]).expect("A buffers");
    fire(&mut shell, &[seated(4, &first[..4], fold_buffered_at(0, 4, 4), RsReset::Held)])
        .expect("four of A's buffer folds");
    let two_fire = shell.state_bytes(4).expect("slot 4 reads back");
    let (worst, coarse, _) = bank_gap(&folded_a4, &truncated);
    eprintln!("read path, truncated fold vs a plain Fold(A[..4]): worst {worst:.3e}, {coarse} coarse (printed)");
    let (worst, coarse, any) = bank_gap(&two_fire, &truncated);
    eprintln!("read path, truncated fold vs buffer-then-fold-four: worst {worst:.3e}, {coarse} coarse, {any} differ");
    assert_eq!(
        (coarse, any),
        (0, 0),
        "a round replaying {k} and folding 4 left banks {worst} off buffering {k} and folding \
         4 in two fires — the truncated commit lands somewhere else on the read path"
    );
    let d = spread(&plain[0], &partial[0]);
    eprintln!("read path, truncated fold: last-row logit spread {d:.4}");
    assert!(d <= LOGIT_FLOOR, "a truncated fold changed what the window computes: {d}");
}
