//! **THE WANDERING-BATCH MEASUREMENT** — the bodies path's reason to exist,
//! read off a workload whose decode batch will not sit still.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --test bodies_bench -- --ignored --nocapture
//! ```
//!
//! The gate beside this file diffs a STEADY stream, which is the workload
//! where a body's advantage is smallest: one composition, one exec, every fire
//! past the warm ones replays. The difference the design paid for shows when
//! the batch WANDERS — a lane joins, a lane drains, the same composition keeps
//! arriving at new lane counts.
//!
//! **THE ARM THIS USED TO BE MEASURED AGAINST IS GONE, AND THAT IS THE
//! MEASUREMENT'S CONCLUSION RATHER THAN ITS LOSS.** There were three arms
//! here: eager, keyed, bodies. The keyed path minted an exec per
//! `(rows, lanes)` and paid a warm pass plus a capture at every count it had
//! not seen, so under this pattern its capture column climbed with the lane
//! counts while the bodies column sat at the COMPOSITION count — one. That is
//! the number the tier-2 campaign acted on: the keyed path was deleted, the
//! router is bodies-or-eager, and what remains to measure is the two arms that
//! ship.
//!
//! So the arms are EAGER (`Graphs::Off`, the oracle and the cost floor for
//! anything a body refuses) and BODIES (`On` with `[engine] bodies` at boot,
//! so the load arms its decode rungs before the pattern starts). The workload
//! prefills four slots and then walks the lane pattern 1-2-3-4-3-2 for
//! `ROUNDS` cycles, feeding each lane its own greedy tail.
//!
//! What makes ONE exec serve every count is that the plans stopped moving with
//! the count, and that is the plan-at-bucket-ceiling work this file is the
//! record of. `Run::planning` carves every schedule a body serves at the KEY's
//! lane ceiling and row total — the fire's lattice point for a whole-fire
//! window and, since the ceiling design's Option B, prefix sums over the key's
//! per-class rung ladder for a windowed one — so the payload numbers a capture
//! bakes are a function of the `BodyKey` and the load; `BodyStats::reshapes`
//! is what says so, and it says so by being zero.
//!
//! The claims are exactly two:
//!
//! 1. **identity** — both arms say the same token, per slot, per step; a
//!    wandering batch changes costs, never numbers.
//! 2. **the counters tell the story** — printed, not asserted, because this
//!    file is a measurement and `bodies_gate.rs` is the claim. What to read:
//!    `captures` sits at the composition count (one, for this workload) rather
//!    than climbing with the lane counts; `reshapes` is ZERO, which is the
//!    ceiling carve's whole deliverable and the column that used to be nonzero
//!    before it; `hits` accounts for the pattern; and `eager_rotating` /
//!    `eager_buffered` are the fires that ran outside every graph, which on
//!    this SKU should be none once the arming has run. Beside them, warm
//!    ms/fire per lane count for both arms — the eager column is what a
//!    refused composition costs today and the body column is what an admitted
//!    one costs, which is the only price comparison left to make.
use std::path::{Path, PathBuf};
use std::time::Instant;

use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// 1-2-3-4-3-2, cycled: every count is visited twice per cycle but at
/// different neighbours, which is what shakes out order-dependent caching.
const PATTERN: [usize; 6] = [1, 2, 3, 4, 3, 2];

const ROUNDS: usize = 4;

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

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn word(query_len: u32) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn ready(what: &str, graphs: Graphs, bodies: bool) -> Option<(Shell, tokenizer::Tokenizer)> {
    if !engine_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let checkpoint = snapshot()?;
    let container = container(&checkpoint)?;
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU");
    let trace = trace(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import for the SKU")(&source)
        .expect("the SKU's import contract fits its own checkpoint");
    drop(source);

    let mut shell = Shell::load(Boot {
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs,
        knobs: engine_cuda::Knobs {
            bodies,
            ..engine_cuda::Knobs::default()
        },
        program_cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: None,
    })
    .expect("the shell loads");
    // Stated on the shell as well as in the `Knobs` above: `bodies` defaults
    // to TRUE since the keyed path died, so the eager arm has to say `false`
    // out loud or it would silently be the bodies arm.
    shell.set_bodies(bodies);
    Some((shell, tokenizer))
}

/// The workload: four prefills, then the pattern. Returns per-slot token
/// streams and (lane_count, millis) per decode fire.
fn wander(shell: &mut Shell, prompts: &[Vec<u32>; 4]) -> ([Vec<u32>; 4], Vec<(usize, f64)>) {
    let mut tails: [Vec<u32>; 4] = Default::default();
    for slot in 0..4u32 {
        shell.open(slot).expect("a slot opens");
        let prompt = &prompts[slot as usize];
        let logits = shell
            .fire(&[Lane {
                slot,
                word: word(prompt.len() as u32),
                tokens: prompt,
            }])
            .expect("a prefill fires");
        tails[slot as usize].push(argmax(&logits[0]));
    }
    let mut costs = Vec::new();
    for _ in 0..ROUNDS {
        for &lanes in &PATTERN {
            let fed: Vec<[u32; 1]> = (0..lanes)
                .map(|slot| [*tails[slot].last().expect("a tail")])
                .collect();
            let batch: Vec<Lane> = (0..lanes)
                .map(|slot| Lane {
                    slot: slot as u32,
                    word: word(1),
                    tokens: &fed[slot],
                })
                .collect();
            let at = Instant::now();
            let logits = shell.fire(&batch).expect("a wandering decode fires");
            costs.push((lanes, at.elapsed().as_secs_f64() * 1000.0));
            for slot in 0..lanes {
                tails[slot].push(argmax(&logits[slot]));
            }
        }
    }
    (tails, costs)
}

/// Warm mean ms/fire per lane count: the second half of each count's visits.
fn by_lanes(costs: &[(usize, f64)]) -> Vec<(usize, f64)> {
    let mut out = Vec::new();
    for lanes in 1..=4usize {
        let all: Vec<f64> = costs
            .iter()
            .filter(|(l, _)| *l == lanes)
            .map(|(_, ms)| *ms)
            .collect();
        if all.is_empty() {
            continue;
        }
        let warm = &all[all.len() / 2..];
        out.push((lanes, warm.iter().sum::<f64>() / warm.len() as f64));
    }
    out
}

#[test]
#[ignore = "real-hardware: needs a CUDA device and a local Qwen3.5-0.8B snapshot; run with -- --ignored"]
fn a_wandering_batch_pays_the_arm_it_rode_and_says_the_same_tokens() {
    let Some((mut eager_shell, tokenizer)) = ready("the wandering bench", Graphs::Off, false)
    else {
        return;
    };
    let prompts: [Vec<u32>; 4] = [
        "The capital of France is",
        "Water boils at",
        "The largest planet is",
        "Two plus two equals",
    ]
    .map(|text| tokenizer.encode(text));

    let (eager_tails, eager_costs) = wander(&mut eager_shell, &prompts);
    let eager_bodies = eager_shell.body_stats();
    drop(eager_shell);

    let Some((mut bodies_shell, _)) = ready("the bodies arm", Graphs::On, true) else {
        return;
    };
    let (bodies_tails, bodies_costs) = wander(&mut bodies_shell, &prompts);
    let bodies_stats = bodies_shell.body_stats();
    drop(bodies_shell);

    // The eager arm's own counters, printed because they are the control: a
    // shell that stated `bodies: false` must show the default value, and a
    // moving number there would say the two columns below are the same path
    // measured twice.
    eprintln!("eager arm:  {eager_bodies}");
    eprintln!("bodies arm: {bodies_stats}");
    eprintln!(
        "warm ms/fire by lanes — eager {:?} | bodies {:?}",
        by_lanes(&eager_costs),
        by_lanes(&bodies_costs)
    );

    assert_eq!(
        eager_tails, bodies_tails,
        "the bodies arm and the eager walk disagree under a wandering batch"
    );
}
