//! **S-2 AND S-3: THE OBSERVABILITY CONTRACT, AND WHAT IT COSTS A FIRE THAT
//! DOES NOT WANT IT** (`.wiki/alto/attn-score.md` §4;
//! `.wiki/alto/campaign.md` §2's S-2 and S-3).
//!
//! `export_axes` gates the capture COLUMN — the per-query log-sum-exp the
//! model text has always exported, that a reader collects after the graph.
//! This file gates the thing that column was never able to be: the per-KEY
//! rectangle a guest epilogue binds, written by `attention.score_capture`
//! into the shell's own slab as the graph runs.
//!
//! ```text
//! (a) S-2, the contract: a captured plane is a probability distribution over
//!     the request's LIVE keys — every live slot finite and non-negative,
//!     the live prefix summing to 1, the declared ceiling past it exactly
//!     zero — and it says so per HEAD, on every exported layer
//! (b) S-2, the tail is not a leftover: a short request after a long one on
//!     the same lane leaves no live garbage past its own kv_len
//! (c) S-2, the numbers DISCRIMINATE: a real prompt's attention sink carries
//!     mass that a uniform row could not, so the rectangle carries the signal
//!     the eviction papers need and not just a well-formed shape
//! (d) S-2, two lanes read their own rows: a shell that observed lane zero
//!     for everybody would be perfectly deterministic and perfectly wrong
//! (e) S-3, zero-cost-when-off: a fire no lane captured is LAUNCH-identical
//!     and TOKEN-identical to the same fire on a shell that has no slab at
//!     all, and the capture arm's observation adds no node to it
//! ```
//!
//! **WHY (e) IS ASKED AS A NODE COUNT AND A CLOCK.** S-3's words are
//! "launch-identical and ms-identical to pre-campaign", and a test cannot
//! diff against a shell that is no longer in the tree. What it can do is the
//! thing `the_second_row_axis_costs_the_first_nothing` does for the carve:
//! ask the invariant in the form that is decidable HERE and is strictly
//! stronger than a stored golden. The observation is a launch or it is not,
//! so the node count of a non-capturing fire is the whole of the launch
//! claim; and the millisecond claim rides on top of it, because a fire with
//! the same launches over the same rows on the same device is the same fire.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 --release \
//!     --test the_observability_contract_holds -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_cuda::{Boot, Graphs, Lane, LayerScores, Seated, Shell};
use model_dsl::{Classify, Platform, Request};

/// The workhorse: the SKU whose model text declares the capture arm.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The published score-row width — the DSL reads this same constant, which is
/// the whole reason it is one.
const KV_MAX: usize = eta_ir::registry::ATTN_SCORE_KV_MAX as usize;

/// Long enough that a uniform row and a real one are different objects, and
/// short enough to prefill in one fire under [`BUDGETS`].
const PROMPT: &str = "The capital of France is Paris. Paris is a large European city \
                      with a long history. Paris is a large European city with a long \
                      history. Paris is a large European city with a long history.";

/// A second prompt, deliberately much shorter, for the stale-tail gate.
const SHORT: &str = "Hello";

const BUDGETS: model_compiler::Budget = model_compiler::Budget {
    max_lanes: 4,
    max_tokens: 256,
    buckets: Vec::new(),
    max_adapters: 0,
};

/// One shell at a time per process — `kernels-cuda`'s scratch slabs are
/// process-global and keyed by name (`serve_smoke.rs` argues it whole).
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The lane word the model's own `Classify` computes — the facts qwen
/// declares, and no third opinion about any of them.
fn word(query_len: u32, captures: bool) -> u64 {
    model::qwen_3::forward::Facts::of(&Request::new(query_len, false).capturing_scores(captures))
        .word()
}

fn seat<'a>(slot: u32, tokens: &'a [u32], captures: bool) -> Seated<'a> {
    let lane = Lane {
        slot,
        word: word(tokens.len() as u32, captures),
        tokens,
    };
    if captures {
        Seated::capturing(lane)
    } else {
        Seated::of(lane)
    }
}

/// Prefill `prompt` on `slot`, capturing or not, and hand back the lane's
/// block of score planes.
fn observe(shell: &mut Shell, slot: u32, prompt: &[u32], captures: bool) -> Vec<f32> {
    shell.open(slot).expect("the slot opens");
    let mut mass: Vec<Vec<LayerScores>> = Vec::new();
    shell
        .fire_captured(&[seat(slot, prompt, captures)], &[], &mut mass)
        .expect("the fire lands");
    shell
        .observed(0)
        .expect("the slab reads back")
        .expect("this load observes")
}

/// One plane of a lane's block, as a slice.
fn plane(block: &[f32], at: usize) -> &[f32] {
    &block[at * KV_MAX..(at + 1) * KV_MAX]
}

// ── (a) the contract ─────────────────────────────────────────────────────

/// **S-2, THE WHOLE OF IT.** Every exported (layer, head) plane of a
/// capturing lane is a probability distribution over that request's live KV
/// positions, and exactly zero on the declared ceiling past them.
///
/// The tolerance is `1e-3` and it is bf16's, not f32's: the keys the capture
/// reads out of the pages are bf16 and the dots are accumulated in f32, so a
/// row of a few hundred bf16 terms lands within about a thousandth of one.
/// Tightening it further would be asserting about the storage dtype rather
/// than about the softmax.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_captured_plane_is_a_distribution_over_the_live_keys() {
    let _one = serialized();
    let Some((mut shell, tokenizer)) = ready("the observability contract") else {
        return;
    };
    let planes = shell.score_planes() as usize;
    let heads = shell.score_heads() as usize;
    let layers = shell.score_layers();
    assert!(
        planes > 0 && heads > 0 && !layers.is_empty(),
        "`{SKU}` observes nothing, so this file is testing nothing: \
         {planes} planes, {heads} heads, layers {layers:?}"
    );
    assert_eq!(
        planes,
        layers.len() * heads,
        "the slab's planes are not this text's exported layers by its heads"
    );

    let prompt = encode(&tokenizer, PROMPT);
    let kv_len = prompt.len();
    assert!(
        kv_len < KV_MAX,
        "the gate's prompt is longer than the ceiling it is asserting about"
    );
    let block = observe(&mut shell, 0, &prompt, true);
    assert_eq!(block.len(), planes * KV_MAX, "one lane's block of planes");

    for (at, layer) in layers.iter().enumerate() {
        for head in 0..heads {
            let row = plane(&block, at * heads + head);
            let live = &row[..kv_len];
            for (key, mass) in live.iter().enumerate() {
                assert!(
                    mass.is_finite() && *mass >= 0.0,
                    "layer {layer} head {head} key {key} is {mass}, which is not a \
                     probability"
                );
            }
            let total: f32 = live.iter().sum();
            assert!(
                (total - 1.0).abs() < 1e-3,
                "layer {layer} head {head} sums to {total} over its {kv_len} live keys; \
                 a captured row is the softmax the attention performed, so it sums to one"
            );
            // THE CEILING PAST THE LIVE PREFIX IS ABSENCE AND NOT A SMALL
            // NUMBER. `!= 0.0` and not `< eps`: a position that does not
            // exist received no attention, and a policy sorts on this row
            // without a sentinel.
            let tail = row[kv_len..].iter().filter(|mass| **mass != 0.0).count();
            assert_eq!(
                tail, 0,
                "layer {layer} head {head} carries {tail} non-zero slots past its \
                 {kv_len} live keys; an eviction policy would keep a position that \
                 does not exist"
            );
        }
    }
}

// ── (b) the tail is not a leftover ───────────────────────────────────────

/// **A SLAB OUTLIVES A FIRE, AND THAT IS THE FAILURE THIS CATCHES.**
///
/// The rectangle is reserved at a ceiling and reused, so a short request
/// landing where a long one was is the one arrangement in which "the tail is
/// zero" can be true by accident on every fire but the second. The kernel
/// writes the whole row every time for exactly this reason; this is the
/// assertion that says it does.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_short_request_leaves_no_tail_of_the_long_one_before_it() {
    let _one = serialized();
    let Some((mut shell, tokenizer)) = ready("the stale-tail gate") else {
        return;
    };
    let planes = shell.score_planes() as usize;
    let long = encode(&tokenizer, PROMPT);
    let short = encode(&tokenizer, SHORT);
    assert!(
        short.len() < long.len(),
        "the gate needs a genuinely shorter second request"
    );

    let first = observe(&mut shell, 0, &long, true);
    let live = first[..long.len()].iter().any(|mass| *mass != 0.0);
    assert!(live, "the long request wrote nothing, so there is no tail to leave");

    let second = observe(&mut shell, 0, &short, true);
    for at in 0..planes {
        let row = plane(&second, at);
        let tail = row[short.len()..]
            .iter()
            .enumerate()
            .filter(|(_, mass)| **mass != 0.0)
            .map(|(key, mass)| (short.len() + key, *mass))
            .collect::<Vec<_>>();
        assert!(
            tail.is_empty(),
            "plane {at} still carries the previous request at {:?}",
            &tail[..tail.len().min(4)]
        );
    }
}

// ── (c) the numbers discriminate ─────────────────────────────────────────

/// **A WELL-FORMED ROW IS NOT YET AN OBSERVATION.** Everything above would
/// pass on a uniform distribution, which carries no information and would
/// make every eviction policy a random one. On a real model the first
/// position is an attention sink and takes mass out of all proportion to a
/// uniform share; requiring that is what separates "the plumbing works" from
/// "the plumbing carries the signal".
///
/// Asked over the LAYER FOLD rather than per head, because sink behaviour is
/// a property of the model and not of every individual head — the curated
/// `tova-attention` gate asks the same question of the same fold.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn the_captured_mass_discriminates_rather_than_being_uniform() {
    let _one = serialized();
    let Some((mut shell, tokenizer)) = ready("the discrimination gate") else {
        return;
    };
    let planes = shell.score_planes() as usize;
    let prompt = encode(&tokenizer, PROMPT);
    let kv_len = prompt.len();
    let block = observe(&mut shell, 0, &prompt, true);

    let mut folded = vec![0.0f32; kv_len];
    for at in 0..planes {
        let row = plane(&block, at);
        for (key, mass) in folded.iter_mut().enumerate() {
            *mass += row[key];
        }
    }
    let total: f32 = folded.iter().sum();
    assert!(
        (total - planes as f32).abs() < 0.05 * planes as f32,
        "the fold over {planes} planes carries {total} of mass, not {planes}"
    );
    let uniform = total / kv_len as f32;
    // **THE BAR IS TWO SHARES AND IT IS NOT ARBITRARY.** What is being ruled
    // out is a UNIFORM row — the one shape that satisfies every assertion
    // above and carries no information at all — so the bar has to be the
    // smallest factor no uniform row can reach and no real sink misses. On
    // this prompt the sink measured 3.4 shares; a factor of two leaves the
    // gate room for a different prompt length without letting a flat row
    // through, and the median check below is what stops a row that is uniform
    // everywhere except position 0.
    assert!(
        folded[0] > 2.0 * uniform,
        "position 0 carries {} against a uniform share of {uniform}; a real model's \
         attention sink is not a uniform row, so this rectangle is not describing \
         the attention the model performed",
        folded[0]
    );
    let mut sorted = folded.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[kv_len / 2];
    let top = sorted[kv_len - 1];
    assert!(
        top > 3.0 * median,
        "the heaviest key carries {top} against a median of {median}; the row is \
         flat, and a policy ranking on it would be picking at random"
    );
}

// ── (d) two lanes read their own rows ────────────────────────────────────

/// **A SHELL THAT OBSERVED LANE ZERO FOR EVERYBODY WOULD BE PERFECTLY
/// DETERMINISTIC AND PERFECTLY WRONG**, and a single-lane fire cannot tell
/// the difference, because lane zero's base offset IS zero. Two capturing
/// lanes of different content in one fire is the smallest arrangement that
/// can.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn two_capturing_lanes_observe_their_own_rows() {
    let _one = serialized();
    let Some((mut shell, tokenizer)) = ready("the per-lane gate") else {
        return;
    };
    let planes = shell.score_planes() as usize;
    // Same LENGTH, different content — a length difference would put the two
    // lanes in different row runs and prove nothing about the base offset.
    let left = encode(&tokenizer, "The capital of France is Paris and the river is");
    let right = encode(&tokenizer, "The capital of Norway is Oslo and the river is");
    assert_eq!(left.len(), right.len(), "the gate needs two lanes of one shape");

    shell.open(0).expect("slot 0 opens");
    shell.open(1).expect("slot 1 opens");
    let mut mass: Vec<Vec<LayerScores>> = Vec::new();
    shell
        .fire_captured(
            &[seat(0, &left, true), seat(1, &right, true)],
            &[],
            &mut mass,
        )
        .expect("the two-lane fire lands");

    let first = shell.observed(0).expect("lane 0 reads").expect("observes");
    let second = shell.observed(1).expect("lane 1 reads").expect("observes");
    let moved = (0..planes * KV_MAX)
        .map(|at| (first[at] - second[at]).abs())
        .fold(0.0f32, f32::max);
    assert!(
        moved > 1e-4,
        "two lanes of different prompts observed the same rectangle to {moved}; \
         the slab's per-lane base is not being read"
    );
}

// ── (e) S-3, zero-cost-when-off ──────────────────────────────────────────

/// **S-3. A FIRE NO LANE CAPTURED IS THE FIRE THIS SHELL ALWAYS FIRED.**
///
/// The observation is a LAUNCH or it is nothing: `Run::capture_scores`
/// returns before it reaches a stream for a load with no slab, for a fire no
/// lane captured, and for a node the score seam does not name. So the claim
/// is asked as the node count of the walk plus the tokens it produced, and
/// asked twice — once on a shell whose lanes capture nothing, once on the
/// same shell after a capturing fire has run on it — because a slab that
/// leaked a launch into the plain path would show up in the first number and
/// a slab that leaked STATE would show up in the second.
///
/// The pre-campaign comparison this stands in for is the one gate (d) of
/// `export_axes` already makes about the capture COLUMN. What is added here
/// is the arm: the column was always written, and the per-key rectangle is
/// new, so "the axis is free when off" has to be re-asked of the new thing.
///
/// # The launch count needs a graph, and this is where the file states `On`
///
/// "The same launches" is not a thing an eager fire can be asked. It has no
/// artifact: the walk issues and forgets, and the only external count of what
/// a fire ran is the node census of something RECORDED. So this test — alone
/// in this file — puts the shell in the tiered mode and states `bodies`, and
/// the count it compares is `BodyStats::nodes`.
///
/// **AND THAT MAKES THE CLAIM STRONGER THAN THE SUBTRACTION IT REPLACES.**
/// The version of this test written against the keyed cache read
/// `Stats::nodes` on a shell loaded `Graphs::Off`, where the counter is zero
/// by construction — it was subtracting zero from zero and calling the result
/// a launch count. What is asked now is what the claim actually means: capture
/// the plain composition's body, run the capturing fire on another lane, and
/// then demand that the plain lane REPLAYS THE BODY IT ALREADY HAD — hits
/// moving, captures unmoved, reshapes unmoved, `nodes` unmoved. A slab that
/// leaked a launch into the plain path would have to change the plain
/// composition's graph to do it, and a graph that changed is a re-capture or a
/// reshape, both of which are counted here.
#[test]
#[ignore = "real-hardware: needs a CUDA device and a local model snapshot; run it with `-- --ignored`, which the self-hosted `pie-worker (engine-cuda)` job does"]
fn a_fire_no_lane_captured_pays_the_observability_axis_nothing() {
    let _one = serialized();
    let Some((mut shell, tokenizer)) = ready("the zero-cost gate") else {
        return;
    };
    let prompt = encode(&tokenizer, PROMPT);

    // The one test in this file that records. `record::WARM_FIRES` walks pass
    // before a composition is captured, so the plain fire is issued three
    // times and the third is the one that mints the body every assertion
    // below is about.
    shell.set_mode(Graphs::On);
    shell.set_bodies(true);

    let mut mass: Vec<Vec<LayerScores>> = Vec::new();
    let mut plain = Vec::new();
    for _ in 0..3 {
        // Re-opened every round so the three fires present ONE composition
        // rather than a lengthening prefill.
        shell.open(0).expect("the slot opens");
        plain = shell
            .fire_captured(&[seat(0, &prompt, false)], &[], &mut mass)
            .expect("the plain fire lands");
    }
    let armed = shell.body_stats();
    assert!(
        armed.captures >= 1,
        "the plain composition was never captured, so there is no graph whose \
         launches this test could count and nothing below asserts anything. A \
         moved `refusals` says the admissibility rule turned it away: {armed}"
    );
    let plain_nodes = armed.nodes;
    assert!(
        mass[0].is_empty(),
        "a lane that captured nothing was handed {} columns",
        mass[0].len()
    );

    // A capturing fire on another slot, so the slab is written and the arm
    // has run at least once in this process.
    shell.open(1).expect("the second slot opens");
    shell
        .fire_captured(&[seat(1, &prompt, true)], &[], &mut mass)
        .expect("the capturing fire lands");
    assert!(
        !mass[0].is_empty(),
        "the capturing lane was handed nothing, so the axis is not running at all \
         and the zero below is a zero about nothing"
    );

    // And the plain fire again, on a re-opened slot, after all of that.
    shell.open(0).expect("the slot reopens");
    let before_again = shell.body_stats();
    let again = shell
        .fire_captured(&[seat(0, &prompt, false)], &[], &mut mass)
        .expect("the plain fire lands again");
    let after = shell.body_stats();
    eprintln!("across the plain fire that followed the capturing one: {after}");

    assert_eq!(
        (
            after.hits - before_again.hits,
            after.captures - before_again.captures,
            after.reshapes - before_again.reshapes,
        ),
        (1, 0, 0),
        "the plain fire that followed a capturing one on another lane did not \
         replay the body the plain composition already had. A moved `captures` \
         or `reshapes` says the plain fire's graph CHANGED across a capturing \
         fire — which is the observability axis costing an off lane something \
         — and a zero hit says it walked instead: before {before_again} / \
         after {after}"
    );
    // `BodyStats::nodes` names the MOST RECENTLY CAPTURED body, so this says
    // two things at once: nothing captured across the capturing fire and the
    // plain fire after it, and the census still describes the plain
    // composition's own graph.
    assert_eq!(
        after.nodes, plain_nodes,
        "the last captured body holds {} launches where the plain composition \
         was recorded with {plain_nodes}; something minted a graph after the \
         slab had been written, and the observability axis is not free when \
         off",
        after.nodes,
    );
    assert_eq!(
        plain[0], again[0],
        "a plain fire's logits moved across a capturing fire on another lane"
    );
    assert!(
        mass[0].is_empty(),
        "the plain lane was handed a capture it did not ask for"
    );
}

// ── boot ─────────────────────────────────────────────────────────────────

fn encode(tokenizer: &tokenizer::Tokenizer, text: &str) -> Vec<u32> {
    tokenizer.encode(text)
}

fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = Path::new(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
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

fn ready(what: &str) -> Option<(Shell, tokenizer::Tokenizer)> {
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
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let source = ztensor_compat::index(&container).expect("the checkpoint opens");
    let contract = model::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);

    let shell = Shell::load(Boot {
        residency: engine_cuda::experts::Plan::default(),
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        budget: BUDGETS,
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::Off,
        // The golden path for every claim in this file but one: what is under
        // test is the score axis, and a recorded fire would put a second
        // subject in the room. S-3 below states `On` on this same shell for
        // its own reason, which it argues where it does it. `bodies` is
        // written out because it defaults to TRUE now and this load's `Off`
        // is what keeps the arming pass from running, not the word.
        knobs: engine_cuda::Knobs {
            bodies: true,
            ..engine_cuda::Knobs::default()
        },
        program_cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        weight_cache_dir: None,
    })
    .expect("the shell loads");
    eprintln!(
        "{SKU} loaded — capture layers {:?}, planes {}, heads {}, observes {}",
        shell.score_layers(),
        shell.score_planes(),
        shell.score_heads(),
        shell.observes_scores(),
    );
    Some((shell, tokenizer))
}
