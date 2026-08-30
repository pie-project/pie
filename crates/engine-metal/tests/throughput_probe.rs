//! **THE BATCHING HALF OF THE NORTH-STAR RACE, MEASURED.**
//!
//! `serve_smoke` says the engine answers correctly and prints one warm
//! ms/fire beside the answer; `session_c_first_light` says the same thing for
//! the three big families. Both of them fire ONE lane. This file fires N of
//! them, and it exists because every number those two print is a number about
//! a machine serving a single sequence — which is the half of the race the
//! engine is not for.
//!
//! Three measurements, one loaded shell each:
//!
//! 1. **The ladder.** N ∈ {1, 4, 8, 16, 32} decode lanes on distinct slots,
//!    each seated by its own short prefill, then a warm window of M decode
//!    fires end to end. ms/fire and aggregate tok/s per rung, printed as a
//!    table. The only thing ASSERTED is that batching buys something — the
//!    numbers are the output, not the gate.
//! 2. **The pipelining delta (article 1).** The same window at N = 16, once
//!    with a step of run-ahead in hand and once in lockstep. The difference is
//!    what `Runahead`'s second frame is worth on this plane, and it is the
//!    Article-1 measurement rather than a claim about it.
//! 3. **The polymorphic-batching invariant.** At N = 16, each lane's greedy
//!    continuation, token for token, against the continuation that lane
//!    produces alone. This is design §0's headline case at sixteen lanes
//!    instead of two, and it is the one arm here that is a GATE — modulo
//!    [`TIE`], which is where the gate is drawn and why: two lanes of the
//!    sixteen part at a step whose top-two bf16 logits are EQUAL, and a coin
//!    landing the same way twice is not a property a batched fire can have.
//!
//! # Why the timed window feeds a rotation and not its own argmax
//!
//! A decode chain feeds each step the previous step's argmax, so it must read
//! the numbers back before it can build the next fire — and a measurement of
//! run-ahead whose workload cannot run ahead measures nothing. Worse, the
//! argmax itself is 248320 host comparisons per lane per step, which lands
//! INSIDE the window it is timing. So the window feeds each lane a rotation
//! over its own prompt's tokens: the kv grows, the positions advance, every
//! kernel does exactly the work it does in production, and the host does not
//! stand in the middle of it. What the tokens SAY is [`the correctness
//! arm`](batching_is_polymorphic)'s business, and that arm feeds argmax the
//! whole way.
//!
//! The rows are still harvested every step, because they are harvested
//! whether or not anybody asks: `Shell::harvest_one` reads the arm's readout
//! seat and widens it to `f32` as part of settling. `rows_of` takes what is
//! already there.
//!
//! # Gating
//!
//! Apple-only at compile time, and every arm SKIPS at run time naming what
//! was missing. Two rows run by default — the 0.8B 4-bit vehicle and
//! gpt-oss-20b, both fast enough to sweep a ladder — and the two ~16 GiB
//! models are behind [`PIE_PROBE_SKUS`](selected), because each of those holds
//! more than half the box for the length of a run.
//!
//! ```text
//! cargo test -p engine-metal --release --test throughput_probe -- --nocapture --test-threads=1
//! PIE_PROBE_SKUS=qwen36 cargo test -p engine-metal --release --test throughput_probe -- --nocapture
//! ```
//!
//! | variable | what it moves |
//! |---|---|
//! | `PIE_PROBE_SKUS` | which rows run — `u4`, `gptoss`, `qwen36`, `gemma4`, comma separated |
//! | `PIE_PROBE_LANES` | the ladder, comma separated |
//! | `PIE_PROBE_STEPS` | decode fires in one warm window |
//! | `PIE_PROBE_TUNING` | `[metal.tuning]` keys, `key=value` comma separated, for A/B-ing a crossover |
//! | `PIE_U4_SNAPSHOT` etc. | where each row's snapshot lives |

#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine::runahead::Runahead;
use engine_metal::{Boot, Landed, Lane, Seated, Shell, StepView};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

// ─────────────────────────────────────────────────────────────────────────────
// The rows
// ─────────────────────────────────────────────────────────────────────────────

/// **ONE MODEL THIS PROBE CAN SWEEP.** The fields are `session_c_first_light`'s
/// [`Family`] minus the audit and plus the three numbers a ladder needs: how
/// wide it may go, whether it runs without being asked, and what its FFN
/// routes.
struct Sku {
    /// The short name `PIE_PROBE_SKUS` selects on.
    name: &'static str,
    /// The catalog row, spelled as the catalog spells it.
    sku: &'static str,
    /// The environment variable that overrides the snapshot directory.
    env: &'static str,
    /// The `models--*` directory in the hugging face cache, exactly.
    repo: &'static str,
    /// The lane word this family's own `Classify` computes.
    word: fn(u32) -> u64,
    /// **The token this family will not decode without** — see
    /// `session_c_first_light::Family::bos`, which is where the measurement
    /// behind this field is written down. It matters here for the same reason
    /// it matters there: the correctness arm compares continuations, and a
    /// gemma sequence with no beginning produces `" la la la la"` in every
    /// lane, which is sixteen lanes agreeing about nothing.
    bos: Option<u32>,
    /// Whether this row runs when `PIE_PROBE_SKUS` is unset.
    ///
    /// **THE TWO BIG ONES ARE OFF, AND IT IS A MEMORY FACT AND NOT A TASTE.**
    /// 15-17 GiB of weights on a 32 GiB box, held for the length of a sweep,
    /// beside a kv pool sized for sixteen lanes.
    by_default: bool,
    /// The widest rung this row's ladder is allowed to reach.
    ///
    /// Thirty-two for the vehicle, sixteen for anything whose weights are
    /// most of the machine: the kv pool is `lanes × context` and it is the one
    /// term of the footprint that grows with the rung.
    ceiling: u32,
    /// `(experts, top_k)` for a routed FFN, `None` for a dense stack.
    ///
    /// Read for one reason: `kernels_metal::linear::moe::should_batch` decides
    /// between the per-row arm and the sorted, batched one on
    /// `pairs = rows × top_k` against `experts × moe_batch_min_per_expert`,
    /// and on the M1 Max that threshold falls between two rungs of this very
    /// ladder. A mixture's throughput curve therefore has a STEP in it, and a
    /// table that did not say which arm each rung took would be a table of
    /// numbers nobody could read.
    routed: Option<(u32, u32)>,
}

/// **THE VEHICLE.** `four_bit_first_light`'s SKU: small enough that the whole
/// ladder fits, fast enough that a rung is seconds, and 4-bit affine
/// throughout — so the crossovers it crosses are the ones the catalog's real
/// models cross.
const U4: Sku = Sku {
    name: "u4",
    sku: "qwen35-d0.8b-mlxu4-kv-bf16",
    env: "PIE_U4_SNAPSHOT",
    repo: "models--mlx-community--Qwen3.5-0.8B-4bit",
    word: |query_len| model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word(),
    bos: None,
    by_default: true,
    ceiling: 32,
    routed: None,
};

/// **GPT-OSS-20B.** Twenty-four layers, 32 experts routed four ways — the row
/// this file's mixture column exists for. See `session_c_first_light::GPT_OSS`
/// for the checkpoint audit; the expert numbers below are that file's
/// ("32 experts routed four ways") and the model text's.
const GPT_OSS: Sku = Sku {
    name: "gptoss",
    sku: "gptoss-20b-mlxu4-mxfp4-kv-bf16",
    env: "PIE_GPTOSS_SNAPSHOT",
    repo: "models--mlx-community--gpt-oss-20b-MXFP4-Q4",
    word: |query_len| model::gpt_oss::forward::Facts::of(&Request::new(query_len, false)).word(),
    bos: None,
    by_default: true,
    ceiling: 16,
    routed: Some((32, 4)),
};

/// **QWEN3.6-27B.** Sixty-four layers, forty-eight of them gated-delta.
/// Opt-in: see [`Sku::by_default`].
const QWEN36: Sku = Sku {
    name: "qwen36",
    sku: "qwen36-27b-mlxu4-kv-bf16",
    env: "PIE_QWEN36_SNAPSHOT",
    repo: "models--mlx-community--Qwen3.6-27B-4bit",
    word: |query_len| model::qwen_3::forward::Facts::of(&Request::new(query_len, false)).word(),
    bos: None,
    by_default: false,
    ceiling: 16,
    routed: None,
};

/// **GEMMA-4-31B.** Sixty layers over two attention shapes. Opt-in.
const GEMMA4: Sku = Sku {
    name: "gemma4",
    sku: "gemma4-31b-mlxu4-kv-bf16",
    env: "PIE_GEMMA4_SNAPSHOT",
    repo: "models--mlx-community--gemma-4-31b-it-4bit",
    word: |query_len| model::gemma_4::forward::Facts::of(&Request::new(query_len, false)).word(),
    // `<bos>`, id 2 — `session_c_first_light::Family::bos` is the measurement.
    bos: Some(2),
    by_default: false,
    ceiling: 16,
    routed: None,
};

// ─────────────────────────────────────────────────────────────────────────────
// The statutes
// ─────────────────────────────────────────────────────────────────────────────

/// **SIXTEEN DISTINCT PROMPTS, AND DISTINCT IS THE WHOLE POINT.** Sixteen
/// lanes carrying one prompt would agree with each other for a reason that has
/// nothing to do with batching — they would agree if the shell handed every
/// lane row zero. These are short (a decode probe wants its prefills cheap and
/// its sequences seated, not a prefill benchmark), different from each other,
/// and different in LENGTH, so no lane's rows begin where another's do.
///
/// Sixteen and not thirty-two because sixteen is where the claim needs them:
/// the ladder's wider rung cycles this list and the invariant arm does not
/// reach it (see [`AT`]).
const PROMPTS: &[&str] = &[
    "The capital of France is",
    "Water boils at",
    "The largest planet in our solar system is",
    "Two plus two equals",
    "The first person to walk on the moon was",
    "The chemical symbol for gold is",
    "Shakespeare wrote a play about a prince of",
    "The speed of light is roughly",
    "The longest river in Africa is",
    "A regular hexagon has",
    "The author of Pride and Prejudice is",
    "In binary, the number four is written",
    "The tallest mountain above sea level is",
    "A leap year happens every",
    "The currency of Japan is",
    "The smallest prime number is",
];

/// How many decode fires one warm window measures, unless `PIE_PROBE_STEPS`
/// says otherwise.
const STEPS: usize = 32;

/// Fires thrown away before the window opens. The first fire of a shape
/// compiles its shader points and grows its slabs, and none of that is decode
/// throughput.
const WARMUP: usize = 8;

/// The ladder, unless `PIE_PROBE_LANES` says otherwise. Clipped per row by
/// [`Sku::ceiling`].
const LADDER: &[u32] = &[1, 4, 8, 16, 32];

/// The rung the pipelining delta and the correctness arm are measured at —
/// clipped to the widest rung the ladder actually reached.
const AT: u32 = 16;

/// Decode fires in the correctness arm, per lane. Short on purpose: sixteen
/// lanes each decoded ALONE is sixteen sequential runs, and the claim is about
/// agreement rather than about length.
const CHECK: usize = 8;

/// Tokens one sequence may hold. Every arm re-seats its lanes, so what has to
/// fit is one prompt plus one warm-up plus one window.
const CONTEXT: u32 = 256;

/// **WHAT BATCHING HAS TO BUY BEFORE THIS FILE IS WILLING TO CALL IT
/// BATCHING.** Aggregate tok/s at the best rung, over the same at one lane.
///
/// A decode fire is memory-bound on the weights: eight lanes read the same
/// weight tile the one lane read, so the aggregate is supposed to climb nearly
/// with N until the arithmetic catches up. A tenth of that would still be a
/// visible improvement and would also be consistent with a shell that had
/// serialized the lanes internally, which is the failure this separates. The
/// measured numbers are printed either way, so a regression that stays over it
/// is still readable.
const MIN_GAIN: f64 = 1.5;

/// **HOW CLOSE A STEP HAS TO BE BEFORE A DISAGREEMENT STOPS BEING ONE.**
///
/// The top-two logit gap, in logits. A batched fire and a solo fire are two
/// different rectangles: the launches tile them differently, so the reductions
/// sum in a different order, and bf16 rounding lands somewhere else. That is
/// not a defect to be tightened away — it is the same thing `serve_smoke`'s
/// two-lane golden means when it compares argmaxes across fires but bytes only
/// within one, and the same thing `session_c_first_light` writes down when
/// `mlx_lm` answers its own prompt two ways at a margin of 0.0625.
///
/// The number is a quarter of a logit, and it is chosen against what it has to
/// separate rather than against taste. `session_c_first_light` measures this
/// family of checkpoints at a two-implementation noise floor of 0.103 rms and
/// records a real mlx-vs-mlx flip at 0.0625; the steps a model has genuinely
/// decided in that file are won by 0.94 to 5.13. A quarter sits an order of
/// magnitude below the smallest DECIDED step and a factor of two above the
/// largest measured tie, which is the widest gap available to put a line in.
/// Every parting is printed with its margin either way, so a drift toward the
/// line is readable long before it crosses it.
const TIE: f32 = 0.25;

/// **ONE SHELL AT A TIME, PER PROCESS.** `session_c_first_light`'s reason,
/// doubled: these loads are large AND these numbers are timings, and a timing
/// taken beside another test's fire is not a timing.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

// ─────────────────────────────────────────────────────────────────────────────
// The preconditions
// ─────────────────────────────────────────────────────────────────────────────

/// Is this row asked for? `PIE_PROBE_SKUS` is a comma-separated list of
/// [`Sku::name`]s; unset means the rows that [`Sku::by_default`] admits.
fn selected(row: &Sku) -> bool {
    match std::env::var("PIE_PROBE_SKUS") {
        Ok(stated) => stated
            .split(',')
            .map(str::trim)
            .any(|name| name.eq_ignore_ascii_case(row.name) || name == "all"),
        Err(_) => row.by_default,
    }
}

/// The rungs this run sweeps: `PIE_PROBE_LANES` or [`LADDER`], clipped to the
/// row's [`ceiling`](Sku::ceiling) and to the sixteen prompts there are.
fn ladder(row: &Sku) -> Vec<u32> {
    let stated: Vec<u32> = match std::env::var("PIE_PROBE_LANES") {
        Ok(list) => list
            .split(',')
            .filter_map(|rung| rung.trim().parse().ok())
            .collect(),
        Err(_) => LADDER.to_vec(),
    };
    let mut rungs: Vec<u32> = stated
        .into_iter()
        .filter(|&rung| rung > 0 && rung <= row.ceiling)
        .collect();
    rungs.sort_unstable();
    rungs.dedup();
    rungs
}

/// How many fires one window measures.
fn steps() -> usize {
    std::env::var("PIE_PROBE_STEPS")
        .ok()
        .and_then(|stated| stated.trim().parse().ok())
        .filter(|&count: &usize| count > 0)
        .unwrap_or(STEPS)
}

/// **A CROSSOVER, MOVED FOR ONE RUN** — `PIE_PROBE_TUNING`, a comma-separated
/// list of `key = value` pairs that becomes the body of a `[metal.tuning]`
/// table in a boot document this process opens before it loads anything.
///
/// Any key that table names is reachable, because a sweep that could only move
/// the ONE constant somebody had already suspected is a sweep that can only
/// confirm. The two this file was written to cross:
///
///   * `moe_batch_min_per_expert` — the mixture column below reports a STEP,
///     and a column that reports one without being able to cross it on
///     purpose can only ever be read as a coincidence. `should_batch` fires at
///     `pairs >= experts × min`, so `0` batches at every width and a number
///     past `lanes × top_k / experts` batches at none.
///   * `qmm_min_batch` — the dense GEMV/GEMM crossover, which decides whether
///     a rung of this ladder reads every weight once or once per lane.
///
/// **ONE ANSWER PER PROCESS, WHICH IS WHY THIS IS A RUN AND NOT AN ARM.**
/// `kernels_metal::tuning` folds the device row and the document at the first
/// `current()` and freezes it — `four_bit_first_light::RESPAWN_KEY` carries
/// the same paragraph — so a sweep that wants both answers runs the binary
/// twice, and a run that states this should also name ONE row in
/// `PIE_PROBE_SKUS` so that the load which freezes the table is the one that
/// was told.
///
/// **AND A GEMM ARM READ OFF A NARROW LADDER IS NOT THE GEMM ARM.** This is
/// the trap `qmm_min_batch = 2` walks into, and it is worth a paragraph
/// because it costs a whole sweep. [`ready`] sizes the load at the WIDEST
/// rung this run asks for, and that ceiling reaches
/// `kernels_metal::linear::quant::mb_rows` as its `capacity` — so a run told
/// `PIE_PROBE_LANES=2,3,4` loads a shell whose activation slot holds four
/// rows, `mb_rows` declines to pad a four-row fire up to a row block it
/// cannot write, and every rung of the "GEMM" column is silently the GEMV.
/// The column looks plausible and is a copy of the other arm. **Name a rung
/// at or above the row block under test** — the sweep behind
/// [`DeviceTuning::qmm_min_batch`] runs `2,3,4,5,6,7,8,16` for exactly this
/// reason, and the sixteen-lane row it collects is the control besides.
///
/// [`DeviceTuning::qmm_min_batch`]: kernels_metal::tuning::DeviceTuning::qmm_min_batch
///
/// **A KEY THIS SPELLS WRONG IS DROPPED IN SILENCE**, because
/// `engine_metal::boot` reads the table advisorily and a shared document is
/// not entitled to refuse a boot over a knob. That is precisely the failure
/// `kernels_metal::tuning`'s header records two false conclusions from, so
/// [`resolved`] prints what the table ACTUALLY says after the fold, and every
/// number below is to be read against that line rather than against this one.
fn stated_tuning() -> Option<String> {
    let stated = std::env::var("PIE_PROBE_TUNING").ok()?;
    let body: String = stated
        .split(',')
        .map(str::trim)
        .filter(|pair| !pair.is_empty())
        .map(|pair| format!("{pair}\n"))
        .collect();
    (!body.is_empty()).then_some(body)
}

/// The crossovers this run is ACTUALLY going to fire, read back out of the
/// frozen table — see [`stated_tuning`] for why a probe that only echoed what
/// it asked for would be the one mistake this knob exists to avoid.
fn resolved() -> String {
    let t = kernels_metal::tuning::current();
    format!(
        "qmm_min_batch {} (moe {}, emulated {}), moe_batch_min_per_expert {}, \
         moe_tile_mid_per {}, sdpa_tile_min_rows_per_request {}",
        t.qmm_min_batch,
        t.qmm_min_batch_moe,
        t.qmm_min_batch_emulated,
        t.moe_batch_min_per_expert,
        t.moe_tile_mid_per,
        t.sdpa_tile_min_rows_per_request,
    )
}

/// A contract lookup the boot door never reaches: the document above is opened
/// for its `[metal.tuning]` table alone, and every shell here is loaded through
/// [`Shell::load`] like every other shell in this directory.
fn no_door(
    _trace: &model_ir::Trace,
    _path: &Path,
) -> Result<checkpoint::contract::ModelContract, String> {
    Err("this door never loads".to_string())
}

/// The snapshot directory: the named override, or this row's own repository in
/// the cache.
///
/// Duplicated from `session_c_first_light` rather than shared, and deliberately:
/// a fixture two test binaries reach into is a file two lanes of work edit.
fn snapshot(row: &Sku) -> Option<PathBuf> {
    if let Ok(stated) = std::env::var(row.env) {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    // The suite runs as root over tailscale ssh, so `HOME` is not the owner's
    // — the cache the checkpoints actually live in is named beside it.
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let snaps = Path::new(home)
            .join(".cache/huggingface/hub")
            .join(row.repo)
            .join("snapshots");
        let mut found: Vec<PathBuf> = std::fs::read_dir(&snaps)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .filter(|path| !containers(path).is_empty())
            .collect();
        found.sort();
        found.into_iter().next()
    })
}

/// Every container of the snapshot, sorted — the plural because three of these
/// four checkpoints are sharded and a contract built over shard one alone
/// refuses at the first tensor that lives in shard two.
fn containers(snapshot: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(snapshot)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            let name = path.file_name()?.to_str()?;
            (name.ends_with(".safetensors") || name.ends_with(".zt")).then_some(path)
        })
        .collect();
    found.sort();
    found
}

/// The winner and **how much it won by** — the top-two logit gap.
///
/// The margin is what separates the two things an argmax disagreement can
/// mean. A step decided by five logits that comes out differently in a batched
/// fire is a wrong answer; a step decided by a hundredth of one is two
/// summation orders rounding a tie two ways, which is a fact about bf16 and
/// not about the window mechanism.
fn top2(logits: &[f32]) -> (u32, f32) {
    let mut best = f32::NEG_INFINITY;
    let mut second = f32::NEG_INFINITY;
    let mut winner = 0u32;
    for (at, &value) in logits.iter().enumerate() {
        if value > best {
            second = best;
            best = value;
            winner = at as u32;
        } else if value > second {
            second = value;
        }
    }
    (winner, best - second)
}

/// Finite, and not a rectangle nothing wrote.
fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    let bad = logits.iter().position(|value| !value.is_finite());
    assert!(
        bad.is_none(),
        "{what} logit {} is {}, and a single NaN means the whole row is noise",
        bad.unwrap_or(0),
        logits[bad.unwrap_or(0)],
    );
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which is a rectangle nothing wrote"
    );
}

/// A loaded shell at the ladder's widest rung, the prompts already encoded, or
/// `None` and a sentence naming what was missing.
///
/// **THE BUDGET IS THE LADDER'S, WHICH IS WHY IT IS AN ARGUMENT.** `max_lanes`
/// is the widest rung and `max_tokens` is the larger of that and the longest
/// prompt — a decode fire is one row per lane and a prefill fire is one lane's
/// whole prompt, and the arena is cut once at the maximum of the two.
fn ready(row: &Sku, rungs: &[u32]) -> Option<(Shell, Vec<Vec<u32>>)> {
    let sku = row.sku;
    if !engine_metal::device::present() {
        eprintln!("skipping {sku}: this machine publishes no Metal device");
        return None;
    }
    let Some(trace) = model::trace_of(sku) else {
        eprintln!("skipping {sku}: the catalog ships no row by that name");
        return None;
    };
    let Some(import) = model::import_of(sku) else {
        eprintln!("skipping {sku}: the catalog ships no import for that row");
        return None;
    };
    let Some(snapshot) = snapshot(row) else {
        eprintln!(
            "skipping {sku}: no snapshot of {} in the hugging face cache — name one in {}",
            row.repo, row.env
        );
        return None;
    };
    if !snapshot.join("tokenizer.json").exists() {
        eprintln!("skipping {sku}: {snapshot:?} ships no tokenizer beside its tensors");
        return None;
    }
    let &widest = rungs.last()?;

    let tokenizer = tokenizer::Tokenizer::from_file(&snapshot.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    // **CYCLED PAST SIXTEEN, AND THE ONE ARM THAT NEEDS THEM DISTINCT STOPS
    // THERE.** The ladder's wide rung is a throughput measurement, where two
    // lanes carrying one prompt are still two sequences in two slots at two
    // kv extents; [`batching_is_polymorphic`] runs at [`AT`] lanes or fewer,
    // which is inside the sixteen distinct ones.
    let prompts: Vec<Vec<u32>> = (0..widest as usize)
        .map(|lane| {
            let mut tokens = Vec::new();
            tokens.extend(row.bos);
            tokens.extend(tokenizer.encode(PROMPTS[lane % PROMPTS.len()]));
            tokens
        })
        .collect();
    let longest = prompts.iter().map(Vec::len).max().unwrap_or(1) as u32;

    let files = containers(&snapshot);
    let source = ztensor_compat::index_all(&files).expect("the checkpoint's shards open as one");
    let contract = import(&source)
        .unwrap_or_else(|why| panic!("{sku}'s import contract does not fit {snapshot:?}: {why}"));
    drop(source);

    if let Some(body) = stated_tuning() {
        let doc = format!("[metal.tuning]\n{body}");
        engine_metal::open(doc.as_bytes(), no_door).expect("the boot document opens");
    }

    let booted = Instant::now();
    // **A REFUSED LOAD IS A SKIP AND NOT A FAILURE**, which is the whole of
    // what "skip gracefully under pressure" can honestly mean from in here: a
    // 17 GiB weight table beside whatever else the box is doing either lands
    // or does not, and the device is the only thing that knows which.
    let shell = match Shell::load(Boot {
        trace: trace(Platform::Metal),
        contract: &contract,
        checkpoint: &snapshot,
        budget: Budget::new(widest, widest.max(longest)),
        profile: None,
        page_size: 16,
        context: CONTEXT,
        slots: widest,
        // **THE DEPTH THE RUNTIME RUNS AT** (article 1's floor): one step
        // executing while the next is already committed behind it. Every
        // number in the table below is taken at this depth; the lockstep arm
        // is a call ORDER on this same shell, which is what makes the delta a
        // measurement of run-ahead rather than of two different loads.
        runahead: Runahead::default(),
        residency: engine_metal::ResidencyPlan::default(),
    }) {
        Ok(shell) => shell,
        Err(why) => {
            eprintln!(
                "skipping {sku}: the load did not fit beside what this box is already \
                 holding — {why}"
            );
            return None;
        }
    };
    let (weights, arena, pools, inputs) = shell.footprint();
    eprintln!(
        "loaded {sku} on {} in {:.1}s — weights {:.2} GiB, arena {:.1} MiB, pools {:.1} MiB, \
         inputs {:.1} MiB, {} frames in flight\n  tuning: {}",
        shell.device_name(),
        booted.elapsed().as_secs_f64(),
        weights as f64 / (1 << 30) as f64,
        arena as f64 / (1 << 20) as f64,
        pools as f64 / (1 << 20) as f64,
        inputs as f64 / (1 << 20) as f64,
        shell.frames_in_flight(),
        resolved(),
    );
    Some((shell, prompts))
}

// ─────────────────────────────────────────────────────────────────────────────
// The fire path, as the runtime spells it
// ─────────────────────────────────────────────────────────────────────────────

/// **ONE STEP, ENQUEUED AND FILED, WITH NOTHING WAITED FOR.**
///
/// The three phases, called where `Metal::submit` calls them: `prepare` makes
/// every host decision, `enqueue` encodes and commits, `settle` files the step
/// on the in-flight ring. Nothing in here synchronizes — the `Landed` it
/// answers is a ticket, and [`Shell::rows_of`] is where a wait can happen.
fn submit(shell: &mut Shell, lanes: &[Seated<'_>]) -> Landed {
    use engine::frame::Shell as FramePhases;
    let prepared = FramePhases::prepare(
        shell,
        StepView {
            lanes,
            attachments: &[],
            done: None,
        },
        None,
    )
    .expect("the step prepares");
    let enqueued = FramePhases::enqueue(shell, prepared).expect("the step enqueues");
    FramePhases::settle(shell, enqueued).expect("the step files its settlement")
}

/// Open `lanes` slots and seat each one with its own prompt — one prefill fire
/// per lane, because a probe of DECODE throughput has no reason to batch the
/// prefills and every reason to keep the arena small.
///
/// `open` is what clears a slot's recurrent banks and resets its extent, so a
/// re-seated lane is a new sequence and not a continuation (palo build log 19).
fn seat(shell: &mut Shell, row: &Sku, prompts: &[Vec<u32>], lanes: u32) {
    for slot in 0..lanes {
        shell.open(slot).expect("the slot opens");
        let prompt = &prompts[slot as usize];
        let fired = shell
            .fire(&[Lane {
                slot,
                word: (row.word)(prompt.len() as u32),
                tokens: prompt,
            }])
            .unwrap_or_else(|why| panic!("lane {slot}'s prefill fires: {why}"));
        finite(&fired[0], &format!("lane {slot}'s prefill"));
    }
}

/// What one warm window measured.
struct Window {
    /// Wall time from the first submit to the last harvest.
    seconds: f64,
    /// The most steps that were airborne at once — sampled on the host between
    /// a submit and the harvest that follows it. **Two is article 1's claim
    /// made observable**; one means the window ran in lockstep whatever it
    /// was asked for.
    airborne: usize,
    /// Encodes the last fire put on the device, off `Shell::last_fire`.
    launches: u32,
}

/// **A WARM WINDOW OF `steps` DECODE FIRES OVER `lanes` SEATED LANES.**
///
/// `runahead` is how many steps may be in the air before the harvest catches
/// up: `1` is what the runtime does at the default depth, `0` is the lockstep
/// spelling — submit, wait, submit, wait — which is F1's shape written as a
/// call order.
///
/// Each lane is fed a rotation over its own prompt's tokens rather than its
/// own argmax; the module header is where that choice is argued.
fn window(
    shell: &mut Shell,
    row: &Sku,
    prompts: &[Vec<u32>],
    lanes: u32,
    steps: usize,
    runahead: usize,
) -> Window {
    let word = (row.word)(1);
    let mut flying: Vec<Landed> = Vec::with_capacity(runahead + 1);
    let mut airborne = 0usize;
    let mut last: Option<Vec<Vec<f32>>> = None;

    let at = Instant::now();
    for step in 0..steps {
        let fed: Vec<[u32; 1]> = (0..lanes)
            .map(|slot| {
                let prompt = &prompts[slot as usize];
                [prompt[step % prompt.len()]]
            })
            .collect();
        let seated: Vec<Seated<'_>> = (0..lanes as usize)
            .map(|slot| {
                Seated::of(Lane {
                    slot: slot as u32,
                    word,
                    tokens: &fed[slot],
                })
            })
            .collect();
        flying.push(submit(shell, &seated));
        // Sampled HERE, on the host, with nothing waited for — the same
        // observation `engine-cuda`'s `two_frames_in_flight` takes off its own
        // airborne counter.
        airborne = airborne.max(shell.airborne_steps());
        if flying.len() > runahead {
            let landed = flying.remove(0);
            last = Some(
                shell
                    .rows_of(&landed)
                    .unwrap_or_else(|why| panic!("step {step}'s rows come back: {why}")),
            );
        }
    }
    for landed in flying {
        last = Some(
            shell
                .rows_of(&landed)
                .unwrap_or_else(|why| panic!("the window's tail comes back: {why}")),
        );
    }
    let seconds = at.elapsed().as_secs_f64();

    // **CHECKED AFTER THE CLOCK STOPS**, because a scan of every logit of
    // every lane is host work of the same order as the fire that produced it.
    // One step's worth is enough to catch the failure this is here for: a
    // residency flake reads as zeros in every lane at once, not in one.
    let rows = last.expect("a window of one or more steps harvested something");
    assert_eq!(
        rows.len(),
        lanes as usize,
        "{} lanes in, {} rows of logits out",
        lanes,
        rows.len()
    );
    for (slot, logits) in rows.iter().enumerate() {
        finite(logits, &format!("lane {slot} at {lanes} lanes"));
    }

    Window {
        seconds,
        airborne,
        launches: shell.last_fire().launches,
    }
}

/// Which arm this fire's mixture took, for a row that has one — `None` for a
/// dense stack, which has no such choice to report.
///
/// Read out of `kernels-metal`'s own selection rather than restated: the same
/// `should_batch` the dispatch calls, against the same process-wide tuning
/// table the device seated at load.
fn mixture(row: &Sku, lanes: u32) -> Option<String> {
    let (experts, top_k) = row.routed?;
    let tuning = kernels_metal::tuning::current();
    let pairs = lanes * top_k;
    let batched =
        kernels_metal::linear::moe::should_batch(pairs, experts, tuning.moe_batch_min_per_expert);
    let tile = kernels_metal::linear::moe::tile_rows(pairs, experts, &tuning);
    Some(match batched {
        true => format!("sorted, {pairs} pairs, tile {tile}"),
        false => format!("per-row, {pairs} pairs"),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// The sweep
// ─────────────────────────────────────────────────────────────────────────────

/// **THE LADDER, THE PIPELINING DELTA, AND THE INVARIANT — ONE LOADED SHELL.**
///
/// One load per row and one row per test, because the two big checkpoints are
/// most of the machine and because a second load to measure a second arm would
/// be measuring two loads.
fn probe(row: &Sku) {
    let _serial = serialized();
    let sku = row.sku;
    if !selected(row) {
        eprintln!(
            "skipping {sku}: {}",
            match std::env::var("PIE_PROBE_SKUS") {
                Ok(stated) => format!("PIE_PROBE_SKUS={stated} does not name `{}`", row.name),
                Err(_) => format!(
                    "this row is opt-in — run it with PIE_PROBE_SKUS={} (it holds most of a \
                     32 GiB box for the length of a sweep)",
                    row.name
                ),
            }
        );
        return;
    }
    let rungs = ladder(row);
    if rungs.is_empty() {
        eprintln!("skipping {sku}: PIE_PROBE_LANES named no rung this row admits");
        return;
    }
    let steps = steps();
    let Some((mut shell, prompts)) = ready(row, &rungs) else {
        return;
    };

    // ── The ladder.
    eprintln!(
        "\n{sku} — {steps} warm decode fires per rung, {} frames in flight, {} tokens of \
         context per lane\n  {:>4}  {:>10}  {:>12}  {:>14}  {:>8}  {:>9}{}",
        shell.frames_in_flight(),
        CONTEXT,
        "N",
        "ms/fire",
        "tok/s/lane",
        "aggregate tok/s",
        "airborne",
        "encodes",
        if row.routed.is_some() { "  mixture" } else { "" },
    );
    let mut aggregate: Vec<(u32, f64)> = Vec::with_capacity(rungs.len());
    for &lanes in &rungs {
        seat(&mut shell, row, &prompts, lanes);
        window(&mut shell, row, &prompts, lanes, WARMUP, 1);
        // Re-seated between the warm-up and the window so that every rung's
        // sequences are the same length when the clock starts — a kv extent
        // is what an attention arm's row count is, and a window measured over
        // longer sequences than its neighbour is not comparable to it.
        seat(&mut shell, row, &prompts, lanes);
        let measured = window(&mut shell, row, &prompts, lanes, steps, 1);
        let per_fire = measured.seconds * 1000.0 / steps as f64;
        let tokens = f64::from(lanes) * steps as f64 / measured.seconds;
        aggregate.push((lanes, tokens));
        eprintln!(
            "  {lanes:>4}  {per_fire:>10.2}  {:>12.1}  {tokens:>14.1}  {:>8}  {:>9}{}",
            tokens / f64::from(lanes),
            measured.airborne,
            measured.launches,
            match mixture(row, lanes) {
                Some(arm) => format!("  {arm}"),
                None => String::new(),
            },
        );
    }

    // ── The pipelining delta, at the widest rung the ladder reached that is
    //    not wider than [`AT`].
    // Not necessarily a rung of the ladder — a slot count is admissible if the
    // load seated it, and the ladder's widest rung is what the load was sized
    // at. Clipped to [`AT`] so the arm stays inside the distinct prompts.
    let at = rungs.last().copied().expect("the ladder has a rung").min(AT);
    seat(&mut shell, row, &prompts, at);
    window(&mut shell, row, &prompts, at, WARMUP, 1);
    seat(&mut shell, row, &prompts, at);
    let pipelined = window(&mut shell, row, &prompts, at, steps, 1);
    seat(&mut shell, row, &prompts, at);
    let lockstep = window(&mut shell, row, &prompts, at, steps, 0);
    let rate = |measured: &Window| f64::from(at) * steps as f64 / measured.seconds;
    eprintln!(
        "\n{sku} — article 1 at {at} lanes: F2 (a step of run-ahead) {:.2} ms/fire, \
         {:.1} tok/s, peak airborne {} | F1 (lockstep) {:.2} ms/fire, {:.1} tok/s, peak \
         airborne {} | delta {:+.1}%",
        pipelined.seconds * 1000.0 / steps as f64,
        rate(&pipelined),
        pipelined.airborne,
        lockstep.seconds * 1000.0 / steps as f64,
        rate(&lockstep),
        lockstep.airborne,
        100.0 * (rate(&pipelined) / rate(&lockstep) - 1.0),
    );

    // ── The invariant.
    batching_is_polymorphic(&mut shell, row, &prompts, at);

    // ── And the one thing this file asserts about a NUMBER. Everything above
    //    is printed; this is the claim that the printing is worth reading.
    let single = aggregate
        .iter()
        .find(|&&(lanes, _)| lanes == 1)
        .map(|&(_, tokens)| tokens);
    let (best_at, best) = aggregate
        .iter()
        .copied()
        .fold((0u32, 0.0f64), |best, rung| if rung.1 > best.1 { rung } else { best });
    if let Some(single) = single {
        eprintln!(
            "{sku}: best aggregate {best:.1} tok/s at {best_at} lanes, {:.2}x the single \
             lane's {single:.1}",
            best / single,
        );
        assert!(
            best >= single * MIN_GAIN,
            "{sku}: the best rung of the ladder turned in {best:.1} tok/s against \
             {single:.1} at one lane, which is {:.2}x — batching a decode is supposed to \
             ride the same weight read, so a fleet that gains less than {MIN_GAIN:.2}x has \
             serialized its lanes somewhere",
            best / single,
        );
    }
}

/// **THE POLYMORPHIC-BATCHING INVARIANT, AT `lanes` LANES** — design §0's
/// headline case, widened.
///
/// Every lane's greedy continuation in an N-lane fire, against the same lane's
/// continuation when it is the only lane in the fire. Argmax equality per lane
/// and per step: a batched fire is a different rectangle, tiled differently by
/// launches of a different shape, so bit equality with a solo fire is not a
/// property this shell promises — and the token is what a serving fleet
/// actually publishes.
///
/// What this separates, that a one-lane smoke cannot: a window whose row
/// interval is off by a lane, a page table indexed by the fire's seriated
/// order rather than the submitted one, a mask plane addressed at lane zero
/// for every lane, and a kv extent taken from the widest lane instead of each.
fn batching_is_polymorphic(shell: &mut Shell, row: &Sku, prompts: &[Vec<u32>], lanes: u32) {
    let word = (row.word)(1);

    // Alone: one lane, in one slot, re-opened each time. `top2` rather than
    // `argmax` because a disagreement is only readable beside the margin that
    // produced it — see [`TIE`].
    let solo: Vec<Vec<(u32, f32)>> = (0..lanes as usize)
        .map(|lane| {
            let prompt = &prompts[lane];
            shell.open(0).expect("slot 0 opens");
            let prefill = shell
                .fire(&[Lane {
                    slot: 0,
                    word: (row.word)(prompt.len() as u32),
                    tokens: prompt,
                }])
                .expect("the solo prefill fires");
            let mut said = vec![top2(&prefill[0])];
            for _ in 0..CHECK {
                let fed = [said.last().expect("a step feeds the last token back").0];
                let step = shell
                    .fire(&[Lane {
                        slot: 0,
                        word,
                        tokens: &fed,
                    }])
                    .expect("the solo decode fires");
                said.push(top2(&step[0]));
            }
            said
        })
        .collect();

    // Together: the same prompts, one lane per slot, one fire per step.
    let mut batched: Vec<Vec<(u32, f32)>> = (0..lanes as usize)
        .map(|lane| {
            let prompt = &prompts[lane];
            shell.open(lane as u32).expect("the slot opens");
            let prefill = shell
                .fire(&[Lane {
                    slot: lane as u32,
                    word: (row.word)(prompt.len() as u32),
                    tokens: prompt,
                }])
                .expect("the batched prefill fires");
            vec![top2(&prefill[0])]
        })
        .collect();
    for _ in 0..CHECK {
        let fed: Vec<[u32; 1]> = batched
            .iter()
            .map(|said| [said.last().expect("a step feeds back").0])
            .collect();
        let seated: Vec<Seated<'_>> = (0..lanes as usize)
            .map(|lane| {
                Seated::of(Lane {
                    slot: lane as u32,
                    word,
                    tokens: &fed[lane],
                })
            })
            .collect();
        let fired = shell
            .fire_seated(&seated)
            .unwrap_or_else(|why| panic!("the {lanes}-lane decode fires: {why}"));
        assert_eq!(fired.len(), lanes as usize, "one row of logits per lane");
        for (lane, logits) in fired.iter().enumerate() {
            batched[lane].push(top2(logits));
        }
    }

    // **WHERE THEY PART, AND BY HOW MUCH THAT STEP WAS DECIDED.** Only the
    // FIRST divergence is a statement about anything: past it the two arms are
    // decoding different sentences, and two different sentences continuing
    // differently is not news.
    let mut parted: Vec<(usize, usize, f32)> = Vec::new();
    for lane in 0..lanes as usize {
        let Some(step) = (0..=CHECK).find(|&step| solo[lane][step].0 != batched[lane][step].0)
        else {
            continue;
        };
        let margin = solo[lane][step].1.min(batched[lane][step].1);
        eprintln!(
            "  lane {lane} parts at step {step}: alone {} by {:.4}, batched {} by {:.4}",
            solo[lane][step].0,
            solo[lane][step].1,
            batched[lane][step].0,
            batched[lane][step].1,
        );
        eprintln!(
            "    alone   {:?}\n    batched {:?}",
            solo[lane].iter().map(|said| said.0).collect::<Vec<_>>(),
            batched[lane].iter().map(|said| said.0).collect::<Vec<_>>(),
        );
        parted.push((lane, step, margin));
    }
    let tightest = solo
        .iter()
        .flatten()
        .chain(batched.iter().flatten())
        .map(|said| said.1)
        .fold(f32::INFINITY, f32::min);
    eprintln!(
        "{}: {} of {lanes} lanes said token for token what they say alone over {CHECK} greedy \
         steps; the tightest step either arm decided was won by {tightest:.4}",
        row.sku,
        lanes as usize - parted.len(),
    );

    // **THE GATE, AND IT IS NOT `parted.is_empty()`.** A batched fire is a
    // different rectangle tiled by launches of a different shape, so the
    // arithmetic is a different summation order and the LOGITS are not
    // promised equal — `serve_smoke`'s two-lane golden says so in its own
    // words. What is promised is that no step the model actually decided comes
    // out the other way, which is [`TIE`]'s line.
    let real: Vec<(usize, usize, f32)> = parted
        .iter()
        .copied()
        .filter(|&(_, _, margin)| margin > TIE)
        .collect();
    assert!(
        real.is_empty(),
        "{}: {real:?} — (lane, step, margin) triples where a {lanes}-lane fire answered \
         differently from the same lane alone at a step that was NOT a tie. Above {TIE:.2} \
         logits the two arms are supposed to agree, and a lane that does not is a window \
         reading rows that are not its own",
        row.sku,
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The arms
// ─────────────────────────────────────────────────────────────────────────────

/// **THE VEHICLE, ALL THE WAY UP THE LADDER.** 0.8B at 4 bits: the only row
/// here whose ceiling is thirty-two lanes.
#[test]
fn qwen35_0p8b_four_bit_decodes_on_many_lanes_at_once() {
    probe(&U4);
}

/// **THE MIXTURE.** gpt-oss-20b, whose routed FFN changes arm partway up this
/// ladder — see [`Sku::routed`] and the mixture column.
#[test]
fn gpt_oss_20b_decodes_on_many_lanes_at_once() {
    probe(&GPT_OSS);
}

/// **QWEN3.6-27B**, opt-in: `PIE_PROBE_SKUS=qwen36`.
#[test]
fn qwen36_27b_decodes_on_many_lanes_at_once() {
    probe(&QWEN36);
}

/// **GEMMA-4-31B**, opt-in: `PIE_PROBE_SKUS=gemma4`.
#[test]
fn gemma4_31b_decodes_on_many_lanes_at_once() {
    probe(&GEMMA4);
}
