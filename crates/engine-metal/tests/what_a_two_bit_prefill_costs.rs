//! **THE TWO NUMBERS THE `WIDTHS` QUESTION IS DECIDED ON.**
//!
//! `kernels_metal::linear::quant::WIDTHS` gates which bit widths the TILED
//! qmm family may be composed at. Two is not on it, so
//! `linear::moe::batched_point` DECLINES a 2-bit routed bank and the prefill
//! falls through to the per-row matvec arm beside it. Flipping the constant
//! to `[2, 4, 8]` buys the tile family for 2-bit prefill and costs whatever
//! the extra points cost to compile. Both halves are measurements and this
//! file is where they are taken.
//!
//! # The arms
//!
//! 1. **[`what_the_composed_census_costs`]** — compiles every point
//!    [`kernels_metal::linear::quant::composed`] names, from a cold
//!    `Pipelines`, and prints the count and the wall. The census is the
//!    WIDTH LIST walked, so this arm's number changes with the constant and
//!    the difference between two builds is the flip's compile bill.
//!
//! 2. **[`what_a_long_two_bit_prefill_costs`]** — prefills [`TOKENS`] tokens
//!    through the dsv4-flash 2-bit miniature, [`RUNS`] times, and prints the
//!    median. It also prints `Shell::compiled()` across the first prefill,
//!    which is how a reader can tell WHICH ARM FIRED without a debugger: the
//!    tiled points are jit-stamped, so a prefill that takes the tile family
//!    mints pipelines it did not have and one that declines mints none.
//!
//! Neither arm is a gate. They print; the only assertions are that the
//! numbers are numbers, because a probe that passes while measuring nothing
//! is worse than a red one.
//!
//! # Why a file of its own and not an arm of `two_bit_moe_first_light`
//!
//! That file fires an eight-token prompt, deliberately — it is asking whether
//! a 2-bit checkpoint loads and returns finite logits, and a long prefill
//! would only make the question slower to answer. The tile/matvec crossover
//! is a WIDE-M question and is invisible at eight rows. Separate files
//! because separate claims, and this one holds the same shell for both.
//!
//! ```text
//! cargo test -p engine-metal --release --test what_a_two_bit_prefill_costs \
//!     -- --nocapture --test-threads=1
//! ```
//!
//! | variable | what it moves |
//! |---|---|
//! | `PIE_U2_PREFILL` | prompt tokens in one prefill (default [`TOKENS`]) |
//! | `PIE_U2_RUNS` | timed prefills, median reported (default [`RUNS`]) |
//! | `PIE_U2_SNAPSHOT` | where the dsv4 2-bit snapshot lives |
//! | `PIE_QWEN4_U2_SNAPSHOT` | where the qwen3.8 2-bit snapshot lives |
//!
//! Arm 2 has two rows now, one per miniature — `what_a_long_two_bit_prefill_costs`
//! for dsv4 and `what_a_long_two_bit_qwen38_prefill_costs` for qwen3.8. Both
//! read the same two knobs, so a sweep over prompt lengths names a length and
//! runs the file, and `--test-threads=1` is what keeps the two shells from
//! being resident at once.

#![cfg(target_vendor = "apple")]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_metal::device::{self, Context, Pipelines};
use engine_metal::{Boot, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

/// **ONE OF THE TWO 2-BIT MINIATURES**, everything that differs between them.
///
/// The arm below was pinned to the dsv4 row for as long as that was the only
/// 2-bit miniature in the tree. There are two now and they answer the WIDTHS
/// question from opposite ends — dsv4 routes six of sixteen experts over five
/// layers, qwen3.8 routes ten of sixteen over four beside an n-gram PLE — so
/// a probe that can only fire one of them prices half the question.
struct Mini {
    /// The short name this row prints its numbers under, so a log holding
    /// both sweeps says which miniature each line came from.
    name: &'static str,
    /// The catalog row, spelled as the catalog spells it.
    sku: &'static str,
    /// The `models--*` directory in the hugging face cache, exactly.
    repo: &'static str,
    /// The environment variable that overrides the snapshot directory.
    env: &'static str,
    /// The lane word this family's own `Classify` computes.
    word: fn(u32) -> u64,
}

/// `two_bit_moe_first_light`'s SKU — five layers, sixteen experts routed six
/// ways.
const DSV4: Mini = Mini {
    name: "dsv4",
    sku: "dsv4-flash-mlxu2-kv-bf16",
    repo: "models--mlx-community--DeepSeek-V4-Flash-2bit-DQ",
    env: "PIE_U2_SNAPSHOT",
    word: |query_len| {
        models::deepseek_v4::forward::Facts::of(&Request::new(query_len, false)).word()
    },
};

/// `qwen4_two_bit_first_light`'s SKU — four layers, sixteen experts routed ten
/// ways, the PLE split eight parts. Its fan-out is the wider of the two, so
/// it enters tile country at fewer rows than [`DSV4`] does.
const QWEN38: Mini = Mini {
    name: "qwen38",
    sku: "qwen38-flash-mlxu2-kv-bf16",
    repo: "models--Sawfwair--Qwen3.8-Flash-Next-MLX-Mixed-2bit",
    env: "PIE_QWEN4_U2_SNAPSHOT",
    word: |query_len| models::qwen_4::forward::Facts::of(&Request::new(query_len, false)).word(),
};

/// **PROMPT TOKENS IN ONE PREFILL.**
///
/// Wide enough that the routed rectangle is unambiguously in tile country:
/// this miniature routes sixteen experts, and `linear::moe::tile_rows` takes
/// the widest tile once a batch carries `moe_tile_wide_per` rows per expert.
/// Eight tokens — what first light fires — is the narrow-M regime where the
/// vector arm wins on any width, so a comparison there would answer a
/// question nobody asked.
const TOKENS: usize = 512;

/// Timed prefills per run. The median of three is the campaign's convention;
/// the first fire is not one of them, it is the warm-up.
const RUNS: usize = 3;

/// The context the shell is loaded with — the prefill plus room for the
/// tokens a decode would append, so the prefill is never the thing that
/// overruns the pool.
const CONTEXT: u32 = 1024;

/// One shell at a time per process: these hold the whole weight table
/// resident and the measurements are only readable one at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A `usize` knob off the environment, or its default.
fn knob(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|text| text.trim().parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

/// The middle of a sorted copy — the campaign reports medians, not means,
/// because one scheduling hiccup moves a mean and does not move this.
fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

// ─────────────────────────────────────────────────────────────────────────────
// Arm 1: what the census costs to compile
// ─────────────────────────────────────────────────────────────────────────────

/// **THE COMPILE BILL OF [`kernels_metal::linear::quant::composed`].**
///
/// Every point the width list reaches, compiled once from a cold cache. The
/// stamped ones each mint their own library — the stamp is part of the source
/// and `Pipelines` keys the library cache on it — so this is not one compile
/// with many lookups, it is one compile per stamped point, and that is
/// exactly the bill a warm-up census would pay.
#[test]
fn what_the_composed_census_costs() {
    let _serial = serialized();
    if !device::present() {
        println!("SKIP the census bill: this machine publishes no Metal device");
        return;
    }
    let device = Context::bind().expect("the device binds");
    let census = kernels_metal::linear::quant::composed();
    let points = census.len();
    assert!(points > 0, "the width list reaches at least one point");

    let pipelines = Pipelines::new();
    let at = Instant::now();
    for fire in census {
        let entry = fire.entrypoint;
        pipelines
            .warm(&device, fire)
            .unwrap_or_else(|why| panic!("`{entry}` compiles: {why}"));
    }
    let wall = at.elapsed().as_secs_f64();
    assert_eq!(
        pipelines.compiled(),
        points as u64,
        "every census point compiled exactly once"
    );
    println!(
        "census: {points} points in {wall:.2}s ({:.0} ms/point) on {}",
        wall * 1000.0 / points as f64,
        device.name(),
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Arm 2: what a long 2-bit prefill costs
// ─────────────────────────────────────────────────────────────────────────────

/// The snapshot: the checkpoint AND the tokenizer beside it.
/// `PIE_U2_SNAPSHOT` overrides where it is looked for.
fn snapshot(mini: &Mini) -> Option<PathBuf> {
    if let Ok(stated) = std::env::var(mini.env) {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    let usable = |path: &Path| path.join("tokenizer.json").exists() && !containers(path).is_empty();
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let snapshots = Path::new(home)
            .join(".cache/huggingface/hub")
            .join(mini.repo)
            .join("snapshots");
        let mut found: Vec<PathBuf> = std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .filter(|path| usable(path))
            .collect();
        found.sort();
        found.into_iter().next()
    })
}

/// **EVERY** container of the snapshot, sorted — the plural is the whole
/// point, and it is `throughput_probe::containers`' paragraph for the same
/// reason: dsv4's miniature is one file and qwen3.8's is two, so a contract
/// read over shard one alone refuses at the first tensor that lives in shard
/// two. It refused at
/// `language_model.model.layers.2.linear_attn.in_proj_qkv.weight`, which is
/// exactly where shard one ends.
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

/// A prompt of exactly `tokens` ids, built by cycling a real encoding.
///
/// Real ids and not a counter: an id past the vocabulary is an embed gather
/// out of bounds, and this probe is about the routed GEMM and not about
/// finding that out the hard way.
fn prompt_of(seed: &[u32], tokens: usize) -> Vec<u32> {
    assert!(!seed.is_empty(), "the seed prompt encodes to something");
    (0..tokens).map(|at| seed[at % seed.len()]).collect()
}

#[test]
fn what_a_long_two_bit_prefill_costs() {
    prices(&DSV4);
}

/// **THE TWIN.** Same rectangle, the other miniature — see [`QWEN38`].
#[test]
fn what_a_long_two_bit_qwen38_prefill_costs() {
    prices(&QWEN38);
}

/// One miniature's prefill, priced.
fn prices(mini: &Mini) {
    let _serial = serialized();
    let (name, sku) = (mini.name, mini.sku);
    let what = format!("the {name} 2-bit prefill probe");
    if !device::present() {
        println!("SKIP {what}: this machine publishes no Metal device");
        return;
    }
    let Some(checkpoint) = snapshot(mini) else {
        println!(
            "SKIP {what}: no {} snapshot with a tokenizer beside it under \
             $HOME/.cache/huggingface/hub — name one in {}",
            mini.repo, mini.env
        );
        return;
    };
    let files = containers(&checkpoint);
    if files.is_empty() {
        println!("SKIP {what}: {checkpoint:?} holds no tensor container");
        return;
    }

    let tokens = knob("PIE_U2_PREFILL", TOKENS);
    let runs = knob("PIE_U2_RUNS", RUNS);
    assert!(
        tokens < CONTEXT as usize,
        "a prefill of {tokens} does not fit a context of {CONTEXT}"
    );

    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let seed = tokenizer.encode("The capital of France is the city of");
    let prompt = prompt_of(&seed, tokens);

    let trace = models::trace_of(sku).expect("the catalog ships the 2-bit SKU");
    let trace = trace(Platform::Metal);
    let source = ztensor_compat::index_all(&files).expect("the checkpoint's shards open as one");
    let contract =
        models::import_of(sku).expect("the catalog ships an import for the SKU")(&source)
            .expect("the 2-bit SKU's import contract fits the real checkpoint");
    drop(source);

    let booted = Instant::now();
    let mut shell = Shell::load(Boot {
        trace,
        contract: &contract,
        checkpoint: &checkpoint,
        // §M-4c, as `serve_smoke` states it: an unstamped snapshot proceeds,
        // and the deployment's facts are stated honestly all the same.
        tp_size: 1,
        precision: models::precision_of(sku)
            .expect("the catalog states this row's precision")
            .to_string(),
        budget: Budget::new(2, CONTEXT),
        patches: None,
        profile: None,
        page_size: 16,
        context: CONTEXT,
        slots: 2,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
    })
    .expect("the 2-bit shell loads");
    let load = booted.elapsed().as_secs_f64();
    println!(
        "loaded {sku} on {} in {load:.1}s — {} points compiled at load",
        shell.device_name(),
        shell.compiled(),
    );

    // The warm-up fire, which is also the census reading: whatever the
    // prefill class needs and the load did not already hold is minted HERE,
    // and the tiled qmm points are exactly the kind of thing that is minted
    // rather than held, because they are jit-stamped.
    let before = shell.compiled();
    let at = Instant::now();
    fire(mini, &mut shell, 0, &prompt);
    let first = at.elapsed().as_secs_f64() * 1000.0;
    let minted = shell.compiled() - before;
    println!(
        "{name}: first prefill of {tokens} tokens: {first:.1} ms, {minted} new points minted \
         (the arm that fired is the one that needed them)"
    );

    let mut millis = Vec::with_capacity(runs);
    for _ in 0..runs {
        let at = Instant::now();
        fire(mini, &mut shell, 0, &prompt);
        millis.push(at.elapsed().as_secs_f64() * 1000.0);
    }
    let warm = shell.compiled() - before - minted;
    assert_eq!(warm, 0, "a warm prefill compiles nothing");
    let mid = median(millis.clone());
    println!(
        "{name}: prefill x{runs} of {tokens} tokens: median {mid:.1} ms ({:.0} tok/s), \
         samples {millis:.1?}",
        tokens as f64 * 1000.0 / mid,
    );
    assert!(mid.is_finite() && mid > 0.0, "the median is a duration");
}

/// One prefill into a freshly opened slot, its logits checked for being
/// numbers at all.
fn fire(mini: &Mini, shell: &mut Shell, slot: u32, prompt: &[u32]) {
    shell.open(slot).expect("the slot opens");
    let out = shell
        .fire(&[Lane {
            slot,
            word: (mini.word)(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    assert_eq!(out.len(), 1, "one lane in, one row of logits out");
    assert!(
        out[0].iter().all(|value| value.is_finite()),
        "a prefill that returns a NaN is not a prefill this probe timed"
    );
}
