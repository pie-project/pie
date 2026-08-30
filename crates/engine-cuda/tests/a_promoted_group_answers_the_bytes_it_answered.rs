//! **B7: a packed group changes tier while the model serves, and answers the
//! same bytes on the other side** (alto streaming §3 item 3; next.md B7).
//!
//! W-5 seated a split-plane bank WHOLE — codes and exponents on one tier — and
//! bought the torn-pair property the cheap way: *a group never changes tier
//! after the load*. W-1 gave that seating a third rung, the mapped artifact,
//! and streaming §3 item 3 wrote down what was still missing in one sentence:
//! **a packed group has no promotion at all**, because `moe_matmul_select_mxfp4`
//! computes each plane's expert base itself and dereferences no table, so the
//! two plane addresses were kernel PARAMETERS and a captured graph holds its
//! parameters forever.
//!
//! The kernel change it named is one load. A streamed group's launch carries a
//! CELL — one 16-byte, 16-byte-aligned word of data at a fixed address holding
//! `(codes, scales)` — and the select reads it with a single
//! `ld.global.v2.u64` before it does the arithmetic it always did. That is one
//! extra load per GROUP per LAUNCH, at one address the whole grid shares; a
//! fully-resident load passes a null cell and pays nothing.
//!
//! ```text
//! (a) THE COUNTERS MOVE. Every packed group now `atomicAdd`s a `u32` once per
//!     routed row per fire, the settle-side readback carries it out, and
//!     `Shell::expert_residency` reports it. Zero before the fires, non-zero
//!     after, and strictly greater after more.
//! (b) A GROUP CHANGES RUNG. Driven through `Shell::promote_group`, a group on
//!     the mapped artifact takes a berth on a faster tier and the group that
//!     held it goes back to the file. `Shell::group_ladder` counts both, and
//!     the residency report's `held` says so of each by name.
//! (c) AND IT ANSWERS THE SAME BYTES. The same prompt, after the ladder has
//!     moved groups in both directions, produces the uncapped load's greedy
//!     tokens and the uncapped load's logits — bit for bit, not nearly. The
//!     pair is one word and the copy lands before the pointer flips, so there
//!     is no state in which a group's codes have moved and its exponents have
//!     not.
//! ```
//!
//! # What is measured and NOT gated
//!
//! Step time, three ways: uncapped, three-tier as the plan seated it, and
//! three-tier after the ladder has walked. The third is expected to be a WASH
//! against the second, and the reason is arithmetic rather than disappointment:
//! the number of berths is fixed by the two budgets, so a swap reassigns which
//! group is on which rung and can never make more fast memory. With uniform
//! demand — every routed bank of gpt-oss-20b is read by every step — any
//! assignment costs the same, which is exactly why
//! `experts::Tier::decide_group`'s strict-improvement rule declines to move
//! anything on its own and why (b) is driven through a door rather than waited
//! for. The ladder pays where demand is NOT uniform: a bank a session never
//! routes to, holding a rung a hot mapped bank wants.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!     --test a_promoted_group_answers_the_bytes_it_answered -- --ignored --nocapture
//! ```
//!
//! # Gating and cost
//!
//! `#[ignore]`d, and it wants what W-1 wants: a CUDA device reporting
//! `pageableMemoryAccess` with ~15 GiB free, the gpt-oss-20b snapshot, and
//! ~15 GiB of scratch disk for the artifact it writes. It loads the model
//! twice, sequentially. Skips with a sentence when any of that is missing.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use engine_cuda::experts::{Budgets, Held, Plan};
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "gptoss-20b-bf16-mxfp4-kv-bf16";

/// **THE TWO CEILINGS**, W-1's exactly: six GiB over a ~12.8 GiB table, so
/// that all three rungs of the ladder carry groups and there is something on
/// the file for the ladder to lift.
const DEVICE: u64 = 4 << 30;
const HOST: u64 = 2 << 30;

const PROMPT: &str = "<|start|>user<|message|>What is the capital of France? \
                      Answer in one word.<|end|>\
                      <|start|>assistant<|channel|>final<|message|>";

/// Long enough that the counters separate from zero twice and that a step time
/// is an average rather than a sample of the first cold fire.
const STEPS: usize = 12;

/// **How many rungs the gate drives.** Each one is ~265 MiB across PCIe and a
/// synchronize, so the claim is stated in a handful rather than in all 48.
const RUNGS: usize = 3;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

fn serialized() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(PoisonError::into_inner)
}

fn word(query_len: u32) -> u64 {
    model::gpt_oss::forward::Facts::of(&Request::new(query_len, false)).word()
}

fn snapshot() -> Option<PathBuf> {
    for key in ["PIE_GPTOSS_SNAPSHOT", "PIE_MASKLESS_SNAPSHOT"] {
        if let Ok(stated) = std::env::var(key) {
            let path = PathBuf::from(stated);
            return path.is_dir().then_some(path);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        Path::new(&home).join(".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .filter_map(|entry| Some(entry.ok()?.path()))
        .find(|path| path.join("tokenizer.json").exists())
}

fn shards(snapshot: &Path) -> Vec<PathBuf> {
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

/// A scratch weight-cache directory of this process's own, removed at the end.
struct Cache(PathBuf);

impl Drop for Cache {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn cache() -> Cache {
    let dir = std::env::temp_dir().join(format!("pie-b7-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    Cache(dir)
}

struct Rig {
    trace: model_ir::Trace,
    contract: checkpoint::contract::ModelContract,
    checkpoint: PathBuf,
    tokenizer: tokenizer::Tokenizer,
}

fn rig(what: &str) -> Option<Rig> {
    if !engine_cuda::device::present() {
        eprintln!("skipping {what}: no CUDA device on this machine");
        return None;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping {what}: no gpt-oss-20b snapshot in the hugging face cache \
             (set PIE_GPTOSS_SNAPSHOT)"
        );
        return None;
    };
    let shards = shards(&checkpoint);
    if shards.is_empty() {
        eprintln!("skipping {what}: {checkpoint:?} holds no tensor container");
        return None;
    }
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let trace = model::trace_of(SKU).expect("the catalog ships the SKU")(Platform::Cuda);
    let source = ztensor_compat::index_all(&shards).expect("the checkpoint's shards open as one");
    let contract = model::import_of(SKU).expect("the catalog ships an import")(&source)
        .expect("the import contract fits its own checkpoint");
    drop(source);
    Some(Rig {
        trace,
        contract,
        checkpoint,
        tokenizer,
    })
}

fn load(rig: &Rig, residency: Plan, cache: Option<&Path>) -> engine_cuda::Result<Shell> {
    Shell::load(Boot {
        trace: rig.trace.clone(),
        contract: &rig.contract,
        checkpoint: &rig.checkpoint,
        budget: Budget::new(4, 256),
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        program_cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        patches: None,
        weight_cache_dir: cache,
        residency,
    })
}

/// A prefill and `STEPS` greedy decodes from a freshly opened slot, feeding
/// the argmax back. Answers the tokens, the logit rows they were chosen from,
/// and the mean DECODE step time — the prefill is excluded because its cost is
/// the prompt's and not the residency's.
fn run(shell: &mut Shell, prompt: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>, f64) {
    shell.open(0).expect("slot 0 opens");
    let mut chosen = Vec::with_capacity(STEPS);
    let mut rows = Vec::with_capacity(STEPS + 1);
    let prefill = shell
        .fire(&[Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt,
        }])
        .expect("the prefill fires");
    finite(&prefill[0], "prefill");
    let mut fed = argmax(&prefill[0]);
    chosen.push(fed);
    rows.push(prefill[0].clone());
    let began = Instant::now();
    for step in 0..STEPS {
        let decode = shell
            .fire(&[Lane {
                slot: 0,
                word: word(1),
                tokens: &[fed],
            }])
            .unwrap_or_else(|why| panic!("decode step {step} fires: {why}"));
        finite(&decode[0], "decode");
        fed = argmax(&decode[0]);
        chosen.push(fed);
        rows.push(decode[0].clone());
    }
    let each = began.elapsed().as_secs_f64() * 1e3 / STEPS as f64;
    (chosen, rows, each)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        assert!(value.is_finite(), "logit {at} is {value}");
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn finite(logits: &[f32], what: &str) {
    assert!(!logits.is_empty(), "{what} produced no logits at all");
    let spread = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - logits.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(
        spread > 1e-3,
        "{what} logits span {spread}, which is a rectangle nothing wrote"
    );
}

/// The same logits, bit for bit — the one claim a residency change may not
/// move.
fn identical(golden: &[Vec<f32>], also: &[Vec<f32>], what: &str) {
    for (step, (a, b)) in golden.iter().zip(also).enumerate() {
        assert_eq!(
            a.len(),
            b.len(),
            "step {step} produced {} logits uncapped and {} {what}",
            a.len(),
            b.len()
        );
        for (at, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "step {step}, logit {at}: uncapped {x}, {what} {y} — the ladder moved a number"
            );
        }
    }
}

/// Every packed group's `(name, tier, hits)`, from the tier's own report. A
/// dense routed bank would report `held: None` and is not one of these; a
/// spilled dense plane reports no hits, because nothing counts it.
fn groups(shell: &Shell) -> Vec<(String, Held, u32)> {
    shell
        .expert_residency()
        .into_iter()
        .filter_map(|bank| {
            let held = bank.held?;
            Some((bank.name, held, *bank.hits.first()?))
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────

/// **(a), (b) and (c).** One uncapped boot to write the artifact and set the
/// golden, one three-tier boot that serves, counts, climbs, and serves again.
#[test]
#[ignore = "real-hardware: needs a CUDA device with HMM and ~15 GiB free, a local \
            gpt-oss-20b snapshot, and ~15 GiB of scratch disk; run it with `-- --ignored`"]
fn a_promoted_group_answers_the_bytes_it_answered() {
    let _one = serialized();
    let Some(rig) = rig("the B7 ladder gate") else {
        return;
    };
    if !engine_cuda::experts::pageable_access() {
        eprintln!(
            "skipping the B7 ladder gate: this device does not report \
             `pageableMemoryAccess`, so there is no mapped tier for a group to be \
             lifted off"
        );
        return;
    }
    let cache = cache();
    let prompt = rig.tokenizer.encode(PROMPT);

    let prospect = engine_cuda::weights::prospect(&rig.trace, &rig.contract, &rig.checkpoint)
        .expect("the load plan pairs every packed bank with its scales");
    let plan = Plan::of(
        &rig.trace,
        &prospect.planes,
        Budgets {
            device: Some(DEVICE),
            host: Some(HOST),
        },
    )
    .expect("a plan past both budgets is planned, not refused");
    assert!(
        plan.spill_demand() > 0,
        "this gate needs something on the file for the ladder to lift"
    );
    assert!(
        !plan.seated().is_empty(),
        "and something in the store for it to lift into"
    );
    eprintln!(
        "planned: {} groups in the store, {} placed elsewhere ({} of them mapped)",
        plan.seated().len(),
        plan.groups().len(),
        plan.groups().iter().filter(|g| g.held == Held::Mapped).count(),
    );

    // ── THE GOLDEN, UNCAPPED — and the boot that WRITES the artifact the
    //    three-tier boot maps and the ladder demotes into.
    let mut resident =
        load(&rig, Plan::default(), Some(&cache.0)).expect("the uncapped shell loads");
    assert!(resident.weights_resident());
    // A resident load opens no tier, so every packed group is handed a null
    // cell and a null counter and the launch is the launch it always made.
    assert!(
        resident.expert_residency().is_empty(),
        "an uncapped load reports no tier at all"
    );
    let (golden, golden_rows, resident_ms) = run(&mut resident, &prompt);
    let says = rig.tokenizer.decode(&golden, false);
    eprintln!("uncapped answers {says:?} at {resident_ms:.2} ms/step");
    drop(resident);

    // ── THE THREE-TIER LOAD.
    let mut shell = load(&rig, plan, Some(&cache.0)).expect("the three-tier shell loads");
    assert!(!shell.weights_resident());

    // (a) THE COUNTERS MOVE. Nothing has fired, so nothing has an opinion.
    let cold = groups(&shell);
    assert!(!cold.is_empty(), "the tier reports its packed groups");
    assert!(
        cold.iter().all(|(_, _, hits)| *hits == 0),
        "a group was counted before a fire: {:?}",
        cold.iter().find(|(_, _, hits)| *hits > 0)
    );

    let (tokens, rows, seated_ms) = run(&mut shell, &prompt);
    assert_eq!(tokens, golden, "the three-tier load chose other tokens");
    identical(&golden_rows, &rows, "three-tier");
    let once = groups(&shell);
    assert!(
        once.iter().all(|(_, _, hits)| *hits > 0),
        "a group was never counted though every fire reads every layer: {:?}",
        once.iter().find(|(_, _, hits)| *hits == 0)
    );
    let (_, _, twice_ms) = run(&mut shell, &prompt);
    let twice = groups(&shell);
    for ((name, _, before), (_, _, after)) in once.iter().zip(&twice) {
        assert!(
            after > before,
            "`{name}` was routed {before} times and then {after}; the counter stopped"
        );
    }
    eprintln!(
        "counters: {} groups, {} .. {} routed rows after one run, {} .. {} after two",
        once.len(),
        once.iter().map(|(_, _, h)| *h).min().unwrap_or(0),
        once.iter().map(|(_, _, h)| *h).max().unwrap_or(0),
        twice.iter().map(|(_, _, h)| *h).min().unwrap_or(0),
        twice.iter().map(|(_, _, h)| *h).max().unwrap_or(0),
    );
    eprintln!("three-tier, as the plan seated it: {seated_ms:.2} ms/step ({twice_ms:.2} warm)");

    // **THE VOTE ITSELF HAS NOT MOVED ANYTHING, AND THAT IS THE RULE WORKING.**
    // Every routed bank of this model is read by every step, so the counters
    // are uniform, `hits(candidate) > hits(occupant)` is false everywhere, and
    // a strict-improvement rule declines. See `Tier::decide_group`.
    let (up, down, held) = shell.group_ladder();
    assert_eq!(
        (up, down),
        (0, 0),
        "a uniform vote moved {up} groups up and {down} down; the rule is supposed to \
         be a steady state here ({held} gaps held back)"
    );

    // ── (b) A GROUP CHANGES RUNG, driven through the door.
    let mapped: Vec<String> = twice
        .iter()
        .filter(|(_, held, _)| *held == Held::Mapped)
        .map(|(name, _, _)| name.clone())
        .take(RUNGS)
        .collect();
    assert!(
        !mapped.is_empty(),
        "the plan put nothing on the file, so there is no rung to climb"
    );
    let mut climbed = Vec::new();
    for name in &mapped {
        match shell
            .promote_group(name)
            .unwrap_or_else(|why| panic!("`{name}` climbs: {why}"))
        {
            Some((from, to)) => {
                assert_eq!(from, Held::Mapped, "`{name}` was on the file");
                assert!(
                    to.rung() < from.rung(),
                    "`{name}` moved from {from:?} to {to:?}, which is not up"
                );
                // **THE REPORT AGREES IMMEDIATELY**, and it is asked here
                // rather than at the end because the ladder is allowed to move
                // this group again: a later forced rung may displace it, and
                // the claim is about the move, not about the final table.
                let (_, now, _) = groups(&shell)
                    .into_iter()
                    .find(|(at, _, _)| at == name)
                    .unwrap_or_else(|| panic!("`{name}` fell out of the residency report"));
                assert_eq!(now, to, "`{name}` climbed to {to:?} and reports {now:?}");
                climbed.push((name.clone(), from, to));
            }
            None => eprintln!("`{name}`: no berth of its shape stands on a faster rung"),
        }
    }
    assert!(
        !climbed.is_empty(),
        "no group could be lifted off the file at all: {mapped:?}"
    );
    let (up, down, held) = shell.group_ladder();
    assert_eq!(
        up as usize,
        climbed.len(),
        "the ladder counted {up} promotions for {} moves",
        climbed.len()
    );
    assert_eq!(
        down, up,
        "every berth here was occupied, so every promotion displaced exactly one group"
    );
    eprintln!("ladder: {up} up, {down} down, {held} gaps held back — {climbed:?}");

    // **THE CENSUS IS CONSERVED, AND THAT IS THE LADDER'S SHAPE AND NOT A
    // DISAPPOINTMENT.** The berths are fixed by the two budgets, so a swap
    // reassigns WHICH group is on which rung and can never make more fast
    // memory. What must change is the assignment.
    let after = groups(&shell);
    let census = |report: &[(String, Held, u32)]| {
        let mut rungs = [0usize; 3];
        for (_, held, _) in report {
            rungs[held.rung() as usize] += 1;
        }
        rungs
    };
    assert_eq!(
        census(&twice),
        census(&after),
        "the ladder changed how many groups sit on each rung, which no swap can do"
    );
    let moved = after
        .iter()
        .zip(&twice)
        .filter(|((name, now, _), (was, then, _))| name == was && now != then)
        .count();
    assert_eq!(
        moved,
        climbed.len() * 2,
        "{} groups changed rung for {} forced promotions; each one is a pair",
        moved,
        climbed.len()
    );
    eprintln!("census by rung (device, pinned, mapped): {:?}", census(&after));

    // ── (c) AND IT ANSWERS THE SAME BYTES. The same prompt over a table whose
    //    groups are on different tiers than they were two runs ago: the greedy
    //    tokens and every logit, bit for bit.
    let (again, again_rows, climbed_ms) = run(&mut shell, &prompt);
    assert_eq!(
        again, golden,
        "after the ladder moved, the load chose {again:?} and the uncapped one chose {golden:?}"
    );
    identical(&golden_rows, &again_rows, "post-ladder");

    eprintln!(
        "step time: uncapped {resident_ms:.2} ms, three-tier {seated_ms:.2} ms \
         ({twice_ms:.2} warm), after {} rungs {climbed_ms:.2} ms",
        climbed.len()
    );
    eprintln!(
        "T2 register after it all: {:?}; ladder {:?}",
        engine_cuda::experts::observed(),
        shell.group_ladder(),
    );
}
