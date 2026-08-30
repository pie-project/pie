//! **W-1, THE CAPABILITY GATE: a model larger than device AND host memory
//! combined produces sensible tokens** (alto streaming §2; the page's own
//! reason to exist).
//!
//! Streaming's owner stated the requirement and it reversed an earlier
//! priority judgment: *serving models that exceed GPU and host memory combined
//! is a required experimental capability.* Capability is the deliverable;
//! speed is the trade, stated and measured. This file is that sentence, made
//! decidable on the hardware this box has.
//!
//! ```text
//! T0 device   `device_weight_budget` = 4 GiB   the dense planes + what fits
//! T1 pinned   `host_weight_budget`   = 2 GiB   whole packed banks, over UVA
//! T2 mapped   no budget — a FILE               the rest, faulted in over HMM
//! ```
//!
//! gpt-oss-20b's weight table is ~12.8 GiB. Four plus two is six. The other
//! ~7 GiB is read by the GPU **directly out of an `mmap` of the warm-boot
//! artifact**: no copy, no page-lock, no host round trip. That is the whole
//! claim, and the absolute numbers are the only thing that separates it from
//! the glm_5-scale instance streaming §2 sketches — the tier mix is the same
//! mix.
//!
//! # Why the artifact and not the checkpoint
//!
//! Streaming §0's precondition is that the T2 source needs **no load-time
//! conversion**, and the checkpoint does not meet it: gpt-oss's gate/up bank
//! is DE-INTERLEAVED at import (`model/src/gpt_oss/import.rs`'s
//! `banked_interleaved` concatenates two strided views), so the bytes in the
//! safetensors shard are not the bytes a kernel reads. The artifact is a
//! snapshot of the DEVICE STORE — every dequant, cast and repack already
//! applied, offsets identical to store offsets — so it is the one file on disk
//! that can be served from as-is. §0 said the format already existed; this
//! gate is what spends it.
//!
//! # The three claims
//!
//! ```text
//! (a) THE CAPABILITY. Both budgets under the banks, and the shell boots and
//!     answers sensible tokens rather than refusing.
//! (b) BIT-IDENTITY across the {T0+T1+T2} mix (W-2). The three-tier load's
//!     greedy tokens and logits are the uncapped load's, exactly.
//! (c) THE REFUSAL IS STILL THERE. The same plan with no artifact to map is
//!     `Fault::Residency` by name — a spilled load with no source is the one
//!     thing this shell cannot serve, and it says so instead of pretending.
//! ```
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!     --test a_model_larger_than_both_budgets_serves -- --ignored --nocapture
//! ```
//!
//! # Gating and cost
//!
//! `#[ignore]`d. It wants a CUDA device that reports `pageableMemoryAccess`
//! (CUDA 12.2+ HMM) with ~15 GiB free, the gpt-oss-20b snapshot, and ~15 GiB
//! of scratch disk for the artifact it writes. It loads the model twice,
//! sequentially. Skips with a sentence when any of that is missing.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_cuda::experts::{Budgets, Held, Plan};
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "gptoss-20b-bf16-mxfp4-kv-bf16";

/// **THE TWO CEILINGS.** Six GiB between them, over a ~12.8 GiB table: what
/// neither holds is the point of the test, and it is most of the model.
const DEVICE: u64 = 4 << 30;
const HOST: u64 = 2 << 30;

const PROMPT: &str = "<|start|>user<|message|>What is the capital of France? \
                      Answer in one word.<|end|>\
                      <|start|>assistant<|channel|>final<|message|>";

/// Short: every fire reads the mapped banks, so the claim is stated in as few
/// steps as can state it.
const STEPS: usize = 6;

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
    let dir = std::env::temp_dir().join(format!("pie-w1-{}", std::process::id()));
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
        // No second row axis: this gate fires text lanes only.
        patches: None,
        // **THE CACHE DIRECTORY IS THE T2 SOURCE'S HOME**, and here it is a
        // gate's own scratch rather than a shared one: the uncapped boot below
        // WRITES the artifact and the capped boot MAPS it, so what is under
        // test is this run's file and not the last run's.
        weight_cache_dir: cache,
        residency,
    })
}

fn run(shell: &mut Shell, prompt: &[u32]) -> (Vec<u32>, Vec<Vec<f32>>) {
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
    (chosen, rows)
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

// ─────────────────────────────────────────────────────────────────────────────

/// **(a), (b) and (c).** One uncapped boot to write the artifact and set the
/// golden, one three-tier boot to serve out of it, and one refusal.
#[test]
#[ignore = "real-hardware: needs a CUDA device with HMM and ~15 GiB free, a local \
            gpt-oss-20b snapshot, and ~15 GiB of scratch disk; run it with `-- --ignored`"]
fn a_model_larger_than_both_budgets_serves() {
    let _one = serialized();
    let Some(rig) = rig("the W-1 capability gate") else {
        return;
    };
    if !engine_cuda::experts::pageable_access() {
        eprintln!(
            "skipping the W-1 capability gate: this device does not report \
             `pageableMemoryAccess`, so a GPU touch of a mapped page cannot fault it \
             in — the T2 arm's one hardware precondition"
        );
        return;
    }
    let cache = cache();
    let prompt = rig.tokenizer.encode(PROMPT);

    let prospect = engine_cuda::weights::prospect(&rig.trace, &rig.contract, &rig.checkpoint)
        .expect("the load plan pairs every packed bank with its scales");
    let full = Plan::of(&rig.trace, &prospect.planes, Budgets::uncapped())
        .expect("an mxfp4 MoE plans")
        .device_demand();
    eprintln!(
        "gpt-oss-20b: {full} bytes of table; the two budgets are {DEVICE} + {HOST} = {}",
        DEVICE + HOST
    );
    assert!(
        full > DEVICE + HOST,
        "this gate is about a model LARGER than both budgets, and this one is {full}"
    );

    // ── THE PLAN, BEFORE ANY DEVICE IS TOUCHED. Three tiers by two budgets.
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
        "two budgets under the table have to spill"
    );
    let (pinned, mapped): (Vec<_>, Vec<_>) = plan
        .groups()
        .iter()
        .partition(|group| group.held == Held::Pinned);
    eprintln!(
        "planned: T0 {} bytes, T1 {} bytes ({} banks), T2 {} bytes ({} banks)",
        plan.device_demand(),
        plan.host_demand(),
        pinned.len(),
        plan.spill_demand(),
        mapped.len(),
    );
    assert!(!mapped.is_empty(), "and the third tier is what holds the rest");

    // ── (c) THE REFUSAL, FIRST, WHILE THE CACHE IS STILL EMPTY. A spilled
    //    plan with no artifact to map is not a slow load; it is a load this
    //    machine cannot serve, and it says which file it wanted.
    let said = match load(&rig, plan.clone(), Some(&cache.0)) {
        Err(why) => format!("{why}"),
        Ok(_) => panic!("a spilled plan with no artifact cannot be served"),
    };
    assert!(
        said.contains("mapped tier"),
        "the refusal names the tier: {said}"
    );
    eprintln!("with an empty cache: {said}");

    // ── THE GOLDEN, UNCAPPED — and the boot that WRITES the artifact the
    //    capped boot will map. `weight_cache_dir` is what makes the warm-boot
    //    file a serving-time source (streaming §0's promotion).
    let mut resident = load(&rig, Plan::default(), Some(&cache.0)).expect("the uncapped shell loads");
    assert!(resident.weights_resident());
    let (golden, golden_rows) = run(&mut resident, &prompt);
    let says = rig.tokenizer.decode(&golden, false);
    eprintln!("uncapped answers: {says:?}");
    drop(resident);

    let artifact = engine_cuda::weight_cache::artifact_path(&cache.0, prospect.resident_key);
    assert!(
        artifact.is_file(),
        "the uncapped boot writes the artifact at {}",
        artifact.display()
    );
    eprintln!(
        "artifact: {} ({} bytes on disk)",
        artifact.display(),
        std::fs::metadata(&artifact).map(|m| m.len()).unwrap_or(0)
    );

    // ── (a) THE CAPABILITY. The same plan, now that the source exists.
    let before = engine_cuda::experts::observed();
    let mut spilled = load(&rig, plan, Some(&cache.0))
        .expect("a model larger than both budgets serves out of the mapped artifact");
    assert!(!spilled.weights_resident());
    let after = engine_cuda::experts::observed();
    assert!(
        after.seated > before.seated && after.bytes > before.bytes,
        "the mapped tier seated nothing: {before:?} -> {after:?}"
    );
    assert_eq!(after.absent, before.absent, "and refused no plane");
    eprintln!(
        "T2 register: {} planes, {} bytes, {} absent, {} loads",
        after.seated - before.seated,
        after.bytes - before.bytes,
        after.absent,
        after.loads
    );

    let banks = spilled.expert_residency();
    assert!(
        banks.iter().any(|bank| bank.held == Some(Held::Mapped)),
        "and the tier reports which banks it mapped"
    );

    let (tokens, rows) = run(&mut spilled, &prompt);
    let also = rig.tokenizer.decode(&tokens, false);
    eprintln!("three-tier answers: {also:?}");
    assert!(
        tokens.iter().collect::<std::collections::BTreeSet<_>>().len() > 1,
        "the three-tier load answered {tokens:?}, which is one token repeated"
    );

    // ── (b) BIT-IDENTITY, {T0 + T1 + T2}. The same kernels over the same
    //    bytes; only which address space they came out of differs.
    assert_eq!(
        tokens, golden,
        "the three-tier load chose {tokens:?} and the uncapped one chose {golden:?}"
    );
    for (step, (a, b)) in golden_rows.iter().zip(&rows).enumerate() {
        for (at, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "step {step}, logit {at}: uncapped {x}, three-tier {y} — a tier moved a number"
            );
        }
    }
}
