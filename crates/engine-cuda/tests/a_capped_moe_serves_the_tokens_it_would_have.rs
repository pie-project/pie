//! **W-5: a capped-budget quantized MoE serves, and says what the uncapped
//! load says** (alto streaming §3 item 6, §2's two gate kinds).
//!
//! The refusal this file replaces was two lines deep and one sentence long.
//! `weights::plane_bytes` asked `model_compiler::arena::elem_bytes` for the
//! size of an `Mxfp4` element; the honest answer is `None` — a four-bit code
//! has no byte — and the shell read that `None` as *"declared in a packed
//! storage element that has no element size"* and refused. gpt-oss-20b is the
//! catalog's only locally-servable MoE and every one of its 48 expert banks is
//! mxfp4, so the cuda shell could not load it AT ALL, capped or not. One tier
//! down, `experts::Plan::of` had a second refusal for the same shape: a
//! quantized bank under a stated `device_weight_budget` was refused rather
//! than held whole, because there was no bank to stream.
//!
//! Both are gone, and what stands in their place is three claims this file
//! makes on real hardware:
//!
//! ```text
//! (a) THE CAPABILITY. A `device_weight_budget` under gpt-oss-20b's ~13.8 GiB
//!     of banks boots and answers sensible tokens. Some of its banks are on
//!     the pinned tier and the kernels read them there, over UVA.
//! (b) BIT-IDENTITY (W-2's {T0 capped} tier mix). The capped load's greedy
//!     tokens are the uncapped load's greedy tokens, and its logits are the
//!     same floats — not nearly, exactly. The two runs launch the same
//!     kernels over the same bytes; only the addresses differ.
//! (c) THE SEATING IS COHERENT. Every plane of a streamed bank is on the same
//!     tier. A group whose codes moved and whose exponents did not is a model
//!     that computes and is wrong — worse than a miss — and the plan says so
//!     of every group it streams.
//! ```
//!
//! # Why a WHOLE bank streams and not a whole expert
//!
//! The dense routed tier (wave D2, `routed_experts_stream.rs`) seats experts
//! one at a time behind an indirection table, because `moe_matmul_select_gemv`
//! LOADS `expert_table[expert]` and dereferences whatever it finds. The mxfp4
//! select does not: `moe_matmul_select_mxfp4` computes
//! `codes + e * n * (k / 2)` and `scales + e * n * (k / 32)` itself, from the
//! routing vector, with no table anywhere in the launch. So there is nothing
//! on the packed path for a per-expert entry to point at, and the unit of
//! residency is the GROUP — the code plane and the exponent plane together,
//! whole, on one tier. `experts.rs`' header argues it; this file measures the
//! result.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda-13 \
//!     --test a_capped_moe_serves_the_tokens_it_would_have -- --ignored --nocapture
//! ```
//!
//! # Gating
//!
//! `#[ignore]`d: it wants a CUDA device with ~15 GiB free AND the gpt-oss-20b
//! snapshot on disk, and it loads the model TWICE (sequentially — the first
//! shell is dropped before the second is built). Skips with a sentence when
//! either is missing, the same convention `masked_axis.rs` uses.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use engine_cuda::experts::{Budgets, Plan};
use engine_cuda::{Boot, Graphs, Lane, Shell};
use model_compiler::Budget;
use model_dsl::{Classify, Platform, Request};

const SKU: &str = "gptoss-20b-bf16-mxfp4-kv-bf16";

/// The harmony turn, written out rather than templated: this binary is its own
/// crate and what it needs is a deterministic prompt, not the chat surface.
/// The `final` channel is opened here so the answer is the answer and not a
/// page of analysis.
const PROMPT: &str = "<|start|>user<|message|>What is the capital of France? \
                      Answer in one word.<|end|>\
                      <|start|>assistant<|channel|>final<|message|>";

/// How many greedy decodes follow the prefill. Short: the claim is an identity
/// and a sensible answer, and both are visible in a handful of tokens.
const STEPS: usize = 12;

/// **What fraction of the whole table the capped load may hold.** Low enough
/// that whole banks land on the pinned tier and high enough that the load is a
/// serving load rather than a demonstration of PCIe.
const CAP: u64 = 7;
const OF: u64 = 10;

/// One shell at a time per process — `kernels-cuda`'s scratch slabs are
/// process-global and keyed by name, and two gpt-oss loads do not fit one card
/// at once anyway.
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

/// Every container in the snapshot, sorted — gpt-oss-20b ships three shards
/// and an import built over shard zero refuses for an unrelated reason.
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

/// The trace, the contract, the checkpoint and the tokenizer — everything both
/// loads share, read once.
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

/// A shell over the rig, at a stated residency.
fn load(rig: &Rig, residency: Plan) -> engine_cuda::Result<Shell> {
    Shell::load(Boot {
        trace: rig.trace.clone(),
        contract: &rig.contract,
        checkpoint: &rig.checkpoint,
        budget: Budget::new(4, 256),
        patches: None,
        profile: None,
        page_size: 16,
        context: 512,
        slots: 4,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: engine_cuda::Knobs::default(),
        program_cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        // The warm-boot artifact cache is off for a gate — a test that shared
        // one would be asserting about the last run — and a streamed load
        // forms no key anyway.
        weight_cache_dir: None,
        residency,
    })
}

/// A prefill and `STEPS` greedy decodes, feeding the argmax back. Answers the
/// tokens it chose and the logit rows it chose them from.
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

/// **(a), (b) and (c) in one boot pair.**
///
/// Two loads of one model, sequential, over the same checkpoint: uncapped
/// first — the golden, and the load that until this wave refused before it
/// reserved a byte — then capped to `CAP/OF` of the table, which puts whole
/// mxfp4 banks on the pinned tier.
#[test]
#[ignore = "real-hardware: needs a CUDA device with ~15 GiB free and a local \
            gpt-oss-20b snapshot; run it with `-- --ignored`"]
fn a_capped_moe_serves_the_tokens_it_would_have() {
    let _one = serialized();
    let Some(rig) = rig("the capped-MoE gate") else {
        return;
    };
    let prompt = rig.tokenizer.encode(PROMPT);
    assert!(prompt.len() > 4, "the harmony turn encodes to something");

    // The pairing the loader records: which other param moves when a bank
    // moves. Read off the load plan, before a byte is landed — the same door
    // `Cuda::load` opens.
    let prospect = engine_cuda::weights::prospect(&rig.trace, &rig.contract, &rig.checkpoint)
        .expect("the load plan pairs every packed bank with its scales");
    let planes = prospect.planes;
    assert!(
        !planes.is_empty(),
        "gpt-oss-20b's banks are split-plane and the plan says so"
    );

    let full = Plan::of(&rig.trace, &planes, Budgets::uncapped())
        .expect("an mxfp4 MoE plans — the `no element size` refusal is gone")
        .device_demand();
    eprintln!("gpt-oss-20b: {full} bytes of weight table, whole");

    // ── THE GOLDEN, UNCAPPED. Everything on the device; no tier opened.
    let mut resident = load(&rig, Plan::default()).expect("the uncapped shell loads");
    assert!(
        resident.weights_resident(),
        "an uncapped load holds the whole table"
    );
    assert!(
        resident.expert_residency().is_empty(),
        "and opens no tier to report on"
    );
    let (golden, golden_rows) = run(&mut resident, &prompt);
    let said = rig.tokenizer.decode(&golden, false);
    eprintln!("uncapped answers: {said:?}");
    drop(resident);

    // ── THE CAPPED LOAD. A budget under the banks: whole groups to T1.
    let budget = full * CAP / OF;
    let plan = Plan::of(&rig.trace, &planes, Budgets::device(budget))
        .expect("a capped mxfp4 MoE plans rather than refusing");
    assert!(plan.streams(), "a budget under the table has to stream");
    assert!(
        !plan.groups().is_empty(),
        "and what it streams is whole packed banks"
    );
    assert!(
        plan.device_demand() <= budget,
        "the plan fits its budget: {} > {budget}",
        plan.device_demand()
    );
    // (c) COHERENCE: no group has one plane on each tier.
    for group in plan.groups() {
        assert!(
            group.planes.len() >= 2,
            "`{}` is split-plane and streams {} plane(s)",
            group.name,
            group.planes.len()
        );
        for plane in &group.planes {
            assert!(
                plan.pinned(plane.param),
                "`{}` streams and its plane {} does not — a torn pair",
                group.name,
                plane.param
            );
        }
    }
    eprintln!(
        "capped: {} bytes on the device ({} the budget), {} of {} banks pinned, \
         {} bytes on the pinned tier",
        plan.device_demand(),
        budget,
        plan.groups().len(),
        planes.len(),
        plan.host_demand(),
    );

    let mut streamed = load(&rig, plan).expect("the capped shell loads");
    assert!(
        !streamed.weights_resident(),
        "a streamed load says so rather than claiming the table"
    );
    // The tier reports the packed banks at zero device slots, which is the
    // truth: their planes are on the pinned tier whole.
    let banks = streamed.expert_residency();
    assert!(
        !banks.is_empty() && banks.iter().all(|bank| bank.slots == 0),
        "every streamed bank here is packed, and a packed bank seats no slot: {banks:?}"
    );

    let (tokens, rows) = run(&mut streamed, &prompt);
    let also = rig.tokenizer.decode(&tokens, false);
    eprintln!("capped answers:   {also:?}");

    // ── (a) SENSIBLE. Not a rectangle of one repeated id, and not empty.
    assert!(
        tokens.iter().collect::<std::collections::BTreeSet<_>>().len() > 1,
        "the capped load answered {tokens:?}, which is one token repeated"
    );
    assert!(
        !said.trim().is_empty(),
        "the uncapped load answered nothing at all"
    );

    // ── (b) BIT-IDENTITY. The same kernels over the same bytes at different
    //    addresses: the floats are the same floats, not nearly the same.
    assert_eq!(
        tokens, golden,
        "the capped load chose {tokens:?} and the uncapped one chose {golden:?}"
    );
    for (step, (a, b)) in golden_rows.iter().zip(&rows).enumerate() {
        assert_eq!(
            a.len(),
            b.len(),
            "step {step} produced {} logits capped and {} uncapped",
            b.len(),
            a.len()
        );
        for (at, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "step {step}, logit {at}: uncapped {x}, capped {y} — residency moved a number"
            );
        }
    }
}
