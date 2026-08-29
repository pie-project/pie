//! **The correction class, end to end through the contract** (alto design §8,
//! decision 17; survey §2 debt 6 — "doors on both sides with nothing
//! between").
//!
//! The adapter axis has been complete on the device since the palo rewrite —
//! banks declared by the model text and reserved at load, a registration verb
//! that writes them, a per-lane id, a fact bit, a routes vector, a kernel —
//! and there was no gate that walked the whole of it through the CONTRACT.
//! `engine-cuda/tests/adapter_banks.rs` walks it through `Shell`, which is the
//! shell's own surface; what nothing asserted is that
//! `engine_api::AdapterRegistration` reaches a bank, that
//! `Lane::adapter` reaches the kernel, and that the two agree.
//!
//! Three claims, in the order they can fail:
//!
//! ```text
//!  1. registration lands   -- `runtime::engine::verbs::register_adapter`
//!                             writes the planes a load's banks reserved
//!  2. the lane routes      -- a lane whose `adapter` is `Some(0)` and whose
//!                             word says so comes back DIFFERENT from base
//!  3. captured == eager    -- and the corrected numbers are the same numbers
//!                             the eager walk computes (article 6)
//! ```
//!
//! Claim 3 is the one that makes claim 2 mean anything: an arm that ran and
//! produced *something* different is not an arm that ran correctly, and the
//! eager walk is this workspace's golden model.
//!
//! ```bash
//! cargo test -p runtime --features engine-cuda-13 --test cuda_adapter_e2e -- --nocapture
//! ```

#![cfg(feature = "_engine-cuda")]

use std::path::{Path, PathBuf};

use engine_api::model_ir::Platform;
use engine_api::{Budgets, Lane, Readout, Step};
use runtime::engine::backend::open;

const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

const PROMPT: &str = "The capital of France is";

#[test]
fn an_adapter_registered_through_the_contract_corrects_the_lane_that_names_it() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the adapter e2e: no CUDA device on this machine");
        return;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping the adapter e2e: no Qwen3.5-0.8B snapshot in the hugging \
             face cache (set PIE_SMOKE_SNAPSHOT)"
        );
        return;
    };
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");
    let prompt = tokenizer.encode(PROMPT);
    assert!(!prompt.is_empty(), "the prompt tokenizes to something");

    // ── 1. BASE, captured. No adapter registered, no lane routed: the
    //    reference every other run is compared against.
    let base = run(&checkpoint, &prompt, Graphs::Captured, Routing::Base);

    // ── 2. CORRECTED, captured. One adapter registered through
    //    `engine_api::AdapterRegistration`, one lane routed to it.
    let corrected = run(&checkpoint, &prompt, Graphs::Captured, Routing::Adapter(0));
    assert_eq!(
        corrected.len(),
        base.len(),
        "a correction changes the numbers, not the shape of them"
    );
    let displacement = max_abs_difference(&base, &corrected);
    eprintln!("base vs corrected: max |delta| = {displacement}");
    assert!(
        displacement > 1e-2,
        "the corrected readout is within {displacement} of the base one — the \
         registration reached no bank, or the lane routed to nobody"
    );

    // ── 3. CORRECTED, eager. Article 6: captured is eager by construction —
    //    one interpreter, two sinks — so the SAME corrected fire through the
    //    golden walk has to answer the same numbers. This is what says the
    //    arm ran RIGHT rather than merely ran.
    let golden = run(&checkpoint, &prompt, Graphs::Eager, Routing::Adapter(0));
    let drift = max_abs_difference(&corrected, &golden);
    eprintln!("captured vs eager (both corrected): max |delta| = {drift}");
    assert_eq!(
        drift, 0.0,
        "the captured corrected fire and the eager one disagree by {drift}; \
         captured is eager by construction (article 6), so a difference here \
         is the correction arm being recorded differently from how it walks"
    );

    // And the eager base is the eager corrected's own control, so a reader
    // does not have to take the captured comparison's word for the axis.
    let golden_base = run(&checkpoint, &prompt, Graphs::Eager, Routing::Base);
    assert!(
        max_abs_difference(&golden_base, &golden) > 1e-2,
        "the eager walk shows no correction either"
    );
    assert_eq!(
        max_abs_difference(&base, &golden_base),
        0.0,
        "and the two BASE walks agree, which is what makes the two corrected \
         ones comparable at all"
    );
}

// ── the run ──────────────────────────────────────────────────────────────

/// Which walk the shell takes. Article 6's two sinks, as a boot key.
#[derive(Clone, Copy)]
enum Graphs {
    /// `[engine] graphs = "on"` — the recorded graph, replayed.
    Captured,
    /// `[engine] graphs = "off"` — the eager walk, this workspace's golden
    /// model, kept off the serving path rather than deleted.
    Eager,
}

/// Whether this fire's lane routes to an adapter row.
#[derive(Clone, Copy)]
enum Routing {
    Base,
    Adapter(u32),
}

/// One load, one optional registration, one fire, one readout.
///
/// A fresh engine per run on purpose: `kernels-cuda`'s scratch slabs are
/// process-global and keyed by name, so one shell at a time is the rule every
/// gate in this workspace follows, and a run that ended is a shell that
/// dropped.
fn run(checkpoint: &Path, prompt: &[u32], graphs: Graphs, routing: Routing) -> Vec<f32> {
    let boot = match graphs {
        Graphs::Captured => "[model]\ndevice = \"cuda:0\"\n[engine]\ngraphs = \"on\"\n",
        Graphs::Eager => "[model]\ndevice = \"cuda:0\"\n[engine]\ngraphs = \"off\"\n",
    };
    let mut engine = open::cuda(boot.as_bytes()).expect("the cuda seam opens");

    // **THE SEATS ARE THE MODEL TEXT'S OWN CAPACITY, ASKED FOR IN FULL.**
    // `max_adapters` is what the DEPLOYMENT intends to register and a bank's
    // leading axis is what the plan seats; `model_compiler::compile` refuses
    // a load whose intent is bigger than the text's, so asking for exactly
    // what the text declares is both the honest ask and the one that
    // exercises the check. This is the same number an operator states as
    // `[model.adapters] seats`.
    let mut request = runtime::engine::load::request(
        checkpoint,
        Platform::Cuda,
        Budgets {
            max_lanes: 2,
            // Small on purpose: the arena reserves `max_tokens` rows of a
            // 248320-wide logit column.
            max_tokens: 64,
            buckets: Vec::new(),
            max_adapters: 0,
            page_size: 16,
            max_context: 128,
            slots: 2,
        },
        // Uncapped: this shell has one weight tier (alto design §7).
        engine_api::Residency::uncapped(),
        0,
        1,
    )
    .expect("the checkpoint identifies and its SKU traces");
    assert_eq!(request.trace.name, SKU);
    let banks = banks_of(&request.trace);
    assert!(
        !banks.is_empty(),
        "{SKU} declares no adapter bank, so this gate has nothing to route to"
    );
    let seats = banks
        .iter()
        .map(|bank| bank.seats)
        .min()
        .expect("checked above");
    request.budgets.max_adapters = seats;

    let loaded = engine.load(request).expect("the checkpoint lands");
    assert_eq!(
        loaded.caps.pools.adapter_banks, seats,
        "the load publishes the capacity it reserved"
    );

    // ── THE REGISTRATION DOOR. `verbs::register_adapter` is the runtime's
    //    public surface for it, and this is what it looks like from a caller
    //    — the same call `worker::embedded_engine` makes for every adapter an
    //    operator declares in `[model.adapters]`.
    if let Routing::Adapter(id) = routing {
        let registration = engine_api::AdapterRegistration {
            id,
            planes: banks
                .iter()
                .map(|bank| engine_api::AdapterPlane {
                    bank: bank.name.clone(),
                    bytes: loud_plane(&bank.name, bank.slot_bytes),
                })
                .collect(),
        };
        runtime::engine::verbs::register_adapter(&mut engine, &registration)
            .expect("the planes reach the banks the load reserved");
    }

    // ── THE FIRE. The lane's `adapter` and the lane's WORD have to agree —
    //    the shell refuses a disagreement by name (`Fault::AdapterWord`), and
    //    that check is the reason the word is computed from the same fact the
    //    field carries rather than stated twice.
    let adapter = match routing {
        Routing::Base => None,
        Routing::Adapter(id) => Some(id),
    };
    let step = Step {
        lanes: vec![Lane {
            slot: 0,
            word: word(prompt.len() as u32, adapter.is_some()),
            tokens: prompt.to_vec(),
            positions: Vec::new(),
            kv: engine_api::KvDelta::default(),
            mask: None,
            adapter,
            drafts: false,
            captures_scores: false,
            rs: engine_api::RsVerb::Fold,
            rs_reset: engine_api::RsReset::Inferred,
            channels: Vec::new(),
            readout: Readout::Last,
        }],
        attachments: Vec::new(),
    };
    step.validate().expect("the submission is one the contract describes");
    let frame = engine_api::FrameSubmission::of(step);
    let mut ticket = engine.submit(&frame).expect("the fire is admitted");
    engine
        .settle_frame(&mut ticket)
        .expect("and the numbers door hands back what it computed");

    let readout = &ticket.steps[0].readouts[0];
    assert_eq!(readout.rows, 1, "`Readout::Last` is one row");
    assert!(
        readout.values.iter().all(|value| value.is_finite()),
        "a non-finite logit is a pool read at the wrong stride"
    );
    readout.values.clone()
}

// ── the banks ────────────────────────────────────────────────────────────

/// One declared adapter bank, read off the plan the runtime traced.
///
/// The runtime knows this without asking the device: a bank is a param the
/// model text marked `registered`, its capacity is that param's leading axis,
/// and one adapter's slot is everything after it.
struct Bank {
    name: String,
    seats: u32,
    slot_bytes: usize,
}

fn banks_of(trace: &engine_api::model_ir::Trace) -> Vec<Bank> {
    trace
        .params
        .iter()
        .filter(|param| param.source == engine_api::model_ir::ParamSource::Registered)
        .map(|param| {
            let seats = param.shape.first().copied().unwrap_or(0);
            let elements: u64 = param.shape.iter().skip(1).product();
            assert_eq!(
                param.dtype,
                engine_api::model_ir::Dtype::Bf16,
                "this gate builds bf16 planes; {} declares {:?}",
                param.name,
                param.dtype
            );
            Bank {
                name: param.name.clone(),
                seats: u32::try_from(seats).expect("a capacity fits a u32"),
                slot_bytes: usize::try_from(elements * 2).expect("a slot fits this host"),
            }
        })
        .collect()
}

/// **FULL CAPACITY, NOT THE ADAPTER'S OWN RANK, AND THE PADDING IS OURS.**
/// The contract says so (`engine_api::adapter`): a plane is one whole slot in
/// the bank's declared dtype and layout, because `A`'s unused ranks are
/// trailing rows and `B`'s are a stride inside every row, and a shell that
/// padded a short plane's prefix would be right for one and wrong for the
/// other.
///
/// Loud on purpose: twenty-four stacked corrections have to take the
/// continuation somewhere the base model would not have gone, or the gate
/// cannot tell an arm that ran from an arm that did not.
fn loud_plane(bank: &str, slot_bytes: usize) -> Vec<u8> {
    let count = slot_bytes / 2;
    let mut bytes = Vec::with_capacity(slot_bytes);
    for at in 0..count {
        let sign = if at % 2 == 0 { 1.0 } else { -1.0 };
        let value = if bank.ends_with(".lora_a") {
            sign * 0.02 * (((at % 11) as f32) + 1.0)
        } else {
            sign * 0.02 * (((at % 7) as f32) + 1.0)
        };
        bytes.extend_from_slice(&bf16_bits(value).to_le_bytes());
    }
    bytes
}

/// f32 to bf16, round-to-nearest-even — the same conversion the loader does,
/// stated here because a test that truncated would be registering a slightly
/// different adapter than the one it describes.
fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// The lane word the model's own `Classify` computes, reached the way the fire
/// path reaches it — through the catalog, keyed by the same `SKU` string every
/// other column is.
fn word(query_len: u32, adapted: bool) -> u64 {
    let classify = runtime::engine::load::classify(SKU).expect("this build ships the gate's SKU");
    classify(&model::Request::new(query_len, false).adapted(adapted))
}

fn max_abs_difference(one: &[f32], other: &[f32]) -> f32 {
    one.iter()
        .zip(other)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

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
