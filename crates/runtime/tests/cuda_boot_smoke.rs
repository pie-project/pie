//! The boot smoke: a real checkpoint through the CONTRACT, and a real token
//! back.
//!
//! # What this is for, and what it is not
//!
//! `engine-cuda/tests/serve_smoke.rs` already asserts that the shell computes
//! — same checkpoint, same prompt, same pinned continuation. It reaches the
//! shell through `Shell::load`/`Shell::fire`, which is the crate's own
//! surface. This test reaches the SAME device through the boundary the runtime
//! actually uses:
//!
//! ```text
//!   backend::open::cuda(boot document)   -> Box<dyn Engine>
//!   engine::load::request(checkpoint)    -> LoadRequest { plan, .. }
//!   Engine::load(request)                -> Loaded { facts, caps }
//!   Engine::fire(Step{lanes})  -> FireTicket { readouts }
//! ```
//!
//! Four things can be wrong here that no test under it can see: the boot
//! document's device key does not reach the shell, the runtime's tracing door
//! picks the wrong SKU for a checkpoint, the `Capabilities` the load answers
//! do not describe the load, and a `Step` the runtime builds is not
//! one the shell composes. Each of them is a fire that runs and says the
//! wrong thing.
//!
//! # Gating
//!
//! A build that names CUDA is not a machine that has it, and neither is a
//! machine that has it a machine with a 1.7 GB checkpoint on its disk. So it
//! skips at RUN time, saying which it was missing, rather than being
//! `#[ignore]`d.
//!
//! ```text
//! RUSTFLAGS="--force-warn missing_docs" \
//!   cargo test -p runtime --features engine-cuda-13 --test cuda_boot_smoke -- --nocapture
//! ```
//!
//! `PIE_SMOKE_SNAPSHOT` overrides where it looks.

#![cfg(feature = "_engine-cuda")]

use std::path::{Path, PathBuf};

use engine::{Budgets, Step, Lane, Readout};
use model_ir::Platform;
use runtime::engine::backend::open;

/// The catalog row this smoke serves, spelled as the catalog spells it.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The prompt, and the reason it is this one: the answer is a single
/// well-known token, so a continuation that is merely fluent still fails.
const PROMPT: &str = "The capital of France is";

/// What a correct load produces here — the same token
/// `engine-cuda/tests/serve_smoke.rs` pinned against the same checkpoint.
/// Two paths to one device must agree about it.
const EXPECTED: &str = " Paris";

/// The snapshot directory: the checkpoint AND the tokenizer that goes with
/// it, because a vocabulary from another snapshot decodes the right ids into
/// the wrong words.
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

/// The lane word the model's own `Classify` computes, reached the way the
/// fire path reaches it.
///
/// It named `model::qwen_3::forward::Facts` directly and said so: the catalog
/// shipped `(sku, tp, TraceFn)`, nothing outside a family's module could say
/// which bit `qo_one` is, and stating a word by naming the family was what a
/// test could do and the runtime could not (`palo B-word`). The catalog carries
/// the classifier now, keyed by the same string every other column is, so this
/// goes through it — and a smoke that asked the wrong family for its bits
/// would now be a compile-time mismatch of one `const SKU`, not a silent
/// disagreement with what production computes.
fn word(query_len: u32) -> u64 {
    let classify = runtime::engine::load::classify(SKU).expect("this build ships the smoke's SKU");
    classify(&model::Request::new(query_len, false))
}

/// Greedy: the highest logit.
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

#[test]
fn a_checkpoint_loads_through_the_contract_and_fires_once() {
    if !engine_cuda::device::present() {
        eprintln!("skipping the boot smoke: no CUDA device on this machine");
        return;
    }
    let Some(checkpoint) = snapshot() else {
        eprintln!(
            "skipping the boot smoke: no Qwen3.5-0.8B snapshot in the hugging \
             face cache (set PIE_SMOKE_SNAPSHOT)"
        );
        return;
    };
    let tokenizer = tokenizer::Tokenizer::from_file(&checkpoint.join("tokenizer.json"))
        .expect("the checkpoint's tokenizer loads");

    // 1. THE DOOR. A boot document, exactly the shape `worker` writes, and
    //    the seam reads the one key it is about.
    let mut engine = open::cuda(b"[model]\ndevice = \"cuda:0\"\n").expect("the cuda seam opens");
    assert_eq!(engine.kind(), "cuda");
    assert!(
        engine.device_facts().is_none(),
        "an engine with no load has bound no device"
    );

    // 2. THE RUNTIME TRACES. `request` identifies the checkpoint against the
    //    catalog, traces that SKU's plan, and states the ceilings; `CompiledModel`
    //    never crosses (decision 18).
    let budgets = Budgets {
        max_lanes: 4,
        // Small on purpose: the arena reserves `max_tokens` rows of a
        // 248320-wide logit column, and this test needs a prompt.
        max_tokens: 256,
        buckets: Vec::new(),
        max_adapters: 0,
        page_size: 16,
        max_context: 512,
        slots: 4,
        // The second row axis: derive, which for a text-only plan is no
        // ladder at all (alto multimodal §5.5).
        max_patches: None,
        max_images: None,
    };
    // **DEPTH 1, AND THE NUMBERS DOOR** (alto F2b). This gate submits one
    // frame at a time and reads its logits, which is exactly the caller
    // `Engine::settle_frame` exists for: `submit` answers with the device
    // still running and empty readouts, and the numbers are asked for — and
    // waited for — by name. One frame in flight because that is what this
    // gate does; a deeper ring would carve slots nothing ever claims.
    let request =
        runtime::engine::load::request(
            &checkpoint,
            Platform::Cuda,
            budgets.clone(),
            // Uncapped: every load in this workspace is fully resident
            // (alto design §7 — the tiers are D2's).
            engine::Residency::uncapped(),
            0,
            1,
        )
            .expect("the checkpoint identifies and its SKU traces");
    assert_eq!(
        request.trace.name, SKU,
        "the checkpoint identifies as the row this smoke is about"
    );

    // 3. THE LOAD, and what it answers about itself.
    let loaded = engine.load(request).expect("the checkpoint lands");
    assert_eq!(loaded.facts.trace_name, SKU);
    assert!(
        loaded.facts.weight_bytes > 0 && loaded.facts.arena_bytes > 0,
        "a load that reserved nothing did not happen: {:?}",
        loaded.facts
    );
    let caps = &loaded.caps;
    assert_eq!(caps.device.backend, "cuda");
    assert_eq!(
        caps.device.domain,
        engine::MemoryDomain::CudaDevice(0),
        "the pages this load holds live on the device the boot document named"
    );
    assert_eq!(caps.pools.kv_page_size, budgets.page_size);
    assert_eq!(caps.limits.max_lanes, budgets.max_lanes);
    assert_eq!(caps.limits.max_context, budgets.max_context);
    assert!(
        caps.profile.vocab > 0 && caps.profile.num_layers > 0,
        "the profile is carried, not reconstructed: {:?}",
        caps.profile
    );
    // The shell resolves the DECODE ENVELOPE on the device and nothing wider
    // (`palo B3`): `program::ports` reads `embed_tokens`, `positions` and
    // `kv_len` off an attached instance's own rings, and a lane's page table
    // is still the caller's. The contract's own negotiation says so rather
    // than a fire discovering it.
    assert!(caps.admits(eta_ir::registry::GeometryClass::Host));
    assert!(caps.admits(eta_ir::registry::GeometryClass::DecodeEnvelope));
    assert!(!caps.admits(eta_ir::registry::GeometryClass::DeviceGeometry));
    // **AND WHAT IT SERVES AND WHAT IT DOES NOT ARE BOTH STATED.** `copy_kv`
    // moves cells between pages of THIS load's own pools — a fork, a graft, a
    // prefix-cache hit — and `Capabilities::kv_copy` says so ahead of any
    // plan. The other three directions name a pinned swap pool this shell
    // does not reserve or a peer mapping it has not opened, and each refuses
    // BY NAME rather than pretending: `KvCopy::default()` is host-pinned on
    // both ends, which is the caller's own memmove.
    //
    // `tests/gpu/tests/cuda_kv_page_graft.rs` is where the served direction is
    // gated on real bytes; this line is only about the negotiation.
    assert!(caps.kv_copy.device_to_device);
    assert!(
        !caps.kv_copy.device_to_host
            && !caps.kv_copy.host_to_device
            && !caps.kv_copy.host_to_host
    );
    assert!(matches!(
        engine.copy_kv(&Default::default()),
        Err(engine::Error::Unsupported { engine: "cuda", .. })
    ));

    // 4. THE FIRE. One lane, the prompt, the shell's own page table.
    let prompt = tokenizer.encode(PROMPT);
    assert!(!prompt.is_empty(), "the prompt tokenizes to something");
    let submission = Step {
        lanes: vec![Lane {
            slot: 0,
            word: word(prompt.len() as u32),
            tokens: prompt.clone(),
            positions: Vec::new(),
            kv: engine::KvDelta::default(),
            mask: None,
            adapter: None,
            drafts: false,
            captures_scores: false,
            rs: engine::RsVerb::Fold,
            rs_reset: engine::RsReset::Inferred,
            channels: Vec::new(),
            readout: Readout::Last,
        }],
        attachments: Vec::new(),
    };
    submission
        .validate()
        .expect("the submission is one the contract describes");
    // ONE STEP IS A FRAME OF ONE, and that is the whole of what `fire`
    // became: the contract's forward verb is `submit(FrameSubmission)`, and
    // the fire this smoke test has always run is the degenerate case.
    let frame = engine::FrameSubmission::of(submission);
    frame
        .validate()
        .expect("and the frame it is the one step of");
    let mut ticket = engine.submit(&frame).expect("the prefill fires");
    // **THE RECEIPT COMES BACK WITH THE DEVICE STILL RUNNING** (article 1),
    // so the readouts are empty until this line — which is the wait, named.
    assert!(
        engine.settles_asynchronously(),
        "the cuda engine answers `submit` before the device is done"
    );
    engine
        .settle_frame(&mut ticket)
        .expect("and the numbers door hands back what the fire computed");

    assert_eq!(ticket.steps.len(), 1, "one step in, one receipt out");
    assert_eq!(ticket.steps[0].readouts.len(), 1, "one lane in, one readout out");
    let readout = &ticket.steps[0].readouts[0];
    assert_eq!(readout.rows, 1, "`Readout::Last` is one row");
    assert_eq!(
        readout.width as usize, readout.values.len(),
        "the readout's stated width is the values it carries"
    );
    assert_eq!(
        readout.width, caps.profile.vocab,
        "a logits row is the vocabulary wide, and the profile says how wide"
    );
    let bad = readout.values.iter().position(|value| !value.is_finite());
    assert!(
        bad.is_none(),
        "logit {} is not finite — a pool read at the wrong stride shows up here first",
        bad.unwrap_or(0)
    );

    // 5. AND IT HAS TO BE RIGHT. A load that ran is not a load that works.
    let token = argmax(&readout.values);
    let text = tokenizer.decode(&[token], false);
    eprintln!("continuation: {text:?}");
    assert_eq!(
        text, EXPECTED,
        "greedy continuation of {PROMPT:?} through the contract was {text:?}, \
         and the same checkpoint through `Shell::fire` answers {EXPECTED:?}"
    );
}
