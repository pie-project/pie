//! The boot smoke, on an Apple GPU: a real checkpoint through the CONTRACT,
//! and a real token back.
//!
//! # What this is for, and what it is not
//!
//! `engine-metal/tests/serve_smoke.rs` already asserts that the shell
//! computes — same checkpoint, same prompt, same pinned continuation — and
//! it reaches the shell through `Shell::load`/`Shell::fire`, which is the
//! crate's own surface. This test reaches the SAME device through the
//! boundary the runtime actually uses:
//!
//! ```text
//!   backend::open::metal(boot document)  -> Box<dyn Engine>
//!   engine::load::request(checkpoint)    -> LoadRequest { plan, .. }
//!   Engine::load(request)                -> Loaded { facts, caps }
//!   Engine::fire(Step{lanes})  -> FireTicket { readouts }
//! ```
//!
//! Four things can be wrong here that no test under it can see: the boot
//! document does not reach the shell, the runtime's tracing door picks the
//! wrong SKU or the wrong PLATFORM for a checkpoint, the `Capabilities` the
//! load answers do not describe the load, and a `Step` the runtime
//! builds is not one the shell composes. Each of them is a fire that runs
//! and says the wrong thing.
//!
//! **AND ONE MORE, WHICH IS THIS FILE'S OWN.** The CUDA smoke traces at
//! `Platform::Cuda` and this one at `Platform::Metal`, so the two run
//! DIFFERENT plans of the same model text against the same weights and pin
//! the same token. That is the strongest statement the pair makes together:
//! the continuation is a property of the model rather than of either
//! backend's arithmetic.
//!
//! # Gating
//!
//! An Apple target is not a machine with a GPU, and neither is a machine
//! with a GPU one with a 1.4 GB checkpoint on its disk. So it skips at RUN
//! time, saying which it was missing, rather than being `#[ignore]`d.
//!
//! ```text
//! cargo test -p runtime --features engine-metal --test metal_boot_smoke -- --nocapture
//! ```
//!
//! `PIE_SMOKE_SNAPSHOT` overrides where it looks.

#![cfg(all(feature = "engine-metal", target_vendor = "apple"))]

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
/// `engine-metal/tests/serve_smoke.rs` pins, which is the same token
/// `engine-cuda`'s pair pins. Two backends and two paths to one answer.
const EXPECTED: &str = " Paris";

/// The snapshot directory: the checkpoint AND the tokenizer that goes with
/// it, because a vocabulary from another snapshot decodes the right ids into
/// the wrong words.
fn snapshot() -> Option<PathBuf> {
    if let Ok(stated) = std::env::var("PIE_SMOKE_SNAPSHOT") {
        let path = PathBuf::from(stated);
        return path.is_dir().then_some(path);
    }
    // The suite runs as root over `tailscale ssh`, so `HOME` is not the
    // owner's — the cache the checkpoint actually lives in is named
    // explicitly beside it.
    let homes = [
        std::env::var("HOME").unwrap_or_default(),
        "/Users/ingim".to_string(),
    ];
    homes.iter().find_map(|home| {
        let snapshots =
            Path::new(home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots");
        std::fs::read_dir(snapshots)
            .ok()?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .find(|path| path.join("tokenizer.json").exists())
    })
}

/// The lane word the model's own `Classify` computes, reached the way the
/// fire path reaches it — through the catalog's own classifier column, keyed
/// by the same string every other column is.
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
    if !engine_metal::device::present() {
        eprintln!("skipping the boot smoke: this machine publishes no Metal device");
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

    // 1. THE DOOR. A boot document, exactly the shape `worker` writes. The
    //    metal seam reads no device key — `MTLCreateSystemDefaultDevice` is
    //    the whole of device selection on this platform — so the document is
    //    handed over for the model id and nothing else.
    let mut engine = open::metal(b"[model]\nid = \"qwen35-d0.8b\"\n").expect("the metal seam opens");
    assert_eq!(engine.kind(), "metal");
    assert!(
        engine.device_facts().is_none(),
        "an engine with no load has bound no device"
    );

    // 2. THE RUNTIME TRACES, AT THIS PLANE'S PLATFORM. `request` identifies
    //    the checkpoint against the catalog, traces that SKU's plan for
    //    `Platform::Metal`, and states the ceilings; `CompiledModel` never crosses
    //    (decision 18).
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
    };
    let request = runtime::engine::load::request(
            &checkpoint,
            Platform::Metal,
            budgets.clone(),
            // Uncapped: every load in this workspace is fully resident
            // (alto design §7 — the tiers are D2's).
            engine::Residency::uncapped(),
            0,
            1,
        )
        .expect("the checkpoint identifies and its SKU traces");
    assert_eq!(
        request.plan.name, SKU,
        "the checkpoint identifies as the row this smoke is about"
    );
    assert_eq!(
        request.plan.platform,
        Platform::Metal,
        "the plan that crosses the boundary was traced for the plane behind it"
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
    assert_eq!(caps.device.backend, "metal");
    assert_eq!(caps.pools.kv_page_size, budgets.page_size);
    assert_eq!(caps.limits.max_lanes, budgets.max_lanes);
    assert_eq!(caps.limits.max_context, budgets.max_context);
    assert!(
        caps.profile.vocab > 0 && caps.profile.num_layers > 0,
        "the profile is carried, not reconstructed: {:?}",
        caps.profile
    );
    // **THIS PLANE RESOLVES THE DECODE ENVELOPE ON THE DEVICE**, so it
    // admits that class and the HOST class beneath it, and nothing wider.
    // `serve::stage` reads `embed_tokens`, `positions` and `kv_len` off an
    // attached instance's own ring at step 0b — the same read the CUDA shell
    // has always made — so a decode loop's sampled token never leaves the
    // device. What stays refused is `DeviceGeometry`: its other four ports
    // describe a page table this shell owns and derives from the seat, and
    // the refusal arrives through the contract's own negotiation at
    // `bind_instance` rather than through a fire that discovers it.
    assert!(caps.admits(eta_ir::registry::GeometryClass::Host));
    assert!(caps.admits(eta_ir::registry::GeometryClass::DecodeEnvelope));
    assert!(!caps.admits(eta_ir::registry::GeometryClass::DeviceGeometry));

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
        media: Vec::new(),
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
    let ticket = engine.submit(&frame).expect("the prefill fires");

    assert_eq!(ticket.steps.len(), 1, "one step in, one receipt out");
    assert_eq!(ticket.steps[0].readouts.len(), 1, "one lane in, one readout out");
    let readout = &ticket.steps[0].readouts[0];
    assert_eq!(readout.rows, 1, "`Readout::Last` is one row");
    assert_eq!(
        readout.width as usize,
        readout.values.len(),
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
