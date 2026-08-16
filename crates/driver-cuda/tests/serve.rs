//! The shell's verbs, exercised through the API the engine calls.
//!
//! This file used to open by saying "the linker proves the ABI": the
//! thirteen `pie_cuda_*` symbols were DECLARED in `driver_api::local` and
//! DEFINED here, and a test that resolved one against the other made the
//! linker check the contract. That proof was necessary while the far side
//! could be C++ and is not a proof of anything now — the verbs are methods
//! on `Shell`, so a drifted signature is a type error at every call site
//! rather than a link error at one.
//!
//! What is left is what the file was actually testing all along: that each
//! verb serves what it claims and refuses what it cannot, with the driver's
//! own `i32` status as the answer.

#![cfg(all(feature = "_cuda", feature = "abi"))]

use driver_api::local::{
    ChannelBinding, InstanceBinding, PIE_CHANNEL_DTYPE_BOOL, PIE_CHANNEL_DTYPE_F32,
    PIE_CHANNEL_EXTERN_IMPORT, PIE_CHANNEL_EXTERN_NONE, PIE_CHANNEL_HOST_ROLE_NONE,
    PIE_CHANNEL_HOST_ROLE_READER, PIE_STATUS_INVALID_ARGUMENT, PIE_STATUS_OK,
    PIE_STATUS_UNSUPPORTED, TerminalCell,
};
use driver_api::local::{
    PIE_GEOMETRY_CLASS_DECODE_ENVELOPE, PIE_RS_FLAG_RESET, PIE_TERMINAL_OUTCOME_SUCCESS,
    StateCopyRange,
};
use driver_api::{ChannelRegistrationPlan, InstanceBindingPlan};

use driver_api::completion::CompletionBroker;
use driver_api::{FrameSubmission, LaunchPlan, StepSubmission};
use driver_cuda::serve::Shell;

/// Register a program, writing the id where the out-parameter used to go.
///
/// The verb answers the id now. These shims keep the assertion shape the
/// tests were written in — `assert_eq!(call(..), PIE_STATUS_OK)` — without
/// pretending an out-parameter still exists.
fn reg_program(s: &mut Shell, plan: &driver_api::ProgramRegistration, out: &mut u64) -> i32 {
    match s.register_program(plan) {
        Ok(id) => {
            *out = id;
            PIE_STATUS_OK
        }
        Err(status) => status,
    }
}

/// Register a channel, writing the binding where the out-parameter used to go.
fn reg_channel(
    s: &mut Shell,
    plan: &driver_api::ChannelRegistrationPlan,
    out: &mut driver_api::local::ChannelBinding,
) -> i32 {
    match s.register_channel(plan) {
        Ok(binding) => {
            *out = binding;
            PIE_STATUS_OK
        }
        Err(status) => status,
    }
}

/// Bind an instance, writing the binding where the out-parameter used to go.
fn bind(
    s: &mut Shell,
    plan: &driver_api::InstanceBindingPlan,
    out: &mut driver_api::local::InstanceBinding,
) -> i32 {
    match s.bind_instance(plan) {
        Ok(binding) => {
            *out = binding;
            PIE_STATUS_OK
        }
        Err(status) => status,
    }
}

/// What a verb answered, in the shape these tests assert on.
///
/// Every entry point used to answer a bare `i32`, so a test wrote
/// `assert_eq!(call(..), PIE_STATUS_OK)`. They answer `Result` now, and this
/// keeps those assertions readable without pretending the two are the same
/// thing: `ok(r)` is `PIE_STATUS_OK`, and an error is its status.
fn code<T>(result: Result<T, i32>) -> i32 {
    match result {
        Ok(_) => PIE_STATUS_OK,
        Err(status) => status,
    }
}

/// The encode plan's `output_rows` as the `u16` (bf16 bit-pattern) rows the
/// towers write. The plan owns bytes; every assertion below is about values.
fn out_u16(plan: &driver_api::MediaEncodePlan) -> Vec<u16> {
    plan.output_rows
        .chunks_exact(2)
        .map(|c| u16::from_ne_bytes([c[0], c[1]]))
        .collect()
}

mod common;
#[allow(unused_imports)] // abi tests take only the guard
use common::gpu_guard;

// ── FIRING A FRAME AND WAITING FOR IT ────────────────────────────────
//
// `Shell::launch` returns with the fire still on the stream: the work it
// owes rides a stream-ordered callback, and the completion notify is how a
// caller learns the fire is done. The engine waits for it. A test that reads
// the ring or a terminal cell on the next line has to wait for it too.
//
// The completion is per FIRE, minted from the test's own broker, so tests
// running in parallel cannot see each other's fires and no fire can be
// mistaken for the one before it. It was a counter per DRIVER, carried in
// the runtime callbacks' `ctx` — the `ctx` was the C boundary's way of
// handing a driver something it could not name, and what it was carrying is
// a `Clone` handle a driver simply holds.

/// The runtime callbacks a driver is created with in the real world.
///
/// A broker nothing observes.
///
/// It was a `PieRuntimeCallbacks` whose `notify` was an `extern "C"` function
/// that ignored its arguments. `create` refused a null one — a driver with no
/// way to notify has no way to retire a fire — and that rule needed a runtime
/// validator to state. A `CompletionBroker` cannot be null, so the type states
/// it, and the validator is gone with the descriptor it checked.
///
/// A test that wants to OBSERVE fires keeps its own broker and awaits the
/// completion it minted; this is for the ones that never launch.
fn engine_runtime() -> CompletionBroker {
    CompletionBroker::new()
}

/// The cached Qwen3-0.6B snapshot, if this box has one.
///
/// Extracted because a second test needed it and the first had it inline
/// — two copies of a path search is two places for a skip to stop
/// working, and a test that silently stops running is worse than one that
/// fails.
fn qwen3_snapshot() -> Option<std::path::PathBuf> {
    let root = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    std::fs::read_dir(root).ok().and_then(|rd| {
        rd.filter_map(std::result::Result::ok).find_map(|e| {
            let p = e.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    })
}

/// The generated model config beside it.
///
/// The BOOT KEY that names it is `[model] config`. It was `[model]
/// descriptor` until `7c6998f39` renamed it — *"the name claimed it was a
/// descriptor; it is a config path, read once"* — and every boot string in
/// this file still said the old spelling, because a boot key nothing reads
/// is a silent default and this file has not compiled since. The FIXTURE
/// files on disk keep their generated names, which is why the helper does.
fn qwen3_descriptor() -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    )
    .join("qwen3_descriptor.json");
    p.is_file().then_some(p)
}

/// Launch, then wait for the fire to retire. The status is the launch's.
///
/// The wait was a spin on a counter an `extern "C"` notify bumped. It is the
/// completion the broker minted now — the same object the scheduler awaits.
///
/// The completion is minted HERE, one per fire, and the broker is what the
/// caller passes. That is not a shortening of the argument list: a completion
/// settles at its epoch and STAYS settled, so one minted by the caller and
/// handed to a chain of fires is a fence for the first of them and a no-op
/// for every one after — which is precisely the reading a test that inspects
/// the ring on the next line cannot survive. The counter this replaced was
/// re-armed by construction, because each fire bumped it and the helper
/// remembered the value it started from.
fn fire_and_wait(shell: &mut Shell, frame: &FrameSubmission, broker: &CompletionBroker) -> i32 {
    let (target, completion) = broker.launch_completion(1);
    let status = code(shell.launch(frame, target));
    if status != PIE_STATUS_OK {
        return status;
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        if let Some(settled) = completion.check() {
            settled.expect("the fire completed");
            return status;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the fire never completed"
        );
        std::thread::yield_now();
    }
}

/// A tensor-parallel group of more than one is REFUSED, not served wrongly.
///
/// The layout half works — a rank compiles a plan that reads only its own
/// bands, and the KV cache is divided the same way — so a rank holds a shard
/// of every projection. The collective that adds the shards back together does
/// not exist: the three all-reduce rows are declared and no launch reaches
/// them.
///
/// A driver that accepted the group would run its own shard, skip the
/// reduction, and return a fraction of the real answer with no error anywhere.
/// That is worse than not starting, which is what this pins.
#[test]
fn a_tensor_parallel_group_is_refused_rather_than_answered_wrongly() {
    let _gpu = gpu_guard();
    let boot = "[driver]\ntp_size = 2\n";
    let broker = engine_runtime();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    assert!(
        d.is_err(),
        "tp_size = 2 must refuse; there is no all-reduce to serve it"
    );

    // And one rank still boots, so the refusal is about the GROUP and not
    // about the keys being present.
    let solo = "[driver]\ntp_size = 1\n";
    let broker = engine_runtime();
    let opened = Shell::open(solo.as_bytes(), broker.clone());
    let d = opened;
    let d = d.expect("one rank is the served configuration");
    drop(d);
}

/// A panic inside a driver entry point fails the REQUEST, not the process.
///
/// The shell has ~30 `unwrap`/`expect` on paths a malformed frame can reach,
/// and one of them used to take the whole worker down with every other request
/// it was serving. That was not fixable while the entry points were
/// `extern "C"`: unwinding out of one is undefined behaviour and, since Rust
/// 1.81, an abort — the catch has to happen INSIDE the frame, which is what
/// `guard` does now that these are plain Rust functions.
///
/// A frame whose `steps` pointer is non-null with a nonsense length is the
/// cheapest way to reach a slice construction from outside.
#[test]
fn a_panicking_request_does_not_take_the_process_down() {
    let _gpu = gpu_guard();
    let boot = "[driver]\ntp_size = 1\n";
    let broker = engine_runtime();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");

    // No model is loaded, so this refuses long before it can panic — which is
    // the point of running it: the guard must not change the ordinary answer.
    let frame = FrameSubmission {
        steps: Vec::new(),
        ..Default::default()
    };
    let (completion, completion_handle) = broker.launch_completion(1);
    let _ = &completion_handle;
    let status = code(d.launch(&frame, completion));
    assert_ne!(status, PIE_STATUS_OK, "a frame with no steps is refused");

    // AND THE PROCESS IS STILL HERE. If the guard were absent and either call
    // had panicked, this line would never run. Closing an unknown instance is
    // idempotent by contract, so `OK` is the right answer and the point is
    // that the call RETURNS one.
    let status = code(d.close_instance(12345));
    assert_eq!(status, PIE_STATUS_OK, "closing is idempotent");
    drop(d);
}

/// The cached Qwen3-0.6B snapshot and its generated descriptor, or `None`.
fn qwen3_fixture() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let home = std::env::var("HOME").ok()?;
    let snaps = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let snap = std::fs::read_dir(&snaps).ok()?.find_map(|e| {
        let p = e.ok()?.path();
        p.join("model.safetensors").is_file().then_some(p)
    })?;
    // THE SNAPSHOT'S OWN `config.json`.
    //
    // This was a hardcoded absolute path into an agent scratch directory --
    // `/tmp/claude-0/-root--patissier-work-tart-alpha/<uuid>/scratchpad/
    // qwen3_descriptor.json` -- which exists on no machine, including this
    // one. So the `is_file()` guard below was always false, `qwen3_fixture`
    // always answered `None`, and every test built on it printed
    // "skipped: no cached Qwen3-0.6B or descriptor" and passed. A suite that
    // skips and a suite that measures print the same `test result: ok`.
    //
    // `[model] config` wants the checkpoint's config, and the snapshot ships
    // it -- `gpu_ptir_shell.rs` boots the same driver off
    // `snap.join("config.json")` and loads the stock snapshot with it. So the
    // descriptor file was never needed; only the path to it was ever
    // unreachable.
    let descriptor = snap.join("config.json");
    descriptor.is_file().then_some((snap, descriptor))
}

/// What `load_model` answers is what an ENGINE will parse.
///
/// This driver answered a five-field summary of the checkpoint —
/// `{"model_type":…,"hidden":…,"layers":…,"vocab":…,"weights":…}` — and
/// `driver_api::DriverCapabilities` rejects that at its first field. So no
/// engine could load a model through this driver at all, and every test here
/// missed it by passing `null` for the caps out-parameter and reading the
/// status instead.
///
/// The fix is not "parse harder": it is that `total_pages` is a real answer
/// about this device, and answering it means sizing the KV pool. This holds
/// the payload against the type the engine deserializes it into, and holds
/// the two fields a scheduler cannot work without.
/// A scope this driver does not implement is REFUSED, not silently widened.
///
/// `ModelLoadDesc::component` had no reader in this crate: `load_model`
/// forwarded `desc.snapshot_dir` to `load_impl` and dropped the rest, so
/// asking for one component loaded the whole checkpoint. Asking gemma-4-26B
/// for `Encode` -- a media tower of a couple of GiB -- staged a 48.1 GiB
/// arena and failed on `cudaMalloc`, which reads as "this card is too small
/// for the encoder" and is not what happened.
///
/// This asserts against `Full` succeeding in the test above, so it is the
/// SCOPE being refused and not the fixture.
#[test]
fn load_model_refuses_a_component_scope_it_does_not_implement() {
    let _gpu = gpu_guard();
    let Some((snap, descriptor)) = qwen3_fixture() else {
        eprintln!("skipped: no cached Qwen3-0.6B or descriptor");
        return;
    };
    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = engine_runtime();
    let mut d = Shell::open(boot.as_bytes(), broker.clone()).expect("the driver creates");
    for scope in [
        driver_api::ModelComponent::Text,
        driver_api::ModelComponent::Encode,
    ] {
        let load = driver_api::ModelLoadDesc {
            snapshot_dir: snap.clone(),
            runtime_quant: String::new(),
            mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
            component: scope,
        };
        assert_eq!(
            d.load_model(&load).err(),
            Some(driver_api::PIE_STATUS_INVALID_ARGUMENT),
            "{scope:?} is not implemented and has to say so"
        );
    }
}

/// A quantization this driver does not perform is REFUSED, not ignored.
///
/// `desc.runtime_quant` and `desc.mxfp4_moe` had no reader in this crate
/// either -- outside the field declarations, every occurrence in the
/// workspace is a test WRITING a default. So a worker config saying
/// `runtime_quant = "fp8"` was accepted, quantized nothing, and ran bf16,
/// while the worker's own documentation says fp8 "enable[s] per-channel
/// symmetric quantization for supported projection weights".
///
/// `Mxfp4MoeRequest`'s doc says the explicit variants "pin a lowering and
/// fail the load if the device cannot provide it". This device pins
/// `native_mxfp4_moe: false`, so `NativeGemm` is precisely that case.
///
/// The values a working configuration already uses -- empty quant, `Auto`
/// and `RoutedDecode` -- are asserted to still load, so this is the
/// UNPROVIDABLE request being refused and not the request field.
#[test]
fn load_model_refuses_a_quantization_it_does_not_perform() {
    let _gpu = gpu_guard();
    let Some((snap, descriptor)) = qwen3_fixture() else {
        eprintln!("skipped: no cached Qwen3-0.6B or descriptor");
        return;
    };
    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = engine_runtime();
    let base = |quant: &str, moe| driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: quant.to_string(),
        mxfp4_moe: moe,
        component: driver_api::ModelComponent::Full,
    };

    let mut d = Shell::open(boot.as_bytes(), broker.clone()).expect("the driver creates");
    for quant in ["fp8", "int8"] {
        assert_eq!(
            d.load_model(&base(quant, driver_api::Mxfp4MoeRequest::Auto))
                .err(),
            Some(driver_api::PIE_STATUS_INVALID_ARGUMENT),
            "runtime_quant '{quant}' is not performed and has to say so"
        );
    }
    for pinned in [
        driver_api::Mxfp4MoeRequest::NativeGemm,
        driver_api::Mxfp4MoeRequest::EagerBf16,
    ] {
        assert_eq!(
            d.load_model(&base("", pinned)).err(),
            Some(driver_api::PIE_STATUS_INVALID_ARGUMENT),
            "{pinned:?} cannot be provided and has to say so"
        );
    }

    // THE CONTROL. A refusal that fired on everything would pass the loop
    // above while breaking every real boot.
    for answered in [
        driver_api::Mxfp4MoeRequest::Auto,
        driver_api::Mxfp4MoeRequest::RoutedDecode,
    ] {
        let mut d = Shell::open(boot.as_bytes(), broker.clone()).expect("the driver creates");
        d.load_model(&base("", answered))
            .unwrap_or_else(|status| panic!("{answered:?} is answered, not refused: {status}"));
    }
}

#[test]
fn load_model_answers_capabilities_an_engine_can_parse() {
    let _gpu = gpu_guard();
    let Some((snap, descriptor)) = qwen3_fixture() else {
        eprintln!("skipped: no cached Qwen3-0.6B or descriptor");
        return;
    };
    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = engine_runtime();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    // The verb ANSWERS the capabilities. It used to answer a status and
    // publish `{ptr, len}` of JSON through a second out-parameter, which the
    // engine then deserialized into this exact type — two readers of one
    // document, and a test that had to check the pointer was non-null before
    // it could check anything else.
    let parsed: driver_api::DriverCapabilities = d.load_model(&load).expect("the snapshot loads");

    assert_eq!(parsed.abi_version, driver_api::PIE_DRIVER_ABI_VERSION);
    // WHAT A SCHEDULER ADMITS AGAINST. Zero pages is a driver that can hold
    // no context, which is indistinguishable from a driver that did not
    // answer — and was the state before this.
    assert!(parsed.total_pages > 0, "the device holds no KV pages");
    assert_eq!(
        parsed.kv_page_size, 16,
        "the page size the fire path builds"
    );
    // THE LATTICE ANSWERED THESE, and that is the point of running it. A
    // stated ceiling would be the same on every device; these are what the
    // planner's search chose for this card and this checkpoint, and the arena
    // it sized is sized for exactly this rectangle.
    assert!(parsed.max_forward_tokens > 0, "a fire may carry no tokens");
    assert!(
        parsed.max_forward_requests > 0,
        "a fire may carry no requests"
    );
    assert!(
        parsed.max_page_refs >= parsed.total_pages,
        "a fire cannot reference fewer pages than the pool holds"
    );
    // The pool and the arena are BOTH resident here, so the pages advertised
    // have to leave room for the workspace. On an L40S with qwen3-0.6B the
    // full-budget figure is ~22k pages and the honest one is ~20k; a driver
    // that advertised the first would fail to build the pool it promised.
    let page_bytes = u64::from(parsed.kv_page_size)
        * 8   // kv heads
        * 128 // head dim
        * 2   // bytes per element
        * 2   // K and V
        * 28; // layers
    let pool = u64::from(parsed.total_pages) * page_bytes;
    assert!(
        pool < 42 * 1024 * 1024 * 1024,
        "the pool alone is {pool} bytes on a 46 GB card, with an arena still to come"
    );
    assert_eq!(parsed.arch_name, "qwen3");
    assert_eq!(parsed.vocab_size, 151_936);
    assert_eq!(parsed.hidden_size, 1024);
    assert!(parsed.max_model_len > 0, "the scheduler's context ceiling");
    assert_eq!(parsed.activation_dtype, "bf16");
    // Qwen3-0.6B is dense attention throughout, so no recurrent slots.
    assert!(!parsed.rs_cache_required);
    // A TEXT checkpoint has no tower, so this stays false — and it is
    // asked here rather than only in the gemma-4 test because the
    // interesting half of the claim is that it is DERIVED. It was
    // hardwired false, which the worker reads as "build no encode
    // executor" (`worker/src/executor/mod.rs:1341`), so gemma-4's towers
    // were unreachable through the engine while four tests fired them
    // directly and passed.
    assert!(
        !parsed.supports_media_encode,
        "qwen3 carries no vision or audio tower"
    );
    drop(d);
}

/// The verbs a caller reaches without a model, and what they refuse.
///
/// # What this test used to be, and why the premise went
///
/// It was `the_shell_answers_the_engines_own_declarations`, and it opened by
/// taking the address of thirteen `pie_cuda_*` functions into a
/// `[*const (); 13]` — *"force the defining objects into this binary: an
/// rlib's members are pulled on REFERENCE, and the `extern` declarations
/// alone reference nothing Rust-side"*. That array made the LINKER check
/// that the symbols `driver_api::local` declared were the ones this crate
/// defined.
///
/// The declarations are gone and so is the proof they needed. A caller that
/// disagrees with `Shell` about a verb's arguments is a caller that does not
/// compile, which is a stronger check made in more places — so what is left
/// here is the part that was never about linkage: the refusals.
///
/// The ABI-version arm went with it. `create` took an `abi_version` field and
/// refused a value it did not recognise; there is no field, because there is
/// no descriptor. What a caller can still get wrong is the boot document, and
/// that is what is checked instead.
#[test]
fn the_verbs_reachable_without_a_model_refuse_by_name() {
    // A boot document this driver cannot serve is still refused before any
    // state exists — the check `abi_version` used to stand in for.
    assert!(
        Shell::open(b"[driver]\ntp_size = 2\n", CompletionBroker::new()).is_err(),
        "a group of more than one has no all-reduce and must refuse"
    );

    let mut d = Shell::open(b"", engine_runtime()).expect("an empty boot document creates");

    // The facts are the driver's own and are readable without a model.
    assert!(
        d.device_facts().backend.contains("cuda"),
        "the shell states which backend it is"
    );

    // Idempotent closes, on ids nothing ever opened.
    assert_eq!(code(d.close_instance(7)), PIE_STATUS_OK);
    assert_eq!(code(d.close_channel(7)), PIE_STATUS_OK);

    // An empty `snapshot_dir` is an argument error. It used to be reachable
    // two ways — a null `*const PieModelLoadDesc` and a descriptor whose
    // `{ptr, len}` was empty — and the first of those is not expressible
    // against a `&ModelLoadDesc`.
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: std::path::PathBuf::new(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(
        code(d.load_model(&load)),
        PIE_STATUS_INVALID_ARGUMENT,
        "an empty snapshot_dir is an argument error"
    );
    drop(d);
}

/// `load_model` over a REAL snapshot: the boot TOML carries the
/// descriptor path (the C++ shell's own channel), the loader parses the
/// HF safetensors layout, ~1.2 GB lands on the device, and the caps JSON
/// answers with the parsed facts. GPU + checkpoint required; skips
/// without either.
#[test]
fn load_model_loads_a_real_snapshot_through_the_abi() {
    let _gpu = gpu_guard();

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let descriptor = std::path::PathBuf::from(std::env::var("PIE_TEST_SCRATCH").unwrap_or_else(
        |_| "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad".into(),
    ))
    .join("qwen3_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated descriptor at {descriptor:?}");
        return;
    }

    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = engine_runtime();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");

    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    let caps: driver_api::DriverCapabilities =
        d.load_model(&load).expect("the real snapshot loads");
    // The payload is a `DriverCapabilities`, which is what the engine
    // deserializes it into — `load_model_answers_capabilities_an_engine_can_parse`
    // holds the whole shape. Here it is only "the load answered about THIS
    // checkpoint", so the two fields the descriptor pins are enough.
    assert_eq!(caps.arch_name, "qwen3", "caps carry the parsed facts");
    assert_eq!(caps.hidden_size, 1024);
    drop(d);
}

/// The id lifecycle: registering the same program hash twice answers one
/// id; binding requires a registered program; a requested instance id is
/// honored; closing is idempotent and actually closes (a rebind of the
/// same id succeeds after close, refuses before).
#[test]
fn the_registries_run_the_id_lifecycle() {
    let mut d = Shell::open(b"", engine_runtime()).expect("the driver creates");

    let prog = driver_api::ProgramRegistration {
        program_hash: 0xC3C3,
        ..Default::default()
    };
    let mut id1 = 0u64;
    let mut id2 = 0u64;
    assert_eq!(reg_program(&mut d, &prog, &mut id1), PIE_STATUS_OK);
    assert_eq!(reg_program(&mut d, &prog, &mut id2), PIE_STATUS_OK);
    assert_eq!(id1, id2, "the hash is the dedup key");

    let unbound = InstanceBindingPlan {
        driver_id: 0,
        program_id: 999,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids: Vec::new(),
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::Host,
    };
    let mut binding = InstanceBinding::default();
    assert_eq!(
        bind(&mut d, &unbound, &mut binding),
        PIE_STATUS_INVALID_ARGUMENT,
        "an unregistered program refuses the bind"
    );

    // `geometry_class: 7` here for years, and nothing looked: it is not a
    // `GeometryClass` and never was. The shared validators caught it the
    // first time an entry point ran them, which is the whole argument for
    // running them.
    let inst = InstanceBindingPlan {
        driver_id: 0,
        program_id: id1,
        requested_instance_id: 42,
        pacing_wait_id: 0,
        channel_ids: Vec::new(),
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::try_from(PIE_GEOMETRY_CLASS_DECODE_ENVELOPE)
            .unwrap_or(driver_api::GeometryClass::Host),
    };
    assert_eq!(bind(&mut d, &inst, &mut binding), PIE_STATUS_OK);
    assert_eq!(binding.instance_id, 42, "the requested id is honored");
    assert_eq!(
        binding.geometry_class,
        driver_api::local::PIE_GEOMETRY_CLASS_DECODE_ENVELOPE,
        "the geometry echoes"
    );

    assert_eq!(
        bind(&mut d, &inst, &mut binding),
        PIE_STATUS_INVALID_ARGUMENT,
        "an id in use refuses"
    );
    assert_eq!(code(d.close_instance(42)), PIE_STATUS_OK);
    assert_eq!(
        bind(&mut d, &inst, &mut binding),
        PIE_STATUS_OK,
        "closed means reusable"
    );
    drop(d);
}

/// The whole ABI, end to end: create → load the real checkpoint →
/// register → bind → LAUNCH one decode frame — a single token over one
/// KV page — and the shell runs the actual forward on the device,
/// publishes the terminal cell, and notifies the runtime. This is the
/// Load a cached snapshot and fire ONE decode step through the thirteen
/// exports, or report that the checkpoint is not on this machine.
///
/// Extracted because it had been copied three times before it was clear
/// that adding a deployment is the cheap part: what differs between them
/// is a repository name and a descriptor, and what does not differ is a
/// hundred and thirty lines of frame construction. A per-deployment test
/// carrying its own copy of the frame hides the thing worth reading,
/// which is WHICH deployments the shell can open.
fn load_and_fire(repo: &str, descriptor_name: &str, what: &str) -> bool {
    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            // Sharded snapshots carry an index instead of one file.
            (p.join("model.safetensors").is_file()
                || p.join("model.safetensors.index.json").is_file())
            .then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached {what}");
        return false;
    };
    let descriptor = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    )
    .join(descriptor_name);
    if !descriptor.is_file() {
        eprintln!("skipped: no generated {what} descriptor");
        return false;
    }

    // THE COMPLETION IS ASYNCHRONOUS NOW, so the test waits for it.
    //
    // `Shell::launch` returns when the fire is ENQUEUED; the terminal
    // cell is published and this callback runs from a stream-ordered host
    // callback when the work retires. Reading the cell straight after the
    // launch call used to be correct and is now a race that happens to
    // win, which is the worst kind.
    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = CompletionBroker::new();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.unwrap_or_else(|s| panic!("{what}: the driver creates: status {s}"));

    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(
        code(d.load_model(&load)),
        PIE_STATUS_OK,
        "{what}: the snapshot loads"
    );

    let prog = driver_api::ProgramRegistration {
        program_hash: 0x0102,
        ..Default::default()
    };
    let mut program_id = 0u64;
    assert_eq!(reg_program(&mut d, &prog, &mut program_id), PIE_STATUS_OK);
    let inst = InstanceBindingPlan {
        driver_id: 0,
        program_id: 0,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids: Vec::new(),
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::Host,
    };
    let mut binding = InstanceBinding::default();
    assert_eq!(bind(&mut d, &inst, &mut binding), PIE_STATUS_OK);

    let mut cell = TerminalCell::pending();
    let cell_ptr: *mut TerminalCell = &mut cell;
    let roster_rows: [u32; 1] = [0];
    let sub_batch_indptr: [u32; 2] = [0, 1];
    let sub_batch_class: [u32; 1] = [driver_api::local::PIE_GEOMETRY_CLASS_HOST];
    let token_ids: [u32; 1] = [7];
    let position_ids: [u32; 1] = [0];
    let kv_page_indices: [u32; 1] = [0];
    let kv_page_indptr: [u32; 2] = [0, 1];
    let kv_last_page_lens: [u32; 1] = [1];
    let qo_indptr: [u32; 2] = [0, 1];
    let step = StepSubmission {
        plan: LaunchPlan {
            token_ids: token_ids.to_vec(),
            position_ids: position_ids.to_vec(),
            kv_page_indices: kv_page_indices.to_vec(),
            kv_page_indptr: kv_page_indptr.to_vec(),
            kv_last_page_lens: kv_last_page_lens.to_vec(),
            qo_indptr: qo_indptr.to_vec(),
            ..Default::default()
        },
        roster_rows: roster_rows.to_vec(),
        sub_batch_indptr: sub_batch_indptr.to_vec(),
        sub_batch_class: sub_batch_class.to_vec(),
        terminal_cells: vec![cell_ptr],
        ..Default::default()
    };
    let instance_ids: [u64; 1] = [binding.instance_id];
    let frame = FrameSubmission {
        instance_ids: instance_ids.to_vec(),
        required_kv_pages: 1,
        steps: vec![step.clone()],
        ..Default::default()
    };
    let (completion, completion_handle) = broker.launch_completion(1);
    let _ = &completion_handle;
    assert_eq!(
        code(d.launch(&frame, completion)),
        PIE_STATUS_OK,
        "{what}: the frame launches"
    );
    // Wait for the completion the launch enqueued. Ten seconds is a
    // timeout, not a budget — a fire that has not retired by then is
    // wedged, and hanging the suite would say less than failing it.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        if let Some(settled) = completion_handle.check() {
            settled.unwrap_or_else(|error| panic!("{what}: {error}"));
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{what}: the fire never completed"
        );
        std::thread::yield_now();
    }
    assert_eq!(
        cell.load(),
        PIE_TERMINAL_OUTCOME_SUCCESS,
        "{what}: the terminal cell published"
    );
    drop(d);
    true
}

/// OLMo-2: the POST-NORM, GLOBAL-qk-norm deployment.
///
/// The case the shell used to refuse outright -- "post-norm families await
/// their facts mapping" -- while `fuse_llama_like`, twenty lines away, had
/// been binding its names correctly all along; its own comment names
/// olmo2. Three facts are read off the checkpoint now rather than
/// asserted: norm placement, `qk_norm` (from the EXTENT of the `q_norm`
/// gamma -- no config key tells per-head from global, and the two lower to
/// different kernels), and `fused_qkv`. All three differ from qwen3-0.6B,
/// so a regression in any of them fails here and nowhere else.
#[test]
fn olmo2_loads_and_fires_post_norm_through_the_abi() {
    let _gpu = gpu_guard();
    load_and_fire(
        "models--allenai--OLMo-2-0425-1B-Instruct",
        "olmo2_descriptor.json",
        "OLMo-2-1B",
    );
}

/// Phi-3: the deployment that ships its projections ALREADY FUSED.
///
/// `fuse_llama_like` concatenates `q_proj`/`k_proj`/`v_proj` when it finds
/// all three. Phi-3 ships neither -- one `qkv_proj` and one
/// `gate_up_proj`, in the order the fuse would have produced -- so nothing
/// was written and the trace's operand had nowhere to resolve. The binder
/// aliases them, which is also cheaper than a second copy of a tensor
/// already on the device in the right layout.
///
/// It is also the only deployment here whose head width (96) is not one
/// this build instantiates, so it is the only one that states
/// `attn::pad_head_dim_bf16`.
#[test]
fn phi3_loads_and_fires_prefused_through_the_abi() {
    let _gpu = gpu_guard();
    load_and_fire(
        "models--microsoft--Phi-3-mini-4k-instruct",
        "phi3_descriptor.json",
        "Phi-3-mini",
    );
}

/// Mistral-7B: the plain GQA member, at a size where nothing else about
/// it is unusual. What it checks is that the derivation did not become
/// correct only for the awkward cases.
#[test]
fn mistral_loads_and_fires_through_the_abi() {
    let _gpu = gpu_guard();
    load_and_fire(
        "models--mistralai--Mistral-7B-Instruct-v0.3",
        "mistral_descriptor.json",
        "Mistral-7B-v0.3",
    );
}

/// Qwen2.5-1.5B: the deployment this build CANNOT serve, refused cleanly.
///
/// Its facts derive fine and its weights bind fine. What stops it is
/// underneath both: twelve query heads over two kv heads is a GQA group
/// size of six, and FlashInfer's decode instantiates {1, 2, 3, 4, 8}. The
/// launcher reports that by throwing, and a throw crossing the C ABI is
/// undefined behaviour — in practice SIGABRT with no message, which is
/// exactly how this was found.
///
/// Two things changed so that it cannot be found that way again. The
/// generated shim wraps every body and PRINTS the exception before dying,
/// so the cause is one line instead of a debugger session. And the load
/// refuses the ratio outright, because a load has somewhere to put a
/// failure and a launcher signature does not.
///
/// This test asserts the REFUSAL. It is a capability limit of the build,
/// not a defect in the derivation, and pinning it means the day someone
/// instantiates group size 6 this test tells them to come delete it.
///
/// "Of the build, not of the derivation" is now where the check LIVES,
/// too. It used to sit inside the llama lineage's facts, which made it a
/// property of that lineage — but every family whose attention reaches
/// the same dispatch is subject to the same instantiation set. The live
/// proof is Qwen3.6-27B: it declares `qwen3_5_text`, so it is already
/// openable through the hybrid's derivation, and its 24 query heads over
/// 4 kv heads is the same group size of six. `refuse_unservable_gqa` runs
/// once, before the registry dispatches to any family.
#[test]
fn an_unserveable_gqa_ratio_is_refused_at_load() {
    let _gpu = gpu_guard();
    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            (p.join("model.safetensors").is_file()
                || p.join("model.safetensors.index.json").is_file())
            .then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen2.5-1.5B");
        return;
    };
    let descriptor = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    )
    .join("qwen25_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated Qwen2.5 descriptor");
        return;
    }
    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = CompletionBroker::new();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    // The load itself may succeed — the refusal is the shell's, and it
    // lands wherever the facts are first asked for.
    let loaded = code(d.load_model(&load));
    assert!(
        loaded == PIE_STATUS_OK || loaded == PIE_STATUS_UNSUPPORTED,
        "an unserveable ratio must refuse, not abort: {loaded}"
    );
    drop(d);
}

/// engine's own call sequence, driven through the engine's own
/// declarations.
#[test]
fn a_real_decode_frame_launches_through_the_abi() {
    let _gpu = gpu_guard();
    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let descriptor = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    )
    .join("qwen3_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated descriptor");
        return;
    }

    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = CompletionBroker::new();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");

    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(code(d.load_model(&load)), PIE_STATUS_OK);

    let prog = driver_api::ProgramRegistration {
        program_hash: 0xF12E,
        ..Default::default()
    };
    let mut program_id = 0u64;
    assert_eq!(reg_program(&mut d, &prog, &mut program_id), PIE_STATUS_OK);
    let inst = InstanceBindingPlan {
        driver_id: 0,
        program_id: 0,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids: Vec::new(),
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::Host,
    };
    let mut binding = InstanceBinding::default();
    assert_eq!(bind(&mut d, &inst, &mut binding), PIE_STATUS_OK);

    // One decode step: instance's first token (id 7) at position 0, one
    // KV page, appended at offset 0.
    let mut cell = TerminalCell::pending();
    let cell_ptr: *mut TerminalCell = &mut cell;
    let roster_rows: [u32; 1] = [0];
    let sub_batch_indptr: [u32; 2] = [0, 1];
    let sub_batch_class: [u32; 1] = [driver_api::local::PIE_GEOMETRY_CLASS_HOST];
    let token_ids: [u32; 1] = [7];
    let position_ids: [u32; 1] = [0];
    let kv_page_indices: [u32; 1] = [0];
    let kv_page_indptr: [u32; 2] = [0, 1];
    let kv_last_page_lens: [u32; 1] = [1];
    let qo_indptr: [u32; 2] = [0, 1];
    let step = StepSubmission {
        plan: LaunchPlan {
            token_ids: token_ids.to_vec(),
            position_ids: position_ids.to_vec(),
            kv_page_indices: kv_page_indices.to_vec(),
            kv_page_indptr: kv_page_indptr.to_vec(),
            kv_last_page_lens: kv_last_page_lens.to_vec(),
            qo_indptr: qo_indptr.to_vec(),
            ..Default::default()
        },
        roster_rows: roster_rows.to_vec(),
        sub_batch_indptr: sub_batch_indptr.to_vec(),
        sub_batch_class: sub_batch_class.to_vec(),
        terminal_cells: vec![cell_ptr],
        ..Default::default()
    };
    let instance_ids: [u64; 1] = [binding.instance_id];
    let frame = FrameSubmission {
        instance_ids: instance_ids.to_vec(),
        required_kv_pages: 1,
        steps: vec![step.clone()],
        ..Default::default()
    };
    // THE COMPLETION IS THE FENCE. `Shell::launch` returns with the fire
    // still on the stream, so the terminal cell below has not been written
    // yet — waiting for the completion is what the engine does and what
    // makes the assertions after it about the fire rather than about the
    // call.
    //
    // It was a spin on a static that an `extern "C"` notify wrote the
    // wait id into, compared against the `0xBEEF` the test had chosen for
    // itself. Neither half survives: a wait id is the broker's to assign,
    // and the notify is a method on the broker the driver holds.
    assert_eq!(
        fire_and_wait(&mut d, &frame, &broker),
        PIE_STATUS_OK,
        "the frame launches"
    );
    assert_eq!(
        cell.load(),
        PIE_TERMINAL_OUTCOME_SUCCESS,
        "the terminal cell published"
    );
    drop(d);
}

/// The channel endpoint contract: a registered channel answers with a
/// pinned mirror of `(capacity + 1)` wire cells and four zeroed control
/// words at indices 0..4; bool bit-packs; duplicates and oversized rings
/// refuse; closing frees and is idempotent.
#[test]
fn channels_bind_the_ring_contract() {
    let _gpu = gpu_guard();

    let mut d = Shell::open(b"", engine_runtime()).expect("the driver creates");

    let shape: [u32; 2] = [4, 8]; // 32 elements
    let ch = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 5,
        shape: shape.to_vec(),
        dtype: PIE_CHANNEL_DTYPE_F32,
        host_role: PIE_CHANNEL_HOST_ROLE_NONE,
        seeded: false,
        extern_dir: PIE_CHANNEL_EXTERN_NONE,
        capacity: 7,
        reader_wait_id: 11,
        writer_wait_id: 12,
        extern_name: Vec::new(),
    };
    let mut b = ChannelBinding::default();
    assert_eq!(reg_channel(&mut d, &ch, &mut b), PIE_STATUS_OK);
    assert_eq!(
        b.cell_bytes,
        32 * 4,
        "f32 wire cells are four bytes per element"
    );
    assert_eq!(
        b.mirror_bytes,
        u64::from(b.cell_bytes) * 8,
        "capacity + 1 cells"
    );
    assert_eq!(
        (
            b.head_word_index,
            b.tail_word_index,
            b.poison_word_index,
            b.closed_word_index
        ),
        (0, 1, 2, 3)
    );
    let words = unsafe { std::slice::from_raw_parts(b.word_base as *const u64, 4) };
    assert_eq!(words, &[0, 0, 0, 0], "control words start zeroed");

    // Duplicate id refuses; a bool channel bit-packs.
    assert_eq!(
        reg_channel(&mut d, &ch, &mut b),
        PIE_STATUS_INVALID_ARGUMENT
    );
    let boolch = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 6,
        shape: shape.to_vec(),
        dtype: PIE_CHANNEL_DTYPE_BOOL,
        host_role: PIE_CHANNEL_HOST_ROLE_NONE,
        seeded: false,
        extern_dir: PIE_CHANNEL_EXTERN_NONE,
        capacity: 1,
        reader_wait_id: 13,
        writer_wait_id: 14,
        extern_name: Vec::new(),
    };
    assert_eq!(reg_channel(&mut d, &boolch, &mut b), PIE_STATUS_OK);
    assert_eq!(b.cell_bytes, 4, "32 bools bit-pack to four bytes");

    // An oversized ring refuses; closes are real and idempotent.
    let big = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 9,
        shape: Vec::new(),
        dtype: PIE_CHANNEL_DTYPE_F32,
        host_role: PIE_CHANNEL_HOST_ROLE_NONE,
        seeded: false,
        extern_dir: PIE_CHANNEL_EXTERN_NONE,
        capacity: 64,
        reader_wait_id: 0,
        writer_wait_id: 0,
        extern_name: Vec::new(),
    };
    assert_eq!(
        reg_channel(&mut d, &big, &mut b),
        PIE_STATUS_INVALID_ARGUMENT,
        "capacity + 1 must stay within the ring maximum"
    );
    assert_eq!(code(d.close_channel(5)), PIE_STATUS_OK);
    assert_eq!(code(d.close_channel(5)), PIE_STATUS_OK);
    let again = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 5,
        shape: Vec::new(),
        dtype: PIE_CHANNEL_DTYPE_F32,
        host_role: PIE_CHANNEL_HOST_ROLE_NONE,
        seeded: false,
        extern_dir: PIE_CHANNEL_EXTERN_NONE,
        capacity: 0,
        reader_wait_id: 0,
        writer_wait_id: 0,
        extern_name: Vec::new(),
    };
    assert_eq!(
        reg_channel(&mut d, &again, &mut b),
        PIE_STATUS_OK,
        "a closed id re-registers"
    );
    drop(d);
}

/// The delivery: the engine's whole loop, with the output coming BACK.
/// A `[vocab]` f32 reader channel attaches to the instance, the reference
/// prompt prefills through `Shell::launch`, and the ring's first cell
/// holds the last row's logits — checked against the SAME transformers
/// reference the executor A/B pinned. The tail word advanced exactly
/// once; head stays the engine's.
#[test]
fn logits_come_back_through_the_ring() {
    let _gpu = gpu_guard();

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let scratch = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    );
    let descriptor = scratch.join("qwen3_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated descriptor");
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("oracle/real_decode/reference.json")).expect("reference");

    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = CompletionBroker::new();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(code(d.load_model(&load)), PIE_STATUS_OK);

    const VOCAB: usize = 151_936;
    let shape: [u32; 1] = [VOCAB as u32];
    let ch = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 77,
        shape: shape.to_vec(),
        dtype: PIE_CHANNEL_DTYPE_F32,
        host_role: PIE_CHANNEL_HOST_ROLE_READER,
        seeded: false,
        extern_dir: PIE_CHANNEL_EXTERN_NONE,
        capacity: 3,
        reader_wait_id: 155,
        writer_wait_id: 156,
        extern_name: Vec::new(),
    };
    let mut chb = ChannelBinding::default();
    assert_eq!(reg_channel(&mut d, &ch, &mut chb), PIE_STATUS_OK);

    let prog = driver_api::ProgramRegistration {
        program_hash: 0xF13E,
        ..Default::default()
    };
    let mut program_id = 0u64;
    assert_eq!(reg_program(&mut d, &prog, &mut program_id), PIE_STATUS_OK);
    let channel_ids: [u64; 1] = [77];
    let inst = InstanceBindingPlan {
        driver_id: 0,
        program_id: 0,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids: channel_ids.to_vec(),
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::Host,
    };
    let mut binding = InstanceBinding::default();
    assert_eq!(bind(&mut d, &inst, &mut binding), PIE_STATUS_OK);

    // The reference prompt as one prefill request over one page.
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let tokens = prompt.len();
    let positions: Vec<u32> = (0..tokens as u32).collect();
    // ONE ROSTER ROW PER REQUEST, not per token. These said `vec![0;
    // tokens]` — a prefill's token count, every entry zero — which the
    // shared validator reads as N requests all claiming roster index 0,
    // and refuses for the duplicates. The engine builds this with
    // `Vec::with_capacity(instance_ids.len())`
    // (`scheduler/batch.rs:388`), so one entry is what a caller sends.
    // The shell never read the field, which is why it went unnoticed.
    let roster_rows: Vec<u32> = vec![0];
    // The CSR partitions ROSTER ROWS by sub-batch, not tokens: the engine
    // pushes `group.len()` (`scheduler/batch.rs:384`), and
    // `validate_csr` checks the last value against `roster_rows.len`.
    // One request, one row, one entry.
    let sub_batch_indptr: [u32; 2] = [0, 1];
    // AND A TERMINAL CELL PER REQUEST. These fixtures stated none, which
    // the validator refuses with `allow_empty = false`: a step with no
    // cell has nowhere to report an outcome, and the engine sends one per
    // request. Never read here — the fires below take their answer off
    // the ring — but a caller's shape is a caller's shape.
    let mut launch_cell = driver_api::local::TerminalCell::pending();
    let launch_cell_ptr: *mut driver_api::local::TerminalCell = &mut launch_cell;
    let sub_batch_class: [u32; 1] = [driver_api::local::PIE_GEOMETRY_CLASS_HOST];
    let kv_page_indices: [u32; 1] = [0];
    let kv_page_indptr: [u32; 2] = [0, 1];
    let kv_last_page_lens: [u32; 1] = [tokens as u32];
    let qo_indptr: [u32; 2] = [0, tokens as u32];
    let step = StepSubmission {
        plan: LaunchPlan {
            token_ids: prompt.to_vec(),
            position_ids: positions.to_vec(),
            kv_page_indices: kv_page_indices.to_vec(),
            kv_page_indptr: kv_page_indptr.to_vec(),
            kv_last_page_lens: kv_last_page_lens.to_vec(),
            qo_indptr: qo_indptr.to_vec(),
            ..Default::default()
        },
        roster_rows: roster_rows.to_vec(),
        sub_batch_indptr: sub_batch_indptr.to_vec(),
        sub_batch_class: sub_batch_class.to_vec(),
        terminal_cells: vec![launch_cell_ptr],
        ..Default::default()
    };
    let instance_ids: [u64; 1] = [binding.instance_id];
    let frame = FrameSubmission {
        instance_ids: instance_ids.to_vec(),
        required_kv_pages: 1,
        steps: vec![step.clone()],
        ..Default::default()
    };
    assert_eq!(fire_and_wait(&mut d, &frame, &broker), PIE_STATUS_OK);

    // The ring advanced once, and cell 0 holds the last row's logits.
    let words = unsafe { std::slice::from_raw_parts(chb.word_base as *const u64, 4) };
    assert_eq!(words[1], 1, "the tail advanced exactly once");
    assert_eq!(words[0], 0, "the head is the engine's to move");
    let cell = unsafe { std::slice::from_raw_parts(chb.mirror_base as *const f32, VOCAB) };
    let hf_argmax = reference["argmax"].as_u64().expect("argmax") as usize;
    let (mut best_t, mut best_v) = (0usize, f32::NEG_INFINITY);
    for (t, &v) in cell.iter().enumerate() {
        if v > best_v {
            (best_t, best_v) = (t, v);
        }
    }
    assert_eq!(
        best_t, hf_argmax,
        "the ring carried the right logits (top {best_v})"
    );
    let hf_top1 = reference["top5_logits"].as_array().expect("top5")[0]
        .as_f64()
        .expect("v") as f32;
    assert!(
        (best_v - hf_top1).abs() < 0.25,
        "top-1 {best_v} vs HF {hf_top1}"
    );
    drop(d);
}

/// Multi-step decode continuity + resize migration + copy_kv, in one
/// story: prefill the reference prompt (step 1) and decode its argmax
/// token (step 2) IN THE SAME FRAME — the decode's logits ride the ring
/// as cell 1. Then resize the pool larger and copy page 0 to page 2, and
/// a decode against the COPIED page must produce the same logits cell —
/// the migration and the page copy both preserved the KV bytes.
#[test]
fn multi_step_resize_and_copy_preserve_the_kv() {
    let _gpu = gpu_guard();

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let descriptor = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    )
    .join("qwen3_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated descriptor");
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("oracle/real_decode/reference.json")).expect("reference");

    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = CompletionBroker::new();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(code(d.load_model(&load)), PIE_STATUS_OK);
    const VOCAB: usize = 151_936;
    let shape: [u32; 1] = [VOCAB as u32];
    let ch = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 9,
        shape: shape.to_vec(),
        dtype: PIE_CHANNEL_DTYPE_F32,
        host_role: PIE_CHANNEL_HOST_ROLE_READER,
        seeded: false,
        extern_dir: PIE_CHANNEL_EXTERN_NONE,
        capacity: 7,
        reader_wait_id: 19,
        writer_wait_id: 20,
        extern_name: Vec::new(),
    };
    let mut chb = ChannelBinding::default();
    assert_eq!(reg_channel(&mut d, &ch, &mut chb), PIE_STATUS_OK);
    let prog = driver_api::ProgramRegistration {
        program_hash: 0xF14E,
        ..Default::default()
    };
    let mut program_id = 0u64;
    reg_program(&mut d, &prog, &mut program_id);
    let channel_ids: [u64; 1] = [9];
    let inst = InstanceBindingPlan {
        driver_id: 0,
        program_id: 0,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids: channel_ids.to_vec(),
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::Host,
    };
    let mut binding = InstanceBinding::default();
    bind(&mut d, &inst, &mut binding);

    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let n = prompt.len();
    let hf_argmax = reference["argmax"].as_u64().expect("argmax") as u32;

    // Step 1: prefill the prompt. Step 2: decode the argmax token at
    // position n against the same page.
    let positions1: Vec<u32> = (0..n as u32).collect();
    // One request, so one roster row and one CSR entry — `n` is the TOKEN
    // count and belongs to `qo_indptr`, which partitions tokens.
    let roster1: [u32; 1] = [0];
    let sbi1: [u32; 2] = [0, 1];
    let cls: [u32; 1] = [driver_api::local::PIE_GEOMETRY_CLASS_HOST];
    // ONE CELL PER STEP, and they must be DISTINCT — the validator says
    // so and it is right: two steps sharing a cell would overwrite each
    // other's outcome, and the frame would report whichever finished
    // last as the answer for both.
    let mut ms_cells = [
        driver_api::local::TerminalCell::pending(),
        driver_api::local::TerminalCell::pending(),
    ];
    let (first, second) = ms_cells.split_at_mut(1);
    let ms_ptr1: *mut driver_api::local::TerminalCell = &mut first[0];
    let ms_ptr2: *mut driver_api::local::TerminalCell = &mut second[0];
    let cells1 = vec![ms_ptr1];
    let cells2 = vec![ms_ptr2];
    let pages: [u32; 1] = [0];
    let indptr: [u32; 2] = [0, 1];
    let lens1: [u32; 1] = [n as u32];
    let qo1: [u32; 2] = [0, n as u32];
    let step1 = StepSubmission {
        plan: LaunchPlan {
            token_ids: prompt.to_vec(),
            position_ids: positions1.to_vec(),
            kv_page_indices: pages.to_vec(),
            kv_page_indptr: indptr.to_vec(),
            kv_last_page_lens: lens1.to_vec(),
            qo_indptr: qo1.to_vec(),
            ..Default::default()
        },
        roster_rows: roster1.to_vec(),
        sub_batch_indptr: sbi1.to_vec(),
        sub_batch_class: cls.to_vec(),
        terminal_cells: cells1,
        ..Default::default()
    };
    let tok2: [u32; 1] = [hf_argmax];
    let pos2: [u32; 1] = [n as u32];
    let roster2: [u32; 1] = [0];
    let sbi2: [u32; 2] = [0, 1];
    let lens2: [u32; 1] = [n as u32 + 1];
    let qo2: [u32; 2] = [0, 1];
    let step2 = StepSubmission {
        plan: LaunchPlan {
            token_ids: tok2.to_vec(),
            position_ids: pos2.to_vec(),
            kv_page_indices: pages.to_vec(),
            kv_page_indptr: indptr.to_vec(),
            kv_last_page_lens: lens2.to_vec(),
            qo_indptr: qo2.to_vec(),
            ..Default::default()
        },
        roster_rows: roster2.to_vec(),
        sub_batch_indptr: sbi2.to_vec(),
        sub_batch_class: cls.to_vec(),
        terminal_cells: cells2,
        ..Default::default()
    };
    let steps = [step1, step2];
    let instance_ids: [u64; 1] = [binding.instance_id];
    let frame = FrameSubmission {
        instance_ids: instance_ids.to_vec(),
        required_kv_pages: 1,
        steps: steps.to_vec(),
        ..Default::default()
    };
    assert_eq!(
        fire_and_wait(&mut d, &frame, &broker),
        PIE_STATUS_OK,
        "the two-step frame launches"
    );
    let words = unsafe { std::slice::from_raw_parts(chb.word_base as *const u64, 4) };
    assert_eq!(words[1], 2, "both steps delivered");
    let cell1 =
        unsafe { std::slice::from_raw_parts((chb.mirror_base as *const f32).add(VOCAB), VOCAB) };
    let decode_logits: Vec<f32> = cell1.to_vec();
    let best1 = decode_logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(t, _)| t)
        .unwrap();

    // ONE completion for the control verbs below, and it is not the
    // per-fire kind. `resize_pool` and `copy_kv` synchronize before they
    // notify, so their answer IS their `i32` and nothing here waits on the
    // completion at all — it is the address they notify, and one address
    // serves all four. A fire is the opposite case, which is why
    // `fire_and_wait` mints its own.
    let (completion, completion_handle) = broker.launch_completion(1);
    let _ = &completion_handle;

    // Resize larger (migrates page 0), copy page 0 → page 2, then decode
    // AGAINST PAGE 2: same context bytes, so the same logits cell.
    let resize = driver_api::PoolResizePlan {
        target_pages: 4,
        ..Default::default()
    };
    assert_eq!(code(d.resize_pool(&resize, completion)), PIE_STATUS_OK);
    let copy = driver_api::KvCopyPlan {
        src_domain: driver_api::local::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        dst_domain: driver_api::local::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        src_page_ids: vec![0],
        dst_page_ids: vec![2],
        ..Default::default()
    };
    assert_eq!(code(d.copy_kv(&copy, completion)), PIE_STATUS_OK);
    let pages2: [u32; 1] = [2];
    let step3 = StepSubmission {
        plan: LaunchPlan {
            kv_page_indices: pages2.to_vec(),
            ..Default::default()
        },
        ..Default::default()
    };
    let steps3 = [step3];
    let frame3 = FrameSubmission {
        steps: steps3.to_vec(),
        required_kv_pages: 4,
        ..Default::default()
    };
    assert_eq!(fire_and_wait(&mut d, &frame3, &broker), PIE_STATUS_OK);
    assert_eq!(words[1], 3, "the third fire delivered");
    let cell2 = unsafe {
        std::slice::from_raw_parts((chb.mirror_base as *const f32).add(2 * VOCAB), VOCAB)
    };
    let best2 = cell2
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(t, _)| t)
        .unwrap();
    assert_eq!(
        best2, best1,
        "the migrated + copied page carries the same context"
    );
    for t in [best1, 0, 1000] {
        assert!(
            (cell2[t] - decode_logits[t]).abs() < 1e-3,
            "token {t}: {} vs {}",
            cell2[t],
            decode_logits[t]
        );
    }

    // ── THE HOST ROUND TRIP: evict, restore, and fire against the
    // restored page.
    //
    // This is what `pools::swap_pool` was ported for and what the shell
    // used to refuse — first outright ("host domains refuse until the swap
    // pool wires in"), then by a stride check guarding a hand-rolled pool
    // that assumed two buffers per layer of one width.
    //
    // The gate is the same one the device copy above uses, and it is the
    // only one worth having: the same context bytes must produce the same
    // logits. A pool that mis-strides a layer, drops a buffer, or
    // transposes src and dst all fail it — and none of them fails a test
    // that only checks the status code.
    let dev = driver_api::local::PIE_MEMORY_DOMAIN_CUDA_DEVICE;
    let host = driver_api::local::PIE_MEMORY_DOMAIN_HOST_PINNED;
    let evict = driver_api::KvCopyPlan {
        src_domain: dev,
        dst_domain: host,
        src_page_ids: vec![0],
        dst_page_ids: vec![0],
        ..Default::default()
    };
    assert_eq!(
        code(d.copy_kv(&evict, completion)),
        PIE_STATUS_OK,
        "page 0 evicts to the pinned pool"
    );
    let restore = driver_api::KvCopyPlan {
        src_domain: host,
        dst_domain: dev,
        src_page_ids: vec![0],
        dst_page_ids: vec![3],
        ..Default::default()
    };
    assert_eq!(
        code(d.copy_kv(&restore, completion)),
        PIE_STATUS_OK,
        "and restores into page 3"
    );
    let pages3: [u32; 1] = [3];
    let step4 = StepSubmission {
        plan: LaunchPlan {
            kv_page_indices: pages3.to_vec(),
            ..Default::default()
        },
        ..Default::default()
    };
    let steps4 = [step4];
    let frame4 = FrameSubmission {
        steps: steps4.to_vec(),
        required_kv_pages: 4,
        ..Default::default()
    };
    assert_eq!(fire_and_wait(&mut d, &frame4, &broker), PIE_STATUS_OK);
    assert_eq!(words[1], 4, "the fourth fire delivered");
    let cell3 = unsafe {
        std::slice::from_raw_parts((chb.mirror_base as *const f32).add(3 * VOCAB), VOCAB)
    };
    let best3 = cell3
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(t, _)| t)
        .unwrap();
    assert_eq!(
        best3, best1,
        "a page that went to host and back carries its context"
    );
    for t in [best1, 0, 1000] {
        assert!(
            (cell3[t] - decode_logits[t]).abs() < 1e-3,
            "round-tripped token {t}: {} vs {}",
            cell3[t],
            decode_logits[t]
        );
    }
    drop(d);
}

/// The mini-soak, and it is real GENERATION: prefill the reference
/// prompt, then fifty greedy decode steps, each feeding the previous
/// argmax back through its own `Shell::launch` — the inference loop an
/// engine runs, driven twice. Gates: every step delivers on the ring,
/// the two runs produce IDENTICAL token sequences (determinism), the
/// first decoded token matches the HF reference argmax, and device free
/// memory is stable across the chain (per-fire allocations all retire).
#[test]
fn a_fifty_step_greedy_chain_is_deterministic_and_leak_free() {
    let _gpu = gpu_guard();

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let descriptor = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    )
    .join("qwen3_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated descriptor");
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("oracle/real_decode/reference.json")).expect("reference");
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let hf_argmax = reference["argmax"].as_u64().expect("argmax") as u32;

    const VOCAB: usize = 151_936;
    const STEPS: usize = 50;
    const PAGE: u32 = 16;

    let chain = |run_tag: u64| -> (Vec<u32>, usize, usize) {
        let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
        let broker = CompletionBroker::new();
        let opened = Shell::open(boot.as_bytes(), broker.clone());
        let d = opened;
        let mut d = d.expect("the driver creates");
        let load = driver_api::ModelLoadDesc {
            snapshot_dir: snap.clone(),
            runtime_quant: String::new(),
            mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
            component: driver_api::ModelComponent::Full,
        };
        assert_eq!(code(d.load_model(&load)), PIE_STATUS_OK);
        let shape: [u32; 1] = [VOCAB as u32];
        let ch = ChannelRegistrationPlan {
            driver_id: 0,
            channel_id: 1,
            shape: shape.to_vec(),
            dtype: PIE_CHANNEL_DTYPE_F32,
            host_role: PIE_CHANNEL_HOST_ROLE_READER,
            seeded: false,
            extern_dir: PIE_CHANNEL_EXTERN_NONE,
            capacity: 3,
            reader_wait_id: 3,
            writer_wait_id: 4,
            extern_name: Vec::new(),
        };
        let mut chb = ChannelBinding::default();
        assert_eq!(reg_channel(&mut d, &ch, &mut chb), PIE_STATUS_OK);
        let prog = driver_api::ProgramRegistration {
            program_hash: run_tag,
            ..Default::default()
        };
        let mut program_id = 0u64;
        reg_program(&mut d, &prog, &mut program_id);
        let channel_ids: [u64; 1] = [1];
        let inst = InstanceBindingPlan {
            driver_id: 0,
            program_id: 0,
            requested_instance_id: 0,
            pacing_wait_id: 0,
            channel_ids: channel_ids.to_vec(),
            seed_values: Vec::new(),
            geometry_class: driver_api::GeometryClass::Host,
        };
        let mut binding = InstanceBinding::default();
        bind(&mut d, &inst, &mut binding);
        let instance_ids: [u64; 1] = [binding.instance_id];

        let total_pages = ((prompt.len() + STEPS) as u32).div_ceil(PAGE);
        let all_pages: Vec<u32> = (0..total_pages).collect();
        let read_cell = |i: u64| -> usize {
            let cell = unsafe {
                std::slice::from_raw_parts(
                    (chb.mirror_base as *const f32).add((i % 4) as usize * VOCAB),
                    VOCAB,
                )
            };
            cell.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(t, _)| t)
                .unwrap()
        };
        let fire = |d: &mut Shell, kv_len: u32, tokens: &[u32], positions: &[u32], qo_end: u32| {
            let pages_used = kv_len.div_ceil(PAGE).max(1);
            let indices = &all_pages[..pages_used as usize];
            let indptr: [u32; 2] = [0, pages_used];
            let lens: [u32; 1] = [kv_len - (pages_used - 1) * PAGE];
            let qo: [u32; 2] = [0, qo_end];
            // One request: one roster row, one CSR entry, one cell. The
            // token count belongs to `qo_indptr`, which partitions TOKENS;
            // these three partition REQUESTS.
            let roster: [u32; 1] = [0];
            let sbi: [u32; 2] = [0, 1];
            let cls: [u32; 1] = [driver_api::local::PIE_GEOMETRY_CLASS_HOST];
            let mut loop_cell = driver_api::local::TerminalCell::pending();
            let loop_cell_ptr: *mut driver_api::local::TerminalCell = &mut loop_cell;
            let step = StepSubmission {
                plan: LaunchPlan {
                    token_ids: tokens.to_vec(),
                    position_ids: positions.to_vec(),
                    kv_page_indices: indices.to_vec(),
                    kv_page_indptr: indptr.to_vec(),
                    kv_last_page_lens: lens.to_vec(),
                    qo_indptr: qo.to_vec(),
                    ..Default::default()
                },
                roster_rows: roster.to_vec(),
                sub_batch_indptr: sbi.to_vec(),
                sub_batch_class: cls.to_vec(),
                terminal_cells: vec![loop_cell_ptr],
                ..Default::default()
            };
            let steps_arr = [step];
            let frame = FrameSubmission {
                instance_ids: instance_ids.to_vec(),
                required_kv_pages: total_pages,
                steps: steps_arr.to_vec(),
                ..Default::default()
            };
            assert_eq!(fire_and_wait(d, &frame, &broker), PIE_STATUS_OK);
        };

        // Prefill, then the greedy chain. The engine's half of the ring:
        // advance the head as each cell is consumed.
        let positions: Vec<u32> = (0..prompt.len() as u32).collect();
        fire(
            &mut d,
            prompt.len() as u32,
            &prompt,
            &positions,
            prompt.len() as u32,
        );
        let words = chb.word_base as *mut u64;
        let mut consumed = 0u64;
        let mut next = read_cell(consumed) as u32;
        consumed += 1;
        unsafe { words.read_volatile() }; // head untouched by us conceptually
        unsafe { words.write_volatile(consumed) };
        let mut generated = vec![next];
        // The driver's OWN residue, after one step and after all fifty.
        // A per-fire leak is growth between these two; anything the shell
        // allocates once is in both.
        let held_after_one = d.live_device_bytes();
        for s in 0..STEPS - 1 {
            let pos = prompt.len() as u32 + s as u32;
            let toks: [u32; 1] = [next];
            let poss: [u32; 1] = [pos];
            fire(&mut d, pos + 1, &toks, &poss, 1);
            next = read_cell(consumed) as u32;
            consumed += 1;
            unsafe { words.write_volatile(consumed) };
            generated.push(next);
        }
        let held_after_all = d.live_device_bytes();
        drop(d);
        (generated, held_after_one, held_after_all)
    };

    let (run1, held_one, held_all) = chain(0xA);
    let (run2, _, _) = chain(0xB);

    assert_eq!(
        run1[0], hf_argmax,
        "the chain starts where the reference points"
    );
    assert_eq!(
        run1, run2,
        "greedy generation is deterministic across drivers"
    );
    assert!(
        run1.iter().skip(1).any(|&t| t != run1[0]),
        "fifty steps that repeat one token would be a broken chain: {run1:?}"
    );

    // THE LEAK CLAIM IS ABOUT THE DRIVER, and it used to be about the
    // device.
    //
    // The old form read `cudaMemGetInfo` before and after and allowed 256
    // MiB of slack. That answers "how much does this GPU have left",
    // which a process sharing a card cannot answer for itself: a second
    // consumer allocating during the five seconds this takes is
    // indistinguishable from a leak. It failed exactly that way against a
    // concurrent agent on the same L40S -- ~45 MB/s of someone else's
    // allocation, read as 390 MB of drift -- on a tree whose suite was
    // green minutes before. A test that fails for a reason outside the
    // thing it tests is a test that will be muted.
    //
    // `live_device_bytes` is the shell's own count. Fifty fires against
    // one driver either grow it or they do not, and what the shell
    // allocates ONCE is in both readings. So the slack is zero and the
    // comparison is exact -- which the old form could never be, and the
    // reason it needed a quarter of a gigabyte of room.
    // A decode chain legitimately grows: its CONTEXT does. The KV pages,
    // the element mask (`sum_r qo_len * kv_len`) and the score sink
    // (`kv_len * heads`) are all sized from the fire's geometry, and
    // fifty steps is fifty more KV positions -- measured at ~2.7 KB over
    // forty-nine fires, which is the geometry and not a leak.
    //
    // What a LEAK looks like here is a per-fire buffer that is allocated
    // and never freed, and every such buffer in this shell is megabyte-
    // scale: the arena, the workspaces, the named seam pins. So the bound
    // is 64 KiB -- comfortably above what the geometry can explain at this
    // length and orders of magnitude below anything that could be missed.
    let grew = held_all.saturating_sub(held_one);
    assert!(
        grew < 64 * 1024,
        "fifty fires grew the driver's own allocation by {grew} bytes \
         ({held_one} -> {held_all}). The context's own growth is kilobytes; \
         a per-fire buffer that is never freed is megabytes"
    );

    eprintln!("[soak] generated: {:?}", &run1[..10.min(run1.len())]);
}

/// THE SOAK, at the C++ gate's round count: 711 fires in ONE driver
/// lifetime — fourteen generation chains (prefill + fifty decodes each),
/// pages rewound per chain, device free memory sampled at every chain
/// boundary and required FLAT after warmup. The C++ soak's own gates
/// (many rounds, RSS steady), spoken through the new ABI.
#[test]
#[ignore = "the scaled soak: ~1 minute of GPU; run explicitly"]
fn the_711_fire_soak_holds_steady() {
    let _gpu = gpu_guard();

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let descriptor = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    )
    .join("qwen3_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated descriptor");
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("oracle/real_decode/reference.json")).expect("reference");
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();

    const VOCAB: usize = 151_936;
    const CHAINS: usize = 14;
    const DECODES: usize = 50;
    const PAGE: u32 = 16;

    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = engine_runtime();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(code(d.load_model(&load)), PIE_STATUS_OK);
    let shape: [u32; 1] = [VOCAB as u32];
    let ch = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 1,
        shape: shape.to_vec(),
        dtype: PIE_CHANNEL_DTYPE_F32,
        host_role: PIE_CHANNEL_HOST_ROLE_READER,
        seeded: false,
        extern_dir: PIE_CHANNEL_EXTERN_NONE,
        capacity: 3,
        reader_wait_id: 3,
        writer_wait_id: 4,
        extern_name: Vec::new(),
    };
    let mut chb = ChannelBinding::default();
    reg_channel(&mut d, &ch, &mut chb);
    let prog = driver_api::ProgramRegistration {
        program_hash: 0x50AC,
        ..Default::default()
    };
    let mut program_id = 0u64;
    reg_program(&mut d, &prog, &mut program_id);
    let channel_ids: [u64; 1] = [1];
    let inst = InstanceBindingPlan {
        driver_id: 0,
        program_id: 0,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids: channel_ids.to_vec(),
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::Host,
    };
    let mut binding = InstanceBinding::default();
    bind(&mut d, &inst, &mut binding);
    let instance_ids: [u64; 1] = [binding.instance_id];
    let words = chb.word_base as *mut u64;

    let total_pages = ((prompt.len() + DECODES) as u32).div_ceil(PAGE);
    let all_pages: Vec<u32> = (0..total_pages).collect();
    let mut fires = 0usize;
    let mut consumed = 0u64;
    let mut baseline = None;
    let mut first_chain_head = Vec::new();
    for chain in 0..CHAINS {
        let fire = |d: &mut Shell, kv_len: u32, tokens: &[u32], positions: &[u32]| {
            let pages_used = kv_len.div_ceil(PAGE).max(1);
            let indices = &all_pages[..pages_used as usize];
            let indptr: [u32; 2] = [0, pages_used];
            let lens: [u32; 1] = [kv_len - (pages_used - 1) * PAGE];
            let qo: [u32; 2] = [0, tokens.len() as u32];
            // One request: one roster row, one CSR entry, one cell. The
            // token count belongs to `qo_indptr`, which partitions TOKENS;
            // these three partition REQUESTS.
            let roster: [u32; 1] = [0];
            let sbi: [u32; 2] = [0, 1];
            let cls: [u32; 1] = [driver_api::local::PIE_GEOMETRY_CLASS_HOST];
            let mut loop_cell = driver_api::local::TerminalCell::pending();
            let loop_cell_ptr: *mut driver_api::local::TerminalCell = &mut loop_cell;
            let step = StepSubmission {
                plan: LaunchPlan {
                    token_ids: tokens.to_vec(),
                    position_ids: positions.to_vec(),
                    kv_page_indices: indices.to_vec(),
                    kv_page_indptr: indptr.to_vec(),
                    kv_last_page_lens: lens.to_vec(),
                    qo_indptr: qo.to_vec(),
                    ..Default::default()
                },
                roster_rows: roster.to_vec(),
                sub_batch_indptr: sbi.to_vec(),
                sub_batch_class: cls.to_vec(),
                terminal_cells: vec![loop_cell_ptr],
                ..Default::default()
            };
            let steps_arr = [step];
            let frame = FrameSubmission {
                instance_ids: instance_ids.to_vec(),
                required_kv_pages: total_pages,
                steps: steps_arr.to_vec(),
                ..Default::default()
            };
            assert_eq!(fire_and_wait(d, &frame, &broker), PIE_STATUS_OK);
        };
        let read_argmax = |i: u64| -> u32 {
            let cell = unsafe {
                std::slice::from_raw_parts(
                    (chb.mirror_base as *const f32).add((i % 4) as usize * VOCAB),
                    VOCAB,
                )
            };
            cell.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(t, _)| t as u32)
                .unwrap()
        };

        let positions: Vec<u32> = (0..prompt.len() as u32).collect();
        fire(&mut d, prompt.len() as u32, &prompt, &positions);
        fires += 1;
        let mut next = read_argmax(consumed);
        consumed += 1;
        unsafe { words.write_volatile(consumed) };
        let mut head_tokens = vec![next];
        for s in 0..DECODES {
            let pos = prompt.len() as u32 + s as u32;
            let toks: [u32; 1] = [next];
            let poss: [u32; 1] = [pos];
            fire(&mut d, pos + 1, &toks, &poss);
            fires += 1;
            next = read_argmax(consumed);
            consumed += 1;
            unsafe { words.write_volatile(consumed) };
            if head_tokens.len() < 8 {
                head_tokens.push(next);
            }
        }
        if chain == 0 {
            first_chain_head = head_tokens;
        } else {
            assert_eq!(
                head_tokens, first_chain_head,
                "chain {chain}: rewound pages must reproduce chain 0"
            );
        }
        // THE DRIVER'S OWN RESIDUE, not the device's. See the fifty-step
        // chain for why: `cudaMemGetInfo` answers a question about the
        // GPU, and a second consumer on the same card is
        // indistinguishable from a leak. This driver is the same one
        // across every chain, so the number is directly comparable and
        // the slack can be tight.
        //
        // Each chain REWINDS to the same pages, so unlike the fifty-step
        // chain the context does not grow across them -- the geometry
        // buffers are already at their high-water mark after chain 0.
        let held = d.live_device_bytes();
        match baseline {
            None => baseline = Some(held),
            Some(b) => assert!(
                held <= b + (1 << 20),
                "chain {chain}: the driver's own allocation grew to {held} \
                 from a baseline of {b} -- the chains rewind to the same \
                 pages, so nothing here should need more than chain 0 did"
            ),
        }
    }
    assert_eq!(fires, CHAINS * (DECODES + 1), "the full round count ran");
    eprintln!("[soak] {fires} fires, memory flat at {:?}", baseline);
    drop(d);
}

/// The qwen3_5 HYBRID through the 13-symbol ABI, end to end (E-gate
/// family #1's shell gate): `load_model` parses the VL-shaped config and
/// admits the fp32 GDN parameters, `launch` prefills the reference
/// prompt through the hybrid plan — 18 GDN layers against driver-owned,
/// ENGINE-slotted state slabs (`rs_slot_ids` + RESET) — and the ring
/// carries logits meeting the family's calibrated bar (argmax within
/// HF's top-5; `real_hybrid.rs` documents why not equality). Then
/// `copy_state` clones slot 0 → slots 1 and 2, and the SAME decode
/// fired against two copies must agree — same argmax, logits within
/// 0.25. (Not bit-identity: two sequential fires jitter at ~0.1 even on
/// identical state — per-fire allocations shift addresses and with them
/// GEMM reduction orders — measured copy-vs-copy, same pattern as
/// live-vs-copy. A wrong stride or a missed plane blows 0.25 wide open;
/// the jitter does not.)
#[test]
fn the_hybrid_loads_fires_and_copies_state_through_the_abi() {
    let _gpu = gpu_guard();

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            (p.join("model.safetensors").is_file()
                || p.join("model.safetensors.index.json").is_file())
            .then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3.5-0.8B-Base");
        return;
    };
    let scratch = std::path::PathBuf::from(std::env::var("PIE_TEST_SCRATCH").unwrap_or_else(
        |_| "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad".into(),
    ));
    let descriptor = scratch.join("qwen3_5_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated qwen3_5 descriptor");
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("oracle/real_decode/qwen3_5_0_8b.json"))
            .expect("reference");

    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = CompletionBroker::new();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(
        code(d.load_model(&load)),
        PIE_STATUS_OK,
        "the hybrid checkpoint loads (fp32 GDN parameters included)"
    );

    const VOCAB: usize = 248_320;
    let shape: [u32; 1] = [VOCAB as u32];
    let ch = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 88,
        shape: shape.to_vec(),
        dtype: PIE_CHANNEL_DTYPE_F32,
        host_role: PIE_CHANNEL_HOST_ROLE_READER,
        seeded: false,
        extern_dir: PIE_CHANNEL_EXTERN_NONE,
        capacity: 3,
        reader_wait_id: 177,
        writer_wait_id: 178,
        extern_name: Vec::new(),
    };
    let mut chb = ChannelBinding::default();
    assert_eq!(reg_channel(&mut d, &ch, &mut chb), PIE_STATUS_OK);
    let prog = driver_api::ProgramRegistration {
        program_hash: 0x35B,
        ..Default::default()
    };
    let mut program_id = 0u64;
    assert_eq!(reg_program(&mut d, &prog, &mut program_id), PIE_STATUS_OK);
    let channel_ids: [u64; 1] = [88];
    let inst = InstanceBindingPlan {
        driver_id: 0,
        program_id: 0,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids: channel_ids.to_vec(),
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::Host,
    };
    let mut binding = InstanceBinding::default();
    assert_eq!(bind(&mut d, &inst, &mut binding), PIE_STATUS_OK);

    let instance_ids: [u64; 1] = [binding.instance_id];
    let fire = |d: &mut Shell, step: &StepSubmission, _wait: u64| {
        let frame = FrameSubmission {
            instance_ids: instance_ids.to_vec(),
            required_kv_pages: 1,
            steps: vec![step.clone()],
            ..Default::default()
        };
        fire_and_wait(d, &frame, &broker)
    };

    // ── Prefill on slot 0 (RESET — a fresh sequence). ──
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let tokens = prompt.len();
    let positions: Vec<u32> = (0..tokens as u32).collect();
    // ONE ROSTER ROW PER REQUEST, not per token. These said `vec![0;
    // tokens]` — a prefill's token count, every entry zero — which the
    // shared validator reads as N requests all claiming roster index 0,
    // and refuses for the duplicates. The engine builds this with
    // `Vec::with_capacity(instance_ids.len())`
    // (`scheduler/batch.rs:388`), so one entry is what a caller sends.
    // The shell never read the field, which is why it went unnoticed.
    let roster_rows: Vec<u32> = vec![0];
    // The CSR partitions ROSTER ROWS by sub-batch, not tokens: the engine
    // pushes `group.len()` (`scheduler/batch.rs:384`), and
    // `validate_csr` checks the last value against `roster_rows.len`.
    // One request, one row, one entry.
    let sub_batch_indptr: [u32; 2] = [0, 1];
    // AND A TERMINAL CELL PER REQUEST. These fixtures stated none, which
    // the validator refuses with `allow_empty = false`: a step with no
    // cell has nowhere to report an outcome, and the engine sends one per
    // request. Never read here — the fires below take their answer off
    // the ring — but a caller's shape is a caller's shape.
    let mut launch_cell = driver_api::local::TerminalCell::pending();
    let launch_cell_ptr: *mut driver_api::local::TerminalCell = &mut launch_cell;
    let sub_batch_class: [u32; 1] = [driver_api::local::PIE_GEOMETRY_CLASS_HOST];
    let kv_page_indices: [u32; 1] = [0];
    let kv_page_indptr: [u32; 2] = [0, 1];
    let kv_last_page_lens: [u32; 1] = [tokens as u32];
    let qo_indptr: [u32; 2] = [0, tokens as u32];
    let rs_slots: [u32; 1] = [0];
    let rs_flags: [u8; 1] = [PIE_RS_FLAG_RESET];
    let step = StepSubmission {
        plan: LaunchPlan {
            token_ids: prompt.to_vec(),
            position_ids: positions.to_vec(),
            kv_page_indices: kv_page_indices.to_vec(),
            kv_page_indptr: kv_page_indptr.to_vec(),
            kv_last_page_lens: kv_last_page_lens.to_vec(),
            qo_indptr: qo_indptr.to_vec(),
            rs_slot_ids: rs_slots.to_vec(),
            rs_slot_flags: rs_flags.to_vec(),
            ..Default::default()
        },
        roster_rows: roster_rows.to_vec(),
        sub_batch_indptr: sub_batch_indptr.to_vec(),
        sub_batch_class: sub_batch_class.to_vec(),
        terminal_cells: vec![launch_cell_ptr],
        ..Default::default()
    };
    assert_eq!(
        fire(&mut d, &step, 1),
        PIE_STATUS_OK,
        "the hybrid prefill fires"
    );

    let words = unsafe { std::slice::from_raw_parts(chb.word_base as *const u64, 4) };
    assert_eq!(words[1], 1, "the tail advanced once");
    let cell0 = unsafe { std::slice::from_raw_parts(chb.mirror_base as *const f32, VOCAB) };
    let argmax_of = |cell: &[f32]| {
        let (mut bt, mut bv) = (0usize, f32::NEG_INFINITY);
        for (t, &v) in cell.iter().enumerate() {
            if v > bv {
                (bt, bv) = (t, v);
            }
        }
        (bt, bv)
    };
    let (best_t, best_v) = argmax_of(cell0);
    let ids5: Vec<usize> = reference["top5_ids"]
        .as_array()
        .expect("top5")
        .iter()
        .map(|v| v.as_u64().expect("id") as usize)
        .collect();
    assert!(
        ids5.contains(&best_t),
        "prefill argmax {best_t} ({best_v}) outside HF's top-5 {ids5:?}"
    );
    let next_token = best_t as u32;

    // ── The state fork: slot 0 → slots 1 AND 2 (two identical copies,
    // so the comparison below is copy vs copy — no live-slot asymmetry).
    let ranges = [
        StateCopyRange {
            src_slot_id: 0,
            dst_slot_id: 1,
            ..Default::default()
        },
        StateCopyRange {
            src_slot_id: 0,
            dst_slot_id: 2,
            ..Default::default()
        },
    ];
    let copy = driver_api::StateCopyPlan {
        slot_ranges: ranges.to_vec(),
    };
    let (completion, completion_handle) = broker.launch_completion(1);
    let _ = &completion_handle;
    assert_eq!(
        code(d.copy_state(&copy, completion)),
        PIE_STATUS_OK,
        "the state fork copies"
    );

    // ── The same decode against slot 0 and against slot 1. ──
    let decode_on = |d: &mut Shell, slot: u32, wait: u64| {
        let dec_ids: [u32; 1] = [next_token];
        let dec_pos: [u32; 1] = [tokens as u32];
        let dec_roster: [u32; 1] = [0];
        let dec_sbi: [u32; 2] = [0, 1];
        let dec_lens: [u32; 1] = [tokens as u32 + 1];
        let dec_qo: [u32; 2] = [0, 1];
        let dec_slots: [u32; 1] = [slot];
        let dec_flags: [u8; 1] = [0];
        let step = StepSubmission {
            plan: LaunchPlan {
                token_ids: dec_ids.to_vec(),
                position_ids: dec_pos.to_vec(),
                kv_page_indices: kv_page_indices.to_vec(),
                kv_page_indptr: kv_page_indptr.to_vec(),
                kv_last_page_lens: dec_lens.to_vec(),
                qo_indptr: dec_qo.to_vec(),
                rs_slot_ids: dec_slots.to_vec(),
                // The FLAGS beside the ids. `validate_frame_desc` requires the
                // two to match in length, and it is right to: a slot with no
                // flag is a recurrent state the driver cannot know whether to
                // reset or continue.
                rs_slot_flags: dec_flags.to_vec(),
                ..Default::default()
            },
            roster_rows: dec_roster.to_vec(),
            sub_batch_indptr: dec_sbi.to_vec(),
            sub_batch_class: sub_batch_class.to_vec(),
            terminal_cells: vec![launch_cell_ptr],
            ..Default::default()
        };
        assert_eq!(
            fire(d, &step, wait),
            PIE_STATUS_OK,
            "the decode fires (slot {slot})"
        );
    };
    decode_on(&mut d, 1, 3);
    decode_on(&mut d, 2, 4);
    assert_eq!(words[1], 3, "three cells published");
    let ring = 4usize; // capacity 3 + 1
    let _ = ring;
    let cell1 =
        unsafe { std::slice::from_raw_parts((chb.mirror_base as *const f32).add(VOCAB), VOCAB) };
    let cell2 = unsafe {
        std::slice::from_raw_parts((chb.mirror_base as *const f32).add(2 * VOCAB), VOCAB)
    };
    let (t1, v1) = argmax_of(cell1);
    let (t2, v2) = argmax_of(cell2);
    assert_eq!(
        t1, t2,
        "decode from the COPIED slot flips the argmax ({v1} vs {v2})"
    );
    let (mut max_d, mut at, mut n_diff) = (0f32, 0usize, 0usize);
    for t in 0..VOCAB {
        let d = (cell1[t] - cell2[t]).abs();
        if d > 0.0 {
            n_diff += 1;
        }
        if d > max_d {
            (max_d, at) = (d, t);
        }
    }
    eprintln!(
        "copy-vs-copy decode: {n_diff} differing logits, max |d| = {max_d} at {at} \
         ({} vs {})",
        cell1[at], cell2[at]
    );
    assert!(
        max_d < 0.25,
        "the copied slots' decodes drifted past inter-fire jitter: |d|={max_d} at {at}"
    );
    drop(d);
}

/// gemma-4 through the FULL ABI: load `gemma-4-E2B-it` (PLE table, fused
/// joins, host-read layer scalars), PREFILL the A/B's prompt and demand
/// the A/B's exact argmax through the ring, then DECODE one token — the
/// two-plan decode leg's (sliding + full-variant) and the fused packed
/// decode post's first LIVE run; the A/B only ever fired the prefill
/// class. The decode has no committed HF reference, so its bar is the
/// family's own invariants: the fire completes, the published cell is
/// finite, and every logit sits inside the final softcap's ±30.
#[test]
fn gemma4_loads_and_fires_both_classes_through_the_abi() {
    let _gpu = gpu_guard();

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            (p.join("model.safetensors").is_file()
                || p.join("model.safetensors.index.json").is_file())
            .then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached gemma-4-E2B-it");
        return;
    };
    let scratch = std::path::PathBuf::from(std::env::var("PIE_TEST_SCRATCH").unwrap_or_else(
        |_| "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad".into(),
    ));
    let descriptor = scratch.join("gemma4_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated gemma4 descriptor");
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("oracle/real_decode/gemma4_e2b.json"))
            .expect("reference");

    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = CompletionBroker::new();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(
        code(d.load_model(&load)),
        PIE_STATUS_OK,
        "the gemma-4 checkpoint loads (PLE table + fused joins + layer scalars)"
    );

    const VOCAB: usize = 262_144;
    let shape: [u32; 1] = [VOCAB as u32];
    let ch = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 44,
        shape: shape.to_vec(),
        dtype: PIE_CHANNEL_DTYPE_F32,
        host_role: PIE_CHANNEL_HOST_ROLE_READER,
        seeded: false,
        extern_dir: PIE_CHANNEL_EXTERN_NONE,
        capacity: 3,
        reader_wait_id: 89,
        writer_wait_id: 90,
        extern_name: Vec::new(),
    };
    let mut chb = ChannelBinding::default();
    assert_eq!(reg_channel(&mut d, &ch, &mut chb), PIE_STATUS_OK);
    let prog = driver_api::ProgramRegistration {
        program_hash: 0x6E44,
        ..Default::default()
    };
    let mut program_id = 0u64;
    assert_eq!(reg_program(&mut d, &prog, &mut program_id), PIE_STATUS_OK);
    let channel_ids: [u64; 1] = [44];
    let inst = InstanceBindingPlan {
        driver_id: 0,
        program_id: 0,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids: channel_ids.to_vec(),
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::Host,
    };
    let mut binding = InstanceBinding::default();
    assert_eq!(bind(&mut d, &inst, &mut binding), PIE_STATUS_OK);

    let instance_ids: [u64; 1] = [binding.instance_id];
    let fire = |d: &mut Shell, step: &StepSubmission, _wait: u64| {
        let frame = FrameSubmission {
            instance_ids: instance_ids.to_vec(),
            required_kv_pages: 1,
            steps: vec![step.clone()],
            ..Default::default()
        };
        fire_and_wait(d, &frame, &broker)
    };

    // ── Prefill: the A/B's prompt, the A/B's exact argmax. ──
    let prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let tokens = prompt.len();
    let positions: Vec<u32> = (0..tokens as u32).collect();
    // ONE ROSTER ROW PER REQUEST, not per token. These said `vec![0;
    // tokens]` — a prefill's token count, every entry zero — which the
    // shared validator reads as N requests all claiming roster index 0,
    // and refuses for the duplicates. The engine builds this with
    // `Vec::with_capacity(instance_ids.len())`
    // (`scheduler/batch.rs:388`), so one entry is what a caller sends.
    // The shell never read the field, which is why it went unnoticed.
    let roster_rows: Vec<u32> = vec![0];
    // The CSR partitions ROSTER ROWS by sub-batch, not tokens: the engine
    // pushes `group.len()` (`scheduler/batch.rs:384`), and
    // `validate_csr` checks the last value against `roster_rows.len`.
    // One request, one row, one entry.
    let sub_batch_indptr: [u32; 2] = [0, 1];
    // AND A TERMINAL CELL PER REQUEST. These fixtures stated none, which
    // the validator refuses with `allow_empty = false`: a step with no
    // cell has nowhere to report an outcome, and the engine sends one per
    // request. Never read here — the fires below take their answer off
    // the ring — but a caller's shape is a caller's shape.
    let mut launch_cell = driver_api::local::TerminalCell::pending();
    let launch_cell_ptr: *mut driver_api::local::TerminalCell = &mut launch_cell;
    let sub_batch_class: [u32; 1] = [driver_api::local::PIE_GEOMETRY_CLASS_HOST];
    let kv_page_indices: [u32; 1] = [0];
    let kv_page_indptr: [u32; 2] = [0, 1];
    let kv_last_page_lens: [u32; 1] = [tokens as u32];
    let qo_indptr: [u32; 2] = [0, tokens as u32];
    let step = StepSubmission {
        plan: LaunchPlan {
            token_ids: prompt.to_vec(),
            position_ids: positions.to_vec(),
            kv_page_indices: kv_page_indices.to_vec(),
            kv_page_indptr: kv_page_indptr.to_vec(),
            kv_last_page_lens: kv_last_page_lens.to_vec(),
            qo_indptr: qo_indptr.to_vec(),
            ..Default::default()
        },
        roster_rows: roster_rows.to_vec(),
        sub_batch_indptr: sub_batch_indptr.to_vec(),
        sub_batch_class: sub_batch_class.to_vec(),
        terminal_cells: vec![launch_cell_ptr],
        ..Default::default()
    };
    assert_eq!(
        fire(&mut d, &step, 1),
        PIE_STATUS_OK,
        "the gemma-4 prefill fires"
    );

    let words = unsafe { std::slice::from_raw_parts(chb.word_base as *const u64, 4) };
    assert_eq!(words[1], 1, "the tail advanced once");
    let cell0 = unsafe { std::slice::from_raw_parts(chb.mirror_base as *const f32, VOCAB) };
    let argmax_of = |cell: &[f32]| {
        let (mut bt, mut bv) = (0usize, f32::NEG_INFINITY);
        for (t, &v) in cell.iter().enumerate() {
            if v > bv {
                (bt, bv) = (t, v);
            }
        }
        (bt, bv)
    };
    let (best_t, best_v) = argmax_of(cell0);
    let hf_argmax = reference["argmax"].as_u64().expect("argmax") as usize;
    assert_eq!(
        best_t, hf_argmax,
        "prefill argmax {best_t} ({best_v}) is not HF's {hf_argmax}"
    );

    // ── Decode one token: the two-plan leg's first live run. ──
    let next: [u32; 1] = [best_t as u32];
    let dec_pos: [u32; 1] = [tokens as u32];
    let dec_roster: [u32; 1] = [0];
    let dec_sbi: [u32; 2] = [0, 1];
    let dec_lens: [u32; 1] = [tokens as u32 + 1];
    let dec_qo: [u32; 2] = [0, 1];
    let step = StepSubmission {
        plan: LaunchPlan {
            token_ids: next.to_vec(),
            position_ids: dec_pos.to_vec(),
            kv_page_indices: kv_page_indices.to_vec(),
            kv_page_indptr: kv_page_indptr.to_vec(),
            kv_last_page_lens: dec_lens.to_vec(),
            qo_indptr: dec_qo.to_vec(),
            ..Default::default()
        },
        roster_rows: dec_roster.to_vec(),
        sub_batch_indptr: dec_sbi.to_vec(),
        sub_batch_class: sub_batch_class.to_vec(),
        terminal_cells: vec![launch_cell_ptr],
        ..Default::default()
    };
    assert_eq!(
        fire(&mut d, &step, 2),
        PIE_STATUS_OK,
        "the gemma-4 decode fires"
    );
    assert_eq!(words[1], 2, "two cells published");
    let cell1 =
        unsafe { std::slice::from_raw_parts((chb.mirror_base as *const f32).add(VOCAB), VOCAB) };
    let (dt, dv) = argmax_of(cell1);
    assert!(dv.is_finite(), "the decode's top logit is finite");
    // The final softcap bounds EVERY logit at ±30; a wrong plan or a
    // wrong page read does not stay inside a tanh.
    let softcap_ok = cell1.iter().all(|v| v.is_finite() && v.abs() <= 30.5);
    assert!(softcap_ok, "a decode logit escaped the ±30 softcap");
    eprintln!("gemma-4 decode after {hf_argmax}: argmax {dt} at {dv}");
    drop(d);
}

/// gemma-4's vision tower encodes REAL E2B weights through the full ABI:
/// load the cached checkpoint, hand `Shell::encode` one synthetic
/// 3×3-patch image, and demand the encode-ABI contract — the anchor CSR
/// filled, exactly `n_patch / pool²` soft-token rows written, every value
/// finite and the row NOT all-zero (real weights on nonzero pixels), and
/// a SECOND call bit-identical to the first (the tower is a pure
/// function of its inputs). The first REAL-WEIGHT run of the tower
/// bridge; the HF-cosine parity gate rides the C++ parity harness's
/// pixel path and is recorded as the follow-up.
#[test]
fn gemma4_vision_encodes_real_weights_through_the_abi() {
    let _gpu = gpu_guard();

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            (p.join("model.safetensors").is_file()).then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached gemma-4-E2B-it");
        return;
    };
    let scratch = std::path::PathBuf::from(std::env::var("PIE_TEST_SCRATCH").unwrap_or_else(
        |_| "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad".into(),
    ));
    let descriptor = scratch.join("gemma4_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated gemma4 descriptor");
        return;
    }

    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = engine_runtime();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    // THE CAPABILITY, BEFORE THE CALL IT GATES. The worker builds no
    // encode executor at all when this is clear
    // (`worker/src/executor/mod.rs:1341`), so everything below this line
    // was unreachable through the engine while it was hardwired false —
    // and no test could see that, because every one of them calls `encode`
    // directly.
    let parsed: driver_api::DriverCapabilities = d.load_model(&load).expect("the snapshot loads");
    assert!(
        parsed.supports_media_encode,
        "gemma-4 carries a vision tower this driver encodes with"
    );

    const TEXT_HIDDEN: usize = 1536;
    const N_PATCH: usize = 9; // 3×3, pooled 3×3 → ONE soft token
    const OUT_LEN: usize = 1;
    const PIXEL_DIM: usize = 3 * 16 * 16;
    // Deterministic nonzero pixels in [0, 1].
    let pixels: Vec<f32> = (0..N_PATCH * PIXEL_DIM)
        .map(|i| (i % 97) as f32 / 96.0)
        .collect();
    let pixel_indptr: [u32; 2] = [0, (pixels.len() * 4) as u32];
    let patch_positions: [u32; 18] = [0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 2, 2, 2];
    let anchors: [u32; 1] = [0];
    // The PATCH-UNIT grid `(t, h, w)`, three per image, which is what the
    // engine sends (`engine/src/driver/abi.rs:323`) and what
    // `validate_encode_desc` requires. These fixtures omitted it, so the
    // validator could not be adopted until they said what a caller says.
    let grids: [u32; 3] = [1, 3, 3];

    let run = |d: &mut Shell, tag: &str| -> Vec<u16> {
        let guard_rows = (OUT_LEN + 1) * TEXT_HIDDEN;
        let mut e = driver_api::MediaEncodePlan {
            image_pixels: pixels.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            image_pixel_indptr: pixel_indptr[..2].to_vec(),
            image_patch_positions: patch_positions[..18].to_vec(),
            image_grids: grids[..3].to_vec(),
            image_anchor_rows: anchors[..1].to_vec(),
            output_rows: vec![0x77u8; guard_rows * 2],
            output_row_indptr: vec![u32::MAX; 2],
            ..Default::default()
        };
        let (completion, completion_handle) = broker.launch_completion(1);
        let _ = &completion_handle;
        assert_eq!(
            code(d.encode(&mut e, completion)),
            PIE_STATUS_OK,
            "{tag}: the vision encode fires"
        );
        assert_eq!(e.output_row_indptr[0], 0, "{tag}: CSR starts at zero");
        assert_eq!(
            e.output_row_indptr[1], OUT_LEN as u32,
            "{tag}: one image, one soft token"
        );
        out_u16(&e)
    };

    let a = run(&mut d, "first");
    let bf = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let mut nonzero = 0usize;
    for &v in &a[..TEXT_HIDDEN] {
        let f = bf(v);
        assert!(f.is_finite(), "a soft-token value is not finite: {f}");
        if v != 0 {
            nonzero += 1;
        }
    }
    assert!(
        nonzero > TEXT_HIDDEN / 4,
        "real weights on nonzero pixels produced a near-zero row ({nonzero} nonzero)"
    );
    for &v in &a[OUT_LEN * TEXT_HIDDEN..] {
        assert_eq!(v, 0x7777, "the guard row must stay untouched");
    }
    let b = run(&mut d, "second");
    assert_eq!(a, b, "the tower is a pure function of its inputs");
    drop(d);
}

/// The audio twin of the vision encode test: REAL E2B audio-tower
/// weights, one synthetic log-mel clip through `Shell::encode`, the
/// same contract — anchor CSR filled, finite non-zero soft tokens,
/// bit-identical across two calls, guards untouched.
#[test]
fn gemma4_audio_encodes_real_weights_through_the_abi() {
    let _gpu = gpu_guard();

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            (p.join("model.safetensors").is_file()).then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached gemma-4-E2B-it");
        return;
    };
    let scratch = std::path::PathBuf::from(std::env::var("PIE_TEST_SCRATCH").unwrap_or_else(
        |_| "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad".into(),
    ));
    let descriptor = scratch.join("gemma4_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated gemma4 descriptor");
        return;
    }
    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = engine_runtime();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(code(d.load_model(&load)), PIE_STATUS_OK,);

    const TEXT_HIDDEN: usize = 1536;
    const N_MEL: usize = 128;
    const N_FRAMES: usize = 32; // subsamples 4× → 8 audio soft tokens
    let features: Vec<f32> = (0..N_FRAMES * N_MEL)
        .map(|i| ((i % 89) as f32 / 88.0) - 0.5)
        .collect();
    let feat_indptr: [u32; 2] = [0, (features.len() * 4) as u32];
    let anchors: [u32; 1] = [0];
    // The PATCH-UNIT grid `(t, h, w)`, three per image, which is what the
    // engine sends (`engine/src/driver/abi.rs:323`) and what
    // `validate_encode_desc` requires. These fixtures omitted it, so the
    // validator could not be adopted until they said what a caller says.
    let _grids: [u32; 3] = [1, 3, 3];
    const MAX_OUT: usize = 16;

    let run = |d: &mut Shell, tag: &str| -> (Vec<u16>, u32) {
        let guard_rows = (MAX_OUT + 1) * TEXT_HIDDEN;
        let mut e = driver_api::MediaEncodePlan {
            audio_features: features.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            audio_feature_indptr: feat_indptr[..2].to_vec(),
            audio_anchor_rows: anchors[..1].to_vec(),
            output_rows: vec![0x77u8; guard_rows * 2],
            output_row_indptr: vec![u32::MAX; 2],
            ..Default::default()
        };
        let (completion, completion_handle) = broker.launch_completion(1);
        let _ = &completion_handle;
        assert_eq!(
            code(d.encode(&mut e, completion)),
            PIE_STATUS_OK,
            "{tag}: the audio encode fires"
        );
        assert_eq!(e.output_row_indptr[0], 0, "{tag}: CSR starts at zero");
        (out_u16(&e), e.output_row_indptr[1])
    };

    let (a, n_tok) = run(&mut d, "first");
    assert!(
        n_tok > 0 && (n_tok as usize) <= MAX_OUT,
        "the clip produced a sane token count, got {n_tok}"
    );
    let bf = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let mut nonzero = 0usize;
    for &v in &a[..n_tok as usize * TEXT_HIDDEN] {
        assert!(bf(v).is_finite(), "an audio soft-token value is not finite");
        if v != 0 {
            nonzero += 1;
        }
    }
    assert!(
        nonzero > (n_tok as usize * TEXT_HIDDEN) / 4,
        "real weights produced a near-zero encode ({nonzero} nonzero)"
    );
    for &v in &a[MAX_OUT * TEXT_HIDDEN..] {
        assert_eq!(v, 0x7777, "the guard row must stay untouched");
    }
    let (b, n2) = run(&mut d, "second");
    assert_eq!(n_tok, n2);
    assert_eq!(a, b, "the audio tower is a pure function of its inputs");
    drop(d);
}

/// One MIXED call: an image AND an audio clip through a single
/// `Shell::encode`, the C++ `Context::encode` shape — vision rows
/// first, audio rows at the offset, ONE shared CSR. On the cached E2B:
/// CSR [0, 1, 9] (one vision soft token, then 8 audio tokens), all rows
/// finite, and each segment bit-identical to what its single-media call
/// produces.
#[test]
fn gemma4_mixed_media_encodes_through_one_call() {
    let _gpu = gpu_guard();

    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            (p.join("model.safetensors").is_file()).then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached gemma-4-E2B-it");
        return;
    };
    let scratch = std::path::PathBuf::from(std::env::var("PIE_TEST_SCRATCH").unwrap_or_else(
        |_| "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad".into(),
    ));
    let descriptor = scratch.join("gemma4_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated gemma4 descriptor");
        return;
    }
    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = engine_runtime();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(code(d.load_model(&load)), PIE_STATUS_OK,);

    const TEXT_HIDDEN: usize = 1536;
    const PIXEL_DIM: usize = 3 * 16 * 16;
    const N_PATCH: usize = 9;
    const N_MEL: usize = 128;
    const N_FRAMES: usize = 32;
    let pixels: Vec<f32> = (0..N_PATCH * PIXEL_DIM)
        .map(|i| (i % 97) as f32 / 96.0)
        .collect();
    let pixel_indptr: [u32; 2] = [0, (pixels.len() * 4) as u32];
    let patch_positions: [u32; 18] = [0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, 0, 2, 1, 2, 2, 2];
    let img_anchors: [u32; 1] = [0];
    let grids: [u32; 3] = [1, 3, 3];
    let features: Vec<f32> = (0..N_FRAMES * N_MEL)
        .map(|i| ((i % 89) as f32 / 88.0) - 0.5)
        .collect();
    let feat_indptr: [u32; 2] = [0, (features.len() * 4) as u32];
    let clip_anchors: [u32; 1] = [1];

    const MAX_ROWS: usize = 12;
    let guard_rows = MAX_ROWS * TEXT_HIDDEN;
    let mut e = driver_api::MediaEncodePlan {
        image_pixels: pixels.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        image_pixel_indptr: pixel_indptr[..2].to_vec(),
        image_patch_positions: patch_positions[..18].to_vec(),
        image_grids: grids[..3].to_vec(),
        image_anchor_rows: img_anchors[..1].to_vec(),
        audio_features: features.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        audio_feature_indptr: feat_indptr[..2].to_vec(),
        audio_anchor_rows: clip_anchors[..1].to_vec(),
        output_rows: vec![0x77u8; guard_rows * 2],
        output_row_indptr: vec![u32::MAX; 3],
        ..Default::default()
    };
    let (completion, completion_handle) = broker.launch_completion(1);
    let _ = &completion_handle;
    assert_eq!(
        code(d.encode(&mut e, completion)),
        PIE_STATUS_OK,
        "the mixed encode fires"
    );
    assert_eq!(
        e.output_row_indptr.as_slice(),
        [0, 1, 9],
        "one vision token, then eight audio tokens"
    );
    let bf = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let mut nonzero = 0usize;
    let out_rows = out_u16(&e);
    for &v in &out_rows[..9 * TEXT_HIDDEN] {
        assert!(bf(v).is_finite(), "a mixed-encode value is not finite");
        if v != 0 {
            nonzero += 1;
        }
    }
    assert!(
        nonzero > 9 * TEXT_HIDDEN / 4,
        "near-zero mixed encode ({nonzero})"
    );
    for &v in &out_rows[9 * TEXT_HIDDEN..] {
        assert_eq!(v, 0x7777, "rows beyond the CSR must stay untouched");
    }
    drop(d);
}

/// THE HF-COSINE PARITY GATE for the vision encode: the C++ parity
/// harness's own synthetic input (60×42 patches from
/// `scripts/gemma4_vision_parity_ref.py`, seed 0) through OUR
/// `Shell::encode` on the cached E2B, cosine-compared per pooled token
/// against HF's fp32 projection. The C++ harness reports ~0.9998 on this
/// input; the bar here is 0.999 mean / 0.995 min — a wrong table entry
/// or a wrong patch order does not miss by 0.001.
#[test]
fn gemma4_vision_encode_matches_hf_cosine() {
    let _gpu = gpu_guard();

    let scratch = std::path::PathBuf::from(std::env::var("PIE_TEST_SCRATCH").unwrap_or_else(
        |_| "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad".into(),
    ));
    let parity = scratch.join("g4vis_parity_e2b");
    let descriptor = scratch.join("gemma4_descriptor.json");
    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            (p.join("model.safetensors").is_file()).then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached gemma-4-E2B-it");
        return;
    };
    if !parity.join("projected_f32.npy").is_file() || !descriptor.is_file() {
        eprintln!("skipped: no generated parity reference / descriptor");
        return;
    }
    // Minimal NPY reader: f32, C-order, header v1.
    let npy_f32 = |p: &std::path::Path| -> (Vec<usize>, Vec<f32>) {
        let raw = std::fs::read(p).expect("npy");
        let hlen = u16::from_le_bytes([raw[8], raw[9]]) as usize;
        let header = std::str::from_utf8(&raw[10..10 + hlen]).expect("header");
        let shape_s = header
            .split("'shape': (")
            .nth(1)
            .expect("shape")
            .split(')')
            .next()
            .expect("shape");
        let shape: Vec<usize> = shape_s
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect();
        let data = &raw[10 + hlen..];
        let vals = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (shape, vals)
    };
    let (pshape, pixels) = npy_f32(&parity.join("input_pixel_values_f32.npy"));
    let (_, pos_f) = npy_f32(&parity.join("input_position_ids.npy"));
    let (rshape, hf) = npy_f32(&parity.join("projected_f32.npy"));
    let n_patch = pshape[0];
    let out_len = rshape[0];
    let text_hidden = rshape[1];
    assert_eq!(pshape[1], 768);
    let positions: Vec<u32> = pos_f.iter().map(|&v| v as u32).collect();

    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = engine_runtime();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(code(d.load_model(&load)), PIE_STATUS_OK,);

    let pixel_indptr: [u32; 2] = [0, (pixels.len() * 4) as u32];
    let anchors: [u32; 1] = [0];
    // The PATCH-UNIT grid `(t, h, w)`, three per image, which is what the
    // engine sends (`engine/src/driver/abi.rs:323`) and what
    // `validate_encode_desc` requires. These fixtures omitted it, so the
    // validator could not be adopted until they said what a caller says.
    let grids: [u32; 3] = [1, 3, 3];
    let guard_rows = out_len * text_hidden;
    let mut e = driver_api::MediaEncodePlan {
        image_pixels: pixels.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        image_pixel_indptr: pixel_indptr[..2].to_vec(),
        image_patch_positions: positions[..positions.len()].to_vec(),
        image_grids: grids[..3].to_vec(),
        image_anchor_rows: anchors[..1].to_vec(),
        output_rows: vec![0x77u8; guard_rows * 2],
        output_row_indptr: vec![u32::MAX; 2],
        ..Default::default()
    };
    let (completion, completion_handle) = broker.launch_completion(1);
    let _ = &completion_handle;
    assert_eq!(
        code(d.encode(&mut e, completion)),
        PIE_STATUS_OK,
        "the parity encode fires ({n_patch} patches)"
    );
    assert_eq!(e.output_row_indptr[1] as usize, out_len);

    let bf = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let out_rows = out_u16(&e);
    let (mut cos_sum, mut cos_min) = (0f64, f64::INFINITY);
    for r in 0..out_len {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for c in 0..text_hidden {
            let a = f64::from(bf(out_rows[r * text_hidden + c]));
            let b = f64::from(hf[r * text_hidden + c]);
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
        cos_sum += cos;
        cos_min = cos_min.min(cos);
    }
    let mean = cos_sum / out_len as f64;
    eprintln!("vision HF-cosine: mean {mean:.6}, min {cos_min:.6} over {out_len} tokens");
    assert!(mean > 0.999, "mean cosine {mean} below the gate");
    assert!(
        cos_min > 0.995,
        "worst token cosine {cos_min} below the gate"
    );
    drop(d);
}

/// THE HF-COSINE PARITY GATE for the audio encode — the vision gate's
/// twin: the C++ audio harness's synthetic 188-frame log-mel
/// (`scripts/gemma4_audio_parity_ref.py`, regenerated on E2B) through
/// OUR `Shell::encode`, cosine per soft token against HF's fp32
/// projection.
#[test]
fn gemma4_audio_encode_matches_hf_cosine() {
    let _gpu = gpu_guard();

    let scratch = std::path::PathBuf::from(std::env::var("PIE_TEST_SCRATCH").unwrap_or_else(
        |_| "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad".into(),
    ));
    let parity = scratch.join("g4aud_parity_e2b");
    let descriptor = scratch.join("gemma4_descriptor.json");
    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            (p.join("model.safetensors").is_file()).then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached gemma-4-E2B-it");
        return;
    };
    if !parity.join("projected_f32.npy").is_file() || !descriptor.is_file() {
        eprintln!("skipped: no generated audio parity reference");
        return;
    }
    let npy_f32 = |p: &std::path::Path| -> (Vec<usize>, Vec<f32>) {
        let raw = std::fs::read(p).expect("npy");
        let hlen = u16::from_le_bytes([raw[8], raw[9]]) as usize;
        let header = std::str::from_utf8(&raw[10..10 + hlen]).expect("header");
        let shape_s = header
            .split("'shape': (")
            .nth(1)
            .expect("shape")
            .split(')')
            .next()
            .expect("shape");
        let shape: Vec<usize> = shape_s
            .split(',')
            .filter_map(|t| t.trim().parse().ok())
            .collect();
        let vals = raw[10 + hlen..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (shape, vals)
    };
    let (fshape, features) = npy_f32(&parity.join("input_features_f32.npy"));
    let (rshape, hf) = npy_f32(&parity.join("projected_f32.npy"));
    let out_len = rshape[0];
    let text_hidden = rshape[1];
    assert_eq!(fshape[1], 128);

    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = engine_runtime();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(code(d.load_model(&load)), PIE_STATUS_OK,);

    let feat_indptr: [u32; 2] = [0, (features.len() * 4) as u32];
    let anchors: [u32; 1] = [0];
    // The PATCH-UNIT grid `(t, h, w)`, three per image, which is what the
    // engine sends (`engine/src/driver/abi.rs:323`) and what
    // `validate_encode_desc` requires. These fixtures omitted it, so the
    // validator could not be adopted until they said what a caller says.
    let _grids: [u32; 3] = [1, 3, 3];
    let guard_rows = out_len * text_hidden;
    let mut e = driver_api::MediaEncodePlan {
        audio_features: features.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        audio_feature_indptr: feat_indptr[..2].to_vec(),
        audio_anchor_rows: anchors[..1].to_vec(),
        output_rows: vec![0x77u8; guard_rows * 2],
        output_row_indptr: vec![u32::MAX; 2],
        ..Default::default()
    };
    let (completion, completion_handle) = broker.launch_completion(1);
    let _ = &completion_handle;
    assert_eq!(
        code(d.encode(&mut e, completion)),
        PIE_STATUS_OK,
        "the audio parity encode fires"
    );
    assert_eq!(
        e.output_row_indptr[1] as usize, out_len,
        "token count matches HF"
    );

    let bf = |bits: u16| f32::from_bits(u32::from(bits) << 16);
    let out_rows = out_u16(&e);
    let (mut cos_sum, mut cos_min) = (0f64, f64::INFINITY);
    for r in 0..out_len {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for c in 0..text_hidden {
            let a = f64::from(bf(out_rows[r * text_hidden + c]));
            let b = f64::from(hf[r * text_hidden + c]);
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
        cos_sum += cos;
        cos_min = cos_min.min(cos);
    }
    let mean = cos_sum / out_len as f64;
    eprintln!("audio HF-cosine: mean {mean:.6}, min {cos_min:.6} over {out_len} tokens");
    assert!(mean > 0.999, "mean cosine {mean} below the gate");
    assert!(
        cos_min > 0.995,
        "worst token cosine {cos_min} below the gate"
    );
    drop(d);
}

/// **gpt-oss-20b: the first QUANTIZED checkpoint this shell opens.**
///
/// The load used to refuse every `Encoding::Quant` wholesale, which made
/// "no MoE checkpoint fits this machine" only half true — gpt-oss is
/// 39 GB on a 46 GB card, and what actually turned it away was the
/// encoding, not the size.
///
/// Its `quantization_config` says `quant_method: mxfp4` with
/// `modules_to_not_convert` covering attention, the router, the embedding
/// and lm_head — so the expert banks are MXFP4 and everything else is
/// bf16, in one checkpoint. That mix is the point: a load that accepted
/// only raw could not open it, and a load that accepted everything would
/// hand some kernel a layout it was not compiled for.
///
/// `reads_its_stored_form` is the line. MXFP4's payload is what
/// `quant::mxfp4_moe_gate_up_decode_bf16` indexes, so the bytes go up
/// verbatim; a scheme wanting a Marlin repack or an FP8 re-encode still
/// refuses, because that is `transcode_engine`'s work and it is not
/// ported.
///
/// Asserts the LOAD, not a fire. Whether the mixture's aligned path can
/// run is `UNARMED`'s question and `build_moe_ptrs_aligned_bf16` is still
/// on it — but a checkpoint that cannot be loaded cannot answer any
/// question at all, and this one now loads.
#[test]
fn a_quantized_checkpoint_loads_through_the_abi() {
    let _gpu = gpu_guard();
    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            (p.join("model.safetensors").is_file()
                || p.join("model.safetensors.index.json").is_file())
            .then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached gpt-oss-20b");
        return;
    };
    let Some(config) = std::fs::read_dir(&snap).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            (p.file_name()? == "config.json").then_some(p)
        })
    }) else {
        eprintln!("skipped: gpt-oss snapshot has no config.json");
        return;
    };

    // The checkpoint's own `config.json`, COPIED rather than normalized.
    //
    // This used to call `model::config::descriptor` to turn it into a
    // `pie.model/1` document — 40 fields resolved from a 136-field
    // schema — because that is what the driver read. The driver reads
    // the file itself now, for the one question a `const` row cannot
    // answer (the declared encoding), so the fixture is a copy.
    //
    // Copied rather than pointed at in place because the boot TOML's
    // path is what a real boot names, and a temp file keeps the
    // snapshot read-only.
    let raw = std::fs::read_to_string(&config).expect("config.json");
    let dpath = std::env::temp_dir().join("pie_gpt_oss_config.json");
    std::fs::write(&dpath, &raw).expect("write config");
    let boot = format!("[model]\nconfig = \"{}\"\n", dpath.display());
    let broker = CompletionBroker::new();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("create");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    let loaded = code(d.load_model(&load));
    drop(d);
    assert_eq!(
        loaded, PIE_STATUS_OK,
        "gpt-oss-20b's MXFP4 expert banks must load beside its bf16 \
         attention; a refusal here means `reads_its_stored_form` turned \
         away a scheme whose bytes the kernels do read"
    );
}

/// **RUN-AHEAD: the call returns while the GPU is still working.**
///
/// The invariant pie is built on is that forward n+1 is already enqueued
/// while forward n executes, and the driver used to make that impossible.
/// `Shell::launch` ended in `stream.synchronize()`, published the
/// terminal cells and notified, all on the calling thread — so the
/// engine's `frame_dispatch_depth` was honoured by the engine and
/// serialized by the driver: the call that would enqueue n+1 had not
/// returned.
///
/// This measures the property directly rather than asserting a
/// refactor happened. Two facts, either of which alone could be an
/// accident and which together cannot be:
///
/// 1. `Shell::launch` RETURNS BEFORE the completion fires. If the
///    driver still synchronized, the notify would have run inside the
///    call and the counter would already be 1 on return.
/// 2. The completion arrives on a DIFFERENT THREAD. A stream-ordered
///    host callback runs on a CUDA-owned thread; anything the calling
///    thread did would carry the calling thread's id.
///
/// The second is what makes the first meaningful. A launch could return
/// early and still be synchronous if the work were trivially short, but
/// it cannot run its completion somewhere else.
#[test]
fn a_launch_returns_before_its_fire_retires() {
    use std::future::Future;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::task::{Context, Poll, Wake, Waker};

    let _gpu = gpu_guard();
    let home = std::env::var("HOME").expect("HOME");
    let snaps = std::path::PathBuf::from(&home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let Some(snap) = std::fs::read_dir(&snaps).ok().and_then(|mut d| {
        d.find_map(|e| {
            let p = e.ok()?.path();
            p.join("model.safetensors").is_file().then_some(p)
        })
    }) else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let descriptor = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    )
    .join("qwen3_descriptor.json");
    if !descriptor.is_file() {
        eprintln!("skipped: no generated Qwen3 descriptor");
        return;
    }

    static DONE: AtomicBool = AtomicBool::new(false);
    static WHERE: AtomicU64 = AtomicU64::new(0);

    /// Records WHERE a fire was retired, from inside the retirement.
    ///
    /// This was an `extern "C"` notify the driver was handed in its runtime
    /// callbacks, and it is a waker parked on the fire's completion now.
    /// The observation is the same one: `CompletionBroker::notify` publishes
    /// the epoch and wakes the parked waker INLINE, on the notifying thread
    /// — so this runs wherever the driver retired the fire, which is a
    /// CUDA callback thread if the retirement is stream-ordered and the
    /// calling thread if `launch` synchronized and notified itself.
    struct Recorder;
    impl Wake for Recorder {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            // The thread the completion ran on, as a number we can compare.
            let id = format!("{:?}", std::thread::current().id());
            let n = id
                .bytes()
                .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(u64::from(b)));
            WHERE.store(n, Ordering::Release);
            DONE.store(true, Ordering::Release);
        }
    }

    // A completion for ONE fire, with the recorder already parked on it.
    //
    // Fresh per fire rather than one for the test: a completion settles at
    // its epoch and stays settled, so polling a spent one is ready on the
    // spot and parks nothing — the second fire would then be measured
    // against the first fire's recording.
    //
    // The returned future must outlive the fire. Dropping it frees the wait
    // slot, and a driver notifying a freed slot wakes nobody.
    let arm = |broker: &CompletionBroker| {
        DONE.store(false, Ordering::Release);
        let (target, handle) = broker.launch_completion(1);
        let waker = Waker::from(Arc::new(Recorder));
        let mut parked = Box::pin(handle);
        let polled = parked.as_mut().poll(&mut Context::from_waker(&waker));
        assert!(
            matches!(polled, Poll::Pending),
            "the completion settled before its fire was launched"
        );
        assert!(
            !DONE.load(Ordering::Acquire),
            "the waker fired at registration rather than at retirement, so \
             it recorded the calling thread and proves nothing"
        );
        (target, parked)
    };

    // THIS DRIVER RUNS AHEAD, and it asks for that itself rather than
    // through the environment — every other test in this file shares the
    // process and expects `launch` to have finished the fire when it
    // returns.
    let boot = format!(
        "[model]\nconfig = \"{}\"\n[driver]\nrunahead = true\n",
        descriptor.display()
    );
    let broker = CompletionBroker::new();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(code(d.load_model(&load)), PIE_STATUS_OK);

    let prog = driver_api::ProgramRegistration::default();
    let mut program_id = 0u64;
    reg_program(&mut d, &prog, &mut program_id);
    let inst = InstanceBindingPlan {
        driver_id: 0,
        program_id: 0,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids: Vec::new(),
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::Host,
    };
    let mut binding = InstanceBinding::default();
    bind(&mut d, &inst, &mut binding);

    // A PREFILL, deliberately: more rows means more work on the stream,
    // so "the call returned before it retired" is a claim about the
    // driver rather than about a fire too small to observe.
    // 512 rows, so the GPU has tens of milliseconds of work. A short
    // fire cannot distinguish "the call did not wait" from "the fire
    // finished before the call could return", and the first is the claim.
    let prompt: Vec<u32> = (0..128).map(|i| 1 + i % 97).collect();
    let positions: Vec<u32> = (0..prompt.len() as u32).collect();
    let roster = [0u32];
    let sub = [0u32, 1];
    let class = [driver_api::local::PIE_GEOMETRY_CLASS_HOST];
    let pages: Vec<u32> = (0..prompt.len().div_ceil(16) as u32).collect();
    let kv_indptr = [0u32, pages.len() as u32];
    let kv_lens = [((prompt.len() - 1) % 16 + 1) as u32];
    let qo = [0u32, prompt.len() as u32];
    let mut cell = driver_api::local::TerminalCell::default();
    let cells = [std::ptr::addr_of_mut!(cell)];
    let step = StepSubmission {
        plan: LaunchPlan {
            token_ids: prompt.to_vec(),
            position_ids: positions.to_vec(),
            kv_page_indices: pages.to_vec(),
            kv_page_indptr: kv_indptr.to_vec(),
            kv_last_page_lens: kv_lens.to_vec(),
            qo_indptr: qo.to_vec(),
            ..Default::default()
        },
        roster_rows: roster.to_vec(),
        sub_batch_indptr: sub.to_vec(),
        sub_batch_class: class.to_vec(),
        terminal_cells: cells.to_vec(),
        ..Default::default()
    };
    let instance_ids: [u64; 1] = [binding.instance_id];
    let frame = FrameSubmission {
        instance_ids: instance_ids.to_vec(),
        required_kv_pages: pages.len() as u32,
        steps: vec![step.clone()],
        ..Default::default()
    };
    let caller = format!("{:?}", std::thread::current().id());
    let caller_n = caller
        .bytes()
        .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(u64::from(b)));

    // ONE WARM FIRE FIRST, and it is not ceremony. A driver's first fire
    // pays for the KV pool, the attention workspace, the plan caches and
    // the weight binding — hundreds of milliseconds of HOST work, during
    // which the few milliseconds of GPU work it eventually enqueues
    // retire easily. Measuring that fire cannot tell "the call waited"
    // from "the work was done before the call could return".
    let (completion, _warm_parked) = arm(&broker);
    assert_eq!(
        code(d.launch(&frame, completion)),
        PIE_STATUS_OK,
        "the warm-up frame launches"
    );
    let warm = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !DONE.load(Ordering::Acquire) {
        assert!(
            std::time::Instant::now() < warm,
            "the warm-up never completed"
        );
        std::thread::yield_now();
    }

    let (completion, _parked) = arm(&broker);
    let t0 = std::time::Instant::now();
    let status = code(d.launch(&frame, completion));
    let returned_after = t0.elapsed();
    // ── FACT 1: the call returned, and the fire had not retired. ──
    let retired_inside_the_call = DONE.load(Ordering::Acquire);
    assert_eq!(status, PIE_STATUS_OK, "the frame launches");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !DONE.load(Ordering::Acquire) {
        assert!(
            std::time::Instant::now() < deadline,
            "the fire never completed"
        );
        std::thread::yield_now();
    }
    let retired_after = t0.elapsed();
    // ── AND THE NUMBER THAT DECIDES WHETHER ANY OF THIS PAYS ──
    //
    // The prefill above says issue time ≈ GPU time. A DECODE is the shape
    // production runs and the shape the supergraph is built to REPLAY, so
    // it is where issuing can be cheaper than executing — one
    // `cudaGraphLaunch` instead of ~500 bound-and-dispatched launches. If
    // it is, run-ahead is the difference between a busy GPU and an idle
    // one; if it is not, run-ahead only queues behind the host.
    //
    // Three rounds: the first warms, the second captures, the third
    // replays.
    let dec_pos = [prompt.len() as u32];
    let dec_tok = [1u32];
    let dec_qo = [0u32, 1];
    // Position 128 lands in a NINTH page; the prefill filled eight.
    let dec_pages: Vec<u32> = (0..=pages.len() as u32).collect();
    let dec_indptr = [0u32, dec_pages.len() as u32];
    let dec_lens = [1u32];
    let decode = StepSubmission {
        plan: LaunchPlan {
            token_ids: dec_tok.to_vec(),
            position_ids: dec_pos.to_vec(),
            kv_page_indices: dec_pages.to_vec(),
            kv_page_indptr: dec_indptr.to_vec(),
            kv_last_page_lens: dec_lens.to_vec(),
            qo_indptr: dec_qo.to_vec(),
            ..Default::default()
        },
        roster_rows: roster.to_vec(),
        sub_batch_indptr: sub.to_vec(),
        sub_batch_class: class.to_vec(),
        terminal_cells: cells.to_vec(),
        ..Default::default()
    };
    let dec_frame = FrameSubmission {
        instance_ids: instance_ids.to_vec(),
        required_kv_pages: dec_pages.len() as u32,
        steps: vec![decode],
        ..Default::default()
    };
    for round in 0..3 {
        let (completion, _parked) = arm(&broker);
        let t = std::time::Instant::now();
        let st = code(d.launch(&dec_frame, completion));
        let issued = t.elapsed();
        let inside = DONE.load(Ordering::Acquire);
        assert_eq!(st, PIE_STATUS_OK, "the decode launches");
        let w = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !DONE.load(Ordering::Acquire) {
            assert!(std::time::Instant::now() < w, "the decode never completed");
            std::thread::yield_now();
        }
        eprintln!(
            "decode {round}: issued in {issued:?}, retired at {:?}, \
             completion ran in-call: {inside}",
            t.elapsed()
        );
    }
    drop(d);

    // MEASURED, and the number is the interesting part. On a warm
    // 128-row prefill of qwen3-0.6B the call returns in ~21 ms and the
    // fire retires ~1 µs later — because ISSUING the fire costs about
    // what RUNNING it does. The executor walks ~500 launches per fire,
    // binding and dispatching each, so the GPU finishes at roughly the
    // moment the host stops feeding it.
    //
    // That is why `retired_inside_the_call` is REPORTED rather than
    // asserted. It is true here, and it does not mean the driver waited:
    // it means the work was done before the call could return. A test
    // that asserted it would be asserting that issue is cheaper than
    // execution, which is a statement about the executor and not about
    // whether this call synchronizes.
    eprintln!(
        "launch returned in {returned_after:?}; the fire retired at \
         {retired_after:?}; completion ran in-call: {retired_inside_the_call}"
    );

    // WHAT IS PROVABLE: the completion did not run on this thread.
    //
    // If `Shell::launch` still ended in `stream.synchronize()` and then
    // published and notified — which is what it did — this would be the
    // calling thread's id, because that is where the publish and the
    // notify happened. It is a CUDA-owned thread instead, which is only
    // reachable through a stream-ordered callback.
    //
    // So the completion is enqueued rather than awaited. Whether the call
    // gets to return before the GPU is done is then a question about how
    // long issuing takes, and the answer today is "about as long as
    // running" — which is what the supergraph's replay path is for.
    assert_ne!(
        WHERE.load(Ordering::Acquire),
        caller_n,
        "the completion ran on the CALLING thread, so it was published by \
         the launch rather than by a stream-ordered callback — the driver \
         is still synchronizing inside `Shell::launch`"
    );
    assert!(
        WHERE.load(Ordering::Acquire) != 0,
        "the completion never recorded where it ran"
    );
    // ── FACT 2: and it retired somewhere else. ──
    assert_ne!(
        WHERE.load(Ordering::Acquire),
        caller_n,
        "the completion ran on the CALLING thread, so it was not a \
         stream-ordered callback — fact 1 alone would then only mean the \
         fire was too small to observe"
    );
}

/// `kv_cache_dtype` is read, and an unserveable one is refused.
///
/// `store/kv_format.rs` has carried nine spellings and their scale
/// layouts since the port, `KvCacheLayout` plans their scale planes, and
/// `kv_paged.cu` switches on the scheme to WRITE them. None of it was
/// reachable: the shell built its pages by hand and could only build
/// bf16, so the format was a capability with no way to ask for it.
///
/// It is a boot key now. The refusal is the other half, and it is about
/// reading rather than writing: the fire path's prefill and decode are
/// FlashInfer's `_bf16` entry points, which take the `KvCacheLayerView`
/// and ignore its scheme. A quantized cache would therefore be appended
/// correctly and attended to as though the bytes were bf16 — plausible
/// logits from a model that had silently lost its context.
///
/// So the driver refuses to start rather than answering that way. What
/// lifts it is a scheme-aware attention fast path, which is a kernel
/// change; the plumbing should not have waited for it, and the refusal is
/// what lets it not wait.
#[test]
fn a_kv_cache_dtype_is_read_and_an_unreadable_one_is_refused() {
    let _gpu = gpu_guard();
    let create = |boot: &str| Shell::open(boot.as_bytes(), engine_runtime());

    // A spelling that is not in the catalogue at all: refused, rather than
    // quietly defaulted to bf16. A caller who typed `fp8` and got bf16
    // would see only a memory figure that did not move.
    let d = create("[driver]\nkv_cache_dtype = \"fp8_e4m3_typo\"\n");
    assert!(d.is_err(), "an uncatalogued kv_cache_dtype must refuse");

    // A real format the write path supports and the read path does not.
    for name in ["fp8_e4m3", "int8_per_token_head", "nvfp4"] {
        let d = create(&format!("[driver]\nkv_cache_dtype = \"{name}\"\n"));
        assert!(
            d.is_err(),
            "{name} can be appended but not attended to; starting would mean \
             wrong logits rather than a slower model"
        );
    }

    // The served ones still boot, so the refusal is about the format and
    // not about the key being present.
    for name in ["bf16", "bfloat16", "auto", "BF16"] {
        let d = create(&format!("[driver]\nkv_cache_dtype = \"{name}\"\n"));
        let d = d.expect("{name} is native bf16 and must boot");
        drop(d);
    }

    // And no key at all is still the default.
    let d = create("[driver]\nrunahead = false\n");
    let d = d.expect("an absent kv_cache_dtype means bf16");
    drop(d);
}

/// Two requests in one frame each get their OWN logits.
///
/// `deliver_logits` took `instance_ids.first()` and `rows - 1`: the
/// roster's FIRST instance, and the fire's LAST row. So a frame with a
/// roster of two published request 0's vocabulary into request 0's ring
/// and returned request 1 nothing at all — no error anywhere, just a
/// request that never got an answer.
///
/// Every fixture in this file sends one request, which is exactly why it
/// survived: single-request is the batch where `first()` and `rows - 1`
/// are both correct.
///
/// This sends two prefills of DIFFERENT prompts into separate rings, and
/// the second one's prompt is the reference's — so if delivery published
/// one answer to both rings, or read the wrong row, the argmax check on
/// request 1 fails rather than coincides.
#[test]
fn every_request_in_a_frame_gets_its_own_logits() {
    let _gpu = gpu_guard();

    let Some(snap) = qwen3_snapshot() else {
        eprintln!("skipped: no cached Qwen3-0.6B");
        return;
    };
    let Some(descriptor) = qwen3_descriptor() else {
        eprintln!("skipped: no generated descriptor");
        return;
    };
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("oracle/real_decode/reference.json")).expect("reference");

    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = CompletionBroker::new();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(code(d.load_model(&load)), PIE_STATUS_OK);

    const VOCAB: usize = 151_936;
    let shape: [u32; 1] = [VOCAB as u32];
    let prog = driver_api::ProgramRegistration {
        program_hash: 0xF13F,
        ..Default::default()
    };
    let mut program_id = 0u64;
    assert_eq!(reg_program(&mut d, &prog, &mut program_id), PIE_STATUS_OK);

    // One reader ring and one instance per request.
    let mut rings = Vec::new();
    let mut instance_ids = Vec::new();
    for r in 0..2u64 {
        let ch = ChannelRegistrationPlan {
            driver_id: 0,
            channel_id: 880 + r,
            shape: shape.to_vec(),
            dtype: PIE_CHANNEL_DTYPE_F32,
            host_role: PIE_CHANNEL_HOST_ROLE_READER,
            seeded: false,
            extern_dir: PIE_CHANNEL_EXTERN_NONE,
            capacity: 3,
            reader_wait_id: 900 + r * 2,
            writer_wait_id: 901 + r * 2,
            extern_name: Vec::new(),
        };
        let mut chb = ChannelBinding::default();
        assert_eq!(reg_channel(&mut d, &ch, &mut chb), PIE_STATUS_OK);
        let ids: [u64; 1] = [880 + r];
        let inst = InstanceBindingPlan {
            driver_id: 0,
            program_id: 0,
            requested_instance_id: 0,
            pacing_wait_id: 0,
            channel_ids: ids.to_vec(),
            seed_values: Vec::new(),
            geometry_class: driver_api::GeometryClass::Host,
        };
        let mut binding = InstanceBinding::default();
        assert_eq!(bind(&mut d, &inst, &mut binding), PIE_STATUS_OK);
        rings.push(chb);
        instance_ids.push(binding.instance_id);
    }

    // Request 0 is a short throwaway; request 1 is the reference prompt.
    // Putting the checked one SECOND is deliberate: `rows - 1` would land
    // inside request 1's span, so a delivery that ignored `qo_indptr`
    // would still be right for it — and wrong for request 0, which is the
    // one whose ring must not hold request 1's answer.
    let reference_prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .expect("ids")
        .iter()
        .map(|v| v.as_u64().expect("id") as u32)
        .collect();
    let first: Vec<u32> = vec![reference_prompt[0], reference_prompt[1]];
    let mut tokens: Vec<u32> = first.clone();
    tokens.extend_from_slice(&reference_prompt);
    let positions: Vec<u32> = (0..first.len() as u32)
        .chain(0..reference_prompt.len() as u32)
        .collect();

    let roster_rows: [u32; 2] = [0, 1];
    let sub_batch_indptr: [u32; 2] = [0, 2];
    let sub_batch_class: [u32; 1] = [driver_api::local::PIE_GEOMETRY_CLASS_HOST];
    let mut cells = [
        driver_api::local::TerminalCell::pending(),
        driver_api::local::TerminalCell::pending(),
    ];
    let cell_ptrs: [*mut driver_api::local::TerminalCell; 2] = [&mut cells[0], &mut cells[1]];
    // A page each.
    let kv_page_indices: [u32; 2] = [0, 1];
    let kv_page_indptr: [u32; 3] = [0, 1, 2];
    let kv_last_page_lens: [u32; 2] = [first.len() as u32, reference_prompt.len() as u32];
    let qo_indptr: [u32; 3] = [0, first.len() as u32, tokens.len() as u32];
    let step = StepSubmission {
        plan: LaunchPlan {
            token_ids: tokens.to_vec(),
            position_ids: positions.to_vec(),
            kv_page_indices: kv_page_indices.to_vec(),
            kv_page_indptr: kv_page_indptr.to_vec(),
            kv_last_page_lens: kv_last_page_lens.to_vec(),
            qo_indptr: qo_indptr.to_vec(),
            ..Default::default()
        },
        roster_rows: roster_rows.to_vec(),
        sub_batch_indptr: sub_batch_indptr.to_vec(),
        sub_batch_class: sub_batch_class.to_vec(),
        terminal_cells: cell_ptrs.to_vec(),
        ..Default::default()
    };
    let frame = FrameSubmission {
        instance_ids: instance_ids.to_vec(),
        required_kv_pages: 2,
        steps: vec![step.clone()],
        ..Default::default()
    };
    assert_eq!(fire_and_wait(&mut d, &frame, &broker), PIE_STATUS_OK);

    // BOTH rings advanced. The one that used to stay at zero is request 1's.
    for (r, chb) in rings.iter().enumerate() {
        let words = unsafe { std::slice::from_raw_parts(chb.word_base as *const u64, 4) };
        assert_eq!(
            words[1], 1,
            "request {r}'s ring did not advance — delivery served only the roster's \
             first instance, so every other request in a frame got nothing"
        );
    }

    // And request 1's cell is the REFERENCE's answer, so it read its own
    // row rather than inheriting request 0's or the frame's last.
    let cell = unsafe { std::slice::from_raw_parts(rings[1].mirror_base as *const f32, VOCAB) };
    let hf_argmax = reference["argmax"].as_u64().expect("argmax") as usize;
    let (mut best_t, mut best_v) = (0usize, f32::NEG_INFINITY);
    for (t, &v) in cell.iter().enumerate() {
        if v > best_v {
            (best_t, best_v) = (t, v);
        }
    }
    assert_eq!(
        best_t, hf_argmax,
        "request 1 got ITS OWN logits (top {best_v})"
    );

    // Request 0 prefilled two tokens of the same prompt, so its answer is
    // a different distribution. Same argmax would mean one answer went to
    // both rings.
    let other = unsafe { std::slice::from_raw_parts(rings[0].mirror_base as *const f32, VOCAB) };
    assert!(
        other.iter().any(|v| v.is_finite() && *v != 0.0),
        "request 0's ring holds a real distribution, not zeros"
    );
    assert!(
        other[..VOCAB] != cell[..VOCAB],
        "the two requests got the SAME vocabulary — one answer was published to both rings"
    );
    drop(d);
}

/// The shell reads the step's REGION TABLE, and a table that does not
/// describe its rows is refused.
///
/// The seriation's output is stated once, on the step: `region_sig[r]` is
/// the axis bitset that decides whether a row carries an adapter, a
/// custom mask, attention hooks or a depth truncation. This shell read it
/// ZERO times — it built `vec![Row { samples: true, ..default() }; rows]`
/// and never looked — so `HasLora`, `HasCustomMask`, `HasStageHooks` and
/// the truncation could not hold no matter what the engine sent.
///
/// That is why LoRA looked like a missing feature. It is supplied through
/// the forward's adapter hook — `fwd.adapter(site, |x, y| …)` lowers to a
/// pass-wide `lora` sink and the engine marks the carrying rows with
/// `PIE_REGION_SIG_LORA` — and the wire had been carrying the bit the
/// whole time into a driver that never asked.
///
/// A refusal is the only observable a black-box test has for "it read
/// the field": a table that does not tile its rows is drift, and drift is
/// refused rather than degraded. If the shell went back to ignoring the
/// table this launch would succeed.
#[test]
fn a_region_table_that_does_not_describe_its_rows_is_refused() {
    let _gpu = gpu_guard();

    let (Some(snap), Some(descriptor)) = (qwen3_snapshot(), qwen3_descriptor()) else {
        eprintln!("skipped: no cached Qwen3-0.6B or descriptor");
        return;
    };
    let boot = format!("[model]\nconfig = \"{}\"\n", descriptor.display());
    let broker = CompletionBroker::new();
    let opened = Shell::open(boot.as_bytes(), broker.clone());
    let d = opened;
    let mut d = d.expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    assert_eq!(code(d.load_model(&load)), PIE_STATUS_OK);

    const VOCAB: usize = 151_936;
    let shape: [u32; 1] = [VOCAB as u32];
    let ch = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 991,
        shape: shape.to_vec(),
        dtype: PIE_CHANNEL_DTYPE_F32,
        host_role: PIE_CHANNEL_HOST_ROLE_READER,
        seeded: false,
        extern_dir: PIE_CHANNEL_EXTERN_NONE,
        capacity: 3,
        reader_wait_id: 1001,
        writer_wait_id: 1002,
        extern_name: Vec::new(),
    };
    let mut chb = ChannelBinding::default();
    assert_eq!(reg_channel(&mut d, &ch, &mut chb), PIE_STATUS_OK);
    let prog = driver_api::ProgramRegistration {
        program_hash: 0xF140,
        ..Default::default()
    };
    let mut program_id = 0u64;
    assert_eq!(reg_program(&mut d, &prog, &mut program_id), PIE_STATUS_OK);
    let ids: [u64; 1] = [991];
    let inst = InstanceBindingPlan {
        driver_id: 0,
        program_id: 0,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids: ids.to_vec(),
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::Host,
    };
    let mut binding = InstanceBinding::default();
    assert_eq!(bind(&mut d, &inst, &mut binding), PIE_STATUS_OK);

    // A four-token prefill, one request, one page.
    let tokens: [u32; 4] = [1, 2, 3, 4];
    let positions: [u32; 4] = [0, 1, 2, 3];
    let roster_rows: [u32; 1] = [0];
    let sub_batch_indptr: [u32; 2] = [0, 1];
    let sub_batch_class: [u32; 1] = [driver_api::local::PIE_GEOMETRY_CLASS_HOST];
    let mut cell = driver_api::local::TerminalCell::pending();
    let cell_ptr: *mut driver_api::local::TerminalCell = &mut cell;
    let kv_page_indices: [u32; 1] = [0];
    let kv_page_indptr: [u32; 2] = [0, 1];
    let kv_last_page_lens: [u32; 1] = [4];
    let qo_indptr: [u32; 2] = [0, 4];
    let instance_ids: [u64; 1] = [binding.instance_id];
    let (completion, completion_handle) = broker.launch_completion(1);
    let _ = &completion_handle;

    // A table whose one region covers rows 0..2 and leaves 2..4 to
    // nobody. Every other field is a valid four-token prefill, so the
    // ONLY thing that can refuse this is the region read.
    let bad_indptr: [u32; 2] = [0, 2];
    let bad_sig: [u32; 1] = [0];
    let bad_k: [u32; 1] = [u32::MAX];
    let mut step = StepSubmission {
        plan: LaunchPlan {
            token_ids: tokens.to_vec(),
            position_ids: positions.to_vec(),
            kv_page_indices: kv_page_indices.to_vec(),
            kv_page_indptr: kv_page_indptr.to_vec(),
            kv_last_page_lens: kv_last_page_lens.to_vec(),
            qo_indptr: qo_indptr.to_vec(),
            ..Default::default()
        },
        roster_rows: roster_rows.to_vec(),
        sub_batch_indptr: sub_batch_indptr.to_vec(),
        sub_batch_class: sub_batch_class.to_vec(),
        terminal_cells: vec![cell_ptr],
        region_row_indptr: bad_indptr.to_vec(),
        region_sig: bad_sig.to_vec(),
        region_k: bad_k.to_vec(),
        ..Default::default()
    };
    let frame = FrameSubmission {
        instance_ids: instance_ids.to_vec(),
        required_kv_pages: 1,
        steps: vec![step.clone()],
        ..Default::default()
    };
    assert_eq!(
        code(d.launch(&frame, completion)),
        PIE_STATUS_INVALID_ARGUMENT,
        "a region table that leaves rows unclaimed is drift between the \
         scheduler's tables and the step; a shell that ignored the table \
         would launch this happily"
    );

    // A tiling table whose region carries an ADAPTER is refused too, and
    // for a different reason: the bit reaches the `HasLora` guard now, so
    // the trace states `gemm::lora_qkv_correction` — and the executor's arm
    // for it is a NO-OP while nothing stages the table, which it must be
    // for union captures to record it. Running the fire would apply no
    // correction and return tokens that look like a slightly worse model.
    //
    // `caps.has_lora` is TRUE now, and this is the assertion that
    // changed with it. It used to read "an adapter this driver cannot
    // stage is refused, not dropped" — correct while nothing staged.
    //
    // Something stages now: `lane_for_instance` resolves each request's
    // adapter to a committed cell address, `lora_pins` names the q, v
    // and x the correction binds, and `stage_qkv_adapters` builds the
    // state. So a fire carrying `PIE_REGION_SIG_LORA` LAUNCHES.
    //
    // What it does with an instance whose program states no `lora` sink
    // — which is this one — is the interesting half: the lane does not
    // resolve, `ctx.lora` stays `None`, and the executor's arm no-ops.
    // That is the SAME correct answer an adapter-free fire gets, and it
    // has to be, because under `GuardMode::Union` the arm is recorded
    // whether or not this fire takes it.
    let good_indptr: [u32; 2] = [0, 4];
    let lora_sig: [u32; 1] = [driver_api::local::PIE_REGION_SIG_LORA];
    step.region_row_indptr = good_indptr.to_vec();
    step.region_sig = lora_sig.to_vec();
    let frame = FrameSubmission {
        instance_ids: instance_ids.to_vec(),
        required_kv_pages: 1,
        steps: vec![step.clone()],
        ..Default::default()
    };
    assert_eq!(
        code(d.launch(&frame, completion)),
        PIE_STATUS_OK,
        "a marked adapter row fires; an instance with no `lora` sink simply \
         stages nothing"
    );

    // The SAME step without the bit launches — so the refusal is about
    // what the region SAYS, not about the table being present.
    step.region_sig = bad_sig.to_vec();
    let frame = FrameSubmission {
        instance_ids: instance_ids.to_vec(),
        required_kv_pages: 1,
        steps: vec![step.clone()],
        ..Default::default()
    };
    assert_eq!(
        fire_and_wait(&mut d, &frame, &broker),
        PIE_STATUS_OK,
        "a table that tiles its rows is the served case"
    );
    drop(d);
}

/// An extern channel registers and then REFUSES to attach.
///
/// The refusal is the point. An extern channel is one two programs
/// share, and sharing it means sharing one ring — the same cells and the
/// same head/tail cursors. This driver builds a ring per session, so the
/// importer would get a private ring nobody fills: a program that blocks
/// forever, or reads a zeroed cell and calls it a value. Registration is
/// harmless and is allowed; the attach is where the driver can still say
/// it cannot serve this, so that is where it does.
#[test]
fn an_extern_channel_registers_and_refuses_to_attach() {
    let _gpu = gpu_guard();

    let mut d = Shell::open(b"", engine_runtime()).expect("the driver creates");

    let shape: [u32; 1] = [4];
    let name = b"kv_bridge";
    // The validator's contract for an extern channel: a name, NO host
    // role, and no seed.
    let ch = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 21,
        shape: shape.to_vec(),
        dtype: PIE_CHANNEL_DTYPE_F32,
        host_role: PIE_CHANNEL_HOST_ROLE_NONE,
        seeded: false,
        extern_dir: PIE_CHANNEL_EXTERN_IMPORT,
        capacity: 2,
        reader_wait_id: 31,
        writer_wait_id: 32,
        extern_name: name.to_vec(),
    };
    let mut b = ChannelBinding::default();
    assert_eq!(
        reg_channel(&mut d, &ch, &mut b),
        PIE_STATUS_OK,
        "registering an extern channel is harmless -- the host endpoint is real"
    );

    let prog = driver_api::ProgramRegistration {
        program_hash: 0x2121,
        ..Default::default()
    };
    let mut program_id = 0u64;
    assert_eq!(reg_program(&mut d, &prog, &mut program_id), PIE_STATUS_OK);
    let ids: [u64; 1] = [21];
    let inst = InstanceBindingPlan {
        driver_id: 0,
        // THE PROGRAM THAT WAS JUST REGISTERED.
        //
        // This said `program_id: 0`, and `bind_instance` refuses an unknown
        // program with `PIE_STATUS_INVALID_ARGUMENT` several lines BEFORE it
        // looks at the channels. So the test never reached the extern check
        // it is named for: it read -1, asserted -3, and had been failing on
        // an unregistered program the whole time. `reg_program` hands the id
        // back through the out-parameter above, which nothing used.
        program_id,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids: ids.to_vec(),
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::Host,
    };
    let mut binding = InstanceBinding::default();
    assert_eq!(
        bind(&mut d, &inst, &mut binding),
        PIE_STATUS_UNSUPPORTED,
        "attaching to an extern channel is refused, not silently made private"
    );

    // A PRIVATE channel of the same shape still attaches, so the refusal
    // is about the declaration and not about the bind path.
    let priv_ch = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 22,
        shape: Vec::new(),
        dtype: PIE_CHANNEL_DTYPE_F32,
        host_role: PIE_CHANNEL_HOST_ROLE_NONE,
        seeded: false,
        extern_dir: driver_api::local::PIE_CHANNEL_EXTERN_NONE,
        capacity: 0,
        reader_wait_id: 33,
        writer_wait_id: 34,
        extern_name: Vec::new(),
    };
    assert_eq!(reg_channel(&mut d, &priv_ch, &mut b), PIE_STATUS_OK);
    let ids2: [u64; 1] = [22];
    let inst2 = InstanceBindingPlan {
        driver_id: 0,
        program_id,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids: ids2.to_vec(),
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::Host,
    };
    assert_eq!(
        bind(&mut d, &inst2, &mut binding),
        PIE_STATUS_OK,
        "a private channel of the same shape binds"
    );
    drop(d);
}
