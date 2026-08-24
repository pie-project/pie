//! A real PTIR program, registered through `Shell` and fired by it.
//!
//! `driver-cuda`'s own tests cover every piece of this and cannot
//! cover the whole: `ptir::Session` runs a program end to end on the GPU,
//! including a value crossing both channel planes, but the corpus in
//! `tests/serve.rs` registers `ProgramRegistration { program_hash, ..Default }`
//! — a descriptor with no bytecode — so `plan.executable` is false,
//! `state.programs` records the hash and nothing more, and `register_program`
//! never adopts a launch package for any of them. The shell shipped untested
//! at that seam and this is the test it owed.
//!
//! # Where this used to live, and why it does not any more
//!
//! It used to live in `engine`, because building a registerable program
//! needs the producer half — `tensor_compiler`'s bind → compile → emit
//! chain — and a dev-dependency from `driver-cuda` on `engine` would have
//! been a cycle (`engine` depends on `driver-cuda` to run one). That
//! reasoning does not apply to `tensor_compiler` itself: it is the CUDA
//! shell's *emitter's* own producer, not the engine's, and
//! `crates/driver-cuda/tests/gpu_ptir_emitted.rs` and `gpu_ptir_fire.rs`
//! already depend on it as a dev-dependency to build exactly this kind of
//! fixture. So the cycle this file was written to avoid was never a cycle
//! for `driver-cuda` — only for `engine` — and the test belongs beside the
//! driver whose seam it tests, alongside its two siblings.
//!
//! # What it claims
//!
//! That a program registered through [`driver_cuda::serve::Shell::register_program`]
//! reaches the fire, reads a value the engine published into a channel, and
//! publishes its answer back — so the caller gets the PROGRAM's output and
//! not raw logits. That is the whole of what an unread `ptir_programs` used
//! to cost.

#![cfg(all(feature = "_cuda", feature = "abi"))]

use driver_api::completion::CompletionBroker;
use driver_api::local::{
    ChannelBinding, InstanceBinding, PIE_CHANNEL_DTYPE_F32, PIE_CHANNEL_DTYPE_I32,
    PIE_CHANNEL_EXTERN_NONE, PIE_CHANNEL_HOST_ROLE_READER, PIE_CHANNEL_HOST_ROLE_WRITER,
    PIE_RS_FLAG_RESET, TerminalCell,
};
use driver_api::{
    ChannelRegistrationPlan, FrameSubmission, GeometryClass, InstanceBindingPlan, LaunchPlan,
    ModelComponent, ModelLoadDesc, Mxfp4MoeRequest, ProgramRegistration, StepSubmission,
};
use driver_cuda::serve::Shell;
use tensor_compiler::codegen::program::{Backend, emit_program};
use tensor_compiler::plan::compile_bound;
use tensor_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use tensor_ir::op::Op;
use tensor_ir::registry::{ModelProfile, Stage};
use tensor_ir::types::{DType as IrDType, Shape};
use tensor_ir::validate::bind;

mod common;
use common::gpu_guard;

/// Lanes the program reduces over.
const LANES: u32 = 8;

/// The cached snapshot both GPU tests here load a model out of — the same
/// catalog row `tests/baker_serve.rs` uses.
const CACHE_DIR: &str = "models--Qwen--Qwen3.5-0.8B-Base";

/// The cached checkpoint this test loads a model from, if this box has one.
///
/// A MODEL IS REQUIRED and that is not incidental: `register_program`
/// compiles only `if plan.executable && state.model.is_some()`, and
/// `run_program` is called from inside the model fire. So a program with
/// no model behind it is registered and never compiled — which is what
/// the first draft of this test measured, silently, in 0.00s.
///
/// IT USED TO NAME `models--Qwen--Qwen3-0.6B`, AND THAT IS WHY BOTH TESTS
/// WERE RED. R3 replaced the guessing loader with `model::catalog()`, and
/// the catalog does not ship a Qwen3-0.6B row — so on a box that HAS that
/// snapshot the skip guard passed, `load_model` reached
/// `baker::identify` and answered "this checkpoint matches no SKU this
/// build ships". The skip and the load disagreed about what a usable
/// checkpoint is. [`CACHE_DIR`] is the row `baker_serve` loads, so the
/// two GPU tests that need a model now ask for the same one.
fn qwen3_snapshot() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(CACHE_DIR)
        .join("snapshots");
    // A SHARD INDEX COUNTS, which is `tests/baker_serve.rs`'s predicate and
    // not what this used to ask. `Qwen3.5-0.8B-Base` ships as
    // `model.safetensors-00001-of-00001.safetensors` beside an index, so a
    // finder that only accepted a single `model.safetensors` skipped a
    // snapshot that is sitting right there — the second half of the same
    // disagreement the doc above records.
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(Result::ok)
        .find_map(|e| {
            let p = e.path();
            (p.join("model.safetensors").is_file()
                || p.join("model.safetensors.index.json").is_file())
            .then_some(p)
        })
}

/// The one-stage epilogue this test registers: read the channel, take an
/// argmax, publish the index.
///
/// An argmax because the observable is an INDEX. `.wiki/driver/progress-metal.md`'s
/// tolerance contract admits one ulp on a magnitude and no slack at all
/// on an argmax, so a disagreement here cannot be rounding — and the
/// seed carries a TIE, which is the case where "the first maximum" is a
/// decision rather than an accident.
fn argmax_program() -> TraceContainer {
    let chan = |shape: Shape, dtype: IrDType, host_role: HostRole, seeded: bool| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded,
    };
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            chan(Shape::vector(LANES), IrDType::F32, HostRole::Writer, false),
            chan(Shape::SCALAR, IrDType::I32, HostRole::Reader, false),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: Vec::new(),
    }
}

/// The emitter's kernel table in the shape `Shell::register_program` takes.
///
/// Three field-identical `EmittedKernel` structs exist in this workspace —
/// `tensor_compiler::codegen::program`'s (what the emitter returns),
/// `driver_api::plan`'s (what a driver adopts to), and
/// `driver_api::local::PieEmittedKernel` (the retired C ABI's) —
/// and `gpu_ptir_emitted.rs` carries the same conversion for the same
/// reason: written out field by field rather than with `..` so that an ABI
/// field added on one side and not the other fails to compile here.
fn as_abi(
    kernels: &[tensor_compiler::codegen::program::EmittedKernel],
) -> Vec<driver_api::plan::EmittedKernel> {
    kernels
        .iter()
        .map(|kernel| driver_api::plan::EmittedKernel {
            kind: kernel.kind,
            stage_index: kernel.stage_index,
            region_index: kernel.region_index,
            entry_name: kernel.entry_name.clone(),
            source: kernel.source.clone(),
            error: kernel.error.clone(),
        })
        .collect()
}

/// Build the registration the engine would hand a driver for this
/// program: the adopted package, the emitted CUDA, and the version they
/// were built with.
fn registration(container: TraceContainer) -> ProgramRegistration {
    let mut profile = ModelProfile::dummy();
    profile.vocab = LANES;
    let bound = bind(container, profile).expect("the container binds");
    let stages = compile_bound(&bound);
    let package = tensor_compiler::codegen::launch::build(&bound, &stages);
    let emitted = emit_program(Backend::Cuda, &stages, &bound);
    ProgramRegistration {
        program_hash: 0x5E5_5107,
        emitted_kernels: as_abi(&emitted),
        emitter_version: Backend::Cuda.emitter_version(),
        launch: package,
        ..Default::default()
    }
}

/// The host plane of a registered channel, as the engine sees it.
///
/// `register_channel` hands back the mirror and word addresses and the
/// engine polls them directly; this reads and writes them the same way,
/// because the point of the test is that the driver and the engine agree
/// about that memory.
struct Endpoint {
    mirror: *mut u8,
    words: *mut u64,
    cell_bytes: usize,
    ring: u64,
}

impl Endpoint {
    fn of(binding: &ChannelBinding) -> Self {
        Self {
            mirror: binding.mirror_base as *mut u8,
            words: binding.word_base as *mut u64,
            cell_bytes: binding.cell_bytes as usize,
            ring: u64::from(binding.capacity) + 1,
        }
    }
    fn publish(&self, cell: &[u8]) {
        let tail = unsafe { self.words.add(1).read_volatile() };
        let slot = (tail % self.ring) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                cell.as_ptr(),
                self.mirror.add(slot * self.cell_bytes),
                cell.len(),
            );
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            self.words.add(1).write_volatile(tail + 1);
        }
    }
    fn take(&self) -> Option<Vec<u8>> {
        let (head, tail) = unsafe {
            (
                self.words.add(0).read_volatile(),
                self.words.add(1).read_volatile(),
            )
        };
        if head == tail {
            return None;
        }
        let slot = (head % self.ring) as usize;
        let mut cell = vec![0u8; self.cell_bytes];
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.mirror.add(slot * self.cell_bytes),
                cell.as_mut_ptr(),
                self.cell_bytes,
            );
            std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
            self.words.add(0).write_volatile(head + 1);
        }
        Some(cell)
    }
}

/// A registered program runs, and its answer comes back through the
/// reader channel.
///
/// The chain is every step the engine takes and nothing stubbed: author a
/// container, bind, compile, emit CUDA, `Shell::register_program`,
/// `Shell::register_channel` twice, `Shell::bind_instance`, publish a seed
/// into the writer's mirror, and read the argmax out of the reader's.
#[test]
fn a_registered_program_reads_a_channel_and_publishes_its_answer() {
    let _gpu = gpu_guard();
    let Some(snap) = qwen3_snapshot() else {
        eprintln!("skipping: no cached {CACHE_DIR}");
        return;
    };
    // The snapshot carries its own `config.json`, and that is all
    // `load_impl` reads out of a `[model] config` path today — see
    // `driver-cuda/src/serve/load.rs`'s account of what a `pie.model/1`
    // descriptor shrank to. Pointing straight at the checkpoint's own file
    // needs no separately generated fixture at all: this test runs against
    // the snapshot exactly as `huggingface-cli` leaves it, redundant tied
    // `lm_head.weight` and all.
    let boot = format!(
        "[model]\nconfig = \"{}\"\n",
        snap.join("config.json").display()
    );
    let broker = CompletionBroker::new();
    let mut shell = match Shell::open(boot.as_bytes(), broker.clone()) {
        Ok(shell) => shell,
        Err(status) => panic!("the driver creates: status {status}"),
    };

    // ── A MODEL FIRST. `register_program` compiles only when one is
    // loaded, and `run_program` is called from inside the model fire, so
    // a program registered against an empty driver is never compiled and
    // never runs. ──
    let load = ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: Mxfp4MoeRequest::Auto,
        component: ModelComponent::Full,
    };
    shell.load_model(&load).expect("the snapshot loads");

    // ── Register the program. ──
    let program = registration(argmax_program());
    let program_id = shell
        .register_program(&program)
        .expect("the program registers");

    // ── Register its two channels, in the order the program indexes
    // them: 0 is what it takes, 1 is what it puts. ──
    let mut bindings = Vec::new();
    for (index, (shape, dtype, host_role)) in [
        (
            vec![LANES],
            PIE_CHANNEL_DTYPE_F32,
            PIE_CHANNEL_HOST_ROLE_WRITER,
        ),
        (
            vec![1u32],
            PIE_CHANNEL_DTYPE_I32,
            PIE_CHANNEL_HOST_ROLE_READER,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let plan = ChannelRegistrationPlan {
            driver_id: 0,
            channel_id: 100 + index as u64,
            shape,
            dtype,
            host_role,
            capacity: 1,
            // Nonzero and distinct, which the shared validator requires:
            // a channel nothing can wait on is not one the runtime can
            // schedule against.
            reader_wait_id: 200 + index as u64 * 2,
            writer_wait_id: 201 + index as u64 * 2,
            seeded: false,
            extern_dir: PIE_CHANNEL_EXTERN_NONE,
            extern_name: Vec::new(),
        };
        let binding = shell
            .register_channel(&plan)
            .unwrap_or_else(|status| panic!("channel {index} registers: status {status}"));
        bindings.push(binding);
    }

    // ── Bind an instance over both, in that order. ──
    let channel_ids: Vec<u64> = bindings.iter().map(|b| b.channel_id).collect();
    let inst = InstanceBindingPlan {
        driver_id: 0,
        program_id,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids,
        seed_values: Vec::new(),
        geometry_class: GeometryClass::Host,
    };
    let instance: InstanceBinding = shell.bind_instance(&inst).expect("the instance binds");

    // ── The engine's side: publish the seed the program will take. ──
    let seed: [f32; LANES as usize] = [2.0, 7.0, 1.0, 7.0, 0.5, 7.0, -3.0, 6.0];
    let input = Endpoint::of(&bindings[0]);
    let output = Endpoint::of(&bindings[1]);
    let bytes: Vec<u8> = seed.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(
        input.cell_bytes,
        bytes.len(),
        "the wire cell is the seed's width"
    );
    input.publish(&bytes);
    assert!(output.take().is_none(), "nothing published before the fire");

    // ── One decode token, which is what carries the program. ──
    let mut cell = TerminalCell::pending();
    let cell_ptr: *mut TerminalCell = &mut cell;
    let step = StepSubmission {
        plan: LaunchPlan {
            token_ids: vec![7],
            position_ids: vec![0],
            kv_page_indices: vec![0],
            kv_page_indptr: vec![0, 1],
            kv_last_page_lens: vec![1],
            qo_indptr: vec![0, 1],
            // A HYBRID SKU WANTS A SLOT PER REQUEST, and the catalog row
            // this test loads is one: `fire::launch` refuses by name
            // ("hybrid fire without rs_slot_ids") before it allocates
            // anything. `RESET` because this is the request's first token
            // and it inherits no recurrent state. Same two lines
            // `serve::load::synthetic_fire` carries, for the same reason.
            rs_slot_ids: vec![0],
            rs_slot_flags: vec![PIE_RS_FLAG_RESET],
            ..Default::default()
        },
        roster_rows: vec![0],
        sub_batch_indptr: vec![0, 1],
        sub_batch_class: vec![0],
        terminal_cells: vec![cell_ptr],
        ..Default::default()
    };
    let frame = FrameSubmission {
        instance_ids: vec![instance.instance_id],
        required_kv_pages: 1,
        steps: vec![step],
        ..Default::default()
    };
    let (target, completion) = broker.launch_completion(1);
    shell.launch(&frame, target).expect("the frame launches");

    // Run-ahead means the call returns with the fire still queued.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        if let Some(settled) = completion.check() {
            settled.expect("the fire completed");
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the fire never completed"
        );
        std::thread::yield_now();
    }

    // ── THE CLAIM. The reader holds the PROGRAM's answer, and it is an
    // index rather than a vocabulary of logits. ──
    let published = output.take().expect(
        "the program published into the reader channel — if this is empty the fire \
         delivered raw logits and `run_program` did not run",
    );
    assert_eq!(published.len(), 4, "an i32 index, not a vocabulary");
    let got = i32::from_le_bytes(published[..4].try_into().expect("four bytes"));
    assert_eq!(got, 1, "argmax over the tie takes the FIRST maximum");

    drop(shell);
}

/// Two requests in one frame each get their OWN answer.
///
/// The shell fired `instance_ids.first()` and nothing else, and then let
/// that one publish suppress the raw-logits fallback for the whole frame.
/// So a two-request batch sampled request 0 and returned request 1
/// NOTHING: no sample, because its program never ran, and no logits,
/// because request 0's had. A batch of one could not see it, which is
/// exactly the batch the test above uses.
///
/// The two instances get DIFFERENT seeds with different argmaxes, so a
/// shell that fired one program and published its answer to both readers
/// would fail rather than coincide.
#[test]
fn every_request_in_a_frame_samples_its_own_row() {
    let _gpu = gpu_guard();
    let Some(snap) = qwen3_snapshot() else {
        eprintln!("skipping: no cached {CACHE_DIR}");
        return;
    };
    let boot = format!(
        "[model]\nconfig = \"{}\"\n",
        snap.join("config.json").display()
    );
    let broker = CompletionBroker::new();
    let mut shell = match Shell::open(boot.as_bytes(), broker.clone()) {
        Ok(shell) => shell,
        Err(status) => panic!("the driver creates: status {status}"),
    };

    let load = ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: Mxfp4MoeRequest::Auto,
        component: ModelComponent::Full,
    };
    shell.load_model(&load).expect("the snapshot loads");

    let program = registration(argmax_program());
    let program_id = shell
        .register_program(&program)
        .expect("the program registers");

    // Two instances of the SAME program, each over its own channel pair.
    let mut instances = Vec::new();
    let mut endpoints = Vec::new();
    for req in 0..2u64 {
        let mut bindings = Vec::new();
        for (index, (shape, dtype, host_role)) in [
            (
                vec![LANES],
                PIE_CHANNEL_DTYPE_F32,
                PIE_CHANNEL_HOST_ROLE_WRITER,
            ),
            (
                vec![1u32],
                PIE_CHANNEL_DTYPE_I32,
                PIE_CHANNEL_HOST_ROLE_READER,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let n = req * 2 + index as u64;
            let plan = ChannelRegistrationPlan {
                driver_id: 0,
                channel_id: 300 + n,
                shape,
                dtype,
                host_role,
                capacity: 1,
                // Distinct across BOTH instances: the validator refuses a
                // reused wait id, and two requests sharing one would be
                // the bug this test is about wearing a different hat.
                reader_wait_id: 400 + n * 2,
                writer_wait_id: 401 + n * 2,
                seeded: false,
                extern_dir: PIE_CHANNEL_EXTERN_NONE,
                extern_name: Vec::new(),
            };
            let binding = shell.register_channel(&plan).unwrap_or_else(|status| {
                panic!("request {req} channel {index} registers: status {status}")
            });
            bindings.push(binding);
        }
        let channel_ids: Vec<u64> = bindings.iter().map(|b| b.channel_id).collect();
        let inst = InstanceBindingPlan {
            driver_id: 0,
            program_id,
            requested_instance_id: 0,
            pacing_wait_id: 0,
            channel_ids,
            seed_values: Vec::new(),
            geometry_class: GeometryClass::Host,
        };
        let instance: InstanceBinding = shell
            .bind_instance(&inst)
            .unwrap_or_else(|status| panic!("instance {req} binds: status {status}"));
        endpoints.push((Endpoint::of(&bindings[0]), Endpoint::of(&bindings[1])));
        instances.push(instance.instance_id);
    }

    // DIFFERENT argmaxes, so publishing one answer to both readers fails.
    let seeds: [[f32; LANES as usize]; 2] = [
        [2.0, 7.0, 1.0, 3.0, 0.5, 4.0, -3.0, 6.0],
        [9.0, 1.0, 1.0, 3.0, 0.5, 4.0, -3.0, 6.0],
    ];
    let expected = [1i32, 0i32];
    for (req, (input, output)) in endpoints.iter().enumerate() {
        let bytes: Vec<u8> = seeds[req].iter().flat_map(|v| v.to_le_bytes()).collect();
        input.publish(&bytes);
        assert!(
            output.take().is_none(),
            "request {req} has nothing before the fire"
        );
    }

    // TWO decode tokens, one per request. `qo_indptr` is what says so,
    // and it is what the shell reads to find each request's logits row.
    let mut cells = [TerminalCell::pending(), TerminalCell::pending()];
    let (first, rest) = cells.split_at_mut(1);
    let cell_ptrs: Vec<*mut TerminalCell> = vec![&mut first[0], &mut rest[0]];
    let step = StepSubmission {
        plan: LaunchPlan {
            token_ids: vec![7, 11],
            position_ids: vec![0, 0],
            kv_page_indices: vec![0, 1],
            kv_page_indptr: vec![0, 1, 2],
            kv_last_page_lens: vec![1, 1],
            qo_indptr: vec![0, 1, 2],
            // One recurrent slot per request, `RESET` on both — see the
            // sibling test above.
            rs_slot_ids: vec![0, 1],
            rs_slot_flags: vec![PIE_RS_FLAG_RESET; 2],
            ..Default::default()
        },
        roster_rows: vec![0, 1],
        sub_batch_indptr: vec![0, 2],
        sub_batch_class: vec![0],
        terminal_cells: cell_ptrs,
        ..Default::default()
    };
    let frame = FrameSubmission {
        instance_ids: instances.to_vec(),
        required_kv_pages: 2,
        steps: vec![step],
        ..Default::default()
    };
    let (target, completion) = broker.launch_completion(1);
    shell
        .launch(&frame, target)
        .expect("the two-request frame launches");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        if let Some(settled) = completion.check() {
            settled.expect("the fire completed");
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the fire never completed"
        );
        std::thread::yield_now();
    }

    // THE CLAIM: BOTH readers hold an answer, and each holds its own.
    for (req, (_, output)) in endpoints.iter().enumerate() {
        let published = output.take().unwrap_or_else(|| {
            panic!(
                "request {req} published nothing — the shell fired only the roster's \
                 first instance, and then let that publish suppress the raw-logits \
                 fallback for the whole frame"
            )
        });
        assert_eq!(published.len(), 4, "request {req}: an i32 index");
        let got = i32::from_le_bytes(published[..4].try_into().expect("four bytes"));
        assert_eq!(got, expected[req], "request {req} sampled its OWN seed");
    }

    drop(shell);
}
