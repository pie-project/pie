//! A real PTIR program, registered through the ABI and fired by the shell.
//!
//! `driver-cuda-new`'s own tests cover every piece of this and cannot
//! cover the whole: `ptir::Session` runs a program end to end on the GPU,
//! including a value crossing both channel planes, but the ABI corpus in
//! that crate registers `PieProgramDesc { program_hash, ..Default }` — a
//! descriptor with no bytecode — so `plan.executable` is false,
//! `ptir_programs` stays empty, and `abi_shell::run_program` returns
//! early in every one of them. The adapter shipped untested and this is
//! the test it owed.
//!
//! It lives HERE rather than beside the driver because building a
//! registerable program needs the producer half — `tensor_compiler`'s
//! bind → compile → emit chain, and `engine`'s `ProgramDescBorrow` to
//! lower the result into the C records `register_program` reads. The
//! engine has both, and a dev-dependency the other way would be a cycle.
//!
//! # What it claims
//!
//! That a program registered through `pie_cuda_register_program` reaches
//! the fire, reads a value the engine published into a channel, and
//! publishes its answer back — so the caller gets the PROGRAM's output
//! and not raw logits. That is the whole of what `ptir_programs` having
//! no reader used to cost.

#![cfg(feature = "driver-cuda-new")]

use driver_api::local::{
    PIE_DRIVER_ABI_VERSION, PIE_STATUS_OK, PieBytes, PieChannelEndpointBinding, PieCompletion,
    PieDriverCaps, PieDriverCreateDesc, PieFrameDesc, PieInstanceBinding, PieInstanceDesc,
    PieModelLoadDesc, PieRuntimeCallbacks, PieStepDesc, PieTerminalCell,
    PieTerminalCellPtrSlice, PieU32Slice, PieU64Slice,
};
use driver_api::plan::{ChannelRegistrationPlan, ProgramRegistration};
use engine::driver::abi::{ChannelDescBorrow, ProgramDescBorrow};
use tensor_compiler::codegen::program::{Backend, emit_program};
use tensor_compiler::plan::compile_bound;
use tensor_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use tensor_ir::op::Op;
use tensor_ir::registry::{ModelProfile, Stage};
use tensor_ir::types::{DType as IrDType, Shape};
use tensor_ir::validate::bind;

/// Lanes the program reduces over.
const LANES: u32 = 8;

/// Bumped by the driver's completion callback, so the test can wait for
/// a fire that returns with work still on the stream.
static FIRED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The cached Qwen3-0.6B snapshot and its generated descriptor.
///
/// A MODEL IS REQUIRED and that is not incidental: `register_program`
/// compiles only `if plan.executable && state.model.is_some()`, and
/// `run_program` is called from inside the model fire. So a program with
/// no model behind it is registered and never compiled — which is what
/// the first draft of this test measured, silently, in 0.00s.
/// One driver at a time.
///
/// Every test here creates a driver, binds device 0 and loads a model,
/// and two of those at once aborts the process — the failure looks like a
/// broken change rather than a contended device, which is the worst way
/// for it to look. `driver-cuda-new`'s own suite has the same mutex for
/// the same reason.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}

fn qwen3_fixture() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let home = std::env::var("HOME").ok()?;
    let snaps = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    let snap = std::fs::read_dir(&snaps).ok()?.find_map(|e| {
        let p = e.ok()?.path();
        p.join("model.safetensors").is_file().then_some(p)
    })?;
    let descriptor = std::path::PathBuf::from(
        "/tmp/claude-0/-root--patissier-work-tart-alpha/\
         7460e4c3-f305-45df-9603-2298b0c0c60e/scratchpad",
    )
    .join("qwen3_descriptor.json");
    descriptor.is_file().then_some((snap, descriptor))
}

/// A driver, or `None` when this box has no CUDA device.
///
/// The same rule every GPU-touching test in this workspace follows: skip
/// rather than fail, because a machine without a card is a machine that
/// cannot answer the question rather than one that answers it wrong.
fn driver_or_skip(
    boot: &str,
    caps: &mut PieDriverCaps,
) -> Option<*mut driver_api::local::PieDriver> {
    unsafe extern "C" fn bump(_ctx: *mut std::ffi::c_void, _wait: u64, _epoch: u64) {
        FIRED.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }
    let desc = PieDriverCreateDesc {
        abi_version: PIE_DRIVER_ABI_VERSION,
        runtime: PieRuntimeCallbacks {
            abi_version: PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            ctx: std::ptr::null_mut(),
            notify: Some(bump),
        },
        config_bytes: PieBytes { ptr: boot.as_ptr(), len: boot.len() },
        ..Default::default()
    };
    let driver = driver_cuda_new::abi_shell::pie_cuda_create(&desc, caps);
    if driver.is_null() {
        eprintln!("[ptir-shell] no CUDA device; skipping");
        return None;
    }
    Some(driver)
}

/// The one-stage epilogue this test registers: read the channel, take an
/// argmax, publish the index.
///
/// An argmax because the observable is an INDEX. `PARITY-INTERP.md`'s
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
        emitted_kernels: emitted
            .iter()
            .map(|k| driver_api::plan::EmittedKernel {
                kind: k.kind,
                stage_index: k.stage_index,
                region_index: k.region_index,
                entry_name: k.entry_name.clone(),
                source: k.source.clone(),
                error: k.error.clone(),
            })
            .collect(),
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
    fn of(binding: &PieChannelEndpointBinding) -> Self {
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
        let (head, tail) =
            unsafe { (self.words.add(0).read_volatile(), self.words.add(1).read_volatile()) };
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
/// container, bind, compile, emit CUDA, lower to the C records,
/// `pie_cuda_register_program`, `pie_cuda_register_channel` twice,
/// `pie_cuda_bind_instance`, publish a seed into the writer's mirror, and
/// read the argmax out of the reader's.
#[test]
fn a_registered_program_reads_a_channel_and_publishes_its_answer() {
    let _gpu = gpu_guard();
    let Some((snap, descriptor)) = qwen3_fixture() else {
        eprintln!("[ptir-shell] no cached Qwen3-0.6B or descriptor; skipping");
        return;
    };
    // The descriptor rides in the boot TOML, which is how `[model]
    // descriptor` reaches a driver for an HF snapshot whose descriptor
    // does not live inside the checkpoint.
    let boot = format!("[model]\ndescriptor = \"{}\"\n", descriptor.display());
    let mut caps = PieDriverCaps::default();
    let Some(driver) = driver_or_skip(&boot, &mut caps) else {
        return;
    };

    // ── A MODEL FIRST. `register_program` compiles only when one is
    // loaded, and `run_program` is called from inside the model fire, so
    // a program registered against an empty driver is never compiled and
    // never runs. ──
    let snap_str = snap.to_string_lossy().into_owned();
    let load = PieModelLoadDesc {
        snapshot_dir: PieBytes { ptr: snap_str.as_ptr(), len: snap_str.len() },
        ..Default::default()
    };
    let mut load_caps = PieDriverCaps::default();
    let status =
        driver_cuda_new::abi_shell::pie_cuda_load_model(driver, &load, &mut load_caps);
    assert_eq!(status, PIE_STATUS_OK, "the snapshot loads");

    // ── Register the program. ──
    let program = registration(argmax_program());
    let borrow = ProgramDescBorrow::new(&program);
    let mut program_id = 0u64;
    let status = driver_cuda_new::abi_shell::pie_cuda_register_program(
        driver,
        borrow.as_raw(),
        &mut program_id,
    );
    assert_eq!(status, PIE_STATUS_OK, "the program registers");

    // ── Register its two channels, in the order the program indexes
    // them: 0 is what it takes, 1 is what it puts. ──
    let mut bindings = Vec::new();
    for (index, (shape, dtype, host_role)) in [
        (vec![LANES], driver_api::local::PIE_CHANNEL_DTYPE_F32, driver_api::local::PIE_CHANNEL_HOST_ROLE_WRITER),
        (vec![1u32], driver_api::local::PIE_CHANNEL_DTYPE_I32, driver_api::local::PIE_CHANNEL_HOST_ROLE_READER),
    ]
    .into_iter()
    .enumerate()
    {
        let plan = ChannelRegistrationPlan {
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
            driver_id: 0,
            seeded: false,
            extern_dir: driver_api::local::PIE_CHANNEL_EXTERN_NONE,
            extern_name: Vec::new(),
        };
        let borrow = ChannelDescBorrow::new(&plan);
        let mut binding = PieChannelEndpointBinding::default();
        let status = driver_cuda_new::abi_shell::pie_cuda_register_channel(
            driver,
            borrow.as_raw(),
            &mut binding,
        );
        assert_eq!(status, PIE_STATUS_OK, "channel {index} registers");
        bindings.push(binding);
    }

    // ── Bind an instance over both, in that order. ──
    let channel_ids: Vec<u64> = bindings.iter().map(|b| b.channel_id).collect();
    let inst = PieInstanceDesc {
        program_id,
        channel_ids: driver_api::local::PieU64Slice {
            ptr: channel_ids.as_ptr(),
            len: channel_ids.len(),
        },
        ..Default::default()
    };
    let mut instance = PieInstanceBinding::default();
    let status =
        driver_cuda_new::abi_shell::pie_cuda_bind_instance(driver, &inst, &mut instance);
    assert_eq!(status, PIE_STATUS_OK, "the instance binds");

    // ── The engine's side: publish the seed the program will take. ──
    let seed: [f32; LANES as usize] = [2.0, 7.0, 1.0, 7.0, 0.5, 7.0, -3.0, 6.0];
    let input = Endpoint::of(&bindings[0]);
    let output = Endpoint::of(&bindings[1]);
    let bytes: Vec<u8> = seed.iter().flat_map(|v| v.to_le_bytes()).collect();
    assert_eq!(input.cell_bytes, bytes.len(), "the wire cell is the seed's width");
    input.publish(&bytes);
    assert!(output.take().is_none(), "nothing published before the fire");

    // ── One decode token, which is what carries the program. ──
    let mut cell = PieTerminalCell {
        outcome: driver_api::local::PIE_TERMINAL_OUTCOME_PENDING,
        reserved0: 0,
    };
    let cell_ptr: *mut PieTerminalCell = &mut cell;
    let u32s = |v: &[u32]| PieU32Slice { ptr: v.as_ptr(), len: v.len() };
    let (roster, sub_indptr, sub_class) = ([0u32], [0u32, 1], [0u32]);
    let (tokens, positions) = ([7u32], [0u32]);
    let (pages, page_indptr, last_lens, qo) = ([0u32], [0u32, 1], [1u32], [0u32, 1]);
    let step = PieStepDesc {
        roster_rows: u32s(&roster),
        sub_batch_indptr: u32s(&sub_indptr),
        sub_batch_class: u32s(&sub_class),
        terminal_cells: PieTerminalCellPtrSlice { ptr: &cell_ptr, len: 1 },
        token_ids: u32s(&tokens),
        position_ids: u32s(&positions),
        kv_page_indices: u32s(&pages),
        kv_page_indptr: u32s(&page_indptr),
        kv_last_page_lens: u32s(&last_lens),
        qo_indptr: u32s(&qo),
        ..Default::default()
    };
    let instance_ids: [u64; 1] = [instance.instance_id];
    let frame = PieFrameDesc {
        abi_version: PIE_DRIVER_ABI_VERSION,
        instance_ids: PieU64Slice { ptr: instance_ids.as_ptr(), len: 1 },
        required_kv_pages: 1,
        steps: driver_api::local::PieStepDescSlice { ptr: &step, len: 1 },
        ..Default::default()
    };
    let completion =
        PieCompletion { wait_id: 0x5E5, target_epoch: 1, terminal_cell: std::ptr::null_mut() };
    let status =
        driver_cuda_new::abi_shell::pie_cuda_launch(driver, &frame, completion);
    assert_eq!(status, PIE_STATUS_OK, "the frame launches");

    // Run-ahead means the call returns with the fire still queued.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while FIRED.load(std::sync::atomic::Ordering::Acquire) == 0 {
        assert!(std::time::Instant::now() < deadline, "the fire never completed");
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

    driver_cuda_new::abi_shell::pie_cuda_destroy(driver);
}

/// Two requests in one frame each get their OWN answer.
///
/// The shell fired `instance_ids.first()` and nothing else, and then let
/// that one publish suppress the raw-logits fallback for the whole frame.
/// So a two-request batch sampled request 0 and returned request 1
/// NOTHING: no sample, because its program never ran, and no logits,
/// because request 0's had. A batch of one could not see it, which is
/// exactly the batch every other test in this file uses.
///
/// The two instances get DIFFERENT seeds with different argmaxes, so a
/// shell that fired one program and published its answer to both readers
/// would fail rather than coincide.
#[test]
fn every_request_in_a_frame_samples_its_own_row() {
    let _gpu = gpu_guard();
    let Some((snap, descriptor)) = qwen3_fixture() else {
        eprintln!("[ptir-shell] no cached Qwen3-0.6B or descriptor; skipping");
        return;
    };
    let boot = format!("[model]\ndescriptor = \"{}\"\n", descriptor.display());
    let mut caps = PieDriverCaps::default();
    let Some(driver) = driver_or_skip(&boot, &mut caps) else {
        return;
    };

    let snap_str = snap.to_string_lossy().into_owned();
    let load = PieModelLoadDesc {
        snapshot_dir: PieBytes { ptr: snap_str.as_ptr(), len: snap_str.len() },
        ..Default::default()
    };
    let mut load_caps = PieDriverCaps::default();
    assert_eq!(
        driver_cuda_new::abi_shell::pie_cuda_load_model(driver, &load, &mut load_caps),
        PIE_STATUS_OK,
        "the snapshot loads"
    );

    let program = registration(argmax_program());
    let borrow = ProgramDescBorrow::new(&program);
    let mut program_id = 0u64;
    assert_eq!(
        driver_cuda_new::abi_shell::pie_cuda_register_program(
            driver,
            borrow.as_raw(),
            &mut program_id
        ),
        PIE_STATUS_OK,
        "the program registers"
    );

    // Two instances of the SAME program, each over its own channel pair.
    let mut instances = Vec::new();
    let mut endpoints = Vec::new();
    for req in 0..2u64 {
        let mut bindings = Vec::new();
        for (index, (shape, dtype, host_role)) in [
            (
                vec![LANES],
                driver_api::local::PIE_CHANNEL_DTYPE_F32,
                driver_api::local::PIE_CHANNEL_HOST_ROLE_WRITER,
            ),
            (
                vec![1u32],
                driver_api::local::PIE_CHANNEL_DTYPE_I32,
                driver_api::local::PIE_CHANNEL_HOST_ROLE_READER,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let n = req * 2 + index as u64;
            let plan = ChannelRegistrationPlan {
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
                driver_id: 0,
                seeded: false,
                extern_dir: driver_api::local::PIE_CHANNEL_EXTERN_NONE,
                extern_name: Vec::new(),
            };
            let borrow = ChannelDescBorrow::new(&plan);
            let mut binding = PieChannelEndpointBinding::default();
            assert_eq!(
                driver_cuda_new::abi_shell::pie_cuda_register_channel(
                    driver,
                    borrow.as_raw(),
                    &mut binding
                ),
                PIE_STATUS_OK,
                "request {req} channel {index} registers"
            );
            bindings.push(binding);
        }
        let channel_ids: Vec<u64> = bindings.iter().map(|b| b.channel_id).collect();
        let inst = PieInstanceDesc {
            program_id,
            channel_ids: driver_api::local::PieU64Slice {
                ptr: channel_ids.as_ptr(),
                len: channel_ids.len(),
            },
            ..Default::default()
        };
        let mut instance = PieInstanceBinding::default();
        assert_eq!(
            driver_cuda_new::abi_shell::pie_cuda_bind_instance(driver, &inst, &mut instance),
            PIE_STATUS_OK,
            "instance {req} binds"
        );
        endpoints.push((Endpoint::of(&bindings[0]), Endpoint::of(&bindings[1])));
        instances.push(instance.instance_id);
    }

    // DIFFERENT argmaxes, so publishing one answer to both readers fails.
    let seeds: [[f32; LANES as usize]; 2] =
        [[2.0, 7.0, 1.0, 3.0, 0.5, 4.0, -3.0, 6.0], [9.0, 1.0, 1.0, 3.0, 0.5, 4.0, -3.0, 6.0]];
    let expected = [1i32, 0i32];
    for (req, (input, output)) in endpoints.iter().enumerate() {
        let bytes: Vec<u8> = seeds[req].iter().flat_map(|v| v.to_le_bytes()).collect();
        input.publish(&bytes);
        assert!(output.take().is_none(), "request {req} has nothing before the fire");
    }

    // TWO decode tokens, one per request. `qo_indptr` is what says so,
    // and it is what the shell reads to find each request's logits row.
    let mut cells = [
        PieTerminalCell { outcome: driver_api::local::PIE_TERMINAL_OUTCOME_PENDING, reserved0: 0 },
        PieTerminalCell { outcome: driver_api::local::PIE_TERMINAL_OUTCOME_PENDING, reserved0: 0 },
    ];
    let cell_ptrs: [*mut PieTerminalCell; 2] = [&mut cells[0], &mut cells[1]];
    let u32s = |v: &[u32]| PieU32Slice { ptr: v.as_ptr(), len: v.len() };
    let (roster, sub_indptr, sub_class) = ([0u32, 1], [0u32, 2], [0u32]);
    let (tokens, positions) = ([7u32, 11], [0u32, 0]);
    let (pages, page_indptr, last_lens, qo) =
        ([0u32, 1], [0u32, 1, 2], [1u32, 1], [0u32, 1, 2]);
    let step = PieStepDesc {
        roster_rows: u32s(&roster),
        sub_batch_indptr: u32s(&sub_indptr),
        sub_batch_class: u32s(&sub_class),
        terminal_cells: PieTerminalCellPtrSlice { ptr: cell_ptrs.as_ptr(), len: 2 },
        token_ids: u32s(&tokens),
        position_ids: u32s(&positions),
        kv_page_indices: u32s(&pages),
        kv_page_indptr: u32s(&page_indptr),
        kv_last_page_lens: u32s(&last_lens),
        qo_indptr: u32s(&qo),
        ..Default::default()
    };
    let frame = PieFrameDesc {
        abi_version: PIE_DRIVER_ABI_VERSION,
        instance_ids: PieU64Slice { ptr: instances.as_ptr(), len: instances.len() },
        required_kv_pages: 2,
        steps: driver_api::local::PieStepDescSlice { ptr: &step, len: 1 },
        ..Default::default()
    };
    let before = FIRED.load(std::sync::atomic::Ordering::Acquire);
    let completion =
        PieCompletion { wait_id: 0x5E6, target_epoch: 1, terminal_cell: std::ptr::null_mut() };
    assert_eq!(
        driver_cuda_new::abi_shell::pie_cuda_launch(driver, &frame, completion),
        PIE_STATUS_OK,
        "the two-request frame launches"
    );

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while FIRED.load(std::sync::atomic::Ordering::Acquire) == before {
        assert!(std::time::Instant::now() < deadline, "the fire never completed");
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

    driver_cuda_new::abi_shell::pie_cuda_destroy(driver);
}
