//! The compile plane against a REAL emitted program, on a real GPU.
//!
//! # What this proves that `gpu_ptir.rs` cannot
//!
//! `gpu_ptir.rs` compiles a kernel this file's author wrote. That checks NVRTC
//! is reachable and that a cubin loads, and it checks nothing about the thing
//! a driver actually receives — which is tens of kilobytes of machine-written
//! CUDA, assembled by the host emitter out of `fused_block0.cuh`, the M1
//! runtime body, and a per-op generated tail, in ONE translation unit with no
//! `#include` and no include path.
//!
//! So this file starts where the engine starts: a `TraceContainer` an inferlet
//! author could have written, bound, compiled to stages, lowered to a
//! `LaunchPackage` by `tensor_compiler::codegen::launch::build`, and emitted
//! for `Backend::Cuda`. That is the exact artefact chain
//! `crates/engine/src/pipeline/program.rs` runs before it calls
//! `register_program`. Then the driver adopts and compiles it.
//!
//! The claim is narrow and worth stating precisely: **the emitter's CUDA and
//! this driver's NVRTC options agree, and the entry-name scheme is really an
//! ABI.** A driver that got either wrong would fail here and pass every
//! GPU-free test in the workspace.
//!
//! What is NOT claimed: that a program runs. The lane table, the device rings
//! and the fire are unwritten, so nothing is launched — and a test that
//! launched a kernel without them would be asserting against uninitialised
//! device memory.
//!
//! Adoption goes through `Boundaries::CUDA`, the vocabulary `serve/load.rs`
//! registers under, and not the bare `adopt_launch_package`'s Metal one. See
//! `gpu_ptir_fire`'s header for what the difference costs and for why a test
//! whose programs name neither vocabulary can hold the wrong one for years.

use driver::{Boundaries, Versions, adopt_launch_package_with};
use driver_cuda::program::{Disk, Runtime, Target};
use tensor_compiler::codegen::program::{Backend, emit_program};
use tensor_compiler::plan::compile_bound;
use tensor_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use tensor_ir::op::Op;
use tensor_ir::registry::{ModelProfile, Stage};
use tensor_ir::types::{DType, Shape};
use tensor_ir::validate::bind;

mod common;
use common::{device_or_skip, gpu_guard};

/// A vocabulary small enough to keep the emitted text readable and large
/// enough that the reduction is a real tree rather than one lane.
const VOCAB: u32 = 128;

fn profile() -> ModelProfile {
    let mut profile = ModelProfile::dummy();
    profile.vocab = VOCAB;
    profile
}

fn chan(shape: Shape, dtype: DType, host_role: HostRole, seeded: bool) -> ChannelDecl {
    ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity: 1,
        host_role,
        seeded,
    }
}

/// The smallest program that is still a decoding program: read a logits-shaped
/// row off a channel, argmax it, publish the token.
///
/// Greedy sampling is the case to pick because it is the one every deployment
/// runs and the one whose emitted region exercises the parts a driver can get
/// wrong: a channel take, a block-wide reduction with a tie-break, and a
/// channel put, all inside one fused region with the pass-atomic commit.
fn greedy_epilogue() -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            chan(Shape::vector(VOCAB), DType::F32, HostRole::None, true),
            chan(Shape::SCALAR, DType::I32, HostRole::Reader, false),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanRead(0),
                Op::ReduceArgmax(0),
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: Vec::new(),
    }
}

/// The emitter's kernel table in the shape a driver receives it.
///
/// Three field-identical `EmittedKernel` structs exist in this workspace —
/// `tensor_compiler::codegen::program`'s (what the emitter returns),
/// `driver_api::plan`'s (what a driver adopts to), and
/// `driver_api::local::PieEmittedKernel` (what crosses the C ABI) — and the
/// engine copies field by field between the first two at
/// `crates/engine/src/driver/abi.rs:129-141`. This reproduces that hop, which
/// is what makes this test measure the emitter-to-driver agreement rather than
/// a shortcut around it.
///
/// Written out field by field rather than with `..` so that an ABI field added
/// on one side and not the other fails to compile here, which is the only
/// place the three copies are ever seen together.
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

/// A cache with a home of its own, so a test never reads or writes the
/// developer's real `$PIE_HOME` — and so the disk assertions below are about
/// this run rather than about whatever a previous one left behind.
fn scratch_disk(name: &str) -> (Disk, std::path::PathBuf) {
    let directory =
        std::env::temp_dir().join(format!("pie-ptir-emitted-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    (Disk::at(&directory), directory)
}

fn target(device: &driver_cuda::device::Device) -> Target {
    let (major, minor) = device.compute_capability().expect("compute capability");
    Target {
        major,
        minor,
        device: u64::try_from(device.ordinal()).unwrap_or(0),
        nvrtc: driver_cuda::program::compile::version().expect("NVRTC must be loadable"),
    }
}

/// The whole chain: author a trace, run the host's real pipeline over it, and
/// compile what comes out.
#[test]
fn the_hosts_own_emitted_cuda_compiles_in_this_driver() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("PTIR emitted-program compile") else {
        return;
    };

    let bound = bind(greedy_epilogue(), profile()).expect("the container binds");
    let stages = compile_bound(&bound);
    let package = tensor_compiler::codegen::launch::build(&bound, &stages);
    let emitted = emit_program(Backend::Cuda, &stages, &bound);
    let kernels = as_abi(&emitted);

    // The emitter's CUDA arm produces fused regions and nothing else: no
    // readiness kernel, no commit kernel, no grouped kernel, because on CUDA
    // those are prebuilt and the fused kernel is already lane-parallel. If
    // this ever stops holding, the driver's single-kind lookup is wrong and
    // this is where that is found out.
    assert!(
        !kernels.is_empty(),
        "the CUDA emitter produced nothing for a program with a fused region; \
         this driver carries no emitter, so there is no slower path"
    );
    for kernel in &kernels {
        assert_eq!(
            kernel.kind,
            driver_api::local::PIE_KERNEL_FUSED,
            "CUDA emits only fused regions; kind {} is a shape this driver \
             does not look up",
            kernel.kind
        );
    }
    let emitted: usize = kernels.iter().filter(|k| !k.source.is_empty()).count();
    assert!(emitted > 0, "every region declined; nothing to compile");
    eprintln!(
        "emitted {} kernel(s), {emitted} with source; largest is {} bytes",
        kernels.len(),
        kernels.iter().map(|k| k.source.len()).max().unwrap_or(0)
    );

    let plan = adopt_launch_package_with(package, Boundaries::CUDA).expect("the driver adopts the package");
    assert!(
        plan.executable,
        "a greedy epilogue must be executable: {}",
        plan.reject_reason.as_deref().unwrap_or("no reason given")
    );

    let (disk, directory) = scratch_disk("compile");
    let mut runtime = Runtime::new(disk);
    let versions = Versions::from_compiler(Backend::Cuda.emitter_version());

    let compiled = runtime
        .compile(0xC3, &plan, &kernels, versions, target(&device))
        .unwrap_or_else(|failure| {
            panic!(
                "the host's own emitted CUDA must compile in this driver: {}",
                failure.reason()
            )
        });

    assert_eq!(
        compiled.stages.len(),
        plan.package.plans.len(),
        "one compiled stage per stage plan"
    );
    let regions: usize = compiled.stages.iter().map(|s| s.regions.len()).sum();
    assert!(regions > 0, "no region compiled, so nothing would launch");
    for stage in compiled.stages.iter() {
        for region in stage.regions.iter() {
            assert!(
                region.module.entry_name().starts_with("ptir_fused_"),
                "the entry-name scheme is an ABI, not a formatting choice: {}",
                region.module.entry_name()
            );
            assert!(
                region.module.block_threads().is_power_of_two(),
                "a generated region reduces by halving blockDim.x"
            );
        }
    }

    let stats = runtime.stats();
    assert_eq!(
        stats.compilations, regions as u64,
        "every generated region must have been compiled exactly once"
    );

    let _ = std::fs::remove_dir_all(&directory);
}

/// Registering the same program twice must compile once. This is the whole
/// reason the program tier exists: a program is registered once and bound many
/// times, and paying a multi-hundred-millisecond NVRTC compile per bind is the
/// difference between a slow model and an unusable one.
#[test]
fn a_second_registration_of_one_program_compiles_nothing() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("PTIR program cache") else {
        return;
    };

    let bound = bind(greedy_epilogue(), profile()).expect("binds");
    let stages = compile_bound(&bound);
    let package = tensor_compiler::codegen::launch::build(&bound, &stages);
    let emitted = emit_program(Backend::Cuda, &stages, &bound);
    let kernels = as_abi(&emitted);
    let plan = adopt_launch_package_with(package, Boundaries::CUDA).expect("adopts");

    let (disk, directory) = scratch_disk("dedup");
    let mut runtime = Runtime::new(disk);
    let versions = Versions::from_compiler(Backend::Cuda.emitter_version());
    let target = target(&device);

    runtime
        .compile(0xC3, &plan, &kernels, versions, target)
        .expect("first compile");
    let after_first = runtime.stats().compilations;
    assert!(
        after_first > 0,
        "the first registration must actually compile"
    );

    runtime
        .compile(0xC3, &plan, &kernels, versions, target)
        .expect("second compile");
    assert_eq!(
        runtime.stats().compilations,
        after_first,
        "the second registration of one program hash must not reach NVRTC"
    );
    assert!(
        runtime.stats().memory_hits > 0,
        "and it must be recorded as a hit rather than silently free"
    );

    let _ = std::fs::remove_dir_all(&directory);
}

/// A fresh process must not recompile what the last one left on disk. Proven
/// with two `Runtime`s over one directory, which is what a restart is from the
/// cache's point of view.
#[test]
fn a_fresh_runtime_answers_from_the_disk_the_last_one_wrote() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("PTIR disk tier") else {
        return;
    };

    let bound = bind(greedy_epilogue(), profile()).expect("binds");
    let stages = compile_bound(&bound);
    let package = tensor_compiler::codegen::launch::build(&bound, &stages);
    let emitted = emit_program(Backend::Cuda, &stages, &bound);
    let kernels = as_abi(&emitted);
    let plan = adopt_launch_package_with(package, Boundaries::CUDA).expect("adopts");

    let (_, directory) = scratch_disk("restart");
    let versions = Versions::from_compiler(Backend::Cuda.emitter_version());
    let target = target(&device);

    let mut first = Runtime::new(Disk::at(&directory));
    first
        .compile(0xC3, &plan, &kernels, versions, target)
        .expect("first process");
    let compiled = first.stats().compilations;
    assert!(compiled > 0);
    drop(first);

    // A different `Runtime` shares nothing but the directory, which is exactly
    // what a restarted process has.
    let mut second = Runtime::new(Disk::at(&directory));
    second
        .compile(0xC3, &plan, &kernels, versions, target)
        .expect("second process");
    assert_eq!(
        second.stats().compilations,
        0,
        "a restart must not pay NVRTC again for a program whose cubins are on \
         disk; {} region(s) were recompiled",
        second.stats().compilations
    );
    assert_eq!(
        second.stats().persistent_hits,
        compiled,
        "and every region must have come from the disk tier"
    );

    let _ = std::fs::remove_dir_all(&directory);
}

/// An emitter bump must miss. The number crosses the ABI precisely so a host
/// that starts emitting different text is not answered out of a cache keyed on
/// versions the driver hardcoded — which is the failure the C++'s
/// `kMetalM1EmitterVersion = 23` had already suffered by the time the host
/// reached 36.
#[test]
fn a_host_side_emitter_bump_recompiles_rather_than_reusing() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("PTIR emitter version") else {
        return;
    };

    let bound = bind(greedy_epilogue(), profile()).expect("binds");
    let stages = compile_bound(&bound);
    let package = tensor_compiler::codegen::launch::build(&bound, &stages);
    let emitted = emit_program(Backend::Cuda, &stages, &bound);
    let kernels = as_abi(&emitted);
    let plan = adopt_launch_package_with(package, Boundaries::CUDA).expect("adopts");

    let (_, directory) = scratch_disk("emitter");
    let target = target(&device);
    let real = Backend::Cuda.emitter_version();

    let mut runtime = Runtime::new(Disk::at(&directory));
    runtime
        .compile(0xC3, &plan, &kernels, Versions::from_compiler(real), target)
        .expect("the real emitter version");
    let after_first = runtime.stats().compilations;

    // A DIFFERENT program hash, so the program tier cannot answer and the
    // question is really about the identity the stage and disk tiers key on.
    runtime
        .compile(
            0xC4,
            &plan,
            &kernels,
            Versions::from_compiler(real + 1),
            target,
        )
        .expect("a bumped emitter version");
    assert!(
        runtime.stats().compilations > after_first,
        "a bumped emitter version must miss every tier; reusing here is how a \
         driver runs code the current host would not have emitted"
    );

    let _ = std::fs::remove_dir_all(&directory);
}
