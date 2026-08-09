//! The PTIR compile plane, against a real GPU.
//!
//! # Why this test exists and what it is allowed to claim
//!
//! `cuda-progress.md` records the rule this suite is written under: *"Non-GPU
//! green is weak evidence. Every dispatch defect found on 2026-08-10 compiled
//! cleanly and passed the full non-GPU battery."* The unit tests beside
//! [`driver_cuda::program::cache`] and [`driver_cuda::program::compile`] pin arithmetic — a cache key, a rounded
//! launch width — and arithmetic is exactly the class of thing that can be
//! right in isolation and wrong against a driver.
//!
//! So this file compiles real CUDA with the real NVRTC, loads the real cubin,
//! and asks the real driver what it thinks. What it deliberately does NOT do
//! is launch a PTIR program: the lane table, the device rings and the fire are
//! not written yet, and a test that pretended otherwise would be the thing
//! §5.D1 refuses — code whose only gate is that it compiles.
//!
//! Skipped without a device, like every other `gpu_*` binary here.

use driver_cuda::program::{Disk, Module, disk_key, compile};

mod common;
use common::{device_or_skip, gpu_guard};

/// A minimal kernel that is legal under the same options a fused region is
/// compiled with, and whose name is spelled the way the host emitter spells
/// one: `ptir_fused_{signature:016x}_r{region}`.
///
/// `extern "C"` matters. NVRTC mangles a C++ name, and the driver looks the
/// entry up by the string the host put in the emitted table — so a kernel
/// without the linkage specifier compiles, loads, and then cannot be found,
/// which is a failure three steps away from its cause.
const SOURCE: &str = r#"
extern "C" __global__ void ptir_fused_00000000deadbeef_r0(
    const unsigned int* input,
    unsigned int* output,
    unsigned int lanes) {
    const unsigned int lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane < lanes) output[lane] = input[lane] + 1u;
}
"#;

const ENTRY: &str = "ptir_fused_00000000deadbeef_r0";

/// The whole compile path, end to end, on the device this build will run on.
///
/// One test rather than four, because the steps are not independent: a cubin
/// is only meaningful if it loads, and a loaded module is only meaningful if
/// its entry resolves. Splitting them would report three failures for one
/// cause.
#[test]
fn a_generated_region_compiles_loads_and_resolves_its_entry() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("PTIR NVRTC compile") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let (nvrtc_major, nvrtc_minor) = compile::version().expect("NVRTC must be loadable");
    eprintln!("compiling for sm_{major}{minor} with NVRTC {nvrtc_major}.{nvrtc_minor}");

    let cubin = compile::compile(SOURCE, &compile::arch_flag(major, minor))
        .expect("a well-formed region must compile");
    assert!(
        cubin.len() > 64,
        "a cubin is an ELF image; {} bytes is not one",
        cubin.len()
    );
    assert_eq!(
        &cubin[..4],
        b"\x7fELF",
        "the real architecture must yield a cubin, not PTX -- a virtual arch \
         would return text the driver would have to JIT a second time"
    );

    let module = Module::load(&cubin, ENTRY).expect("the cubin must load and carry its entry");
    assert_eq!(module.entry_name(), ENTRY);
    assert!(
        module.block_threads().is_power_of_two(),
        "the generated reductions halve blockDim.x, so a width that is not a \
         power of two folds some lanes twice and others never; got {}",
        module.block_threads()
    );
    assert!(
        (32..=1024).contains(&module.block_threads()),
        "a launch width outside [32, 1024] is not launchable: {}",
        module.block_threads()
    );
}

/// NVRTC's rejection must be [`Deterministic`], because that is what decides
/// whether the answer is remembered — and a program that is rejected on every
/// fire, recompiled each time, is the difference between a slow model and an
/// unusable one.
///
/// [`Deterministic`]: driver_cuda::program::FailureKind::Deterministic
#[test]
fn a_source_nvrtc_rejects_is_deterministic_and_carries_its_log() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("PTIR NVRTC rejection") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");

    let error = compile::compile(
        "extern \"C\" __global__ void k() { this_symbol_does_not_exist(); }",
        &compile::arch_flag(major, minor),
    )
    .expect_err("an undeclared identifier must not compile");

    assert_eq!(
        error.kind,
        driver_cuda::program::FailureKind::Deterministic,
        "a source NVRTC rejects will be rejected identically forever; not \
         remembering that is a recompile per fire"
    );
    assert!(
        error.message.contains("this_symbol_does_not_exist"),
        "the log is the whole diagnostic and must reach the caller: {}",
        error.message
    );
}

/// An entry name the cubin does not carry must fail at the lookup rather than
/// producing a `Module` whose function is null — and the module must not leak
/// on the way out, which is why the failure path unloads.
#[test]
fn a_missing_entry_is_refused_rather_than_answered_with_a_null_function() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("PTIR entry lookup") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let cubin = compile::compile(SOURCE, &compile::arch_flag(major, minor)).expect("compile");

    let error = Module::load(&cubin, "ptir_fused_0000000000000000_r9")
        .expect_err("an entry the cubin does not carry must not resolve");
    assert_eq!(error.call(), "cuModuleGetFunction");
}

/// The disk tier's whole claim: a cubin written by one run is loadable by the
/// next. Proven against a real cubin and a real module load rather than
/// against bytes, because "it round-trips" and "it still runs" are different
/// statements and only the second one matters.
#[test]
fn a_cubin_survives_the_disk_cache_and_still_loads() {
    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("PTIR disk cache") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let cubin = compile::compile(SOURCE, &compile::arch_flag(major, minor)).expect("compile");

    let directory = std::env::temp_dir().join(format!("pie-ptir-gpu-disk-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let disk = Disk::at(&directory);
    let key = disk_key("identity-for-this-test", SOURCE);

    disk.store(&key, 0, ENTRY, &cubin);
    let restored = disk
        .load(&key, 0, ENTRY)
        .expect("what was stored must come back");
    assert_eq!(restored, cubin, "the image must survive byte for byte");

    let module = Module::load(&restored, ENTRY).expect("a cached cubin must still load");
    assert_eq!(module.entry_name(), ENTRY);

    // The regression the source fingerprint exists to prevent: an edit to the
    // emitted text bumps no version number, so if the source were not in the
    // key this lookup would hit and yesterday's kernel would answer.
    let edited = disk_key("identity-for-this-test", &format!("{SOURCE}\n// edited"));
    assert_eq!(
        disk.load(&edited, 0, ENTRY),
        None,
        "an edited source must miss; hitting here is how a kernel change \
         silently does nothing"
    );

    let _ = std::fs::remove_dir_all(&directory);
}

/// The two control kernels are prebuilt on CUDA, absent from the kernels
/// archive, and therefore compiled here. This is the test that says the
/// forty lines of hand-written CUDA in `driver_cuda::program::run` are legal CUDA — which
/// nothing else can, because they never reach a compiler at build time.
#[test]
fn the_control_kernels_compile_and_both_entry_points_resolve() {
    use driver_cuda::program::{Control, run};

    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("PTIR control kernels") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");

    let directory = std::env::temp_dir().join(format!("pie-ptir-control-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let disk = Disk::at(&directory);

    let compiled = Control::compile(&disk, &compile::arch_flag(major, minor), "control-test")
        .expect("the readiness and commit kernels must compile");
    assert_eq!(compiled.readiness().entry_name(), run::READINESS_ENTRY);
    assert_eq!(compiled.commit().entry_name(), run::COMMIT_ENTRY);

    // Both are single-thread kernels guarded on `threadIdx.x == 0`, so the
    // launch width is irrelevant to correctness — but it must still be a
    // legal one, and a zero here would be a launch failure at the first fire.
    assert!(compiled.readiness().block_threads() >= 32);
    assert!(compiled.commit().block_threads() >= 32);

    // The second call must not recompile: the pair belongs to no program, so
    // paying NVRTC for it per program would be the cost this cache exists to
    // avoid.
    let again = Control::compile(&disk, &compile::arch_flag(major, minor), "control-test")
        .expect("the second compile must be answered from disk");
    assert_eq!(again.readiness().entry_name(), run::READINESS_ENTRY);

    let _ = std::fs::remove_dir_all(&directory);
}

/// The device ring, against the control kernels that advance it.
///
/// This is the first test in which the two halves meet: the host lays out the
/// cells and cursors, and the KERNEL — forty lines of CUDA compiled at run
/// time — decides readiness and moves the cursors. Every assertion below is
/// about a value the device wrote.
#[test]
fn the_control_kernels_gate_and_advance_the_device_ring() {
    use driver_cuda::device::{Allocator, OwnedStream};
    use driver_cuda::program::{ChannelShape, Control, Rings, run, launch_control};
    use driver::tensor_ir::DType;

    let _gpu = gpu_guard();
    let Some(device) = device_or_skip("PTIR device ring") else {
        return;
    };
    let (major, minor) = device.compute_capability().expect("compute capability");
    let stream = OwnedStream::new(0).expect("stream");
    let alloc = Allocator::new();

    let directory = std::env::temp_dir().join(format!("pie-ptir-ring-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let disk = Disk::at(&directory);
    let kernels = Control::compile(&disk, &compile::arch_flag(major, minor), "ring-test")
        .expect("control kernels compile");

    // Two channels: an input holding one f32 lane, and an output. Capacity 1,
    // so each ring is two slots and the sentinel is doing real work.
    let shapes = [
        ChannelShape {
            numel: 1,
            dtype: DType::F32,
            capacity: 1,
        },
        ChannelShape {
            numel: 1,
            dtype: DType::F32,
            capacity: 1,
        },
    ];
    let mut rings = Rings::new(&alloc, &shapes, stream.as_ref()).expect("rings");

    // Nothing is seeded yet, so a pass that must TAKE channel 0 is not ready.
    let verdict = launch_control::readiness(&kernels, &rings, &[0], &[1], &alloc, stream.as_ref())
        .expect("readiness launches");
    assert!(
        !verdict,
        "an empty input ring must refuse the pass; a driver that ran anyway \
         would read the zeroed cell as a real value"
    );

    // Seed channel 0. Now the same question must answer yes.
    rings
        .seed(0, 0, &7.5f32.to_le_bytes(), stream.as_ref())
        .expect("seed the input");
    stream.as_ref().synchronize().expect("sync");
    let verdict = launch_control::readiness(&kernels, &rings, &[0], &[1], &alloc, stream.as_ref())
        .expect("readiness launches");
    assert!(
        verdict,
        "a seeded input and an empty output ring is a ready pass"
    );

    let before = rings.cursors(stream.as_ref()).expect("cursors");
    assert!(before[0].is_readable(), "the seeded cell is published");
    assert_eq!(before[0].depth(2), 1, "one unconsumed item");
    assert_eq!(before[1].depth(2), 0, "the output is still empty");

    // Commit: consume channel 0, publish channel 1.
    launch_control::commit(&kernels, &rings, &[0], &[1], true, &alloc, stream.as_ref())
        .expect("commit launches");
    stream.as_ref().synchronize().expect("sync");

    let after = rings.cursors(stream.as_ref()).expect("cursors");
    assert_eq!(after[0].head, 1, "the taken channel's consumer advanced");
    assert_eq!(after[0].depth(2), 0, "and its item is consumed");
    assert!(!after[0].is_readable(), "its full bit is cleared");
    assert_eq!(after[1].tail, 1, "the put channel's producer advanced");
    assert_eq!(after[1].depth(2), 1, "and its cell is published");
    assert!(after[1].is_readable());

    // A commit with the pass flag clear must move nothing. This is the dummy
    // run: a blocked fire still launches every kernel, over the same cells,
    // and declines only at the publish.
    let held = rings.cursors(stream.as_ref()).expect("cursors");
    launch_control::commit(&kernels, &rings, &[0], &[1], false, &alloc, stream.as_ref())
        .expect("commit launches");
    stream.as_ref().synchronize().expect("sync");
    assert_eq!(
        rings.cursors(stream.as_ref()).expect("cursors"),
        held,
        "a pass that did not commit must leave every cursor where it was"
    );

    let _ = std::fs::remove_dir_all(&directory);
}
