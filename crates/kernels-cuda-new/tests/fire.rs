//! The crate fires a kernel on a real device.
//!
//! Everything above this file is a claim about text: a row names an
//! instantiation, a header set resolves an include, a cache key spans what
//! produced a cubin. None of it is worth anything if the kernel does not run
//! and produce the number, and there is exactly one way to find that out.
//!
//! # What has to be true for this to pass
//!
//! A GPU, a driver, and `libnvrtc.so` reachable by `dlopen`. There is
//! deliberately no toolkit in that list: the crate carries its sources, NVRTC
//! carries the compiler, and nothing links. A machine that can run a CUDA
//! program can run this even if it has never had `nvcc` installed — which is
//! the property the whole design is arranged around, and the only place it is
//! actually demonstrated.
//!
//! Without a device the target still COMPILES and every test skips with a
//! reason. A skip is not a pass and the message says so; the alternative — a
//! test that silently succeeds on a laptop — is how a broken launch path ships.

#![cfg(feature = "_cuda")]

use kernels_cuda_new::runtime::{self, ArgValue, Dims, Stream, cache};
use kernels_cuda_new::unit;

/// `sm_XY` for the current device, or a stated reason there is none.
///
/// Every test opens with this. `cache::arch` is also what a fire uses, so a
/// skip here is the same question the launch path asks, not a proxy for it.
///
/// It also binds the thread, which the tests need for their OWN driver-API
/// calls rather than for the crate's: `cuMemAlloc_v2` below is as much a
/// driver-API call as `cuLaunchKernel` is, and a test thread that has not
/// forced the primary context cannot allocate the buffer it means to launch
/// over. `fire` does this for itself; a caller doing driver-API work beside a
/// fire has to do it for itself too, which is worth demonstrating here.
fn arch_or_skip(what: &str) -> Option<&'static str> {
    match cache::arch() {
        Some(arch) => match cache::bind_context() {
            Ok(()) => Some(arch),
            Err(why) => {
                eprintln!("SKIP {what}: no usable context ({why})");
                None
            }
        },
        None => {
            eprintln!("SKIP {what}: no CUDA device is current");
            None
        }
    }
}

/// Every unit compiles, and the compile is what the cache holds.
///
/// The first fire pays for NVRTC; this is the seam `cache::warm` exists to
/// move, so warming is also how the compile gets measured without a launch in
/// the way.
#[test]
fn every_unit_compiles_on_this_device() {
    let Some(arch) = arch_or_skip("every_unit_compiles_on_this_device") else { return };

    for (name, outcome) in cache::warm() {
        let rows = outcome.unwrap_or_else(|why| panic!("{name} will not compile for {arch}: {why}"));
        assert!(rows > 0, "{name} compiled and resolved no entries");
    }
}

/// A row with no launcher anywhere still fires.
///
/// `norm::scalar_mul_bf16` is the proof, because its `.cu` was DELETED: there
/// is no `pie_k_norm_scalar_mul_bf16` in any archive, no host launcher holding
/// a `<<<>>>`, and no C++ caller. If this passes, the kernel that ran was
/// compiled from text in this binary — there is no other candidate.
#[test]
fn a_row_with_no_launcher_fires() {
    let Some(_) = arch_or_skip("a_row_with_no_launcher_fires") else { return };

    let symbol = "norm::scalar_mul_bf16";
    assert!(runtime::hosts(symbol), "{symbol} is hosted by no unit");

    let n: usize = 4096;
    let bytes = n * 2;
    let mut device_ptr = 0u64;
    // SAFETY: `device_ptr` is a live out-parameter and `bytes` is non-zero.
    let code = unsafe { cudarc::driver::sys::cuMemAlloc_v2(&raw mut device_ptr, bytes) };
    assert_eq!(code, cudarc::driver::sys::CUresult::CUDA_SUCCESS, "allocation");

    // bf16 1.0 is 0x3F80 -- the top half of an f32 1.0, which is the whole of
    // what bf16 is. Writing it with a fill means the expected answer after a
    // multiply by two is 0x4000 and needs no host-side conversion to state.
    let ones = vec![0x3F80u16; n];
    // SAFETY: the allocation above is `bytes` long and `ones` is exactly that.
    let code = unsafe {
        cudarc::driver::sys::cuMemcpyHtoD_v2(device_ptr, ones.as_ptr().cast(), bytes)
    };
    assert_eq!(code, cudarc::driver::sys::CUresult::CUDA_SUCCESS, "upload");

    let values = [
        ArgValue::Ptr(device_ptr as *mut std::ffi::c_void),
        ArgValue::F32(2.0),
        ArgValue::Usize(n),
    ];
    // `..Default::default()` for the axes an elementwise rule never reads.
    // `Dims` gained a head and expert vocabulary when the rest of
    // `LaunchRule` was ported; a flat pointwise fire states the three
    // extents it has and leaves the rest zero, which is what a rule that
    // reads them would refuse on rather than silently launch nothing.
    let dims = Dims {
        rows: 1,
        width: u32::try_from(n).unwrap(),
        in_width: u32::try_from(n).unwrap(),
        ..Default::default()
    };

    // SAFETY: the pointer is a live device allocation of `n` bf16 elements,
    // the values match the row's operands, and the null stream is always live.
    unsafe { runtime::fire(symbol, dims, &values, Stream::NULL) }
        .unwrap_or_else(|why| panic!("{symbol} would not fire: {why}"));

    let code = unsafe { cudarc::driver::sys::cuCtxSynchronize() };
    assert_eq!(code, cudarc::driver::sys::CUresult::CUDA_SUCCESS, "synchronise");

    let mut back = vec![0u16; n];
    // SAFETY: same allocation, same length.
    let code = unsafe {
        cudarc::driver::sys::cuMemcpyDtoH_v2(back.as_mut_ptr().cast(), device_ptr, bytes)
    };
    assert_eq!(code, cudarc::driver::sys::CUresult::CUDA_SUCCESS, "download");
    // SAFETY: the allocation is still live and is freed exactly once.
    unsafe { cudarc::driver::sys::cuMemFree_v2(device_ptr) };

    assert!(
        back.iter().all(|&h| h == 0x4000),
        "the kernel ran and did not double: first differing element is {:?}",
        back.iter().position(|&h| h != 0x4000)
    );
}

/// The generated typed façade fires the same kernel as the dynamic path.
///
/// This is the property that makes `api` worth generating rather than writing:
/// both call sites come from one row, so they cannot disagree about the
/// operand order. A hand-written wrapper that transposed two pointers would
/// pass every test that only ever called the dynamic form.
#[test]
fn the_typed_api_and_the_dynamic_path_agree() {
    let Some(_) = arch_or_skip("the_typed_api_and_the_dynamic_path_agree") else { return };

    let n: usize = 1024;
    let bytes = n * 2;
    let mut device_ptr = 0u64;
    // SAFETY: live out-parameter, non-zero size.
    unsafe { cudarc::driver::sys::cuMemAlloc_v2(&raw mut device_ptr, bytes) };

    let ones = vec![0x3F80u16; n];
    // SAFETY: the allocation is `bytes` long.
    unsafe { cudarc::driver::sys::cuMemcpyHtoD_v2(device_ptr, ones.as_ptr().cast(), bytes) };

    // No `Dims` here, and that is the whole difference between the two
    // surfaces. The generated wrapper took one because a row states a
    // `LaunchRule` and the geometry is derived from `Dims` at the call site;
    // `x::norm::scalar_mul_bf16` computes its own grid from the extent it
    // takes, so there is no vocabulary of axes for a caller to fill in
    // wrongly. The sibling test above still builds one, because
    // `runtime::fire` is the dynamic path and still binds by symbol.
    // SAFETY: a live device allocation of `n` bf16 elements and the null stream.
    let fired = unsafe {
        kernels_cuda_new::x::norm::scalar_mul_bf16(
            device_ptr as *mut kernels_cuda_new::x::abi::bf16,
            2.0,
            n,
            std::ptr::null_mut(),
        )
    };
    // `api::norm_scalar_mul_bf16` was here until north star §6 half A retired
    // `emit.rs`. It was the emitter's ONLY caller in the tree — `model-loader`
    // had already crossed to `x::quant`'s four host programs and said so in
    // its own module doc, so a 1,070-line generator and a `pub mod api`
    // survived the whole sweep on this one line.
    //
    // The replacement is not the same shape and the difference is the point.
    // The generated function took `dims` and returned `Result`, because a row
    // states a `LaunchRule` and the geometry is computed from `Dims` at the
    // call site. `x::norm::scalar_mul_bf16` takes the extent it actually uses
    // and returns `Fired`, which can say `Declined` — the refusal is a value
    // here, where the generator could only drop the row and leave a comment.
    assert!(
        matches!(fired, kernels_cuda_new::x::Fired::Launched),
        "the typed entry fires: {fired:?}"
    );

    // SAFETY: no outstanding work beyond the launch above.
    unsafe { cudarc::driver::sys::cuCtxSynchronize() };

    let mut back = vec![0u16; n];
    // SAFETY: same allocation, same length.
    unsafe { cudarc::driver::sys::cuMemcpyDtoH_v2(back.as_mut_ptr().cast(), device_ptr, bytes) };
    // SAFETY: freed exactly once.
    unsafe { cudarc::driver::sys::cuMemFree_v2(device_ptr) };

    assert!(back.iter().all(|&h| h == 0x4000), "the typed entry did not double");
}

/// A symbol no unit holds is refused as unknown, not as broken.
///
/// The distinction is the reason `fire` returns a `Result` rather than the
/// `bool` its ancestor returned: *"not mine"* means a caller should try
/// another dispatcher, and *"mine and broken"* means it should stop. One value
/// cannot say both, and the version that tried lost the reason.
#[test]
fn an_unhosted_symbol_is_unknown() {
    let symbol = "norm::a_kernel_nobody_wrote";
    assert!(!runtime::hosts(symbol));

    // SAFETY: the values are empty and the call returns before touching CUDA.
    let refusal = unsafe { runtime::fire(symbol, Dims { rows: 1, width: 1, in_width: 1, ..Default::default() }, &[], Stream::NULL) };
    assert!(
        matches!(refusal, Err(runtime::Error::Unknown { .. })),
        "an unhosted symbol answered {refusal:?}"
    );
}

/// Compiling twice hands back the same module.
///
/// The cache is per (unit, architecture) for the process, which is the design
/// decision `lib.rs` states: a fire happens once per kernel per layer per
/// token, so a second NVRTC compile on the launch path would be a stall
/// nothing recovers from. Address equality is the only honest way to check it.
#[test]
fn a_unit_compiles_once() {
    let Some(_) = arch_or_skip("a_unit_compiles_once") else { return };

    let (index, u) = unit::unit_of("norm::scalar_mul_bf16").expect("some unit hosts it");
    let first = cache::module(index, u).expect("compiles");
    let second = cache::module(index, u).expect("compiles");
    assert!(std::ptr::eq(first, second), "the unit was compiled a second time");
}
