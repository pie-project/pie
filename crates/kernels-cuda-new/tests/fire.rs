//! The crate fires a kernel on a real device.
//!
//! Everything above this file is a claim about text: a body names an
//! instantiation, a header set resolves an include, a cache key spans what
//! produced a cubin. `every_instantiation_compiles.rs` checks that every one
//! of those strings COMPILES, which is a claim about text too. None of it is
//! worth anything if the kernel does not run and produce the number, and there
//! is exactly one way to find that out.
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

use std::ffi::c_void;

use cudarc::driver::sys::{CUresult, cuCtxSynchronize, cuMemAlloc_v2, cuMemFree_v2, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2};
use kernels_cuda_new::jit::{ArgValue, Ctx, cache};

/// `sm_XY` for the current device, or a stated reason there is none.
///
/// Every test opens with this. `cache::arch` is also what a fire uses, so a
/// skip here is the same question the launch path asks, not a proxy for it.
///
/// It also binds the thread, which the tests need for their OWN driver-API
/// calls rather than for the crate's: `cuMemAlloc_v2` below is as much a
/// driver-API call as `cuLaunchKernel` is, and a test thread that has not
/// forced the primary context cannot allocate the buffer it means to launch
/// over. A fire does this for itself; a caller doing driver-API work beside a
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

/// `n` bf16 elements on the device, filled with 1.0.
///
/// bf16 1.0 is `0x3F80` — the top half of an f32 1.0, which is the whole of
/// what bf16 is. Filling with it means the expected answer after a multiply by
/// two is `0x4000` and needs no host-side conversion to state.
struct Ones {
    ptr: u64,
    n: usize,
}

impl Ones {
    fn new(n: usize) -> Self {
        let bytes = n * 2;
        let mut ptr = 0u64;
        // SAFETY: `ptr` is a live out-parameter and `bytes` is non-zero.
        let code = unsafe { cuMemAlloc_v2(&raw mut ptr, bytes) };
        assert_eq!(code, CUresult::CUDA_SUCCESS, "allocation");

        let ones = vec![0x3F80u16; n];
        // SAFETY: the allocation above is `bytes` long and `ones` is exactly that.
        let code = unsafe { cuMemcpyHtoD_v2(ptr, ones.as_ptr().cast(), bytes) };
        assert_eq!(code, CUresult::CUDA_SUCCESS, "upload");
        Self { ptr, n }
    }

    /// Synchronise, read back, and assert every element doubled.
    fn assert_doubled(&self, what: &str) {
        // SAFETY: no outstanding work beyond the launch under test.
        let code = unsafe { cuCtxSynchronize() };
        assert_eq!(code, CUresult::CUDA_SUCCESS, "{what}: synchronise");

        let mut back = vec![0u16; self.n];
        // SAFETY: same allocation, same length.
        let code = unsafe { cuMemcpyDtoH_v2(back.as_mut_ptr().cast(), self.ptr, self.n * 2) };
        assert_eq!(code, CUresult::CUDA_SUCCESS, "{what}: download");
        assert!(
            back.iter().all(|&h| h == 0x4000),
            "{what}: the kernel ran and did not double — first differing element is {:?}",
            back.iter().position(|&h| h != 0x4000)
        );
    }
}

impl Drop for Ones {
    fn drop(&mut self) {
        // SAFETY: the allocation is still live and is freed exactly once.
        unsafe { cuMemFree_v2(self.ptr) };
    }
}

/// A kernel with no launcher anywhere still fires.
///
/// `norm::scalar_mul_bf16` is the proof, because its `.cu` was DELETED: there
/// is no `pie_k_norm_scalar_mul_bf16` in any archive, no host launcher holding
/// a `<<<>>>`, and no C++ caller. If this passes, the kernel that ran was
/// compiled from text in this binary — there is no other candidate.
#[test]
fn a_kernel_with_no_launcher_fires() {
    let Some(_) = arch_or_skip("a_kernel_with_no_launcher_fires") else { return };

    let symbol = "norm::scalar_mul_bf16";
    assert!(kernels_cuda_new::routine(symbol).is_some(), "{symbol} is declared by no family");

    let buffer = Ones::new(4096);
    let args = [
        ArgValue::Ptr(buffer.ptr as *mut c_void),
        ArgValue::F32(2.0),
        ArgValue::Usize(buffer.n),
    ];
    // SAFETY: the pointer is a live device allocation of `n` bf16 elements,
    // the values match the routine's signature, and the null stream is always
    // live.
    unsafe { kernels_cuda_new::call(symbol, &args, std::ptr::null_mut()) }
        .unwrap_or_else(|why| panic!("{symbol} would not fire: {why:?}"));

    buffer.assert_doubled(symbol);
}

/// The dynamic path and the routine's own `fn` fire the same kernel.
///
/// This is the property that makes the extractor shape worth its machinery:
/// `call` reaches the body through an argument table DERIVED from the `fn`
/// signature, so the two call sites cannot disagree about operand order. A
/// hand-written dispatch arm that transposed the scalar and the extent would
/// pass every test that only ever called one of them.
///
/// The difference between the two is the vocabulary, not the kernel.
/// `x::norm::scalar_mul_bf16` takes a typed pointer and a `usize` and computes
/// its own grid; `call` takes `&[ArgValue]` and a symbol, and is what a trace
/// reaches. Both end at the same `ctx.launch`.
#[test]
fn the_dynamic_path_and_the_routine_agree() {
    let Some(_) = arch_or_skip("the_dynamic_path_and_the_routine_agree") else { return };

    let buffer = Ones::new(1024);

    // SAFETY: a live device allocation of `n` bf16 elements and the null stream.
    let ctx = unsafe { Ctx::on(std::ptr::null_mut()) };
    let fired = kernels_cuda_new::x::norm::scalar_mul_bf16(
        &ctx,
        buffer.ptr as *mut kernels_cuda_new::x::abi::bf16,
        2.0,
        buffer.n,
    );
    assert_eq!(fired, Ok(()), "the routine fires");
    buffer.assert_doubled("the routine");

    // The same buffer, halved back and doubled again through the symbol, so
    // the two paths are compared on one allocation rather than on two runs
    // that could differ in setup.
    let args = [
        ArgValue::Ptr(buffer.ptr as *mut c_void),
        ArgValue::F32(0.5),
        ArgValue::Usize(buffer.n),
    ];
    // SAFETY: as above — the allocation is live and the arguments match.
    unsafe { kernels_cuda_new::call("norm::scalar_mul_bf16", &args, std::ptr::null_mut()) }
        .expect("the dynamic path fires");
    // SAFETY: no outstanding work beyond the launch above.
    assert_eq!(unsafe { cuCtxSynchronize() }, CUresult::CUDA_SUCCESS);

    let args = [
        ArgValue::Ptr(buffer.ptr as *mut c_void),
        ArgValue::F32(2.0),
        ArgValue::Usize(buffer.n),
    ];
    // SAFETY: as above.
    unsafe { kernels_cuda_new::call("norm::scalar_mul_bf16", &args, std::ptr::null_mut()) }
        .expect("the dynamic path fires");
    buffer.assert_doubled("the dynamic path");
}

/// A symbol no family declares is refused as undeclared, not as broken.
///
/// The distinction is why the dynamic path returns a `Refusal` rather than the
/// `bool` its ancestor returned: *"not mine"* means a caller should try another
/// dispatcher, and *"mine and broken"* means it should stop. One value cannot
/// say both, and the version that tried lost the reason.
///
/// No device needed, and no `arch_or_skip`: the refusal is decided before
/// anything touches CUDA, which is itself the claim.
#[test]
fn an_undeclared_symbol_is_refused_as_undeclared() {
    let symbol = "norm::a_kernel_nobody_wrote";
    assert!(kernels_cuda_new::routine(symbol).is_none());

    // SAFETY: the values are empty and the call returns before touching CUDA.
    let refusal = unsafe { kernels_cuda_new::call(symbol, &[], std::ptr::null_mut()) };
    assert_eq!(refusal, Err(kernels::Refusal::Undeclared), "an undeclared symbol answered {refusal:?}");
}

/// Resolving one symbol twice hands back the same function.
///
/// The cache is per (root, instantiation, architecture) for the process, which
/// is the design decision `jit/cache.rs` states: a fire happens once per kernel
/// per layer per token, so a second NVRTC compile on the launch path would be a
/// stall nothing recovers from. Address equality is the only honest way to
/// check it — a second compile would produce an equal-looking handle from a
/// different module.
#[test]
fn a_symbol_compiles_once() {
    let Some(_) = arch_or_skip("a_symbol_compiles_once") else { return };

    use kernels_cuda_new::x::norm::elementwise;
    let first = cache::resolve(&elementwise::ROOT, elementwise::inst::SCALAR_MUL_BF16)
        .expect("compiles");
    let second = cache::resolve(&elementwise::ROOT, elementwise::inst::SCALAR_MUL_BF16)
        .expect("compiles");
    assert!(std::ptr::eq(first, second), "the symbol was compiled a second time");
}

/// Two instantiations out of one root are two functions, not one.
///
/// The unit shape compiled a whole row list into one module, so this question
/// could not be asked of it. Per-symbol compilation makes the root a shared
/// INPUT rather than a shared output, and a cache keyed on the root alone
/// would hand both instantiations the first one's entry point — which would
/// run the wrong kernel and pass every test that fires only one of them.
#[test]
fn two_instantiations_of_one_root_are_distinct() {
    let Some(_) = arch_or_skip("two_instantiations_of_one_root_are_distinct") else { return };

    use kernels_cuda_new::x::norm::elementwise;
    let mul = cache::resolve(&elementwise::ROOT, elementwise::inst::SCALAR_MUL_BF16)
        .expect("compiles");
    let other = cache::resolve(&elementwise::ROOT, elementwise::inst::RESIDUAL_ADD_BF16)
        .expect("compiles");
    assert!(
        !std::ptr::eq(mul, other),
        "two instantiations of `{}` resolved to one function",
        elementwise::ROOT.name
    );
}
