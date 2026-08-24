#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(all(feature = "_cuda", not(any(feature = "cuda-12", feature = "cuda-13"))))]
compile_error!(
    "kernels-cuda's runtime needs exactly one CUDA runtime version: \
     enable `cuda-12` or `cuda-13`, matching the libcudart this binary will load"
);

#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!(
    "kernels-cuda: `cuda-12` and `cuda-13` are mutually exclusive -- a binary \
     loads one libcudart, and the two disagree on `cudaGraphAddNode`'s arity"
);

pub use kernels::{Cap, KernelSig, LaunchRule, Lit, Refusal, Source, Ty};

pub use kernels::routine::{In, InOut, Out};

pub mod jit;

pub mod routine;

pub mod source;

pub mod raises;

pub mod driver_internal;

pub mod dist;

pub mod comm;

pub mod tile;

pub mod attn;
pub mod gemm;
pub mod graph;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
/// GENERATED from `kernels::points` × this plane's `*_CLAIMS`; see the
/// file's own header and `tests/points_dispatch_is_current.rs`.
pub mod points_dispatch;
pub mod quant;
pub mod rope;
pub mod sample;
pub mod ssm;

/// The device-side checks for the kernels this crate declares AND launches
/// itself — the two `#[claims]` bodies that are launchers rather than
/// delegations. Behind `_cuda` because it fires, and a unit module rather
/// than a `tests/` file because `cudarc` is optional and an optional
/// dev-dependency is not a thing Cargo has.
#[cfg(all(test, feature = "_cuda"))]
mod devtest;

#[cfg(feature = "_cuda")]
pub mod tower;
pub mod views;
pub mod vision;

pub type Plane = crate::jit::Cuda;

#[cfg(not(target_family = "wasm"))]
#[::linkme::distributed_slice]
pub static CUDA_ROUTINES: [::kernels::routine::Routine<Plane>];

#[cfg(not(target_family = "wasm"))]
pub use CUDA_ROUTINES as ROUTINES;

#[cfg(target_family = "wasm")]
#[doc(hidden)]
pub struct Registered(pub ::kernels::routine::Routine<Plane>);

#[cfg(target_family = "wasm")]
::inventory::collect!(Registered);

pub fn rows() -> impl Iterator<Item = &'static ::kernels::routine::Routine<Plane>> {
    #[cfg(not(target_family = "wasm"))]
    {
        ROUTINES.iter()
    }
    #[cfg(target_family = "wasm")]
    {
        ::inventory::iter::<Registered>.into_iter().map(|r| &r.0)
    }
}

#[cfg(feature = "_cuda")]
#[cfg_attr(docsrs, doc(cfg(any(feature = "cuda-12", feature = "cuda-13"))))]
pub use jit::Error;

pub use jit::ArgValue;

#[must_use]
pub fn sigs() -> &'static [KernelSig] {
    static ROWS: std::sync::OnceLock<&'static [KernelSig]> = std::sync::OnceLock::new();
    ROWS.get_or_init(|| {
        let mut rows: Vec<KernelSig> = Vec::new();
        for r in crate::rows() {
            {
                let symbol: &'static str = String::leak(r.symbol());
                rows.push(KernelSig {
                    name: symbol,
                    symbol,
                    args: r.args,
                    whole: r.whole,
                    depth_prefix_plan: r.depth_prefix_plan,
                    sources: r.sources,
                    derived: r.derived,
                    internal: r.internal,
                    no_join: r.no_join,
                    driver: r.driver,
                    canon: r.canon,
                    point: r.point,
                    out_rule: r.out_rule,
                    ..SIG_BASE
                });
            }
        }
        Vec::leak(rows)
    })
}

const SIG_BASE: KernelSig = KernelSig {
    name: "",
    symbol: "",
    whole: false,
    depth_prefix_plan: false,
    args: &[],
    sources: &[],
    derived: &[],
    axes: &[],
    internal: false,
    no_join: false,
    driver: false,
    canon: None,
    point: &[],
    out_rule: &[],
};

#[must_use]
pub fn routine(symbol: &str) -> Option<&'static jit::Routine> {
    rows().find(|r| r.answers(symbol))
}

/// The row for `symbol` AT a dtype point — how a caller that knows the
/// statement's value dtype selects among the rows `dtypes(..)` stamped.
/// Falls back to the first row when no row names the point (a single-point
/// routine's row has an empty or singleton `point`), so a caller with no
/// hint gets exactly what [`routine`] gives.
#[must_use]
pub fn routine_at(symbol: &str, point: &str) -> Option<&'static jit::Routine> {
    rows()
        .find(|r| r.answers(symbol) && r.point.contains(&point))
        .or_else(|| routine(symbol))
}

/// Fire a symbol on a stream, with no cuBLAS handle and no environment.
///
/// # Safety
///
/// [`call_answering`]'s, with both of the pointers it names left null.
pub unsafe fn call(
    symbol: &str,
    args: &[ArgValue],
    stream: *mut core::ffi::c_void,
) -> Result<(), kernels::Refusal> {
    unsafe { call_with_cublas(symbol, args, stream, core::ptr::null_mut()) }
}

/// [`call`], carrying a cuBLAS handle for the routines that want one.
///
/// # Safety
///
/// [`call_answering`]'s.
pub unsafe fn call_with_cublas(
    symbol: &str,
    args: &[ArgValue],
    stream: *mut core::ffi::c_void,
    cublas: *mut core::ffi::c_void,
) -> Result<(), kernels::Refusal> {
    unsafe { call_answering(symbol, args, stream, cublas, None) }
}

/// Fire a symbol, answering its asks out of `env`.
///
/// # Safety
///
/// Every pointer in `args` must address device memory of the extent the
/// symbol reads, and the kinds must match the signature the symbol was
/// compiled from -- `ArgValue` carries a kind but not a length, so an
/// argument of the right kind and the wrong size passes every check here
/// and faults on device. `stream` must be a live stream in the current
/// context, and `cublas`, if not null, a live handle set to that stream.
pub unsafe fn call_answering(
    symbol: &str,
    args: &[ArgValue],
    stream: *mut core::ffi::c_void,
    cublas: *mut core::ffi::c_void,
    env: Option<&dyn kernels::routine::Answers<jit::Cuda>>,
) -> Result<(), kernels::Refusal> {
    let Some(routine) = routine(symbol) else {
        return Err(kernels::Refusal::Undeclared);
    };

    let ctx = unsafe { jit::Ctx::on(stream).with_cublas(cublas) };
    let ctx = match env {
        Some(env) => ctx.with_env(env),
        None => ctx,
    };
    (routine.body)(&ctx, args)
}

pub use crate::jit::abi::Pointee as RoutineElem;

/// Stage a peel window's `(start, len)` pair into a named device scratch
/// slot, for the `devwin` kernels whose ABI reads `win[0]`/`win[1]` from
/// device memory.
///
/// The pair used to arrive as `keys::PeelWindow`, a driver-owned device
/// buffer; it is lowering-spliced now (design-no-ask §3, category E), so the
/// routine takes the two `Const<i32>`s the splice writes and stages them
/// itself. Eight bytes, stream-ordered against the launch that reads them.
pub(crate) fn stage_peel_window(
    ctx: &jit::Ctx<'_>,
    name: &'static str,
    start: i32,
    len: i32,
) -> Result<*mut u32, Refusal> {
    let win = ctx.scratch(name, 2 * core::mem::size_of::<u32>())?;
    let bounds = [start.unsigned_abs(), len.unsigned_abs()];
    #[cfg(feature = "_cuda")]
    {
        // Pageable host memory: `cudaMemcpyAsync` stages it before
        // returning, so the stack pair may die after this call.
        let bytes = unsafe {
            core::slice::from_raw_parts(
                bounds.as_ptr().cast::<u8>(),
                core::mem::size_of_val(&bounds),
            )
        };
        unsafe { jit::device::upload(win, bytes, ctx.stream())? };
    }
    #[cfg(not(feature = "_cuda"))]
    let _ = bounds;
    Ok(win.cast::<u32>())
}
