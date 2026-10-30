//! Block-FP8 projections (128x128-block e4m3 codes, their ue8m0 exponents
//! laid in atoms) against the rows quantized to e4m3 in 1x128 blocks (W8A8)
//! on cuBLASLt's block-scaled FP8 GEMM (`BLK128x128_32F` A scales,
//! `VEC128_32F` B scales): the wgmma GEMM where there is no tcgen05 one.
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::cublaslt::sys as lt;
use cudarc::runtime::sys::{cudaGetDevice, cudaMemcpyAsync, cudaMemcpyKind};
use dtype::Dtype;

use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, kernel, refuse, stated};
use crate::tensor::Tensor;

const ACT_QUANT: &str = "linear/act_quant.cu";

const OK: lt::cublasStatus_t = lt::cublasStatus_t::CUBLAS_STATUS_SUCCESS;
/// The workspace slab cuBLASLt's dense products share (`linear::dense`'s),
/// requested at one size, the profile's `kernel.gemm.workspace_mb`, so the
/// load's scratch keeps one shape whatever products run.
const LT_WORKSPACE: &str = "linear.lt_workspace";

/// A shape's descriptors and its algorithm. The scale pointers are set on
/// `desc` a call at a time, under the lock.
struct Plan {
    desc: lt::cublasLtMatmulDesc_t,
    a: lt::cublasLtMatrixLayout_t,
    b: lt::cublasLtMatrixLayout_t,
    d: lt::cublasLtMatrixLayout_t,
    algo: lt::cublasLtMatmulAlgo_t,
}

// SAFETY: the descriptors are host objects, used under the plan's lock.
unsafe impl Send for Plan {}

impl Drop for Plan {
    fn drop(&mut self) {
        // A build that failed part way leaves the later handles null.
        for layout in [self.d, self.b, self.a] {
            if !layout.is_null() {
                // SAFETY: a layout `build` created for this plan alone, freed
                // once here.
                let _ = unsafe { lt::cublasLtMatrixLayoutDestroy(layout) };
            }
        }
        if !self.desc.is_null() {
            // SAFETY: the descriptor `build` created for this plan alone,
            // freed once here, after the layouts.
            let _ = unsafe { lt::cublasLtMatmulDescDestroy(self.desc) };
        }
    }
}

type Key = (i32, u32, u32, u32, usize, bool);

struct Lt {
    handles: HashMap<i32, usize>,
    plans: HashMap<Key, Option<Arc<Mutex<Plan>>>>,
}

fn state() -> &'static Mutex<Lt> {
    static STATE: OnceLock<Mutex<Lt>> = OnceLock::new();
    STATE.get_or_init(|| {
        Mutex::new(Lt {
            handles: HashMap::new(),
            plans: HashMap::new(),
        })
    })
}

fn device() -> i32 {
    let mut device = 0;
    // SAFETY: `device` is a local out-pointer; on an error it keeps its 0.
    let _ = unsafe { cudaGetDevice(&raw mut device) };
    device
}

/// The descriptors of a `[m, k] x [n, k]^T` product, and the first algorithm
/// cuBLASLt offers for it; `None` where it offers none.
fn build(
    handle: lt::cublasLtHandle_t,
    m: u32,
    n: u32,
    k: u32,
    workspace: usize,
    mx: bool,
) -> Option<Plan> {
    let mut plan = Plan {
        desc: std::ptr::null_mut(),
        a: std::ptr::null_mut(),
        b: std::ptr::null_mut(),
        d: std::ptr::null_mut(),
        // SAFETY: `cublasLtMatmulAlgo_t` is an opaque array of u64 words;
        // all-zero bytes are a valid value, overwritten below.
        algo: unsafe { core::mem::zeroed() },
    };
    // SAFETY: `plan.desc` is a local out-pointer; the compute and scale
    // types are the enum values cuBLASLt documents.
    let created = unsafe {
        lt::cublasLtMatmulDescCreate(
            &raw mut plan.desc,
            lt::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            lt::cudaDataType::CUDA_R_32F,
        )
    };
    if created != OK {
        return None;
    }
    // SAFETY: `plan.desc` was just created; every caller below passes a
    // pointer to a local of exactly `bytes` bytes, of the type the attribute
    // names, and cuBLASLt copies it before returning.
    let set = |attribute, value: *const c_void, bytes| unsafe {
        lt::cublasLtMatmulDescSetAttribute(plan.desc, attribute, value, bytes) == OK
    };
    use lt::cublasLtMatmulDescAttributes_t as A;
    let (ta, tb) = (
        cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
        cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
    );
    // `mx`: MXFP8, a ue8m0 exponent a 1x32 block on both sides, in tcgen05's atoms
    let (sa, sb) = if mx {
        (
            lt::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
            lt::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0,
        )
    } else {
        (
            lt::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F,
            lt::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F,
        )
    };
    let op = std::mem::size_of::<cudarc::cublas::sys::cublasOperation_t>();
    let mode = std::mem::size_of::<lt::cublasLtMatmulMatrixScale_t>();
    if !(set(
        A::CUBLASLT_MATMUL_DESC_TRANSA,
        std::ptr::from_ref(&ta).cast(),
        op,
    ) && set(
        A::CUBLASLT_MATMUL_DESC_TRANSB,
        std::ptr::from_ref(&tb).cast(),
        op,
    ) && set(
        A::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
        std::ptr::from_ref(&sa).cast(),
        mode,
    ) && set(
        A::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
        std::ptr::from_ref(&sb).cast(),
        mode,
    )) {
        return None;
    }
    // The heuristic asks for scale pointers; each call sets its own.
    let stand_in = 256usize as *const c_void;
    let pointer = std::mem::size_of::<*const c_void>();
    if !(set(
        A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        std::ptr::from_ref(&stand_in).cast(),
        pointer,
    ) && set(
        A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        std::ptr::from_ref(&stand_in).cast(),
        pointer,
    )) {
        return None;
    }
    // The weights `k x n` column-major (their rows row-major) under op T, the
    // rows `k x m`, the product `n x m`: `[m, n]` row-major.
    use lt::cudaDataType as T;
    for (layout, ty, rows, cols) in [
        (&raw mut plan.a, T::CUDA_R_8F_E4M3, k, n),
        (&raw mut plan.b, T::CUDA_R_8F_E4M3, k, m),
        (&raw mut plan.d, T::CUDA_R_16BF, n, m),
    ] {
        // SAFETY: `layout` points at a field of the local `plan`, which
        // outlives the call; the dims are plain integers.
        if unsafe {
            lt::cublasLtMatrixLayoutCreate(
                layout,
                ty,
                u64::from(rows),
                u64::from(cols),
                i64::from(rows),
            )
        } != OK
        {
            return None;
        }
    }
    let mut pref: lt::cublasLtMatmulPreference_t = std::ptr::null_mut();
    // SAFETY: `pref` is a local out-pointer.
    if unsafe { lt::cublasLtMatmulPreferenceCreate(&raw mut pref) } != OK {
        return None;
    }
    // SAFETY: the heuristic result is a C struct of the opaque algo words,
    // integers, a status enum whose zero is SUCCESS and a float; all-zero
    // bytes are a valid value.
    let mut found: lt::cublasLtMatmulHeuristicResult_t = unsafe { core::mem::zeroed() };
    let mut returned = 0;
    // SAFETY: `pref` was just created and `workspace` is a local usize of
    // the size passed. `handle` is the live handle `plan_for` keeps for this
    // device, the descriptors were all created above, and `found` and
    // `returned` are locals with room for the one result asked for.
    let asked = unsafe {
        lt::cublasLtMatmulPreferenceSetAttribute(
            pref,
            lt::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            std::ptr::from_ref(&workspace).cast(),
            std::mem::size_of::<usize>(),
        ) == OK
            && lt::cublasLtMatmulAlgoGetHeuristic(
                handle,
                plan.desc,
                plan.a,
                plan.b,
                plan.d,
                plan.d,
                pref,
                1,
                &raw mut found,
                &raw mut returned,
            ) == OK
    };
    // SAFETY: the preference created above, used by nothing after here.
    let _ = unsafe { lt::cublasLtMatmulPreferenceDestroy(pref) };
    if !asked || returned < 1 {
        // SAFETY: takes no arguments; it clears a sticky error a refused
        // heuristic may leave.
        let _ = unsafe { cudarc::runtime::sys::cudaGetLastError() };
        return None;
    }
    plan.algo = found.algo;
    Some(plan)
}

/// The handle and plan for this device's `[m, k] x [n, k]^T`, built once.
fn plan_for(
    m: u32,
    n: u32,
    k: u32,
    workspace: usize,
    mx: bool,
) -> Option<(lt::cublasLtHandle_t, Arc<Mutex<Plan>>)> {
    let mut st = state().lock().expect("lock poisoned");
    let dev = device();
    let handle = match st.handles.get(&dev) {
        Some(&h) => h as lt::cublasLtHandle_t,
        None => {
            let mut h: lt::cublasLtHandle_t = std::ptr::null_mut();
            // SAFETY: `h` is a local out-pointer; the handle it receives is
            // kept in `st.handles` for the process and never destroyed.
            if unsafe { lt::cublasLtCreate(&raw mut h) } != OK || h.is_null() {
                return None;
            }
            st.handles.insert(dev, h as usize);
            h
        }
    };
    let plan = st
        .plans
        .entry((dev, m, n, k, workspace, mx))
        .or_insert_with(|| build(handle, m, n, k, workspace, mx).map(|p| Arc::new(Mutex::new(p))))
        .clone()?;
    Some((handle, plan))
}

/// A plane's block scales in cuBLASLt's `BLK128x128_32F` layout, converted
/// once from its atoms and kept by the atoms' address: a block-FP8 plane
/// (codes and atoms, two planes) is never a rotated tenant, so its atoms sit
/// where the load put them. A miss converts on a thread of its own, on its own
/// stream, so a capturing stream's graph never holds the conversion: every
/// graph reads the one finished copy.
fn weight_scales(op: &'static str, atoms: u64, n: u32, k: u32) -> Result<u64, Error> {
    use cudarc::runtime::sys as rt;
    /// The converted scales by device, atoms' address, `n` and `k`.
    type Converted = Mutex<HashMap<(i32, u64, u32, u32), u64>>;
    static SCALES: OnceLock<Converted> = OnceLock::new();
    let dev = device();
    let key = (dev, atoms, n, k);
    let mut held = SCALES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("lock poisoned");
    if let Some(&at) = held.get(&key) {
        return Ok(at);
    }
    let kbp = (k / 128).div_ceil(4) * 4;
    let bytes = (n / 128) as usize * kbp as usize * 4;
    let made = std::thread::scope(|scope| {
        scope
            .spawn(|| -> Result<u64, Error> {
                let fail = |what: &str, code: rt::cudaError_t| {
                    refuse(
                        op,
                        format!("`{what}` answered {} converting the scales", code as i32),
                    )
                };
                // SAFETY: the device the caller runs on, a fresh allocation
                // and stream this thread owns, a launch it waits out.
                unsafe {
                    let code = rt::cudaSetDevice(dev);
                    if code != rt::cudaError::cudaSuccess {
                        return Err(fail("cudaSetDevice", code));
                    }
                    let mut at: *mut c_void = std::ptr::null_mut();
                    let code = rt::cudaMalloc(&raw mut at, bytes.max(4));
                    if code != rt::cudaError::cudaSuccess {
                        return Err(fail("cudaMalloc", code));
                    }
                    let mut stream: rt::cudaStream_t = std::ptr::null_mut();
                    let code =
                        rt::cudaStreamCreateWithFlags(&raw mut stream, rt::cudaStreamNonBlocking);
                    if code != rt::cudaError::cudaSuccess {
                        return Err(fail("cudaStreamCreateWithFlags", code));
                    }
                    let ctx = Ctx::on(stream.cast());
                    let fired = ctx.fire(
                        op,
                        Fire::at(kernel(ACT_QUANT, "::pie::linear::fp8b128_atoms_to_lt")).apply(
                            Launch::grid([((n / 128) * kbp).div_ceil(256), 1, 1], [256, 1, 1]),
                        ),
                        &[
                            ArgValue::Ptr(atoms),
                            ArgValue::Ptr(at as u64),
                            stated(op, n)?.arg(),
                            stated(op, k)?.arg(),
                        ],
                    );
                    let code = rt::cudaStreamSynchronize(stream);
                    let _ = rt::cudaStreamDestroy(stream);
                    fired?;
                    if code != rt::cudaError::cudaSuccess {
                        return Err(fail("cudaStreamSynchronize", code));
                    }
                    Ok(at as u64)
                }
            })
            .join()
            .unwrap_or_else(|_| Err(refuse(op, "the scales' conversion thread panicked")))
    })?;
    held.insert(key, made);
    Ok(made)
}

/// `y = q(act) . (W 2^(s - 127))^T` for an `n x k` 128x128-block FP8 plane
/// (`codes`, its exponents in tcgen05's atoms at `atoms`), `q` the rows'
/// e4m3 1x128 quantization. `Ok(false)` for a shape cuBLASLt does not take.
pub fn matmul(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: u64,
    atoms: u64,
    y: &mut Tensor,
) -> Result<bool, Error> {
    let (m, n, k) = (act.rows, y.width, act.width);
    if act.dtype != Dtype::Bf16
        || y.dtype != Dtype::Bf16
        || y.rows != m
        || m == 0
        || !n.is_multiple_of(128)
        || !k.is_multiple_of(128)
    {
        return Ok(false);
    }
    // cuBLASLt takes the rows in whole fours; the padding rows are zero.
    let mpad = m.div_ceil(4) * 4;
    let workspace = (ctx.tuning().gemm_workspace_mb as usize) << 20;
    let Some((handle, plan)) = plan_for(mpad, n, k, workspace, false) else {
        return Ok(false);
    };
    let blocks = k / 128;
    let scratch = |name, bytes: u64| -> Result<u64, Error> {
        Ok(ctx.scratch_stream(op, name, usize::try_from(bytes).unwrap_or(usize::MAX))? as u64)
    };
    let xq = scratch("linear.lt_fp8_x", u64::from(mpad) * u64::from(k))?;
    let xs = scratch("linear.lt_fp8_xs", u64::from(blocks) * u64::from(mpad) * 4)?;
    let ws_scales = weight_scales(op, atoms, n, k)?;
    let d = if mpad == m {
        y.ptr
    } else {
        scratch("linear.lt_fp8_d", u64::from(mpad) * u64::from(n) * 2)?
    };
    let lanes = u64::from(mpad) * u64::from(blocks) * 16;
    ctx.fire(
        op,
        Fire::at(kernel(
            ACT_QUANT,
            "::pie::linear::quantize_rows_e4m3b128_lt",
        ))
        .apply(Launch::grid(
            [u32::try_from(lanes.div_ceil(128)).unwrap_or(u32::MAX), 1, 1],
            [128, 1, 1],
        )),
        &[
            act.arg(),
            ArgValue::Ptr(xq),
            ArgValue::Ptr(xs),
            stated(op, m)?.arg(),
            stated(op, mpad)?.arg(),
            stated(op, k)?.arg(),
        ],
    )?;
    run(
        ctx, op, handle, &plan, workspace, codes, ws_scales, xq, xs, d, y, m, mpad, n, k,
    )
}

/// The plan's product of `codes` (scales at `a_scales`) and the quantized rows
/// `xq` (scales at `xs`) into `d`, its first `m` rows copied to `y` where padded.
#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &Ctx,
    op: &'static str,
    handle: lt::cublasLtHandle_t,
    plan: &Mutex<Plan>,
    workspace: usize,
    codes: u64,
    ws_scales: u64,
    xq: u64,
    xs: u64,
    d: u64,
    y: &mut Tensor,
    m: u32,
    mpad: u32,
    n: u32,
    k: u32,
) -> Result<bool, Error> {
    let _ = mpad;
    let plan = plan.lock().expect("plan lock poisoned");
    let slab = ctx.scratch_stream(op, LT_WORKSPACE, workspace)?;
    use lt::cublasLtMatmulDescAttributes_t as A;
    let pointer = std::mem::size_of::<*const c_void>();
    let (a_scale, b_scale) = (
        ws_scales as usize as *const c_void,
        xs as usize as *const c_void,
    );
    let (alpha, beta) = (1.0f32, 0.0f32);
    // SAFETY: `plan` is locked for the whole block, so no other call sets
    // its scale pointers between the sets and the matmul that reads them;
    // each set passes a local pointer of `pointer` bytes. `handle` is this
    // device's live handle, the descriptors are the plan's, `alpha`/`beta`
    // are host f32 locals as the 32F compute type takes, and `slab` is a
    // `workspace`-byte scratch slab on `ctx`. `xq`, `xs` and `d` are slabs
    // sized above for `mpad` rows; `codes` and the scales from `atoms` are
    // the caller's weight plane, its size the caller's to vouch for.
    let status = unsafe {
        if lt::cublasLtMatmulDescSetAttribute(
            plan.desc,
            A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            std::ptr::from_ref(&a_scale).cast(),
            pointer,
        ) != OK
            || lt::cublasLtMatmulDescSetAttribute(
                plan.desc,
                A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                std::ptr::from_ref(&b_scale).cast(),
                pointer,
            ) != OK
        {
            return Err(refuse(op, "cuBLASLt refused the block scales' pointers"));
        }
        lt::cublasLtMatmul(
            handle,
            plan.desc,
            std::ptr::from_ref(&alpha).cast(),
            codes as usize as *const c_void,
            plan.a,
            xq as usize as *const c_void,
            plan.b,
            std::ptr::from_ref(&beta).cast(),
            d as usize as *const c_void,
            plan.d,
            d as usize as *mut c_void,
            plan.d,
            &raw const plan.algo,
            slab,
            workspace,
            ctx.stream().cast(),
        )
    };
    if status != OK {
        return Err(refuse(
            op,
            format!("cuBLASLt's block-scaled FP8 GEMM answered {status:?} at M={m} N={n} K={k}"),
        ));
    }
    if d != y.ptr {
        // The padded product's first `m` rows are the output's.
        // SAFETY: `d` is the `mpad * n` bf16 scratch slab the matmul wrote
        // and `m <= mpad`; `y` is the caller's `m x n` bf16 output. Both are
        // on `ctx`'s stream, live by `Ctx::on`'s contract.
        let copied = unsafe {
            cudaMemcpyAsync(
                y.ptr as usize as *mut c_void,
                d as usize as *const c_void,
                m as usize * n as usize * 2,
                cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                ctx.stream().cast(),
            )
        };
        if copied != cudarc::runtime::sys::cudaError::cudaSuccess {
            return Err(refuse(
                op,
                format!("the padded product's copy answered {copied:?}"),
            ));
        }
    }
    Ok(true)
}

/// `y = q(act) . W^T` for an `n x k` MXFP8 plane (`codes`, a ue8m0 exponent a
/// 1x32 block in tcgen05's atoms at `atoms`), `q` the rows' e4m3 1x32 (MXFP8)
/// quantization, as the checkpoint's activations are declared: cuBLASLt's
/// block-scaled FP8 GEMM where there is no tcgen05 one (sm_120). `Ok(false)`
/// for a shape or device cuBLASLt does not take.
pub fn matmul_mx(
    ctx: &Ctx,
    op: &'static str,
    act: Tensor,
    codes: u64,
    atoms: u64,
    y: &mut Tensor,
) -> Result<bool, Error> {
    use crate::linear::act_quant;
    let (m, n, k) = (act.rows, y.width, act.width);
    if act.dtype != Dtype::Bf16
        || y.dtype != Dtype::Bf16
        || y.rows != m
        || m == 0
        || !n.is_multiple_of(128)
        || !k.is_multiple_of(128)
    {
        return Ok(false);
    }
    // the rows in whole 128s, the scales' tile: the padding rows' products are dropped
    let mpad = m.div_ceil(128) * 128;
    let workspace = (ctx.tuning().gemm_workspace_mb as usize) << 20;
    let Some((handle, plan)) = plan_for(mpad, n, k, workspace, true) else {
        return Ok(false);
    };
    let scratch = |name, bytes: u64| -> Result<u64, Error> {
        Ok(ctx.scratch_stream(op, name, usize::try_from(bytes).unwrap_or(usize::MAX))? as u64)
    };
    let xq = scratch("linear.lt_mx_x", u64::from(mpad) * u64::from(k))?;
    let atom_bytes = (mpad / 128) as usize * (k / 128) as usize * 512;
    let xs = act_quant::atoms_of(ctx, op, "linear.lt_mx_xs", atom_bytes)?;
    act_quant::quantize(
        ctx,
        op,
        act_quant::Format::E4m3 { gran: 32 },
        act_quant::Source::rows(act, None),
        xq,
        xs,
        atom_bytes,
    )?;
    let d = if mpad == m {
        y.ptr
    } else {
        scratch("linear.lt_mx_d", u64::from(mpad) * u64::from(n) * 2)?
    };
    run(
        ctx, op, handle, &plan, workspace, codes, atoms, xq, xs, d, y, m, mpad, n, k,
    )
}
