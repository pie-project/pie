use std::sync::OnceLock;

use crate::runtime::{ArgValue, Args, Launch, Stream, cache};

/// Warps per block in the row-per-warp form — `gemv.cu:329`'s
const WARPS: u32 = 4;

/// Warps per block in the split-K form on Blackwell — `gemv.cu:342`'s
const SPLIT_WARPS_B: u32 = 4;

/// Warps per block in the split-K form everywhere else — `gemv.cu:352`'s
const SPLIT_WARPS: u32 = 8;

/// A warp, which is the first block axis of all four launches.
const WARP_LANES: u32 = 32;

/// The row count below which K is split INSIDE the block — `gemv.cu:317`'s
const SPLIT_K_MAX_ROWS: i32 = 4096;

/// The largest grid the row-per-warp form will open — `gemv.cu:381`'s
const MAX_BLOCKS: i64 = 2_147_483_647;

/// The split-K row, four warps, unroll 2 — [`super::GEMV`]'s first row.
const SPLITK_W4_U2: &str = "gemm::gemv_splitk_bf16_w4_u2";

/// The split-K row, eight warps, unroll 1 — `GEMV_SIGS[1]`.
const SPLITK_W8_U1: &str = "gemm::gemv_splitk_bf16_w8_u1";

/// The row-per-warp row, four warps, unroll 2 — `GEMV_SIGS[2]`.
const ROW_W4_U2: &str = "gemm::gemv_bf16_w4_u2";

/// The row-per-warp row, four warps, unroll 4 — `GEMV_SIGS[3]`.
const ROW_W4_U4: &str = "gemm::gemv_bf16_w4_u4";

/// Why [`gemv_bf16`] did not launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decline {
    /// `N <= 0`, `K <= 0`, or `K % 8 != 0` — `gemv.cu:311`.
    Shape,
    /// `weight`, `act` or `out` was null — `gemv.cu:312`.
    Null,
    /// `weight` or `act` was not 16-byte aligned — `gemv.cu:313`.
    Misaligned,
    /// The row-per-warp grid would not fit — `gemv.cu:381`. See `MAX_BLOCKS`
    Grid,
}

/// What [`gemv_bf16`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub enum Gemv {
    /// The launch is on the stream. Exactly one kernel was enqueued.
    Launched,
    /// Nothing was enqueued. **Use cuBLAS for this shape.**
    Declined(Decline),
}

/// How deep to unroll the row walk: 2 on Blackwell and later, 4 below.
fn unroll_depth() -> i32 {
    static DEPTH: OnceLock<i32> = OnceLock::new();
    *DEPTH.get_or_init(|| {
        use cudarc::driver::sys as dr;
        use cudarc::runtime::sys as rt;

        let mut ordinal: i32 = 0;
        // SAFETY: `ordinal` is a live, writable out-parameter for the call.
        let code = unsafe { rt::cudaGetDevice(&raw mut ordinal) };
        if code != rt::cudaError::cudaSuccess {
            return 4;
        }
        if cudarc::driver::result::init().is_err() {
            return 4;
        }
        let mut device: dr::CUdevice = 0;
        // SAFETY: `device` is a valid, writable handle slot, and the driver
        let code = unsafe { dr::cuDeviceGet(&raw mut device, ordinal) };
        if code != dr::CUresult::CUDA_SUCCESS {
            return 4;
        }
        let mut major: i32 = 0;
        // SAFETY: `major` is valid and writable; `device` came from
        let code = unsafe {
            dr::cuDeviceGetAttribute(
                &raw mut major,
                dr::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device,
            )
        };
        if code != dr::CUresult::CUDA_SUCCESS {
            return 4;
        }
        if major >= 10 { 2 } else { 4 }
    })
}

/// Single-row bf16 GEMV: `out[n] = sum_k W[n][k] * x[k] + bias[n] + beta * out[n]`.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn gemv_bf16(
    weight: *const std::ffi::c_void,
    act: *const std::ffi::c_void,
    bias: *const std::ffi::c_void,
    out: *mut std::ffi::c_void,
    n: i32,
    k: i32,
    stream: *mut std::ffi::c_void,
    beta: f32,
) -> Gemv {
    if n <= 0 || k <= 0 || k % 8 != 0 {
        return Gemv::Declined(Decline::Shape);
    }
    if weight.is_null() || act.is_null() || out.is_null() {
        return Gemv::Declined(Decline::Null);
    }
    if !aligned16(weight) || !aligned16(act) {
        return Gemv::Declined(Decline::Misaligned);
    }

    let values = [
        ArgValue::Ptr(weight.cast_mut()),
        ArgValue::Ptr(act.cast_mut()),
        ArgValue::Ptr(bias.cast_mut()),
        ArgValue::Ptr(out),
        ArgValue::I32(n),
        ArgValue::I32(k),
        ArgValue::F32(beta),
    ];

    if n <= SPLIT_K_MAX_ROWS {
        let (symbol, warps) = if unroll_depth() == 2 {
            (SPLITK_W4_U2, SPLIT_WARPS_B)
        } else {
            (SPLITK_W8_U1, SPLIT_WARPS)
        };
        fire(
            symbol,
            Launch {
                grid: [n.unsigned_abs(), 1, 1],
                block: [WARP_LANES, warps, 1],
                smem: 0,
            },
            &values,
            stream,
        );
        return Gemv::Launched;
    }

    let warps = i64::from(WARPS);
    let blocks = (i64::from(n) + warps - 1) / warps;
    if blocks > MAX_BLOCKS {
        return Gemv::Declined(Decline::Grid);
    }
    let Ok(grid_x) = u32::try_from(blocks) else {
        return Gemv::Declined(Decline::Grid);
    };

    let symbol = if unroll_depth() == 2 { ROW_W4_U2 } else { ROW_W4_U4 };
    fire(
        symbol,
        Launch {
            grid: [grid_x, 1, 1],
            block: [WARP_LANES, WARPS, 1],
            smem: 0,
        },
        &values,
        stream,
    );
    Gemv::Launched
}

/// `gemv.cu:299` — `(reinterpret_cast<std::uintptr_t>(p) & 15u) == 0`.
fn aligned16(p: *const std::ffi::c_void) -> bool {
    p.addr() & 15 == 0
}

/// Resolve one row through the JIT table, bind the operands, launch.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
fn fire(symbol: &'static str, launch: Launch, values: &[ArgValue], stream: *mut std::ffi::c_void) {
    let Some((index, unit)) = crate::unit::unit_of(symbol) else {
        panic!("{symbol} is in no JIT unit — this driver and its kernel table disagree");
    };
    let Some(sig) = unit.row(symbol).map(|row| row.sig) else {
        panic!("{symbol} named unit `{}` and is not one of its rows", unit.name);
    };
    let module = match cache::module(index, unit) {
        Ok(module) => module,
        Err(why) => panic!("{symbol}: unit `{}` would not compile or load: {why}", unit.name),
    };
    let mut args = match Args::bind(sig, values) {
        Ok(args) => args,
        Err(why) => panic!("{symbol}: {why}"),
    };
    // SAFETY: the caller holds the fire's stream live across the launch — the
    let stream = unsafe { Stream::from_runtime(stream) };
    if let Err(why) = module.fire(sig, launch, &mut args, stream) {
        panic!("{symbol}: {why}");
    }
}
