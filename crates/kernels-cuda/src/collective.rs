//! `Collective`: the tensor-parallel collectives, NCCL on the stream. One
//! entry per IR variant; every rank traces the same plan, so a collective
//! here is a sync point of an SPMD fire, enqueued like any launch and never
//! synchronised on.
//!
//! Communicator setup (`ncclGetUniqueId`, rank exchange,
//! `ncclCommInitRank`) is runtime/engine boot business, outside the plan;
//! the [`Ctx`] arrives with the communicator already open, or with none,
//! and a collective on a comm-less context is a typed refusal. A live
//! communicator can only exist because the engine already loaded `libnccl`
//! to build it, so no entry probes for the library.

use crate::error::Error;

use crate::jit::Ctx;
use crate::tensor::Tensor;

/// `buf = Σ_ranks buf`, in place (the IR aliases `buf_out` onto `buf`).
pub fn all_reduce(ctx: &Ctx, buf: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "collective.all_reduce";
    let comm = ctx.comm(OP)?;

    #[cfg(feature = "cuda")]
    {
        use cudarc::nccl::sys as nccl;

        let dtype = wire_dtype(OP, buf.dtype)?;
        let Some((send, count)) = message(*buf) else {
            return Ok(());
        };
        let code = unsafe {
            nccl::ncclAllReduce(
                send,
                send.cast_mut(),
                count,
                dtype,
                nccl::ncclRedOp_t::ncclSum,
                comm.cast(),
                ctx.stream().cast(),
            )
        };
        answered(OP, "ncclAllReduce", code)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (comm, buf);
        Err(crate::jit::runtimeless(OP))
    }
}

/// Concatenates each rank's `x` into `y` on every rank, rank-major.
pub fn all_gather(ctx: &Ctx, x: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "collective.all_gather";
    let comm = ctx.comm(OP)?;
    debug_assert_eq!(x.dtype, y.dtype, "a gather does not change the dtype");
    debug_assert!(
        x.elements() > 0 && y.elements() % x.elements() == 0,
        "the gathered rectangle is a whole number of shards"
    );

    #[cfg(feature = "cuda")]
    {
        use cudarc::nccl::sys as nccl;

        let dtype = wire_dtype(OP, x.dtype)?;
        let Some((send, count)) = message(x) else {
            return Ok(());
        };
        let code = unsafe {
            nccl::ncclAllGather(
                send,
                y.ptr as usize as *mut core::ffi::c_void,
                count,
                dtype,
                comm.cast(),
                ctx.stream().cast(),
            )
        };
        answered(OP, "ncclAllGather", code)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = comm;
        Err(crate::jit::runtimeless(OP))
    }
}

/// Sums `x` across ranks, leaving each rank its own shard in `y`.
pub fn reduce_scatter(ctx: &Ctx, x: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "collective.reduce_scatter";
    let comm = ctx.comm(OP)?;
    debug_assert_eq!(x.dtype, y.dtype, "a reduction does not change the dtype");
    debug_assert!(
        y.elements() > 0 && x.elements() % y.elements() == 0,
        "the reduced rectangle is a whole number of shards"
    );

    #[cfg(feature = "cuda")]
    {
        use cudarc::nccl::sys as nccl;

        let dtype = wire_dtype(OP, x.dtype)?;
        let Some((recv, count)) = message(*y) else {
            return Ok(());
        };
        let code = unsafe {
            nccl::ncclReduceScatter(
                x.ptr as usize as *const core::ffi::c_void,
                recv.cast_mut(),
                count,
                dtype,
                nccl::ncclRedOp_t::ncclSum,
                comm.cast(),
                ctx.stream().cast(),
            )
        };
        answered(OP, "ncclReduceScatter", code)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = comm;
        Err(crate::jit::runtimeless(OP))
    }
}

/// The handle's dtype as NCCL spells it on the wire.
#[cfg(feature = "cuda")]
fn wire_dtype(
    op: &'static str,
    dtype: dtype::Dtype,
) -> Result<cudarc::nccl::sys::ncclDataType_t, Error> {
    use cudarc::nccl::sys::ncclDataType_t as t;

    Ok(crate::jit::dtype_dispatch!(op, dtype, {
        Bf16 => t::ncclBfloat16,
        F16 => t::ncclFloat16,
        F32 => t::ncclFloat32,
    }))
}

/// The rank-local message: its address and element count. `None` for an
/// empty rectangle — a conditioned fire may legitimately move nothing, and
/// a refusal here would kill the whole fire under graph capture.
#[cfg(feature = "cuda")]
fn message(t: Tensor) -> Option<(*const core::ffi::c_void, usize)> {
    let count = usize::try_from(t.elements()).ok()?;
    (count > 0).then_some((t.ptr as usize as *const core::ffi::c_void, count))
}

#[cfg(feature = "cuda")]
fn answered(
    op: &'static str,
    call: &'static str,
    code: cudarc::nccl::sys::ncclResult_t,
) -> Result<(), Error> {
    if code == cudarc::nccl::sys::ncclResult_t::ncclSuccess {
        return Ok(());
    }
    Err(crate::jit::Fault::Device {
        call,
        code: code as i32,
    }
    .at(op))
}
