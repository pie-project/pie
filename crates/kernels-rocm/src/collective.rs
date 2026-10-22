use crate::error::Error;

use crate::jit::{Ctx, refuse};
use crate::tensor::Tensor;

/// Bytes of one `vllm::Signal`, rounded up to what a rank allocates.
pub const SIGNAL_BYTES: u64 = 4096;

/// A rank's view of a group whose ranks map one another's memory; its
/// collectives read the peers' stages directly instead of going through NCCL.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Peers {
    pub rank: u32,
    pub world: u32,
    pub stage: u64,
    pub stage_bytes: u64,
    pub signal: u64,
    /// `vllm::RankData`: every rank's stage, in rank order.
    pub stages: u64,
    /// `vllm::RankSignals`: every rank's signal block, in rank order.
    pub signals: u64,
}

impl Peers {
    /// No peer kernels on ROCm: the handshake refuses, and NCCL carries the
    /// group's messages.
    pub fn answer(&self, _ctx: &Ctx, _status: u64) -> Result<(), Error> {
        Err(Error::Unsupported {
            op: "collective.peers_answer",
        })
    }
}

pub fn all_reduce(ctx: &Ctx, buf: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "collective.all_reduce";
    let comm = ctx.comm(OP)?;

    #[cfg(feature = "rocm")]
    {
        let nccl = rccl(OP)?;
        let dtype = wire_dtype(OP, buf.dtype)?;
        let Some((send, count)) = message(*buf) else {
            return Ok(());
        };
        let code = unsafe {
            (nccl.ncclAllReduce)(
                send,
                send.cast_mut(),
                count,
                dtype,
                sys::NCCL_SUM,
                comm,
                ctx.stream(),
            )
        };
        answered(OP, "ncclAllReduce", code)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (comm, buf);
        Err(crate::jit::runtimeless(OP))
    }
}

pub fn all_gather(ctx: &Ctx, x: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "collective.all_gather";
    let comm = ctx.comm(OP)?;
    debug_assert_eq!(x.dtype, y.dtype, "a gather does not change the dtype");
    debug_assert!(
        x.elements() > 0 && y.elements().is_multiple_of(x.elements()),
        "the gathered rectangle is a whole number of shards"
    );

    #[cfg(feature = "rocm")]
    {
        let nccl = rccl(OP)?;
        let dtype = wire_dtype(OP, x.dtype)?;
        let Some((send, count)) = message(x) else {
            return Ok(());
        };

        if x.rows <= 1 {
            let code = unsafe {
                (nccl.ncclAllGather)(
                    send,
                    y.ptr as usize as *mut core::ffi::c_void,
                    count,
                    dtype,
                    comm,
                    ctx.stream(),
                )
            };
            return answered(OP, "ncclAllGather", code);
        }

        let world = u32::try_from(y.elements() / x.elements()).map_err(|_| {
            refuse(
                OP,
                "the gathered rectangle is more shards than a u32 counts",
            )
        })?;
        let bytes = usize::try_from(y.elements().saturating_mul(y.dtype.bytes_ceil()))
            .map_err(|_| refuse(OP, "the gathered rectangle does not fit this address space"))?;
        let stage = ctx.scratch(OP, "all_gather_stage", bytes)?;
        let code = unsafe { (nccl.ncclAllGather)(send, stage, count, dtype, comm, ctx.stream()) };
        answered(OP, "ncclAllGather", code)?;
        crate::gather::rows::gather_width_concat(ctx, stage as usize as u64, y, x.width, world)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = comm;
        Err(crate::jit::runtimeless(OP))
    }
}

/// Runs `body`'s collectives as one NCCL group, so several small gathers on
/// the same communicator and stream go out as one launch instead of one each.
pub fn grouped<R>(op: &'static str, body: impl FnOnce() -> Result<R, Error>) -> Result<R, Error> {
    #[cfg(feature = "rocm")]
    {
        let nccl = rccl(op)?;
        answered(op, "ncclGroupStart", unsafe { (nccl.ncclGroupStart)() })?;
        let ran = body();
        let ended = answered(op, "ncclGroupEnd", unsafe { (nccl.ncclGroupEnd)() });
        let out = ran?;
        ended?;
        Ok(out)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = op;
        body()
    }
}

pub fn reduce_scatter(ctx: &Ctx, x: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "collective.reduce_scatter";
    let comm = ctx.comm(OP)?;
    if x.rows > 1 {
        return Err(refuse(
            OP,
            format!(
                "this scatter keeps each rank its columns of every row (the IR shapes it \
                 `[rows, width / world]`) and ncclReduceScatter hands each rank a \
                 contiguous block of the flat buffer; the two are the same layout only at \
                 one row, and this fire brought {}. Scattering wider wants a permute before \
                 the collective, which is not built.",
                x.rows,
            ),
        ));
    }
    debug_assert_eq!(x.dtype, y.dtype, "a reduction does not change the dtype");
    debug_assert!(
        y.elements() > 0 && x.elements().is_multiple_of(y.elements()),
        "the reduced rectangle is a whole number of shards"
    );

    #[cfg(feature = "rocm")]
    {
        let nccl = rccl(OP)?;
        let dtype = wire_dtype(OP, x.dtype)?;
        let Some((recv, count)) = message(*y) else {
            return Ok(());
        };
        let code = unsafe {
            (nccl.ncclReduceScatter)(
                x.ptr as usize as *const core::ffi::c_void,
                recv.cast_mut(),
                count,
                dtype,
                sys::NCCL_SUM,
                comm,
                ctx.stream(),
            )
        };
        answered(OP, "ncclReduceScatter", code)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = comm;
        Err(crate::jit::runtimeless(OP))
    }
}

#[cfg(feature = "rocm")]
use crate::hip::sys;

#[cfg(feature = "rocm")]
pub(crate) fn rccl(op: &'static str) -> Result<&'static sys::Rccl, Error> {
    sys::rccl().map_err(|detail| refuse(op, detail))
}

#[cfg(feature = "rocm")]
fn wire_dtype(op: &'static str, dtype: dtype::Dtype) -> Result<sys::NcclDataType, Error> {
    Ok(crate::jit::dtype_dispatch!(op, dtype, {
        Bf16 => sys::NCCL_BFLOAT16,
        F16 => sys::NCCL_FLOAT16,
        F32 => sys::NCCL_FLOAT32,
    }))
}

#[cfg(feature = "rocm")]
fn message(t: Tensor) -> Option<(*const core::ffi::c_void, usize)> {
    let count = usize::try_from(t.elements()).ok()?;
    (count > 0).then_some((t.ptr as usize as *const core::ffi::c_void, count))
}

#[cfg(feature = "rocm")]
pub(crate) fn answered(
    op: &'static str,
    call: &'static str,
    code: sys::NcclResult,
) -> Result<(), Error> {
    if code == sys::NCCL_SUCCESS {
        return Ok(());
    }
    Err(crate::jit::Fault::Device { call, code }.at(op))
}
