#![allow(unsafe_code)]
use crate::error::Error;

use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, kernel, refuse};
use crate::tensor::Tensor;

const FILE: &str = "collective/all_reduce.cu";

const THREADS: u32 = 512;

/// `p2p::kMaxBlocks`: the signal block holds one flag row per block.
const MAX_BLOCKS: u64 = 36;

/// Bytes of one `p2p::Signal`, rounded up to what a rank allocates.
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
    /// `p2p::RankData`: every rank's stage, in rank order.
    pub stages: u64,
    /// `p2p::RankSignals`: every rank's signal block, in rank order.
    pub signals: u64,
}

impl Peers {
    fn carries(&self, t: &Tensor) -> bool {
        t.dtype == dtype::Dtype::Bf16 && aligned16(t.ptr) && (t.elements() * 2).is_multiple_of(16)
    }

    fn entry(&self, reduce: bool) -> Option<&'static str> {
        Some(match (reduce, self.world) {
            (true, 2) => "::pie::collective::all_reduce_peers<__nv_bfloat16, 2>",
            (false, 2) => "::pie::collective::all_gather_peers<__nv_bfloat16, 2>",
            (true, 4) => "::pie::collective::all_reduce_peers<__nv_bfloat16, 4>",
            (false, 4) => "::pie::collective::all_gather_peers<__nv_bfloat16, 4>",
            (true, 8) => "::pie::collective::all_reduce_peers<__nv_bfloat16, 8>",
            (false, 8) => "::pie::collective::all_gather_peers<__nv_bfloat16, 8>",
            _ => return None,
        })
    }

    /// Launches this rank's half of the check `peers_answer` makes; the
    /// verdict lands in `status`, 0 when every peer answered.
    pub fn answer(&self, ctx: &Ctx, status: u64) -> Result<(), Error> {
        ctx.fire(
            "collective.peers_answer",
            Fire::at(kernel(FILE, "::pie::collective::peers_answer"))
                .apply(Launch::grid([1, 1, 1], [1, 1, 1])),
            &[
                ArgValue::Ptr(self.stages),
                ArgValue::Ptr(self.signals),
                ArgValue::Ptr(self.signal),
                ArgValue::I32(self.rank as i32),
                ArgValue::I32(self.world as i32),
                ArgValue::Ptr(status),
            ],
        )
    }

    fn launch(packs: u64) -> Launch {
        let blocks = packs.div_ceil(u64::from(THREADS)).clamp(1, MAX_BLOCKS) as u32;
        Launch::grid([blocks, 1, 1], [THREADS, 1, 1])
    }

    fn all_reduce(&self, ctx: &Ctx, op: &'static str, buf: &Tensor) -> Result<(), Error> {
        let entry = self
            .entry(true)
            .ok_or_else(|| refuse(op, "no peer kernel for this world"))?;
        let total = buf.elements() * 2;
        let chunk = self.stage_bytes & !15;
        let mut done = 0;
        while done < total {
            let bytes = chunk.min(total - done);
            copy(ctx, op, self.stage, buf.ptr + done, bytes)?;
            let packs = bytes / 16;
            ctx.fire(
                op,
                Fire::at(kernel(FILE, entry)).apply(Self::launch(packs)),
                &[
                    ArgValue::Ptr(self.stages),
                    ArgValue::Ptr(self.signals),
                    ArgValue::Ptr(self.signal),
                    ArgValue::Ptr(buf.ptr + done),
                    ArgValue::I32(self.rank as i32),
                    ArgValue::I32(
                        i32::try_from(packs).map_err(|_| refuse(op, "a chunk past i32"))?,
                    ),
                ],
            )?;
            done += bytes;
        }
        Ok(())
    }

    fn all_gather(&self, ctx: &Ctx, op: &'static str, x: &Tensor, y: &Tensor) -> Result<(), Error> {
        let entry = self
            .entry(false)
            .ok_or_else(|| refuse(op, "no peer kernel for this world"))?;
        let row = u64::from(x.width) * 2;
        let per = (self.stage_bytes / row).min(u64::from(x.rows));
        if per == 0 {
            return Err(refuse(op, "one row is wider than the peer stage"));
        }
        let width = i32::try_from(row / 16).map_err(|_| refuse(op, "a row past i32 packs"))?;
        let mut first = 0;
        while first < u64::from(x.rows) {
            let rows = per.min(u64::from(x.rows) - first);
            copy(ctx, op, self.stage, x.ptr + first * row, rows * row)?;
            ctx.fire(
                op,
                Fire::at(kernel(FILE, entry))
                    .apply(Self::launch(rows * row / 16 * u64::from(self.world))),
                &[
                    ArgValue::Ptr(self.stages),
                    ArgValue::Ptr(self.signals),
                    ArgValue::Ptr(self.signal),
                    ArgValue::Ptr(y.ptr + first * row * u64::from(self.world)),
                    ArgValue::I32(self.rank as i32),
                    ArgValue::I32(rows as i32),
                    ArgValue::I32(width),
                ],
            )?;
            first += rows;
        }
        Ok(())
    }
}

fn copy(ctx: &Ctx, op: &'static str, dst: u64, src: u64, bytes: u64) -> Result<(), Error> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::runtime::sys as rt;

        // SAFETY: the stream is `ctx`'s, live by `Ctx::on`'s contract. The
        // callers bound `bytes` by the peer stage's `stage_bytes`, so `dst`
        // stays inside the stage `Ctx::with_peers` vouched for; `src` is a
        // range of the caller's tensor.
        let code = unsafe {
            rt::cudaMemcpyAsync(
                dst as usize as *mut core::ffi::c_void,
                src as usize as *const core::ffi::c_void,
                bytes as usize,
                rt::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                ctx.stream().cast(),
            )
        };
        if code != rt::cudaError::cudaSuccess {
            return Err(refuse(
                op,
                format!(
                    "`cudaMemcpyAsync` answered {} staging a peer message",
                    code as i32
                ),
            ));
        }
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ctx, dst, src, bytes);
        Err(crate::jit::runtimeless(op))
    }
}

pub fn all_reduce(ctx: &Ctx, buf: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "collective.all_reduce";
    if let Some(peers) = ctx.peers()
        && peers.carries(buf)
    {
        return peers.all_reduce(ctx, OP, buf);
    }
    let comm = ctx.comm(OP)?;

    #[cfg(feature = "cuda")]
    {
        use cudarc::nccl::sys as nccl;

        let dtype = wire_dtype(OP, buf.dtype)?;
        let Some((send, count)) = message(*buf) else {
            return Ok(());
        };
        // SAFETY: `comm` is the live communicator `Ctx::with_comm` vouched
        // for and the stream is `ctx`'s. The reduce is in place over the
        // `count` elements of `buf`, a device allocation the caller owns.
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

pub fn all_gather(ctx: &Ctx, x: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "collective.all_gather";
    if let Some(peers) = ctx.peers()
        && peers.carries(&x)
        && x.width.is_multiple_of(8)
        && aligned16(y.ptr)
        && y.rows == x.rows
        && y.width == x.width * peers.world
    {
        return peers.all_gather(ctx, OP, &x, y);
    }
    let comm = ctx.comm(OP)?;
    assert_eq!(x.dtype, y.dtype, "a gather does not change the dtype");
    assert!(
        x.elements() > 0 && y.elements().is_multiple_of(x.elements()),
        "the gathered rectangle is a whole number of shards"
    );

    #[cfg(feature = "cuda")]
    {
        use cudarc::nccl::sys as nccl;

        let dtype = wire_dtype(OP, x.dtype)?;
        let Some((send, count)) = message(x) else {
            return Ok(());
        };

        if x.rows <= 1 {
            // SAFETY: `comm` and the stream are `ctx`'s, live by its
            // contracts. NCCL reads `count` elements of `x` and writes
            // `count * world` into `y`; that `y` is that large is the
            // caller's to keep beyond the assertion above, which holds it to whole shards.
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
        // SAFETY: `comm` and the stream are `ctx`'s. `stage` is a slab of
        // `y.elements()` elements, which is `count * world` by the
        // division above; `send` is `x`'s `count` elements.
        let code = unsafe {
            nccl::ncclAllGather(send, stage, count, dtype, comm.cast(), ctx.stream().cast())
        };
        answered(OP, "ncclAllGather", code)?;
        crate::gather::gather_width_concat(ctx, stage as usize as u64, y, x.width, world)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = comm;
        Err(crate::jit::runtimeless(OP))
    }
}

/// Runs `body`'s collectives as one NCCL group, so several small gathers on
/// the same communicator and stream go out as one launch instead of one each.
pub fn grouped<R>(op: &'static str, body: impl FnOnce() -> Result<R, Error>) -> Result<R, Error> {
    #[cfg(feature = "cuda")]
    {
        use cudarc::nccl::sys as nccl;

        // SAFETY: takes no arguments; it opens a group on this thread that
        // the `ncclGroupEnd` below closes whatever `body` returns.
        answered(op, "ncclGroupStart", unsafe { nccl::ncclGroupStart() })?;
        let ran = body();
        // SAFETY: closes the group opened above, on the same thread.
        let ended = answered(op, "ncclGroupEnd", unsafe { nccl::ncclGroupEnd() });
        let out = ran?;
        ended?;
        Ok(out)
    }
    #[cfg(not(feature = "cuda"))]
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
    assert_eq!(x.dtype, y.dtype, "a reduction does not change the dtype");
    assert!(
        y.elements() > 0 && x.elements().is_multiple_of(y.elements()),
        "the reduced rectangle is a whole number of shards"
    );

    #[cfg(feature = "cuda")]
    {
        use cudarc::nccl::sys as nccl;

        let dtype = wire_dtype(OP, x.dtype)?;
        let Some((recv, count)) = message(*y) else {
            return Ok(());
        };
        // SAFETY: `comm` and the stream are `ctx`'s. NCCL writes `count`
        // elements of `y` and reads `count * world` of `x`; that `x` holds
        // that many is the caller's to keep beyond the assertion above,
        // which holds it to whole shards.
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

#[cfg(feature = "cuda")]
fn message(t: Tensor) -> Option<(*const core::ffi::c_void, usize)> {
    let count = usize::try_from(t.elements()).ok()?;
    (count > 0).then_some((t.ptr as usize as *const core::ffi::c_void, count))
}

#[cfg(feature = "cuda")]
pub(crate) fn answered(
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
