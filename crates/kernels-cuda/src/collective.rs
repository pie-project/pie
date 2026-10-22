use crate::error::Error;

use crate::jit::{ArgValue, Ctx, Fire, Launch, aligned16, refuse};
use crate::tensor::Tensor;

const FILE: &str = "collective/all_reduce.cuh";

const THREADS: u32 = 512;

/// `vllm::kMaxBlocks`: the signal block holds one flag row per block.
const MAX_BLOCKS: u64 = 36;

/// Bytes of one `vllm::Signal`, rounded up to what a rank allocates.
pub const SIGNAL_BYTES: u64 = 4096;

/// Packs (16 bytes) of one rank's push slot: what one launch of the push
/// kernel's blocks owns, 512 a block.
pub const PUSH_PACKS: u64 = MAX_BLOCKS * 512;

/// Bytes past a rank's stage of its push slots: two alternating sets, one
/// slot for every rank of a group as wide as `world`.
#[must_use]
pub const fn push_bytes(world: u64) -> u64 {
    2 * world * PUSH_PACKS * 16
}

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
    /// The ranks reach one another only through host memory: `stage` is the
    /// group's pinned, mapped block (`host_bytes`: the slots, `stage_bytes`
    /// a gather slot), `stages` its flags there, and `signal` this rank's
    /// epoch counts on its device (`HOST_COUNT_BYTES`).
    pub host: bool,
}

/// Flags a rank holds in each lane of the host path: `kHostBlocks`, one a
/// block.
const HOST_BLOCKS: u64 = 36;

/// Bytes of a rank's slot in the host path's reduce lane: a decode's
/// residuals; past it NCCL reduces.
const HOST_REDUCE_BYTES: u64 = 64 << 10;

/// The host path's two lanes, each with its own slots, flags and epochs, so
/// each always launches the same blocks: a block's slots and epoch are its
/// own, and the packs a block stores are the packs its peer block reads in
/// every call of its lane. Reduces run one block (a few kilobytes and one
/// flag to poll), gathers eight (a vocabulary's logits).
const HOST_REDUCE_GRID: u32 = 1;
const HOST_GATHER_GRID: u32 = 8;

/// Bytes of the pinned block the host path maps for a group as wide as
/// `world` whose gather slots are `slot` bytes: both lanes' slot sets and
/// flags.
#[must_use]
pub const fn host_bytes(world: u64, slot: u64) -> u64 {
    2 * world * (HOST_REDUCE_BYTES + slot) + 2 * world * HOST_BLOCKS * 4
}

/// Where the flags start in that block: past both lanes' slot sets.
#[must_use]
pub const fn host_flags_at(world: u64, slot: u64) -> u64 {
    2 * world * (HOST_REDUCE_BYTES + slot)
}

/// Bytes of a rank's epoch counts on its device: both lanes'.
pub const HOST_COUNT_BYTES: u64 = 2 * HOST_BLOCKS * 4;

impl Peers {
    fn carries(&self, t: &Tensor) -> bool {
        t.dtype == dtype::Dtype::Bf16 && aligned16(t.ptr) && (t.elements() * 2).is_multiple_of(16)
    }

    /// Whether the host path's lane for a reduce (or a gather) holds a
    /// rank's `bytes`; always where the ranks map one another.
    fn holds(&self, reduce: bool, bytes: u64) -> bool {
        !self.host
            || bytes
                <= if reduce {
                    HOST_REDUCE_BYTES
                } else {
                    self.stage_bytes
                }
    }

    /// The lane's launch and its slots, slot bytes, flags and counts.
    fn host_lane(&self, reduce: bool) -> (Launch, u64, u64, u64, u64) {
        let world = u64::from(self.world);
        let flags = world * HOST_BLOCKS * 4;
        if reduce {
            (
                Launch::grid([HOST_REDUCE_GRID, 1, 1], [THREADS, 1, 1]),
                self.stage,
                HOST_REDUCE_BYTES,
                self.stages,
                self.signal,
            )
        } else {
            (
                Launch::grid([HOST_GATHER_GRID, 1, 1], [THREADS, 1, 1]),
                self.stage + 2 * world * HOST_REDUCE_BYTES,
                self.stage_bytes,
                self.stages + flags,
                self.signal + HOST_BLOCKS * 4,
            )
        }
    }

    fn host_entry(&self, reduce: bool) -> Option<&'static str> {
        Some(match (reduce, self.world) {
            (true, 2) => "::pie::collective::all_reduce_host<__nv_bfloat16, 2>",
            (false, 2) => "::pie::collective::all_gather_host<__nv_bfloat16, 2>",
            (true, 4) => "::pie::collective::all_reduce_host<__nv_bfloat16, 4>",
            (false, 4) => "::pie::collective::all_gather_host<__nv_bfloat16, 4>",
            (true, 8) => "::pie::collective::all_reduce_host<__nv_bfloat16, 8>",
            (false, 8) => "::pie::collective::all_gather_host<__nv_bfloat16, 8>",
            _ => return None,
        })
    }

    fn push_entry(&self) -> Option<&'static str> {
        Some(match self.world {
            2 => "::pie::collective::all_reduce_push<__nv_bfloat16, 2>",
            4 => "::pie::collective::all_reduce_push<__nv_bfloat16, 4>",
            8 => "::pie::collective::all_reduce_push<__nv_bfloat16, 8>",
            _ => return None,
        })
    }

    fn push_gather_entry(&self) -> Option<&'static str> {
        Some(match self.world {
            2 => "::pie::collective::all_gather_push<__nv_bfloat16, 2>",
            4 => "::pie::collective::all_gather_push<__nv_bfloat16, 4>",
            8 => "::pie::collective::all_gather_push<__nv_bfloat16, 8>",
            _ => return None,
        })
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
            Fire::at(FILE, "::pie::collective::peers_answer")
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
        let total = buf.elements() * 2;
        if self.host {
            let entry = self
                .host_entry(true)
                .ok_or_else(|| refuse(op, "no host-path kernel for this world"))?;
            let packs = total / 16;
            let (launch, slots, slot_bytes, flags, counts) = self.host_lane(true);
            return ctx.fire(
                op,
                Fire::at(FILE, entry).apply(launch),
                &[
                    ArgValue::Ptr(flags),
                    ArgValue::Ptr(counts),
                    ArgValue::Ptr(slots),
                    ArgValue::I64(slot_bytes as i64),
                    ArgValue::Ptr(buf.ptr),
                    ArgValue::I32(self.rank as i32),
                    ArgValue::I32(
                        i32::try_from(packs).map_err(|_| refuse(op, "a message past i32"))?,
                    ),
                ],
            );
        }
        // A message the push slots hold goes in one launch, no staging copy.
        if total / 16 <= PUSH_PACKS
            && let Some(entry) = self.push_entry()
        {
            let packs = total / 16;
            let threads = if packs >= 512 {
                512
            } else {
                packs.next_multiple_of(32).max(32) as u32
            };
            return ctx.fire(
                op,
                Fire::at(FILE, entry).apply(Launch::grid(
                    [packs.div_ceil(512).max(1) as u32, 1, 1],
                    [threads, 1, 1],
                )),
                &[
                    ArgValue::Ptr(self.stages),
                    ArgValue::Ptr(self.signals),
                    ArgValue::Ptr(self.signal),
                    ArgValue::Ptr(buf.ptr),
                    ArgValue::I32(self.rank as i32),
                    ArgValue::I32(packs as i32),
                    ArgValue::I64(self.stage_bytes as i64),
                    ArgValue::I32(PUSH_PACKS as i32),
                ],
            );
        }
        let entry = self
            .entry(true)
            .ok_or_else(|| refuse(op, "no peer kernel for this world"))?;
        let chunk = self.stage_bytes & !15;
        let mut done = 0;
        while done < total {
            let bytes = chunk.min(total - done);
            copy(ctx, op, self.stage, buf.ptr + done, bytes)?;
            let packs = bytes / 16;
            ctx.fire(
                op,
                Fire::at(FILE, entry).apply(Self::launch(packs)),
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
        // A shard the push slots hold goes in one launch, no staging copy.
        let shard = x.elements() * 2 / 16;
        if !self.host
            && shard <= PUSH_PACKS
            && let Some(entry) = self.push_gather_entry()
        {
            let threads = if shard >= 512 {
                512
            } else {
                shard.next_multiple_of(32).max(32) as u32
            };
            return ctx.fire(
                op,
                Fire::at(FILE, entry).apply(Launch::grid(
                    [shard.div_ceil(512).max(1) as u32, 1, 1],
                    [threads, 1, 1],
                )),
                &[
                    ArgValue::Ptr(self.stages),
                    ArgValue::Ptr(self.signals),
                    ArgValue::Ptr(self.signal),
                    ArgValue::Ptr(x.ptr),
                    ArgValue::Ptr(y.ptr),
                    ArgValue::I32(self.rank as i32),
                    ArgValue::I32(x.rows as i32),
                    ArgValue::I32((u64::from(x.width) * 2 / 16) as i32),
                    ArgValue::I64(self.stage_bytes as i64),
                    ArgValue::I32(PUSH_PACKS as i32),
                ],
            );
        }
        if self.host {
            let entry = self
                .host_entry(false)
                .ok_or_else(|| refuse(op, "no host-path kernel for this world"))?;
            let width = i32::try_from(u64::from(x.width) * 2 / 16)
                .map_err(|_| refuse(op, "a row past i32 packs"))?;
            let (launch, slots, slot_bytes, flags, counts) = self.host_lane(false);
            return ctx.fire(
                op,
                Fire::at(FILE, entry).apply(launch),
                &[
                    ArgValue::Ptr(flags),
                    ArgValue::Ptr(counts),
                    ArgValue::Ptr(slots),
                    ArgValue::I64(slot_bytes as i64),
                    ArgValue::Ptr(x.ptr),
                    ArgValue::Ptr(y.ptr),
                    ArgValue::I32(self.rank as i32),
                    ArgValue::I32(x.rows as i32),
                    ArgValue::I32(width),
                ],
            );
        }
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
                Fire::at(FILE, entry).apply(Self::launch(rows * row / 16 * u64::from(self.world))),
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
        && peers.holds(true, buf.elements() * 2)
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
        && peers.holds(false, x.elements() * 2)
        && x.width.is_multiple_of(8)
        && aligned16(y.ptr)
        && y.rows == x.rows
        && y.width == x.width * peers.world
    {
        return peers.all_gather(ctx, OP, &x, y);
    }
    let comm = ctx.comm(OP)?;
    debug_assert_eq!(x.dtype, y.dtype, "a gather does not change the dtype");
    debug_assert!(
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
        let code = unsafe {
            nccl::ncclAllGather(send, stage, count, dtype, comm.cast(), ctx.stream().cast())
        };
        answered(OP, "ncclAllGather", code)?;
        crate::layout::gather_width_concat(ctx, stage as usize as u64, y, x.width, world)
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

        answered(op, "ncclGroupStart", unsafe { nccl::ncclGroupStart() })?;
        let ran = body();
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
    debug_assert_eq!(x.dtype, y.dtype, "a reduction does not change the dtype");
    debug_assert!(
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
