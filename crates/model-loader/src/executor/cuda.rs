//! The arena on a discrete CUDA device, and the transforms that run
//! there, behind `feature = "cuda"` (the crate without it builds
//! unchanged, since [`ArenaBacking`] is the only seam). The caller owns
//! the device ALLOCATION -- [`CudaArena::new`] takes a pointer and a
//! length, not an allocator -- and this owns everything from that
//! pointer to the finished weights: staging, transfers, transforms.
//!
//! The staging buffer is PINNED (two alternating slots so a copy can
//! overlap the next fill; an oversized write bypasses staging), and
//! `Cast`/`Scale`/`Encode` run on the device via [`kernels_cuda::quant`]
//! when both operands are already resident, avoiding a host round trip.

use std::borrow::Cow;
use std::ffi::c_void;

use cudarc::runtime::sys as rt;
// This file is the one typed caller of `kernels-cuda` that never goes
// through the binder: every census in the refactor reads `sigs()` and
// `ROUTINES`, which only see rows a FIRE dispatches. A weight-loading
// pass calls these `fn`s directly and is invisible to all of them, so
// the gates measure the binder's surface and this is not on it.
//
// `scale_rows` takes a fat `Out` rather than a slot, sized by
// `extent_2d`'s `(rows, width)` pair rather than two separate `i32`s.
use kernels_cuda::quant;
use kernels_cuda::{In, InOut, Out};

use crate::error::Error;
use crate::executor::arena::{ArenaBacking, TileMapOp};
use crate::plan::passes::tile::{
    CUDA_CAST_FP32_TO_BF16, CUDA_QUANTIZE_BF16_TO_FP8, CUDA_QUANTIZE_BF16_TO_MXFP4,
    CUDA_SCALE_ROWS_BF16,
};

/// How much pinned host memory one staging slot holds, absent a
/// caller-stated budget. Only a ceiling: [`CudaArena::new`] pins the
/// smaller of this and the plan's `max_tile_bytes`, so a small model
/// does not reserve as if it were a large one.
const STAGING_SLOT_CEILING: usize = 32 * 1024 * 1024;

fn cuda(what: &str, status: rt::cudaError) -> Error {
    Error::Contract(format!("cuda arena: {what} failed: {status:?}"))
}

fn check(what: &str, status: rt::cudaError) -> Result<(), Error> {
    if status == rt::cudaError::cudaSuccess {
        Ok(())
    } else {
        Err(cuda(what, status))
    }
}

/// Pinned host memory, freed on drop.
///
/// Its own type rather than a `Vec` because the whole point is that the
/// allocation is page-locked: a `Vec`'s pages are not, and the transfer out of
/// them is the synchronous one this exists to avoid.
struct PinnedBuf {
    ptr: *mut u8,
    len: usize,
}

impl PinnedBuf {
    fn new(len: usize) -> Result<Self, Error> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a live local and `len` is the allocation we want.
        check("cudaHostAlloc", unsafe {
            rt::cudaHostAlloc(&raw mut ptr, len, rt::cudaHostAllocDefault)
        })?;
        Ok(Self {
            ptr: ptr.cast::<u8>(),
            len,
        })
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `cudaHostAlloc` and is freed once.
        unsafe {
            rt::cudaFreeHost(self.ptr.cast());
        }
    }
}

/// One pinned staging slot and the event that says its copy has landed.
///
/// The event is per SLOT rather than per copy because that is the question
/// asked of it: "may I overwrite these bytes yet". A stream-wide drain answers
/// a stricter question and costs the overlap.
struct StagingSlot {
    buf: PinnedBuf,
    done: rt::cudaEvent_t,
}

impl Drop for StagingSlot {
    fn drop(&mut self) {
        // SAFETY: created by `cudaEventCreate` and destroyed once.
        unsafe {
            rt::cudaEventDestroy(self.done);
        }
    }
}

/// A `LoadPlan`'s persistent arena in CUDA global memory.
///
/// Holds the allocation but does not own it: see the module doc.
pub struct CudaArena {
    base: *mut u8,
    len: usize,
    stream: rt::cudaStream_t,
    /// Pinned staging, alternated so one copy can be in flight while the
    /// executor fills the next slot. Empty when pinning failed — the writes
    /// then take the pageable path they always took, which is slower and
    /// correct.
    staging: Vec<StagingSlot>,
    next_slot: usize,
    device_transforms: bool,
}

impl CudaArena {
    /// Wrap `len` bytes of device memory at `base`, ordering every copy
    /// on `stream`. Staging slots are sized to the smaller of
    /// `max_write_bytes` and [`STAGING_SLOT_CEILING`], so a small model
    /// pins a small pool.
    ///
    /// # Safety
    ///
    /// `base` must point at `len` bytes of device memory that outlive
    /// this value, and `stream` must be a live stream in the same context.
    ///
    /// # Errors
    ///
    /// Never -- pinning is best effort; the `Result` keeps the signature
    /// stable for a future failure here.
    pub unsafe fn new(
        base: *mut c_void,
        len: usize,
        max_write_bytes: usize,
        stream: *mut c_void,
    ) -> Result<Self, Error> {
        let slot = max_write_bytes.clamp(1, STAGING_SLOT_CEILING);
        // Best effort: a driver that cannot pin two slots still loads. The
        // event is recorded before first use is possible, so a slot's first
        // synchronize is on an event that has never been recorded --
        // `cudaEventSynchronize` on such an event returns immediately, which
        // is the answer we want ("nothing is reading this slot yet").
        let staging = (0..2)
            .map(|_| {
                let buf = PinnedBuf::new(slot)?;
                let mut done: rt::cudaEvent_t = std::ptr::null_mut();
                // SAFETY: `done` is a live local.
                check("cudaEventCreate", unsafe {
                    rt::cudaEventCreate(&raw mut done)
                })?;
                Ok(StagingSlot { buf, done })
            })
            .collect::<Result<Vec<_>, Error>>()
            .unwrap_or_default();
        Ok(Self {
            base: base.cast::<u8>(),
            len,
            stream: stream.cast(),
            staging,
            next_slot: 0,
            device_transforms: device_transforms_enabled(),
        })
    }

    /// The same arena with device transforms forced off.
    ///
    /// The host path is the reference implementation, so this is how a caller
    /// gets the two answers to compare. It is also the honest response to a
    /// checkpoint that trips a kernel: load it on the host and report it.
    #[must_use]
    pub const fn host_transforms_only(mut self) -> Self {
        self.device_transforms = false;
        self
    }

    /// The device address `bytes` into the arena.
    fn at(&self, offset: usize) -> *mut c_void {
        // SAFETY: every span the executor hands over was resolved against
        // `ArenaBacking::len`, which is this arena's length.
        unsafe { self.base.add(offset).cast() }
    }

    fn bounds(&self, offset: usize, len: usize) -> Result<(), Error> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| Error::Contract("arena span overflows".into()))?;
        if end > self.len {
            return Err(Error::Contract(format!(
                "arena span {offset}..{end} is out of bounds ({})",
                self.len
            )));
        }
        Ok(())
    }
}

/// Drain the stream before the pinned staging goes away.
///
/// [`ArenaBacking::finish`] is the ordinary path; this covers a load
/// that fails mid-schedule and returns through `?` without reaching it.
/// [`Self::write`] leaves its copy IN FLIGHT by design, so the source of
/// a live `cudaMemcpyAsync` is a [`PinnedBuf`] this arena owns --
/// `PinnedBuf::drop` reaching `cudaFreeHost` on a buffer the copy engine
/// is still reading is undefined.
///
/// Unconditional rather than a flag: "is a copy still running" is the
/// stream's question, answered for free when the answer is no. The
/// error is dropped since a drop cannot report it.
impl Drop for CudaArena {
    fn drop(&mut self) {
        let _ = ArenaBacking::finish(self);
    }
}

/// Whether device-side load transforms are on
/// (`PIE_LOADER_DEVICE_TRANSFORMS=0` turns them off). Defaulted ON since
/// that is what this module exists for; the switch exists to bisect a
/// numerical disagreement against the host executor without a rebuild.
fn device_transforms_enabled() -> bool {
    !matches!(
        std::env::var("PIE_LOADER_DEVICE_TRANSFORMS").as_deref(),
        Ok("0")
    )
}

impl ArenaBacking for CudaArena {
    fn len(&self) -> usize {
        self.len
    }

    /// A device read is a STAGING COPY, and it synchronizes.
    ///
    /// Both are acceptable here and neither is on the write path: the executor
    /// reads the arena only to feed a transform whose input is a tensor it
    /// already wrote there, while it writes the arena once per file extent.
    fn read(&self, offset: usize, len: usize) -> Result<Cow<'_, [u8]>, Error> {
        self.bounds(offset, len)?;
        let mut out = vec![0u8; len];
        if len == 0 {
            return Ok(Cow::Owned(out));
        }
        // SAFETY: the span is in bounds and `out` is `len` bytes.
        check("cudaMemcpyAsync D2H", unsafe {
            rt::cudaMemcpyAsync(
                out.as_mut_ptr().cast(),
                self.at(offset).cast_const(),
                len,
                rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                self.stream,
            )
        })?;
        // SAFETY: `stream` is live.
        check("cudaStreamSynchronize", unsafe {
            rt::cudaStreamSynchronize(self.stream)
        })?;
        Ok(Cow::Owned(out))
    }

    /// Staged through PINNED memory, enqueued, not awaited --
    /// [`CudaArena::finish`] drains it. The caller's `bytes` is reused,
    /// so copying into a pinned slot first is what lets the copy stay in
    /// flight; alternating two slots lets the executor fill one while
    /// the other crosses.
    ///
    /// **A slot is only waited on when REUSED**, and only for its own
    /// copy -- draining the stream here instead would buy nothing.
    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        self.bounds(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        let slot_bytes = self.staging.first().map_or(0, |slot| slot.buf.len);
        if bytes.len() > slot_bytes {
            // Larger than a slot, or nothing pinned: the pageable path, which
            // is synchronous in effect and is what every write used to be.
            // Ordered on the same stream, so it cannot pass a staged copy.
            // SAFETY: the span is in bounds and `bytes` is host memory.
            return check("cudaMemcpyAsync H2D", unsafe {
                rt::cudaMemcpyAsync(
                    self.at(offset),
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    self.stream,
                )
            });
        }
        let slot = self.next_slot;
        self.next_slot = (slot + 1) % self.staging.len();
        let staged = &self.staging[slot];
        // The copy that last read this slot. With two slots this is the
        // copy-before-last, so the wait is for a transfer that has had a whole
        // extent read to finish in.
        // SAFETY: the event is live and owned by this slot.
        check("cudaEventSynchronize", unsafe {
            rt::cudaEventSynchronize(staged.done)
        })?;
        // SAFETY: the slot holds `slot_bytes >= bytes.len()` pinned bytes and
        // the copy above proved nothing is still reading them.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), staged.buf.ptr, bytes.len());
        }
        // SAFETY: the span is in bounds and the source is the pinned slot.
        check("cudaMemcpyAsync H2D", unsafe {
            rt::cudaMemcpyAsync(
                self.at(offset),
                staged.buf.ptr.cast(),
                bytes.len(),
                rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                self.stream,
            )
        })?;
        // SAFETY: both handles are live.
        check("cudaEventRecord", unsafe {
            rt::cudaEventRecord(staged.done, self.stream)
        })
    }

    fn fill(&mut self, offset: usize, len: usize, byte: u8) -> Result<(), Error> {
        self.bounds(offset, len)?;
        if len == 0 {
            return Ok(());
        }
        // SAFETY: the span is in bounds.
        check("cudaMemsetAsync", unsafe {
            rt::cudaMemsetAsync(self.at(offset), i32::from(byte), len, self.stream)
        })
    }

    /// Wait for every copy this arena enqueued. [`Self::write`] returns
    /// while its `cudaMemcpyAsync` is still crossing (a slot is only
    /// waited on when REUSED), so the arena holds a partly written model
    /// until the stream drains; the executor calls this once, after the
    /// last instruction.
    ///
    /// # Errors
    ///
    /// The stream faulted while draining the writes -- the first point a
    /// copy that failed long ago can be reported.
    fn finish(&mut self) -> Result<(), Error> {
        // SAFETY: `stream` is the caller's live stream.
        check("cudaStreamSynchronize", unsafe {
            rt::cudaStreamSynchronize(self.stream)
        })
    }

    /// Yes, unless device transforms were turned off. One bit: WHICH
    /// transforms run on the device is the plan's answer, named per
    /// instruction by `plan::passes::tile::cuda_kernel`.
    fn runs_named_kernels(&self) -> bool {
        self.device_transforms
    }

    /// Launch the row the plan named. **No decision is made here** --
    /// dtype matching and shape rules live in the compiler now, so this
    /// is a pure lookup: an unknown symbol means the compiler named a
    /// row this build has no launcher for (drift between two halves of
    /// one tree), refused rather than silently run on the host.
    fn run_tile_map(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        match op.kernel {
            CUDA_CAST_FP32_TO_BF16 => self.cast(op),
            CUDA_SCALE_ROWS_BF16 => self.scale(op),
            CUDA_QUANTIZE_BF16_TO_MXFP4 => self.encode_mxfp4(op),
            CUDA_QUANTIZE_BF16_TO_FP8 => self.encode_fp8(op),
            other => Err(Error::Contract(format!(
                "the plan names kernel `{other}`, which this build has no \
                 launcher for"
            ))),
        }
    }
}

impl CudaArena {
    // `Ctx::on` takes a raw `*mut c_void` rather than a borrowed
    // `Stream`, so the compiler no longer checks that the stream
    // outlives the arena -- each `unsafe` block below asserts it
    // instead, true by `CudaArena::new`'s safety contract since these
    // are all `&mut self` methods and the arena is therefore alive at
    // each call.

    /// `quant::cast_fp32_to`, at `bf16`: elementwise over a flat byte run,
    /// addressed by element count `n` rather than the plan's 2-D shape
    /// (a `Cast` is not indexed by rows/cols). `x::quant::elementwise`
    /// turns `n` into `Launch::flat(n, 256)` internally.
    fn cast(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        self.bounds(op.src.offset, op.src.len)?;
        self.bounds(op.dst.offset, op.dst.len)?;
        let n = op.src.len / 4;
        // A cast of more than 4 Gi elements is refused rather than
        // truncated (truncation would leave the tail holding whatever
        // the arena had). Refused HERE, first, so this is a load error
        // naming the plan rather than a `panic!` inside the host
        // program, which has no `Result` to put it in.
        u32::try_from(n).map_err(|_| {
            plan_disagrees("the cast covers more elements than a 32-bit launch extent states")
        })?;
        // SAFETY: both spans are in bounds, the plan chose this row for an
        // f32 source and a bf16 destination, and the stream is live for the
        // launch.
        let width = i32::try_from(n)
            .map_err(|_| plan_disagrees("the cast covers more elements than one row can state"))?;
        let ctx = unsafe { kernels_cuda::jit::Ctx::on(self.stream.cast()) };
        let fired = quant::cast_fp32_to::<kernels_cuda::jit::abi::bf16>(
            &ctx,
            // ONE ROW OF `n`: the cast is elementwise and the routine reads
            // its extent off the destination, which is what a traced fire's
            // binder mints for it too.
            In {
                ptr: self.at(op.src.offset).cast(),
                rows: 1,
                width,
            },
            Out {
                ptr: self.at(op.dst.offset).cast(),
                rows: 1,
                width,
            },
        );
        declined(CUDA_CAST_FP32_TO_BF16, fired)
    }

    /// `quant::scale_rows`, at `bf16`, the per-row multiply, in place. `rows`
    /// picks the block (`grid.x`) and `width` is read by the kernel's
    /// own loop over columns; both now live inside
    /// `x::quant::route_rows`, next to the `<<<>>>` they size.
    fn scale(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        let (rows, cols) = self.extent_2d(op)?;
        let factors = op
            .factors
            .ok_or_else(|| plan_disagrees("scale_rows reads per-group factors"))?;
        self.bounds(op.dst.offset, op.dst.len)?;
        self.bounds(factors.offset, factors.len)?;
        // SAFETY: both spans are in bounds; the kernel writes `dst` in place,
        // which is what the plan checked before naming this row.
        let ctx = unsafe { kernels_cuda::jit::Ctx::on(self.stream.cast()) };
        let fired = quant::scale_rows::<kernels_cuda::jit::abi::bf16>(
            &ctx,
            // `InOut`: the kernel scales `dst` IN PLACE, which the comment
            // above already says and the mark now says too.
            InOut {
                ptr: self.at(op.dst.offset).cast(),
                rows: as_int(rows)?,
                width: as_int(cols)?,
            },
            In {
                ptr: self.at(factors.offset).cast_const().cast(),
                rows: 0,
                width: 0,
            },
        );
        declined(CUDA_SCALE_ROWS_BF16, fired)
    }

    /// `quant::quantize_bf16_to_mxfp4_e2m1_per_block`: `rows` sizes the
    /// block, `cols` is read directly by the packer. The host program
    /// divides `cols` into 32-element groups internally -- exact, since
    /// the plan has already refused a `cols` that is not a whole number
    /// of them.
    fn encode_mxfp4(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        let (rows, cols) = self.extent_2d(op)?;
        let scales = self.encode_scales(op)?;
        // SAFETY: every span is bounds-checked by `encode_scales`; the row
        // takes a source, a payload, an E8M0 scale array and one extent.
        let ctx = unsafe { kernels_cuda::jit::Ctx::on(self.stream.cast()) };
        let fired = unsafe {
            quant::quantize_bf16_to_mxfp4_e2m1_per_block(
                &ctx,
                self.at(op.src.offset).cast_const().cast(),
                self.at(op.dst.offset).cast(),
                self.at(scales.offset).cast(),
                as_int(rows)?,
                as_int(cols)?,
            )
        };
        declined(CUDA_QUANTIZE_BF16_TO_MXFP4, fired)
    }

    /// `quant::quantize_bf16_to_fp8_e4m3_per_channel`: a per-row absmax
    /// reduction, `Launch::per_row(rows, 256)` with 32 bytes of shared
    /// memory sized to that fixed block width; `cols` stays an operand
    /// the reduction does not otherwise need.
    fn encode_fp8(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        let (rows, cols) = self.extent_2d(op)?;
        let scales = self.encode_scales(op)?;
        // SAFETY: as above; the scales are `f32` for this row.
        let ctx = unsafe { kernels_cuda::jit::Ctx::on(self.stream.cast()) };
        let fired = unsafe {
            quant::quantize_bf16_to_fp8_e4m3_per_channel(
                &ctx,
                self.at(op.src.offset).cast_const().cast(),
                self.at(op.dst.offset).cast(),
                self.at(scales.offset).cast(),
                as_int(rows)?,
                as_int(cols)?,
            )
        };
        declined(CUDA_QUANTIZE_BF16_TO_FP8, fired)
    }

    /// The 2-D extent every row above takes, in the plan's own units.
    /// An `Err` rather than a decline: the plan chose a row that needs
    /// a shape, so its absence here means the two disagree.
    fn extent_2d(&self, op: &TileMapOp<'_>) -> Result<(u32, u32), Error> {
        op.shape
            .ok_or_else(|| plan_disagrees("the row takes a 2-D extent"))
    }

    /// `Encode`'s second destination, bounds-checked along with the first.
    fn encode_scales(
        &self,
        op: &TileMapOp<'_>,
    ) -> Result<crate::executor::arena::ArenaSpan, Error> {
        let scales = op
            .dst_scales
            .ok_or_else(|| plan_disagrees("an Encode publishes payload AND scales"))?;
        self.bounds(op.src.offset, op.src.len)?;
        self.bounds(op.dst.offset, op.dst.len)?;
        self.bounds(scales.offset, scales.len)?;
        Ok(scales)
    }
}

/// An extent as the `int` the `__global__` reads it in.
fn as_int(extent: u32) -> Result<i32, Error> {
    i32::try_from(extent).map_err(|_| plan_disagrees("the extent does not fit an `int`"))
}

/// The compiler named a row whose operands are not what arrived. Never
/// a fallback: the host path would produce the right bytes, which is
/// exactly why it must not run and hide a compiler that chose wrongly.
fn plan_disagrees(what: &str) -> Error {
    Error::Contract(format!("cuda arena: the plan named a kernel but {what}"))
}

/// The host program would not launch the row the plan named. Never a
/// fall-back to the host: these kernels quantise and cast WEIGHTS, so a
/// wrong answer here is a checkpoint that loads, runs, and is quietly
/// wrong -- the refusal is propagated and the load fails with the row
/// named.
///
/// **One narrowing, and it is real.** Of `runtime::Error`'s variants,
/// only `Refusal::Empty` (a collapsed rectangle) arrives here as an
/// `Err`; an unknown symbol or a unit that will not compile PANICS
/// instead, since a host program has no `Result` to put them in and
/// both are drift between this build and its own device text rather
/// than a condition a load can report.
fn declined(symbol: &str, fired: Result<(), kernels_cuda::Refusal>) -> Result<(), Error> {
    fired.map_err(|why| {
        Error::Contract(format!(
            "cuda arena: the plan named `{symbol}` and the routine declined it: {why:?}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use crate::plan::passes::tile::{
        CUDA_CAST_FP32_TO_BF16, CUDA_QUANTIZE_BF16_TO_FP8, CUDA_QUANTIZE_BF16_TO_MXFP4,
        CUDA_SCALE_ROWS_BF16,
    };

    /// Every symbol the compiler may name is a row this build can FIRE.
    ///
    /// The constants live in `plan::passes::tile` so a CUDA plan
    /// compiles on a machine with no CUDA, which means nothing there can
    /// check them; this is the other half, catching a renamed or
    /// removed row here instead of at load time.
    ///
    /// Asks `routine()` rather than scanning a table: a symbol it
    /// answers for has a host program the dispatch will actually reach,
    /// which is what this file's four launches need to be true.
    ///
    /// TWO OF THE FOUR ARE NOT ROUTINES, and the split is the point. A
    /// `Routine` is what a TRACE states, and no trace states a load-time
    /// weight transform — so `kernels-cuda` holds the two quantisers as
    /// plain `unsafe fn`s. Their names are checked by the COMPILER at
    /// `encode_mxfp4`/`encode_fp8` instead, which is a stronger check than
    /// this one and needs no list; what stays here is the pair whose string
    /// still has to resolve against a registry.
    #[test]
    fn every_symbol_the_plan_may_name_is_a_row_this_build_can_fire() {
        for symbol in [CUDA_CAST_FP32_TO_BF16, CUDA_SCALE_ROWS_BF16] {
            assert!(
                kernels_cuda::routine(symbol).is_some(),
                "the loader may compile a plan naming `{symbol}`, and \
                 `kernels-cuda` has no routine for it"
            );
        }

        // The other half of the same claim: a plan may name these two, and
        // nothing in `kernels-cuda`'s registry answers for either. If one
        // ever gains a `#[routine]`, it belongs in the loop above.
        for symbol in [CUDA_QUANTIZE_BF16_TO_MXFP4, CUDA_QUANTIZE_BF16_TO_FP8] {
            assert!(
                kernels_cuda::routine(symbol).is_none(),
                "`{symbol}` is a registry row again; move it into the loop \
                 above, which is the check its string then needs"
            );
        }
    }
}
