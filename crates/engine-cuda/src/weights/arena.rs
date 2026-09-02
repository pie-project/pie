//! Load arena on a discrete CUDA device: staging, transfers, and device
//! transforms (Cast/Scale/Encode) for a caller-owned device allocation,
//! behind `feature = "cuda"`.

use std::borrow::Cow;
use std::ffi::c_void;

use cudarc::runtime::sys as rt;
// This file calls kernels-cuda directly, bypassing the binder, so a
// weight-loading pass calling these fns is invisible to census tooling.
use kernels_cuda::linear::quant;
use kernels_cuda::{Ctx, Tensor};
use model_ir::Dtype;

use checkpoint::error::Error;
use checkpoint::executor::arena::{ArenaBacking, TileMapOp};
use checkpoint::plan::passes::tile::{CUDA_CAST_FP32_TO_BF16, CUDA_SCALE_ROWS_BF16};

/// Ceiling on pinned host memory per staging slot. [`CudaArena::new`] pins
/// the smaller of this and the plan's `max_tile_bytes`.
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

/// Pinned host memory, freed on drop. A dedicated type since page-locking
/// matters — a `Vec`'s pages are not locked, and the synchronous transfer
/// out of them is what this exists to avoid.
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

/// One pinned staging slot plus the event marking its copy landed. Per-slot
/// rather than stream-wide, so the wait is only for the slot being reused.
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
    /// Pinned staging, alternated so one copy can overlap the next fill.
    /// Empty if pinning failed — falls back to the slower pageable path.
    staging: Vec<StagingSlot>,
    next_slot: usize,
    device_transforms: bool,
}

impl CudaArena {
    /// Wraps `len` bytes of device memory at `base`, ordering every copy
    /// on `stream`. Staging slots are sized to the smaller of
    /// `max_write_bytes` and [`STAGING_SLOT_CEILING`].
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
        // best effort: an engine that can't pin two slots still loads. The
        // event is recorded before first use, so a slot's first
        // synchronize (on an unrecorded event) returns immediately.
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
            device_transforms: true,
        })
    }

    /// The same arena with device transforms forced off: the host path is
    /// the reference implementation (for comparison), and the fallback
    /// when a checkpoint trips a kernel.
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

/// Drains the stream before the pinned staging goes away.
///
/// [`ArenaBacking::finish`] is the ordinary path; this covers a load that
/// fails mid-schedule and returns through `?` without reaching it.
/// [`Self::write`] leaves its copy in flight by design, so a live
/// `cudaMemcpyAsync` can still be reading a [`PinnedBuf`] this arena owns.
impl Drop for CudaArena {
    fn drop(&mut self) {
        let _ = ArenaBacking::finish(self);
    }
}

impl ArenaBacking for CudaArena {
    fn len(&self) -> usize {
        self.len
    }

    /// A device read is a staging copy, and it synchronizes.
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

    /// Staged through pinned memory, enqueued but not awaited —
    /// [`CudaArena::finish`] drains it. Alternating two slots lets the
    /// executor fill one while the other crosses.
    fn write(&mut self, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        self.bounds(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        let slot_bytes = self.staging.first().map_or(0, |slot| slot.buf.len);
        if bytes.len() > slot_bytes {
            // larger than a slot, or nothing pinned: the pageable path
            // (synchronous in effect), still ordered on the same stream.
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
        // waits on the copy that last read this slot (with two slots: the
        // copy-before-last).
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

    /// Waits for every copy this arena enqueued; the executor calls this
    /// once, after the last instruction.
    ///
    /// # Errors
    ///
    /// The stream faulted while draining the writes.
    fn finish(&mut self) -> Result<(), Error> {
        // SAFETY: `stream` is the caller's live stream.
        check("cudaStreamSynchronize", unsafe {
            rt::cudaStreamSynchronize(self.stream)
        })
    }

    /// Yes, unless device transforms were turned off.
    fn runs_named_kernels(&self) -> bool {
        self.device_transforms
    }

    /// Launches the row the plan named; an unknown symbol means drift
    /// between two halves of one tree, refused rather than run on the host.
    fn run_tile_map(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        match op.kernel {
            CUDA_CAST_FP32_TO_BF16 => self.cast(op),
            CUDA_SCALE_ROWS_BF16 => self.scale(op),
            other => Err(Error::Contract(format!(
                "the plan names kernel `{other}`, which this build has no \
                 launcher for"
            ))),
        }
    }
}

impl CudaArena {
    /// This arena's stream as the context an entry fires on. Built per
    /// call rather than held as a field, since [`Ctx`] carries no state.
    fn ctx(&self) -> Ctx {
        // SAFETY: `stream` is live for as long as this arena is, by
        // `CudaArena::new`'s safety contract, and this is a `&self` method.
        unsafe { Ctx::on(self.stream.cast()) }
    }

    /// The device address `bytes` into the arena, as the `u64` a [`Tensor`]
    /// carries (the host never dereferences it).
    fn device_ptr(&self, offset: usize) -> u64 {
        self.at(offset) as u64
    }

    /// `quant::cast_fp32_to`, at `bf16`: elementwise over a flat byte run,
    /// addressed by element count rather than the plan's 2-D shape.
    fn cast(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        self.bounds(op.src.offset, op.src.len)?;
        self.bounds(op.dst.offset, op.dst.len)?;
        let n = op.src.len / 4;
        // refused rather than truncated past 4 Gi elements
        let width = u32::try_from(n).map_err(|_| {
            plan_disagrees("the cast covers more elements than a 32-bit launch extent states")
        })?;
        let src = Tensor::new(self.device_ptr(op.src.offset), 1, width, Dtype::F32);
        let mut dst = Tensor::new(self.device_ptr(op.dst.offset), 1, width, Dtype::Bf16);
        let fired = quant::cast_fp32_to(&self.ctx(), src, &mut dst);
        declined(CUDA_CAST_FP32_TO_BF16, fired)
    }

    /// `quant::scale_rows`, at `bf16`, the per-row multiply, in place.
    /// `rows` picks the block and `cols` sizes it; the factors are one row
    /// of `cols`, indexed by column, hence a `[1, cols]` handle.
    fn scale(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        let (rows, cols) = self.extent_2d(op)?;
        let factors = op
            .factors
            .ok_or_else(|| plan_disagrees("scale_rows reads per-group factors"))?;
        self.bounds(op.dst.offset, op.dst.len)?;
        self.bounds(factors.offset, factors.len)?;
        let l = Tensor::new(self.device_ptr(factors.offset), 1, cols, Dtype::Bf16);
        // scales `dst` in place
        let mut buf = Tensor::new(self.device_ptr(op.dst.offset), rows, cols, Dtype::Bf16);
        let fired = quant::scale_rows(&self.ctx(), l, &mut buf);
        declined(CUDA_SCALE_ROWS_BF16, fired)
    }

    /// The 2-D extent every row above takes. An `Err` rather than a
    /// decline, since a missing shape means the plan and this disagree.
    fn extent_2d(&self, op: &TileMapOp<'_>) -> Result<(u32, u32), Error> {
        op.shape
            .ok_or_else(|| plan_disagrees("the row takes a 2-D extent"))
    }
}

/// The compiler named a row whose operands are not what arrived. Never
/// falls back to the host: that would hide a compiler that chose wrongly.
fn plan_disagrees(what: &str) -> Error {
    Error::Contract(format!("cuda arena: the plan named a kernel but {what}"))
}

/// The host program would not launch the row the plan named. Never falls
/// back to the host: a wrong answer here would be a checkpoint that loads,
/// runs, and is quietly wrong.
fn declined(symbol: &str, fired: Result<(), kernels_cuda::Error>) -> Result<(), Error> {
    fired.map_err(|why| {
        Error::Contract(format!(
            "cuda arena: the plan named `{symbol}` and the routine declined it: {why:?}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use checkpoint::plan::passes::tile::{CUDA_CAST_FP32_TO_BF16, CUDA_SCALE_ROWS_BF16};

    // Checks every symbol the compiler may name is a path this build
    // actually calls: the macro resolves the function path (a rename fails
    // to compile) so the plan's kernel-word spelling can't drift from it.
    #[test]
    fn every_symbol_the_plan_may_name_is_a_path_this_build_calls() {
        macro_rules! spelled {
            ($ns:ident :: $f:ident) => {{
                let _ = kernels_cuda::linear::$ns::$f;
                concat!(stringify!($ns), "::", stringify!($f))
            }};
        }
        assert_eq!(CUDA_CAST_FP32_TO_BF16, spelled!(quant::cast_fp32_to));
        assert_eq!(CUDA_SCALE_ROWS_BF16, spelled!(quant::scale_rows));
    }
}
