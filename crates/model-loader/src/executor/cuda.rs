//! The arena on a discrete CUDA device, and the transforms that run there.
//!
//! [`super::arena`] made the arena a BACKING so the executor would not have to
//! know whether its bytes were reachable by the host. That left one thing
//! undone: the backing still decided which KERNEL a transform ran, and the
//! only implementation of it lived in a driver. So "the loader loads" was true
//! of every decision except the ones that need a GPU, and a second consumer
//! wanting a device load had to write this file again.
//!
//! It lives here now, behind `feature = "cuda"`. The crate without that
//! feature is what it always was — `half`, `ztensor`, `serde` — and
//! `pie model convert` on a machine with no toolkit builds it unchanged. The
//! feature is the only `#[cfg]` this adds: [`ArenaBacking`] was already the
//! seam, so nothing in `executor/host.rs` learns that a GPU exists.
//!
//! # What the caller still owns, and why
//!
//! The device ALLOCATION. [`CudaArena::new`] takes a pointer and a length,
//! not an allocator, because a consumer that has a pool, a VMM reservation or
//! a suballocated slab must be able to hand a span of it over — and because
//! "hand the loader an arena" is the whole shape this is for. What the loader
//! owns is everything between that pointer and the finished weights: the
//! staging, the transfers, the transforms.
//!
//! # Two things this does beyond addressing
//!
//! **The staging buffer is PINNED.** `cudaMemcpyAsync` out of pageable memory
//! is asynchronous in name only — the runtime stages it internally and the
//! call blocks — which made every byte of a load cross at roughly half the
//! achievable rate with no overlap at all. The executor hands over a borrowed
//! `&[u8]` it reuses, so the bytes have to be copied somewhere before the copy
//! can be left in flight; copying them into pinned memory is that somewhere.
//!
//! Two slots, alternated: one can be in flight while the executor fills the
//! next. A write larger than a slot bypasses staging and goes synchronously,
//! because a 39 GB checkpoint must not be able to make this allocate
//! proportionally to a tensor.
//!
//! **`Cast`, `Scale` and `Encode` run HERE**, on the device, when both
//! operands are already in the arena. The host path for those is a device read
//! that synchronizes, an arithmetic loop, and a device write — a full round
//! trip to compute something `kernels-cuda` has a kernel for.

use std::borrow::Cow;
use std::ffi::c_void;

use cudarc::runtime::sys as rt;

use crate::error::Error;
use crate::executor::arena::{ArenaBacking, TileMapOp};
use crate::plan::passes::tile::{
    CUDA_CAST_FP32_TO_BF16, CUDA_QUANTIZE_BF16_TO_FP8, CUDA_QUANTIZE_BF16_TO_MXFP4,
    CUDA_SCALE_ROWS_BF16,
};
use crate::plan::{TILE_MAP_CAST, TILE_MAP_ENCODE, TILE_MAP_SCALE};

/// How much pinned host memory one staging slot holds, when the caller states
/// no budget of its own.
///
/// Only a ceiling. [`CudaArena::new`] takes the plan's `max_tile_bytes` and
/// pins the smaller of the two, because pinned memory is a scarce global
/// resource and a small model must not reserve as if it were a large one.
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
    /// Wrap `len` bytes of device memory at `base`, ordering every copy on
    /// `stream`.
    ///
    /// `max_write_bytes` is the plan's `target.max_tile_bytes` — the largest
    /// single `write` the executor can make. The staging slots are sized to
    /// the smaller of it and [`STAGING_SLOT_CEILING`], so a small model pins a
    /// small pool and a write that would not fit takes the pageable path
    /// rather than growing one.
    ///
    /// # Safety
    ///
    /// `base` must point at `len` bytes of device memory that outlive this
    /// value, and `stream` must be a live stream in the same context.
    ///
    /// # Errors
    ///
    /// Never — pinning is best effort, and a backing that could not pin still
    /// loads. The result is kept so that a future failure here does not change
    /// the signature.
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

    /// Wait for every copy this arena enqueued.
    ///
    /// A load is not finished when `execute_plan_into_backing` returns — the
    /// last writes are still in flight — so a caller that reads the arena
    /// without this reads a partly written model.
    ///
    /// # Errors
    ///
    /// The stream faulted while draining the writes.
    pub fn finish(&self) -> Result<(), Error> {
        // SAFETY: `stream` is the caller's live stream.
        check("cudaStreamSynchronize", unsafe {
            rt::cudaStreamSynchronize(self.stream)
        })
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

/// Whether device-side load transforms are on. `PIE_LOADER_DEVICE_TRANSFORMS=0`
/// turns them off.
///
/// Defaulted ON: the device path is the one this module exists for, and an env
/// var that must be set to get the intended behaviour is a footgun. The off
/// switch is here for bisecting a numerical disagreement against the host
/// executor without a rebuild.
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

    /// Staged through PINNED memory, enqueued, not awaited — [`CudaArena::finish`]
    /// drains it.
    ///
    /// `bytes` is host memory the executor owns and reuses, so a copy left in
    /// flight out of it would race the next extent read. Copying into a pinned
    /// slot first is what lets the copy actually stay in flight; alternating
    /// two slots is what makes that useful, because the executor can fill one
    /// while the other is crossing.
    ///
    /// **A slot is only waited on when it is REUSED**, and then only for its
    /// own copy — that is the whole overlap. Draining the stream here instead
    /// would pay the extra `memcpy` and buy nothing.
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

    /// Whichever transforms the PLAN named a kernel for.
    ///
    /// The kinds are still the unit the executor asks about, so this is still
    /// a mask; what it no longer is, is a guess. The plan named a row per
    /// instruction (`plan::passes::tile::cuda_kernel`), so claiming a kind
    /// here costs nothing — an instruction the table does not cover carries no
    /// kernel and never reaches [`Self::run_tile_map`].
    fn tile_map_caps(&self) -> u32 {
        if self.device_transforms {
            TILE_MAP_CAST | TILE_MAP_SCALE | TILE_MAP_ENCODE
        } else {
            0
        }
    }

    /// Launch the row the plan named, or hand the instruction back.
    ///
    /// **No decision is made here.** This used to be ~120 lines of dtype
    /// matching, shape derivation and block-size arithmetic, ending in
    /// `Ok(false)` — a *decline* — whenever the operands were a shape no
    /// kernel covered. That answer was correct and unobservable: the load
    /// finished, the bytes were right, and the transform had quietly run on
    /// the host at a fraction of the speed.
    ///
    /// Those rules are in the compiler now, where the tensor's name is still
    /// in hand. What is left is a lookup: a `kernel` the plan states is a row
    /// this backing must be able to launch, and `None` is the plan saying the
    /// host runs this one.
    ///
    /// So the three answers are two. `Ok(false)` survives only for `None`,
    /// which is not a decline but a plan read correctly, and an unknown symbol
    /// is an `Err`: it means the compiler named a row this build does not
    /// have, and falling back would hide the drift behind a slower answer.
    fn run_tile_map(&mut self, op: &TileMapOp<'_>) -> Result<bool, Error> {
        let Some(kernel) = op.transform.kernel.as_deref() else {
            return Ok(false);
        };
        match kernel {
            CUDA_CAST_FP32_TO_BF16 => self.cast(op),
            CUDA_SCALE_ROWS_BF16 => self.scale(op),
            CUDA_QUANTIZE_BF16_TO_MXFP4 => self.encode_mxfp4(op),
            CUDA_QUANTIZE_BF16_TO_FP8 => self.encode_fp8(op),
            other => Err(Error::Contract(format!(
                "the plan names kernel `{other}` for a {:?}, which this build \
                 has no launcher for",
                op.kind
            ))),
        }
    }
}

impl CudaArena {
    /// `quant::cast_fp32_to_bf16`.
    ///
    /// The element count is the one arithmetic fact left here, and it is
    /// addressing rather than selection: the row takes a count, and the spans
    /// are bytes.
    fn cast(&mut self, op: &TileMapOp<'_>) -> Result<bool, Error> {
        self.bounds(op.src.offset, op.src.len)?;
        self.bounds(op.dst.offset, op.dst.len)?;
        // SAFETY: both spans are in bounds, and the plan chose this row for an
        // f32 source and a bf16 destination.
        unsafe {
            kernels_cuda::ffi::pie_k_quant_cast_fp32_to_bf16(
                self.at(op.src.offset).cast_const(),
                self.at(op.dst.offset),
                op.src.len / 4,
                self.stream.cast(),
            );
        }
        Ok(true)
    }

    /// `quant::scale_rows_bf16`, the per-row multiply, in place.
    fn scale(&mut self, op: &TileMapOp<'_>) -> Result<bool, Error> {
        let (rows, cols) = self.extent_2d(op)?;
        let factors = op
            .factors
            .ok_or_else(|| plan_disagrees("scale_rows_bf16 reads per-group factors"))?;
        self.bounds(op.dst.offset, op.dst.len)?;
        self.bounds(factors.offset, factors.len)?;
        // SAFETY: both spans are in bounds; the kernel writes `dst` in place,
        // which is what the plan checked before naming this row.
        unsafe {
            kernels_cuda::ffi::pie_k_quant_scale_rows_bf16(
                self.at(op.dst.offset),
                self.at(factors.offset).cast_const(),
                rows,
                cols,
                self.stream.cast(),
            );
        }
        Ok(true)
    }

    /// `quant::quantize_bf16_to_mxfp4_e2m1_per_block`.
    fn encode_mxfp4(&mut self, op: &TileMapOp<'_>) -> Result<bool, Error> {
        let (rows, cols) = self.extent_2d(op)?;
        let scales = self.encode_scales(op)?;
        // SAFETY: every span is bounds-checked by `encode_scales`; the row
        // takes a source, a payload, an E8M0 scale array and two extents.
        unsafe {
            kernels_cuda::ffi::pie_k_quant_quantize_bf16_to_mxfp4_e2m1_per_block(
                self.at(op.src.offset).cast_const(),
                self.at(op.dst.offset).cast::<u8>(),
                self.at(scales.offset).cast::<u8>(),
                rows,
                cols,
                self.stream.cast(),
            );
        }
        Ok(true)
    }

    /// `quant::quantize_bf16_to_fp8_e4m3_per_channel`.
    fn encode_fp8(&mut self, op: &TileMapOp<'_>) -> Result<bool, Error> {
        let (rows, cols) = self.extent_2d(op)?;
        let scales = self.encode_scales(op)?;
        // SAFETY: as above; the scales are `f32` for this row.
        unsafe {
            kernels_cuda::ffi::pie_k_quant_quantize_bf16_to_fp8_e4m3_per_channel(
                self.at(op.src.offset).cast_const(),
                self.at(op.dst.offset).cast::<u8>(),
                self.at(scales.offset).cast::<f32>(),
                rows,
                cols,
                self.stream.cast(),
            );
        }
        Ok(true)
    }

    /// The 2-D extent every row above takes, as the `int`s they take it in.
    ///
    /// An `Err` rather than a decline, for the reason the unknown-symbol arm
    /// is: the plan states a shape and chose a row that needs one, so its
    /// absence here is the two disagreeing.
    fn extent_2d(&self, op: &TileMapOp<'_>) -> Result<(i32, i32), Error> {
        let (rows, cols) = op
            .shape
            .ok_or_else(|| plan_disagrees("the row takes a 2-D extent"))?;
        let (Ok(rows), Ok(cols)) = (i32::try_from(rows), i32::try_from(cols)) else {
            return Err(plan_disagrees("the extent does not fit an `int`"));
        };
        Ok((rows, cols))
    }

    /// `Encode`'s second destination, bounds-checked along with the first.
    fn encode_scales(&self, op: &TileMapOp<'_>) -> Result<crate::executor::arena::ArenaSpan, Error> {
        let scales = op
            .dst_scales
            .ok_or_else(|| plan_disagrees("an Encode publishes payload AND scales"))?;
        self.bounds(op.src.offset, op.src.len)?;
        self.bounds(op.dst.offset, op.dst.len)?;
        self.bounds(scales.offset, scales.len)?;
        Ok(scales)
    }
}

/// The compiler named a row whose operands are not what arrived.
///
/// Never a fallback. The host path would produce the right bytes, and that is
/// exactly why it must not run: it would hide a compiler that chose wrongly
/// behind an answer that looks fine and is slower.
fn plan_disagrees(what: &str) -> Error {
    Error::Contract(format!(
        "cuda arena: the plan named a kernel but {what}"
    ))
}

#[cfg(test)]
mod tests {
    use crate::plan::passes::tile::{
        CUDA_CAST_FP32_TO_BF16, CUDA_QUANTIZE_BF16_TO_FP8, CUDA_QUANTIZE_BF16_TO_MXFP4,
        CUDA_SCALE_ROWS_BF16,
    };

    /// Every symbol the compiler may name is a row this build can launch.
    ///
    /// The constants live in `plan::passes::tile` so that a CUDA plan compiles
    /// on a machine with no CUDA — which means nothing there can check them.
    /// This is the other half: with the feature on, the table is in the graph,
    /// and a row that was renamed or removed fails here instead of becoming a
    /// plan whose kernel `run_tile_map` reports as unknown at load time.
    #[test]
    fn every_symbol_the_plan_may_name_is_a_row_in_the_table() {
        for symbol in [
            CUDA_CAST_FP32_TO_BF16,
            CUDA_SCALE_ROWS_BF16,
            CUDA_QUANTIZE_BF16_TO_MXFP4,
            CUDA_QUANTIZE_BF16_TO_FP8,
        ] {
            assert!(
                kernels_cuda::quant::KERNELS
                    .iter()
                    .any(|k| k.symbol == symbol),
                "the loader may compile a plan naming `{symbol}`, and \
                 `kernels_cuda::quant` has no such row"
            );
        }
    }
}
