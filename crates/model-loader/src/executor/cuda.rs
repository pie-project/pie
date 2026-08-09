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
//! seam, so nothing in `executor/walk.rs` learns that a GPU exists.
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
//! trip to compute something `kernels-cuda-new` has a kernel for.
//!
//! # The four rows are fired, not linked
//!
//! They used to be `kernels_cuda::ffi::pie_k_*` — `extern "C"` symbols an
//! ahead-of-time archive defines, each forwarding into a C++ host launcher
//! holding a `<<<>>>`. That made this feature imply nvcc, CMake and a CUDA
//! toolkit in the one crate whose toolkit-free build is what lets
//! `pie model convert` run anywhere. They were then
//! `kernels_cuda_new::api`'s generated functions, and are now
//! [`kernels_cuda_new::x::quant`]'s four host programs, which reach the same
//! device text through NVRTC at run time.
//!
//! The paragraph below is kept as written, because the property it names is
//! the one that survived two moves and is still the reason this file is
//! short. Its middle bullet is the one §5 step 5 RETIRED, and how is worth
//! reading in its own words first:
//!
//! > The generator is the same one, reading the same rows: `emit_rust_api`
//! > replaced `emit_c_shim`, so *"a row that changes its operands changes
//! > both call sites or fails to compile"* still holds — that property is
//! > the whole reason the shim was worth having. What changed is three
//! > things at each call site, and each is a decision rather than a
//! > translation:
//! >
//! > * **A stream is not an operand.** It is `cuLaunchKernel`'s sixth
//! >   parameter, outside the `void**`, so it left the argument list and
//! >   became `Stream` — see `CudaArena::fire_on`.
//! > * **An extent the launch rule recovers is not an operand either.**
//! >   Every `rows` argument is gone: three of these four rows put the row
//! >   on `grid.x`, so the rule derives it from `Dims` and the `__global__`
//! >   never took it. What each row's rule reads, and what this file
//! >   therefore has to supply, is written beside each launch.
//! > * **A refusal is a `Result`.** The C symbol returned nothing, so a
//! >   rejected launch was invisible and the load finished holding whatever
//! >   the arena had. See `refused` for what happens now and why it is not
//! >   a fallback.
//!
//! # `rows` is an argument again, and that is the port
//!
//! §5 step 5 took `quant` into fn-world: the eleven contracts state no
//! `operands` and no `LaunchRule`, and the geometry each rule used to compute
//! is a `Launch` written beside the `<<<>>>` it came from, inside the host
//! `fn`. So the middle bullet inverts. There is no `Dims` to fill, no rule to
//! recover `rows` from, and no nine zero fields to explain — `rect` is
//! deleted with them — and every extent is an ordinary `i32` parameter in the
//! kernel's own order.
//!
//! **That inversion is not cosmetic and this file is where it would have
//! bitten.** A `unit!` device row carries `LaunchRule::Unstated`, so these
//! four calls left through `api::quant_*` would have compiled unchanged and
//! then refused every launch at load time with `Ungeometric::Unstated` — a
//! checkpoint that converts on one build and fails on the next, with an
//! error naming a rule nobody wrote. Calling the host programs is what makes
//! the change a compile-time one.
//!
//! The first and third bullets stand. The stream is still outside the
//! argument list, spelled `*mut c_void` because that is what
//! `x::fire::fire` takes; and a refusal is still a `Result`, now
//! [`declined`] rather than the deleted `refused`, for the same reason and
//! with one honest narrowing recorded there.

use std::borrow::Cow;
use std::ffi::c_void;

use cudarc::runtime::sys as rt;
use kernels_cuda_new::x::quant;

use crate::error::Error;
use crate::executor::arena::{ArenaBacking, TileMapOp};
use crate::plan::passes::tile::{
    CUDA_CAST_FP32_TO_BF16, CUDA_QUANTIZE_BF16_TO_FP8, CUDA_QUANTIZE_BF16_TO_MXFP4,
    CUDA_SCALE_ROWS_BF16,
};

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
/// [`Self::write`] leaves its copy IN FLIGHT by design — that is what the
/// slots buy — so the source of a live `cudaMemcpyAsync` is a [`PinnedBuf`]
/// this arena owns. On the happy path [`ArenaBacking::finish`] has already
/// drained; on the path out of a failed load nothing had, and
/// `PinnedBuf::drop` reached `cudaFreeHost` on a buffer the copy engine was
/// still reading. That is undefined, and the fault it eventually produces
/// does not name this site.
///
/// Unconditional rather than a flag, because the question "is a copy still
/// running" is the stream's to answer and it answers it for free when the
/// answer is no. The error is dropped: a drop cannot report, and a stream that
/// faulted here has already recorded it for the next call on the context.
/// Drain the stream before the pinned staging goes away.
///
/// [`ArenaBacking::finish`] is the ordinary path and the executor calls it, so
/// this is the extraordinary one: a load that failed mid-schedule returns
/// through `?` without reaching it. [`Self::write`] leaves its copy IN FLIGHT
/// by design — that is what the slots buy — so the source of a live
/// `cudaMemcpyAsync` is a [`PinnedBuf`] this arena owns, and `PinnedBuf::drop`
/// reaches `cudaFreeHost` on a buffer the copy engine is still reading. That
/// is undefined, and the fault it eventually produces does not name this site.
///
/// Unconditional rather than a flag, because "is a copy still running" is the
/// stream's question and it answers it for free when the answer is no. The
/// error is dropped: a drop cannot report, and a stream that faulted here has
/// already recorded it against the context.
impl Drop for CudaArena {
    fn drop(&mut self) {
        let _ = ArenaBacking::finish(self);
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

    /// Wait for every copy this arena enqueued.
    ///
    /// The backing that needs this verb, and the reason it is on the trait.
    /// [`Self::write`] returns while its `cudaMemcpyAsync` is still crossing —
    /// a slot is waited on only when it is REUSED — so the arena holds a
    /// partly written model until the stream drains. The executor calls this
    /// once, after the last instruction, which is where the overlap has
    /// already been paid for and there is nothing left to overlap with.
    ///
    /// # Errors
    ///
    /// The stream faulted while draining the writes. This is where a copy that
    /// failed long ago is finally reported, because it is the first point at
    /// which it has certainly either happened or not.
    fn finish(&mut self) -> Result<(), Error> {
        // SAFETY: `stream` is the caller's live stream.
        check("cudaStreamSynchronize", unsafe {
            rt::cudaStreamSynchronize(self.stream)
        })
    }

    /// Yes, unless device transforms were turned off.
    ///
    /// One bit, and it is the only one this backing has to offer: WHICH
    /// transforms run on the device is the plan's answer, named per
    /// instruction by `plan::passes::tile::cuda_kernel` with the tensor's name
    /// still in hand. This used to be a `u32` returning `CAST | SCALE |
    /// ENCODE` unconditionally — a constant wearing a bitmask's clothes,
    /// which could only ever claim more kinds than the plan had named rows
    /// for.
    fn runs_named_kernels(&self) -> bool {
        self.device_transforms
    }

    /// Launch the row the plan named.
    ///
    /// **No decision is made here.** This used to be ~120 lines of dtype
    /// matching, shape derivation and block-size arithmetic, ending in
    /// `Ok(false)` — a *decline* — whenever the operands were a shape no
    /// kernel covered. That answer was correct and unobservable: the load
    /// finished, the bytes were right, and the transform had quietly run on
    /// the host at a fraction of the speed.
    ///
    /// Those rules are in the compiler now. What is left is a lookup, and it
    /// has no second answer: the executor offers an op only for an instruction
    /// the plan named a row for, so an unknown symbol here means the compiler
    /// named a row this build does not have. That is drift between two halves
    /// of one tree, and falling back to the host would hide it behind an
    /// answer that looks fine and is slower.
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
    // `fire_on` STOOD HERE. It borrowed `&self` to produce a
    // `runtime::Stream<'_>` whose lifetime could not outlive the arena, and
    // its doc called that *"what the lifetime exists to make unspellable"* —
    // a launch queued on a destroyed stream. `x::fire::fire` takes a raw
    // `*mut c_void` and re-wraps it with the same `Stream::from_runtime`, so
    // the borrow could not be carried across and the four calls below pass
    // `self.stream.cast()` directly.
    //
    // The obligation did not disappear with the type: `CudaArena::new`'s
    // safety contract already says the stream outlives the arena, every call
    // site below is inside an `unsafe` block that names it, and the four are
    // `&mut self` methods, so the arena is alive at each. What is lost is
    // that the compiler checked it. Written down here because a borrow that
    // stops being checked and starts being asserted is exactly the kind of
    // change that leaves no trace.

    /// `quant::cast_fp32_to_bf16`.
    ///
    /// The element count is the one arithmetic fact left here, and it is
    /// addressing rather than selection: the row takes a count, and the spans
    /// are bytes.
    ///
    /// **What the rule recovered:** nothing. `LaunchRule::Elementwise` read
    /// `rows * width` to size a flat grid, and `n` stayed an operand because
    /// the kernel tests its own index against it — the distinction between an
    /// extent a rule RECOVERS and one a kernel READS. So the rectangle was
    /// stated as one row of `n`, which is the shape this transform actually
    /// has: a `Cast` is over a byte run and the plan's 2-D shape, if it has
    /// one, is not what the kernel indexes by.
    ///
    /// The rule is gone and the distinction it drew is now visible in the
    /// signature: `x::quant::cast_fp32_to_bf16` takes `n` once, and
    /// `x::quant::elementwise` turns it into `Launch::flat(n, 256)` inside.
    /// There was never a second number to state.
    fn cast(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        self.bounds(op.src.offset, op.src.len)?;
        self.bounds(op.dst.offset, op.dst.len)?;
        let n = op.src.len / 4;
        // A cast of more than 4 Gi elements — a 16 GiB f32 tensor in one
        // span — is refused rather than truncated, because the truncation is
        // a grid that covers a prefix and leaves the tail holding whatever
        // the arena had.
        //
        // The bound used to be `Dims`' nine `u32`s against the archive's
        // `size_t n`; it is now `x::quant::cast_fp32_to_bf16`'s own extent
        // check, which is a `panic!` rather than a refusal because a host
        // program has no `Result` to put it in. Refusing HERE, first, is what
        // keeps this a load error naming the plan instead of an abort.
        u32::try_from(n).map_err(|_| {
            plan_disagrees("the cast covers more elements than a 32-bit launch extent states")
        })?;
        // SAFETY: both spans are in bounds, the plan chose this row for an
        // f32 source and a bf16 destination, and the stream is live for the
        // launch.
        let ctx = unsafe { kernels_cuda_new::jit::Ctx::on(self.stream.cast()) };
        let fired = quant::cast_fp32_to_bf16(
            &ctx,
            self.at(op.src.offset).cast(),
            self.at(op.dst.offset).cast(),
            n,
        );
        declined(CUDA_CAST_FP32_TO_BF16, fired)
    }

    /// `quant::scale_rows_bf16`, the per-row multiply, in place.
    ///
    /// **What the rule recovered:** `rows`. `LaunchRule::RouteRows` was one
    /// block per row on `grid.x`, so the row count left the argument list and
    /// the `__global__` never took it; `width` stayed an operand because the
    /// kernel's `for (c = threadIdx.x; c < width; c += blockDim.x)` reads it.
    /// The rule read `width` too — as the BLOCK width, rounded up to a warp
    /// and capped at 1024 — which is the one number that differs from the
    /// archive's fixed 256 and is inert for exactly that stride.
    ///
    /// `rows` is a parameter again, because `x::quant::scale_rows_bf16` puts
    /// it in `x::quant::route_rows` itself — same expression, same cap, one
    /// copy — and the `__global__` still does not take it. The block-width
    /// note above is the reason that expression had to move rather than be
    /// re-derived: it is a MEASURED divergence from the archive, not a
    /// restatement of it.
    fn scale(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        let (rows, cols) = self.extent_2d(op)?;
        let factors = op
            .factors
            .ok_or_else(|| plan_disagrees("scale_rows_bf16 reads per-group factors"))?;
        self.bounds(op.dst.offset, op.dst.len)?;
        self.bounds(factors.offset, factors.len)?;
        // SAFETY: both spans are in bounds; the kernel writes `dst` in place,
        // which is what the plan checked before naming this row.
        let ctx = unsafe { kernels_cuda_new::jit::Ctx::on(self.stream.cast()) };
        let fired = quant::scale_rows_bf16(
            &ctx,
            self.at(op.dst.offset).cast(),
            self.at(factors.offset).cast_const().cast(),
            as_int(rows)?,
            as_int(cols)?,
        );
        declined(CUDA_SCALE_ROWS_BF16, fired)
    }

    /// `quant::quantize_bf16_to_mxfp4_e2m1_per_block`.
    ///
    /// **What the rule recovered:** `rows`, as in [`Self::scale`] — the same
    /// `RouteRows`, the same `<<<rows, ...>>>`. `cols` stayed an operand
    /// because the packer computes three row bases from it.
    ///
    /// **The `cols / 32` this call site used to pass is GONE, and losing it
    /// silently would have been a real regression.** Its note read: *"The
    /// block width the rule derives is stated as the group count, `cols / 32`,
    /// and not as `cols`: the kernel's loop is over 32-element MXFP4 blocks,
    /// one whole block per thread, so a block sized on the element width
    /// would launch 32 threads for every one with work."* That division was
    /// this file compensating for a rule that could not see the kernel's
    /// blocking, and it worked only because the caller knew. `x::quant`'s
    /// host program does the same division INSIDE, beside the `<<<>>>` it
    /// came from, so this call site passes the honest `cols` and the 32 lives
    /// once, next to the loop that justifies it.
    ///
    /// The rest of the note still holds: the plan has already refused a
    /// `cols` that is not a whole number of groups, so the division is exact
    /// — and it is the same truncation the kernel's own `groups = cols / 32`
    /// performs either way.
    fn encode_mxfp4(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        let (rows, cols) = self.extent_2d(op)?;
        let scales = self.encode_scales(op)?;
        // SAFETY: every span is bounds-checked by `encode_scales`; the row
        // takes a source, a payload, an E8M0 scale array and one extent.
        let ctx = unsafe { kernels_cuda_new::jit::Ctx::on(self.stream.cast()) };
        let fired = quant::quantize_bf16_to_mxfp4_e2m1_per_block(
            &ctx,
            self.at(op.src.offset).cast_const().cast(),
            self.at(op.dst.offset).cast(),
            self.at(scales.offset).cast(),
            as_int(rows)?,
            as_int(cols)?,
        );
        declined(CUDA_QUANTIZE_BF16_TO_MXFP4, fired)
    }

    /// `quant::quantize_bf16_to_fp8_e4m3_per_channel`.
    ///
    /// **What the rule recovered:** `rows`, and the block and the shared
    /// memory with it. `LaunchRule::Rms` was `<<<rows, 256, (256 / 32) * 4>>>`
    /// — the archive's launch digit for digit — and it read nothing else,
    /// because the block width is fixed by the reduction rather than by the
    /// row: `absmax` folds warp by warp into a shared array sized on 256, and
    /// a block of another width reads past it. `cols` stays an operand.
    ///
    /// That launch is `x::quant::rms` now — `Launch::per_row(rows, 256)`
    /// with 32 bytes of shared memory — and the `(256 / 32) * 4` is written
    /// out there, where the reduction that fixes it can be read in the same
    /// glance.
    ///
    /// The paragraph that followed is retired with `rect`: *"`width` is
    /// supplied anyway, and is the source's row width. No rule reads it here;
    /// a zero left in a field nobody can look up is the thing `Dims` asks
    /// callers not to do."* There is no unread field to fill.
    fn encode_fp8(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        let (rows, cols) = self.extent_2d(op)?;
        let scales = self.encode_scales(op)?;
        // SAFETY: as above; the scales are `f32` for this row.
        let ctx = unsafe { kernels_cuda_new::jit::Ctx::on(self.stream.cast()) };
        let fired = quant::quantize_bf16_to_fp8_e4m3_per_channel(
            &ctx,
            self.at(op.src.offset).cast_const().cast(),
            self.at(op.dst.offset).cast(),
            self.at(scales.offset).cast(),
            as_int(rows)?,
            as_int(cols)?,
        );
        declined(CUDA_QUANTIZE_BF16_TO_FP8, fired)
    }

    /// The 2-D extent every row above takes, in the plan's own units.
    ///
    /// An `Err` rather than a decline, for the reason the unknown-symbol arm
    /// is: the plan states a shape and chose a row that needs one, so its
    /// absence here is the two disagreeing.
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

// `rect` STOOD HERE and is deleted with the `Dims` it built. Its doc was a
// careful account of nine fields a load-time transform must not invent —
// *"query heads, head width, rotary channels, expert counts, a request count,
// an AltUp stream count ... A loader that filled them with plausible numbers
// would be stating a model geometry that nothing checked"* — and the whole
// question is gone rather than answered: a host program takes the extents its
// `__global__` takes, so there is no wider struct to leave zeroed and no
// convention to follow about which zero means "absent" and which means
// "empty". That is `x::launch::Launch`'s four fields against `Dims`' twelve,
// and it is the clearest small measure of what fn-world removes.

/// An extent as the `int` the `__global__` reads it in.
fn as_int(extent: u32) -> Result<i32, Error> {
    i32::try_from(extent).map_err(|_| plan_disagrees("the extent does not fit an `int`"))
}

/// The compiler named a row whose operands are not what arrived.
///
/// Never a fallback. The host path would produce the right bytes, and that is
/// exactly why it must not run: it would hide a compiler that chose wrongly
/// behind an answer that looks fine and is slower.
fn plan_disagrees(what: &str) -> Error {
    Error::Contract(format!("cuda arena: the plan named a kernel but {what}"))
}

// `refused` STOOD HERE and its consumer set is empty: the four call sites it
// served now call `x::quant`'s host programs, which return `Fired` and not
// `Result<(), runtime::Error>`. Its argument is not lost — [`declined`] below
// carries it forward and states the one narrowing — but its own words are
// worth keeping, because they are the reason both functions exist:
//
// > **This is the answer the `pie_k_*` symbols could not give.** They
// > returned `void`: a launch refused for a collapsed rectangle, a unit NVRTC
// > would not compile, or an argument list that did not match the row
// > produced no value to inspect, and the load went on to publish a tensor
// > holding whatever the arena happened to contain. These four kernels
// > quantise and cast WEIGHTS — a wrong answer here is a checkpoint that
// > loads, runs, and is quietly wrong — so the refusal is propagated and the
// > load fails with the row named.
// >
// > It is deliberately not a fall back to the host, for `plan_disagrees`'
// > reason and one more: every variant of the JIT's error is drift between
// > this build and the table it was built from — an unknown symbol, a unit
// > that will not compile, a rectangle that collapsed to nothing — and none
// > of them is a condition that running the same transform somewhere else
// > resolves. It would only make the drift slower to find.

/// The host program would not launch the row the plan named.
///
/// `refused`'s successor, and it keeps that function's whole argument:
/// these four kernels quantise and cast WEIGHTS, a wrong answer here is a
/// checkpoint that loads, runs and is quietly wrong, so the refusal is
/// propagated and the load fails with the row named. It is not a fall back to
/// the host, for the reasons recorded in the block comment above.
///
/// **One narrowing, and it is real.** `refused` took every variant of
/// `runtime::Error` — an unknown symbol, a unit NVRTC would not compile, an
/// argument list that did not match the row, a rectangle that collapsed. In
/// fn-world only the LAST of those arrives here as a value: `x::fire::fire`
/// panics on the first three, because a host program has no `Result` to put
/// them in and every one of them is drift between this build and its own
/// tables rather than a condition a load can report. So the three that used
/// to be a `Contract` error are an abort now.
///
/// That is a worse diagnostic and a better one at once, and the honest
/// statement is both halves: worse, because a load that could have named the
/// plan now dies in a panic message; better, because those three were never
/// recoverable and reporting them as "the plan named a kernel but ..." was
/// telling the operator to look at the checkpoint when the bug is in the
/// tree. `Refusal::Empty` — the collapsed rectangle — is the one this
/// function still sees, and it is still an `Err` and not a no-op, exactly as
/// `launch::eval`'s `Ungeometric::Empty` was: a transform the plan asked for
/// and the arena did not perform leaves a tensor holding whatever was there.
fn declined(
    symbol: &str,
    fired: Result<(), kernels_cuda_new::x::Refusal>,
) -> Result<(), Error> {
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
    /// The constants live in `plan::passes::tile` so that a CUDA plan compiles
    /// on a machine with no CUDA — which means nothing there can check them.
    /// This is the other half: with the feature on, the JIT is in the graph,
    /// and a row that was renamed or removed fails here instead of becoming a
    /// plan whose kernel `run_tile_map` reports as unknown at load time.
    ///
    /// It asks `runtime::hosts` rather than scanning a table, and the change
    /// is not cosmetic. A row present in `kernels_cuda::quant::KERNELS` was a
    /// row someone had DECLARED; a symbol `hosts` answers for is one a
    /// compiled unit carries the text of and `fire` will resolve — which is
    /// the question this file's four launches actually ask. The rows were
    /// spelled differently on the JIT side until recently
    /// (`quant::cast_f32_to_bf16` for `quant::cast_fp32_to_bf16`), and that
    /// is exactly the drift a table scan would have called green.
    #[test]
    fn every_symbol_the_plan_may_name_is_a_row_this_build_can_fire() {
        for symbol in [
            CUDA_CAST_FP32_TO_BF16,
            CUDA_SCALE_ROWS_BF16,
            CUDA_QUANTIZE_BF16_TO_MXFP4,
            CUDA_QUANTIZE_BF16_TO_FP8,
        ] {
            assert!(
                kernels_cuda_new::runtime::hosts(symbol),
                "the loader may compile a plan naming `{symbol}`, and no unit \
                 in `kernels-cuda-new` hosts it"
            );
        }
    }
}
