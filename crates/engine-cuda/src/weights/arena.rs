//! The load arena on a discrete CUDA device, and the transforms that run
//! there, behind `feature = "_cuda"` (the crate without a runtime selected
//! builds unchanged, since [`ArenaBacking`] is the only seam). The caller
//! owns the device ALLOCATION -- [`CudaArena::new`] takes a pointer and a
//! length, not an allocator -- and this owns everything from that
//! pointer to the finished weights: staging, transfers, transforms.
//!
//! The staging buffer is PINNED (two alternating slots so a copy can
//! overlap the next fill; an oversized write bypasses staging), and
//! `Cast`/`Scale`/`Encode` run on the device via
//! [`kernels_cuda::linear::quant`] when both operands are already resident,
//! avoiding a host round trip.
//!
//! # Why it lives HERE and not in the loader
//!
//! It stood in `checkpoint`'s executor/cuda.rs -- the crate was called
//! model-loader at the time -- behind an optional `cuda`
//! feature that dragged `kernels-cuda` -- and through it `kernels` and
//! `model-ir` -- into the graph of a crate whose whole value is that it
//! compiles a CUDA plan on a machine with no CUDA. Nothing in any engine's
//! `src/` ever turned that feature on: the only target in the tree that did
//! was the loader's own parity test, which is now
//! `tests/gpu_transform_parity.rs` beside this crate's other gates.
//!
//! The loader keeps the VOCABULARY -- [`ArenaBacking`], [`TileMapOp`] and
//! the four `CUDA_*` kernel words a plan carries -- because those say
//! *what* transform to run, which is a compiler's sentence. Running it is a
//! device's, and this is the crate that owns the device: it already deps
//! `kernels-cuda`, `cudarc` and `checkpoint`, so moving the file here grew
//! nothing.
//!
//! Beside [`super`] rather than at the crate root because `weights.rs` is
//! the one module that drives `checkpoint::executor::Execution` at all --
//! its header argues why the transforms run on the host today, and this is
//! the other answer to the same question. `crate::arena` is a different
//! arena entirely: the fire's slot table, per launch, not per load.

use std::borrow::Cow;
use std::ffi::c_void;

use cudarc::runtime::sys as rt;
// This file is the one typed caller of `kernels-cuda` that never goes
// through the binder: every census in the refactor reads `sigs()` and
// `ROUTINES`, which only see rows a FIRE dispatches. A weight-loading
// pass calls these `fn`s directly and is invisible to all of them, so
// the gates measure the binder's surface and this is not on it.
//
// THE HANDLE VOCABULARY CHANGED UNDER THIS FILE ONCE, and the repair is
// worth recording because nothing caught it for a release. The plane used to
// speak `In`/`Out`/`InOut` marks carrying `{ptr, rows, width}` around a
// `Tensor<T>` whose scalar was a type parameter; it now speaks one
// dtype-erased [`Tensor`] with the extents folded onto the handle, and the
// entries moved from `kernels_cuda::quant` to `kernels_cuda::linear::quant`.
// This file was the only caller and did not follow, so `--features cuda`
// stopped compiling while every default build stayed green — see
// `.wiki/palo/design.md`'s step-4 note, which routed around it with host
// transforms.
//
// `&`/`&mut` on a `Tensor` is INTENT, not borrow discipline
// (`crates/kernels-cuda/src/tensor.rs` states the rule): an entry takes what it writes by `&mut`, and
// that signature is the whole record of write intent. So `scale_rows`'s
// in-place destination is the `&mut` operand and its per-column factors are
// the by-value one, which is the same claim the old `InOut`/`In` pair made.
use kernels_cuda::linear::quant;
use kernels_cuda::{Ctx, Tensor};
// The same enum `kernels_cuda::Tensor` is written in -- `dtype::Dtype`,
// re-exported at the path the rest of this crate already spells it.
use model_ir::Dtype;

use checkpoint::error::Error;
use checkpoint::executor::arena::{ArenaBacking, TileMapOp};
use checkpoint::plan::passes::tile::{CUDA_CAST_FP32_TO_BF16, CUDA_SCALE_ROWS_BF16};

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
        // Best effort: an engine that cannot pin two slots still loads. The
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
            device_transforms: true,
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

// `fn device_transforms_enabled()` STOOD HERE and read
// `PIE_LOADER_DEVICE_TRANSFORMS` out of the process environment. It came with
// this file from the loader, where article 9 did not apply, and arrived in a
// shell, where it does: *shells read no environment; every knob is typed*
// (alto design §1, gated by `tests/no_shell_reads_the_environment.rs`, which
// is how it was caught).
//
// It did not survive the reading. Its own doc argued the name was worth
// keeping because "`engine-cuda`'s `Boot` parses it into `[engine]
// device_transforms` and has a test pinning every false spelling" — and that
// is not true of this tree: `device_transforms` appears in no `Boot` field and
// in no config key, and nothing anywhere SETS the variable, including the
// parity test whose header cites it as the way to select the host path. It
// gated a bit that was `true` in every run there has ever been.
//
// The typed knob it claimed to duplicate already existed one screen up:
// [`CudaArena::host_transforms_only`], a `const fn` a caller applies to the
// arena it is about to hand over. That is the switch, and it is the one the
// host-vs-device comparison should reach for.

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
            // **AND TWO QUANTIZE ROWS STOOD HERE** — bf16 to MXFP4 and bf16
            // to FP8, the load-time encode §M-3 shut. The table is the two
            // rows a serving load still needs: a cast and a row scale.
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

    /// This arena's stream as the context an entry fires on.
    ///
    /// Built per call rather than held as a field: a [`Ctx`] is what a
    /// engine's `Run` mints per fire, it carries no state of its own (the jit
    /// cache and the scratch slabs are process-global behind it), and holding
    /// one would only restate `self.stream` in a second place that could
    /// disagree with the first.
    fn ctx(&self) -> Ctx {
        // SAFETY: `stream` is live for as long as this arena is, by
        // `CudaArena::new`'s safety contract, and this is a `&self` method.
        unsafe { Ctx::on(self.stream.cast()) }
    }

    /// The device address `bytes` into the arena, as the integer a
    /// [`Tensor`] carries. `Tensor::ptr` is a `u64` rather than a pointer
    /// because the host never dereferences it.
    fn device_ptr(&self, offset: usize) -> u64 {
        self.at(offset) as u64
    }

    /// `quant::cast_fp32_to`, at `bf16`: elementwise over a flat byte run,
    /// addressed by element count rather than the plan's 2-D shape (a `Cast`
    /// is not indexed by rows/cols). The entry reads that count off the
    /// DESTINATION handle (`Tensor::elements`) and picks its instantiation
    /// off the destination's dtype, so `Dtype::Bf16` here is what
    /// `CUDA_CAST_FP32_TO_BF16`'s name has always meant and the turbofish
    /// used to say.
    fn cast(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        self.bounds(op.src.offset, op.src.len)?;
        self.bounds(op.dst.offset, op.dst.len)?;
        let n = op.src.len / 4;
        // A cast of more than 4 Gi elements is refused rather than
        // truncated (truncation would leave the tail holding whatever
        // the arena had). Refused HERE, first, so this is a load error
        // naming the plan rather than a `panic!` inside the host
        // program, which has no `Result` to put it in.
        let width = u32::try_from(n).map_err(|_| {
            plan_disagrees("the cast covers more elements than a 32-bit launch extent states")
        })?;
        // ONE ROW OF `n`: the cast is elementwise, and `rows * width` is the
        // extent the entry counts in either way.
        let src = Tensor::new(self.device_ptr(op.src.offset), 1, width, Dtype::F32);
        let mut dst = Tensor::new(self.device_ptr(op.dst.offset), 1, width, Dtype::Bf16);
        let fired = quant::cast_fp32_to(&self.ctx(), src, &mut dst);
        declined(CUDA_CAST_FP32_TO_BF16, fired)
    }

    /// `quant::scale_rows`, at `bf16`, the per-row multiply, in place.
    ///
    /// `buf.rows` picks the block (`grid.x`) and `buf.width` sizes it and is
    /// read by the kernel's own loop over columns; both live inside
    /// `quant::route_rows`, next to the `<<<>>>` they size. The factors are
    /// one row of `cols` — the device text indexes them `l[c]`, by COLUMN,
    /// which is why they are a `[1, cols]` handle and not a `[rows, 1]` one.
    fn scale(&mut self, op: &TileMapOp<'_>) -> Result<(), Error> {
        let (rows, cols) = self.extent_2d(op)?;
        let factors = op
            .factors
            .ok_or_else(|| plan_disagrees("scale_rows reads per-group factors"))?;
        self.bounds(op.dst.offset, op.dst.len)?;
        self.bounds(factors.offset, factors.len)?;
        let l = Tensor::new(self.device_ptr(factors.offset), 1, cols, Dtype::Bf16);
        // `&mut`: the kernel scales `dst` IN PLACE, which the plan checked
        // before naming this row.
        let mut buf = Tensor::new(self.device_ptr(op.dst.offset), rows, cols, Dtype::Bf16);
        let fired = quant::scale_rows(&self.ctx(), l, &mut buf);
        declined(CUDA_SCALE_ROWS_BF16, fired)
    }

    /// The 2-D extent every row above takes, in the plan's own units.
    /// An `Err` rather than a decline: the plan chose a row that needs
    /// a shape, so its absence here means the two disagree.
    fn extent_2d(&self, op: &TileMapOp<'_>) -> Result<(u32, u32), Error> {
        op.shape
            .ok_or_else(|| plan_disagrees("the row takes a 2-D extent"))
    }
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
/// **NOTHING PANICS ANY MORE, and the paragraph that stood here said it
/// did.** It read: only `KernelError::Zero` (a collapsed rectangle) arrives
/// as an `Err`, while an unknown symbol or a unit that will not compile
/// panics, since a host program has no `Result` to put them in. That was
/// true of the routine layer and is not true of `Ctx::fire`, whose every
/// failure — no carried unit by that name, an empty grid, an instantiation
/// NVRTC refuses, a launch the runtime rejects — comes back as a
/// [`kernels_cuda::Error`] attributed to the op. So every one of them
/// reaches here, and every one of them fails the load with the row named,
/// which is what this function always claimed to be for.
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

    /// Every symbol the compiler may name is a PATH this build calls.
    ///
    /// The constants live in `plan::passes::tile` so a CUDA plan compiles on
    /// a machine with no CUDA, which means nothing there can check them; this
    /// is the other half, catching a renamed transform here instead of at
    /// load time.
    ///
    /// THE PATH IS THE BINDING AND THE STRING IS ITS SPELLING.
    /// `run_tile_map` above MATCHES these strings and then calls the
    /// functions by path, so what can drift is the spelling, not the call.
    /// The macro binds the function item — which is what makes a rename a
    /// compile error here — and composes the string out of the same tokens,
    /// so the two cannot disagree without this failing.
    ///
    /// This replaced a lookup in `kernels_cuda::routine(symbol)`, the by-name
    /// registry the routine layer carried. The registry is deleted; the
    /// question it was asked survives, and the compiler answers more of it.
    ///
    /// **IT CHECKED FOUR AND CHECKS TWO** (§M-3). The other two were the
    /// quantisers — bf16 to MXFP4 and bf16 to FP8 — and they went with the
    /// load-time encode door: no device mask carries an encode, so no plan
    /// can name their words, and a build with no launcher for a word no plan
    /// carries has nothing to drift from.
    ///
    /// THE STRINGS SAY `quant::`, THE PATHS SAY `linear::quant::`, AND THAT
    /// IS DELIBERATE. The entries moved namespace when the kernel plane was
    /// re-cut; the strings did not, because a plan's kernel word is a name
    /// this crate owns and writes into checked-in goldens, not a Rust path it
    /// mirrors. The macro therefore takes the path and the spelling
    /// separately, and this test is what holds them to each other.
    #[test]
    fn every_symbol_the_plan_may_name_is_a_path_this_build_calls() {
        macro_rules! spelled {
            ($ns:ident :: $f:ident) => {{
                // THE PATH, RESOLVED. A rename that missed the constant
                // stops this line from compiling.
                let _ = kernels_cuda::linear::$ns::$f;
                concat!(stringify!($ns), "::", stringify!($f))
            }};
        }
        assert_eq!(CUDA_CAST_FP32_TO_BF16, spelled!(quant::cast_fp32_to));
        assert_eq!(CUDA_SCALE_ROWS_BF16, spelled!(quant::scale_rows));
    }
}
