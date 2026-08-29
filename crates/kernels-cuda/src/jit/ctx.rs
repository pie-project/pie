//! The context every entry takes: the stream an engine `Run` wraps, with the
//! cuBLAS handle and device probes beside it. `fire` resolves a [`Fire`]'s
//! unit through the module cache and enqueues the launch — enqueue only,
//! never sync.

use core::ffi::c_void;

use kernels::KernelError;

use crate::jit::{ArgValue, refuse};

/// Dispatch geometry: grid x block, dynamic shared memory, and the
/// cooperative flag a grid-synchronising kernel needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub smem: u32,
    pub cooperative: bool,
}

impl Launch {
    /// One thread per element, flattened: `ceil(n / block)` blocks.
    #[must_use]
    pub const fn flat(n: u32, block: u32) -> Self {
        let grid = if block == 0 { 0 } else { n.div_ceil(block) };
        Self {
            grid: [grid, 1, 1],
            block: [block, 1, 1],
            smem: 0,
            cooperative: false,
        }
    }

    /// One block per row.
    #[must_use]
    pub const fn per_row(rows: u32, block: u32) -> Self {
        Self {
            grid: [rows, 1, 1],
            block: [block, 1, 1],
            smem: 0,
            cooperative: false,
        }
    }

    #[must_use]
    pub const fn grid(grid: [u32; 3], block: [u32; 3]) -> Self {
        Self {
            grid,
            block,
            smem: 0,
            cooperative: false,
        }
    }

    #[must_use]
    pub const fn smem(mut self, bytes: u32) -> Self {
        self.smem = bytes;
        self
    }

    #[must_use]
    pub const fn cooperative(mut self) -> Self {
        self.cooperative = true;
        self
    }

    #[must_use]
    pub const fn empty(&self) -> bool {
        self.grid[0] == 0
            || self.grid[1] == 0
            || self.grid[2] == 0
            || self.block[0] == 0
            || self.block[1] == 0
            || self.block[2] == 0
    }
}

/// One launch, fully named: the `.cuh` unit under `kernels/`, the
/// instantiation NVRTC lowers, and the geometry it dispatches at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fire {
    pub file: &'static str,

    /// The C++ instantiation expression — `::pie::` spelling, dtype stamped
    /// in (composed names go through [`symbol`](crate::jit::symbol)).
    pub entrypoint: &'static str,

    pub launch: Launch,
}

impl Fire {
    #[must_use]
    pub const fn at(file: &'static str, entrypoint: &'static str) -> Self {
        Self {
            file,
            entrypoint,
            launch: Launch::grid([0, 0, 0], [0, 0, 0]),
        }
    }

    #[must_use]
    pub const fn apply(mut self, launch: Launch) -> Self {
        self.launch = launch;
        self
    }
}

/// **ONE CONTEXT'S SCRATCH SLABS**, minted by [`Slabs::open`] and freed by
/// [`Slabs::release`].
///
/// A scratch slab is the workspace an entry may not allocate per fire, and
/// the question this handle answers is WHOSE. It used to be nobody's: the
/// slabs were keyed by a static name and shared by every `Ctx` in the
/// process, so two shells staged into one another's planes and both computed
/// (build log 18 measured the garbage; the tree's workaround was one shell
/// per process). An arena makes the sharing explicit — one per CUDA context,
/// and within it one slab per `(name, stream)`, which is the other half of
/// the same key (build log 24's `EXCLUSIVE` list existed only because two
/// forked arms shared a name).
///
/// Handles are `Copy` and mean nothing but an identity; the storage is
/// process-global behind them, like the jit cache beside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slabs(u32);

impl Slabs {
    /// **THE ARENA A BARE [`Ctx::on`] FIRES AGAINST.** Shared by everything
    /// that never asked for one of its own — the model-loader's transform
    /// executor, benches, a test that wants a stream and nothing else — and
    /// still per stream inside, so sharing it is not sharing a slab.
    pub const PROCESS: Slabs = Slabs(0);

    /// A fresh arena, disjoint from every other. One per `Context`.
    #[must_use]
    pub fn open() -> Slabs {
        static NEXT: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(1);
        Slabs(NEXT.fetch_add(1, core::sync::atomic::Ordering::Relaxed))
    }

    /// Tell this arena it fires on `stream` too.
    ///
    /// **CALL IT BEFORE THE FIRST FIRE ON THAT STREAM.** Growth is broadcast
    /// across an arena's attached streams — that is what lets the shell's
    /// EAGER warm pass, which runs on one stream, size the slabs the CAPTURE
    /// pass will read on the others (`jit::device`'s header argues it). A
    /// stream attached late gets its slab on the name's next growth, and a
    /// capture is where there is no next growth.
    ///
    /// # Safety
    ///
    /// `stream` must be a live `cudaStream_t`, and must not outlive this
    /// arena's [`release`](Slabs::release).
    pub unsafe fn attach(self, stream: *mut c_void) {
        #[cfg(feature = "_cuda")]
        {
            crate::jit::device::attach(self.0, stream);
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = stream;
        }
    }

    /// Free every slab in this arena.
    ///
    /// The context's teardown, and nothing else: a slab is read only by a
    /// launch enqueued on one of the arena's own streams, so a context that
    /// has synchronized has nothing left pointing at one. Freeing is what
    /// makes a second shell in one process cost what the first did.
    pub fn release(self) {
        #[cfg(feature = "_cuda")]
        {
            crate::jit::device::release(self.0);
        }
    }
}

/// **THIS FIRE'S ROW COUNT AND THE BUCKET IT ROUNDS UP TO** — the whole of
/// what D4 (`.wiki/palo/cuda-abi.md` §3) sends down to the entries.
///
/// A `Pad` is not a permission to round; it is the pair of numbers that makes
/// rounding CHECKABLE, and [`Ctx::opaque_rows`] is the only reader. `rows` is
/// the fire's TOTAL row count — the extent an Always region's launch is handed
/// — and `bucket` is what `engine::fire::compose` rounded it up to. An engine
/// that arms neither leaves the default, which is `rows == bucket == 0`: no
/// extent equals it, so nothing is ever padded and the plane is byte-for-byte
/// the one that existed before this field did. That default is also what a
/// WINDOWED region is armed with, which is where the boundary is enforced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pad {
    /// The fire's total token rows.
    pub rows: u32,
    /// The smallest lattice point that holds them (`Composition::bucket`).
    pub bucket: u32,
}

/// The stream and its companions. Long-lived state (jit cache, scratch
/// slabs, device probes) is process-global behind it; the `Ctx` itself is
/// what an engine `Run` builds per fire and lends to every entry.
pub struct Ctx {
    stream: *mut c_void,
    cublas: *mut c_void,
    comm: *mut c_void,
    slabs: Slabs,

    /// D4's number, written per REGION by the engine's walk ([`Ctx::arm`]).
    ///
    /// **A `Cell` BECAUSE THE CONTEXT OUTLIVES THE FIRE AND THE NUMBER DOES
    /// NOT.** One `Ctx` is minted per stream at LOAD (`device::Context::bind`,
    /// `open_lanes`) and lent to every entry of every fire after it, so the
    /// fire-lived half of what an entry needs cannot be a constructor
    /// argument without re-minting a cuBLAS handle per fire. A `Cell<Pad>` is
    /// eight bytes the shell stamps between two fires; it is never read across
    /// threads, because a `Ctx` holds raw handles and is neither `Send` nor
    /// `Sync` already.
    pad: core::cell::Cell<Pad>,
}

impl Ctx {
    /// # Safety
    /// `stream` must be a live `cudaStream_t` for as long as this context
    /// fires on it.
    #[must_use]
    pub const unsafe fn on(stream: *mut c_void) -> Self {
        Self {
            stream,
            cublas: core::ptr::null_mut(),
            comm: core::ptr::null_mut(),
            slabs: Slabs::PROCESS,
            pad: core::cell::Cell::new(Pad {
                rows: 0,
                bucket: 0,
            }),
        }
    }

    /// The same context, firing against `slabs` rather than the process
    /// arena — what a shell that opened its own [`Slabs`] hands every entry.
    #[must_use]
    pub const fn with_slabs(mut self, slabs: Slabs) -> Self {
        self.slabs = slabs;
        self
    }

    /// # Safety
    /// `handle` must be a live `cublasHandle_t` bound to this context's
    /// stream.
    #[must_use]
    pub const unsafe fn with_cublas(mut self, handle: *mut c_void) -> Self {
        self.cublas = handle;
        self
    }

    /// # Safety
    /// `comm` must be a live `ncclComm_t` whose clique this context's stream
    /// belongs to, for as long as this context fires collectives on it.
    #[must_use]
    pub const unsafe fn with_comm(mut self, comm: *mut c_void) -> Self {
        self.comm = comm;
        self
    }

    #[must_use]
    pub const fn stream(&self) -> *mut c_void {
        self.stream
    }

    pub fn cublas(&self, op: &'static str) -> Result<*mut c_void, KernelError> {
        if self.cublas.is_null() {
            return Err(refuse(op, "this context carries no cuBLAS handle"));
        }
        Ok(self.cublas)
    }

    /// The NCCL communicator a tensor-parallel run carries. Absent on a
    /// single-rank context — a collective fired there is a typed refusal,
    /// not a hang.
    pub fn comm(&self, op: &'static str) -> Result<*mut c_void, KernelError> {
        if self.comm.is_null() {
            return Err(refuse(op, "this context carries no communicator"));
        }
        Ok(self.comm)
    }

    /// A named scratch slab, grown but never shrunk — the workspace an entry
    /// may not allocate per fire (graph capture forbids it).
    ///
    /// **THE KEY IS `(arena, name, stream)`, AND NONE OF THE THREE IS
    /// DECORATION.** The arena is this context's [`Slabs`], so two shells in
    /// one process no longer stage into one another's planes; the stream is
    /// the one this `Ctx` fires on, so two arms of a P6 fork group get two
    /// slabs and the compiler no longer has to order them apart. `jit::device`
    /// carries the measurements both halves come from.
    ///
    /// **The contract, both ways.** Growth allocates a fresh block, which
    /// would split a capture in progress across two addresses. The engine's
    /// side: warm every scratch-consuming entry with an eager fire at full
    /// fire shape before capturing, so a captured fire only ever re-reads a
    /// slab that is already big enough — and the warm pass may fire on ONE
    /// stream, because growth is broadcast across every stream the arena has
    /// been told about. This plane's side: the cheap runtime guard in
    /// `device::take` — if this context's stream is mid-capture
    /// (`cudaStreamIsCapturing`) and the slab would have to grow, the fire
    /// comes back as a [`KernelError::Backend`] refusal naming the
    /// un-warmed slab instead of corrupting the capture.
    ///
    /// **AND WHAT NEITHER SIDE CAN PROMISE**: that no LATER fire grows a name
    /// a graph already baked. The shapes a serving load brings are not
    /// bounded by the ones it has brought, so growth after a capture is
    /// ordinary, and `jit::device` answers it by RETIRING the superseded
    /// block rather than freeing it — the address a recorded graph holds
    /// stays its own for the life of the arena. The module comment there
    /// carries the measurement.
    pub fn scratch(
        &self,
        op: &'static str,
        name: &'static str,
        bytes: usize,
    ) -> Result<*mut c_void, KernelError> {
        #[cfg(feature = "_cuda")]
        {
            crate::jit::device::take(self.slabs.0, self.stream, name, bytes)
                .map_err(|fault| fault.at(op))
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = (name, bytes);
            Err(crate::jit::runtimeless(op))
        }
    }

    /// Which arena this context's slabs come from.
    #[must_use]
    pub const fn slabs(&self) -> Slabs {
        self.slabs
    }

    /// **STAMP D4'S TWO NUMBERS ON THIS CONTEXT, FOR THE LAUNCH ABOUT TO RUN.**
    ///
    /// **PER REGION, NOT PER FIRE**, and that is the whole of why this is a
    /// setter on a long-lived context rather than an argument to a
    /// constructor. The engine calls it from `Run::ctx` — the lookup every
    /// dispatch arm goes through to find its stream — with the walk's cursor
    /// already on the node about to launch, so what it stamps is the pad THIS
    /// REGION is allowed. A region whose window is not the whole fire is
    /// stamped `Pad::default()`, because padding a window writes the next
    /// class's rows of the same column.
    ///
    /// A `Pad` whose `bucket` is not above its `rows` arms nothing — which is
    /// what a deployment with no lattice (`Composition::bucket == rows`), a
    /// shell with `PIE_CUDA_PAD=off`, and a windowed region all hand over, so
    /// none of the three needs an arm of its own here.
    pub fn arm(&self, pad: Pad) {
        self.pad.set(pad);
    }

    /// Put the pad back to "no fire is running". Every launch after it is
    /// quantized by nothing, which is what a warm pass, a weight transform or
    /// a bench firing on this stream between two fires must see.
    pub fn disarm(&self) {
        self.pad.set(Pad::default());
    }

    /// What this fire told the entries to round to, for a caller that wants
    /// to report it.
    #[must_use]
    pub fn pad(&self) -> Pad {
        self.pad.get()
    }

    /// **THE ROW COUNT AN OPAQUE CALLEE IS TOLD ABOUT** — D4, in one call.
    ///
    /// An entry that hands its shape to a library planner nobody publishes the
    /// arm table of (cuBLASLt: `.wiki/palo/cuda-abi.md` §1 counted 151
    /// arm-switching nodes and a non-monotone splitK band on the gemm's M
    /// alone) asks for its `M` here instead of using the extent it was handed.
    /// Every kernel this tree OWNS keeps the live extent and the zero-row
    /// contract; this is not a general rounding service.
    ///
    /// **THE SAFETY ARGUMENT IS TWO GATES, AND ONLY THE SECOND ONE IS HERE.**
    /// Padding is legal exactly when the rows `[rows, bucket)` a padded call
    /// reads and writes belong to NOBODY:
    ///
    /// * **In bounds.** The arena reserves every `Dim::Tokens` column at
    ///   `max_tokens` rows and hands out static offsets (`engine_cuda::arena`),
    ///   and P0 refuses a lattice with a bucket past that ceiling
    ///   (`model_compiler`'s `accept`: "list a bucket past the token ceiling").
    ///   So a fire's tail rows are reserved bytes, not somebody's allocation.
    /// * **Harmless.** A gemm is ROW-INDEPENDENT: output row `i` is a function
    ///   of input row `i` and the weight. Garbage in the tail rows —
    ///   uninitialized bytes included, since nothing stages them — contaminates
    ///   tail rows and stops there. An entry that REDUCED over rows would fold
    ///   that garbage into a real row and compute rather than fault, which is
    ///   why the property is stated here, at the quantization, and not left to
    ///   be rediscovered by whoever adds the next opaque entry.
    /// * **Nobody's.** This is the clause the comparison enforces. A WINDOWED
    ///   launch's tail is not the fire's tail — it is the NEXT class's rows of
    ///   the same column, and under a merge or a co-tenant those are real
    ///   bytes somebody reads. Padding one is a clobber that computes.
    ///
    /// **The first gate is the engine's, and it is the one that decides.**
    /// `engine_cuda::run::Run::ctx` compares the region's WINDOW against the
    /// composition — span at row zero, covering every row, not gathered, no
    /// segment list — and arms this context with `Pad::default()` when any
    /// clause fails. That is the question asked where the answer lives: an
    /// entry sees extents, and an extent cannot tell the fire's rows from a
    /// window that happens to hold as many.
    ///
    /// **The second gate is this comparison, and it is a belt.** Inside a
    /// region that MAY pad, the pad still applies only to an extent that is the
    /// fire's row count — so an entry that hands over something other than the
    /// rows of the rectangle it was given (half of them, twice them, a
    /// `Dim::Const` width) gets its own number back rather than a rounded one.
    ///
    /// **The residue, named.** The two gates together cannot separate a
    /// token-shaped extent from a LANE-shaped one in a fire that carries one
    /// row per lane, because both are then the same integer and a `Dim::Lanes`
    /// column is reserved at `max_lanes` rather than at `max_tokens`. No opaque
    /// callee on this plane is lane-shaped — the catalog's cuBLASLt entries all
    /// take `Dim::Tokens` rectangles — and one added later must not read this
    /// function without the engine first learning to say which axis it is
    /// quantizing.
    #[must_use]
    pub fn opaque_rows(&self, rows: i32) -> i32 {
        let pad = self.pad.get();
        // Disarmed, or a deployment with no lattice: `bucket_of` answers
        // `rows` itself and there is nothing above it to round to.
        if pad.bucket <= pad.rows {
            return rows;
        }
        // NOT THE FULL FIRE — a window, and windows are never padded.
        if rows < 0 || rows.unsigned_abs() != pad.rows {
            return rows;
        }
        // The `max` is the seatbelt, not the arithmetic: `bucket >= rows` is
        // `bucket_of`'s post-condition and a bucket past `i32` is a lattice no
        // `Dim::Tokens` column was ever cut for.
        i32::try_from(pad.bucket).unwrap_or(rows).max(rows)
    }

    #[allow(clippy::unused_self)]
    #[must_use]
    pub fn compute_capability_major(&self) -> Option<u32> {
        #[cfg(feature = "_cuda")]
        {
            crate::jit::device::compute_capability_major()
        }
        #[cfg(not(feature = "_cuda"))]
        {
            None
        }
    }

    #[allow(clippy::unused_self)]
    #[must_use]
    pub fn multiprocessors(&self) -> Option<u32> {
        #[cfg(feature = "_cuda")]
        {
            crate::jit::device::multiprocessors()
        }
        #[cfg(not(feature = "_cuda"))]
        {
            None
        }
    }

    /// Enqueue one launch. `Ok` means the launch is on the stream, not that
    /// it ran; every failure comes back attributed to `op`.
    pub fn fire(&self, op: &'static str, fire: Fire, args: &[ArgValue]) -> Result<(), KernelError> {
        let Some(root) = crate::jit::Root::of(fire.file) else {
            return Err(refuse(
                op,
                format!("no carried unit is named `{}`", fire.file),
            ));
        };
        if fire.launch.empty() {
            return Err(refuse(op, "the grid is empty"));
        }
        self.issue(op, &root, fire.entrypoint, fire.launch, args)
    }

    #[cfg(feature = "_cuda")]
    fn issue(
        &self,
        op: &'static str,
        root: &crate::jit::Root,
        instantiation: &'static str,
        launch: Launch,
        args: &[ArgValue],
    ) -> Result<(), KernelError> {
        let resolved = match crate::jit::cache::resolve(root, instantiation) {
            Ok(resolved) => resolved,
            Err(why) => return Err(said(root.name, instantiation, why).at(op)),
        };

        let mut bound = crate::jit::abi::Bound::new(args);

        let fired = unsafe {
            crate::jit::launch::issue(resolved.function, launch, bound.slots_mut(), self.stream)
        };
        match fired {
            Ok(()) => Ok(()),
            Err(why) => Err(said(root.name, instantiation, why).at(op)),
        }
    }

    #[cfg(not(feature = "_cuda"))]
    #[allow(clippy::unused_self, clippy::needless_pass_by_value)]
    fn issue(
        &self,
        op: &'static str,
        _root: &crate::jit::Root,
        _instantiation: &'static str,
        _launch: Launch,
        _args: &[ArgValue],
    ) -> Result<(), KernelError> {
        Err(crate::jit::runtimeless(op))
    }
}

/// Report a refusal once per instantiation — the same broken row is fired
/// once per layer per token, and the caller already gets the error back.
#[cfg(feature = "_cuda")]
fn said(root: &str, instantiation: &str, why: crate::jit::Fault) -> crate::jit::Fault {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};

    static SAID: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let said = SAID.get_or_init(|| Mutex::new(HashSet::new()));
    if let Ok(mut said) = said.lock()
        && said.insert(instantiation.to_owned())
    {
        tracing::error!(
            root,
            instantiation,
            why = %why,
            "a device instantiation will not fire"
        );
    }
    why
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A context on no stream. Nothing below touches the device — the
    /// quantization is arithmetic over two `u32`s, and a test that needed a
    /// GPU to check it would not be run.
    fn bare() -> Ctx {
        // SAFETY: no method called here fires, so the null stream is never
        // handed to the runtime.
        unsafe { Ctx::on(core::ptr::null_mut()) }
    }

    #[test]
    fn an_unarmed_context_quantizes_nothing() {
        let ctx = bare();
        for rows in [0, 1, 3, 9, 4096] {
            assert_eq!(
                ctx.opaque_rows(rows),
                rows,
                "a context no fire armed hands back the extent it was given"
            );
        }
    }

    #[test]
    fn a_deployment_with_no_lattice_is_its_own_bucket_and_pads_by_zero() {
        let ctx = bare();
        // `compose::bucket_of` answers `rows` itself when `Budget::buckets`
        // is empty, and the shell's `PIE_CUDA_PAD=off` arm says the same
        // thing by choice. Both must be the identity.
        ctx.arm(Pad { rows: 9, bucket: 9 });
        assert_eq!(ctx.opaque_rows(9), 9);
    }

    #[test]
    fn the_full_fires_extent_rounds_up_to_the_bucket() {
        let ctx = bare();
        ctx.arm(Pad {
            rows: 9,
            bucket: 16,
        });
        assert_eq!(
            ctx.opaque_rows(9),
            16,
            "an Always launch is handed the fire's rows and computes the bucket's"
        );
    }

    /// **THE BOUNDARY THE DESIGN NAMES** (`.wiki/palo/cuda-abi.md` §3): an
    /// Always launch's tail is the fire's tail and belongs to nobody, but a
    /// WINDOWED launch padded past its window writes the next class's rows of
    /// the same column — a clobber that computes.
    ///
    /// The gate that decides it is the engine's (`Run::ctx` never arms a
    /// windowed region at all); this is the belt under it, and what it holds
    /// is that an extent which is not the fire's own row count is never
    /// rounded even inside a region that may pad.
    #[test]
    fn a_windowed_extent_is_never_padded_however_close_it_comes() {
        let ctx = bare();
        ctx.arm(Pad {
            rows: 9,
            bucket: 16,
        });
        for windowed in [1, 3, 8, 10, 16] {
            assert_eq!(
                ctx.opaque_rows(windowed),
                windowed,
                "{windowed} is not this fire's row count, so it is somebody's window"
            );
        }
    }

    #[test]
    fn disarming_puts_the_extent_back_the_way_the_fire_found_it() {
        let ctx = bare();
        ctx.arm(Pad {
            rows: 9,
            bucket: 16,
        });
        ctx.disarm();
        assert_eq!(
            ctx.opaque_rows(9),
            9,
            "the pad is the fire's, and the stream outlives the fire"
        );
    }

    /// A bucket no `Dim::Tokens` column was ever cut for is not a licence to
    /// launch: the seatbelt hands back the live extent rather than a negative
    /// `M`.
    #[test]
    fn a_bucket_past_the_kernels_int_falls_back_on_the_live_extent() {
        let ctx = bare();
        ctx.arm(Pad {
            rows: 9,
            bucket: u32::MAX,
        });
        assert_eq!(ctx.opaque_rows(9), 9);
    }
}
