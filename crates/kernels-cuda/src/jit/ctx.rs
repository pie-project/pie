//! The context every entry takes: the stream a driver `Run` wraps, with the
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

/// The stream and its companions. Long-lived state (jit cache, scratch
/// slabs, device probes) is process-global behind it; the `Ctx` itself is
/// what a driver `Run` builds per fire and lends to every entry.
pub struct Ctx {
    stream: *mut c_void,
    cublas: *mut c_void,
    comm: *mut c_void,
    slabs: Slabs,
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
    /// **The contract, both ways.** Growth is `cudaFree` + `cudaMalloc`,
    /// which would poison a capture in progress. The driver's side: warm
    /// every scratch-consuming entry with an eager fire at full fire shape
    /// before capturing, so a captured fire only ever re-reads a slab that
    /// is already big enough — and the warm pass may fire on ONE stream,
    /// because growth is broadcast across every stream the arena has been
    /// told about. This plane's side: the cheap runtime guard in
    /// `device::take` — if this context's stream is mid-capture
    /// (`cudaStreamIsCapturing`) and the slab would have to grow, the fire
    /// comes back as a [`KernelError::Backend`] refusal naming the
    /// un-warmed slab instead of corrupting the capture.
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
