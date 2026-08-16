use core::ffi::c_void;

use kernels::routine::{Backend, Refusal};

use crate::jit::{ArgValue, Root};

/// This backend, as the machinery names it.
///
/// A marker: never constructed, carrying only the two concrete types the
/// `kernels` machinery is generic over.
#[derive(Clone, Copy, Debug)]
pub struct Cuda;

impl Backend for Cuda {
    type Value = ArgValue;
    type Ctx<'a> = Ctx;
}

/// One launch's geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// Blocks per axis.
    pub grid: [u32; 3],
    /// Threads per block per axis.
    pub block: [u32; 3],
    /// Dynamic shared memory, in bytes.
    pub smem: u32,
    /// Every block of this grid must be resident at once — `grid.sync()`.
    ///
    /// A per-LAUNCH attribute rather than a property of the kernel, which is
    /// why it lives here: `cuLaunchKernelEx` takes it, and the same entry
    /// point could in principle be launched either way.
    pub cooperative: bool,
}

impl Launch {
    /// ONE THREAD PER ELEMENT, in blocks of `block`.
    #[must_use]
    pub const fn flat(n: u32, block: u32) -> Self {
        let grid = if block == 0 { 0 } else { n.div_ceil(block) };
        Self { grid: [grid, 1, 1], block: [block, 1, 1], smem: 0, cooperative: false }
    }

    /// ONE BLOCK PER ROW, `block` threads wide.
    #[must_use]
    pub const fn per_row(rows: u32, block: u32) -> Self {
        Self { grid: [rows, 1, 1], block: [block, 1, 1], smem: 0, cooperative: false }
    }

    /// A grid stated on all three axes.
    #[must_use]
    pub const fn grid(grid: [u32; 3], block: [u32; 3]) -> Self {
        Self { grid, block, smem: 0, cooperative: false }
    }

    /// The same launch with `bytes` of dynamic shared memory.
    ///
    /// There is no `smem_opt_in` beside it. The >48 KiB opt-in is not a
    /// decision an author makes -- it follows from the number -- and stating
    /// it twice is what let the two disagree.
    #[must_use]
    pub const fn smem(mut self, bytes: u32) -> Self {
        self.smem = bytes;
        self
    }

    /// The same launch, cooperatively.
    #[must_use]
    pub const fn cooperative(mut self) -> Self {
        self.cooperative = true;
        self
    }

    /// Nothing to launch — an axis is zero.
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

/// What a routine body launches through.
///
/// The whole of what it HOLDS is two pointers: the stream the call was made
/// on, and the engine's cuBLAS handle when the caller had one to lend. Both
/// are per-call, which is why they are fields.
///
/// It also ANSWERS for the scratch allocator, the multiprocessor count and
/// the compute capability, and those are not fields and must not become
/// fields. Each is a property of the device rather than of a call, so each is
/// a process-wide static in `jit/device.rs` reached through a method here.
/// Copying one into every `Ctx` would make a per-call value out of a
/// per-device fact, and the way that fails is a `Ctx` built on one stream
/// answering with another device's multiprocessor count.
///
/// The distinction is worth the paragraph because the two halves read the
/// same at a call site — `ctx.stream()` and `ctx.multiprocessors()` — and
/// only one of them is something the caller could have got wrong.
pub struct Ctx {
    stream: *mut c_void,
    cublas: *mut c_void,
}

impl Ctx {
    /// A context for one call, on `stream`, with no cuBLAS handle.
    ///
    /// # Safety
    ///
    /// `stream` must name a live CUDA stream for as long as this value is used
    /// to launch — which outlives the launch itself, since a launch is
    /// asynchronous and ends when the stream is synchronised.
    #[must_use]
    pub const unsafe fn on(stream: *mut c_void) -> Self {
        Self { stream, cublas: core::ptr::null_mut() }
    }

    /// The same context, carrying the engine's cuBLAS handle.
    ///
    /// **Carried, not owned, and that is deliberate.** The plan had `Ctx` mint
    /// its own per-device handles; the engine's is configured with
    /// `CUBLAS_TENSOR_OP_MATH`, and `driver-cuda/src/device/cublas.rs` records
    /// the argument against a second one — it would be a second place for the
    /// math mode to be true, and `cublasDestroy` costs 3.2 ms. So the one
    /// handle the engine already makes and destroys is the one that crosses.
    ///
    /// # Safety
    ///
    /// `handle` must be a live `cublasHandle_t` for as long as this value is
    /// used, and its stream must be the one this context was built on — a
    /// GEMM on a different stream from the layer around it is a race with no
    /// error.
    #[must_use]
    pub const unsafe fn with_cublas(mut self, handle: *mut c_void) -> Self {
        self.cublas = handle;
        self
    }

    /// The stream every launch through this context goes to.
    #[must_use]
    pub const fn stream(&self) -> *mut c_void {
        self.stream
    }

    /// The engine's cuBLAS handle.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] if this context was built without one — which is
    /// every context `call()` makes, since a trace statement cannot name a
    /// handle. A cuBLAS routine is reachable from a caller that has one.
    pub fn cublas(&self) -> Result<*mut c_void, Refusal> {
        if self.cublas.is_null() {
            return Err(Refusal::Absent { what: "a cuBLAS handle" });
        }
        Ok(self.cublas)
    }

    /// A device scratch buffer of at least `bytes`, kept under `name`.
    ///
    /// Grow-only and process-lifetime, so the address a routine is handed
    /// stays valid for launches already in flight. Named, so two routines
    /// wanting scratch do not hand each other the same bytes.
    ///
    /// This is where a family's own `static Mutex<..>` over a `cudaMalloc`
    /// belongs: the buffer is a property of the device, not of a kernel.
    ///
    /// # Errors
    ///
    /// [`Refusal::Device`] if the allocation fails.
    #[allow(clippy::unused_self)]
    pub fn scratch(&self, name: &'static str, bytes: usize) -> Result<*mut c_void, Refusal> {
        #[cfg(feature = "_cuda")]
        {
            crate::jit::device::take(name, bytes)
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = (name, bytes);
            Err(Refusal::Device { why: "this build selected no CUDA runtime" })
        }
    }

    /// This device's compute-capability major number, or `None` if the driver
    /// will not say.
    ///
    /// `None` rather than a refusal, because a caller picking a tuning
    /// constant from it has a defensible answer for an unknown device.
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

    /// How many multiprocessors this device has.
    ///
    /// # Errors
    ///
    /// [`Refusal::Device`] if the driver will not say.
    #[allow(clippy::unused_self)]
    pub fn multiprocessors(&self) -> Result<u32, Refusal> {
        #[cfg(feature = "_cuda")]
        {
            crate::jit::device::multiprocessors()
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Refusal::Device { why: "this build selected no CUDA runtime" })
        }
    }

    /// Launch one instantiation of `root`, compiling it if this process has
    /// not yet.
    ///
    /// `instantiation` is the C++ expression NVRTC is asked to lower — the
    /// fully qualified template-id, not a label. It is written in the body
    /// because choosing it IS the body's job.
    ///
    /// # Errors
    ///
    /// [`Refusal::Empty`] for a grid with a zero axis, and
    /// [`Refusal::Device`] if the compile, the load or the launch refused —
    /// the detail goes to the log, once per instantiation, because a refusing
    /// kernel is fired once per layer per token.
    ///
    /// # Safety
    ///
    /// Every [`ArgValue::Ptr`] must address device memory live and large
    /// enough for the parameter the kernel reads it as. Nothing here checks
    /// that and nothing can: it is the same obligation every `<<<>>>` carried.
    /// Fire one instantiation out of one carried file.
    ///
    /// The file and the template-id are the two things a launch is, and both
    /// are named here rather than reached by path. A `static ROOT` beside a
    /// `mod inst` used to hold them; that is one indirection per launch for
    /// two strings that only ever have one reader, and `Root` is derivable
    /// from the file's name in full — see [`Root::new`] and `CONFIGURED`.
    ///
    /// # Errors
    ///
    /// [`Refusal::Undeclared`] if `file` names nothing the binary carries.
    /// That is a compile error at a `Root::new`, and here it cannot be: the
    /// name arrives as an argument. `every_instantiation_compiles` reads these
    /// literals out of the source and puts each through NVRTC, so the miss is
    /// caught at `cargo test` rather than at the first fire on a GPU.
    ///
    /// # Safety
    ///
    /// Every pointer in `args` must address live device memory of the extent
    /// the kernel reads it as, and must stay live across the launch.
    pub unsafe fn launch(
        &self,
        file: &'static str,
        instantiation: &str,
        launch: Launch,
        args: &[ArgValue],
    ) -> Result<(), Refusal> {
        let Some(root) = Root::of(file) else {
            return Err(Refusal::Undeclared);
        };
        // SAFETY: the caller's obligation, forwarded.
        unsafe { self.launch_at(&root, instantiation, launch, args) }
    }

    /// The same, for a root that is not one carried file compiled one way.
    ///
    /// The two lattices only: FA2 and XQA name their points apart from the
    /// file they share, so a point cannot be reached by naming a file. See
    /// [`Root::variant`].
    ///
    /// # Safety
    ///
    /// [`Ctx::launch`]'s.
    pub unsafe fn launch_at(
        &self,
        root: &Root,
        instantiation: &str,
        launch: Launch,
        args: &[ArgValue],
    ) -> Result<(), Refusal> {
        if launch.empty() {
            return Err(Refusal::Empty { what: "the grid" });
        }
        // SAFETY: the caller's obligation, forwarded.
        unsafe { self.fire(root, instantiation, launch, args) }
    }

    #[cfg(feature = "_cuda")]
    unsafe fn fire(
        &self,
        root: &Root,
        instantiation: &str,
        launch: Launch,
        args: &[ArgValue],
    ) -> Result<(), Refusal> {
        let resolved = match crate::jit::cache::resolve(root, instantiation) {
            Ok(resolved) => resolved,
            Err(why) => return Err(said(root.name, instantiation, &why.to_string())),
        };
        // SAFETY: `ArgValue::Bytes`' contract, forwarded from this function's.
        let mut bound = unsafe { crate::jit::value::Bound::new(args) };
        // SAFETY: the entry point came from a module this process keeps
        // loaded; the slots are one per parameter and live across the call;
        // the caller vouches for the pointers.
        let fired = unsafe {
            crate::jit::launch::issue(resolved.function, launch, bound.slots_mut(), self.stream)
        };
        match fired {
            Ok(()) => Ok(()),
            Err(why) => Err(said(root.name, instantiation, &why.to_string())),
        }
    }

    #[cfg(not(feature = "_cuda"))]
    #[allow(clippy::unused_self, clippy::needless_pass_by_value)]
    unsafe fn fire(
        &self,
        _root: &Root,
        _instantiation: &str,
        _launch: Launch,
        _args: &[ArgValue],
    ) -> Result<(), Refusal> {
        Err(Refusal::Device { why: "this build selected no CUDA runtime" })
    }
}

/// Say what went wrong, once per instantiation, and refuse.
///
/// The refusal itself is a `Copy` value and carries no message: the same
/// broken kernel is fired once per layer per token, so the detail is logged
/// once rather than returned every time.
#[cfg(feature = "_cuda")]
fn said(root: &str, instantiation: &str, why: &str) -> Refusal {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};

    static SAID: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let said = SAID.get_or_init(|| Mutex::new(HashSet::new()));
    if let Ok(mut said) = said.lock()
        && said.insert(instantiation.to_owned())
    {
        tracing::error!(root, instantiation, why, "a device instantiation will not fire");
    }
    Refusal::Device { why: "the compile, the load or the launch refused; see the log" }
}
