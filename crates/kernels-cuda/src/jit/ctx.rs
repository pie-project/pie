//! What a routine body launches through: the backend marker, one launch's
//! geometry, and the per-call context.
//!
//! [`Cuda`] names this backend to the `kernels` machinery; [`Launch`] is a
//! grid, a block, dynamic shared memory and the cooperative flag; [`Ctx`]
//! holds only borrowed per-call state — the stream, the engine's cuBLAS
//! handle if one was lent, and this rank's [`Plane`] if the fire is a
//! collective's. Device-wide facts (scratch, multiprocessor count, compute
//! capability) are process-wide statics in `jit/device.rs`, reached through
//! methods here.

use kernels::routine::Fire;
use core::ffi::c_void;

use kernels::routine::{Backend, Extent, Refusal};

use crate::comm::Plane;
use crate::jit::{ArgValue, Root};

/// This backend, as the machinery names it.
///
/// A marker: never constructed, carrying only the two concrete types the
/// `kernels` machinery is generic over.
#[derive(Clone, Copy, Debug)]
pub struct Cuda;

impl Backend for Cuda {
    type Value = ArgValue;
    type Ctx<'a> = Ctx<'a>;

    fn region(value: &ArgValue) -> Result<Extent, Refusal> {
        match *value {
            ArgValue::Region { rows, width, .. } => Ok(Extent { rows, width }),
            // `Absent`, not `Kind`: the value carries no `Ty` mismatch, just
            // no width/rows to report.
            _ => Err(Refusal::Absent {
                what: "a region's shape: the bound value carries only an address",
            }),
        }
    }
}

// THE ASKING SIDE, over the resolver the driver lent. `Asks` is
// blanket-implemented for every `Answers`, so this is the whole of what this
// plane states to give its bodies `ctx.ask::<C, keys::X>()`, `ctx.params()`
// and `ctx.absent()`.
impl kernels::routine::Answers<Cuda> for Ctx<'_> {
    fn resolve(
        &self,
        ty: kernels::Ty,
        source: kernels::Source,
    ) -> Result<ArgValue, Refusal> {
        self.env
            .ok_or(Refusal::Unstated {
                what: "a fact, on a context built for a hand-written call: \
                       nothing behind it holds this fire's answers",
            })?
            .resolve(ty, source)
    }
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
    /// A per-launch attribute: the same entry point can be launched either way.
    pub cooperative: bool,
}

impl kernels::routine::Geometry for Launch {
    fn apply_to(self, fire: Fire) -> Fire {
        self.apply_to_impl(fire)
    }
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
    /// No separate `smem_opt_in`: the >48 KiB opt-in follows from `bytes`.
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

    /// This geometry, applied to a [`Fire`].
    ///
    /// `Fire` counts LANES and CUDA counts BLOCKS, which is the same pair from
    /// the other end: `lanes = grid * block`, and `Fire::grid()` divides back
    /// to exactly the grid stated here.
    #[must_use]
    fn apply_to_impl(self, fire: Fire) -> Fire {
        fire.geometry(
            [
                self.grid[0].saturating_mul(self.block[0]),
                self.grid[1].saturating_mul(self.block[1]),
                self.grid[2].saturating_mul(self.block[2]),
            ],
            self.block,
            self.smem,
            self.cooperative,
        )
    }

    /// This geometry, as the shared [`Fire`] at `file`'s `entrypoint`.
    ///
    /// `Fire` counts LANES and CUDA counts BLOCKS, which is the same pair from
    /// the other end: `lanes = grid * block`, and `Fire::grid()` divides back
    /// to exactly the grid stated here. Counting the total is what lets one
    /// field mean one thing on four planes.
    #[must_use]
    pub const fn at(self, file: &'static str, entrypoint: &'static str) -> Fire {
        Fire {
            file,
            entrypoint,
            unit: "",
            lanes: [
                self.grid[0].saturating_mul(self.block[0]),
                self.grid[1].saturating_mul(self.block[1]),
                self.grid[2].saturating_mul(self.block[2]),
            ],
            group: self.block,
            smem: self.smem,
            cooperative: self.cooperative,
        }
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
/// Holds only per-call BORROWED state as fields: the stream, the engine's
/// cuBLAS handle (if lent), and this rank's plane (if the fire is a
/// collective's). The scratch allocator, multiprocessor count and compute
/// capability are device-wide facts, not per-call, so they stay process-wide
/// statics in `jit/device.rs` reached through a method here instead.
pub struct Ctx<'a> {
    stream: *mut c_void,
    cublas: *mut c_void,
    comm: Option<Plane>,
    /// WHAT THIS FIRE ANSWERS, for a body that asks.
    ///
    /// `None` on a context built for a hand-written call, which is every
    /// `Ctx::on` in a driver arm: those pass their arguments directly and ask
    /// nothing. A body that asks on one of them gets [`Refusal::Unstated`],
    /// which is the honest answer -- there is no fire behind it to ask.
    ///
    /// A trait object because `kernels-cuda` cannot name the driver's
    /// resolver, exactly as the shader planes cannot name their encoder.
    env: Option<&'a (dyn kernels::routine::Answers<Cuda> + 'a)>,
    /// What the three raw pointers above are borrowed FROM.
    ///
    /// The stream and the cuBLAS handle already had to outlive every launch
    /// made through this value -- `Ctx::on`'s `# Safety` says so in prose. A
    /// lifetime says it to the compiler instead, and makes this plane's `&Ctx`
    /// read as the other three planes' `&Ctx<'_>`, which is what a shader
    /// plane's `dyn Encode + 'a` has always spelled.
    held: core::marker::PhantomData<&'a ()>,
}

impl<'a> Ctx<'a> {
    /// A context for one call, on `stream`, with no cuBLAS handle and no
    /// plane.
    ///
    /// # Safety
    ///
    /// `stream` must name a live CUDA stream for as long as this value is used
    /// to launch — which outlives the launch itself, since a launch is
    /// asynchronous and ends when the stream is synchronised.
    #[must_use]
    pub const unsafe fn on(stream: *mut c_void) -> Self {
        Self {
            stream,
            cublas: core::ptr::null_mut(),
            comm: None,
            env: None,
            held: core::marker::PhantomData,
        }
    }

    /// The same context, carrying the engine's cuBLAS handle.
    ///
    /// Carried, not owned: a second handle would be a second place the
    /// `CUBLAS_TENSOR_OP_MATH` math mode must hold, and `cublasDestroy` costs
    /// 3.2 ms, so the engine's own handle is the one that crosses.
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

    /// The same context, carrying this rank's tensor-parallel plane.
    ///
    /// Carried, not owned: the `Plane` is a `Copy` handle (eight peer
    /// `Signal*` slots, a `RankData*` and a fusion workspace address, each a
    /// mapping of another process' allocation via `cudaIpcOpenMemHandle`).
    /// The owning lifecycle — registering buffers and closing the mappings
    /// in `Drop` — stays in the shell; a second owner here would close IPC
    /// mappings peers are still reducing through, which hangs rather than
    /// merely costing.
    ///
    /// # Safety
    ///
    /// `plane` must describe a live P2P plane whose peers stay mapped for as
    /// long as this value is used to launch — which outlives the launch,
    /// a collective being asynchronous like anything else on a stream.
    #[must_use]
    pub const unsafe fn with_comm(mut self, plane: Plane) -> Self {
        self.comm = Some(plane);
        self
    }

    /// The same context, able to ANSWER a body that asks.
    ///
    /// What `Env` parameters used to arrive through. The driver resolves a
    /// `(Ty, Source)` pair for every argument it binds already — that is
    /// `kernels::bind::one` over its own `Holds` — and this lends the same
    /// resolver to the body, so a fact only the fire can answer stops having
    /// to be a parameter for the column to carry it.
    #[must_use]
    pub const fn with_env(mut self, env: &'a (dyn kernels::routine::Answers<Cuda> + 'a)) -> Self {
        self.env = Some(env);
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

    /// This rank's tensor-parallel plane.
    ///
    /// # Errors
    ///
    /// [`Refusal::Absent`] if this context was built without one, which is
    /// every context `call()` makes and every context a single-rank
    /// deployment builds.
    ///
    /// `comm`'s bodies do not forward this `Refusal`: they answer
    /// `Decline::NoInstance`, which `bind/arms/comm.rs` routes to NCCL on —
    /// an absent plane is a routing answer, not a failure.
    pub fn comm(&self) -> Result<Plane, Refusal> {
        let Some(plane) = self.comm else {
            return Err(Refusal::Absent { what: "a tensor-parallel plane" });
        };
        Ok(plane)
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
    /// fully qualified template-id, not a label — chosen by the caller.
    ///
    /// # Errors
    ///
    /// [`Refusal::Empty`] for a grid with a zero axis; [`Refusal::Undeclared`]
    /// if `file` names nothing the binary carries (caught at `cargo test` by
    /// `every_instantiation_compiles`, not here); [`Refusal::Device`] if the
    /// compile, the load or the launch refused — the detail goes to the log,
    /// once per instantiation, because a refusing kernel is fired once per
    /// layer per token.
    ///
    /// # The pointer obligation, and where it lives
    ///
    /// Every [`ArgValue::Ptr`] in `args` must address live device memory of
    /// the extent the kernel reads it as, and must stay live across the
    /// launch. Nothing here checks that; it is the same obligation every
    /// `<<<>>>` carried.
    ///
    /// It is the ROUTINE's obligation and not this call's, which is why this
    /// is not `unsafe fn`. A routine takes its pointers as `In<E>`/`Out<E>`
    /// and documents them under its own `# Safety`; the `unsafe { }` that used
    /// to sit here wrapped the whole argument list and so marked no particular
    /// pointer, while making a CUDA body legible as a CUDA body at a glance --
    /// four hundred and sixty-five of them, against none on the other three
    /// planes running the same dispatch.
    pub fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
        // A NAMED UNIT OR THE FILE'S OWN. `Root::variant` is how FA2 and XQA
        // reach a point in a file they compile several ways; everything else
        // states no unit and the file names its root.
        let root = if fire.unit.is_empty() {
            match Root::of(fire.file) {
                Some(root) => root,
                None => return Err(Refusal::Undeclared),
            }
        } else {
            Root::variant(fire.unit, fire.file)
        };
        let launch = Launch {
            grid: fire.grid(),
            block: fire.group,
            smem: fire.smem,
            cooperative: fire.cooperative,
        };
        // SAFETY: the routine's obligation, forwarded -- see this method's doc.
        unsafe { self.launch_at(&root, fire.entrypoint, launch, args) }
    }

    /// The same, for a root the caller already holds.
    ///
    /// # Safety
    ///
    /// [`Ctx::fire`]'s obligation, which is the routine's.
    unsafe fn launch_at(
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
        unsafe { self.issue(root, instantiation, launch, args) }
    }

    /// Resolve `instantiation` in `root` and issue it.
    ///
    /// Private and named apart from [`Ctx::fire`]: that one is the ROUTINE's
    /// door and takes a [`Fire`], this one is what it calls once the module is
    /// known. They were both `fire` for a while, which is one inherent impl
    /// with two methods of one name.
    #[cfg(feature = "_cuda")]
    unsafe fn issue(
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

    /// [`Ctx::issue`] where the build selected no CUDA runtime.
    #[cfg(not(feature = "_cuda"))]
    #[allow(clippy::unused_self, clippy::needless_pass_by_value)]
    unsafe fn issue(
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
