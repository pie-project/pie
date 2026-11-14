#![allow(unsafe_code)]
use core::ffi::c_void;

use crate::error::Error;

use crate::jit::{ArgValue, Kernel, refuse};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub smem: u32,
}

impl Launch {
    #[must_use]
    pub const fn flat(n: u32, block: u32) -> Self {
        let grid = if block == 0 { 0 } else { n.div_ceil(block) };
        Self {
            grid: [grid, 1, 1],
            block: [block, 1, 1],
            smem: 0,
        }
    }

    #[must_use]
    pub const fn per_row(rows: u32, block: u32) -> Self {
        Self {
            grid: [rows, 1, 1],
            block: [block, 1, 1],
            smem: 0,
        }
    }

    #[must_use]
    pub const fn grid(grid: [u32; 3], block: [u32; 3]) -> Self {
        Self {
            grid,
            block,
            smem: 0,
        }
    }

    #[must_use]
    pub const fn smem(mut self, bytes: u32) -> Self {
        self.smem = bytes;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fire {
    pub kernel: &'static Kernel,

    pub launch: Launch,
}

impl Fire {
    #[must_use]
    pub const fn at(kernel: &'static Kernel) -> Self {
        Self {
            kernel,
            launch: Launch::grid([0, 0, 0], [0, 0, 0]),
        }
    }

    #[must_use]
    pub const fn apply(mut self, launch: Launch) -> Self {
        self.launch = launch;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slabs(u32);

/// What names a slab: the arena it is one of ([`Slabs::id`]), its name, the
/// region that takes it (or `NO_REGION`), and the stream it is scoped to (0
/// where it is a region's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlabKey {
    pub arena: u32,
    pub name: &'static str,
    pub region: u32,
    pub scope: usize,
}

/// What a placed slab is aligned to.
pub const PLACED_ALIGN: usize = 256;

impl Slabs {
    pub const PROCESS: Slabs = Slabs(0);

    /// The arena a [`SlabKey`] names these slabs by.
    #[must_use]
    pub const fn id(self) -> u32 {
        self.0
    }

    /// The bytes `slabs` take laid one after another, each of `(key,
    /// bytes)` at [`PLACED_ALIGN`].
    #[must_use]
    pub fn placed_bytes(slabs: &[(SlabKey, usize)]) -> usize {
        slabs.iter().fold(0usize, |at, (_, bytes)| {
            at.saturating_add(bytes.next_multiple_of(PLACED_ALIGN))
        })
    }

    /// Lays `slabs` one after another from `base`: a take of one answers
    /// its place there, and nothing is allocated for it. A slab laid before
    /// is laid anew.
    ///
    /// # Safety
    ///
    /// `base` must be device memory of [`Slabs::placed_bytes`] bytes, zeroed,
    /// that the caller keeps backed for as long as the slabs' arenas live.
    pub unsafe fn place(base: *mut c_void, slabs: &[(SlabKey, usize)]) {
        #[cfg(feature = "rocm")]
        {
            crate::jit::device::place(base, slabs);
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (base, slabs);
        }
    }

    /// The bytes takes asked past what was placed for them (a slab not
    /// placed, or placed smaller) since the process began: each one a
    /// launch refused.
    #[must_use]
    pub fn unplaced_bytes() -> usize {
        #[cfg(feature = "rocm")]
        {
            crate::jit::device::unplaced_bytes()
        }
        #[cfg(not(feature = "rocm"))]
        {
            0
        }
    }

    #[must_use]
    pub fn open() -> Slabs {
        static NEXT: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(1);
        Slabs(NEXT.fetch_add(1, core::sync::atomic::Ordering::Relaxed))
    }

    /// # Safety
    ///
    /// `stream` must be a live HIP stream that outlives every launch
    /// this context is used for.
    pub unsafe fn attach(self, stream: *mut c_void) {
        #[cfg(feature = "rocm")]
        {
            crate::jit::device::attach(self.0, stream);
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = stream;
        }
    }

    /// Forgets the slabs laid for these arenas; their memory is the placing
    /// caller's to free.
    ///
    /// # Safety
    ///
    /// Nothing reads the slabs again through these arenas: no `Ctx` carrying
    /// them fires after this, and no graph captured on their stream is
    /// replayed. A `Slabs` is `Copy`, so that is the caller's to keep across
    /// every copy.
    pub unsafe fn release(self) {
        #[cfg(feature = "rocm")]
        {
            crate::jit::device::release(self.0);
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pad {
    pub rows: u32,
    pub bucket: u32,
}

impl Pad {
    /// The rows a kernel launches over for an operand of `rows` rows: the
    /// fire's rows are padded to the bucket, whatever axis they were read on.
    #[must_use]
    pub fn extent(self, rows: u32) -> u32 {
        if self.bucket > self.rows && rows == self.rows {
            self.bucket
        } else {
            rows
        }
    }
}

pub const NO_REGION: u32 = u32::MAX;

/// The slab rocBLAS's calls run on ([`Ctx::blas_workspace`]).
const BLAS_WORKSPACE: &str = "linear.rocblas_workspace";

/// The tickets a stream holds ([`Ctx::tickets`]): every launch that draws
/// them asks for this many, so one laid slab serves every one.
pub const TICKETS: u32 = 1 << 16;

#[cfg(feature = "rocm")]
pub const SHARED_REGION: u32 = u32::MAX - 1;

pub struct Ctx {
    stream: *mut c_void,
    owner: *mut c_void,
    blas: *mut c_void,
    comm: *mut c_void,
    /// This rank's place in its tensor-parallel group, for the launches
    /// that only need to know it (a vocab-sharded embed): a side stream's
    /// context carries it but no communicator, so collectives stay on the
    /// main stream.
    rank: Option<u32>,
    peers: Option<crate::collective::Peers>,
    slabs: Slabs,

    pad: core::cell::Cell<Pad>,

    stage: core::cell::Cell<u64>,
    region: core::cell::Cell<u32>,
    lane: core::cell::Cell<u32>,

    tuning: crate::tuning::Tuning,
}

impl Ctx {
    /// # Safety
    ///
    /// `stream` must be a live HIP stream that outlives this context.
    #[must_use]
    pub const unsafe fn on(stream: *mut c_void) -> Self {
        Self {
            stream,
            owner: stream,
            blas: core::ptr::null_mut(),
            comm: core::ptr::null_mut(),
            rank: None,
            peers: None,
            slabs: Slabs::PROCESS,
            pad: core::cell::Cell::new(Pad { rows: 0, bucket: 0 }),
            stage: core::cell::Cell::new(0),
            region: core::cell::Cell::new(NO_REGION),
            lane: core::cell::Cell::new(0),
            tuning: crate::tuning::Tuning::DEFAULT,
        }
    }

    /// Launches that run in `owner`'s order — a conditional body, captured
    /// on its own stream and replayed inside `owner`'s graph — share
    /// `owner`'s per-stream slabs, so what a load laid there serves them.
    #[must_use]
    pub const fn in_order_of(mut self, owner: *mut c_void) -> Self {
        self.owner = owner;
        self
    }

    #[must_use]
    pub const fn with_tuning(mut self, tuning: crate::tuning::Tuning) -> Self {
        self.tuning = tuning;
        self
    }

    pub fn set_tuning(&mut self, tuning: crate::tuning::Tuning) {
        self.tuning = tuning;
    }

    #[must_use]
    pub const fn tuning(&self) -> crate::tuning::Tuning {
        self.tuning
    }

    #[must_use]
    pub const fn with_slabs(mut self, slabs: Slabs) -> Self {
        self.slabs = slabs;
        self
    }

    /// # Safety
    ///
    /// `handle` must be a live rocBLAS handle that outlives this context.
    #[must_use]
    pub const unsafe fn with_blas(mut self, handle: *mut c_void) -> Self {
        self.blas = handle;
        self
    }

    /// # Safety
    ///
    /// `comm` must be a live communicator that outlives this context.
    #[must_use]
    pub const unsafe fn with_comm(mut self, comm: *mut c_void) -> Self {
        self.comm = comm;
        self
    }

    /// # Safety
    ///
    /// Every pointer in `peers` must stay live, and mapped on this context's
    /// device, for as long as this context fires collectives.
    #[must_use]
    pub const unsafe fn with_peers(mut self, peers: crate::collective::Peers) -> Self {
        self.peers = Some(peers);
        self
    }

    #[must_use]
    pub const fn with_rank(mut self, rank: Option<u32>) -> Self {
        self.rank = rank;
        self
    }

    #[must_use]
    pub const fn rank(&self) -> Option<u32> {
        self.rank
    }

    #[must_use]
    pub const fn peers(&self) -> Option<crate::collective::Peers> {
        self.peers
    }

    #[must_use]
    pub const fn stream(&self) -> *mut c_void {
        self.stream
    }

    pub fn blas(&self, op: &'static str) -> Result<*mut c_void, Error> {
        if self.blas.is_null() {
            return Err(refuse(op, "this context carries no rocBLAS handle"));
        }
        Ok(self.blas)
    }

    /// Where this context's rocBLAS calls take their workspace: one slab a
    /// stream, at the most any call on it asks (`hip::blas::Workspace`), so
    /// rocBLAS allocates none of its own.
    pub fn blas_workspace(
        &self,
        op: &'static str,
    ) -> impl FnMut(usize) -> Result<u64, Error> + '_ {
        move |bytes| {
            self.scratch_stream(op, BLAS_WORKSPACE, bytes)
                .map(|at| at as usize as u64)
        }
    }

    pub fn comm(&self, op: &'static str) -> Result<*mut c_void, Error> {
        if self.comm.is_null() {
            return Err(refuse(op, "this context carries no communicator"));
        }
        Ok(self.comm)
    }

    pub fn scratch(
        &self,
        op: &'static str,
        name: &'static str,
        bytes: usize,
    ) -> Result<*mut c_void, Error> {
        #[cfg(feature = "rocm")]
        {
            crate::jit::device::take(
                self.slabs.0,
                self.stream,
                name,
                self.region.get(),
                bytes,
            )
            .map_err(|fault| fault.at(op))
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (name, bytes);
            Err(crate::jit::runtimeless(op))
        }
    }

    /// A slab one per stream, whatever region asks: for a tile a launch
    /// fills and the next launch on the same stream consumes, so a model
    /// with hundreds of such regions holds one tile a stream, not one a
    /// region.
    pub fn scratch_stream(
        &self,
        op: &'static str,
        name: &'static str,
        bytes: usize,
    ) -> Result<*mut c_void, Error> {
        #[cfg(feature = "rocm")]
        {
            crate::jit::device::take(
                self.slabs.0,
                self.owner,
                name,
                NO_REGION,
                bytes,
            )
            .map_err(|fault| fault.at(op))
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (name, bytes);
            Err(crate::jit::runtimeless(op))
        }
    }

    /// `count` i32 tickets, zero as laid, one slab a stream:
    /// a launch that draws them puts every one back to zero before it ends
    /// (its last block), so the next launch on the stream finds them zero.
    pub fn tickets(&self, op: &'static str, count: u32) -> Result<u64, Error> {
        #[cfg(feature = "rocm")]
        {
            crate::jit::device::take(
                self.slabs.0,
                self.owner,
                "pie.tickets",
                NO_REGION,
                count as usize * 4,
            )
            .map(|p| p as u64)
            .map_err(|fault| fault.at(op))
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = count;
            Err(crate::jit::runtimeless(op))
        }
    }

    pub fn scratch_shared(
        &self,
        op: &'static str,
        name: &'static str,
        bytes: usize,
    ) -> Result<*mut c_void, Error> {
        #[cfg(feature = "rocm")]
        {
            let region = SHARED_REGION.saturating_sub(self.lane.get().min(1 << 8));
            crate::jit::device::take(self.slabs.0, self.stream, name, region, bytes)
                .map_err(|fault| fault.at(op))
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = (name, bytes);
            Err(crate::jit::runtimeless(op))
        }
    }

    #[must_use]
    pub const fn slabs(&self) -> Slabs {
        self.slabs
    }

    pub fn arm(&self, pad: Pad) {
        self.pad.set(pad);
    }

    pub fn disarm(&self) {
        self.pad.set(Pad::default());
    }

    pub fn arm_stage(&self, addr: u64) {
        self.stage.set(addr);
    }

    pub fn disarm_stage(&self) {
        self.stage.set(0);
    }

    pub fn arm_lane(&self, lane: u32) {
        self.lane.set(lane);
    }
    pub fn arm_region(&self, region: u32) {
        self.region.set(region);
    }

    pub fn disarm_region(&self) {
        self.region.set(NO_REGION);
    }

    #[must_use]
    pub fn stage(&self) -> ArgValue<'static> {
        ArgValue::Ptr(self.stage.get())
    }

    #[must_use]
    pub fn pad(&self) -> Pad {
        self.pad.get()
    }

    #[must_use]
    pub fn opaque_rows(&self, rows: i32) -> i32 {
        let Ok(stated) = u32::try_from(rows) else {
            return rows;
        };
        i32::try_from(self.pad.get().extent(stated))
            .unwrap_or(rows)
            .max(rows)
    }

    #[allow(clippy::unused_self)]
    #[must_use]
    pub fn multiprocessors(&self) -> Option<u32> {
        #[cfg(feature = "rocm")]
        {
            crate::jit::device::multiprocessors()
        }
        #[cfg(not(feature = "rocm"))]
        {
            None
        }
    }

    pub fn fire(&self, op: &'static str, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
        if fire.kernel.root.is_none() {
            return Err(refuse(
                op,
                format!("no carried unit is named `{}`", fire.kernel.file),
            ));
        }
        if fire.launch.empty() {
            return Err(refuse(op, "the grid is empty"));
        }
        self.issue(op, fire.kernel, fire.launch, args)
    }

    #[cfg(feature = "rocm")]
    fn issue(
        &self,
        op: &'static str,
        kernel: &'static Kernel,
        launch: Launch,
        args: &[ArgValue],
    ) -> Result<(), Error> {
        let resolved = match kernel.resolve() {
            Ok(resolved) => resolved,
            Err(why) => return Err(said(kernel, why).at(op)),
        };

        let mut bound = crate::jit::abi::Bound::new(args);

        // SAFETY: `resolved.function` is a kernel of a module the cache never
        // unloads; `bound` owns every cell its slots point at and outlives the
        // call; `self.stream` is live per `Ctx::on`. That `args` has one value
        // per kernel parameter, in order, is each entry's to keep: the
        // instantiation's signature is not checked here.
        let fired = unsafe {
            crate::jit::launch::issue(resolved.function, launch, bound.slots_mut(), self.stream)
        };
        match fired {
            Ok(()) => Ok(()),
            Err(why) => Err(said(kernel, why).at(op)),
        }
    }

    #[cfg(not(feature = "rocm"))]
    #[allow(clippy::unused_self, clippy::needless_pass_by_value)]
    fn issue(
        &self,
        op: &'static str,
        _kernel: &'static Kernel,
        _launch: Launch,
        _args: &[ArgValue],
    ) -> Result<(), Error> {
        Err(crate::jit::runtimeless(op))
    }
}

/// Whether a device instantiation that will not fire (it does not compile,
/// or its module does not load) panics rather than falling back: on under
/// `PIE_STRICT_KERNELS=1` or [`strict_kernels`], as the test harnesses run.
/// Off, the fallback serves (another chip's toolchain may lack what one
/// unit asks) and the fault is said once.
#[must_use]
pub fn kernels_strict() -> bool {
    STRICT.load(std::sync::atomic::Ordering::Relaxed)
        || std::env::var_os("PIE_STRICT_KERNELS").is_some_and(|v| v == "1")
}

/// Turns [`kernels_strict`] on for this process.
pub fn strict_kernels() {
    STRICT.store(true, std::sync::atomic::Ordering::Relaxed);
}

static STRICT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(feature = "rocm")]
fn said(kernel: &Kernel, why: crate::jit::Fault) -> crate::jit::Fault {
    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};

    let (root, instantiation) = (kernel.file, kernel.name);
    assert!(
        !kernels_strict(),
        "strict kernels: `{root}` {instantiation} will not fire: {why}"
    );
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

    fn bare() -> Ctx {
        // SAFETY: no method called here fires, so the null stream is never
        // handed to the runtime.
        unsafe { Ctx::on(core::ptr::null_mut()) }
    }

    #[test]
    fn ctx_every_case() {
        an_unarmed_context_quantizes_nothing();
        the_full_fires_extent_rounds_up_to_the_bucket();
        disarming_puts_the_extent_back_the_way_the_fire_found_it();
    }

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
}
