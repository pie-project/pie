#![allow(unsafe_code)]
use core::ffi::c_void;

use crate::error::Error;

use crate::jit::{ArgValue, Kernel, refuse};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub smem: u32,
    pub cooperative: bool,
    /// CTAs a cluster along x; 1 launches none.
    pub cluster: u32,
    /// Programmatic dependent launch: this kernel may start while the one
    /// before it on the stream finishes, and waits on it in-kernel
    /// (`griddepcontrol.wait`) before it reads anything that one wrote. Set
    /// only on a kernel that waits so (sm_90 and later).
    pub pdl: bool,
}

impl Launch {
    #[must_use]
    pub const fn flat(n: u32, block: u32) -> Self {
        let grid = if block == 0 { 0 } else { n.div_ceil(block) };
        Self {
            grid: [grid, 1, 1],
            block: [block, 1, 1],
            smem: 0,
            cooperative: false,
            cluster: 1,
            pdl: false,
        }
    }

    #[must_use]
    pub const fn per_row(rows: u32, block: u32) -> Self {
        Self {
            grid: [rows, 1, 1],
            block: [block, 1, 1],
            smem: 0,
            cooperative: false,
            cluster: 1,
            pdl: false,
        }
    }

    #[must_use]
    pub const fn grid(grid: [u32; 3], block: [u32; 3]) -> Self {
        Self {
            grid,
            block,
            smem: 0,
            cooperative: false,
            cluster: 1,
            pdl: false,
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

    /// Launches `ctas` consecutive CTAs along x as one cluster; the grid's
    /// x must be a multiple of it.
    #[must_use]
    pub const fn cluster(mut self, ctas: u32) -> Self {
        self.cluster = ctas;
        self
    }

    /// Launches as a programmatic dependent of the stream's previous kernel
    /// where `on`; see [`Launch::pdl`].
    #[must_use]
    pub const fn programmatic(mut self, on: bool) -> Self {
        self.pdl = on;
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

impl Slabs {
    pub const PROCESS: Slabs = Slabs(0);

    #[must_use]
    pub fn census() -> Vec<(&'static str, usize)> {
        #[cfg(feature = "cuda")]
        {
            crate::jit::device::census()
        }
        #[cfg(not(feature = "cuda"))]
        {
            Vec::new()
        }
    }

    #[must_use]
    pub fn open() -> Slabs {
        static NEXT: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(1);
        Slabs(NEXT.fetch_add(1, core::sync::atomic::Ordering::Relaxed))
    }

    /// # Safety
    ///
    /// `stream` must be a live CUDA stream that outlives every launch
    /// this context is used for.
    pub unsafe fn attach(self, stream: *mut c_void) {
        #[cfg(feature = "cuda")]
        {
            crate::jit::device::attach(self.0, stream);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = stream;
        }
    }

    pub fn release(self) {
        #[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
pub const SHARED_REGION: u32 = u32::MAX - 1;

pub struct Ctx {
    stream: *mut c_void,
    owner: *mut c_void,
    cublas: *mut c_void,
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
    /// `stream` must be a live CUDA stream that outlives this context.
    #[must_use]
    pub const unsafe fn on(stream: *mut c_void) -> Self {
        Self {
            stream,
            owner: stream,
            cublas: core::ptr::null_mut(),
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
    /// `owner`'s per-stream slabs, so what a load warmed there serves them.
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
    /// `handle` must be a live cuBLAS handle bound to this context's stream.
    #[must_use]
    pub const unsafe fn with_cublas(mut self, handle: *mut c_void) -> Self {
        self.cublas = handle;
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

    /// The context's cuBLAS handle, set to sum a split K's partials in the
    /// compute type: cuBLAS's default adds them in a bf16 or f16 output's own
    /// type, and over a cancelling K that is off by percents (sm89 splits a
    /// hundred-row bf16 product so).
    pub fn cublas(&self, op: &'static str) -> Result<*mut c_void, Error> {
        if self.cublas.is_null() {
            return Err(refuse(op, "this context carries no cuBLAS handle"));
        }
        #[cfg(feature = "cuda")]
        {
            use cudarc::cublas::sys::{cublasMath_t, cublasSetMathMode};
            // SAFETY: a live handle this context carries.
            let _ = unsafe {
                cublasSetMathMode(
                    self.cublas.cast(),
                    cublasMath_t::CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION,
                )
            };
        }
        Ok(self.cublas)
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
        #[cfg(feature = "cuda")]
        {
            crate::jit::device::take(
                self.slabs.0,
                self.stream,
                self.stream,
                name,
                self.region.get(),
                bytes,
            )
            .map_err(|fault| fault.at(op))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (name, bytes);
            Err(crate::jit::runtimeless(op))
        }
    }

    /// A slab one per stream, whatever region asks: for a tile a launch
    /// fills and the next launch on the same stream consumes, so a model
    /// with hundreds of such regions holds one tile a stream, not one a
    /// region. A capture sees the slab as it was warmed, so arming fires
    /// the widest plane eagerly first.
    pub fn scratch_stream(
        &self,
        op: &'static str,
        name: &'static str,
        bytes: usize,
    ) -> Result<*mut c_void, Error> {
        #[cfg(feature = "cuda")]
        {
            crate::jit::device::take(
                self.slabs.0,
                self.stream,
                self.owner,
                name,
                NO_REGION,
                bytes,
            )
            .map_err(|fault| fault.at(op))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (name, bytes);
            Err(crate::jit::runtimeless(op))
        }
    }

    /// [`Ctx::scratch_stream`], zeroed when fresh: for counts a kernel
    /// leaves at zero between launches.
    pub fn scratch_stream_zeroed(
        &self,
        op: &'static str,
        name: &'static str,
        bytes: usize,
    ) -> Result<*mut c_void, Error> {
        #[cfg(feature = "cuda")]
        {
            crate::jit::device::take_zeroed(
                self.slabs.0,
                self.stream,
                self.owner,
                name,
                NO_REGION,
                bytes,
            )
            .map_err(|fault| fault.at(op))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (name, bytes);
            Err(crate::jit::runtimeless(op))
        }
    }

    pub fn scratch_shared(
        &self,
        op: &'static str,
        name: &'static str,
        bytes: usize,
    ) -> Result<*mut c_void, Error> {
        #[cfg(feature = "cuda")]
        {
            let region = SHARED_REGION.saturating_sub(self.lane.get().min(1 << 8));
            crate::jit::device::take(self.slabs.0, self.stream, self.stream, name, region, bytes)
                .map_err(|fault| fault.at(op))
        }
        #[cfg(not(feature = "cuda"))]
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
        #[cfg(feature = "cuda")]
        {
            crate::jit::device::multiprocessors()
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// The dynamic shared memory one block may opt into on this device.
    #[allow(clippy::unused_self)]
    #[must_use]
    pub fn smem_per_block_optin(&self) -> Option<u32> {
        #[cfg(feature = "cuda")]
        {
            crate::jit::device::max_shared_memory_per_block_optin()
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// Whether a block asking `bytes` of dynamic shared memory can launch
    /// here; true where the device does not say.
    #[must_use]
    pub fn smem_fits(&self, bytes: u32) -> bool {
        self.smem_per_block_optin().is_none_or(|max| bytes <= max)
    }

    /// The shared memory one SM holds across its resident blocks.
    #[allow(clippy::unused_self)]
    #[must_use]
    pub fn smem_per_sm(&self) -> Option<u32> {
        #[cfg(feature = "cuda")]
        {
            crate::jit::device::max_shared_memory_per_sm()
        }
        #[cfg(not(feature = "cuda"))]
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
        if trace_fires() {
            let now = std::time::Instant::now();
            let gap = {
                static LAST: std::sync::Mutex<Option<std::time::Instant>> =
                    std::sync::Mutex::new(None);
                let mut last = LAST.lock().expect("last lock poisoned");
                let gap = last.map_or(0, |at| now.duration_since(at).as_micros());
                *last = Some(now);
                gap
            };
            eprintln!("fire: {gap} {op} {}", fire.kernel.name);
        }
        self.issue(op, fire.kernel, fire.launch, args)
    }

    #[cfg(feature = "cuda")]
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

        if let Some(why) = mismatch(&resolved.params, args) {
            return Err(refuse(op, format!("`{}`: {why}", kernel.name)));
        }
        let mut bound = crate::jit::abi::Bound::new(args);

        // SAFETY: `resolved.function` lives for the process (the cache
        // leaks it) and was loaded in the context `resolve` bound to this
        // thread; `self.stream` is live by `Ctx::on`'s contract. `bound`
        // owns one slot per argument and outlives the call, and `mismatch`
        // held the arguments to the kernel's parameters, so the driver reads
        // no slot past its value. That every pointer among them is a live
        // device address is not checked here: `fire` takes it on trust from
        // its caller.
        let fired = unsafe {
            crate::jit::launch::issue(resolved.function, launch, bound.slots_mut(), self.stream)
        };
        match fired {
            Ok(()) => Ok(()),
            Err(why) => Err(said(kernel, why).at(op)),
        }
    }

    #[cfg(not(feature = "cuda"))]
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

/// Why `args` cannot fill a kernel whose parameters are `params` bytes each,
/// or `None` when each argument is exactly its parameter's width.
#[cfg(any(feature = "cuda", test))]
fn mismatch(params: &[usize], args: &[ArgValue]) -> Option<String> {
    if params.len() != args.len() {
        return Some(format!(
            "the kernel takes {} parameters and the fire passes {}",
            params.len(),
            args.len()
        ));
    }
    for (at, (&size, arg)) in params.iter().zip(args).enumerate() {
        if arg.width() != size {
            return Some(format!(
                "parameter {at} is {size} bytes and the fire passes {} of {}",
                arg.kind(),
                arg.width()
            ));
        }
    }
    None
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

#[cfg(feature = "cuda")]
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

fn trace_fires() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_CUDA_TRACE_FIRES").is_some_and(|v| v == "1"))
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

    #[test]
    fn a_fire_is_held_to_its_kernels_parameters() {
        let args = [ArgValue::F32(2.0), ArgValue::Ptr(0), ArgValue::I32(3)];
        assert_eq!(mismatch(&[4, 8, 4], &args), None, "every width matches");
        assert!(
            mismatch(&[4, 8, 4, 8], &args).is_some_and(|why| why.contains("takes 4")),
            "a missing argument would have the driver read past the slots"
        );
        assert!(
            mismatch(&[4, 8, 8], &args).is_some_and(|why| why.contains("parameter 2")),
            "an i32 into an 8-byte parameter is refused"
        );
    }
}
