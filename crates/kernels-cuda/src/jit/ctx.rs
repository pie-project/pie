//! The context every entry takes: the stream an engine `Run` wraps, with the
//! cuBLAS handle and device probes beside it. `fire` resolves a [`Fire`]'s
//! unit through the module cache and enqueues the launch — enqueue only,
//! never sync.

use core::ffi::c_void;

use crate::error::Error;

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

/// One context's scratch slabs, minted by [`Slabs::open`] and freed by
/// [`Slabs::release`]. One arena per CUDA context, one slab per
/// `(name, region)` inside it, so two shells never stage into each other's
/// planes. Handles are `Copy` and mean nothing but an identity; the storage
/// is process-global behind them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slabs(u32);

impl Slabs {
    /// The arena a bare [`Ctx::on`] fires against. Shared by everything that
    /// never asked for one of its own; still per-stream inside, so sharing
    /// it is not sharing a slab.
    pub const PROCESS: Slabs = Slabs(0);

    /// A fresh arena, disjoint from every other. One per `Context`.
    #[must_use]
    pub fn open() -> Slabs {
        static NEXT: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(1);
        Slabs(NEXT.fetch_add(1, core::sync::atomic::Ordering::Relaxed))
    }

    /// Tell this arena it fires on `stream` too.
    ///
    /// # Safety
    ///
    /// `stream` must be a live `cudaStream_t`, and must not outlive this
    /// arena's [`release`](Slabs::release).
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

    /// Free every slab in this arena. The context's teardown: a context that
    /// has synchronized has nothing left pointing at one of its slabs.
    pub fn release(self) {
        #[cfg(feature = "cuda")]
        {
            crate::jit::device::release(self.0);
        }
    }
}

/// This fire's row count and the bucket it rounds up to. Not a permission to
/// round — the pair of numbers that makes rounding checkable, read only by
/// [`Ctx::opaque_rows`]. The default `rows == bucket == 0` (nothing armed, or
/// a windowed region) matches no extent, so nothing is ever padded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pad {
    /// The fire's total token rows.
    pub rows: u32,
    /// The smallest lattice point that holds them (`Composition::bucket`).
    pub bucket: u32,
}

/// The region a caller outside the walk asks under (a warm pass, a weight
/// transform, a bench, a test). Not one shared slot: `jit::device`'s key
/// falls back to the asking stream at this value, since such callers share
/// `Slabs::PROCESS` and run at once.
pub const NO_REGION: u32 = u32::MAX;

/// The stream and its companions. Long-lived state (jit cache, scratch
/// slabs, device probes) is process-global behind it; the `Ctx` itself is
/// what an engine `Run` builds per fire and lends to every entry.
pub struct Ctx {
    stream: *mut c_void,
    cublas: *mut c_void,
    comm: *mut c_void,
    slabs: Slabs,

    /// The padding quantum, written per region by the engine's walk
    /// ([`Ctx::arm`]). A `Cell` because the context outlives the fire and
    /// this value does not. Never read across threads (`Ctx` is neither
    /// `Send` nor `Sync`).
    pad: core::cell::Cell<Pad>,

    /// The device address of this region's live-geometry words, or `0` (the
    /// staged-geometry seat). Four `u32` live at that address in the
    /// engine's order (`engine_cuda::window`): `[rows, row_offset, lanes,
    /// lane_offset]` — a row-gridded entry reads 0 and 1, a request-gridded
    /// one (chunked linear-attention scans) reads 2 and 3. `0` is the whole
    /// off switch: [`ArgValue::ABSENT`] is `Ptr(0)`, so a disarmed stage and
    /// the null seat every entry passes today are the same argument bytes.
    stage: core::cell::Cell<u64>,
    /// Which template region this context is firing, stamped by the
    /// engine's walk at the same instant as [`Pad`] and the staged seat.
    /// Read only by the scratch slab's key ([`Ctx::scratch`]).
    /// [`NO_REGION`] for a caller that is not walking a template.
    region: core::cell::Cell<u32>,
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
            stage: core::cell::Cell::new(0),
            region: core::cell::Cell::new(NO_REGION),
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

    pub fn cublas(&self, op: &'static str) -> Result<*mut c_void, Error> {
        if self.cublas.is_null() {
            return Err(refuse(op, "this context carries no cuBLAS handle"));
        }
        Ok(self.cublas)
    }

    /// The NCCL communicator a tensor-parallel run carries. Absent on a
    /// single-rank context — a collective fired there is a typed refusal,
    /// not a hang.
    pub fn comm(&self, op: &'static str) -> Result<*mut c_void, Error> {
        if self.comm.is_null() {
            return Err(refuse(op, "this context carries no communicator"));
        }
        Ok(self.comm)
    }

    /// A named scratch slab, grown but never shrunk — the workspace an entry
    /// may not allocate per fire (graph capture forbids it). Keyed by
    /// `(arena, name, region)`, so two forked arms of a walk get two slabs.
    ///
    /// Growth allocates a fresh block, which would split a capture in
    /// progress: the engine warms every scratch-consuming entry with an
    /// eager fire before capturing, and `device::take` refuses a growth
    /// mid-capture. `jit::device` retires a superseded block rather than
    /// freeing it, so an address a recorded graph holds stays valid.
    ///
    /// New slabs must be sized only by the launch's own row count and
    /// load-fixed constants: every slab is monotone in per-launch rows.
    pub fn scratch(
        &self,
        op: &'static str,
        name: &'static str,
        bytes: usize,
    ) -> Result<*mut c_void, Error> {
        #[cfg(feature = "cuda")]
        {
            crate::jit::device::take(self.slabs.0, self.stream, name, self.region.get(), bytes)
                .map_err(|fault| fault.at(op))
        }
        #[cfg(not(feature = "cuda"))]
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

    /// Stamp the padding quantum's two numbers on this context, for the
    /// launch about to run. Per region, not per fire. A region whose window
    /// is not the whole fire is stamped `Pad::default()`, since padding a
    /// window would write the next class's rows of the same column.
    pub fn arm(&self, pad: Pad) {
        self.pad.set(pad);
    }

    /// Put the pad back to "no fire is running". Every launch after it is
    /// quantized by nothing, which is what a warm pass, a weight transform or
    /// a bench firing on this stream between two fires must see.
    pub fn disarm(&self) {
        self.pad.set(Pad::default());
    }

    /// Stamp the staged-geometry seat: the device address of the four `u32`
    /// this region's live geometry will be read from at run time. Per
    /// region, like [`arm`](Ctx::arm). An entry that supports the seat
    /// passes [`stage`](Ctx::stage) as its `win` argument; one that does not
    /// is unaffected.
    pub fn arm_stage(&self, addr: u64) {
        self.stage.set(addr);
    }

    /// Put the stage back to "no body is being served" — the null seat.
    pub fn disarm_stage(&self) {
        self.stage.set(0);
    }

    /// Stamp which template region is firing, for the one reader there is:
    /// the scratch slab's key ([`scratch`](Ctx::scratch)). Per region, like
    /// [`arm`](Ctx::arm) and [`arm_stage`](Ctx::arm_stage).
    pub fn arm_region(&self, region: u32) {
        self.region.set(region);
    }

    /// Put the region back to [`NO_REGION`] — the shared slot every caller
    /// outside a walk asks under. The twin of [`disarm`](Ctx::disarm).
    pub fn disarm_region(&self) {
        self.region.set(NO_REGION);
    }

    /// The staged-geometry seat as the argument an entry passes: the
    /// live-geometry words' address, or [`ArgValue::ABSENT`] when nothing
    /// armed one (same variant, since `ABSENT` is `Ptr(0)` and `0` is the
    /// disarmed stage).
    #[must_use]
    pub fn stage(&self) -> ArgValue {
        ArgValue::Ptr(self.stage.get())
    }

    /// What this fire told the entries to round to, for a caller that wants
    /// to report it.
    #[must_use]
    pub fn pad(&self) -> Pad {
        self.pad.get()
    }

    /// The row count an opaque callee is told about, in one call. An entry
    /// that hands its shape to a library planner (e.g. cuBLASLt) asks for
    /// its `M` here instead of using the raw extent. Not a general rounding
    /// service — kernels this tree owns keep the live extent.
    ///
    /// Padding is legal only when the rows `[rows, bucket)` belong to
    /// nobody: in-bounds, harmless (row-independent), and nobody's (a
    /// windowed launch's tail is the next class's real rows, so it is never
    /// padded — enforced by only rounding an extent equal to the fire's own
    /// row count).
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
        #[cfg(feature = "cuda")]
        {
            crate::jit::device::compute_capability_major()
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
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

    /// Enqueue one launch. `Ok` means the launch is on the stream, not that
    /// it ran; every failure comes back attributed to `op`.
    pub fn fire(&self, op: &'static str, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
        let Some(root) = crate::jit::Root::of(fire.file) else {
            return Err(refuse(
                op,
                format!("no carried unit is named `{}`", fire.file),
            ));
        };
        if fire.launch.empty() {
            return Err(refuse(op, "the grid is empty"));
        }
        if trace_fires() {
            let now = std::time::Instant::now();
            let gap = {
                static LAST: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
                let mut last = LAST.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let gap = last.map_or(0, |at| now.duration_since(at).as_micros());
                *last = Some(now);
                gap
            };
            eprintln!("fire: {gap} {op} {}", fire.entrypoint);
        }
        self.issue(op, &root, fire.entrypoint, fire.launch, args)
    }

    #[cfg(feature = "cuda")]
    fn issue(
        &self,
        op: &'static str,
        root: &crate::jit::Root,
        instantiation: &'static str,
        launch: Launch,
        args: &[ArgValue],
    ) -> Result<(), Error> {
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

    #[cfg(not(feature = "cuda"))]
    #[allow(clippy::unused_self, clippy::needless_pass_by_value)]
    fn issue(
        &self,
        op: &'static str,
        _root: &crate::jit::Root,
        _instantiation: &'static str,
        _launch: Launch,
        _args: &[ArgValue],
    ) -> Result<(), Error> {
        Err(crate::jit::runtimeless(op))
    }
}

/// Report a refusal once per instantiation — the same broken row is fired
/// once per layer per token, and the caller already gets the error back.
#[cfg(feature = "cuda")]
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

}

/// `PIE_CUDA_TRACE_FIRES=1` prints every launch before it is issued — with
/// `CUDA_LAUNCH_BLOCKING=1` the last line names a kernel that never returns.
fn trace_fires() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_CUDA_TRACE_FIRES").is_some_and(|v| v == "1"))
}
