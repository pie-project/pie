use core::ffi::c_void;

use crate::error::Error;

use crate::jit::{ArgValue, refuse};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub smem: u32,
    pub cooperative: bool,
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
        }
    }

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fire {
    pub file: &'static str,

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slabs(u32);

impl Slabs {
    pub const PROCESS: Slabs = Slabs(0);

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

pub const NO_REGION: u32 = u32::MAX;

pub struct Ctx {
    stream: *mut c_void,
    cublas: *mut c_void,
    comm: *mut c_void,
    slabs: Slabs,

    pad: core::cell::Cell<Pad>,

    stage: core::cell::Cell<u64>,
    region: core::cell::Cell<u32>,
}

impl Ctx {
    /// # Safety
    ///
    /// `stream` must be a live CUDA stream that outlives this context.
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
            crate::jit::device::take(self.slabs.0, self.stream, name, self.region.get(), bytes)
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

    pub fn arm_region(&self, region: u32) {
        self.region.set(region);
    }

    pub fn disarm_region(&self) {
        self.region.set(NO_REGION);
    }

    #[must_use]
    pub fn stage(&self) -> ArgValue {
        ArgValue::Ptr(self.stage.get())
    }

    #[must_use]
    pub fn pad(&self) -> Pad {
        self.pad.get()
    }

    #[must_use]
    pub fn opaque_rows(&self, rows: i32) -> i32 {
        let pad = self.pad.get();
        if pad.bucket <= pad.rows {
            return rows;
        }
        if rows < 0 || rows.unsigned_abs() != pad.rows {
            return rows;
        }
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

}

