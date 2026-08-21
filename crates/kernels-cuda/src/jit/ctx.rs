use core::ffi::c_void;
use kernels::routine::Fire;

use kernels::routine::{Backend, Extent, Refusal};

use crate::comm::Plane;
use crate::jit::{ArgValue, Root};

#[derive(Clone, Copy, Debug)]
pub struct Cuda;

impl Backend for Cuda {
    type Value = ArgValue;
    type Ctx<'a> = Ctx<'a>;

    fn region(value: &ArgValue) -> Result<Extent, Refusal> {
        match *value {
            ArgValue::Region { rows, width, .. } => Ok(Extent { rows, width }),
            _ => Err(Refusal::Absent {
                what: "a region's shape: the bound value carries only an address",
            }),
        }
    }
}

impl kernels::routine::Answers<Cuda> for Ctx<'_> {
    fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
        self.env
            .ok_or(Refusal::Unstated {
                what: "a fact, on a context built for a hand-written call: \
                       nothing behind it holds this fire's answers",
            })?
            .resolve(ty, source)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub smem: u32,
    pub cooperative: bool,
}

impl kernels::routine::Geometry for Launch {
    fn apply_to(self, fire: Fire) -> Fire {
        self.apply_to_impl(fire)
    }
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
            stamp: "",
        }
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

pub struct Ctx<'a> {
    stream: *mut c_void,
    cublas: *mut c_void,
    comm: Option<Plane>,
    env: Option<&'a (dyn kernels::routine::Answers<Cuda> + 'a)>,
    held: core::marker::PhantomData<&'a ()>,
}

impl<'a> Ctx<'a> {
    /// A context on a stream, answering nothing else.
    ///
    /// # Safety
    ///
    /// `stream` must be a live CUDA stream in the current context, and must
    /// stay live for as long as the returned `Ctx` is used to fire.
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

    /// The same context, carrying a cuBLAS handle for the routines that want one.
    ///
    /// # Safety
    ///
    /// `handle` must be a live `cublasHandle_t` in the current context, and
    /// its stream must be this context's -- a handle set to another stream
    /// orders its work against the wrong queue.
    #[must_use]
    pub const unsafe fn with_cublas(mut self, handle: *mut c_void) -> Self {
        self.cublas = handle;
        self
    }

    /// The same context, carrying the communicator a collective fires on.
    ///
    /// # Safety
    ///
    /// `plane` must name a communicator that is live and whose every rank is
    /// making the same call in the same order; a collective that one rank
    /// skips hangs the rest.
    #[must_use]
    pub const unsafe fn with_comm(mut self, plane: Plane) -> Self {
        self.comm = Some(plane);
        self
    }

    #[must_use]
    pub const fn with_env(mut self, env: &'a (dyn kernels::routine::Answers<Cuda> + 'a)) -> Self {
        self.env = Some(env);
        self
    }

    #[must_use]
    pub const fn stream(&self) -> *mut c_void {
        self.stream
    }

    pub fn cublas(&self) -> Result<*mut c_void, Refusal> {
        if self.cublas.is_null() {
            return Err(Refusal::Absent {
                what: "a cuBLAS handle",
            });
        }
        Ok(self.cublas)
    }

    pub fn comm(&self) -> Result<Plane, Refusal> {
        let Some(plane) = self.comm else {
            return Err(Refusal::Absent {
                what: "a tensor-parallel plane",
            });
        };
        Ok(plane)
    }

    #[allow(clippy::unused_self)]
    pub fn scratch(&self, name: &'static str, bytes: usize) -> Result<*mut c_void, Refusal> {
        #[cfg(feature = "_cuda")]
        {
            crate::jit::device::take(name, bytes)
        }
        #[cfg(not(feature = "_cuda"))]
        {
            let _ = (name, bytes);
            Err(Refusal::Device {
                why: "this build selected no CUDA runtime",
            })
        }
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
    pub fn multiprocessors(&self) -> Result<u32, Refusal> {
        #[cfg(feature = "_cuda")]
        {
            crate::jit::device::multiprocessors()
        }
        #[cfg(not(feature = "_cuda"))]
        {
            Err(Refusal::Device {
                why: "this build selected no CUDA runtime",
            })
        }
    }

    pub fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
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

        unsafe { self.launch_at(&root, fire.entrypoint, launch, args) }
    }

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

        unsafe { self.issue(root, instantiation, launch, args) }
    }

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

        let mut bound = unsafe { crate::jit::value::Bound::new(args) };

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
    unsafe fn issue(
        &self,
        _root: &Root,
        _instantiation: &str,
        _launch: Launch,
        _args: &[ArgValue],
    ) -> Result<(), Refusal> {
        Err(Refusal::Device {
            why: "this build selected no CUDA runtime",
        })
    }
}

#[cfg(feature = "_cuda")]
fn said(root: &str, instantiation: &str, why: &str) -> Refusal {
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
            why,
            "a device instantiation will not fire"
        );
    }
    Refusal::Device {
        why: "the compile, the load or the launch refused; see the log",
    }
}
