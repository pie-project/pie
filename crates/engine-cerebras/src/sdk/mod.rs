//! A pure-Rust binding to the Cerebras SDK host runtime (`cerebras::SdkRuntime`): it dlopens
//! `libsdkruntime.so` and calls its mangled C++ entry points, layouts mirrored in `abi` and
//! [`sys`]; no Python, shim or build script. The SDK's ELFs hard-code `/cb/...` and `/cbcore/...`,
//! so the unpacked image must be symlinked there. Runtimes must all be made on one thread, and a
//! C++ exception aborts the process, so the SDK's cheap argument checks are duplicated here.

#![allow(unsafe_code)]

pub(crate) mod abi;
pub mod sys;

pub use kernels_cerebras::compile::toolchain::Arch;

impl From<Target> for Arch {
    fn from(t: Target) -> Self {
        match t {
            Target::Wse2 => Arch::Wse2,
            Target::Wse3 => Arch::Wse3,
        }
    }
}

use abi::{
    ArtifactsMem, CxxString, CxxVecU32, MemcpyOptions, RuntimeMem, SimPlatform, SystemPlatformMem,
    Task,
};
use libloading::Library;
use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Why an SDK call could not be made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A shared library could not be loaded.
    Library(String, String),
    /// An entry point is missing from the loaded library.
    MissingSymbol(String, String),
    /// `bin/out_rpc.json` is missing or malformed.
    Rpc(String),
    /// The program exports no symbol of that name.
    UnknownSymbol(String),
    /// A symbol was used as the wrong kind (variable vs function).
    WrongKind(String, RpcKind),
    /// A host buffer does not hold `w * h * elems_per_pe` words.
    Size {
        symbol: String,
        expected: usize,
        actual: usize,
    },
    /// A path is not valid UTF-8.
    Path(PathBuf),
    /// Every runtime of a process must be created on the thread that created
    /// its first one: the SDK's message system asserts on that and would abort.
    MainThread,
    /// The SDK's `bin` directory cannot join `PATH` (a path holding the
    /// separator): its helpers would not be found.
    PathJoin(PathBuf, String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Library(lib, e) => write!(f, "cannot load {lib}: {e}"),
            Error::MissingSymbol(s, e) => write!(f, "missing SDK entry point {s}: {e}"),
            Error::Rpc(e) => write!(f, "cannot read rpc table: {e}"),
            Error::UnknownSymbol(s) => write!(f, "program exports no symbol {s:?}"),
            Error::WrongKind(s, k) => write!(f, "symbol {s:?} is a {k:?}, not usable here"),
            Error::Size {
                symbol,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "buffer for {symbol:?} holds {actual} words, rectangle needs {expected}"
                )
            }
            Error::Path(p) => write!(f, "path is not UTF-8: {}", p.display()),
            Error::MainThread => write!(
                f,
                "the SDK runtime must be created on the process's main thread"
            ),
            Error::PathJoin(dir, e) => write!(
                f,
                "the SDK's helpers in {} cannot join PATH: {e}",
                dir.display()
            ),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// The loaded SDK libraries and their resolved entry points.
pub struct Sdk {
    api: sys::Api,
    // Order matters on drop: the runtime library goes first.
    _runtime: Library,
    _artifacts: Library,
    _platform: Library,
}

impl Sdk {
    /// Default library directory: the SDK image's `/cbcore/lib`.
    pub fn default_lib_dir() -> PathBuf {
        kernels_cerebras::compile::toolchain::Toolchain::default().sdk_lib()
    }

    pub fn open() -> Result<Arc<Self>> {
        Self::open_at(&Self::default_lib_dir())
    }

    /// Whether the SDK libraries are on this box, without loading them.
    pub fn available() -> bool {
        Self::default_lib_dir().join("libsdkruntime.so").is_file()
    }

    pub fn open_at(lib_dir: &Path) -> Result<Arc<Self>> {
        let load = |name: &str| {
            let path = lib_dir.join(name);
            // SAFETY: loading runs the libraries' initializers, as cs_python's import does.
            unsafe { Library::new(&path) }
                .map_err(|e| Error::Library(path.display().to_string(), e.to_string()))
        };
        let platform = load("libsdk_execution_platform.so")?;
        let artifacts = load("libsdk_compile_artifacts.so")?;
        let runtime = load("libsdkruntime.so")?;
        let api = sys::Api::resolve(&artifacts, &platform, &runtime)?;
        prepend_path(&lib_dir.join("../bin"))?;
        Ok(Arc::new(Sdk {
            api,
            _runtime: runtime,
            _artifacts: artifacts,
            _platform: platform,
        }))
    }
}

/// The thread that created this process's first runtime: the SDK asserts later ones are made on
/// it.
static OWNER: std::sync::OnceLock<std::thread::ThreadId> = std::sync::OnceLock::new();

fn owning_thread() -> bool {
    *OWNER.get_or_init(|| std::thread::current().id()) == std::thread::current().id()
}

/// `dir` first on `PATH` (the SDK's helpers run from there); a `PATH` it cannot join is refused.
fn prepend_path(dir: &Path) -> Result<()> {
    let current = std::env::var_os("PATH").unwrap_or_default();
    let joined = prepended(dir, &current)?;
    // SAFETY: called once at open time, before the SDK spawns any helper.
    unsafe { std::env::set_var("PATH", joined) };
    Ok(())
}

/// `dir` ahead of the search path `current`, refused by name where it cannot
/// join one.
fn prepended(dir: &Path, current: &std::ffi::OsStr) -> Result<std::ffi::OsString> {
    let mut parts = vec![dir.to_path_buf()];
    parts.extend(std::env::split_paths(current));
    std::env::join_paths(parts).map_err(|e| Error::PathJoin(dir.to_path_buf(), e.to_string()))
}

/// The simulator's threads by default: the host's available parallelism (refused if unknown).
pub fn host_threads() -> std::result::Result<u32, String> {
    let n = std::thread::available_parallelism().map_err(|e| {
        format!("the host's available parallelism: {e} (state [engine] num_threads)")
    })?;
    Ok(u32::try_from(n.get()).unwrap_or(u32::MAX))
}

/// Which wafer generation the program was compiled for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Wse2,
    Wse3,
}

/// Fabric simulator settings (`SimfabConfig`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Simulator {
    pub target: Target,
    pub num_threads: u32,
    pub suppress_trace: bool,
    pub dump_core: bool,
}

impl Simulator {
    pub fn new(target: Target, num_threads: u32) -> Self {
        Simulator {
            target,
            num_threads,
            suppress_trace: true,
            dump_core: false,
        }
    }
}

/// Where a program runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Platform {
    Simulator(Simulator),
    /// A CS system reached at `IP:port` (the SDK's `cmaddr`); a failed connect aborts the process.
    System {
        cmaddr: String,
    },
}

/// Element order of a rectangle copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    RowMajor = 0,
    ColMajor = 1,
}

/// Width of each element on the wire; 16-bit copies pack two elements per `u32` host word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Bits32 = 0,
    Bits16 = 1,
}

/// A rectangle of PEs, in the program's logical coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub fn single() -> Self {
        Rect {
            x: 0,
            y: 0,
            w: 1,
            h: 1,
        }
    }
    fn pes(&self) -> usize {
        (self.w.max(0) as usize) * (self.h.max(0) as usize)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RpcKind {
    Var,
    Func,
}

/// One entry of the compiler's `bin/out_rpc.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpcSymbol {
    pub id: u16,
    pub kind: RpcKind,
    pub ty: String,
    /// A function's parameter count (`inputs`), 0 for variables.
    pub inputs: usize,
}

/// The exported symbols of a compiled program, keyed by exported name.
#[derive(Debug, Clone, Default)]
pub struct Rpc {
    symbols: HashMap<String, RpcSymbol>,
}

impl Rpc {
    pub fn read(artifacts_dir: &Path) -> Result<Self> {
        let path = artifacts_dir.join("bin/out_rpc.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| Error::Rpc(format!("{}: {e}", path.display())))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| Error::Rpc(e.to_string()))?;
        let list = v["rpc_symbols"]
            .as_array()
            .ok_or_else(|| Error::Rpc("no rpc_symbols array".into()))?;
        let mut symbols = HashMap::new();
        for s in list {
            let name = s["name"]
                .as_str()
                .ok_or_else(|| Error::Rpc("symbol without name".into()))?;
            let id = s["id"]
                .as_u64()
                .ok_or_else(|| Error::Rpc(format!("symbol {name} without id")))?;
            let kind = match s["kind"].as_str() {
                Some("Var") => RpcKind::Var,
                Some("Func") => RpcKind::Func,
                other => return Err(Error::Rpc(format!("symbol {name} has kind {other:?}"))),
            };
            let ty = s["type"].as_str().unwrap_or_default().to_string();
            let inputs = s["inputs"].as_array().map_or(0, Vec::len);
            symbols.insert(
                name.to_string(),
                RpcSymbol {
                    id: id as u16,
                    kind,
                    ty,
                    inputs,
                },
            );
        }
        Ok(Rpc { symbols })
    }

    pub fn get(&self, name: &str) -> Result<&RpcSymbol> {
        self.symbols
            .get(name)
            .ok_or_else(|| Error::UnknownSymbol(name.to_string()))
    }

    fn id(&self, name: &str, kind: RpcKind) -> Result<u16> {
        let s = self.get(name)?;
        if s.kind != kind {
            return Err(Error::WrongKind(name.to_string(), s.kind));
        }
        Ok(s.id)
    }

    pub fn symbols(&self) -> &HashMap<String, RpcSymbol> {
        &self.symbols
    }
}

enum PlatformMem {
    Simulator(Box<SimPlatform>),
    System(Box<SystemPlatformMem>),
}

impl PlatformMem {
    fn as_ptr(&self) -> *const c_void {
        match self {
            PlatformMem::Simulator(p) => p.as_ptr(),
            PlatformMem::System(p) => p.as_ptr(),
        }
    }
}

/// A loaded program on a platform: one `cerebras::SdkRuntime`. Calls block; `stop` runs on drop.
pub struct Runtime {
    sdk: Arc<Sdk>,
    rpc: Rpc,
    this: Box<RuntimeMem>,
    _artifacts: Box<ArtifactsMem>,
    _platform: PlatformMem,
    running: bool,
    /// Blocking copies and launches so far: calls, time, words.
    pub io: IoStats,
    /// The simulator's host-to-device writes issued since its last landed one.
    pub unlanded: usize,
    /// A PE of the last such write (where a read lands them).
    pub landing: Option<Rect>,
    /// The words last written to each kept array since the load, by symbol and rectangle origin: a
    /// later write sends the PEs' changed words alone.
    pub kept: std::collections::HashMap<(String, i32, i32), Vec<u32>>,
    /// `rowp` as the device holds it where its program advances it (`state.advances`); `None`
    /// after a load or a failed launch.
    pub rows_held: Option<Vec<u32>>,
}

/// A runtime's blocking calls by kind: how many, how long, how many words.
#[derive(Debug, Clone, Copy, Default)]
pub struct IoStats {
    pub h2d: (u64, u64, u64),
    pub d2h: (u64, u64, u64),
    pub launch: (u64, u64),
    /// The PEs' stamps read back (apart from `d2h`).
    pub stamps: (u64, u64, u64),
    /// Launches whose end was polled ([`crate::reading::Reading::poll`]): how many, and the time
    /// from each call to a one-word read of its first PE coming back.
    pub done: (u64, u64),
}

impl IoStats {
    fn add(slot: &mut (u64, u64, u64), t0: std::time::Instant, words: usize) {
        slot.0 += 1;
        slot.1 += t0.elapsed().as_nanos() as u64;
        slot.2 += words as u64;
    }
}

impl std::fmt::Display for IoStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ms = |ns: u64| ns as f64 / 1e6;
        write!(
            f,
            "h2d {} calls {:.1} ms {} words; d2h {} calls {:.1} ms {} words; launch {} calls {:.1} ms",
            self.h2d.0,
            ms(self.h2d.1),
            self.h2d.2,
            self.d2h.0,
            ms(self.d2h.1),
            self.d2h.2,
            self.launch.0,
            ms(self.launch.1)
        )
    }
}

impl Runtime {
    /// Opens the compiler output at `artifacts_dir` on `platform`; `msg_level` is the SDK's log
    /// threshold (cs_python uses `"WARNING"`).
    pub fn new(
        sdk: Arc<Sdk>,
        artifacts_dir: &Path,
        platform: Platform,
        msg_level: &str,
    ) -> Result<Self> {
        if !owning_thread() {
            return Err(Error::MainThread);
        }
        let rpc = Rpc::read(artifacts_dir)?;
        let dir = artifacts_dir
            .to_str()
            .ok_or_else(|| Error::Path(artifacts_dir.to_path_buf()))?;
        let api = sdk.api;

        let artifacts = ArtifactsMem::zeroed();
        let dir_s = CxxString::new(dir);
        // SAFETY: zeroed over-allocated storage and a live std::string mirror.
        unsafe { (api.artifacts_ctor)(artifacts.as_ptr(), &*dir_s) };

        let platform = match platform {
            Platform::Simulator(s) => PlatformMem::Simulator(SimPlatform::new(
                s.num_threads,
                s.suppress_trace,
                s.dump_core,
                s.target == Target::Wse3,
            )),
            Platform::System { cmaddr } => {
                let mem = SystemPlatformMem::zeroed();
                let addr = CxxString::new(&cmaddr);
                // SAFETY: zeroed over-allocated storage and a live std::string mirror.
                unsafe { (api.platform_ctor)(mem.as_ptr(), &*addr) };
                PlatformMem::System(mem)
            }
        };

        let this = RuntimeMem::zeroed();
        let mut msg = CxxString::new(msg_level);
        let mut cslc_prefix = CxxString::new("");
        let mut out_prefix = CxxString::new("");
        // SAFETY: arguments follow the exported constructor; the strings are owned by this frame.
        unsafe {
            (api.runtime_ctor)(
                this.as_ptr(),
                artifacts.as_ptr(),
                platform.as_ptr(),
                msg.as_mut().as_mut_ptr(),
                true,
                false,
                false,
                cslc_prefix.as_mut().as_mut_ptr(),
                out_prefix.as_mut().as_mut_ptr(),
            )
        };
        Ok(Runtime {
            sdk,
            rpc,
            this,
            _artifacts: artifacts,
            _platform: platform,
            running: false,
            unlanded: 0,
            landing: None,
            kept: std::collections::HashMap::new(),
            rows_held: None,
            io: IoStats::default(),
        })
    }

    /// Whether it runs on the fabric simulator.
    #[must_use]
    pub fn simulated(&self) -> bool {
        matches!(self._platform, PlatformMem::Simulator(_))
    }

    pub fn rpc(&self) -> &Rpc {
        &self.rpc
    }

    fn api(&self) -> sys::Api {
        self.sdk.api
    }

    fn this(&self) -> *mut c_void {
        self.this.as_ptr()
    }

    /// Loads the program's ELFs onto the fabric.
    pub fn load(&mut self) {
        self.kept.clear();
        self.rows_held = None;
        // SAFETY: constructed runtime.
        unsafe { (self.api().load)(self.this()) }
    }

    /// Starts execution; RPCs and copies are accepted from here on.
    pub fn run(&mut self) {
        // SAFETY: constructed runtime.
        unsafe { (self.api().run)(self.this()) }
        self.running = true;
    }

    /// Stops the program. Idempotent.
    pub fn stop(&mut self) {
        if self.running {
            // SAFETY: constructed, running runtime.
            unsafe { (self.api().stop)(self.this()) }
            self.running = false;
        }
    }

    fn finish(&self, task: &mut Task) {
        // SAFETY: task was produced by this runtime and not yet released.
        unsafe {
            (self.api().task_wait)(self.this(), task);
            (self.api().task_dtor)(task);
        }
    }

    fn check_size(
        &self,
        symbol: &str,
        rect: Rect,
        elems_per_pe: usize,
        words: usize,
    ) -> Result<()> {
        let expected = rect.pes() * elems_per_pe;
        if words != expected {
            return Err(Error::Size {
                symbol: symbol.into(),
                expected,
                actual: words,
            });
        }
        Ok(())
    }

    /// Copies `data` into the exported variable `symbol` on every PE of `rect`,
    /// `elems_per_pe` elements each.
    pub fn memcpy_h2d(
        &mut self,
        symbol: &str,
        data: &[u32],
        rect: Rect,
        elems_per_pe: usize,
        order: Order,
        data_type: DataType,
    ) -> Result<()> {
        let id = self.rpc.id(symbol, RpcKind::Var)?;
        self.check_size(symbol, rect, elems_per_pe, data.len())?;
        let t0 = std::time::Instant::now();
        let opts = MemcpyOptions::new(false, data_type as i32, order as i32, false);
        let mut task = Task::empty();
        // SAFETY: data outlives the blocking copy; the SDK only reads it.
        unsafe {
            (self.api().memcpy_h2d)(
                &mut task,
                self.this(),
                id,
                data.as_ptr() as *mut c_void,
                rect.x,
                rect.y,
                rect.w,
                rect.h,
                elems_per_pe as i32,
                &opts,
            )
        };
        self.finish(&mut task);
        IoStats::add(&mut self.io.h2d, t0, data.len());
        Ok(())
    }

    /// Copies the exported variable `symbol` from every PE of `rect` into `out`.
    pub fn memcpy_d2h(
        &mut self,
        symbol: &str,
        out: &mut [u32],
        rect: Rect,
        elems_per_pe: usize,
        order: Order,
        data_type: DataType,
    ) -> Result<()> {
        let id = self.rpc.id(symbol, RpcKind::Var)?;
        self.check_size(symbol, rect, elems_per_pe, out.len())?;
        let t0 = std::time::Instant::now();
        let opts = MemcpyOptions::new(false, data_type as i32, order as i32, false);
        let mut task = Task::empty();
        // SAFETY: out outlives the blocking copy and holds exactly the words checked above.
        unsafe {
            (self.api().memcpy_d2h)(
                &mut task,
                self.this(),
                out.as_mut_ptr() as *mut c_void,
                id,
                rect.x,
                rect.y,
                rect.w,
                rect.h,
                elems_per_pe as i32,
                &opts,
            )
        };
        self.finish(&mut task);
        IoStats::add(&mut self.io.d2h, t0, out.len());
        Ok(())
    }

    /// Copies row `r` of `data` (`elems_per_pe` elements a row) into `symbol` on every PE of row
    /// `r` of `rect`: one copy of the rectangle's height, not its area.
    pub fn memcpy_h2d_rowbcast(
        &mut self,
        symbol: &str,
        data: &[u32],
        rect: Rect,
        elems_per_pe: usize,
        data_type: DataType,
    ) -> Result<()> {
        let id = self.rpc.id(symbol, RpcKind::Var)?;
        let expected = (rect.h.max(0) as usize) * elems_per_pe;
        if data.len() != expected {
            return Err(Error::Size {
                symbol: symbol.into(),
                expected,
                actual: data.len(),
            });
        }
        let rowbcast = self.api().memcpy_h2d_rowbcast.ok_or_else(|| {
            Error::MissingSymbol(
                "cerebras::SdkRuntime::memcpy_h2d_rowbcast".into(),
                "this SDK has no row-broadcast copy (the batched builds' tables need it)".into(),
            )
        })?;
        let t0 = std::time::Instant::now();
        let opts = MemcpyOptions::new(false, data_type as i32, Order::RowMajor as i32, false);
        let mut task = Task::empty();
        // SAFETY: data outlives the blocking copy; the SDK only reads it.
        unsafe {
            (rowbcast)(
                &mut task,
                self.this(),
                id,
                data.as_ptr() as *mut c_void,
                rect.x,
                rect.y,
                rect.w,
                rect.h,
                elems_per_pe as i32,
                &opts,
            )
        };
        self.finish(&mut task);
        IoStats::add(&mut self.io.h2d, t0, data.len());
        Ok(())
    }

    /// Launches the exported function `name` with `args` and waits for it.
    pub fn launch(&mut self, name: &str, args: &[u32]) -> Result<()> {
        self.rpc.id(name, RpcKind::Func)?;
        let t0 = std::time::Instant::now();
        let name_s = CxxString::new(name);
        let vec = CxxVecU32::borrow(args);
        let opts = MemcpyOptions::new(false, 0, 0, false);
        let mut task = Task::empty();
        // SAFETY: mirrors are live for the duration of the call.
        unsafe { (self.api().call)(&mut task, self.this(), &*name_s, &vec, &opts) };
        self.finish(&mut task);
        self.io.launch.0 += 1;
        self.io.launch.1 += t0.elapsed().as_nanos() as u64;
        Ok(())
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        self.stop();
        // SAFETY: constructed runtime, destroyed exactly once.
        unsafe { (self.api().runtime_dtor)(self.this()) };
        // The artifacts object has no exported destructor; its two strings leak.
    }
}
