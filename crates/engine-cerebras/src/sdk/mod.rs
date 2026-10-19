//! A pure-Rust binding to the Cerebras SDK host runtime (`cerebras::SdkRuntime`).
//!
//! The SDK ships no C API and no headers: its host runtime is a C++ library,
//! `libsdkruntime.so`, whose only public surface is the pybind module used by
//! `cs_python`. This module dlopens the SDK's shared libraries at runtime and
//! calls their mangled C++ entry points directly, with the object layouts and
//! calling conventions mirrored in [`abi`] and [`sys`]. There is no Python,
//! no C++ shim and no build script; the crate builds on a box without the SDK.
//!
//! # Host setup
//!
//! The SDK's ELF files hard-code their interpreter and rpath under `/cb/...`
//! and its helper binaries under `/cbcore/...`, so the unpacked SDK image
//! must be reachable at those absolute paths. With the image at
//! `/root/cs_sdk/rootfs` that is two symlinks:
//!
//! ```text
//! ln -s /root/cs_sdk/rootfs/cb /cb
//! ln -s /root/cs_sdk/rootfs/cbcore /cbcore
//! ```
//!
//! No container, PRoot or environment variables are needed after that.
//! [`Sdk::open`] looks for the libraries in `$PIE_CEREBRAS_SDK_LIB`, then
//! `/cbcore/lib`, and prepends the sibling `bin/` directory to `PATH` because
//! the runtime shells out to helpers such as `cself_default`.
//!
//! # What is and is not covered
//!
//! Covered: the simulator platform, a CS system by `cmaddr` (untested: no
//! hardware here), `load`/`run`/`stop`, blocking host to device and device to
//! host copies, RPC launches with `u32` arguments, symbol ids read from the
//! compiler's `bin/out_rpc.json`. Not yet covered: streaming copies,
//! non-blocking tasks, ports.
//!
//! The simulator writes `sim.log`, `simconfig.json`, `sim_stats.json` and
//! `wio_flows_tmpdir.*` into the process's current directory, and a stale
//! scratch directory from an aborted run makes the next start throw. Run
//! from a scratch directory (`fabric-run` changes into its spec's directory).
//!
//! The SDK initialises its message system when the first runtime of a
//! process is created and asserts that every later runtime is created on
//! that same thread; [`Runtime::new`] refuses on another thread instead.
//! Together with the one-simulator-per-process rule this is why programs are
//! run in dedicated processes (`fabric-run`).
//!
//! A C++ exception thrown inside the SDK cannot cross this boundary and
//! aborts the process. The SDK's own argument checks are therefore
//! duplicated here where they are cheap (buffer sizes, symbol kinds).

#![allow(unsafe_code)]

pub mod abi;
pub mod cslc;
pub mod sys;

pub use cslc::{Arch, Compile, Cslc};

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
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// The loaded SDK libraries and their resolved entry points.
pub struct Sdk {
    api: sys::Api,
    lib_dir: PathBuf,
    // Order matters on drop: the runtime library goes first.
    _runtime: Library,
    _artifacts: Library,
    _platform: Library,
}

impl Sdk {
    /// Default library directory: `$PIE_CEREBRAS_SDK_LIB` or `/cbcore/lib`.
    pub fn default_lib_dir() -> PathBuf {
        std::env::var_os("PIE_CEREBRAS_SDK_LIB")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/cbcore/lib"))
    }

    pub fn open() -> Result<Arc<Self>> {
        Self::open_at(&Self::default_lib_dir())
    }

    /// Whether the SDK libraries are on this box, without loading them
    /// (loading notes the cwd and writes simulator logs there later).
    pub fn available() -> bool {
        Self::default_lib_dir().join("libsdkruntime.so").is_file()
    }

    pub fn open_at(lib_dir: &Path) -> Result<Arc<Self>> {
        let load = |name: &str| {
            let path = lib_dir.join(name);
            // SAFETY: loading the SDK's libraries runs their initializers, which
            // is exactly what cs_python does when it imports the pybind module.
            unsafe { Library::new(&path) }
                .map_err(|e| Error::Library(path.display().to_string(), e.to_string()))
        };
        let platform = load("libsdk_execution_platform.so")?;
        let artifacts = load("libsdk_compile_artifacts.so")?;
        let runtime = load("libsdkruntime.so")?;
        let api = sys::Api::resolve(&artifacts, &platform, &runtime)?;
        prepend_path(&lib_dir.join("../bin"));
        Ok(Arc::new(Sdk {
            api,
            lib_dir: lib_dir.to_path_buf(),
            _runtime: runtime,
            _artifacts: artifacts,
            _platform: platform,
        }))
    }

    pub fn lib_dir(&self) -> &Path {
        &self.lib_dir
    }
}

/// The thread that created this process's first runtime. The SDK's message
/// system is initialised then and asserts that later initialisations happen
/// on the same thread, so a runtime built elsewhere would abort the process.
static OWNER: std::sync::OnceLock<std::thread::ThreadId> = std::sync::OnceLock::new();

fn owning_thread() -> bool {
    *OWNER.get_or_init(|| std::thread::current().id()) == std::thread::current().id()
}

fn prepend_path(dir: &Path) {
    let current = std::env::var_os("PATH").unwrap_or_default();
    let mut parts = vec![dir.to_path_buf()];
    parts.extend(std::env::split_paths(&current));
    if let Ok(joined) = std::env::join_paths(parts) {
        // SAFETY: called once at open time, before the SDK spawns any helper.
        unsafe { std::env::set_var("PATH", joined) };
    }
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
    pub fn new(target: Target) -> Self {
        Simulator {
            target,
            num_threads: 16,
            suppress_trace: true,
            dump_core: false,
        }
    }
}

/// Where a program runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Platform {
    Simulator(Simulator),
    /// A CS system reached at `IP:port` (the SDK's `cmaddr`). Building it
    /// contacts the system; if that fails the SDK throws, which aborts this
    /// process (see the module docs).
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

/// Width of each element on the wire. Host buffers are always `u32` words;
/// 16-bit copies pack two device elements per word.
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
            symbols.insert(
                name.to_string(),
                RpcSymbol {
                    id: id as u16,
                    kind,
                    ty,
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

/// A loaded program on a platform: one `cerebras::SdkRuntime`.
///
/// Calls are blocking; every task the SDK hands back is waited on and
/// released before the method returns. `stop` runs on drop if it has not
/// been called.
pub struct Runtime {
    sdk: Arc<Sdk>,
    rpc: Rpc,
    this: Box<RuntimeMem>,
    _artifacts: Box<ArtifactsMem>,
    _platform: PlatformMem,
    running: bool,
}

impl Runtime {
    /// Opens the compiler output at `artifacts_dir` (the `-o` directory given
    /// to `cslc`) on `platform`. `msg_level` is the SDK's log threshold
    /// (`"WARNING"` is what cs_python uses).
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
        // SAFETY: argument order and types follow the exported constructor
        // signature; the three by-value strings are owned by this frame.
        unsafe {
            (api.runtime_ctor)(
                this.as_ptr(),
                artifacts.as_ptr(),
                platform.as_ptr(),
                &mut *msg,
                true,
                false,
                false,
                &mut *cslc_prefix,
                &mut *out_prefix,
            )
        };
        Ok(Runtime {
            sdk,
            rpc,
            this,
            _artifacts: artifacts,
            _platform: platform,
            running: false,
        })
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
        Ok(())
    }

    /// Launches the exported function `name` with `args` and waits for it.
    pub fn launch(&mut self, name: &str, args: &[u32]) -> Result<()> {
        self.rpc.id(name, RpcKind::Func)?;
        let name_s = CxxString::new(name);
        let vec = CxxVecU32::borrow(args);
        let opts = MemcpyOptions::new(false, 0, 0, false);
        let mut task = Task::empty();
        // SAFETY: mirrors are live for the duration of the call.
        unsafe { (self.api().call)(&mut task, self.this(), &*name_s, &vec, &opts) };
        self.finish(&mut task);
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

/// Reinterprets `f32` values as the `u32` words the copy calls take.
pub fn f32_words(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Reinterprets copied-back words as `f32`.
pub fn words_f32(v: &[u32]) -> Vec<f32> {
    v.iter().map(|x| f32::from_bits(*x)).collect()
}
