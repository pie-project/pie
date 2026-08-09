//! Generate `api.rs`: one typed Rust function per row, into `OUT_DIR`.
//!
//! # Why a build script writes it
//!
//! The direct-call surface is one function per row, and the rows are Rust
//! data — `static` arrays of `DeviceKernel`, in this crate's own
//! `src/families/*.rs` and in `kernels-cuda`. Nothing in the language reads
//! one of those at compile time: a `macro_rules!` cannot iterate a static,
//! and a proc-macro would be a third crate that has to depend on the tables
//! to expand against them, which the tables' own crate cannot then depend on.
//! A build script can just call the emitter, which is how `kernels-cuda`
//! already generates its `ffi.rs` and its C shim. The alternative is not a
//! cleverer macro, it is a hand-written wrapper per row — the thing
//! `emit_c_shim` exists so that nobody writes.
//!
//! A checked-in `api.rs` with a test that regenerates and diffs it was the
//! other candidate and it loses on one property: a stale file still COMPILES.
//! A row whose operands changed under a hand-refreshed façade gives its
//! callers a function that binds the old list, and `Args::bind` refuses it at
//! the launch — the run-time failure §12.4 says the generation exists to
//! convert into a build-time one. Generated into `OUT_DIR`, it cannot be
//! stale by construction.
//!
//! # Why the emitter is included by `#[path]`, and why its siblings are too
//!
//! A build script cannot depend on the crate it builds, so `src/emit.rs` is
//! pulled in as a module rather than reached through `kernels_cuda_new::`.
//! `kernels-cuda/build.rs` records the same constraint about `abi.rs` and
//! draws the consequence in one sentence: *"a module here that reached for a
//! sibling, or for `crate::`, would have to be pulled in with it."* It then
//! pulls in fourteen of them, because that is what reading the rows costs.
//!
//! This script pulls in four, for the same reason and after the same defect.
//! The emitter used to read `kernels_cuda::norm_device`'s two statics — the
//! only rows that existed when the pilot shipped — and once the families
//! began declaring their own it covered **ten of the hundred and thirty-five
//! the crate then declared**. Every other row had no typed entry point at
//! all, reachable only through `runtime::fire` with a string, which is
//! precisely the surface §12.4 says the generated façade exists to replace.
//! The rows live in `src/families/*.rs`, so this script reads
//! `src/families/*.rs`.
//!
//! `crate::` inside an included module resolves against THIS file, since a
//! build script is its own crate root — so the module tree below has to be
//! spelled with the library's own names, and is. `families` reaches for
//! `crate::device` and `crate::unit`; `unit` reaches for `crate::device`,
//! `crate::source` and `crate::families`; the closure stops there.
//! `#[allow(dead_code, unused_imports)]` on each is
//! `kernels-cuda/build.rs`'s trick with one lint added: this script calls two
//! of `emit`'s items and none of anything else's, and `device.rs`'s
//! `pub use` re-export is unused HERE and load-bearing in the library.
//! Silencing both here is what keeps a genuine unused-code warning in the
//! library readable.
//!
//! The cost is worth stating because it is met the first time a table is
//! edited: the tables are now compiled TWICE, once into this script and once
//! into the library, so a broken row is two errors rather than one and the
//! script's copy is reported first. That is the same bargain
//! `kernels-cuda/build.rs` makes with fourteen modules, and the alternative
//! to paying it is not reading the rows.
//!
//! # The one module that is a stub, and why it cannot be the file
//!
//! `crate::source` below is not `src/source.rs`. That file carries the header
//! set through `include!(concat!(env!("OUT_DIR"), "/carried.rs"))` — a file
//! THIS script writes, at the bottom of `main`. Including it would ask the
//! compiler for a file the script has not run yet: a clean build fails at the
//! include, and an incremental one succeeds off the previous run's output,
//! which is the worse half of the two orders because it makes the failure
//! look intermittent.
//!
//! What the tables take from that module is small and none of it is a source:
//! `Unit`'s `root` field, `norm`'s two roots, and the three items
//! `Unit::cache_key` folds. So the stub carries no text at all rather than a
//! second `include_str!` of the same two files — a copy drifts, an empty
//! string cannot, and nothing in this script hands a `Unit` to NVRTC. The
//! two functions `panic!` rather than answer: a cache key computed over an
//! empty header set is a WRONG key, and a wrong key is the one bug
//! `program/cache.rs` documents as unfindable.
//!
//! # What this script does NOT need
//!
//! A CUDA toolkit. `kernels-cuda/build.rs` needs nvcc, CMake and
//! `cuda_runtime.h` because it compiles an archive; this one writes a text
//! file, so the crate builds on a machine that has never seen a GPU — which
//! is the whole claim layers 1 and 2 make in `src/lib.rs`, and it would be an
//! odd claim to make while the build script quietly broke it. The modules
//! pulled in below are layer-1 and layer-2 data for the same reason: a row is
//! a `KernelSig` and a unit is a `&'static str`, and neither has ever needed
//! `cudarc`.

// This crate's own tables and emitter, read by the build script that runs
// them. The names are the library's, because `crate::` inside these files
// resolves against this one -- see the header for the whole argument.
#[allow(dead_code, unused_imports)]
#[path = "src/device.rs"]
mod device;
#[allow(dead_code, unused_imports)]
#[path = "src/families/mod.rs"]
mod families;
#[allow(dead_code, unused_imports)]
#[path = "src/unit.rs"]
mod unit;
// The fn-world floor, because `families::ALL` names `x::rope::UNITS` — and
// the REAL floor, not a stub: `carried::generate` walks `unit::UNITS` into
// the header set, and a stub here would carry a different set of files than
// the library holds. Cargo compiles a build script under the package's feature
// `cfg`s (the `CARGO_FEATURE_*` variables are the RUN-time spelling of the
// same fact), so with `cuda-12` on, `x`'s `_cuda` half compiles here too —
// against the `runtime`/`execution`/`table` stubs below, which exist to be
// compiled and never run, exactly as `mod source` does.
#[allow(dead_code, unused_imports)]
#[path = "src/x/mod.rs"]
mod x;

/// `src/runtime`'s signatures as far as `x`'s `_cuda` half names them, and no
/// further — `mod source`'s argument, one module over. This script walks rows
/// and never fires: every body here is `unreachable!()`, and what keeps the
/// two sides honest is the loud direction — a fn-world file that grows a new
/// `crate::runtime` dependency is a compile failure of this script naming the
/// missing item.
#[allow(dead_code, unused_imports, clippy::unused_self)]
#[cfg(feature = "_cuda")]
mod runtime {
    use core::ffi::c_void;

    /// The marshalling cell `x::Abi::arg` answers with.
    pub enum ArgValue {
        I32(i32),
        U32(u32),
        F32(f32),
        Bool(bool),
        I64(i64),
        Usize(usize),
        Ptr(*mut c_void),
        Bytes { ptr: *const u8, len: usize },
    }

    /// What `cuLaunchKernel` takes; `x::launch::Launch` converts into it.
    pub struct Launch {
        pub grid: [u32; 3],
        pub block: [u32; 3],
        pub smem: u32,
    }

    pub struct Args;
    impl Args {
        pub fn bind(
            _sig: &kernels::KernelSig,
            _values: &[ArgValue],
        ) -> Result<Self, &'static str> {
            unreachable!("the build script never fires")
        }
    }

    pub struct Stream;
    impl Stream {
        pub unsafe fn from_runtime(_raw: *mut c_void) -> Self {
            unreachable!("the build script never fires")
        }
    }

    pub mod cache {
        pub struct Module;
        impl Module {
            pub fn fire(
                &self,
                _sig: &kernels::KernelSig,
                _launch: super::Launch,
                _args: &mut super::Args,
                _stream: super::Stream,
            ) -> Result<(), &'static str> {
                unreachable!("the build script never fires")
            }
        }
        pub fn module(_index: usize, _unit: &crate::unit::Unit) -> Result<Module, &'static str> {
            unreachable!("the build script never fires")
        }
    }
}

/// `src/execution.rs` and `src/table/` as far as `x::route` names them.
/// Compiled, never run: this script emits from rows and routes nothing.
#[allow(dead_code)]
#[cfg(feature = "_cuda")]
mod execution {
    pub enum Service {
        DriverOp,
    }
    pub fn service(_symbol: &str) -> Option<Service> {
        unreachable!("the build script never routes")
    }
}

#[allow(dead_code)]
#[cfg(feature = "_cuda")]
mod table {
    pub fn sig(_symbol: &str) -> Option<&'static kernels::KernelSig> {
        unreachable!("the build script never routes")
    }
}

/// `src/source.rs` as far as the tables need it, and no further — see the
/// header for why it cannot be the file itself.
#[allow(dead_code, unused_imports)]
mod source {
    /// One header, and the name an `#include` spells to reach it. The
    /// library's type, restated because `Unit::cache_key_with` takes a slice
    /// of it; a field added there and missing here is a build failure of this
    /// script, which is the only direction that can go wrong silently.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Header {
        pub name: &'static str,
        pub text: &'static str,
    }

    /// No headers, because this script resolves no `#include`: it walks rows.
    pub const DEVICE_HEADERS: &[Header] = &[];

    /// Empty for [`DEVICE_HEADERS`]' reason, and separate for the library's:
    /// there it is the device set joined with the vendored one, and a unit
    /// that compiles against FlashInfer names it instead. This script never
    /// compiles, so the distinction costs nothing here — but the NAME has to
    /// exist, because `src/unit.rs` imports it and this module is what that
    /// import resolves against inside the build script.
    pub const ALL_HEADERS: &[Header] = &[];

    /// The unit roots a family names symbolically.
    ///
    /// Empty strings, not a second `include_str!`. The emitter reads
    /// `KernelSig`s and never a root's text, and a copy of the two paths here
    /// would be a second place for a moved `.cuh` to be right or wrong.
    pub mod roots {
        pub const NORM_ALTUP_AUX: &str = "";
        pub const NORM_ELEMENTWISE: &str = "";
    }

    /// Refused rather than answered: with [`DEVICE_HEADERS`] empty this would
    /// hash nothing and return a plausible number, and a cubin cache served
    /// under a key that spans less than what produced it is the failure
    /// `driver-cuda/src/program/cache.rs` records in the past tense.
    pub fn digest(_headers: &[Header]) -> u64 {
        panic!("build.rs carries no header set: a cache key computed here would be wrong")
    }

    /// Refused for [`digest`]'s reason — a key half computed from real text
    /// and half from a stub is a key nothing can tell from a real one.
    pub(crate) fn fnv1a64(_bytes: &[u8]) -> u64 {
        panic!("build.rs carries no source text: a cache key computed here would be wrong")
    }
}

mod carried;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // The rows themselves. Cargo would rebuild this script anyway -- every
    // file below is one of its compile inputs, and rustc reports them -- but
    // the list is stated because it is also the ANSWER to "what does the
    // generated API depend on", and a reader asking that should not have to
    // reconstruct it from a `#[path]` attribute.
    println!("cargo:rerun-if-changed=src/device.rs");
    println!("cargo:rerun-if-changed=src/unit.rs");
    println!("cargo:rerun-if-changed=src/families");
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    // `emit::emit_rust_api(&emit::all_rows())` wrote `api.rs` here until north
    // star §6 half A. What it walked was `unit::rows()` and NOT
    // `table::ROW_TABLES` — the device row list, which every fn-world family
    // still contributes to, because a unit's rows carry the instantiation
    // strings NVRTC lowers. So the generator was never going to run dry on
    // its own: the row list it read is the one the JIT needs.
    //
    // The three `rerun-if-changed` lines above stay. They are `carried`'s
    // now — `device.rs`, `unit.rs` and `families/` are what decides the
    // header set, and that is the half of this script that survives.

    carried::generate(&out);
}
