//! The per-symbol JIT: a root, an instantiation, and one entry point.
//!
//! Compilation happens per instantiation: [`Root`] holds the device text,
//! [`Ctx`] is what a routine body launches through, [`ArgValue`] is one bound
//! argument, [`Abi`] says how a crossing type spells itself in C++, and
//! [`nvrtc`]/[`Error`] are the compiler and its refusal type.
//!
//! `nvrtc`, `error` and `abi` live here, not in `runtime`/`x`, because each
//! previously pointed the wrong way across a module boundary.

pub mod abi;
mod arg;
/// One instantiation, compiled at most once per process, and its entry point.
#[cfg(feature = "_cuda")]
pub mod cache;
mod ctx;
#[cfg(feature = "_cuda")]
mod error;
#[cfg(feature = "_cuda")]
mod launch;
/// NVRTC, as this crate asks it: one root, one instantiation, one cubin.
#[cfg(feature = "_cuda")]
pub mod nvrtc;
mod root;
#[cfg(feature = "_cuda")]
pub(crate) mod device;
/// The page-locked host buffer a capturable H2D must read from.
pub mod pinned;
/// One bound argument's value, as the launch ABI carries it.
pub mod value;

pub use abi::{Abi, ByValue, Layout, fp8_kind};
pub use ctx::{Ctx, Cuda, Launch};
#[cfg(feature = "_cuda")]
pub use error::Error;
pub use pinned::PinnedBytes;
pub use root::{Headers, Root, Toolchain};
pub use value::ArgValue;

/// Is `p` 16-byte aligned — the test a body makes before it picks a vector
/// width. A host-side check, made before any launch (see
/// `kernels/gemm/gemv.cuh:91`); shared here, feature-free, so any routine
/// body may call it from either build.
#[must_use]
pub fn aligned16(p: *const core::ffi::c_void) -> bool {
    p.addr() & 15 == 0
}

/// This entry point's name, as a `&'static str`.
///
/// # WHY A NAME HAS TO OUTLIVE THE STATEMENT THAT BUILT IT
///
/// [`kernels::routine::Fire`] takes `&'static str`, and that is right for the
/// three shader planes: an entry point there is a fixed name in a module
/// compiled ahead of time, so the string is a literal and the bound costs
/// nothing. This plane JITs TEMPLATES, and a template's entry point is not
/// known until the element type is — `::pie::attn::attn_sink_rescale<__nv_bfloat16>`
/// is built by `format!` at the call, out of `T::CPP`.
///
/// Widening `Fire` to a `Cow` or a lifetime would spend four planes'
/// signatures on one plane's habit, and the habit is bounded: the names come
/// from a finite cross of routines and element types, and every one of them is
/// about to be compiled and cached under this very string.
///
/// Feature-free and beside [`aligned16`] for the reason [`value`]'s header
/// gives: a routine body NAMES a symbol whether or not this build can launch
/// one, so the naming cannot sit behind `_cuda` with the cache that consumes it.
#[must_use]
pub fn symbol(name: &str) -> &'static str {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static NAMES: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let mut map = NAMES
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(found) = map.get(name) {
        return found;
    }
    // Leaked on purpose, exactly as `cache::slot` leaks: the name is live for
    // the rest of the process either way, and leaking is what lets the borrow
    // escape the lock.
    let fresh: &'static str = Box::leak(name.to_owned().into_boxed_str());
    map.insert(name.to_owned(), fresh);
    fresh
}


/// One routine, in this backend's instantiation of the machinery.
pub type Routine = kernels::routine::Routine<Cuda>;

// `Family` STOOD HERE, and a namespace on the routine deleted it.
//
// It held two things: a `&'static str` and a `&'static [Routine]`. The string
// was never stated -- `family!(ROUTINES)` passed `module_path!()` and
// `segment_after_crate` took the first segment after the crate root -- so the
// whole type was ONE DERIVABLE STRING ATTACHED TO A GROUP.
//
// `module_path!()` expands wherever it is written, and `#[routine]` is written
// inside the same module. So the string derives per ROUTINE, the group has
// nothing left to hold, and fifteen `FAMILY` statics and the `FAMILIES` list
// that indexed them are one flat list of rows.
//
// It also settled a question the grouping had to explain in a comment on
// `FAMILIES`: `attn::xqa` and `attn::fa2` were *"two families sharing one
// namespace, as long as no name repeats"*. Both modules are under `attn`, so
// `namespace` answers `"attn"` for both and there is nothing to say.
//
// `segment_after_crate` moved with it, to `kernels::routine::namespace`.

// THE PLANE'S `routine!` WRAPPER STOOD HERE AND HAS NO CALLERS.
//
// It filled this backend in so a membership list could name only the
// `fn`. There is no membership list: `#[routine]` builds the row beside
// the `fn` and a distributed slice collects it, so the only caller of
// `kernels::routine!` is the attribute, which names the backend through
// `crate::Plane`.
