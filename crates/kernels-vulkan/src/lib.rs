mod capability;
pub use crate::capability::Capability;

pub mod module;
pub use crate::module::{MODULES, code, embedded};

pub mod runtime;

pub mod points;
pub mod routine;
pub mod views;

pub mod attn;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod quant;
pub mod rope;
pub mod ssm;

/// Every entrypoint this tree can produce, from `build.rs`'s walk of the
/// `// pie:instantiate` lines.
///
/// # THE ROUTINE LAYER IS RETIRED HERE, AND THIS IS WHAT REPLACED IT
///
/// A `#[routine]` registry stood beside this: a `linkme` slice of
/// `Routine<Vulkan>` rows, `rows()`, `declared()` and `routines()` over it,
/// and 101 launcher fns filling it. Every launch now lives in a `#[claims]`
/// block, and the registry went with the fns because on THIS plane nothing
/// read a row. `model-ir`'s `Backend` has a cuda arm and a metal arm and no
/// vulkan arm, so `canon_symbol` — the walk that kept two rows alive in
/// `kernels-metal` — never reached this crate; the by-name reader was
/// `driver-vulkan`, which left the workspace at R3.
///
/// So the entrypoint census is the whole of what this crate declares about
/// its shaders, and it comes from the shaders themselves rather than from a
/// second list in Rust.
#[must_use]
pub fn entrypoints() -> Vec<String> {
    module::CENSUS.iter().map(|n| (*n).to_owned()).collect()
}
