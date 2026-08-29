//! `pie-bin` library shim — exposes the composition-root seam so a boot gate can
//! drive it. **No logic lives here**; `main.rs` is a thin shell over these
//! modules and `run_standalone` is the one public composition seam.
//!
//! This is the composition root, not a role crate — its `[lib]` exists purely to
//! make the single compose seam testable, which is distinct from the worker's
//! old lib+bin anti-pattern (a *role* crate flipping identity by feature).
//!
//! The gate it was written for was tests/boot_smoke.rs, deleted with the dummy
//! engine it booted; the seam is driven by the Vulkan gates in `tests/gpu/`
//! now, which call the same `run_standalone` against a real device.

pub mod compose;
pub mod derive;
pub mod local;
pub mod ops;
pub mod sweep;
pub mod ui;

pub use compose::{StandaloneHandle, run_standalone};
