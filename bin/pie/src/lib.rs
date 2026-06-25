//! `pie-bin` library shim — exposes the composition-root seam so the standalone
//! boot smoke can drive it. **No logic lives here**; `main.rs` is a thin shell
//! over these modules and `run_standalone` is the one public composition seam.
//!
//! This is the composition root, not a role crate — its `[lib]` exists purely to
//! make the single compose seam testable (`bin/pie/tests/boot_smoke.rs` → green
//! once golf's P5a `compose.rs` overlays), which is distinct from the worker's
//! old lib+bin anti-pattern (a *role* crate flipping identity by feature).

pub mod compose;
pub mod derive;
pub mod ops;
pub mod paths;

pub use compose::{Mode, StandaloneHandle, run_standalone};
