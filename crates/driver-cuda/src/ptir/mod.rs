//! PTIR on CUDA: the device half of the channel plane.
//!
//! # Where the line is
//!
//! A PTIR program is a user's decoding logic — sample this, compare that,
//! push a token onto a channel — traced by `tensor-dsl`, compiled by
//! `tensor-compiler`, and delivered to this driver through the ABI as
//! [`PieProgramDesc`](driver_api::local::PieProgramDesc): a launch package
//! that says what the program *is*, and a table of emitted CUDA sources that
//! say how its fusable regions run.
//!
//! Almost none of executing one is CUDA. Adopting the launch package, indexing
//! the emitted table, keying the compile cache, laying out the lane table,
//! deciding whether a fire may run, evaluating the reference pass — all of it
//! is arithmetic over the ABI's records, all of it is identical on Metal, and
//! all of it lives in [`driver_pipeline`], which two shells share and neither
//! owns. The C++ answered the same question by keeping THREE hand-written
//! copies of one golden model, and `.wiki/driver/progress-metal.md` opens by counting them.
//!
//! What is left — this module — is the part that genuinely names a CUDA
//! symbol:
//!
//! * [`nvrtc`] — one self-contained translation unit in, one cubin out. No
//!   include path, real `sm_XY` arch, and three float flags that are a
//!   reproducibility contract rather than a tuning knob.
//! * [`module`] — the cubin loaded: `CUmodule`, `CUfunction`, and the launch
//!   width the compiled function's register pressure permits, rounded down to
//!   a power of two because the generated reductions halve `blockDim.x`.
//! * [`disk`] — cubins that survive the process, keyed on the identity *plus
//!   the emitted source's fingerprint*, which is what stops a template edit
//!   from silently reusing yesterday's kernel.
//! * [`control`] — the readiness and commit kernels. They are prebuilt on
//!   CUDA rather than emitted, and the C++'s copies are private to a crate
//!   this one replaces, so they are compiled here through the same NVRTC path
//!   as every emitted region.
//! * [`launch`] — `cuLaunchKernel` and its argument marshalling, which is
//!   unchecked in both arity and type, so the marshalling lives in one place
//!   rather than at each call site.
//! * [`params`] — the one device record whose CUDA layout is NOT the shared
//!   crate's. `Status` and `ValueDesc` cross as themselves; `OpParams` is 64
//!   bytes there and 88 here, and the difference is silent rather than loud.
//! * [`ring`] — the device-side channel rings: the cells the kernels read and
//!   write, and the four cursors the control kernels advance. Native bytes on
//!   the device, bit-packed on the wire, and the difference is invisible until
//!   the first bool channel.
//! * [`bridge`] — where that difference LANDS: the copy between the pinned
//!   host mirror the engine polls and the device rings the kernels use. The
//!   two planes are different memory by construction, so a fire pulls its
//!   inputs across and pushes its outputs back.
//! * [`runtime`] — the three tiers and the negative cache, assembled past the
//!   last failure so a program that fails halfway installs nothing.
//!
//! * [`fire`] — one stage prepared and launched: the lane table, the
//!   descriptors, the params, the offsets, the scratch and the side tables,
//!   then one CTA per lane.
//!
//! # What is not here yet
//!
//! Multi-lane grouping, the intrinsic bindings a logits-reading program needs,
//! and the ticketed channel path that lets a table be staged ahead of the fire
//! that uses it. The single-lane epilogue runs, which is the shape every
//! decode loop has.

pub mod bridge;
pub mod control;
pub mod disk;
pub mod fire;
pub mod launch;
pub mod module;
pub mod nvrtc;
pub mod params;
pub mod ring;
pub mod session;
pub mod runtime;

pub use control::{Control, MAX_RING};
pub use disk::{Disk, disk_key};
pub use fire::{INTRINSIC_SLOTS, Prepared};
pub use launch::{Args, launch_control};
pub use module::Module;
pub use nvrtc::{CompileError, FailureKind};
pub use params::{CudaOpParams, params_bytes};
pub use ring::{ChannelShape, Cursors, Rings, native_cell_bytes};
pub use runtime::{Compiled, Programs, Region, Runtime, Stage, Target};
