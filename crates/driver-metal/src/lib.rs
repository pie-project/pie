//! The Metal execution shell, in Rust.
//!
//! This was `driver-metal-new`, grown beside the C++ `driver-metal` rather
//! than inside it, on the reasoning that a rewrite which has to keep the old
//! one working is a rewrite that can be abandoned halfway without a revert.
//! The C++ shell was retired on 2026-08-10 and this took its name; what
//! follows describes the split it was built with, which is why it still
//! measures itself against a crate that is no longer in the tree.
//!
//! # What is here, and why it is shaped this way
//!
//! The C++ shell is ~42k lines, of which 13 files and ~8.6k lines name a
//! Metal or Objective-C type at all. The other 80% is scheduling, geometry,
//! pool arithmetic and plan interpretation -- logic that never touches the
//! GPU and is only in C++ because it was written next to the part that does.
//! So the split here is by that line rather than by subsystem:
//!
//! * [`layout`] and [`lowering`] are portable. They compile and test on any
//!   host, including the Linux boxes the rest of the workspace is developed
//!   on, because their inputs are text and integers.
//! * `gpu` is Apple-only and is where every `unsafe` message send lives.
//!   It is **one** gate, not five: `.wiki/driver/real-metal-north-star.md`
//!   §6 records what four gates inside one module cost the last time.
//!
//! The portable half is not a convenience. It is the half that can be tested
//! without a GPU, and keeping it importable is what stops it from drifting
//! back into the untestable half.
//!
//! # The gate is a feature, and that is the point
//!
//! `metal-4`, not `cfg(target_vendor = "apple")`, because:
//!
//! > **A platform cfg cannot be tested. A feature can.**
//!
//! On macOS `target_vendor = "apple"` is always true, so the portable half
//! was never compiled; on Linux it is always false, so the Apple half never
//! was. No machine built both, and no single job could catch a reference
//! across the boundary — which is exactly how three of them got in and
//! stayed. `cargo test -p driver-metal` builds the portable half and
//! `--features metal-4` builds all of it, both on the Mac the people who
//! work on this crate already have.
//!
//! There is no default feature on purpose. A default would have to be turned
//! OFF to reach the portable half, and rule 4 of
//! `.wiki/driver/north-star.md` is that a check which can be skipped will
//! be. A consumer that wants the serving path says so — `engine`'s
//! `driver-metal` feature does, in one line.
//!
//! # Where a thing lives
//!
//! The device rows are code spans and not links because this table is in the
//! portable half, which is compiled without them.
//!
//! | | |
//! |---|---|
//! | [`layout`] | how big, where, how many |
//! | [`lowering`] | fire shape → symbols, grids, operands |
//! | `device` | the only place `queue`, `heap`, `MTLBuffer` are the right words |
//! | `weights` | checkpoint → device |
//! | `pools` | what `layout` planned, allocated |
//! | `bind` | compile the symbols, stage the tables, dispatch |
//! | `fire` | what one fire keeps between steps |
//! | `program` | user programs: compile, cache, channel, run |
//! | `serve` | the transfers the engine asks for |
//!
//! [`batch`] and [`model`] are what is left of the driver's model knowledge,
//! and what is left is small on purpose. `facts.rs` — a private `ModelFacts`
//! parsed out of a `pie.model/1` JSON descriptor, plus the `arch_stem` that
//! turned `Qwen3MoeForCausalLM` into a dispatch key — is DELETED.
//! `.wiki/driver/real-metal-north-star.md` §4 states that this crate spelling
//! family names is a fact that failed to reach the crate that owns it; the
//! fact reached it. A checkpoint is matched to a `model::catalog` row by its
//! TENSORS, the row projects a `model::deployment::Deployment`, and what is
//! left here turns that value into the Metal-side numbers a kernel is
//! launched with.
//!
//! # Ownership
//!
//! The C++ shell hands out `void*` for every Metal object, because its header
//! is included by plain C++ translation units that cannot name an `id<>`.
//! Nothing here does. `Retained<ProtocolObject<dyn MTLBuffer>>` is the same
//! pointer with the retain/release already correct, and the reason the port
//! is worth doing at all is that the lifetime bugs the `void*` boundary can
//! express stop being representable.

#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout, clippy::print_stderr)]

pub mod batch;
pub mod channel;
mod error;
pub mod layout;
pub mod loader;
pub mod lowering;
pub mod model;

pub use error::{Error, Result};
pub use layout::{Batch, Region, Request};

// `metal-4` needs an Apple target: Metal has no implementation elsewhere, so
// enabling it on Linux would produce a wall of unresolved `objc2` paths
// instead of a sentence. Build WITHOUT it for the half of this crate that
// answers questions no GPU changes.
#[cfg(all(feature = "metal-4", not(target_vendor = "apple")))]
compile_error!(
    "`metal-4` needs an Apple target: Metal has no implementation elsewhere. \
     Build without it for the half of this crate that answers questions no \
     GPU changes."
);

#[cfg(feature = "metal-4")]
pub mod bind;
#[cfg(feature = "metal-4")]
pub mod device;
#[cfg(feature = "metal-4")]
pub mod fire;
#[cfg(feature = "metal-4")]
pub mod pools;
#[cfg(feature = "metal-4")]
pub mod program;
#[cfg(feature = "metal-4")]
pub mod serve;
#[cfg(feature = "metal-4")]
pub mod weights;

// # The device half
//
// Everything in the seven modules below needs a device to be correct;
// everything above them does not. That is the whole of the boundary, and
// `.wiki/driver/real-metal-north-star.md` §6 states why it is drawn once
// rather than per subsystem:
//
// > Four gates inside one module is how `tables` and `resolve` came to sit
// > ungated beside gated siblings and reach across.
//
// The cut is *does answering this need a device*, not *is this about the
// GPU*: `layout::tuning` is entirely about the GPU and is above the line,
// because its inputs are two integers.
//
// These seven used to sit under a `gpu` module whose whole job was to carry
// one `#[cfg]` for all of them. The nesting bought a structural guarantee --
// a file placed inside the subtree could not forget the gate -- and charged
// a path segment for it on every reference in the crate, in a crate that is
// a Metal driver, where `gpu::device::allocator` says GPU twice and means it
// once. It also grew a second job: re-exporting eighty-six names its children
// already re-exported, so `gpu::Pool` and `device::Pool` both resolved.
//
// The guarantee was narrower than it read. Three of the four ways to lose
// the gate are caught by the portable build, which CI runs: an ungated room
// that names a gated one, a portable module that `use`s a device room, and a
// declared-but-absent room all fail to compile. What neither the compiler nor
// the subtree caught is a self-contained device module placed at `src/` --
// beside `layout`, which is where someone unfamiliar with the
// split would put it. `objc2` is an unconditional dependency, so that
// compiles clean on a Mac and breaks on Linux. The subtree could only protect
// files placed INSIDE it; a file placed outside it was exactly as unprotected
// then as now. `tests/layering.rs` covers that case and says so with a
// measurement.
//
// The device half is reached through the room that owns the name, and only
// through it.
//
// Sixty-five names used to be re-exported flat from the crate root too, so
// every one of them had two paths: `driver_metal::device::Stepper` and
// `driver_metal::device::Stepper` named one type. That is §5's "one concept,
// two names" inside a single crate, and it is what let the engine reach past
// the seam into whatever it liked -- a facade is only a facade if the
// alternative is not also public.
//
// Nine of the sixty-five had a caller through the flat path. The other
// fifty-six existed because adding a name to a list is cheaper than deciding
// where it belongs (`.wiki/driver/real-metal-north-star.md` §9, "everything
// else goes private").
//
// What stays crate-root is the PORTABLE half above: `Error` and the layout
// types. Those answer questions no GPU changes, and they are
// the same on a machine with no Metal at all. There used to be a third —
// `facts::{ModelFacts, ModelFamily}`, a model definition living at the root
// of a DRIVER — and it is deleted rather than moved: a row in
// `model::catalog` answers what it answered, and it answers it once for
// every driver instead of once per driver.
// `tests/layering.rs` holds this to the two of them.
//
// # `unsafe`
//
// Every objc2 message send is `unsafe`, so the device half cannot carry the
// workspace's `unsafe_code = "forbid"`. What it carries instead is the rule
// that an `unsafe` block states the invariant it is relying on -- Metal's own
// API contract does not stop being a contract because it is written in
// Objective-C.
