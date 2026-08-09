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
//! The `gpu::*` rows are code spans and not links because this table is in
//! the portable half, which is compiled without them.
//!
//! | | |
//! |---|---|
//! | [`layout`] | how big, where, how many |
//! | [`lowering`] | fire shape → symbols, grids, operands |
//! | `gpu::device` | the only place `queue`, `heap`, `MTLBuffer` are the right words |
//! | `gpu::weights` | checkpoint → device |
//! | `gpu::pools` | what `layout` planned, allocated |
//! | `gpu::bind` | compile the symbols, stage the tables, dispatch |
//! | `gpu::fire` | what one fire keeps between steps |
//! | `gpu::program` | user programs: compile, cache, channel, run |
//! | `gpu::serve` | the transfers the engine asks for |
//!
//! [`batch`], [`facts`] and [`model`] are what is left of the driver's model
//! knowledge. They still spell family names, which
//! `.wiki/driver/real-metal-north-star.md` §4 states is a fact that failed to
//! reach the crate that owns it, and they are going to `crates/model` whole.
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
pub mod facts;
pub mod layout;
pub mod loader;
pub mod lowering;
pub mod model;

pub use error::{Error, Result};
pub use facts::{ModelFacts, ModelFamily};
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
pub mod gpu;

// The device half is reached through `gpu::`, and only through it.
//
// Sixty-five names used to be re-exported flat from here, so every one of
// them had two paths: `driver_metal::Stepper` and `driver_metal::gpu::Stepper`
// named one type. That is §5's "one concept, two names" inside a single
// crate, and it is what let the engine reach past the seam into whatever it
// liked -- a facade is only a facade if the alternative is not also public.
//
// Nine of the sixty-five had a caller through the flat path. The other
// fifty-six existed because adding a name to a list is cheaper than deciding
// where it belongs (`.wiki/driver/real-metal-north-star.md` §9, "everything
// else goes private").
//
// What stays crate-root is the PORTABLE half above: `Error`, the layout
// types, the model facts. Those answer questions no GPU changes, and they are
// the same on a machine with no Metal at all.
// `tests/layering.rs` holds this to the three of them.
