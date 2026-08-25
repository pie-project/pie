//! The WebGPU execution shell: what it takes to actually FIRE the modules
//! `kernels-wgpu` states. Pure Rust, no `-sys` crate in the closure
//! (`tests/pure.rs`); `naga` parses WGSL anywhere, so [`reflect`] checks a
//! module against its row with no adapter; `native` gates what must ask one.
//!
//! **WebGPU has no push constants.** A launch's scalars are the fields of ONE
//! uniform buffer the shell writes and binds at `@group(1) @binding(0)`, so
//! [`binding::Params`] carries one variant where `driver-vulkan` carries two,
//! and [`binding::params_from`] places the run at the offsets `naga` says the
//! module reads them at. [`baker::encode`] packs the same block for the claim
//! plane.
//!
//! **A dispatch may not bind one buffer both readable and writable**, tracked
//! per ALLOCATION, so [`binding::Arena`]'s one buffer trips it on every real
//! plan: `device::Device::run_all` shadows the offending read ranges. WGSL has
//! no 16-bit storage type, so bf16 crosses as `array<u32>` and a decode lane
//! owns the PAIR -- [`geometry::Rule::SdpaVector`] halves the workgroup too.
//!
//! **The seam is exactly this, and nothing else:**
//!
//! ```ignore
//! impl driver_wgpu::binding::Allocation for device::Buffer {
//!     fn size(&self) -> u64 { self.size }
//! }
//! impl driver_wgpu::binding::Resolve for Store {
//!     type Buffer = device::Buffer;
//!     fn weight(&self, name: &str) -> Option<&device::Buffer>;
//!     fn named(&self, value: model_ir::plan::ValueId) -> Option<&device::Buffer>;
//!     // Defaulted to `None`; state them only where the text needs them.
//!     fn kv(&self, layer: u16, values: bool) -> Option<&device::Buffer>;
//!     fn slab(&self, layer: u16, which: &'static str) -> Option<&device::Buffer>;
//!     fn number(&self, which: binding::FireNumber) -> Option<u32>;
//!     fn table(&self, which: binding::FireTable) -> Option<&device::Buffer>;
//! }
//! ```
//!
//! # THE LEGACY WALK IS GONE FROM THIS CRATE
//!
//! [`baker`] is what executes a plan. What executed one before was
//! `model_compiler::lower` -- a `Lowered` of `Launch`es over a flat run of
//! `Arg`s -- and that crate module is deleted, along with
//! `kernels::routine::Routine`, which was the arm-and-body pair a launch was
//! fired through. Everything here that named either is gone.
//!
//! FOUR MODULES WENT WHOLE, because nothing was left in them that anything
//! read:
//!
//! * `dispatch` -- `plan_one`/`plan_all` were the JOIN, turning one `Launch`
//!   into a `Dispatch` by asking `binding` where an operand lives and
//!   `geometry` how many workgroups a rule wants. `Dispatch`, `Built`,
//!   `Sources` and `Undispatchable` were its vocabulary; `baker::dispatch` has
//!   its own, generated from the claim table rather than assembled from a row.
//!   `Geometry` -- the fire-wide model shape -- went with it, and
//!   `baker::stage::KvGeometry` is what carries that fact now.
//! * `lowering` -- `hold::Handles` held a statement's operands and minted a
//!   handle per ask, `bind` read a signature's `sources` column into them
//!   through `kernels::bind`, `views` built the host aggregates a `Ty::Raised`
//!   operand wants, `cached` kept the derived lowerings by fire shape, and
//!   `routine::plan` ran a crossed body against a `Planner`. `hold::LIVE` was
//!   the roster of crossed STEMS the fork consulted; `kernels_wgpu::
//!   points_dispatch::CLAIMED` is the roster now, and it is a different one.
//! * `shell` -- `Shell` was one assembled server, and its whole reason to
//!   exist was checking that the pieces a caller supplied separately agreed.
//!   Its methods all went through `turns::Serving`.
//! * `encode` -- `Encoder` was `Encode` over a real adapter for the routine
//!   plane, and answered a body's asks through `lowering::bind::one`.
//!   [`baker::encode::Encoder`] is the same seam for the claim plane, and
//!   unlike this one it names no `wgpu` type at all.
//!
//! What the surviving modules lost is recorded in each of them:
//! `binding::{extent, resolve, bind, params, Unbindable}`,
//! `serve::{Fire, fire, record, logits, Unread}`,
//! `resources::Frame::seriation`, `turns::{Serving, Unstepped}` and
//! `frames::Launched`.

// WAS `forbid`, AND THE ONE THING THAT TOOK IT DOWN IS NAMED HERE.
//
// The manifest's note still holds for all but one line: `wgpu` is a safe API
// and this crate is safe Rust, including the whole device half. The exception
// is `wgpu::ExperimentalFeatures::enabled()`, which is the only way to ask an
// adapter for `EXPERIMENTAL_COOPERATIVE_MATRIX` and is `unsafe` because its
// contract reads *"there may be UB-containing bugs in these apis"* -- an
// admission about wgpu's own implementation, not a proof obligation this
// crate can discharge by reasoning about a pointer.
//
// So the ratchet is `deny` and not `forbid`, and there is exactly ONE
// `#[expect(unsafe_code)]` in the tree, in `device.rs` beside the device
// request, with the argument for it written out there. `deny` still makes any
// second one a compile error, and `expect` makes the first one a compile error
// the day it stops being needed. Grep for `unsafe_code` to audit this crate;
// two hits is the whole story.
#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout)]

// THE EXECUTOR, and it is ungated on purpose rather than by omission.
//
// `baker/` is the whole of what this driver does with a model: trace a catalog
// row for this plane, bind its lanes, walk a `model_compiler::program::Program`
// and drive each statement through `kernels_wgpu::points_dispatch` into a
// `#[claims]` body. Not one line of it names a `wgpu` type, because the thing a
// claim body talks to is `dyn Encode` and the driver is what implements that —
// so the walk, the marks, the bound statement and the resolve pass are all
// checkable with no adapter in the process (`tests/the_walk_is_the_program.rs`
// does exactly that).
//
// `driver-metal` arrives at the same place from the other side: ITS portable
// half exists because a compiler will not accept `objc2` off Apple, and the
// executor landing in it is a happy consequence. Here both halves build
// everywhere and the ungating is the design.
pub mod baker;
pub mod walk;

// Ungated: binding, geometry and reflection are arithmetic over a plan, a
// rectangle and a shader SOURCE, so none of them needs `wgpu`.
//
// `dispatch` and `lowering` STOOD HERE and are deleted whole; see this
// module's header for what each held and what carries the claim now.
//
// `runtime` STOOD HERE TOO and went at R5, with its subject. It held
// `Streams` — value id → the fire table a plan's runtime stream stages in —
// and it read the value ids off a `ForwardPlan`, the LEGACY traced form.
// Nothing had built one since R3, so `Streams::of` had no caller, the
// `Model::runtime` field it fed had no constructor, and the whole channel
// was a translation nobody asked for. `baker/walk.rs`'s `runtime` is what
// answers a stream on this driver now, by NAME and per statement.
pub mod binding;
pub mod facts;
// Not a module of this crate: `driver::names` is the one table, and the copy
// that stood here was byte-for-byte identical to `driver-vulkan`'s 412 lines.
// Re-exported rather than referred to directly so `driver_wgpu::names` keeps
// answering, which is what this crate's own `tests/checkpoint.rs` asks.
pub use driver::names;
pub mod programs;
pub mod reflect;
pub mod rope;
pub mod skip;

// The device half: everything below needs an ADAPTER to answer.
#[cfg(feature = "native")]
pub mod device;
// STILL GATED, AND FOR A DIFFERENT REASON THAN IT WAS. It was here because it
// named `turns::Serving`; that type is gone and this one still cannot leave,
// because it names `frames::{Unlaunched, member_requests}` -> `turns::Step` ->
// `serve::{Fired, Logits, Unfired}` -> `device::Failed`. Nothing in `envelope`
// itself touches a device, and nothing in `frames` or `turns` does either; the
// whole chain is gated by one refusal type that carries what `wgpu` said.
// Ungating it is a `serve`-side question, not this file's.
#[cfg(feature = "native")]
pub mod envelope;
#[cfg(feature = "native")]
pub mod frames;
#[cfg(feature = "native")]
pub mod serve;
// `encode` and `shell` STOOD HERE, both `native`-gated, and both are deleted.
#[cfg(feature = "native")]
pub mod turns;

// UNGATED both: `resources`' `Shape`, `Request` and `Frame` are integer
// arithmetic over page numbers, and `pages` speaks only in those.
pub mod pages;
pub mod resources;

pub use binding::{Allocation, Arena, Bound, Resolve};
/// The tier vocabulary this driver was built against.
///
/// `kernels::sig_in` was re-exported beside it -- "the row a symbol belongs
/// to" -- and so was `KERNELS`, because a caller asking "how wide a fire can
/// this driver take" needed the same table the dispatcher read and a second
/// copy would be a second thing that can drift. There is no row table, no
/// `sig_in` to look one up with, and no dispatcher reading one.
pub use kernels_wgpu::Capability;
pub use reflect::Declared;
