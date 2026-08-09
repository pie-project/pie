//! The WebGPU execution shell: what it takes to actually FIRE the modules
//! `kernels-wgpu` states.
//!
//! `kernels-wgpu` is a table of 99 rows over 480 entrypoints and a WGSL tree.
//! It knows what each entrypoint's operands are, which storage binding each
//! takes, what its uniform block looks like and which device features it wants
//! — and it deliberately knows nothing about adapters, queues, bind groups or
//! command encoders. This crate is that half.
//!
//! # Why this crate exists beside two shells that already work
//!
//! Not because WebGPU is better. Because of what it COSTS: `wgpu` is pure
//! Rust, `naga` compiles the shaders in-process, and there is no `-sys` crate
//! anywhere in the closure. `tests/pure.rs` is the claim and it runs with
//! `native` ON, which is the version of the claim that matters — a portable
//! half that is pure while the device half needs a toolchain would be a
//! technicality.
//!
//! The three splits are three different lines and it is worth being exact
//! about which is which. `driver-metal` splits on what a COMPILER will accept:
//! no Linux host can build an `objc2` message send. `driver-vulkan` splits on
//! what needs a GPU to be PRESENT: `ash` is a loader, so every line compiles
//! anywhere and only running needs a driver. Here both halves compile anywhere
//! AND the shader front end RUNS anywhere, so [`reflect`] parses the WGSL a
//! fire will dispatch and checks it against the row on a machine with no
//! adapter in it. `native` gates only what has to ask an adapter a question.
//!
//! # The one place a whole design differs, and it is not cosmetic
//!
//! **WebGPU has no push constants.** `wgpu` exposes them as
//! `Features::PUSH_CONSTANTS`, a native-only extension no WebGPU
//! implementation is obliged to offer and the browser backend cannot offer at
//! all, so a driver that depended on them would run on `wgpu` and not on
//! WebGPU — giving up the only thing this backend has that its siblings do
//! not.
//!
//! So `driver-vulkan`'s `Params::Push(Vec<u8>)` has no counterpart. A launch's
//! scalars are the fields of ONE uniform buffer the shell writes and binds at
//! `@group(1) @binding(0)`, and that buffer is a binding like any other. Which
//! means `driver-vulkan`'s SECOND parameter variant — `Params::Block { bytes,
//! at }`, its fallback for a module that read its scalars from a storage
//! buffer instead — stops being a different KIND of answer. Both are "these
//! bytes, in a buffer, at that slot". [`binding::Params`] carries one variant
//! where Vulkan carries two, with the slot as data; see its own docs.
//!
//! That also deletes a whole discovery step. `driver-vulkan` finds a
//! parameter block by SIZE, scanning the module's bindings for one whose bytes
//! match the scalar count, because the shader's ABI is the only place the
//! answer exists. Here `kernels_wgpu::bindings` states it from the row, so
//! [`binding::descriptors`] reads the placement off the table and the
//! reflection is a CHECK on it rather than the source of it.
//!
//! # The second place a whole design differs, and it was found by running it
//!
//! **A dispatch may not bind one buffer both readable and writable.** WebGPU
//! makes each dispatch a *usage scope*, and within one a buffer carries any
//! number of readable usages or exactly one writable usage and never both;
//! `wgpu-core`'s `invalid_resource_state` is the rule as code. Disjoint ranges
//! do not help, because a buffer has no subresources and the tracking is per
//! ALLOCATION.
//!
//! [`binding::Arena`] is ONE buffer holding every activation, so this is the
//! shape of every launch of every real plan: `rms_single_row`'s `x` is an arena
//! range and its `out` is another. Vulkan binds that without comment and Metal
//! has no length to disagree about; here it is refused outright.
//!
//! `device::Device::run_all` answers it by SHADOWING — copying each offending
//! read range into a scratch buffer first — which is correct and costs a copy of
//! every input of every rectangle. `device::Device::check` names the condition
//! instead, for a caller that wants the diagnosis. The real answer is for the
//! arena not to be one allocation, and that belongs to [`binding::Arena`] rather
//! than to the device half; the shadow is what makes a plan run in the meantime,
//! and `serve::Fired::shadowed` reports how many were needed so the cost is
//! visible rather than inferred.
//!
//! # What is here
//!
//! [`geometry`] is the division a rule becomes. `dispatch_workgroups` counts
//! workgroups exactly as `vkCmdDispatch` does, so the arithmetic is
//! `driver-vulkan`'s verbatim and so are the reasons attached to each
//! rounding — with **one exception, which is worth knowing about before
//! reading either file.** WGSL has no 16-bit storage type, so a bf16 tensor
//! crosses as `array<u32>` with two values to a word, and a decode-attention
//! lane owns the PAIR rather than the channel: `sdpa_vector.wgsl` declares
//! `@workgroup_size(PIE_HEAD_DIM / 2)` where the GLSL declares
//! `local_size_x = PIE_HEAD_DIM`. So [`geometry::Rule::SdpaVector`] halves
//! with it. That is not a rounding difference to be absorbed by a tail guard:
//! the shader reads `num_workgroups.x` as its query-head COUNT, so the Vulkan
//! expression would build a grid twice as wide AND tell every lane the model
//! has twice the heads it has.
//! `geometry::tests::a_decode_attention_module_is_half_the_head_it_serves`
//! holds the factor against every such module `naga` can read.
//!
//! Two things are added, and both are WebGPU's:
//! `max_compute_workgroups_per_dimension` has a guaranteed floor of 65535,
//! which a wide enough elementwise launch reaches, so
//! [`geometry::groups_within`] refuses it BY NAME rather than letting `wgpu`
//! reject the encode with a message about a number; and
//! [`geometry::MAX_WORKGROUPS_PER_DIMENSION`] states the floor so the portable
//! half can name it without `wgpu` present.
//!
//! [`reflect`] reads a module back: its workgroup, its bindings, its uniform
//! block, whether it reads its own workgroup count. It is `driver-vulkan`'s
//! `spirv` module with the parser deleted — `naga` is a WGSL front end in
//! Rust, so 993 lines of word-stream walking become a walk over a
//! `naga::Module`. Its own docs say which of the Vulkan questions survive and
//! which have no analog, which is the most useful prose in the crate.
//!
//! [`lowering`] and [`dispatch`] turn one of a plan's rectangles into one
//! `dispatch_workgroups`: which buffers, at which offsets, with which scalars,
//! over which grid. [`binding`] is where a row's operand ORDER meets a
//! module's binding order, which are not the same order and were not the same
//! order for 2898 of the 3992 rectangles three real texts state on the Vulkan
//! side. Nothing on this backend would report it either: every operand is a
//! storage buffer, so a bind group typed by the LAYOUT accepts them shuffled.
//!
//! [`facts`] is the first line the engine's seam reads. Its measured half
//! takes a `u32` rather than a device, which is a small change from
//! `driver-vulkan` with a real consequence: the whole answer is testable with
//! no adapter, where that crate's facts test can only check its two constants.
//!
//! [`names`] is the one thing that stays per-checkpoint: a plan binds
//! `layer.0.down.zeros` and a loader publishes
//! `layers.0.mlp.down_proj.biases`. Measured against a real compiled load
//! plan on the Vulkan side, 704 of 704 names disagreed; through this table,
//! none do. It is backend-agnostic and is carried across unchanged, because a
//! naming convention is not a property of an API.
//!
//! `src/pages.rs` is the book that outlives a fire: which conversation owns
//! which page of the cache. Also backend-agnostic — handing page numbers out
//! by hand gives two users each other's history with nothing to notice, and
//! that is true whatever dispatches the attention. It is written and NOT yet
//! declared, because it speaks in `resources::{Shape, Request}`; see the
//! module list below.
//!
//! [`rope`] is the rotary ladder a rescaling config asks for: numbers a plan
//! never mentions because they belong to a DEPLOYMENT rather than to a model.
//!
//! [`programs`] serves the five registration verbs — programs, channels,
//! instances, and closing the last two — none of which touches a device. Its
//! whole body is the conversion between the ABI's records and the `driver`
//! crate's, which already owns the plane for the other shells. It also RUNS a
//! program, on the host, through `driver`'s reference interpreter; what drives
//! that loop is [`frames::run_programs`] rather than the shell, because the
//! registry is alive from the engine seam's `create` and the shell is not.
//!
//! [`frames`] is the engine's `FrameSubmission` turned into fires: the page
//! CSRs converted to [`resources::Request`]s, the fields this driver does not
//! implement refused by their own names, and the programs of each step fired
//! over the rows it read out. It is the half of `shell::Shell::launch` that
//! needs no adapter, which is why every one of its refusals has a test on a
//! machine with no GPU.
//!
//! # This crate has no `unsafe` at all, and that is a real advantage
//!
//! Both siblings have to drop `unsafe_code = "forbid"` from the workspace lint
//! table, because every `ash` entry point and every `objc2` message send is
//! unsafe by construction. `driver-vulkan` then keeps a weaker guarantee for
//! its portable half only, checked by a test that reads its own module list
//! out of this file and scans for the word.
//!
//! `wgpu` is a safe API. So the forbid below covers the WHOLE crate including
//! the device half, the lint does the job that test was standing in for, and
//! `tests/pure.rs` does not carry the scan — a test that a compiler attribute
//! already enforces is a test that can only be wrong.
//!
//! # What is not here
//!
//! Nothing of the device half — it is `device`, `resources`, `serve`, `shell`
//! and `turns`, all behind `native`, all written against the interface this
//! half publishes rather than the other way round.
//!
//! **The seam is exactly this, and nothing else:**
//!
//! ```ignore
//! // In `device.rs`. `size` is the only thing binding asks of an allocation;
//! // `PartialEq` is a supertrait because two `Bound`s are the same range when
//! // they name the same MEMORY, and a caller holding two handles to one
//! // buffer must find its two identical ranges equal.
//! impl driver_wgpu::binding::Allocation for device::Buffer {
//!     fn size(&self) -> u64;
//! }
//!
//! // In whatever owns the weights and the fire's tables.
//! impl driver_wgpu::binding::Resolve for Store {
//!     type Buffer = device::Buffer;
//!     fn weight(&self, name: &str) -> Option<&device::Buffer>;
//!     fn named(&self, value: model_compiler::trace::ValueId) -> Option<&device::Buffer>;
//!     // Defaulted to `None`; state them only where the text needs them.
//!     fn kv(&self, layer: u16, values: bool) -> Option<&device::Buffer>;
//!     fn number(&self, which: binding::FireNumber) -> Option<u32>;
//!     fn table(&self, which: binding::FireTable) -> Option<&device::Buffer>;
//! }
//! ```
//!
//! Everything else is already generic over that associated type:
//! [`binding::Bound<'a, B>`] is the range a bind-group entry addresses,
//! [`binding::Arena<'a, B>`] is the frame's one buffer, and
//! [`dispatch::plan_one`] answers a `Dispatch<'a, R::Buffer>`. The device half
//! names no type of this crate's beyond those, and this crate names no type of
//! the device half's at all — which is what lets every offset, extent and
//! refusal be tested with [`binding::Placeholder`] and no adapter.
//!
//! Weights, past a store that holds bytes under a name. Nothing in `src/`
//! loads a checkpoint: `Arg::Weight` carries a name and no WIDTH, so a *plan*
//! does not say how large a tensor is, and a driver that depended on a
//! checkpoint FORMAT would be a driver that could not be handed bytes.
//!
//! A sampler. Sampling is policy — temperature, top-p, penalties, a grammar —
//! and a driver that held one would be a driver a server had to argue with.

// The manifest deliberately does not take the workspace lint table, and the
// reason here is the OPPOSITE of the siblings'. They drop it because it forbids
// `unsafe_code` and they cannot keep that; this crate can, since `wgpu` is a
// safe API and nothing in either half needs an unsafe block. Restating the
// table by hand is what puts the forbid next to the argument for it -- a reader
// looking for this crate's guarantees reads `lib.rs`, not a manifest.
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout)]

// Every module below is ungated, and that is this crate's shape rather than an
// oversight. Binding, geometry, lowering, dispatch and reflection are all
// arithmetic over a plan, a row and a shader SOURCE -- `naga` parses WGSL
// anywhere -- so none of them needs `wgpu` to be present and gating them would
// only make the default build unable to answer questions it can answer.
pub mod binding;
pub mod dispatch;
pub mod facts;
pub mod geometry;
pub mod lowering;
pub mod names;
pub mod programs;
pub mod reflect;
pub mod rope;

// The device half. Everything below needs an ADAPTER to answer -- not a
// compiler, which is the line `driver-metal` splits on, and not merely a
// driver, which is `driver-vulkan`'s. `wgpu` is the only thing behind this
// gate.
//
// `device` is the seam the whole portable half was written against: it
// implements `binding::Allocation` for its `Buffer` and that type is the
// `Resolve::Buffer` every caller names.
#[cfg(feature = "native")]
pub mod device;
// The engine's frame, turned into fires. Gated with the rest of the serving
// path because it names `turns::Serving`, though nothing in it touches a
// device -- which is what lets `frames::unserved_in` and `frames::pages_named`
// be tested with no adapter at all.
#[cfg(feature = "native")]
pub mod envelope;
#[cfg(feature = "native")]
pub mod frames;
#[cfg(feature = "native")]
pub mod serve;
#[cfg(feature = "native")]
pub mod shell;
#[cfg(feature = "native")]
pub mod turns;

// UNGATED, both of them, which is a change from `driver-vulkan` and is the
// change this crate's split was drawn for.
//
// `resources` holds six things and only three of them need a device. `Shape` is
// the cache's arithmetic, `Request` is what one conversation contributes to one
// fire, and `Frame` derives every table a fire states from those two -- all of
// it integer arithmetic over page numbers, and all of it exactly the part a
// machine with no GPU should be able to check. `Pool`, `Weights` and `Model`
// hold buffers and are gated INSIDE the module.
//
// That is what lets `pages` be ungated too. It is the book of which
// conversation owns which page -- `Book::fork` returns the moves rather than
// performing them -- and it speaks in `resources::{Shape, Request}` and nothing
// else. `driver-vulkan` gates its identical copy on `native` only because those
// two types live beside device handles there, which is a fact about where they
// were put rather than about what they are.
pub mod pages;
pub mod resources;

pub use binding::{Allocation, Arena, Bound, Resolve, Unbindable, bind, resolve};
pub use geometry::{Dims, Local, Module, Rule, Tile, Ungeometric, groups, groups_within, lanes};
/// The row a symbol belongs to, from the table above.
pub use kernels::sig_in;
/// The kernel table and the tier vocabulary this driver was built against.
///
/// Re-exported because a caller that must ask "how wide a fire can this
/// driver take" needs the same table the dispatcher reads, and a second copy
/// -- an engine depending on `kernels-wgpu` directly -- is a second thing that
/// can drift.
pub use kernels_wgpu::{Capability, KERNELS};
pub use lowering::{Call, Mismatch, Value, pack};
pub use reflect::Declared;
