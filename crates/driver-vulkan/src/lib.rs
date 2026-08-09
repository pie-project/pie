//! The Vulkan execution shell: what it takes to actually FIRE the modules
//! `kernels-vulkan` compiles.
//!
//! `kernels-vulkan` is a table and 665 SPIR-V modules. It knows what each
//! entrypoint's operands are, what its push block looks like, and which device
//! features it needs — and it deliberately knows nothing about instances,
//! queues, descriptor pools or command buffers. This crate is that half.
//!
//! # Why this is not a port of `driver-metal`
//!
//! It shares that crate's vocabulary — the same [`kernels::LaunchRule`], the
//! same `Dims` field names — and it should, because a disagreement about which
//! rule a row names would be a real defect rather than a backend difference.
//!
//! But the thing a rule ANSWERS is not the same. Metal's encoder takes a thread
//! count and a threadgroup and sizes the group at dispatch time.
//! `vkCmdDispatch` takes only a count of workgroups, and how wide one is was
//! decided when `glslc` ran. So the driver's arithmetic is a division by a
//! number it does not choose, against a divisor that varies per module, and
//! [`geometry`] is that division, written down with the reason each rounding
//! goes the way it does.
//!
//! # What the split is for, which is not what `driver-metal`'s is for
//!
//! `driver-metal` is split by what a COMPILER will accept: no Linux host can
//! build an `objc2` message send, so its portable half exists to be buildable
//! away from a Mac.
//!
//! Vulkan is a loader, not a platform. Every line here compiles on every host
//! in the tree. The `native` feature gates what needs a GPU to be PRESENT, so
//! the portable half is defined by what can be PROVED without one — and that is
//! a much better deal than the Metal side got, because the device half is
//! testable on the same machine this crate is written on, against a validation
//! layer that turns a silent misuse into a failed test.
//!
//! # What is here, and the order it was built in
//!
//! [`geometry`] was deliberately first. Every kernel in this tree that was
//! wrong after the Vulkan port was wrong in its LAUNCH SHAPE and not in its
//! arithmetic, because an undershot Vulkan grid writes nothing, leaves the
//! buffer's birth zeros in the gap, and returns success from every call
//! involved. Getting the division right, and being able to check it against
//! each module's own declared workgroup, is the part of a Vulkan shell that
//! carries the defects.
//!
//! [`spirv`] reads a module back: its bindings, its push offsets, its declared
//! workgroup. Every claim this crate makes about a module is measured from the
//! module rather than assumed from the row that names it, and where the two
//! are computed separately they are checked against each other.
//!
//! [`lowering`] and [`dispatch`] turn one of a plan's rectangles into one
//! `vkCmdDispatch`: which buffers, at which offsets, with which scalars, over
//! which grid. [`binding`] is where a row's operand ORDER meets a module's
//! binding order, which are not the same order and were not the same order for
//! 2898 of the 3992 rectangles three real texts state.
//!
//! [`device`] is the Vulkan itself, and the only place in the crate with
//! `unsafe` in it. [`Device::run`](device::Device::run) submits one dispatch
//! and waits; [`run_all`](device::Device::run_all) records a whole plan into
//! one command buffer with a barrier between each pair, and the two are checked
//! to agree over a real plan.
//!
//! [`serve`] is a whole fire: plan every rectangle, allocate every scalar
//! block, build every pipeline, record, submit once, wait. Three passes and
//! not one loop, and the module says why each boundary is where it is. It also
//! holds the last mile, [`serve::logits`]: a fire's distributions are a range
//! of its own arena, and reading them needs an element width the lowering
//! states and a driver must not assume.
//!
//! [`turns`] is one fire after another over the same cache: grow, frame,
//! lower, stage, fire, read. Everything below it is per-fire, and the things
//! that can only be wrong ACROSS fires -- a conversation's pages, its
//! positions, and a pipeline cache that must stop growing -- live there
//! because there is nowhere else they could.
//!
//! [`pages`] is the book that outlives a fire: which conversation owns which
//! page of the cache. `Frame::of` refuses two requests in one fire naming the
//! same page, but a conversation spans thousands of fires, and handing page
//! numbers out by hand -- which every test here did before it existed -- gives
//! two users each other's history with nothing to notice.
//!
//! [`resources`] and [`rope`] are the memory and the numbers a plan never
//! mentions, because they belong to a DEPLOYMENT rather than to a model: the
//! paged KV cache, the tables a fire assembles, and the rotary ladder a
//! rescaling config asks for. A text that stated any of them would be right for
//! one server and quietly wrong for the next.
//!
//! # What every module here has been mutated against
//!
//! Every module in this crate has been swept once: a check is deleted, or a
//! value replaced, or two values swapped, and the suite is run. A mutation
//! that survives is a claim nothing was reading.
//!
//! It found more than it should have. `binding` handed the shader two cache
//! strides nothing read back. Three head-shape overrides fired six hundred
//! times and were checked zero. Five `.max(1)` clamps guarded an input
//! nothing sent -- one was dead, one was hiding a disagreement with the
//! shader, one let a rowless fire dispatch nothing and return `Ok`. All three
//! refusals `serve::logits` makes before it reads a byte were made against
//! nothing. And deleting the SPIR-V walk's zero-length refusal does not fail
//! the suite; it HANGS it, which is what a corrupt module would do to a
//! driver.
//!
//! Three survivors are recorded rather than fixed, each with its reason at
//! the line: `dims_of`'s `in_width` has no consumer in any text here,
//! `plan_one`'s empty-grid refusal is unreachable from a real plan and is
//! witnessed by `tests/device.rs` alone, and `Bound::at`'s alignment clamp
//! needs a driver that reports zero alignment.
//!
//! # What is not here
//!
//! Weights, past a store that holds bytes under a name. Nothing loads a
//! checkpoint, and it cannot: `Arg::Weight` carries a name and no WIDTH, so a
//! plan does not say how large a tensor is. Until a loader supplies real
//! sizes, a whole plan exercises its plumbing rather than computing anything.
//!
//! That loader is not missing work in this crate, which is worth stating
//! because it looks like it should be. `tests/checkpoint.rs` measured a real
//! `Qwen/Qwen3-0.6B` snapshot against a real qwen3 plan: ZERO of 704 weight
//! names agree. The plan says `layer.0.down` where the checkpoint says
//! `model.layers.0.mlp.down_proj.weight`, and the plan wants `embed.scales`
//! and `embed.zeros`, which no bfloat16 checkpoint holds under any spelling
//! because they are outputs of quantizing. Loading is therefore a CONVERSION,
//! it already has a home in `model-loader`, and what this crate owes is
//! exactly `Weights::hold` -- a name, some bytes, and no opinion about where
//! they came from.
//!
//! A sampler. [`turns::Serving::step`] drives fire after fire and
//! [`serve::logits`] names where the distribution is, but nothing chooses a
//! token from it. That is deliberate and matches `driver-metal`: sampling is
//! policy -- temperature, top-p, penalties, a grammar -- and a driver that
//! held one would be a driver a server had to argue with. The seam is the
//! `Logits` a step returns.

// The manifest deliberately does not take the workspace lint table, because it
// forbids `unsafe_code` and every `ash` entry point is unsafe. The rest of that
// table is worth having, so it is restated here without that one name -- and
// the portable half keeps its own guarantee a different way, by containing no
// `unsafe` at all, which `tests/pure.rs` asserts by reading the modules this
// file does not gate.
#![deny(missing_docs)]
#![deny(
    clippy::todo,
    clippy::unimplemented,
    clippy::dbg_macro,
    clippy::mem_forget
)]
#![deny(clippy::print_stdout)]

#[cfg(feature = "native")]
pub mod binding;
#[cfg(feature = "native")]
pub mod device;
#[cfg(feature = "native")]
pub mod dispatch;
pub mod geometry;
pub mod lowering;
// Pure Rust, and gated only because it speaks in `resources`' `Shape` and
// `Request` -- which are pure data in a module that also holds device
// handles. Splitting that module to ungate this one would buy nothing today.
#[cfg(feature = "native")]
pub mod pages;
#[cfg(feature = "native")]
pub mod resources;
pub mod rope;
#[cfg(feature = "native")]
pub mod serve;
pub mod spirv;
#[cfg(feature = "native")]
pub mod turns;

#[cfg(feature = "native")]
pub use binding::{Arena, Resolve, Unbindable, bind, resolve};
pub use geometry::{Dims, Local, Module, Rule, Tile, Ungeometric, groups, lanes};
pub use lowering::{Call, Mismatch, Value, pack};
pub use spirv::Declared;
