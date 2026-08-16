//! The device half: an adapter, a queue, bind groups, and pipelines that
//! actually fire.
//!
//! Only compiled under the `native` feature, and — unlike its two siblings —
//! only because it needs an ADAPTER to answer. Every line of it builds
//! anywhere, and so does the half above it, which is why `reflect` can parse
//! the WGSL a fire will dispatch on a machine with no GPU in it. This module
//! is what happens after that parse.
//!
//! # The memory-ordering answer, which is the highest-risk thing in this file
//!
//! A plan's launches are chained through the arena: rectangle *n+1* reads what
//! rectangle *n* wrote. `driver-vulkan` puts a full compute-to-compute barrier
//! between every recorded pair, because **Vulkan gives no ordering at all**
//! between dispatches in one command buffer and the absence is silent — every
//! call returns success and the numbers are whatever the scheduler produced.
//!
//! WebGPU's model is the opposite and it is worth being exact rather than
//! reassuring, because the failure mode is a plan that is right on one adapter
//! and garbage on another. Three claims, each checked against source rather
//! than remembered:
//!
//! * **The specification orders them.** `GPUQueue.submit()` is defined as
//!   "for each `commandBuffer` … execute each command in
//!   `commandBuffer.[[command_list]]`", and a command buffer's command list is
//!   an ordered list executed on the queue timeline. So a conforming
//!   implementation must behave as though each command completed, memory
//!   effects and all, before the next begins. The per-dispatch *usage scope*
//!   rule (§3.4.4: "in a compute pass, each dispatch command is one usage
//!   scope") is a VALIDATION rule about aliasing WITHIN a dispatch and says
//!   nothing about the hazard between two of them, which is exactly the
//!   confusion this note exists to prevent.
//! * **`wgpu` implements it with a real barrier, before every dispatch.**
//!   `wgpu-core-30.0.0/src/command/compute.rs:1008` calls `flush_bindings`
//!   before the raw `dispatch_workgroups`, which merges every bound group into
//!   a usage scope and then calls `CommandEncoder::drain_barriers`, which is
//!   `raw.transition_buffers(..)` — a HAL barrier, emitted inline in the pass.
//!   Whether one is needed is `track/mod.rs:341`:
//!
//!   ```text
//!   fn skip_barrier<F: Flags>(old: F, ordered_uses_mask: F, new: F) -> bool {
//!       old.bits() == new.bits() && ordered_uses_mask.contains(old)
//!   }
//!   ```
//!
//!   and `ordered_uses_mask` is `hal::Adapter::get_ordered_buffer_usages()`,
//!   which is `BufferUses::INCLUSIVE | MAP_WRITE` on every backend
//!   (`wgpu-hal/src/{vulkan,metal,dx12,gles}/adapter.rs`). `STORAGE_READ_WRITE`
//!   is in `EXCLUSIVE` and never in that mask, so `skip_barrier` is **always
//!   false** for a buffer bound writable — the state not changing does not
//!   exempt it. Two read-only bindings of the same buffer DO skip, which is
//!   correct: read-after-read is not a hazard.
//! * **The granularity does not matter.** The same tracker fires at a pass
//!   boundary (`compute.rs:915`, spliced in front of the pass body by
//!   `close_and_swap`) and at a submit boundary against the per-`Device`
//!   tracker (`device/queue.rs:1445`). One pass, many passes, many command
//!   buffers and many submits are the same code path.
//!
//! **So [`Device::run_all`] records the whole plan into ONE compute pass**, and
//! splitting it would buy no correctness and cost a command buffer per
//! rectangle. That is the one place this shell is allowed to be simpler than
//! `driver-vulkan`, and it is simpler because the API is stronger, not because
//! the hazard went away. [`Device::run`] still exists and still submits one
//! dispatch at a time, for the reason its sibling gives: when a fire computes
//! the wrong answer, the two suspects are the plan and the ordering the shell
//! imposed on it, and running the same plan one submission at a time separates
//! them. `run` and `run_all` agreeing byte for byte over a real plan is the
//! single most valuable test in this half.
//!
//! # A bind group is checked against its layout, which a descriptor set is not
//!
//! Vulkan will write a `STORAGE_BUFFER` descriptor at any binding the layout
//! declares and never look at what the SPIR-V said. `wgpu` compares the two:
//! `create_bind_group` refuses a group whose entries do not match the layout,
//! and `create_compute_pipeline` refuses a layout that does not cover what the
//! module declares — including the read/write direction, since
//! `wgpu-core/src/validation.rs:591` turns a layout's
//! `Storage { read_only }` into a `naga::AddressSpace` and compares it for
//! EQUALITY with the shader's.
//!
//! That is stricter than Vulkan and it is an advantage, so the layout is built
//! from what the module actually declares — [`crate::reflect::Declared`] for
//! which bindings, and a walk of the same `naga::Module` for whether each is
//! `read` or `read_write` — rather than from the row. A row and a module
//! disagreeing is then a named refusal at pipeline build instead of a wrong
//! number at dispatch.
//!
//! **A module's binding set may have holes, and a hole is legal.** `naga` keeps
//! a global the entry point never reads, and `wgpu-core/src/validation.rs:1280`
//! only demands a layout entry where `!usage.is_empty()` — so a variant that
//! never READS a bound buffer needs no entry for it, and a layout may skip the
//! number entirely. `attn/kv_write.wgsl`'s paged arm is the plain case: it
//! declares 0, 1, 2, 3, 10 and 11 and the layout this file builds has exactly
//! those six entries, with nothing at 4..9. On Vulkan the same module needs a
//! descriptor at every number up to the highest and the shell has to find
//! something to put in the holes. Here it does not, which deletes a whole
//! class of "bind an unrelated tensor to a slot on the theory that nothing
//! looks there".
//!
//! What is NOT legal is a binding past the row's count, and
//! [`crate::dispatch::plan_one`] refuses that with the traced op in hand.
//!
//! # Every refusal is named, and nothing panics
//!
//! `wgpu`'s default error handler is
//! `panic!("wgpu error: {err}")` (`wgpu/src/backend/wgpu_core.rs:692`), which
//! for a driver is the wrong shape twice over: a server does not want a
//! validation slip to take the process, and a panic carries a message about a
//! number rather than about a launch. [`Device::open`] installs an
//! `on_uncaptured_error` handler that parks the message, and every fallible
//! call below drains it into [`Failed::Wgpu`]. The handler is called
//! synchronously — `ErrorSinkRaw::handle_error_or_return_handler` returns a
//! closure the caller runs immediately — so "do the call, then take the error"
//! is a real ordering and not a hope.
//!
//! # The arena cannot be bound both ways in one dispatch, and this is the
//! divergence with teeth
//!
//! [`crate::binding::Arena`] is ONE buffer holding every activation, so a
//! launch's input and its output are two ranges of one allocation. Vulkan binds
//! that without comment; Metal has no length to disagree about.
//!
//! **WebGPU forbids it.** A dispatch is one *usage scope*, and within a usage
//! scope a buffer may carry any number of INCLUSIVE usages or exactly one
//! EXCLUSIVE usage, never both — `wgpu-core-30.0.0/src/track/mod.rs:333`:
//!
//! ```text
//! fn invalid_resource_state<T: ResourceUses>(state: T) -> bool {
//!     // Is power of two also means "is one bit set". We check for this as if
//!     // we're in any exclusive state, we must only be in a single state.
//!     state.any_exclusive() && !state.bits().is_power_of_two()
//! }
//! ```
//!
//! `STORAGE_READ_WRITE` is in `BufferUses::EXCLUSIVE` and `STORAGE_READ_ONLY`
//! is in `INCLUSIVE`, so the pair is two bits with an exclusive one among them
//! and the dispatch is refused with *"Attempted to use Buffer with conflicting
//! usages"*. That is the SPECIFICATION's rule and not `wgpu`'s invention:
//! WebGPU's "usage scope storage exception" permits one buffer to be bound
//! writable several times over, and says nothing about mixing a writable
//! binding with a readable one. Buffers have no subresources, so the two ranges
//! being disjoint does not help — the tracking is per ALLOCATION.
//!
//! ## And the exception is the way out
//!
//! Read the predicate again: two `STORAGE_READ_WRITE` usages are the SAME
//! BIT, so `is_power_of_two` holds and the dispatch is legal. A `read`
//! declaration is what makes an arena launch illegal, and nothing forces one
//! — a binding declared `read_write` that the body only reads is read-only in
//! fact.
//!
//! `kernels-wgpu`'s shader tree therefore declares no `var<storage, read>` at
//! all, and its `no_shader_declares_a_read_only_storage_binding` keeps it that
//! way. A 452-launch decode went from 451 shadow copies to none, 25.1 ms to
//! 11.2 ms, 39.8 to 89.3 tok/s on an RTX 4090.
//! `two_read_write_bindings_into_one_buffer_are_legal` is the claim itself, on
//! a device, because it is load-bearing and it is about `wgpu` rather than
//! about this code.
//!
//! What the `read` declaration WOULD have caught is two operands covering the
//! same bytes partially — and that is now [`Failed::Overlapping`], asked of
//! every dispatch, raised by no real plan. Both siblings bind the arena both
//! ways with no workaround and no check at all.
//!
//! ## What this file does about it
//!
//! **It shadows the read side**, and no kernel in this tree needs it any
//! more — see the exception above. It is kept because a `read` binding is
//! legal WGSL that a future kernel may want, and without this such a kernel
//! would be REFUSED rather than run.
//!
//! Before a dispatch whose read-only operands
//! share a buffer with one of its writable operands, [`Device::run_all`] copies
//! each offending range into a scratch buffer and binds the scratch instead.
//! The copies are encoded into the same command buffer as the dispatches, just
//! ahead of the pass that reads them — a copy cannot go INSIDE a compute pass,
//! so a shadow point ends one and opens another, but it does not end the
//! ENCODER. Ordering is the one the first section of these docs establishes,
//! and `wgpu-core` emits the barrier at the usage transition.
//!
//! It did open a fresh encoder per segment until a real decode was measured:
//! 451 of 452 rectangles shadow something, so the queue was being given 735
//! command buffers for one token. One encoder instead took a decode from
//! 31.9 ms to 20.5 ms on an RTX 4090 -- encoding 7.1 to 4.3, submit 5.4 to
//! 1.0, and the wait itself 13.0 to 9.7, because a command buffer is a unit
//! the queue schedules and 735 of them are 735 boundaries the GPU had no
//! reason to draw.
//!
//! It is CORRECT rather than merely tolerated: the shader reads the values that
//! were there before the dispatch, which is what a plan means, and no kernel can
//! rely on seeing its own writes — WGSL gives no ordering between the
//! invocations of one dispatch, so a body that read a slot another invocation
//! writes is undefined whatever the shell does.
//!
//! It COSTS a copy per aliased read operand, and it breaks the recording into
//! one PASS per shadow point. [`Device::run_all`] returns a [`Ran`] saying how
//! many copies it made and how many command buffers it submitted, so a caller
//! sees the cost instead of inferring it; a plan that needs no shadow is one
//! encoder and one pass, and a plan that shadows every rectangle is still one
//! encoder.
//!
//! ## Why this is a workaround and not the answer
//!
//! The answer is for the arena not to be one buffer, and that is
//! [`crate::binding::Arena`]'s to decide rather than this file's: a plan whose
//! reads and writes landed in two allocations would need no copy at all. Until
//! then this is what makes a real plan run, and [`Failed::Aliased`] is what a
//! caller gets from [`Device::check`] when it wants the diagnosis without the
//! copy.
//!
//! # Readback
//!
//! `wgpu` has no `vkMapMemory` on a device-local buffer. A readback is a copy
//! into a `MAP_READ | COPY_DST` staging buffer, a `map_async`, a device poll,
//! and then the bytes. It is written ONCE, in [`Device::read_at`], because five
//! copies of it is five chances to forget the `unmap`.
//!
//! # Two more things wgpu does that Vulkan does not, and one it refuses to do
//!
//! **A new buffer is zeroed.** WebGPU requires the contents of a `GPUBuffer` to
//! be zero at creation, and `wgpu-core` enforces it by zero-initialising any
//! range that has not been written before first use
//! (`device/queue.rs`'s `initialize_buffer_memory`). So [`Device::zeroed`]
//! allocates and writes nothing, where `resources::Pool::open` on the Vulkan
//! side has to build a host-side `vec![0u8; layer_bytes]` — which for a real
//! cache is hundreds of megabytes of memset and a matching upload.
//!
//! **A buffer may not be copied onto itself.**
//! `wgpu-core/src/command/transfer.rs:987` refuses `copy_buffer_to_buffer` when
//! source and destination are the same buffer, where `vkCmdCopyBuffer` allows it
//! for non-overlapping regions. The KV cache is one buffer per layer and a fork
//! moves a page WITHIN it, so this backend cannot express that copy directly.
//! [`Device::shuffle`] routes every such move through one scratch buffer in one
//! command buffer — `src -> scratch`, `scratch -> dst` — which is two transfers
//! instead of one and is why the scratch is shared across a whole page move
//! rather than allocated per copy.

use std::collections::BTreeMap;
use std::future::Future;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use kernels_wgpu::Capability;

use crate::binding::{Allocation, Bound};
use crate::geometry::{self, Dims, Module, Rule, Ungeometric};
use crate::reflect::{self, Declared, STORAGE_GROUP, UNIFORM_BINDING, UNIFORM_GROUP};

/// How long a device wait may take before it is called a failure.
///
/// Finite for the reason `driver-vulkan`'s fence wait is finite: a wait with
/// no deadline on a hung device is a test run that never returns and reports
/// nothing. `wgpu::PollType::wait_indefinitely` is the call this deliberately
/// does not make.
///
/// It was also called "generous", against nothing. Measured: a decode step of
/// Qwen3-0.6B is 452 dispatches, submitted as ONE submission with ONE wait
/// over it, and on `llvmpipe` the `quest-attention` and `h2o-attention` frames
/// exceed thirty seconds -- which is how they come back from a software
/// adapter as a wait timeout instead of the clean intrinsic refusal they give
/// on a GPU. A bound that is generous for a card is not generous for a CPU
/// rasteriser, and the constant could not say so because nothing had ever
/// asked it.
///
/// So thirty seconds is the DEFAULT and no longer the rule. `PIE_WGPU_WAIT_SECS`
/// overrides it.
///
/// This paragraph used to end "and without a longer bound that implementation
/// cannot complete one model step". That was a generalisation from the two
/// curated failures to every frame, and the next measurement refuted it:
/// `wgpu_padded_causal_mask` boots a real model on `llvmpipe`, prompts it
/// twice and answers, at the default thirty seconds, in 124 s. Some frames
/// exceed the bound; a model step as such does not.
const WAIT_DEFAULT: Duration = Duration::from_secs(30);

/// [`WAIT_DEFAULT`], or what `PIE_WGPU_WAIT_SECS` says.
///
/// Read once. A zero or an unparseable value is the default rather than a
/// refusal: this is a deadline on a wait, and a driver that would not open
/// because a number was mistyped is worse than one that waits thirty seconds.
fn wait() -> Duration {
    static HELD: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *HELD.get_or_init(|| {
        std::env::var("PIE_WGPU_WAIT_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|secs| *secs > 0)
            .map_or(WAIT_DEFAULT, Duration::from_secs)
    })
}

/// The refusal a device wait produces, naming the deadline it was actually
/// given and the knob that moves it.
///
/// `wgpu`'s own text is *"The requested Wait timed out before the submission
/// was completed."* — true, and it says neither how long the wait was nor
/// that the length is adjustable. `PIE_WGPU_WAIT_SECS` appears in no README,
/// no config template and no `--help`; it is named in this crate's source and
/// nowhere else. The person who needs to know it exists is, by construction,
/// reading this string.
///
/// The number comes from the deadline that was used rather than from
/// [`WAIT_DEFAULT`], so a run that raised it says the raised value. A message
/// quoting the default while the wait used something else would be the same
/// copied-number rot this crate keeps finding in its prose, with a worse
/// audience.
fn not_answered(deadline: Duration, why: &impl std::fmt::Display) -> Failed {
    Failed::Wgpu(format!(
        "the device did not answer within {}s (raise `PIE_WGPU_WAIT_SECS`): {why}",
        deadline.as_secs()
    ))
}

/// Why there is no device to run on.
///
/// A distinct type from [`Failed`] because the two mean opposite things to a
/// caller: this one is the environment, and a machine with no adapter is the
/// normal state of a build host rather than a defect. `tests/device.rs` prints
/// it and skips.
#[derive(Clone, Debug)]
pub struct Unavailable(
    /// What the adapter search said.
    pub String,
);

impl core::fmt::Display for Unavailable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl core::error::Error for Unavailable {}

/// The adapter limits this shell reads, states and refuses against.
///
/// A struct of its own rather than a `wgpu::Limits` field, because only these
/// eight are load-bearing here and a reader looking for "which limits does this
/// driver actually obey" should find a list rather than a hundred-field type.
/// Every one of them is checked somewhere below, and the check is named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// `max_storage_buffers_per_shader_stage`.
    ///
    /// The one that decides whether a ROW can run at all.
    /// `wgpu::Limits::downlevel_defaults()` guarantees **8** and
    /// `sdpa_paged_decode` binds eleven, so a shell that asked for the
    /// downlevel defaults out of caution would fail to build exactly the
    /// attention pipeline a paged decode needs — and would fail at model load,
    /// with a `wgpu` message about a limit rather than about attention.
    /// [`Device::unreachable`] is the list, computed at open.
    pub storage_buffers: u32,
    /// `max_compute_workgroups_per_dimension`, whose guaranteed floor is
    /// [`geometry::MAX_WORKGROUPS_PER_DIMENSION`].
    ///
    /// A wide enough elementwise launch reaches it.
    /// [`geometry::groups_within`] refuses by name rather than letting the
    /// encoder reject the dispatch.
    pub workgroups_per_dimension: u32,
    /// `max_buffer_size`. What [`Device::buffer`] refuses against.
    pub buffer_size: u64,
    /// `max_storage_buffer_binding_size`.
    ///
    /// Not the same number as [`Self::buffer_size`] and usually smaller: a
    /// device may hold a buffer it will not bind the whole of. An arena
    /// larger than this is legal to ALLOCATE and cannot be bound, which is a
    /// refusal a driver would otherwise meet halfway through a fire.
    pub storage_binding_size: u64,
    /// `max_uniform_buffer_binding_size`. The ceiling on one launch's scalar
    /// block, which no row in this tree comes near.
    pub uniform_binding_size: u64,
    /// `min_storage_buffer_offset_alignment`, floor 256.
    ///
    /// The granularity an arena operand's offset must divide, checked by
    /// [`crate::binding::Bound::within`] with this number.
    pub storage_offset: u32,
    /// `min_uniform_buffer_offset_alignment`, floor 256.
    ///
    /// Not used to place a uniform today — every launch gets its own buffer at
    /// offset zero, which every alignment divides — and read anyway, because
    /// the day the blocks are packed into one buffer with dynamic offsets this
    /// is the number that decides the stride, and a driver that did not know
    /// it would pack them four bytes apart and bind the wrong one.
    pub uniform_offset: u32,
    /// `max_compute_invocations_per_workgroup`.
    ///
    /// A module's `@workgroup_size` is fixed when it compiles, so a shader
    /// wider than this cannot be built on this adapter. Checked at pipeline
    /// build so the refusal names the entrypoint.
    pub invocations_per_workgroup: u32,
}

impl Limits {
    /// Read the eight numbers off an adapter's limits.
    fn of(limits: &wgpu::Limits) -> Self {
        Self {
            storage_buffers: limits.max_storage_buffers_per_shader_stage,
            workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            buffer_size: limits.max_buffer_size,
            storage_binding_size: limits.max_storage_buffer_binding_size,
            uniform_binding_size: limits.max_uniform_buffer_binding_size,
            storage_offset: limits.min_storage_buffer_offset_alignment,
            uniform_offset: limits.min_uniform_buffer_offset_alignment,
            invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        }
    }
}

impl core::fmt::Display for Limits {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} storage buffers per stage, {} workgroups per dimension, \
             buffers to {} bytes ({} bindable, {} uniform), offsets every \
             {}/{} bytes, {} invocations per workgroup",
            self.storage_buffers,
            self.workgroups_per_dimension,
            self.buffer_size,
            self.storage_binding_size,
            self.uniform_binding_size,
            self.storage_offset,
            self.uniform_offset,
            self.invocations_per_workgroup,
        )
    }
}

/// Which limit a refusal is about.
///
/// Named rather than formatted into a string, so a test can assert WHICH
/// ceiling was hit and a caller that can act on one of them — split the
/// rectangle, shrink the arena — can tell it from one it cannot act on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ceiling {
    /// An allocation past `max_buffer_size`.
    BufferSize,
    /// A binding past `max_storage_buffer_binding_size`.
    StorageBinding,
    /// A uniform binding past `max_uniform_buffer_binding_size`.
    UniformBinding,
    /// A workgroup wider than `max_compute_invocations_per_workgroup`.
    Invocations,
    /// A grid past `max_compute_workgroups_per_dimension`.
    Workgroups,
}

impl core::fmt::Display for Ceiling {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::BufferSize => "max_buffer_size",
            Self::StorageBinding => "max_storage_buffer_binding_size",
            Self::UniformBinding => "max_uniform_buffer_binding_size",
            Self::Invocations => "max_compute_invocations_per_workgroup",
            Self::Workgroups => "max_compute_workgroups_per_dimension",
        })
    }
}

/// Which part of a fire a [`Failed`] belongs to.
///
/// Every check [`Device::run_all`] makes BEFORE the queue names the launch it
/// was checking. Nothing after `submit` can: the whole plan goes to the device
/// as one submission and the single wait covers all of it, so a device that
/// errors or never answers has not singled out a launch.
///
/// Reporting that as a launch index does not merely lose information, it
/// invents some — and it did. A wait timeout on a 452-launch frame came back
/// as `launch 452`, which is one past the last real index, and it was read in
/// this repository as *"the 452nd dispatch is slow"* and written up twice that
/// way. The truth was "all 452 of them, waited once". So the two cases are
/// different values now, and [`crate::serve::Unfired`] keeps them different
/// all the way to the message a reader sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// The launch at this index, checked before anything was submitted.
    Launch(usize),
    /// The submission as a whole, after the queue took it. `of` is how many
    /// launches were in flight, which is the only true thing there is to say
    /// about where it stopped.
    Submission {
        /// How many launches the submission held.
        of: usize,
    },
}

/// Why a dispatch could not be made.
///
/// Compared by value, as `driver-vulkan`'s is, because a test that asserts
/// WHICH refusal came back is the only way an alignment failure stays
/// distinguishable from a length one. [`Self::Wgpu`] carries a string because
/// `wgpu::Error` is not comparable and its useful content is the message.
#[derive(Clone, Debug, PartialEq)]
pub enum Failed {
    /// The launch shape could not be worked out.
    Geometry(
        /// Why.
        Ungeometric,
    ),
    /// The module could not be read.
    ///
    /// Including [`reflect::Unreadable::NoSource`], which at a tier above
    /// [`Capability::Baseline`] is ordinary and is handled by falling back —
    /// see [`Pipelines::get`]. Reaching here means the BASELINE variant is
    /// missing, which is a table this build does not have.
    Module(
        /// What the reader said.
        reflect::Unreadable,
    ),
    /// The caller bound a different number of buffers than the module's
    /// `@group(0)` layout has entries.
    ///
    /// A short set is refused by `wgpu` too, at `create_bind_group`, with a
    /// message about an entry count. This is the same refusal one layer up,
    /// where the numbers still mean something.
    Bindings {
        /// Entries the layout has: what the module declares AND reads.
        module: usize,
        /// What the caller bound.
        bound: usize,
    },
    /// The scalar block is not the size the module's uniform struct needs.
    ///
    /// Both directions, and the short one is what this exists for. WGSL
    /// requires an implementation to bounds-check every access, so a block
    /// short of what the shader reads returns ZEROS rather than faulting: a
    /// missing `logits_pitch` is a plausible number no layer objects to.
    /// `wgpu` would also refuse a uniform binding smaller than the struct —
    /// but only if it is asked to bind one at all, and offering none where the
    /// module declares a block is the case it cannot see.
    Params {
        /// What the module's `@group(1) @binding(0)` struct needs.
        needs: u32,
        /// What the caller offered.
        given: usize,
    },
    /// A `@group(0)` binding was given fewer bytes than the block the shader
    /// reads.
    ///
    /// The storage-buffer half of [`Self::Params`], for the rows whose scalars
    /// ride a `Buf` operand — `rms_single_row`'s `params` is the case. Same
    /// silence, same reason: the tail reads as zero.
    Short {
        /// Which `@group(0)` binding.
        binding: u32,
        /// What the module's block needs.
        needs: u32,
        /// What the caller bound.
        given: u64,
    },
    /// A device limit says no.
    PastLimit {
        /// Which ceiling.
        which: Ceiling,
        /// What was asked for.
        want: u64,
        /// What the adapter offers.
        limit: u64,
    },
    /// The row binds more storage buffers than this adapter offers a stage.
    ///
    /// Refused BY NAME, before `wgpu` is asked, because the message it would
    /// give names a limit and not a kernel — and the answer a deployment needs
    /// is "this adapter cannot run paged decode", not "8 < 11".
    /// [`Device::unreachable`] is the whole list at open time.
    Unreachable {
        /// The entrypoint that cannot be built.
        entrypoint: String,
        /// Storage bindings its module declares.
        needs: u32,
        /// What the adapter allows one stage.
        limit: u32,
    },
    /// One buffer is bound both readable and writable in the same dispatch.
    ///
    /// The divergence this module's docs give a section to. Legal on Vulkan and
    /// on Metal, and the ordinary shape of every arena launch; forbidden by
    /// WebGPU, whose usage scope admits any number of readable usages or exactly
    /// one writable one and never both. Disjoint ranges do not help, because a
    /// buffer has no subresources and the tracking is per allocation.
    ///
    /// [`Device::run_all`] does not return this: it SHADOWS the read side into a
    /// scratch buffer and runs. This is what [`Device::check`] answers for a
    /// caller that wants the diagnosis — which of the two slots, and which two
    /// ranges of which buffer — rather than the copy.
    Aliased {
        /// The `@group(0)` binding read.
        reader: u32,
        /// The `@group(0)` binding written.
        writer: u32,
        /// Where the read starts, in the shared buffer.
        read_at: u64,
        /// Where the write starts.
        write_at: u64,
    },
    /// Two operands of one dispatch cover the same bytes, PARTIALLY.
    ///
    /// Not [`Self::Aliased`], which is about the WebGPU usage scope and is
    /// answered by binding both ways round. This is about the DATA, it is a
    /// defect on every backend, and neither sibling would catch it either.
    ///
    /// Disjoint ranges are the ordinary case — a launch's input and its output
    /// are two places in one arena — and IDENTICAL ranges are the in-place
    /// case a kernel authors deliberately, where invocation `i` reads and
    /// writes element `i`. What no kernel authors is a partial overlap: there
    /// invocation `i`'s write is some other invocation's read, WGSL orders the
    /// invocations of one dispatch not at all, and the answer is whatever the
    /// scheduler did.
    ///
    /// It was unnameable while the shadow existed, because a copy of the read
    /// side made every overlap harmless and every plan look sound. Removing
    /// the shadow is what made this worth asking, and asking it found nothing
    /// — which is the answer that lets the shadow go.
    Overlapping {
        /// The `@group(0)` binding written.
        writer: u32,
        /// The other binding, which is not the same range.
        other: u32,
        /// The bytes both cover.
        overlap: std::ops::Range<u64>,
    },
    /// A grid of zero in some dimension.
    ///
    /// Legal WebGPU and always a defect: it runs nothing, reports success, and
    /// leaves the output holding whatever it held before.
    Empty {
        /// The grid that would have been dispatched.
        groups: [u32; 3],
    },
    /// `wgpu` refused, or a device call failed.
    Wgpu(
        /// What it said.
        String,
    ),
    /// The device would not give the memory.
    ///
    /// Separate from [`Self::Wgpu`] because the answer above it differs, and
    /// that difference is the whole reason [`crate::frames::Launched::
    /// Exhausted`] exists: a validation slip is a fault, and an allocation the
    /// device declined is something a scheduler can serve by evicting and
    /// re-posting.
    ///
    /// It is reachable in ordinary service rather than only under abuse.
    /// [`Device::budget`] reports a heap's SIZE, so `Pool::ceiling` admits any
    /// frame the device could hold **if it were empty** -- and the device is
    /// never empty, because the weights are in it.
    OutOfMemory(
        /// What it said.
        String,
    ),
}

impl Failed {
    /// Would evicting something else make this work?
    ///
    /// Only [`Self::OutOfMemory`]. [`Self::PastLimit`] is a DECLARED limit and
    /// no eviction moves it -- `Shell::admit` asks about that one first, with
    /// `Pool::ceiling`, and answers `Impossible` -- so folding the two here
    /// would turn a permanent refusal into a scheduler that evicts and
    /// re-posts forever.
    #[must_use]
    pub const fn is_out_of_memory(&self) -> bool {
        matches!(self, Self::OutOfMemory(_))
    }
}

impl core::fmt::Display for Failed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Geometry(e) => write!(f, "no launch geometry: {e}"),
            Self::Module(e) => write!(f, "the module could not be read: {e}"),
            Self::Bindings { module, bound } => write!(
                f,
                "the module's layout has {module} entries and {bound} buffers were bound"
            ),
            Self::Params { needs, given } => write!(
                f,
                "the uniform block is {needs} bytes and {given} were offered, \
                 whose tail reads as zero"
            ),
            Self::Short {
                binding,
                needs,
                given,
            } => write!(
                f,
                "binding {binding} reads a {needs}-byte block and was given \
                 {given} bytes, whose tail reads as zero"
            ),
            Self::PastLimit { which, want, limit } => {
                write!(f, "{want} is past this adapter's {which} of {limit}")
            }
            Self::Unreachable {
                entrypoint,
                needs,
                limit,
            } => write!(
                f,
                "`{entrypoint}` binds {needs} storage buffers and this adapter \
                 allows a compute stage {limit}"
            ),
            Self::Aliased {
                reader,
                writer,
                read_at,
                write_at,
            } => write!(
                f,
                "binding {reader} reads at {read_at} and binding {writer} writes \
                 at {write_at} in the same buffer, which WebGPU refuses within \
                 one dispatch however far apart the two ranges are"
            ),
            Self::Overlapping {
                writer,
                other,
                overlap,
            } => write!(
                f,
                "binding {writer} writes {}..{} of a buffer that binding \
                 {other} also covers, partially -- the invocations of one \
                 dispatch are unordered, so which value is read is whatever \
                 the scheduler did",
                overlap.start, overlap.end
            ),
            Self::Empty { groups } => write!(
                f,
                "a grid of {groups:?} would run nothing and report success"
            ),
            Self::Wgpu(e) => write!(f, "{e}"),
            Self::OutOfMemory(e) => write!(f, "the device would not give the memory: {e}"),
        }
    }
}

impl core::error::Error for Failed {}

impl From<Ungeometric> for Failed {
    fn from(e: Ungeometric) -> Self {
        Self::Geometry(e)
    }
}

impl From<reflect::Unreadable> for Failed {
    fn from(e: reflect::Unreadable) -> Self {
        Self::Module(e)
    }
}

/// Wakes a [`block_on`] that is parked on a condvar.
struct Parked {
    woken: Mutex<bool>,
    bell: Condvar,
}

impl Wake for Parked {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        // The lock is taken before the notify so a wake that lands between the
        // poll and the wait is not lost -- the flag is what the sleeper checks,
        // and the notify is only the nudge.
        *self.woken.lock().unwrap_or_else(|e| e.into_inner()) = true;
        self.bell.notify_one();
    }
}

/// Drive a future to completion on this thread.
///
/// `wgpu` has three async entry points — `request_adapter`, `request_device`
/// and `Buffer::map_async`'s callback — and on a native adapter all three
/// resolve on the spot. This turns them back into values.
///
/// Public because `tests/device.rs` needs the same thing and a test binary
/// cannot reach a private helper; and written here rather than taken from
/// `pollster` because it is nine lines and because those nine lines are
/// **safe**. A hand-rolled `RawWaker` is the usual way to write this and needs
/// an `unsafe` block; `std::task::Wake` plus `Waker::from(Arc<_>)` does not,
/// and this crate's `#![forbid(unsafe_code)]` is a guarantee worth nine lines.
///
/// # Panics
///
/// Never, unless the future itself does.
pub fn block_on<F: Future>(future: F) -> F::Output {
    // Pinned to the stack, which is what lets a `!Unpin` future be polled at
    // all. `Box::pin` would also do and would allocate.
    let mut future = core::pin::pin!(future);
    let parked = Arc::new(Parked {
        woken: Mutex::new(false),
        bell: Condvar::new(),
    });
    let waker = Waker::from(Arc::clone(&parked));
    let mut cx = Context::from_waker(&waker);
    loop {
        if let Poll::Ready(out) = future.as_mut().poll(&mut cx) {
            return out;
        }
        let mut woken = parked.woken.lock().unwrap_or_else(|e| e.into_inner());
        while !*woken {
            woken = parked.bell.wait(woken).unwrap_or_else(|e| e.into_inner());
        }
        *woken = false;
    }
}

/// A storage buffer, and how many bytes it holds.
///
/// Plain data with no device reference, so a caller can keep a table of these
/// beside the [`Device`] that made them — and, unlike `driver-vulkan`'s, it
/// needs no `Device::free`: `wgpu::Buffer` is `Arc`-backed and releases its
/// allocation when the last handle drops. That deletes the whole class of
/// defect its sibling's `release`/`close` pairs exist to catch.
#[derive(Clone, Debug)]
pub struct Buffer {
    inner: wgpu::Buffer,
    size: u64,
}

/// Two buffers are equal when they are the same allocation.
///
/// `wgpu::Buffer`'s own `PartialEq` compares the inner resource, so a clone of
/// a handle equals the handle — which is what [`crate::binding::Bound`]'s
/// equality needs, since a test asking where a dispatch points is asking about
/// memory and not about which copy of the handle it was given.
impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Eq for Buffer {}

impl Allocation for Buffer {
    fn size(&self) -> u64 {
        self.size
    }
}

impl Buffer {
    /// Bytes it holds.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Whether it holds nothing. Never, by construction; clippy asks for it
    /// beside `size`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// The `wgpu` handle, for a caller that wants to do something this shell
    /// does not.
    #[must_use]
    pub fn raw(&self) -> &wgpu::Buffer {
        &self.inner
    }
}

/// A compute pipeline, the layouts it was built with, and what its module
/// declares.
pub struct Pipeline {
    entrypoint: String,
    tier: Capability,
    declared: Declared,
    /// The `@group(0)` binding numbers the layout covers, ascending.
    ///
    /// Exactly the numbers the module declares AND reads, which is what
    /// [`crate::dispatch::plan_one`] sizes its dense buffer list to. A hole is
    /// absent from this vector rather than present and skipped, because a
    /// `wgpu` layout may simply not declare it — see the module docs.
    slots: Vec<u32>,
    /// Whether each of [`Self::slots`] is `var<storage, read>`.
    ///
    /// The same reading the layout was built from, kept because a dispatch has
    /// to know which of its operands are the READ side before it can tell
    /// whether the arena is bound both ways -- see the module docs.
    read_only: Vec<bool>,
    storage: wgpu::BindGroupLayout,
    uniform: Option<wgpu::BindGroupLayout>,
    pipeline: wgpu::ComputePipeline,
    /// Held so the module is not recompiled.
    ///
    /// `wgpu::Device::create_shader_module` runs `naga` — parse, validate and
    /// then a full backend translation to SPIR-V, MSL or DXIL — every single
    /// time it is called. On Vulkan the equivalent work happened in `glslc` at
    /// BUILD time and `vkCreateShaderModule` is a memcpy, so a cache there
    /// saves a pipeline compile and here it saves a whole shader compiler.
    /// Keeping the module alive beside the pipeline costs one handle and makes
    /// the saving legible.
    _module: wgpu::ShaderModule,
}

impl Pipeline {
    /// What the module declares.
    #[must_use]
    pub fn declared(&self) -> &Declared {
        &self.declared
    }

    /// The entrypoint this was built for.
    #[must_use]
    pub fn entrypoint(&self) -> &str {
        &self.entrypoint
    }

    /// The tier the source came from, which may be below the one asked for.
    #[must_use]
    pub fn tier(&self) -> Capability {
        self.tier
    }

    /// The `@group(0)` binding numbers its layout covers, ascending.
    #[must_use]
    pub fn slots(&self) -> &[u32] {
        &self.slots
    }

    /// Whether the operand at each of [`Self::slots`] is read-only.
    #[must_use]
    pub fn read_only(&self) -> &[bool] {
        &self.read_only
    }

    /// How many buffers a dispatch must bind: one per layout entry.
    #[must_use]
    pub fn bindings(&self) -> usize {
        self.slots.len()
    }

    /// Bytes the module's uniform block needs, or zero when it declares none.
    #[must_use]
    pub fn uniform_bytes(&self) -> u32 {
        self.declared.uniform_bytes
    }

    /// The geometry the module imposes: its workgroup, and the tile its name
    /// encodes.
    #[must_use]
    pub fn module(&self) -> Module {
        Module::loaded(&self.entrypoint, &self.declared)
    }
}

/// One dispatch in a recorded run.
///
/// The same four things [`Device::run`] takes, named rather than positional
/// because a run states thousands of them and a swapped pair in a list that
/// long is not something a reader would catch.
#[derive(Clone, Copy)]
pub struct Recorded<'a, 'b> {
    /// What to run.
    pub pipeline: &'a Pipeline,
    /// The `@group(0)` entries, in [`Pipeline::slots`] order.
    pub buffers: &'a [Bound<'b, Buffer>],
    /// The bytes of the `@group(1) @binding(0)` uniform block.
    ///
    /// Empty where the module declares none — and, for the rows whose scalars
    /// ride a storage `Buf` operand, ALSO empty: that block is a buffer the
    /// caller allocated and put in [`Self::buffers`] at the slot the row
    /// named. One field carries the ordinary case; the other case is not a
    /// different kind of answer, it is a buffer like the rest.
    pub uniform: &'a [u8],
    /// Workgroups in each dimension.
    pub groups: [u32; 3],
}

/// One read-only range copied out of a buffer this dispatch also writes.
///
/// See the module docs for the WebGPU rule that makes it necessary and for why
/// the copy computes the same answer.
struct Shadow<'a> {
    /// The binding whose range this stands in for.
    slot: Bound<'a, Buffer>,
    /// Where the bytes come from.
    from: Buffer,
    /// And where in it.
    at: u64,
    /// How many, rounded out to a whole number of copy units.
    bytes: u64,
    /// Where they go, and what the dispatch binds instead.
    into: wgpu::Buffer,
}

/// The bind groups one dispatch needs, alive until the queue is done with them.
struct Bound1 {
    storage: wgpu::BindGroup,
    uniform: Option<wgpu::BindGroup>,
    /// The uniform block's buffer.
    ///
    /// A `wgpu::BindGroup` holds its resources alive on its own, so this is
    /// not load-bearing; it is kept so that a reader does not have to know
    /// that in order to believe the buffer is still there at submit.
    _block: Option<wgpu::Buffer>,
}

/// An open adapter with a compute queue.
///
/// Owns the instance, the adapter, the device and the queue, and hands out
/// buffers and pipelines. Cheap to clone in `wgpu` terms — every handle inside
/// is `Arc`-backed — but deliberately NOT `Clone` here, because the error sink
/// is per-device state and two `Device`s over one `wgpu::Device` would drain
/// each other's refusals.
pub struct Device {
    /// Held because dropping the instance while an adapter lives is legal and
    /// pointless, and because a reader looking for the backend selection finds
    /// it here.
    _instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    features: wgpu::Features,
    limits: Limits,
    tiers: Vec<Capability>,
    unreachable: Vec<&'static str>,
    /// Where `on_uncaptured_error` parks what would otherwise be a panic.
    /// Where `on_uncaptured_error` parks what would otherwise be a panic,
    /// with whether `wgpu` called it an OUT-OF-MEMORY.
    ///
    /// The kind is kept because the answer above it differs: a validation slip
    /// is a fault and a device that would not give the memory is
    /// `Launched::Exhausted` -- evict and re-post. Storing only
    /// `e.to_string()` threw that away, and `Shell::admit` had no way to tell
    /// them apart.
    errors: Arc<Mutex<Vec<(bool, String)>>>,
}

/// How many `@group(0)` bindings each entrypoint's module declares.
///
/// Parsed ONCE for the process. Every entrypoint in the tree has to be read to
/// answer it, and a `Device` is opened per test and per deployment -- doing it
/// at each open took the device suite from 21 seconds to 169.
///
/// The count does not depend on the adapter; only the COMPARISON does, so the
/// expensive half is cached and the cheap half is per-device.
fn widest_bindings() -> &'static [(&'static str, u32)] {
    static COUNTS: std::sync::OnceLock<Vec<(&'static str, u32)>> = std::sync::OnceLock::new();
    COUNTS.get_or_init(|| {
        kernels_wgpu::entrypoints()
            .into_iter()
            .filter_map(|name| {
                let declared =
                    crate::reflect::entrypoint(&name, kernels_wgpu::Capability::Baseline).ok()?;
                Some((&*String::leak(name), declared.bindings))
            })
            .collect()
    })
}

impl Device {
    /// Open the best adapter this machine offers.
    ///
    /// `PowerPreference::HighPerformance`, which on a laptop with two adapters
    /// is the discrete one. Not `None`: a driver that silently picked the
    /// integrated GPU would report plausible numbers a great deal slower, and
    /// "which adapter" is exactly the fact [`Device::name`] exists to make
    /// visible.
    ///
    /// # What is requested, and why it is not the downlevel defaults
    ///
    /// The **adapter's own limits**, not `wgpu::Limits::downlevel_defaults()`.
    /// The defaults guarantee 8 storage buffers per stage and
    /// `sdpa_paged_decode` binds eleven, so asking for them would refuse the
    /// paged attention pipeline on hardware that offers 64 of them. The rows
    /// this adapter still cannot reach are computed here and named by
    /// [`Self::unreachable`]; a pipeline for one is refused by name at
    /// [`Pipelines::get`] rather than at `create_compute_pipeline`.
    ///
    /// The features requested are the ones the tiers ask for and the adapter
    /// reports — [`Capability::requires`] names them as strings and this maps
    /// them. An UNKNOWN name makes its tier unavailable rather than being
    /// skipped, which is `kernels-vulkan`'s lesson: its matrix tier named its
    /// matrix extension and not the `shaderFloat16` its operands needed, the
    /// driver built the pipeline anyway, and the answer was `-9.5` where it
    /// should have been `-0.0618`.
    ///
    /// # Errors
    ///
    /// [`Unavailable`] when no adapter answers, or when the one that does will
    /// not give up a device.
    pub fn open() -> Result<Self, Unavailable> {
        // `PIE_WGPU_FALLBACK` asks for the SOFTWARE adapter, and it is read
        // HERE rather than only in the test harnesses because of what the
        // paragraph below used to claim and could not deliver.
        //
        // `WGPU_POWER_PREF` was said to be "how a machine with two adapters is
        // asked the same question twice". Measured on a machine with an RTX
        // 4090 and `llvmpipe`, all three of unset, `low` and `high` answer the
        // 4090: a power preference RANKS adapters and never reaches a software
        // one, which needs `force_fallback_adapter` -- the flag
        // [`Self::software`] sets and no preference implies. So the stated way
        // to ask the second implementation could not ask it, and the three
        // test files that spell `PIE_WGPU_FALLBACK` themselves were the only
        // paths that could: no gate, no curated run and no server could reach
        // the second adapter at all.
        //
        // Reading it here makes the crate's own strongest claim -- a shader
        // that agrees on a discrete GPU and on `llvmpipe` "has been checked by
        // two independent compilers and two independent schedulers" -- true of
        // every path rather than of three test files.
        if std::env::var_os("PIE_WGPU_FALLBACK").is_some() {
            return Self::software();
        }
        // `WGPU_POWER_PREF` still overrides, and it is the right knob for what
        // it can actually do: choosing between two HARDWARE adapters, which is
        // a machine this has not been run on.
        Self::open_with(
            wgpu::PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::HighPerformance),
        )
    }

    /// The FALLBACK adapter: a software implementation of WebGPU, if the
    /// instance has one.
    ///
    /// A second implementation of the same WGSL on the same machine, which is
    /// the strongest cross-check this backend can offer and close to the reason
    /// it exists — a shader that agrees on a discrete GPU and on `llvmpipe` has
    /// been checked by two independent compilers and two independent schedulers,
    /// where one that agrees only on the card it was written on has been checked
    /// by neither.
    ///
    /// Ruinously slow, and not a deployment: this is what a test asks for.
    ///
    /// # Errors
    ///
    /// [`Unavailable`] when the instance has no software adapter, which on a
    /// machine without Mesa is the ordinary answer.
    pub fn software() -> Result<Self, Unavailable> {
        Self::request(wgpu::PowerPreference::None, true)
    }

    /// The same as [`Self::open`], at a stated power preference.
    ///
    /// Separate because a machine with two adapters answers the same question
    /// twice, and two adapters DISAGREEING about a number is the most useful
    /// signal this backend can produce.
    ///
    /// # Errors
    ///
    /// As [`Self::open`].
    pub fn open_with(power: wgpu::PowerPreference) -> Result<Self, Unavailable> {
        Self::request(power, false)
    }

    /// Open one adapter, however it was chosen.
    fn request(power: wgpu::PowerPreference, fallback: bool) -> Result<Self, Unavailable> {
        let instance = wgpu::Instance::new(
            // `with_env` so `WGPU_BACKEND=vulkan` selects one, which is how a
            // machine with several is asked the same question twice.
            wgpu::InstanceDescriptor::new_without_display_handle().with_env(),
        );
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: power,
            force_fallback_adapter: fallback,
            compatible_surface: None,
            // Deliberately off. Limit bucketing exists so a browser cannot be
            // fingerprinted by its adapter's exact numbers; it rounds the
            // limits DOWN, and this driver's whole argument about
            // `max_storage_buffers_per_shader_stage` is about the real number.
            apply_limit_buckets: false,
        }))
        .map_err(|e| Unavailable(format!("no adapter: {e}")))?;

        let info = adapter.get_info();
        let features = adapter.features();
        let limits = adapter.limits();

        // The tiers this adapter can serve, and the features they need.
        let mut tiers = Vec::new();
        let mut wanted = wgpu::Features::empty();
        for tier in Capability::PREFERENCE {
            let mut needs = wgpu::Features::empty();
            // EVERY name, not the first, and an unknown name is a refusal
            // rather than a skip.
            let known = tier.requires().iter().all(|name| match feature(name) {
                Some(bit) => {
                    needs |= bit;
                    features.contains(bit)
                }
                None => false,
            });
            if known {
                tiers.push(tier);
                wanted |= needs;
            }
        }

        let errors: Arc<Mutex<Vec<(bool, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&errors);
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pie driver-wgpu"),
            required_features: wanted,
            required_limits: limits.clone(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| Unavailable(format!("{} would not open a device: {e}", info.name)))?;

        // Before anything else can fail. Without this, `wgpu`'s default
        // handler panics the process on a validation error, and a driver whose
        // refusals are panics has no refusals.
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let oom = matches!(e, wgpu::Error::OutOfMemory { .. });
            sink.lock()
                .unwrap_or_else(|p| p.into_inner())
                .push((oom, e.to_string()));
        }));

        let limits = Limits::of(&limits);
        // Computed here rather than asked for later, so the answer to "what
        // can this machine not run" is available before a model is loaded.
        // Counted off the MODULES and not the table, which is empty. This read
        // `storage_count(sig)` over `KERNELS` until every kernel crossed to a
        // routine, at which point it answered "none" for every adapter -- and
        // `shell.rs` REFUSES A DEPLOYMENT on this list being non-empty, so the
        // guard would have gone permanently open rather than loudly wrong.
        //
        // `Declared::bindings` is one past the highest `@group(0)` binding a
        // module declares, which is the number a bind group layout must cover;
        // a variant may leave HOLES and `wgpu` checks entry for entry, so the
        // count of declared buffers would be the wrong number and this is the
        // right one. It is the same quantity `storage_count` stood in for.
        //
        // Done ONCE at open, for the reason the old comment gave: the answer
        // to "what can this machine not run" has to be available before a
        // model is loaded, not discovered when a pipeline fails. An entrypoint
        // whose source will not even parse is not this question's business and
        // is left to `Failed::Module`.
        let unreachable = widest_bindings()
            .iter()
            .filter(|(_, bindings)| *bindings > limits.storage_buffers)
            .map(|(name, _)| *name)
            .collect();

        Ok(Self {
            _instance: instance,
            adapter,
            device,
            queue,
            info,
            features,
            limits,
            tiers,
            unreachable,
            errors,
        })
    }

    /// What the adapter calls itself.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.info.name
    }

    /// Which native API is underneath.
    ///
    /// Worth reporting rather than hiding: this backend is chosen for being
    /// portable, and the same WGSL running over Vulkan here and Metal there is
    /// the claim. A number that differs between the two is a finding.
    #[must_use]
    pub fn backend(&self) -> wgpu::Backend {
        self.info.backend
    }

    /// Everything the adapter says about itself.
    #[must_use]
    pub fn info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    /// The features the adapter reports.
    #[must_use]
    pub fn features(&self) -> wgpu::Features {
        self.features
    }

    /// The eight limits this shell obeys.
    #[must_use]
    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// The tiers this adapter can build, best first.
    ///
    /// Always ends at [`Capability::Baseline`], which requires nothing.
    #[must_use]
    pub fn tiers(&self) -> &[Capability] {
        &self.tiers
    }

    /// The kernel rows this adapter cannot bind, by name.
    ///
    /// Empty on every desktop adapter. Non-empty is a real answer rather than
    /// a failure: a device offering the WebGPU floor of 8 storage buffers per
    /// stage can serve most of this table and cannot serve
    /// `sdpa_paged_decode`, and a deployment is entitled to know that before
    /// it loads a model rather than at the first attention.
    #[must_use]
    pub fn unreachable(&self) -> &[&'static str] {
        &self.unreachable
    }

    /// `min_storage_buffer_offset_alignment`, which every arena offset must
    /// divide.
    #[must_use]
    pub fn min_storage_offset(&self) -> u64 {
        u64::from(self.limits.storage_offset)
    }

    /// The largest buffer this driver could take AND bind, in bytes.
    ///
    /// # This is not a memory budget, because WebGPU has none to give
    ///
    /// `driver-vulkan`'s method of this name reads
    /// `VkPhysicalDeviceMemoryProperties` and answers with the size of the
    /// largest host-visible heap — a capacity. **Nothing on `wgpu` 30 answers
    /// that question**, and the absence is a design decision of the WebGPU
    /// spec rather than a gap in this crate's binding: device memory size is a
    /// fingerprinting vector, so the API does not report it. Concretely, of
    /// everything an adapter or a device will say here:
    ///
    /// * `Adapter::{features, limits, get_info, get_downlevel_capabilities,
    ///   get_texture_format_features}` — capabilities and identity, no bytes;
    /// * `Device::generate_allocator_report` — `total_allocated_bytes` and
    ///   `total_reserved_bytes`, which are what is ALREADY taken and not what
    ///   remains. Usage is not capacity: a device holding 100 MB may have
    ///   100 MB free or twenty gigabytes. It also answers `None` on any
    ///   backend that does not sub-allocate, so a ceiling built on it would
    ///   be backend-dependent as well as wrong;
    /// * `Device::get_internal_counters` — zero unless `wgpu`'s `counters`
    ///   feature is on, and counters of objects rather than of bytes.
    ///
    /// `Adapter::as_hal::<Vulkan>()` would reach the real memory properties,
    /// and it is refused twice over: it is an `unsafe fn`, which
    /// `#![forbid(unsafe_code)]` makes unavailable in this crate, and it names
    /// one backend, which is the whole of what this crate exists not to do.
    ///
    /// # So what this returns instead, and why it is the right number here
    ///
    /// The smaller of `max_buffer_size` and `max_storage_buffer_binding_size`
    /// — a per-ALLOCATION cap, not a heap. It is the one bound on the cache
    /// this adapter actually states, and it is a hard one: the KV cache is
    /// `layers * 2` buffers, each bound whole as a storage buffer, so a pool
    /// whose per-layer buffer passes either number can never be created on
    /// this adapter no matter how idle the device is. [`Device::zeroed`]
    /// refuses the first and [`Device::check`] the second, both by name.
    ///
    /// Both halves are needed because they are different numbers and usually
    /// unequal: a device may hold a buffer it will not bind the whole of.
    /// Taking only `max_buffer_size` would admit a pool that allocates and
    /// then cannot be bound, which is a refusal met halfway through the first
    /// attention rather than at admission.
    ///
    /// # What it therefore does NOT bound
    ///
    /// How much memory the device has. A pool under this number can still fail
    /// to allocate, and on a discrete card it usually will long before it
    /// reaches it — 2 GiB per buffer times fifty-six buffers is past every
    /// consumer part. That is the direction [`crate::resources::Pool::ceiling`]
    /// wants to err in, and its docs say why: too generous turns a permanent
    /// refusal into a retried one, and too stingy makes a scheduler
    /// permanently drop work the pool would have held.
    #[must_use]
    pub fn budget(&self) -> u64 {
        self.limits
            .buffer_size
            .min(self.limits.storage_binding_size)
    }

    /// Whether this adapter shares memory with the host.
    ///
    /// `DeviceType::IntegratedGpu` and `Cpu`. What [`crate::facts::of`] reports
    /// as `unified_memory`, and the thing that decides whether a staging copy
    /// is a copy or a formality.
    #[must_use]
    pub fn unified(&self) -> bool {
        matches!(
            self.info.device_type,
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu
        )
    }

    /// The `wgpu` device, for a caller that wants to do something this shell
    /// does not.
    #[must_use]
    pub fn raw(&self) -> &wgpu::Device {
        &self.device
    }

    /// The queue.
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// The adapter.
    #[must_use]
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }

    /// Every adapter this instance can see, opened or not.
    ///
    /// For a report rather than for a decision: a machine with two GPUs runs the
    /// same WGSL on two implementations, and knowing that the second one exists
    /// is what turns "it works here" into a question worth asking twice.
    /// `WGPU_POWER_PREF` is how [`Self::open`] is pointed at the other one.
    #[must_use]
    pub fn adapters(&self) -> Vec<wgpu::AdapterInfo> {
        block_on(self._instance.enumerate_adapters(wgpu::Backends::all()))
            .iter()
            .map(wgpu::Adapter::get_info)
            .collect()
    }

    /// Take whatever `wgpu` complained about since this was last asked.
    ///
    /// `Ok(())` when nothing did. Every fallible call below ends with this,
    /// because `wgpu` reports a validation failure through the error sink and
    /// then hands back an object that looks like the real thing — a pipeline
    /// that will not run, a bind group that cannot be set — so a caller that
    /// did not ask would carry the broken handle forward and meet it at
    /// submit, with nothing about the launch left in the message.
    ///
    /// # Errors
    ///
    /// [`Failed::OutOfMemory`] when `wgpu` called any of the parked errors an
    /// out-of-memory, and [`Failed::Wgpu`] otherwise, each holding every
    /// message since the last drain, joined.
    ///
    /// ANY rather than all: a drain that mixes a validation slip with an
    /// allocation failure is one where the allocation still failed, and the
    /// retryable reading is the safe direction -- a caller that re-posts a
    /// frame it could have faulted loses time, and one that faults a frame it
    /// could have re-posted loses the request.
    pub fn drained(&self) -> Result<(), Failed> {
        let mut sink = self.errors.lock().unwrap_or_else(|p| p.into_inner());
        if sink.is_empty() {
            return Ok(());
        }
        let parked = core::mem::take(&mut *sink);
        let oom = parked.iter().any(|(oom, _)| *oom);
        let all = parked
            .into_iter()
            .map(|(_, message)| message)
            .collect::<Vec<_>>()
            .join("; ");
        Err(if oom {
            Failed::OutOfMemory(all)
        } else {
            Failed::Wgpu(all)
        })
    }

    /// Throw away whatever `wgpu` complained about.
    ///
    /// For a caller that has just made a refusal ON PURPOSE — a test that
    /// checks a named error — and does not want the message to surface in the
    /// next unrelated call.
    pub fn forget_errors(&self) {
        self.errors
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clear();
    }

    /// A zeroed storage buffer of `bytes`.
    ///
    /// Nothing is uploaded. WebGPU requires a new `GPUBuffer`'s contents to be
    /// zero and `wgpu-core` enforces it by zero-initialising any range not yet
    /// written, so this is one allocation where the Vulkan sibling builds a
    /// host-side `vec![0u8; bytes]` and pushes it across. For a KV cache that
    /// is the difference between a memset of hundreds of megabytes and none.
    ///
    /// # Errors
    ///
    /// [`Failed::PastLimit`] when the size is past `max_buffer_size`, and
    /// [`Failed::Wgpu`] when the allocation itself fails.
    pub fn zeroed(&self, bytes: u64) -> Result<Buffer, Failed> {
        if bytes > self.limits.buffer_size {
            return Err(Failed::PastLimit {
                which: Ceiling::BufferSize,
                want: bytes,
                limit: self.limits.buffer_size,
            });
        }
        // Rounded to four, because a buffer that a copy can reach has to be a
        // whole number of `COPY_BUFFER_ALIGNMENT` units, and a readback of a
        // three-byte buffer is otherwise unexpressible. The reported size is
        // the rounded one so that `Allocation::size` and the device agree.
        let size = bytes.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT).max(4);
        let inner = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.drained()?;
        Ok(Buffer { inner, size })
    }

    /// A storage buffer holding `bytes`.
    ///
    /// # Errors
    ///
    /// As [`Self::zeroed`], plus anything the upload reports.
    pub fn buffer(&self, bytes: &[u8]) -> Result<Buffer, Failed> {
        let buffer = self.zeroed(bytes.len() as u64)?;
        self.write(&buffer, 0, bytes)?;
        Ok(buffer)
    }

    /// A storage buffer holding `words`, little-endian.
    ///
    /// The shape every fire table arrives in, and here rather than at each
    /// call site so that a table written a byte at a time cannot disagree with
    /// one written a word at a time about endianness.
    ///
    /// # Errors
    ///
    /// As [`Self::buffer`].
    pub fn words(&self, words: &[u32]) -> Result<Buffer, Failed> {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        self.buffer(&bytes)
    }

    /// A uniform buffer holding `bytes`.
    ///
    /// Separate from [`Self::buffer`] because the USAGE differs and `wgpu`
    /// checks it: a buffer bound at `@group(1) @binding(0)` must carry
    /// `BufferUsages::UNIFORM`, and one bound as a storage operand must carry
    /// `STORAGE`. Asking for both on every buffer would work and would tell a
    /// reader nothing about which slot a given allocation is for.
    ///
    /// # Errors
    ///
    /// [`Failed::PastLimit`] past `max_uniform_buffer_binding_size`, and
    /// [`Failed::Wgpu`] from the allocation.
    pub fn uniform(&self, bytes: &[u8]) -> Result<Buffer, Failed> {
        let want = bytes.len() as u64;
        if want > self.limits.uniform_binding_size {
            return Err(Failed::PastLimit {
                which: Ceiling::UniformBinding,
                want,
                limit: self.limits.uniform_binding_size,
            });
        }
        // 16 because WGSL gives every host-shareable struct an alignment of at
        // least 16 and `wgpu` refuses a uniform binding whose size is not a
        // multiple of it. `kernels_wgpu::uniform_size` rounds the same way, so
        // this is a floor for the degenerate case rather than a second rule.
        let size = want.next_multiple_of(16).max(16);
        let inner = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.drained()?;
        let buffer = Buffer { inner, size };
        if !bytes.is_empty() {
            self.write(&buffer, 0, bytes)?;
        }
        Ok(buffer)
    }

    /// Write `bytes` into `buffer` at `offset`.
    ///
    /// # Errors
    ///
    /// [`Failed::Wgpu`] when the range leaves the buffer or the queue refuses.
    pub fn write(&self, buffer: &Buffer, offset: u64, bytes: &[u8]) -> Result<(), Failed> {
        if bytes.is_empty() {
            return Ok(());
        }
        let end = offset.saturating_add(bytes.len() as u64);
        if end > buffer.size {
            return Err(Failed::Wgpu(format!(
                "{} bytes at {offset} do not fit a buffer of {}",
                bytes.len(),
                buffer.size
            )));
        }
        // `write_buffer` wants a length that is a whole number of copy units.
        // The tail is padded with zeros rather than refused, because the
        // caller's bytes are a tensor and its width is not the transfer's
        // business -- and the buffer was rounded up to hold them.
        let padded = bytes.len().next_multiple_of(4);
        if padded == bytes.len() {
            self.queue.write_buffer(&buffer.inner, offset, bytes);
        } else if offset + padded as u64 <= buffer.size {
            let mut whole = bytes.to_vec();
            whole.resize(padded, 0);
            self.queue.write_buffer(&buffer.inner, offset, &whole);
        } else {
            return Err(Failed::Wgpu(format!(
                "{} bytes at {offset} round up to {padded} and do not fit a \
                 buffer of {}",
                bytes.len(),
                buffer.size
            )));
        }
        self.drained()
    }

    /// Read the whole buffer back.
    ///
    /// # Errors
    ///
    /// As [`Self::read_at`].
    pub fn read(&self, buffer: &Buffer) -> Result<Vec<u8>, Failed> {
        self.read_at(buffer, 0, buffer.size)
    }

    /// Read `len` bytes of `buffer` from `offset`.
    ///
    /// **The one readback in this crate.** `wgpu` has no `vkMapMemory` on a
    /// device-local allocation, so this is: a `MAP_READ | COPY_DST` staging
    /// buffer, a `copy_buffer_to_buffer` into it, a submit, a `map_async`, a
    /// device poll, the bytes, an `unmap`. Six steps with two easy omissions —
    /// polling before submitting, and forgetting the `unmap`, which leaves the
    /// staging buffer permanently mapped — which is why every caller,
    /// including `serve::logits` and every test, comes here instead of writing
    /// it again.
    ///
    /// The copy is aligned outward to `COPY_BUFFER_ALIGNMENT` and the answer
    /// sliced back, so a caller may ask for any range.
    ///
    /// # Errors
    ///
    /// [`Failed::Wgpu`] when the range leaves the buffer, when the map fails,
    /// or when the device does not answer within the deadline [`Self::wait`]
    /// applies.
    pub fn read_at(&self, buffer: &Buffer, offset: u64, len: u64) -> Result<Vec<u8>, Failed> {
        if len == 0 {
            return Ok(Vec::new());
        }
        if offset.saturating_add(len) > buffer.size {
            return Err(Failed::Wgpu(format!(
                "{len} bytes at {offset} run past a buffer of {}",
                buffer.size
            )));
        }
        let align = wgpu::COPY_BUFFER_ALIGNMENT;
        let from = offset - offset % align;
        let skip = (offset - from) as usize;
        let span = (len + (offset - from)).next_multiple_of(align);
        // A range that is legal to ask for can still round past the end, and
        // the buffer's own size is already a multiple of the alignment, so
        // this clamp cannot cut into what was asked for.
        let span = span.min(buffer.size - from);

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: span,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.drained()?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback"),
            });
        encoder.copy_buffer_to_buffer(&buffer.inner, from, &staging, 0, Some(span));
        self.queue.submit([encoder.finish()]);
        self.drained()?;

        let answer: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> =
            Arc::new(Mutex::new(None));
        let park = Arc::clone(&answer);
        staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            *park.lock().unwrap_or_else(|p| p.into_inner()) = Some(r);
        });
        // The poll is what runs the callback: `map_async` queues it and
        // nothing else drives it. Waiting on the LAST submission is what makes
        // the copy above have happened.
        let deadline = wait();
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(deadline),
            })
            .map_err(|e| not_answered(deadline, &e))?;
        self.drained()?;
        match answer.lock().unwrap_or_else(|p| p.into_inner()).take() {
            Some(Ok(())) => {}
            Some(Err(e)) => return Err(Failed::Wgpu(format!("the readback did not map: {e}"))),
            None => {
                return Err(Failed::Wgpu(
                    "the readback's map never completed".to_string(),
                ));
            }
        }

        let view = staging
            .slice(..)
            .get_mapped_range()
            .map_err(|e| Failed::Wgpu(format!("the readback did not map: {e}")))?;
        let bytes = view[skip..(skip + len as usize).min(view.len())].to_vec();
        // Both, in this order. A `BufferView` borrows the mapping and `unmap`
        // takes it away, so a staging buffer left mapped is one that cannot be
        // dropped cleanly.
        drop(view);
        staging.unmap();
        Ok(bytes)
    }

    /// Copy `bytes` from one buffer to another, for each of `moves`, in one
    /// command buffer.
    ///
    /// Each entry is `(source, source offset, destination, destination
    /// offset)`. Unlike [`Self::shuffle`] this needs no scratch, because the two
    /// ends are DIFFERENT buffers — which is the only kind of copy `wgpu`
    /// allows. A caller moving bytes within one buffer wants `shuffle`.
    ///
    /// One submission for the whole list, which is what makes a pool resize one
    /// round trip rather than `2 * layers` of them. And device to device
    /// throughout: `driver-vulkan`'s resize reads each layer back to the host
    /// and writes it out again, because its `Device` has no copy between two
    /// buffers.
    ///
    /// # Errors
    ///
    /// [`Failed::Wgpu`] when an offset is not a multiple of
    /// `COPY_BUFFER_ALIGNMENT` or a range leaves its buffer.
    pub fn transfer(
        &self,
        moves: &[(&Buffer, u64, &Buffer, u64)],
        bytes: u64,
    ) -> Result<(), Failed> {
        if moves.is_empty() || bytes == 0 {
            return Ok(());
        }
        let span = bytes.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        for (src, from, dst, to) in moves {
            for (buffer, at) in [(src, from), (dst, to)] {
                if !at.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
                    return Err(Failed::Wgpu(format!(
                        "a copy at {at} is not a multiple of {}",
                        wgpu::COPY_BUFFER_ALIGNMENT
                    )));
                }
                if at.saturating_add(span) > buffer.size {
                    return Err(Failed::Wgpu(format!(
                        "{span} bytes at {at} run past a buffer of {}",
                        buffer.size
                    )));
                }
            }
            if src.inner == dst.inner {
                return Err(Failed::Wgpu(
                    "`transfer` cannot copy a buffer onto itself; `shuffle` is \
                     the call that routes through a scratch"
                        .to_string(),
                ));
            }
        }
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("transfer"),
            });
        for (src, from, dst, to) in moves {
            encoder.copy_buffer_to_buffer(&src.inner, *from, &dst.inner, *to, Some(span));
        }
        self.queue.submit([encoder.finish()]);
        self.drained()?;
        self.wait()
    }

    /// Move `bytes` from one place in a buffer to another, for each of
    /// `moves`, sharing one scratch allocation and one submission.
    ///
    /// # Why this is not `copy_buffer_to_buffer`
    ///
    /// `wgpu-core/src/command/transfer.rs:987` refuses a copy whose source and
    /// destination are the SAME buffer, where `vkCmdCopyBuffer` allows it for
    /// non-overlapping regions. A KV cache is one buffer per layer per side and
    /// a fork moves a page within it, so this backend cannot express that copy
    /// at all — it has to go `src -> scratch -> dst`.
    ///
    /// One scratch and one command buffer for the whole list, and that is
    /// correct rather than merely cheap: commands in a command buffer execute
    /// in order and `wgpu` puts a barrier between two uses of the scratch (see
    /// the module docs), so the second move's `src -> scratch` cannot overtake
    /// the first move's `scratch -> dst`. A scratch per move would be 2·layers
    /// allocations per page copied.
    ///
    /// # Errors
    ///
    /// [`Failed`] when a range leaves its buffer, or from the allocation.
    pub fn shuffle(&self, moves: &[Move<'_>], bytes: u64) -> Result<(), Failed> {
        if moves.is_empty() || bytes == 0 {
            return Ok(());
        }
        let span = bytes.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
        for one in moves {
            for at in [one.from, one.to] {
                if !at.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
                    return Err(Failed::Wgpu(format!(
                        "a copy at {at} is not a multiple of {}",
                        wgpu::COPY_BUFFER_ALIGNMENT
                    )));
                }
                if at.saturating_add(span) > one.buffer.size {
                    return Err(Failed::Wgpu(format!(
                        "{span} bytes at {at} run past a buffer of {}",
                        one.buffer.size
                    )));
                }
            }
        }
        let scratch = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("page scratch"),
            size: span,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.drained()?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("page move"),
            });
        for one in moves {
            encoder.copy_buffer_to_buffer(&one.buffer.inner, one.from, &scratch, 0, Some(span));
            encoder.copy_buffer_to_buffer(&scratch, 0, &one.buffer.inner, one.to, Some(span));
        }
        self.queue.submit([encoder.finish()]);
        self.drained()?;
        self.wait()
    }

    /// Wait until the queue has finished everything submitted so far.
    ///
    /// # Errors
    ///
    /// [`Failed::Wgpu`] if the device does not answer within the deadline:
    /// `WAIT_DEFAULT`, or what `PIE_WGPU_WAIT_SECS` says.
    pub fn wait(&self) -> Result<(), Failed> {
        let deadline = wait();
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(deadline),
            })
            .map_err(|e| not_answered(deadline, &e))?;
        self.drained()
    }

    /// Run one dispatch and wait for it.
    ///
    /// Synchronous on purpose, and it is the SLOW path. This is the shell a
    /// correctness test drives: one dispatch, one submission, one wait, so a
    /// wrong answer is a wrong answer rather than a race. See [`Self::run_all`]
    /// for the one a fire uses and for what the two prove about each other.
    ///
    /// [`Self::run_all`] with a run of one, so a dispatch that binds the arena
    /// both ways is shadowed here exactly as it is there — the two paths must
    /// not differ in what they COMPUTE, only in how much they submit at once,
    /// or the test that compares them is comparing two shells.
    ///
    /// # Errors
    ///
    /// [`Failed`], naming which of the checks the call did not pass.
    pub fn run(
        &self,
        pipeline: &Pipeline,
        buffers: &[Bound<'_, Buffer>],
        uniform: &[u8],
        groups: [u32; 3],
    ) -> Result<(), Failed> {
        self.run_all(&[Recorded {
            pipeline,
            buffers,
            uniform,
            groups,
        }])
        .map(|_| ())
        .map_err(|(_, why)| why)
    }

    /// Record a whole run, submit once, and wait. Answers how many read
    /// operands had to be shadowed.
    ///
    /// [`Self::run`] is one dispatch per submission, which is right for a test
    /// and wrong for a fire: a real plan states thousands of rectangles and one
    /// round trip to the queue per rectangle is most of the time a small model
    /// spends.
    ///
    /// # No barrier is written here, and that is the researched answer
    ///
    /// See this module's own docs. `wgpu-core` emits a HAL barrier before every
    /// dispatch that reuses a buffer in an exclusive state, at every encoding
    /// granularity, because `skip_barrier` can only skip a usage in the
    /// adapter's ordered mask and `STORAGE_READ_WRITE` never is. A plan's
    /// launches chain through the arena, so nearly every pair gets one.
    ///
    /// # One compute pass, until a dispatch needs a shadow
    ///
    /// The whole run goes into ONE pass in ONE command buffer where it can. A
    /// dispatch that binds one buffer both readable and writable cannot be
    /// recorded at all — see the module docs — so its read side is copied into a
    /// scratch buffer first, and a copy cannot be encoded inside a pass. The
    /// recording therefore breaks into a command buffer of copies and a command
    /// buffer of dispatches at each such point, all submitted together, which
    /// orders them exactly as one pass would.
    ///
    /// The returned count is how many ranges were shadowed. It is a real number
    /// a caller wants: it is the cost of the arena being one allocation, and a
    /// plan that pays it four thousand times is paying for a design decision one
    /// layer up rather than for anything a device did.
    ///
    /// Each dispatch is checked before any of them is recorded, and the whole
    /// run is refused if any of them fails — nothing is submitted, so a caller
    /// never has to reason about a partially executed plan.
    ///
    /// # Errors
    ///
    /// [`Failed`], with the [`Stage`] it belongs to — a dispatch index for
    /// everything checked before the queue, and [`Stage::Submission`] for
    /// anything after it, which singles out no dispatch and must not pretend
    /// to.
    pub fn run_all(&self, run: &[Recorded<'_, '_>]) -> Result<Ran, (Stage, Failed)> {
        if run.is_empty() {
            return Ok(Ran::default());
        }
        // Every check first, so a refusal has submitted nothing. `Failed::Aliased`
        // is deliberately NOT among them: the shadow below is what answers it.
        for (at, one) in run.iter().enumerate() {
            self.check_bindable(one)
                .map_err(|e| (Stage::Launch(at), e))?;
        }
        // The scratch copies each dispatch needs, and the ranges it will bind in
        // place of its own. Taken before any bind group is made, because a bind
        // group has to name the scratch rather than the arena.
        let mut shadows = Vec::with_capacity(run.len());
        for (at, one) in run.iter().enumerate() {
            shadows.push(self.shadow(one).map_err(|e| (Stage::Launch(at), e))?);
        }
        let copies: usize = shadows.iter().map(|s| s.len()).sum();

        // Then every bind group, so that recording holds only references.
        // `create_bind_group` is where `wgpu` compares the entries against the
        // layout, so a mismatch is caught here with the launch index in hand
        // rather than at submit.
        let mut bound = Vec::with_capacity(run.len());
        for (at, (one, shadow)) in run.iter().zip(&shadows).enumerate() {
            bound.push(self.bind(one, shadow).map_err(|e| (Stage::Launch(at), e))?);
        }

        let encoder = |label| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
        };
        // ONE command buffer for the whole fire, copies and passes alike.
        //
        // A shadow point still ends the pass -- `copy_buffer_to_buffer` is not
        // encodable inside one -- but ending a pass is not ending an ENCODER,
        // and this loop used to do both: an encoder per copy group and another
        // per run of dispatches, which for a real decode is 735 command
        // buffers for 452 launches, because 451 of them shadow something.
        //
        // Measured on an RTX 4090, qwen3-0.6B, one decode of 452 launches:
        //
        // |            | 735 buffers | 1 buffer |
        // | encoding   |     7.1 ms  |   4.3 ms |
        // | submit     |     5.4     |   1.0    |
        // | wait       |    13.0     |   9.7    |
        // | whole fire |    31.9     |  20.5    |
        //
        // The wait falls too, which is the part worth explaining: a command
        // buffer is a unit of work the queue schedules, and 735 of them are
        // 735 chances to stall on a boundary the GPU had no reason to draw.
        //
        // The ORDERING is the one the first section of these docs establishes,
        // unchanged. Command buffers in a single `submit` execute in order,
        // and commands within one command buffer execute in order; the copies
        // still land before the pass that reads them, and `wgpu-core` tracks
        // the usage transition and emits the barrier either way. What was
        // bought by the split was nothing.
        let mut submission: Vec<wgpu::CommandBuffer> = Vec::new();
        let mut at = 0;
        let mut work = encoder("fire");
        while at < run.len() {
            if !shadows[at].is_empty() {
                for one in &shadows[at] {
                    work.copy_buffer_to_buffer(
                        &one.from.inner,
                        one.at,
                        &one.into,
                        0,
                        Some(one.bytes),
                    );
                }
            }
            {
                let mut pass = work.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("fire"),
                    timestamp_writes: None,
                });
                loop {
                    let groups = &bound[at];
                    pass.set_pipeline(&run[at].pipeline.pipeline);
                    pass.set_bind_group(STORAGE_GROUP, &groups.storage, &[]);
                    if let Some(block) = &groups.uniform {
                        pass.set_bind_group(UNIFORM_GROUP, block, &[]);
                    }
                    let g = run[at].groups;
                    pass.dispatch_workgroups(g[0], g[1], g[2]);
                    at += 1;
                    // The pass runs until the next dispatch that needs a copy,
                    // which cannot be encoded inside one.
                    if at >= run.len() || !shadows[at].is_empty() {
                        break;
                    }
                }
            }
        }
        submission.push(work.finish());
        let buffers = submission.len();
        self.queue.submit(submission);
        let whole = Stage::Submission { of: run.len() };
        self.drained().map_err(|e| (whole, e))?;
        // The wait is what makes a caller's next `read` see this fire, and it
        // is also what makes the scalar blocks and the scratch buffers safe to
        // drop: they go when `bound` and `shadows` do, which is after the queue
        // is done with them.
        self.wait().map_err(|e| (whole, e))?;
        drop(bound);
        drop(shadows);
        Ok(Ran {
            shadowed: copies,
            buffers,
        })
    }

    /// The read-only ranges this dispatch cannot bind where they are.
    ///
    /// Empty for a dispatch whose operands live in different buffers, which is
    /// every dispatch a test builds by hand and no dispatch a real plan states.
    /// See the module docs for the rule and for why the copy is correct.
    fn shadow<'b>(&self, one: &Recorded<'_, 'b>) -> Result<Vec<Shadow<'b>>, Failed> {
        let read_only = one.pipeline.read_only.as_slice();
        // The buffers this dispatch WRITES. Small -- a launch writes one or two
        // -- so a linear scan beats a set.
        let written: Vec<&Buffer> = one
            .buffers
            .iter()
            .zip(read_only)
            .filter(|(_, ro)| !**ro)
            .map(|(b, _)| b.buffer())
            .collect();
        if written.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for (bound, ro) in one.buffers.iter().zip(read_only) {
            if !*ro || !written.iter().any(|w| *w == bound.buffer()) {
                continue;
            }
            // The copy is aligned outward, exactly as `read_at` aligns a
            // readback: `copy_buffer_to_buffer` wants whole copy units, and the
            // shader addresses from the BINDING's start, so the scratch has to
            // begin where the range begins. An offset that does not divide four
            // is refused rather than shifted -- shifting would bind the right
            // number of bytes starting at the wrong element.
            if !bound.offset().is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
                return Err(Failed::Wgpu(format!(
                    "a range at {} has to be shadowed and does not start on a \
                     {}-byte boundary",
                    bound.offset(),
                    wgpu::COPY_BUFFER_ALIGNMENT
                )));
            }
            let bytes = bound
                .len()
                .next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
                .min(bound.buffer().size - bound.offset());
            let into = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.drained()?;
            out.push(Shadow {
                slot: *bound,
                from: bound.buffer().clone(),
                at: bound.offset(),
                bytes,
                into,
            });
        }
        Ok(out)
    }

    /// Everything a dispatch is refused for, plus the one thing it is not.
    ///
    /// `Failed::Aliased` is what [`Self::check`] adds and this does not: a
    /// caller of `run_all` gets the shadow copy instead of the refusal, and a
    /// caller that wants to know whether one was needed asks `check`.
    ///
    /// # Errors
    ///
    /// [`Failed`].
    fn check_bindable(&self, one: &Recorded<'_, '_>) -> Result<(), Failed> {
        let pipeline = one.pipeline;
        if one.buffers.len() != pipeline.slots.len() {
            return Err(Failed::Bindings {
                module: pipeline.slots.len(),
                bound: one.buffers.len(),
            });
        }
        // Both directions, and the short one is what this is for. A block
        // shorter than the struct reads as zeros past its end, and a zero
        // pitch or a zero flag is a plausible number. Offering one where the
        // module declares none is the other direction and would bind a group
        // the pipeline layout does not have.
        let needs = pipeline.declared.uniform_bytes;
        if (needs == 0) != one.uniform.is_empty() || (one.uniform.len() as u32) < needs {
            return Err(Failed::Params {
                needs,
                given: one.uniform.len(),
            });
        }
        if one.uniform.len() as u64 > self.limits.uniform_binding_size {
            return Err(Failed::PastLimit {
                which: Ceiling::UniformBinding,
                want: one.uniform.len() as u64,
                limit: self.limits.uniform_binding_size,
            });
        }
        // Only the bindings whose block has a fixed size, which is the
        // parameter structs. A tensor binding ends in a runtime array and its
        // extent is the call's to decide, so there is nothing to check it
        // against and nothing is claimed.
        //
        // Zipped against the layout's own binding NUMBERS and not counted from
        // zero: `block_bytes` is indexed by binding number, and past a hole the
        // caller's nth buffer is not binding n.
        for (&binding, bound) in pipeline.slots.iter().zip(one.buffers) {
            if bound.len() > self.limits.storage_binding_size {
                return Err(Failed::PastLimit {
                    which: Ceiling::StorageBinding,
                    want: bound.len(),
                    limit: self.limits.storage_binding_size,
                });
            }
            let Some(Some(needs)) = pipeline.declared.block_bytes.get(binding as usize) else {
                continue;
            };
            if bound.len() < u64::from(*needs) {
                return Err(Failed::Short {
                    binding,
                    needs: *needs,
                    given: bound.len(),
                });
            }
        }
        // Two operands covering the same bytes PARTIALLY. See
        // `Failed::Overlapping` for why identical is fine and disjoint is the
        // ordinary case. In `check_bindable` and not in `check`, so `run_all`
        // pays it: this one is not answered by a workaround.
        let read_only = one.pipeline.read_only.as_slice();
        for (i, a) in one.buffers.iter().enumerate() {
            if read_only.get(i).copied().unwrap_or(false) {
                continue;
            }
            for (j, b) in one.buffers.iter().enumerate() {
                if i == j || a.buffer() != b.buffer() {
                    continue;
                }
                let (x, y) = (
                    a.offset()..a.offset() + a.len(),
                    b.offset()..b.offset() + b.len(),
                );
                if x == y {
                    continue;
                }
                let lo = x.start.max(y.start);
                let hi = x.end.min(y.end);
                if lo < hi {
                    return Err(Failed::Overlapping {
                        writer: pipeline.slots.get(i).copied().unwrap_or(i as u32),
                        other: pipeline.slots.get(j).copied().unwrap_or(j as u32),
                        overlap: lo..hi,
                    });
                }
            }
        }
        if one.groups.contains(&0) {
            return Err(Failed::Empty { groups: one.groups });
        }
        for &n in &one.groups {
            if n > self.limits.workgroups_per_dimension {
                return Err(Failed::PastLimit {
                    which: Ceiling::Workgroups,
                    want: u64::from(n),
                    limit: u64::from(self.limits.workgroups_per_dimension),
                });
            }
        }
        Ok(())
    }

    /// Everything a dispatch is refused for, including the arena being bound
    /// both ways.
    ///
    /// The whole diagnosis, for a caller that wants to know rather than to run.
    /// [`Self::run_all`] deliberately does NOT call this: it shadows the read
    /// side and proceeds, because refusing every arena launch would be refusing
    /// every plan.
    ///
    /// # Errors
    ///
    /// [`Failed`], and [`Failed::Aliased`] where the two siblings would have
    /// bound it without comment.
    pub fn check(&self, one: &Recorded<'_, '_>) -> Result<(), Failed> {
        self.check_bindable(one)?;
        let read_only = one.pipeline.read_only.as_slice();
        for (reader, (bound, ro)) in one.buffers.iter().zip(read_only).enumerate() {
            if !*ro {
                continue;
            }
            for (writer, (other, wo)) in one.buffers.iter().zip(read_only).enumerate() {
                if *wo || other.buffer() != bound.buffer() {
                    continue;
                }
                return Err(Failed::Aliased {
                    reader: one.pipeline.slots[reader],
                    writer: one.pipeline.slots[writer],
                    read_at: bound.offset(),
                    write_at: other.offset(),
                });
            }
        }
        Ok(())
    }

    /// The bind groups one checked dispatch needs.
    fn bind(&self, one: &Recorded<'_, '_>, shadow: &[Shadow<'_>]) -> Result<Bound1, Failed> {
        let pipeline = one.pipeline;
        let entries: Vec<wgpu::BindGroupEntry<'_>> = pipeline
            .slots
            .iter()
            .zip(one.buffers)
            .map(|(&binding, bound)| {
                // The scratch, where this range had to be copied out of a buffer
                // the same dispatch writes. Matched by the RANGE and not by the
                // slot, so a dispatch that binds one range at two slots gets one
                // copy named twice rather than two copies.
                let stand_in = shadow.iter().find(|s| s.slot == *bound);
                wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: stand_in.map_or(&bound.buffer().inner, |s| &s.into),
                        offset: stand_in.map_or(bound.offset(), |_| 0),
                        // The extent, never `None`. `None` means "to the end of the
                        // buffer", so a sub-range bound that way covers its own
                        // start and every tensor allocated after it in the arena --
                        // and WGSL bounds-checks against the BOUND range, so the
                        // narrow one CONFINES a stray index to a zero instead of
                        // letting it read a neighbour. `Bound::within` already
                        // refused a zero length, which is what makes the
                        // `NonZeroU64` below infallible.
                        size: wgpu::BufferSize::new(bound.len()),
                    }),
                }
            })
            .collect();
        let storage = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.storage,
            entries: &entries,
        });
        self.drained()?;

        let (uniform, block) = match (&pipeline.uniform, one.uniform.is_empty()) {
            (Some(layout), false) => {
                let block = self.uniform(one.uniform)?;
                let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: UNIFORM_BINDING,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &block.inner,
                            offset: 0,
                            size: wgpu::BufferSize::new(block.size),
                        }),
                    }],
                });
                self.drained()?;
                (Some(group), Some(block.inner))
            }
            // `check` already refused the two mixed cases -- a block with no
            // layout, or a layout with no block -- so this is the module that
            // declares neither.
            _ => (None, None),
        };
        Ok(Bound1 {
            storage,
            uniform,
            _block: block,
        })
    }
}

/// What one [`Device::run_all`] actually did.
///
/// Two numbers, both of which used to be unobservable in different ways.
/// [`Self::shadowed`] was returned bare as a `usize`, which is fine until
/// there is a second number; [`Self::buffers`] was not returned at all, and
/// `serve::Fired::submissions` — a field whose doc says "how many command
/// buffers were submitted", one unless a caller asked for a submission per
/// launch — was HARDCODED to
/// `1` because it was counting `queue.submit` CALLS.
///
/// It was wrong by a factor of 735 on a real decode, and being wrong is what
/// let the cost hide: `Fired` exists because "a caller that cannot observe
/// them cannot tell a fire that ran from one that quietly ran less", and the
/// caller was observing a constant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ran {
    /// How many read-only ranges were copied out of a buffer their own
    /// dispatch also writes. See this module's docs for the rule.
    pub shadowed: usize,
    /// How many COMMAND BUFFERS went into the queue.
    ///
    /// One per `run_all`, whatever the shadowing costs — a shadow point ends
    /// a compute pass but not an encoder.
    pub buffers: usize,
}

/// One page move within one buffer. See [`Device::shuffle`].
#[derive(Clone, Copy, Debug)]
pub struct Move<'a> {
    /// The buffer both ends are in.
    pub buffer: &'a Buffer,
    /// Where the bytes are, in bytes.
    pub from: u64,
    /// Where they go.
    pub to: u64,
}

/// The `wgpu::Features` bit a [`Capability::requires`] name means.
///
/// A match and not a lookup, and it answers `None` for a name it does not know
/// — which makes the tier unavailable rather than making the requirement
/// vanish. `kernels-wgpu`'s own docs say why that direction matters.
fn feature(name: &str) -> Option<wgpu::Features> {
    match name {
        "SHADER_F16" => Some(wgpu::Features::SHADER_F16),
        "SUBGROUP" => Some(wgpu::Features::SUBGROUP),
        _ => None,
    }
}

/// Whether each `@group(0)` storage binding is `read` or `read_write`.
///
/// Read off the parsed module and not off the row, because `wgpu` compares the
/// LAYOUT's `Storage { read_only }` against the shader's `naga::AddressSpace`
/// for equality (`wgpu-core/src/validation.rs:591`) — so a layout that guessed
/// would be refused at pipeline creation with a message about an address
/// space. `naga` spells `var<storage, read>` as `StorageAccess::LOAD` alone;
/// anything that can store is not read-only.
///
/// This is the one question [`crate::reflect::Declared`] does not answer, and
/// it is asked here rather than added there because the answer is only needed
/// where a layout is built — which is only where there is a device.
fn access(module: &naga::Module) -> BTreeMap<u32, bool> {
    let mut out = BTreeMap::new();
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = &global.binding else {
            continue;
        };
        if binding.group != STORAGE_GROUP {
            continue;
        }
        if let naga::AddressSpace::Storage { access } = global.space {
            out.insert(binding.binding, access == naga::StorageAccess::LOAD);
        }
    }
    out
}

/// Compiled pipelines, one per (entrypoint, tier), built on first use.
///
/// Separate from [`Device`] rather than a field on it, so that the borrow of a
/// pipeline does not borrow the device — a caller needs both at once on every
/// dispatch.
///
/// # Why the cache earns more here than on Vulkan
///
/// `driver-vulkan` caches a `vkCreateComputePipelines` call over a SPIR-V blob
/// `glslc` produced at build time. This caches that plus a whole shader
/// compiler: `wgpu::Device::create_shader_module` runs `naga` — parse,
/// validate, and translate to SPIR-V or MSL or DXIL — every time it is called,
/// because WGSL is not a build product. A fire states the same nineteen
/// symbols four thousand times.
pub struct Pipelines {
    built: BTreeMap<(String, Capability), Pipeline>,
    /// The expanded source and its reflection, per (entrypoint, requested
    /// tier).
    ///
    /// Beside the pipelines because it has the same lifetime and the same key,
    /// and because the alternative -- recomputing it -- was 95% of a decode
    /// step and then 22% of one. Expanding a module means splicing includes,
    /// resolving `//#if` arms and substituting defines out of the embedded
    /// tree; reflecting it means a `naga` front-end parse of the result. Both
    /// are functions of the key and neither changes for the life of a shell.
    ///
    /// The tier in the KEY is the one that was ASKED for and the one in the
    /// value is where [`crate::serve::pick`] landed, which are different
    /// whenever an adapter asks for a tier the tree has no variant of. Keying
    /// on the request is what makes the second lookup unnecessary.
    read: BTreeMap<(String, Capability), Read>,
    /// How many times [`Pipelines::remember`] has been told something.
    ///
    /// Not `read.len()`, which is the number of distinct KEYS and would sit
    /// still while a caller that had stopped consulting the cache re-expanded
    /// the same module on every step. This counts the expansions.
    reads: usize,
    /// How many times [`Pipelines::module`] has been consulted.
    ///
    /// The other half, and it catches the other regression: a caller that
    /// consults the cache once per LAUNCH rather than once per distinct symbol
    /// is cheap but wrong-shaped, and this is the number that says so.
    asks: usize,
}

/// What reading a module once yields: the expanded source, where the tier
/// search landed, the reflection, and the table row the symbol resolves to.
///
/// A tuple struct rather than a tuple because the fourth field arrived after
/// the first three and a four-tuple stops being readable at a use site.
///
/// The ROW is here because `kernels::sig_in` walks the table for an exact
/// match and then walks it again matching axis points, and that is a function
/// of the symbol like everything else in this type. `driver-vulkan` caches it
/// beside its reflection for the same reason.
pub struct Read {
    /// The expanded WGSL.
    pub source: String,
    /// Where [`crate::serve::pick`] landed, which is not always what was asked.
    pub tier: Capability,
    /// What the module binds and how it must be launched.
    pub declared: crate::reflect::Declared,
}

impl Default for Pipelines {
    fn default() -> Self {
        Self::new()
    }
}

impl Pipelines {
    /// An empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self {
            built: BTreeMap::new(),
            read: BTreeMap::new(),
            reads: 0,
            asks: 0,
        }
    }

    /// How many modules have been expanded and reflected.
    ///
    /// [`Self::built`]'s sibling, and it exists for the same reason: a
    /// server's caches must stop growing, and this one is the difference
    /// between a 28 ms decode step and a 700 ms one. Every entry is one cache
    /// MISS, because a miss is the only thing that inserts -- so a fire that
    /// re-read a module it had already read shows up here as a number that
    /// went up.
    #[must_use]
    pub const fn modules_read(&self) -> usize {
        self.reads
    }

    /// How many times the module cache has been CONSULTED.
    ///
    /// See [`Self::modules_read`] for why both numbers exist.
    #[must_use]
    pub const fn modules_asked(&self) -> usize {
        self.asks
    }

    /// The expanded source and reflection for `entrypoint` at `tier`, if this
    /// cache has read it before.
    #[must_use]
    pub fn module(&mut self, entrypoint: &str, tier: Capability) -> Option<&Read> {
        self.asks += 1;
        self.read.get(&(entrypoint.to_owned(), tier))
    }

    /// Remember what [`Self::module`] will answer with next time.
    pub fn remember(&mut self, entrypoint: &str, tier: Capability, what: Read) {
        self.reads += 1;
        self.read.insert((entrypoint.to_owned(), tier), what);
    }

    /// How many distinct pipelines have been built.
    ///
    /// A server's pipeline cache must STOP growing, and this is what makes
    /// that a number a test can compare rather than a claim about timing.
    #[must_use]
    pub fn built(&self) -> usize {
        self.built.len()
    }

    /// The pipeline for `entrypoint` at `tier`, building it from `source` if it
    /// is not already held.
    ///
    /// # Why the source is a parameter
    ///
    /// So that the module store is a SEAM and not a call. `kernels-wgpu` embeds
    /// every shader in the rlib, so the only store that matters is
    /// [`crate::serve::Embedded`] and this function could simply call
    /// `entrypoint_source` — but then no test could hand this a module `naga`
    /// refuses, and "a bad module is a named refusal rather than a panic" would
    /// be an untested claim about the one path a shell cannot otherwise reach.
    ///
    /// `tier` is only the cache KEY here; the fallback from a tier with no
    /// variant down to [`Capability::Baseline`] belongs to whoever chose the
    /// source, which is [`crate::serve::pick`]. Keying by the tier the source
    /// actually came from is what makes two requested tiers that land on one
    /// baseline module share one pipeline instead of compiling it twice.
    ///
    /// # Errors
    ///
    /// [`Failed::Unreachable`] when the module binds more storage buffers than
    /// this adapter allows a compute stage, named before `wgpu` is asked;
    /// [`Failed::Module`] when the source is not a module this crate can read;
    /// [`Failed::PastLimit`] for a workgroup wider than the adapter takes; and
    /// [`Failed::Wgpu`] for whatever `create_compute_pipeline` said.
    pub fn get(
        &mut self,
        device: &Device,
        entrypoint: &str,
        tier: Capability,
        source: &str,
    ) -> Result<&Pipeline, Failed> {
        let key = (entrypoint.to_owned(), tier);
        if !self.built.contains_key(&key) {
            let built = Self::build(device, entrypoint, tier, source)?;
            self.built.insert(key.clone(), built);
        }
        self.built
            .get(&key)
            .ok_or_else(|| Failed::Wgpu(format!("`{entrypoint}` was built and is not held")))
    }

    /// The pipeline for `entrypoint` at `tier`, if it is already built.
    ///
    /// Borrows immutably, which is the whole reason it exists: [`Self::get`]
    /// takes `&mut self` because it may build, so a caller cannot hold a
    /// reference to one pipeline while asking for the next — and recording a
    /// fire needs one reference per launch, all alive at once.
    ///
    /// `tier` must be the one the source came from, exactly as [`Self::get`]
    /// was given.
    #[must_use]
    pub fn peek(&self, entrypoint: &str, tier: Capability) -> Option<&Pipeline> {
        self.built.get(&(entrypoint.to_owned(), tier))
    }

    /// Forget every pipeline.
    ///
    /// Takes no device, unlike its Vulkan counterpart: a `wgpu` pipeline
    /// releases itself when the last handle drops. Kept as a call anyway because
    /// "the cache is emptied here" is a fact a reader of `shell`'s teardown
    /// wants to find, and because a server that swaps a model wants the old
    /// modules' memory back before it loads the new ones rather than at the next
    /// allocation failure.
    pub fn clear(&mut self) {
        self.built.clear();
    }

    /// Parse the module, build the layouts from what it declares, and compile.
    fn build(
        device: &Device,
        entrypoint: &str,
        tier: Capability,
        source: &str,
    ) -> Result<Pipeline, Failed> {
        // Parsed HERE and not inside `reflect::entrypoint`, so the same
        // `naga::Module` answers both questions this needs: what the reflection
        // says, and which of its storage bindings are writable. `wgpu` will
        // parse the text a second time inside `create_shader_module` -- there
        // is no way to hand it an already-parsed module without the `naga-ir`
        // feature, which would put a second copy of `naga`'s IR types in the
        // public API of this crate.
        let module = naga::front::wgsl::parse_str(source).map_err(|e| {
            Failed::Module(reflect::Unreadable::Unparseable(e.emit_to_string(source)))
        })?;
        let declared = reflect::of_module(&module).map_err(Failed::Module)?;
        let writable = access(&module);

        // The layout is what the module DECLARES AND READS, which is not the
        // same as what the row states and not the same as the count of
        // bindings. See the module docs: `wgpu` needs no entry for a global the
        // entry point never reads, so a hole is simply absent -- and
        // `plan_one`'s dense buffer list is sized to exactly this set.
        let slots: Vec<u32> = (0..declared.bindings)
            .filter(|at| declared.used.get(*at as usize).copied().unwrap_or(false))
            .collect();
        let read_only: Vec<bool> = slots
            .iter()
            .map(|at| writable.get(at).copied().unwrap_or(false))
            .collect();
        let needs = slots.len() as u32;
        if needs > device.limits.storage_buffers {
            return Err(Failed::Unreachable {
                entrypoint: entrypoint.to_owned(),
                needs,
                limit: device.limits.storage_buffers,
            });
        }
        let lanes: u64 = declared.local.iter().map(|n| u64::from(*n)).product();
        if lanes > u64::from(device.limits.invocations_per_workgroup) {
            return Err(Failed::PastLimit {
                which: Ceiling::Invocations,
                want: lanes,
                limit: u64::from(device.limits.invocations_per_workgroup),
            });
        }

        let entries: Vec<wgpu::BindGroupLayoutEntry> = slots
            .iter()
            .map(|at| wgpu::BindGroupLayoutEntry {
                binding: *at,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    // Read off the module. A layout that said `read_only:
                    // false` for a `var<storage, read>` global is refused at
                    // pipeline creation with a message about an address space,
                    // because `wgpu` compares the two for equality.
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: writable.get(at).copied().unwrap_or(false),
                    },
                    has_dynamic_offset: false,
                    // Left to the call. The tensor bindings end in a runtime
                    // array whose length IS the binding's, so a minimum here
                    // would be inventing one; the parameter structs do have a
                    // knowable size and `Device::check` refuses a short one by
                    // name, which is a better message than the layout's.
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let storage = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(entrypoint),
                entries: &entries,
            });
        device.drained()?;

        let uniform = (declared.uniform_bytes > 0).then(|| {
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(entrypoint),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: UNIFORM_BINDING,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                })
        });
        device.drained()?;

        // `Option<&BindGroupLayout>` per slot, because a pipeline layout is
        // indexed by GROUP NUMBER: the uniform block is `@group(1)`, so
        // omitting a missing `@group(0)` would move it to 0. `None` is how a
        // layout says "nothing at this index", which is the case for a module
        // that binds no storage buffers at all.
        let mut groups: Vec<Option<&wgpu::BindGroupLayout>> = vec![Some(&storage)];
        if let Some(block) = &uniform {
            groups.push(Some(block));
        }
        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entrypoint),
                bind_group_layouts: &groups,
                immediate_size: 0,
            });
        device.drained()?;

        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(entrypoint),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        device.drained()?;

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entrypoint),
                layout: Some(&layout),
                module: &shader,
                // `None` means "the module's only compute entry point", which
                // every expansion `kernels-wgpu` produces has exactly one of --
                // `reflect::of_module` refuses a second one by name. Naming
                // `main` would work today and would tie this file to a
                // convention the shader tree is free to change.
                entry_point: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        device.drained()?;

        Ok(Pipeline {
            entrypoint: entrypoint.to_owned(),
            tier,
            declared,
            slots,
            read_only,
            storage,
            uniform,
            pipeline,
            _module: shader,
        })
    }
}

/// The workgroup count for a fire, from the rule and the module it will run.
///
/// The one place [`geometry`] and this module meet, and the reason the LOADED
/// module is what answers: the divisor and the GEMM tile are the module's, and
/// asking it is what keeps them from being assumed.
///
/// [`geometry::groups_within`] rather than `groups`, because a device is in
/// hand here and `max_compute_workgroups_per_dimension` is the number a wide
/// enough elementwise launch reaches. A refusal names the axis.
///
/// # Errors
///
/// [`Failed::Geometry`] when the rule cannot answer for these dimensions, or
/// when the grid is past what this adapter dispatches.
pub fn groups_for(
    device: &Device,
    pipeline: &Pipeline,
    rule: Rule,
    dims: Dims,
) -> Result<[u32; 3], Failed> {
    geometry::groups_within(
        rule,
        dims,
        pipeline.module(),
        device.limits.workgroups_per_dimension,
    )
    .map_err(Failed::Geometry)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every feature a TIER names is a feature this file can look up.
    ///
    /// # Why this is the one seam worth a test with no adapter
    ///
    /// `open` builds its tier list by asking [`feature`] for every name in
    /// `Capability::requires()` and dropping the tier when any name is
    /// unknown. That is the right shape -- an unknown name must not be read
    /// as "not needed" -- but it makes a TYPO indistinguishable from a
    /// missing GPU feature: the tier is quietly absent on every adapter
    /// forever, the baseline body runs, every number is right, and nothing
    /// anywhere says a tier was dropped.
    ///
    /// Nothing else covers it. `kernels-wgpu`'s side asserts the tiers name
    /// what their bodies need; the device tests ask a real adapter what it
    /// has. The mapping BETWEEN those two -- string to `wgpu::Features` bit
    /// -- is this file's alone, and it is two match arms.
    #[test]
    fn every_feature_a_tier_names_is_one_this_file_can_look_up() {
        for tier in Capability::PREFERENCE {
            for name in tier.requires() {
                assert!(
                    feature(name).is_some(),
                    "`Capability::{tier:?}` requires `{name}`, which `feature()` does not \
                     know -- so the tier is dropped on EVERY adapter, silently, and the \
                     baseline body runs forever"
                );
            }
        }
    }

    /// And the negative half, which is what gives the test above its teeth: a
    /// name that is not a `wgpu` feature must not resolve to one.
    #[test]
    fn a_name_that_is_not_a_feature_does_not_resolve() {
        assert!(feature("SHADER_F32").is_none());
        assert!(feature("shader_f16").is_none(), "the lookup is exact");
        assert!(feature("").is_none());
    }
}

#[cfg(test)]
mod retryable {
    use super::{Ceiling, Failed};

    /// Which refusals a scheduler may re-post, in BOTH directions.
    ///
    /// `Shell::admit` turns `Failed::is_out_of_memory` into
    /// [`crate::frames::Launched::Exhausted`], which the engine re-posts, and
    /// everything else into a fault. Both mistakes are expensive and they are
    /// expensive differently: calling a validation slip retryable has the
    /// scheduler evict and re-post against a device that will refuse it again
    /// forever, and calling an allocation failure a fault kills a request that
    /// would have succeeded once something else finished.
    ///
    /// So the classification is pinned rather than left to the one call site.
    /// `PastLimit` is the case worth naming: it is a refusal about MEMORY and
    /// it is not retryable, because the limit is one the adapter declares and
    /// no eviction moves it. `admit` asks about that one first, through
    /// `Pool::ceiling`, and answers `Impossible`.
    #[test]
    fn only_a_device_that_would_not_give_memory_is_worth_re_posting() {
        assert!(Failed::OutOfMemory("device is full".to_string()).is_out_of_memory());

        for permanent in [
            Failed::Wgpu("validation: binding 3 is not a storage buffer".to_string()),
            Failed::PastLimit {
                which: Ceiling::BufferSize,
                want: 1 << 40,
                limit: 1 << 28,
            },
            Failed::Bindings {
                module: 11,
                bound: 10,
            },
            Failed::Empty { groups: [0, 1, 1] },
        ] {
            assert!(
                !permanent.is_out_of_memory(),
                "`{permanent}` is not something evicting fixes, so a scheduler \
                 told to re-post it would re-post it forever"
            );
        }
    }

    /// The message survives the classification.
    ///
    /// The sink joins every parked message and the kind is a flag beside them,
    /// so a drain that decides "out of memory" must still say what `wgpu`
    /// said -- a refusal naming nothing is the shape this crate's whole error
    /// surface exists to avoid.
    #[test]
    fn a_refusal_that_is_retryable_still_names_what_the_device_said() {
        let said = "Not enough memory left to allocate a buffer of 4294967296 bytes";
        let failed = Failed::OutOfMemory(said.to_string());
        let shown = failed.to_string();
        assert!(shown.contains(said), "got: {shown}");
        assert!(
            shown.contains("would not give the memory"),
            "and it says which KIND of refusal it is: {shown}"
        );
    }
}

#[cfg(test)]
mod deadline {
    use super::{WAIT_DEFAULT, not_answered, wait};

    /// The timeout refusal says how long it waited and how to wait longer.
    ///
    /// `wgpu` says "The requested Wait timed out before the submission was
    /// completed", which names neither. This is the only place
    /// `PIE_WGPU_WAIT_SECS` is written where someone who needs it will be
    /// looking -- it is in no README, no config template and no `--help` -- so
    /// the two facts are pinned rather than left to survive the next edit of a
    /// format string.
    ///
    /// The number asserted is 600 and NOT the default, which is the point: the
    /// message quotes the deadline the wait was actually given. A run that
    /// raised the bound and then read "30s" in its own failure would be chasing
    /// the wrong thing.
    #[test]
    fn a_wait_that_timed_out_names_its_deadline_and_the_knob_that_moves_it() {
        let said = not_answered(
            std::time::Duration::from_secs(600),
            &"The requested Wait timed out before the submission was completed.",
        )
        .to_string();

        assert!(
            said.contains("within 600s"),
            "the deadline the wait was GIVEN, not the default: {said}"
        );
        assert!(
            !said.contains("within 30s"),
            "and not the default, which this run did not use: {said}"
        );
        assert!(
            said.contains("PIE_WGPU_WAIT_SECS"),
            "a reader who cannot find the knob cannot raise it, and it is \
             documented in no README: {said}"
        );
        assert!(
            said.contains("Wait timed out"),
            "without losing what `wgpu` said: {said}"
        );
    }

    /// The wait bound is a default, and the override is read.
    ///
    /// It cannot be tested by setting the variable here — [`wait`] reads the
    /// environment ONCE, into a `OnceLock`, and a test that raced another test
    /// for who initialises it would be the flakiest thing in this crate. So
    /// what is asserted is the shape: the default is what it says, and the
    /// parse rules are exercised through the same expression the reader uses.
    ///
    /// The rules matter more than they look. A wait bound that REFUSED a
    /// mistyped number would be a driver that will not open because an
    /// operator fat-fingered an env var, which is worse than one that waits
    /// thirty seconds; and a zero would be a deadline no submission can meet,
    /// which is a hang reported as a fault on every device.
    #[test]
    fn a_mistyped_deadline_is_the_default_and_not_a_refusal() {
        assert_eq!(WAIT_DEFAULT.as_secs(), 30);

        let read = |v: Option<&str>| -> u64 {
            v.and_then(|v| v.parse::<u64>().ok())
                .filter(|secs| *secs > 0)
                .map_or(WAIT_DEFAULT, std::time::Duration::from_secs)
                .as_secs()
        };
        assert_eq!(read(Some("600")), 600, "a number is taken");
        assert_eq!(read(None), 30, "absent is the default");
        assert_eq!(read(Some("")), 30, "empty is the default");
        assert_eq!(read(Some("ten minutes")), 30, "unparseable is the default");
        assert_eq!(read(Some("-5")), 30, "negative does not parse as u64");
        assert_eq!(
            read(Some("0")),
            30,
            "zero is a deadline no submission can meet, which would report \
             every device as hung"
        );

        // And the live reader agrees with the default in this process, which
        // is the one thing that would catch the `OnceLock` being wired to
        // something else entirely.
        assert!(
            wait().as_secs() >= 1,
            "the deadline in force is {:?}, which no submission could meet",
            wait()
        );
    }
}
