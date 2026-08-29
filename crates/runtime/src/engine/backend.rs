//! The engine registry: an [`EngineSpec`] and the [`Engine`] it is paired
//! with, addressed by `EngineId`.
//!
//! # What this module stopped being
//!
//! It was the dispatcher. `EngineBox` was an `enum` of five variants and
//! fourteen `match`es over them — seventy arms, every body a forward — plus
//! two more `match`es (`kind`, `device_domain`) that answered an engine's own
//! properties on the engine's behalf.
//!
//! All of it is [`engine_api::Engine`] now, and `EngineBox` is a
//! `Box<dyn Engine>`. What that deleted, besides the arms:
//!
//! * **The size tuning.** Two variants were `Box`ed with `size_of`
//!   measurements in their doc comments, a third carried an
//!   `expect(clippy::large_enum_variant)` saying the real fix was "eighteen
//!   call sites in code no backend here owns", and every registry entry paid
//!   the widest variant's width on every build. A trait object is one word.
//! * **A test that could only exist because of the shape.**
//!   `each_backend_names_its_own_memory` checked that this crate had not
//!   answered one backend's memory domain for another — a mistake only
//!   possible while the `match` lived here. An engine states its own domain
//!   now, so the test has nothing left to catch and is gone with the arm.
//! * **The layering lie.** This module's header said "strictly leaf: no
//!   `crate::{store,scheduler,pipeline,...}` imports" while
//!   `register_program` called `crate::pipeline::program::lookup` and a test
//!   called `crate::scheduler::device_domain`. The host-codegen splice that
//!   needed the first is `pipeline::program::with_host_codegen`'s now, called
//!   by the scheduler that owns the engine handle; the claim is true as
//!   written for the first time.
//!
//! Selecting a backend is `open`'s: one function per device, each answering
//! the same `Box<dyn Engine>`.

use std::sync::{OnceLock, RwLock};

use anyhow::{Result, anyhow};
use engine_api::transfer::MemoryDomain;
use engine_api::Engine;
use tensor_ir::registry::PortMask;

/// One execution device, behind the contract.
///
/// A `Box` rather than an `enum`: see the module header for what the five
/// variants cost. `dyn` dispatch adds one indirection per verb, and every
/// verb here is per-frame or rarer — the CUDA seam this replaced reached its
/// engine through a C ABI, a `*mut c_void` cast and seven descriptor
/// validators, so the trait object is strictly cheaper than what was there.
pub type EngineBox = Box<dyn Engine>;

#[derive(Debug, Clone, Copy)]
pub struct SchedulerLimits {
    pub max_forward_requests: usize,
    pub max_forward_tokens: usize,
    pub max_page_refs: usize,
}

#[derive(Debug, Clone)]
pub struct EngineSpec {
    pub num_kv_pages: usize,
    pub limits: SchedulerLimits,
    /// Which descriptor ports this engine resolves on the device.
    ///
    /// Was a private thirteen-bit `u32` numbering that disagreed with the
    /// port registry's own; it is `tensor_ir::registry`'s mask now
    /// (decision 19), so the two cannot drift.
    pub device_geometry_port_mask: PortMask,
    /// Does this engine resolve a step's descriptor ports when the step RUNS,
    /// rather than for the whole frame before any of it runs?
    ///
    /// False is the safe answer and the CUDA one: `FramePrepare` does every
    /// step's host work at frame entry, so a slot chained behind an earlier
    /// slot of the same frame asks for a cell nobody has produced yet, and
    /// `pipeline::fire` refuses that frame by name.
    ///
    /// True says the engine's `launch` interleaves the two halves -- convert
    /// one step, fire it, let its program run, then convert the next -- which
    /// is what makes a chained slot's tokens exist by the time they are read.
    /// `engine-vulkan` does this: its `launch` calls `envelope::fill` inside
    /// the per-step loop and answers `Filled::Early` for a channel that is
    /// still empty, rather than reading every step up front.
    pub resolves_geometry_per_step: bool,
    /// Which memory a KV page of this engine's lives in.
    ///
    /// Set by [`register_engine_backend`] from the BACKEND, not by whoever
    /// built the spec: it is a fact about the engine being registered, and a
    /// caller that could state it could state it wrongly. Every literal a
    /// caller writes here is overwritten.
    ///
    /// It exists because the scheduler used to stamp
    /// `PIE_MEMORY_DOMAIN_CUDA_DEVICE` on every `KvCopyPlan` it made, at nine
    /// sites, regardless of which engine the plan was for. On CUDA that is
    /// right by accident. On any other backend it names somebody else's
    /// memory, and an engine that checks the domain -- which is the only
    /// defence against a copy between two unrelated pools -- refuses every
    /// prefix-cache hit and every swap.
    pub device_domain: MemoryDomain,
}

impl EngineSpec {
    pub fn scheduler_limits(&self) -> SchedulerLimits {
        self.limits
    }
}

/// Opening a device: one function per backend, all answering the contract.
///
/// Free functions rather than `EngineBox::*_create` constructors, because
/// `EngineBox` is an alias for `Box<dyn Engine>` and has no inherent impl to
/// hang them off any more — and because what they have in common is the
/// ANSWER, not the receiver.
///
/// Each takes the boot bytes it is given and reads what it needs. The boot
/// TOML is the runtime's format on purpose: an engine that parsed it would be
/// the second thing entitled to an opinion about the file's shape, and the
/// two would drift.
pub mod open {
    // Every function below is gated on an engine feature, so with none
    // selected -- which is how the workspace clippy gate builds this crate --
    // the import has no user and `-D warnings` refuses the crate.
    #[cfg(any(
        feature = "_engine-cuda",
        all(feature = "engine-metal", target_vendor = "apple")
    ))]
    use super::{EngineBox, Result};

    /// Open a CUDA device.
    ///
    /// # Errors
    ///
    /// No device, or a boot config this engine refuses.
    #[cfg(feature = "_engine-cuda")]
    pub fn cuda(config_bytes: &[u8]) -> Result<EngineBox> {
        Ok(Box::new(super::cuda::open(config_bytes)?))
    }

    /// Open one CUDA device per rank, as one engine.
    ///
    /// **`palo B-tp`: ONE RANK, AND IT REFUSES THE REST BY NAME.** The group
    /// this replaced was a leader shell and N follower shells behind one
    /// `Engine`, fanning a `Vec<ModelLoadDesc>` across a thread scope and
    /// cross-checking that the ranks agreed about the model. None of that
    /// shape survives: a rank is not a load
    /// ([`LoadRequest`](engine_api::LoadRequest) is one plan, `Shard::Cut` is
    /// in the plan), the shell states tp=1 in `weights.rs` — `StorageTarget::
    /// for_backend(Cuda, 0, 1)` — and no collective ever fires in v1
    /// (`serve.rs`'s "what v1 does not do"). A multi-rank launch is refused
    /// here rather than opened and served wrong.
    ///
    /// What the successor needs: a rank index and a width reaching
    /// `StorageTarget`, an `Engine` that owns N shells and drives their
    /// streams in lockstep, and NCCL ordering — decision 5, "collectives
    /// never elided; descriptor rank-replicated".
    ///
    /// # Errors
    ///
    /// An empty rank list, more than one rank, or a device that failed to
    /// open.
    #[cfg(feature = "_engine-cuda")]
    pub fn cuda_group(config_blobs: Vec<Vec<u8>>) -> Result<(EngineBox, usize)> {
        match config_blobs.len() {
            0 => Err(super::anyhow!("a cuda group requires at least one rank")),
            1 => Ok((cuda(&config_blobs[0])?, 1)),
            ranks => Err(super::anyhow!(
                "this build serves one cuda rank and was asked for {ranks}: \
                 tensor parallelism is palo B-tp, and a group opened as a \
                 single rank would load every shard onto one device"
            )),
        }
    }

    /// Open the system's default Metal device.
    ///
    /// **THERE IS NO DEVICE KEY TO READ**, where the CUDA door reads
    /// `[model] device`: Metal selects with `MTLCreateSystemDefaultDevice`
    /// and a Mac has one GPU. The boot document is still handed over,
    /// because the seam reads what it is about and a document that says
    /// nothing about this engine is the ordinary case rather than an error.
    ///
    /// # Errors
    ///
    /// A boot document that is not TOML. Binding the device happens at
    /// [`Engine::load`](engine_api::Engine::load), not here — `Shell::load`
    /// is one call that binds, bakes and lands, and there is nothing to bind
    /// before a plan says what to bake.
    #[cfg(all(feature = "engine-metal", target_vendor = "apple"))]
    pub fn metal(config_bytes: &[u8]) -> Result<EngineBox> {
        Ok(Box::new(super::metal::open(config_bytes)?))
    }
}

// `settle_control` STOOD HERE. A seam whose memory is coherent — where
// `copy_kv` is a `memmove` — had nobody to hand a completion target to, so it
// published the terminal cell and notified the broker itself, and the comment
// on it recorded an 850-second hang from a seam that did one and not the
// other. Both halves are `crate::engine::verbs::settled` now, for every seam
// at once: the contract's control verbs answer `Result<()>` and the work is
// done when they return, so the completion the runtime hands its waiters is
// one that is already settled. There is no half of it left to forget.

#[cfg(feature = "_engine-cuda")]
mod cuda;
// TARGET-GATED as well as feature-gated, and now for the plainest reason
// there is: there is a shell behind this door and it binds an `MTLDevice`.
// `engine-metal` itself still builds and host-tests on any OS — its device
// half is `cfg(target_vendor = "apple")` and its refusing twin answers
// `Fault::Deviceless` elsewhere — but an `Engine` impl that cannot bind
// anything is not one this registry should hand a scheduler, and `worker`'s
// `EngineOptions::Metal` arm is Apple-gated on the same reading.
#[cfg(all(feature = "engine-metal", target_vendor = "apple"))]
mod metal;
mod remote;

// `pub use cuda::CudaEngine` STOOD HERE. There is no such type: the `Engine`
// impl is `engine_cuda::Cuda`, in the crate that owns the device, and this
// module's CUDA arm is a boot-config reader that answers one
// (`backend::cuda::open`). Re-exporting the shell's own type through here
// would be this crate claiming an engine it does not implement.
// `pub use metal::MetalEngine` STOOD HERE, and it is gone for the reason the
// CUDA line above it never existed: the `Engine` impl is `engine_metal::Metal`,
// in the crate that owns the device, and this module's metal arm is a
// boot-config reader that answers one (`backend::metal::open`). What stood
// here was a REFUSING engine this crate defined itself, back when there was
// no shell to open — every verb `Error::Unsupported`. There is a shell.
pub use remote::{RemoteDisconnectHandle, RemoteEngine};

// One function DOES come through, and it is not an engine: `palo B3`'s
// envelope counter, which is the only observable of a negative (the token
// that did not travel to the host). See `cuda::envelopes_resolved`.
#[cfg(feature = "_engine-cuda")]
pub use cuda::envelopes_resolved;
// And its palo-E sibling: the fold's motion mirror, which is how a
// runtime-level gate sees the next-fire hint land (`cuda::fold_observed`).
#[cfg(feature = "_engine-cuda")]
pub use cuda::fold_observed;

struct EngineRegistration {
    spec: EngineSpec,
    /// The engine, until a scheduler claims it.
    ///
    /// `Option` for the HANDOFF and not for a registration that never had one:
    /// `register_engine_backend` always installs an engine, and
    /// [`take_engine_backend`] moves it out exactly once, into the
    /// `EngineLoop` that owns it from then on. The spec stays behind because
    /// [`get_spec`] is read after the claim.
    ///
    /// There used to be a second constructor — `register_engine(spec)` — that
    /// installed `None` here, for a registration with a spec and no engine.
    /// Nothing called it. It was defined, re-exported from `crate::engine`,
    /// and had zero callers in the workspace, so the `None` it existed to
    /// produce was a state the registry could describe and never reach.
    backend: Option<EngineBox>,
}

fn registry() -> &'static RwLock<Vec<Option<EngineRegistration>>> {
    static REGISTRY: OnceLock<RwLock<Vec<Option<EngineRegistration>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

pub fn register_engine_backend(mut spec: EngineSpec, backend: EngineBox) -> usize {
    let mut engines = registry().write().unwrap();
    let id = engines.len();
    // The BACKEND states it, still — but through `device_facts`, which is
    // where an engine says what machine it is. An engine with no device of its
    // own (the remote seam) answers `None`, and host-pinned is the honest
    // reading of "these pages are not on a device of mine".
    spec.device_domain = backend
        .device_facts()
        .map_or(MemoryDomain::HostPinned, |facts| facts.domain);
    engines.push(Some(EngineRegistration {
        spec,
        backend: Some(backend),
    }));
    id
}

pub fn get_spec(engine_id: usize) -> Result<EngineSpec> {
    registry()
        .read()
        .unwrap()
        .get(engine_id)
        .and_then(|d| d.as_ref().map(|r| r.spec.clone()))
        .ok_or_else(|| anyhow!("unknown engine {engine_id}"))
}

pub fn take_engine_backend(engine_id: usize) -> Result<EngineBox> {
    let mut engines = registry().write().unwrap();
    let Some(Some(engine)) = engines.get_mut(engine_id) else {
        return Err(anyhow!("unknown engine {engine_id}"));
    };
    engine
        .backend
        .take()
        .ok_or_else(|| anyhow!("engine {engine_id} has no backend installed"))
}

pub fn unregister_engine(engine_id: usize) -> Result<()> {
    let mut engines = registry().write().unwrap();
    let Some(slot) = engines.get_mut(engine_id) else {
        return Err(anyhow!("unknown engine {engine_id}"));
    };
    slot.take();
    Ok(())
}

// The `tests` module STOOD HERE, and held exactly two things: that
// `settle_control` publishes before it notifies, and that every host-side
// seam calls it rather than minting a completion of its own. Both went with
// the helper and its three seams (see the note above). This module has no
// host-side seam left to make a claim about.
