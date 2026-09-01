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
//! All of it is [`engine::Engine`] now, and `EngineBox` is a
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
//!
//! # What this module stopped KNOWING
//!
//! It also held both shells' boot-config readers — `backend/cuda.rs` (347
//! lines) and `backend/metal.rs` (72) — which parsed the boot TOML into
//! `engine_cuda::{DeviceBoot, Graphs, Knobs}` and `engine_metal::DeviceBoot`.
//! They lived here because that is where the call happened to be made, and the
//! price was that this crate named five of a shell's types and adding a
//! backend meant editing a crate that is not the backend.
//!
//! They are `engine_cuda::boot` and `engine_metal::boot` now. What is left is
//! three lines per entry point, and the whole of what a new shell costs THIS
//! crate is:
//!
//! ```text
//!   Cargo.toml   one optional path dependency + its feature line
//!   backend.rs   one `#[cfg]`-gated `pub fn` in `open`, three lines long
//! ```
//!
//! and nothing else in `crates/runtime/src/`. No `trait EngineKind`, no
//! registry, no `register()` list: which shells exist in a binary is a
//! COMPILE-TIME fact — CUDA needs `cudarc`, Metal needs `objc2` and
//! `target_vendor = "apple"` — so a runtime table over `#[cfg]`-gated
//! implementations would be a table populated under the same `#[cfg]`s. That
//! trades a `match` for a `register()` and leaves the arrow pointing the same
//! wrong way.

use std::sync::{OnceLock, RwLock};

use anyhow::{Result, anyhow};
use engine::transfer::MemoryDomain;
use engine::Engine;
use eta_ir::registry::PortMask;

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
    /// port registry's own; it is `eta_ir::registry`'s mask now
    /// (decision 19), so the two cannot drift.
    pub device_geometry_port_mask: PortMask,
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
/// Each hands the boot bytes it is given to the shell that owns the types
/// they parse into, and boxes what comes back.
///
/// **"THE BOOT TOML IS THE RUNTIME'S FORMAT ON PURPOSE"** STOOD HERE, on the
/// argument that *"an engine that parsed it would be the second thing entitled
/// to an opinion about the file's shape, and the two would drift."* The half
/// about the FORMAT is still true and is still enforced — the worker writes
/// the document and [`crate::config`] rules on what a key may say — but the
/// conclusion drawn from it was wrong, and it cost 419 lines. Every one of
/// those lines parsed into a SHELL's type (`engine_cuda::{DeviceBoot, Graphs,
/// Knobs}`, `engine_metal::DeviceBoot`), so this crate named five structs it
/// could not otherwise have heard of, and a backend could not be added without
/// editing a crate that is not the backend.
///
/// The reader is `engine_cuda::boot` and `engine_metal::boot` now, one per
/// shell, each reading only the keys it parses into its own type. There is
/// still exactly one reader per key, which is all the drift argument ever
/// asked for; what changed is which crate it is in.
///
/// Free functions rather than `EngineBox::*_create` constructors, because
/// `EngineBox` is an alias for `Box<dyn Engine>` and has no inherent impl to
/// hang them off any more — and because what they have in common is the
/// ANSWER, not the receiver. The shell answers its own concrete type and the
/// boxing happens here, for the same reason: `EngineBox` is this crate's
/// alias, and a shell that returned one would know the name of the crate that
/// opens it.
///
/// **THE ONE THING THAT STILL CROSSES INWARD** is
/// [`crate::engine::load::contract_for`], passed as a `fn` pointer. A shell
/// cannot depend on `runtime` — that is the cycle — so the load door is a
/// parameter of every `open` below, exactly as it has always been a parameter
/// of `Cuda::new`.
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
    /// No device, or a boot config that shell refuses. The sentence is the
    /// shell's; this crate only gives it an `anyhow` skin, because the two
    /// sides of this seam have different error vocabularies and neither
    /// should have to name the other's.
    #[cfg(feature = "_engine-cuda")]
    pub fn cuda(config_bytes: &[u8]) -> Result<EngineBox> {
        engine_cuda::open(config_bytes, crate::engine::load::contract_for)
            .map(|engine| Box::new(engine) as EngineBox)
            .map_err(::anyhow::Error::msg)
    }

    /// Open one CUDA device per rank, as one engine.
    ///
    /// **`palo B-tp`: ONE RANK, AND IT REFUSES THE REST BY NAME.** The group
    /// this replaced was a leader shell and N follower shells behind one
    /// `Engine`, fanning a `Vec<ModelLoadDesc>` across a thread scope and
    /// cross-checking that the ranks agreed about the model. None of that
    /// shape survives: a rank is not a load
    /// ([`LoadRequest`](engine::LoadRequest) is one plan, `Shard::Cut` is
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
    /// **THE REFUSAL IS THIS CRATE'S AND STAYED WHEN THE READER LEFT.** Every
    /// other line of the CUDA door moved to `engine_cuda::boot`, and this one
    /// did not, because what it refuses is not a device fact. `engine_cuda::
    /// open` takes ONE document and answers ONE `Cuda`; it is never handed a
    /// list and has no way to learn that a launcher held three. The fan-out
    /// being refused is a shape of this crate's registry — N boot documents
    /// collapsing into one `EngineId` — so the arity policy belongs to the
    /// party that owns the registry. The shell's half of the same claim is
    /// already stated, and stated better, by `open`'s signature: it takes
    /// `&[u8]`, singular.
    ///
    /// It is also why a third shell does not owe the workspace a `*_group`.
    /// This exists because `worker`'s CUDA launch path fans a `Vec<Vec<u8>>`,
    /// not because opening a device needs it.
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

    /// **RUN THE COLD HALF OF A LOAD AND KEEP ONLY THE FILE IT WRITES** (§M
    /// wave M-1: `.wiki/alto/zt-as-serving-artifact.md`).
    ///
    /// The door `pie model import` reaches the CUDA shell through. It opens a
    /// device from the SAME boot document [`cuda`] above is given, runs the
    /// cold half of a load — bake, land, write the tier artifact — and tears
    /// the device down without arming a thing. No engine is registered and
    /// nothing is returned: what survives the call is the file in `[model]
    /// weight_cache_dir`, which is what makes the deployment's first real
    /// serve a warm one.
    ///
    /// It sits in this module rather than beside `Engine`'s verbs because
    /// preparing is not one: there is no load to be a verb ABOUT. It is the
    /// same shape as `cuda` — a boot document in, a device errand run — and
    /// the same argument makes it live here, in the crate that links both the
    /// shell and the catalog the load door reads.
    ///
    /// # Errors
    ///
    /// No device, a boot config the shell refuses, a checkpoint no SKU
    /// claims, or whatever the bake and the landing said. The sentence is the
    /// shell's, in an `anyhow` skin, for [`cuda`]'s reason.
    #[cfg(feature = "_engine-cuda")]
    pub fn prepare_cuda(config_bytes: &[u8], request: engine::LoadRequest) -> Result<()> {
        let engine = engine_cuda::open(config_bytes, crate::engine::load::contract_for)
            .map_err(::anyhow::Error::msg)?;
        engine.prepare(request).map_err(::anyhow::Error::from)
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
    /// [`Engine::load`](engine::Engine::load), not here — `Shell::load`
    /// is one call that binds, bakes and lands, and there is nothing to bind
    /// before a plan says what to bake.
    #[cfg(all(feature = "engine-metal", target_vendor = "apple"))]
    pub fn metal(config_bytes: &[u8]) -> Result<EngineBox> {
        engine_metal::open(config_bytes, crate::engine::load::contract_for)
            .map(|engine| Box::new(engine) as EngineBox)
            .map_err(::anyhow::Error::msg)
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

// `mod cuda;` AND `mod metal;` STOOD HERE, and both files are gone: the boot
// readers are `engine_cuda::boot` and `engine_metal::boot` now, in the crates
// that declare the types they parse into. What was on those two lines is what
// survives, and it survives on the two `open` functions above.
//
// The metal one was TARGET-gated as well as feature-gated, for the plainest
// reason there is: there is a shell behind that door and it binds an
// `MTLDevice`. `engine-metal` itself still builds and host-tests on any OS —
// its device half is `cfg(target_vendor = "apple")` and its refusing twin
// answers `Fault::Deviceless` elsewhere — but an `Engine` impl that cannot
// bind anything is not one this registry should hand a scheduler, and
// `worker`'s `EngineOptions::Metal` arm is Apple-gated on the same reading.
// That gate is unchanged; it is spelled on `open::metal`.
mod remote;

// `pub use cuda::CudaEngine` STOOD HERE. There is no such type: the `Engine`
// impl is `engine_cuda::Cuda`, in the crate that owns the device, and this
// module's CUDA arm was a boot-config reader that answered one. Re-exporting
// the shell's own type through here would be this crate claiming an engine it
// does not implement. The reader has since gone the same way for the same
// reason — `engine_cuda::boot` — and what is left is `open::cuda`.
// `pub use metal::MetalEngine` STOOD HERE, and it is gone for the reason the
// CUDA line above it never existed: the `Engine` impl is `engine_metal::Metal`,
// in the crate that owns the device. What stood here was a REFUSING engine
// this crate defined itself, back when there was no shell to open — every verb
// `Error::Unsupported`. There is a shell.
pub use remote::{RemoteDisconnectHandle, RemoteEngine};

/// How many descriptor-port envelopes the CUDA shell has resolved off guest
/// device rings in this process (`palo B3`).
///
/// **THE ONE OBSERVABLE OF A NEGATIVE.** Device-carried decode's whole claim
/// is that a chained fire's token did not travel to the host, and a round trip
/// that does not happen leaves no trace. What DOES happen is one envelope
/// resolved per attached device-carried lane per fire, so a serving gate
/// asserts on this: zero says every decode serialized through the host plane,
/// `>= decodes` says the shell read the token off the ring the epilogue wrote.
///
/// **THIS IS THE ONE PLACE THIS CRATE STILL SPELLS `engine_cuda`, AND IT IS
/// NOT THE BOOT PATH.** Re-exported here rather than reached for directly
/// because `engine-cuda` is a private link of this crate — `_engine-cuda` is
/// what gates it — and a test that named the shell crate would be a test that
/// could not build without a GPU feature it does not select. That is an
/// observability argument, not a layering one: nothing about opening or
/// serving an engine passes through here.
#[cfg(feature = "_engine-cuda")]
#[must_use]
pub fn envelopes_resolved() -> u64 {
    engine_cuda::Shell::envelopes_resolved()
}

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
