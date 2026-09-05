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
//! They went to `engine_cuda::boot` and `engine_metal::boot`, and the CUDA
//! half has since stopped being a reader at all: the boot crosses as the
//! shell's own typed `engine_cuda::DeviceBoot`, re-exported below as the one
//! door the worker reaches it through. What is left is three lines per entry
//! point, and the whole of what a new shell costs THIS crate is:
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
use engine::Engine;
use engine::transfer::MemoryDomain;
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
    /// The most tokens one sequence may hold (`FireLimits::max_context`).
    /// Zero states no ceiling. Enforced on the host-geometry fire path
    /// (`pipeline::fire`): the engine's own check runs only for a fire that
    /// supplies no page table, and this runtime always supplies one.
    pub max_context: usize,
}

#[derive(Debug, Clone)]
pub struct EngineSpec {
    pub num_kv_pages: usize,
    pub limits: SchedulerLimits,
    /// Which descriptor ports this engine resolves on the device.
    ///
    /// `eta_ir::registry`'s own mask, not a numbering of this module's, so
    /// the two cannot drift.
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
/// **ONE AUTHORITY PER KEY.** The boot format is the runtime's — the worker
/// writes the document and [`crate::config`] rules on what a key may say — but
/// the PARSING belongs to whoever owns the types parsed into: the metal reader
/// is `engine_metal::boot`, and the CUDA door takes the typed
/// [`engine_cuda::DeviceBoot`] itself, so the struct is the schema and there is
/// no second reader to drift. A backend can be added without editing a crate
/// that is not the backend.
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
        feature = "cuda",
        feature = "vulkan",
        feature = "wgpu",
        all(feature = "metal", target_vendor = "apple")
    ))]
    use super::{EngineBox, Result};

    /// Open a CUDA device.
    ///
    /// # Errors
    ///
    /// A boot that shell refuses — today that is one knob out of range. The
    /// sentence is the shell's; this crate only gives it an `anyhow` skin,
    /// because the two sides of this seam have different error vocabularies
    /// and neither should have to name the other's.
    #[cfg(feature = "cuda")]
    pub fn cuda(boot: engine_cuda::DeviceBoot) -> Result<EngineBox> {
        engine_cuda::open(boot, crate::engine::load::contract_for, |name| models::sku(name).map(|sku| sku.classify))
            .map(|engine| Box::new(engine) as EngineBox)
            .map_err(::anyhow::Error::msg)
    }

    /// Open one CUDA device per rank, as one engine.
    ///
    /// One boot opens through [`cuda`]; two or more open as a
    /// tensor-parallel group (`engine_cuda::open_group`): rank `i` is
    /// `boots[i]`, the communicators are opened together, and the group
    /// answers every verb as one engine, rank 0 speaking. The load that
    /// follows must name a SKU of the group's width (`…-tp<n>`), which is
    /// how each rank's plan comes to read its own band and carry the
    /// collectives.
    ///
    /// # Errors
    ///
    /// An empty rank list, a device or communicator that failed to open.
    #[cfg(feature = "cuda")]
    pub fn cuda_group(mut boots: Vec<engine_cuda::DeviceBoot>) -> Result<(EngineBox, usize)> {
        match boots.len() {
            0 => Err(super::anyhow!("a cuda group requires at least one rank")),
            1 => Ok((cuda(boots.remove(0))?, 1)),
            ranks => engine_cuda::open_group(boots, crate::engine::load::contract_for, |name| {
                models::sku(name).map(|sku| sku.classify)
            })
            .map(|group| (Box::new(group) as EngineBox, ranks))
            .map_err(::anyhow::Error::msg),
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
    /// [`Engine::load`](engine::Engine::load), not here — `Shell::load`
    /// is one call that binds, bakes and lands, and there is nothing to bind
    /// before a plan says what to bake.
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    pub fn metal(config_bytes: &[u8]) -> Result<EngineBox> {
        engine_metal::open(config_bytes, crate::engine::load::contract_for)
            .map(|engine| Box::new(engine) as EngineBox)
            .map_err(::anyhow::Error::msg)
    }

    /// Open a Vulkan device.
    ///
    /// **FEATURE-GATED AND NOTHING ELSE**, where the Metal door above is also
    /// target-gated: a Vulkan loader is a thing a machine either has or has
    /// not, on every platform pie builds for, so there is no target whose
    /// answer is known in advance. Which device is `[vulkan] device_index` in
    /// the boot document, so this door — like Metal's — takes bytes rather
    /// than a typed boot, and the shell reads what it is about.
    ///
    /// # Errors
    ///
    /// A boot document that is not TOML, or one whose `[vulkan]` table states
    /// a knob the shell refuses. Binding the device happens at
    /// [`Engine::load`](engine::Engine::load), not here.
    #[cfg(feature = "vulkan")]
    pub fn vulkan(config_bytes: &[u8]) -> Result<EngineBox> {
        engine_vulkan::open(config_bytes, crate::engine::load::contract_for)
            .map(|engine| Box::new(engine) as EngineBox)
            .map_err(::anyhow::Error::msg)
    }

    /// Open a wgpu device.
    ///
    /// **FEATURE-GATED AND NOTHING ELSE**, like the Vulkan door above: wgpu
    /// picks its own backend (Vulkan, Metal, DX12) at run time, so there is
    /// no target whose answer is known in advance and no target half to pair
    /// the feature with. Which adapter is `[wgpu] adapter_index` in the boot
    /// document, so this door takes bytes rather than a typed boot, and the
    /// shell reads what it is about.
    ///
    /// # Errors
    ///
    /// A boot document that is not TOML, or one whose `[wgpu]` table states a
    /// knob the shell refuses. Requesting the adapter happens at
    /// [`Engine::load`](engine::Engine::load), not here.
    #[cfg(feature = "wgpu")]
    pub fn wgpu(config_bytes: &[u8]) -> Result<EngineBox> {
        engine_wgpu::open(config_bytes, crate::engine::load::contract_for)
            .map(|engine| Box::new(engine) as EngineBox)
            .map_err(::anyhow::Error::msg)
    }
}

/// **THE ONE DOOR THE WORKER REACHES THE CUDA BOOT TYPES THROUGH.** The
/// worker has no `engine-cuda` dependency of its own — that is the point of
/// this registry — so the boot it assembles for [`open::cuda`] is spelled in
/// types re-exported here, and `ordinal_of` beside them because "which
/// ordinal does `cuda:1` name" is a CUDA naming fact no other crate should
/// re-derive.
#[cfg(feature = "cuda")]
pub use engine_cuda::{DeviceBoot, Graphs, Knobs, ordinal_of, Recording, World};

// `open::metal` is TARGET-gated as well as feature-gated, for the plainest
// reason there is: there is a shell behind that door and it binds an
// `MTLDevice`. `engine-metal` itself builds and host-tests on any OS — its
// device half is `cfg(target_vendor = "apple")` and its refusing twin answers
// `Fault::Deviceless` elsewhere — but an `Engine` impl that cannot bind
// anything is not one this registry should hand a scheduler, and `worker`'s
// `EngineOptions::Metal` arm is Apple-gated on the same reading.
mod remote;

// No engine type is re-exported from here: the `Engine` impls are
// `engine_cuda::Cuda` and `engine_metal::Metal`, in the crates that own the
// devices, and re-exporting one through this module would be this crate
// claiming an engine it does not implement. `open::cuda` and `open::metal` are
// what this module offers.
pub use remote::{RemoteDisconnectHandle, RemoteEngine};

/// How many descriptor-port envelopes the CUDA shell has resolved off guest
/// device rings in this process.
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
/// because `engine-cuda` is a private link of this crate — the `cuda` feature is
/// what gates it — and a test that named the shell crate would be a test that
/// could not build without a GPU feature it does not select. That is an
/// observability argument, not a layering one: nothing about opening or
/// serving an engine passes through here.
#[cfg(feature = "cuda")]
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
