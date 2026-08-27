//! The driver registry: a [`DriverSpec`] and the [`Driver`] it is paired
//! with, addressed by `DriverId`.
//!
//! # What this module stopped being
//!
//! It was the dispatcher. `DriverBackend` was an `enum` of five variants and
//! fourteen `match`es over them — seventy arms, every body a forward — plus
//! two more `match`es (`kind`, `device_domain`) that answered a driver's own
//! properties on the driver's behalf.
//!
//! All of it is [`driver_api::Driver`] now, and `DriverBackend` is a
//! `Box<dyn Driver>`. What that deleted, besides the arms:
//!
//! * **The size tuning.** Two variants were `Box`ed with `size_of`
//!   measurements in their doc comments, a third carried an
//!   `expect(clippy::large_enum_variant)` saying the real fix was "eighteen
//!   call sites in code no backend here owns", and every registry entry paid
//!   the widest variant's width on every build. A trait object is one word.
//! * **A test that could only exist because of the shape.**
//!   `each_backend_names_its_own_memory` checked that this crate had not
//!   answered one backend's memory domain for another — a mistake only
//!   possible while the `match` lived here. A driver states its own domain
//!   now, so the test has nothing left to catch and is gone with the arm.
//! * **The layering lie.** This module's header said "strictly leaf: no
//!   `crate::{store,scheduler,pipeline,...}` imports" while
//!   `register_program` called `crate::pipeline::program::lookup` and a test
//!   called `crate::scheduler::device_domain`. The host-codegen splice that
//!   needed the first is `pipeline::program::with_host_codegen`'s now, called
//!   by the scheduler that owns the driver handle; the claim is true as
//!   written for the first time.
//!
//! Selecting a backend is `open`'s: one function per device, each answering
//! the same `Box<dyn Driver>`.

use std::sync::{OnceLock, RwLock};

use anyhow::{Result, anyhow};
use driver_api::transfer::MemoryDomain;
use driver_api::Driver;
use tensor_ir::registry::PortMask;

/// One execution device, behind the contract.
///
/// A `Box` rather than an `enum`: see the module header for what the five
/// variants cost. `dyn` dispatch adds one indirection per verb, and every
/// verb here is per-frame or rarer — the CUDA seam this replaced reached its
/// driver through a C ABI, a `*mut c_void` cast and seven descriptor
/// validators, so the trait object is strictly cheaper than what was there.
pub type DriverBackend = Box<dyn Driver>;

#[derive(Debug, Clone, Copy)]
pub struct SchedulerLimits {
    pub max_forward_requests: usize,
    pub max_forward_tokens: usize,
    pub max_page_refs: usize,
}

#[derive(Debug, Clone)]
pub struct DriverSpec {
    pub num_kv_pages: usize,
    pub limits: SchedulerLimits,
    /// Which descriptor ports this driver resolves on the device.
    ///
    /// Was a private thirteen-bit `u32` numbering that disagreed with the
    /// port registry's own; it is `tensor_ir::registry`'s mask now
    /// (decision 19), so the two cannot drift.
    pub device_geometry_port_mask: PortMask,
    /// Does this driver resolve a step's descriptor ports when the step RUNS,
    /// rather than for the whole frame before any of it runs?
    ///
    /// False is the safe answer and the CUDA one: `FramePrepare` does every
    /// step's host work at frame entry, so a slot chained behind an earlier
    /// slot of the same frame asks for a cell nobody has produced yet, and
    /// `pipeline::fire` refuses that frame by name.
    ///
    /// True says the driver's `launch` interleaves the two halves -- convert
    /// one step, fire it, let its program run, then convert the next -- which
    /// is what makes a chained slot's tokens exist by the time they are read.
    /// `driver-vulkan` does this: its `launch` calls `envelope::fill` inside
    /// the per-step loop and answers `Filled::Early` for a channel that is
    /// still empty, rather than reading every step up front.
    pub resolves_geometry_per_step: bool,
    /// Which memory a KV page of this driver's lives in.
    ///
    /// Set by [`register_driver_backend`] from the BACKEND, not by whoever
    /// built the spec: it is a fact about the driver being registered, and a
    /// caller that could state it could state it wrongly. Every literal a
    /// caller writes here is overwritten.
    ///
    /// It exists because the scheduler used to stamp
    /// `PIE_MEMORY_DOMAIN_CUDA_DEVICE` on every `KvCopyPlan` it made, at nine
    /// sites, regardless of which driver the plan was for. On CUDA that is
    /// right by accident. On any other backend it names somebody else's
    /// memory, and a driver that checks the domain -- which is the only
    /// defence against a copy between two unrelated pools -- refuses every
    /// prefix-cache hit and every swap.
    pub device_domain: MemoryDomain,
}

impl DriverSpec {
    pub fn scheduler_limits(&self) -> SchedulerLimits {
        self.limits
    }
}

/// Opening a device: one function per backend, all answering the contract.
///
/// Free functions rather than `DriverBackend::*_create` constructors, because
/// there is no `DriverBackend` type to hang them off any more — and because
/// what they have in common is the ANSWER, not the receiver.
///
/// Each takes the boot bytes it is given and reads what it needs. The boot
/// TOML is the engine's format on purpose: a driver that parsed it would be
/// the second thing entitled to an opinion about the file's shape, and the
/// two would drift.
pub mod open {
    // Every function below is gated on a driver feature, so with none
    // selected -- which is how the workspace clippy gate builds this crate --
    // the import has no user and `-D warnings` refuses the crate.
    #[cfg(any(
        feature = "_driver-cuda",
        all(feature = "driver-metal", target_vendor = "apple")
    ))]
    use super::{DriverBackend, Result};

    /// Open a CUDA device.
    ///
    /// # Errors
    ///
    /// No device, or a boot config this driver refuses.
    #[cfg(feature = "_driver-cuda")]
    pub fn cuda(config_bytes: &[u8]) -> Result<DriverBackend> {
        Ok(Box::new(super::cuda::open(config_bytes)?))
    }

    /// Open one CUDA device per rank, as one driver.
    ///
    /// **`palo B-tp`: ONE RANK, AND IT REFUSES THE REST BY NAME.** The group
    /// this replaced was a leader shell and N follower shells behind one
    /// `Driver`, fanning a `Vec<ModelLoadDesc>` across a thread scope and
    /// cross-checking that the ranks agreed about the model. None of that
    /// shape survives: a rank is not a load
    /// ([`LoadRequest`](driver_api::LoadRequest) is one plan, `Shard::Cut` is
    /// in the plan), the shell states tp=1 in `weights.rs` — `StorageTarget::
    /// for_backend(Cuda, 0, 1)` — and no collective ever fires in v1
    /// (`serve.rs`'s "what v1 does not do"). A multi-rank launch is refused
    /// here rather than opened and served wrong.
    ///
    /// What the successor needs: a rank index and a width reaching
    /// `StorageTarget`, a `Driver` that owns N shells and drives their
    /// streams in lockstep, and NCCL ordering — decision 5, "collectives
    /// never elided; descriptor rank-replicated".
    ///
    /// # Errors
    ///
    /// An empty rank list, more than one rank, or a device that failed to
    /// open.
    #[cfg(feature = "_driver-cuda")]
    pub fn cuda_group(config_blobs: Vec<Vec<u8>>) -> Result<(DriverBackend, usize)> {
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

    /// Open the default Metal 4 device.
    ///
    /// # Errors
    ///
    /// No Metal 4 device, or a device whose queue could not be created.
    #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
    pub fn metal(config_bytes: &[u8]) -> Result<DriverBackend> {
        Ok(Box::new(super::metal::MetalDriver::create(config_bytes)?))
    }
}

// `settle_control` STOOD HERE. A seam whose memory is coherent — where
// `copy_kv` is a `memmove` — had nobody to hand a completion target to, so it
// published the terminal cell and notified the broker itself, and the comment
// on it recorded an 850-second hang from a seam that did one and not the
// other. Both halves are `crate::driver::verbs::settled` now, for every seam
// at once: the contract's control verbs answer `Result<()>` and the work is
// done when they return, so the completion the engine hands its waiters is
// one that is already settled. There is no half of it left to forget.

#[cfg(feature = "_driver-cuda")]
mod cuda;
// TARGET-GATED as well as feature-gated, and the reason has moved. It used to
// be that `driver-metal` was Apple-only at the crate level, so the feature
// alone was not a build that links. That crate names no Metal API any more —
// it is the dispatch layer, and it builds and tests on any OS — and the seam
// behind this gate has no shell to be Apple about either, so the module
// itself is portable and the test beside it passes on Linux with the target
// clause lifted. The gate stays because `worker`'s `DriverOptions::Metal`
// arm and its whole option struct are Apple-gated: ungating the door alone
// would open one nothing on this platform can reach. It comes off with the
// shell (`palo B-metal`), when there is something on the other side.
#[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
mod metal;
mod remote;

// `pub use cuda::CudaDriver` STOOD HERE. There is no such type: the `Driver`
// impl is `driver_cuda::Cuda`, in the crate that owns the device, and this
// module's CUDA arm is a boot-config reader that answers one
// (`backend::cuda::open`). Re-exporting the shell's own type through here
// would be this crate claiming a driver it does not implement.
#[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
pub use metal::MetalDriver;
pub use remote::{RemoteDisconnectHandle, RemoteDriver};

// One function DOES come through, and it is not a driver: `palo B3`'s
// envelope counter, which is the only observable of a negative (the token
// that did not travel to the host). See `cuda::envelopes_resolved`.
#[cfg(feature = "_driver-cuda")]
pub use cuda::envelopes_resolved;

struct DriverRegistration {
    spec: DriverSpec,
    /// The driver, until a scheduler claims it.
    ///
    /// `Option` for the HANDOFF and not for a registration that never had one:
    /// `register_driver_backend` always installs a driver, and
    /// [`take_driver_backend`] moves it out exactly once, into the
    /// `DriverLane` that owns it from then on. The spec stays behind because
    /// [`get_spec`] is read after the claim.
    ///
    /// There used to be a second constructor — `register_driver(spec)` — that
    /// installed `None` here, for a registration with a spec and no driver.
    /// Nothing called it. It was defined, re-exported from `crate::driver`,
    /// and had zero callers in the workspace, so the `None` it existed to
    /// produce was a state the registry could describe and never reach.
    backend: Option<DriverBackend>,
}

fn registry() -> &'static RwLock<Vec<Option<DriverRegistration>>> {
    static REGISTRY: OnceLock<RwLock<Vec<Option<DriverRegistration>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

pub fn register_driver_backend(mut spec: DriverSpec, backend: DriverBackend) -> usize {
    let mut drivers = registry().write().unwrap();
    let id = drivers.len();
    // The BACKEND states it, still — but through `device_facts`, which is
    // where a driver says what machine it is. A driver with no device of its
    // own (the remote seam) answers `None`, and host-pinned is the honest
    // reading of "these pages are not on a device of mine".
    spec.device_domain = backend
        .device_facts()
        .map_or(MemoryDomain::HostPinned, |facts| facts.domain);
    drivers.push(Some(DriverRegistration {
        spec,
        backend: Some(backend),
    }));
    id
}

pub fn get_spec(driver_id: usize) -> Result<DriverSpec> {
    registry()
        .read()
        .unwrap()
        .get(driver_id)
        .and_then(|d| d.as_ref().map(|r| r.spec.clone()))
        .ok_or_else(|| anyhow!("unknown driver {driver_id}"))
}

pub fn take_driver_backend(driver_id: usize) -> Result<DriverBackend> {
    let mut drivers = registry().write().unwrap();
    let Some(Some(driver)) = drivers.get_mut(driver_id) else {
        return Err(anyhow!("unknown driver {driver_id}"));
    };
    driver
        .backend
        .take()
        .ok_or_else(|| anyhow!("driver {driver_id} has no backend installed"))
}

pub fn unregister_driver(driver_id: usize) -> Result<()> {
    let mut drivers = registry().write().unwrap();
    let Some(slot) = drivers.get_mut(driver_id) else {
        return Err(anyhow!("unknown driver {driver_id}"));
    };
    slot.take();
    Ok(())
}

// The `tests` module STOOD HERE, and held exactly two things: that
// `settle_control` publishes before it notifies, and that every host-side
// seam calls it rather than minting a completion of its own. Both went with
// the helper and its three seams (see the note above). This module has no
// host-side seam left to make a claim about.
