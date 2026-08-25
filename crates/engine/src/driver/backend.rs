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
use driver_api::{DeviceDomain, Driver};

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
    pub device_geometry_port_mask: u32,
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
    pub device_domain: DeviceDomain,
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
        Ok(Box::new(super::cuda::CudaDriver::create(config_bytes)?))
    }

    /// Open one CUDA device per rank, as one driver.
    ///
    /// # Errors
    ///
    /// Any rank's device failed to open.
    #[cfg(feature = "_driver-cuda")]
    pub fn cuda_group(config_blobs: Vec<Vec<u8>>) -> Result<(DriverBackend, usize)> {
        let (driver, ranks) = super::cuda::CudaDriver::create_group(config_blobs)?;
        Ok((Box::new(driver), ranks))
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

/// Settle a control-plane submission a HOST-SIDE seam has already finished.
///
/// A seam whose memory is coherent — where `copy_kv` is a `memmove` — has
/// nobody to hand the completion target to, so it publishes the terminal cell
/// and notifies the broker itself. `cuda` and `remote` never use this and must
/// not: they are asynchronous, so each hands the whole target to whatever
/// finishes the work.
///
/// BOTH HALVES, IN THAT ORDER. The 850-second hang this exists to prevent came
/// from a seam that minted a `control_completion` and dropped the target: it
/// parked a real `pie run`, with the scheduler's watchdog naming it exactly —
/// `in_flight_control: KV copy pipeline Some(..) settled=false`. Publishing
/// without notifying trips the engine's own check on the way out.
#[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
pub(crate) fn settle_control(
    broker: &driver_api::CompletionBroker,
) -> driver_api::SubmissionCompletion {
    let (raw, completion) = broker.control_completion(1);
    if !raw.terminal_cell.is_null() {
        // SAFETY: the broker owns this cell for the life of the completion it
        // was minted with, and `publish` is a release store into an
        // `AtomicU32` the engine only ever reads.
        unsafe {
            (*raw.terminal_cell).publish(driver_api::PIE_TERMINAL_OUTCOME_SUCCESS);
        }
    }
    broker.notify(completion.wait_id(), raw.target_epoch);
    completion
}

#[cfg(feature = "_driver-cuda")]
mod cuda;
// TARGET-GATED as well as feature-gated, unlike the seams that stood beside it:
// `driver-metal` is Apple-only at the crate level, so the feature alone is not
// a build that links. Vulkan is a loader and wgpu is pure Rust; neither needed
// this, and both are still out at P5.
#[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
mod metal;
mod remote;

#[cfg(feature = "_driver-cuda")]
pub use cuda::CudaDriver;
#[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
pub use metal::MetalDriver;
pub use remote::{RemoteDisconnectHandle, RemoteDriver};

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
    spec.device_domain = backend.device_domain();
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
