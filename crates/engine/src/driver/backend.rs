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
    use super::{DriverBackend, Result};

    /// Open a CUDA device.
    ///
    /// # Errors
    ///
    /// No device, or a boot config this driver refuses.
    #[cfg(feature = "driver-cuda")]
    pub fn cuda(config_bytes: &[u8]) -> Result<DriverBackend> {
        Ok(Box::new(super::cuda::CudaDriver::create(config_bytes)?))
    }

    /// Open one CUDA device per rank, as one driver.
    ///
    /// # Errors
    ///
    /// Any rank's device failed to open.
    #[cfg(feature = "driver-cuda")]
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

    /// Open a Vulkan device.
    ///
    /// # Errors
    ///
    /// No Vulkan device, or no readable SPIR-V module directory.
    #[cfg(feature = "driver-vulkan")]
    pub fn vulkan(config_bytes: &[u8]) -> Result<DriverBackend> {
        Ok(Box::new(super::vulkan::VulkanDriver::create(config_bytes)?))
    }

    /// Open a WebGPU adapter.
    ///
    /// Needs no SDK, no loader and no vendor runtime -- only an adapter --
    /// which is the whole argument for this backend. There is no module
    /// directory to find, because `kernels-wgpu` ships its shaders in the
    /// rlib.
    ///
    /// # Errors
    ///
    /// No adapter, or an adapter whose storage-buffer limit cannot bind this
    /// build's attention kernels.
    #[cfg(feature = "driver-wgpu")]
    pub fn wgpu(config_bytes: &[u8]) -> Result<DriverBackend> {
        Ok(Box::new(super::wgpu::WgpuDriver::create(config_bytes)?))
    }
}

#[cfg(feature = "driver-cuda")]
mod cuda;
#[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
mod metal;
// NOT target-gated, unlike the Metal seam above it. Vulkan is a loader rather
// than a platform: the same crate builds and runs on Linux, Windows and
// Android, so the feature is the whole gate.
mod remote;
#[cfg(feature = "driver-vulkan")]
mod vulkan;
// Not target-gated either, and for a wider reason than Vulkan's. Vulkan is a
// loader; this is a loader with no C in it. `driver-wgpu`'s whole closure is
// pure Rust -- no SDK, no ICD, no `-sys` crate -- so the feature can be turned
// on wherever this crate builds at all, which is not something the other three
// seams can say.
#[cfg(feature = "driver-wgpu")]
mod wgpu;

#[cfg(feature = "driver-cuda")]
pub use cuda::CudaDriver;
#[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
pub use metal::MetalDriver;
pub use remote::{RemoteDisconnectHandle, RemoteDriver};
#[cfg(feature = "driver-vulkan")]
pub use vulkan::VulkanDriver;
#[cfg(feature = "driver-wgpu")]
pub use wgpu::WgpuDriver;

struct DriverRegistration {
    spec: DriverSpec,
    backend: Option<DriverBackend>,
}

fn registry() -> &'static RwLock<Vec<Option<DriverRegistration>>> {
    static REGISTRY: OnceLock<RwLock<Vec<Option<DriverRegistration>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

pub fn register_driver(spec: DriverSpec) -> usize {
    let mut drivers = registry().write().unwrap();
    let id = drivers.len();
    drivers.push(Some(DriverRegistration {
        spec,
        backend: None,
    }));
    id
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

#[cfg(test)]
mod tests {
    /// Every verb of the wgpu seam is reachable through `dyn Driver`, with no
    /// adapter, and each one either serves or refuses in words.
    ///
    /// # What this is guarding
    ///
    /// An `impl` that compiles proves a method EXISTS. It does not prove the
    /// method does anything: `todo!()`, `unimplemented!()` and a silent
    /// `Ok(())` all type-check, and the last of those is the one that matters
    /// here -- a verb that answered success without doing the work would take
    /// a KV copy the scheduler then believes happened. So every verb is
    /// called, through the trait object, and its answer is read.
    ///
    /// It matters more than it did: four of these verbs are DEFAULT methods
    /// now, so this is also the test that a driver which overrides none of
    /// them still refuses by name rather than silently succeeding.
    ///
    /// No adapter is opened, which is what makes this a CI test rather than a
    /// GPU one: ten of the fourteen verbs are host code. `encode` and
    /// `copy_state` refuse by name; `launch`, `copy_kv` and `resize_pool`
    /// refuse because there is no shell yet, which is the same refusal a real
    /// driver gives between `create` and `load_model`; the registry five are
    /// served.
    #[cfg(feature = "driver-wgpu")]
    #[test]
    fn the_wgpu_variant_answers_every_verb_without_a_device() {
        use ::driver_api::FrameLaunchOutcome;

        // The trait object, not the concrete driver: what is under test is
        // that every verb is reachable THROUGH the contract.
        let mut backend: super::DriverBackend =
            Box::new(super::wgpu::WgpuDriver::without_adapter());

        assert_eq!(backend.kind(), "wgpu", "the seam's name in the handshake");
        assert_eq!(
            backend.device_domain(),
            ::driver_api::PIE_MEMORY_DOMAIN_WEBGPU_DEVICE
        );
        assert!(
            backend.export_kv_handle().is_none(),
            "there is no cross-process sharing path in WebGPU to export"
        );
        // In-process: this is a no-op that must not panic.
        backend.disconnect("a message nobody is listening to");

        // THE REGISTRY FIVE, which are alive before any model. A program with
        // one epilogue stage is the emptiest package the registry accepts.
        let program = backend
            .register_program(&::driver_api::ProgramRegistration {
                program_hash: 0x_c0_ff_ee,
                launch: ::driver_api::plan::LaunchPackage {
                    channels: vec![::driver_api::plan::LaunchChannel {
                        id: 9,
                        capacity: 2,
                        dtype: ::driver_api::PIE_CHANNEL_DTYPE_F32,
                        flags: ::driver_api::PIE_CHANNEL_HOST_VISIBLE,
                        extern_dir: -1,
                        readiness: ::driver_api::PIE_READINESS_UNTOUCHED,
                        shape: vec![4],
                        extern_name: vec![],
                    }],
                    stages: vec![::driver_api::plan::LaunchStage {
                        kind: 3,
                        ..Default::default()
                    }],
                    plans: vec![::driver_api::plan::LaunchStagePlan::default()],
                    ..Default::default()
                },
                ..Default::default()
            })
            .expect("a one-stage program registers before any model is loaded");

        let channel = backend
            .register_channel(&::driver_api::ChannelRegistrationPlan {
                driver_id: 4,
                channel_id: 9,
                shape: vec![4],
                dtype: ::driver_api::PIE_CHANNEL_DTYPE_F32,
                host_role: ::driver_api::PIE_CHANNEL_HOST_ROLE_WRITER,
                seeded: false,
                extern_dir: ::driver_api::PIE_CHANNEL_EXTERN_NONE,
                capacity: 2,
                reader_wait_id: 11,
                writer_wait_id: 12,
                extern_name: Vec::new(),
            })
            .expect("a channel matching the program's declaration");
        assert_eq!(
            (
                channel.driver_id,
                channel.reader_wait_id,
                channel.writer_wait_id
            ),
            (4, 11, 12),
            "the three fields the driver does not answer came from the wrong \
             side, so this channel signals nobody"
        );

        let instance = backend
            .bind_instance(&::driver_api::InstanceBindingPlan {
                driver_id: 4,
                program_id: program,
                // Zero is "any", not instance zero.
                requested_instance_id: 0,
                pacing_wait_id: 13,
                channel_ids: vec![9],
                seed_values: Vec::new(),
                geometry_class: ::driver_api::geometry::GeometryClass::Host,
            })
            .expect("an instance over the one channel");
        assert_eq!(instance.program_id, program);
        assert_eq!(instance.pacing_wait_id, 13);

        backend
            .close_instance(instance.instance_id)
            .expect("closes");
        backend.close_channel(9).expect("closes");

        // THE REFUSALS. Each is read, not merely counted: a refusal whose
        // words do not name the field leaves the caller to guess.
        //
        // `launch` is served on this seam and cannot be served HERE: a frame
        // is fired over a model's cache, and there is no model. So the words
        // to check are the ones every pre-load verb gives -- which verb, and
        // what it was waiting for -- and not the name of a missing feature.
        let said = backend
            .launch(&::driver_api::FrameSubmission {
                instance_ids: vec![1],
                kv_translation: vec![0],
                kv_translation_indptr: vec![0, 1],
                required_kv_pages: 1,
                steps: Vec::new(),
            })
            .map(|outcome| match outcome {
                FrameLaunchOutcome::Launched(_) => "launched",
                FrameLaunchOutcome::Exhausted => "exhausted",
                FrameLaunchOutcome::Impossible => "impossible",
            })
            .expect_err("a frame before a load has no cache to fire over");
        let said = said.to_string();
        assert!(
            said.contains("launch") && said.contains("load_model"),
            "the refusal names the verb and what it was waiting for: {said}"
        );

        // `let Err(..) else` rather than `expect_err`, here and below: the
        // Ok type is `SubmissionCompletion`, which is deliberately not
        // `Debug` -- it owns a raw terminal cell.
        let Err(said) = backend.encode(&mut ::driver_api::MediaEncodePlan::default()) else {
            panic!("there is no encode entry point in this driver")
        };
        let said = said.to_string();
        assert!(said.contains("encode"), "{said}");

        let Err(said) = backend.copy_state(&::driver_api::StateCopyPlan::default()) else {
            panic!("no model this driver serves holds a recurrent state")
        };
        let said = said.to_string();
        assert!(said.contains("recurrent"), "{said}");

        // AND THE TWO THAT NEED A MODEL. Before `load_model` there is no pool
        // to move a page within, and the answer says which verb was early
        // rather than reporting a success nothing performed.
        let Err(said) = backend.copy_kv(&::driver_api::KvCopyPlan {
            src_domain: ::driver_api::PIE_MEMORY_DOMAIN_WEBGPU_DEVICE,
            dst_domain: ::driver_api::PIE_MEMORY_DOMAIN_WEBGPU_DEVICE,
            src_page_ids: vec![0],
            dst_page_ids: vec![1],
            ..Default::default()
        }) else {
            panic!("a copy before a load has no pool to copy within")
        };
        let said = said.to_string();
        assert!(
            said.contains("copy_kv") && said.contains("load_model"),
            "the refusal names the verb and what it was waiting for: {said}"
        );

        let Err(said) = backend.resize_pool(&::driver_api::PoolResizePlan {
            pool_id: ::driver_api::PIE_ELASTIC_POOL_KV,
            target_pages: 8,
            ..Default::default()
        }) else {
            panic!("a resize before a load has no pool to resize")
        };
        let said = said.to_string();
        assert!(said.contains("resize_pool"), "{said}");

        // `load_model` with no descriptor, which is the one shape of it that
        // needs no checkpoint on disk.
        let said = backend
            .load_model(Vec::new())
            .expect_err("zero descriptors is not one model")
            .to_string();
        assert!(
            said.contains("descriptor"),
            "the refusal counts what it was given: {said}"
        );
    }
}
