//! Driver specs, backend storage (the `DriverId` registry), and concrete
//! backend dispatch. Scheduler-handle lookup lives in the scheduler layer;
//! this module keeps only what the driver ABI itself owns: `DriverSpec`
//! plus the optional `DriverBackend` it's paired with.

use std::sync::{OnceLock, RwLock};

use anyhow::{Result, anyhow};

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

#[cfg(feature = "driver-cuda")]
pub use cuda::CudaDriver;
#[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
pub use metal::MetalDriver;
pub use remote::{RemoteDisconnectHandle, RemoteDriver};
#[cfg(feature = "driver-vulkan")]
pub use vulkan::VulkanDriver;

use crate::driver::channel::RegisteredChannel;
use crate::driver::command::{
    ChannelRegistrationPlan, KvCopyPlan, MediaEncodePlan, PoolResizePlan, ProgramRegistration,
    StateCopyPlan,
};
use crate::driver::completion::SubmissionCompletion;
use crate::driver::instance::{BoundInstance, InstanceBindingPlan};
use crate::driver::submission::FrameSubmission;

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
    pub device_domain: ::driver_api::PieMemoryDomain,
}

impl DriverSpec {
    pub fn scheduler_limits(&self) -> SchedulerLimits {
        self.limits
    }
}

/// Outcome of a frame launch post: admission is folded into the launch call
/// (ABI v14), so a post either enters the driver with one completion, or
/// reports why it cannot right now.
pub enum FrameLaunchOutcome {
    /// The frame was admitted and posted; one completion settles it.
    Launched(SubmissionCompletion),
    /// Admission is full right now; the engine re-posts later.
    Exhausted,
    /// The frame can never fit within the driver's physical budget ceiling.
    Impossible,
}

// Fires on `Remote`, and only once a small variant exists to compare it
// against: `RemoteDriver` is 560 bytes inline and the boxed Vulkan arm is 8.
// Boxing `Remote` too is the real answer and is not this change's to make --
// it is eighteen call sites in code no backend here owns.
#[cfg_attr(
    feature = "driver-vulkan",
    expect(
        clippy::large_enum_variant,
        reason = "\
    the large variant is `Remote`, which this change does not own"
    )
)]
pub enum DriverBackend {
    #[cfg(feature = "driver-cuda")]
    Cuda(CudaDriver),
    #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
    Metal(MetalDriver),
    /// Boxed, and it is the only variant that is. `VulkanDriver` carries the
    /// device, the queue, the pipeline cache and the module map inline --
    /// five times the next-largest variant -- and every `DriverBackend` in
    /// the registry, on every backend, would pay that width.
    #[cfg(feature = "driver-vulkan")]
    Vulkan(Box<VulkanDriver>),
    Remote(RemoteDriver),
}

impl DriverBackend {
    pub fn kind(&self) -> &'static str {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(_) => "cuda",
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(_) => "metal",
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(_) => "vulkan",
            Self::Remote(_) => "remote",
        }
    }

    #[cfg(feature = "driver-cuda")]
    pub fn cuda_create(config_bytes: &[u8]) -> Result<(Self, ::driver_api::DeviceFacts)> {
        let (driver, facts) = CudaDriver::create(config_bytes)?;
        Ok((Self::Cuda(driver), facts))
    }

    /// Open the Metal device. A library call: no ABI crossing, because the
    /// driver on the other side is Rust.
    ///
    /// # Errors
    ///
    /// No Metal 4 device.
    #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
    pub fn metal_create(config_bytes: &[u8]) -> Result<(Self, ::driver_api::DeviceFacts)> {
        let (driver, facts) = MetalDriver::create(config_bytes)?;
        Ok((Self::Metal(driver), facts))
    }

    /// Open a Vulkan device. A library call: no ABI crossing, because the
    /// driver it talks to is Rust.
    ///
    /// # Errors
    ///
    /// No Vulkan device, or no readable SPIR-V module directory.
    #[cfg(feature = "driver-vulkan")]
    pub fn vulkan_create(config_bytes: &[u8]) -> Result<(Self, ::driver_api::DeviceFacts)> {
        let (driver, facts) = VulkanDriver::create(config_bytes)?;
        Ok((Self::Vulkan(Box::new(driver)), facts))
    }

    #[cfg(feature = "driver-cuda")]
    pub fn cuda_group_create(
        config_blobs: Vec<Vec<u8>>,
    ) -> Result<(Self, Vec<::driver_api::DeviceFacts>)> {
        let (driver, facts) = CudaDriver::create_group(config_blobs)?;
        Ok((Self::Cuda(driver), facts))
    }

    pub fn load_model(
        &mut self,
        descs: Vec<::driver_api::ModelLoadDesc>,
    ) -> Result<::driver_api::DriverCapabilities> {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.load_model(descs),
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(driver) => driver.load_model(descs),
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(driver) => driver.load_model(descs),
            Self::Remote(driver) => driver.load_model(descs),
        }
    }

    /// The backend whose kernels this driver wants the host to generate, or
    /// `None` when it generates its own (or needs none). The variant already
    /// says which native backend it is, so this needs no capability round-trip.
    fn codegen_backend(&self) -> Option<&'static str> {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(_) => Some("cuda"),
            // A remote driver's own backend does its generation on the far
            // side.
            _ => None,
        }
    }

    pub fn register_program(&mut self, desc: &ProgramRegistration) -> Result<u64> {
        // Attach whatever this driver reads and the caller did not already
        // supply. Generation is memoised per program per backend, so a
        // re-registration costs a lookup.
        let registered = crate::pipeline::program::lookup(desc.program_hash);
        let codegen_backend = self.codegen_backend();

        // The driver no longer carries an emitter, so a fused region with no
        // host source is a registration failure rather than a slower path.
        let emitted = codegen_backend
            .filter(|_| desc.emitted_kernels.is_empty())
            .and_then(|backend| {
                registered
                    .as_ref()
                    .and_then(|program| program.emitted(backend))
            });

        // The region analysis is the other half of the CUDA emitter's own
        // contract -- which regions bind, and how the kernel's intrinsic side
        // tables are laid out -- so it only means anything to a driver running
        // those kernels.
        let region_analysis = if desc.region_analysis.is_empty() && codegen_backend == Some("cuda")
        {
            registered
                .as_ref()
                .map(|program| program.region_analysis())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let owned;
        let desc = if emitted.is_some() || !region_analysis.is_empty() {
            let mut next = desc.clone();
            if let Some(emitted) = emitted {
                next.emitter_version = emitted.emitter_version;
                next.emitted_kernels = emitted
                    .kernels
                    .iter()
                    .map(|kernel| ::driver_api::EmittedKernel {
                        kind: kernel.kind,
                        stage_index: kernel.stage_index,
                        region_index: kernel.region_index,
                        entry_name: kernel.entry_name.clone(),
                        source: kernel.source.clone(),
                        error: kernel.error.clone(),
                    })
                    .collect();
            }
            if !region_analysis.is_empty() {
                next.region_analysis = region_analysis;
            }
            owned = next;
            &owned
        } else {
            desc
        };
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.register_program(desc),
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(driver) => driver.register_program(desc),
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(driver) => driver.register_program(desc),
            Self::Remote(driver) => driver.register_program(desc),
        }
    }

    pub fn register_channel(
        &mut self,
        desc: &ChannelRegistrationPlan,
    ) -> Result<RegisteredChannel> {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.register_channel(desc),
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(driver) => driver.register_channel(desc),
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(driver) => driver.register_channel(desc),
            Self::Remote(driver) => driver.register_channel(desc),
        }
    }

    pub fn bind_instance(&mut self, desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.bind_instance(desc),
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(driver) => driver.bind_instance(desc),
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(driver) => driver.bind_instance(desc),
            Self::Remote(driver) => driver.bind_instance(desc),
        }
    }

    /// Post one sealed frame. Admission is folded into the call: the driver
    /// evaluates the frame-union demand and either admits (one completion
    /// settles the whole frame) or reports Exhausted/Impossible without side
    /// effects.
    pub fn launch(&mut self, desc: &FrameSubmission) -> Result<FrameLaunchOutcome> {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.launch(desc),
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(driver) => driver.launch(desc),
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(driver) => driver.launch(desc),
            Self::Remote(driver) => driver.launch(desc),
        }
    }

    pub fn encode(&mut self, plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.encode(plan),
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(driver) => driver.encode(plan),
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(driver) => driver.encode(plan),
            Self::Remote(driver) => driver.encode(plan),
        }
    }

    pub fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<SubmissionCompletion> {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.copy_kv(desc),
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(driver) => driver.copy_kv(desc),
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(driver) => driver.copy_kv(desc),
            Self::Remote(driver) => driver.copy_kv(desc),
        }
    }

    pub fn copy_state(&mut self, desc: &StateCopyPlan) -> Result<SubmissionCompletion> {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.copy_state(desc),
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(driver) => driver.copy_state(desc),
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(driver) => driver.copy_state(desc),
            Self::Remote(driver) => driver.copy_state(desc),
        }
    }

    pub fn resize_pool(&mut self, desc: &PoolResizePlan) -> Result<SubmissionCompletion> {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.resize_pool(desc),
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(driver) => driver.resize_pool(desc),
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(driver) => driver.resize_pool(desc),
            Self::Remote(driver) => driver.resize_pool(desc),
        }
    }

    pub fn close_instance(&mut self, id: u64) -> Result<()> {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.close_instance(id),
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(driver) => driver.close_instance(id),
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(driver) => driver.close_instance(id),
            Self::Remote(driver) => driver.close_instance(id),
        }
    }

    pub fn close_channel(&mut self, id: u64) -> Result<()> {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.close_channel(id),
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(driver) => driver.close_channel(id),
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(driver) => driver.close_channel(id),
            Self::Remote(driver) => driver.close_channel(id),
        }
    }

    pub fn export_kv_handle(&self) -> Option<::driver_api::KvHandle> {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.export_kv_handle(),
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(driver) => driver.export_kv_handle(),
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(driver) => driver.export_kv_handle(),
            Self::Remote(_) => None,
        }
    }

    /// Which memory this backend's KV pages live in.
    ///
    /// Answered by the variant, which is the only place that knows. See
    /// [`DriverSpec::device_domain`] for what was there before and what it
    /// cost.
    pub fn device_domain(&self) -> ::driver_api::PieMemoryDomain {
        match self {
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(_) => ::driver_api::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
            // `METAL_PRIVATE` and not `METAL_SHARED`: the shared tag is for
            // memory the CPU addresses directly, and a KV page is a private
            // buffer the driver copies through its own encoder.
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(_) => ::driver_api::PIE_MEMORY_DOMAIN_METAL_PRIVATE,
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(_) => ::driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE,
            // The domain is the REMOTE driver's and this side cannot see it.
            // CUDA is what every plan carried before this method existed, so
            // answering it here changes nothing for the one backend that was
            // ever right by accident, and leaves the question where it
            // belongs: on the ABI, which does not carry the answer yet.
            Self::Remote(_) => ::driver_api::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
        }
    }

    pub fn disconnect(&self, message: impl Into<String>) {
        // `Remote` is the only variant this question has ever been asked of,
        // and since the interpreter backend went it is the only one LEFT that
        // is not feature-gated -- so the pattern is irrefutable. Written as a
        // `match` rather than a bare call so a second ungated variant lands
        // here as a non-exhaustive arm instead of silently sharing this one.
        match self {
            Self::Remote(driver) => driver.disconnect(message),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(_) => {}
            #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
            Self::Metal(_) => {}
            // In-process, like the two above it: there is no connection to
            // drop. A Vulkan device that has gone away is a lost device at
            // the next submit, which the driver reports there.
            #[cfg(feature = "driver-vulkan")]
            Self::Vulkan(_) => {}
        }
    }
}

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
    /// Every backend names a domain, and no two name the same one.
    ///
    /// # What this is guarding
    ///
    /// `DriverSpec::device_domain` is stamped onto every `KvCopyPlan` the
    /// scheduler builds, and it replaced a hardcoded
    /// `PIE_MEMORY_DOMAIN_CUDA_DEVICE` at nine sites. A driver checks the tag
    /// -- `driver-vulkan`'s `Pool::copy_plan` refuses a plan whose ends are
    /// not both `VULKAN_DEVICE` -- so a backend that answered somebody else's
    /// domain would be refused on every prefix-cache hit and every swap, and
    /// a backend that answered a domain nobody validates would be accepted
    /// into a pool it does not own.
    ///
    /// Distinctness is the property that matters: the tag is only useful as a
    /// discriminator. `Remote` is excluded because it deliberately answers
    /// CUDA -- the remote side's real domain does not cross the ABI, and CUDA
    /// is what every plan carried before this existed.
    /// A REGISTERED Vulkan backend puts its own domain on its spec.
    ///
    /// The test above compares constants, which cannot see a match arm
    /// returning the wrong one. This goes through the path the scheduler uses
    /// -- construct the backend, register it, read the spec back -- so
    /// answering CUDA for `Self::Vulkan` fails here.
    ///
    /// Needs a device and the module directory `kernels-vulkan` built, so it
    /// skips when `PIE_KERNELS_VULKAN_SPV_DIR` is unset or no Vulkan device
    /// answers. Skipping is stated rather than silent.
    #[cfg(feature = "driver-vulkan")]
    #[test]
    fn a_registered_vulkan_backend_carries_the_vulkan_domain() {
        if std::env::var("PIE_KERNELS_VULKAN_SPV_DIR").is_err() {
            eprintln!("skipped: PIE_KERNELS_VULKAN_SPV_DIR is unset");
            return;
        }
        let Ok((backend, _facts)) = super::DriverBackend::vulkan_create(b"") else {
            eprintln!("skipped: no Vulkan device answered");
            return;
        };
        assert_eq!(
            backend.device_domain(),
            ::driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE
        );
        // `register_driver_backend` overwrites whatever the caller wrote, so
        // the literal here is deliberately the wrong one.
        let id = super::register_driver_backend(
            super::DriverSpec {
                num_kv_pages: 1,
                limits: super::SchedulerLimits {
                    max_forward_requests: 1,
                    max_forward_tokens: 1,
                    max_page_refs: 1,
                },
                device_geometry_port_mask: 0,
                device_domain: ::driver_api::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
            },
            backend,
        );
        let spec = super::get_spec(id).expect("the driver just registered");
        assert_eq!(
            spec.device_domain,
            ::driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE,
            "the spec carries the backend's domain, not the caller's"
        );
        assert_eq!(
            crate::scheduler::device_domain(id),
            ::driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE,
            "and the scheduler reads the same answer"
        );
        let _ = super::unregister_driver(id);
    }

    #[test]
    #[allow(clippy::vec_init_then_push, reason = "the entries are cfg-gated")]
    fn each_backend_names_its_own_memory() {
        // Built by pushing rather than as a literal because the entries are
        // feature-gated; `vec![]` cannot carry a `cfg` per element.
        let mut seen: Vec<(&str, ::driver_api::PieMemoryDomain)> = Vec::new();
        #[cfg(feature = "driver-cuda")]
        seen.push(("cuda", ::driver_api::PIE_MEMORY_DOMAIN_CUDA_DEVICE));
        #[cfg(all(feature = "driver-metal", target_vendor = "apple"))]
        seen.push(("metal", ::driver_api::PIE_MEMORY_DOMAIN_METAL_PRIVATE));
        #[cfg(feature = "driver-vulkan")]
        seen.push(("vulkan", ::driver_api::PIE_MEMORY_DOMAIN_VULKAN_DEVICE));

        for (name, domain) in &seen {
            assert!(
                ::driver_api::pie_memory_domain_is_valid(*domain),
                "{name} names domain {domain}, which the ABI does not define"
            );
            assert_ne!(
                *domain,
                ::driver_api::PIE_MEMORY_DOMAIN_HOST_PINNED,
                "{name} names host memory for its device pages"
            );
        }
        for (i, (a, da)) in seen.iter().enumerate() {
            for (b, db) in &seen[i + 1..] {
                assert_ne!(da, db, "{a} and {b} both claim domain {da}");
            }
        }
    }
}
