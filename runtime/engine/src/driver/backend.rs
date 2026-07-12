//! Driver specs, backend storage (the `DriverId` registry), and concrete
//! backend dispatch. Scheduler-handle lookup lives in the scheduler layer;
//! this module keeps only what the driver ABI itself owns: `DriverSpec`
//! plus the optional `DriverBackend` it's paired with.

use std::sync::{OnceLock, RwLock};

use anyhow::{Result, anyhow};

#[cfg(feature = "driver-cuda")]
mod cuda;
mod dummy;
#[cfg(feature = "driver-metal")]
mod metal;

#[cfg(feature = "driver-cuda")]
pub use cuda::CudaDriver;
pub use dummy::DummyDriver;
#[cfg(feature = "driver-metal")]
pub use metal::MetalDriver;

use crate::driver::channel::RegisteredChannel;
use crate::driver::command::{
    ChannelRegistrationPlan, KvCopyPlan, PoolResizePlan, ProgramRegistration, StateCopyPlan,
};
use crate::driver::completion::SubmissionCompletion;
use crate::driver::instance::{BoundInstance, InstanceBindingPlan};
use crate::driver::submission::LaunchSubmission;

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
}

impl DriverSpec {
    pub fn scheduler_limits(&self) -> SchedulerLimits {
        self.limits
    }
}

pub enum DriverBackend {
    Dummy(DummyDriver),
    #[cfg(feature = "driver-cuda")]
    Cuda(CudaDriver),
    #[cfg(feature = "driver-metal")]
    Metal(MetalDriver),
}

impl DriverBackend {
    pub fn dummy(
        options: pie_driver_dummy_lib::DummyDriverOptions,
    ) -> Result<(Self, pie_driver_abi::DriverCapabilities)> {
        let driver = DummyDriver::new(options);
        let caps = driver.capabilities().clone();
        Ok((Self::Dummy(driver), caps))
    }

    #[cfg(feature = "driver-cuda")]
    pub fn cuda_create(config_bytes: &[u8]) -> Result<(Self, pie_driver_abi::DriverCapabilities)> {
        let (driver, caps) = CudaDriver::create(config_bytes)?;
        Ok((Self::Cuda(driver), caps))
    }

    #[cfg(feature = "driver-cuda")]
    pub fn cuda_group_create(
        config_blobs: Vec<Vec<u8>>,
    ) -> Result<(Self, pie_driver_abi::DriverCapabilities)> {
        let (driver, caps) = CudaDriver::create_group(config_blobs)?;
        Ok((Self::Cuda(driver), caps))
    }

    #[cfg(feature = "driver-metal")]
    pub fn metal_create(config_bytes: &[u8]) -> Result<(Self, pie_driver_abi::DriverCapabilities)> {
        let (driver, caps) = MetalDriver::create(config_bytes)?;
        Ok((Self::Metal(driver), caps))
    }

    pub fn capabilities(&self) -> &pie_driver_abi::DriverCapabilities {
        match self {
            Self::Dummy(driver) => driver.capabilities(),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.capabilities(),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.capabilities(),
        }
    }

    pub fn register_program(&mut self, desc: &ProgramRegistration) -> Result<u64> {
        match self {
            Self::Dummy(driver) => driver.register_program(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.register_program(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.register_program(desc),
        }
    }

    pub fn register_channel(
        &mut self,
        desc: &ChannelRegistrationPlan,
    ) -> Result<RegisteredChannel> {
        match self {
            Self::Dummy(driver) => driver.register_channel(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.register_channel(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.register_channel(desc),
        }
    }

    pub fn bind_instance(&mut self, desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        match self {
            Self::Dummy(driver) => driver.bind_instance(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.bind_instance(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.bind_instance(desc),
        }
    }

    pub fn launch(&mut self, desc: &LaunchSubmission) -> Result<SubmissionCompletion> {
        match self {
            Self::Dummy(driver) => driver.launch(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.launch(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.launch(desc),
        }
    }

    pub fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<SubmissionCompletion> {
        match self {
            Self::Dummy(driver) => driver.copy_kv(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.copy_kv(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.copy_kv(desc),
        }
    }

    pub fn copy_state(&mut self, desc: &StateCopyPlan) -> Result<SubmissionCompletion> {
        match self {
            Self::Dummy(driver) => driver.copy_state(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.copy_state(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.copy_state(desc),
        }
    }

    pub fn resize_pool(&mut self, desc: &PoolResizePlan) -> Result<SubmissionCompletion> {
        match self {
            Self::Dummy(driver) => driver.resize_pool(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.resize_pool(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.resize_pool(desc),
        }
    }

    pub fn close_instance(&mut self, id: u64) -> Result<()> {
        match self {
            Self::Dummy(driver) => driver.close_instance(id),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.close_instance(id),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.close_instance(id),
        }
    }

    pub fn close_channel(&mut self, id: u64) -> Result<()> {
        match self {
            Self::Dummy(driver) => driver.close_channel(id),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.close_channel(id),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.close_channel(id),
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

pub fn register_driver_backend(spec: DriverSpec, backend: DriverBackend) -> usize {
    let mut drivers = registry().write().unwrap();
    let id = drivers.len();
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
