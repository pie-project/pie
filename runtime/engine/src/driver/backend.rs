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
mod remote;

#[cfg(feature = "driver-cuda")]
pub use cuda::CudaDriver;
pub use dummy::DummyDriver;
#[cfg(feature = "driver-metal")]
pub use metal::MetalDriver;
pub use remote::{RemoteDisconnectHandle, RemoteDriver};

use crate::driver::channel::RegisteredChannel;
use crate::driver::command::{
    ChannelRegistrationPlan, KvCopyPlan, MediaEncodePlan, PoolResizePlan, ProgramRegistration,
    StateCopyPlan,
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
    pub device_geometry_port_mask: u32,
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
    Remote(RemoteDriver),
}

impl DriverBackend {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Dummy(_) => "dummy",
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(_) => "cuda",
            #[cfg(feature = "driver-metal")]
            Self::Metal(_) => "metal",
            Self::Remote(_) => "remote",
        }
    }

    pub fn dummy(
        options: pie_driver_dummy_lib::DummyDriverOptions,
    ) -> Result<(Self, pie_driver_abi::DeviceFacts)> {
        let driver = DummyDriver::new(options);
        let facts = driver.device_facts().clone();
        Ok((Self::Dummy(driver), facts))
    }

    #[cfg(feature = "driver-cuda")]
    pub fn cuda_create(config_bytes: &[u8]) -> Result<(Self, pie_driver_abi::DeviceFacts)> {
        let (driver, facts) = CudaDriver::create(config_bytes)?;
        Ok((Self::Cuda(driver), facts))
    }

    #[cfg(feature = "driver-cuda")]
    pub fn cuda_group_create(
        config_blobs: Vec<Vec<u8>>,
    ) -> Result<(Self, Vec<pie_driver_abi::DeviceFacts>)> {
        let (driver, facts) = CudaDriver::create_group(config_blobs)?;
        Ok((Self::Cuda(driver), facts))
    }

    #[cfg(feature = "driver-metal")]
    pub fn metal_create(config_bytes: &[u8]) -> Result<(Self, pie_driver_abi::DeviceFacts)> {
        let (driver, facts) = MetalDriver::create(config_bytes)?;
        Ok((Self::Metal(driver), facts))
    }

    pub fn load_model(
        &mut self,
        descs: Vec<pie_driver_abi::ModelLoadDesc>,
    ) -> Result<pie_driver_abi::DriverCapabilities> {
        match self {
            Self::Dummy(driver) => {
                let [desc] = descs.as_slice() else {
                    return Err(anyhow!(
                        "dummy model load requires exactly one descriptor, got {}",
                        descs.len()
                    ));
                };
                driver.load_model(desc)
            }
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.load_model(descs),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => {
                let [desc] = descs.as_slice() else {
                    return Err(anyhow!(
                        "metal model load requires exactly one descriptor, got {}",
                        descs.len()
                    ));
                };
                driver.load_model(desc)
            }
            Self::Remote(driver) => driver.load_model(descs),
        }
    }

    pub fn register_program(&mut self, desc: &ProgramRegistration) -> Result<u64> {
        match self {
            Self::Dummy(driver) => driver.register_program(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.register_program(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.register_program(desc),
            Self::Remote(driver) => driver.register_program(desc),
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
            Self::Remote(driver) => driver.register_channel(desc),
        }
    }

    pub fn bind_instance(&mut self, desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        match self {
            Self::Dummy(driver) => driver.bind_instance(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.bind_instance(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.bind_instance(desc),
            Self::Remote(driver) => driver.bind_instance(desc),
        }
    }

    pub fn launch(&mut self, desc: &LaunchSubmission) -> Result<SubmissionCompletion> {
        match self {
            Self::Dummy(driver) => driver.launch(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.launch(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.launch(desc),
            Self::Remote(driver) => driver.launch(desc),
        }
    }

    pub fn encode(&mut self, plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        match self {
            Self::Dummy(driver) => driver.encode(plan),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.encode(plan),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.encode(plan),
            Self::Remote(driver) => driver.encode(plan),
        }
    }

    pub fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<SubmissionCompletion> {
        match self {
            Self::Dummy(driver) => driver.copy_kv(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.copy_kv(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.copy_kv(desc),
            Self::Remote(driver) => driver.copy_kv(desc),
        }
    }

    pub fn copy_state(&mut self, desc: &StateCopyPlan) -> Result<SubmissionCompletion> {
        match self {
            Self::Dummy(driver) => driver.copy_state(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.copy_state(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.copy_state(desc),
            Self::Remote(driver) => driver.copy_state(desc),
        }
    }

    pub fn resize_pool(&mut self, desc: &PoolResizePlan) -> Result<SubmissionCompletion> {
        match self {
            Self::Dummy(driver) => driver.resize_pool(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.resize_pool(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.resize_pool(desc),
            Self::Remote(driver) => driver.resize_pool(desc),
        }
    }

    pub fn close_instance(&mut self, id: u64) -> Result<()> {
        match self {
            Self::Dummy(driver) => driver.close_instance(id),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.close_instance(id),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.close_instance(id),
            Self::Remote(driver) => driver.close_instance(id),
        }
    }

    pub fn close_channel(&mut self, id: u64) -> Result<()> {
        match self {
            Self::Dummy(driver) => driver.close_channel(id),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.close_channel(id),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.close_channel(id),
            Self::Remote(driver) => driver.close_channel(id),
        }
    }

    pub fn export_kv_handle(&self) -> Option<pie_driver_abi::KvHandle> {
        match self {
            Self::Dummy(driver) => driver.export_kv_handle(),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.export_kv_handle(),
            #[cfg(feature = "driver-metal")]
            Self::Metal(_) => None,
            Self::Remote(_) => None,
        }
    }

    pub fn disconnect(&self, message: impl Into<String>) {
        if let Self::Remote(driver) = self {
            driver.disconnect(message);
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
