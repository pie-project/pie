use anyhow::Result;

use crate::driver::completion::Completion;
#[cfg(feature = "driver-cuda")]
use crate::driver::ffi::cuda::CudaDriver;
#[cfg(feature = "driver-metal")]
use crate::driver::ffi::metal::MetalDriver;
use crate::driver::frame::{
    BoundInstance, ChannelRegistrationPlan, InstanceBindingPlan, KvCopyPlan, LaunchSubmission,
    PoolResizePlan, ProgramRegistration, RegisteredChannel, StateCopyPlan,
};

pub trait LocalDriver: Send {
    fn capabilities(&self) -> &pie_driver_abi::DriverCapabilities;
    fn register_program(&mut self, desc: &ProgramRegistration) -> Result<u64>;
    fn register_channel(&mut self, desc: &ChannelRegistrationPlan) -> Result<RegisteredChannel>;
    fn bind_instance(&mut self, desc: &InstanceBindingPlan) -> Result<BoundInstance>;
    fn launch(&mut self, desc: &LaunchSubmission) -> Result<Completion>;
    fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<Completion>;
    fn copy_state(&mut self, desc: &StateCopyPlan) -> Result<Completion>;
    fn resize_pool(&mut self, desc: &PoolResizePlan) -> Result<Completion>;
    fn close_instance(&mut self, id: u64) -> Result<()>;
    fn close_channel(&mut self, id: u64) -> Result<()>;
}

pub enum NativeDriver {
    Dummy(crate::driver::registry::DummyLocalDriver),
    #[cfg(feature = "driver-cuda")]
    Cuda(CudaDriver),
    #[cfg(feature = "driver-metal")]
    Metal(MetalDriver),
}

impl NativeDriver {
    pub fn dummy(
        options: pie_driver_dummy_lib::DummyDriverOptions,
    ) -> Result<(Self, pie_driver_abi::DriverCapabilities)> {
        let driver = crate::driver::registry::DummyLocalDriver::new(options);
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
}

impl LocalDriver for NativeDriver {
    fn capabilities(&self) -> &pie_driver_abi::DriverCapabilities {
        match self {
            Self::Dummy(driver) => driver.capabilities(),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.capabilities(),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.capabilities(),
        }
    }

    fn register_program(&mut self, desc: &ProgramRegistration) -> Result<u64> {
        match self {
            Self::Dummy(driver) => driver.register_program(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.register_program(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.register_program(desc),
        }
    }

    fn register_channel(&mut self, desc: &ChannelRegistrationPlan) -> Result<RegisteredChannel> {
        match self {
            Self::Dummy(driver) => driver.register_channel(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.register_channel(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.register_channel(desc),
        }
    }

    fn bind_instance(&mut self, desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        match self {
            Self::Dummy(driver) => driver.bind_instance(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.bind_instance(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.bind_instance(desc),
        }
    }

    fn launch(&mut self, desc: &LaunchSubmission) -> Result<Completion> {
        match self {
            Self::Dummy(driver) => driver.launch(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.launch(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.launch(desc),
        }
    }

    fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<Completion> {
        match self {
            Self::Dummy(driver) => driver.copy_kv(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.copy_kv(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.copy_kv(desc),
        }
    }

    fn copy_state(&mut self, desc: &StateCopyPlan) -> Result<Completion> {
        match self {
            Self::Dummy(driver) => driver.copy_state(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.copy_state(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.copy_state(desc),
        }
    }

    fn resize_pool(&mut self, desc: &PoolResizePlan) -> Result<Completion> {
        match self {
            Self::Dummy(driver) => driver.resize_pool(desc),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.resize_pool(desc),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.resize_pool(desc),
        }
    }

    fn close_instance(&mut self, id: u64) -> Result<()> {
        match self {
            Self::Dummy(driver) => driver.close_instance(id),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.close_instance(id),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.close_instance(id),
        }
    }

    fn close_channel(&mut self, id: u64) -> Result<()> {
        match self {
            Self::Dummy(driver) => driver.close_channel(id),
            #[cfg(feature = "driver-cuda")]
            Self::Cuda(driver) => driver.close_channel(id),
            #[cfg(feature = "driver-metal")]
            Self::Metal(driver) => driver.close_channel(id),
        }
    }
}
