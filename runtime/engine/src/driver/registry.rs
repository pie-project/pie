//! Driver specs and native-backend storage — the `DriverId` registry.
//! Scheduler-handle lookup lives in the scheduler layer; this module keeps
//! only what the driver ABI itself owns, `DriverSpec`/`NativeDriver`.

use std::sync::{OnceLock, RwLock};

use anyhow::{Result, anyhow};

use crate::driver::backend::NativeDriver;
use crate::driver::completion::{Completion, CompletionBroker};
use crate::driver::frame::{
    BoundInstance, ChannelRegistrationPlan, InstanceBindingPlan, KvCopyDescBorrow, KvCopyPlan,
    LaunchDescBorrow, LaunchSubmission, PoolResizePlan, ProgramRegistration, RegisteredChannel,
    StateCopyPlan,
};

pub struct DummyLocalDriver {
    inner: pie_driver_dummy_lib::DummyDriver,
    broker: CompletionBroker,
}

unsafe impl Send for DummyLocalDriver {}
unsafe impl Sync for DummyLocalDriver {}

impl DummyLocalDriver {
    pub fn new(options: pie_driver_dummy_lib::DummyDriverOptions) -> Self {
        let broker = CompletionBroker::new();
        let inner =
            pie_driver_dummy_lib::DummyDriver::with_runtime(options, broker.runtime_callbacks());
        Self { inner, broker }
    }

    pub fn capabilities(&self) -> &pie_driver_abi::DriverCapabilities {
        self.inner.capabilities()
    }
    pub fn register_program(&mut self, desc: &ProgramRegistration) -> Result<u64> {
        let borrowed = crate::driver::frame::ProgramDescBorrow::new(desc);
        self.inner.register_program(borrowed.as_raw())
    }
    pub fn register_channel(
        &mut self,
        desc: &ChannelRegistrationPlan,
    ) -> Result<RegisteredChannel> {
        let borrowed = crate::driver::frame::ChannelDescBorrow::new(desc);
        let binding = self.inner.register_channel(borrowed.as_raw())?;
        pie_driver_abi::validate_channel_endpoint_binding(&binding, borrowed.as_raw())
            .map_err(|error| anyhow!(error))?;
        Ok(RegisteredChannel {
            driver_id: desc.driver_id,
            binding,
            reader_wait_id: desc.reader_wait_id,
            writer_wait_id: desc.writer_wait_id,
        })
    }
    pub fn bind_instance(&mut self, desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        let borrowed = crate::driver::frame::InstanceDescBorrow::new(desc);
        let binding = self.inner.bind_instance(borrowed.as_raw())?;
        if let Err(error) = desc.validate_binding(&binding) {
            let _ = self.inner.close_instance(binding.instance_id);
            return Err(error);
        }
        Ok(BoundInstance::new(
            desc.driver_id,
            desc.program_id,
            binding,
            desc.pacing_wait_id,
        ))
    }
    pub fn launch(&mut self, desc: &LaunchSubmission) -> Result<Completion> {
        let (raw, completion) = self.broker.launch_completion(1);
        let borrowed = LaunchDescBorrow::from_submission(desc);
        self.inner.launch(borrowed.as_raw(), raw)?;
        Ok(completion)
    }
    pub fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<Completion> {
        let (raw, completion) = self.broker.pie_completion(1);
        let borrowed = KvCopyDescBorrow::new(desc);
        self.inner.copy_kv(borrowed.as_raw(), raw)?;
        Ok(completion)
    }
    pub fn copy_state(&mut self, desc: &StateCopyPlan) -> Result<Completion> {
        let (raw, completion) = self.broker.pie_completion(1);
        let borrowed = crate::driver::frame::StateCopyDescBorrow::new(desc);
        self.inner.copy_state(borrowed.as_raw(), raw)?;
        Ok(completion)
    }
    pub fn resize_pool(&mut self, desc: &PoolResizePlan) -> Result<Completion> {
        let (raw, completion) = self.broker.pie_completion(1);
        let borrowed = crate::driver::frame::PoolResizeDescBorrow::new(desc);
        self.inner.resize_pool(borrowed.as_raw(), raw)?;
        Ok(completion)
    }
    pub fn close_instance(&mut self, id: u64) -> Result<()> {
        self.inner.close_instance(id)
    }
    pub fn close_channel(&mut self, id: u64) -> Result<()> {
        self.inner.close_channel(id)
    }
}

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

struct DriverRegistration {
    spec: DriverSpec,
    native: Option<NativeDriver>,
}

fn registry() -> &'static RwLock<Vec<Option<DriverRegistration>>> {
    static REGISTRY: OnceLock<RwLock<Vec<Option<DriverRegistration>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Vec::new()))
}

pub fn register_driver(spec: DriverSpec) -> usize {
    let mut drivers = registry().write().unwrap();
    let id = drivers.len();
    drivers.push(Some(DriverRegistration { spec, native: None }));
    id
}

pub fn register_native_driver(spec: DriverSpec, native: NativeDriver) -> usize {
    let mut drivers = registry().write().unwrap();
    let id = drivers.len();
    drivers.push(Some(DriverRegistration {
        spec,
        native: Some(native),
    }));
    id
}

pub async fn get_spec(driver_id: usize) -> Result<DriverSpec> {
    registry()
        .read()
        .unwrap()
        .get(driver_id)
        .and_then(|d| d.as_ref().map(|r| r.spec.clone()))
        .ok_or_else(|| anyhow!("unknown driver {driver_id}"))
}

pub fn take_native_driver(driver_id: usize) -> Result<NativeDriver> {
    let mut drivers = registry().write().unwrap();
    let Some(Some(driver)) = drivers.get_mut(driver_id) else {
        return Err(anyhow!("unknown driver {driver_id}"));
    };
    driver
        .native
        .take()
        .ok_or_else(|| anyhow!("driver {driver_id} has no native backend installed"))
}

pub fn unregister_driver(driver_id: usize) -> Result<()> {
    let mut drivers = registry().write().unwrap();
    let Some(slot) = drivers.get_mut(driver_id) else {
        return Err(anyhow!("unknown driver {driver_id}"));
    };
    slot.take();
    Ok(())
}
