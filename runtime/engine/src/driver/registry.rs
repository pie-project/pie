use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};

use anyhow::{Result, anyhow};

use crate::driver::backend::NativeDriver;
use crate::driver::completion::{Completion, CompletionBroker};
use crate::driver::frame::{
    BoundInstance, InstanceBindingPlan, KvCopyDescBorrow, KvCopyPlan, LaunchDescBorrow,
    LaunchSubmission, PoolResizePlan, ProgramRegistration, StateCopyPlan,
};
use crate::inference::scheduler::SchedulerHandle;

pub struct DummyLocalDriver {
    inner: pie_driver_dummy_lib::DummyDriver,
    broker: CompletionBroker,
    next_epoch: AtomicU64,
}

unsafe impl Send for DummyLocalDriver {}
unsafe impl Sync for DummyLocalDriver {}

impl DummyLocalDriver {
    pub fn new(options: pie_driver_dummy_lib::DummyDriverOptions) -> Self {
        let broker = CompletionBroker::new();
        let inner =
            pie_driver_dummy_lib::DummyDriver::with_runtime(options, broker.runtime_callbacks());
        Self {
            inner,
            broker,
            next_epoch: AtomicU64::new(1),
        }
    }

    pub fn capabilities(&self) -> &pie_driver_abi::DriverCapabilities {
        self.inner.capabilities()
    }
    fn next_epoch(&self) -> u64 {
        self.next_epoch.fetch_add(1, Ordering::Relaxed)
    }
    pub fn register_program(&mut self, desc: &ProgramRegistration) -> Result<u64> {
        let borrowed = crate::driver::frame::ProgramDescBorrow::new(desc);
        self.inner.register_program(borrowed.as_raw())
    }
    pub fn bind_instance(&mut self, desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        let borrowed = crate::driver::frame::InstanceDescBorrow::new(desc);
        let binding = self.inner.bind_instance(borrowed.as_raw())?;
        if let Err(error) =
            crate::driver::binding_validation::validate_instance_binding(&binding, desc)
        {
            let _ = self.inner.close_instance(binding.instance_id);
            return Err(error);
        }
        Ok(BoundInstance::new(
            desc.driver_id,
            desc.program_id,
            binding,
            desc.pacing_wait_id,
            desc.channel_waits.clone(),
        ))
    }
    pub fn launch(&mut self, desc: &LaunchSubmission) -> Result<Completion> {
        let epoch = self.next_epoch();
        let (raw, completion) = self.broker.pie_completion(epoch);
        let borrowed = LaunchDescBorrow::from_submission(desc);
        self.inner.launch(borrowed.as_raw(), raw)?;
        Ok(completion)
    }
    pub fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<Completion> {
        let epoch = self.next_epoch();
        let (raw, completion) = self.broker.pie_completion(epoch);
        let borrowed = KvCopyDescBorrow::new(desc);
        self.inner.copy_kv(borrowed.as_raw(), raw)?;
        Ok(completion)
    }
    pub fn copy_state(&mut self, desc: &StateCopyPlan) -> Result<Completion> {
        let epoch = self.next_epoch();
        let (raw, completion) = self.broker.pie_completion(epoch);
        let borrowed = crate::driver::frame::StateCopyDescBorrow::new(desc);
        self.inner.copy_state(borrowed.as_raw(), raw)?;
        Ok(completion)
    }
    pub fn resize_pool(&mut self, desc: &PoolResizePlan) -> Result<Completion> {
        let epoch = self.next_epoch();
        let (raw, completion) = self.broker.pie_completion(epoch);
        let borrowed = crate::driver::frame::PoolResizeDescBorrow::new(desc);
        self.inner.resize_pool(borrowed.as_raw(), raw)?;
        Ok(completion)
    }
    pub fn close_instance(&mut self, id: u64) -> Result<()> {
        self.inner.close_instance(id)
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
    scheduler: Option<SchedulerHandle>,
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
        native: None,
        scheduler: None,
    }));
    id
}

pub fn register_native_driver(spec: DriverSpec, native: NativeDriver) -> usize {
    let mut drivers = registry().write().unwrap();
    let id = drivers.len();
    drivers.push(Some(DriverRegistration {
        spec,
        native: Some(native),
        scheduler: None,
    }));
    id
}

pub(crate) fn install_scheduler_handle(driver_id: usize, scheduler: SchedulerHandle) -> Result<()> {
    let mut drivers = registry().write().unwrap();
    let Some(Some(driver)) = drivers.get_mut(driver_id) else {
        return Err(anyhow!("unknown driver {driver_id}"));
    };
    driver.scheduler = Some(scheduler);
    Ok(())
}

pub(crate) fn clear_scheduler_handle(driver_id: usize) -> Result<()> {
    let mut drivers = registry().write().unwrap();
    let Some(Some(driver)) = drivers.get_mut(driver_id) else {
        return Err(anyhow!("unknown driver {driver_id}"));
    };
    driver.scheduler = None;
    Ok(())
}

pub(crate) fn scheduler_handle(driver_id: usize) -> Result<SchedulerHandle> {
    registry()
        .read()
        .unwrap()
        .get(driver_id)
        .and_then(|d| d.as_ref().and_then(|r| r.scheduler.clone()))
        .ok_or_else(|| anyhow!("driver {driver_id} has no scheduler"))
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
