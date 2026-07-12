use anyhow::{Result, anyhow};

use crate::driver::abi::{
    ChannelDescBorrow, InstanceDescBorrow, KvCopyDescBorrow, LaunchDescBorrow,
    PoolResizeDescBorrow, ProgramDescBorrow, StateCopyDescBorrow,
};
use crate::driver::channel::RegisteredChannel;
use crate::driver::command::{
    ChannelRegistrationPlan, KvCopyPlan, PoolResizePlan, ProgramRegistration, StateCopyPlan,
};
use crate::driver::completion::{CompletionBroker, SubmissionCompletion};
use crate::driver::instance::{BoundInstance, InstanceBindingPlan};
use crate::driver::submission::LaunchSubmission;

fn sync_status(status: i32, op: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(anyhow!("{op} failed with status {status}"))
    }
}
use pie_driver_abi::{
    PieBytes, PieChannelEndpointBinding, PieDriver, PieDriverCaps, PieDriverCreateDesc,
    pie_metal_bind_instance, pie_metal_close_channel, pie_metal_close_instance, pie_metal_copy_kv,
    pie_metal_copy_state, pie_metal_create, pie_metal_destroy, pie_metal_launch,
    pie_metal_register_channel, pie_metal_register_program, pie_metal_resize_pool,
};

pub struct MetalDriver {
    driver: *mut PieDriver,
    broker: CompletionBroker,
    capabilities: pie_driver_abi::DriverCapabilities,
}

impl MetalDriver {
    pub fn create(config_bytes: &[u8]) -> Result<(Self, pie_driver_abi::DriverCapabilities)> {
        let broker = CompletionBroker::new();
        let desc = PieDriverCreateDesc {
            abi_version: pie_driver_abi::PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            config_bytes: PieBytes {
                ptr: config_bytes.as_ptr(),
                len: config_bytes.len(),
            },
            runtime: broker.runtime_callbacks(),
        };
        let mut caps = PieDriverCaps::default();
        let driver = unsafe { pie_metal_create(&desc, &mut caps) };
        if driver.is_null() {
            return Err(anyhow!("pie_metal_create returned null"));
        }
        let capabilities = match parse_caps(caps) {
            Ok(capabilities) => capabilities,
            Err(error) => {
                unsafe { pie_metal_destroy(driver) };
                return Err(error);
            }
        };
        Ok((
            Self {
                driver,
                broker,
                capabilities: capabilities.clone(),
            },
            capabilities,
        ))
    }

    pub fn unsupported() -> Result<Self> {
        Err(anyhow!("Metal local driver is not available in this build"))
    }
    pub fn capabilities(&self) -> &pie_driver_abi::DriverCapabilities {
        &self.capabilities
    }
    pub fn register_program(&mut self, plan: &ProgramRegistration) -> Result<u64> {
        let borrowed = ProgramDescBorrow::new(plan);
        let mut out = 0u64;
        sync_status(
            unsafe { pie_metal_register_program(self.driver, borrowed.as_raw(), &mut out) },
            "pie_metal_register_program",
        )?;
        Ok(out)
    }
    pub fn register_channel(
        &mut self,
        plan: &ChannelRegistrationPlan,
    ) -> Result<RegisteredChannel> {
        let borrowed = ChannelDescBorrow::new(plan);
        let mut binding = PieChannelEndpointBinding::default();
        sync_status(
            unsafe { pie_metal_register_channel(self.driver, borrowed.as_raw(), &mut binding) },
            "pie_metal_register_channel",
        )?;
        pie_driver_abi::validate_channel_endpoint_binding(&binding, borrowed.as_raw())
            .map_err(|error| anyhow!(error))?;
        Ok(RegisteredChannel {
            driver_id: plan.driver_id,
            binding,
            reader_wait_id: plan.reader_wait_id,
            writer_wait_id: plan.writer_wait_id,
        })
    }
    pub fn bind_instance(&mut self, plan: &InstanceBindingPlan) -> Result<BoundInstance> {
        let borrowed = InstanceDescBorrow::new(plan);
        let mut binding = pie_driver_abi::PieInstanceBinding::default();
        sync_status(
            unsafe { pie_metal_bind_instance(self.driver, borrowed.as_raw(), &mut binding) },
            "pie_metal_bind_instance",
        )?;
        if let Err(error) = plan.validate_binding(&binding) {
            let _ = unsafe { pie_metal_close_instance(self.driver, binding.instance_id) };
            return Err(error);
        }
        Ok(BoundInstance::new(
            plan.driver_id,
            plan.program_id,
            binding,
            plan.pacing_wait_id,
        ))
    }
    pub fn launch(&mut self, plan: &LaunchSubmission) -> Result<SubmissionCompletion> {
        let target_epoch = 1;
        let (raw, completion) = self.broker.launch_completion(target_epoch);
        let borrowed = LaunchDescBorrow::from_submission(plan);
        sync_status(
            unsafe { pie_metal_launch(self.driver, borrowed.as_raw(), raw) },
            "pie_metal_launch",
        )?;
        Ok(completion)
    }
    pub fn copy_kv(&mut self, plan: &KvCopyPlan) -> Result<SubmissionCompletion> {
        let target_epoch = 1;
        let (raw, completion) = self.broker.pie_completion(target_epoch);
        let borrowed = KvCopyDescBorrow::new(plan);
        sync_status(
            unsafe { pie_metal_copy_kv(self.driver, borrowed.as_raw(), raw) },
            "pie_metal_copy_kv",
        )?;
        Ok(completion)
    }
    pub fn copy_state(&mut self, plan: &StateCopyPlan) -> Result<SubmissionCompletion> {
        let target_epoch = 1;
        let (raw, completion) = self.broker.pie_completion(target_epoch);
        let borrowed = StateCopyDescBorrow::new(plan);
        sync_status(
            unsafe { pie_metal_copy_state(self.driver, borrowed.as_raw(), raw) },
            "pie_metal_copy_state",
        )?;
        Ok(completion)
    }
    pub fn resize_pool(&mut self, plan: &PoolResizePlan) -> Result<SubmissionCompletion> {
        let target_epoch = 1;
        let (raw, completion) = self.broker.pie_completion(target_epoch);
        let borrowed = PoolResizeDescBorrow::new(plan);
        sync_status(
            unsafe { pie_metal_resize_pool(self.driver, borrowed.as_raw(), raw) },
            "pie_metal_resize_pool",
        )?;
        Ok(completion)
    }
    pub fn close_instance(&mut self, instance_id: u64) -> Result<()> {
        sync_status(
            unsafe { pie_metal_close_instance(self.driver, instance_id) },
            "pie_metal_close_instance",
        )
    }
    pub fn close_channel(&mut self, channel_id: u64) -> Result<()> {
        sync_status(
            unsafe { pie_metal_close_channel(self.driver, channel_id) },
            "pie_metal_close_channel",
        )
    }
}
unsafe impl Send for MetalDriver {}
unsafe impl Sync for MetalDriver {}
impl Drop for MetalDriver {
    fn drop(&mut self) {
        self.broker.close_all("metal driver dropped");
        if !self.driver.is_null() {
            unsafe { pie_metal_destroy(self.driver) };
        }
    }
}

fn parse_caps(caps: PieDriverCaps) -> Result<pie_driver_abi::DriverCapabilities> {
    if caps.json_bytes.is_null() {
        return Err(anyhow!("driver creation returned null capability payload"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(caps.json_bytes, caps.json_len) };
    serde_json::from_slice(bytes).map_err(|e| anyhow!("driver capability payload parse: {e}"))
}

#[allow(dead_code)]
pub unsafe fn _touch_create_symbol() {
    let _ = pie_metal_create;
}
