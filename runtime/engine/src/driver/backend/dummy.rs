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

pub struct DummyDriver {
    inner: pie_driver_dummy_lib::DummyDriver,
    broker: CompletionBroker,
}

unsafe impl Send for DummyDriver {}
unsafe impl Sync for DummyDriver {}

impl DummyDriver {
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
        let borrowed = ProgramDescBorrow::new(desc);
        self.inner.register_program(borrowed.as_raw())
    }

    pub fn register_channel(
        &mut self,
        desc: &ChannelRegistrationPlan,
    ) -> Result<RegisteredChannel> {
        let borrowed = ChannelDescBorrow::new(desc);
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
        let borrowed = InstanceDescBorrow::new(desc);
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

    pub fn launch(&mut self, desc: &LaunchSubmission) -> Result<SubmissionCompletion> {
        let (raw, completion) = self.broker.launch_completion(1);
        let borrowed = LaunchDescBorrow::from_submission(desc);
        self.inner.launch(borrowed.as_raw(), raw)?;
        Ok(completion)
    }

    pub fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<SubmissionCompletion> {
        let (raw, completion) = self.broker.pie_completion(1);
        let borrowed = KvCopyDescBorrow::new(desc);
        self.inner.copy_kv(borrowed.as_raw(), raw)?;
        Ok(completion)
    }

    pub fn copy_state(&mut self, desc: &StateCopyPlan) -> Result<SubmissionCompletion> {
        let (raw, completion) = self.broker.pie_completion(1);
        let borrowed = StateCopyDescBorrow::new(desc);
        self.inner.copy_state(borrowed.as_raw(), raw)?;
        Ok(completion)
    }

    pub fn resize_pool(&mut self, desc: &PoolResizePlan) -> Result<SubmissionCompletion> {
        let (raw, completion) = self.broker.pie_completion(1);
        let borrowed = PoolResizeDescBorrow::new(desc);
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
