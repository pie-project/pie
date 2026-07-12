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
use pie_driver_abi::{
    PieBytes, PieChannelEndpointBinding, PieDriver, PieDriverCaps, PieDriverCreateDesc,
    PieModelLoadDesc, pie_cuda_bind_instance, pie_cuda_close_channel, pie_cuda_close_instance,
    pie_cuda_copy_kv, pie_cuda_copy_state, pie_cuda_create, pie_cuda_destroy, pie_cuda_launch,
    pie_cuda_load_model, pie_cuda_register_channel, pie_cuda_register_program,
    pie_cuda_resize_pool,
};

struct CudaDriverHandle {
    driver: *mut PieDriver,
    broker: CompletionBroker,
    device_facts: pie_driver_abi::DeviceFacts,
}

impl CudaDriverHandle {
    fn create(config_bytes: &[u8]) -> Result<Self> {
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
        let driver = unsafe { pie_cuda_create(&desc, &mut caps) };
        if driver.is_null() {
            return Err(anyhow!("pie_cuda_create returned null"));
        }
        let device_facts: pie_driver_abi::DeviceFacts = match parse_json(caps, "device facts") {
            Ok(device_facts) => device_facts,
            Err(error) => {
                unsafe { pie_cuda_destroy(driver) };
                return Err(error);
            }
        };
        Ok(Self {
            driver,
            broker,
            device_facts,
        })
    }

    fn device_facts(&self) -> &pie_driver_abi::DeviceFacts {
        &self.device_facts
    }

    fn load_model(
        &mut self,
        desc: &pie_driver_abi::ModelLoadDesc,
    ) -> Result<pie_driver_abi::DriverCapabilities> {
        let snapshot = desc
            .snapshot_dir
            .to_str()
            .ok_or_else(|| anyhow!("model snapshot path must be UTF-8"))?;
        let raw = PieModelLoadDesc {
            abi_version: pie_driver_abi::PIE_DRIVER_ABI_VERSION,
            reserved0: 0,
            compiler_version: desc.compiler_version,
            load_plan_bytes: PieBytes {
                ptr: desc.load_plan_bytes.as_ptr(),
                len: desc.load_plan_bytes.len(),
            },
            snapshot_dir: PieBytes {
                ptr: snapshot.as_ptr(),
                len: snapshot.len(),
            },
        };
        let mut caps = PieDriverCaps::default();
        sync_status(
            unsafe { pie_cuda_load_model(self.driver, &raw, &mut caps) },
            "pie_cuda_load_model",
        )?;
        let capabilities: pie_driver_abi::DriverCapabilities =
            parse_json(caps, "model capabilities")?;
        Ok(capabilities)
    }

    fn register_program(&mut self, plan: &ProgramRegistration) -> Result<u64> {
        let borrowed = ProgramDescBorrow::new(plan);
        let mut out = 0u64;
        sync_status(
            unsafe { pie_cuda_register_program(self.driver, borrowed.as_raw(), &mut out) },
            "pie_cuda_register_program",
        )?;
        Ok(out)
    }

    fn register_channel(&mut self, plan: &ChannelRegistrationPlan) -> Result<RegisteredChannel> {
        let borrowed = ChannelDescBorrow::new(plan);
        let mut binding = PieChannelEndpointBinding::default();
        sync_status(
            unsafe { pie_cuda_register_channel(self.driver, borrowed.as_raw(), &mut binding) },
            "pie_cuda_register_channel",
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

    fn bind_instance(&mut self, plan: &InstanceBindingPlan) -> Result<BoundInstance> {
        let borrowed = InstanceDescBorrow::new(plan);
        let mut binding = pie_driver_abi::PieInstanceBinding::default();
        sync_status(
            unsafe { pie_cuda_bind_instance(self.driver, borrowed.as_raw(), &mut binding) },
            "pie_cuda_bind_instance",
        )?;
        if let Err(error) = plan.validate_binding(&binding) {
            let _ = unsafe { pie_cuda_close_instance(self.driver, binding.instance_id) };
            return Err(error);
        }
        Ok(BoundInstance::new(
            plan.driver_id,
            plan.program_id,
            binding,
            plan.pacing_wait_id,
        ))
    }

    fn launch(&mut self, plan: &LaunchSubmission) -> Result<SubmissionCompletion> {
        let target_epoch = 1;
        let (raw, completion) = self.broker.launch_completion(target_epoch);
        let borrowed = LaunchDescBorrow::from_submission(plan);
        sync_status(
            unsafe { pie_cuda_launch(self.driver, borrowed.as_raw(), raw) },
            "pie_cuda_launch",
        )?;
        Ok(completion)
    }

    fn copy_kv(&mut self, plan: &KvCopyPlan) -> Result<SubmissionCompletion> {
        let target_epoch = 1;
        let (raw, completion) = self.broker.pie_completion(target_epoch);
        let borrowed = KvCopyDescBorrow::new(plan);
        sync_status(
            unsafe { pie_cuda_copy_kv(self.driver, borrowed.as_raw(), raw) },
            "pie_cuda_copy_kv",
        )?;
        Ok(completion)
    }

    fn copy_state(&mut self, plan: &StateCopyPlan) -> Result<SubmissionCompletion> {
        let target_epoch = 1;
        let (raw, completion) = self.broker.pie_completion(target_epoch);
        let borrowed = StateCopyDescBorrow::new(plan);
        sync_status(
            unsafe { pie_cuda_copy_state(self.driver, borrowed.as_raw(), raw) },
            "pie_cuda_copy_state",
        )?;
        Ok(completion)
    }

    fn resize_pool(&mut self, plan: &PoolResizePlan) -> Result<SubmissionCompletion> {
        let target_epoch = 1;
        let (raw, completion) = self.broker.pie_completion(target_epoch);
        let borrowed = PoolResizeDescBorrow::new(plan);
        sync_status(
            unsafe { pie_cuda_resize_pool(self.driver, borrowed.as_raw(), raw) },
            "pie_cuda_resize_pool",
        )?;
        Ok(completion)
    }

    fn close_instance(&mut self, instance_id: u64) -> Result<()> {
        sync_status(
            unsafe { pie_cuda_close_instance(self.driver, instance_id) },
            "pie_cuda_close_instance",
        )
    }

    fn close_channel(&mut self, channel_id: u64) -> Result<()> {
        sync_status(
            unsafe { pie_cuda_close_channel(self.driver, channel_id) },
            "pie_cuda_close_channel",
        )
    }
}

unsafe impl Send for CudaDriverHandle {}
unsafe impl Sync for CudaDriverHandle {}

impl Drop for CudaDriverHandle {
    fn drop(&mut self) {
        self.broker.close_all("cuda driver dropped");
        if !self.driver.is_null() {
            unsafe { pie_cuda_destroy(self.driver) };
        }
    }
}

pub struct CudaDriver {
    leader: CudaDriverHandle,
    followers: Vec<CudaDriverHandle>,
}

impl CudaDriver {
    pub fn create(config_bytes: &[u8]) -> Result<(Self, pie_driver_abi::DeviceFacts)> {
        let (driver, mut facts) = Self::create_group(vec![config_bytes.to_vec()])?;
        let facts = facts
            .pop()
            .ok_or_else(|| anyhow!("cuda create returned no device facts"))?;
        Ok((driver, facts))
    }

    pub fn create_group(
        config_blobs: Vec<Vec<u8>>,
    ) -> Result<(Self, Vec<pie_driver_abi::DeviceFacts>)> {
        if config_blobs.is_empty() {
            return Err(anyhow!("cuda group requires at least one rank config"));
        }

        let mut joins = Vec::with_capacity(config_blobs.len());
        for (rank, config_bytes) in config_blobs.into_iter().enumerate() {
            let thread = std::thread::Builder::new()
                .name(format!("pie-cuda-init-rank-{rank}"))
                .spawn(move || CudaDriverHandle::create(&config_bytes))
                .map_err(|err| anyhow!("spawn cuda rank {rank} init thread: {err}"))?;
            joins.push(thread);
        }

        let mut created = Vec::with_capacity(joins.len());
        let mut first_error = None;
        for (rank, join) in joins.into_iter().enumerate() {
            match join.join() {
                Ok(Ok(driver)) => created.push(driver),
                Ok(Err(err)) => {
                    if first_error.is_none() {
                        first_error = Some(anyhow!("cuda rank {rank} create failed: {err:#}"));
                    }
                }
                Err(_) => {
                    if first_error.is_none() {
                        first_error = Some(anyhow!("cuda rank {rank} init thread panicked"));
                    }
                }
            }
        }

        if let Some(err) = first_error {
            drop(created);
            return Err(err);
        }

        let mut created = created;
        let leader = created.remove(0);
        let mut device_facts = Vec::with_capacity(created.len() + 1);
        device_facts.push(leader.device_facts().clone());
        device_facts.extend(created.iter().map(|driver| driver.device_facts().clone()));
        Ok((
            Self {
                leader,
                followers: created,
            },
            device_facts,
        ))
    }

    pub fn unsupported() -> Result<Self> {
        Err(anyhow!("CUDA local driver is not available in this build"))
    }

    pub fn device_facts(&self) -> Vec<pie_driver_abi::DeviceFacts> {
        std::iter::once(self.leader.device_facts().clone())
            .chain(
                self.followers
                    .iter()
                    .map(|driver| driver.device_facts().clone()),
            )
            .collect()
    }

    pub fn load_model(
        &mut self,
        descs: Vec<pie_driver_abi::ModelLoadDesc>,
    ) -> Result<pie_driver_abi::DriverCapabilities> {
        if descs.len() != self.followers.len() + 1 {
            return Err(anyhow!(
                "cuda model-load descriptor count {} does not match rank count {}",
                descs.len(),
                self.followers.len() + 1
            ));
        }
        let mut results = Vec::with_capacity(descs.len());
        let (leader_desc, follower_descs) = descs
            .split_first()
            .ok_or_else(|| anyhow!("cuda group has no model-load descriptor"))?;
        let Self { leader, followers } = self;
        std::thread::scope(|scope| {
            let mut joins = Vec::with_capacity(descs.len());
            joins.push(scope.spawn(|| leader.load_model(leader_desc)));
            for (follower, desc) in followers.iter_mut().zip(follower_descs) {
                joins.push(scope.spawn(move || follower.load_model(desc)));
            }
            for join in joins {
                results.push(
                    join.join()
                        .map_err(|_| anyhow!("cuda rank model-load thread panicked"))?,
                );
            }
            Ok::<(), anyhow::Error>(())
        })?;
        let mut capabilities = results.into_iter().collect::<Result<Vec<_>>>()?.into_iter();
        let leader = capabilities
            .next()
            .ok_or_else(|| anyhow!("cuda group has no leader capabilities"))?;
        if capabilities.any(|caps| {
            caps.arch_name != leader.arch_name
                || caps.vocab_size != leader.vocab_size
                || caps.kv_page_size != leader.kv_page_size
        }) {
            return Err(anyhow!(
                "cuda tensor-parallel ranks reported incompatible model capabilities"
            ));
        }
        Ok(leader)
    }

    pub fn register_program(&mut self, plan: &ProgramRegistration) -> Result<u64> {
        self.leader.register_program(plan)
    }

    pub fn register_channel(
        &mut self,
        plan: &ChannelRegistrationPlan,
    ) -> Result<RegisteredChannel> {
        self.leader.register_channel(plan)
    }

    pub fn bind_instance(&mut self, plan: &InstanceBindingPlan) -> Result<BoundInstance> {
        self.leader.bind_instance(plan)
    }

    pub fn launch(&mut self, plan: &LaunchSubmission) -> Result<SubmissionCompletion> {
        self.leader.launch(plan)
    }

    pub fn copy_kv(&mut self, plan: &KvCopyPlan) -> Result<SubmissionCompletion> {
        self.leader.copy_kv(plan)
    }

    pub fn copy_state(&mut self, plan: &StateCopyPlan) -> Result<SubmissionCompletion> {
        self.leader.copy_state(plan)
    }

    pub fn resize_pool(&mut self, plan: &PoolResizePlan) -> Result<SubmissionCompletion> {
        self.leader.resize_pool(plan)
    }

    pub fn close_instance(&mut self, instance_id: u64) -> Result<()> {
        self.leader.close_instance(instance_id)
    }

    pub fn close_channel(&mut self, channel_id: u64) -> Result<()> {
        self.leader.close_channel(channel_id)
    }
}

unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

impl Drop for CudaDriver {
    fn drop(&mut self) {
        self.followers.clear();
    }
}

pub(crate) fn sync_status(status: i32, op: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(anyhow!("{op} failed with status {status}"))
    }
}

fn parse_json<T: serde::de::DeserializeOwned>(caps: PieDriverCaps, label: &str) -> Result<T> {
    if caps.json_bytes.is_null() {
        return Err(anyhow!("driver returned null {label} payload"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(caps.json_bytes, caps.json_len) };
    serde_json::from_slice(bytes).map_err(|e| anyhow!("driver {label} payload parse: {e}"))
}

#[allow(dead_code)]
pub unsafe fn _touch_create_symbol() {
    let _ = pie_cuda_create;
}
