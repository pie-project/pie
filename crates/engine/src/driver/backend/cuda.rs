use anyhow::{Result, anyhow};

use ::driver_api::{
    BoundInstance, ChannelRegistrationPlan, CompletionBroker, Driver, FrameLaunchOutcome,
    FrameSubmission, InstanceBindingPlan, KvCopyPlan, MediaEncodePlan, PoolResizePlan,
    ProgramRegistration, RegisteredChannel, StateCopyPlan, SubmissionCompletion,
};
// THE DRIVER, AS A RUST TYPE.
//
// This was thirteen free functions and a `*mut PieDriver`, imported from
// `driver_cuda::serve` and called with the descriptor structs `crate::driver::abi`
// built for them. Before that it was the same thirteen names reached through an
// `unsafe extern "C"` block, which the linker resolved against this same
// workspace: both sides were Rust, and the declaration existed only because the
// driver on the far side used to be C++.
//
// What the shape cost, measured: seven `#[repr(C)]` descriptors that existed to
// be built here and taken apart there, two `*DescBorrow` marshallers, a JSON
// blob handed back through an out-parameter because a C function that fails
// cannot return two things, and a null check plus a validator call on every
// entry. `driver_cuda::serve::Shell` is a struct with methods now.
use driver_cuda::serve::Shell;

struct CudaDriverHandle {
    shell: Shell,
    broker: CompletionBroker,
    kv_handle: Option<::driver_api::KvHandle>,
}

/// Turn the shell's status into an error that names the verb.
///
/// `driver-cuda` still answers `i32` internally — it is that crate's own
/// convention across several thousand lines, and no longer an ABI now that
/// nothing foreign reads it. This is the one place it is translated.
fn status<T>(result: std::result::Result<T, i32>, verb: &'static str) -> Result<T> {
    result.map_err(|code| anyhow!("driver-cuda {verb} failed with status {code}"))
}

impl CudaDriverHandle {
    fn create(config_bytes: &[u8]) -> Result<Self> {
        let broker = CompletionBroker::new();
        let shell = status(Shell::open(config_bytes, broker.clone()), "create")?;
        Ok(Self {
            shell,
            broker,
            kv_handle: None,
        })
    }

    fn device_facts(&self) -> &::driver_api::DeviceFacts {
        self.shell.device_facts()
    }

    fn load_model(
        &mut self,
        desc: &::driver_api::ModelLoadDesc,
    ) -> Result<::driver_api::DriverCapabilities> {
        let capabilities = status(self.shell.load_model(desc), "load_model")?;
        self.kv_handle = capabilities.kv_handle.clone();
        Ok(capabilities)
    }

    fn register_program(&mut self, plan: &ProgramRegistration) -> Result<u64> {
        status(self.shell.register_program(plan), "register_program")
    }

    fn register_channel(&mut self, plan: &ChannelRegistrationPlan) -> Result<RegisteredChannel> {
        let binding = status(self.shell.register_channel(plan), "register_channel")?;
        // Still checked, and against the PLAN rather than against a
        // `#[repr(C)]` copy of it. This is the direction the type system
        // cannot check: what a driver ANSWERED.
        ::driver_api::validate_channel_endpoint_binding(&binding, plan)
            .map_err(|error| anyhow!(error))?;
        Ok(RegisteredChannel {
            driver_id: plan.driver_id,
            binding,
            reader_wait_id: plan.reader_wait_id,
            writer_wait_id: plan.writer_wait_id,
        })
    }

    fn bind_instance(&mut self, plan: &InstanceBindingPlan) -> Result<BoundInstance> {
        let binding = status(self.shell.bind_instance(plan), "bind_instance")?;
        if let Err(error) = plan.validate_binding(&binding) {
            let _ = self.shell.close_instance(binding.instance_id);
            return Err(error);
        }
        Ok(BoundInstance::new(
            plan.driver_id,
            plan.program_id,
            binding,
            plan.pacing_wait_id,
        ))
    }

    fn launch(&mut self, frame: &FrameSubmission) -> Result<FrameLaunchOutcome> {
        let target_epoch = 1;
        let (target, completion) = self.broker.launch_completion(target_epoch);
        match self.shell.launch(frame, target) {
            Err(::driver_api::PIE_STATUS_EXHAUSTED) => Ok(FrameLaunchOutcome::Exhausted),
            Err(::driver_api::PIE_STATUS_IMPOSSIBLE) => Ok(FrameLaunchOutcome::Impossible),
            other => {
                status(other, "launch")?;
                Ok(FrameLaunchOutcome::Launched(completion))
            }
        }
    }

    fn encode(&mut self, plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        let (target, completion) = self.broker.control_completion(1);
        status(self.shell.encode(plan, target), "encode")?;
        Ok(completion)
    }

    fn copy_kv(&mut self, plan: &KvCopyPlan) -> Result<SubmissionCompletion> {
        let (target, completion) = self.broker.control_completion(1);
        status(self.shell.copy_kv(plan, target), "copy_kv")?;
        Ok(completion)
    }

    fn copy_state(&mut self, plan: &StateCopyPlan) -> Result<SubmissionCompletion> {
        let (target, completion) = self.broker.control_completion(1);
        status(self.shell.copy_state(plan, target), "copy_state")?;
        Ok(completion)
    }

    fn resize_pool(&mut self, plan: &PoolResizePlan) -> Result<SubmissionCompletion> {
        let (target, completion) = self.broker.control_completion(1);
        status(self.shell.resize_pool(plan, target), "resize_pool")?;
        Ok(completion)
    }

    fn close_instance(&mut self, instance_id: u64) -> Result<()> {
        status(self.shell.close_instance(instance_id), "close_instance")
    }

    fn close_channel(&mut self, channel_id: u64) -> Result<()> {
        status(self.shell.close_channel(channel_id), "close_channel")
    }

    fn export_kv_handle(&self) -> Option<::driver_api::KvHandle> {
        self.kv_handle.clone()
    }
}

// STILL `unsafe impl`, and the reason changed rather than went away.
//
// The pair used to cover a `*mut PieDriver` — an erased handle the C boundary
// made unprovable. That pointer is gone, and this did not become derivable: a
// `Shell` holds the device's own raw handles inline (a `cublasContext`, CUDA
// events, the arena's `c_void` bases), none of which is `Send` to the
// compiler, and no amount of taking C out of the CALL changes what a CUDA
// context is.
//
// What makes it sound is unchanged, and it is the driver's rule rather than
// this seam's: every verb takes `&mut self`, so exactly one thread touches a
// shell at a time. `create_group` moves each rank's handle to its own init
// thread and joins before any verb runs, which is the one place the move
// actually happens.
unsafe impl Send for CudaDriverHandle {}
unsafe impl Sync for CudaDriverHandle {}

impl Drop for CudaDriverHandle {
    fn drop(&mut self) {
        // The shell's own `Drop` drains the stream and frees the pinned
        // endpoints; what is left here is telling whoever is still waiting.
        self.broker.close_all("cuda driver dropped");
    }
}

pub struct CudaDriver {
    leader: CudaDriverHandle,
    followers: Vec<CudaDriverHandle>,
}

impl CudaDriver {
    /// Open one device.
    ///
    /// # Errors
    ///
    /// No device, or a boot config this driver refuses.
    pub fn create(config_bytes: &[u8]) -> Result<Self> {
        let (driver, ranks) = Self::create_group(vec![config_bytes.to_vec()])?;
        if ranks != 1 {
            return Err(anyhow!("cuda create opened {ranks} ranks, expected one"));
        }
        Ok(driver)
    }

    /// Open one device per rank config, as one driver.
    ///
    /// Answers how many ranks opened rather than a `Vec<DeviceFacts>`: the
    /// facts are readable off the driver ([`Driver::device_facts`] for the
    /// leader, [`Self::rank_facts`] for all of them), and a second copy
    /// handed back at create was one the caller had to keep in step.
    ///
    /// # Errors
    ///
    /// An empty rank list, or any rank whose device failed to open.
    pub fn create_group(config_blobs: Vec<Vec<u8>>) -> Result<(Self, usize)> {
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
        let ranks = created.len() + 1;
        Ok((
            Self {
                leader,
                followers: created,
            },
            ranks,
        ))
    }

    /// This build has no CUDA driver.
    ///
    /// # Errors
    ///
    /// Always.
    pub fn unsupported() -> Result<Self> {
        Err(anyhow!("CUDA local driver is not available in this build"))
    }

    /// Every rank's facts, leader first.
    ///
    /// Named apart from [`Driver::device_facts`], which answers the leader's
    /// by reference as the contract requires. This is the GROUP's, and a
    /// caller that did not open a group has no use for it.
    #[must_use]
    pub fn rank_facts(&self) -> Vec<::driver_api::DeviceFacts> {
        std::iter::once(self.leader.device_facts().clone())
            .chain(
                self.followers
                    .iter()
                    .map(|driver| driver.device_facts().clone()),
            )
            .collect()
    }
}

impl Driver for CudaDriver {
    fn kind(&self) -> &'static str {
        "cuda"
    }

    fn device_domain(&self) -> ::driver_api::DeviceDomain {
        ::driver_api::PIE_MEMORY_DOMAIN_CUDA_DEVICE
    }

    /// The one driver that takes host-generated kernels.
    ///
    /// The other three generate their own or need none, and take the trait's
    /// default. `Remote` takes it too, and deliberately: the far side does
    /// its own generation, even when the device behind it is a CUDA card.
    fn codegen_backend(&self) -> Option<&'static str> {
        Some("cuda")
    }

    /// The LEADER's facts.
    ///
    /// A group is one driver with one contract, and the rank that answers for
    /// it is rank 0. Every rank's facts are [`CudaDriver::rank_facts`], which
    /// only a caller that opened a group has any use for.
    fn device_facts(&self) -> Option<&::driver_api::DeviceFacts> {
        Some(self.leader.device_facts())
    }

    fn load_model(
        &mut self,
        descs: Vec<::driver_api::ModelLoadDesc>,
    ) -> Result<::driver_api::DriverCapabilities> {
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

    fn register_program(&mut self, plan: &ProgramRegistration) -> Result<u64> {
        self.leader.register_program(plan)
    }

    fn register_channel(&mut self, plan: &ChannelRegistrationPlan) -> Result<RegisteredChannel> {
        self.leader.register_channel(plan)
    }

    fn bind_instance(&mut self, plan: &InstanceBindingPlan) -> Result<BoundInstance> {
        self.leader.bind_instance(plan)
    }

    fn launch(&mut self, frame: &FrameSubmission) -> Result<FrameLaunchOutcome> {
        self.leader.launch(frame)
    }

    fn encode(&mut self, plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        if !self.followers.is_empty() {
            return Err(anyhow!(
                "media encode does not support tensor-parallel groups"
            ));
        }
        self.leader.encode(plan)
    }

    fn copy_kv(&mut self, plan: &KvCopyPlan) -> Result<SubmissionCompletion> {
        self.leader.copy_kv(plan)
    }

    fn copy_state(&mut self, plan: &StateCopyPlan) -> Result<SubmissionCompletion> {
        self.leader.copy_state(plan)
    }

    fn resize_pool(&mut self, plan: &PoolResizePlan) -> Result<SubmissionCompletion> {
        self.leader.resize_pool(plan)
    }

    fn close_instance(&mut self, instance_id: u64) -> Result<()> {
        self.leader.close_instance(instance_id)
    }

    fn close_channel(&mut self, channel_id: u64) -> Result<()> {
        self.leader.close_channel(channel_id)
    }

    fn export_kv_handle(&self) -> Option<::driver_api::KvHandle> {
        self.followers
            .is_empty()
            .then(|| self.leader.export_kv_handle())
            .flatten()
    }
}

unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

impl Drop for CudaDriver {
    fn drop(&mut self) {
        self.followers.clear();
    }
}

// `sync_status` STOOD HERE: it turned a nonzero `i32` into
// "{op} failed with status {status}". Nothing in the crate called it. The FFI
// this wrapped stopped returning bare status codes to this layer, so the one
// thing it did -- name the operation beside the number -- is now done where
// the call is made, with the operation in scope rather than passed in as a
// string.
