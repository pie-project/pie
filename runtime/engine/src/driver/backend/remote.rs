use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, ensure};
use pie_driver_abi::{
    ExecutorRequest, ExecutorResponse, ExecutorRpcClient, RemoteBindInstance, RemoteChannelValue,
    RemoteError, RemoteErrorKind, RemoteLaunch, RemoteRegisterChannel, ScratchGrant,
    TerminalCellState,
};

use crate::driver::channel::RegisteredChannel;
use crate::driver::command::{
    ChannelRegistrationPlan, KvCopyPlan, MediaEncodePlan, PoolResizePlan, ProgramRegistration,
    StateCopyPlan,
};
use crate::driver::completion::{CompletionBroker, SubmissionCompletion};
use crate::driver::instance::{BoundInstance, InstanceBindingPlan};
use crate::driver::submission::LaunchSubmission;

const RPC_DEADLINE: Duration = Duration::from_secs(300);

pub struct RemoteDriver {
    client: ExecutorRpcClient,
    runtime: tokio::runtime::Handle,
    broker: CompletionBroker,
    connected: Arc<AtomicBool>,
    capabilities: pie_driver_abi::DriverCapabilities,
    grant: ScratchGrant,
    programs: Mutex<HashMap<u64, u64>>,
    channels: Mutex<HashMap<u64, u64>>,
    instances: Mutex<HashMap<u64, u64>>,
    next_local_instance: AtomicU64,
}

#[derive(Clone)]
pub struct RemoteDisconnectHandle {
    broker: CompletionBroker,
    connected: Arc<AtomicBool>,
}

impl RemoteDisconnectHandle {
    pub fn disconnect(&self, message: impl Into<String>) {
        if self.connected.swap(false, Ordering::AcqRel) {
            self.broker.close_all(message);
        }
    }

    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Acquire)
    }
}

impl RemoteDriver {
    pub fn new(
        client: ExecutorRpcClient,
        runtime: tokio::runtime::Handle,
        capabilities: pie_driver_abi::DriverCapabilities,
        grant: ScratchGrant,
    ) -> Self {
        Self {
            client,
            runtime,
            broker: CompletionBroker::new(),
            connected: Arc::new(AtomicBool::new(true)),
            capabilities,
            grant,
            programs: Mutex::new(HashMap::new()),
            channels: Mutex::new(HashMap::new()),
            instances: Mutex::new(HashMap::new()),
            next_local_instance: AtomicU64::new(1),
        }
    }

    pub fn capabilities(&self) -> &pie_driver_abi::DriverCapabilities {
        &self.capabilities
    }

    pub fn grant(&self) -> ScratchGrant {
        self.grant
    }

    pub fn disconnect_handle(&self) -> RemoteDisconnectHandle {
        RemoteDisconnectHandle {
            broker: self.broker.clone(),
            connected: Arc::clone(&self.connected),
        }
    }

    pub fn disconnect(&self, message: impl Into<String>) {
        self.disconnect_handle().disconnect(message);
    }

    fn ensure_connected(&self) -> Result<()> {
        ensure!(
            self.connected.load(Ordering::Acquire),
            "remote driver is disconnected"
        );
        Ok(())
    }

    fn context() -> tarpc::context::Context {
        let mut context = tarpc::context::current();
        context.deadline = Instant::now() + RPC_DEADLINE;
        context
    }

    fn execute_blocking(
        &self,
        request: ExecutorRequest,
    ) -> Result<Result<ExecutorResponse, RemoteError>> {
        self.ensure_connected()?;
        let client = self.client.clone();
        let result = self
            .runtime
            .block_on(async move { client.execute(Self::context(), request).await });
        match result {
            Ok(response) => Ok(response),
            Err(error) => {
                self.disconnect(format!("executor transport lost: {error}"));
                Err(anyhow!("executor transport failed: {error}"))
            }
        }
    }

    fn response(
        &self,
        request: ExecutorRequest,
        expected: &'static str,
    ) -> Result<ExecutorResponse> {
        match self.execute_blocking(request)? {
            Ok(response) => Ok(response),
            Err(error) => {
                if error.kind == RemoteErrorKind::Disconnected {
                    self.disconnect(error.to_string());
                }
                Err(anyhow!("executor rejected {expected}: {error}"))
            }
        }
    }

    fn next_local_instance_id(&self, requested: u64) -> Result<u64> {
        if requested != 0 {
            ensure!(
                !self.instances.lock().unwrap().contains_key(&requested),
                "remote local instance {requested} is already bound"
            );
            return Ok(requested);
        }
        loop {
            let candidate = self.next_local_instance.fetch_add(1, Ordering::Relaxed);
            if candidate != 0 && !self.instances.lock().unwrap().contains_key(&candidate) {
                return Ok(candidate);
            }
        }
    }

    fn translate_instances(&self, local: &[u64]) -> Result<Vec<u64>> {
        let instances = self.instances.lock().unwrap();
        local
            .iter()
            .map(|id| {
                instances
                    .get(id)
                    .copied()
                    .ok_or_else(|| anyhow!("remote local instance {id} is not bound"))
            })
            .collect()
    }

    fn publish_terminal(pointers: &[usize], states: &[TerminalCellState]) -> Result<()> {
        ensure!(
            pointers.len() == states.len(),
            "executor returned {} terminal cells for {} local cells",
            states.len(),
            pointers.len()
        );
        for (&address, state) in pointers.iter().zip(states) {
            ensure!(address != 0, "local terminal cell pointer is null");
            let cell = address as *mut pie_driver_abi::PieTerminalCell;
            unsafe {
                (*cell).reserved0 = state.reserved0;
                std::sync::atomic::AtomicU32::from_ptr(std::ptr::addr_of_mut!((*cell).outcome))
                    .store(state.outcome, Ordering::Release);
            }
        }
        Ok(())
    }

    fn spawn_launch_rpc(
        &self,
        request: ExecutorRequest,
        terminal_pointers: Vec<usize>,
        completion: SubmissionCompletion,
    ) {
        let client = self.client.clone();
        let broker = self.broker.clone();
        let connected = Arc::clone(&self.connected);
        let wait_id = completion.wait_id();
        let target_epoch = completion.target_epoch();
        let task_completion = completion.clone();
        self.runtime.spawn(async move {
            let settled = match client.execute(Self::context(), request).await {
                Ok(Ok(ExecutorResponse::Terminal(remote))) => {
                    Self::publish_terminal(&terminal_pointers, &remote.per_request)
                }
                Ok(Ok(other)) => Err(anyhow!(
                    "executor returned unexpected launch response {other:?}"
                )),
                Ok(Err(error)) => {
                    if error.kind == RemoteErrorKind::Disconnected {
                        connected.store(false, Ordering::Release);
                        broker.close_all(error.to_string());
                    }
                    Err(anyhow!("executor rejected launch: {error}"))
                }
                Err(error) => {
                    connected.store(false, Ordering::Release);
                    broker.close_all(format!("executor transport lost: {error}"));
                    Err(anyhow!("executor launch transport failed: {error}"))
                }
            };
            match settled {
                Ok(()) => broker.notify(wait_id, target_epoch),
                Err(error) => task_completion.close(error.to_string()),
            }
        });
    }

    pub fn load_model(
        &mut self,
        _descs: Vec<pie_driver_abi::ModelLoadDesc>,
    ) -> Result<pie_driver_abi::DriverCapabilities> {
        self.ensure_connected()?;
        Ok(self.capabilities.clone())
    }

    pub fn register_program(&mut self, desc: &ProgramRegistration) -> Result<u64> {
        if let Some(&program_id) = self.programs.lock().unwrap().get(&desc.program_hash) {
            return Ok(program_id);
        }
        let response = self.response(
            ExecutorRequest::RegisterProgram(desc.clone()),
            "register_program",
        )?;
        let ExecutorResponse::ProgramRegistered(program_id) = response else {
            return Err(anyhow!(
                "executor returned unexpected register_program response {response:?}"
            ));
        };
        self.programs
            .lock()
            .unwrap()
            .insert(desc.program_hash, program_id);
        Ok(program_id)
    }

    pub fn register_channel(
        &mut self,
        desc: &ChannelRegistrationPlan,
    ) -> Result<RegisteredChannel> {
        if self.channels.lock().unwrap().contains_key(&desc.channel_id) {
            return Err(anyhow!(
                "remote local channel {} is already registered",
                desc.channel_id
            ));
        }
        let response = self.response(
            ExecutorRequest::RegisterChannel(RemoteRegisterChannel {
                local_channel_id: desc.channel_id,
                shape: desc.shape.clone(),
                dtype: desc.dtype,
                host_role: desc.host_role,
                seeded: desc.seeded,
                extern_dir: desc.extern_dir,
                capacity: desc.capacity,
                extern_name: desc.extern_name.clone(),
            }),
            "register_channel",
        )?;
        let ExecutorResponse::ChannelRegistered(binding) = response else {
            return Err(anyhow!(
                "executor returned unexpected register_channel response {response:?}"
            ));
        };
        ensure!(
            binding.local_channel_id == desc.channel_id,
            "executor channel correlation mismatch"
        );
        self.channels
            .lock()
            .unwrap()
            .insert(desc.channel_id, binding.executor_channel_id);
        let mut native = pie_driver_abi::PieChannelEndpointBinding::default();
        native.channel_id = desc.channel_id;
        Ok(RegisteredChannel {
            driver_id: desc.driver_id,
            binding: native,
            reader_wait_id: desc.reader_wait_id,
            writer_wait_id: desc.writer_wait_id,
        })
    }

    pub fn bind_instance(&mut self, desc: &InstanceBindingPlan) -> Result<BoundInstance> {
        self.ensure_connected()?;
        let local_instance_id = self.next_local_instance_id(desc.requested_instance_id)?;
        let channels = self.channels.lock().unwrap();
        let remote_channel_ids = desc
            .channel_ids
            .iter()
            .map(|channel| {
                channels
                    .get(channel)
                    .copied()
                    .ok_or_else(|| anyhow!("remote local channel {channel} is not registered"))
            })
            .collect::<Result<Vec<_>>>()?;
        let remote_seed_values = desc
            .seed_values
            .iter()
            .map(|value| {
                Ok(RemoteChannelValue {
                    channel_id: channels.get(&value.channel).copied().ok_or_else(|| {
                        anyhow!("remote seed channel {} is not registered", value.channel)
                    })?,
                    bytes: value.bytes.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        drop(channels);
        let response = self.response(
            ExecutorRequest::BindInstance(RemoteBindInstance {
                local_instance_id,
                program_id: desc.program_id,
                channel_ids: remote_channel_ids,
                seed_values: remote_seed_values,
            }),
            "bind_instance",
        )?;
        let ExecutorResponse::InstanceBound(binding) = response else {
            return Err(anyhow!(
                "executor returned unexpected bind_instance response {response:?}"
            ));
        };
        ensure!(
            binding.local_instance_id == local_instance_id,
            "executor bind correlation mismatch: {} != {}",
            binding.local_instance_id,
            local_instance_id
        );
        ensure!(
            binding.executor_instance_id != 0,
            "executor returned zero instance id"
        );
        self.instances
            .lock()
            .unwrap()
            .insert(local_instance_id, binding.executor_instance_id);
        Ok(BoundInstance::new(
            desc.driver_id,
            desc.program_id,
            pie_driver_abi::PieInstanceBinding {
                instance_id: local_instance_id,
            },
            desc.pacing_wait_id,
        ))
    }

    pub fn launch(&mut self, desc: &LaunchSubmission) -> Result<SubmissionCompletion> {
        self.ensure_connected()?;
        let remote_instances = self.translate_instances(&desc.instance_ids)?;
        let terminal_count = u32::try_from(desc.terminal_cells.len())
            .context("remote launch terminal count exceeds u32")?;
        ensure!(
            desc.terminal_cells.iter().all(|cell| !cell.is_null()),
            "remote launch contains a null terminal cell"
        );
        let request = ExecutorRequest::Launch(RemoteLaunch {
            plan: desc.plan.clone(),
            instance_ids: remote_instances,
            terminal_count,
            kv_translation: desc.kv_translation.clone(),
            kv_translation_indptr: desc.kv_translation_indptr.clone(),
            program_row_indptr: desc.program_row_indptr.clone(),
            logical_fire_ids: desc.logical_fire_ids.clone(),
            channel_expected_head: desc.channel_expected_head.clone(),
            channel_expected_tail: desc.channel_expected_tail.clone(),
            channel_ticket_indptr: desc.channel_ticket_indptr.clone(),
        });
        let completion = self.broker.launch_completion(1).1;
        self.spawn_launch_rpc(
            request,
            desc.terminal_cells
                .iter()
                .map(|cell| *cell as usize)
                .collect(),
            completion.clone(),
        );
        Ok(completion)
    }

    pub fn encode(&mut self, _plan: &mut MediaEncodePlan) -> Result<SubmissionCompletion> {
        Err(anyhow!(
            "nested media encode through a remote driver is unsupported"
        ))
    }

    pub fn copy_kv(&mut self, desc: &KvCopyPlan) -> Result<SubmissionCompletion> {
        self.ensure_connected()?;
        let (raw, completion) = self.broker.pie_completion(1);
        self.spawn_launch_rpc(
            ExecutorRequest::CopyKv(desc.clone()),
            vec![raw.terminal_cell as usize],
            completion.clone(),
        );
        Ok(completion)
    }

    pub fn copy_state(&mut self, _desc: &StateCopyPlan) -> Result<SubmissionCompletion> {
        Err(anyhow!(
            "remote driver does not support recurrent-state copies"
        ))
    }

    pub fn resize_pool(&mut self, _desc: &PoolResizePlan) -> Result<SubmissionCompletion> {
        Err(anyhow!(
            "remote executor pools are fixed for the lease lifetime"
        ))
    }

    pub fn close_instance(&mut self, local_id: u64) -> Result<()> {
        let remote_id = self
            .instances
            .lock()
            .unwrap()
            .remove(&local_id)
            .ok_or_else(|| anyhow!("remote local instance {local_id} is not bound"))?;
        let response =
            self.response(ExecutorRequest::CloseInstance(remote_id), "close_instance")?;
        ensure!(
            matches!(response, ExecutorResponse::Closed),
            "executor returned unexpected close_instance response {response:?}"
        );
        Ok(())
    }

    pub fn close_channel(&mut self, local_id: u64) -> Result<()> {
        let remote_id = self
            .channels
            .lock()
            .unwrap()
            .remove(&local_id)
            .ok_or_else(|| anyhow!("remote local channel {local_id} is not registered"))?;
        let response = self.response(ExecutorRequest::CloseChannel(remote_id), "close_channel")?;
        ensure!(
            matches!(response, ExecutorResponse::Closed),
            "executor returned unexpected close_channel response {response:?}"
        );
        Ok(())
    }
}

impl Drop for RemoteDriver {
    fn drop(&mut self) {
        self.disconnect("remote driver dropped");
    }
}
