//! [`SchedulerHandle`]: what the rest of the runtime asks of an engine's
//! scheduler (registrations, fires, copies, pool moves, closes), each sent
//! to its loop and answered there.

use super::*;

impl SchedulerHandle {
    pub(crate) fn device_domain(&self) -> ::engine::MemoryDomain {
        self.inner.device_domain
    }

    pub(super) fn send(&self, item: SchedulerItem) -> Result<()> {
        if !self.inner.accepting.load(Ordering::SeqCst) {
            return Err(anyhow!("scheduler shutting down"));
        }
        self.inner.active_senders.fetch_add(1, Ordering::SeqCst);
        if !self.inner.accepting.load(Ordering::SeqCst) {
            self.finish_send();
            return Err(anyhow!("scheduler shutting down"));
        }
        let result = self
            .inner
            .tx
            .send(item)
            .map_err(|_| anyhow!("scheduler channel closed"));
        self.finish_send();
        result
    }

    pub(super) fn finish_send(&self) {
        if self.inner.active_senders.fetch_sub(1, Ordering::SeqCst) == 1
            && !self.inner.accepting.load(Ordering::SeqCst)
        {
            let _guard = self.inner.shutdown_gate.lock().unwrap();
            self.inner.shutdown_wait.notify_all();
        }
    }

    pub(super) fn begin_shutdown(&self) {
        if !self.inner.accepting.swap(false, Ordering::SeqCst) {
            return;
        }
        let mut guard = self.inner.shutdown_gate.lock().unwrap();
        while self.inner.active_senders.load(Ordering::SeqCst) != 0 {
            guard = self.inner.shutdown_wait.wait(guard).unwrap();
        }
        let _ = self.inner.tx.send(SchedulerItem::Stop);
    }

    pub(super) async fn request<T>(
        &self,
        make: impl FnOnce(tokio::sync::oneshot::Sender<T>) -> SchedulerItem,
    ) -> Result<T> {
        let (response, receiver) = tokio::sync::oneshot::channel();
        self.send(make(response))?;
        receiver
            .await
            .map_err(|_| anyhow!("scheduler channel closed"))
    }

    /// Registers `program`: its host codegen on a blocking thread, then
    /// the lane hands back its compile, which runs on a blocking thread
    /// while the lane fires on, and the lane then makes it the program. Its
    /// instances bind once this answers. Every failure names the program's
    /// hash; an engine's prepare holds nothing its finish must undo, so a
    /// compile that panics leaves nothing behind.
    pub(super) async fn register(
        &self,
        program: ProgramRegistration,
    ) -> Result<::engine::ProgramId> {
        let hash = program.program_hash;
        let program = match self.inner.codegen_backend.clone() {
            Some(backend) => crate::rt::spawn_blocking(move || {
                let coded = match crate::pipeline::program::with_host_codegen(
                    &program,
                    Some(backend.as_str()),
                ) {
                    std::borrow::Cow::Owned(coded) => Some(coded),
                    std::borrow::Cow::Borrowed(_) => None,
                };
                coded.unwrap_or(program)
            })
            .await
            .map_err(|e| anyhow!("program 0x{hash:016x}: its host codegen did not finish: {e}"))?,
            None => program,
        };
        let compile = self
            .request(|response| SchedulerItem::PrepareProgram {
                program: Box::new(program),
                response,
            })
            .await?
            .map_err(|e| anyhow!("program 0x{hash:016x}: {e:#}"))?;
        let prepared = crate::rt::spawn_blocking(compile)
            .await
            .map_err(|e| anyhow!("program 0x{hash:016x}: its compile did not finish: {e}"))?;
        self.request(|response| SchedulerItem::FinishProgram { prepared, response })
            .await?
            .map_err(|e| anyhow!("program 0x{hash:016x}: {e:#}"))
    }

    /// The id of `program`, registered once however many binds ask at
    /// once: the first registers it in a task of its own, which a bind's
    /// cancellation does not stop (its finish is always reached, so no
    /// compile is lost and no prepared program is left behind), and every
    /// bind waits on its answer.
    pub(super) async fn program_id(
        &self,
        pid: ProcessId,
        program: ProgramRegistration,
    ) -> Result<::engine::ProgramId> {
        self.program_id_with(pid, program, |this, program| async move {
            this.register(program).await
        })
        .await
    }

    /// [`SchedulerHandle::program_id`], registering by `register`.
    pub(super) async fn program_id_with<F, Fut>(
        &self,
        pid: ProcessId,
        program: ProgramRegistration,
        register: F,
    ) -> Result<::engine::ProgramId>
    where
        F: FnOnce(SchedulerHandle, ProgramRegistration) -> Fut,
        Fut: std::future::Future<Output = Result<::engine::ProgramId>> + Send + 'static,
    {
        let hash = program.program_hash;
        let mut answer = {
            let mut ids = self.inner.program_ids.lock().unwrap();
            match ids.get(&hash) {
                Some((_, launch)) if launch != &program.launch => {
                    return Err(anyhow!("program hash collision for 0x{hash:016x}"));
                }
                Some((Known::Ready(id), _)) => return Ok(*id),
                Some((Known::Registering(answer), _)) => answer.clone(),
                None => {
                    let (sender, answer) = tokio::sync::watch::channel(None);
                    ids.insert(
                        hash,
                        (Known::Registering(answer.clone()), program.launch.clone()),
                    );
                    let this = self.clone();
                    let unanswered = Unanswered {
                        handle: self.clone(),
                        hash,
                        answer: Some(sender),
                    };
                    let registering = register(this, program);
                    crate::rt::spawn(async move {
                        let registered = registering.await;
                        unanswered.answer(registered.map_err(|e| format!("{e:#}")));
                    });
                    answer
                }
            }
        };
        let answered = crate::inferlet::process::said(
            "bind: register the program",
            pid,
            answer.wait_for(Option::is_some),
        )
        .await
        .map_err(|_| anyhow!("program 0x{hash:016x}: its registration ended unanswered"))?;
        match &*answered {
            Some(Ok(id)) => Ok(*id),
            Some(Err(why)) => Err(anyhow!("{why}")),
            None => Err(anyhow!(
                "program 0x{hash:016x}: its registration ended unanswered"
            )),
        }
    }

    pub(crate) fn stats(&self) -> Arc<SchedulerStats> {
        Arc::clone(&self.inner.stats)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn submit_prebuilt_tracked_with_copy(
        &self,
        request: crate::engine::FireRequest,
        instance_id: ::engine::InstanceId,
        completion: WorkItemCompletion,
        process_id: ProcessId,
        pipeline_id: ProcessId,
        prelaunch_copy: Option<::engine::KvCopy>,
        prelaunch_state_copy: Option<StateCopy>,
        frame: Option<FrameStamp>,
        lora_program: bool,
    ) -> Result<()> {
        self.send(SchedulerItem::Launch {
            pending: PendingRequest::direct(
                request,
                instance_id,
                completion,
                Some(process_id),
                Some(pipeline_id),
                prelaunch_copy,
                prelaunch_state_copy,
                frame,
                lora_program,
            ),
        })
    }

    pub fn frame_truncate(&self, lane: ProcessId, seq: u64, submitted: u32) -> Result<()> {
        self.send(SchedulerItem::FrameTruncate {
            lane,
            seq,
            submitted,
        })
    }

    pub(crate) fn nudge(&self) -> Result<()> {
        self.send(SchedulerItem::Nudge)
    }

    pub async fn register_channels_bind(
        &self,
        pipeline_id: Option<ProcessId>,
        plans: Vec<ChannelRegistration>,
        program: ProgramRegistration,
        mut bind: InstanceBindingPlan,
    ) -> Result<(Vec<RegisteredChannel>, BoundInstance)> {
        let program_id = self
            .program_id(pipeline_id.unwrap_or_default(), program)
            .await?;
        bind.binding.program = program_id;
        // The fire term of the program at these extents counts while the
        // bind is under way, and the activation the fires may take holds it
        // before the bind is sent (`store::shared::programs`). The engine is
        // asked the pair's facts once, at its first bind.
        let _open = match crate::store::registry::try_get(0, bind.engine_id)
            .and_then(|stores| stores.shared)
        {
            Some(shared) => {
                let key = crate::store::shared::programs::FireKey {
                    program: program_id,
                    extents: bind.binding.extents,
                };
                if !shared.lock().unwrap().knows_fire(&key) {
                    self.request(|response| SchedulerItem::FireFacts { key, response })
                        .await??;
                }
                // What the bind adds to the engine's standing guest bytes,
                // asked on the lane before anything is bound: held with the
                // term, refused with it by name, and backed with it below
                // the carve before the bind is sent.
                let standing = self
                    .request(|response| SchedulerItem::StandingAfter {
                        binding: Box::new(bind.binding.clone()),
                        response,
                    })
                    .await??;
                let open = if bind.float {
                    crate::store::shared::programs::ProgramOpen::float(&shared, key, standing)
                } else {
                    crate::store::shared::programs::ProgramOpen::new(&shared, key, standing)
                }
                .map_err(|why| anyhow!("pipeline: {why}"))?;
                let engine = bind.engine_id;
                let pid = pipeline_id.unwrap_or_default();
                crate::inferlet::process::said(
                    "bind: the program's fire span",
                    pid,
                    crate::store::shared::programs::fund_span(
                        &shared,
                        || crate::store::shared::fire_cap(engine),
                        &crate::store::shared::programs::Planner { pid, quorum: pid },
                        &crate::store::shared::programs::LaneMover {
                            engine,
                            epoch_now: || crate::store::shared::programs::epoch_of(engine),
                        },
                    ),
                )
                .await
                .map_err(|why| {
                    anyhow!("pipeline: a program's fires need a wider activation span: {why}")
                })?;
                Some(open)
            }
            None => None,
        };
        let pid = pipeline_id.unwrap_or_default();
        let lane_bind = || {
            let (plans, bind) = (plans.clone(), bind.clone());
            async move {
                crate::inferlet::process::said(
                    "bind: the lane's bind",
                    pid,
                    self.request(|response| SchedulerItem::RegisterChannelsBind {
                        pipeline_id,
                        plans,
                        bind,
                        response,
                    }),
                )
                .await?
            }
        };
        // Suspended meanwhile, its channels parked with it: the bind waits
        // for it to be resident again, as a fire does, never refused.
        let resident = || async move {
            match crate::planner::planner_of(pid) {
                Some(planner) => {
                    crate::inferlet::process::said(
                        "bind: resident again",
                        pid,
                        planner.wait_resident(pid),
                    )
                    .await
                }
                None => Err(crate::planner::PlannerError::Cancelled),
            }
        };
        // Short of a free run for its guest state, it waits on the walk to
        // commit a segment, as a kv admission does, never refused while a
        // segment is left to commit within the budget.
        let engine = bind.engine_id;
        let grown = |prefix: u32, run: u32| async move {
            let shared = crate::store::registry::try_get(0, engine)
                .and_then(|stores| stores.shared)
                .ok_or_else(|| format!("engine {engine} has no shared pool to grow"))?;
            crate::inferlet::process::said(
                "bind: a segment for its guest state",
                pid,
                crate::store::roles::grow_for_guest(&shared, engine, prefix, run),
            )
            .await
        };
        // Short of room the budget holds, with no segment left to commit,
        // it waits for room given back (a process completing, or parked),
        // as an admission waits on pages, never refused for it.
        let room = || {
            crate::store::registry::try_get(0, engine)
                .and_then(|stores| stores.roles)
                .map(|roles| roles.lock().unwrap().room_watch())
        };
        // Whether `pieces` runs of `run` chunks and `singles` chunks beside
        // them stand free for its state.
        let fits = |run: u32, pieces: u32, singles: u32| {
            crate::store::registry::try_get(0, engine)
                .and_then(|stores| stores.roles)
                .is_some_and(|roles| roles.lock().unwrap().room_free(run, pieces, singles))
        };
        // Holding guest state of its own, it is suspended instead: waiting,
        // it would hold room its own bind lacks.
        let holds = || {
            crate::store::registry::try_get(0, engine)
                .and_then(|stores| stores.roles)
                .is_some_and(|roles| roles.lock().unwrap().holds_guest_state(pid))
        };
        // No process is moved off the device for room idle cache holds.
        let idle = |run: u32, pieces: u32, singles: u32| async move {
            let Some(planner) = crate::planner::planner_of(pid).filter(|_| holds()) else {
                return Ok(crate::planner::BindRoomWait::IdleOut);
            };
            crate::inferlet::process::said(
                "bind: idle cache's room for its guest state",
                pid,
                planner.wait_idle_room(pid, run, pieces, singles),
            )
            .await
            .map_err(|why| why.to_string())
        };
        let suspend = |run: u32, pieces: u32, singles: u32| {
            holds()
                && crate::planner::planner_of(pid)
                    .is_some_and(|planner| planner.suspend_for_room(pid, run, pieces, singles))
        };
        // Waiting, it is a waiter in the planner's queue: idle cache goes
        // first, then a resident is parked for it, as for a restore short
        // of room for its roles.
        let planned = |run: u32, pieces: u32, singles: u32| async move {
            let planner = crate::planner::planner_of(pid)
                .ok_or_else(|| "its process has no planner".to_string())?;
            crate::inferlet::process::said(
                "bind: room for its guest state",
                pid,
                planner.wait_bind_room(pid, run, pieces, singles),
            )
            .await
            .map_err(|why| why.to_string())
        };
        // However the bind ends, the pool no longer counts its need.
        let leave = move || {
            if let Some(roles) =
                crate::store::registry::try_get(0, engine).and_then(|stores| stores.roles)
            {
                roles.lock().unwrap().leave_guest_wait(pid);
            }
        };
        // Waiting at the bind for room, the process is a victim the
        // planner may park for an older head, as an idle one is.
        let waiting = move |waits: bool| {
            if let Some(planner) = crate::planner::planner_of(pid) {
                planner.set_bind_waiting(pid, waits);
            }
        };
        // Before it waits, what its process has in flight lands and gives
        // its leases back: nothing else finalizes it while its guest is in
        // this bind.
        let settle = move || crate::pipeline::fire::settle_process_fires(pid);
        let (registered, _, bound) = crate::store::roles::bind_when_resident(
            pid, lane_bind, resident, grown, room, fits, idle, suspend, planned, leave, waiting, settle,
        )
        .await?;
        Ok((registered, bound))
    }

    pub(crate) fn copy_tracked(&self, plan: PreLaunchCopy) -> Result<ControlCompletion> {
        let completion = ControlCompletion::new();
        self.send(SchedulerItem::CopyTracked {
            plan,
            completion: completion.clone(),
        })?;
        Ok(completion)
    }

    pub(crate) async fn host_kv(&self, op: HostKvOp) -> Result<Vec<u8>> {
        self.request(|response| SchedulerItem::HostKv { op, response })
            .await?
    }

    /// Queues `op` on the lane without waiting on it: what the lane answers
    /// arrives on the receiver.
    pub(crate) fn host_kv_queued(
        &self,
        op: HostKvOp,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<Vec<u8>>>> {
        let (response, receiver) = tokio::sync::oneshot::channel();
        self.send(SchedulerItem::HostKv { op, response })?;
        Ok(receiver)
    }

    /// Between frames, on the lane that owns the engine.
    pub(crate) async fn pool(&self, op: PoolOp) -> Result<()> {
        self.request(|response| SchedulerItem::Pool { op, response })
            .await?
    }

    pub fn close_instance(&self, id: ::engine::InstanceId, pacing_wait_id: u64) -> Result<()> {
        self.send(SchedulerItem::CloseInstance { id, pacing_wait_id })
    }

    pub fn close_channel(&self, id: ::engine::ChannelId) -> Result<()> {
        self.send(SchedulerItem::CloseChannel { id })
    }

    pub fn close_channels(&self, ids: Vec<::engine::ChannelId>) -> Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        self.send(SchedulerItem::CloseChannels { ids })
    }
}
