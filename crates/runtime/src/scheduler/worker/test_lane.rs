use super::*;

pub(crate) struct TestLane {
    engine: usize,
    rx: tokio::sync::mpsc::UnboundedReceiver<SchedulerItem>,
}

/// One copy handed to the lane, unlanded.
pub(crate) struct LaneCopy {
    /// Down to the disk tier.
    pub(crate) to_disk: bool,
    /// Up from the disk tier.
    pub(crate) from_disk: bool,
    /// The slots it copies, shape by shape (a shared pool's every copy).
    pub(crate) slots: Option<::engine::chunk::SlotCopy>,
    completion: ControlCompletion,
}

impl LaneCopy {
    /// The copy lands: its completion resolves.
    pub(crate) fn land(self) {
        self.completion.resolve(&Ok(()));
    }

    /// The copy did not land (`engine::Landing::Failed`): its
    /// destination is not to be trusted, its sources hold what they
    /// held.
    pub(crate) fn fail_landing(self, why: &str) {
        self.completion.resolve(&Err(anyhow::Error::new(
            crate::engine::completion::CopyLandingFailed(why.to_string()),
        )));
    }

    /// The tier it copies down to is full (`engine::Error::TierFull`):
    /// nothing on the device changed.
    pub(crate) fn refuse_tier_full(self) {
        self.completion
            .resolve(&Err(anyhow::Error::new(::engine::Error::TierFull(
                "the disk tier's slot file is full (test lane)".into(),
            ))));
    }
}

impl TestLane {
    pub(crate) fn install(engine: usize) -> Self {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SchedulerItem>();
        let handle = SchedulerHandle {
            inner: Arc::new(SchedulerControl {
                tx,
                active_senders: AtomicUsize::new(0),
                shutdown_wait: Condvar::new(),
                shutdown_gate: Mutex::new(()),
                program_ids: Mutex::new(HashMap::new()),
                accepting: AtomicBool::new(true),
                stats: Arc::new(SchedulerStats::default()),
                device_domain: ::engine::MemoryDomain::Device {
                    ordinal: 0,
                    reach: ::engine::DeviceReach::Peer,
                },
                codegen_backend: None,
            }),
        };
        crate::scheduler::set_test_handle(engine, Some(handle));
        Self { engine, rx }
    }

    /// The copies submitted since the last take, in order; every pool
    /// request is answered, a rebind by `rebind` as the lane runs it
    /// (`store::roles::rebind`), every other item dropped.
    pub(crate) fn take(
        &mut self,
        rebind: &mut dyn FnMut(::engine::Rebind) -> Result<()>,
    ) -> Vec<LaneCopy> {
        let mut copies = Vec::new();
        while let Ok(item) = self.rx.try_recv() {
            match item {
                SchedulerItem::CopyTracked { plan, completion } => {
                    let disk = ::engine::MemoryDomain::LocalDisk;
                    let (to_disk, from_disk) = match &plan {
                        PreLaunchCopy::Slots(copy) => (copy.dst == disk, copy.src == disk),
                        PreLaunchCopy::Kv(copy) => (copy.dst == disk, copy.src == disk),
                        PreLaunchCopy::State(_) => (false, false),
                    };
                    let slots = match &plan {
                        PreLaunchCopy::Slots(copy) => Some(copy.clone()),
                        _ => None,
                    };
                    copies.push(LaneCopy {
                        to_disk,
                        from_disk,
                        slots,
                        completion,
                    });
                }
                SchedulerItem::Pool {
                    op: PoolOp::Rebind(change),
                    response,
                } => {
                    let _ = response.send(rebind(change));
                }
                SchedulerItem::Pool { response, .. } => {
                    let _ = response.send(Ok(()));
                }
                SchedulerItem::HostKv { response, .. } => {
                    let _ = response.send(Ok(Vec::new()));
                }
                _ => {}
            }
        }
        copies
    }
}

impl Drop for TestLane {
    fn drop(&mut self) {
        crate::scheduler::set_test_handle(self.engine, None);
    }
}
