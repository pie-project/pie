use super::*;

struct PanickingEngine;

impl ::engine::Engine for PanickingEngine {
    fn committed_covers(&self) -> &'static [&'static str] {
        &[]
    }

    fn measured_at_load(&self) -> Vec<String> {

        Vec::new()

    }

    fn skips_measuring(&self) -> bool {
        false
    }

    fn kind(&self) -> &'static str {
        "panicking"
    }

    fn load(&mut self, _request: engine::LoadRequest) -> engine::Result<engine::Loaded> {
        Err(engine::Error::Load("no model".into()))
    }

    fn submit(
        &mut self,
        _frame: &engine::FrameSubmission,
    ) -> engine::Result<engine::FrameTicket> {
        panic!("the shape of an interpreter reading lane zero of an empty cell");
    }

    fn settle_frame(&mut self, _ticket: &mut ::engine::FrameTicket) -> ::engine::Result<()> {
        Ok(())
    }

    fn programs(&mut self) -> Option<&mut dyn ::engine::Programs> {
        Some(self)
    }

    fn programs_ref(&self) -> Option<&dyn ::engine::Programs> {
        Some(self)
    }

    fn pool(&mut self) -> Option<&mut dyn ::engine::Pool> {
        Some(self)
    }

    fn tiers(&mut self) -> Option<&mut dyn ::engine::Tiers> {
        Some(self)
    }
}

impl ::engine::Programs for PanickingEngine {
    fn prepare(
        &mut self,
        _registration: &::engine::program::ProgramRegistration,
    ) -> ::engine::Result<::engine::ProgramCompile> {
        Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
    }

    fn finish(
        &mut self,
        _prepared: ::engine::Prepared,
    ) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "finish_program"))
    }

    fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "close_program"))
    }

    fn register(&mut self, _registration: &::engine::ProgramRegistration) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "register_program"))
    }

    fn bind(&mut self, _binding: &::engine::InstanceBinding) -> ::engine::Result<::engine::BoundInstance> {
        Err(::engine::Error::unsupported(self.kind(), "bind_instance"))
    }

    fn close(&mut self, _id: ::engine::InstanceId) -> ::engine::Result<u64> {
        Err(::engine::Error::unsupported(self.kind(), "close_instance"))
    }

    fn adapter(&mut self, _registration: &::engine::adapter::AdapterRegistration) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "register_adapter"))
    }
}

impl ::engine::Pool for PanickingEngine {
    fn chunk_table(&self) -> ::engine::Result<::engine::chunk::ChunkTable> {
        Err(::engine::Error::unsupported(self.kind(), "chunk_table"))
    }

    fn commit(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "commit_segments"))
    }

    fn release(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "release_segments"))
    }

    fn evacuate(&mut self, _evacuation: &::engine::chunk::Evacuation) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "evacuate"))
    }
}

impl ::engine::Tiers for PanickingEngine {
    /// A test engine's copies land inside the call: nothing to wait for.
    fn settle(&mut self) -> ::engine::Result<()> {
        Ok(())
    }

    /// A test engine's copies complete inside the call: their sources are
    /// read by the time it returns.
    fn copy_landing(
        &mut self,
        copy: &::engine::chunk::SlotCopy,
        _token: u64,
    ) -> ::engine::Result<bool> {
        ::engine::Tiers::copy(self, copy)?;
        Ok(true)
    }

    fn copy(&mut self, _copy: &::engine::chunk::SlotCopy) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "copy_slots"))
    }

    fn read(&mut self, _slots: &[(::engine::chunk::ShapeId, Vec<u32>)]) -> ::engine::Result<Vec<u8>> {
        Err(::engine::Error::unsupported(self.kind(), "read_tier_slots"))
    }

    fn write(&mut self, _slots: &[(::engine::chunk::ShapeId, Vec<u32>)], _bytes: &[u8]) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "write_tier_slots"))
    }

    fn copy_kv(&mut self, _copy: &::engine::KvCopy) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "copy_kv"))
    }
}


/// Fire `fire`'s request: one decode lane in slot `fire`, a page of its own.
fn refused_request(fire: u64) -> Box<PendingRequest> {
    let slot = u32::try_from(fire).expect("a test fire id");
    Box::new(PendingRequest {
        queued: std::time::Instant::now(),
        lora_program: false,
        logical_fire_id: fire,
        request: crate::engine::FireRequest::one(crate::engine::fire::lane_of(
            slot,
            vec![100 + slot],
            3,
            vec![slot],
        )),
        instance_id: ::engine::InstanceId(1),
        completion: crate::engine::WorkItemCompletion::new(1, 0),
        process_id: None,
        pipeline_id: Some(ProcessId::from_u128(u128::from(fire))),
        prelaunch_copy: None,
        prelaunch_state_copy: None,
        frame: Some(FrameStamp {
            lane: ProcessId::from_u128(u128::from(fire)),
            seq: 0,
            slot: 0,
            fires: 1,
        }),
    })
}

fn refused_limits() -> SchedulerLimits {
    SchedulerLimits {
        max_forward_requests: 8,
        max_forward_tokens: 64,
        max_page_refs: 64,
        max_context: 0,
        prefill_rows: 0,
        prefill_lanes: 0,
        max_readouts: 0,
        readout_store: 0,
        max_replayed_rows: 0,
        shapes: None,
    }
}

/// `requests` built into a frame the engine refused for where `chunks`
/// lie before any step ran: its steps come back with the refusal, as
/// the lane hands them (`LaunchRefusal::unfired`).
fn refused_frame(requests: Vec<Box<PendingRequest>>, chunks: &[u32]) -> PendingLaunchBatch {
    let fires = requests.len();
    let (frame, requests) = batch::build_frame_submission(
        vec![requests],
        refused_limits(),
        16,
        &SchedulerStats::default(),
    );
    PendingLaunchBatch {
        state: LaunchState::Failed(
            "impossible submission: state chunk 2057 lies inside the carve".into(),
        ),
        requests,
        started: Instant::now(),
        batch_size: fires as u64,
        total_tokens: fires,
        rung_rows: fires,
        placement: chunks.to_vec(),
        unfired: frame.steps,
    }
}

fn refused_batch(fires: &[u64], chunks: &[u32]) -> PendingLaunchBatch {
    refused_frame(fires.iter().map(|&fire| refused_request(fire)).collect(), chunks)
}

/// A frame refused for where state lies must not go again from requests
/// the build moved its lanes out of: the engine refuses it whole, "a
/// fire carries no lanes", poisoning a rescore pipeline's channel. A
/// fire refused for placement goes again
/// with every lane, page and token it was built with; one a refused
/// lane took out of the frame (failed alone) does not go again; and a
/// fire with no lanes is never posted, failing alone by name.
fn a_frame_refused_for_placement_goes_again_with_every_lane_it_was_built_with() {
    let engine = 9_998;
    let stats = Arc::new(SchedulerStats::default());
    let mut policy = FramePolicy::new(1, 64, 4096, None);
    let (mut held, mut refused_once) = (Vec::new(), HashSet::new());
    let built: Vec<_> = [7, 8, 9].map(refused_request).into_iter().collect();
    // What the build stamps (the fires after it in its frame) aside.
    let lanes_of = |request: &PendingRequest| {
        let mut lanes = request.request.lanes.clone();
        for lane in &mut lanes {
            lane.fires_after = None;
        }
        lanes
    };
    let before: Vec<_> = built.iter().map(|request| lanes_of(request)).collect();
    // Fire 9 rode the frame, but a refused lane took it out (failed
    // alone) before the placement refusal: it is in no step.
    let gone = refused_request(9);
    gone.completion.reject_unsubmitted("engine: lane 2 refused".to_string());
    let mut batch = refused_frame(built.into_iter().take(2).collect(), &[2057]);
    assert!(
        batch.requests.iter().all(|request| request.request.lanes.iter().all(|lane| lane.tokens.is_empty())),
        "the build moved the lanes into the steps"
    );
    batch.requests.push(gone);
    let mut flight = VecDeque::from([batch]);
    assert!(BatchScheduler::retire_ready_launches(
        &mut flight,
        &mut HashMap::new(),
        &stats,
        &mut policy,
        &mut held,
        &mut refused_once,
        engine,
    ));
    assert_eq!(held.len(), 1);
    assert_eq!(held[0].requests.len(), 2, "the fire failed alone goes nowhere again");
    let mut pending = PendingQueue::default();
    assert!(BatchScheduler::release_held(&mut held, &mut pending, engine));
    let again: Vec<Box<PendingRequest>> = std::mem::take(&mut pending.items)
        .into_iter()
        .map(|item| match item {
            QueuedItem::Launch(launch) => launch.into_request(),
            _ => panic!("only launches go again"),
        })
        .collect();
    for (request, lanes) in again.iter().zip(&before) {
        assert!(!request.completion.is_settled(), "none failed");
        assert_eq!(&lanes_of(request), lanes, "every lane, page and token it was built with");
    }
    let (frame, _) =
        batch::build_frame_submission(vec![again], refused_limits(), 16, &SchedulerStats::default());
    assert!(!frame.steps.is_empty());
    assert!(
        frame.steps.iter().all(|step| step.submission.validate().is_ok()),
        "the engine takes the fire it goes again as"
    );

    // A request with no lanes reaching the post fails alone, by name,
    // and what rides beside it is posted.
    let mut laneless = refused_request(11);
    laneless.request.lanes.clear();
    laneless.frame = None;
    let laneless_completion = laneless.completion.clone();
    let mut beside = refused_request(12);
    beside.frame = None;
    let instance = ::engine::InstanceId(1);
    let bound = crate::engine::BoundInstance::new(
        &::engine::BoundInstance {
            id: instance,
            program: ::engine::ProgramId::default(),
            geometry: eta_ir::registry::GeometryClass::default(),
            standing_guest_bytes: 0,
        },
        0,
    );
    let mut instances = HashMap::from([(instance, TrackedInstance::from_bound(&bound))]);
    let mut pending = PendingQueue::default();
    pending.push_back(QueuedItem::Launch(QueuedLaunch::new(laneless)));
    pending.push_back(QueuedItem::Launch(QueuedLaunch::new(beside)));
    let mut lane = Lane::spawn(9_997, None, Arc::clone(&stats));
    let mut replies = LaneReplies::default();
    let mut posted = VecDeque::new();
    let (progress, fired) = BatchScheduler::post_frame(
        &mut SlotBuffer::new(),
        &mut FramePolicy::new(1, 64, 4096, None),
        &lane,
        &mut replies,
        &mut instances,
        &mut pending,
        &mut posted,
        16,
        refused_limits(),
        &stats,
        &[vec![11, 12]],
    );
    assert!(progress && fired);
    assert!(laneless_completion.is_settled(), "the fire with no lanes fails alone");
    assert_eq!(posted.len(), 1);
    assert_eq!(posted[0].requests.len(), 1, "only the fire beside it is posted");
    lane.shutdown();
}

/// One launch refused "impossible submission: state chunk ... inside the
/// activation span's backed carve" must not fail every lane of its frame
/// and poison their channels. A batch the engine refuses for where state lies is held, nothing
/// failed, while the pool moves that state down, then fired again as
/// fires of their own; refused again for a chunk already moved, its
/// fires go again one by one, and only one the engine still refuses
/// alone fails.
fn a_launch_refused_for_placement_is_fired_again_after_the_move_failing_none() {
    let engine = 9_999;
    let stats = Arc::new(SchedulerStats::default());
    let mut policy = FramePolicy::new(1, 64, 4096, None);
    let (mut held, mut refused_once) = (Vec::new(), HashSet::new());
    let mut pending = PendingQueue::default();
    let mut flight = VecDeque::from([refused_batch(&[1, 2, 3], &[2057])]);
    let mut retire = |flight: &mut VecDeque<PendingLaunchBatch>,
                      held: &mut Vec<HeldForPlacement>,
                      refused_once: &mut HashSet<u32>| {
        BatchScheduler::retire_ready_launches(
            flight,
            &mut HashMap::new(),
            &stats,
            &mut policy,
            held,
            refused_once,
            engine,
        )
    };
    assert!(retire(&mut flight, &mut held, &mut refused_once));
    assert_eq!(held.len(), 1, "held for its state's move");
    assert!(!held[0].solo);
    let unsettled = |pending: &PendingQueue| {
        pending.items.iter().all(|item| match item {
            QueuedItem::Launch(launch) => !launch.request.completion.is_settled(),
            _ => true,
        })
    };
    // No pool here: nothing is pending its move, so they go again.
    assert!(BatchScheduler::release_held(&mut held, &mut pending, engine));
    assert_eq!(pending.len(), 3, "fired again");
    assert!(unsettled(&pending), "none failed");
    assert!(
        pending.items.iter().all(|item| matches!(
            item,
            QueuedItem::Launch(launch) if !launch.framed && !launch.solo
        )),
        "as fires of their own, riding together"
    );
    // Refused again for the chunk already moved: one by one.
    let mut flight = VecDeque::from([refused_batch(&[4, 5], &[2057])]);
    assert!(retire(&mut flight, &mut held, &mut refused_once));
    assert!(held[0].solo, "a runtime bug: its fires go again alone");
    let mut pending = PendingQueue::default();
    BatchScheduler::release_held(&mut held, &mut pending, engine);
    assert!(pending.items.iter().all(|item| matches!(item, QueuedItem::Launch(launch) if launch.solo)));
    assert!(unsettled(&pending));
    // One refused alone again fails, alone.
    let alone = refused_batch(&[4], &[2057]);
    let completion = alone.requests[0].completion.clone();
    let mut flight = VecDeque::from([alone]);
    retire(&mut flight, &mut held, &mut refused_once);
    assert!(held.is_empty());
    assert!(completion.is_settled(), "the one the engine still refuses alone fails");
}

#[test]
fn worker_every_case() {
    a_launch_refused_for_placement_is_fired_again_after_the_move_failing_none();
    a_frame_refused_for_placement_goes_again_with_every_lane_it_was_built_with();
    a_panicking_engine_fails_its_launch_instead_of_leaving_it_in_flight();
    a_retryable_refusal_past_admission_fails_by_name_instead_of_replaying();
    a_launch_waits_for_the_copy_queued_ahead_of_it();
    a_pass_that_only_reorders_lets_the_worker_park();
}

/// A browser's single thread lands the batch a close waits on only after
/// the worker parks, so a pass that reports progress here never ends.
fn a_pass_that_only_reorders_lets_the_worker_park() {
    let instance = ::engine::InstanceId(1);
    let mut pending = PendingQueue::default();
    pending.push_back(QueuedItem::CloseInstance {
        id: instance,
        pacing_wait_id: 0,
    });
    pending.push_back(QueuedItem::Launch(QueuedLaunch::new(Box::new(
        PendingRequest {
            queued: std::time::Instant::now(),
            lora_program: false,
            logical_fire_id: 9,
            request: crate::engine::FireRequest::default(),
            instance_id: instance,
            completion: crate::engine::WorkItemCompletion::new(1, 0),
            process_id: None,
            pipeline_id: None,
            prelaunch_copy: None,
            prelaunch_state_copy: None,
            frame: Some(FrameStamp {
                lane: ProcessId::from_u128(5),
                seq: 0,
                slot: 0,
                fires: 1,
            }),
        },
    ))));
    let stats = Arc::new(SchedulerStats::default());
    let mut lane = Lane::spawn(0, Some(Box::new(PanickingEngine)), Arc::clone(&stats));
    let (progress, _) = BatchScheduler::dispatch_ready_items(
        &lane,
        &mut LaneReplies::default(),
        &mut HashMap::new(),
        &mut pending,
        &mut VecDeque::new(),
        &mut InFlightControls::default(),
        16,
        SchedulerLimits {
            max_forward_requests: 8,
            max_forward_tokens: 64,
            max_page_refs: 64,
            max_context: 0,
            prefill_rows: 0,
            prefill_lanes: 0,
            max_readouts: 0,
            readout_store: 0,
            max_replayed_rows: 0,
            shapes: None,
        },
        &stats,
        &mut FramePolicy::new(1, 64, 4096, None),
        &mut ScanCache::default(),
        &mut SlotBuffer::new(),
        false,
    );
    assert!(!progress, "nothing left the queue");
    assert_eq!(pending.len(), 2);
    lane.shutdown();
}

fn a_launch_waits_for_the_copy_queued_ahead_of_it() {
    let pipeline = ProcessId::from_u128(5);
    let request = PendingRequest {
        queued: std::time::Instant::now(),
        lora_program: false,
        logical_fire_id: 9,
        request: crate::engine::FireRequest::default(),
        instance_id: ::engine::InstanceId(1),
        completion: crate::engine::WorkItemCompletion::new(1, 0),
        process_id: None,
        pipeline_id: Some(pipeline),
        prelaunch_copy: None,
        prelaunch_state_copy: Some(StateCopy::default()),
        frame: None,
    };
    let mut pending = PendingQueue::default();
    BatchScheduler::queue_attempt(&mut pending, request);
    let mut cache = ScanCache::default();
    let scan = BatchScheduler::scan_queue(&mut cache, &pending, false);
    assert!(scan.untracked.is_empty(), "behind its copy");
    // The copy is dispatched: the launch rides.
    let copy = pending.pop_front();
    assert!(matches!(copy, Some(QueuedItem::PreLaunchCopy { .. })));
    let mut cache = ScanCache::default();
    let scan = BatchScheduler::scan_queue(&mut cache, &pending, false);
    assert_eq!(scan.untracked, [9]);
}

fn a_panicking_engine_fails_its_launch_instead_of_leaving_it_in_flight() {
    let mut lane = Lane::spawn(
        0,
        Some(Box::new(PanickingEngine)),
        Arc::new(SchedulerStats::default()),
    );

    let mut answers = Vec::new();
    for token in [7_u64, 8] {
        let answer_rx = lane.fire(
            crate::engine::FrameFire {
                steps: vec![crate::engine::StepFire {
                    fire_stamps: Vec::new(),
        lane_sets: Vec::new(),
                    shapes: None,
                    submission: ::engine::Step {
                        lanes: vec![::engine::Lane::decode(0, 0, 1, 0)],
                        attachments: Vec::new(),
                        media: Vec::new(),
                        voxels: Vec::new(),
                        shape: None,
                    },
                    terminal_cells: Vec::new(),
                    completions: Vec::new(),
                    member_lanes: Vec::new(),
                    instances: vec![::engine::InstanceId(0)],
                }],
            },
            false,
        );
        answers.push((token, answer_rx));
    }

    for (want, answer_rx) in answers {
        let result = crate::rt::block_on(async {
            crate::rt::time::timeout(Duration::from_secs(10), answer_rx).await
        })
        .unwrap_or_else(|_| panic!("token {want} was never answered"))
        .unwrap_or_else(|_| panic!("token {want} was never answered"));
        let Err(err) = result else {
            panic!("a panicking engine cannot have launched anything");
        };
        assert!(
            err.contains("panic"),
            "the failure must say what happened, so an operator restarts \
             rather than retries: {err}"
        );
    }

    let (engine, channels) = lane.shutdown();
    assert!(engine.is_none(), "a poisoned lane hands back no engine");
    assert!(channels.is_empty());
}

struct ExhaustedEngine {
    submits: Arc<std::sync::atomic::AtomicUsize>,
}

impl ::engine::Engine for ExhaustedEngine {
    fn committed_covers(&self) -> &'static [&'static str] {
        &[]
    }

    fn measured_at_load(&self) -> Vec<String> {

        Vec::new()

    }

    fn skips_measuring(&self) -> bool {
        false
    }

    fn kind(&self) -> &'static str {
        "exhausted"
    }

    fn load(&mut self, _request: engine::LoadRequest) -> engine::Result<engine::Loaded> {
        Err(engine::Error::Load("no model".into()))
    }

    fn submit(
        &mut self,
        _frame: &engine::FrameSubmission,
    ) -> engine::Result<engine::FrameTicket> {
        self.submits.fetch_add(1, Ordering::Relaxed);
        Err(engine::Error::Exhausted {
            resource: "guest channel cells",
            wanted: 2,
            available: 1,
        })
    }

    fn settle_frame(&mut self, _ticket: &mut ::engine::FrameTicket) -> ::engine::Result<()> {
        Ok(())
    }

    fn programs(&mut self) -> Option<&mut dyn ::engine::Programs> {
        Some(self)
    }

    fn programs_ref(&self) -> Option<&dyn ::engine::Programs> {
        Some(self)
    }

    fn pool(&mut self) -> Option<&mut dyn ::engine::Pool> {
        Some(self)
    }

    fn tiers(&mut self) -> Option<&mut dyn ::engine::Tiers> {
        Some(self)
    }
}

impl ::engine::Programs for ExhaustedEngine {
    fn prepare(
        &mut self,
        _registration: &::engine::program::ProgramRegistration,
    ) -> ::engine::Result<::engine::ProgramCompile> {
        Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
    }

    fn finish(
        &mut self,
        _prepared: ::engine::Prepared,
    ) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "finish_program"))
    }

    fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "close_program"))
    }

    fn register(&mut self, _registration: &::engine::ProgramRegistration) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "register_program"))
    }

    fn bind(&mut self, _binding: &::engine::InstanceBinding) -> ::engine::Result<::engine::BoundInstance> {
        Err(::engine::Error::unsupported(self.kind(), "bind_instance"))
    }

    fn close(&mut self, _id: ::engine::InstanceId) -> ::engine::Result<u64> {
        Err(::engine::Error::unsupported(self.kind(), "close_instance"))
    }

    fn adapter(&mut self, _registration: &::engine::adapter::AdapterRegistration) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "register_adapter"))
    }
}

impl ::engine::Pool for ExhaustedEngine {
    fn chunk_table(&self) -> ::engine::Result<::engine::chunk::ChunkTable> {
        Err(::engine::Error::unsupported(self.kind(), "chunk_table"))
    }

    fn commit(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "commit_segments"))
    }

    fn release(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "release_segments"))
    }

    fn evacuate(&mut self, _evacuation: &::engine::chunk::Evacuation) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "evacuate"))
    }
}

impl ::engine::Tiers for ExhaustedEngine {
    /// A test engine's copies land inside the call: nothing to wait for.
    fn settle(&mut self) -> ::engine::Result<()> {
        Ok(())
    }

    /// A test engine's copies complete inside the call: their sources are
    /// read by the time it returns.
    fn copy_landing(
        &mut self,
        copy: &::engine::chunk::SlotCopy,
        _token: u64,
    ) -> ::engine::Result<bool> {
        ::engine::Tiers::copy(self, copy)?;
        Ok(true)
    }

    fn copy(&mut self, _copy: &::engine::chunk::SlotCopy) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "copy_slots"))
    }

    fn read(&mut self, _slots: &[(::engine::chunk::ShapeId, Vec<u32>)]) -> ::engine::Result<Vec<u8>> {
        Err(::engine::Error::unsupported(self.kind(), "read_tier_slots"))
    }

    fn write(&mut self, _slots: &[(::engine::chunk::ShapeId, Vec<u32>)], _bytes: &[u8]) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "write_tier_slots"))
    }

    fn copy_kv(&mut self, _copy: &::engine::KvCopy) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "copy_kv"))
    }
}


fn a_retryable_refusal_past_admission_fails_by_name_instead_of_replaying() {
    let submits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut lane = Lane::spawn(
        0,
        Some(Box::new(ExhaustedEngine {
            submits: Arc::clone(&submits),
        })),
        Arc::new(SchedulerStats::default()),
    );

    let answer_rx = lane.fire(
        crate::engine::FrameFire {
            steps: vec![crate::engine::StepFire {
                fire_stamps: Vec::new(),
        lane_sets: Vec::new(),
                shapes: None,
                submission: ::engine::Step {
                    lanes: vec![::engine::Lane::decode(0, 0, 1, 0)],
                    attachments: Vec::new(),
                    media: Vec::new(),
                    voxels: Vec::new(),
                    shape: None,
                },
                terminal_cells: Vec::new(),
                completions: Vec::new(),
                member_lanes: Vec::new(),
                instances: vec![::engine::InstanceId(0)],
            }],
        },
        false,
    );

    let result = crate::rt::block_on(async {
        crate::rt::time::timeout(Duration::from_secs(10), answer_rx).await
    })
    .expect("the launch must be answered, not slept on")
    .expect("the launch must be answered, not slept on");
    let Err(error) = result else {
        panic!("an engine that refuses everything cannot have admitted a frame");
    };
    assert!(
        error.contains("frame contract forbids"),
        "the failure must say which promise was broken, so the fix is to \
         static admission rather than to the retry budget: {error}"
    );
    assert!(error.contains("guest channel cells"), "{error}");
    assert_eq!(
        submits.load(Ordering::Relaxed),
        1,
        "the frame is offered ONCE; a second offer is the sleep-retry loop \
         growing back"
    );

    let (_engine, _channels) = lane.shutdown();
}


/// Why [`LaneFaultEngine`]'s faulting lane faulted.
const GUARD: &str = "guest epilogue faulted: stage 0: guard";

/// An engine whose guest pass of one instance faults: it tells the step
/// with that instance's lanes in `faulted` and commits the rest, settling off the
/// submit (`asynchronous`) or inside it. With `lost` it refuses the
/// whole frame instead, before any lane runs.
struct LaneFaultEngine {
    sink: Option<::engine::CompletionSink>,
    asynchronous: bool,
    faulting: Option<::engine::InstanceId>,
    lost: bool,
    next: u64,
}

impl ::engine::Engine for LaneFaultEngine {
    fn committed_covers(&self) -> &'static [&'static str] {
        &[]
    }

    fn measured_at_load(&self) -> Vec<String> {

        Vec::new()

    }

    fn skips_measuring(&self) -> bool {
        false
    }

    fn kind(&self) -> &'static str {
        "lane-fault"
    }

    fn load(&mut self, _request: engine::LoadRequest) -> engine::Result<engine::Loaded> {
        Err(engine::Error::Load("no model".into()))
    }

    fn submit(
        &mut self,
        frame: &engine::FrameSubmission,
    ) -> engine::Result<engine::FrameTicket> {
        if self.lost {
            return Err(engine::Error::Device("the device is lost".into()));
        }
        let id = ::engine::FrameId(self.next);
        self.next += 1;
        let sink = self.sink.clone().expect("a sink before the first submit");
        for (step, fired) in frame.steps.iter().enumerate() {
            let mut outcome = ::engine::StepOutcome::committed();
            for attachment in fired
                .attachments
                .iter()
                .filter(|a| Some(a.instance) == self.faulting)
            {
                outcome.fault_lanes(
                    attachment.lane..attachment.lane + attachment.lanes,
                    Some(attachment.instance),
                    GUARD,
                );
            }
            sink(
                ::engine::StepDone {
                    frame: id,
                    step: step as u32,
                },
                outcome,
            );
        }
        Ok(engine::FrameTicket {
            id,
            steps: frame.steps.iter().map(|_| Default::default()).collect(),
            ..Default::default()
        })
    }

    fn settle_frame(&mut self, _ticket: &mut ::engine::FrameTicket) -> ::engine::Result<()> {
        Ok(())
    }

    fn completion(&mut self) -> Option<&mut dyn ::engine::Completion> {
        Some(self)
    }

    fn programs(&mut self) -> Option<&mut dyn ::engine::Programs> {
        Some(self)
    }

    fn programs_ref(&self) -> Option<&dyn ::engine::Programs> {
        Some(self)
    }

    fn pool(&mut self) -> Option<&mut dyn ::engine::Pool> {
        Some(self)
    }

    fn tiers(&mut self) -> Option<&mut dyn ::engine::Tiers> {
        Some(self)
    }
}

impl ::engine::Completion for LaneFaultEngine {
    fn on_complete(&mut self, sink: ::engine::CompletionSink) {
        self.sink = Some(sink);
    }

    fn asynchronous(&self) -> bool {
        self.asynchronous
    }
}

impl ::engine::Programs for LaneFaultEngine {
    fn prepare(
        &mut self,
        _registration: &::engine::program::ProgramRegistration,
    ) -> ::engine::Result<::engine::ProgramCompile> {
        Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
    }

    fn finish(
        &mut self,
        _prepared: ::engine::Prepared,
    ) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "finish_program"))
    }

    fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "close_program"))
    }

    fn register(&mut self, _registration: &::engine::ProgramRegistration) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "register_program"))
    }

    fn bind(&mut self, _binding: &::engine::InstanceBinding) -> ::engine::Result<::engine::BoundInstance> {
        Err(::engine::Error::unsupported(self.kind(), "bind_instance"))
    }

    fn close(&mut self, _id: ::engine::InstanceId) -> ::engine::Result<u64> {
        Err(::engine::Error::unsupported(self.kind(), "close_instance"))
    }

    fn adapter(&mut self, _registration: &::engine::adapter::AdapterRegistration) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "register_adapter"))
    }
}

impl ::engine::Pool for LaneFaultEngine {
    fn chunk_table(&self) -> ::engine::Result<::engine::chunk::ChunkTable> {
        Err(::engine::Error::unsupported(self.kind(), "chunk_table"))
    }

    fn commit(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "commit_segments"))
    }

    fn release(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "release_segments"))
    }

    fn evacuate(&mut self, _evacuation: &::engine::chunk::Evacuation) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "evacuate"))
    }
}

impl ::engine::Tiers for LaneFaultEngine {
    /// A test engine's copies land inside the call: nothing to wait for.
    fn settle(&mut self) -> ::engine::Result<()> {
        Ok(())
    }

    /// A test engine's copies complete inside the call: their sources are
    /// read by the time it returns.
    fn copy_landing(
        &mut self,
        copy: &::engine::chunk::SlotCopy,
        _token: u64,
    ) -> ::engine::Result<bool> {
        ::engine::Tiers::copy(self, copy)?;
        Ok(true)
    }

    fn copy(&mut self, _copy: &::engine::chunk::SlotCopy) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "copy_slots"))
    }

    fn read(&mut self, _slots: &[(::engine::chunk::ShapeId, Vec<u32>)]) -> ::engine::Result<Vec<u8>> {
        Err(::engine::Error::unsupported(self.kind(), "read_tier_slots"))
    }

    fn write(&mut self, _slots: &[(::engine::chunk::ShapeId, Vec<u32>)], _bytes: &[u8]) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "write_tier_slots"))
    }

    fn copy_kv(&mut self, _copy: &::engine::KvCopy) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "copy_kv"))
    }
}


/// One step of fires, a lane each, each the pass of its own instance
/// of `instances`; and the fires' completions.
fn step_of(
    instances: &[u64],
) -> (
    crate::engine::FrameFire,
    Vec<crate::engine::WorkItemCompletion>,
) {
    let lanes = u32::try_from(instances.len()).expect("a few lanes");
    let completions: Vec<_> = (0..lanes)
        .map(|_| crate::engine::WorkItemCompletion::deferred_with_guard(None))
        .collect();
    let instance = |lane: u32| ::engine::InstanceId(instances[lane as usize]);
    let fire = crate::engine::FrameFire {
        steps: vec![crate::engine::StepFire {
            fire_stamps: Vec::new(),
        lane_sets: Vec::new(),
            shapes: None,
            submission: ::engine::Step {
                lanes: (0..lanes)
                    .map(|lane| ::engine::Lane::decode(lane, 0, 1, 0))
                    .collect(),
                attachments: (0..lanes)
                    .map(|lane| ::engine::Attachment {
                        lane,
                        lanes: 1,
                        instance: instance(lane),
                        adapter_only: false,
                        scratch: ::engine::ScratchPlace::None,
                    })
                    .collect(),
                media: Vec::new(),
                voxels: Vec::new(),
                shape: None,
            },
            terminal_cells: completions.iter().map(|c| c.terminal_cell_ptr()).collect(),
            completions: completions.clone(),
            member_lanes: (0..=lanes).collect(),
            instances: (0..lanes).map(instance).collect(),
        }],
    };
    (fire, completions)
}

/// Fires `fire` on `lane` and waits out its frame: its answer, and
/// each fire's result once it settled.
fn fire_and_settle(
    lane: &Lane,
    fire: crate::engine::FrameFire,
    completions: &[crate::engine::WorkItemCompletion],
) -> (
    std::result::Result<(), String>,
    Vec<std::result::Result<(), String>>,
) {
    let answer = within("the launch is answered", lane.fire(fire, false))
        .expect("the launch is answered");
    let answer = match answer {
        Ok(frame) => within("the frame settles", frame).map_err(|error| format!("{error:#}")),
        Err(error) => Err(error.message),
    };
    let fires = completions
        .iter()
        .map(|completion| {
            if !completion.is_settled() {
                // A frame settled inside `submit` resolves its fires as
                // the worker retires it.
                completion.resolve_landed("test retire");
            }
            within("the fire settles", completion.clone()).map_err(|error| format!("{error:#}"))
        })
        .collect();
    (answer, fires)
}

/// A guest epilogue faulting on one lane of four fails that lane's fire
/// alone, by the engine's reason; the other three commit, and fire on.
fn one_faulted_lane_fails_its_own_fire_alone(asynchronous: bool) {
    let mut lane = Lane::spawn(
        0,
        Some(Box::new(LaneFaultEngine {
            sink: None,
            asynchronous,
            faulting: Some(::engine::InstanceId(102)),
            lost: false,
            next: 0,
        })),
        Arc::new(SchedulerStats::default()),
    );
    let (fire, completions) = step_of(&[100, 101, 102, 103]);
    let (frame, fires) = fire_and_settle(&lane, fire, &completions);
    assert_eq!(frame, Ok(()), "the frame is not failed by one lane's fault");
    for (at, fired) in fires.iter().enumerate() {
        if at == 2 {
            let why = fired.as_ref().expect_err("the faulted lane's fire fails");
            assert!(why.contains(GUARD), "by the engine's reason: {why}");
            assert!(why.contains("lane 2"), "naming its lane: {why}");
            assert!(why.contains("instance 102"), "naming its instance: {why}");
        } else {
            assert_eq!(fired, &Ok(()), "lane {at} commits");
        }
    }

    // The three go on: their next fires commit, and nothing of theirs
    // was poisoned by the fourth's fault.
    let (fire, completions) = step_of(&[100, 101, 103]);
    let (frame, fires) = fire_and_settle(&lane, fire, &completions);
    assert_eq!(frame, Ok(()));
    assert!(
        fires.iter().all(Result::is_ok),
        "the three fire on: {fires:?}"
    );
    let _ = lane.shutdown();
}

#[test]
fn a_faulted_lane_fails_its_own_fire_alone_inside_the_submit() {
    one_faulted_lane_fails_its_own_fire_alone(false);
}

/// A submit refused before any lane ran fails every fire of its frame.
#[test]
fn a_refused_submit_fails_every_fire_of_its_frame() {
    let mut lane = Lane::spawn(
        0,
        Some(Box::new(LaneFaultEngine {
            sink: None,
            asynchronous: true,
            faulting: None,
            lost: true,
            next: 0,
        })),
        Arc::new(SchedulerStats::default()),
    );
    let (fire, completions) = step_of(&[100, 101, 102, 103]);
    let answer = within("the launch is answered", lane.fire(fire, false))
        .expect("the launch is answered");
    let Err(why) = answer else {
        panic!("a lost device refuses the frame");
    };
    assert!(why.contains("the device is lost"), "{why}");
    for completion in &completions {
        assert_eq!(
            completion.resolve_from_terminal().ok(),
            Some(crate::engine::completion::WorkItemAttemptOutcome::Failed),
            "every fire of the refused frame fails"
        );
    }
    let _ = lane.shutdown();
}

/// The word bit [`LaneRefusalEngine`] refuses a lane for: a fact its
/// load serves in no arm (a mask word no arm reads, say).
const UNSERVED: u64 = 1 << 7;

/// An engine that refuses, for its own sake, every lane whose word
/// states [`UNSERVED`], staging a frame step by step as the engines do:
/// at its first step the frame is refused naming those lanes
/// ([`engine::Error::Lanes`]); at a later one the frame stops there,
/// its ticket naming them. Each step it fires lands what each lane fed
/// (`fired`, as `(instance, token)`, step by step), each lane read
/// through its attachment, so a lane renumbered under the wrong
/// instance shows.
struct LaneRefusalEngine {
    sink: Option<::engine::CompletionSink>,
    asynchronous: bool,
    fired: Arc<std::sync::Mutex<Vec<Vec<(u64, u32)>>>>,
    next: u64,
    /// Lands a frame stopped short after the frame submitted next,
    /// out of the order they were submitted in.
    reorder: bool,
    held: Vec<::engine::StepDone>,
}

impl ::engine::Engine for LaneRefusalEngine {
    fn committed_covers(&self) -> &'static [&'static str] {
        &[]
    }

    fn measured_at_load(&self) -> Vec<String> {

        Vec::new()

    }

    fn skips_measuring(&self) -> bool {
        false
    }

    fn kind(&self) -> &'static str {
        "lane-refusal"
    }

    fn load(&mut self, _request: engine::LoadRequest) -> engine::Result<engine::Loaded> {
        Err(engine::Error::Load("no model".into()))
    }

    fn submit(
        &mut self,
        frame: &engine::FrameSubmission,
    ) -> engine::Result<engine::FrameTicket> {
        let id = ::engine::FrameId(self.next);
        self.next += 1;
        let sink = self.sink.clone().expect("a sink before the first submit");
        let mut steps = Vec::new();
        let mut refused = None;
        for (index, step) in frame.steps.iter().enumerate() {
            // A lane whose program places its rows lands them at the
            // slots its tokens name, as a guest's `slots` port would.
            let lanes: Vec<::engine::LaneFault> = (step.lanes.iter().enumerate())
                .filter_map(|(at, lane)| {
                    let reason = if lane.word & UNSERVED != 0 {
                        Some("its word reads a mask in no arm this load applies one in".into())
                    } else if lane.list == ::engine::ListKind::Translation {
                        ::engine::fire::misplaced_write(lane.kv.writable(), &lane.tokens)
                    } else {
                        None
                    };
                    reason.map(|reason| ::engine::LaneFault {
                        lane: u32::try_from(at).expect("a few lanes"),
                        instance: None,
                        reason,
                    })
                })
                .collect();
            if !lanes.is_empty() {
                let error = engine::Error::Lanes(lanes);
                if !error.stops_frame_at(index) {
                    return Err(error);
                }
                refused = Some(error);
                break;
            }
            let fed = step
                .attachments
                .iter()
                .map(|attachment| {
                    let lane = &step.lanes[attachment.lane as usize];
                    assert_eq!(
                        u64::from(lane.slot),
                        attachment.instance.0,
                        "an attachment reads its own instance's lane"
                    );
                    (attachment.instance.0, lane.tokens[0])
                })
                .collect();
            self.fired.lock().unwrap().push(fed);
            steps.push(::engine::FireTicket::default());
        }
        let landings: Vec<::engine::StepDone> = (0..steps.len())
            .map(|step| ::engine::StepDone {
                frame: id,
                step: u32::try_from(step).expect("a few steps"),
            })
            .collect();
        if self.reorder && refused.is_some() {
            self.held.extend(landings);
        } else {
            for at in landings.into_iter().chain(std::mem::take(&mut self.held)) {
                sink(at, ::engine::StepOutcome::committed());
            }
        }
        Ok(engine::FrameTicket::of(id, steps, refused))
    }

    fn settle_frame(&mut self, _ticket: &mut ::engine::FrameTicket) -> ::engine::Result<()> {
        Ok(())
    }

    fn completion(&mut self) -> Option<&mut dyn ::engine::Completion> {
        Some(self)
    }

    fn programs(&mut self) -> Option<&mut dyn ::engine::Programs> {
        Some(self)
    }

    fn programs_ref(&self) -> Option<&dyn ::engine::Programs> {
        Some(self)
    }

    fn pool(&mut self) -> Option<&mut dyn ::engine::Pool> {
        Some(self)
    }

    fn tiers(&mut self) -> Option<&mut dyn ::engine::Tiers> {
        Some(self)
    }
}

impl ::engine::Completion for LaneRefusalEngine {
    fn on_complete(&mut self, sink: ::engine::CompletionSink) {
        self.sink = Some(sink);
    }

    fn asynchronous(&self) -> bool {
        self.asynchronous
    }
}

impl ::engine::Programs for LaneRefusalEngine {
    fn prepare(
        &mut self,
        _registration: &::engine::program::ProgramRegistration,
    ) -> ::engine::Result<::engine::ProgramCompile> {
        Err(::engine::Error::unsupported(self.kind(), "prepare_program"))
    }

    fn finish(
        &mut self,
        _prepared: ::engine::Prepared,
    ) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "finish_program"))
    }

    fn close_program(&mut self, _program: ::engine::ProgramId) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "close_program"))
    }

    fn register(&mut self, _registration: &::engine::ProgramRegistration) -> ::engine::Result<::engine::ProgramId> {
        Err(::engine::Error::unsupported(self.kind(), "register_program"))
    }

    fn bind(&mut self, _binding: &::engine::InstanceBinding) -> ::engine::Result<::engine::BoundInstance> {
        Err(::engine::Error::unsupported(self.kind(), "bind_instance"))
    }

    fn close(&mut self, _id: ::engine::InstanceId) -> ::engine::Result<u64> {
        Err(::engine::Error::unsupported(self.kind(), "close_instance"))
    }

    fn adapter(&mut self, _registration: &::engine::adapter::AdapterRegistration) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "register_adapter"))
    }
}

impl ::engine::Pool for LaneRefusalEngine {
    fn chunk_table(&self) -> ::engine::Result<::engine::chunk::ChunkTable> {
        Err(::engine::Error::unsupported(self.kind(), "chunk_table"))
    }

    fn commit(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "commit_segments"))
    }

    fn release(&mut self, _segments: u32) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "release_segments"))
    }

    fn evacuate(&mut self, _evacuation: &::engine::chunk::Evacuation) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "evacuate"))
    }
}

impl ::engine::Tiers for LaneRefusalEngine {
    /// A test engine's copies land inside the call: nothing to wait for.
    fn settle(&mut self) -> ::engine::Result<()> {
        Ok(())
    }

    /// A test engine's copies complete inside the call: their sources are
    /// read by the time it returns.
    fn copy_landing(
        &mut self,
        copy: &::engine::chunk::SlotCopy,
        _token: u64,
    ) -> ::engine::Result<bool> {
        ::engine::Tiers::copy(self, copy)?;
        Ok(true)
    }

    fn copy(&mut self, _copy: &::engine::chunk::SlotCopy) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "copy_slots"))
    }

    fn read(&mut self, _slots: &[(::engine::chunk::ShapeId, Vec<u32>)]) -> ::engine::Result<Vec<u8>> {
        Err(::engine::Error::unsupported(self.kind(), "read_tier_slots"))
    }

    fn write(&mut self, _slots: &[(::engine::chunk::ShapeId, Vec<u32>)], _bytes: &[u8]) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "write_tier_slots"))
    }

    fn copy_kv(&mut self, _copy: &::engine::KvCopy) -> ::engine::Result<()> {
        Err(::engine::Error::unsupported(self.kind(), "copy_kv"))
    }
}


/// A frame of steps of fires, a lane each, each the pass of its own
/// instance: step `s` is `steps[s]`, `(instance, unserved)` a fire. A
/// lane's slot is its instance and its token `1000 * step + instance`.
/// The fires' completions, step by step.
fn frame_of(
    steps: &[&[(u64, bool)]],
) -> (
    crate::engine::FrameFire,
    Vec<Vec<crate::engine::WorkItemCompletion>>,
) {
    let lanes: Vec<Vec<(u64, ::engine::Lane)>> = steps
        .iter()
        .enumerate()
        .map(|(at, fires)| {
            fires
                .iter()
                .map(|&(instance, unserved)| {
                    let word = if unserved { UNSERVED | 1 } else { 1 };
                    let token = u32::try_from(1000 * at as u64 + instance).expect("small");
                    let slot = u32::try_from(instance).expect("small");
                    (instance, ::engine::Lane::decode(slot, word, token, 0))
                })
                .collect()
        })
        .collect();
    frame_of_lanes(&lanes)
}

/// A frame of steps of fires, a lane each, each the pass of its own
/// instance: step `s` is `steps[s]`, `(instance, lane)` a fire.
fn frame_of_lanes(
    steps: &[Vec<(u64, ::engine::Lane)>],
) -> (
    crate::engine::FrameFire,
    Vec<Vec<crate::engine::WorkItemCompletion>>,
) {
    let mut fire = crate::engine::FrameFire::default();
    let mut all = Vec::new();
    for fires in steps {
        let lanes = u32::try_from(fires.len()).expect("a few lanes");
        let completions: Vec<_> = (0..lanes)
            .map(|_| crate::engine::WorkItemCompletion::deferred_with_guard(None))
            .collect();
        fire.steps.push(crate::engine::StepFire {
            fire_stamps: Vec::new(),
        lane_sets: Vec::new(),
            shapes: None,
            submission: ::engine::Step {
                lanes: fires.iter().map(|(_, lane)| lane.clone()).collect(),
                attachments: (0..lanes)
                    .map(|lane| ::engine::Attachment {
                        lane,
                        lanes: 1,
                        instance: ::engine::InstanceId(fires[lane as usize].0),
                        adapter_only: false,
                        scratch: ::engine::ScratchPlace::None,
                    })
                    .collect(),
                media: Vec::new(),
                voxels: Vec::new(),
                shape: None,
            },
            terminal_cells: completions.iter().map(|c| c.terminal_cell_ptr()).collect(),
            completions: completions.clone(),
            member_lanes: (0..=lanes).collect(),
            instances: fires.iter().map(|&(id, _)| ::engine::InstanceId(id)).collect(),
        });
        all.push(completions);
    }
    (fire, all)
}

/// Fires `steps` on a [`LaneRefusalEngine`]: the frame's answer, each
/// fire's result step by step, and what each step the engine fired fed.
#[allow(clippy::type_complexity, reason = "a test's three answers")]
fn fire_refusing(
    asynchronous: bool,
    steps: &[&[(u64, bool)]],
) -> (
    std::result::Result<(), String>,
    Vec<Vec<std::result::Result<(), String>>>,
    Vec<Vec<(u64, u32)>>,
) {
    fire_refusing_landing(asynchronous, false, steps)
}

/// [`fire_refusing`], with `reorder` landing a frame stopped short
/// after the one submitted next.
#[allow(clippy::type_complexity, reason = "a test's three answers")]
fn fire_refusing_landing(
    asynchronous: bool,
    reorder: bool,
    steps: &[&[(u64, bool)]],
) -> (
    std::result::Result<(), String>,
    Vec<Vec<std::result::Result<(), String>>>,
    Vec<Vec<(u64, u32)>>,
) {
    fire_refusing_frame(asynchronous, reorder, frame_of(steps))
}

/// Fires `frame` on a [`LaneRefusalEngine`], as [`fire_refusing`] does.
#[allow(clippy::type_complexity, reason = "a test's three answers")]
fn fire_refusing_frame(
    asynchronous: bool,
    reorder: bool,
    (fire, completions): (
        crate::engine::FrameFire,
        Vec<Vec<crate::engine::WorkItemCompletion>>,
    ),
) -> (
    std::result::Result<(), String>,
    Vec<Vec<std::result::Result<(), String>>>,
    Vec<Vec<(u64, u32)>>,
) {
    let fired = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut lane = Lane::spawn(
        0,
        Some(Box::new(LaneRefusalEngine {
            sink: None,
            asynchronous,
            fired: Arc::clone(&fired),
            next: 0,
            reorder,
            held: Vec::new(),
        })),
        Arc::new(SchedulerStats::default()),
    );
    let flat: Vec<_> = completions.iter().flatten().cloned().collect();
    let (frame, results) = if reorder {
        // Every fire of a launch landing out of order is failed by the
        // lane as it lands, whenever after the frame that does.
        let answer = within("the launch is answered", lane.fire(fire, false))
            .expect("the launch is answered");
        let answer = match answer {
            Ok(frame) => {
                within("the frame settles", frame).map_err(|error| format!("{error:#}"))
            }
            Err(error) => Err(error.message),
        };
        let results = flat
            .iter()
            .map(|completion| {
                within("the fire settles", completion.clone())
                    .map_err(|error| format!("{error:#}"))
            })
            .collect();
        (answer, results)
    } else {
        fire_and_settle(&lane, fire, &flat)
    };
    let mut results = results.into_iter();
    let by_step = completions
        .iter()
        .map(|step| results.by_ref().take(step.len()).collect())
        .collect();
    let _ = lane.shutdown();
    let fed = fired.lock().unwrap().clone();
    (frame, by_step, fed)
}

fn refused(result: &std::result::Result<(), String>) -> bool {
    result
        .as_ref()
        .is_err_and(|why| why.contains("lane") && why.contains("refused"))
}

/// One lane of four refused at a fire's staging fails its own fire
/// alone, by the engine's reason; the other three fire once each, in
/// the same submission, with what they fed.
fn one_refused_lane_fails_its_own_fire_alone(asynchronous: bool) {
    let (frame, fires, fed) = fire_refusing(
        asynchronous,
        &[&[(100, false), (101, false), (102, true), (103, false)]],
    );
    assert_eq!(frame, Ok(()), "the frame is not failed by one lane's refusal");
    let why = fires[0][2].as_ref().expect_err("the refused lane's fire fails");
    assert!(why.contains("reads a mask in no arm"), "by the engine's reason: {why}");
    assert!(why.contains("lane 2"), "naming its lane: {why}");
    for at in [0, 1, 3] {
        assert_eq!(fires[0][at], Ok(()), "fire {at} commits");
    }
    assert_eq!(
        fed,
        vec![vec![(100, 100), (101, 101), (103, 103)]],
        "the three fire once, each with its own token"
    );
}

#[test]
fn a_refused_lane_fails_its_own_fire_alone_inside_the_submit() {
    one_refused_lane_fails_its_own_fire_alone(false);
}

/// A lane refused at a frame's second step: the first step fired whole
/// (its fires commit, once), and the second fires again without the
/// refused fire, whose earlier fire stands.
fn a_lane_refused_past_the_first_step_stops_the_frame_there(asynchronous: bool) {
    let (frame, fires, fed) = fire_refusing(
        asynchronous,
        &[
            &[(100, false), (101, false), (102, false)],
            &[(100, false), (101, true), (102, false)],
        ],
    );
    assert_eq!(frame, Ok(()));
    assert!(fires[0].iter().all(Result::is_ok), "the first step commits: {fires:?}");
    assert!(refused(&fires[1][1]), "the refused fire fails: {:?}", fires[1][1]);
    assert_eq!(fires[1][0], Ok(()));
    assert_eq!(fires[1][2], Ok(()));
    assert_eq!(
        fed,
        vec![
            vec![(100, 100), (101, 101), (102, 102)],
            vec![(100, 1100), (102, 1102)],
        ],
        "no step fires twice, and none of the others' tokens is lost"
    );
}

#[test]
fn a_lane_refused_past_the_first_step_stops_the_frame_there_inside_the_submit() {
    a_lane_refused_past_the_first_step_stops_the_frame_there(false);
}

/// A launch split into two engine frames whose second lands before its
/// first is refused by name: every fire of the launch fails, none on
/// another frame's tokens.
#[test]
fn a_split_launch_landing_out_of_order_is_refused_by_name() {
    let (_frame, fires, fed) = fire_refusing_landing(
        true,
        true,
        &[
            &[(100, false), (101, false)],
            &[(100, false), (101, true)],
        ],
    );
    assert_eq!(fed.len(), 2, "the engine fired both frames: {fed:?}");
    assert!(refused(&fires[1][1]), "{:?}", fires[1][1]);
    for fired in [&fires[0][0], &fires[0][1], &fires[1][0]] {
        let why = fired.as_ref().expect_err("no fire takes an out-of-order frame's tokens");
        assert!(
            why.contains("settled its engine frame 1 of 2 while frame 0 was expected"),
            "named: {why}"
        );
    }
}

/// A lane whose program places its rows at a slot below its set's
/// frozen prefix, or past what it reserved, is refused by name, the
/// write never made; the lanes beside it, writing their own slots,
/// fire once each in the same submission.
#[test]
fn a_row_placed_outside_its_writable_slots_fails_its_own_fire_alone() {
    // Each lane's set froze slots 0..8 and reserved through 16; its
    // program places its row at the slot its token names.
    let placed = |instance: u64, slot: u32| {
        let mut lane = ::engine::Lane::decode(u32::try_from(instance).unwrap(), 1, slot, 0);
        lane.list = ::engine::ListKind::Translation;
        lane.kv.folded = 8;
        lane.kv.write_end = 16;
        (instance, lane)
    };
    for asynchronous in [true, false] {
        let steps = vec![vec![
            placed(100, 8),
            placed(101, 3),
            placed(102, 15),
            placed(103, 16),
        ]];
        let (frame, fires, fed) =
            fire_refusing_frame(asynchronous, false, frame_of_lanes(&steps));
        assert_eq!(frame, Ok(()), "the frame is not failed by one lane's write");
        let below = fires[0][1]
            .as_ref()
            .expect_err("the lane writing below fails");
        assert!(
            below.contains("lane 1") && below.contains("kv slot 3") && below.contains("frozen"),
            "named, by the slot it placed: {below}"
        );
        let past = fires[0][3]
            .as_ref()
            .expect_err("the lane writing past fails");
        assert!(
            past.contains("lane 3") && past.contains("kv slot 16"),
            "{past}"
        );
        assert_eq!(
            fires[0][0],
            Ok(()),
            "a row at the frozen prefix's end lands"
        );
        assert_eq!(fires[0][2], Ok(()), "a row inside what it reserved lands");
        assert_eq!(
            fed,
            vec![vec![(100, 8), (102, 15)]],
            "the lanes writing their own slots fire once; the refused never fire"
        );
    }
}


fn within<T>(what: &str, wait: impl std::future::Future<Output = T>) -> T {
    crate::rt::block_on(async { crate::rt::time::timeout(Duration::from_secs(10), wait).await })
        .unwrap_or_else(|_| panic!("timed out: {what}"))
}

/// A handle on no scheduler: what a program's registration by content
/// hash keeps, driven by a registration of the test's own.
fn bare_handle() -> SchedulerHandle {
    let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
    SchedulerHandle {
        inner: Arc::new(SchedulerControl {
            tx,
            active_senders: AtomicUsize::new(0),
            shutdown_wait: Condvar::new(),
            shutdown_gate: Mutex::new(()),
            program_ids: Mutex::new(HashMap::new()),
            accepting: AtomicBool::new(true),
            stats: Arc::new(SchedulerStats::default()),
            device_domain: ::engine::MemoryDomain::HostPinned,
            codegen_backend: None,
        }),
    }
}

async fn panics_registering() -> Result<::engine::ProgramId> {
    tokio::time::sleep(Duration::from_millis(20)).await;
    panic!("the compile panics")
}

fn registration(hash: u64) -> ProgramRegistration {
    ProgramRegistration {
        program_hash: hash,
        ..ProgramRegistration::default()
    }
}

/// A registration that panics refuses each bind waiting on it by the
/// program's hash, and leaves no entry: the next bind registers afresh.
#[test]
fn a_registration_that_panics_is_refused_by_name_and_forgotten() {
    let handle = bare_handle();
    let refused = crate::rt::block_on(async {
        let binds = (0..3).map(|_| {
            handle.program_id_with(ProcessId::default(), registration(0xbad1), |_, _| {
                panics_registering()
            })
        });
        futures::future::join_all(binds).await
    });
    for bind in refused {
        let why = bind.expect_err("refused").to_string();
        assert!(why.contains("0x000000000000bad1"), "by name: {why}");
    }
    assert!(
        !handle
            .inner
            .program_ids
            .lock()
            .unwrap()
            .contains_key(&0xbad1),
        "no entry is left"
    );
    let retried = crate::rt::block_on(handle.program_id_with(
        ProcessId::default(),
        registration(0xbad1),
        |_, _| async { Ok(::engine::ProgramId(3)) },
    ));
    assert_eq!(retried.expect("registers afresh"), ::engine::ProgramId(3));
}
