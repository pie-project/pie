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


#[test]
fn worker_every_case() {
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


fn within<T>(what: &str, wait: impl std::future::Future<Output = T>) -> T {
    crate::rt::block_on(async { crate::rt::time::timeout(Duration::from_secs(10), wait).await })
        .unwrap_or_else(|_| panic!("timed out: {what}"))
}
