use std::sync::Mutex;

use super::*;
use crate::engine::completion::FrameSettlements;
use crate::engine::{CompletionBroker, FrameFire, StepFire, WorkItemCompletion};
use crate::store::kv::KvStore;

/// Takes every frame and lands it only when told; its channel take fails.
struct TakeFails;

impl ::engine::Engine for TakeFails {
    fn kind(&self) -> &'static str {
        "take-fails"
    }

    fn load(&mut self, _request: ::engine::LoadRequest) -> ::engine::Result<::engine::Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(
        &mut self,
        frame: &::engine::FrameSubmission,
    ) -> ::engine::Result<::engine::FrameTicket> {
        Ok(::engine::FrameTicket {
            id: ::engine::FrameId(7),
            steps: vec![Default::default(); frame.steps.len()],
        })
    }

    fn settles_asynchronously(&self) -> bool {
        true
    }

    fn take_channel(
        &mut self,
        _instance: ::engine::InstanceId,
        _channel: u32,
    ) -> ::engine::Result<Option<Vec<u8>>> {
        Err(::engine::Error::Program("the ring is gone".into()))
    }
}

/// A frame the engine took fails after (its channel take): its fires fail,
/// but only once the engine lands it. Failed at submission, their epochs
/// (and leases) would end, and the pages they write would go to another lane
/// while the engine still runs over them.
#[test]
fn a_frame_failing_after_the_engine_took_it_holds_its_pages_until_it_lands() {
    let instance = ::engine::InstanceId(1);
    let mut channels = ChannelJoin::new();
    let channel = ::engine::ChannelId(3);
    channels.insert(
        RegisteredChannel::new(0, channel, 4, 4, 0, 0),
        eta_ir::container::HostRole::Reader,
    );
    channels.bind(instance, vec![channel]);

    // The fire's pages, let go under it: back only when its epoch ends.
    let kv = Arc::new(Mutex::new(KvStore::new(4, [9; 32])));
    let fire = {
        let mut kv = kv.lock().unwrap();
        let ws = kv.create_working_set();
        kv.reserve(ws, 4).unwrap();
        let mut granted = kv.reserve_device_pages(4).unwrap();
        kv.ensure_backed_reserved(ws, 4, &mut granted).unwrap();
        let fire = kv.begin_fire();
        let epoch = kv.current_epoch();
        kv.release_working_set(ws, epoch);
        kv.retire_idle();
        fire
    };
    let completion = WorkItemCompletion::new(41, 0);
    {
        let kv = Arc::clone(&kv);
        completion.on_settled(move || kv.lock().unwrap().settle(fire));
    }

    let mut frame = FrameFire {
        steps: vec![StepFire {
            submission: ::engine::Step {
                lanes: vec![::engine::Lane::decode(0, 0, 1, 0)],
                attachments: vec![::engine::Attachment {
                    lane: 0,
                    lanes: 1,
                    instance,
                    adapter_only: false,
                }],
                media: Vec::new(),
                voxels: Vec::new(),
            },
            terminal_cells: vec![completion.terminal_cell_ptr()],
            completions: vec![completion.clone()],
            instances: vec![instance],
        }],
    };
    let mut engine: EngineBox = Box::new(TakeFails);
    let (_launch_tx, launch_rx) = crate::rt::channel::unbounded::<LaneRequest>();
    let broker = CompletionBroker::new();
    let settlements = FrameSettlements::new();
    let mut landed = HashMap::new();
    let submitted = Lane::fire_frame(
        &mut engine,
        &channels,
        None,
        &mut frame,
        &launch_rx,
        &mut None,
        &broker,
        &settlements,
        &mut landed,
    );
    assert!(
        submitted.is_ok(),
        "the engine took the frame: it is in flight, not refused"
    );
    assert!(
        !completion.is_settled(),
        "a fire the engine has yet to land was settled"
    );
    assert_eq!(
        kv.lock().unwrap().available_pages(),
        0,
        "a page the frame in flight writes went back"
    );

    settlements.settled(
        ::engine::FrameId(7),
        0,
        &::engine::StepOutcome::Committed,
        &broker,
    );
    assert!(completion.is_settled(), "it lands");
    assert!(
        crate::rt::block_on(completion.clone()).is_err(),
        "and fails, saying so"
    );
    kv.lock().unwrap().retire_idle();
    assert_eq!(
        kv.lock().unwrap().available_pages(),
        4,
        "its pages are back"
    );
}

/// Takes every frame and keeps what it was handed.
struct Records(Arc<Mutex<Vec<::engine::FrameSubmission>>>);

impl ::engine::Engine for Records {
    fn kind(&self) -> &'static str {
        "records"
    }

    fn load(&mut self, _request: ::engine::LoadRequest) -> ::engine::Result<::engine::Loaded> {
        Err(::engine::Error::Load("no model".into()))
    }

    fn submit(
        &mut self,
        frame: &::engine::FrameSubmission,
    ) -> ::engine::Result<::engine::FrameTicket> {
        self.0.lock().unwrap().push(frame.clone());
        Ok(::engine::FrameTicket {
            id: ::engine::FrameId(9),
            steps: vec![Default::default(); frame.steps.len()],
        })
    }

    fn settles_asynchronously(&self) -> bool {
        true
    }
}

fn member(
    instance: u64,
    tokens: Vec<u32>,
    pages: Vec<u32>,
) -> Box<crate::scheduler::worker::PendingRequest> {
    let mut request = crate::engine::FireRequest::one(crate::engine::fire::lane_of(
        u32::try_from(instance).unwrap(),
        tokens,
        0,
        pages,
    ));
    request.boundary_program = true;
    Box::new(crate::scheduler::worker::PendingRequest {
        lora_program: false,
        logical_fire_id: instance,
        request,
        instance_id: ::engine::InstanceId(instance),
        completion: WorkItemCompletion::new(100 + instance, 0),
        process_id: None,
        pipeline_id: None,
        prelaunch_copy: None,
        prelaunch_state_copy: None,
        frame: None,
    })
}

/// Where instance `instance`'s lanes lie in `step`, and the rows before
/// them: the engine reads a lane's rows (its readout's last row among them)
/// from the step's lanes in order.
fn rows_of(step: &::engine::Step, instance: u64) -> (Vec<::engine::Lane>, u32) {
    let at = step
        .attachments
        .iter()
        .find(|a| a.instance == ::engine::InstanceId(instance))
        .expect("the member's program is attached");
    let (first, count) = (at.lane as usize, at.lanes as usize);
    let before = step.lanes[..first].iter().map(::engine::Lane::rows).sum();
    (step.lanes[first..first + count].to_vec(), before)
}

/// A frame mixes a late prompt's prefill with decoders. One decoder process
/// fails before the frame is built (its fire refused, cancelled), and
/// another after the frame is built but before it is submitted: neither
/// re-cuts the frame. The late lane's attachment names its own lane, whose
/// last row is the prompt's last token, at the rows its own lane spans.
#[test]
fn a_decoder_failing_around_a_mixed_frame_leaves_the_late_lane_reading_its_own_row() {
    const LATE: u64 = 5;
    let prompt: Vec<u32> = (1000..1013).collect();
    let decoders = [1u64, 2, 3];
    let build = |failed_before: Option<u64>| {
        let mut wave: Vec<_> = decoders
            .iter()
            .filter(|&&d| Some(d) != failed_before)
            .map(|&d| {
                // A batch decoder: several lanes in one member.
                let mut m = member(d, vec![10 + d as u32], vec![d as u32]);
                let mut second = m.request.lanes[0].clone();
                second.tokens = vec![40 + d as u32];
                m.request.lanes.push(second);
                m
            })
            .collect();
        wave.insert(1, member(LATE, prompt.clone(), vec![20, 21]));
        crate::scheduler::batch::build_frame_submission(
            vec![wave],
            crate::engine::SchedulerLimits {
                max_forward_requests: 8,
                max_forward_tokens: 64,
                max_page_refs: 64,
                max_context: 0,
                prefill_rows: 0,
                prefill_lanes: 0,
                max_readouts: 0,
            },
            16,
            &crate::scheduler::stats::SchedulerStats::default(),
        )
    };
    for failed_before in [None, Some(1), Some(2), Some(3)] {
        let (mut frame, requests) = build(failed_before);
        // Another decoder fails after the build: its fire is cancelled, and
        // the frame stays as it was built.
        if let Some(gone) = requests
            .iter()
            .find(|r| r.instance_id.0 != LATE && Some(r.instance_id.0) != failed_before)
        {
            gone.completion.request_cancel();
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut engine: EngineBox = Box::new(Records(Arc::clone(&seen)));
        let (_launch_tx, launch_rx) = crate::rt::channel::unbounded::<LaneRequest>();
        let broker = CompletionBroker::new();
        let settlements = FrameSettlements::new();
        Lane::fire_frame(
            &mut engine,
            &ChannelJoin::new(),
            None,
            &mut frame,
            &launch_rx,
            &mut None,
            &broker,
            &settlements,
            &mut HashMap::new(),
        )
        .expect("the engine takes the frame");
        let seen = seen.lock().unwrap();
        let steps = &seen[0].steps;
        let step = steps
            .iter()
            .find(|step| step.attachments.iter().any(|a| a.instance.0 == LATE))
            .expect("the late lane is in the frame");
        let (lanes, before) = rows_of(step, LATE);
        assert_eq!(lanes.len(), 1, "{failed_before:?}");
        assert_eq!(lanes[0].tokens, prompt, "the late lane is its own");
        assert_eq!(lanes[0].readout, ::engine::fire::Readout::Last);
        assert_eq!(lanes[0].kv.pages, vec![20, 21]);
        let total: u32 = step.lanes.iter().map(::engine::Lane::rows).sum();
        let last = before + lanes[0].rows() - 1;
        assert!(last < total);
        assert_eq!(
            total,
            13 + 2 * (decoders.len() as u32 - u32::from(failed_before.is_some()))
        );
        // Every other attachment's span lies apart from the late lane's.
        for at in step.attachments.iter().filter(|a| a.instance.0 != LATE) {
            let (theirs, start) = rows_of(step, at.instance.0);
            let end = start + theirs.iter().map(::engine::Lane::rows).sum::<u32>();
            assert!(
                end <= before || start > last,
                "{:?} spans rows {start}..{end}, over the late lane's last row {last}",
                at.instance
            );
            let d = at.instance.0 as u32;
            assert_eq!(
                theirs
                    .iter()
                    .map(|lane| lane.tokens.clone())
                    .collect::<Vec<_>>(),
                vec![vec![10 + d], vec![40 + d]],
                "{:?} reads lanes that are not its own",
                at.instance
            );
        }
        let attached: Vec<u64> = steps
            .iter()
            .flat_map(|step| step.attachments.iter().map(|a| a.instance.0))
            .collect();
        if let Some(gone) = failed_before {
            assert!(
                !attached.contains(&gone),
                "a member failed before the build rode"
            );
        }
    }
}
