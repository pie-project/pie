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
