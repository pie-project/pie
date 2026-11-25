use super::*;
use crate::pipeline::channel::ChannelCell;
use eta_ir::container::ChannelDecl;
use eta_ir::types::{Dtype, Shape};

fn channel(role: HostRole, capacity: u32, seeded: bool) -> Arc<Mutex<ChannelCell>> {
    let mut cell = ChannelCell::new(vec![1], Dtype::U32, capacity);
    cell.bind(&ChannelDecl {
        shape: Shape::new(&[1]).expect("a one-element cell"),
        dtype: eta_ir::container::ChanDType::Concrete(Dtype::U32),
        capacity,
        host_role: role,
        seeded,
    });
    Arc::new(Mutex::new(cell))
}

fn slot(cells: &[Arc<Mutex<ChannelCell>>], accesses: &[(bool, bool)]) -> SlotAccess {
    SlotAccess {
        cells: cells.to_vec(),
        accesses: accesses.to_vec(),
    }
}

#[test]
fn admission_every_case() {
    a_device_ring_frame_that_overflows_is_refused_by_name();
    a_frame_that_drains_more_writer_cells_than_are_staged_is_refused();
    a_reader_ring_too_small_for_the_frames_writes_is_refused();
    two_passes_putting_one_ring_get_distinct_cells_in_dispatch_order();
}

/// A channel is its process's: two passes of the process put one device
/// ring, each a fire of its own lane, and the engine picks each put's
/// cell from the ring's cursor as it builds the step. In each way the
/// fire that reserved later can reach the engine first, the two get
/// distinct cells, in the order they are dispatched, with no rule about
/// the order they reserved in.
fn two_passes_putting_one_ring_get_distinct_cells_in_dispatch_order() {
    use crate::scheduler::worker::PendingRequest;
    type Submissions = Vec<Vec<Vec<u64>>>;
    // Each case as the submissions it dispatches, each a list of waves of
    // the fires' instances: 1 reserved first, 2 second.
    let cases: [(&str, Submissions); 7] = [
        ("in reservation order", vec![vec![vec![1, 2]]]),
        (
            "the seal leaves the earlier frame for later",
            vec![vec![vec![2]], vec![vec![1]]],
        ),
        ("a held prompt waits a wave", vec![vec![vec![2], vec![1]]]),
        (
            "the earlier lane stands later in the wave",
            vec![vec![vec![2, 1]]],
        ),
        (
            "a boundary falls between them",
            vec![vec![vec![2]], vec![vec![1]]],
        ),
        (
            "the later is a rider ahead of the framed",
            vec![vec![vec![2]], vec![vec![1]]],
        ),
        (
            "the later is framed ahead of a rider",
            vec![vec![vec![2]], vec![vec![1]]],
        ),
    ];
    let limits = crate::engine::SchedulerLimits {
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
    };
    for (case, submissions) in cases {
        let ring = channel(HostRole::None, 4, false);
        let cells: BoundCells = vec![Arc::clone(&ring)];
        let id = ring.lock().unwrap().global_id;
        let reservations = [
            TicketReservation::new(&cells, &[(false, true)]),
            TicketReservation::new(&cells, &[(false, true)]),
        ];
        let fire = |instance: u64| {
            let mut request = crate::engine::FireRequest::one(crate::engine::fire::lane_of(
                0,
                vec![instance as u32],
                0,
                vec![instance as u32],
            ));
            request.single_token_mode = true;
            reservations[instance as usize - 1].apply_to(&mut request);
            assert_eq!(
                request.ring_moves,
                [crate::engine::fire::RingMove {
                    channel: id,
                    takes: false,
                    puts: true,
                }],
                "the fire carries the end it moves and no place"
            );
            Box::new(PendingRequest {
                queued: std::time::Instant::now(),
                lora_program: false,
                logical_fire_id: instance,
                request,
                instance_id: ::engine::InstanceId(instance),
                completion: crate::engine::WorkItemCompletion::new(instance, 0),
                process_id: Some(uuid::Uuid::from_u128(1)),
                pipeline_id: Some(uuid::Uuid::from_u128(u128::from(instance))),
                prelaunch_copy: None,
                prelaunch_state_copy: None,
                frame: None,
            })
        };
        // The engine's ring cursor, moved by each put as it builds a step.
        let mut cursor = 0u64;
        let mut cells_of = std::collections::BTreeMap::new();
        let mut dispatched = Vec::new();
        for waves in submissions {
            let waves = waves
                .into_iter()
                .map(|wave| wave.into_iter().map(fire).collect())
                .collect();
            let (frame, _) = crate::scheduler::batch::build_frame_submission(
                waves,
                limits,
                16,
                &crate::scheduler::stats::SchedulerStats::default(),
            );
            for step in &frame.steps {
                let before = cursor;
                for instance in &step.instances {
                    cells_of.insert(instance.0, cursor);
                    cursor += 1;
                    dispatched.push(instance.0);
                }
                assert!(cursor - before <= 1, "{case}: one put of the ring a step");
            }
        }
        let cells: Vec<u64> = dispatched.iter().map(|fire| cells_of[fire]).collect();
        assert_eq!(cells, [0, 1], "{case}: distinct cells in dispatch order");
        assert_eq!(ring.lock().unwrap().device_ring_backlog(), 2, "{case}");
        for reservation in reservations {
            reservation.commit();
        }
    }
}

fn a_device_ring_frame_that_overflows_is_refused_by_name() {
    let ring = channel(HostRole::None, 1, false);
    let slots = [
        slot(std::slice::from_ref(&ring), &[(false, true)]),
        slot(std::slice::from_ref(&ring), &[(false, true)]),
    ];
    let refusal = prove_frame_admissible(2, &slots).expect_err("two publishes, capacity one");
    assert!(
        refusal.contains("device-ring occupancy past capacity 1"),
        "{refusal}"
    );
    assert!(refusal.contains("frame slot 1"), "{refusal}");
}

fn a_frame_that_drains_more_writer_cells_than_are_staged_is_refused() {
    let writer = channel(HostRole::Writer, 4, false);
    writer
        .lock()
        .unwrap()
        .put_ref(&[0u8; 4])
        .expect("one staged cell");
    let one = [slot(std::slice::from_ref(&writer), &[(true, false)])];
    assert_eq!(prove_frame_admissible(2, &one), Ok(()));

    let two = [
        slot(std::slice::from_ref(&writer), &[(true, false)]),
        slot(std::slice::from_ref(&writer), &[(true, false)]),
    ];
    let refusal = prove_frame_admissible(2, &two).expect_err("two consumes, one staged");
    assert!(
        refusal.contains("consumes 2 host-writer cell(s)"),
        "{refusal}"
    );
    assert!(refusal.contains("only 1 are staged"), "{refusal}");
}

fn a_reader_ring_too_small_for_the_frames_writes_is_refused() {
    let reader = channel(HostRole::Reader, 1, false);
    let slots = [
        slot(std::slice::from_ref(&reader), &[(false, true)]),
        slot(std::slice::from_ref(&reader), &[(false, true)]),
    ];
    let refusal = prove_frame_admissible(2, &slots).expect_err("two writes, capacity one");
    assert!(
        refusal.contains("frame would need 2 reader cell(s)"),
        "{refusal}"
    );
    assert!(refusal.contains("2k-1 = 3"), "{refusal}");

    let roomy = channel(HostRole::Reader, 3, false);
    let ok = [
        slot(std::slice::from_ref(&roomy), &[(false, true)]),
        slot(std::slice::from_ref(&roomy), &[(false, true)]),
    ];
    assert_eq!(prove_frame_admissible(2, &ok), Ok(()));
}
