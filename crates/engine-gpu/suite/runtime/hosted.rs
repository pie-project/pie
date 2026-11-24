//! What a fire reads back to the host, lane by lane: a lane a program is
//! attached to hands its rows to the program on the device and the host
//! reads none of them; a `Last` or `Rows(r)` lane no program reads gets
//! exactly its rows; a `None` lane hosts nothing. A program reading its
//! lane's own readout plane is still served, fire after fire (a replayed
//! one among them, where the backend records).

use engine::Engine;
use engine::channel::{ChannelId, ChannelRegistration};
use engine::fire::{Attachment, FrameSubmission, Lane, Readout, Step};
use engine::program::InstanceBinding;
use engine_gpu::serve::Serve;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::Request;

use super::outputs::{Loaded, Rows, WIDTH, loaded, output_decls, registration_of};
use crate::backend;

/// A program that reads its lane's `logits` row and puts it, as read, to a
/// channel the host takes: a sampler's read, with the sampling left out.
fn reader() -> TraceContainer {
    TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![ChannelDecl {
            shape: Shape::matrix(1, WIDTH),
            dtype: ChanDType::Concrete(EtaDtype::F32),
            capacity: 2,
            host_role: HostRole::Reader,
            seeded: false,
        }],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(1, WIDTH),
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 0, value: 0 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// The table's row of `token`, as bits.
fn row_of(table: &[f32], token: u32) -> Vec<u32> {
    table[(token * WIDTH) as usize..][..WIDTH as usize]
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn each_lane_hosts_what_it_reads_out_and_a_program_reads_its_rows_on_the_device() {
    let Some(Loaded {
        mut engine,
        caps,
        trace,
        planes,
        dir,
    }) = loaded("hosted", &Rows)
    else {
        return;
    };
    let table = planes["embed"].1.clone();
    let program = engine
        .register_program(&registration_of(&output_decls(&caps), reader(), 0x4057))
        .expect("the reader registers");
    let chunks = &caps.caps.chunks;
    let mut slots =
        super::slots::Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let channel = ChannelId(1);
    slots.register(
        &mut engine,
        ChannelRegistration {
            id: channel,
            shape: vec![1, WIDTH],
            dtype: ChanDType::Concrete(EtaDtype::F32),
            host_role: HostRole::Reader,
            seeded: false,
            extern_dir: None,
            capacity: 2,
            extern_name: Vec::new(),
            slots: Vec::new(),
        },
    );
    const ROWS: u32 = 3;
    let binding = slots.binding(
        &engine,
        InstanceBinding {
            program,
            channels: vec![channel],
            seeds: Vec::new(),
            geometry: GeometryClass::Host,
            extents: engine::program::BindExtents {
                row_count: 1,
                token_count: ROWS,
                sampled_rows: 1,
                query_len: ROWS,
                ..engine::program::BindExtents::default()
            },
            device_resolved: Default::default(),
            instance_slots: Vec::new(),
            ring_slots: Vec::new(),
            adapter_slots: Vec::new(),
            scratch: engine::ScratchPlace::None,
            pair: engine::ScratchPlace::None,
            pair_generation: 0,
        },
        &[channel],
    );
    let instance = engine
        .bind_instance(&binding)
        .expect("the reader binds")
        .id;
    let word = trace.facts.word(&Request::new(ROWS, false));
    let lane = |slot: u32, tokens: &[u32], readout: Readout| Lane {
        slot,
        word,
        tokens: tokens.to_vec(),
        readout,
        kv_less: true,
        ..Lane::default()
    };

    // Twice, other tokens the second time: the same shape, so where the
    // backend records, the second fire replays the first's recording and
    // the program reads the rows where the replay left them.
    for (round, base) in [(0u32, 0u32), (1, 5)] {
        let tokens: Vec<Vec<u32>> = (0..4)
            .map(|lane| (0..ROWS).map(|k| (base + lane * 3 + k * 7) % WIDTH).collect())
            .collect();
        for readout in [Readout::Last, Readout::None] {
            let before = engine.shell_mut().expect("loaded").readout_tally();
            let step = Step {
                lanes: vec![
                    // Its program reads its rows on the device: the host none.
                    lane(0, &tokens[0], readout.clone()),
                    // No program: the host reads its last row.
                    lane(1, &tokens[1], Readout::Last),
                    // No program, rows 2 and 0: the host reads exactly those.
                    lane(2, &tokens[2], Readout::Rows(vec![2, 0])),
                    // No program, no readout: nothing.
                    lane(3, &tokens[3], Readout::None),
                ],
                attachments: vec![Attachment {
                    lane: 0,
                    lanes: 1,
                    instance,
                    adapter_only: false,
                    scratch: engine::ScratchPlace::None,
                }],
                media: Vec::new(),
                voxels: Vec::new(),
                shape: None,
            };
            let mut ticket = engine
                .submit(&FrameSubmission::of(step))
                .expect("the frame fires");
            engine.settle_frame(&mut ticket).expect("the frame settles");
            let readouts = &ticket.steps[0].readouts;
            let what = format!("round {round}, the read lane's readout {readout:?}");

            assert!(
                readouts[0].values.is_empty(),
                "{what}: a lane its program reads hosts no row"
            );
            assert_eq!(
                bits(&readouts[1].values),
                row_of(&table, tokens[1][2]),
                "{what}: a Last lane hosts its last row"
            );
            assert_eq!(readouts[2].rows, 2, "{what}: a Rows lane hosts its rows");
            assert_eq!(
                bits(&readouts[2].values),
                [row_of(&table, tokens[2][2]), row_of(&table, tokens[2][0])].concat(),
                "{what}: a Rows lane hosts exactly the rows it names, in its order"
            );
            assert!(
                readouts[3].values.is_empty(),
                "{what}: a None lane hosts nothing"
            );

            // The program read its lane's own row, wherever it lay.
            let read = engine
                .take_channel(instance, 0)
                .expect("the channel reads")
                .expect("the reader put its row");
            let read: Vec<u32> = read
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c).to_bits())
                .collect();
            assert_eq!(
                read,
                row_of(&table, tokens[0][2]),
                "{what}: the program read its lane's last row"
            );

            let after = engine.shell_mut().expect("loaded").readout_tally();
            let (hosted, seated) = (after.0 - before.0, after.1 - before.1);
            assert_eq!(hosted, 3, "{what}: the host read back three rows, no more");
            // Each hosted row is copied into the seat; the program's row too
            // where it reads the arm's seat, and nowhere where it reads the
            // span.
            let copied = if <backend::Api as Serve>::READS_SPAN { 3 } else { 4 };
            assert!(
                seated == copied * u64::from(WIDTH) * 2 || seated == copied * u64::from(WIDTH) * 4,
                "{what}: {seated} bytes copied into the seat, {copied} rows of {WIDTH}"
            );
        }
    }
    eprintln!(
        "a program read its row on the device ({}), the host read back only its lanes' rows",
        if <backend::Api as Serve>::READS_SPAN {
            "in the span"
        } else {
            "in the arm's seat"
        }
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
