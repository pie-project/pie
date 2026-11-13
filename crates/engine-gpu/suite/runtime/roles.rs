//! A guest instance whose launch tables lie in `Instance` pool slots and
//! whose kept cell lies in `Channel` slots (REDESIGN step 4) reads the same
//! outputs after an evacuation moved their chunks and after a park moved
//! them to other refs; without the rebind it reads the wrong ones.

use engine::ChannelId;
use engine::Engine;
use engine::channel::{ChannelRegistration, ChannelSeed};
use engine::chunk::{Evacuation, ShapeId, SlotCopy};
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine::program::{InstanceBinding, InstanceId, ProgramRegistration, Rebind, RingSlots};
use engine::transfer::MemoryDomain;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{Axis, EngineCaps, GeometryClass, OutputDecl, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::Request;
use poem_ir::InputAxis;

use super::outputs::{LAYERS, NAME, Outputs, VOCAB, WIDTH, contract_for, planes, write};
use super::slots::Slots;
use crate::backend;

/// Channels: `ids` (doubled) and `stack` the host reads, `t` it writes,
/// `last` the instance's own (seeded; `WIDE` copies of the pass's id, which
/// the device keeps between passes where it keeps cells: one of capacity 1
/// and 64 KiB, so a run of `Channel` slots), `seen` the host reads: twice
/// what `last` held when the pass began.
fn program() -> TraceContainer {
    let decl = |shape: Shape, dtype: EtaDtype, host_role: HostRole, capacity: u32| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(dtype),
        capacity,
        host_role,
        seeded: false,
    };
    let ids = || Op::Output {
        index: 0,
        lane: 0,
        shape: Shape::matrix(1, 1),
        dtype: EtaDtype::I32,
    };
    TraceContainer {
        names: vec!["ids".to_string(), "stack".to_string()],
        channels: vec![
            decl(Shape::matrix(1, 1), EtaDtype::I32, HostRole::Reader, 2),
            decl(
                Shape::matrix(LAYERS, WIDTH),
                EtaDtype::F32,
                HostRole::Reader,
                2,
            ),
            decl(Shape::matrix(1, 1), EtaDtype::F32, HostRole::Writer, 2),
            ChannelDecl {
                seeded: true,
                ..decl(Shape::matrix(1, WIDE), EtaDtype::I32, HostRole::None, 1)
            },
            decl(Shape::matrix(1, WIDE), EtaDtype::I32, HostRole::Reader, 2),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                // Doubled on the device, so a pass whose tables are gone
                // shows: its puts would be the heap's stale words.
                ids(),
                Op::Add(0, 0),
                Op::ChanPut { chan: 0, value: 1 },
                Op::Output {
                    index: 1,
                    lane: 0,
                    shape: Shape::matrix(LAYERS, WIDTH),
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 1, value: 2 },
                Op::ChanTake(3),
                Op::Add(3, 3),
                Op::ChanPut { chan: 4, value: 4 },
                ids(),
                Op::Broadcast {
                    value: 5,
                    shape: Shape::matrix(1, WIDE),
                },
                Op::ChanPut { chan: 3, value: 6 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

/// What the seeded `last` holds before the first pass.
const SEED: i32 = -5;

/// The ids `last` carries: 64 KiB of them, what the device keeps between
/// passes without a deferral.
const WIDE: u32 = 16 << 10;

struct Bound {
    engine: backend::Engine,
    instance: InstanceId,
    channels: Vec<ChannelId>,
    t_input: u32,
    word: u64,
    /// The argmax of each embedding row.
    argmax: Vec<i32>,
}

impl Bound {
    /// One pass reading token `token` last: (`ids`, `seen`).
    fn pass(&mut self, token: u32) -> Result<(i32, i32), String> {
        let lane = Lane {
            slot: 0,
            word: self.word,
            tokens: vec![3, 11, token],
            readout: Readout::Last,
            kv_less: true,
            inputs: vec![InputFeed {
                id: self.t_input,
                channel: self.channels[2],
            }],
            ..Lane::default()
        };
        let mut ticket = self
            .engine
            .submit(&FrameSubmission::of(Step {
                lanes: vec![lane],
                attachments: vec![Attachment {
                    lane: 0,
                    lanes: 1,
                    instance: self.instance,
                    adapter_only: false,
                }],
                media: Vec::new(),
                voxels: Vec::new(),
            }))
            .map_err(|e| e.to_string())?;
        self.engine
            .settle_frame(&mut ticket)
            .map_err(|e| e.to_string())?;
        let mut take = |dense: u32| -> Result<i32, String> {
            let cell = self
                .engine
                .take_channel(self.instance, dense)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| format!("channel {dense} holds no cell"))?;
            let first = i32::from_le_bytes(cell[..4].try_into().expect("an i32"));
            let even = cell
                .as_chunks::<4>()
                .0
                .iter()
                .all(|word| i32::from_le_bytes(*word) == first);
            if !even {
                return Err(format!("channel {dense}'s cell holds unlike ids"));
            }
            Ok(first)
        };
        let got = (take(0)?, take(4)?);
        // The stacked rows are another test's: drained so the ring has room.
        self.engine
            .take_channel(self.instance, 1)
            .map_err(|e| e.to_string())?;
        Ok(got)
    }
}

#[test]
fn an_instance_in_pool_slots_fires_through_moves() {
    let boot = backend::DeviceBoot::default();
    if backend::Context::bind(&boot, engine::Allowance::WHOLE).is_err() {
        eprintln!("not asked: no device");
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "pie-{}-roles-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let path = dir.join(format!("{}-{NAME}.zt", backend::NAME));
    let planes = planes();
    write(&path, &planes);
    let argmax: Vec<i32> = (0..VOCAB as usize)
        .map(|token| {
            let row = &planes["embed"].1[token * WIDTH as usize..][..WIDTH as usize];
            (0..row.len())
                .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                .expect("a row") as i32
        })
        .collect();

    let trace = poem::trace_hybrid(NAME, &Outputs, backend::PLATFORM);
    let facts = trace.facts.clone();
    let t_input = trace
        .inputs
        .iter()
        .position(|input| input.name == "t")
        .expect("the plan reads `t`") as u32;
    let mut engine = backend::Engine::new(boot, contract_for);
    let loaded = engine
        .load(LoadRequest {
            trace,
            checkpoint: Checkpoint::Path(path),
            budgets: Budgets {
                max_lanes: 4,
                max_tokens: 16,
                buckets: vec![16],
                max_adapters: 0,
                page_size: 16,
                max_context: 64,
                context_declared: false,
                slots: 4,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency: Residency {
                host_kv_budget: Some(64 << 20),
                ..Residency::default()
            },
            ordinal: 0,
            frames_in_flight: 1,
        })
        .expect("the miniature loads");
    let chunks = loaded.caps.chunks.clone();
    let top = chunks.segment_chunks;
    assert!(top >= 10, "segment 0 holds the test's chunks: {top}");
    assert!(
        chunks.host_chunks >= 2,
        "host chunks: {}",
        chunks.host_chunks
    );
    // The guests' chunks, near the top of the committed segment: tables,
    // cells, two blank ones, two the first evacuation moves onto, and the
    // two the second evacuation and an unpark land in.
    let [
        tables,
        cells,
        blank,
        blank_cells,
        onto,
        onto_cells,
        fresh_tables,
        fresh_cells,
    ] = [1, 2, 3, 4, 5, 6, 7, 8].map(|k| top - k);
    let mut slots = Slots::of(&chunks, tables, cells);
    let instance_shape = slots.instance.expect("an Instance shape");
    let channel_shape = slots.channel;
    assert_eq!(
        channel_shape.is_some(),
        backend::NAME == "vulkan",
        "Vulkan keeps cells on the device; wgpu keeps every cell on the host"
    );

    let mut decls: Vec<OutputDecl> = Vec::new();
    for decl in loaded.caps.classes.classes.iter().flat_map(|c| &c.outputs) {
        if decls.iter().any(|seen| seen.name == decl.name) {
            continue;
        }
        decls.push(OutputDecl {
            name: decl.name.clone(),
            axis: match decl.axis {
                InputAxis::Lane => Axis::Lane,
                InputAxis::Row => Axis::Row,
                InputAxis::Voxel => Axis::Voxel,
            },
            width: decl.width,
            dtype: match decl.dtype {
                poem_ir::Dtype::I32 => EtaDtype::I32,
                _ => EtaDtype::F32,
            },
            layers: decl.layers,
            readout: decl.readout,
        });
    }
    let bound = eta_ir::validate::bind(program(), vec![decls], EngineCaps::default())
        .expect("the program binds");
    let stages = eta_compiler::plan::compile_bound(&bound);
    let program = engine
        .register_program(&ProgramRegistration {
            program_hash: 0x5107,
            launch: eta_compiler::codegen::launch::build(&bound, &stages),
            ..Default::default()
        })
        .expect("the program registers");
    // Channel 3 (`last`) is the instance's own; the rest are registered.
    let channels: Vec<ChannelId> = (1..=5).map(ChannelId).collect();
    let registered: Vec<ChannelId> = [0usize, 1, 2, 4].map(|k| channels[k]).to_vec();
    for (k, (shape, dtype, host_role)) in [
        (vec![1, 1], EtaDtype::I32, HostRole::Reader),
        (vec![LAYERS, WIDTH], EtaDtype::F32, HostRole::Reader),
        (vec![1, 1], EtaDtype::F32, HostRole::Writer),
        (vec![1, WIDE], EtaDtype::I32, HostRole::Reader),
    ]
    .into_iter()
    .enumerate()
    {
        slots.register(
            &mut engine,
            ChannelRegistration {
                id: registered[k],
                shape,
                dtype: ChanDType::Concrete(dtype),
                host_role,
                seeded: false,
                extern_dir: None,
                capacity: 2,
                extern_name: Vec::new(),
                slots: Vec::new(),
            },
        );
    }
    let rows = 3u32;
    let binding = slots.binding(
        &engine,
        InstanceBinding {
            program,
            channels: channels.clone(),
            seeds: vec![ChannelSeed {
                channel: 3,
                bytes: SEED.to_le_bytes().repeat(WIDE as usize),
            }],
            geometry: GeometryClass::Host,
            extents: engine::program::BindExtents {
                row_count: 1,
                token_count: rows,
                sampled_rows: 1,
                query_len: rows,
                ..engine::program::BindExtents::default()
            },
            device_resolved: Default::default(),
            instance_slots: Vec::new(),
            ring_slots: Vec::new(),
        },
        &registered,
    );
    assert!(!binding.instance_slots.is_empty(), "the tables take slots");
    assert_eq!(
        !binding.ring_slots.is_empty(),
        channel_shape.is_some(),
        "the own kept channel takes slots where cells lie on the device"
    );
    let instance = engine
        .bind_instance(&binding)
        .expect("the instance binds")
        .id;
    assert!(
        engine
            .publish_channel(instance, 2, &0.375f32.to_le_bytes())
            .expect("the timestep publishes"),
        "the ring had room"
    );
    let mut bound = Bound {
        engine,
        instance,
        channels,
        t_input,
        word: facts.word(&Request::new(rows, false)),
        argmax,
    };
    // Rounds read distinct tokens whose argmax differ from each other and
    // from zero, so a stale or blank read shows.
    let tokens: Vec<u32> = {
        let mut picked: Vec<u32> = Vec::new();
        for token in 0..VOCAB {
            let id = bound.argmax[token as usize];
            if id != 0 && picked.iter().all(|&p| bound.argmax[p as usize] != id) {
                picked.push(token);
            }
        }
        picked
    };
    assert!(tokens.len() >= 4, "enough distinct rows: {tokens:?}");
    let want = |bound: &Bound, token: u32| bound.argmax[token as usize];
    // `last` carries the previous pass's ids: on the device where it keeps
    // cells, else through the host ring.
    let mut last = SEED;
    let round = |bound: &mut Bound, token: u32, last: &mut i32, what: &str| {
        let got = bound
            .pass(token)
            .unwrap_or_else(|why| panic!("{what}: the pass fires: {why}"));
        assert_eq!(
            got,
            (2 * want(bound, token), 2 * *last),
            "{what}: (ids, seen), each doubled"
        );
        *last = want(bound, token);
    };
    round(&mut bound, tokens[0], &mut last, "first pass");
    round(&mut bound, tokens[1], &mut last, "second pass");

    // Every slot the instance holds: its tables', then its own kept cell's.
    let held: Vec<(ShapeId, Vec<u32>)> =
        std::iter::once((instance_shape.id, binding.instance_slots.clone()))
            .chain(binding.ring_slots.iter().map(|ring| {
                (
                    channel_shape.expect("a channel shape").id,
                    ring.slots.clone(),
                )
            }))
            .collect();
    let cells_held = channel_shape.is_some();

    // Each guest chunk onto another, and a blank chunk onto the place each
    // left: what the guest bound there is gone. The engine rebinds as it
    // evacuates.
    let moves = |pairs: [(u32, u32); 4]| Evacuation {
        moves: pairs
            .into_iter()
            .enumerate()
            .filter(|&(k, _)| cells_held || k % 2 == 0)
            .map(|(_, pair)| pair)
            .collect(),
    };
    bound
        .engine
        .evacuate(&moves([
            (tables, onto),
            (cells, onto_cells),
            (blank, tables),
            (blank_cells, cells),
        ]))
        .expect("the evacuation");
    round(&mut bound, tokens[2], &mut last, "after an evacuation");

    // The same again on the shell alone, which moves the chunks and leaves
    // the guests bound where they were: the pass reads blank tables. (The
    // guest chunks sit on `onto`, the blank ones on their old places.)
    let before = last;
    bound
        .engine
        .shell_mut()
        .expect("loaded")
        .evacuate(&moves([
            (tables, fresh_tables),
            (cells, fresh_cells),
            (blank, onto),
            (blank_cells, onto_cells),
        ]))
        .expect("a bare evacuation");
    let stale = bound.pass(tokens[3]);
    assert!(
        stale.as_ref().ok() != Some(&(2 * want(&bound, tokens[3]), 2 * before)),
        "without a rebind the pass reads blank tables, and answered {stale:?}"
    );
    // The rebind the bare evacuation skipped: the same refs, placed anew.
    bound
        .engine
        .rebind_slots(Rebind::Instance {
            id: bound.instance,
            instance: binding.instance_slots.clone(),
            rings: binding.ring_slots.clone(),
        })
        .expect("the slots rebind");
    // The stale pass kept its ids where the cell no longer is: `seen` reads
    // the pass before it.
    last = before;
    round(&mut bound, tokens[1], &mut last, "after the rebind");

    // Park and unpark: each slot copied to the host tier, its device place
    // blanked, copied back into fresh refs.
    let device = <backend::Api as engine_gpu::api::Host>::domain(bound.engine.boot());
    let copy = |engine: &mut backend::Engine, shape: ShapeId, src, dst, pairs: Vec<(u32, u32)>| {
        engine
            .copy_slots(&SlotCopy {
                shape,
                src,
                dst,
                slots: pairs,
                runs: Vec::new(),
            })
            .expect("the slots copy");
    };
    let mut fresh: Vec<Vec<u32>> = Vec::new();
    for (k, (shape, refs)) in held.iter().enumerate() {
        let per = if k == 0 {
            instance_shape.per_chunk
        } else {
            channel_shape.expect("a channel shape").per_chunk
        };
        // Host chunk 0 holds the tables, 1 the cells.
        let host: Vec<u32> = (0..refs.len() as u32)
            .map(|i| k.min(1) as u32 * per + i)
            .collect();
        let pairs = |from: &[u32], to: &[u32]| -> Vec<(u32, u32)> {
            from.iter().copied().zip(to.iter().copied()).collect()
        };
        copy(
            &mut bound.engine,
            *shape,
            device,
            MemoryDomain::HostPinned,
            pairs(refs, &host),
        );
        // The device place blanked from a slot of a blank chunk.
        let blank_ref = if k == 0 { blank } else { blank_cells } * per;
        let blanks: Vec<u32> = refs.iter().map(|_| blank_ref).collect();
        copy(
            &mut bound.engine,
            *shape,
            device,
            device,
            pairs(&blanks, refs),
        );
        let chunk = if k == 0 { onto } else { onto_cells };
        let back: Vec<u32> = (0..refs.len() as u32).map(|i| chunk * per + i).collect();
        copy(
            &mut bound.engine,
            *shape,
            MemoryDomain::HostPinned,
            device,
            pairs(&host, &back),
        );
        fresh.push(back);
    }
    let stale = bound.pass(tokens[0]);
    assert!(
        stale.as_ref().ok() != Some(&(2 * want(&bound, tokens[0]), 2 * last)),
        "unparked without a rebind the pass reads blank tables, and answered {stale:?}"
    );
    let before = last;
    bound
        .engine
        .rebind_slots(Rebind::Instance {
            id: bound.instance,
            instance: fresh[0].clone(),
            rings: binding
                .ring_slots
                .iter()
                .zip(&fresh[1..])
                .map(|(ring, slots)| RingSlots {
                    channel: ring.channel,
                    slots: slots.clone(),
                })
                .collect(),
        })
        .expect("the unparked slots rebind");
    last = before;
    round(&mut bound, tokens[2], &mut last, "after a park and unpark");
    round(&mut bound, tokens[3], &mut last, "a pass later");

    drop(bound);
    let _ = std::fs::remove_dir_all(&dir);
}
