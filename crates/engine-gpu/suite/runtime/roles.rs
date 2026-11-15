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
use super::slots::{Shape as RoleShape, Slots};
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

/// The miniature loaded with a host tier and the program's instance bound
/// over `Instance` slots in chunk `top - 1` and, where the device keeps
/// cells, `Channel` slots in `top - 2`; two passes fired so both hold the
/// bytes a served guest's do.
struct Rig {
    bound: Bound,
    binding: InstanceBinding,
    instance_shape: RoleShape,
    channel_shape: Option<RoleShape>,
    /// The committed segment's chunks; the guests' lie near its top.
    top: u32,
    /// Tokens whose argmax differ from each other and from zero.
    tokens: Vec<u32>,
    /// What `last` holds now.
    last: i32,
    dir: std::path::PathBuf,
}

impl Rig {
    fn want(&self, token: u32) -> i32 {
        self.bound.argmax[token as usize]
    }

    /// One pass of `token`, checked against what `last` held.
    fn round(&mut self, token: u32, what: &str) {
        let got = self
            .bound
            .pass(token)
            .unwrap_or_else(|why| panic!("{what}: the pass fires: {why}"));
        assert_eq!(
            got,
            (2 * self.want(token), 2 * self.last),
            "{what}: (ids, seen), each doubled"
        );
        self.last = self.want(token);
    }

    /// Every slot the instance holds: its tables', then its own kept cell's.
    fn held(&self) -> Vec<(RoleShape, Vec<u32>)> {
        std::iter::once((self.instance_shape, self.binding.instance_slots.clone()))
            .chain(self.binding.ring_slots.iter().map(|ring| {
                (
                    self.channel_shape.expect("a channel shape"),
                    ring.slots.clone(),
                )
            }))
            .collect()
    }
}

/// [`Rig`], or `None` without a device.
fn rig(what: &str) -> Option<Rig> {
    let boot = backend::DeviceBoot::default();
    if backend::Context::bind(&boot, backend::ALLOWANCE).is_err() {
        eprintln!("not asked: no device");
        return None;
    }
    let dir = std::env::temp_dir().join(format!(
        "pie-{}-{what}-{}",
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
                device: backend::ALLOWANCE,
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
    // The guests' tables and cells, near the top of the committed segment.
    let mut slots = Slots::of(&chunks, top - 1, top - 2);
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
            adapter_slots: Vec::new(),
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
    let bound = Bound {
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
    // `last` carries the previous pass's ids: on the device where it keeps
    // cells, else through the host ring.
    let mut rig = Rig {
        bound,
        binding,
        instance_shape,
        channel_shape,
        top,
        tokens,
        last: SEED,
        dir,
    };
    rig.round(rig.tokens[0], "first pass");
    rig.round(rig.tokens[1], "second pass");
    Some(rig)
}

#[test]
fn an_instance_in_pool_slots_fires_through_moves() {
    let Some(mut rig) = rig("roles") else {
        return;
    };
    let top = rig.top;
    // The guests' chunks: tables, cells, two blank ones, two the first
    // evacuation moves onto, and the two the second evacuation and an
    // unpark land in.
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
    let (instance_shape, channel_shape) = (rig.instance_shape, rig.channel_shape);
    let binding = rig.binding.clone();
    let tokens = rig.tokens.clone();
    let held: Vec<(ShapeId, Vec<u32>)> = rig
        .held()
        .into_iter()
        .map(|(shape, refs)| (shape.id, refs))
        .collect();
    let cells_held = channel_shape.is_some();
    let want = |bound: &Bound, token: u32| bound.argmax[token as usize];
    let mut last = rig.last;
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
    let bound = &mut rig.bound;

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
    round(bound, tokens[2], &mut last, "after an evacuation");

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
            adapter: Vec::new(),
        })
        .expect("the slots rebind");
    // The stale pass kept its ids where the cell no longer is: `seen` reads
    // the pass before it.
    last = before;
    round(bound, tokens[1], &mut last, "after the rebind");

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
            adapter: Vec::new(),
        })
        .expect("the unparked slots rebind");
    last = before;
    round(bound, tokens[2], &mut last, "after a park and unpark");
    round(bound, tokens[3], &mut last, "a pass later");

    let _ = std::fs::remove_dir_all(&rig.dir);
}

/// The bytes of `refs` of `shape` read off the device where they lie, one
/// slot after another, once every frame and copy has landed.
fn device_bytes(engine: &mut backend::Engine, shape: &RoleShape, refs: &[u32]) -> Vec<u8> {
    let shell = engine.shell_mut().expect("loaded");
    shell.drain().expect("the frames land");
    let pools = shell.pools();
    let bytes = pools.slot_bytes(shape.id).expect("the slot's bytes") as usize;
    let mut out = vec![0u8; refs.len() * bytes];
    for (k, &slot) in refs.iter().enumerate() {
        let (buffer, at) = pools.slot_place(shape.id, slot).expect("the slot's place");
        engine_gpu::backend::Buffer::<backend::Api>::read(
            buffer,
            at,
            &mut out[k * bytes..][..bytes],
        )
        .expect("the slot reads");
    }
    out
}

/// A pattern no guest writes, seeded by `seed`, over `refs` of `shape` on
/// the device; returns it.
fn clobber(engine: &mut backend::Engine, shape: &RoleShape, refs: &[u32], seed: u8) -> Vec<u8> {
    let shell = engine.shell_mut().expect("loaded");
    shell.drain().expect("the frames land");
    let pools = shell.pools();
    let bytes = pools.slot_bytes(shape.id).expect("the slot's bytes") as usize;
    let pattern: Vec<u8> = (0..refs.len() * bytes)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect();
    for (k, &slot) in refs.iter().enumerate() {
        let (buffer, at) = pools.slot_place(shape.id, slot).expect("the slot's place");
        engine_gpu::backend::Buffer::<backend::Api>::write_shared(
            buffer,
            at,
            &pattern[k * bytes..][..bytes],
        )
        .expect("the slot writes");
    }
    pattern
}

/// A suspend and restore as the runtime's role store moves one
/// (`prepare_park`, then `prepare_unpark` into fresh device refs and a
/// rebind): each `Instance` and `Channel` slot the instance holds reaches
/// the host tier byte for byte, comes back byte for byte after its device
/// place was clobbered, and the instance fires as before. Without the copy
/// back the fresh refs hold other bytes.
#[test]
fn a_role_slot_round_trips_through_the_host_tier() {
    let Some(mut rig) = rig("role-tier") else {
        return;
    };
    let token = rig.tokens[2];
    // To a fixed point: `last` holds `token`'s ids, so every pass of it
    // reads (and leaves) the same.
    rig.round(token, "a pass to the fixed point");
    let fired = rig
        .bound
        .pass(token)
        .expect("the pass before the suspend fires");
    assert_eq!(
        fired,
        (2 * rig.want(token), 2 * rig.want(token)),
        "the pass before the suspend"
    );

    let held = rig.held();
    assert_eq!(
        held.len(),
        1 + usize::from(rig.channel_shape.is_some()),
        "the tables' slots, and the kept cell's where the device keeps cells"
    );
    let device = <backend::Api as engine_gpu::api::Host>::domain(rig.bound.engine.boot());
    let pairs = |from: &[u32], to: &[u32]| -> Vec<(u32, u32)> {
        from.iter().copied().zip(to.iter().copied()).collect()
    };
    let mut fresh: Vec<Vec<u32>> = Vec::new();
    for (k, (shape, refs)) in held.iter().enumerate() {
        let what = if k == 0 { "Instance" } else { "Channel" };
        let engine = &mut rig.bound.engine;
        let copy = |engine: &mut backend::Engine, src, dst, slots: Vec<(u32, u32)>| {
            engine
                .copy_slots(&SlotCopy {
                    shape: shape.id,
                    src,
                    dst,
                    slots,
                    runs: Vec::new(),
                })
                .unwrap_or_else(|why| panic!("{what}: the slots copy: {why}"));
        };
        let per = shape.per_chunk;
        // Host chunk 0 holds the tables, 1 the cell; device chunks
        // `top - 5` and `top - 6` the fresh refs an unpark takes.
        let host: Vec<u32> = (0..refs.len() as u32).map(|i| k as u32 * per + i).collect();
        let back: Vec<u32> = (0..refs.len() as u32)
            .map(|i| (rig.top - 5 - k as u32) * per + i)
            .collect();

        // The slots' bytes as the guest left them.
        let original = device_bytes(engine, shape, refs);
        assert!(
            original.iter().any(|&b| b != 0),
            "{what}: the slots hold the guest's bytes"
        );

        // Park: device to the host tier, which then holds the same bytes.
        copy(engine, device, MemoryDomain::HostPinned, pairs(refs, &host));
        let tier = engine
            .read_tier_slots(&[(shape.id, host.clone())])
            .unwrap_or_else(|why| panic!("{what}: the host tier reads: {why}"));
        assert!(
            tier == original,
            "{what}: the host tier holds the slots' bytes"
        );

        // The device places the park gave up, clobbered; the fresh refs
        // the unpark takes hold another guest's leavings.
        let clobbered = clobber(engine, shape, refs, 0xa5);
        assert!(
            device_bytes(engine, shape, refs) == clobbered && clobbered != original,
            "{what}: the clobber landed"
        );
        let leavings = clobber(engine, shape, &back, 0x5a);
        // The negative control: restored without the copy back, the fresh
        // refs do not hold the slots' bytes.
        let skipped = device_bytes(engine, shape, &back);
        assert!(
            skipped == leavings && skipped != original,
            "{what}: without the copy back the fresh refs hold other bytes"
        );

        // Unpark: the host tier into the fresh refs, byte for byte.
        copy(
            engine,
            MemoryDomain::HostPinned,
            device,
            pairs(&host, &back),
        );
        let restored = device_bytes(engine, shape, &back);
        assert!(
            restored == original,
            "{what}: the fresh refs hold the slots' bytes ({} of {} differ)",
            restored
                .iter()
                .zip(&original)
                .filter(|(a, b)| a != b)
                .count(),
            original.len()
        );
        eprintln!(
            "{}: {what}: {} slots, {} bytes, round trip through the host tier",
            backend::NAME,
            refs.len(),
            original.len()
        );
        fresh.push(back);
    }

    // The rebind an unpark ends with, then the same pass as before.
    rig.bound
        .engine
        .rebind_slots(Rebind::Instance {
            id: rig.bound.instance,
            instance: fresh[0].clone(),
            rings: rig
                .binding
                .ring_slots
                .iter()
                .zip(&fresh[1..])
                .map(|(ring, slots)| RingSlots {
                    channel: ring.channel,
                    slots: slots.clone(),
                })
                .collect(),
            adapter: Vec::new(),
        })
        .expect("the restored slots rebind");
    let after = rig
        .bound
        .pass(token)
        .expect("the pass after the restore fires");
    assert_eq!(after, fired, "the pass after the restore reads as before");

    let _ = std::fs::remove_dir_all(&rig.dir);
}
