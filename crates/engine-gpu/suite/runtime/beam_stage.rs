//! The guest-program contract (`eta_compiler::contract`) on this engine:
//! beam_search's step at widths 2 and 4, registered, bound with its seeds,
//! fired over the case's logits seated as the case binds them (one lane's
//! seat holding every row, and a lane a row laid last first with NaN gaps),
//! and every channel the host reads what the interpreter puts there. A seat
//! no row of the program reads is refused by name.
//!
//! The instance binds on an engine that loaded a miniature whose logits are
//! the case's width; the fire is the plane's, over seats in a buffer of the
//! test's own.

use std::collections::BTreeMap;

use checkpoint::file::emit::{self, Object};
use checkpoint::serving::Stamp;
use engine::ChannelId;
use engine::Engine;
use engine::channel::{ChannelRegistration, ChannelSeed};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use engine::program::{InstanceBinding, ProgramRegistration};
use engine_gpu::program::Programs as _;
use eta_compiler::contract::{Binding, Case, Seat, beam_stage};
use eta_ir::registry::GeometryClass;
use poem::{Dtype, ForwardHybrid, Input, StateRows, Weight, ops};

use super::outputs::contract_for;
use super::slots::Slots;
use crate::backend;

const NAME: &str = "beam-logits-mini";
/// The table's rows: the tokens a lane may carry.
const TOKENS: u32 = 4;
const VOCAB: u32 = eta_compiler::contract::QWEN35_VOCAB;

struct Logits;

impl ForwardHybrid for Logits {
    fn states(&self) -> StateRows {
        StateRows::new()
    }
    fn forward(&self, inputs: Input) {
        let table = Weight::sym("embed", [u64::from(TOKENS), u64::from(VOCAB)], Dtype::Bf16);
        let x = ops::gather::embed(&inputs.tokens(), &table, TOKENS);
        let x = ops::gather::gather_rows(&x, &inputs.readout_rows());
        poem::output("logits", &x, None);
    }
}

/// The table, bf16 values in [-8, 8) as f32.
fn table() -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    (0..TOKENS as usize * VOCAB as usize)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 16.0 - 8.0;
            f32::from_bits(x.to_bits() & 0xffff_0000)
        })
        .collect()
}

fn write(path: &std::path::Path, values: &[f32]) {
    let raw: Vec<u8> = values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let objects = [Object::leaf(
        "embed",
        vec![u64::from(TOKENS), u64::from(VOCAB)],
        ztensor::Leaf::BF16,
        &raw,
    )];
    emit::write(
        path,
        &Stamp::of(backend::NAME, NAME),
        &BTreeMap::new(),
        4096,
        &objects,
        |o, p, _| panic!("{o}/{p} is not streamed here"),
    )
    .expect("the artifact writes");
}

/// What the device puts in each of `case`'s readers, fired over the case's
/// logits seated at `seats`, or why the fire was refused.
fn fired(
    engine: &mut backend::Engine,
    loaded: &engine::load::Loaded,
    case: &Case,
    seats: &[Seat],
) -> Result<Vec<Vec<u8>>, String> {
    // Each fire's program and channels are its own.
    static FIRES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let fire = FIRES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let rows = case.rows;
    let stages = eta_compiler::plan::compile_bound(&case.bound);
    let program = engine
        .register_program(&ProgramRegistration {
            program_hash: 0xbea0_0000 + fire,
            launch: eta_compiler::codegen::launch::build(&case.bound, &stages),
            ..Default::default()
        })
        .expect("the program registers");
    let chunks = &loaded.caps.chunks;
    let mut slots = Slots::of(chunks, chunks.segment_chunks - 1, chunks.segment_chunks - 2);
    let base = 1 + 64 * fire;
    let channels: Vec<ChannelId> = case
        .bound
        .container
        .channels
        .iter()
        .enumerate()
        .map(|(at, decl)| {
            let id = ChannelId(base + at as u64);
            slots.register(
                engine,
                ChannelRegistration {
                    id,
                    shape: decl.shape.dims()[..decl.shape.rank()].to_vec(),
                    dtype: decl.dtype.clone(),
                    host_role: decl.host_role,
                    seeded: decl.seeded,
                    extern_dir: None,
                    capacity: decl.capacity,
                    extern_name: Vec::new(),
                    slots: Vec::new(),
                },
            );
            id
        })
        .collect();
    let binding = slots.binding(
        engine,
        InstanceBinding {
            program,
            channels: channels.clone(),
            seeds: case
                .wire_seeds()
                .into_iter()
                .map(|(channel, bytes)| ChannelSeed { channel, bytes })
                .collect(),
            geometry: GeometryClass::Host,
            extents: engine::program::BindExtents {
                row_count: rows,
                token_count: rows,
                sampled_rows: rows,
                query_len: rows,
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
        &channels,
    );
    let instance = engine
        .bind_instance(&binding)
        .expect("the instance binds")
        .id;

    let device = engine
        .shell
        .as_ref()
        .expect("the miniature is loaded")
        .device();
    let bytes: Vec<u8> = case.buffer().iter().flat_map(|x| x.to_le_bytes()).collect();
    let mut buffer = backend::Buffer::zeroed(
        &device,
        bytes.len() as u64,
        engine_gpu::books::owners::GUEST,
    )
    .expect("the logits' buffer");
    buffer.write(0, &bytes).expect("the logits upload");
    let lanes = seats
        .iter()
        .map(|seat| seat.lane as usize + 1)
        .max()
        .unwrap_or(1);
    let mut table: Vec<Vec<engine_gpu::guest::run::Readout<'_, backend::Api>>> =
        vec![Vec::new(); lanes];
    for seat in seats {
        // A seat holds every row from its offset on that the buffer has.
        let held = (bytes.len() as u64 - seat.offset) / 4;
        let width = if case.binding == Binding::OneLane && seat.lane == 0 {
            (held as u32 / seat.row_stride.max(1)).min(rows) * seat.width
        } else {
            seat.width
        };
        table[seat.lane as usize].push(engine_gpu::guest::run::Readout {
            output: "logits",
            seat: &buffer,
            at: seat.offset,
            width,
            dtype: eta_ir::Dtype::F32,
        });
    }
    let fired = engine.programs.fire_seated(&device, instance, table);
    let got = fired.map_err(|why| why.to_string()).map(|()| {
        case.readers
            .iter()
            .map(|&channel| {
                engine
                    .take_channel(instance, channel)
                    .expect("the channel reads")
                    .expect("the channel holds a cell")
            })
            .collect()
    });
    engine
        .close_instance(instance)
        .expect("the instance closes");
    got
}

#[test]
fn a_beam_step_answers_what_the_interpreter_answers() {
    let boot = crate::gpu::device_boot();
    if backend::Context::bind(&boot, backend::ALLOWANCE).is_err() {
        eprintln!("not asked: no device");
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "pie-{}-beam-stage-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let path = dir.join(format!("{}-{NAME}.zt", backend::NAME));
    let table = table();
    write(&path, &table);
    let trace = poem::trace_hybrid(NAME, &Logits, backend::PLATFORM);
    let mut engine = backend::Engine::new(boot, contract_for);
    let loaded = engine
        .load(LoadRequest {
            trace: trace.clone(),
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
                device: crate::gpu::share(),
                ..Residency::default()
            },
            ordinal: 0,
            frames_in_flight: 1,
        })
        .expect("the miniature loads");
    for case in beam_stage() {
        let want = case.expected();
        let got = fired(&mut engine, &loaded, &case, &case.seats())
            .unwrap_or_else(|why| panic!("{}: {why}", case.name));
        for ((channel, got), want) in case.readers.iter().zip(&got).zip(&want) {
            assert_eq!(got, want, "{}: channel {channel}", case.name);
        }
        eprintln!("{}: every reader the interpreter's", case.name);
    }
    // A seat no row of the program reaches is refused by name, not dropped:
    // lanes 0 and 3 of an output read at lane 0 are no rows 0 and 1.
    let case = &beam_stage()[0];
    let mut seats = case.seats();
    seats.push(Seat {
        lane: 3,
        ..seats[0]
    });
    let refused =
        fired(&mut engine, &loaded, case, &seats).expect_err("a seat past the rows is refused");
    assert!(refused.contains("bound at lane 3"), "{refused}");
    eprintln!("a seat past the rows: {refused}");
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
