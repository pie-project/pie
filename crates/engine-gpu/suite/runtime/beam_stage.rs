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

/// A one-row top-k program over the logits: the row's top `k` (cast to
/// bf16 first where `bf16`), its values and indices put on two channels.
fn top_k_case(name: &'static str, k: u32, bf16: bool) -> Case {
    use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
    use eta_ir::op::Op;
    use eta_ir::registry::{Axis, EngineCaps, OutputDecl, Stage};
    use eta_ir::types::{Dtype as EtaDtype, Shape};
    let value = if bf16 { EtaDtype::Bf16 } else { EtaDtype::F32 };
    let decl = |dtype: EtaDtype| ChannelDecl {
        shape: Shape::vector(k),
        dtype: ChanDType::Concrete(dtype),
        capacity: 2,
        host_role: HostRole::Reader,
        seeded: false,
    };
    let mut ops = vec![Op::Output {
        index: 0,
        lane: 0,
        shape: Shape::matrix(1, VOCAB),
        dtype: EtaDtype::F32,
    }];
    let row = if bf16 {
        ops.push(Op::Cast {
            value: 0,
            dtype: EtaDtype::Bf16,
        });
        1
    } else {
        0
    };
    // Flattened, as the beam step lays its candidates: a one-row top-k
    // (`steps::one_row`), where the readout's rows are a dynamic axis.
    ops.push(Op::Reshape {
        value: row,
        shape: Shape::vector(VOCAB),
    });
    let input = row + 1;
    ops.push(Op::TopK { input, k });
    ops.push(Op::ChanPut {
        chan: 0,
        value: input + 1,
    });
    ops.push(Op::ChanPut {
        chan: 1,
        value: input + 2,
    });
    let container = TraceContainer {
        names: vec!["logits".to_string()],
        channels: vec![decl(value), decl(EtaDtype::U32)],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    };
    let logits = OutputDecl {
        name: "logits".into(),
        axis: Axis::Row,
        width: VOCAB,
        dtype: EtaDtype::F32,
        layers: 0,
        readout: true,
    };
    let bound = eta_ir::validate::bind(container, vec![vec![logits]], EngineCaps::default())
        .expect("the top-k program binds");
    Case {
        name,
        binding: Binding::OneLane,
        bound,
        rows: 1,
        vocab: VOCAB,
        seeds: Vec::new(),
        logits: Vec::new(),
        readers: vec![0, 1],
    }
}

/// The row's top `k` as the sort orders them: keys descending (the
/// runtime's `sort_key`: NaN last, -0 as 0), ties by index.
fn sorted_top(row: &[f32], k: usize) -> Vec<usize> {
    let key = |x: f32| -> (u8, f32) {
        if x.is_nan() {
            (1, 0.0)
        } else {
            (0, -(x + 0.0))
        }
    };
    let mut order: Vec<usize> = (0..row.len()).collect();
    order.sort_by(|&a, &b| {
        key(row[a])
            .partial_cmp(&key(row[b]))
            .expect("keys order")
            .then(a.cmp(&b))
    });
    order.truncate(k);
    order
}

/// Whether the `k`-th key's ties straddle the cut: some of them in the top
/// `k`, some not, so only the index order decides which.
fn ties_straddle(row: &[f32], k: usize) -> bool {
    let top = sorted_top(row, k);
    let Some(&last) = top.last() else {
        return false;
    };
    let at = row[last];
    let tied = |x: f32| x == at || (x.is_nan() && at.is_nan());
    let inside = top.iter().filter(|&&i| tied(row[i])).count();
    let all = row.iter().filter(|&&x| tied(x)).count();
    inside > 0 && all > inside
}

/// Rows a one-row top-k selects over, each with its `k`, every one with
/// ties straddling the cut: heavy ties at and around the `k`-th key,
/// `-inf`s among NaNs, a few NaNs, a bf16 row; and the rows the select
/// hands to the sort: `k` reaching into the NaNs, ties past its
/// candidates, `k` past its candidates.
fn top_k_cases() -> Vec<Case> {
    let width = VOCAB as usize;
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut place = |row: &mut Vec<f32>, count: usize, value: f32| {
        let mut put = 0;
        while put < count {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let at = (state % width as u64) as usize;
            if row[at] == -5.0 {
                row[at] = value;
                put += 1;
            }
        }
    };
    // 30 keys tied at the top, 40 tied below them, the rest lower still.
    let mut tied = vec![-5.0f32; width];
    place(&mut tied, 30, 3.0);
    place(&mut tied, 40, 2.0);
    // 10 finite keys, then 60 tied at -inf, the rest NaN.
    let mut infs = vec![-5.0f32; width];
    place(&mut infs, 10, 1.0);
    place(&mut infs, 60, f32::NEG_INFINITY);
    for x in &mut infs {
        if *x == -5.0 {
            *x = f32::NAN;
        }
    }
    // The tied row with 20 NaN among its lowest keys.
    let mut nans = tied.clone();
    for x in nans.iter_mut().filter(|x| **x == -5.0).take(20) {
        *x = f32::NAN;
    }
    // 10 keys, the rest NaN.
    let mut mostly_nan = vec![f32::NAN; width];
    for (at, x) in mostly_nan.iter_mut().enumerate().skip(100).take(10) {
        *x = (at % 3) as f32;
    }
    // 2000 keys tied at the top: more than the select's candidates.
    let mut crowd = vec![-5.0f32; width];
    place(&mut crowd, 2000, 4.0);
    let with = |mut case: Case, row: Vec<f32>| {
        case.logits = row;
        case
    };
    vec![
        with(top_k_case("top 1 of 30 tied", 1, false), tied.clone()),
        with(top_k_case("top 4 of 30 tied", 4, false), tied.clone()),
        with(
            top_k_case("top 50 across two tie groups", 50, false),
            tied.clone(),
        ),
        with(top_k_case("top 4 of 30 tied, bf16", 4, true), tied.clone()),
        with(top_k_case("top 40 into the -inf ties", 40, false), infs),
        with(top_k_case("top 50 beside NaN", 50, false), nans),
        with(
            top_k_case("top 20 reaching into NaN", 20, false),
            mostly_nan,
        ),
        with(
            top_k_case("top 4 of 2000 tied, past the candidates", 4, false),
            crowd,
        ),
        with(
            top_k_case("top width-1, past the candidates", VOCAB - 1, false),
            tied,
        ),
    ]
}

/// A one-row top-k selects its `k`-th key and sorts its candidates
/// (`steps::Ladder::Select`), and answers what the sort answers: the
/// interpreter's top-k, values and indices, where ties straddle the cut,
/// beside NaN and `-inf`, and where it hands the row to the sort.
#[test]
fn a_top_k_by_select_answers_what_the_sort_answers() {
    let boot = crate::gpu::device_boot();
    if backend::Context::bind(&boot, backend::ALLOWANCE).is_err() {
        eprintln!("not asked: no device");
        return;
    }
    let dir = std::env::temp_dir().join(format!(
        "pie-{}-top-k-select-{}",
        backend::NAME,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let path = dir.join(format!("{}-{NAME}.zt", backend::NAME));
    write(&path, &table());
    let trace = poem::trace_hybrid(NAME, &Logits, backend::PLATFORM);
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
                device: crate::gpu::share(),
                ..Residency::default()
            },
            ordinal: 0,
            frames_in_flight: 1,
            steps: Default::default(),
        })
        .expect("the miniature loads");
    for case in top_k_cases() {
        let k = case.bound.container.channels[0].shape.dims()[0] as usize;
        assert!(
            ties_straddle(&case.logits, k),
            "{}: the k-th key's ties straddle the cut",
            case.name
        );
        let want = case.expected();
        let got = fired(&mut engine, &loaded, &case, &case.seats())
            .unwrap_or_else(|why| panic!("{}: {why}", case.name));
        for ((channel, got), want) in case.readers.iter().zip(&got).zip(&want) {
            assert_eq!(got, want, "{}: channel {channel}", case.name);
        }
        eprintln!("{}: the sort's values and indices", case.name);
    }
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
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
            steps: Default::default(),
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
