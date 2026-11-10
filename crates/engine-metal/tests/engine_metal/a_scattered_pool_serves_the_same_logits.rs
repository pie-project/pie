#![cfg(target_vendor = "apple")]

//! A model served out of the chunk pool lands the same logits, bit for bit,
//! wherever its slots sit: its kv pages and its recurrent record in the
//! seats' own runs, or scattered over chunks the runtime names by slot refs
//! (pages strided across three chunks, the record in a chunk of its own),
//! with those chunks evacuated mid-decode out of the pool's top segment,
//! which is then given back, the decode running on through the moved
//! chunks. Two miniatures: a dense one and a gated-delta hybrid.
//!
//! (Before the per-plane pools were deleted this script also ran over them in
//! the same process and landed the same bits.)

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine::chunk::{Evacuation, Growth};
use engine_metal::serve::Seated;
use engine_metal::store::layout::{FIXED, LIST};
use engine_metal::{Boot, Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{ParamSource, Trace};

const PAGE: u32 = 16;
const CONTEXT: u32 = 256;
const SEATS: u32 = 4;

fn fnv(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A checkpoint of `trace`'s own planes seeded by name: norms near one, the
/// rest small.
fn write_checkpoint(path: &Path, trace: &Trace) {
    let mut writer = ztensor::Writer::create(path).expect("the fixture writes");
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    for param in planes {
        let count = param.shape.iter().product::<u64>() as usize;
        let norm = param.name.ends_with("norm");
        let mut seed = fnv(&param.name);
        let mut next = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            seed >> 33
        };
        let unit = |bits: u64| (bits as f32 / (1u64 << 31) as f32) - 0.5;
        let value = |bits: u64| {
            if norm {
                1.0 + 0.05 * unit(bits)
            } else {
                0.08 * unit(bits)
            }
        };
        let (leaf, bytes): (ztensor::Leaf, Vec<u8>) = match param.dtype {
            Dtype::Bf16 => (
                ztensor::Leaf::BF16,
                (0..count)
                    .flat_map(|_| ((value(next()).to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
            ),
            Dtype::F32 => (
                ztensor::Leaf::F32,
                (0..count)
                    .flat_map(|_| value(next()).to_le_bytes())
                    .collect(),
            ),
            other => panic!(
                "`{}` is {other:?}, not a dtype the fixture seeds",
                param.name
            ),
        };
        writer
            .add(param.name.as_str(), param.shape.clone(), leaf, &bytes)
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer.finish().expect("the fixture finishes");
}

struct Fixture {
    trace: Trace,
    contract: ModelContract,
    container: PathBuf,
}

fn fixture(model: &str) -> Fixture {
    let package = poem_compiler::catalog::package_of(model).expect("the model is listed");
    let deploy = poem_compiler::catalog::one_rank(Dtype::Bf16, Dtype::Bf16);
    let trace = package
        .trace(model, &deploy, model, Platform::Metal)
        .unwrap_or_else(|why| panic!("`{model}` traces: {why:#}"));
    let dir = std::env::temp_dir().join(format!("pie-bm-pool-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("the fixture dir");
    let container = dir.join(format!("{model}.zt"));
    if !container.exists() {
        write_checkpoint(&container, &trace);
    }
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Metal)
        .expect("the plan reads its own planes");
    Fixture {
        trace,
        contract,
        container,
    }
}

fn load(fixture: &Fixture) -> Shell {
    Shell::load(Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(SEATS, 64),
        patches: None,
        voxels: None,
        profile: None,
        page_size: PAGE,
        context: CONTEXT,
        slots: SEATS,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_metal::ResidencyPlan::default(),
        profiles: None,
    })
    .unwrap_or_else(|why| panic!("the shell loads: {why}"))
}

fn word(trace: &Trace, rows: u32) -> u64 {
    trace.facts.word(&Request::new(rows, false))
}

fn argmax(logits: &[f32]) -> u32 {
    (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap_or(0) as u32
}

/// Where the script's two lanes keep their state.
enum Placement {
    /// Seat-addressed: each lane's seat runs.
    Seats,
    /// Slot refs the runtime would name: pages strided over chunks
    /// `first .. first + 3` from the top down, lane `l`'s record in chunk
    /// `first + 3 + l`.
    Refs {
        first: u32,
        per_list: u32,
        per_fixed: u32,
    },
}

impl Placement {
    fn page(&self, lane: u32, page: u32) -> u32 {
        match *self {
            Placement::Seats => unreachable!(),
            Placement::Refs {
                first, per_list, ..
            } => {
                let chunk = first + 2 - (page % 3);
                let index = (7 * page + 3 + 61 * lane) % per_list;
                chunk * per_list + index
            }
        }
    }

    fn record(&self, lane: u32) -> u32 {
        match *self {
            Placement::Seats => unreachable!(),
            Placement::Refs {
                first, per_fixed, ..
            } => (first + 3 + lane) * per_fixed + (lane * 3 + 1) % per_fixed,
        }
    }
}

const PROMPTS: [&[u32]; 2] = [
    &[
        7, 42, 11, 200, 3, 99, 128, 65, 17, 4, 150, 23, 77, 1, 9, 30, 31, 101, 5, 6, 200, 2, 13,
        14, 15, 66, 67, 68, 69, 70, 100, 110, 120, 130, 140, 150, 160,
    ],
    &[
        5, 4, 3, 2, 1, 250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110,
    ],
];

/// Prefills each lane alone, then decodes both together `steps` times;
/// after `evacuate_at` decodes, `between` runs (an evacuation). Every
/// logit row, in order.
fn serve(
    shell: &mut Shell,
    trace: &Trace,
    place: &Placement,
    steps: usize,
    evacuate_at: usize,
    between: &mut dyn FnMut(&mut Shell),
) -> Vec<Vec<f32>> {
    let refs = matches!(place, Placement::Refs { .. });
    let mut held = [0u32; 2];
    let mut fed: [Vec<u32>; 2] = [PROMPTS[0].to_vec(), PROMPTS[1].to_vec()];
    let mut rows = Vec::new();
    let tables = |held: [u32; 2], rows: [u32; 2]| -> [Vec<u32>; 2] {
        [0, 1].map(|lane| {
            if !refs {
                return Vec::new();
            }
            let pages = (held[lane] + rows[lane]).div_ceil(PAGE);
            (0..pages)
                .map(|page| place.page(lane as u32, page))
                .collect()
        })
    };
    let fire = |shell: &mut Shell, lanes: &[(u32, &[u32], u32, &[u32])]| -> Vec<Vec<f32>> {
        let seated: Vec<Seated<'_>> = lanes
            .iter()
            .map(|&(lane, tokens, have, pages)| Seated {
                pages,
                held: Some(have),
                rs_slot: refs.then(|| place.record(lane)),
                refs,
                ..Seated::of(Lane {
                    slot: lane,
                    word: word(trace, tokens.len() as u32),
                    tokens,
                })
            })
            .collect();
        shell.fire_seated(&seated).expect("the fire lands")
    };
    // A fresh record for each lane: the refs-addressed ones are zeroed by
    // their first fire (`RsReset::Inferred` with nothing held).
    for lane in 0..2usize {
        let table = tables(held, [fed[0].len() as u32, fed[1].len() as u32]);
        let got = fire(
            shell,
            &[(lane as u32, &fed[lane], held[lane], &table[lane])],
        );
        held[lane] += fed[lane].len() as u32;
        fed[lane] = vec![argmax(&got[0])];
        rows.extend(got);
    }
    for step in 0..steps {
        if step == evacuate_at {
            between(shell);
        }
        let table = tables(held, [1, 1]);
        let got = fire(
            shell,
            &[
                (0, &fed[0], held[0], &table[0]),
                (1, &fed[1], held[1], &table[1]),
            ],
        );
        for lane in 0..2 {
            held[lane] += 1;
            fed[lane] = vec![argmax(&got[lane])];
        }
        rows.extend(got);
    }
    rows
}

/// The chunks the scattered placement holds: three of pages, two records.
const HELD: u32 = 5;

/// Panics at the first row whose logits differ in a bit.
fn same(what: &str, got: &[Vec<f32>], want: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len(), "{what}: rows");
    for (at, (got, want)) in got.iter().zip(want).enumerate() {
        let differ = got
            .iter()
            .zip(want)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert!(
            differ == 0 && got.len() == want.len(),
            "{what}: row {at} of {} differs in {differ} of {} logits (argmax {} against {})",
            want.len(),
            want.len(),
            argmax(got),
            argmax(want)
        );
    }
}

fn check(model: &str, hybrid: bool) {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture(model);
    const STEPS: usize = 12;

    let mut shell = load(&fixture);
    let facts = shell.pool_chunk_facts();
    facts.validate().expect("the facts hang together");
    eprintln!(
        "{model}: {} chunks of {} B, {} a segment, segments {:?}, shapes {:?}",
        facts.total_chunks,
        facts.chunk_bytes,
        facts.segment_chunks,
        facts.segments,
        facts
            .shapes
            .iter()
            .map(|s| (s.id.0, s.kind, s.slot_bytes, s.slots_per_chunk))
            .collect::<Vec<_>>()
    );
    assert!(facts.shapes.iter().all(|s| s.pinned.is_none()));
    assert_eq!(facts.shape(FIXED).is_some(), hybrid);
    let seats = serve(
        &mut shell,
        &fixture.trace,
        &Placement::Seats,
        STEPS,
        usize::MAX,
        &mut |_| {},
    );
    drop(shell);

    // Scattered over a segment of their own above the seats' runs, then
    // evacuated below it and that segment given back, mid-decode.
    let mut shell = load(&fixture);
    let facts = shell.pool_chunk_facts();
    let layout = shell.pool_layout();
    let segment = facts.segment_chunks;
    let low = 1 + layout.seat_chunks();
    let kept = (low + HELD).div_ceil(segment);
    let first = kept * segment;
    let commit = kept + HELD.div_ceil(segment);
    assert!(
        facts.segments.max >= commit,
        "{model}: the pool spans {commit} segments"
    );
    let per = |id| facts.shape(id).map_or(1, |s| s.slots_per_chunk);
    let place = Placement::Refs {
        first,
        per_list: per(LIST),
        per_fixed: per(FIXED),
    };
    shell.commit_segments(commit).expect("the segments commit");
    let sparse = facts.segments.growth == Growth::Bottom;
    let scattered = serve(
        &mut shell,
        &fixture.trace,
        &place,
        STEPS,
        STEPS / 2,
        &mut |shell| {
            let moves = (0..HELD).map(|at| (first + at, low + at)).collect();
            shell
                .evacuate(&Evacuation { moves })
                .expect("the evacuation lands");
            if sparse {
                shell
                    .release_segments(kept)
                    .expect("the top segments go back");
                let after = shell.pool_chunk_facts();
                assert_eq!(after.segments.committed, kept);
            }
        },
    );
    same(
        &format!("{model}: scattered and evacuated slots against the seats' runs"),
        &scattered,
        &seats,
    );
}

#[test]
fn a_scattered_dense_pool_serves_the_same_logits() {
    check("qwen3-micro-text", false);
}

#[test]
fn a_scattered_hybrid_pool_serves_the_same_logits() {
    check("qwen35-tiny", true);
}

/// The pools a GDN hybrid's catalog trace lays on this device, no weights
/// landed: one chunk space with a list, a record and an rs buffer shape, none
/// pinned; segments commit from the bottom and go back.
#[test]
#[allow(unsafe_code)]
fn a_reserved_hybrid_pool_states_shared_chunks() {
    use engine_metal::store::layout::{BUFFER, Layout};
    use engine_metal::store::{Pools, Rows};

    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let name = "qwen38-flash-next-mini-bf16-kv-bf16";
    let trace = poem_compiler::catalog::deployment(name)
        .expect("the deployment is listed")
        .trace(Platform::Metal);
    let device = engine_metal::Context::bind().expect("the system device");
    let facts = engine_metal::store::kv::probe(&trace).expect("the kv facts");
    let rows = Rows::plan(&trace, PAGE, &facts).expect("the rows lay out");
    let rs = engine_metal::rs::Layout::read(&trace).expect("the rs layout");
    let buffer = rs
        .as_ref()
        .map_or(0, |rs| engine_metal::rs::Buffers::page_layout(rs, PAGE).1);
    let bytes = [rows.list_bytes, rows.fixed_bytes, buffer];
    let chunk = Layout::chunk_bytes(bytes[0], bytes[1], bytes[2], 2 << 20);
    let segment = engine_metal::store::pool::segment_chunks(chunk);
    let layout =
        Layout::new(chunk, 3 * segment, segment, PAGE, bytes, SEATS, 16).expect("a layout");
    let paging = engine_metal::store::kv::Paging::of(PAGE, 256, SEATS, 1).expect("a paging");
    let mut pools = Pools::reserve(&device, rows, layout, paging, Vec::new()).expect("a pool");
    let facts = pools.chunk_facts();
    eprintln!(
        "{name}: {} chunks of {} B, segments {:?}, shapes {:?}",
        facts.total_chunks,
        facts.chunk_bytes,
        facts.segments,
        facts
            .shapes
            .iter()
            .map(|s| (s.id.0, s.kind, s.slot_bytes, s.slots_per_chunk))
            .collect::<Vec<_>>()
    );
    facts.validate().expect("the facts hang together");
    for id in [LIST, FIXED, BUFFER] {
        let shape = facts.shape(id).expect("the shape stands");
        assert!(shape.pinned.is_none());
        assert!(shape.slot_bytes * u64::from(shape.slots_per_chunk) <= facts.chunk_bytes);
    }
    assert_eq!(facts.chunk_bytes % facts.shape(LIST).unwrap().slot_bytes, 0);
    assert_eq!(facts.segments.max, 3);
    assert!(facts.act_chunks.is_empty(), "activation lives in the arena");
    if facts.segments.growth != Growth::Bottom {
        assert_eq!(facts.segments.committed, 3);
        return;
    }
    assert_eq!(facts.segments.committed, 0);
    pools.commit_segments(2).expect("two segments");
    assert_eq!(pools.chunk_facts().segments.committed, 2);
    assert!(pools.committed_bytes() >= 2 * u64::from(segment) * chunk);
    // SAFETY: nothing ran on this pool.
    unsafe { pools.release_segments(1) }.expect("one goes back");
    assert_eq!(pools.chunk_facts().segments.committed, 1);
}
