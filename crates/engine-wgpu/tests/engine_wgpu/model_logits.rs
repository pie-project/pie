//! Whole miniature models served by the shell, end to end: two lanes prefill
//! alone and then decode together, every logit row kept. The rows are
//! written bit for bit to `$PIE_WGPU_LOGITS/<model>.bin` when that is set, so
//! a build can be held against another one (the chunk pool against the
//! per-row slabs it replaced) byte for byte.

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine::chunk::Evacuation;
use engine_wgpu::serve::Seated;
use engine_wgpu::store::layout::{FIXED, LIST};
use engine_wgpu::{Boot, Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{ParamSource, Trace};

pub const PAGE: u32 = 16;
pub const CONTEXT: u32 = 256;
pub const SEATS: u32 = 4;

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
            // Ids (a hash router's token-to-expert table): all 0, an id
            // every table holds.
            Dtype::I64 => (ztensor::Leaf::I(64), vec![0u8; 8 * count]),
            Dtype::I32 => (ztensor::Leaf::I(32), vec![0u8; 4 * count]),
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

pub struct Fixture {
    pub trace: Trace,
    pub contract: ModelContract,
    pub container: PathBuf,
}

/// The model's fixture, cut to `layers` where stated.
pub fn fixture(model: &str, layers: Option<u32>) -> Fixture {
    let package = poem_compiler::catalog::package_of(model).expect("the model is listed");
    let deploy = poem::star::Deploy {
        layers,
        ..poem_compiler::catalog::one_rank(Dtype::Bf16, Dtype::Bf16)
    };
    let trace = package
        .trace(model, &deploy, model, Platform::Wgpu)
        .unwrap_or_else(|why| panic!("`{model}` traces: {why:#}"));
    let dir = std::env::temp_dir().join(format!("pie-wgpu-pool-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("the fixture dir");
    let container = dir.join(format!("{model}-{}.zt", layers.unwrap_or(0)));
    if !container.exists() {
        write_checkpoint(&container, &trace);
    }
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Wgpu)
        .expect("the plan reads its own planes");
    Fixture {
        trace,
        contract,
        container,
    }
}

pub fn load(fixture: &Fixture, context: u32) -> Option<Shell> {
    load_at(fixture, context, None)
}

/// [`load`], the pool's recipe aiming at chunks of `chunk` bytes where
/// stated.
pub fn load_at(fixture: &Fixture, context: u32, chunk: Option<u64>) -> Option<Shell> {
    let device = engine_wgpu::api::DeviceBoot {
        chunk_bytes: chunk,
        ..engine_wgpu::api::DeviceBoot::default()
    };
    if let Err(why) = engine_wgpu::Context::bind(&device, 0.25) {
        eprintln!("skipping: no wgpu device ({why})");
        return None;
    }
    Some(
        Shell::load(Boot {
            trace: fixture.trace.clone(),
            contract: &fixture.contract,
            checkpoint: &fixture.container,
            budget: Budget::new(SEATS, 256),
            patches: None,
            voxels: None,
            profile: None,
            page_size: PAGE,
            context,
            slots: SEATS,
            runahead: engine::runahead::Runahead::F1,
            device: &device,
            residency: engine::load::Residency::default(),
            handed: None,
        })
        .unwrap_or_else(|why| panic!("the shell loads: {why}")),
    )
}

pub fn word(trace: &Trace, rows: u32) -> u64 {
    trace.facts.word(&Request::new(rows, false))
}

pub fn argmax(logits: &[f32]) -> u32 {
    (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap_or(0) as u32
}

/// Two prompts long enough to span pages.
pub fn prompts() -> [Vec<u32>; 2] {
    [
        (0..150u32).map(|at| (at * 37 + 11) % 256).collect(),
        (0..90u32).map(|at| (at * 53 + 5) % 256).collect(),
    ]
}

/// Prefills each lane alone (seat-addressed), then decodes both together
/// `steps` times. Every logit row, in order.
pub fn serve_seats(shell: &mut Shell, trace: &Trace, steps: usize) -> Vec<Vec<f32>> {
    let mut fed: [Vec<u32>; 2] = prompts();
    let mut held = [0u32; 2];
    let mut rows = Vec::new();
    for lane in 0..2usize {
        let seated = Seated {
            held: Some(held[lane]),
            ..Seated::of(Lane {
                slot: lane as u32,
                word: word(trace, fed[lane].len() as u32),
                tokens: &fed[lane],
            })
        };
        let got = shell.fire_seated(&[seated]).expect("the prefill lands");
        held[lane] += fed[lane].len() as u32;
        fed[lane] = vec![argmax(&got[0])];
        rows.extend(got);
    }
    for _ in 0..steps {
        let seated: Vec<Seated<'_>> = (0..2usize)
            .map(|lane| Seated {
                held: Some(held[lane]),
                ..Seated::of(Lane {
                    slot: lane as u32,
                    word: word(trace, 1),
                    tokens: &fed[lane],
                })
            })
            .collect();
        let got = shell.fire_seated(&seated).expect("the decode lands");
        for lane in 0..2 {
            held[lane] += 1;
            fed[lane] = vec![argmax(&got[lane])];
        }
        rows.extend(got);
    }
    rows
}

/// Writes `rows` to `$PIE_WGPU_LOGITS/<name>.bin` when that is set.
pub fn dump(name: &str, rows: &[Vec<f32>]) {
    let Some(dir) = std::env::var_os("PIE_WGPU_LOGITS") else {
        return;
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("the dump dir");
    let bytes: Vec<u8> = rows
        .iter()
        .flat_map(|row| row.iter().flat_map(|v| v.to_bits().to_le_bytes()))
        .collect();
    std::fs::write(dir.join(format!("{name}.bin")), bytes).expect("the dump writes");
}

fn seats(model: &str, layers: Option<u32>) {
    let fixture = fixture(model, layers);
    let Some(mut shell) = load(&fixture, CONTEXT) else {
        return;
    };
    let rows = serve_seats(&mut shell, &fixture.trace, 12);
    assert!(rows.iter().all(|row| row.iter().all(|v| v.is_finite())));
    dump(&format!("seats-{model}"), &rows);
}

#[test]
fn a_dense_miniature_serves_two_lanes() {
    seats("qwen3-micro-text", None);
}

#[test]
fn a_hybrid_miniature_serves_two_lanes() {
    seats("qwen35-tiny", None);
}

#[test]
fn a_windowed_miniature_serves_two_lanes() {
    seats("gemma4-e4b-mini-l1", None);
}

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
            "{what}: row {at} differs in {differ} of {} logits (argmax {} against {})",
            want.len(),
            argmax(got),
            argmax(want)
        );
    }
}

/// Slot refs the runtime would name: pages strided over chunks `first ..
/// first + 3` from the top down, piece `p` of lane `l`'s record in chunk
/// `first + 3 + l * pieces + p`.
struct Refs {
    first: u32,
    per_list: u32,
    per_fixed: u32,
    pieces: u32,
}

impl Refs {
    fn page(&self, lane: u32, page: u32) -> u32 {
        let chunk = self.first + 2 - (page % 3);
        let index = (7 * page + 3 + 61 * lane) % self.per_list;
        chunk * self.per_list + index
    }

    fn record(&self, lane: u32) -> Vec<u32> {
        (0..self.pieces)
            .map(|p| {
                (self.first + 3 + lane * self.pieces + p) * self.per_fixed
                    + (lane * 3 + 1) % self.per_fixed
            })
            .collect()
    }

    /// The chunks these refs hold: three of pages and two records'.
    fn held(&self) -> u32 {
        3 + 2 * self.pieces
    }
}

/// As [`serve_seats`], every lane addressed by slot refs; after
/// `evacuate_at` decodes, `between` runs (an evacuation).
fn serve_refs(
    shell: &mut Shell,
    trace: &Trace,
    refs: &Refs,
    steps: usize,
    evacuate_at: usize,
    between: &mut dyn FnMut(&mut Shell),
) -> Vec<Vec<f32>> {
    let mut held = [0u32; 2];
    let mut fed: [Vec<u32>; 2] = prompts();
    let page = shell.paging().page_size;
    let records = [refs.record(0), refs.record(1)];
    let mut rows = Vec::new();
    let table = |lane: usize, upto: u32| -> Vec<u32> {
        (0..upto.div_ceil(page))
            .map(|p| refs.page(lane as u32, p))
            .collect()
    };
    let fire = |shell: &mut Shell, lanes: &[(usize, &[u32], u32)]| -> Vec<Vec<f32>> {
        let tables: Vec<Vec<u32>> = lanes
            .iter()
            .map(|&(lane, tokens, have)| table(lane, have + tokens.len() as u32))
            .collect();
        let seated: Vec<Seated<'_>> = lanes
            .iter()
            .zip(&tables)
            .map(|(&(lane, tokens, have), pages)| Seated {
                pages,
                held: Some(have),
                refs: true,
                record: Some(&records[lane][..]),
                ..Seated::of(Lane {
                    slot: lane as u32,
                    word: word(trace, tokens.len() as u32),
                    tokens,
                })
            })
            .collect();
        shell.fire_seated(&seated).expect("the fire lands")
    };
    for lane in 0..2usize {
        let got = fire(shell, &[(lane, &fed[lane], held[lane])]);
        held[lane] += fed[lane].len() as u32;
        fed[lane] = vec![argmax(&got[0])];
        rows.extend(got);
    }
    for step in 0..steps {
        if step == evacuate_at {
            between(shell);
        }
        let got = fire(shell, &[(0, &fed[0], held[0]), (1, &fed[1], held[1])]);
        for lane in 0..2 {
            held[lane] += 1;
            fed[lane] = vec![argmax(&got[lane])];
        }
        rows.extend(got);
    }
    rows
}

/// A model served out of the chunk pool lands the same logits, bit for bit,
/// wherever its slots sit: in the seats' own runs, or scattered over chunks
/// the runtime names by slot refs in a segment of their own, those chunks
/// evacuated mid-decode below it and that segment given back, the decode
/// running on through the moved chunks.
fn scattered(model: &str) -> Vec<Vec<f32>> {
    scattered_at(model, None)
}

/// [`scattered`] over a pool whose recipe aims at `chunk`-byte chunks;
/// the seats' rows.
fn scattered_at(model: &str, chunk: Option<u64>) -> Vec<Vec<f32>> {
    const STEPS: usize = 12;
    let fixture = fixture(model, None);
    let Some(mut shell) = load_at(&fixture, CONTEXT, chunk) else {
        return Vec::new();
    };
    let seats = serve_seats(&mut shell, &fixture.trace, STEPS);
    drop(shell);

    let mut shell = load_at(&fixture, CONTEXT, chunk).expect("loaded before");
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
    let layout = shell.pool_layout();
    let segment = facts.segment_chunks;
    let low = 1 + layout.seat_chunks();
    let per = |id| facts.shape(id).map_or(1, |s| s.slots_per_chunk);
    let pieces = match facts.shape(FIXED).map(|s| s.kind) {
        Some(engine::chunk::SlotKind::Fixed { pieces }) => pieces,
        _ => 1,
    };
    let held = 3 + 2 * pieces;
    let kept = (low + held).div_ceil(segment);
    let first = kept * segment;
    let commit = (first + held).div_ceil(segment);
    assert!(
        facts.segments.max >= commit,
        "{model}: the pool spans {commit} segments, it has {}",
        facts.segments.max
    );
    let refs = Refs {
        first,
        per_list: per(LIST),
        per_fixed: per(FIXED),
        pieces,
    };
    assert_eq!(refs.held(), held);
    shell.commit_segments(commit).expect("the segments commit");
    let scattered = serve_refs(
        &mut shell,
        &fixture.trace,
        &refs,
        STEPS,
        STEPS / 2,
        &mut |shell| {
            let moves = (0..held).map(|at| (first + at, low + at)).collect();
            shell
                .evacuate(&Evacuation { moves })
                .expect("the evacuation lands");
            shell
                .release_segments(kept)
                .expect("the top segments go back");
            assert_eq!(shell.pool_chunk_facts().segments.committed, kept);
        },
    );
    same(
        &format!("{model}: scattered and evacuated slots against the seats' runs"),
        &scattered,
        &seats,
    );
    seats
}

#[test]
fn a_scattered_dense_pool_serves_the_same_logits() {
    scattered("qwen3-micro-text");
}

#[test]
fn a_scattered_hybrid_pool_serves_the_same_logits() {
    scattered("qwen35-tiny");
}

/// A model with a compressed kv pool: its compressor side planes, a cell a
/// token of every physical list page, follow the pages an evacuation moves.
/// Ignored: dsv4-flash-mini does not serve here, before the pool as after
/// (before it, its side planes overran a binding; now its index top-k
/// refuses the index pool's pitch).
#[test]
#[ignore = "dsv4-flash-mini does not serve on this backend"]
fn a_scattered_compressed_pool_serves_the_same_logits() {
    scattered("dsv4-flash-mini");
}

/// A record cut over several slots (a chunk smaller than it, as
/// `PIE_GOLDEN_CHUNK` forces on CUDA): the same logits, bit for bit, as the
/// record in one slot, its pieces scattered and evacuated as above.
#[test]
fn a_record_in_pieces_serves_the_same_logits() {
    let whole = scattered("qwen35-tiny");
    let chunk = std::env::var("PIE_WGPU_CHUNK").map_or(64 << 10, |bytes| {
        bytes.parse().expect("PIE_WGPU_CHUNK is bytes")
    });
    let pieces = scattered_at("qwen35-tiny", Some(chunk));
    same(
        "qwen35-tiny: a record in pieces against one whole",
        &pieces,
        &whole,
    );
}

#[test]
fn a_scattered_windowed_pool_serves_the_same_logits() {
    scattered("gemma4-e4b-mini-l1");
}
