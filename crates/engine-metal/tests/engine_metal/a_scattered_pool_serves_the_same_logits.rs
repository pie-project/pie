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
use engine::fire::{FoldLen, Lineage, WindowFold};
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
    for (name, shape, leaf, bytes) in seeded(trace) {
        writer
            .add(name.as_str(), shape, leaf, &bytes)
            .unwrap_or_else(|why| panic!("`{name}`: {why}"));
    }
    writer.finish().expect("the fixture finishes");
}

/// The same planes as a serving artifact stamped for `trace`'s Metal
/// deployment: the file a deployed model loads warm, out of its mapping.
pub(super) fn write_serving(path: &Path, trace: &Trace) {
    use checkpoint::file::emit::{self, Object};
    let planes = seeded(trace);
    let objects: Vec<Object<'_>> = planes
        .iter()
        .map(|(name, shape, leaf, bytes)| Object::leaf(name, shape.clone(), *leaf, bytes))
        .collect();
    emit::write(
        path,
        &checkpoint::serving::Stamp::of(trace.platform.backend(), &trace.name),
        &std::collections::BTreeMap::new(),
        16384,
        &objects,
        |object, plane, _| panic!("{object}/{plane} is not streamed here"),
    )
    .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

/// The values of the bf16 plane `name` of `count` elements the fixture
/// writes ([`seeded`]), as f32: what a host oracle reads for it.
pub(super) fn seeded_bf16(name: &str, count: usize) -> Vec<f32> {
    let norm = name.ends_with("norm");
    let mut seed = fnv(name);
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value = if norm { 1.0 + 0.05 * unit } else { 0.08 * unit };
            f32::from_bits(value.to_bits() & 0xffff_0000)
        })
        .collect()
}

/// `trace`'s checkpoint planes seeded by name, by name.
pub(super) fn seeded(trace: &Trace) -> Vec<(String, Vec<u64>, ztensor::Leaf, Vec<u8>)> {
    let mut out = Vec::new();
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    // A packed plane's scales and zero points are made with it when the
    // import quantizes it: the container holds none of their own.
    let made: std::collections::BTreeSet<String> = planes
        .iter()
        .filter(|param| packed(param.dtype))
        .flat_map(|param| {
            [
                poem::scales_name(&param.name),
                poem::biases_name(&param.name),
                poem::global_scales_name(&param.name),
                poem::input_scales_name(&param.name),
            ]
        })
        .collect();
    planes.retain(|param| !made.contains(&param.name));
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    for param in planes {
        // The shape before the import packs it: a packed plane's codes are
        // stored narrower than the values they hold.
        let shape = poem::Weight::of_plane(param).shape;
        let count = shape.iter().product::<u64>() as usize;
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
            // A packed plane as the bf16 it is quantized from.
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
            // A hash-routing table: each row names experts 0.. of its width.
            Dtype::I64 => {
                let width = param.shape.last().copied().unwrap_or(1).max(1);
                (
                    ztensor::Leaf::I64,
                    (0..count as u64)
                        .flat_map(|at| (at % width).to_le_bytes())
                        .collect(),
                )
            }
            other if packed(other) => (
                ztensor::Leaf::BF16,
                (0..count)
                    .flat_map(|_| ((value(next()).to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
            ),
            // Block scale exponents (near one) and byte planes, as served.
            Dtype::E8m0 => (
                ztensor::Leaf::E8M0,
                (0..count).map(|_| (118 + next() % 6) as u8).collect(),
            ),
            Dtype::U8 => (
                ztensor::Leaf::U8,
                (0..count).map(|_| next() as u8).collect(),
            ),
            other => panic!(
                "`{}` is {other:?}, not a dtype the fixture seeds",
                param.name
            ),
        };
        out.push((param.name.clone(), shape, leaf, bytes));
    }
    out
}

/// Whether a plane stored as `dtype` is packed codes the import quantizes
/// from bf16.
fn packed(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::Mxfp4
            | Dtype::Nvfp4
            | Dtype::E4m3tensor
            | Dtype::U4g64
            | Dtype::U8g64
            | Dtype::U4g32
            | Dtype::U4g64tiled
            | Dtype::U2g32
            | Dtype::U2g64
            | Dtype::U2g128
    )
}

/// The fixture of catalog deployment `name` (listed, or spelled): a
/// deployment with packed planes loads from a serving artifact alone, which
/// the test kit's import writes from the seeded floats (a load converts
/// nothing).
pub(super) fn deployed_fixture(name: &str) -> Fixture {
    let trace = poem_compiler::catalog::deployment(name)
        .cloned()
        .or_else(|| poem_compiler::catalog::repository().parse(name))
        .unwrap_or_else(|| panic!("no package spells `{name}`"))
        .trace(Platform::Metal);
    let encoded = trace
        .params
        .iter()
        .any(|param| param.source == ParamSource::Checkpoint && packed(param.dtype));
    if !encoded {
        return fixture_from(trace, name);
    }
    let (dir, owned) = fixture_dir();
    let floats = dir.join(format!("{name}.floats.zt"));
    if !floats.exists() {
        write_checkpoint(&floats, &trace);
    }
    let source = ztensor::Source::open(&floats).expect("the seeded floats open");
    let imported = poem::import::own_contract(&source, &trace.params, 1, Platform::Metal)
        .expect("the seeded floats are read by the plan's own names");
    drop(source);
    let container = dir.join(format!("{name}.serving.zt"));
    if !container.exists() {
        let tiled: Vec<&str> = trace
            .params
            .iter()
            .filter(|param| param.dtype == Dtype::U4g64tiled)
            .map(|param| param.name.as_str())
            .collect();
        checkpoint::testkit::artifact::import(
            &floats,
            &imported,
            &checkpoint::serving::Stamp::of(trace.platform.backend(), &trace.name),
            &tiled,
            &container,
        )
        .unwrap_or_else(|why| panic!("{name} imports: {why}"));
    }
    // The load reads the artifact's planes as stored.
    let source = ztensor::Source::open(&container).expect("the artifact opens");
    let contract = poem::import::own_contract(&source, &trace.params, 1, Platform::Metal)
        .expect("the artifact is read by the plan's own names");
    drop(source);
    Fixture {
        trace,
        contract,
        container,
        owned,
    }
}

/// Where a fixture's files go: `PIE_FIXTURE_DIR`, kept across processes (a
/// probe), or a directory of its own under the temp dir, which the fixture
/// removes when dropped (`owned`), passing or failing.
fn fixture_dir() -> (PathBuf, Option<PathBuf>) {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if let Some(kept) = std::env::var_os("PIE_FIXTURE_DIR") {
        let dir = PathBuf::from(kept);
        std::fs::create_dir_all(&dir).expect("the fixture dir");
        return (dir, None);
    }
    let dir = std::env::temp_dir().join(format!(
        "pie-bm-pool-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).expect("the fixture dir");
    (dir.clone(), Some(dir))
}

pub(super) struct Fixture {
    pub(super) trace: Trace,
    pub(super) contract: ModelContract,
    pub(super) container: PathBuf,
    /// The directory the fixture made for itself, removed with it.
    owned: Option<PathBuf>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Some(dir) = self.owned.take() {
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}

fn fixture(model: &str) -> Fixture {
    fixture_of(model, None)
}

/// The model's fixture, cut to `layers` where stated.
pub(super) fn fixture_of(model: &str, layers: Option<u32>) -> Fixture {
    let package = poem_compiler::catalog::package_of(model).expect("the model is listed");
    let deploy = poem::star::Deploy {
        layers,
        ..poem_compiler::catalog::one_rank(Dtype::Bf16, Dtype::Bf16)
    };
    let trace = package
        .trace(model, &deploy, model, Platform::Metal)
        .unwrap_or_else(|why| panic!("`{model}` traces: {why:#}"));
    fixture_from(trace, &format!("{model}-{}", layers.unwrap_or(0)))
}

/// The fixture of `trace`, its checkpoint written under `name`.
pub(super) fn fixture_from(trace: Trace, name: &str) -> Fixture {
    // `PIE_FIXTURE_DIR` keeps the fixtures across processes (a probe).
    let (dir, owned) = fixture_dir();
    let container = dir.join(format!("{name}.zt"));
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
        owned,
    }
}

pub(super) fn load_with(fixture: &Fixture, context: u32) -> Shell {
    load_near(fixture, context, None)
}

/// The shell, its pool's recipe aiming its chunk near `chunk` bytes where
/// stated.
pub(super) fn load_near(fixture: &Fixture, context: u32, chunk: Option<u64>) -> Shell {
    load_ahead(fixture, context, chunk, engine::runahead::Runahead::F1)
}

/// [`load_near`] keeping `runahead`'s frames in flight.
pub(super) fn load_ahead(
    fixture: &Fixture,
    context: u32,
    chunk: Option<u64>,
    runahead: engine::runahead::Runahead,
) -> Shell {
    load_boot(fixture, context, chunk, runahead, None)
}

/// [`load_near`] keeping what it measures in `profiles` and taking what an
/// earlier load stored there. The loads a test compares to the bit share
/// one: each load otherwise times its own kernel.dense curves and may fit
/// another ladder (FOLLOWUPS F57: a 40-row prompt on mpp in one load and on
/// tile in the other), so their logits agree within a tolerance, not to the
/// bit.
pub(super) fn load_stored(
    fixture: &Fixture,
    context: u32,
    chunk: Option<u64>,
    profiles: &Profiles,
) -> Shell {
    load_boot(
        fixture,
        context,
        chunk,
        engine::runahead::Runahead::F1,
        Some(profiles.0.clone()),
    )
}

/// A profile directory of one test's own, gone when dropped: the loads that
/// test compares share it ([`load_stored`]), and no other test reads what
/// they stored.
pub(super) struct Profiles(PathBuf);

impl Profiles {
    pub(super) fn new(test: &str) -> Profiles {
        let dir = std::env::temp_dir().join(format!("pie-{test}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Profiles(dir)
    }
}

impl Drop for Profiles {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn load_boot(
    fixture: &Fixture,
    context: u32,
    chunk: Option<u64>,
    runahead: engine::runahead::Runahead,
    profiles: Option<PathBuf>,
) -> Shell {
    let residency = engine_metal::ResidencyPlan::default();
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
        kv_budget: None,
        allowance: engine::Allowance::WHOLE,
        runahead,
        residency: match chunk {
            Some(bytes) => residency.with_chunk_bytes(bytes),
            None => residency,
        },
        profiles,
        top: engine_metal::Top::EveryRung,
        measure: engine::rates::Measuring::Time,
    })
    .and_then(|mut shell| shell.raise_whole().map(|()| shell))
    .unwrap_or_else(|why| panic!("the shell loads: {why}"))
}

pub(super) fn word(trace: &Trace, rows: u32) -> u64 {
    trace.facts.word(&Request::new(rows, false))
}

pub(super) fn argmax(logits: &[f32]) -> u32 {
    (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap_or(0) as u32
}

/// Where the script's two lanes keep their state.
enum Placement {
    /// Seat-addressed: each lane's seat runs.
    Seats,
    /// Slot refs the runtime would name: pages strided over chunks
    /// `first .. first + 3` from the top down, the lanes' pages interleaved
    /// in each and scattered over its slots (no two share one), piece `p`
    /// of lane `l`'s record in chunk `first + 3 + l * pieces + p`, and on a
    /// ring load lane `l`'s ring piece `p` in chunk
    /// `first + 3 + 2 * pieces + l * ring_pieces + p`.
    Refs {
        first: u32,
        /// The kv list's slots a chunk; `None` where the load states no kv
        /// list (every layer windowed), so no fire reads a page table.
        per_list: Option<u32>,
        per_fixed: u32,
        pieces: u32,
        /// The ring's pieces and the window buffer's rows a slot, where the
        /// load lays rings: each lane folds every fire through its own.
        ring: Option<(u32, u32)>,
    },
}

/// Page `page` of lane `lane` among the three chunks `base .. base + 3` of
/// `per` slots each: chunks from the top down, the lanes in turn in each,
/// at an odd stride coprime to `per`.
fn scatter(base: u32, per: u32, lane: u32, page: u32) -> u32 {
    let chunk = base + 2 - (page % 3);
    let nth = (page / 3) * 2 + lane;
    let stride = (7..)
        .step_by(2)
        .find(|&s| gcd(s, per) == 1)
        .expect("an odd stride coprime to the slots");
    let index = (stride * nth + 3) % per;
    chunk * per + index
}

impl Placement {
    fn page(&self, lane: u32, page: u32) -> u32 {
        match *self {
            Placement::Refs {
                first,
                per_list: Some(per),
                ..
            } => scatter(first, per, lane, page),
            _ => unreachable!("a page table of a placement with a kv list"),
        }
    }

    /// Lane `lane`'s ring pieces, `[run, cow]` chunk refs, where the load
    /// lays rings.
    fn ring(&self, lane: u32) -> Option<Vec<[u32; 2]>> {
        match *self {
            Placement::Refs {
                first,
                pieces,
                ring: Some((ring, _)),
                ..
            } => Some(
                (0..ring)
                    .map(|p| first + 3 + 2 * pieces + lane * ring + p)
                    .map(|chunk| [chunk, chunk])
                    .collect(),
            ),
            _ => None,
        }
    }

    /// How a lane of `have` rows folds a fire of `rows` through its ring:
    /// every row, its buffer based past them.
    fn fold(&self, have: u32, rows: u32) -> Option<WindowFold> {
        match *self {
            Placement::Refs {
                ring: Some((_, tokens)), ..
            } => Some(WindowFold {
                base: (have + rows) / tokens * tokens,
                fold: FoldLen::Host(rows),
                lineage: Lineage::Set,
            }),
            _ => None,
        }
    }

    /// The lane's kv page table and window table over its first `rows`.
    fn tables(&self, lane: u32, rows: u32, page: u32) -> (Vec<u32>, Vec<Vec<u32>>) {
        match *self {
            Placement::Seats => (Vec::new(), Vec::new()),
            Placement::Refs { per_list, .. } => (
                per_list.map_or_else(Vec::new, |_| {
                    (0..rows.div_ceil(page))
                        .map(|p| self.page(lane, p))
                        .collect()
                }),
                Vec::new(),
            ),
        }
    }

    fn record(&self, lane: u32) -> Vec<u32> {
        match *self {
            Placement::Seats => Vec::new(),
            Placement::Refs {
                first,
                per_fixed,
                pieces,
                ..
            } => (0..pieces)
                .map(|p| (first + 3 + lane * pieces + p) * per_fixed + (lane * 3 + 1) % per_fixed)
                .collect(),
        }
    }

    /// Every slot ref of both lanes' tables over `rows` rows is its own, in
    /// the kv list and in the window list.
    fn distinct(&self, rows: u32, page: u32) -> bool {
        let tables = [self.tables(0, rows, page), self.tables(1, rows, page)];
        let pages: Vec<u32> = tables.iter().flat_map(|t| t.0.clone()).collect();
        let windows: Vec<u32> = tables.iter().flat_map(|t| t.1.concat()).collect();
        let unique = |refs: &[u32]| {
            refs.iter().collect::<std::collections::BTreeSet<_>>().len() == refs.len()
        };
        unique(&pages) && unique(&windows)
    }

    /// The chunks this placement holds: three of pages, the records, and
    /// both lanes' rings where the load lays them.
    fn held(&self) -> u32 {
        match *self {
            Placement::Seats => 0,
            Placement::Refs { pieces, ring, .. } => 3 + 2 * pieces + 2 * ring.map_or(0, |(ring, _)| ring),
        }
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Two prompts long enough to span pages of the recipe's tokens.
fn prompts() -> [Vec<u32>; 2] {
    [
        (0..150u32).map(|at| (at * 37 + 11) % 256).collect(),
        (0..90u32).map(|at| (at * 53 + 5) % 256).collect(),
    ]
}

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
    let mut fed: [Vec<u32>; 2] = prompts();
    let page = shell.paging().page_size;
    assert!(
        !refs || place.distinct(CONTEXT, page),
        "the lanes' pages share no slot"
    );
    let records = [place.record(0), place.record(1)];
    let rings = [place.ring(0), place.ring(1)];
    let mut rows = Vec::new();
    let tables = |held: [u32; 2], rows: [u32; 2]| -> [(Vec<u32>, Vec<Vec<u32>>); 2] {
        [0, 1].map(|lane| place.tables(lane as u32, held[lane] + rows[lane], page))
    };
    type Fired<'t> = (u32, &'t [u32], u32, &'t (Vec<u32>, Vec<Vec<u32>>));
    let fire = |shell: &mut Shell, lanes: &[Fired<'_>]| -> Vec<Vec<f32>> {
        let seated: Vec<Seated<'_>> = lanes
            .iter()
            .map(|&(lane, tokens, have, (pages, windows))| Seated {
                replay_stated: None,
                pages,
                windows,
                ring: rings[lane as usize].as_deref(),
                window: place.fold(have, tokens.len() as u32),
                held: Some(have),
                record: refs.then(|| records[lane as usize].as_slice()),
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

/// The most a logit of one placement may differ from the same logit of
/// another, `ulps` bf16 units in the last place of the larger magnitude:
/// the lanes' rows meet the kernels in another order once their pages lie
/// elsewhere, and bf16 operands round each partial sum to 8 significant
/// bits, so the same row may land a few such units apart (no batch
/// invariance is promised; an argmax apart, or a logit past this, is a bug).
fn bf16_bound(a: f32, b: f32, ulps: f32) -> f32 {
    let magnitude = a.abs().max(b.abs()).max(f32::MIN_POSITIVE);
    // A bf16's unit in the last place: 2^(exponent - 7).
    ulps * 2f32.powi(magnitude.log2().floor() as i32 - 7)
}

/// Panics at the first row whose argmax differs, or a logit of which lies
/// past [`bf16_bound`] of two units from the other's.
pub(super) fn close(what: &str, got: &[Vec<f32>], want: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len(), "{what}: rows");
    for (at, (got, want)) in got.iter().zip(want).enumerate() {
        assert_eq!(got.len(), want.len(), "{what}: row {at}'s logits");
        assert_eq!(argmax(got), argmax(want), "{what}: row {at}'s argmax");
        let worst = got
            .iter()
            .zip(want)
            .map(|(&a, &b)| ((a - b).abs(), bf16_bound(a, b, 2.0), a, b))
            .filter(|&(apart, bound, ..)| !(apart <= bound))
            .collect::<Vec<_>>();
        assert!(
            worst.is_empty(),
            "{what}: row {at}: {} of {} logits past two bf16 units, first {:?}",
            worst.len(),
            want.len(),
            worst.first()
        );
    }
}

/// The bound passes two bf16 units of a logit and refuses three, and an
/// argmax apart is refused at once whatever the logits' distance.
#[test]
fn close_holds_two_bf16_units_and_the_argmax() {
    let unit = 2f32.powi(-7); // a bf16 unit at 1.0
    let row = vec![1.0f32, 0.5, -3.0];
    close("two units", &[vec![1.0 + 2.0 * unit, 0.5, -3.0]], &[row.clone()]);
    let three = std::panic::catch_unwind(|| {
        close("three units", &[vec![1.0 + 3.0 * unit, 0.5, -3.0]], &[vec![1.0f32, 0.5, -3.0]]);
    });
    assert!(three.is_err(), "three bf16 units apart is refused");
    let swapped = std::panic::catch_unwind(|| {
        close("argmax", &[vec![0.5f32, 0.5001, -3.0]], &[vec![0.5001f32, 0.5, -3.0]]);
    });
    assert!(swapped.is_err(), "an argmax apart is refused");
}

/// Panics at the first row whose logits differ in a bit.
pub(super) fn same(what: &str, got: &[Vec<f32>], want: &[Vec<f32>]) {
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
    check_near(model, hybrid, None, &Profiles::new(&format!("scattered-{model}")));
}

/// [`check`] over a pool whose chunk aims near `chunk` bytes where stated:
/// the seats' logits.
/// The seats' run and the scattered one are two loads compared to the bit:
/// both load in `profiles`, so they fire the same kernels (a caller that
/// compares the result with another `check_near` passes the same one).
fn check_near(model: &str, hybrid: bool, chunk: Option<u64>, profiles: &Profiles) -> Vec<Vec<f32>> {
    let fixture = fixture(model);
    const STEPS: usize = 12;
    let load = |fixture: &Fixture| load_stored(fixture, CONTEXT, chunk, profiles);

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
    let low = layout.free_chunk();
    let per = |id| facts.shape(id).map_or(1, |s| s.slots_per_chunk);
    let per_list = facts.shape(LIST).map(|s| s.slots_per_chunk);
    // A windowed model's rings: each lane's by slot ref too, so the
    // scattered serve moves every row the fire reads.
    let ring = shell
        .pool_ring()
        .map(|ring| (ring.pieces, layout.window(0).expect("a ring load's window buffer").tokens));
    let place = Placement::Refs {
        first: 0,
        per_list,
        per_fixed: per(FIXED),
        pieces: layout.pieces,
        ring,
    };
    let held = place.held();
    let kept = (low + held).div_ceil(segment);
    let first = kept * segment;
    let commit = kept + held.div_ceil(segment);
    assert!(
        facts.segments.max >= commit,
        "{model}: the pool spans {commit} segments"
    );
    let place = Placement::Refs {
        first,
        per_list,
        per_fixed: per(FIXED),
        pieces: layout.pieces,
        ring,
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
            let moves = (0..held).map(|at| (first + at, low + at)).collect();
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
    close(
        &format!("{model}: scattered and evacuated slots against the seats' runs"),
        &scattered,
        &seats,
    );
    seats
}

/// An evacuation refused past its copies and its rebind (a holder whose refs
/// the moved table cannot resolve, beamx40 on d821e8d1c) is taken back whole:
/// the refusal answered, the chunk table as it stood, and the lanes decode on
/// at the refs they lay in, the seats' logits bit for bit. Before, the copies
/// and the claims stood moved while the refusal told the runtime nothing
/// moved, and the two maps parted.
#[test]
fn a_refused_evacuation_moves_nothing() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    for model in ["qwen3-micro-text", "qwen35-tiny"] {
        let fixture = fixture(model);
        const STEPS: usize = 8;
        let profiles = Profiles::new(&format!("refused-evacuation-{model}"));
        let mut shell = load_stored(&fixture, CONTEXT, None, &profiles);
        let seats = serve(&mut shell, &fixture.trace, &Placement::Seats, STEPS, usize::MAX, &mut |_| {});
        drop(shell);

        let mut shell = load_stored(&fixture, CONTEXT, None, &profiles);
        let facts = shell.pool_chunk_facts();
        let layout = shell.pool_layout();
        let segment = facts.segment_chunks;
        let low = layout.free_chunk();
        let per = |id| facts.shape(id).map_or(1, |s| s.slots_per_chunk);
        let place = |first| Placement::Refs {
            first,
            per_list: facts.shape(LIST).map(|s| s.slots_per_chunk),
            per_fixed: per(FIXED),
            pieces: layout.pieces,
            ring: None,
        };
        let held = place(0).held();
        let kept = (low + held).div_ceil(segment);
        let first = kept * segment;
        shell
            .commit_segments(kept + held.div_ceil(segment))
            .expect("the segments commit");
        let scattered = serve(&mut shell, &fixture.trace, &place(first), STEPS, STEPS / 2, &mut |shell| {
            let before = shell.chunk_table().clone();
            let moves = (0..held).map(|at| (first + at, low + at)).collect();
            engine_metal::serve::fail_next_repoint();
            let refused = shell.evacuate(&Evacuation { moves });
            assert!(refused.is_err(), "{model}: the injected refusal is answered");
            assert_eq!(shell.chunk_table(), &before, "{model}: a refused evacuation moves nothing");
        });
        close(
            &format!("{model}: decoding on past a refused evacuation, against the seats' runs"),
            &scattered,
            &seats,
        );
    }
}

#[test]
fn a_scattered_dense_pool_serves_the_same_logits() {
    check("qwen3-micro-text", false);
}

#[test]
fn a_scattered_windowed_pool_serves_the_same_logits() {
    check("gemma4-e4b-mini-l1", false);
}

#[test]
fn a_scattered_hybrid_pool_serves_the_same_logits() {
    check("qwen35-tiny", true);
}

/// Caps each buffer of a pool loaded while it lives at one segment, so a
/// small model's range spans several buffers; lifted when dropped.
struct OverBuffers;

impl OverBuffers {
    fn cap() -> OverBuffers {
        engine_metal::store::pool::cap_parts(1);
        OverBuffers
    }
}

impl Drop for OverBuffers {
    fn drop(&mut self) {
        engine_metal::store::pool::cap_parts(0);
    }
}

/// The range over several buffers, a segment each: kv pages, records and
/// the activation span in buffers past the first, read through each
/// buffer's place from the first, give the seats' logits.
#[test]
fn a_pool_over_buffers_serves_the_same_logits() {
    if !engine_metal::device::present() {
        return;
    }
    let _over = OverBuffers::cap();
    check("qwen3-micro-text", false);
    check("qwen35-tiny", true);
}

/// A page id names its slot by its chunk in its buffer and its index in
/// that chunk where a paged shape's slots do not tile a chunk: a window list
/// and an rs list at such a chunk, their pages in buffers past the first,
/// serve the seats' logits.
#[test]
fn split_page_ids_over_buffers_serve_the_same_logits() {
    if !engine_metal::device::present() {
        return;
    }
    let _over = OverBuffers::cap();
    for (model, hybrid) in [("gemma4-e4b-mini-l1", false), ("qwen35-tiny", true)] {
        let fixture = fixture(model);
        // The scattered placement lays each lane's pages over three
        // chunks: a chunk holds a dozen list slots at least.
        let chunk = [3u64 << 19, 5 << 19, 7 << 19, 9 << 19, 11 << 19, 13 << 19]
            .into_iter()
            .find(|&chunk| {
                let shell = load_near(&fixture, CONTEXT, Some(chunk));
                let (ids, parts) = shell.page_ids();
                let list = shell.pool_layout().list.map_or(u32::MAX, |l| l.per_chunk);
                ids.shift.is_some() && parts > 1 && list >= 12
            })
            .unwrap_or_else(|| panic!("{model}: a chunk its paged slots do not tile"));
        let shell = load_near(&fixture, CONTEXT, Some(chunk));
        let (ids, parts) = shell.page_ids();
        eprintln!(
            "{model}: chunk {} B over {parts} buffers, a page id's index {} bits",
            ids.chunk_bytes,
            ids.shift.unwrap_or(0)
        );
        drop(shell);
        check_near(model, hybrid, Some(chunk), &Profiles::new(&format!("split-{model}")));
    }
}

/// Where every paged shape's slots tile the chunk, a page id is its slot in
/// its buffer (`kv_page.h` compiles `page * stride`, with the buffer's place
/// where there are several); where one does not, its chunk and index. The
/// profile's own chunk on each fixture here.
#[test]
fn page_ids_split_exactly_where_a_paged_shape_does_not_tile() {
    if !engine_metal::device::present() {
        return;
    }
    for model in ["qwen3-micro-text", "gemma4-e4b-mini-l1", "qwen35-tiny"] {
        let shell = load_near(&fixture(model), CONTEXT, None);
        let (ids, parts) = shell.page_ids();
        let layout = shell.pool_layout();
        let tiles = layout
            .list
            .iter()
            .chain(layout.window(0).map(|w| &w.slots))
            .chain(layout.rs.iter())
            .all(|slots| layout.chunk_bytes.is_multiple_of(slots.slot_bytes));
        eprintln!("{model}: {parts} buffer(s), slots tile the chunk: {tiles}, {ids:?}");
        assert_eq!(
            ids.shift.is_none(),
            tiles,
            "{model}: split exactly where a shape does not tile"
        );
    }
}

/// A gated-delta hybrid's record cut into pieces on device: its chunk aimed
/// at 256 KiB, the 1.6 MiB record lies in pieces of their own, each piece of
/// each lane in a chunk of its own once scattered by slot ref (and evacuated
/// mid-decode); the logits are the same bits as the seats' runs, and as the
/// whole record's under the profile's own chunk.
#[test]
fn a_record_in_pieces_serves_the_whole_records_logits() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let model = "qwen35-tiny";
    const CHUNK: u64 = 256 << 10;
    let pieces = load_near(&fixture(model), CONTEXT, Some(CHUNK))
        .pool_layout()
        .pieces;
    assert!(
        pieces > 1,
        "{model}'s record is whole in a {CHUNK}-byte chunk"
    );
    eprintln!("{model}: the record in {pieces} pieces");
    let profiles = Profiles::new("record-in-pieces");
    let cut = check_near(model, true, Some(CHUNK), &profiles);
    let whole = check_near(model, true, None, &profiles);
    same(
        &format!("{model}: the record in {pieces} pieces against whole"),
        &cut,
        &whole,
    );
}

/// The pools a GDN hybrid's catalog trace lays on this device, no weights
/// landed: one chunk space with a list, a record and an rs buffer shape, none
/// pinned; segments commit from the bottom and go back.
#[test]
#[allow(unsafe_code)]
fn a_reserved_hybrid_pool_states_shared_chunks() {
    use engine_metal::store::layout::Layout;
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
    let budgets = poem_compiler::Budgets::of(Budget::new(SEATS, 64));
    let recipe = engine_metal::chunks::recipe(&trace, &budgets).expect("a recipe");
    let rs = engine_metal::rs::Layout::read(&trace).expect("the rs layout");
    let rows = Rows::plan(&trace, &recipe, &facts, rs.as_ref())
        .unwrap_or_else(|why| panic!("{name}: the recipe and this shell disagree: {why}"));
    let chunk = rows.chunk_bytes;
    let segment = engine_metal::store::pool::segment_chunks(chunk);
    let layout = Layout::new(
        chunk,
        3 * segment,
        segment,
        rows.tokens,
        [rows.list_bytes, rows.fixed_bytes],
        rows.pieces,
        SEATS,
        4,
    )
    .expect("a layout")
    // The state maps' run of the engine's tables, as a load sizes it.
    .with_tables(
        u32::try_from(Pools::map_bytes(rows.state_rows(), 4).div_ceil(chunk))
            .expect("a chunk count"),
    );
    let paging = engine_metal::store::kv::Paging::of(rows.tokens, 256, SEATS, 1).expect("a paging");
    let bands = rows.bands.clone();
    let mut pools = Pools::reserve(&device, rows, layout, paging).expect("a pool");
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
    eprintln!("{name}: bands {bands:?}");
    assert_eq!(
        pools.bands().iter().sum::<u32>() as usize,
        bands.iter().map(Vec::len).sum::<usize>()
    );
    for id in [LIST, FIXED] {
        let shape = facts.shape(id).expect("the shape stands");
        assert!(shape.pinned.is_none());
        assert!(shape.slot_bytes * u64::from(shape.slots_per_chunk) <= facts.chunk_bytes);
    }
    assert_eq!(facts.chunk_bytes % facts.shape(LIST).unwrap().slot_bytes, 0);
    assert_eq!(facts.segments.max, 3);
    assert!(
        facts.act_chunks.is_empty(),
        "a layout with no activation span states none"
    );
    if facts.segments.growth != Growth::Bottom {
        assert_eq!(facts.segments.committed, 3);
        return;
    }
    // The engine's tables run is backed from the load: its segments, and
    // no other.
    assert_eq!(
        facts.segments.committed,
        (1 + pools.layout().tables).div_ceil(segment)
    );
    pools.commit_segments(2).expect("two segments");
    assert_eq!(pools.chunk_facts().segments.committed, 2);
    assert!(pools.committed_bytes() >= 2 * u64::from(segment) * chunk);
    // SAFETY: nothing ran on this pool.
    unsafe { pools.release_segments(1) }.expect("one goes back");
    assert_eq!(pools.chunk_facts().segments.committed, 1);
}
