//! Whole miniature models served by the shell, end to end: two lanes prefill
//! alone and then decode together, every logit row kept. The rows are
//! written bit for bit to `$PIE_VK_LOGITS/<model>.bin` (`$PIE_WGPU_LOGITS`
//! on wgpu) when that is set, so a build can be held against another one
//! (the chunk pool against the per-row slabs it replaced) byte for byte.

use std::path::{Path, PathBuf};

use crate::backend::{self, Boot, Context, DeviceBoot, Lane, Seated, Shell};
use checkpoint::contract::ModelContract;
use engine::chunk::Evacuation;
use engine_gpu::store::layout::{FIXED, LIST};
use poem::{Dtype, Request};
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

/// The packed formats: codes the import makes, with their scales (and zero
/// points), from the floats a checkpoint stores.
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

/// The values of the bf16 plane `name` of `count` elements the fixture
/// writes ([`write_checkpoint`]), as f32: what a host oracle reads for it.
pub fn seeded_bf16(name: &str, count: usize) -> Vec<f32> {
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

/// A checkpoint of `trace`'s own planes seeded by name: norms near one, the
/// rest small. A packed plane is stored as the bf16 it is quantized from, at
/// its unpacked shape (its scales are made by the import, so none is
/// stored); a plain fp8, block-exponent or byte plane as the bytes served.
fn write_checkpoint(path: &Path, trace: &Trace) {
    let mut writer = ztensor::Writer::create(path).expect("the fixture writes");
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
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
            // e4m3 codes of a small magnitude, each finite.
            Dtype::E4m3 => (
                ztensor::Leaf::E4M3,
                (0..count)
                    .map(|_| {
                        let bits = next();
                        ((bits & 0x80) | (0x18 + (bits >> 8) % 0x28)) as u8
                    })
                    .collect(),
            ),
            // Block exponents near one.
            Dtype::E8m0 => (
                ztensor::Leaf::E8M0,
                (0..count).map(|_| (118 + next() % 6) as u8).collect(),
            ),
            Dtype::U8 => (
                ztensor::Leaf::U8,
                (0..count).map(|_| next() as u8).collect(),
            ),
            dtype if packed(dtype) => (
                ztensor::Leaf::BF16,
                (0..count)
                    .flat_map(|_| ((value(next()).to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
            ),
            other => panic!(
                "`{}` is {other:?}, not a dtype the fixture seeds",
                param.name
            ),
        };
        writer
            .add(param.name.as_str(), shape, leaf, &bytes)
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
    fixture_in(package, model, layers, "")
}

/// [`fixture`] traced out of `package`, its checkpoint file named apart by
/// `variant`.
pub fn fixture_in(
    package: &poem::star::Package,
    model: &str,
    layers: Option<u32>,
    variant: &str,
) -> Fixture {
    let deploy = poem::star::Deploy {
        layers,
        ..poem_compiler::catalog::one_rank(Dtype::Bf16, Dtype::Bf16)
    };
    fixture_deployed(package, model, &deploy, variant)
}

/// [`fixture_in`] under `deploy`. A deployment serving a plane in another
/// form than the floats it is stored as (packed codes, fp8) loads from a
/// serving artifact only (a load converts nothing): the seeded floats are
/// imported into one (`checkpoint::testkit::artifact::import`) for this
/// platform.
pub fn fixture_deployed(
    package: &poem::star::Package,
    model: &str,
    deploy: &poem::star::Deploy,
    variant: &str,
) -> Fixture {
    let trace = package
        .trace(model, deploy, model, backend::PLATFORM)
        .unwrap_or_else(|why| panic!("`{model}` traces: {why:#}"));
    // The written checkpoints are the same bits every time (written from the
    // trace alone), so every run shares one directory: a run per process
    // used to leave its own copy behind (gigabytes each), and per-test runs
    // filled the disk. `$PIE_VK_FIXTURES` (`$PIE_WGPU_FIXTURES`) names
    // another.
    let dir = std::env::var_os(backend::FIXTURES).map_or_else(
        || std::env::temp_dir().join(format!("pie-{}-fixtures", backend::NAME)),
        PathBuf::from,
    );
    std::fs::create_dir_all(&dir).expect("the fixture dir");
    let encoded = trace.params.iter().any(|param| {
        param.source == ParamSource::Checkpoint
            && !matches!(
                param.dtype,
                Dtype::Bf16 | Dtype::F32 | Dtype::I64 | Dtype::I32
            )
    });
    let stem = format!("{model}{variant}-{}", deploy.layers.unwrap_or(0));
    let container = dir.join(if encoded {
        format!("{stem}.serving.zt")
    } else {
        format!("{stem}.zt")
    });
    if !container.exists() {
        // Written aside and renamed in: tests running at once may each
        // write it, and none opens a file half written.
        let aside = dir.join(format!(
            "{stem}.{}.{:?}.part",
            std::process::id(),
            std::thread::current().id()
        ));
        if encoded {
            let floats = dir.join(format!("{stem}.floats.zt"));
            if !floats.exists() {
                let seeded = dir.join(format!(
                    "{stem}.floats.{}.{:?}.part",
                    std::process::id(),
                    std::thread::current().id()
                ));
                write_checkpoint(&seeded, &trace);
                std::fs::rename(&seeded, &floats).expect("the floats land");
            }
            let source = ztensor::Source::open(&floats).expect("the floats open");
            let contract = poem::import::own_contract(&source, &trace.params, 1, backend::PLATFORM)
                .expect("the floats are read by the plan's own names");
            drop(source);
            let tiled: Vec<&str> = trace
                .params
                .iter()
                .filter(|param| param.dtype == Dtype::U4g64tiled)
                .map(|param| param.name.as_str())
                .collect();
            checkpoint::testkit::artifact::import(
                &floats,
                &contract,
                &checkpoint::serving::Stamp::of(backend::NAME, model),
                &tiled,
                &aside,
            )
            .unwrap_or_else(|why| panic!("`{model}` imports: {why}"));
        } else {
            write_checkpoint(&aside, &trace);
        }
        std::fs::rename(&aside, &container).expect("the fixture lands");
    }
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract = poem::import::own_contract(&source, &trace.params, 1, backend::PLATFORM)
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
    load_paged(fixture, PAGE, context, chunk)
}

/// [`load_at`] with kv pages of `page` tokens.
pub fn load_paged(fixture: &Fixture, page: u32, context: u32, chunk: Option<u64>) -> Option<Shell> {
    load_with(
        fixture,
        page,
        context,
        chunk,
        engine::load::Residency::default(),
    )
}

/// Held over a load: a driver reports what it holds a process, so a load
/// prices what its pipelines take while no other test's load allocates.
/// Taken after the test's share of the device (`gpu::share`), which a test
/// holding this never waits on.
static LOADING: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn loading() -> std::sync::MutexGuard<'static, ()> {
    crate::gpu::share();
    LOADING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// [`load_paged`] for fires of up to `lanes` lanes and `tokens` tokens, a
/// seat a lane.
pub fn load_lanes(
    fixture: &Fixture,
    page: u32,
    context: u32,
    lanes: u32,
    tokens: u32,
) -> Option<Shell> {
    let _alone = loading();
    let device = crate::gpu::device_boot();
    let residency = allowed(&device, engine::load::Residency::default())?;
    let boot = Boot {
        budget: Budget::new(lanes, tokens),
        slots: lanes,
        ..boot(fixture, page, context, &device, residency)
    };
    Some(Shell::load(boot).unwrap_or_else(|why| panic!("the shell loads: {why}")))
}

/// [`load_paged`] under `residency` (a host tier's budget, say).
pub fn load_with(
    fixture: &Fixture,
    page: u32,
    context: u32,
    chunk: Option<u64>,
    residency: engine::load::Residency,
) -> Option<Shell> {
    let _alone = loading();
    load_alone(fixture, page, context, chunk, residency)
}

/// [`load`] as the engine raises a served load: one whose programs it
/// places once the shell is raised, so the shell fires nothing before
/// (`Boot::places_programs`).
pub fn load_served(fixture: &Fixture, context: u32) -> Option<Shell> {
    let _alone = loading();
    let device = crate::gpu::device_boot();
    let residency = allowed(&device, engine::load::Residency::default())?;
    Some(
        Shell::load(Boot {
            places_programs: true,
            ..boot(fixture, PAGE, context, &device, residency)
        })
        .unwrap_or_else(|why| panic!("the shell loads: {why}")),
    )
}

/// [`load_with`], [`LOADING`] held by the caller.
fn load_alone(
    fixture: &Fixture,
    page: u32,
    context: u32,
    chunk: Option<u64>,
    residency: engine::load::Residency,
) -> Option<Shell> {
    let device = DeviceBoot {
        chunk_bytes: chunk,
        ..crate::gpu::device_boot()
    };
    let residency = allowed(&device, residency)?;
    Some(
        Shell::load(boot(fixture, page, context, &device, residency))
            .unwrap_or_else(|why| panic!("the shell loads: {why}")),
    )
}

/// [`load`] on a device whose largest buffer is `max_buffer` bytes, as a
/// device that binds less does, its pool `kv_budget` bytes in chunks of
/// `chunk` (the kernels bind at most four segments of it, none past that
/// buffer).
pub fn load_capped(
    fixture: &Fixture,
    context: u32,
    max_buffer: u64,
    kv_budget: u64,
    chunk: u64,
) -> Option<Shell> {
    let _alone = loading();
    let device = DeviceBoot {
        max_buffer: Some(max_buffer),
        chunk_bytes: Some(chunk),
        ..crate::gpu::device_boot()
    };
    let residency = allowed(
        &device,
        engine::load::Residency {
            device_kv_budget: Some(kv_budget),
            ..engine::load::Residency::default()
        },
    )?;
    Some(
        Shell::load(boot(fixture, PAGE, context, &device, residency))
            .unwrap_or_else(|why| panic!("the shell loads: {why}")),
    )
}

/// The boot of a suite load of `fixture` on `device`.
fn boot<'a>(
    fixture: &'a Fixture,
    page: u32,
    context: u32,
    device: &'a DeviceBoot,
    residency: engine::load::Residency,
) -> Boot<'a> {
    Boot {
        trace: fixture.trace.clone(),
        contract: &fixture.contract,
        checkpoint: &fixture.container,
        budget: Budget::new(SEATS, 256),
        patches: None,
        voxels: None,
        profile: None,
        page_size: page,
        context,
        slots: SEATS,
        runahead: engine::runahead::Runahead::F1,
        device,
        residency,
        handed: None,
        check: None,
        places_programs: false,
    }
}

/// `residency` with the device's budget the test's share of the device
/// (`gpu::share`) where it states none; `None` where no device comes up.
fn allowed(
    device: &DeviceBoot,
    residency: engine::load::Residency,
) -> Option<engine::load::Residency> {
    // The device is shared: a driver that fails to come up is asked again
    // before the test is skipped.
    let mut tries = 0;
    while let Err(why) = Context::bind(device, backend::ALLOWANCE) {
        tries += 1;
        if tries == 5 {
            eprintln!("skipping: no {} device ({why})", backend::NAME);
            return None;
        }
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    let share = crate::gpu::share();
    Some(engine::load::Residency {
        device: engine::Allowance {
            budget: residency.device.budget.or(share.budget),
        },
        ..residency
    })
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

/// Writes `rows` to `$PIE_VK_LOGITS/<name>.bin` (`$PIE_WGPU_LOGITS`) when
/// that is set.
pub fn dump(name: &str, rows: &[Vec<f32>]) {
    let Some(dir) = std::env::var_os(backend::LOGITS) else {
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

/// The state chunks `shell`'s pool holds below its activation span.
fn state_chunks(shell: &Shell) -> u32 {
    shell.pools().layout().total_chunks - shell.pools().span_chunks()
}

/// Every seat `shell` lays has its runs below the activation span: a
/// seat-addressed fire of its last seat names no chunk the pool does not
/// hold for state.
fn seats_lie_in_state(shell: &Shell) {
    let layout = *shell.pools().layout();
    let state = state_chunks(shell);
    assert!(
        1 + layout.seat_chunks() <= state,
        "{} seats' runs take {} chunks and the pool holds {state} for state",
        shell.paging().slots,
        1 + layout.seat_chunks()
    );
    for shape in layout.shapes().filter(|shape| shape.seat_slots > 0) {
        let last = shape
            .seat_ref(shape.seat_slots - 1)
            .expect("the last seat's ref");
        assert!(
            shape.split(last).0 < state,
            "shape {} names chunk {} past the {state} state chunks",
            shape.id,
            shape.split(last).0
        );
    }
}

/// A hybrid load lays the seats its widest seat-addressed fire asks for,
/// within its pool: a fire of every seat at once, the last among them,
/// serves. Asked for more seats than its pool holds the runs of, the load
/// is refused by name, never laid with a seat's record or rs run past the
/// state chunks (where a fire of it names a chunk the pool does not hold,
/// or the activation span's).
#[test]
fn a_hybrid_load_lays_the_seats_its_widest_seat_addressed_fire_asks_for() {
    let fixture = fixture("qwen35-tiny", None);
    let Some(mut shell) = load(&fixture, CONTEXT) else {
        return;
    };
    assert_eq!(shell.paging().slots, SEATS);
    seats_lie_in_state(&shell);
    // Every seat at once, as many rows each as the fire's budget shares.
    let [prompt, _] = prompts();
    let rows = (shell.budget().max_tokens / SEATS).min(prompt.len() as u32);
    let prompt = &prompt[..rows as usize];
    let seated: Vec<Seated<'_>> = (0..SEATS)
        .map(|seat| Seated {
            held: Some(0),
            ..Seated::of(Lane {
                slot: seat,
                word: word(&fixture.trace, rows),
                tokens: prompt,
            })
        })
        .collect();
    let got = shell
        .fire_seated(&seated)
        .unwrap_or_else(|why| panic!("a fire of all {SEATS} seats lands: {why}"));
    assert_eq!(got.len(), SEATS as usize);
    assert!(got.iter().all(|row| row.iter().all(|v| v.is_finite())));
    let laid = *shell.pools().layout();
    drop(shell);

    // A pool holding the runs of `SEATS` seats, asked for more seats than
    // its list run alone holds.
    let list = laid.list.expect("the hybrid pages its attention rows");
    let asked = SEATS * (2 + laid.seat_chunks()) * list.per_chunk;
    let _alone = loading();
    let device = crate::gpu::device_boot();
    let Some(residency) = allowed(
        &device,
        engine::load::Residency {
            device_kv_budget: Some(u64::from(1 + laid.seat_chunks()) * laid.chunk_bytes),
            ..engine::load::Residency::default()
        },
    ) else {
        return;
    };
    let refused = match Shell::load(Boot {
        slots: asked,
        ..boot(&fixture, PAGE, CONTEXT, &device, residency)
    }) {
        Ok(shell) => panic!(
            "{asked} seats laid over {} state chunks",
            state_chunks(&shell)
        ),
        Err(why) => why.to_string(),
    };
    assert!(
        refused.contains("seats' runs"),
        "refused for another reason: {refused}"
    );
}

/// A served hybrid load (no seats stated, its lanes naming their slots by
/// ref) lays the seats its own naming walk fires, whatever its budget: a
/// pool holding a few seats' runs loads, and its seats lie in it. Asked
/// for a fixed count of seats, a served load at such a budget laid runs
/// past its pool and was refused at its naming walk ("this fire wants ..
/// chunks a slot ref may name").
#[test]
fn a_served_hybrid_load_lays_the_seats_its_own_fires_take() {
    use engine::Engine;
    use engine::load::{Budgets, Checkpoint, LoadRequest};
    fn contract_for(
        trace: &Trace,
        path: &Path,
    ) -> Result<checkpoint::contract::ModelContract, String> {
        let source = ztensor::Source::open(path).map_err(|why| why.to_string())?;
        poem::import::own_contract(&source, &trace.params, 1, backend::PLATFORM)
            .map_err(|why| why.to_string())
    }
    let fixture = fixture("qwen35-tiny", None);
    let Some(shell) = load(&fixture, CONTEXT) else {
        return;
    };
    // The runs of `SEATS` seats: more than the naming walk's.
    let laid = *shell.pools().layout();
    drop(shell);
    let device = crate::gpu::device_boot();
    let Some(residency) = allowed(
        &device,
        engine::load::Residency {
            device_kv_budget: Some(u64::from(1 + laid.seat_chunks()) * laid.chunk_bytes),
            ..engine::load::Residency::default()
        },
    ) else {
        return;
    };
    let _alone = loading();
    let mut engine = backend::Engine::new(device, contract_for);
    engine
        .load(LoadRequest {
            trace: fixture.trace.clone(),
            checkpoint: Checkpoint::Path(fixture.container.clone()),
            budgets: Budgets {
                max_lanes: SEATS,
                max_tokens: CONTEXT,
                buckets: vec![CONTEXT],
                max_adapters: 0,
                page_size: PAGE,
                max_context: CONTEXT,
                context_declared: false,
                slots: 0,
                max_patches: None,
                max_images: None,
                max_voxels: None,
                max_clips: None,
            },
            residency,
            ordinal: 0,
            frames_in_flight: 1,
        })
        .unwrap_or_else(|why| panic!("the served load lays the seats its pool holds: {why}"));
    let shell = engine.loaded().expect("the shell stands");
    assert!(shell.paging().slots < SEATS);
    seats_lie_in_state(shell);
}

/// Fires `model` at every rung its load admits (one lane of the rung's
/// rows, every seat sharing them, then a decode of those seats) and holds
/// each fire to the pipelines the load built: its walks named every kernel
/// a fire asks for, so none is built while serving.
fn every_rung_fires_what_the_load_built(model: &str) {
    let fixture = fixture(model, None);
    let Some(mut shell) = load(&fixture, CONTEXT) else {
        return;
    };
    let built = shell.compiled();
    let named = shell.named().clone();
    assert_eq!(
        built,
        named.kernels.len() as u64,
        "the load built other pipelines than the ones its walks named"
    );
    let reserved = shell.device().books().reserved();
    let rungs: Vec<u32> = shell
        .pool_chunk_facts()
        .act_chunks
        .iter()
        .map(|&(rung, _)| rung)
        .collect();
    assert!(!rungs.is_empty(), "the load admits a rung");
    let tokens: Vec<u32> = (0..CONTEXT).map(|at| (at * 37 + 11) % 256).collect();
    let mut fires = 0;
    for &rung in &rungs {
        let rung = rung.clamp(1, CONTEXT);
        for lanes in [1, SEATS.min(rung)] {
            let each = (rung / lanes) as usize;
            for (held, rows) in [(0, each), (each as u32, 1)] {
                if held + rows as u32 > CONTEXT {
                    continue;
                }
                let seated: Vec<Seated<'_>> = (0..lanes)
                    .map(|lane| Seated {
                        held: Some(held),
                        ..Seated::of(Lane {
                            slot: lane,
                            word: word(&fixture.trace, rows as u32),
                            tokens: &tokens[..rows],
                        })
                    })
                    .collect();
                let got = shell.fire_seated(&seated).unwrap_or_else(|why| {
                    panic!("{lanes} lane(s) of {rows} rows at rung {rung} fire: {why}")
                });
                assert!(got.iter().all(|row| row.iter().all(|v| v.is_finite())));
                fires += 1;
                assert_eq!(
                    shell.compiled(),
                    built,
                    "{lanes} lane(s) of {rows} rows at rung {rung} built a pipeline the load's \
                     walks did not name"
                );
            }
        }
    }
    eprintln!(
        "{model}: {} kernels named and built, {} dispatches a fire at most, rungs {rungs:?}, \
         {fires} fires, {} bytes reserved at load and {} after",
        named.kernels.len(),
        named.dispatches,
        reserved,
        shell.device().books().reserved()
    );
}

/// A load names its kernels and its dispatch bound from fires at its
/// largest rung alone; the fires at each other rung alone name the same
/// kernels, the same bound and the same classes the kernels refuse.
fn every_rung_names_what_the_largest_names(model: &str) {
    let fixture = fixture(model, None);
    let device = crate::gpu::device_boot();
    let _alone = loading();
    let Some(residency) = allowed(&device, engine::load::Residency::default()) else {
        return;
    };
    let at = |rung: Option<u32>| {
        Shell::named_at(
            boot(&fixture, PAGE, CONTEXT, &device, residency.clone()),
            rung,
        )
    };
    // A load this device refuses names nothing; it is refused in the words
    // the load is, and said so.
    let largest = match at(None) {
        Ok(largest) => largest,
        Err(why) => {
            let load = Shell::load(boot(&fixture, PAGE, CONTEXT, &device, residency.clone()));
            match load {
                Err(refused) if refused.to_string() == why.to_string() => {
                    eprintln!("{model}: not named, as the load is refused: {why}");
                    return;
                }
                _ => panic!("{model} names nothing, and its load is not refused so: {why}"),
            }
        }
    };
    let named = |rung: Option<u32>| {
        at(rung).unwrap_or_else(|why| panic!("{model} names at {rung:?}: {why}"))
    };
    assert!(!largest.rungs.is_empty(), "{model}: the load admits a rung");
    let classes = |asks: &engine_gpu::serve::Asks| -> Vec<usize> {
        asks.unservable.iter().map(|held| held.class).collect()
    };
    for &rung in &largest.rungs {
        let at = named(Some(rung));
        assert_eq!(
            at.kernels, largest.kernels,
            "{model}: the kernels at rung {rung}"
        );
        assert_eq!(
            at.dispatches, largest.dispatches,
            "{model}: the dispatch bound at rung {rung}"
        );
        assert_eq!(
            classes(&at),
            classes(&largest),
            "{model}: the unservable classes at rung {rung}"
        );
    }
    let keys: Vec<&str> = largest.kernels.iter().map(|kernel| kernel.key).collect();
    eprintln!(
        "{model}: rungs {:?} each name {} kernels (set {:016x}), {} dispatches a fire at most, \
         unservable {:?}",
        largest.rungs,
        largest.kernels.len(),
        fnv(&keys.join(" ")),
        largest.dispatches,
        classes(&largest)
    );
}

#[test]
fn every_rung_of_a_dense_miniature_names_what_the_largest_names() {
    every_rung_names_what_the_largest_names("qwen3-micro-text");
}

#[test]
fn every_rung_of_a_hybrid_miniature_names_what_the_largest_names() {
    every_rung_names_what_the_largest_names("qwen35-tiny");
}

#[test]
fn every_rung_of_a_windowed_miniature_names_what_the_largest_names() {
    every_rung_names_what_the_largest_names("gemma4-e4b-mini-l1");
}

#[test]
fn every_rung_of_a_miniature_with_unservable_classes_names_what_the_largest_names() {
    every_rung_names_what_the_largest_names("gemma4-e4b-mini-l6");
}

/// A fire whose lanes mix classes so that a region's mask comes in runs: a
/// prompt and a decode under an adapter, which run the adapter's region,
/// with a plain decode between them in the class order. Where the load
/// copies windows the region's rows are copied into one (its attention
/// reads the fire's masks, which the copy carries), launching as the same
/// fire without the plain decode does; else it launches a run at a time.
/// Either way the fire asks for no pipeline the load did not build and
/// encodes no more dispatches than the load named.
///
/// The lanes are the order's, read off the compiled model, not assumed:
/// since the mask is per-lane data (5b538b76d, 20a6c0cdf) a lane's mask is
/// no class bit, so the seriated order lays every prompt class before every
/// decode class, and a plain prompt beside an adapted one (this test's fire
/// before the fold) cuts no region.
#[test]
fn a_mixed_fire_that_cuts_a_region_fires_from_what_the_load_built() {
    use poem_exec::fire::fallback;
    use poem_ir::{ClassSet, RowAxis};

    let fixture = fixture("qwen3-micro-text", None);
    let Some(mut shell) = load(&fixture, CONTEXT) else {
        return;
    };
    let built = shell.compiled();
    let named = shell.named().clone();
    let tokens: Vec<u32> = (0..CONTEXT).map(|at| (at * 37 + 11) % 256).collect();
    const ROWS: usize = 8;
    let plain = |rows: usize| word(&fixture.trace, rows as u32);
    let adapted = |rows: usize| {
        fixture
            .trace
            .facts
            .word(&Request::new(rows as u32, false).adapted(true))
    };
    assert_ne!(
        adapted(ROWS),
        plain(ROWS),
        "the fixture tells an adapted lane apart"
    );

    // The adapted prompt, the plain decode and the adapted decode lie in
    // that turn of the class order, and a region holds both adapted lanes'
    // classes and not the plain decode's: the fire cuts it.
    {
        let compiled = shell.compiled_model();
        let axis = RowAxis::Tokens;
        let classes = compiled.classes.classes.len();
        let class = |word: u64| {
            compiled
                .classes
                .classes
                .iter()
                .position(|class| class.words.contains(&word))
                .unwrap_or_else(|| panic!("no class holds word {word:#x}"))
        };
        let order = compiled
            .order_for(axis)
            .map(|order| order.class_order(&ClassSet::of(0..classes)))
            .unwrap_or_default();
        let turn = |word: u64| {
            order
                .iter()
                .position(|&at| usize::from(at) == class(word))
                .expect("the order holds every class")
        };
        let (a, b, c) = (adapted(ROWS), plain(1), adapted(1));
        assert!(
            turn(a) < turn(b) && turn(b) < turn(c),
            "the class order {order:?} does not lay the plain decode between the adapted lanes"
        );
        let cut = compiled.template().iter().enumerate().any(|(at, region)| {
            compiled.axis_of(at) == axis
                && !fallback::promised(compiled, axis, region)
                && region.mask.contains(class(a))
                && region.mask.contains(class(c))
                && !region.mask.contains(class(b))
        });
        assert!(
            cut,
            "no region holds the adapted lanes and not the plain decode"
        );
    }

    // Three plain prompts seat their rows: no region is cut.
    let seated: Vec<Seated<'_>> = (0..3)
        .map(|lane| Seated {
            held: Some(0),
            ..Seated::of(Lane {
                slot: lane,
                word: plain(ROWS),
                tokens: &tokens[..ROWS],
            })
        })
        .collect();
    shell
        .fire_seated(&seated)
        .expect("three plain prompts fire");
    shell.walked_serving();
    shell.recorded_serving();

    let lane = |slot: u32, held: u32, word: u64, rows: usize| Seated {
        held: Some(held),
        ..Seated::of(Lane {
            slot,
            word,
            tokens: &tokens[..rows],
        })
    };
    fn under(seated: Seated<'_>) -> Seated<'_> {
        Seated {
            adapter: Some(0),
            ..seated
        }
    }
    // A new prompt under an adapter and a decode of the third seat under
    // it: the adapter's region in one run.
    let whole = [
        under(lane(0, 0, adapted(ROWS), ROWS)),
        under(lane(2, ROWS as u32, adapted(1), 1)),
    ];
    shell.fire_seated(&whole).expect("the adapted fire fires");
    let (_, whole) = shell.walked_serving();

    // The same with a plain decode of the second seat between them.
    let seated = [
        under(lane(0, 0, adapted(ROWS), ROWS)),
        lane(1, ROWS as u32, plain(1), 1),
        under(lane(2, ROWS as u32 + 1, adapted(1), 1)),
    ];
    let got = shell.fire_seated(&seated).expect("the mixed fire fires");
    assert!(got.iter().all(|row| row.iter().all(|v| v.is_finite())));
    let (dispatches, launches) = shell.walked_serving();
    if shell.copies() {
        assert_eq!(
            launches, whole,
            "the mixed fire's cut region was not copied into one window"
        );
    } else {
        assert!(
            launches > whole,
            "the mixed fire cut no region: {launches} launches, {whole} without the plain decode"
        );
    }
    assert_eq!(
        shell.compiled(),
        built,
        "the mixed fire built a pipeline the load's walks did not name"
    );
    assert!(
        dispatches <= u64::from(named.dispatches),
        "the mixed fire encoded {dispatches} dispatches and the load named {} at most",
        named.dispatches
    );
    eprintln!(
        "mixed fire: {launches} launches ({whole} with no region cut), {dispatches} dispatches \
         of {} named at most, {} pipelines built since the load, {} walked to be kept",
        named.dispatches,
        shell.compiled() - built,
        shell.recorded_serving()
    );
}

/// A masked lane in a mixed fire (a masked prompt, a plain decode, a
/// prompt under an adapter) fires with every lane in it, launches as the same fire stating
/// no mask does (a mask is data: it cuts no region of its own, and where the
/// load copies windows the copy carries its rows' entries), and the masked
/// prompt answers what it answers alone through its mask (up to contract
/// A's near-tie), which hides keys (it answers otherwise unmasked).
#[test]
fn a_masked_lane_in_a_mixed_fire_launches_and_answers_as_unmasked_lanes_do() {
    use engine::fire::{Mask, Masking};
    use eta_compiler::contract::{Drift, near_tie};

    let fixture = fixture("qwen3-micro-text", None);
    let Some(mut shell) = load(&fixture, CONTEXT) else {
        return;
    };
    // Every fire walks, so each counts its launches.
    shell.walk_every_fire();
    let tokens: Vec<u32> = (0..CONTEXT).map(|at| (at * 37 + 11) % 256).collect();
    const ROWS: usize = 8;
    let plain = |rows: usize| word(&fixture.trace, rows as u32);
    let adapted = fixture
        .trace
        .facts
        .word(&Request::new(ROWS as u32, false).adapted(true));
    let seat: Vec<Seated<'_>> = (0..3)
        .map(|lane| Seated {
            held: Some(0),
            ..Seated::of(Lane {
                slot: lane,
                word: plain(ROWS),
                tokens: &tokens[..ROWS],
            })
        })
        .collect();
    shell.fire_seated(&seat).expect("three plain prompts fire");
    // A mask a row over the lane's rows: rows 2 and 5 hide key 1, the rest
    // admit every key (a row reads none past its own whatever it says).
    let mask = Masking::Rows(
        (0..ROWS as u64)
            .map(|row| match row {
                2 | 5 => Mask::new(vec![0, 1, 1, ROWS as u32 - 2], ROWS as u64),
                _ => Mask::new(vec![0, ROWS as u32], ROWS as u64),
            })
            .collect(),
    );
    let lane = |slot: u32, held: u32, word: u64, rows: usize| Seated {
        held: Some(held),
        ..Seated::of(Lane {
            slot,
            word,
            tokens: &tokens[..rows],
        })
    };
    let mixed = |masked: bool| {
        [
            Seated {
                mask: masked.then_some(&mask),
                ..lane(0, 0, plain(ROWS), ROWS)
            },
            lane(1, ROWS as u32, plain(1), 1),
            Seated {
                adapter: Some(0),
                ..lane(2, 0, adapted, ROWS)
            },
        ]
    };
    shell.walked_serving();
    let unmasked = shell
        .fire_seated(&mixed(false))
        .expect("the mixed fire fires");
    let (_, plain_launches) = shell.walked_serving();
    let got = shell
        .fire_seated(&mixed(true))
        .expect("the mixed fire with a masked lane fires");
    let (_, launches) = shell.walked_serving();
    assert_eq!(
        launches, plain_launches,
        "the masked lane's rows launched otherwise than unmasked ones"
    );
    assert!(
        got[0]
            .iter()
            .zip(&unmasked[0])
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "the masked lane's mask hid nothing"
    );
    let alone = shell
        .fire_seated(&[Seated {
            mask: Some(&mask),
            ..lane(0, 0, plain(ROWS), ROWS)
        }])
        .expect("the masked prompt fires alone")
        .swap_remove(0);
    let (a, b) = (&alone, &got[0]);
    let (ta, tb) = (argmax(a) as usize, argmax(b) as usize);
    if ta != tb {
        let mut top = [f32::NEG_INFINITY; 2];
        for &v in a {
            if v > top[0] {
                top = [v, top[0]];
            } else if v > top[1] {
                top[1] = v;
            }
        }
        let parting = near_tie(top[0] - top[1], &[ta, tb], Drift::Rows { a, b }, None);
        assert!(
            parting.tie,
            "the masked lane parts from itself alone on no near-tie: {parting:?}"
        );
    }
    eprintln!(
        "mixed fire with a masked lane: {launches} launches ({plain_launches} unmasked), \
         token {tb} beside {ta} alone"
    );
}

#[test]
fn a_dense_miniature_fires_every_rung_from_what_the_load_built() {
    every_rung_fires_what_the_load_built("qwen3-micro-text");
}

#[test]
fn a_hybrid_miniature_fires_every_rung_from_what_the_load_built() {
    every_rung_fires_what_the_load_built("qwen35-tiny");
}

/// A hybrid's lane that buffers its recurrent rows (a beam's, a draft's)
/// moves them through the buffer and commits as of the fold: kernels no
/// folding fire asks for, which the load names and builds all the same.
#[test]
fn a_hybrid_miniatures_buffering_fires_build_nothing_the_load_did_not() {
    use engine::fire::{FoldLen, Lineage, RsVerb};
    let fixture = fixture("qwen35-tiny", None);
    let Some(mut shell) = load(&fixture, CONTEXT) else {
        return;
    };
    let built = shell.compiled();
    let per_page = shell
        .pools()
        .rs_rows()
        .expect("a hybrid's pool lays an rs list")
        .tokens
        .max(1);
    shell.open(0).expect("the slot opens");
    let tokens: Vec<u32> = (0..8).map(|at| (at * 37 + 11) % 256).collect();
    let buffer = |folded: u32, fold: u32| RsVerb::Buffer {
        pages: vec![0; 8u32.div_ceil(per_page) as usize],
        base: 0,
        folded,
        fold: FoldLen::Host(fold),
        lineage: Lineage::Set,
    };
    // A prefill that folds, two rows buffered, then one row that replays
    // them and folds them.
    let fires = [
        (0u32, 4usize, RsVerb::Fold),
        (4, 2, buffer(4, 0)),
        (6, 1, buffer(4, 2)),
    ];
    for (held, rows, rs) in &fires {
        let seated = [Seated {
            rs,
            held: Some(*held),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, *rows as u32),
                tokens: &tokens[*held as usize..*held as usize + rows],
            })
        }];
        let got = shell
            .fire_seated(&seated)
            .unwrap_or_else(|why| panic!("{rows} row(s) held {held} under {rs:?} fire: {why}"));
        assert!(got.iter().all(|row| row.iter().all(|v| v.is_finite())));
        assert_eq!(
            shell.compiled(),
            built,
            "{rows} row(s) held {held} under {rs:?} built a pipeline the load's walks did not name"
        );
    }
}

/// A decode's body replays what it fired the first time: two buckets of
/// one load alternate on the same arms' fire blocks, a yield and a restore
/// of the pool's segments between, and every fire of a bucket lands the
/// logits its first fire did (a replay that skips its head copy because the
/// block holds its image already reads that image).
#[test]
fn a_decodes_bodies_replay_their_logits_across_buckets_and_a_yield() {
    let fixture = fixture("qwen3-micro-text", None);
    let Some(mut shell) = load(&fixture, CONTEXT) else {
        return;
    };
    let tokens: Vec<u32> = (0..SEATS).map(|at| (at * 37 + 11) % 256).collect();
    let decode = |shell: &mut Shell, lanes: u32| -> Vec<Vec<f32>> {
        let seated: Vec<Seated<'_>> = (0..lanes)
            .map(|lane| Seated {
                held: Some(4),
                ..Seated::of(Lane {
                    slot: lane,
                    word: word(&fixture.trace, 1),
                    tokens: &tokens[lane as usize..lane as usize + 1],
                })
            })
            .collect();
        shell
            .fire_seated(&seated)
            .unwrap_or_else(|why| panic!("a decode of {lanes} lane(s) fires: {why}"))
    };
    let wide = SEATS.min(2);
    let first = (decode(&mut shell, 1), decode(&mut shell, wide));
    for round in 0..3 {
        if round == 1 {
            let segments = shell.pool_chunk_facts().segments;
            if segments.growth != engine::chunk::Growth::Fixed && segments.committed > 1 {
                let committed = segments.committed;
                shell.release_segments(1).expect("the top segments go back");
                shell.commit_segments(committed).expect("the segments commit again");
            }
        }
        assert_eq!(decode(&mut shell, 1), first.0, "round {round}: one lane");
        assert_eq!(decode(&mut shell, wide), first.1, "round {round}: {wide} lanes");
    }
}

#[test]
fn a_windowed_miniature_fires_every_rung_from_what_the_load_built() {
    every_rung_fires_what_the_load_built("gemma4-e4b-mini-l1");
}

/// Every byte a served miniature holds on the device is in the device's
/// ledger under an owner, and the ledger holds within what the device has.
#[test]
fn a_served_miniature_is_in_the_ledger_whole() {
    use engine::memory::Class;
    // The driver reports what it holds the process, every device summed:
    // no other test holds one while this one reads it.
    crate::gpu::sole();
    let fixture = fixture("qwen3-micro-text", None);
    let _alone = loading();
    let Some(mut shell) = load_alone(
        &fixture,
        PAGE,
        CONTEXT,
        None,
        engine::load::Residency::default(),
    ) else {
        return;
    };
    let loaded = shell.device().outside();
    let built = shell.compiled();
    assert_eq!(
        built,
        shell.named().kernels.len() as u64,
        "the load built other pipelines than the ones its walks named"
    );
    serve_seats(&mut shell, &fixture.trace, 2);
    let books = shell.device().books();
    let ledger = books.ledger();
    assert_eq!(ledger.check(), Ok(()), "{ledger}");
    assert_eq!(books.audit().gap(), 0, "{ledger}");
    // Serving builds no pipeline: every one a fire asks for was named by
    // the load's walks and built before it reserved.
    let served = shell.device().outside();
    eprintln!("outside: {loaded} at load, {served} after serving, {built} pipelines built");
    assert_eq!(
        shell.compiled(),
        built,
        "a serving fire built a pipeline the load's walks did not name"
    );
    assert!(
        served <= loaded,
        "serving took {} bytes of the driver's own past what the load left it",
        served - loaded
    );
    for class in [Class::Model, Class::State, Class::Fire] {
        assert!(
            ledger.bytes(class) > 0,
            "nothing entered as {class:?}: {ledger}"
        );
    }
}

/// State lent in the activation span under a narrow rung's carve stands
/// through that rung's fires, the first (recorded where the backend
/// records) and every replay: no fire writes past the chunks its rung
/// states.
#[test]
fn state_lent_in_the_span_outlives_a_narrow_rungs_fires() {
    use engine::chunk::{SlotCopy, SlotKind};
    use engine::transfer::MemoryDomain;
    let fixture = fixture("qwen3-micro-text", None);
    let Some(mut shell) = load_with(
        &fixture,
        PAGE,
        CONTEXT,
        Some(64 << 10),
        engine::load::Residency {
            host_kv_budget: Some(256 << 20),
            // State well within the kernels' index of the pool (8 GiB of
            // 64 KiB chunks), whatever the device has: a pool whose state
            // reaches it states every rung at the span's chunks and lends
            // none (`store::pool::sized`), so a share of a large device
            // would leave no narrow rung to test.
            device_kv_budget: Some(256 << 20),
            ..engine::load::Residency::default()
        },
    ) else {
        return;
    };
    let device = <backend::Api as engine_gpu::api::Host>::domain(&crate::gpu::device_boot());
    // The prefills take the top rung; the decodes after them two lanes.
    let mut fed: [Vec<u32>; 2] = prompts();
    let mut held = [0u32; 2];
    for lane in 0..2usize {
        let seated = Seated {
            held: Some(held[lane]),
            ..Seated::of(Lane {
                slot: lane as u32,
                word: word(&fixture.trace, fed[lane].len() as u32),
                tokens: &fed[lane],
            })
        };
        let got = shell.fire_seated(&[seated]).expect("the prefill lands");
        held[lane] += fed[lane].len() as u32;
        fed[lane] = vec![argmax(&got[0])];
    }
    let facts = shell.pool_chunk_facts();
    let total = facts.total_chunks;
    let rung = |tokens: u32| {
        facts
            .act_chunks
            .iter()
            .find(|&&(rung, _)| rung >= tokens)
            .map_or(0, |&(_, chunks)| chunks)
    };
    let (narrow, widest) = (rung(2), facts.act_chunks_max());
    assert!(
        widest > narrow,
        "a two-lane rung lends span chunks: {:?}",
        facts.act_chunks
    );
    let list = facts
        .shapes
        .iter()
        .find(|shape| shape.id == LIST && matches!(shape.kind, SlotKind::List { .. }))
        .expect("a kv list");
    let per = list.slots_per_chunk;
    // Every chunk of the span the narrow rung lends.
    let lent: Vec<u32> = (total - widest..total - narrow).collect();
    let device_refs: Vec<u32> = lent
        .iter()
        .flat_map(|&chunk| (0..per).map(move |i| chunk * per + i))
        .collect();
    let slots = device_refs.len() as u32;
    let host_in: Vec<u32> = (0..slots).collect();
    let host_out: Vec<u32> = (slots..2 * slots).collect();
    let pattern: Vec<u8> = (0..u64::from(slots) * list.slot_bytes)
        .map(|at| (at.wrapping_mul(2_654_435_761) >> 13) as u8 | 1)
        .collect();
    shell
        .write_tier_slots(&[(LIST, host_in.clone())], &pattern)
        .expect("the host slots take the pattern");
    let pairs = |from: &[u32], to: &[u32]| from.iter().copied().zip(to.iter().copied()).collect();
    shell
        .copy_slots(&SlotCopy {
            shape: LIST,
            src: MemoryDomain::HostPinned,
            dst: device,
            slots: pairs(&host_in, &device_refs),
            runs: Vec::new(),
        })
        .expect("the pattern lands in the lent chunks");
    for _ in 0..6 {
        let seated: Vec<Seated<'_>> = (0..2usize)
            .map(|lane| Seated {
                held: Some(held[lane]),
                ..Seated::of(Lane {
                    slot: lane as u32,
                    word: word(&fixture.trace, 1),
                    tokens: &fed[lane],
                })
            })
            .collect();
        let got = shell.fire_seated(&seated).expect("the decode lands");
        for lane in 0..2 {
            held[lane] += 1;
            fed[lane] = vec![argmax(&got[lane])];
        }
    }
    shell
        .copy_slots(&SlotCopy {
            shape: LIST,
            src: device,
            dst: MemoryDomain::HostPinned,
            slots: pairs(&device_refs, &host_out),
            runs: Vec::new(),
        })
        .expect("the lent chunks read back");
    let back = shell
        .read_tier_slots(&[(LIST, host_out)])
        .expect("the host slots read");
    let first = back.iter().zip(&pattern).position(|(a, b)| a != b);
    assert_eq!(
        first, None,
        "a two-lane fire wrote lent span chunks {lent:?} (rung chunks {narrow} of {widest}, \
         total {total})"
    );
}

/// A park to the host tier is told landed only once it has read its device
/// slots (or has read them when it returns): what the runtime gives those
/// slots to next (any shape, written by anything, not a fire alone) cannot
/// reach the parked bytes. The slots are overwritten from the host the
/// moment the park is told landed, outside any frame, and the host tier
/// still holds what they held.
#[test]
fn a_park_is_told_landed_once_it_has_read_its_slots() {
    use engine::chunk::{SlotCopy, SlotKind};
    use engine::transfer::MemoryDomain;
    use engine_gpu::backend::Buffer as _;
    let fixture = fixture("qwen3-micro-text", None);
    let Some(mut shell) = load_with(
        &fixture,
        PAGE,
        CONTEXT,
        Some(64 << 10),
        engine::load::Residency {
            host_kv_budget: Some(256 << 20),
            ..engine::load::Residency::default()
        },
    ) else {
        return;
    };
    let device = <backend::Api as engine_gpu::api::Host>::domain(&DeviceBoot::default());
    let host = MemoryDomain::HostPinned;
    let facts = shell.pool_chunk_facts();
    shell
        .commit_segments(facts.segments.max.min(4))
        .expect("the segments commit");
    let list = facts
        .shapes
        .iter()
        .find(|shape| shape.id == LIST && matches!(shape.kind, SlotKind::List { .. }))
        .expect("a kv list");
    let per = list.slots_per_chunk;
    let n = per.min(4);
    // Device slots of chunk 1 (chunk 0 is the null chunk).
    let slots: Vec<u32> = (per..per + n).collect();
    let (seed, parked, back): (Vec<u32>, Vec<u32>, Vec<u32>) = (
        (0..n).collect(),
        (n..2 * n).collect(),
        (2 * n..3 * n).collect(),
    );
    let pattern: Vec<u8> = (0..u64::from(n) * list.slot_bytes)
        .map(|at| (at * 13 + 7) as u8 ^ (at >> 9) as u8)
        .collect();
    let pairs = |from: &[u32], to: &[u32]| from.iter().copied().zip(to.iter().copied()).collect();
    let copy = |src, dst, slots| SlotCopy {
        shape: LIST,
        src,
        dst,
        slots,
        runs: Vec::new(),
    };
    shell
        .write_tier_slots(&[(LIST, seed.clone())], &pattern)
        .expect("the host slots take the pattern");
    for round in 0..8u8 {
        shell
            .copy_slots(&copy(host, device, pairs(&seed, &slots)))
            .expect("the slots land on the device");
        // On the queue past the call where the host chunks are mapped, told
        // landed from its fence; else waited out in the call.
        let (told, landed) = std::sync::mpsc::channel();
        let now = shell
            .copy_slots_landing(
                &copy(device, host, pairs(&slots, &parked)),
                Some(Box::new(move || {
                    let _ = told.send(());
                })),
            )
            .expect("the slots park");
        if !now {
            landed
                .recv_timeout(std::time::Duration::from_secs(30))
                .expect("the park is told landed");
        }
        // The slots go to their next owner, which writes them at once.
        for &slot in &slots {
            let (buffer, at) = shell
                .pools()
                .slot_place(LIST, slot)
                .expect("a device slot's place");
            buffer
                .write_shared(at, &vec![0x5a ^ round; list.slot_bytes as usize])
                .expect("the next owner writes the slot");
        }
        let held = shell
            .read_tier_slots(&[(LIST, parked.clone())])
            .expect("the parked slots read");
        assert!(
            held == pattern,
            "round {round}: the host tier holds what the slots held when the park landed"
        );
        // The overwrite reached the device slots.
        shell
            .copy_slots(&copy(device, host, pairs(&slots, &back)))
            .expect("the overwritten slots read back");
        let theirs = shell
            .read_tier_slots(&[(LIST, back.clone())])
            .expect("the overwritten slots read");
        assert!(
            theirs.iter().all(|&byte| byte == 0x5a ^ round),
            "round {round}: the next owner's bytes are on the device"
        );
    }
    shell.settle_copies().expect("every copy settles");
}

/// A masked fire after a partial fold: a lane's mask speaks of the keys from
/// its committed boundary on (`KvDelta::folded`). Past a boundary at 100 of
/// a 150-key prompt, a mask over the buffer and the fire's own rows lands
/// the same logits, bit for bit, as the same bits over the whole context
/// from a boundary at 0, its committed columns set; the mask is read (it
/// lands other logits than all set), and one short of its buffer and rows
/// is refused for that extent.
#[test]
fn a_masked_fire_after_a_partial_fold_admits_the_committed_keys() {
    let fixture = fixture("qwen3-micro-text", None);
    let Some(mut shell) = load(&fixture, CONTEXT) else {
        return;
    };
    let [prompt, _] = prompts();
    let have = prompt.len() as u32;
    shell
        .fire_seated(&[Seated {
            held: Some(0),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, have),
                tokens: &prompt,
            })
        }])
        .expect("the prefill lands");

    const ROWS: u32 = 4;
    const FOLDED: u32 = 100;
    let tokens: Vec<u32> = (0..ROWS).map(|at| (at * 29 + 3) % 256).collect();
    let masked = fixture.trace.facts.word(&Request::new(ROWS, true));
    let buffer = u64::from(have - FOLDED);
    // Hides every third buffered key; admits the fire's own rows.
    let hidden = |column: u64| column < buffer && column % 3 == 1;
    let bits = |committed: u64, hide: &dyn Fn(u64) -> bool| {
        let total = committed + buffer + u64::from(ROWS);
        let mut runs = vec![0u32];
        let mut admitted = false;
        for column in 0..total {
            let keep = column < committed || !hide(column - committed);
            if keep != admitted {
                runs.push(0);
                admitted = keep;
            }
            *runs.last_mut().unwrap() += 1;
        }
        engine::fire::Masking::Extent(engine::fire::Mask::new(runs, total))
    };
    let mut fire = |folded: u32, mask: &engine::fire::Masking| {
        shell.fire_seated(&[Seated {
            held: Some(have),
            folded,
            mask: Some(mask),
            ..Seated::of(Lane {
                slot: 0,
                word: masked,
                tokens: &tokens,
            })
        }])
    };
    let past = fire(FOLDED, &bits(0, &hidden)).expect("the masked fire lands");
    let whole = fire(0, &bits(u64::from(FOLDED), &hidden)).expect("the whole mask lands");
    same("a mask past the boundary against the whole", &past, &whole);
    let all = fire(FOLDED, &bits(0, &|_| false)).expect("an all-set mask lands");
    assert!(
        past[0]
            .iter()
            .zip(&all[0])
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "the mask hid nothing"
    );
    let short = {
        let engine::fire::Masking::Extent(mask) = bits(0, &hidden) else {
            unreachable!()
        };
        let total = mask.total - 1;
        engine::fire::Masking::Extent(engine::fire::Mask::new(mask.runs, total))
    };
    let refused = fire(FOLDED, &short).expect_err("a mask short of its extent");
    assert!(
        refused
            .to_string()
            .contains(&format!("will read {}", buffer + u64::from(ROWS))),
        "{refused}"
    );
}

/// A lane's attention mask is per-row data, never its class (the mask fold):
/// a masked beam lane beside three plain decodes fires in one fire and one
/// body, the launches four plain decodes make (where the backend records,
/// the plain fire's recording replayed, none of its own kept), and each lane
/// answers what it answers alone (contract A: a token parts only on a
/// near-tie). The beam's mask is read: it answers otherwise unmasked.
#[test]
fn a_masked_beam_beside_plain_decodes_answers_as_each_alone() {
    use engine::fire::{Mask, Masking};
    use eta_compiler::contract::{Drift, near_tie};

    const PLAIN: u32 = 3;
    const LANES: u32 = PLAIN + 1;
    const PROMPT: u32 = 40;
    const STEPS: u32 = 6;
    let fixture = fixture("qwen3-micro-text", None);
    // Each lane's slot pair: batched in slot `l`, alone in slot `LANES + l`.
    let Some(mut shell) = load_lanes(&fixture, PAGE, CONTEXT, 2 * LANES, 256) else {
        return;
    };
    let trace = fixture.trace.clone();
    let beam = PLAIN as usize;
    // Every key but a stretch of the prompt (`8..16`), so a lane reading no
    // mask answers otherwise.
    let beam_mask =
        |columns: u32| Masking::Extent(Mask::new(vec![0, 8, 8, columns - 16], u64::from(columns)));
    // `(slot, tokens, held, mask)` a lane.
    type Row<'a> = (u32, &'a [u32], u32, Option<&'a Masking>);
    let fire = |shell: &mut Shell, lanes: &[Row<'_>]| -> Vec<Vec<f32>> {
        let seated: Vec<Seated<'_>> = lanes
            .iter()
            .map(|&(slot, tokens, held, mask)| Seated {
                held: Some(held),
                mask,
                ..Seated::of(Lane {
                    slot,
                    word: word(&trace, tokens.len() as u32),
                    tokens,
                })
            })
            .collect();
        shell.fire_seated(&seated).expect("the fire serves")
    };

    let mut last = [[0u32; 2]; LANES as usize];
    for lane in 0..LANES {
        let tokens: Vec<u32> = (0..PROMPT)
            .map(|t| (t * 7 + 11 + lane * 13) % 256)
            .collect();
        for (side, slot) in [(0usize, lane), (1, LANES + lane)] {
            let rows = fire(&mut shell, &[(slot, &tokens, 0, None)]);
            last[lane as usize][side] = argmax(&rows[0]);
        }
    }
    shell.walked_serving();
    shell.recorded_serving();
    let (mut same, mut ties) = (0u32, 0u32);
    let mut parted = [false; LANES as usize];
    for step in 0..STEPS {
        let held = PROMPT + step;
        let mask = beam_mask(held + 1);
        let next: Vec<[u32; 1]> = (0..LANES as usize).map(|l| [last[l][0]]).collect();
        let rows = |masked: bool| -> Vec<Row<'_>> {
            (0..LANES)
                .map(|l| {
                    let mask = (masked && l as usize == beam).then_some(&mask);
                    (l, &next[l as usize][..], held, mask)
                })
                .collect()
        };
        // The same lanes stating no mask first: the same keys written.
        let plain = fire(&mut shell, &rows(false));
        let (_, plain_launches) = shell.walked_serving();
        shell.recorded_serving();
        let batched = fire(&mut shell, &rows(true));
        let (_, launches) = shell.walked_serving();
        assert_eq!(
            shell.recorded_serving(),
            0,
            "step {step}: the fire with a masked lane kept a recording of its own"
        );
        assert!(
            launches == 0 || launches == plain_launches,
            "step {step}: the fire with a masked lane made {launches} launches, the plain \
             one {plain_launches}"
        );
        assert!(
            plain[beam]
                .iter()
                .zip(&batched[beam])
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "step {step}: the beam's mask hid nothing"
        );
        for lane in 0..LANES as usize {
            let token = [last[lane][1]];
            let alone = fire(
                &mut shell,
                &[(
                    LANES + lane as u32,
                    &token,
                    held,
                    (lane == beam).then_some(&mask),
                )],
            )
            .swap_remove(0);
            let (a, b) = (&alone, &batched[lane]);
            let (ta, tb) = (argmax(a) as usize, argmax(b) as usize);
            if !parted[lane] && ta != tb {
                let mut top = [f32::NEG_INFINITY; 2];
                for &v in a {
                    if v > top[0] {
                        top = [v, top[0]];
                    } else if v > top[1] {
                        top[1] = v;
                    }
                }
                let parting = near_tie(top[0] - top[1], &[ta, tb], Drift::Rows { a, b }, None);
                eprintln!("lane {lane} parts at step {step}: {parting:?}");
                assert!(
                    parting.tie,
                    "lane {lane} (beam: {}) parts at step {step} on no near-tie",
                    lane == beam
                );
                parted[lane] = true;
                ties += 1;
            } else if !parted[lane] {
                same += 1;
            }
            last[lane] = [tb as u32, ta as u32];
        }
    }
    eprintln!("{same} lane steps the same token, {ties} parted on a near-tie");
    assert!(same > 0, "no lane step answered as it does alone");
}

/// A row's causal bound is its row order, the cache row; a stated position
/// rotates the row and bounds nothing. A lane's four rows land at slots
/// 0..4 and all state position 0: the last row still reads the three before
/// it, so two lanes whose middle tokens differ answer differently (bounded
/// by the stated position, the last row read slot 0 alone and the two
/// answered the same). The rotation follows the stated position: stating
/// 0 everywhere answers otherwise than stating nothing, and stating the
/// slots answers the same, bit for bit.
#[test]
fn a_stated_position_rotates_and_bounds_no_key() {
    let fixture = fixture("qwen3-micro-text", None);
    let Some(mut shell) = load(&fixture, CONTEXT) else {
        return;
    };
    let mut last = |slot: u32, tokens: &[u32], positions: &[u32]| {
        let seated = Seated {
            held: Some(0),
            positions,
            ..Seated::of(Lane {
                slot,
                word: word(&fixture.trace, tokens.len() as u32),
                tokens,
            })
        };
        shell
            .fire_seated(&[seated])
            .expect("the fire lands")
            .remove(0)
    };
    let zeros = [0u32; 4];
    let one = last(0, &[11, 48, 85, 122], &zeros);
    let two = last(1, &[11, 200, 7, 122], &zeros);
    let apart = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    };
    assert!(
        apart(&one, &two) > 1e-3,
        "the last row's logits are the same whatever the rows before it say: its \
         attention was bounded by its stated position, not its cache row"
    );
    let unstated = last(0, &[11, 48, 85, 122], &[]);
    let slots = last(0, &[11, 48, 85, 122], &[0, 1, 2, 3]);
    same(
        "positions stated as the slots",
        &[slots],
        &[unstated.clone()],
    );
    assert!(
        apart(&one, &unstated) > 1e-3,
        "stating position 0 everywhere rotated the rows as their slots"
    );
}

/// Panics at the first row whose logits differ in a bit.
pub fn same(what: &str, got: &[Vec<f32>], want: &[Vec<f32>]) {
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
/// first + 3` from the top down, the lanes' pages interleaved in each and
/// scattered over its slots, piece `p` of lane `l`'s record in chunk
/// `first + 3 + l * pieces + p`. No two (lane, page) share a slot while a
/// chunk holds every lane's pages in it ([`Refs::distinct`]).
struct Refs {
    first: u32,
    per_list: u32,
    per_fixed: u32,
    pieces: u32,
}

impl Refs {
    fn page(&self, lane: u32, page: u32) -> u32 {
        let chunk = self.first + 2 - (page % 3);
        let nth = (page / 3) * 2 + lane;
        let stride = (7..)
            .step_by(2)
            .find(|&s| gcd(s, self.per_list) == 1)
            .expect("an odd stride coprime to the slots");
        let index = (stride * nth + 3) % self.per_list;
        chunk * self.per_list + index
    }

    /// Every slot ref of both lanes' `pages` pages is its own.
    fn distinct(&self, pages: u32) -> bool {
        let refs: std::collections::BTreeSet<u32> = (0..2)
            .flat_map(|lane| (0..pages).map(move |p| self.page(lane, p)))
            .collect();
        refs.len() == 2 * pages as usize
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

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
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
    assert!(
        refs.distinct(CONTEXT.div_ceil(page)),
        "the lanes' pages share no slot"
    );
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
/// evacuated mid-decode below it and that segment given back (where the
/// pool grows; a pool laid whole at its load keeps it), the decode running
/// on through the moved chunks.
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
            // A pool laid whole at its load (`Growth::Fixed`, as wgpu lays
            // it) stands whole: it gives no segment back.
            let stands = if facts.segments.growth == engine::chunk::Growth::Fixed {
                facts.segments.max
            } else {
                kept
            };
            assert_eq!(shell.pool_chunk_facts().segments.committed, stands);
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

/// A model with a compressed kv pool: its compressor state, rows of the list
/// slots it describes, moves with the slots an evacuation moves.
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
    let chunk = std::env::var(backend::CHUNK)
        .map_or(64 << 10, |bytes| bytes.parse().expect("the chunk is bytes"));
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

/// A window list the runtime trims (CONTRACT-BC 19, 21): slots wholly
/// behind the window read the null chunk (ref 0). One lane prefills past
/// the window in pieces of a fire's rows and decodes on; its logits are the
/// same bits whether every list holds every page (seat-addressed) or only
/// the window's, scattered over chunks by slot ref and trimmed as the
/// runtime trims, each step, those chunks evacuated mid-decode below a
/// segment given back.
fn trimmed_window(model: &str, layers: Option<u32>, page: u32, context: u32, prompt: u32) {
    trimmed_window_of(&fixture(model, layers), model, page, context, prompt);
}

fn trimmed_window_of(fixture: &Fixture, model: &str, page: u32, context: u32, prompt: u32) {
    use engine::chunk::{ShapeId, SlotKind};

    const STEPS: usize = 16;
    let prompt: Vec<u32> = (0..prompt).map(|at| (at * 37 + 11) % 256).collect();
    let run = |trimmed: bool| -> Option<Vec<Vec<f32>>> {
        let mut shell = load_paged(fixture, page, context, None)?;
        let facts = shell.pool_chunk_facts();
        facts.validate().expect("the facts hang together");
        eprintln!(
            "{model}: {} chunks of {} B, {} a segment, shapes {:?}",
            facts.total_chunks,
            facts.chunk_bytes,
            facts.segment_chunks,
            facts
                .shapes
                .iter()
                .map(|s| (s.id.0, s.kind, s.slot_bytes, s.slots_per_chunk))
                .collect::<Vec<_>>()
        );
        // What the load holds unservable, and what a bind that reads all
        // such a class writes is answered.
        let named = shell.named();
        if let Some(held) = named.unservable.first() {
            let classes = shell.classes();
            let reads: Vec<(String, u32)> = classes.classes[held.class]
                .outputs
                .iter()
                .map(|decl| (decl.name.clone(), 0))
                .collect();
            eprintln!(
                "{model}: lane classes {:?} of {} held unservable; a bind reading {:?}: {:?}",
                named
                    .unservable
                    .iter()
                    .map(|held| held.class)
                    .collect::<Vec<_>>(),
                classes.classes.len(),
                reads
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>(),
                engine_gpu::naming::refuses(&named.unservable, &classes, &reads),
            );
        }
        // Each list the lane pages: its shape, tokens a slot, window,
        // slots a chunk.
        let lists: Vec<(ShapeId, u32, Option<u32>, u32)> = facts
            .shapes
            .iter()
            .filter_map(|s| match s.kind {
                SlotKind::List { tokens, window, .. }
                    if s.role == engine::chunk::SlotRole::State =>
                {
                    Some((s.id, tokens, window, s.slots_per_chunk))
                }
                _ => None,
            })
            .collect();
        assert!(
            lists.iter().any(|l| l.2.is_some()),
            "{model}: a window list at context {context}"
        );
        let layout = shell.pool_layout();
        let segment = facts.segment_chunks;
        let low = 1 + layout.seat_chunks();
        // Each list's run of chunks, one after another.
        let mut runs = Vec::new();
        let mut held = 0u32;
        for &(_, tokens, _, per) in &lists {
            runs.push(held);
            held += context.div_ceil(tokens).div_ceil(per);
        }
        let kept = (low + held).div_ceil(segment);
        let first = kept * segment;
        let commit = (first + held).div_ceil(segment);
        assert!(
            facts.segments.max >= commit,
            "{model}: room for the scattered run"
        );
        if trimmed {
            shell.commit_segments(commit).expect("the segments commit");
        }
        // Slot `k` of list `l`: in its run's chunk `k / per`, strided there
        // by a step prime to `per`, so each page has a slot of its own.
        let slot_ref = |l: usize, k: u32| {
            let per = lists[l].3;
            let step = [7u32, 11, 13, 17]
                .into_iter()
                .find(|step| !per.is_multiple_of(*step))
                .expect("a step prime to the slots a chunk");
            (first + runs[l] + k / per) * per + (k * step + 3) % per
        };
        let mut rows_out = Vec::new();
        let mut have = 0u32;
        let mut fed: Vec<u32> = prompt.clone();
        for step in 0..=STEPS {
            if trimmed && step == STEPS / 2 {
                let moves = (0..held).map(|at| (first + at, low + at)).collect();
                shell
                    .evacuate(&Evacuation { moves })
                    .expect("the evacuation lands");
                shell
                    .release_segments(kept)
                    .expect("the top segments go back");
            }
            let pieces: Vec<Vec<u32>> = fed.chunks(256).map(<[u32]>::to_vec).collect();
            let mut last = Vec::new();
            for tokens_in in pieces {
                let n = tokens_in.len() as u32;
                let after = have + n;
                // Trimmed as the runtime trims: a slot whose last row lies
                // below `committed_end - window` is the null chunk.
                let tables: Vec<(ShapeId, Vec<u32>)> = lists
                    .iter()
                    .enumerate()
                    .map(|(l, &(id, tokens, window, _))| {
                        let refs = (0..after.div_ceil(tokens))
                            .map(|k| match window {
                                Some(w) if (k + 1) * tokens - 1 + w < have => 0,
                                _ => slot_ref(l, k),
                            })
                            .collect();
                        (id, refs)
                    })
                    .collect();
                let pages: &[u32] = tables
                    .iter()
                    .find(|(id, _)| *id == LIST)
                    .map_or(&[], |(_, refs)| refs.as_slice());
                let seated = Seated {
                    held: Some(have),
                    refs: trimmed,
                    pages: if trimmed { pages } else { &[] },
                    windows: if trimmed { &tables } else { &[] },
                    ..Seated::of(Lane {
                        slot: 0,
                        word: word(&fixture.trace, n),
                        tokens: &tokens_in,
                    })
                };
                last = shell.fire_seated(&[seated]).expect("the fire lands");
                have = after;
            }
            fed = vec![argmax(&last[0])];
            rows_out.extend(last);
        }
        Some(rows_out)
    };
    let Some(whole) = run(false) else {
        return;
    };
    let trimmed = run(true).expect("loaded before");
    same(&format!("{model}: trimmed against whole"), &trimmed, &whole);
}

/// A model whose every kv row is windowed: the window list is the All
/// axis's table.
#[test]
fn a_trimmed_window_list_reads_as_the_whole_one() {
    trimmed_window("gemma4-e4b-mini-l1", None, PAGE, 1024, 700);
}

/// A window list beside the kv list, its own table second, its slots of
/// `T_w` tokens beside 32-token kv pages.
#[test]
fn a_trimmed_window_list_beside_the_kv_list_reads_as_the_whole_one() {
    trimmed_window("gemma4-e4b-mini-l6", None, 32, 1024, 700);
}

/// Two window lists beside the kv list (CONTRACT-BC 21): Gemma's miniature
/// with every other sliding layer reading back 256 rows instead of 512,
/// each window its own list, table and trim.
#[test]
fn two_trimmed_window_lists_read_as_the_whole_ones() {
    const MODEL: &str = "gemma4-e4b-mini-l6";
    let package = poem_compiler::catalog::package_of(MODEL).expect("the model is listed");
    let (_, text) = package
        .files()
        .find(|(_, text)| text.contains("def forward(m, inputs):"))
        .expect("the package states its forward");
    let start = text.find("def forward(m, inputs):").expect("found above");
    let end = start
        + text[start..]
            .find("\ndef ")
            .expect("a function after the forward");
    let stated = "        d, kv_heads, win = geometry(m, at.reading)\n";
    let forward = text[start..end].replace(
        stated,
        &format!("{stated}        win = 256 if win != None and l % 2 == 1 else win\n"),
    );
    assert_ne!(forward, text[start..end], "the forward reads its geometry");
    let narrowed = poem_compiler::catalog::replacing(MODEL, "forward", &forward)
        .unwrap_or_else(|why| panic!("the narrowed forward loads: {why}"));
    let fixture = fixture_in(&narrowed, MODEL, None, "-two-windows");
    let windows: Vec<Option<u32>> = fixture
        .trace
        .caches
        .iter()
        .filter_map(|row| match row {
            poem_ir::CacheRow::Kv { window, .. } => Some(*window),
            poem_ir::CacheRow::State { .. } => None,
        })
        .collect();
    assert!(
        windows.contains(&Some(256)) && windows.contains(&Some(512)) && windows.contains(&None),
        "rows of two windows and full rows: {windows:?}"
    );
    trimmed_window_of(&fixture, MODEL, 32, 1024, 700);
}

/// Every shape a parked set carries comes back bit for bit through the
/// tiers below the device: kv list pages, the hybrid's records and its
/// rs-list rows, device -> host -> device, device -> disk -> device and
/// host -> disk -> host (kv pages host -> disk -> device too), the disk
/// budget divided into disk chunks laid out as the device's, as the load
/// states it. Words no bf16 holds (every one's low half set, NaN
/// payloads among them); two sets parked in one copy, in a disk order not
/// their own; their device pages taken by another owner and overwritten
/// between park and restore, so the restores land in other pages, two of
/// them issued back to back.
#[test]
fn a_tier_round_trip_is_bit_exact_for_every_parked_shape() {
    use engine::chunk::{SlotCopy, SlotShape};
    use engine::transfer::MemoryDomain;
    use engine_gpu::backend::Buffer as _;
    let fixture = fixture("qwen35-tiny", None);
    let Some(mut shell) = load_with(
        &fixture,
        PAGE,
        CONTEXT,
        None,
        engine::load::Residency {
            host_kv_budget: Some(256 << 20),
            ..engine::load::Residency::default()
        },
    ) else {
        return;
    };
    let dir = std::env::temp_dir().join(format!(
        "pie-{}-tier-exact-{}",
        backend::NAME,
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("the slot file's dir");
    let device = <backend::Api as engine_gpu::api::Host>::domain(&crate::gpu::device_boot());
    let (host, disk) = (MemoryDomain::HostPinned, MemoryDomain::LocalDisk);
    const BUDGET: u64 = 64 << 20;
    let chunks = shell
        .seat_disk_kv(Some(BUDGET), Some(&dir), SEATS, CONTEXT)
        .expect("the slot file opens");
    // The budget divides into disk chunks of a pool chunk each (on a direct
    // I/O boundary), laid out as the device's: any shape's slots take any
    // chunk, packed as there.
    let chunk_bytes = shell.pools().disk_chunk_bytes();
    let pool_chunk = shell.pool_chunk_facts().chunk_bytes;
    assert_eq!(chunk_bytes, pool_chunk.next_multiple_of(4096));
    assert!(
        shell
            .pools()
            .disk_division()
            .iter()
            .all(|&(_, per, slot)| u64::from(per) * slot <= pool_chunk)
    );
    assert_eq!(u64::from(chunks), BUDGET / chunk_bytes);
    assert!(chunks >= 12, "the slot file holds {chunks} chunks");
    let facts = shell.pool_chunk_facts();
    assert_eq!(facts.disk_chunks, chunks, "the facts state the disk chunks");
    shell
        .commit_segments(facts.segments.max)
        .expect("the segments commit");
    assert!(facts.host_chunks >= 58, "{} host chunks", facts.host_chunks);
    let shape = |id| -> SlotShape {
        facts
            .shapes
            .iter()
            .find(|shape| shape.id == id)
            .cloned()
            .unwrap_or_else(|| panic!("the hybrid states shape {id}"))
    };
    let (list, fixed, rs) = (
        shape(LIST),
        shape(FIXED),
        shape(engine_gpu::store::layout::RS),
    );
    // Slots `n` of chunk `chunk` (device chunk 0 is the null chunk; the
    // test fires nothing, so any other chunk is the test's).
    let refs = |shape: &SlotShape, chunk: u32, n: u32| -> Vec<u32> {
        assert!(shape.pinned.is_none(), "shape {} shares the pool", shape.id);
        assert!(
            n <= shape.slots_per_chunk,
            "shape {} holds {n} a chunk",
            shape.id
        );
        (0..n).map(|i| chunk * shape.slots_per_chunk + i).collect()
    };
    // A slot's words, every one's low half set: no bf16 (nor a narrowing
    // of f32 to one) keeps them. Some are NaNs with payloads, some
    // denormals.
    let pattern = |seed: u32, bytes: u64| -> Vec<u8> {
        (0..bytes / 4)
            .flat_map(|at| {
                let x = (at as u32 ^ seed.wrapping_mul(0x9e37_79b9)).wrapping_mul(0x85eb_ca6b);
                let word = match at % 7 {
                    0 => 0x7fc0_0000 | (x & 0x003f_ffff) | 1,
                    1 => x & 0x007f_ffff | 1,
                    _ => x ^ (x >> 15) | 0x0001_0001,
                };
                word.to_le_bytes()
            })
            .collect()
    };
    let write = |shell: &Shell, id, slot: u32, bytes: &[u8]| {
        let (buffer, at) = shell.pools().slot_place(id, slot).expect("a slot's place");
        buffer
            .write_shared(at, bytes)
            .expect("the slot takes its bytes");
    };
    let read = |shell: &mut Shell, id, slot: u32, len: u64| -> Vec<u8> {
        shell.settle_copies().expect("every copy settles");
        let (buffer, at) = shell.pools().slot_place(id, slot).expect("a slot's place");
        let mut out = vec![0u8; len as usize];
        buffer.read(at, &mut out).expect("the slot reads");
        out
    };
    let copy = |id, src, dst, slots: Vec<(u32, u32)>| SlotCopy {
        shape: id,
        src,
        dst,
        slots,
        runs: Vec::new(),
    };
    let zip = |a: &[u32], b: &[u32]| -> Vec<(u32, u32)> {
        a.iter().copied().zip(b.iter().copied()).collect()
    };
    let check = |what: &str, got: &[u8], want: &[u8]| {
        assert_eq!(got.len(), want.len(), "{what}: length");
        if let Some(at) = got.iter().zip(want).position(|(a, b)| a != b) {
            panic!(
                "{what}: byte {at} of {} differs ({:#04x} for {:#04x}); {} bytes differ",
                want.len(),
                got[at],
                want[at],
                got.iter().zip(want).filter(|(a, b)| a != b).count()
            );
        }
    };

    // Device -> disk -> device: two sets, parked in one copy.
    let n = list.slots_per_chunk.min(3);
    let set_a = refs(&list, 1, n);
    let set_b: Vec<u32> = refs(&list, 2, n.min(2));
    let mut held = Vec::new();
    for (k, &slot) in set_a.iter().chain(&set_b).enumerate() {
        let bytes = pattern(100 + k as u32, list.slot_bytes);
        write(&shell, LIST, slot, &bytes);
        held.push(bytes);
    }
    let parked: Vec<u32> = set_a.iter().chain(&set_b).copied().collect();
    // Disk slots out of order, past the first.
    let disk_refs: Vec<u32> = (0..parked.len() as u32).rev().map(|i| 3 + 2 * i).collect();
    shell
        .copy_slots(&copy(LIST, device, disk, zip(&parked, &disk_refs)))
        .expect("both sets park to disk");
    // Their pages go to another owner, which writes them.
    for &slot in &parked {
        write(&shell, LIST, slot, &vec![0xa5; list.slot_bytes as usize]);
    }
    // Restored into other pages, in another order: two restores back to
    // back, one a set each.
    let (from_a, from_b) = disk_refs.split_at(set_a.len());
    let into_a: Vec<u32> = refs(&list, 3, n).into_iter().rev().collect();
    let into_b: Vec<u32> = refs(&list, 4, n.min(2));
    shell
        .copy_slots(&copy(LIST, disk, device, zip(from_a, &into_a)))
        .expect("set a restores");
    shell
        .copy_slots(&copy(LIST, disk, device, zip(from_b, &into_b)))
        .expect("set b restores");
    for (k, &slot) in into_a.iter().chain(&into_b).enumerate() {
        let got = read(&mut shell, LIST, slot, list.slot_bytes);
        check(
            &format!("kv page {k} device -> disk -> device"),
            &got,
            &held[k],
        );
    }
    // Both sets back in one copy too, onto the pages they left.
    shell
        .copy_slots(&copy(LIST, disk, device, zip(&disk_refs, &parked)))
        .expect("both sets restore in one copy");
    for (k, &slot) in parked.iter().enumerate() {
        let got = read(&mut shell, LIST, slot, list.slot_bytes);
        check(&format!("kv page {k} restored in one copy"), &got, &held[k]);
    }

    // The host tier's way down: host -> disk, then disk -> device and
    // disk -> host.
    let host_in: Vec<u32> = (0..parked.len() as u32).collect();
    let seeded: Vec<u8> = (0..parked.len() as u32)
        .flat_map(|k| pattern(200 + k, list.slot_bytes))
        .collect();
    shell
        .write_tier_slots(&[(LIST, host_in.clone())], &seeded)
        .expect("the host slots take the pattern");
    let spilled: Vec<u32> = (0..parked.len() as u32).map(|i| 20 + 3 * i).rev().collect();
    shell
        .copy_slots(&copy(LIST, host, disk, zip(&host_in, &spilled)))
        .expect("the host slots go on down to disk");
    // The host slots go back and are written over.
    shell
        .write_tier_slots(&[(LIST, host_in.clone())], &vec![0x3c; seeded.len()])
        .expect("the host slots are written over");
    let into: Vec<u32> = refs(&list, 5, n)
        .into_iter()
        .chain(refs(&list, 6, n.min(2)))
        .collect();
    shell
        .copy_slots(&copy(LIST, disk, device, zip(&spilled, &into)))
        .expect("the spilled pages come up to the device");
    let page = list.slot_bytes as usize;
    for (k, &slot) in into.iter().enumerate() {
        let got = read(&mut shell, LIST, slot, list.slot_bytes);
        check(
            &format!("kv page {k} host -> disk -> device"),
            &got,
            &seeded[k * page..(k + 1) * page],
        );
    }
    let host_back: Vec<u32> = (0..parked.len() as u32).map(|i| 40 + i).collect();
    shell
        .copy_slots(&copy(LIST, disk, host, zip(&spilled, &host_back)))
        .expect("the spilled pages come up to the host");
    let got = shell
        .read_tier_slots(&[(LIST, host_back)])
        .expect("the host slots read");
    check("kv pages host -> disk -> host", &got, &seeded);

    // The hybrid's records and rs rows, which park with the kv pages: to the
    // host tier and to disk, and from the host tier down to disk.
    for (k, shape) in [fixed, rs].into_iter().enumerate() {
        let id = shape.id;
        let per = shape.slots_per_chunk;
        let m = per.min(2);
        for (t, tier) in [host, disk].into_iter().enumerate() {
            let chunk = 8 + 8 * k as u32 + 4 * t as u32;
            let (a, b) = (refs(&shape, chunk, m), refs(&shape, chunk + 1, m));
            let sets: Vec<u32> = a.iter().chain(&b).copied().collect();
            let held: Vec<Vec<u8>> = sets
                .iter()
                .enumerate()
                .map(|(i, &slot)| {
                    let seed = 300 + 32 * k as u32 + 16 * t as u32 + i as u32;
                    let bytes = pattern(seed, shape.slot_bytes);
                    write(&shell, id, slot, &bytes);
                    bytes
                })
                .collect();
            // Tier refs out of order, in chunks of their own.
            let base = if tier == host { 50 } else { 4 } + 4 * k as u32;
            let tier_refs: Vec<u32> = (0..sets.len() as u32)
                .rev()
                .map(|i| base * per + i)
                .collect();
            shell
                .copy_slots(&copy(id, device, tier, zip(&sets, &tier_refs)))
                .expect("both sets park");
            for &slot in &sets {
                write(&shell, id, slot, &vec![0x5a; shape.slot_bytes as usize]);
            }
            let into: Vec<u32> = refs(&shape, chunk + 2, m)
                .into_iter()
                .rev()
                .chain(refs(&shape, chunk + 3, m))
                .collect();
            let (tier_a, tier_b) = tier_refs.split_at(a.len());
            let (into_a, into_b) = into.split_at(a.len());
            shell
                .copy_slots(&copy(id, tier, device, zip(tier_a, into_a)))
                .expect("set a restores");
            shell
                .copy_slots(&copy(id, tier, device, zip(tier_b, into_b)))
                .expect("set b restores");
            for (i, &slot) in into.iter().enumerate() {
                let got = read(&mut shell, id, slot, shape.slot_bytes);
                check(
                    &format!("shape {id} slot {i} device -> {tier:?} -> device"),
                    &got,
                    &held[i],
                );
            }
        }
        // Host -> disk -> host.
        let host_in: Vec<u32> = (0..2).map(|i| (50 + 4 * k as u32) * per + i).collect();
        let seeded: Vec<u8> = (0..2)
            .flat_map(|i| pattern(500 + 2 * k as u32 + i, shape.slot_bytes))
            .collect();
        shell
            .write_tier_slots(&[(id, host_in.clone())], &seeded)
            .expect("the host slots take the pattern");
        let down: Vec<u32> = (0..2).rev().map(|i| (5 + 4 * k as u32) * per + i).collect();
        shell
            .copy_slots(&copy(id, host, disk, zip(&host_in, &down)))
            .expect("the host slots go down to disk");
        shell
            .write_tier_slots(&[(id, host_in.clone())], &vec![0x3c; seeded.len()])
            .expect("the host slots are written over");
        let back: Vec<u32> = (0..2).map(|i| (51 + 4 * k as u32) * per + i).collect();
        shell
            .copy_slots(&copy(id, disk, host, zip(&down, &back)))
            .expect("the disk slots come up to the host");
        let got = shell
            .read_tier_slots(&[(id, back)])
            .expect("the host slots read");
        check(&format!("shape {id} host -> disk -> host"), &got, &seeded);
    }
    shell.settle_copies().expect("every copy settles");
    drop(shell);
    let _ = std::fs::remove_dir_all(&dir);
}


/// A never-written cell of the pool reads back as NaN: bf16 `0x7fc0` in each
/// half of a word, itself an f32 NaN. A kernel that reads such a cell and
/// weighs it by a zero (`0 * NaN` is NaN) answers NaN; one that leaves it out
/// by its bound answers as over the zeroes a fresh pool holds.
const POISON: u32 = 0x7fc0_7fc0;

/// Every byte of `shell`'s pool, all its segments committed, set to
/// `pattern` a word ([`POISON`], or 0): the kv pages past the rows a lane
/// writes, the recurrent state, a fire's arena and its kernels' scratch, as
/// recycled memory may hold anything.
fn fill(shell: &mut Shell, pattern: u32) {
    use engine_gpu::backend::Buffer as _;
    shell.drain().expect("every frame lands first");
    let facts = shell.pool_chunk_facts();
    shell
        .commit_segments(facts.segments.max)
        .expect("the segments commit");
    const STRIPE: usize = 8 << 20;
    let stripe: Vec<u8> = std::iter::repeat_n(pattern.to_le_bytes(), STRIPE / 4)
        .flatten()
        .collect();
    for buffer in shell.pools().segment_buffers() {
        let bytes = buffer.bytes();
        let mut at = 0;
        while at < bytes {
            let len = (bytes - at).min(STRIPE as u64);
            buffer
                .write_shared(at, &stripe[..len as usize])
                .expect("the fill lands");
            at += len;
        }
        // The fill reads back at the segment's ends.
        for at in [0, bytes - 4] {
            let mut word = [0u8; 4];
            buffer.read(at, &mut word).expect("the fill reads back");
            assert_eq!(u32::from_le_bytes(word), pattern, "the fill lands at byte {at}");
        }
    }
}

/// Whether two logit rows of one prompt agree: the same argmax, and every
/// logit within a hair of the row's scale (lanes at other offsets of a fire
/// may round apart, none may read another's cells).
fn agree(a: &[f32], b: &[f32]) -> bool {
    let scale = a.iter().fold(0f32, |top, v| top.max(v.abs())).max(1.0);
    a.len() == b.len()
        && argmax(a) == argmax(b)
        && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-2 * scale)
}

/// The seats a poisoned load holds: the widest case's six lanes, and two
/// more each round's lanes move onto.
const POISON_SEATS: usize = 8;

/// A prefill over a pool whose every never-written cell is NaN, for each
/// prompt length (none a whole page or tile of keys: 33 and 65 one key
/// past one, 31 one short) on one lane and on four lanes of one prompt, and
/// four lanes of a 35-row prompt beside two other prompts; each then
/// decodes. Every lane's logits finite, the lanes of one prompt alike, and
/// alike to the same fire over a pool of zeroes. A round runs over the
/// pool poisoned afresh with its seats opened, the first on seats `0..`,
/// the second each lane moved onto a seat another held, so its pages and
/// state are recycled ones holding NaN wherever nothing is written.
fn poisoned(model: &str) {
    let fixture = fixture(model, None);
    let Some(mut shell) = load_lanes(&fixture, PAGE, CONTEXT, POISON_SEATS as u32, 256) else {
        return;
    };
    // Four lanes a prompt where a fire holds their rows (three of 65).
    let most = shell.budget().max_tokens as usize;
    for len in [1u32, 31, 33, 35, 47, 65] {
        for same in [1usize, 4.min(most / len as usize)] {
            poisoned_case(&mut shell, &fixture.trace, model, len, same, 0);
        }
    }
    poisoned_case(&mut shell, &fixture.trace, model, 35, 4, 2);
    // The poison reaches what a fire reads: a decode over a lane's held
    // keys answers otherwise once they are poisoned under it (NaN, or the
    // splits whose keys it poisoned left out of the fold).
    fill(&mut shell, 0);
    shell.open(0).expect("the seat opens");
    let prompt: Vec<u32> = (0..33u32).map(|at| (at * 37 + 11) % 256).collect();
    let lane = |tokens, held| Seated {
        held: Some(held),
        ..Seated::of(Lane {
            slot: 0,
            word: word(&fixture.trace, 1),
            tokens,
        })
    };
    shell
        .fire_seated(&[Seated {
            held: Some(0),
            ..Seated::of(Lane {
                slot: 0,
                word: word(&fixture.trace, 33),
                tokens: &prompt,
            })
        }])
        .expect("the prefill lands");
    let token = [1u32];
    let held = shell
        .fire_seated(&[lane(&token[..], 33)])
        .expect("the decode lands")
        .swap_remove(0);
    fill(&mut shell, POISON);
    let poisoned = shell
        .fire_seated(&[lane(&token[..], 33)])
        .expect("the decode over poisoned keys lands")
        .swap_remove(0);
    assert!(
        held.iter().zip(&poisoned).any(|(a, b)| a.to_bits() != b.to_bits()),
        "{model}: a decode over poisoned held keys answers as over its own: the poison missed \
         the pool"
    );
}

/// [`poisoned`]'s rounds of `same` lanes of one `len`-row prompt and
/// `others` lanes of other prompts.
fn poisoned_case(shell: &mut Shell, trace: &Trace, model: &str, len: u32, same: usize, others: usize) {
    const STEPS: usize = 3;
    let lanes = same + others;
    let case = format!("{model} {same}x{len}+{others}");
    let shared: Vec<u32> = (0..len).map(|at| (at * 37 + 11) % 256).collect();
    let prompts: Vec<Vec<u32>> = (0..lanes)
        .map(|lane| match lane - same.min(lane) {
            _ if lane < same => shared.clone(),
            0 => (0..61u32).map(|at| (at * 53 + 5) % 256).collect(),
            _ => (0..23u32).map(|at| (at * 29 + 3) % 256).collect(),
        })
        .collect();
    let check = |what: &str, rows: &[Vec<f32>]| {
        for (lane, row) in rows.iter().enumerate() {
            let bad = row.iter().filter(|v| !v.is_finite()).count();
            assert!(
                bad == 0,
                "{case} {what}: lane {lane} has {bad} of {} logits not finite",
                row.len()
            );
        }
        for lane in 1..same {
            assert!(
                agree(&rows[0], &rows[lane]),
                "{case} {what}: lane {lane} answers otherwise than lane 0 on the same prompt \
                 (argmax {} against {})",
                argmax(&rows[lane]),
                argmax(&rows[0])
            );
        }
    };
    let round = |shell: &mut Shell, pattern: u32, seat: &dyn Fn(usize) -> u32| -> Vec<Vec<f32>> {
        fill(shell, pattern);
        for lane in 0..lanes {
            shell.open(seat(lane)).expect("the seat opens");
        }
        let seated: Vec<Seated<'_>> = (0..lanes)
            .map(|lane| Seated {
                held: Some(0),
                ..Seated::of(Lane {
                    slot: seat(lane),
                    word: word(trace, prompts[lane].len() as u32),
                    tokens: &prompts[lane],
                })
            })
            .collect();
        let got = shell.fire_seated(&seated).expect("the prefill lands");
        check("prefill", &got);
        let mut held: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
        let mut fed: Vec<Vec<u32>> = got.iter().map(|row| vec![argmax(row)]).collect();
        let mut rows = got;
        for step in 0..STEPS {
            let seated: Vec<Seated<'_>> = (0..lanes)
                .map(|lane| Seated {
                    held: Some(held[lane]),
                    ..Seated::of(Lane {
                        slot: seat(lane),
                        word: word(trace, 1),
                        tokens: &fed[lane],
                    })
                })
                .collect();
            let got = shell.fire_seated(&seated).expect("the decode lands");
            check(&format!("decode {step}"), &got);
            for lane in 0..lanes {
                held[lane] += 1;
                fed[lane] = vec![argmax(&got[lane])];
            }
            rows.extend(got);
        }
        rows
    };
    let zeroes = round(shell, 0, &|lane| lane as u32);
    let fresh = round(shell, POISON, &|lane| lane as u32);
    // Each lane on a seat another lane (or none) held.
    let moved = |lane: usize| ((lane + 2) % POISON_SEATS) as u32;
    let recycled = round(shell, POISON, &moved);
    for (what, rows) in [("fresh", &fresh), ("recycled", &recycled)] {
        for (at, (a, b)) in zeroes.iter().zip(rows).enumerate() {
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "{case}: row {at} over a poisoned pool ({what} seats) answers otherwise than \
                 over zeroes (argmax {} against {})",
                argmax(b),
                argmax(a)
            );
        }
    }
    eprintln!("{case}: every lane finite and as over zeroes, fresh and recycled");
}

#[test]
fn a_dense_miniature_over_a_poisoned_pool_answers_finite() {
    poisoned("qwen3-micro-text");
}

#[test]
fn a_hybrid_miniature_over_a_poisoned_pool_answers_finite() {
    poisoned("qwen35-tiny");
}

#[test]
fn a_windowed_miniature_over_a_poisoned_pool_answers_finite() {
    poisoned("gemma4-e4b-mini-l1");
}

