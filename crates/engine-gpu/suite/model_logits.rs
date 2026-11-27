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
    // trace alone), so every run shares one directory: a copy per process or per
    // test would be gigabytes each. `$PIE_VK_FIXTURES` (`$PIE_WGPU_FIXTURES`) names
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

#[allow(
    clippy::disallowed_methods,
    reason = "the lock guards no state, so a holder that panicked left none half-written"
)]
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
    let device = DeviceBoot {
        chunk_bytes: chunk,
        ..crate::gpu::device_boot()
    };
    load_on(
        fixture,
        page,
        context,
        &device,
        residency,
        engine::runahead::Runahead::F1,
    )
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
    load_booted(
        fixture,
        page,
        context,
        &device,
        residency,
        engine::runahead::Runahead::F1,
    )
}

/// [`load_with`] on the device `device` boots, running ahead by `runahead`
/// (an arm a frame in flight).
pub fn load_on(
    fixture: &Fixture,
    page: u32,
    context: u32,
    device: &DeviceBoot,
    residency: engine::load::Residency,
    runahead: engine::runahead::Runahead,
) -> Option<Shell> {
    let _alone = loading();
    load_booted(fixture, page, context, device, residency, runahead)
}

/// [`load_on`], [`LOADING`] held by the caller.
fn load_booted(
    fixture: &Fixture,
    page: u32,
    context: u32,
    device: &DeviceBoot,
    residency: engine::load::Residency,
    runahead: engine::runahead::Runahead,
) -> Option<Shell> {
    let residency = allowed(device, residency)?;
    Some(
        Shell::load(Boot {
            runahead,
            ..boot(fixture, page, context, device, residency)
        })
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

/// Lane `slot` seat-addressed, `held` rows before `tokens`.
fn seat<'a>(trace: &Trace, slot: u32, held: u32, tokens: &'a [u32]) -> Seated<'a> {
    Seated {
        held: Some(held),
        ..Seated::of(Lane {
            slot,
            word: word(trace, tokens.len() as u32),
            tokens,
        })
    }
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

/// Every fire is served from what the load built: on each family, at every
/// rung the load admits (one lane of the rung's rows, every seat sharing
/// them, then a decode of those seats), and on a hybrid the fires that
/// buffer their recurrent rows and commit them at a fold (kernels no folding
/// fire asks for), no pipeline is built after the load.
#[test]
fn every_fire_is_served_from_what_the_load_built() {
    use engine::fire::{FoldLen, Lineage, RsVerb};
    let tokens: Vec<u32> = (0..CONTEXT).map(|at| (at * 37 + 11) % 256).collect();
    for model in ["qwen3-micro-text", "qwen35-tiny", "gemma4-e4b-mini-l1"] {
        let fixture = fixture(model, None);
        let Some(mut shell) = load(&fixture, CONTEXT) else {
            return;
        };
        let built = shell.compiled();
        assert_eq!(
            built,
            shell.named().kernels.len() as u64,
            "{model}: the load built other pipelines than the ones its walks named"
        );
        let fire = |shell: &mut Shell, seated: &[Seated<'_>], built: u64, what: &str| {
            let got = shell
                .fire_seated(seated)
                .unwrap_or_else(|why| panic!("{model}: {what} fire(s): {why}"));
            assert!(got.iter().all(|row| row.iter().all(|v| v.is_finite())));
            assert_eq!(
                shell.compiled(),
                built,
                "{model}: {what} built a pipeline the load's walks did not name"
            );
        };
        let lane = |slot: u32, held: u32, rows: usize| {
            seat(&fixture.trace, slot, held, &tokens[held as usize..][..rows])
        };
        let rungs: Vec<u32> = shell
            .pool_chunk_facts()
            .act_chunks
            .iter()
            .map(|&(rung, _)| rung)
            .collect();
        assert!(!rungs.is_empty(), "{model}: the load admits a rung");
        for &rung in &rungs {
            let rung = rung.clamp(1, CONTEXT);
            for lanes in [1, SEATS.min(rung)] {
                let each = (rung / lanes) as usize;
                for (held, rows) in [(0, each), (each as u32, 1)] {
                    if held + rows as u32 > CONTEXT {
                        continue;
                    }
                    let seated: Vec<Seated<'_>> = (0..lanes).map(|l| lane(l, held, rows)).collect();
                    fire(
                        &mut shell,
                        &seated,
                        built,
                        &format!("{lanes} x {rows} rows at rung {rung}"),
                    );
                }
            }
        }
        if shell.pools().rs_rows().is_none() {
            continue;
        }
        // A prefill that folds, two rows buffered, then one row that
        // replays them and folds them, on a load of their own.
        drop(shell);
        let mut shell = load(&fixture, CONTEXT).expect("loaded before");
        let built = shell.compiled();
        let per_page = shell
            .pools()
            .rs_rows()
            .expect("a hybrid's rs list")
            .tokens
            .max(1);
        shell.open(0).expect("the slot opens");
        let buffer = |folded: u32, fold: u32| RsVerb::Buffer {
            pages: vec![0; 8u32.div_ceil(per_page) as usize],
            base: 0,
            folded,
            fold: FoldLen::Host(fold),
            lineage: Lineage::Set,
        };
        for (held, rows, rs) in [
            (0u32, 4usize, RsVerb::Fold),
            (4, 2, buffer(4, 0)),
            (6, 1, buffer(4, 2)),
        ] {
            let seated = [Seated {
                rs: &rs,
                ..lane(0, held, rows)
            }];
            fire(
                &mut shell,
                &seated,
                built,
                &format!("{rows} row(s) held {held} under {rs:?}"),
            );
        }
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
            .map(|lane| seat(&fixture.trace, lane, 4, &tokens[lane as usize..][..1]))
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
                shell
                    .commit_segments(committed)
                    .expect("the segments commit again");
            }
        }
        assert_eq!(decode(&mut shell, 1), first.0, "round {round}: one lane");
        assert_eq!(
            decode(&mut shell, wide),
            first.1,
            "round {round}: {wide} lanes"
        );
    }
}

/// Every byte a served miniature holds on the device is in the device's
/// ledger under an owner, the ledger within what the device has, and
/// serving takes none of the driver's own past what the load left it.
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
    serve_seats(&mut shell, &fixture.trace, 2);
    let books = shell.device().books();
    let ledger = books.ledger();
    assert_eq!(ledger.check(), Ok(()), "{ledger}");
    assert_eq!(books.audit().gap(), 0, "{ledger}");
    let served = shell.device().outside();
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

/// A park to the host tier is told landed only once it has read its device
/// slots: overwritten from the host the moment the park is told landed,
/// outside any frame, the host tier still holds what they held.
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
    let (seed, parked): (Vec<u32>, Vec<u32>) = ((0..n).collect(), (n..2 * n).collect());
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
    }
    shell.settle_copies().expect("every copy settles");
}

/// A masked beam lane beside three plain decodes, in one fire: each lane
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
                mask,
                ..seat(&trace, slot, held, tokens)
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
    let mut same = 0u32;
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
        let batched = fire(&mut shell, &rows(true));
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
                assert!(
                    parting.tie,
                    "lane {lane} (beam: {}) parts at step {step} on no near-tie",
                    lane == beam
                );
                parted[lane] = true;
            } else if !parted[lane] {
                same += 1;
            }
            last[lane] = [tb as u32, ta as u32];
        }
    }
    assert!(same > 0, "no lane step answered as it does alone");
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
    let windowed = shell.pools().rings().is_some();
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
                folded: have,
                window: folds_all(windowed, have, tokens.len() as u32),
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

#[test]
fn a_scattered_windowed_pool_serves_the_same_logits() {
    scattered("gemma4-e4b-mini-l1");
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

/// A lane folding every row it brings this fire (`Lane::window` of a
/// windowed load; `None` elsewhere): its buffer table starts at the folded
/// index after the fire, and the fire leaves no row in it.
pub fn folds_all(windowed: bool, have: u32, rows: u32) -> Option<engine::fire::WindowFold> {
    windowed.then(|| engine::fire::WindowFold {
        base: (have + rows) / PAGE * PAGE,
        fold: engine::fire::FoldLen::Host(rows),
        lineage: engine::fire::Lineage::Set,
    })
}

/// Logit rows alike within a bf16-derived bound: every logit within 2^-6
/// of its row's largest magnitude (a few bf16 ulps of the rows that feed
/// it), and the same argmax, or one that ties it within that bound.
pub fn near(what: &str, got: &[Vec<f32>], want: &[Vec<f32>]) {
    assert_eq!(got.len(), want.len(), "{what}: rows");
    let mut worst = 0f32;
    for (at, (got, want)) in got.iter().zip(want).enumerate() {
        assert_eq!(got.len(), want.len(), "{what}: row {at}'s logits");
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
        let err = got
            .iter()
            .zip(want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        worst = worst.max(err / scale);
        assert!(
            err <= scale / 64.0,
            "{what}: row {at} differs by {err} of {scale} (argmax {} against {})",
            argmax(got),
            argmax(want)
        );
        let (g, w) = (argmax(got) as usize, argmax(want) as usize);
        assert!(
            g == w || got[g] - got[w] <= scale / 64.0,
            "{what}: row {at}'s argmax {g} ({}) against {w} ({})",
            got[g],
            got[w]
        );
    }
    eprintln!("{what}: worst {worst:.2e} of a row's largest logit");
}

/// The prompt a windowed check feeds: 184 rows, past a window of 64 twice.
fn windowed_prompt() -> Vec<u32> {
    (0..184).map(|at| (at * 37 + 11) % 256).collect()
}

/// [`load_with`] at a context of 1024 under a device `budget` where stated.
fn load_budget(fixture: &Fixture, page: u32, budget: Option<u64>) -> Option<Shell> {
    let device = engine::Allowance { budget };
    load_with(
        fixture,
        page,
        1024,
        None,
        engine::load::Residency {
            device,
            ..Default::default()
        },
    )
}

/// One seat-addressed lane prefilling `prompt` in fires of 64 rows and
/// decoding `steps` on, each step fed `feed`'s token or else its own
/// argmax: the logit row of each step, and the tokens it fed.
fn windowed_rows(
    fixture: &Fixture,
    page: u32,
    budget: Option<u64>,
    prompt: &[u32],
    steps: usize,
    feed: Option<&[u32]>,
) -> Option<(Vec<Vec<f32>>, Vec<u32>)> {
    let mut shell = load_budget(fixture, page, budget)?;
    let mut have = 0u32;
    let (mut out, mut tokens) = (Vec::new(), Vec::new());
    let mut fed = prompt.to_vec();
    for _ in 0..=steps {
        let mut last = Vec::new();
        for piece in fed.chunks(64) {
            last = shell
                .fire_seated(&[seat(&fixture.trace, 0, have, piece)])
                .expect("the fire lands");
            have += piece.len() as u32;
        }
        let next = feed
            .and_then(|feed| feed.get(tokens.len()).copied())
            .unwrap_or_else(|| argmax(&last[0]));
        tokens.push(next);
        fed = vec![next];
        out.extend(last);
    }
    tokens.pop();
    Some((out, tokens))
}

/// The dense answer to [`windowed_rows`]: the prompt and the first `k`
/// tokens in one fire, for each step `k`, its last row. One fire reads
/// every key from its own rows through the window's mask, so no ring,
/// buffer or fold stands between a row and the keys it reads.
fn dense_rows(fixture: &Fixture, page: u32, prompt: &[u32], tokens: &[u32]) -> Vec<Vec<f32>> {
    let mut shell = load_budget(fixture, page, None).expect("loaded before");
    (0..=tokens.len())
        .map(|k| {
            let rows: Vec<u32> = prompt.iter().chain(&tokens[..k]).copied().collect();
            let got = shell
                .fire_seated(&[seat(&fixture.trace, 0, 0, &rows)])
                .expect("the fire lands");
            got.into_iter().last().expect("a readout row")
        })
        .collect()
}

/// A windowed layer as ring + buffer (CONTRACT-BC 19, 21) answers what
/// one dense fire does: a prefill past its window (narrowed to 64 rows)
/// in fires of 64 (each fire's own rows read from its fire slots and
/// folded into the ring), and 16 decodes reading the ring. A model of a
/// layer only: deeper, a miniature's logits move with how a prompt is cut
/// into fires (bf16 rounding through its random weights, F77).
#[test]
fn a_windowed_layer_answers_what_one_dense_fire_does() {
    const MODEL: &str = "gemma4-e4b-mini-l1";
    let fixture = narrowed(MODEL, None, "64 if win != None else win", "-w64");
    let prompt = windowed_prompt();
    let Some((rows, tokens)) = windowed_rows(&fixture, PAGE, None, &prompt, 16, None) else {
        return;
    };
    let want = dense_rows(&fixture, PAGE, &prompt, &tokens);
    near(
        &format!("{MODEL}: ring + buffer against one dense fire"),
        &rows,
        &want,
    );
}

/// A deep windowed model through its rings, held to the same model with
/// its windows taken out, fed the same rows in the same fires (all inside
/// its window).
fn same_cut_check(model: &str, layers: Option<u32>, page: u32, budget: Option<u64>, steps: usize) {
    let windowed = fixture_in(
        poem_compiler::catalog::package_of(model).expect("the model is listed"),
        model,
        layers,
        "",
    );
    let window = poem_exec::store::kv::declared_window(&windowed.trace).unwrap_or(0);
    let prompt = windowed_prompt();
    assert!(
        window as usize > prompt.len() + steps,
        "{model}: a window past the rows"
    );
    let plain = narrowed(model, layers, "None", "-plain");
    assert_eq!(
        poem_exec::store::kv::declared_window(&plain.trace),
        None,
        "{model}: no window left"
    );
    let Some((rows, tokens)) = windowed_rows(&windowed, page, budget, &prompt, steps, None) else {
        return;
    };
    let (want, _) =
        windowed_rows(&plain, page, budget, &prompt, steps, Some(&tokens)).expect("loaded before");
    near(
        &format!("{model}: ring + buffer against plain attention, fire for fire"),
        &rows,
        &want,
    );
}

/// Windowed layers beside full ones, the buffer's slot the kv page.
#[test]
fn windowed_layers_beside_full_ones_answer_what_plain_attention_does() {
    same_cut_check("gemma4-e4b-mini-l6", None, 32, None, 16);
}

/// Gemma's shared tail: its layers write no kv and read their owner
/// layers' rings (the 30-layer miniature: six tail layers over 24 owners).
#[test]
fn a_shared_tail_answers_what_plain_attention_does() {
    same_cut_check("gemma4-e4b-mini-l30", None, PAGE, Some(20 << 30), 2);
}

/// `model` (cut to `layers`) with its sliding layers' windows as `rule`
/// states them (a Starlark expression of `win` and the layer `l`), in its
/// attention and its readout tail's alike, or none (`"None"`).
fn narrowed(model: &str, layers: Option<u32>, rule: &str, variant: &str) -> Fixture {
    let package = poem_compiler::catalog::package_of(model).expect("the model is listed");
    let (_, text) = package
        .files()
        .find(|(_, text)| text.contains("def forward(m, inputs):"))
        .expect("the package states its forward");
    let start = text.find("def forward(m, inputs):").expect("found above");
    let end = start
        + text[start..]
            .find("\ndef ")
            .expect("a function after the forward");
    let mut forward = text[start..end].to_string();
    for stated in [
        "        d, kv_heads, win = geometry(m, at.reading)\n",
        "        d, _, win = geometry(m, at.reading)\n",
    ] {
        assert!(
            forward.contains(stated),
            "{model}'s forward reads its geometry: {stated:?}"
        );
        forward = forward.replace(stated, &format!("{stated}        win = {rule}\n"));
    }
    let narrowed = poem_compiler::catalog::replacing(model, "forward", &forward)
        .unwrap_or_else(|why| panic!("the narrowed forward loads: {why}"));
    fixture_in(&narrowed, model, layers, variant)
}

/// Slot refs of one shape `id` the runtime would hand out: slot `k` of
/// the run from chunk `first`, `per` a chunk, strided over each chunk.
fn slot_of(per: u32, first: u32, k: u32) -> u32 {
    let step = [7u32, 11, 13, 17]
        .into_iter()
        .find(|step| !per.is_multiple_of(*step))
        .expect("a step prime to the slots a chunk");
    (first + k / per) * per + (k * step + 3) % per
}

/// The lanes of the ring tests: refs in a run of chunks past the seats'.
struct RingRefs {
    fixed: u32,
    buffer: u32,
    first: u32,
    pieces: u32,
}

impl RingRefs {
    fn of(shell: &mut Shell) -> RingRefs {
        let layout = shell.pool_layout();
        let first = 1 + layout.seat_chunks();
        let refs = RingRefs {
            fixed: layout
                .fixed
                .expect("a windowed load holds records")
                .per_chunk,
            buffer: layout
                .buffer
                .expect("a windowed load holds a window buffer")
                .per_chunk,
            first,
            pieces: layout.pieces,
        };
        let segment = layout.segment_chunks.max(1);
        let most = shell.pool_chunk_facts().segments.max;
        shell
            .commit_segments((first + 16).div_ceil(segment).min(most))
            .expect("the segments commit");
        refs
    }

    /// Record `r`: a piece each from chunks `first ..`.
    fn record(&self, r: u32) -> Vec<u32> {
        (0..self.pieces)
            .map(|p| slot_of(self.fixed, self.first, r * self.pieces + p))
            .collect()
    }

    /// Buffer slot `k` of owner `owner`, in the chunks past the records'.
    fn slot(&self, owner: u32, k: u32) -> u32 {
        slot_of(self.buffer, self.first + 8, owner * 64 + k)
    }
}

/// A lane addressed by slot refs on a windowed load: its record, its
/// buffer table from `base`, and its fold.
struct RingLane<'a> {
    slot: u32,
    record: &'a [u32],
    have: u32,
    folded: u32,
    base: u32,
    fold: u32,
    table: Vec<u32>,
    /// Its fold counted on the device (`FoldLen::Device`, `fold` the count
    /// its program holds at the fire).
    device: bool,
}

fn fire_rings(
    shell: &mut Shell,
    trace: &Trace,
    lanes: &[(&RingLane<'_>, &[u32])],
) -> Vec<Vec<f32>> {
    let tables: Vec<Vec<(engine::chunk::ShapeId, Vec<u32>)>> = lanes
        .iter()
        .map(|(lane, _)| vec![(engine_gpu::store::layout::WINDOW, lane.table.clone())])
        .collect();
    let seated: Vec<Seated<'_>> = lanes
        .iter()
        .zip(&tables)
        .map(|((lane, tokens), table)| Seated {
            held: Some(lane.have),
            refs: true,
            record: Some(lane.record),
            folded: lane.folded,
            windows: table,
            window: Some(engine::fire::WindowFold {
                base: lane.base,
                fold: if lane.device {
                    engine::fire::FoldLen::Device
                } else {
                    engine::fire::FoldLen::Host(lane.fold)
                },
                lineage: engine::fire::Lineage::Set,
            }),
            ..Seated::of(Lane {
                slot: lane.slot,
                word: word(trace, tokens.len() as u32),
                tokens,
            })
        })
        .collect();
    let folds: Vec<engine_gpu::store::frozen::DeviceFold> = lanes
        .iter()
        .map(|(lane, _)| engine_gpu::store::frozen::DeviceFold {
            count: lane.device.then_some(lane.fold),
            reports: 0,
        })
        .collect();
    shell.fire_folded(&seated, &folds).expect("the fire lands")
}

/// The logit rows of `tokens` fed one lane alone, folding every row, the
/// prompt in fires of 200 and then a row a fire.
fn alone(fixture: &Fixture, prompt: &[u32], tokens: &[u32]) -> Vec<Vec<f32>> {
    let mut shell = load_paged(fixture, PAGE, 1024, None).expect("loaded before");
    let mut have = 0u32;
    let mut out = Vec::new();
    for piece in prompt.chunks(200).chain(tokens.chunks(1)) {
        let got = shell
            .fire_seated(&[seat(&fixture.trace, 0, have, piece)])
            .expect("the fire lands");
        have += piece.len() as u32;
        out.extend(got);
    }
    out
}

/// Lane 0 of `record` fed `prompt` in fires of 200, folding every row into
/// its ring: the last fire's logit rows.
fn ring_prefill(shell: &mut Shell, trace: &Trace, record: &[u32], prompt: &[u32]) -> Vec<Vec<f32>> {
    let mut have = 0u32;
    let mut last = Vec::new();
    for piece in prompt.chunks(200) {
        let n = piece.len() as u32;
        let lane = RingLane {
            slot: 0,
            record,
            have,
            folded: have,
            base: (have + n) / PAGE * PAGE,
            fold: n,
            table: Vec::new(),
            device: false,
        };
        last = fire_rings(shell, trace, &[(&lane, piece)]);
        have += n;
    }
    last
}

/// A stop-truncate leaves the ring untouched (CONTRACT-BC 19): a lane
/// prefills 600 rows (all folded), decodes five rows it buffers, folding
/// none, then drops the last three and decodes other rows from there: it
/// answers what a lane fed the kept rows alone answers, so the dropped
/// rows reached no ring place (ring places of rows its window still reads).
#[test]
fn a_stop_truncate_leaves_the_ring_untouched() {
    const MODEL: &str = "gemma4-e4b-mini-l1";
    let fixture = fixture(MODEL, None);
    let Some(mut shell) = load_paged(&fixture, PAGE, 1024, None) else {
        return;
    };
    let refs = RingRefs::of(&mut shell);
    let record = refs.record(0);
    let prompt: Vec<u32> = (0..600u32).map(|at| (at * 53 + 5) % 256).collect();
    let last = ring_prefill(&mut shell, &fixture.trace, &record, &prompt);
    let folded = prompt.len() as u32;
    let base = folded / PAGE * PAGE;
    let buffered = |have: u32| -> RingLane<'_> {
        RingLane {
            slot: 0,
            record: &record,
            have,
            folded,
            base,
            fold: 0,
            table: (0..(have + 1 - base).div_ceil(PAGE))
                .map(|k| refs.slot(0, k))
                .collect(),
            device: false,
        }
    };
    let mut next = argmax(&last[0]);
    let mut tokens = Vec::new();
    for step in 0..5 {
        let got = fire_rings(
            &mut shell,
            &fixture.trace,
            &[(&buffered(folded + step), &[next][..])],
        );
        tokens.push(next);
        next = argmax(&got[0]);
    }
    // Truncated to its first two buffered rows, then other rows.
    tokens.truncate(2);
    let mut rows = Vec::new();
    let mut fed = (tokens[1] + 7) % 256;
    for step in 0..4 {
        let got = fire_rings(
            &mut shell,
            &fixture.trace,
            &[(&buffered(folded + 2 + step), &[fed][..])],
        );
        tokens.push(fed);
        rows.push(got[0].clone());
        fed = argmax(&got[0]);
    }
    let want = alone(&fixture, &prompt, &tokens);
    near(
        &format!("{MODEL}: decoding on past a stop-truncate"),
        &rows,
        &want[want.len() - rows.len()..],
    );
}

/// Every shape a parked set carries (kv list pages, the hybrid's records
/// and its rs-list rows) comes back bit for bit through the tiers below the
/// device: device -> host -> device and device -> disk -> device into other
/// slots, the parked slots written by their next owner between. Words no bf16 holds (every one's low half set, NaN payloads
/// and denormals among them); tier refs in an order not the set's own.
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
    shell
        .seat_disk_kv(Some(64 << 20), Some(&dir), SEATS, CONTEXT)
        .expect("the slot file opens");
    let facts = shell.pool_chunk_facts();
    shell
        .commit_segments(facts.segments.max)
        .expect("the segments commit");
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
    let copy = |id, src, dst, from: &[u32], to: &[u32]| SlotCopy {
        shape: id,
        src,
        dst,
        slots: from.iter().copied().zip(to.iter().copied()).collect(),
        runs: Vec::new(),
    };
    for (k, id) in [LIST, FIXED, engine_gpu::store::layout::RS]
        .into_iter()
        .enumerate()
    {
        let k = k as u32;
        let shape: SlotShape = facts
            .shapes
            .iter()
            .find(|shape| shape.id == id)
            .cloned()
            .unwrap_or_else(|| panic!("the hybrid states shape {id}"));
        let (per, bytes) = (shape.slots_per_chunk, shape.slot_bytes);
        let m = per.min(3);
        // `m` slots of `chunk`, in order or reversed (device chunk 0 is the
        // null chunk; the test fires nothing, so any other is the test's).
        let refs = |chunk: u32, rev: bool| -> Vec<u32> {
            let mut refs: Vec<u32> = (0..m).map(|i| chunk * per + i).collect();
            if rev {
                refs.reverse();
            }
            refs
        };
        for (t, tier) in [host, disk].into_iter().enumerate() {
            let t = t as u32;
            let (from, into) = (
                refs(1 + 4 * k + 2 * t, false),
                refs(2 + 4 * k + 2 * t, true),
            );
            let held: Vec<Vec<u8>> = (0..m)
                .map(|i| pattern(100 + 16 * k + 4 * t + i, bytes))
                .collect();
            for (&slot, bytes) in from.iter().zip(&held) {
                let (buffer, at) = shell.pools().slot_place(id, slot).expect("a slot's place");
                buffer
                    .write_shared(at, bytes)
                    .expect("the slot takes its bytes");
            }
            let parked = refs(if tier == host { 48 + 3 * k } else { 2 + 3 * k }, true);
            shell
                .copy_slots(&copy(id, device, tier, &from, &parked))
                .expect("the set parks");
            // Its slots go to another owner, which writes them.
            for &slot in &from {
                let (buffer, at) = shell.pools().slot_place(id, slot).expect("a slot's place");
                buffer
                    .write_shared(at, &vec![0x5a; bytes as usize])
                    .expect("the next owner writes the slot");
            }
            shell
                .copy_slots(&copy(id, tier, device, &parked, &into))
                .expect("the set restores");
            shell.settle_copies().expect("every copy settles");
            for (i, &slot) in into.iter().enumerate() {
                let (buffer, at) = shell.pools().slot_place(id, slot).expect("a slot's place");
                let mut got = vec![0u8; bytes as usize];
                buffer.read(at, &mut got).expect("the slot reads");
                assert!(
                    got == held[i],
                    "shape {id} slot {i} device -> {tier:?} -> device"
                );
            }
        }
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
            assert_eq!(
                u32::from_le_bytes(word),
                pattern,
                "the fill lands at byte {at}"
            );
        }
    }
}

/// The seats a poisoned load holds: the widest case's six lanes, and two
/// more each round's lanes move onto.
const POISON_SEATS: usize = 8;

/// A prefill over a pool whose every never-written cell is NaN (a lane of
/// 31 rows, one short of a page or tile of keys; four lanes of a 35-row
/// prompt beside two other prompts), each then decoding: every lane's
/// logits finite and the same bits as over a pool of zeroes, on fresh
/// seats and on seats another lane held.
#[test]
fn a_hybrid_miniature_over_a_poisoned_pool_answers_finite() {
    const MODEL: &str = "qwen35-tiny";
    let fixture = fixture(MODEL, None);
    let Some(mut shell) = load_lanes(&fixture, PAGE, CONTEXT, POISON_SEATS as u32, 256) else {
        return;
    };
    for (len, same, others) in [(31, 1, 0), (35, 4, 2)] {
        poisoned_case(&mut shell, &fixture.trace, MODEL, len, same, others);
    }
}

/// The poisoned-pool test's rounds of `same` lanes of one `len`-row prompt and
/// `others` lanes of other prompts, over zeroes and over poison.
fn poisoned_case(
    shell: &mut Shell,
    trace: &Trace,
    model: &str,
    len: u32,
    same: usize,
    others: usize,
) {
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
    };
    let round =
        |shell: &mut Shell, pattern: u32, seat_of: &dyn Fn(usize) -> u32| -> Vec<Vec<f32>> {
            fill(shell, pattern);
            for lane in 0..lanes {
                shell.open(seat_of(lane)).expect("the seat opens");
            }
            let seated: Vec<Seated<'_>> = (0..lanes)
                .map(|lane| seat(trace, seat_of(lane), 0, &prompts[lane]))
                .collect();
            let got = shell.fire_seated(&seated).expect("the prefill lands");
            check("prefill", &got);
            let mut held: Vec<u32> = prompts.iter().map(|p| p.len() as u32).collect();
            let mut fed: Vec<Vec<u32>> = got.iter().map(|row| vec![argmax(row)]).collect();
            let mut rows = got;
            for step in 0..STEPS {
                let seated: Vec<Seated<'_>> = (0..lanes)
                    .map(|lane| seat(trace, seat_of(lane), held[lane], &fed[lane]))
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
}

/// A fold the device counts (`FoldLen::Device`): a lane prefills 600 rows
/// (all folded), decodes five rows it buffers, then fires a row whose
/// program's count folds four of its six (600..604), the rest left: the
/// row it carried at 604 in its buffer, its own at 605 landing there. It
/// decodes on from the folded index the count left, answering as a lane
/// fed its rows alone.
#[test]
fn a_device_counted_fold_folds_what_its_count_says() {
    const MODEL: &str = "gemma4-e4b-mini-l1";
    let fixture = fixture(MODEL, None);
    let Some(mut shell) = load_paged(&fixture, PAGE, 1024, None) else {
        return;
    };
    let refs = RingRefs::of(&mut shell);
    let record = refs.record(0);
    let prompt: Vec<u32> = (0..600u32).map(|at| (at * 29 + 3) % 256).collect();
    let last = ring_prefill(&mut shell, &fixture.trace, &record, &prompt);
    let table = |upto: u32, base: u32| -> Vec<u32> {
        (0..(upto - base).div_ceil(PAGE))
            .map(|k| refs.slot(0, k))
            .collect()
    };
    let mut folded = prompt.len() as u32;
    let mut next = argmax(&last[0]);
    let mut tokens = Vec::new();
    let mut rows = Vec::new();
    for step in 0..10u32 {
        let have = 600 + step;
        // Row 605 folds by the device's count of 4; the rows after fold
        // nothing until then, every row from 609 on all it carries.
        let (fold, device) = match have {
            605 => (4, true),
            h if h >= 609 => (h + 1 - folded, false),
            _ => (0, false),
        };
        let base = folded / PAGE * PAGE;
        let lane = RingLane {
            slot: 0,
            record: &record,
            have,
            folded,
            base,
            fold,
            table: table(have + 1, base),
            device,
        };
        let got = fire_rings(&mut shell, &fixture.trace, &[(&lane, &[next][..])]);
        tokens.push(next);
        rows.push(got[0].clone());
        next = argmax(&got[0]);
        folded += fold;
    }
    let want = alone(&fixture, &prompt, &tokens);
    near(
        &format!("{MODEL}: a device-counted fold"),
        &rows,
        &want[want.len() - rows.len()..],
    );
}

/// [`load_capped`] with kv pages of `page` tokens, under `residency` (its
/// pool's budget, and whether the weights start held).
pub fn load_capped_with(
    fixture: &Fixture,
    page: u32,
    context: u32,
    max_buffer: u64,
    chunk: u64,
    residency: engine::load::Residency,
) -> Option<Shell> {
    let _alone = loading();
    let device = DeviceBoot {
        max_buffer: Some(max_buffer),
        chunk_bytes: Some(chunk),
        ..crate::gpu::device_boot()
    };
    let residency = allowed(&device, residency)?;
    Some(
        Shell::load(boot(fixture, page, context, &device, residency))
            .unwrap_or_else(|why| panic!("the shell loads: {why}")),
    )
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
        std::slice::from_ref(&unstated),
    );
    assert!(
        apart(&one, &unstated) > 1e-3,
        "stating position 0 everywhere rotated the rows as their slots"
    );
}
