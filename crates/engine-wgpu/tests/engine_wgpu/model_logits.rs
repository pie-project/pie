//! Whole miniature models served by the shell, end to end: two lanes prefill
//! alone and then decode together, every logit row kept. The rows are
//! written bit for bit to `$PIE_WGPU_LOGITS/<model>.bin` when that is set, so
//! a build can be held against another one (the chunk pool against the
//! per-row slabs it replaced) byte for byte.

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine_wgpu::serve::Seated;
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
    let device = engine_wgpu::api::DeviceBoot::default();
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
