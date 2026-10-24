//! A model whose products of one activation the compiler joins (q, k and v
//! into one weight, split after) answers what the same model answers with
//! each product apart: the joined weight is its legs' rows one after the
//! other, the same values. The joined product is wider, so its kernel tiles
//! it otherwise and rounds the last bf16 place apart, over the layers a few
//! of them: on an L40S, 0.057 of a spread of 6.8 for qwen3 0.6b's 28.

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine_cuda::{Lane, Shell};
use poem::{Dtype, Join, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{ParamSource, Trace};

/// A dense model whose package states q, k and v apart.
const ROW: &str = "qwen3-0.6b-bf16-kv-bf16";
const CEILING: u32 = 256;
const PAGE: u32 = 16;

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fnv(name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in name.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// A plane's bf16 values, by its name.
fn values(name: &str, count: usize) -> Vec<u16> {
    let norm = name.ends_with("norm");
    let mut seed = fnv(name);
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let unit = ((seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
            let value = if norm { 1.0 + 0.05 * unit } else { 0.08 * unit };
            (value.to_bits() >> 16) as u16
        })
        .collect()
}

/// `trace`'s planes, a joined one its legs' values (`joins`) in a row.
fn write_checkpoint(path: &Path, trace: &Trace, joins: &[Join], unjoined: &Trace) {
    let mut writer = ztensor::Writer::create(path).expect("the checkpoint opens");
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    let count = |t: &Trace, name: &str| -> usize {
        let param = t
            .params
            .iter()
            .find(|p| p.name == name)
            .expect("a leg's param");
        param.shape.iter().product::<u64>() as usize
    };
    for param in planes {
        assert_eq!(param.dtype, Dtype::Bf16, "`{}` is not bf16", param.name);
        let plane: Vec<u16> = match joins.iter().find(|j| j.name == param.name) {
            Some(join) => join
                .legs
                .iter()
                .flat_map(|leg| values(leg, count(unjoined, leg)))
                .collect(),
            None => values(&param.name, count(trace, &param.name)),
        };
        let bytes: Vec<u8> = plane.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer
            .add(
                param.name.as_str(),
                param.shape.clone(),
                ztensor::Leaf::BF16,
                &bytes,
            )
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer.finish().expect("the checkpoint closes");
}

fn load(trace: &Trace, joins: &[Join], unjoined: &Trace, tag: &str) -> (Shell, Scratch) {
    let dir = std::env::temp_dir().join(format!("pie-pack-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the scratch directory");
    let container = dir.join("model.zt");
    write_checkpoint(&container, trace, joins, unjoined);
    let source = ztensor::Source::open(&container).expect("the fixture opens");
    let contract: ModelContract =
        poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
            .expect("a container of the plan's own planes is read by its own names");
    drop(source);
    let staged = Shell::stage(engine_cuda::Boot {
        trace: trace.clone(),
        contract: &contract,
        checkpoint: &container,
        budget: Budget::new(1, CEILING),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CEILING,
        slots: 1,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: engine_cuda::Knobs::default(),
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the model loads");
    (staged.finish().expect("the model arms"), Scratch(dir))
}

fn logits(trace: &Trace, joins: &[Join], unjoined: &Trace, tag: &str, prompt: &[u32]) -> Vec<f32> {
    let (mut shell, _dir) = load(trace, joins, unjoined, tag);
    shell.open(0).unwrap();
    let word = trace.facts.word(&Request::new(prompt.len() as u32, false));
    let mut out = shell
        .fire(&[Lane {
            slot: 0,
            word,
            tokens: prompt,
        }])
        .unwrap();
    out.remove(0)
}

#[test]
fn a_joined_projection_answers_what_its_legs_answer() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let row = poem_compiler::catalog::deployment(ROW).expect("the row is listed");
    let unjoined = row
        .package
        .trace(&row.model.id, &row.deploy, &row.name, Platform::Cuda)
        .expect("the row traces");
    let joins = poem_compiler::pack::joins(&unjoined);
    assert!(!joins.is_empty(), "the compiler joins the row's q, k and v");
    let joined = row.trace(Platform::Cuda);

    let prompt: Vec<u32> = (0..41).map(|t| (t * 7 + 11) % 4096).collect();
    let apart = logits(&unjoined, &[], &unjoined, "apart", &prompt);
    let together = logits(&joined, &joins, &unjoined, "joined", &prompt);
    assert_eq!(apart.len(), together.len());
    let spread = apart.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - apart.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(spread > 1e-3, "the logits are a rectangle nothing wrote");
    let worst = apart
        .iter()
        .zip(&together)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("joined vs apart: worst gap {worst} over a spread of {spread}");
    assert!(
        worst <= 0.015 * spread,
        "a joined product answers its legs: worst gap {worst} over a spread of {spread}"
    );
    let top = |l: &[f32]| {
        (0..l.len())
            .max_by(|&a, &b| l[a].total_cmp(&l[b]))
            .expect("a vocabulary")
    };
    assert_eq!(
        top(&apart),
        top(&together),
        "the joined model picks the same token"
    );
}
