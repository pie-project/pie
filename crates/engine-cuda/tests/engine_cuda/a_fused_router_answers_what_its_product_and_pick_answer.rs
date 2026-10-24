//! A routed row whose router product and top-k the compiler folds into one
//! `fused.matmul_moe_route` answers what the row answers with the rule off.
//! Off a tcgen05 chip (an L40S) the fused kernel declines every fire, so the
//! op runs the product into its logits and the pick over them: the same
//! launches as the pair, the same logits to the bit.

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine_cuda::{Knobs, Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{Fused, Linear, Operation, ParamSource, Trace};

const CEILING: u32 = 256;
const PAGE: u32 = 16;

/// Routed rows of bf16 planes: Qwen3-MoE's softmax, GLM-4.5-Air's biased
/// sigmoid and DeepSeek-V4's biased sqrt-softplus (its hashed layers apart).
const ROWS: [&str; 3] = [
    "qwen3-30b-a3b-mini-bf16-kv-bf16",
    "glm45-air-mini-bf16-kv-bf16",
    "dsv4-flash-mini-bf16-kv-bf16",
];

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

fn write_checkpoint(path: &Path, trace: &Trace) {
    let mut writer = ztensor::Writer::create(path).expect("the checkpoint opens");
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    // A hashed route's table names experts a token, each one of the
    // fewest experts any route of the row picks over.
    let experts = trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Linear(Linear::MoeRoute { experts, .. }) => Some(u64::from(*experts)),
            _ => None,
        })
        .min()
        .unwrap_or(1);
    for param in planes {
        if param.dtype == Dtype::I64 {
            let picks = param.shape.last().copied().unwrap_or(1);
            let count: u64 = param.shape.iter().product();
            let bytes: Vec<u8> = (0..count)
                .flat_map(|at| (((at / picks) * 3 + at % picks) % experts).to_le_bytes())
                .collect();
            writer
                .add(
                    param.name.as_str(),
                    param.shape.clone(),
                    ztensor::Leaf::I64,
                    &bytes,
                )
                .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
            continue;
        }
        let plane = values(&param.name, param.shape.iter().product::<u64>() as usize);
        let (leaf, bytes): (ztensor::Leaf, Vec<u8>) = match param.dtype {
            Dtype::Bf16 => (
                ztensor::Leaf::BF16,
                plane.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            Dtype::F32 => (
                ztensor::Leaf::F32,
                plane
                    .iter()
                    .flat_map(|v| f32::from_bits(u32::from(*v) << 16).to_le_bytes())
                    .collect(),
            ),
            other => panic!(
                "`{}` is {other:?}, which this fixture does not write",
                param.name
            ),
        };
        writer
            .add(param.name.as_str(), param.shape.clone(), leaf, &bytes)
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer.finish().expect("the checkpoint closes");
}

/// The row's logits over `prompt`, the router rule on or off.
fn logits(trace: &Trace, container: &Path, fused: bool, prompt: &[u32]) -> Vec<f32> {
    let source = ztensor::Source::open(container).expect("the fixture opens");
    let contract: ModelContract =
        poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
            .expect("a container of the plan's own planes is read by its own names");
    drop(source);
    let chip = engine_cuda::chip::profile(&format!("[fuse]\nmatmul_moe_route = {fused}\n"));
    let staged = Shell::stage(engine_cuda::Boot {
        trace: trace.clone(),
        contract: &contract,
        checkpoint: container,
        budget: Budget::new(1, CEILING),
        patches: None,
        voxels: None,
        page_size: PAGE,
        context: CEILING,
        slots: 1,
        kv_budget: None,
        memory_share: engine::load::DEFAULT_DEVICE_MEMORY_SHARE,
        ordinal: 0,
        knobs: Knobs {
            chip: Some(chip),
            ..Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the row loads");
    let mut shell = staged.finish().expect("the row arms");
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

fn check(name: &str) {
    let row =
        poem_compiler::catalog::deployment(name).unwrap_or_else(|| panic!("`{name}` is listed"));
    let trace = row.trace(Platform::Cuda);
    let formed = poem_compiler::fuse::fuse(trace.clone(), engine_cuda::FUSIONS.iter())
        .nodes
        .iter()
        .filter(|node| matches!(node.op, Operation::Fused(Fused::MatmulMoeRoute { .. })))
        .count();
    assert!(formed > 0, "{name}: no router folds");

    let dir = std::env::temp_dir().join(format!(
        "pie-router-{}-{}",
        name.replace('.', "_"),
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the scratch directory");
    let _scratch = Scratch(dir.clone());
    let container = dir.join("model.zt");
    write_checkpoint(&container, &trace);

    let vocab = row.model.vocab.max(1);
    // A prefill's rows and a decode's one.
    for count in [41u32, 1] {
        let prompt: Vec<u32> = (0..count).map(|t| (t * 7 + 11) % vocab.min(4096)).collect();
        let fused = logits(&trace, &container, true, &prompt);
        let apart = logits(&trace, &container, false, &prompt);
        let spread = apart.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - apart.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            spread > 1e-3,
            "{name}: the logits are a rectangle nothing wrote"
        );
        assert!(
            fused.iter().all(|v| v.is_finite()),
            "{name}: a non-finite logit"
        );
        eprintln!("{name}: {formed} routers fold, {count} rows, spread {spread}");
        assert_eq!(
            fused, apart,
            "{name} over {count} rows: the fused router answers apart"
        );
    }
}

#[test]
fn a_fused_router_answers_what_its_product_and_pick_answer() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    let failed: Vec<String> = ROWS
        .iter()
        .filter_map(|name| {
            std::panic::catch_unwind(|| check(name)).err().map(|why| {
                let why = why
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| why.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_default();
                format!("{name}: {why}")
            })
        })
        .collect();
    assert!(failed.is_empty(), "{}", failed.join("\n"));
}
