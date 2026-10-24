//! A routed MLP formed as one mega-MoE (`fused.moe_mega`) that the kernel
//! declines (on any chip short of tcgen05, or past `kernel.moe.mega_until`
//! routes) runs as its legs: the same logits, to the bit, as the model with
//! the fusion off.

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine_cuda::{Lane, Shell};
use poem::{Dtype, Platform, Request};
use poem_compiler::Budget;
use poem_ir::{Fused, Operation, ParamSource, Trace};

/// Rows whose routed MLPs the mega-MoE forms: GLM's bf16 banks under the
/// weighted sum.
const ROWS: [&str; 1] = ["glm45-air-mini-bf16-kv-bf16"];
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

fn write_checkpoint(path: &Path, trace: &Trace) {
    let mut writer = ztensor::Writer::create(path).expect("the checkpoint opens");
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    for param in planes {
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

/// The logits of `prompt` under `trace`, the mega-MoE's fusion on or off.
fn logits(trace: &Trace, container: &Path, mega: bool, prompt: &[u32]) -> Vec<f32> {
    let source = ztensor::Source::open(container).expect("the fixture opens");
    let contract: ModelContract =
        poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
            .expect("a container of the plan's own planes is read by its own names");
    drop(source);
    let chip = engine_cuda::chip::profile(&format!(
        "[engine]\ngraphs = \"off\"\n[fuse]\nmoe_mega = {mega}\n"
    ));
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
        knobs: engine_cuda::Knobs {
            chip: Some(chip),
            ..engine_cuda::Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: false,
        world: engine_cuda::World::default(),
        expert: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .expect("the model loads");
    let mut shell = staged.finish().expect("the model arms");
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
fn a_declined_mega_moe_answers_what_its_legs_answer() {
    if !engine_cuda::device::present() {
        eprintln!("skipping: no CUDA device on this machine");
        return;
    }
    for name in ROWS {
        let row = poem_compiler::catalog::deployment(name)
            .unwrap_or_else(|| panic!("`{name}` is listed"));
        let trace = row.trace(Platform::Cuda);
        let fused = poem_compiler::fuse::fuse(trace.clone(), engine_cuda::FUSIONS.iter());
        assert!(
            fused
                .nodes
                .iter()
                .any(|node| matches!(node.op, Operation::Fused(Fused::MoeMega { .. }))),
            "{name}: the mega-MoE forms"
        );
        let dir = std::env::temp_dir().join(format!(
            "pie-mega-{}-{}",
            name.replace('.', "_"),
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("the scratch directory");
        let _scratch = Scratch(dir.clone());
        let container = dir.join("model.zt");
        write_checkpoint(&container, &trace);

        let vocab = row.model.vocab.max(1);
        let prompt: Vec<u32> = (0..41).map(|t| (t * 7 + 11) % vocab.min(4096)).collect();
        let apart = logits(&trace, &container, false, &prompt);
        let mega = logits(&trace, &container, true, &prompt);
        let spread = apart.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - apart.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            spread > 1e-3,
            "{name}: the logits are a rectangle nothing wrote"
        );
        assert!(
            apart
                .iter()
                .zip(&mega)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: the mega-MoE's logits differ from its legs'"
        );
    }
}
