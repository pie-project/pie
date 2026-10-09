//! What the VAE reference tests share: where the artifact, the snapshot and
//! the golden dump are found, how a VAE-only plan loads, and how its answer
//! is scored against the reference.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine_cuda::{Boot, Graphs, Knobs, Recording, Shell};
use poem::Trace;
use poem::star::Package;
use poem_compiler::{Budget, VoxelLadder};

fn dir_of(var: &str, default: &str) -> PathBuf {
    std::env::var_os(var).map_or_else(|| PathBuf::from(default), PathBuf::from)
}

/// The imported artifact `file` under `PIE_IMAGEGEN_ARTIFACTS`.
#[must_use]
pub fn artifact(file: &str) -> PathBuf {
    let path = dir_of("PIE_IMAGEGEN_ARTIFACTS", "/root/.cache/pie-imagegen").join(file);
    assert!(
        path.is_file(),
        "no {} (run `pie model import` first, or set PIE_IMAGEGEN_ARTIFACTS)",
        path.display()
    );
    path
}

/// The golden dump `dir` under `PIE_IMAGEGEN_GOLDEN`.
#[must_use]
pub fn golden(dir: &str) -> PathBuf {
    let dir = dir_of("PIE_IMAGEGEN_GOLDEN", "/root/.cache/pie-imagegen/golden").join(dir);
    assert!(
        dir.join("shapes.json").is_file(),
        "no golden dump at {} (run the family's golden script with --vae, or set \
         PIE_IMAGEGEN_GOLDEN)",
        dir.display()
    );
    dir
}

/// The golden dump's `shapes.json`.
#[must_use]
pub fn shapes(gold: &Path) -> serde_json::Value {
    serde_json::from_slice(&std::fs::read(gold.join("shapes.json")).expect("shapes.json"))
        .expect("shapes.json parses")
}

/// The box and the channel count `shapes` states for `key`.
#[must_use]
pub fn boxed(shapes: &serde_json::Value, key: &str) -> ([u32; 3], usize) {
    let at = &shapes[key];
    let get = |name: &str| at[name].as_u64().expect("a box extent") as u32;
    (
        [get("t"), get("h"), get("w")],
        at["channels"].as_u64().expect("channels") as usize,
    )
}

fn hub() -> PathBuf {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join("hub");
    }
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface/hub")
}

/// The Hugging Face snapshot of `repo` that holds every one of `files`.
#[must_use]
pub fn snapshot(repo: &str, files: &[&str]) -> PathBuf {
    let snapshots = hub()
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(&snapshots)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| entry.path())
        .find(|path| files.iter().all(|file| path.join(file).is_file()))
        .unwrap_or_else(|| panic!("no {repo} snapshot holding {files:?} under the hub cache"))
}

#[must_use]
pub fn f32s(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[must_use]
pub fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&v| {
            let bits = v.to_bits();
            let rounding = 0x7fff + ((bits >> 16) & 1);
            (((bits + rounding) >> 16) as u16).to_le_bytes()
        })
        .collect()
}

pub struct Score {
    pub cos: f64,
    pub max_abs: f64,
    pub mean_abs: f64,
}

#[must_use]
pub fn score(got: &[f32], want: &[f32]) -> Score {
    assert_eq!(got.len(), want.len(), "one value per reference value");
    let (mut dot, mut gg, mut ww, mut max_abs, mut sum_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let (g, w) = (f64::from(*g), f64::from(*w));
        dot += g * w;
        gg += g * g;
        ww += w * w;
        let err = (g - w).abs();
        max_abs = max_abs.max(err);
        sum_abs += err;
    }
    Score {
        cos: dot / (gg.sqrt() * ww.sqrt()).max(1e-30),
        max_abs,
        mean_abs: sum_abs / want.len().max(1) as f64,
    }
}

/// The one-rank deployment the catalog row `row` states.
#[must_use]
pub fn deploy(row: &str) -> poem::star::Deploy {
    let row = models::deployment(row).unwrap_or_else(|| panic!("no `{row}` row"));
    models::star::deploy(&row.deploy)
}

/// `id`'s package with a forward that runs one of its VAE's readings alone,
/// over every row, and holds no caches.
#[must_use]
pub fn one_arm(id: &str, decode: bool) -> Package {
    let package = models::star::package_of(id).unwrap_or_else(|| panic!("no package holds {id}"));
    let files_at = format!("{}files/", poem::star::ATTRIBUTE);
    let mut files: Vec<(String, String)> = package
        .attributes()
        .into_iter()
        .filter_map(|(key, source)| Some((key.strip_prefix(&files_at)?.to_string(), source)))
        .collect();
    for (file, source) in &mut files {
        if file == "forward.star" {
            *source = source
                .replace("def caches(m, c):", "def every_cache(m, c):")
                .replace("def forward(m, inputs):", "def every_reading(m, inputs):")
                + &format!(
                    "\ndef caches(m, c):\n    pass\n\ndef forward(m, inputs):\n    return {}(inputs, m.vae)\n",
                    if decode { "decode" } else { "encode" }
                );
        }
    }
    let files: Vec<(&str, &str)> = files
        .iter()
        .map(|(f, s)| (f.as_str(), s.as_str()))
        .collect();
    Package::new(package.name(), &files).unwrap_or_else(|why| panic!("{why:#}"))
}

/// Cuts `contract` to the tensors `trace` reads and those they are derived
/// from.
pub fn keep_what_reads(contract: &mut ModelContract, trace: &Trace) {
    let mut keep: BTreeSet<String> = trace.params.iter().map(|p| p.name.clone()).collect();
    loop {
        let more: Vec<String> = contract
            .tensors
            .iter()
            .filter(|t| keep.contains(&t.name))
            .flat_map(|t| t.expr.outputs().into_iter().map(str::to_string))
            .filter(|name: &String| !keep.contains(name))
            .collect();
        if more.is_empty() {
            break;
        }
        keep.extend(more);
    }
    contract.tensors.retain(|t| keep.contains(&t.name));
}

/// A VAE-only plan loaded on device 0 with room for `max_voxels`, its slot 0
/// open.
#[must_use]
pub fn load(trace: Trace, contract: &ModelContract, checkpoint: &Path, max_voxels: u32) -> Shell {
    let mut shell = Shell::load(Boot {
        trace,
        contract,
        checkpoint,
        budget: Budget::new(2, 16),
        patches: None,
        voxels: Some(VoxelLadder::new(max_voxels, 2)),
        profile: None,
        page_size: 16,
        context: 64,
        slots: 2,
        pages: 4,
        ordinal: 0,
        graphs: Graphs::Off,
        knobs: Knobs {
            recording: Recording::Off,
            ..Knobs::default()
        },
        cache_dir: None,
        runahead: engine::runahead::Runahead::F1,
        residency: engine_cuda::experts::Plan::default(),
        deferred_tier: true,
        world: engine_cuda::World::default(),
        comm: core::ptr::null_mut(),
    })
    .unwrap_or_else(|why| panic!("the VAE plan does not load: {why}"));
    shell.open(0).expect("slot 0 opens");
    shell
}
