//! The fixtures of the expert-parallel group tests: the MoE micro, its
//! checkpoint (every expert; a rank's load keeps its own slots), prompts,
//! an engine's load and steps, and how close two ranks' logits are held to
//! one rank's.

use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Readout, Step};
use engine::load::{Budgets, Checkpoint, LoadRequest, Residency};
use poem::{Dtype, Platform};
use poem_ir::{ParamSource, Trace};

pub const MICRO: &str = "qwen3-a3b-micro";

/// The group's ranks: `PIE_EP_RANKS` (2, 4 or 8), two by default.
pub fn ranks() -> u32 {
    let ranks = std::env::var("PIE_EP_RANKS").map_or(2, |word| {
        word.parse()
            .unwrap_or_else(|_| panic!("PIE_EP_RANKS is a rank count, not `{word}`"))
    });
    assert!(
        matches!(ranks, 2 | 4 | 8),
        "an expert-parallel group is 2, 4 or 8 ranks, not {ranks}"
    );
    ranks
}

/// Each rank's device: rank `r` on device `r % count`, so two ranks run on
/// two peers and a larger group shares each device among its ranks.
pub fn ordinals(ranks: u32) -> Vec<i32> {
    let count = engine_cuda::device::ctx::count().max(1) as u32;
    (0..ranks).map(|rank| (rank % count) as i32).collect()
}

/// The share of its device a load may hold, less what is in use as it
/// begins (the ranks loaded before it, other processes): every load's kv
/// is small and fixed, so the ranks sharing a device fit beside each other.
pub const SHARE: f64 = 0.95;

/// Each load's kv.
pub const KV: u64 = 64 << 20;

pub struct Scratch(pub PathBuf);

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

/// A plane's values, by its name: the same for one rank's whole bank and
/// a rank's slots cut from it.
pub fn values(name: &str, count: usize) -> Vec<u16> {
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

/// `whole`'s planes, every expert of every bank: what each rank of a group
/// reads its own slots from.
pub fn write_checkpoint(path: &Path, whole: &Trace) {
    let mut writer = ztensor::Writer::create(path).expect("the checkpoint opens");
    let mut planes: Vec<_> = whole
        .params
        .iter()
        .filter(|param| param.source == ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    for param in planes {
        assert_eq!(param.dtype, Dtype::Bf16, "`{}` is not bf16", param.name);
        assert!(
            param.experts.is_none(),
            "`{}` is a rank's part of a bank",
            param.name
        );
        let count: usize = param.shape.iter().product::<u64>() as usize;
        let bytes: Vec<u8> = values(&param.name, count)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
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

pub fn prompt(len: u32, seed: u32) -> Vec<u32> {
    (0..len).map(|t| (t * 7 + 11 + seed * 131) % 2048).collect()
}

pub fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: the vocabularies differ");
    let spread = want.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        - want.iter().copied().fold(f32::INFINITY, f32::min);
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        spread > 1e-3,
        "{what}: the logits are a rectangle nothing wrote"
    );
    // The experts' products see other row counts on two ranks (every rank's
    // received rows, not one rank's own), so a kernel of another tiling
    // rounds the last bf16 place: 0.0625 of a spread of 8 is one ulp there.
    assert!(
        worst <= 0.015 * spread,
        "{what}: worst gap {worst} over a spread of {spread}"
    );
}

// An engine's load and steps, as a worker drives one.

pub fn contract_for(trace: &Trace, path: &Path) -> Result<ModelContract, String> {
    let source = ztensor::Source::open(path).map_err(|why| why.to_string())?;
    poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .map_err(|why| why.to_string())
}

pub fn request(trace: &Trace, checkpoint: &Path) -> LoadRequest {
    LoadRequest {
        trace: trace.clone(),
        checkpoint: Checkpoint::Path(checkpoint.to_path_buf()),
        budgets: Budgets {
            max_lanes: 2,
            max_tokens: 256,
            buckets: vec![16, 64, 256],
            max_adapters: 0,
            page_size: 16,
            max_context: 256,
            context_declared: false,
            slots: 2,
            max_patches: None,
            max_images: None,
            max_voxels: None,
            max_clips: None,
        },
        // Ranks sharing a device: each its share of it, and a small kv.
        residency: Residency {
            device_memory_share: SHARE,
            device_kv_budget: Some(KV),
            host_kv_budget: Some(64 << 20),
            disk_kv_budget: Some(0),
            ..Residency::default()
        },
        ordinal: -1,
        frames_in_flight: 1,
    }
}

/// The diagnostics every boot of the process states, as a worker's boots
/// would: `PIE_EP_DIAG`'s words (the sync guard: `sync-guard`,
/// `sync-guard=panic`), none by default.
pub fn diagnostics() -> engine_cuda::Diagnostics {
    std::env::var("PIE_EP_DIAG").map_or_else(
        |_| engine_cuda::Diagnostics::default(),
        |words| {
            words
                .parse()
                .unwrap_or_else(|why| panic!("PIE_EP_DIAG: {why}"))
        },
    )
}

/// A rank's boot on `ordinal`, or a one-rank engine's.
pub fn boot(ordinal: i32) -> engine_cuda::DeviceBoot {
    let mut boot = engine_cuda::DeviceBoot {
        ordinal,
        ..engine_cuda::DeviceBoot::default()
    };
    boot.knobs.diagnostics = diagnostics();
    boot.knobs.work_queues = Some(work_queues());
    boot
}

/// The work queues this binary's constructor started CUDA with.
pub fn work_queues() -> u32 {
    std::env::var("CUDA_DEVICE_MAX_CONNECTIONS")
        .ok()
        .and_then(|word| word.trim().parse().ok())
        .unwrap_or(engine_cuda::ep::DEFAULT_WORK_QUEUES)
}

/// `whole`'s checkpoint, every expert, in a directory of its own.
pub fn checkpoint(whole: &Trace, tag: &str) -> (Scratch, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("pie-ep-engine-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("the scratch directory");
    let path = dir.join("micro.zt");
    write_checkpoint(&path, whole);
    (Scratch(dir), path)
}

pub fn step(slot: u32, word: u64, tokens: &[u32]) -> Step {
    Step {
        lanes: vec![Lane {
            slot,
            word,
            tokens: tokens.to_vec(),
            readout: Readout::Last,
            ..Lane::default()
        }],
        attachments: Vec::new(),
        media: Vec::new(),
        voxels: Vec::new(),
    }
}

/// One step's last-row logits, submitted and settled.
pub fn answer(engine: &mut dyn Engine, step: &Step) -> Vec<f32> {
    let mut ticket = engine
        .submit(&FrameSubmission::of(step.clone()))
        .expect("the frame fires");
    engine.settle_frame(&mut ticket).expect("the frame settles");
    let readout = &ticket.steps[0].readouts[0];
    readout.values.clone()
}
