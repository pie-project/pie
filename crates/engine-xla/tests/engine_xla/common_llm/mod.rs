//! Catalog language models with random bf16 weights, loaded through the
//! engine and fired as the runtime fires them: the bitwise gates between
//! two builds read what these fires answer.

#![allow(dead_code)]

use std::path::PathBuf;

use engine::Engine;
use engine::fire::{FrameSubmission, Lane, Readout, Step};
use engine::load::{Budgets, Checkpoint, LoadRequest, Loaded, Residency};
use engine_xla::{DeviceBoot, Xla};
use poem::{Platform, Request};

use crate::common_dit::contract_for;

pub const PAGE: u32 = 16;
pub const SLOTS: u32 = 4;

fn fnv(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A checkpoint of `trace`'s own planes seeded by name and `seed`: norms
/// near one, the rest small (engine-metal's fixture writer).
fn write_checkpoint(path: &std::path::Path, trace: &poem_ir::Trace, seed: u64) {
    let mut writer = ztensor::Writer::create(path).expect("the fixture writes");
    let mut planes: Vec<_> = trace
        .params
        .iter()
        .filter(|param| param.source == poem_ir::ParamSource::Checkpoint)
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));
    for param in planes {
        let count = param.shape.iter().product::<u64>() as usize;
        let norm = param.name.ends_with("norm");
        let mut state = fnv(&param.name) ^ seed;
        let mut next = || {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            state >> 33
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
            poem::Dtype::Bf16 => (
                ztensor::Leaf::BF16,
                (0..count)
                    .flat_map(|_| ((value(next()).to_bits() >> 16) as u16).to_le_bytes())
                    .collect(),
            ),
            poem::Dtype::F32 => (
                ztensor::Leaf::F32,
                (0..count)
                    .flat_map(|_| value(next()).to_le_bytes())
                    .collect(),
            ),
            other => panic!("`{}` is {other:?}, not a dtype the fixture seeds", param.name),
        };
        writer
            .add(param.name.as_str(), param.shape.clone(), leaf, &bytes)
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer.finish().expect("the fixture finishes");
}

pub struct Model {
    pub engine: Xla,
    pub loaded: Loaded,
    pub facts: poem_ir::Facts,
    pub vocab: u32,
    pub context: u32,
    dir: PathBuf,
}

impl Drop for Model {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Model {
    /// Model `name` of the catalog, cut to `layers` where stated, its
    /// weights drawn from `seed`, seated at `context` tokens.
    pub fn load(name: &str, layers: Option<u32>, context: u32, seed: u64) -> Model {
        let package = poem_compiler::catalog::package_of(name).expect("the model is listed");
        let deploy = poem::star::Deploy {
            layers,
            ..poem_compiler::catalog::one_rank(poem::Dtype::Bf16, poem::Dtype::Bf16)
        };
        let trace = package
            .trace(name, &deploy, name, Platform::Xla)
            .unwrap_or_else(|why| panic!("`{name}` traces: {why:#}"));
        let dir = std::env::temp_dir().join(format!(
            "pie-xla-llm-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        let path = dir.join("weights.zt");
        write_checkpoint(&path, &trace, seed);
        let vocab = trace
            .params
            .iter()
            .find(|p| p.name.contains("embed"))
            .and_then(|p| p.shape.first().copied())
            .map_or(1000, |v| u32::try_from(v).unwrap_or(1000));
        let facts = trace.facts.clone();
        let mut engine = Xla::new(DeviceBoot::default(), contract_for);
        let residency = Residency {
            device_kv_budget: crate::common::kv_budget(&trace, context, SLOTS, SLOTS * context / PAGE),
            ..Residency::default()
        };
        let loaded = engine
            .load(LoadRequest {
                trace,
                checkpoint: Checkpoint::Path(path),
                budgets: Budgets {
                    max_lanes: 8,
                    max_tokens: 128,
                    buckets: Vec::new(),
                    max_adapters: 0,
                    page_size: PAGE,
                    max_context: context,
                    context_declared: false,
                    slots: SLOTS,
                    max_patches: None,
                    max_images: None,
                    max_voxels: None,
                    max_clips: None,
                },
                residency,
                ordinal: 0,
                frames_in_flight: 1,
            })
            .expect("the model loads");
        Model {
            engine,
            loaded,
            facts,
            vocab: vocab.max(2),
            context,
            dir,
        }
    }

    pub fn word(&self, rows: u32) -> u64 {
        self.facts.word(&Request::new(rows, false))
    }

    /// Submits one step and settles it: each lane's readout values.
    pub fn fire(&mut self, lanes: Vec<Lane>) -> Vec<Vec<f32>> {
        let mut ticket = self
            .engine
            .submit(&FrameSubmission::of(Step {
                lanes,
                attachments: Vec::new(),
                media: Vec::new(),
                voxels: Vec::new(),
            }))
            .expect("the frame fires");
        self.engine
            .settle_frame(&mut ticket)
            .expect("the frame settles");
        ticket.steps[0]
            .readouts
            .iter()
            .map(|r| r.values.clone())
            .collect()
    }
}

/// `n` token ids below `vocab`, drawn from `seed`.
pub fn tokens(seed: u64, n: usize, vocab: u32) -> Vec<u32> {
    let mut s = seed ^ 0x2545_f491_4f6c_dd1d;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) % u64::from(vocab)) as u32
        })
        .collect()
}

/// A lane's readout as bits, for a bitwise gate.
pub fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// One sequence the gates drive: the slot the runtime seats it at and how
/// many rows it holds.
#[derive(Clone, Copy, Debug)]
pub struct Seq {
    pub slot: u32,
    pub held: u32,
}

impl Seq {
    pub fn lane(&self, model: &Model, tokens: Vec<u32>) -> Lane {
        let rows = tokens.len() as u32;
        let after = self.held + rows;
        let pages: Vec<u32> = (0..after.div_ceil(PAGE))
            .map(|p| self.slot * (model.context / PAGE) + p)
            .collect();
        Lane {
            slot: self.slot,
            word: model.word(rows),
            tokens,
            kv: engine::fire::KvDelta {
                held: self.held,
                pages,
                translation: Vec::new(),
                window: Vec::new(),
            },
            readout: Readout::Last,
            rs_slot: Some(self.slot),
            ..Lane::default()
        }
    }
}

/// The scenario every bitwise gate fires: two prompts, a two-lane decode,
/// then a third prompt fired beside a decode, and with `long` a prompt of
/// that many tokens (past a window) decoded on. Every readout, in order.
pub fn scenario(model: &mut Model, long: Option<u32>) -> Vec<Vec<u32>> {
    let vocab = model.vocab;
    let mut out = Vec::new();
    let mut a = Seq { slot: 0, held: 0 };
    let mut b = Seq { slot: 1, held: 0 };
    let mut c = Seq { slot: 2, held: 0 };
    let lane = a.lane(model, tokens(1, 37, vocab));
    out.extend(model.fire(vec![lane]).iter().map(|v| bits(v)));
    a.held += 37;
    let lane = b.lane(model, tokens(2, 20, vocab));
    out.extend(model.fire(vec![lane]).iter().map(|v| bits(v)));
    b.held += 20;
    for step in 0..4u64 {
        let la = a.lane(model, tokens(10 + step, 1, vocab));
        let lb = b.lane(model, tokens(20 + step, 1, vocab));
        out.extend(model.fire(vec![la, lb]).iter().map(|v| bits(v)));
        a.held += 1;
        b.held += 1;
    }
    let la = a.lane(model, tokens(30, 1, vocab));
    let lc = c.lane(model, tokens(3, 50, vocab));
    out.extend(model.fire(vec![la, lc]).iter().map(|v| bits(v)));
    a.held += 1;
    c.held += 50;
    for step in 0..2u64 {
        let la = a.lane(model, tokens(40 + step, 1, vocab));
        let lc = c.lane(model, tokens(50 + step, 1, vocab));
        out.extend(model.fire(vec![la, lc]).iter().map(|v| bits(v)));
        a.held += 1;
        c.held += 1;
    }
    if let Some(n) = long {
        let mut d = Seq { slot: 3, held: 0 };
        // Prompted in two fires, then decoded beside `a`.
        let first = n / 2;
        let lane = d.lane(model, tokens(60, first as usize, vocab));
        out.extend(model.fire(vec![lane]).iter().map(|v| bits(v)));
        d.held += first;
        let lane = d.lane(model, tokens(61, (n - first) as usize, vocab));
        out.extend(model.fire(vec![lane]).iter().map(|v| bits(v)));
        d.held += n - first;
        for step in 0..3u64 {
            let la = a.lane(model, tokens(70 + step, 1, vocab));
            let ld = d.lane(model, tokens(80 + step, 1, vocab));
            out.extend(model.fire(vec![la, ld]).iter().map(|v| bits(v)));
            a.held += 1;
            d.held += 1;
        }
    }
    out
}
