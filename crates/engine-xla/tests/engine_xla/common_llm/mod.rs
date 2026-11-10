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

/// Load budgets the speed probe sets for the loads it makes.
static KNOBS: std::sync::Mutex<Vec<(String, u32)>> = std::sync::Mutex::new(Vec::new());

/// Sets load budget `name` (`SLOTS`, `LANES`, `TOKENS`) for later loads.
pub fn set_knob(name: &str, v: u32) {
    let mut knobs = KNOBS.lock().expect("the knobs");
    knobs.retain(|(n, _)| n != name);
    knobs.push((name.to_string(), v));
}

/// A load budget: what [`set_knob`] set, else `PIE_XLA_TEST_<name>`, else
/// `default`.
fn knob(name: &str, default: u32) -> u32 {
    let set = KNOBS
        .lock()
        .expect("the knobs")
        .iter()
        .find(|(n, _)| n == name)
        .map(|&(_, v)| v);
    set.or_else(|| {
        std::env::var(format!("PIE_XLA_TEST_{name}"))
            .ok()
            .and_then(|v| v.parse().ok())
    })
    .unwrap_or(default)
}

/// Working sets a load seats.
pub fn slots() -> u32 {
    knob("SLOTS", SLOTS)
}

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
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
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
        Model::load_with(name, layers, context, seed, None)
    }

    /// [`Model::load`] with a slot file under `disk` (1 MiB of kv slots).
    pub fn load_with(
        name: &str,
        layers: Option<u32>,
        context: u32,
        seed: u64,
        disk: Option<&std::path::Path>,
    ) -> Model {
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
            // `PIE_XLA_TEST_EXTRA_MIB` grows the pool past what the
            // scenario needs: readouts must not depend on the pool's size.
            device_kv_budget: crate::common::kv_budget(
                &trace,
                context,
                slots(),
                slots() * context / PAGE,
            )
            .map(|bytes| {
                bytes
                    + std::env::var("PIE_XLA_TEST_EXTRA_MIB")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .map_or(0, |mib| mib << 20)
            }),
            disk_kv_dir: disk.map(std::path::Path::to_path_buf),
            disk_kv_budget: disk.map(|_| 64 << 20),
            ..Residency::default()
        };
        let loaded = engine
            .load(LoadRequest {
                trace,
                checkpoint: Checkpoint::Path(path),
                budgets: Budgets {
                    max_lanes: knob("LANES", 8),
                    max_tokens: knob("TOKENS", 128),
                    buckets: Vec::new(),
                    max_adapters: 0,
                    page_size: PAGE,
                    max_context: context,
                    context_declared: false,
                    slots: slots(),
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

/// The runtime's part in a pool this load states shared (`ChunkFacts`
/// shapes unpinned): chunks handed to shapes as they fill, slots taken from
/// the top of each chunk down, so no slot sits where an identity map would
/// put it.
#[derive(Debug, Default)]
pub struct Refs {
    pub facts: Option<engine::chunk::ChunkFacts>,
    next_chunk: u32,
    open: std::collections::BTreeMap<u32, (u32, u32)>,
}

impl Refs {
    pub fn of(model: &Model) -> Refs {
        Refs::of_facts(model.loaded.caps.chunks.clone())
    }

    pub fn of_facts(facts: Option<engine::chunk::ChunkFacts>) -> Refs {
        let facts = facts.filter(|facts| {
            !facts.shapes.is_empty() && facts.shapes.iter().all(|s| s.pinned.is_none())
        });
        let next_chunk = 0;
        Refs {
            facts,
            next_chunk,
            open: Default::default(),
        }
    }

    pub fn shared(&self) -> bool {
        self.facts.is_some()
    }

    /// A fresh slot ref of shape `id`.
    pub fn take(&mut self, id: engine::chunk::ShapeId) -> u32 {
        let facts = self.facts.as_ref().expect("a shared pool");
        let spc = facts.shape(id).expect("a stated shape").slots_per_chunk;
        let (chunk, used) = match self.open.get(&id.0) {
            Some(&(chunk, used)) if used < spc => (chunk, used),
            _ => {
                while facts.is_reserved(self.next_chunk) {
                    self.next_chunk += 1;
                }
                let chunk = self.next_chunk;
                self.next_chunk += 1;
                assert!(chunk < facts.total_chunks, "the pool ran out of chunks");
                (chunk, 0)
            }
        };
        self.open.insert(id.0, (chunk, used + 1));
        chunk * spc + (spc - 1 - used)
    }

    pub fn shapes(
        &self,
        want: impl Fn(&engine::chunk::SlotShape) -> bool,
    ) -> Vec<engine::chunk::SlotShape> {
        self.facts.as_ref().map_or_else(Vec::new, |f| {
            f.shapes.iter().filter(|s| want(s)).cloned().collect()
        })
    }
}

/// One sequence the gates drive: the slot the runtime seats it at, how
/// many rows it holds and, in a shared pool, its slot refs.
#[derive(Clone, Debug, Default)]
pub struct Seq {
    pub slot: u32,
    pub held: u32,
    pub kv: Vec<u32>,
    pub window: Vec<u32>,
    pub record: Vec<(engine::chunk::ShapeId, Vec<[u32; 2]>)>,
}

impl Seq {
    pub fn new(slot: u32) -> Seq {
        Seq {
            slot,
            ..Seq::default()
        }
    }

    pub fn lane(&mut self, model: &Model, refs: &mut Refs, tokens: Vec<u32>) -> Lane {
        let word = model.word(tokens.len() as u32);
        self.lane_of(model.context, word, refs, tokens)
    }

    /// The lane as the runtime fires it: legacy page ids in a degenerate
    /// pool, slot refs (`tables`, `fixed`) in a shared one.
    pub fn lane_of(&mut self, context: u32, word: u64, refs: &mut Refs, tokens: Vec<u32>) -> Lane {
        use engine::chunk::{SlotKind, SlotRole};
        let rows = tokens.len() as u32;
        let after = self.held + rows;
        let need = after.div_ceil(PAGE) as usize;
        let mut lane = Lane {
            slot: self.slot,
            word,
            tokens,
            readout: Readout::Last,
            rs_slot: Some(self.slot),
            ..Lane::default()
        };
        if !refs.shared() {
            lane.kv = engine::fire::KvDelta {
                held: self.held,
                pages: (0..need as u32)
                    .map(|p| self.slot * (context / PAGE) + p)
                    .collect(),
                translation: Vec::new(),
                window: Vec::new(),
            };
            return lane;
        }
        for list in refs.shapes(|s| {
            matches!(s.kind, SlotKind::List { window: None, .. }) && s.role == SlotRole::State
        }) {
            while self.kv.len() < need {
                let r = refs.take(list.id);
                self.kv.push(r);
            }
            lane.tables.push((list.id, self.kv[..need].to_vec()));
        }
        for list in refs.shapes(|s| {
            matches!(
                s.kind,
                SlotKind::List {
                    window: Some(_),
                    ..
                }
            )
        }) {
            let SlotKind::List {
                tokens,
                window: Some(w),
                ..
            } = list.kind
            else {
                unreachable!()
            };
            let need = after.div_ceil(tokens) as usize;
            while self.window.len() < need {
                let r = refs.take(list.id);
                self.window.push(r);
            }
            // Trimmed behind the window of the rows committed so far.
            let table: Vec<u32> = (0..need)
                .map(|i| {
                    let last = (i as u32 + 1) * tokens - 1;
                    if last + w < self.held {
                        0
                    } else {
                        self.window[i]
                    }
                })
                .collect();
            lane.tables.push((list.id, table));
        }
        if self.record.is_empty() {
            for fixed in refs.shapes(|s| matches!(s.kind, SlotKind::Fixed { .. })) {
                let SlotKind::Fixed { pieces } = fixed.kind else {
                    unreachable!()
                };
                let pairs = (0..pieces)
                    .map(|_| {
                        let r = refs.take(fixed.id);
                        [r, r]
                    })
                    .collect();
                self.record.push((fixed.id, pairs));
            }
        }
        lane.fixed = self.record.clone();
        lane.kv = engine::fire::KvDelta {
            held: self.held,
            pages: self.kv[..need.min(self.kv.len())].to_vec(),
            translation: Vec::new(),
            window: Vec::new(),
        };
        lane
    }
}

/// The scenario every bitwise gate fires: two prompts, a two-lane decode,
/// then a third prompt fired beside a decode, and with `long` a prompt of
/// that many tokens (past a window) decoded on. Every readout, in order.
pub fn scenario(model: &mut Model, long: Option<u32>) -> Vec<Vec<u32>> {
    let vocab = model.vocab;
    let mut out = Vec::new();
    let mut refs = Refs::of(model);
    let mut a = Seq::new(0);
    let mut b = Seq::new(1);
    let mut c = Seq::new(2);
    let lane = a.lane(model, &mut refs, tokens(1, 37, vocab));
    out.extend(model.fire(vec![lane]).iter().map(|v| bits(v)));
    a.held += 37;
    let lane = b.lane(model, &mut refs, tokens(2, 20, vocab));
    out.extend(model.fire(vec![lane]).iter().map(|v| bits(v)));
    b.held += 20;
    for step in 0..4u64 {
        let la = a.lane(model, &mut refs, tokens(10 + step, 1, vocab));
        let lb = b.lane(model, &mut refs, tokens(20 + step, 1, vocab));
        out.extend(model.fire(vec![la, lb]).iter().map(|v| bits(v)));
        a.held += 1;
        b.held += 1;
    }
    let la = a.lane(model, &mut refs, tokens(30, 1, vocab));
    let lc = c.lane(model, &mut refs, tokens(3, 50, vocab));
    out.extend(model.fire(vec![la, lc]).iter().map(|v| bits(v)));
    a.held += 1;
    c.held += 50;
    for step in 0..2u64 {
        let la = a.lane(model, &mut refs, tokens(40 + step, 1, vocab));
        let lc = c.lane(model, &mut refs, tokens(50 + step, 1, vocab));
        out.extend(model.fire(vec![la, lc]).iter().map(|v| bits(v)));
        a.held += 1;
        c.held += 1;
    }
    if let Some(n) = long {
        let mut d = Seq::new(3);
        // Prompted in fires of at most 100 rows, then decoded beside `a`.
        let mut piece = 0u64;
        while d.held < n {
            let rows = (n - d.held).min(100);
            let lane = d.lane(model, &mut refs, tokens(60 + piece, rows as usize, vocab));
            out.extend(model.fire(vec![lane]).iter().map(|v| bits(v)));
            d.held += rows;
            piece += 1;
        }
        for step in 0..3u64 {
            let la = a.lane(model, &mut refs, tokens(70 + step, 1, vocab));
            let ld = d.lane(model, &mut refs, tokens(80 + step, 1, vocab));
            out.extend(model.fire(vec![la, ld]).iter().map(|v| bits(v)));
            a.held += 1;
            d.held += 1;
        }
    }
    out
}
