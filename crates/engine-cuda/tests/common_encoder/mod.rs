#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use checkpoint::file::emit::{self, Object};
use checkpoint::serving::Stamp;
use engine::Engine;
use engine::channel::ChannelRegistration;
use engine::fire::{
    Attachment, Boundary, FrameSubmission, Lane, LaneStream, PortFeed, PortKind, Readout, Step,
};
use engine::load::{Budgets, Checkpoint, LoadRequest, Loaded, Residency};
use engine::program::{InstanceBinding, ProgramRegistration};
use engine_cuda::Graphs;
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, ModelProfile, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use eta_ir::validate::bind;
use poem::{
    Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Trace, Value, Weight, ops, seam,
    trace_hybrid,
};

pub const WIDTH: u32 = 32;
pub const HEAD_DIM: u32 = 64;
pub const HEADS: u32 = 2;
pub const MAX_LEN: u32 = 64;
pub const NUM_BUCKETS: u32 = 32;
pub const MAX_DISTANCE: f32 = 128.0;
pub const SM_SCALE: f32 = 0.125;
pub const NAME: &str = "encoder-mini";

pub struct EncoderLayer;

impl ForwardHybrid for EncoderLayer {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) -> Value {
        let w = |name: &str, out: u32, inner: u32| {
            Weight::sym(name, [u64::from(out), u64::from(inner)], Dtype::Bf16)
        };
        let x = inputs.latents(0, WIDTH, Dtype::Bf16);
        let table = ops::elemwise::relative_bucket_bias(
            inputs.recorder(),
            &Weight::sym(
                "rel_bias",
                [u64::from(NUM_BUCKETS), u64::from(HEADS)],
                Dtype::Bf16,
            ),
            MAX_LEN,
            NUM_BUCKETS,
            MAX_DISTANCE,
            true,
        );
        let q = ops::linear::matmul(&x, &w("q", HEADS * HEAD_DIM, WIDTH));
        let k = ops::linear::matmul(&x, &w("k", HEADS * HEAD_DIM, WIDTH));
        let v = ops::linear::matmul(&x, &w("v", HEADS * HEAD_DIM, WIDTH));
        let perm = inputs.row_permutation();
        let indptr = inputs.lane_indptr();
        let o = ops::attn::ragged(
            &ops::layout::pack_rows(&q, &perm),
            &ops::layout::pack_rows(&k, &perm),
            &ops::layout::pack_rows(&v, &perm),
            &indptr,
            &indptr,
            HEAD_DIM,
            SM_SCALE,
            ops::attn::relative_bias(&table, MAX_LEN),
        );
        let o = ops::layout::unpack_rows(&o, &perm);
        let y = ops::linear::matmul(&o, &w("o", WIDTH, HEADS * HEAD_DIM));
        let out = ops::elemwise::residual_add(&x, &y);
        seam::at(seam::HIDDEN, &[&out]);
        out
    }
}

pub fn trace() -> Trace {
    trace_hybrid(NAME, &EncoderLayer, Platform::Cuda)
}

pub fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(round)) >> 16) as u16
}

pub fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

pub fn bf(x: f32) -> f32 {
    from_bf16(to_bf16(x))
}

pub struct Lcg(u64);

impl Lcg {
    pub fn seeded(seed: u64) -> Lcg {
        Lcg(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    pub fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (self.0 >> 40) as u32;
        (bits as f32 / 8_388_608.0) - 1.0
    }
}

pub struct Weights {
    pub planes: BTreeMap<String, (Vec<u64>, Vec<f32>)>,
}

impl Weights {
    pub fn random(trace: &Trace, seed: u64) -> Weights {
        let mut rng = Lcg::seeded(seed);
        let mut planes = BTreeMap::new();
        for param in &trace.params {
            let shape: Vec<u64> = param.shape.clone();
            let count: u64 = shape.iter().product();
            let fan_in = shape.last().copied().unwrap_or(1) as f32;
            let scale = 1.0 / fan_in.sqrt();
            let values: Vec<f32> = (0..count).map(|_| bf(rng.unit() * scale)).collect();
            planes.insert(param.name.clone(), (shape, values));
        }
        Weights { planes }
    }

    pub fn get(&self, name: &str) -> &[f32] {
        &self
            .planes
            .get(name)
            .unwrap_or_else(|| panic!("no plane {name}"))
            .1
    }

    pub fn write(&self, dir: &Path) -> PathBuf {
        let path = dir.join(format!("cuda-{NAME}.zt"));
        let bytes: Vec<(String, Vec<u64>, Vec<u8>)> = self
            .planes
            .iter()
            .map(|(name, (shape, values))| {
                (
                    name.clone(),
                    shape.clone(),
                    values
                        .iter()
                        .flat_map(|v| to_bf16(*v).to_le_bytes())
                        .collect(),
                )
            })
            .collect();
        let objects: Vec<Object<'_>> = bytes
            .iter()
            .map(|(name, shape, raw)| Object::leaf(name, shape.clone(), ztensor::Leaf::BF16, raw))
            .collect();
        emit::write(
            &path,
            &Stamp::of("cuda", NAME),
            &BTreeMap::new(),
            4096,
            &objects,
            |o, p, _| panic!("{o}/{p} is not streamed here"),
        )
        .expect("the artifact writes");
        path
    }
}

pub fn contract_for(trace: &Trace, path: &Path) -> Result<ModelContract, String> {
    let source = ztensor_compat::index(path).map_err(|why| why.to_string())?;
    poem::import::own_contract(&source, &trace.params, 1, Platform::Cuda)
        .map_err(|why| why.to_string())
}

fn matmul_bf16(x: &[f32], rows: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * n];
    for r in 0..rows {
        for c in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc = x[r * k + i].mul_add(w[c * k + i], acc);
            }
            y[r * n + c] = bf(acc);
        }
    }
    y
}

pub fn bucket(d: i64, bidirectional: bool, mut num_buckets: i64, max_distance: f32) -> i64 {
    let mut out = 0;
    let n = if bidirectional {
        num_buckets /= 2;
        if d > 0 {
            out += num_buckets;
        }
        d.abs()
    } else {
        -d.min(0)
    };
    let max_exact = num_buckets / 2;
    if n < max_exact {
        return out + n;
    }
    let x = (n as f32 / max_exact as f32).ln();
    let ratio = (f64::from(max_distance) / max_exact as f64).ln() as f32;
    let large = max_exact + (x / ratio * (num_buckets - max_exact) as f32).trunc() as i64;
    out + large.min(num_buckets - 1)
}

fn bias(embedding: &[f32], h: usize, d: i64) -> f32 {
    let b = bucket(d, true, i64::from(NUM_BUCKETS), MAX_DISTANCE) as usize;
    embedding[b * HEADS as usize + h]
}

fn attention(q: &[f32], k: &[f32], v: &[f32], rows: usize, embedding: &[f32]) -> Vec<f32> {
    let hd = HEAD_DIM as usize;
    let heads = HEADS as usize;
    let width = heads * hd;
    let mut o = vec![0f32; rows * width];
    for h in 0..heads {
        for i in 0..rows {
            let qi = &q[i * width + h * hd..][..hd];
            let scores: Vec<f32> = (0..rows)
                .map(|j| {
                    let kj = &k[j * width + h * hd..][..hd];
                    let dot = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>();
                    dot * SM_SCALE + bias(embedding, h, j as i64 - i as i64)
                })
                .collect();
            let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - peak).exp()).collect();
            let mass: f32 = weights.iter().sum();
            for d in 0..hd {
                let mut acc = 0f32;
                for j in 0..rows {
                    acc += bf(weights[j] / mass) * v[j * width + h * hd + d];
                }
                o[i * width + h * hd + d] = bf(acc);
            }
        }
    }
    o
}

pub fn reference(weights: &Weights, x: &[f32], rows: usize) -> Vec<f32> {
    let w = WIDTH as usize;
    let width = (HEADS * HEAD_DIM) as usize;
    let q = matmul_bf16(x, rows, w, weights.get("q"), width);
    let k = matmul_bf16(x, rows, w, weights.get("k"), width);
    let v = matmul_bf16(x, rows, w, weights.get("v"), width);
    let o = attention(&q, &k, &v, rows, weights.get("rel_bias"));
    let y = matmul_bf16(&o, rows, width, weights.get("o"), w);
    y.iter().zip(x).map(|(a, b)| bf(a + b)).collect()
}

pub fn epilogue(rows: u32) -> TraceContainer {
    let decl = |host_role: HostRole| ChannelDecl {
        shape: Shape::matrix(rows, WIDTH),
        dtype: ChanDType::Concrete(EtaDtype::F32),
        capacity: 2,
        host_role,
        seeded: false,
    };
    TraceContainer {
        names: Vec::new(),
        channels: vec![decl(HostRole::Writer), decl(HostRole::Reader)],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![Op::ChanTake(0), Op::ChanPut { chan: 1, value: 0 }],
        }],
        externs: Vec::new(),
    }
}

pub struct LaneHandles {
    pub instance: u64,
    pub latent: u64,
    pub out: u64,
    pub rows: u32,
}

pub struct Rig {
    pub engine: engine_cuda::Cuda,
    pub loaded: Loaded,
    pub facts: poem_ir::Facts,
    pub programs: BTreeMap<u32, u64>,
    next_channel: u64,
    _dir: tempfile::TempDir,
}

impl Rig {
    pub fn load(weights: &Weights, max_tokens: u32, buckets: Vec<u32>, graphs: Graphs) -> Rig {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let path = weights.write(dir.path());
        let boot = engine_cuda::DeviceBoot {
            graphs,
            ..engine_cuda::DeviceBoot::default()
        };
        let mut engine = engine_cuda::open(boot, contract_for).expect("the engine opens");
        let trace = trace();
        let facts = trace.facts.clone();
        let loaded = engine
            .load(LoadRequest {
                trace,
                checkpoint: Checkpoint::Path(path.clone()),
                budgets: Budgets {
                    max_lanes: 8,
                    max_tokens,
                    buckets,
                    max_adapters: 0,
                    page_size: 16,
                    max_context: 64,
                    slots: 8,
                    pages: 64,
                    max_patches: None,
                    max_images: None,
                    max_voxels: None,
                    max_clips: None,
                },
                residency: Residency::default(),
                ordinal: 0,
                frames_in_flight: 1,
            })
            .expect("the layer loads");
        Rig {
            engine,
            loaded,
            facts,
            programs: BTreeMap::new(),
            next_channel: 1,
            _dir: dir,
        }
    }

    pub fn profile(&self) -> &ModelProfile {
        &self.loaded.caps.profile
    }

    fn program(&mut self, rows: u32) -> u64 {
        if let Some(id) = self.programs.get(&rows) {
            return *id;
        }
        let bound = bind(epilogue(rows), self.profile().clone()).expect("the epilogue binds");
        let stages = compile_bound(&bound);
        let launch = eta_compiler::codegen::launch::build(&bound, &stages);
        let backend = Backend::parse("cuda").expect("the cuda backend");
        let registration = ProgramRegistration {
            program_hash: 0xe4c ^ u64::from(rows),
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        };
        let id = self
            .engine
            .register_program(&registration)
            .expect("the program registers");
        self.programs.insert(rows, id);
        id
    }

    fn channel(&mut self, shape: Vec<u32>, host_role: HostRole) -> u64 {
        let id = self.next_channel;
        self.next_channel += 1;
        self.engine
            .register_channel(&ChannelRegistration {
                id,
                shape,
                dtype: ChanDType::Concrete(EtaDtype::F32),
                host_role,
                seeded: false,
                extern_dir: None,
                capacity: 2,
                extern_name: Vec::new(),
            })
            .expect("the channel registers");
        id
    }

    pub fn lane(&mut self, rows: u32) -> LaneHandles {
        let program = self.program(rows);
        let latent = self.channel(vec![rows, WIDTH], HostRole::Writer);
        let out = self.channel(vec![rows, WIDTH], HostRole::Reader);
        let bound = self
            .engine
            .bind_instance(&InstanceBinding {
                program,
                channels: vec![latent, out],
                seeds: Vec::new(),
                geometry: GeometryClass::Host,
                extents: engine::program::BindExtents {
                    row_count: rows,
                    token_count: rows,
                    sampled_rows: rows,
                    query_len: rows,
                    ..engine::program::BindExtents::default()
                },
            })
            .expect("the instance binds");
        LaneHandles {
            instance: bound.id,
            latent,
            out,
            rows,
        }
    }

    pub fn publish(&mut self, instance: u64, dense: u32, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let accepted = self
            .engine
            .publish_channel(instance, dense, &bytes)
            .expect("the cell publishes");
        assert!(accepted, "the ring had room");
    }
}

pub fn lane(rig: &Rig, slot: u32, handles: &LaneHandles, group: u32) -> Lane {
    Lane {
        slot,
        word: rig.facts.word(&Request::new(handles.rows, false)),
        tokens: vec![0; handles.rows as usize],
        readout: Readout::Rows((0..handles.rows).collect()),
        stream: LaneStream::Text,
        group: Some(group),
        reading: 0,
        ports: vec![PortFeed {
            kind: PortKind::Latents,
            port: 0,
            channel: handles.latent,
        }],
        ..Lane::default()
    }
}

pub fn attach(lane: u32, handles: &LaneHandles) -> Attachment {
    Attachment {
        lane,
        instance: handles.instance,
        at: Boundary::Epilogue,
    }
}

pub fn frame(lanes: Vec<Lane>, attachments: Vec<Attachment>) -> FrameSubmission {
    FrameSubmission::of(Step {
        lanes,
        attachments,
        media: Vec::new(),
        voxels: Vec::new(),
    })
}

pub fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 4.0e-2 * w.abs().max(1.0),
            "{what} at {i}: device {g} against host {w}"
        );
    }
}
