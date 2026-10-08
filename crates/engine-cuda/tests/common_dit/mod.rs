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
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::{GeometryClass, ModelProfile, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use eta_ir::validate::bind;
use poem::fact;
use poem::{
    Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Platform, RaggedMask, Request, RopeForm,
    Stream, Trace, Value, Weight, ops, seam, trace_hybrid,
};

pub const WIDTH: u32 = 32;
pub const HEAD_DIM: u32 = 64;
pub const FREQ: u32 = 16;
pub const SM_SCALE: f32 = 0.125;
pub const THETA: f32 = 10_000.0;
pub const NAME: &str = "dit-mini";

thread_local! {
    /// The facts of the trace this thread's rig loaded, which its lanes are
    /// worded by.
    static FACTS: std::cell::RefCell<poem_ir::Facts> = std::cell::RefCell::default();
}

pub fn classify(request: &Request) -> u64 {
    FACTS.with(|facts| facts.borrow().word(request))
}

fn remember(trace: &Trace) {
    FACTS.with(|facts| *facts.borrow_mut() = trace.facts.clone());
}

pub struct DoubleBlock;

impl ForwardHybrid for DoubleBlock {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) -> Value {
        let (txt, img) = (
            inputs.on(fact::stream(Stream::Text)),
            inputs.on(!fact::stream(Stream::Text)),
        );
        let w = |name: &str, out: u32, inner: u32| {
            Weight::sym(name, [u64::from(out), u64::from(inner)], Dtype::Bf16)
        };
        let x_txt = txt.latents(0, WIDTH, Dtype::Bf16);
        let x_img = img.latents(1, WIDTH, Dtype::Bf16);
        let t = inputs.lane_vector(0, 1);
        let pos = inputs.axis_positions(0, 2);
        let emb = ops::elemwise::sinusoid(&t, FREQ, THETA, true, 1.0);
        let emb = ops::elemwise::silu(&emb);
        let m = ops::linear::matmul(&emb, &w("ada", 2 * WIDTH, FREQ));
        let lanes = inputs.request_of_token();
        let condition = |x: &Value| {
            let normed = ops::elemwise::layernorm_no_scale(x, 1e-6);
            ops::elemwise::modulate(&normed, &m, Some(&lanes), ModulateForm::ScaleShift)
        };
        let h_txt = condition(&x_txt);
        let h_img = condition(&x_img);
        let project = |h: &Value, prefix: &str| {
            (
                ops::linear::matmul(h, &w(&format!("{prefix}.q"), HEAD_DIM, WIDTH)),
                ops::linear::matmul(h, &w(&format!("{prefix}.k"), HEAD_DIM, WIDTH)),
                ops::linear::matmul(h, &w(&format!("{prefix}.v"), HEAD_DIM, WIDTH)),
            )
        };
        let (qt, kt, vt) = project(&h_txt, "txt");
        let (qi, ki, vi) = project(&h_img, "img");
        let q = Value::merge(vec![qt, qi]);
        let k = Value::merge(vec![kt, ki]);
        let v = Value::merge(vec![vt, vi]);
        let dims = [HEAD_DIM / 2, HEAD_DIM / 2, 0, 0];
        let thetas = [THETA, THETA, 0.0, 0.0];
        let q = ops::elemwise::rope_axes(
            &q,
            &pos,
            dims,
            thetas,
            RopeForm::Interleaved,
            HEAD_DIM,
            HEAD_DIM,
        );
        let k = ops::elemwise::rope_axes(
            &k,
            &pos,
            dims,
            thetas,
            RopeForm::Interleaved,
            HEAD_DIM,
            HEAD_DIM,
        );
        let perm = inputs.row_permutation();
        let indptr = inputs.group_indptr();
        let o = ops::attn::ragged(
            &ops::layout::pack_rows(&q, &perm),
            &ops::layout::pack_rows(&k, &perm),
            &ops::layout::pack_rows(&v, &perm),
            &indptr,
            &indptr,
            HEAD_DIM,
            SM_SCALE,
            RaggedMask::GroupBlockDiagonal,
        );
        let o = ops::layout::unpack_rows(&o, &perm);
        let (o_txt, o_img) = (
            o.on(fact::stream(Stream::Text)),
            o.on(!fact::stream(Stream::Text)),
        );
        let y_txt = ops::linear::matmul(&o_txt, &w("txt.o", WIDTH, HEAD_DIM));
        let y_img = ops::linear::matmul(&o_img, &w("img.o", WIDTH, HEAD_DIM));
        let r_txt = ops::elemwise::residual_add(&x_txt, &y_txt);
        let r_img = ops::elemwise::residual_add(&x_img, &y_img);
        let out = Value::merge(vec![r_txt, r_img]);
        seam::at(seam::VELOCITY, &[&out]);
        out
    }
}

pub fn trace() -> Trace {
    trace_hybrid(NAME, &DoubleBlock, Platform::Cuda)
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

pub struct HostRequest {
    pub text: Vec<f32>,
    pub image: Vec<f32>,
    pub text_rows: usize,
    pub image_rows: usize,
    pub timestep: f32,
    pub positions: Vec<[f32; 2]>,
}

pub fn matmul_bf16(x: &[f32], rows: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
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

pub fn matmul_f32(x: &[f32], rows: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut y = vec![0f32; rows * n];
    for r in 0..rows {
        for c in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc = x[r * k + i].mul_add(w[c * k + i], acc);
            }
            y[r * n + c] = acc;
        }
    }
    y
}

pub fn sinusoid(t: f32) -> Vec<f32> {
    let half = (FREQ / 2) as usize;
    let angles: Vec<f32> = (0..half)
        .map(|i| t * (-THETA.ln() * i as f32 / half as f32).exp())
        .collect();
    let mut row: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
    row.extend(angles.iter().map(|a| a.sin()));
    row
}

pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}

pub fn condition(x: &[f32], rows: usize, m: &[f32]) -> Vec<f32> {
    let w = WIDTH as usize;
    let mut y = vec![0f32; rows * w];
    for r in 0..rows {
        let row = &x[r * w..(r + 1) * w];
        let mean = row.iter().sum::<f32>() / w as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / w as f32;
        let inv = 1.0 / (var + 1e-6).sqrt();
        for c in 0..w {
            let normed = (row[c] - mean) * inv;
            y[r * w + c] = bf(normed.mul_add(1.0 + m[c], m[w + c]));
        }
    }
    y
}

pub fn rope(x: &mut [f32], rows: usize, positions: &[[f32; 2]]) {
    let hd = HEAD_DIM as usize;
    let block = hd / 2;
    for r in 0..rows {
        for axis in 0..2 {
            let base = axis * block;
            for i in 0..block / 2 {
                let angle = positions[r][axis] * THETA.powf(-2.0 * i as f32 / block as f32);
                let (s, c) = angle.sin_cos();
                let a = x[r * hd + base + 2 * i];
                let b = x[r * hd + base + 2 * i + 1];
                x[r * hd + base + 2 * i] = bf(a * c - b * s);
                x[r * hd + base + 2 * i + 1] = bf(a * s + b * c);
            }
        }
    }
}

pub fn attention(q: &[f32], k: &[f32], v: &[f32], rows: usize) -> Vec<f32> {
    let hd = HEAD_DIM as usize;
    let mut o = vec![0f32; rows * hd];
    for i in 0..rows {
        let scores: Vec<f32> = (0..rows)
            .map(|j| (0..hd).map(|d| q[i * hd + d] * k[j * hd + d]).sum::<f32>() * SM_SCALE)
            .collect();
        let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scores.iter().map(|s| (s - peak).exp()).collect();
        let mass: f32 = weights.iter().sum();
        for d in 0..hd {
            let mut acc = 0f32;
            for j in 0..rows {
                acc += bf(weights[j] / mass) * v[j * hd + d];
            }
            o[i * hd + d] = bf(acc);
        }
    }
    o
}

pub fn reference(weights: &Weights, request: &HostRequest) -> (Vec<f32>, Vec<f32>) {
    let w = WIDTH as usize;
    let hd = HEAD_DIM as usize;
    let emb = silu(&sinusoid(request.timestep));
    let m = matmul_f32(&emb, 1, FREQ as usize, weights.get("ada"), 2 * w);
    let h_txt = condition(&request.text, request.text_rows, &m);
    let h_img = condition(&request.image, request.image_rows, &m);
    let project = |h: &[f32], rows: usize, prefix: &str| {
        (
            matmul_bf16(h, rows, w, weights.get(&format!("{prefix}.q")), hd),
            matmul_bf16(h, rows, w, weights.get(&format!("{prefix}.k")), hd),
            matmul_bf16(h, rows, w, weights.get(&format!("{prefix}.v")), hd),
        )
    };
    let (mut qt, mut kt, vt) = project(&h_txt, request.text_rows, "txt");
    let (mut qi, mut ki, vi) = project(&h_img, request.image_rows, "img");
    let text_positions = &request.positions[..request.text_rows];
    let image_positions = &request.positions[request.text_rows..];
    rope(&mut qt, request.text_rows, text_positions);
    rope(&mut kt, request.text_rows, text_positions);
    rope(&mut qi, request.image_rows, image_positions);
    rope(&mut ki, request.image_rows, image_positions);
    let rows = request.text_rows + request.image_rows;
    let cat = |a: &[f32], b: &[f32]| {
        let mut out = a.to_vec();
        out.extend_from_slice(b);
        out
    };
    let o = attention(&cat(&qt, &qi), &cat(&kt, &ki), &cat(&vt, &vi), rows);
    let o_txt = &o[..request.text_rows * hd];
    let o_img = &o[request.text_rows * hd..];
    let y_txt = matmul_bf16(o_txt, request.text_rows, hd, weights.get("txt.o"), w);
    let y_img = matmul_bf16(o_img, request.image_rows, hd, weights.get("img.o"), w);
    let fold =
        |y: &[f32], x: &[f32]| -> Vec<f32> { y.iter().zip(x).map(|(a, b)| bf(a + b)).collect() };
    (fold(&y_txt, &request.text), fold(&y_img, &request.image))
}

pub fn epilogue(rows: u32) -> TraceContainer {
    let writer = |shape: Shape| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(EtaDtype::F32),
        capacity: 2,
        host_role: HostRole::Writer,
        seeded: false,
    };
    let reader = |shape: Shape| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(EtaDtype::F32),
        capacity: 2,
        host_role: HostRole::Reader,
        seeded: false,
    };
    let ops = vec![
        Op::ChanTake(0),
        Op::IntrinsicVal {
            intr: IntrinsicId::Velocity,
            shape: Shape::matrix(rows, WIDTH),
            dtype: EtaDtype::F32,
        },
        Op::Add(0, 1),
        Op::ChanPut { chan: 3, value: 2 },
    ];
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            writer(Shape::matrix(rows, WIDTH)),
            writer(Shape::matrix(1, 1)),
            writer(Shape::matrix(rows, 2)),
            reader(Shape::matrix(rows, WIDTH)),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops,
        }],
        externs: Vec::new(),
    }
}

pub struct LaneHandles {
    pub instance: u64,
    pub latent: u64,
    pub timestep: u64,
    pub positions: u64,
    pub out: u64,
    pub rows: u32,
}

pub struct Rig {
    pub engine: engine_cuda::Cuda,
    pub loaded: Loaded,
    pub programs: BTreeMap<u32, u64>,
    next_channel: u64,
    _dir: tempfile::TempDir,
}

impl Rig {
    pub fn load(weights: &Weights, max_tokens: u32, buckets: Vec<u32>) -> Rig {
        Rig::load_plan(trace(), weights, max_tokens, buckets)
    }

    pub fn load_plan(plan: Trace, weights: &Weights, max_tokens: u32, buckets: Vec<u32>) -> Rig {
        Rig::load_recording(
            plan,
            weights,
            max_tokens,
            buckets,
            engine_cuda::Recording::default(),
        )
    }

    pub fn load_recording(
        plan: Trace,
        weights: &Weights,
        max_tokens: u32,
        buckets: Vec<u32>,
        recording: engine_cuda::Recording,
    ) -> Rig {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let path = weights.write(dir.path());
        let mut boot = engine_cuda::DeviceBoot::default();
        boot.knobs.recording = recording;
        let mut engine = engine_cuda::open(boot, contract_for).expect("the engine opens");
        let loaded = engine
            .load(LoadRequest {
                trace: {
                    remember(&plan);
                    plan
                },
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
            .expect("the miniature loads");
        Rig {
            engine,
            loaded,
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
        let id = self.register(epilogue(rows), u64::from(rows));
        self.programs.insert(rows, id);
        id
    }

    pub fn register(&mut self, container: TraceContainer, salt: u64) -> u64 {
        let bound = bind(container, self.profile().clone()).expect("the epilogue binds");
        let stages = compile_bound(&bound);
        let launch = eta_compiler::codegen::launch::build(&bound, &stages);
        let backend = Backend::parse("cuda").expect("the cuda backend");
        let registration = ProgramRegistration {
            program_hash: 0xd17 ^ salt,
            emitted_kernels: emit_program(backend, &stages, &bound),
            emitter_version: backend.emitter_version(),
            region_analysis: eta_compiler::codegen::cuda::region_analysis::analyze_program(&stages),
            launch,
            ..Default::default()
        };
        self.engine
            .register_program(&registration)
            .expect("the program registers")
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
        self.lane_of(program, rows, WIDTH)
    }

    pub fn lane_of(&mut self, program: u64, rows: u32, width: u32) -> LaneHandles {
        let latent = self.channel(vec![rows, WIDTH], HostRole::Writer);
        let timestep = self.channel(vec![1, 1], HostRole::Writer);
        let positions = self.channel(vec![rows, 2], HostRole::Writer);
        let out = self.channel(vec![rows, width], HostRole::Reader);
        let bound = self
            .engine
            .bind_instance(&InstanceBinding {
                program,
                channels: vec![latent, timestep, positions, out],
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
            timestep,
            positions,
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

    pub fn take(&mut self, instance: u64, dense: u32) -> Vec<f32> {
        let bytes = self
            .engine
            .take_channel(instance, dense)
            .expect("the channel reads")
            .expect("the channel holds a cell");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

pub fn lane(slot: u32, handles: &LaneHandles, stream: LaneStream, group: u32) -> Lane {
    let port = match stream {
        LaneStream::Text => 0,
        _ => 1,
    };
    Lane {
        slot,
        word: classify(&Request::new(handles.rows, false).on_stream(match stream {
            LaneStream::Text => Stream::Text,
            _ => Stream::Image,
        })),
        tokens: vec![0; handles.rows as usize],
        readout: Readout::Rows((0..handles.rows).collect()),
        stream,
        group: Some(group),
        reading: 0,
        ports: vec![
            PortFeed {
                kind: PortKind::Latents,
                port,
                channel: handles.latent,
            },
            PortFeed {
                kind: PortKind::LaneVector,
                port: 0,
                channel: handles.timestep,
            },
            PortFeed {
                kind: PortKind::AxisPositions,
                port: 0,
                channel: handles.positions,
            },
        ],
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
