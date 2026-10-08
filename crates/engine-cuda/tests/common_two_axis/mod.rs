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
    StepVoxels,
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
use poem::ops::spatial::{self, Conv};
use poem::{
    Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Platform, RaggedMask, Request, RopeForm,
    Trace, Value, Weight, ops, seam, trace_hybrid,
};

pub const WIDTH: u32 = 32;
pub const HEAD_DIM: u32 = 64;
pub const FREQ: u32 = 16;
pub const SM_SCALE: f32 = 0.125;
pub const THETA: f32 = 10_000.0;
pub const C_IN: u32 = 8;
pub const C_OUT: u32 = 4;
pub const TAPS: u32 = 27;
pub const NAME: &str = "two-axis-mini";
pub const VAE_READING: u8 = 1;

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

#[must_use]
pub fn word(reading: u8) -> u64 {
    let request = Request::new(1, false);
    if reading == VAE_READING {
        classify(&request.in_reading("vae"))
    } else {
        classify(&request)
    }
}

pub struct TwoAxis;

impl ForwardHybrid for TwoAxis {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) -> Value {
        let (vae, dit) = (
            inputs.on(fact::reading("vae")),
            inputs.on(!fact::reading("vae")),
        );
        let w = |name: &str, out: u32, inner: u32| {
            Weight::sym(name, [u64::from(out), u64::from(inner)], Dtype::Bf16)
        };

        let x = dit.latents(0, WIDTH, Dtype::Bf16);
        let t = dit.lane_vector(0, 1);
        let pos = dit.axis_positions(0, 2);
        let emb = ops::elemwise::sinusoid(&t, FREQ, THETA, true, 1.0);
        let emb = ops::elemwise::silu(&emb);
        let m = ops::linear::matmul(&emb, &w("ada", 2 * WIDTH, FREQ));
        let lanes = dit.request_of_token();
        let normed = ops::elemwise::layernorm_no_scale(&x, 1e-6);
        let h = ops::elemwise::modulate(&normed, &m, Some(&lanes), ModulateForm::ScaleShift);
        let q = ops::linear::matmul(&h, &w("q", HEAD_DIM, WIDTH));
        let k = ops::linear::matmul(&h, &w("k", HEAD_DIM, WIDTH));
        let v = ops::linear::matmul(&h, &w("v", HEAD_DIM, WIDTH));
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
        let perm = dit.row_permutation();
        let indptr = dit.group_indptr();
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
        let y = ops::linear::matmul(&o, &w("o", WIDTH, HEAD_DIM));
        let out = ops::elemwise::residual_add(&x, &y);
        seam::at(seam::VELOCITY, &[&out]);

        let grid = vae.grid();
        let xv = vae.voxels(0, C_IN, Dtype::Bf16);
        let conv = Weight::sym(
            "conv",
            [u64::from(C_OUT), u64::from(C_IN) * u64::from(TAPS)],
            Dtype::Bf16,
        )
        .conv_taps_major(C_IN, TAPS);
        let (pixels, pixel_grid) = spatial::conv3d(&xv, &grid, &conv, None, Conv::same3(), None);
        seam::at(seam::PIXELS, &[&pixels, &pixel_grid]);

        out
    }
}

#[must_use]
pub fn trace() -> Trace {
    trace_hybrid(NAME, &TwoAxis, Platform::Cuda)
}

#[must_use]
pub fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(round)) >> 16) as u16
}

#[must_use]
pub fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

#[must_use]
pub fn bf(x: f32) -> f32 {
    from_bf16(to_bf16(x))
}

pub struct Lcg(u64);

impl Lcg {
    #[must_use]
    pub fn seeded(seed: u64) -> Lcg {
        Lcg(seed | 1)
    }
    pub fn unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }
}

pub struct Weights {
    pub planes: BTreeMap<String, (Vec<u64>, Vec<f32>)>,
}

impl Weights {
    #[must_use]
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

    #[must_use]
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

#[must_use]
pub fn conv_reference(weights: &Weights, clip: [u32; 3], x: &[f32]) -> Vec<f32> {
    let [t, h, w] = clip.map(|n| n as usize);
    let (c_in, c_out) = (C_IN as usize, C_OUT as usize);
    let plane = weights.get("conv");
    let mut y = vec![0f32; t * h * w * c_out];
    for ti in 0..t {
        for hi in 0..h {
            for wi in 0..w {
                let row = (ti * h + hi) * w + wi;
                for oc in 0..c_out {
                    let mut acc = 0f32;
                    for (tap, (dt, dh, dw)) in taps().enumerate() {
                        let (st, sh, sw) = (ti as i32 + dt, hi as i32 + dh, wi as i32 + dw);
                        if st < 0
                            || sh < 0
                            || sw < 0
                            || st >= t as i32
                            || sh >= h as i32
                            || sw >= w as i32
                        {
                            continue;
                        }
                        let src = ((st as usize * h + sh as usize) * w + sw as usize) * c_in;
                        for ic in 0..c_in {
                            let at = oc * (c_in * TAPS as usize) + ic * TAPS as usize + tap;
                            acc += bf(x[src + ic]) * plane[at];
                        }
                    }
                    y[row * c_out + oc] = bf(acc);
                }
            }
        }
    }
    y
}

fn taps() -> impl Iterator<Item = (i32, i32, i32)> {
    (-1..=1).flat_map(|dt| (-1..=1).flat_map(move |dh| (-1..=1).map(move |dw| (dt, dh, dw))))
}

#[must_use]
pub fn pixel_epilogue(clip: [u32; 3], rows: u32) -> TraceContainer {
    let cell = Shape::new(&[clip[1], clip[2], C_IN]).expect("the clip's box");
    let plane = Shape::matrix(rows, C_OUT);
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            ChannelDecl {
                shape: cell,
                dtype: ChanDType::Concrete(EtaDtype::F32),
                capacity: 2,
                host_role: HostRole::Writer,
                seeded: false,
            },
            ChannelDecl {
                shape: plane,
                dtype: ChanDType::Concrete(EtaDtype::F32),
                capacity: 2,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::IntrinsicVal {
                    intr: IntrinsicId::Pixels,
                    shape: plane,
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: Vec::new(),
    }
}

#[must_use]
pub fn velocity_epilogue(rows: u32) -> TraceContainer {
    let writer = |shape: Shape| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(EtaDtype::F32),
        capacity: 2,
        host_role: HostRole::Writer,
        seeded: false,
    };
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            writer(Shape::matrix(rows, WIDTH)),
            writer(Shape::matrix(1, 1)),
            writer(Shape::matrix(rows, 2)),
            ChannelDecl {
                shape: Shape::matrix(rows, WIDTH),
                dtype: ChanDType::Concrete(EtaDtype::F32),
                capacity: 2,
                host_role: HostRole::Reader,
                seeded: false,
            },
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::IntrinsicVal {
                    intr: IntrinsicId::Velocity,
                    shape: Shape::matrix(rows, WIDTH),
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 3, value: 1 },
            ],
        }],
        externs: Vec::new(),
    }
}

pub struct Rig {
    pub engine: engine_cuda::Cuda,
    pub loaded: Loaded,
    next_channel: u64,
    _dir: tempfile::TempDir,
}

impl Rig {
    pub fn load(weights: &Weights, max_tokens: u32, buckets: Vec<u32>, max_voxels: u32) -> Rig {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let path = weights.write(dir.path());
        let mut engine = engine_cuda::open(engine_cuda::DeviceBoot::default(), contract_for)
            .expect("the engine opens");
        let loaded = engine
            .load(LoadRequest {
                trace: {
                    let trace = trace();
                    remember(&trace);
                    trace
                },
                checkpoint: Checkpoint::Path(path),
                budgets: Budgets {
                    max_lanes: 4,
                    max_tokens,
                    buckets,
                    max_adapters: 0,
                    page_size: 16,
                    max_context: 64,
                    slots: 4,
                    pages: 32,
                    max_patches: None,
                    max_images: None,
                    max_voxels: Some(max_voxels),
                    max_clips: Some(2),
                },
                residency: Residency::default(),
                ordinal: 0,
                frames_in_flight: 1,
            })
            .expect("the two-axis miniature loads");
        Rig {
            engine,
            loaded,
            next_channel: 1,
            _dir: dir,
        }
    }

    #[must_use]
    pub fn profile(&self) -> &ModelProfile {
        &self.loaded.caps.profile
    }

    pub fn register(&mut self, container: TraceContainer, salt: u64) -> u64 {
        let bound = bind(container, self.profile().clone()).expect("the epilogue binds");
        let stages = compile_bound(&bound);
        let launch = eta_compiler::codegen::launch::build(&bound, &stages);
        let backend = Backend::parse("cuda").expect("the cuda backend");
        let registration = ProgramRegistration {
            program_hash: 0x2a ^ salt,
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

    pub fn channel(&mut self, shape: Vec<u32>, host_role: HostRole) -> u64 {
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

    pub fn bind(&mut self, program: u64, channels: Vec<u64>, rows: u32) -> u64 {
        self.engine
            .bind_instance(&InstanceBinding {
                program,
                channels,
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
            .expect("the instance binds")
            .id
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

#[must_use]
pub fn vae_lane(slot: u32, channel: u64) -> Lane {
    Lane {
        slot,
        word: word(VAE_READING),
        tokens: vec![0],
        readout: Readout::None,
        stream: LaneStream::Image,
        group: None,
        reading: VAE_READING,
        ports: vec![PortFeed {
            kind: PortKind::Voxels,
            port: 0,
            channel,
        }],
        ..Lane::default()
    }
}

#[must_use]
pub fn dit_lane(slot: u32, rows: u32, latent: u64, timestep: u64, positions: u64) -> Lane {
    Lane {
        slot,
        word: word(0),
        tokens: vec![0; rows as usize],
        readout: Readout::Rows((0..rows).collect()),
        stream: LaneStream::Image,
        group: Some(0),
        reading: 0,
        ports: vec![
            PortFeed {
                kind: PortKind::Latents,
                port: 0,
                channel: latent,
            },
            PortFeed {
                kind: PortKind::LaneVector,
                port: 0,
                channel: timestep,
            },
            PortFeed {
                kind: PortKind::AxisPositions,
                port: 0,
                channel: positions,
            },
        ],
        ..Lane::default()
    }
}

#[must_use]
pub fn attach(lane: u32, instance: u64) -> Attachment {
    Attachment {
        lane,
        instance,
        at: Boundary::Epilogue,
    }
}

#[must_use]
pub fn frame(
    lanes: Vec<Lane>,
    attachments: Vec<Attachment>,
    voxels: Vec<StepVoxels>,
) -> FrameSubmission {
    FrameSubmission::of(Step {
        lanes,
        attachments,
        media: Vec::new(),
        voxels,
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
