
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use checkpoint::file::emit::{self, Object};
use checkpoint::serving::Stamp;
use engine::Engine;
use engine::channel::ChannelRegistration;
use engine::fire::{Attachment, FrameSubmission, InputFeed, Lane, Readout, Step, StepVoxels};
use engine::load::{Budgets, Checkpoint, LoadRequest, Loaded, Residency};
use engine::program::{InstanceBinding, ProgramRegistration};
use engine::{ChannelId, InstanceId, ProgramId};
use eta_compiler::codegen::program::{Backend, emit_program};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem::fact;
use poem::ops::conv::{self, ConvShape};
use poem::{
    Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Platform, RaggedMask, Request, RopeForm,
    Trace, Weight, ops, trace_hybrid,
};
use poem_ir::InputAxis;

pub const WIDTH: u32 = 32;
pub const HEAD_DIM: u32 = 64;
pub const FREQ: u32 = 16;
pub const SM_SCALE: f32 = 0.125;
pub const THETA: f32 = 10_000.0;
pub const C_IN: u32 = 8;
pub const C_OUT: u32 = 4;
pub const TAPS: u32 = 27;
pub const NAME: &str = "two-axis-mini";

pub struct TwoAxis;

impl ForwardHybrid for TwoAxis {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) {
        inputs
            .recorder()
            .declare_switch("reading", &["denoise", "vae"]);
        let (vae, dit) = (
            inputs.on(fact::switch_is("reading", "vae")),
            inputs.on(!fact::switch_is("reading", "vae")),
        );
        let w = |name: &str, out: u32, inner: u32| {
            Weight::sym(name, [u64::from(out), u64::from(inner)], Dtype::Bf16)
        };

        let x = dit.input("latents", Dtype::Bf16, InputAxis::Row, WIDTH, None);
        let t = dit.input("timestep", Dtype::F32, InputAxis::Lane, 1, None);
        let pos = dit.input("positions", Dtype::F32, InputAxis::Row, 2, None);
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
            &ops::gather::pack_rows(&q, &perm),
            &ops::gather::pack_rows(&k, &perm),
            &ops::gather::pack_rows(&v, &perm),
            &indptr,
            &indptr,
            HEAD_DIM,
            SM_SCALE,
            RaggedMask::GroupBlockDiagonal,
        );
        let o = ops::gather::unpack_rows(&o, &perm);
        let y = ops::linear::matmul(&o, &w("o", WIDTH, HEAD_DIM));
        let out = ops::elemwise::residual_add(&x, &y);
        poem::output("velocity", &out, None);

        let grid = vae.grid();
        let xv = vae.input("voxels", Dtype::Bf16, InputAxis::Voxel, C_IN, None);
        let conv = Weight::sym(
            "conv",
            [u64::from(C_OUT), u64::from(C_IN) * u64::from(TAPS)],
            Dtype::Bf16,
        )
        .conv_taps_major(C_IN, TAPS);
        let (pixels, pixel_grid) = conv::conv3d(&xv, &grid, &conv, None, ConvShape::same3(), None);
        poem::output("pixels", &pixels, Some(&pixel_grid));
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
        names: vec!["pixels".to_string()],
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
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: plane,
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 1, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
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
        names: vec!["velocity".to_string()],
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
                Op::Output {
                    index: 0,
                    lane: 0,
                    shape: Shape::matrix(rows, WIDTH),
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 3, value: 1 },
            ],
        }],
        externs: Vec::new(),
        inputs: Vec::new(),
    }
}

pub struct Rig {
    pub engine: engine_cuda::Cuda,
    pub loaded: Loaded,
    pub facts: poem_ir::Facts,
    /// The plan's guest inputs by name, in id order.
    pub inputs: Vec<String>,
    next_channel: u64,
    _dir: tempfile::TempDir,
}

impl Rig {
    pub fn load(weights: &Weights, max_tokens: u32, buckets: Vec<u32>, max_voxels: u32) -> Rig {
        let dir = tempfile::tempdir().expect("a scratch directory");
        let path = weights.write(dir.path());
        let mut engine = engine_cuda::open(engine_cuda::DeviceBoot::default(), contract_for)
            .expect("the engine opens");
        let trace = trace();
        let facts = trace.facts.clone();
        let inputs = trace
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();
        let loaded = engine
            .load(LoadRequest {
                trace,
                checkpoint: Checkpoint::Path(path),
                budgets: Budgets {
                    max_lanes: 4,
                    max_tokens,
                    buckets,
                    max_adapters: 0,
                    page_size: 16,
                    max_context: 64,
                    context_declared: false,
                    slots: 4,
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
            facts,
            inputs,
            next_channel: 1,
            _dir: dir,
        }
    }

    #[must_use]
    pub fn input(&self, name: &str) -> u32 {
        let id = self
            .inputs
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("the plan reads no input `{name}`"));
        u32::try_from(id).expect("fewer than 2^32 inputs")
    }

    /// The word of a one-row lane running `reading`, or the plan's token
    /// reading when `None`.
    #[must_use]
    pub fn word(&self, reading: Option<&str>) -> u64 {
        let request = Request::new(1, false);
        self.facts.word(&match reading {
            Some(reading) => request.switched("reading", reading),
            None => request,
        })
    }

    #[must_use]
    pub fn vae_lane(&self, slot: u32, channel: ChannelId) -> Lane {
        Lane {
            slot,
            word: self.word(Some("vae")),
            tokens: vec![0],
            readout: Readout::None,
            group: None,
            inputs: vec![InputFeed {
                id: self.input("voxels"),
                channel,
            }],
            ..Lane::default()
        }
    }

    #[must_use]
    pub fn dit_lane(
        &self,
        slot: u32,
        rows: u32,
        latent: ChannelId,
        timestep: ChannelId,
        positions: ChannelId,
    ) -> Lane {
        Lane {
            slot,
            word: self.word(None),
            tokens: vec![0; rows as usize],
            readout: Readout::Rows((0..rows).collect()),
            group: Some(0),
            inputs: vec![
                InputFeed {
                    id: self.input("latents"),
                    channel: latent,
                },
                InputFeed {
                    id: self.input("timestep"),
                    channel: timestep,
                },
                InputFeed {
                    id: self.input("positions"),
                    channel: positions,
                },
            ],
            ..Lane::default()
        }
    }

    pub fn reads_out(&self, name: &str) -> Option<u32> {
        self.loaded
            .caps
            .classes
            .classes
            .iter()
            .flat_map(|class| &class.outputs)
            .find(|decl| decl.name == name)
            .map(|decl| decl.width)
    }

    pub fn register(&mut self, container: TraceContainer, salt: u64) -> ProgramId {
        let bound = crate::bound::loaded(container, &self.loaded.caps).expect("the epilogue binds");
        let stages = compile_bound(&bound);
        let launch = eta_compiler::codegen::launch::build(&bound, &stages);
        let backend =
            Backend::parse(engine_cuda::api::CODEGEN).expect("the engine's codegen backend");
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

    pub fn channel(&mut self, shape: Vec<u32>, host_role: HostRole) -> ChannelId {
        let id = ChannelId(self.next_channel);
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

    pub fn bind(&mut self, program: ProgramId, channels: Vec<ChannelId>, rows: u32) -> InstanceId {
        self.engine
            .bind_instance(&InstanceBinding {
                program,
                channels,
                seeds: Vec::new(),
                geometry: GeometryClass::Host,
                device_resolved: Default::default(),
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

    pub fn publish(&mut self, instance: InstanceId, dense: u32, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let accepted = self
            .engine
            .publish_channel(instance, dense, &bytes)
            .expect("the cell publishes");
        assert!(accepted, "the ring had room");
    }

    pub fn take(&mut self, instance: InstanceId, dense: u32) -> Vec<f32> {
        let bytes = self
            .engine
            .take_channel(instance, dense)
            .expect("the channel reads")
            .expect("the channel holds a cell");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }
}

#[must_use]
pub fn attach(lane: u32, instance: InstanceId) -> Attachment {
    Attachment {
        lane,
        lanes: 1,
        instance,
        adapter_only: false,
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
