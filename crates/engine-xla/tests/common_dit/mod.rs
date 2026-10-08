//! The diffusion miniatures engine-cuda's `common_dit` and `common_two_axis`
//! gate on, for the xla engine: a double-stream DiT block (latent, lane
//! vector and axis-position ports, grouped ragged attention over a row
//! permutation, a velocity seam), and a two-axis plan whose VAE reading
//! convolves a voxel port into a pixels seam. Weights are drawn in-process,
//! the answers checked against host references.
//!
//! The lanes attach a carrier program that takes each fed channel's cell
//! (so every fire reads the next cell, as engine-cuda's epilogues make it);
//! guest intrinsics over velocity and pixels are the guest runtime's to
//! lower, so the gates read the engine's own readouts.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use checkpoint::contract::ModelContract;
use engine::Engine;
use engine::channel::ChannelRegistration;
use engine::fire::{
    Attachment, Boundary, FrameSubmission, Lane, LaneStream, PortFeed, PortKind, Readout, Step,
    StepVoxels,
};
use engine::load::{Budgets, Checkpoint, LoadRequest, Loaded, Residency};
use engine::program::{InstanceBinding, ProgramRegistration};
use engine_xla::{DeviceBoot, Xla};
use eta_compiler::plan::compile_bound;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::Op;
use eta_ir::registry::{GeometryClass, ModelProfile, Stage};
use eta_ir::types::{Dtype as EtaDtype, Shape};
use eta_ir::validate::bind;
use poem::fact;
use poem::ops::spatial::{self, Conv};
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
pub const C_IN: u32 = 8;
pub const C_OUT: u32 = 4;
pub const TAPS: u32 = 27;
pub const VAE_READING: u8 = 1;

/// Whether a PJRT plugin is here to run on; device gates skip otherwise.
#[must_use]
pub fn device() -> Option<engine_xla::bench::DeviceLock> {
    let lock = engine_xla::bench::lock_device();
    // The last holder lets go of the device lock a moment before libtpu
    // lets go of the chip: a busy chip is retried for a while.
    let mut tries = 0;
    let found = loop {
        match engine_xla::pjrt::Api::discover(None) {
            Err(why)
                if tries < 60
                    && (why.to_string().contains("in use")
                        || why.to_string().contains("lockfile")) =>
            {
                tries += 1;
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
            other => break other,
        }
    };
    match found {
        Ok(_) => Some(lock),
        Err(why) if why.to_string().contains("in use") || why.to_string().contains("lockfile") => {
            panic!("the device is held outside the device lock: {why}")
        }
        Err(why) => {
            eprintln!("no PJRT plugin, skipping: {why}");
            None
        }
    }
}

// ------------------------------------------------------------ the DiT block

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
    trace_hybrid(NAME, &DoubleBlock, Platform::Xla)
}

// ------------------------------------------------------------ the two-axis plan

/// The word of a lane of the two-axis plan running `reading` (1 is the VAE).
#[must_use]
pub fn vae_word(reading: u8) -> u64 {
    let request = Request::new(1, false);
    let request = if reading == VAE_READING {
        request.in_reading("vae")
    } else {
        request
    };
    two_axis().facts.word(&request)
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
        let emb = ops::elemwise::silu(&ops::elemwise::sinusoid(&t, FREQ, THETA, true, 1.0));
        let m = ops::linear::matmul(&emb, &w("ada", 2 * WIDTH, FREQ));
        let lanes = dit.request_of_token();
        let normed = ops::elemwise::layernorm_no_scale(&x, 1e-6);
        let h = ops::elemwise::modulate(&normed, &m, Some(&lanes), ModulateForm::ScaleShift);
        let y = ops::linear::matmul(&h, &w("o", WIDTH, WIDTH));
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

pub fn two_axis() -> Trace {
    trace_hybrid("two-axis-mini", &TwoAxis, Platform::Xla)
}

// ------------------------------------------------------------ numbers

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

    /// Every plane as bf16, in an unstamped container.
    pub fn write(&self, path: &Path) {
        let mut writer = ztensor::Writer::create(path).expect("the container opens");
        for (name, (shape, values)) in &self.planes {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|v| to_bf16(*v).to_le_bytes())
                .collect();
            writer
                .add(name, shape.clone(), ztensor::Leaf::BF16, &bytes)
                .unwrap_or_else(|why| panic!("`{name}`: {why}"));
        }
        writer.finish().expect("the container closes");
    }
}

pub fn contract_for(trace: &Trace, path: &Path) -> Result<ModelContract, String> {
    let source = ztensor_compat::index(path).map_err(|why| why.to_string())?;
    poem::import::own_contract(&source, &trace.params, 1, Platform::Xla)
        .map_err(|why| why.to_string())
}

// ------------------------------------------------------------ host references

pub struct HostRequest {
    pub text: Vec<f32>,
    pub image: Vec<f32>,
    pub text_rows: usize,
    pub image_rows: usize,
    pub timestep: f32,
    pub positions: Vec<[f32; 2]>,
}

pub fn matmul(x: &[f32], rows: usize, k: usize, w: &[f32], n: usize, round: bool) -> Vec<f32> {
    let mut y = vec![0f32; rows * n];
    for r in 0..rows {
        for c in 0..n {
            let mut acc = 0f32;
            for i in 0..k {
                acc = x[r * k + i].mul_add(w[c * k + i], acc);
            }
            y[r * n + c] = if round { bf(acc) } else { acc };
        }
    }
    y
}

pub fn modulation(weights: &Weights, timestep: f32) -> Vec<f32> {
    let half = (FREQ / 2) as usize;
    let angles: Vec<f32> = (0..half)
        .map(|i| timestep * (-THETA.ln() * i as f32 / half as f32).exp())
        .collect();
    let mut row: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
    row.extend(angles.iter().map(|a| a.sin()));
    let emb: Vec<f32> = row.iter().map(|v| v / (1.0 + (-v).exp())).collect();
    matmul(
        &emb,
        1,
        FREQ as usize,
        weights.get("ada"),
        2 * WIDTH as usize,
        false,
    )
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
    for (r, at) in positions.iter().enumerate().take(rows) {
        for (axis, &pos) in at.iter().enumerate() {
            let base = axis * block;
            for i in 0..block / 2 {
                let angle = pos * THETA.powf(-2.0 * i as f32 / block as f32);
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

/// One request's text and image velocities through the double block.
pub fn reference(weights: &Weights, request: &HostRequest) -> (Vec<f32>, Vec<f32>) {
    reference_with(weights, request, true)
}

/// `reference`, with the two streams attending jointly or each only to
/// itself (an attention class table that keeps text and image apart).
pub fn reference_with(
    weights: &Weights,
    request: &HostRequest,
    joint: bool,
) -> (Vec<f32>, Vec<f32>) {
    let w = WIDTH as usize;
    let hd = HEAD_DIM as usize;
    let m = modulation(weights, request.timestep);
    let h_txt = condition(&request.text, request.text_rows, &m);
    let h_img = condition(&request.image, request.image_rows, &m);
    let project = |h: &[f32], rows: usize, prefix: &str| {
        (
            matmul(h, rows, w, weights.get(&format!("{prefix}.q")), hd, true),
            matmul(h, rows, w, weights.get(&format!("{prefix}.k")), hd, true),
            matmul(h, rows, w, weights.get(&format!("{prefix}.v")), hd, true),
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
    let o = if joint {
        attention(&cat(&qt, &qi), &cat(&kt, &ki), &cat(&vt, &vi), rows)
    } else {
        cat(
            &attention(&qt, &kt, &vt, request.text_rows),
            &attention(&qi, &ki, &vi, request.image_rows),
        )
    };
    let o_txt = &o[..request.text_rows * hd];
    let o_img = &o[request.text_rows * hd..];
    let y_txt = matmul(o_txt, request.text_rows, hd, weights.get("txt.o"), w, true);
    let y_img = matmul(o_img, request.image_rows, hd, weights.get("img.o"), w, true);
    let fold =
        |y: &[f32], x: &[f32]| -> Vec<f32> { y.iter().zip(x).map(|(a, b)| bf(a + b)).collect() };
    (fold(&y_txt, &request.text), fold(&y_img, &request.image))
}

/// The two-axis plan's VAE reading: a `same3` convolution, natural weights.
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
                    let mut tap = 0;
                    for dt in -1i32..=1 {
                        for dh in -1i32..=1 {
                            for dw in -1i32..=1 {
                                let (st, sh, sw) = (ti as i32 + dt, hi as i32 + dh, wi as i32 + dw);
                                let inside = st >= 0
                                    && sh >= 0
                                    && sw >= 0
                                    && st < t as i32
                                    && sh < h as i32
                                    && sw < w as i32;
                                if inside {
                                    let src =
                                        ((st as usize * h + sh as usize) * w + sw as usize) * c_in;
                                    for ic in 0..c_in {
                                        let at =
                                            oc * (c_in * TAPS as usize) + ic * TAPS as usize + tap;
                                        acc += bf(x[src + ic]) * plane[at];
                                    }
                                }
                                tap += 1;
                            }
                        }
                    }
                    y[row * c_out + oc] = bf(acc);
                }
            }
        }
    }
    y
}

// ------------------------------------------------------------ the engine rig

/// A program that takes one cell off each of its first `fed` channels.
pub fn carrier(shapes: &[Shape], fed: u32) -> TraceContainer {
    TraceContainer {
        names: Vec::new(),
        channels: shapes
            .iter()
            .map(|shape| ChannelDecl {
                shape: *shape,
                dtype: ChanDType::Concrete(EtaDtype::F32),
                capacity: 2,
                host_role: HostRole::Writer,
                seeded: false,
            })
            .collect(),
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: (0..fed).map(Op::ChanTake).collect(),
        }],
        externs: Vec::new(),
    }
}

pub struct LaneHandles {
    pub instance: u64,
    pub latent: u64,
    pub timestep: u64,
    pub positions: u64,
    pub rows: u32,
}

pub struct Rig {
    pub engine: Xla,
    pub loaded: Loaded,
    programs: BTreeMap<u32, u64>,
    next_channel: u64,
    dir: PathBuf,
}

impl Drop for Rig {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

impl Rig {
    pub fn load(
        plan: Trace,
        weights: &Weights,
        max_tokens: u32,
        buckets: Vec<u32>,
        max_voxels: Option<u32>,
    ) -> Rig {
        let dir = std::env::temp_dir().join(format!(
            "pie-xla-{}-{}-{}",
            plan.name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos())
        ));
        std::fs::create_dir_all(&dir).expect("a scratch directory");
        let path = dir.join("weights.zt");
        weights.write(&path);
        let mut engine = Xla::new(DeviceBoot::default(), contract_for);
        let loaded = engine
            .load(LoadRequest {
                trace: {
                    remember(&plan);
                    plan
                },
                checkpoint: Checkpoint::Path(path),
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
                    max_voxels,
                    max_clips: max_voxels.map(|_| 2),
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
            dir,
        }
    }

    pub fn profile(&self) -> &ModelProfile {
        &self.loaded.caps.profile
    }

    pub fn register(&mut self, container: TraceContainer) -> u64 {
        let bound = bind(container, self.profile().clone()).expect("the program binds");
        let launch = eta_compiler::codegen::launch::build(&bound, &compile_bound(&bound));
        self.engine
            .register_program(&ProgramRegistration {
                launch,
                ..Default::default()
            })
            .expect("the program registers")
    }

    pub fn channel(&mut self, shape: Vec<u32>, host_role: HostRole) -> u64 {
        self.channel_of(shape, EtaDtype::F32, host_role)
    }

    pub fn channel_of(&mut self, shape: Vec<u32>, dtype: EtaDtype, host_role: HostRole) -> u64 {
        let id = self.next_channel;
        self.next_channel += 1;
        self.engine
            .register_channel(&ChannelRegistration {
                id,
                shape,
                dtype: ChanDType::Concrete(dtype),
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

    /// A DiT lane's instance: latent, timestep and positions channels, each
    /// fire taking one cell off each.
    pub fn lane(&mut self, rows: u32) -> LaneHandles {
        let program = match self.programs.get(&rows) {
            Some(id) => *id,
            None => {
                let shapes = [
                    Shape::matrix(rows, WIDTH),
                    Shape::matrix(1, 1),
                    Shape::matrix(rows, 2),
                ];
                let id = self.register(carrier(&shapes, 3));
                self.programs.insert(rows, id);
                id
            }
        };
        let latent = self.channel(vec![rows, WIDTH], HostRole::Writer);
        let timestep = self.channel(vec![1, 1], HostRole::Writer);
        let positions = self.channel(vec![rows, 2], HostRole::Writer);
        let instance = self.bind(program, vec![latent, timestep, positions], rows);
        LaneHandles {
            instance,
            latent,
            timestep,
            positions,
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Submits one step and settles it: each lane's readout.
    pub fn fire(
        &mut self,
        lanes: Vec<Lane>,
        attachments: Vec<Attachment>,
        voxels: Vec<StepVoxels>,
    ) -> Vec<engine::fire::LaneReadout> {
        let mut ticket = self
            .engine
            .submit(&FrameSubmission::of(Step {
                lanes,
                attachments,
                media: Vec::new(),
                voxels,
            }))
            .expect("the frame fires");
        self.engine
            .settle_frame(&mut ticket)
            .expect("the frame settles");
        ticket.steps[0].readouts.clone()
    }
}

/// A double-block lane: its stream's latent port, the timestep and the
/// positions, in attention group `group`.
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

pub fn attach(lane: u32, instance: u64) -> Attachment {
    Attachment {
        lane,
        instance,
        at: Boundary::Epilogue,
    }
}

pub fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        worst = worst.max((g - w).abs());
        assert!(
            (g - w).abs() <= 4.0e-2 * w.abs().max(1.0),
            "{what} at {i}: device {g} against host {w}"
        );
    }
    eprintln!("{what}: max |err| {worst:.4}");
}
