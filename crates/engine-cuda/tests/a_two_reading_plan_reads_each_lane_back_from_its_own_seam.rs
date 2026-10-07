#![cfg(feature = "cuda")]

mod common_dit;

use common_dit::{
    Lcg, NAME, Rig, StreamFacts, WIDTH, Weights, assert_close, attach, bf, frame, lane,
};
use engine::Engine;
use engine::fire::{LaneStream, PortKind, ReadoutSeam};
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype as EtaDtype, Shape};
use poem_dsl::{
    Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Platform, Stream, Trace, Value, Weight,
    ops, seam, trace_hybrid,
};

const FREQ: u32 = 16;
const HIDDEN: u32 = 48;
const T_SCALE: f32 = 1000.0;
const EMB_SCALE: f32 = 0.75;

struct TwoReadings;

impl ForwardHybrid for TwoReadings {
    type Facts = StreamFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<StreamFacts>) -> Value {
        let (txt, img) = inputs.split(&StreamFacts::on(Stream::Text));
        let x_txt = txt.latents(0, WIDTH, Dtype::Bf16);
        let enc = Weight::sym("enc", [u64::from(HIDDEN), u64::from(WIDTH)], Dtype::Bf16);
        let h = ops::linear::matmul(&x_txt, &enc);
        seam::at(seam::HIDDEN, &[&h]);
        let x_img = img.latents(1, WIDTH, Dtype::Bf16);
        let t = img.lane_vector(0, 1);
        let emb = ops::elemwise::silu(&ops::elemwise::sinusoid(&t, FREQ, 10_000.0, true, T_SCALE));
        let emb = ops::elemwise::mul_scalar(EMB_SCALE, &emb);
        let ada = Weight::sym("ada", [2 * u64::from(WIDTH), u64::from(FREQ)], Dtype::Bf16);
        let m = ops::linear::matmul(&emb, &ada);
        let lanes = img.request_of_token();
        let hm = ops::elemwise::modulate(
            &ops::elemwise::layernorm_no_scale(&x_img, 1e-6),
            &m,
            Some(&lanes),
            ModulateForm::ScaleShift,
        );
        let den = Weight::sym("den", [u64::from(WIDTH), u64::from(WIDTH)], Dtype::Bf16);
        let v = ops::linear::matmul(&hm, &den);
        seam::at(seam::VELOCITY, &[&v]);
        v
    }
}

fn plan() -> Trace {
    trace_hybrid(NAME, &TwoReadings, Platform::Cuda)
}

fn hidden_reference(weights: &Weights, x: &[f32], rows: usize) -> Vec<f32> {
    let (w, hd) = (WIDTH as usize, HIDDEN as usize);
    let enc = weights.get("enc");
    let mut out = vec![0f32; rows * hd];
    for r in 0..rows {
        for c in 0..hd {
            let mut acc = 0f32;
            for i in 0..w {
                acc = x[r * w + i].mul_add(enc[c * w + i], acc);
            }
            out[r * hd + c] = bf(acc);
        }
    }
    out
}

fn velocity_reference(weights: &Weights, x: &[f32], rows: usize, timestep: f32) -> Vec<f32> {
    let w = WIDTH as usize;
    let half = (FREQ / 2) as usize;
    let t = timestep * T_SCALE;
    let angles: Vec<f32> = (0..half)
        .map(|i| t * (-10_000f32.ln() * i as f32 / half as f32).exp())
        .collect();
    let mut emb: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
    emb.extend(angles.iter().map(|a| a.sin()));
    let emb: Vec<f32> = emb
        .iter()
        .map(|v| v / (1.0 + (-v).exp()) * EMB_SCALE)
        .collect();
    let ada = weights.get("ada");
    let m: Vec<f32> = (0..2 * w)
        .map(|c| {
            let mut acc = 0f32;
            for i in 0..FREQ as usize {
                acc = emb[i].mul_add(ada[c * FREQ as usize + i], acc);
            }
            acc
        })
        .collect();
    let den = weights.get("den");
    let mut out = vec![0f32; rows * w];
    for r in 0..rows {
        let row = &x[r * w..(r + 1) * w];
        let mean = row.iter().sum::<f32>() / w as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / w as f32;
        let inv = 1.0 / (var + 1e-6).sqrt();
        let h: Vec<f32> = (0..w)
            .map(|c| bf(((row[c] - mean) * inv).mul_add(1.0 + m[c], m[w + c])))
            .collect();
        for c in 0..w {
            let mut acc = 0f32;
            for i in 0..w {
                acc = h[i].mul_add(den[c * w + i], acc);
            }
            out[r * w + c] = bf(acc);
        }
    }
    out
}

fn epilogue(rows: u32, intrinsic: IntrinsicId, width: u32) -> TraceContainer {
    let decl = |shape: Shape, host_role: HostRole| ChannelDecl {
        shape,
        dtype: ChanDType::Concrete(EtaDtype::F32),
        capacity: 2,
        host_role,
        seeded: false,
    };
    TraceContainer {
        names: Vec::new(),
        channels: vec![
            decl(Shape::matrix(rows, WIDTH), HostRole::Writer),
            decl(Shape::matrix(1, 1), HostRole::Writer),
            decl(Shape::matrix(rows, 2), HostRole::Writer),
            decl(Shape::matrix(rows, width), HostRole::Reader),
        ],
        ports: Vec::new(),
        stages: vec![StageProgram {
            stage: Stage::Epilogue,
            ops: vec![
                Op::ChanTake(0),
                Op::IntrinsicVal {
                    intr: intrinsic,
                    shape: Shape::matrix(rows, width),
                    dtype: EtaDtype::F32,
                },
                Op::ChanPut { chan: 3, value: 1 },
            ],
        }],
        externs: Vec::new(),
    }
}

#[test]
fn each_lane_reads_back_its_own_arms_seam() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let plan = plan();
    let weights = Weights::random(&plan, 37);
    let mut rig = Rig::load_plan(plan, &weights, 64, vec![16, 64]);
    assert!(rig.profile().has_velocity, "the velocity seam is plan-wide");
    assert_eq!(rig.profile().velocity_width, WIDTH);

    let w = WIDTH as usize;
    let mut rng = Lcg::seeded(39);
    let (text_rows, image_rows) = (4usize, 5usize);
    let text: Vec<f32> = (0..text_rows * w).map(|_| bf(rng.unit())).collect();
    let image: Vec<f32> = (0..image_rows * w).map(|_| bf(rng.unit())).collect();
    let timestep = 0.0005;
    let want_hidden = hidden_reference(&weights, &text, text_rows);
    let want_velocity = velocity_reference(&weights, &image, image_rows, timestep);

    let text_program = rig.register(epilogue(text_rows as u32, IntrinsicId::Hidden, HIDDEN), 1);
    let image_program = rig.register(epilogue(image_rows as u32, IntrinsicId::Velocity, WIDTH), 2);
    let t = rig.lane_of(text_program, text_rows as u32, HIDDEN);
    let i = rig.lane_of(image_program, image_rows as u32, WIDTH);
    rig.publish(t.instance, 0, &text);
    rig.publish(t.instance, 1, &[timestep]);
    rig.publish(i.instance, 0, &image);
    rig.publish(i.instance, 1, &[timestep]);

    let mut text_lane = lane(0, &t, LaneStream::Text, 0);
    let mut image_lane = lane(1, &i, LaneStream::Image, 1);
    for lane in [&mut text_lane, &mut image_lane] {
        lane.ports.retain(|f| f.kind != PortKind::AxisPositions);
    }
    text_lane.ports.retain(|f| f.kind != PortKind::LaneVector);
    let mut ticket = rig
        .engine
        .submit(&frame(
            vec![text_lane, image_lane],
            vec![attach(0, &t), attach(1, &i)],
        ))
        .expect("the frame fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the frame settles");
    let readouts = &ticket.steps[0].readouts;
    assert_eq!(readouts[0].seam, ReadoutSeam::Hidden);
    assert_eq!(readouts[0].width, HIDDEN);
    assert_close(
        &readouts[0].values,
        &want_hidden,
        "text lane: its arm's hidden",
    );
    assert_eq!(readouts[1].seam, ReadoutSeam::Velocity);
    assert_eq!(readouts[1].width, WIDTH);
    assert_close(
        &readouts[1].values,
        &want_velocity,
        "image lane: its arm's velocity",
    );

    assert_close(
        &rig.take(t.instance, 3),
        &want_hidden,
        "text lane: hidden() intrinsic",
    );
    assert_close(
        &rig.take(i.instance, 3),
        &want_velocity,
        "image lane: velocity() intrinsic",
    );
}
