#![cfg(feature = "cuda")]

mod common_dit;

use common_dit::{
    HEAD_DIM, Lcg, NAME, Rig, SM_SCALE, StreamFacts, THETA, WIDTH, Weights, assert_close, attach,
    bf, condition, frame, lane, matmul_bf16, matmul_f32, rope, silu, sinusoid,
};
use engine::Engine;
use engine::fire::{LaneStream, ReadoutSeam};
use poem_dsl::{
    Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Platform, RaggedMask, RopeForm, Stream,
    Trace, Value, Weight, ops, seam, trace_hybrid,
};

const FREQ: u32 = 16;

struct ReferenceBlock;

impl ForwardHybrid for ReferenceBlock {
    type Facts = StreamFacts;
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input<StreamFacts>) -> Value {
        let (txt, img) = inputs.split(&StreamFacts::on(Stream::Text));
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
        let tags = inputs.reference_tags();
        let o = ops::attn::ragged(
            &ops::layout::pack_rows(&q, &perm),
            &ops::layout::pack_rows(&k, &perm),
            &ops::layout::pack_rows(&v, &perm),
            &indptr,
            &indptr,
            HEAD_DIM,
            SM_SCALE,
            RaggedMask::ReferenceSelfOnly {
                q_tags: tags.id(),
                kv_tags: tags.id(),
            },
        );
        let o = ops::layout::unpack_rows(&o, &perm);
        let (o_txt, o_img) = o.split(&StreamFacts::on(Stream::Text));
        let y_txt = ops::linear::matmul(&o_txt, &w("txt.o", WIDTH, HEAD_DIM));
        let y_img = ops::linear::matmul(&o_img, &w("img.o", WIDTH, HEAD_DIM));
        let r_txt = ops::elemwise::residual_add(&x_txt, &y_txt);
        let r_img = ops::elemwise::residual_add(&x_img, &y_img);
        let out = Value::merge(vec![r_txt, r_img]);
        seam::at(seam::VELOCITY, &[&out]);
        out
    }
}

fn plan() -> Trace {
    trace_hybrid(NAME, &ReferenceBlock, Platform::Cuda)
}

struct HostLane {
    rows: Vec<f32>,
    count: usize,
    positions: Vec<[f32; 2]>,
    reference: bool,
    text: bool,
}

fn reference(weights: &Weights, timestep: f32, lanes: &[HostLane]) -> Vec<Vec<f32>> {
    let w = WIDTH as usize;
    let hd = HEAD_DIM as usize;
    let emb = silu(&sinusoid(timestep));
    let m = matmul_f32(&emb, 1, FREQ as usize, weights.get("ada"), 2 * w);
    let mut q_all = Vec::new();
    let mut k_all = Vec::new();
    let mut v_all = Vec::new();
    let mut tags = Vec::new();
    let mut spans = Vec::new();
    for (at, lane) in lanes.iter().enumerate() {
        let prefix = if lane.text { "txt" } else { "img" };
        let h = condition(&lane.rows, lane.count, &m);
        let mut q = matmul_bf16(&h, lane.count, w, weights.get(&format!("{prefix}.q")), hd);
        let mut k = matmul_bf16(&h, lane.count, w, weights.get(&format!("{prefix}.k")), hd);
        let v = matmul_bf16(&h, lane.count, w, weights.get(&format!("{prefix}.v")), hd);
        rope(&mut q, lane.count, &lane.positions);
        rope(&mut k, lane.count, &lane.positions);
        let start = tags.len();
        q_all.extend(q);
        k_all.extend(k);
        v_all.extend(v);
        tags.extend(std::iter::repeat_n(
            if lane.reference { at as i32 } else { -1 },
            lane.count,
        ));
        spans.push(start..tags.len());
    }
    let rows = tags.len();
    let mut o = vec![0f32; rows * hd];
    for i in 0..rows {
        let keys: Vec<usize> = (0..rows)
            .filter(|&j| tags[i] < 0 || tags[j] == tags[i])
            .collect();
        let scores: Vec<f32> = keys
            .iter()
            .map(|&j| {
                (0..hd)
                    .map(|d| q_all[i * hd + d] * k_all[j * hd + d])
                    .sum::<f32>()
                    * SM_SCALE
            })
            .collect();
        let peak = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights_: Vec<f32> = scores.iter().map(|s| (s - peak).exp()).collect();
        let mass: f32 = weights_.iter().sum();
        for d in 0..hd {
            let mut acc = 0f32;
            for (&j, p) in keys.iter().zip(&weights_) {
                acc += bf(p / mass) * v_all[j * hd + d];
            }
            o[i * hd + d] = bf(acc);
        }
    }
    lanes
        .iter()
        .zip(&spans)
        .map(|(lane, span)| {
            let prefix = if lane.text { "txt" } else { "img" };
            let y = matmul_bf16(
                &o[span.start * hd..span.end * hd],
                lane.count,
                hd,
                weights.get(&format!("{prefix}.o")),
                w,
            );
            y.iter().zip(&lane.rows).map(|(a, b)| bf(a + b)).collect()
        })
        .collect()
}

#[test]
fn each_reference_lane_attends_itself_and_the_rest_see_everything() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let plan = plan();
    let weights = Weights::random(&plan, 43);
    let mut rig = Rig::load_plan(plan, &weights, 64, vec![16, 32, 64]);

    let w = WIDTH as usize;
    let mut rng = Lcg::seeded(47);
    let timestep = 0.6;
    let counts = [3usize, 5, 4, 6];
    let kinds = [(true, false), (false, false), (false, true), (false, true)];
    let mut host: Vec<HostLane> = Vec::new();
    let mut at_row = 0usize;
    for (&count, &(text, reference)) in counts.iter().zip(&kinds) {
        host.push(HostLane {
            rows: (0..count * w).map(|_| bf(rng.unit())).collect(),
            count,
            positions: (0..count)
                .map(|r| [(at_row + r) as f32, if reference { 10.0 } else { 0.0 }])
                .collect(),
            reference,
            text,
        });
        at_row += count;
    }
    let want = reference(&weights, timestep, &host);

    let order = [2usize, 0, 3, 1];
    let mut lanes = Vec::new();
    let mut attachments = Vec::new();
    let mut readback: Vec<usize> = Vec::new();
    for (slot, &at) in order.iter().enumerate() {
        let lane_host = &host[at];
        let handles = rig.lane(lane_host.count as u32);
        rig.publish(handles.instance, 0, &lane_host.rows);
        rig.publish(handles.instance, 1, &[timestep]);
        rig.publish(
            handles.instance,
            2,
            &lane_host
                .positions
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<f32>>(),
        );
        let stream = match (lane_host.text, lane_host.reference) {
            (true, _) => LaneStream::Text,
            (false, true) => LaneStream::Reference,
            (false, false) => LaneStream::Image,
        };
        let mut submitted = lane(slot as u32, &handles, stream, 0);
        submitted.word = common_dit::classify(
            &poem_dsl::Request::new(lane_host.count as u32, false).on_stream(match stream {
                LaneStream::Text => Stream::Text,
                LaneStream::Reference => Stream::Reference,
                _ => Stream::Image,
            }),
        );
        attachments.push(attach(lanes.len() as u32, &handles));
        lanes.push(submitted);
        readback.push(at);
    }
    let mut ticket = rig
        .engine
        .submit(&frame(lanes, attachments))
        .expect("the frame fires");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the frame settles");
    let readouts = &ticket.steps[0].readouts;
    assert_eq!(readouts.len(), 4);
    for (submitted, &at) in readouts.iter().zip(&readback) {
        assert_eq!(submitted.seam, ReadoutSeam::Velocity);
        assert_close(
            &submitted.values,
            &want[at],
            &format!(
                "lane {at} ({}) velocity",
                match (host[at].text, host[at].reference) {
                    (true, _) => "text",
                    (false, true) => "reference",
                    _ => "image",
                }
            ),
        );
    }
}
