#![cfg(feature = "cuda")]

mod common_dit;

use common_dit::{Lcg, NAME, Rig, StreamFacts, WIDTH, Weights, attach, bf, frame, lane};
use engine::Engine;
use engine::fire::{LaneStream, ReadoutSeam};
use engine_cuda::Recording;
use eta_ir::container::{ChanDType, ChannelDecl, HostRole, StageProgram, TraceContainer};
use eta_ir::op::{IntrinsicId, Op};
use eta_ir::registry::Stage;
use eta_ir::types::{Dtype as EtaDtype, Shape};
use model_dsl::{
    Dtype, ForwardHybrid, HybridSpec, Input, ModulateForm, Platform, Stream, Trace, Value, Weight,
    ops, seam, trace_hybrid,
};

const VOCAB: u32 = 64;
const HEAD_DIM: u32 = 64;
const HEADS: u32 = 1;
const SM_SCALE: f32 = 0.125;
const FREQ: u32 = 16;
const KV_ROW: &str = "trunk.kv";
const TEXT_ROWS: u32 = 6;

struct TokensAndLatents;

impl ForwardHybrid for TokensAndLatents {
    type Facts = StreamFacts;

    fn caches(&self) -> HybridSpec {
        let mut spec = HybridSpec::new();
        let space = spec.kv_space(Dtype::Bf16);
        let plane = u64::from(HEADS) * u64::from(HEAD_DIM);
        spec.kv(space, KV_ROW, [plane, plane]);
        spec
    }

    fn forward(&self, inputs: Input<StreamFacts>) -> Value {
        let (txt, img) = inputs.split(&StreamFacts::on(Stream::Text));
        let w = |name: &str, out: u32, inner: u32| {
            Weight::sym(name, [u64::from(out), u64::from(inner)], Dtype::Bf16)
        };

        let table = Weight::sym("embed", [u64::from(VOCAB), u64::from(WIDTH)], Dtype::Bf16);
        let x = ops::layout::embed(&txt.tokens(), &table, VOCAB);
        let q = ops::linear::matmul(&x, &w("txt.q", HEAD_DIM, WIDTH));
        let k = ops::linear::matmul(&x, &w("txt.k", HEAD_DIM, WIDTH));
        let v = ops::linear::matmul(&x, &w("txt.v", HEAD_DIM, WIDTH));
        let pages = txt.kv(KV_ROW);
        ops::attn::kv_append(
            &k,
            &v,
            pages,
            &txt.write_page(KV_ROW),
            &txt.write_offset(KV_ROW),
        );
        let plan = ops::attn::plan_prefill(&txt, HEADS, HEADS, HEAD_DIM, None);
        let o = ops::attn::prefill(&q, &plan, pages, None, HEAD_DIM, HEADS, SM_SCALE);
        let h = ops::linear::matmul(&o, &w("txt.o", WIDTH, HEAD_DIM));
        seam::at(seam::HIDDEN, &[&h]);

        let x_img = img.latents(1, WIDTH, Dtype::Bf16);
        let t = img.lane_vector(0, 1);
        let emb = ops::elemwise::silu(&ops::elemwise::sinusoid(&t, FREQ, 10_000.0, true, 1.0));
        let m = ops::linear::matmul(&emb, &w("ada", 2 * WIDTH, FREQ));
        let hm = ops::elemwise::modulate(
            &ops::elemwise::layernorm_no_scale(&x_img, 1e-6),
            &m,
            Some(&img.request_of_token()),
            ModulateForm::ScaleShift,
        );
        let velocity = ops::linear::matmul(&hm, &w("den", WIDTH, WIDTH));
        seam::at(seam::VELOCITY, &[&velocity]);
        velocity
    }
}

fn plan() -> Trace {
    trace_hybrid(NAME, &TokensAndLatents, Platform::Cuda)
}

fn epilogue(rows: u32, width: u32) -> TraceContainer {
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
                    intr: IntrinsicId::Hidden,
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
fn the_token_reading_fires_alone() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let plan = plan();
    let weights = Weights::random(&plan, 41);
    let mut rig = Rig::load_recording(
        plan,
        &weights,
        64,
        vec![16, 64],
        Recording::Bodies { mem_megabytes: 0 },
    );

    let program = rig.register(epilogue(TEXT_ROWS, WIDTH), 7);
    let handles = rig.lane_of(program, TEXT_ROWS, WIDTH);
    let mut rng = Lcg::seeded(43);
    let cell: Vec<f32> = (0..TEXT_ROWS as usize * WIDTH as usize)
        .map(|_| bf(rng.unit()))
        .collect();
    rig.publish(handles.instance, 0, &cell);

    let mut text = lane(0, &handles, LaneStream::Text, 0);
    text.tokens = (0..TEXT_ROWS).map(|r| r % VOCAB).collect();
    text.ports.clear();

    let mut ticket = rig
        .engine
        .submit(&frame(vec![text], vec![attach(0, &handles)]))
        .expect("a plan whose token reading carries kv fires that reading alone");
    rig.engine
        .settle_frame(&mut ticket)
        .expect("the frame settles");

    let readout = &ticket.steps[0].readouts[0];
    assert_eq!(readout.seam, ReadoutSeam::Hidden, "the token arm's seam");
    assert_eq!(readout.rows, TEXT_ROWS, "one row per token");
    assert_eq!(readout.width, WIDTH, "the trunk's residual width");

    let out = rig.take(handles.instance, 3);
    assert_eq!(out.len(), TEXT_ROWS as usize * WIDTH as usize);
    assert!(
        out.iter().all(|v| v.is_finite()),
        "the hidden rows the epilogue read back are finite"
    );
    assert!(
        out.iter().any(|v| *v != 0.0),
        "the hidden rows are not a field of zeros, which is what an \
         unwritten arena column would read as"
    );
}
