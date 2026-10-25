use poem::fact;
use poem::{
    Dtype, ForwardHybrid, HybridSpec, Input, Platform, Predicate, RaggedMask, Request, Selection,
    Trace, Value, Weight, ops, seam, trace_hybrid,
};
use poem_ir::{Attn, Def, Gather, GeomKind, Guard, Operation, RuntimeInput};

const AUDIO_WIDTH: u32 = 32;
const VIDEO_WIDTH: u32 = 48;
const HEAD_DIM: u32 = 16;
const HEADS: u64 = 4;

struct CrossAttention;

impl ForwardHybrid for CrossAttention {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) -> Value {
        inputs
            .recorder()
            .declare_switch("stream", &["text", "image", "video", "audio"]);
        let ([audio, video], _rest) = inputs.partition([
            fact::switch_is("stream", "audio"),
            fact::switch_is("stream", "video"),
        ]);
        let wq = Weight::sym(
            "audio.q",
            [HEADS * u64::from(HEAD_DIM), u64::from(AUDIO_WIDTH)],
            Dtype::Bf16,
        );
        let wk = Weight::sym(
            "video.k",
            [HEADS * u64::from(HEAD_DIM), u64::from(VIDEO_WIDTH)],
            Dtype::Bf16,
        );
        let wv = Weight::sym(
            "video.v",
            [HEADS * u64::from(HEAD_DIM), u64::from(VIDEO_WIDTH)],
            Dtype::Bf16,
        );
        let wo = Weight::sym(
            "audio.o",
            [u64::from(AUDIO_WIDTH), HEADS * u64::from(HEAD_DIM)],
            Dtype::Bf16,
        );

        let xa = audio.input(
            "audio",
            Dtype::Bf16,
            poem_ir::InputAxis::Row,
            AUDIO_WIDTH,
            None,
        );
        let xv = video.input(
            "video",
            Dtype::Bf16,
            poem_ir::InputAxis::Row,
            VIDEO_WIDTH,
            None,
        );
        let q = ops::gather::pack_rows(&ops::linear::matmul(&xa, &wq), &audio.row_permutation());
        let k = ops::gather::pack_rows(&ops::linear::matmul(&xv, &wk), &video.row_permutation());
        let v = ops::gather::pack_rows(&ops::linear::matmul(&xv, &wv), &video.row_permutation());
        let o = ops::attn::ragged(
            &q,
            &k,
            &v,
            &audio.lane_indptr(),
            &video.lane_indptr(),
            HEAD_DIM,
            0.25,
            RaggedMask::None,
        );
        let o = ops::gather::unpack_rows(&o, &audio.row_permutation());
        let out = ops::linear::matmul(&o, &wo);
        seam::at(seam::VELOCITY, &[&out]);
        out
    }
}

/// The guard `p` stands for in `trace`.
fn cond_of(trace: &Trace, p: &Predicate) -> Guard {
    p.guard(&mut trace.facts.clone())
}

#[test]
fn a_ragged_attention_joins_two_arms_every_case() {
    let trace = trace_hybrid("cross", &CrossAttention, Platform::Cuda);
    queries_off_one_arm_and_keys_off_another_trace_under_the_or_of_both(&trace);
    every_other_op_still_refuses_two_arms();
    a_lanes_stream_is_its_fact_word(&trace);
}

fn queries_off_one_arm_and_keys_off_another_trace_under_the_or_of_both(trace: &Trace) {
    assert!(trace.caches.is_empty(), "a denoiser declares no kv space");

    let audio = cond_of(trace, &fact::switch_is("stream", "audio"));
    let video = Guard::and(
        !audio.clone(),
        cond_of(trace, &fact::switch_is("stream", "video")),
    );

    let (at, ragged) = trace
        .nodes
        .iter()
        .enumerate()
        .find(|(_, node)| matches!(node.op, Operation::Attn(Attn::Ragged { .. })))
        .expect("the text holds the one ragged attention it was written for");
    let Operation::Attn(Attn::Ragged {
        head_dim,
        kv_heads,
        q_indptr,
        kv_indptr,
        ..
    }) = &ragged.op
    else {
        unreachable!()
    };
    assert_eq!(*head_dim, HEAD_DIM);
    assert_eq!(*kv_heads, 4, "kv heads are read off k's width");

    let both = Guard::or(audio.clone(), video.clone());
    assert!(
        ragged.guard.equivalent(&both),
        "the ragged node is guarded by {:?}, not the join of audio and video",
        ragged.guard
    );
    assert!(
        !ragged.guard.equivalent(&audio) && !ragged.guard.equivalent(&video),
        "the join is wider than either arm"
    );

    let unpack = trace.nodes[at + 1..]
        .iter()
        .find(|node| matches!(node.op, Operation::Gather(Gather::UnpackRows { .. })))
        .expect("the answer is unpacked onto the audio rows");
    assert!(
        unpack.guard.equivalent(&audio),
        "the unpack is guarded by {:?}, not the queries' arm",
        unpack.guard
    );
    assert_eq!(
        unpack.guard, audio,
        "and spelled exactly as the audio arm spells itself, so the next op accepts it"
    );

    let selection = |id: poem_ir::ValueId| match &trace.values[id.0 as usize].def {
        Def::Input(RuntimeInput::Geometry {
            space: 0,
            kind: GeomKind::LaneIndptr { select },
        }) => *select,
        other => panic!("a CSR is a lane indptr in the token space, not {other:?}"),
    };
    assert_eq!(selection(*q_indptr), Selection::of(&audio).unwrap());
    assert_eq!(selection(*kv_indptr), Selection::of(&video).unwrap());
    assert_ne!(selection(*q_indptr), selection(*kv_indptr));
    let perms = trace
        .values
        .iter()
        .filter(|decl| matches!(decl.def, Def::Input(RuntimeInput::RowPermutation { .. })))
        .count();
    assert_eq!(
        perms, 2,
        "one permutation per side, deduplicated per selection"
    );
}

struct TwoArmsIntoOneAdd;

impl ForwardHybrid for TwoArmsIntoOneAdd {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) -> Value {
        inputs
            .recorder()
            .declare_switch("stream", &["text", "image", "video", "audio"]);
        let (audio, video) = (
            inputs.on(fact::switch_is("stream", "audio")),
            inputs.on(!fact::switch_is("stream", "audio")),
        );
        let w = Weight::sym("w", [8, 8], Dtype::Bf16);
        let a = ops::linear::matmul(
            &audio.input("latents", Dtype::Bf16, poem_ir::InputAxis::Row, 8, None),
            &w,
        );
        let b = ops::linear::matmul(
            &video.input("latents", Dtype::Bf16, poem_ir::InputAxis::Row, 8, None),
            &w,
        );
        ops::elemwise::add(&a, &b)
    }
}

fn every_other_op_still_refuses_two_arms() {
    let refused =
        std::panic::catch_unwind(|| trace_hybrid("mixed", &TwoArmsIntoOneAdd, Platform::Cuda));
    let message = match refused {
        Ok(_) => panic!("an add over two arms traced"),
        Err(payload) => payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
    };
    assert!(
        message.contains("elemwise.binary") && message.contains("different split arms"),
        "the refusal names the op and the rule: {message}"
    );
}

fn a_lanes_stream_is_its_fact_word(trace: &Trace) {
    let audio = cond_of(trace, &fact::switch_is("stream", "audio"));
    let video = cond_of(trace, &fact::switch_is("stream", "video"));
    let word = trace
        .facts
        .word(&Request::new(16, false).switched("stream", "video"));
    assert!(video.holds(word));
    assert!(!audio.holds(word));
    let text = trace.facts.word(&Request::new(16, false));
    assert!(
        !audio.holds(text) && !video.holds(text),
        "a lane that names no stream is on neither arm"
    );
}
