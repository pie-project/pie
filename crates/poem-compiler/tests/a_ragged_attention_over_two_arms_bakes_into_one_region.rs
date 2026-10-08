mod common;

use poem::fact;
use poem::{
    Dtype, ForwardHybrid, HybridSpec, Input, RaggedMask, Request, Stream, Value, Weight, ops, seam,
    trace_hybrid,
};
use poem_compiler::Placement;
use poem_ir::{Attention, Def, Layout, Linear, Operation, RuntimeInput};

const AUDIO: u64 = 32;
const VIDEO: u64 = 48;
const HEAD_DIM: u32 = 16;
const HEADS: u64 = 4;

struct CrossAttention;

impl ForwardHybrid for CrossAttention {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }
    fn forward(&self, inputs: Input) -> Value {
        let ([audio, video], _rest) =
            inputs.partition([fact::stream(Stream::Audio), fact::stream(Stream::Video)]);
        let inner = HEADS * u64::from(HEAD_DIM);
        let wq = Weight::sym("audio.q", [inner, AUDIO], Dtype::Bf16);
        let wk = Weight::sym("video.k", [inner, VIDEO], Dtype::Bf16);
        let wv = Weight::sym("video.v", [inner, VIDEO], Dtype::Bf16);
        let wo = Weight::sym("audio.o", [AUDIO, inner], Dtype::Bf16);
        let xa = audio.latents(0, AUDIO as u32, Dtype::Bf16);
        let xv = video.latents(1, VIDEO as u32, Dtype::Bf16);
        let q = ops::layout::pack_rows(&ops::linear::matmul(&xa, &wq), &audio.row_permutation());
        let k = ops::layout::pack_rows(&ops::linear::matmul(&xv, &wk), &video.row_permutation());
        let v = ops::layout::pack_rows(&ops::linear::matmul(&xv, &wv), &video.row_permutation());
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
        let o = ops::layout::unpack_rows(&o, &audio.row_permutation());
        let out = ops::linear::matmul(&o, &wo);
        seam::at(seam::VELOCITY, &[&out]);
        out
    }
}

#[test]
fn the_join_is_the_attentions_alone_and_the_velocity_lives_to_the_end() {
    for platform in common::PLATFORMS {
        let trace = trace_hybrid("cross", &CrossAttention, platform);
        assert!(trace.caches.is_empty());
        assert!(
            !trace.seams.iter().any(|seam| seam.seam == seam::OUT.name),
            "no logits"
        );

        let compiled = common::bake(&trace).unwrap_or_else(|e| panic!("{platform:?}: {e}"));

        let audio_class = compiled
            .classes
            .class_of(
                trace
                    .facts
                    .word(&Request::new(1, false).on_stream(Stream::Audio))
                    & compiled.classes.mask,
            )
            .expect("an audio lane has a class");
        let video_class = compiled
            .classes
            .class_of(
                trace
                    .facts
                    .word(&Request::new(1, false).on_stream(Stream::Video))
                    & compiled.classes.mask,
            )
            .expect("a video lane has a class");
        assert_ne!(audio_class, video_class);

        let region_of = |at: usize| {
            compiled
                .template()
                .iter()
                .find(|region| region.nodes.contains(&(at as u32)))
                .unwrap_or_else(|| panic!("node {at} is in a region"))
        };

        let (ragged_at, _) = trace
            .nodes
            .iter()
            .enumerate()
            .find(|(_, node)| matches!(node.op, Operation::Attention(Attention::Ragged { .. })))
            .expect("one ragged attention");
        let ragged = region_of(ragged_at);
        assert!(
            ragged.mask.contains(audio_class) && ragged.mask.contains(video_class),
            "{platform:?}: the ragged region's mask {:?} holds both classes",
            ragged.mask.iter().collect::<Vec<_>>()
        );
        for (at, node) in trace.nodes.iter().enumerate() {
            if at == ragged_at {
                continue;
            }
            let region = region_of(at);
            assert!(
                !(region.mask.contains(audio_class) && region.mask.contains(video_class)),
                "{platform:?}: node {at} ({}) also runs over both classes",
                poem_ir::Operands::name(&node.op)
            );
        }

        let (audio_proj, _) = trace
            .nodes
            .iter()
            .enumerate()
            .find(|(_, node)| matches!(node.op, Operation::Linear(Linear::Matmul { .. })))
            .expect("the audio projection is the first matmul");
        let region = region_of(audio_proj);
        assert!(region.mask.contains(audio_class) && !region.mask.contains(video_class));
        let (video_proj, _) = trace
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| matches!(node.op, Operation::Linear(Linear::Matmul { .. })))
            .nth(1)
            .expect("the key projection is the second matmul");
        let region = region_of(video_proj);
        assert!(
            region.mask.contains(video_class) && !region.mask.contains(audio_class),
            "{platform:?}: the key projection runs over the video window"
        );
        assert!(
            compiled.classes.dead.is_empty(),
            "{platform:?}: every node is demanded somewhere: {:?}",
            compiled.classes.dead
        );

        let velocity = trace
            .seams
            .iter()
            .find(|seam| seam.seam == seam::VELOCITY.name)
            .and_then(|seam| seam.values.first().copied())
            .expect("the velocity seam names its value");
        let at = velocity.0 as usize;
        assert!(
            matches!(compiled.arena.placements[at], Placement::Arena { .. }),
            "{platform:?}: the velocity is an arena rectangle, not {:?}",
            compiled.arena.placements[at]
        );
        let span = compiled.arena.spans[at].expect("a rectangle has a life");
        assert_eq!(
            span.last as usize,
            trace.nodes.len(),
            "{platform:?}: the velocity lives past the last node, for the reader after the graph"
        );

        for (id, decl) in trace.values.iter().enumerate() {
            if let Def::Input(input) = &decl.def {
                assert_eq!(compiled.arena.placements[id], Placement::Runtime(*input));
            }
        }
        assert!(
            trace
                .values
                .iter()
                .any(|decl| matches!(decl.def, Def::Input(RuntimeInput::RowPermutation { .. })))
        );
        assert!(
            trace
                .nodes
                .iter()
                .any(|node| matches!(node.op, Operation::Layout(Layout::PackRows { .. })))
        );
    }
}
