#![allow(dead_code)]

use poem::fact;
use poem::{Dtype, ForwardHybrid, HybridSpec, Input, RaggedMask, Stream, Value, Weight, ops, seam};

pub const AUDIO_WIDTH: u32 = 32;
pub const VIDEO_WIDTH: u32 = 48;
pub const HEAD_DIM: u32 = 16;
pub const HEADS: u64 = 4;

pub struct CrossAttention;

impl ForwardHybrid for CrossAttention {
    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input) -> Value {
        let ([audio, video], _rest) =
            inputs.partition([fact::stream(Stream::Audio), fact::stream(Stream::Video)]);
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

        let xa = audio.latents(0, AUDIO_WIDTH, Dtype::Bf16);
        let xv = video.latents(1, VIDEO_WIDTH, Dtype::Bf16);
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
