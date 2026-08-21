use crate::deployment::{Deployment, Refusal};
use crate::manifest::{Manifest, TensorSpec};

use super::spec::CsmFacts;

pub const NO_DEPLOYMENT: &str = "csm is three stacks — a 16-layer backbone, a 4-layer depth \
     decoder that emits 32 residual codebooks per frame, and a Mimi \
     codec — and a Deployment describes one; this build has no speech \
     decode loop to drive them and no `Deployment` field that could \
     carry the depth decoder or the codec";

pub const NO_TRACE: &str = "csm has no traced forward text: there is no `csm/forward` module, \
     so neither the backbone's frame pass nor the depth decoder's \
     per-codebook pass has ever been written in the tracing eDSL";

#[must_use]
pub fn manifest(f: &CsmFacts) -> Manifest {
    let b = &f.backbone;
    let d = &f.depth;
    let c = &f.codec;
    let (hidden, inter) = (u64::from(b.hidden), u64::from(b.intermediate));
    let (dh, dinter) = (u64::from(d.hidden), u64::from(d.intermediate));

    Manifest::new(b.layers)

        .with(TensorSpec::required(
            "embed_text_tokens",
            [u64::from(b.text_vocab), hidden],
        ))
        .with(TensorSpec::required("backbone_model.norm", [hidden]))

        .either(
            !f.tied_embeddings,
            "lm_head",
            [u64::from(b.audio_vocab), hidden],
        )
        .with(TensorSpec::required(
            "backbone_model.layer.{}.self_attn.q_proj",
            [u64::from(b.q_width()), hidden],
        ))
        .with(TensorSpec::required(
            "backbone_model.layer.{}.self_attn.k_proj",
            [u64::from(b.kv_width()), hidden],
        ))
        .with(TensorSpec::required(
            "backbone_model.layer.{}.self_attn.v_proj",
            [u64::from(b.kv_width()), hidden],
        ))
        .with(TensorSpec::required(
            "backbone_model.layer.{}.self_attn.o_proj",
            [hidden, u64::from(b.q_width())],
        ))
        .with(TensorSpec::required(
            "backbone_model.layer.{}.mlp.gate_proj",
            [inter, hidden],
        ))
        .with(TensorSpec::required(
            "backbone_model.layer.{}.mlp.up_proj",
            [inter, hidden],
        ))
        .with(TensorSpec::required(
            "backbone_model.layer.{}.mlp.down_proj",
            [hidden, inter],
        ))
        .with(TensorSpec::required(
            "backbone_model.layer.{}.input_layernorm",
            [hidden],
        ))
        .with(TensorSpec::required(
            "backbone_model.layer.{}.post_attention_layernorm",
            [hidden],
        ))

        .either(
            !f.tied_codebooks,
            "backbone_model.embed_tokens.embed_audio_tokens",
            [u64::from(d.code_table_rows()), hidden],
        )

        .with(TensorSpec::required(
            "depth_decoder.model.embed_tokens",
            [u64::from(d.code_table_rows()), u64::from(d.backbone_hidden)],
        ))

        .with(TensorSpec::required(
            "depth_decoder.model.inputs_embeds_projector",
            [dh, u64::from(d.backbone_hidden)],
        ))
        .with(TensorSpec::required("depth_decoder.model.norm", [dh]))

        .with(TensorSpec::required(
            "depth_decoder.codebooks_head",
            [u64::from(d.head_slices()), dh, u64::from(d.vocab)],
        ))
        .with(TensorSpec::required(
            "depth_decoder.model.layer.{}.self_attn.q_proj",
            [u64::from(d.q_width()), dh],
        ))
        .with(TensorSpec::required(
            "depth_decoder.model.layer.{}.self_attn.k_proj",
            [u64::from(d.kv_width()), dh],
        ))
        .with(TensorSpec::required(
            "depth_decoder.model.layer.{}.self_attn.v_proj",
            [u64::from(d.kv_width()), dh],
        ))
        .with(TensorSpec::required(
            "depth_decoder.model.layer.{}.self_attn.o_proj",
            [dh, u64::from(d.q_width())],
        ))
        .with(TensorSpec::required(
            "depth_decoder.model.layer.{}.mlp.gate_proj",
            [dinter, dh],
        ))
        .with(TensorSpec::required(
            "depth_decoder.model.layer.{}.mlp.up_proj",
            [dinter, dh],
        ))
        .with(TensorSpec::required(
            "depth_decoder.model.layer.{}.mlp.down_proj",
            [dh, dinter],
        ))
        .with(TensorSpec::required(
            "depth_decoder.model.layer.{}.input_layernorm",
            [dh],
        ))
        .with(TensorSpec::required(
            "depth_decoder.model.layer.{}.post_attention_layernorm",
            [dh],
        ))

        .with(TensorSpec::required(
            "codec_model.quantizer.semantic_residual_vector_quantizer.layer.{}.codebook.embed_sum",
            [u64::from(c.codebook_size), u64::from(c.codebook_dim)],
        ))
        .with(TensorSpec::required(
            "codec_model.quantizer.acoustic_residual_vector_quantizer.layer.{}.codebook.embed_sum",
            [u64::from(c.codebook_size), u64::from(c.codebook_dim)],
        ))

        .with(TensorSpec::required(
            "codec_model.quantizer.semantic_residual_vector_quantizer.input_proj",
            [u64::from(c.codebook_dim), u64::from(c.hidden), 1],
        ))
        .with(TensorSpec::required(
            "codec_model.quantizer.acoustic_residual_vector_quantizer.output_proj",
            [u64::from(c.hidden), u64::from(c.codebook_dim), 1],
        ))

        .with(TensorSpec::required(
            "codec_model.encoder_transformer.layer.{}.self_attn.q_proj",
            [u64::from(c.hidden), u64::from(c.hidden)],
        ))
        .with(TensorSpec::required(
            "codec_model.decoder_transformer.layer.{}.self_attn.q_proj",
            [u64::from(c.hidden), u64::from(c.hidden)],
        ))

        .with(TensorSpec::required(
            "codec_model.decoder_transformer.layer.{}.self_attn_layer_scale.scale",
            [u64::from(c.hidden)],
        ))

        .with(TensorSpec::present("codec_model.encoder.layer.{}.conv"))
        .with(TensorSpec::present("codec_model.decoder.layer.{}.conv"))
        .with(TensorSpec::present("codec_model.upsample.conv"))
        .with(TensorSpec::present("codec_model.downsample.conv"))
}

pub fn deployment(f: &CsmFacts) -> Result<Deployment, Refusal> {
    let _ = f;
    Err(Refusal::Unsupported(NO_DEPLOYMENT))
}

pub fn trace(f: &CsmFacts) -> Result<model_ir::trace::ForwardPlan, Refusal> {
    let _ = f;
    Err(Refusal::Unsupported(NO_TRACE))
}
