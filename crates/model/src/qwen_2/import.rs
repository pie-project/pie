//! Qwen2 as llama.cpp spells it, in the vocabulary the catalog reads.
//!
//! GGUF is a third weight vocabulary beside the trace's `layer.7.qkv` and
//! HuggingFace's `model.layers.7.self_attn.q_proj.weight`. It calls that
//! tensor `blk.7.attn_q.weight`, and until something maps it an imported GGUF
//! is an artifact no catalog row can identify -- 291 tensors, every one of
//! them missing from every row.
//!
//! # Why this is a rename and nothing more
//!
//! llama.cpp's converter is a name map plus a per-architecture
//! `modify_tensors` hook, and Qwen2's hook does nothing to the DATA:
//! `Qwen2Model.modify_tensors` only prefixes `model.` for a checkpoint
//! published without it, then defers. Compare `LlamaModel`, which sets
//! `undo_permute = True` and rearranges every Q and K row so that llama.cpp's
//! rope reads them in its own order. That reordering is why llama is not the
//! family this landed with first: a permutation applied when it was not
//! wanted, or skipped when it was, is not an error. It is a model that loads,
//! serves, and answers slightly wrong, and pie has no cheap check that
//! notices.
//!
//! Qwen2 has no such hook, so the whole of the conversion is what the file
//! calls each tensor -- and a rename is a claim that can be checked by
//! reading it.
//!
//! # Why it lives here and not in `shared`
//!
//! `shared`'s bar is "**more than one generation binds it**, not 'it looks
//! reusable'". One generation does. The `blk.{bid}.` split is genuinely
//! architecture-independent in llama.cpp -- `TENSOR_NAMES` takes no
//! architecture argument -- and it will still be that on the day a second
//! family wants it, which is the day it moves.
//!
//! # The one thing this map cannot say
//!
//! `output.weight`. The 0.5B row is `tied_embeddings: true`, and pie spells a
//! tie as the head's ABSENCE, so an artifact carrying one is a different
//! model as far as the catalog is concerned. This file maps the name anyway
//! and lets identification say so, because the alternative is worse: GGUF
//! states no tie key at all, so dropping the tensor here would be this
//! module guessing at a fact the file does not contain -- and the tensor is
//! not redundant, since llama.cpp quantized this head to Q8_0 while the
//! embedding it would be tied to is Q4_0.

/// The HuggingFace name for a GGUF tensor, or `None` if this map has none.
///
/// `None` is a refusal and not a skip. A tensor silently left out is a model
/// that loads with a layer missing, and the caller turns this into an error
/// naming the tensor.
#[must_use]
pub fn hf_name(gguf: &str) -> Option<String> {
    // The suffix rides along untouched: `.weight` and `.bias` mean the same
    // thing in both vocabularies, and splitting them off leaves a stem that
    // is one table lookup.
    let (stem, suffix) = match gguf.rsplit_once('.') {
        Some((stem, tail @ ("weight" | "bias"))) => (stem, tail),
        _ => return None,
    };
    let hf = match split_layer(stem) {
        Some((layer, member)) => format!("model.layers.{layer}.{}", LAYER.get_member(member)?),
        // Not `model.`-prefixed, because HuggingFace does not prefix it: the
        // head sits outside the decoder that `model.` names.
        None if stem == "output" => "lm_head".to_string(),
        None => format!("model.{}", MODEL.get_member(stem)?),
    };
    Some(format!("{hf}.{suffix}"))
}

/// `blk.7.attn_q` as `(7, "attn_q")`.
///
/// The index is parsed rather than matched so that a malformed `blk.x.` falls
/// through to the unmapped case instead of being read as layer 0.
fn split_layer(stem: &str) -> Option<(u32, &str)> {
    let rest = stem.strip_prefix("blk.")?;
    let (index, member) = rest.split_once('.')?;
    Some((index.parse().ok()?, member))
}

/// A small ordered table, looked up by scan.
///
/// Fifteen rows total. A `HashMap` here would cost a build and an allocation
/// per import to beat a scan over twelve strings, and the table reads as a
/// table this way -- which is the point of stating it rather than sharing it.
struct Table(&'static [(&'static str, &'static str)]);

impl Table {
    fn get_member(&self, member: &str) -> Option<&'static str> {
        self.0
            .iter()
            .find(|(gguf, _)| *gguf == member)
            .map(|(_, hf)| *hf)
    }
}

/// The per-layer members, all twelve of them.
///
/// `attn_norm` and `ffn_norm` are the pair worth reading twice: llama.cpp
/// names them for what they precede, HuggingFace for where they sit in the
/// block, so `ffn_norm` is `post_attention_layernorm` and NOT anything with
/// "ffn" in it. The two vocabularies disagree about which end of the residual
/// a norm belongs to, and this row is that disagreement.
const LAYER: Table = Table(&[
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("attn_norm", "input_layernorm"),
    ("ffn_norm", "post_attention_layernorm"),
    ("ffn_gate", "mlp.gate_proj"),
    ("ffn_up", "mlp.up_proj"),
    ("ffn_down", "mlp.down_proj"),
]);

/// The model-level tables. `output` is handled above, outside `model.`.
const MODEL: Table = Table(&[("token_embd", "embed_tokens"), ("output_norm", "norm")]);

#[cfg(test)]
mod tests {
    use super::hf_name;

    /// Every name a real Qwen2 GGUF holds is mapped.
    ///
    /// The list is the fifteen distinct patterns read off
    /// `Qwen2.5-0.5B-Instruct-Q4_0.gguf` (291 tensors, 24 layers), which is
    /// the only thing that makes this test worth more than the map: both
    /// sides of a map written from imagination are the same imagination.
    #[test]
    fn every_tensor_a_qwen2_gguf_publishes_has_a_name() {
        let published = [
            ("blk.3.attn_k.bias", "model.layers.3.self_attn.k_proj.bias"),
            (
                "blk.3.attn_k.weight",
                "model.layers.3.self_attn.k_proj.weight",
            ),
            ("blk.3.attn_norm.weight", "model.layers.3.input_layernorm.weight"),
            (
                "blk.3.attn_output.weight",
                "model.layers.3.self_attn.o_proj.weight",
            ),
            ("blk.3.attn_q.bias", "model.layers.3.self_attn.q_proj.bias"),
            (
                "blk.3.attn_q.weight",
                "model.layers.3.self_attn.q_proj.weight",
            ),
            ("blk.3.attn_v.bias", "model.layers.3.self_attn.v_proj.bias"),
            (
                "blk.3.attn_v.weight",
                "model.layers.3.self_attn.v_proj.weight",
            ),
            ("blk.3.ffn_down.weight", "model.layers.3.mlp.down_proj.weight"),
            ("blk.3.ffn_gate.weight", "model.layers.3.mlp.gate_proj.weight"),
            (
                "blk.3.ffn_norm.weight",
                "model.layers.3.post_attention_layernorm.weight",
            ),
            ("blk.3.ffn_up.weight", "model.layers.3.mlp.up_proj.weight"),
            ("output.weight", "lm_head.weight"),
            ("output_norm.weight", "model.norm.weight"),
            ("token_embd.weight", "model.embed_tokens.weight"),
        ];
        for (gguf, hf) in published {
            assert_eq!(hf_name(gguf).as_deref(), Some(hf), "mapping {gguf}");
        }
    }

    /// The index is the layer's, not a fixed one.
    ///
    /// A map that dropped the index would send all 24 layers to the same
    /// name, and the artifact would hold one layer's worth of tensors under
    /// names that all collide -- which a name map is uniquely able to do
    /// silently.
    #[test]
    fn the_layer_index_is_carried_through() {
        for layer in [0u32, 7, 23, 199] {
            assert_eq!(
                hf_name(&format!("blk.{layer}.attn_q.weight")).as_deref(),
                Some(format!("model.layers.{layer}.self_attn.q_proj.weight").as_str())
            );
        }
    }

    /// A name this map does not know answers `None` rather than something.
    ///
    /// The refusal is the safety property: the caller turns `None` into an
    /// error naming the tensor, so a Qwen2 variant that publishes a tensor
    /// this table predates stops the import instead of writing an artifact
    /// with a hole in it.
    #[test]
    fn an_unknown_name_is_refused_rather_than_guessed() {
        for unknown in [
            "blk.3.attn_q_norm.weight",   // qwen3's, not qwen2's
            "blk.3.ffn_gate_inp.weight",  // a mixture's router
            "blk.x.attn_q.weight",        // not an index
            "blk.3.attn_q",               // no suffix
            "blk.3.attn_q.scales",        // a suffix this map does not carry
            "token_embd",                 // no suffix
            "rope_freqs.weight",          // computed, not published, by pie
        ] {
            assert_eq!(hf_name(unknown), None, "should not map {unknown}");
        }
    }
}
