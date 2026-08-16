//! Qwen2's tensor names, in pie's vocabulary and in the two foreign ones.
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
//! # The day the split moved to `shared`, as its own doc predicted
//!
//! This file used to argue that the `blk.{bid}.` split stays here because
//! "`shared`'s bar is **more than one generation binds it**", and that it
//! "will still be architecture-independent on the day a second family wants
//! it, which is the day it moves". Four generations wanted it, each with its
//! own copy of the same `split_layer` and the same `Table`. That day arrived,
//! and the machinery is now [`crate::shared::vocabulary`]. What stayed is
//! what was never shared: the rows.
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

use crate::shared::vocabulary::{Member, Vocab};

/// Every tensor Qwen2 publishes, and what each vocabulary calls it.
///
/// # `attn_norm` and `ffn_norm` are the pair worth reading twice
///
/// llama.cpp names a norm for what it PRECEDES and HuggingFace for where it
/// SITS, so `ffn_norm` is `post_attention_layernorm` and not anything with
/// "ffn" in it. The two vocabularies disagree about which end of the residual
/// a norm belongs to, and these two rows are that disagreement.
///
/// # The one row this table cannot get right
///
/// `output` / `lm_head`. The 0.5B row is `tied_embeddings: true`, and pie
/// spells a tie as the head's ABSENCE, so an artifact carrying one is a
/// different model as far as the catalog is concerned. The row is here
/// anyway and identification is left to say so, because the alternative is
/// worse: GGUF states no tie key at all, so dropping the tensor here would be
/// this table guessing at a fact the file does not contain -- and the tensor
/// is not redundant, since llama.cpp quantized this head to Q8_0 while the
/// embedding it would be tied to is Q4_0.
///
/// # The rows with no `gguf` column
///
/// `q_norm` and `post_feedforward_layernorm` are published by rows in
/// `crate::qwen_2` and by no Qwen2 GGUF measured here. They are not omissions
/// in the GGUF column: a family's table answers for every vocabulary that
/// names the family, and a member only one of them has is a `None`.
pub const VOCAB: Vocab = Vocab(&[
    // ── Inside a decoder layer ───────────────────────────────────────
    Member::gguf(
        "model.layers.{layer}.self_attn.q_proj",
        "blk.{layer}.attn_q",
    ),
    Member::gguf(
        "model.layers.{layer}.self_attn.k_proj",
        "blk.{layer}.attn_k",
    ),
    Member::gguf(
        "model.layers.{layer}.self_attn.v_proj",
        "blk.{layer}.attn_v",
    ),
    Member::gguf(
        "model.layers.{layer}.self_attn.o_proj",
        "blk.{layer}.attn_output",
    ),
    Member::same("model.layers.{layer}.self_attn.q_norm"),
    Member::gguf(
        "model.layers.{layer}.input_layernorm",
        "blk.{layer}.attn_norm",
    ),
    Member::gguf(
        "model.layers.{layer}.post_attention_layernorm",
        "blk.{layer}.ffn_norm",
    ),
    Member::same("model.layers.{layer}.post_feedforward_layernorm"),
    Member::gguf("model.layers.{layer}.mlp.gate_proj", "blk.{layer}.ffn_gate"),
    Member::gguf("model.layers.{layer}.mlp.up_proj", "blk.{layer}.ffn_up"),
    Member::gguf("model.layers.{layer}.mlp.down_proj", "blk.{layer}.ffn_down"),
    // ── Outside it ───────────────────────────────────────────────────
    Member::gguf("model.embed_tokens", "token_embd"),
    Member::gguf("model.norm", "output_norm"),
    // Not under `model.`, because the head sits outside the decoder that
    // `model.` names. A whole-name table states that; a stem table carved it
    // out by hand.
    Member::gguf("lm_head", "output"),
]);

#[cfg(test)]
mod tests {
    use super::VOCAB;

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
            (
                "blk.3.attn_norm.weight",
                "model.layers.3.input_layernorm.weight",
            ),
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
            (
                "blk.3.ffn_down.weight",
                "model.layers.3.mlp.down_proj.weight",
            ),
            (
                "blk.3.ffn_gate.weight",
                "model.layers.3.mlp.gate_proj.weight",
            ),
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
            assert_eq!(VOCAB.from_gguf(gguf).as_deref(), Some(hf), "mapping {gguf}");
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
                VOCAB
                    .from_gguf(&format!("blk.{layer}.attn_q.weight"))
                    .as_deref(),
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
            "blk.3.attn_q_norm.weight",  // qwen3's, not qwen2's
            "blk.3.ffn_gate_inp.weight", // a mixture's router
            "blk.x.attn_q.weight",       // not an index
            "blk.3.attn_q",              // no suffix
            "blk.3.attn_q.scales",       // a suffix this map does not carry
            "token_embd",                // no suffix
            "rope_freqs.weight",         // computed, not published, by pie
        ] {
            assert_eq!(VOCAB.from_gguf(unknown), None, "should not map {unknown}");
        }
    }
}
