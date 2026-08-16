//! The llama-like lineage's tensor names, in every vocabulary that spells them.
//!
//! Sibling of [`crate::qwen_2::import`], and the family that one deferred:
//! "a permutation applied when it was not wanted, or skipped when it was, is
//! not an error. It is a model that loads, serves, and answers slightly
//! wrong, and pie has no cheap check that notices."
//!
//! # Why this is the one import table that lives in `shared`
//!
//! It used to be `llama_2::import`, "named for the earliest generation that
//! carries it rather than for a family of its own" -- and `llama_2` has no
//! catalog rows at all, so the table that feeds Llama 3's rows lived in a
//! directory nothing reachable is written in.
//!
//! Both halves of the sharing were then measured. On the HuggingFace side,
//! the rows of `llama_3`, `mistral_3`, `olmo_2`, `olmo_3` and `phi_3` name
//! the SAME THIRTEEN tensors, character for character -- five generations,
//! one member set. On the GGUF side, `general.architecture` is `llama` for
//! Llama 2, Llama 3, Mistral and everything else llama.cpp folds into that
//! converter, so the input vocabulary is one too.
//!
//! That is `shared`'s bar met twice over -- the same bar
//! [`crate::shared::llama_like::contract`] met when ten generations reached
//! across a sibling edge for it. Being written first is a fact about who
//! wrote it, not a claim of ownership.
//!
//! # The `gguf` column belongs to one architecture, not to five generations
//!
//! `phi_3` and `olmo_2` are llama-like to pie and have their OWN
//! `general.architecture` in llama.cpp (`phi3`, `olmo2`). Neither has an arm
//! in [`crate::ingest`], so neither can reach the column below -- the GGUF
//! door is keyed on the architecture, and the HuggingFace door never reads
//! that column at all. The column is the `llama` architecture's, and a
//! generation naming this table for its HuggingFace names is not claiming
//! llama.cpp spells it that way.
//!
//! Which pie row an import identifies as is decided afterwards, by the
//! tensors and shapes the artifact actually holds.
//!
//! # The permutation, measured rather than ported
//!
//! `LlamaModel` sets `undo_permute = True`, which rearranges the rows of
//! every Q and K projection so llama.cpp's rope reads them in its own order.
//! Reading the direction off the converter is how this gets written
//! backwards, so it was read off the files instead:
//! `Llama-3.2-1B-Instruct-BF16.gguf` against its own safetensors release,
//! both BF16, so nothing but the reordering could differ.
//!
//! Every tensor matched bit for bit except `attn_q` and `attn_k`, and for
//! those the row map is, within each head:
//!
//! ```text
//! gguf[h*hd + 2k]     ==  hf[h*hd + k]
//! gguf[h*hd + 2k + 1] ==  hf[h*hd + hd/2 + k]
//! ```
//!
//! llama.cpp stores the two halves of a rope pair ADJACENT; HuggingFace
//! stores them half a head apart. The regrouping is per head and never
//! crosses one, which is worth stating because the obvious reading of the
//! converter -- `reshape(n_head, 2, hd/2).swapaxes(0, 1)` -- moves rows
//! BETWEEN heads, and it was checked and does not reproduce the file.
//!
//! The head count is `n_head` for Q and `n_head_kv` for K, because K has one
//! row group per KV head. On Llama-3.2-1B those are 32 and 8, and using 32
//! for both produces a K that is the right shape and the wrong model.
//!
//! # Nothing here moves a byte
//!
//! The regrouping is declared, not performed: it is `head` bands of rows,
//! taken two apart, which the loader writes as `Concat` of `Stride`. Both
//! survive a blocked payload, since a GGUF block runs along the row and this
//! only ever moves whole rows -- so a Q4_0 llama regroups without being
//! decoded first.

use crate::shared::vocabulary::{Member, Vocab, gguf_member};
use model_loader::checkpoint::Attributes;
/// Every tensor the llama-like lineage publishes, and what each vocabulary
/// calls it.
///
/// Thirteen rows, which is the exact set the rows of `crate::llama_3`,
/// `crate::mistral_3`, `crate::olmo_2`, `crate::olmo_3` and `crate::phi_3`
/// name between them. `crate::qwen_2` and `crate::qwen_3` are llama-like in
/// `forward` and NOT here -- they add a `q_norm`, and qwen2 adds three
/// attention biases -- which is why they keep their own tables and this one
/// does not try to cover them.
///
/// The `attn_norm`/`ffn_norm` pair carries the trap it does everywhere:
/// llama.cpp names a norm for what it PRECEDES and HuggingFace for where it
/// SITS, so `ffn_norm` is `post_attention_layernorm`. Gemma is the family
/// where copying that row would go wrong -- see `crate::gemma_3::import`.
///
/// `rope_freqs.weight` has no row. It is not a weight; see [`is_derived`].
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
    // No GGUF column: the `llama` converter publishes no query norm, and
    // rows in `crate::llama_3` and `crate::olmo_2` ask for one.
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
    Member::gguf("lm_head", "output"),
]);

/// Tensors llama.cpp computed and pie computes for itself.
///
/// `rope_freqs.weight` is llama.cpp's precomputed rope frequency table --
/// Llama 3's rope scaling, evaluated at conversion time. No HuggingFace
/// release publishes it, no catalog row names it, and pie builds its own
/// from the config. Carrying it would put a tensor in the artifact that
/// nothing can identify; refusing it would stop every Llama 3 import, since
/// the file always has one. It is dropped, which is the only answer that is
/// true: it is not a weight.
#[must_use]
pub fn is_derived(gguf: &str) -> bool {
    gguf == "rope_freqs.weight"
}

/// How many row groups this tensor's rows regroup by, or `None` if its rows
/// are already in pie's order.
///
/// Reads the head counts from the file rather than from a table, because they
/// are what the file states about itself and a table would be a second place
/// for them to be wrong.
#[must_use]
pub fn regroup_heads(attributes: &Attributes, gguf: &str) -> Option<u32> {
    let (_, member) = gguf_member(gguf)?;
    let key = match member {
        "attn_q" => "llama.attention.head_count",
        "attn_k" => "llama.attention.head_count_kv",
        _ => return None,
    };
    match attributes.get(key)? {
        model_loader::checkpoint::Attribute::Uint(heads) => u32::try_from(*heads).ok(),
        _ => None,
    }
    .filter(|heads| *heads > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_loader::checkpoint::Attribute;

    fn attrs() -> Attributes {
        Attributes::from_pairs([
            (
                "general.architecture".to_string(),
                Attribute::Text("llama".to_string()),
            ),
            (
                "llama.attention.head_count".to_string(),
                Attribute::Uint(32),
            ),
            (
                "llama.attention.head_count_kv".to_string(),
                Attribute::Uint(8),
            ),
        ])
    }

    /// The twelve patterns `Llama-3.2-1B-Instruct-BF16.gguf` actually holds,
    /// read off the file rather than imagined.
    #[test]
    fn every_tensor_a_llama_gguf_publishes_has_an_answer() {
        let published = [
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
            (
                "blk.3.attn_q.weight",
                "model.layers.3.self_attn.q_proj.weight",
            ),
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
            ("output_norm.weight", "model.norm.weight"),
            ("token_embd.weight", "model.embed_tokens.weight"),
            // Not in the 1B, which ties its head; Llama 2 publishes one.
            ("output.weight", "lm_head.weight"),
        ];
        for (gguf, hf) in published {
            assert_eq!(VOCAB.from_gguf(gguf).as_deref(), Some(hf), "mapping {gguf}");
        }
        // The twelfth pattern, and the only one with no name.
        assert!(is_derived("rope_freqs.weight"));
        assert_eq!(VOCAB.from_gguf("rope_freqs.weight"), None);
    }

    /// Q regroups by `n_head`, K by `n_head_kv`, and nothing else regroups.
    ///
    /// The two counts differ on every GQA model, so a pass that used one for
    /// both would produce a K of the right shape whose rows are grouped four
    /// heads at a time -- exactly the failure this family was deferred over.
    #[test]
    fn only_q_and_k_regroup_and_they_regroup_by_their_own_head_counts() {
        let attributes = attrs();
        assert_eq!(regroup_heads(&attributes, "blk.0.attn_q.weight"), Some(32));
        assert_eq!(regroup_heads(&attributes, "blk.0.attn_k.weight"), Some(8));
        for still in [
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_gate.weight",
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
        ] {
            assert_eq!(regroup_heads(&attributes, still), None, "{still}");
        }
    }

    /// A file that does not state its head count regroups nothing rather than
    /// guessing a count.
    #[test]
    fn a_missing_head_count_is_not_a_default() {
        let bare = Attributes::from_pairs([(
            "general.architecture".to_string(),
            Attribute::Text("llama".to_string()),
        )]);
        assert_eq!(regroup_heads(&bare, "blk.0.attn_q.weight"), None);
    }

    /// The layer index is the layer's, not a fixed one.
    #[test]
    fn the_layer_index_is_carried_through() {
        for layer in [0u32, 7, 31, 199] {
            assert_eq!(
                VOCAB.from_gguf(&format!("blk.{layer}.attn_q.weight")).as_deref(),
                Some(format!("model.layers.{layer}.self_attn.q_proj.weight").as_str())
            );
        }
    }

    /// A name this map does not know answers `None` rather than something.
    #[test]
    fn an_unknown_name_is_refused_rather_than_guessed() {
        for unknown in [
            "blk.3.attn_q_norm.weight",
            "blk.3.ffn_gate_inp.weight",
            "blk.x.attn_q.weight",
            "blk.3.attn_q",
            "blk.3.attn_q.scales",
            "token_embd",
        ] {
            assert_eq!(VOCAB.from_gguf(unknown), None, "should not map {unknown}");
        }
    }
}
