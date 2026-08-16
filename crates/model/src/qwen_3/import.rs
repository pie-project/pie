//! Qwen3 as llama.cpp spells it, in the vocabulary the catalog reads.
//!
//! # Why this is a rename and nothing more
//!
//! Checked, not assumed. A BF16 GGUF of `Qwen3-0.6B` and the BF16
//! safetensors release it was converted from agree **bit for bit** on
//! `attn_q`, `attn_k`, `attn_v`, `attn_q_norm` and `token_embd` -- so
//! llama.cpp's `Qwen3Model` touches no data on the way in, and this file has
//! only to say what each tensor is called. Compare `llama_2::import`, whose
//! Q and K come back at cosine 0.02 against the same kind of twin because
//! `LlamaModel` reorders every rope row.
//!
//! That check is the reason this module is 200 lines and not a guess. A name
//! map is cheap to write from a converter's source and expensive to be wrong
//! about, because a permutation applied when it was not wanted is not an
//! error -- it is a model that loads, serves, and answers slightly wrong.
//!
//! # Why it is a file of its own next to `qwen_2::import`
//!
//! The two tables are nine rows the same and two rows different, and the
//! temptation is to share the nine. They are not shared. A GGUF ingest pass
//! is per **input vocabulary**, and the whole value of stating one is that a
//! family's answer can be read in one place and checked against one file on
//! disk. A table assembled from a base plus a delta cannot be read that way,
//! and the day llama.cpp changes one of those nine rows for one architecture
//! -- which is exactly the kind of thing it does -- the shared version is a
//! silent wrong answer for the other.
//!
//! # What Qwen3 has that Qwen2 does not, and the reverse
//!
//! `attn_q_norm` / `attn_k_norm`: Qwen3 normalizes each head's queries and
//! keys before rope, and those are 128-element tensors (one head dimension,
//! not one hidden size), so a map that sent them anywhere plausible would be
//! caught by shape identification rather than by name. They are still stated
//! rather than left out, because `None` here stops the import.
//!
//! No `attn_q.bias`: Qwen3 dropped the QKV biases Qwen2 carried. The suffix
//! split below still admits `.bias` -- it costs nothing, and refusing a
//! suffix this family happens not to publish would be this module asserting a
//! fact about the checkpoint that the checkpoint states for itself.
//!
//! No `output.weight`: `Qwen3-0.6B` ties its head, and llama.cpp writes no
//! head tensor when the tie holds. The name is mapped anyway for the variants
//! that do publish one, on the same reasoning as `qwen_2::import`: GGUF
//! states no tie key, so dropping the tensor here would be a guess at a fact
//! the file does not contain.

/// The HuggingFace name for a GGUF tensor, or `None` if this map has none.
///
/// `None` is a refusal and not a skip. A tensor silently left out is a model
/// that loads with a layer missing, and the caller turns this into an error
/// naming the tensor.
#[must_use]
pub fn hf_name(gguf: &str) -> Option<String> {
    let (stem, suffix) = match gguf.rsplit_once('.') {
        Some((stem, tail @ ("weight" | "bias"))) => (stem, tail),
        _ => return None,
    };
    let hf = match split_layer(stem) {
        Some((layer, member)) => format!("model.layers.{layer}.{}", LAYER.get_member(member)?),
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
struct Table(&'static [(&'static str, &'static str)]);

impl Table {
    fn get_member(&self, member: &str) -> Option<&'static str> {
        self.0
            .iter()
            .find(|(gguf, _)| *gguf == member)
            .map(|(_, hf)| *hf)
    }
}

/// The per-layer members, all eleven of them.
///
/// `attn_norm` and `ffn_norm` are the pair worth reading twice: llama.cpp
/// names them for what they precede, HuggingFace for where they sit in the
/// block, so `ffn_norm` is `post_attention_layernorm` and NOT anything with
/// "ffn" in it.
const LAYER: Table = Table(&[
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("attn_q_norm", "self_attn.q_norm"),
    ("attn_k_norm", "self_attn.k_norm"),
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

    /// Every name a real Qwen3 GGUF holds is mapped.
    ///
    /// The list is the thirteen distinct patterns read off
    /// `Qwen3-0.6B-BF16.gguf` (310 tensors, 28 layers), which is the only
    /// thing that makes this test worth more than the map: both sides of a
    /// map written from imagination are the same imagination.
    #[test]
    fn every_tensor_a_qwen3_gguf_publishes_has_a_name() {
        let published = [
            (
                "blk.3.attn_k.weight",
                "model.layers.3.self_attn.k_proj.weight",
            ),
            (
                "blk.3.attn_k_norm.weight",
                "model.layers.3.self_attn.k_norm.weight",
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
                "blk.3.attn_q_norm.weight",
                "model.layers.3.self_attn.q_norm.weight",
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
        ];
        for (gguf, hf) in published {
            assert_eq!(hf_name(gguf).as_deref(), Some(hf), "mapping {gguf}");
        }
    }

    /// The two head norms do not collide with each other or with the
    /// projections they sit beside.
    ///
    /// `attn_q` and `attn_q_norm` share a prefix, and a table looked up by
    /// prefix rather than by whole member would send the norm to the
    /// projection's name -- a 128-element tensor under a 2048x1024 tensor's
    /// name, in a family where both exist in every layer.
    #[test]
    fn a_head_norm_is_not_read_as_the_projection_it_precedes() {
        for (norm, proj) in [("attn_q", "q"), ("attn_k", "k")] {
            let n = hf_name(&format!("blk.0.{norm}_norm.weight")).unwrap();
            let p = hf_name(&format!("blk.0.{norm}.weight")).unwrap();
            assert_eq!(n, format!("model.layers.0.self_attn.{proj}_norm.weight"));
            assert_eq!(p, format!("model.layers.0.self_attn.{proj}_proj.weight"));
            assert_ne!(n, p);
        }
    }

    /// The index is the layer's, not a fixed one.
    #[test]
    fn the_layer_index_is_carried_through() {
        for layer in [0u32, 7, 27, 199] {
            assert_eq!(
                hf_name(&format!("blk.{layer}.attn_q_norm.weight")).as_deref(),
                Some(format!("model.layers.{layer}.self_attn.q_norm.weight").as_str())
            );
        }
    }

    /// A name this map does not know answers `None` rather than something.
    #[test]
    fn an_unknown_name_is_refused_rather_than_guessed() {
        for unknown in [
            "blk.3.ffn_gate_inp.weight",  // a mixture's router, not this one
            "blk.3.ffn_gate_exps.weight", // qwen3moe's stacked experts
            "blk.x.attn_q.weight",        // not an index
            "blk.3.attn_q_norm",          // no suffix
            "blk.3.attn_q.scales",        // a suffix this map does not carry
            "token_embd",                 // no suffix
            "rope_freqs.weight",          // llama's, and not published here
        ] {
            assert_eq!(hf_name(unknown), None, "should not map {unknown}");
        }
    }
}
