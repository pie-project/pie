//! Qwen3-MoE as llama.cpp spells it, in the vocabulary the catalog reads.
//!
//! # The experts are the whole difference
//!
//! Everything outside the MLP is `qwen_3::import` again -- same eight
//! attention members, same two norms, same two model-level names. The MLP is
//! not: a mixture has a router and `E` experts where a dense block has one
//! `gate_proj`/`up_proj`/`down_proj` triple, and llama.cpp **stacks** the
//! experts. `blk.3.ffn_gate_exps.weight` is one `[E, I, H]` tensor where
//! HuggingFace publishes `E` separate `[I, H]` ones.
//!
//! So this map answers a question the other four do not: one name in, many
//! names out. [`is_stacked`] is where it says which, and `Ingest::Unstack`
//! is the shape that carries it.
//!
//! # Why it is taken apart rather than fused
//!
//! `shared::moe::hf_moe_expert_stacks` already joins `E` per-expert slabs
//! into the `[E, 2I, H]` `gate_up_proj` the fused-MoE forward binds, in the
//! `[up | gate]` order flashinfer's grouped GEMM reads. Fusing here would be
//! a second statement of that order, in a module that has no way to check it,
//! and the two would have to be kept in agreement by hand.
//!
//! Taking the stack apart instead leaves the join where it already is and
//! buys the check that matters: the artifact this produces holds exactly the
//! names the safetensors release produces, so the two can be compared object
//! by object. They were, and they agree -- see below.
//!
//! # Why this is a rename and a slice and nothing more
//!
//! Checked, not assumed, and by bytes. Against `Qwen3-30B-A3B` BF16 in both
//! spellings, read at the same tensors:
//!
//! | GGUF | HuggingFace | |
//! |---|---|---|
//! | `blk.0.ffn_gate_exps.weight` expert 0 | `...mlp.experts.0.gate_proj.weight` | identical |
//! | `blk.0.ffn_gate_exps.weight` expert 1 | `...mlp.experts.1.gate_proj.weight` | identical |
//! | `blk.0.ffn_up_exps.weight` expert 0 | `...mlp.experts.0.up_proj.weight` | identical |
//! | `blk.0.ffn_down_exps.weight` expert 0 | `...mlp.experts.0.down_proj.weight` | identical |
//!
//! Three things at once, and all three had to be measured rather than
//! recalled. The experts stack in index order, so slab `e` is expert `e`.
//! Nothing is transposed -- `ffn_gate_exps` declares GGUF dims
//! `[2048, 768, 128]`, which read fastest-first is `[128, 768, 2048]` =
//! `[E, I, H]`, the same `[I, H]` per expert HuggingFace ships, and
//! `ffn_down_exps` is `[E, H, I]` the same way. And `Qwen3MoeModel` moves no
//! data, exactly as `Qwen3Model` does not.
//!
//! A transpose would have been invisible here if it existed: `Qwen3-30B-A3B`
//! has `H = 2048` and `I = 768`, which do differ, but gpt-oss has
//! `H = I = 2880` and would have hidden it completely. That is why the check
//! is bytes against a twin and not a shape assertion.
//!
//! # No `attn_q.bias`, and an `output.weight` that may or may not be there
//!
//! Qwen3 dropped the QKV biases. The suffix split still admits `.bias`, on
//! the same reasoning as `qwen_3::import`: refusing a suffix this family
//! happens not to publish would be this module asserting a fact the
//! checkpoint states for itself. `output.weight` is mapped for the same
//! reason -- `Qwen3-30B-A3B` publishes one, a tied variant would not, and
//! GGUF states no tie key to read.

/// The HuggingFace name for a GGUF tensor, or `None` if this map has none.
///
/// For a tensor [`is_stacked`] answers `true` for, the string returned is a
/// TEMPLATE with a single `{}` where the instance index goes, and not a name.
/// The two are answered by one function because they are one row of one
/// table: `ffn_gate_exps` is the stacked spelling of `gate_proj` and there is
/// no name for it that is not per-expert.
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

/// Whether llama.cpp published this tensor as one stack of many.
///
/// True for exactly the three routed-expert tensors, which are also exactly
/// the three whose mapped name holds a `{}`. Stated as its own function
/// rather than inferred from the `{}` so that a template introduced by a
/// typo is a refused name and not a silent unstacking of a tensor that has
/// one instance.
///
/// `ffn_gate_inp` is deliberately NOT here: the router is one `[E, H]`
/// tensor in both vocabularies, and its leading extent being `E` is the trap
/// -- a rule that unstacked whatever led with the expert count would take the
/// router apart into `E` rows.
#[must_use]
pub fn is_stacked(gguf: &str) -> bool {
    let Some((stem, "weight")) = gguf.rsplit_once('.') else {
        return false;
    };
    let Some((_, member)) = split_layer(stem) else {
        return false;
    };
    matches!(member, "ffn_gate_exps" | "ffn_up_exps" | "ffn_down_exps")
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

/// The per-layer members, all twelve of them.
///
/// `attn_norm` and `ffn_norm` are the pair worth reading twice: llama.cpp
/// names them for what they precede, HuggingFace for where they sit in the
/// block, so `ffn_norm` is `post_attention_layernorm` and NOT anything with
/// "ffn" in it. Gemma is the family where copying that row would go wrong --
/// see `gemma_3::import` -- and it is a row this family does share the
/// meaning of, which is exactly why the table is spelled out here anyway.
///
/// `ffn_gate_inp` is the router and `mlp.gate.weight` is what HuggingFace
/// calls it: a `gate` that selects experts, and not the `gate_proj` half of
/// a gated MLP. The two are one edit apart and one is `[E, H]` while the
/// other is `[I, H]`.
const LAYER: Table = Table(&[
    ("attn_q", "self_attn.q_proj"),
    ("attn_k", "self_attn.k_proj"),
    ("attn_v", "self_attn.v_proj"),
    ("attn_output", "self_attn.o_proj"),
    ("attn_q_norm", "self_attn.q_norm"),
    ("attn_k_norm", "self_attn.k_norm"),
    ("attn_norm", "input_layernorm"),
    ("ffn_norm", "post_attention_layernorm"),
    ("ffn_gate_inp", "mlp.gate"),
    ("ffn_gate_exps", "mlp.experts.{}.gate_proj"),
    ("ffn_up_exps", "mlp.experts.{}.up_proj"),
    ("ffn_down_exps", "mlp.experts.{}.down_proj"),
]);

/// The model-level tables. `output` is handled above, outside `model.`.
const MODEL: Table = Table(&[("token_embd", "embed_tokens"), ("output_norm", "norm")]);

#[cfg(test)]
mod tests {
    use super::{hf_name, is_stacked};

    /// Every name a real Qwen3-MoE GGUF holds is mapped.
    ///
    /// The list is the fifteen distinct patterns read off
    /// `Qwen3-30B-A3B-BF16.gguf` (579 tensors, 48 layers, 128 experts),
    /// which is the only thing that makes this test worth more than the map:
    /// both sides of a map written from imagination are the same
    /// imagination.
    #[test]
    fn every_tensor_a_qwen3moe_gguf_publishes_has_a_name() {
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
                "blk.3.ffn_norm.weight",
                "model.layers.3.post_attention_layernorm.weight",
            ),
            (
                "blk.3.ffn_gate_inp.weight",
                "model.layers.3.mlp.gate.weight",
            ),
            (
                "blk.3.ffn_gate_exps.weight",
                "model.layers.3.mlp.experts.{}.gate_proj.weight",
            ),
            (
                "blk.3.ffn_up_exps.weight",
                "model.layers.3.mlp.experts.{}.up_proj.weight",
            ),
            (
                "blk.3.ffn_down_exps.weight",
                "model.layers.3.mlp.experts.{}.down_proj.weight",
            ),
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output_norm.weight", "model.norm.weight"),
            ("output.weight", "lm_head.weight"),
        ];
        for (gguf, hf) in published {
            assert_eq!(hf_name(gguf).as_deref(), Some(hf), "mapping {gguf}");
        }
    }

    /// Exactly the routed experts unstack, and exactly they hold a `{}`.
    ///
    /// The two facts are checked against each other over the whole published
    /// set, because a template that does not unstack puts a literal `{}` in
    /// the artifact and an unstacking with no template writes `E` tensors
    /// under one name.
    #[test]
    fn the_routed_experts_are_the_stacked_ones_and_nothing_else_is() {
        let stacked = [
            "blk.3.ffn_gate_exps.weight",
            "blk.3.ffn_up_exps.weight",
            "blk.3.ffn_down_exps.weight",
        ];
        let flat = [
            "blk.3.ffn_gate_inp.weight",
            "blk.3.attn_q.weight",
            "blk.3.ffn_norm.weight",
            "token_embd.weight",
            "output.weight",
        ];
        for name in stacked {
            assert!(is_stacked(name), "{name} stacks");
            assert!(hf_name(name).unwrap().contains("{}"), "{name} templates");
        }
        for name in flat {
            assert!(!is_stacked(name), "{name} does not stack");
            assert!(!hf_name(name).unwrap().contains("{}"), "{name} is a name");
        }
    }

    /// The router is not an expert stack.
    ///
    /// `ffn_gate_inp` leads with `E` exactly as the three expert tensors do,
    /// and it is one tensor on both sides. A rule that read the leading
    /// extent instead of the name would take it apart into `E` rows of the
    /// hidden size, each declared as a `[H]` weight, and every shape check
    /// downstream would pass.
    #[test]
    fn the_router_is_one_tensor_and_stays_one() {
        assert!(!is_stacked("blk.0.ffn_gate_inp.weight"));
        assert_eq!(
            hf_name("blk.0.ffn_gate_inp.weight").as_deref(),
            Some("model.layers.0.mlp.gate.weight")
        );
    }

    /// The index is the layer's, not a fixed one.
    #[test]
    fn the_layer_index_is_carried_through() {
        for layer in [0u32, 7, 47, 199] {
            assert_eq!(
                hf_name(&format!("blk.{layer}.ffn_up_exps.weight")).as_deref(),
                Some(format!("model.layers.{layer}.mlp.experts.{{}}.up_proj.weight").as_str())
            );
        }
    }

    /// A name this map does not know answers `None` rather than something.
    #[test]
    fn an_unknown_name_is_refused_rather_than_guessed() {
        for unknown in [
            "blk.3.ffn_gate.weight",       // the dense MLP this family has none of
            "blk.3.ffn_up.weight",         // likewise
            "blk.3.ffn_gate_shexp.weight", // qwen2moe's shared expert, not published here
            "blk.x.attn_q.weight",         // not an index
            "blk.3.attn_q_norm",           // no suffix
            "blk.3.attn_q.scales",         // a suffix this map does not carry
            "token_embd",                  // no suffix
            "rope_freqs.weight",           // llama's, and not published here
        ] {
            assert_eq!(hf_name(unknown), None, "should not map {unknown}");
            assert!(!is_stacked(unknown), "should not unstack {unknown}");
        }
    }
}
