//! NO `(1 + w)` FOLD ON EITHER LEG, and the ten `plus_one` rows that used to
//! stand here were wrong about the checkpoint rather than wrong about the
//! law.
//!
//! Gemma-2 and Gemma-3 store rmsnorm banks centred on zero and multiply by
//! `1 + w`; qwen3.5 does the same, which is why `norm.rmsnorm_plus_one`
//! exists and why qwen's texts state it. Gemma-4 DROPPED that convention:
//! its banks hold the final scale. Two independent measurements say so, and
//! both are against the shipped E4B safetensors.
//!
//! The first is internal to the checkpoint. `q_norm` and `k_norm` are
//! constant across their `head_dim` and their product carries the attention
//! temperature -- `q * k * sqrt(head_dim)` is 2.000 at all 35 sliding layers
//! (1.4142 at all 7 global ones), which is exactly why `Attn::sm_scale` is
//! 1.0 and nothing else divides by `sqrt(d)`. Read as offsets the same
//! product is 35.3 to 36.5 with no structure at all, and layer 20's `q_norm`
//! -- exactly 1.0, a scale of one -- would have to be an offset of one.
//!
//! The second is the model's, end to end and against a third party. These
//! bytes go in with no fold and gemma-4-E4B runs: 890 of 890 steps fire,
//! the argmax is 785 ("ite") at 7.5938, and six teacher-forced positions
//! match a transformers 5.15.1 forward on the same cached checkpoint —
//! three bit-equal and three within one ulp. The join says the same thing
//! from the table's side: 575 demanded / 575 satisfied, every satisfied
//! param's storage matching its plan repr.
//!
//! (This paragraph used to cite `driver-cuda/tests/real_gemma4.rs` firing
//! `NormVariant::Plain`. That file does not exist and `NormVariant` is
//! retired vocabulary — the convention is a POINT now,
//! `norm.rmsnorm{,_per_head}_plus_one` beside `norm.rmsnorm{,_per_head}`,
//! and this text states the plain one.)
//!
//! NO GGUF LEG. There was one — 90 rows of `blk.{l}.*` spellings — and it
//! could not run for three independent reasons: all three drivers select a
//! `safetensors*` base by name, `model::snapshot` reads safetensors and
//! nothing else, and the E4B leg said `scalar_of`, a verb the interpreter
//! refused unconditionally. Nothing had ever held a GGUF file against those
//! names, which is the whole objection: a production table nobody can run
//! is a table nobody has checked, and this file's own law is that the
//! checkpoint decides. `every_registered_verb_is_one_the_interpreter_runs`
//! is what stops the next one.
//!
//! THE LAW THIS RESTS ON: import may rewrite bytes, load may only view. The
//! permission to rewrite is a permission to reach the CANONICAL form -- a
//! re-layout, a pack, a fold whose matching point the text then states. It
//! is not a permission to change what a number MEANS. A fold at import and a
//! plain point in the text agree with each other whether or not either is
//! true of the file, so their agreement proves nothing; the checkpoint is
//! the third party, and it is the one that decides.

use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::load::{Import, SfBase, copy, pack, slice};

use super::model::{AttnBanks, Model};

pub fn import_hf<B: SfBase, W1: Dtype, K: KvDtype, const TP: usize>(
    m: &Model<W1, K, TP>,
) -> Import {
    let mut i = Import::new::<B>();
    i.write("embed", copy("embed_tokens"));
    i.write("final_norm", copy("norm"));
    for (l, w) in m.layers.iter().enumerate() {
        i.write(
            format!("layer.{l}.attn_norm"),
            copy(format!("layer.{l}.input_layernorm")),
        );
        i.write(
            format!("layer.{l}.post_attn_norm"),
            copy(format!("layer.{l}.post_attention_layernorm")),
        );
        i.write(
            format!("layer.{l}.pre_ffw_norm"),
            copy(format!("layer.{l}.pre_feedforward_layernorm")),
        );
        i.write(
            format!("layer.{l}.post_ffw_norm"),
            copy(format!("layer.{l}.post_feedforward_layernorm")),
        );
        i.write(
            format!("layer.{l}.q_norm"),
            copy(format!("layer.{l}.self_attn.q_norm")),
        );
        match &w.attn.banks {
            AttnBanks::Owned { .. } => {
                i.write(
                    format!("layer.{l}.k_norm"),
                    copy(format!("layer.{l}.self_attn.k_norm")),
                );
                i.write(
                    format!("layer.{l}.qkv"),
                    pack([
                        format!("layer.{l}.self_attn.q_proj"),
                        format!("layer.{l}.self_attn.k_proj"),
                        format!("layer.{l}.self_attn.v_proj"),
                    ]),
                );
            }
            AttnBanks::Shared { .. } => {
                i.write(
                    format!("layer.{l}.q_proj"),
                    copy(format!("layer.{l}.self_attn.q_proj")),
                );
            }
        }
        i.write(
            format!("layer.{l}.o_proj"),
            copy(format!("layer.{l}.self_attn.o_proj")),
        );
        i.write(
            format!("layer.{l}.gate_up"),
            pack([
                format!("layer.{l}.mlp.gate_proj"),
                format!("layer.{l}.mlp.up_proj"),
            ]),
        );
        i.write(
            format!("layer.{l}.down"),
            copy(format!("layer.{l}.mlp.down_proj")),
        );
    }
    if m.ple.is_some() {
        i.write("ple.model_proj", copy("per_layer_model_projection"));
        i.write("ple.model_norm", copy("per_layer_projection_norm"));
        // ONE SLICE PER LAYER OUT OF THE ONE CHECKPOINT TENSOR. See
        // `PleLayer::table`: `embed_tokens_per_layer` is `[vocab, layers *
        // ple_dim]` and no shader plane can bind it whole.
        let ple_dim = m.ple.as_ref().map_or(0, |p| u64::from(p.dim));
        for l in 0..m.layers.len() {
            i.write(
                format!("layer.{l}.ple_table"),
                slice("embed_tokens_per_layer", 1, l as u64 * ple_dim, ple_dim),
            );
            i.write(
                format!("layer.{l}.ple_gate"),
                copy(format!("layer.{l}.per_layer_input_gate")),
            );
            i.write(
                format!("layer.{l}.ple_proj"),
                copy(format!("layer.{l}.per_layer_projection")),
            );
            i.write(
                format!("layer.{l}.ple_norm"),
                copy(format!("layer.{l}.post_per_layer_input_norm")),
            );
            // The HF release files this as a `[1]` tensor of its own, beside
            // the norm rather than inside it -- `layers.{l}.layer_scalar`,
            // which the legacy alias table reaches under the role name
            // `scalar` (`shared/weight_names.rs`). A plain copy, then.
            i.write(
                format!("layer.{l}.ple_scalar"),
                copy(format!("layer.{l}.layer_scalar")),
            );
        }
    }
    i
}
