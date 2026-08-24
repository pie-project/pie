use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::load::{Import, SfBase, copy, deinterleave};

use super::model::Model;

/// gpt-oss's fused gate/up rows are INTERLEAVED in the checkpoint — row `2i`
/// is gate `i` and row `2i + 1` is up `i` — and the canonical form is the
/// deinterleaved one, `[gate | up]` in halves.
///
/// THE CANONICAL FORM IS THE FIRE-READY FORM, and the fire that decides is
/// `mlp.swiglu_clamp_alpha`. The text projects one `[routes, 2 * inter]` row
/// and hands it to that point, whose claimed body reads `packed[row + i]` and
/// `packed[row + I + i]` — gate half first, up half second. So the bank rows
/// that produce it have to be in that order, and a routed GEMM cannot reorder
/// its own result. The permutation is free here (once, offline, a byte move)
/// and would be a kernel that does not exist there.
///
/// The AXIS is 1 on every one of these: the tensors are `[experts, rows, ..]`
/// and the interleaving is under the expert fan, never across it.
const FUSED: u32 = 2;

const ROW_AXIS: u32 = 1;

pub fn import_hf<B: SfBase, W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize>(
    m: &Model<W1, W2, K, TP>,
) -> Import {
    let mut i = Import::new::<B>();
    i.write("embed", copy("embed_tokens"));
    i.write("final_norm", copy("norm"));
    i.write("lm_head", copy("lm_head"));
    for l in 0..m.layers.len() {
        i.write(
            format!("layer.{l}.attn_norm"),
            copy(format!("layer.{l}.input_layernorm")),
        );
        i.write(
            format!("layer.{l}.mlp_norm"),
            copy(format!("layer.{l}.post_attention_layernorm")),
        );
        i.write(
            format!("layer.{l}.q_proj"),
            copy(format!("layer.{l}.self_attn.q_proj")),
        );
        i.write(
            format!("layer.{l}.q_bias"),
            copy(format!("layer.{l}.self_attn.q_proj.bias")),
        );
        i.write(
            format!("layer.{l}.k_proj"),
            copy(format!("layer.{l}.self_attn.k_proj")),
        );
        i.write(
            format!("layer.{l}.k_bias"),
            copy(format!("layer.{l}.self_attn.k_proj.bias")),
        );
        i.write(
            format!("layer.{l}.v_proj"),
            copy(format!("layer.{l}.self_attn.v_proj")),
        );
        i.write(
            format!("layer.{l}.v_bias"),
            copy(format!("layer.{l}.self_attn.v_proj.bias")),
        );
        i.write(
            format!("layer.{l}.o_proj"),
            copy(format!("layer.{l}.self_attn.o_proj")),
        );
        i.write(
            format!("layer.{l}.o_bias"),
            copy(format!("layer.{l}.self_attn.o_proj.bias")),
        );
        i.write(
            format!("layer.{l}.attn_sinks"),
            copy(format!("layer.{l}.self_attn.sinks")),
        );
        i.write(
            format!("layer.{l}.router"),
            copy(format!("layer.{l}.mlp.router")),
        );
        i.write(
            format!("layer.{l}.router_bias"),
            copy(format!("layer.{l}.mlp.router.bias")),
        );
        // THE BANK IS TWO PLANES AND THE CHECKPOINT SHIPS THEM AS TWO, so
        // every row here is a byte move. gpt-oss's release holds
        // `gate_up_proj_blocks` `[E, 2I, H/32, 16]` and `gate_up_proj_scales`
        // `[E, 2I, H/32]` — the codes and their E8M0 block exponents — and
        // NOTHING repacks them: the Marlin repack the legacy repr is named
        // after is reached only through a `native_mxfp4_moe` capability that
        // no driver in this tree states true. What the kernel reads is these
        // bytes, so these bytes are the canonical form.
        //
        // The rows are deinterleaved on the way through, and the three rows
        // of one bank agree about it: codes, scales and bias all carry gate
        // `i` at row `2i` in the checkpoint and all three land as halves.
        i.bank::<W2>(
            format!("layer.{l}.expert_gate_up_bank"),
            [
                deinterleave(
                    format!("layer.{l}.mlp.experts.gate_up_proj_blocks"),
                    ROW_AXIS,
                    FUSED,
                ),
                deinterleave(
                    format!("layer.{l}.mlp.experts.gate_up_proj_scales"),
                    ROW_AXIS,
                    FUSED,
                ),
            ],
        );
        i.write(
            format!("layer.{l}.expert_gate_up_bias"),
            deinterleave(
                format!("layer.{l}.mlp.experts.gate_up_proj_bias"),
                ROW_AXIS,
                FUSED,
            ),
        );
        // The down bank fuses nothing, so its rows are already `[E, H, I]`
        // and every plane is a straight copy.
        i.bank::<W2>(
            format!("layer.{l}.expert_down_bank"),
            [
                copy(format!("layer.{l}.mlp.experts.down_proj_blocks")),
                copy(format!("layer.{l}.mlp.experts.down_proj_scales")),
            ],
        );
        i.write(
            format!("layer.{l}.expert_down_bias"),
            copy(format!("layer.{l}.mlp.experts.down_proj_bias")),
        );
    }
    i
}
