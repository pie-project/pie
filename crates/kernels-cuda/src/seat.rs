#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Rows(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Lanes(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Routes(pub u32);

impl Rows {
    #[must_use]
    pub const fn fan(self, fan: u32) -> Routes {
        Routes(self.0 * fan)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reads {
    Nothing,
    Rows,
    RowsAndLanes,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct EntryInfo {
    pub name: &'static str,
    pub reads: Reads,
}

const fn entry(name: &'static str, reads: Reads) -> EntryInfo {
    EntryInfo { name, reads }
}

pub const ENTRIES: &[EntryInfo] = &[
    entry("attention.decode", Reads::RowsAndLanes),
    entry("attention.decode_lse", Reads::RowsAndLanes),
    entry("attention.decode_rel", Reads::RowsAndLanes),
    entry("attention.masked", Reads::RowsAndLanes),
    entry("attention.prefill", Reads::RowsAndLanes),
    entry("attention.prefill_lse", Reads::RowsAndLanes),
    entry("attention.ragged", Reads::RowsAndLanes),
    entry("attention.prefill_rel", Reads::RowsAndLanes),
    entry("attention.plan_decode", Reads::Nothing),
    entry("attention.plan_prefill", Reads::Nothing),
    entry("attention.index_layernorm_rope", Reads::Rows),
    entry("attention.index_rope", Reads::Rows),
    entry("attention.index_topk", Reads::Rows),
    entry("attention.kv_append", Reads::Rows),
    entry("attention.kv_append_shared", Reads::Rows),
    entry("attention.merge_lse", Reads::Rows),
    entry("attention.mla_decode_selected", Reads::Rows),
    entry("attention.mla_latents", Reads::Rows),
    entry("attention.mla_latents_rope", Reads::Rows),
    entry("attention.mla_prefill_selected", Reads::Rows),
    entry("attention.mla_split_q_b", Reads::Rows),
    entry("attention.ple_ngram_ids", Reads::Rows),
    entry("attention.pool_lse", Reads::Rows),
    entry("attention.short_conv", Reads::Rows),
    entry("attention.sink", Reads::Rows),
    entry("attention.ssm_causal_conv1d", Reads::Rows),
    entry("attention.ssm_gated_delta", Reads::Rows),
    entry("attention.ssm_gdn_prep", Reads::Rows),
    entry("attention.ssm_kda_step", Reads::Rows),
    entry("attention.ple_ngram_ids_chunked", Reads::RowsAndLanes),
    entry("attention.short_conv_chunked", Reads::RowsAndLanes),
    entry("attention.ssm_causal_conv1d_chunked", Reads::RowsAndLanes),
    entry("attention.ssm_gated_delta_chunked", Reads::RowsAndLanes),
    entry("attention.ssm_kda_chunked", Reads::RowsAndLanes),
    entry("elementwise.add", Reads::Rows),
    entry("elementwise.add_bias", Reads::Rows),
    entry("elementwise.clamp", Reads::Rows),
    entry("elementwise.clamp_learned", Reads::Rows),
    entry("elementwise.gate_sigmoid_mul", Reads::Rows),
    entry("elementwise.gate_sigmoid_mul_heads", Reads::Rows),
    entry("elementwise.gated_residual_add", Reads::Rows),
    entry("elementwise.gated_residual_norm_modulate", Reads::Rows),
    entry("elementwise.hc_expand", Reads::Rows),
    entry("elementwise.hc_fold", Reads::Rows),
    entry("elementwise.hc_gates", Reads::Rows),
    entry("elementwise.hc_inject", Reads::Rows),
    entry("elementwise.hc_mix", Reads::Rows),
    entry("elementwise.hc_rmsnorm_f32", Reads::Rows),
    entry("elementwise.layernorm", Reads::Rows),
    entry("elementwise.layernorm_no_scale", Reads::Rows),
    entry("elementwise.modulate", Reads::Rows),
    entry("elementwise.mul", Reads::Rows),
    entry("elementwise.mul_scalar", Reads::Rows),
    entry("elementwise.norm_modulate", Reads::Rows),
    entry("elementwise.ple_gate", Reads::Rows),
    entry("elementwise.relative_bucket_bias", Reads::Nothing),
    entry("elementwise.residual_add", Reads::Rows),
    entry("elementwise.residual_add_rmsnorm", Reads::Rows),
    entry("elementwise.rmsnorm", Reads::Rows),
    entry("elementwise.rmsnorm_gated", Reads::Rows),
    entry("elementwise.rmsnorm_gated_by", Reads::Rows),
    entry("elementwise.rmsnorm_grouped_plus_one", Reads::Rows),
    entry("elementwise.rmsnorm_no_scale", Reads::Rows),
    entry("elementwise.rmsnorm_per_head", Reads::Rows),
    entry("elementwise.rmsnorm_per_head_plus_one", Reads::Rows),
    entry("elementwise.rmsnorm_plus_one", Reads::Rows),
    entry("elementwise.rope_axes", Reads::Rows),
    entry("elementwise.rope_full", Reads::Rows),
    entry("elementwise.rope_mrope", Reads::Rows),
    entry("elementwise.rope_partial", Reads::Rows),
    entry("elementwise.rope_partial_last", Reads::Rows),
    entry("elementwise.rope_partial_q", Reads::Rows),
    entry("elementwise.rope_yarn", Reads::Rows),
    entry("elementwise.scale", Reads::Rows),
    entry("elementwise.silu", Reads::Rows),
    entry("elementwise.silu_scaled", Reads::Rows),
    entry("elementwise.sinusoid", Reads::Rows),
    entry("elementwise.tanh", Reads::Rows),
    entry("layout.embed", Reads::Rows),
    entry("layout.embed_concat", Reads::Rows),
    entry("layout.embed_weighted", Reads::Rows),
    entry("layout.pack_rows", Reads::Rows),
    entry("layout.scatter_live_rows", Reads::Rows),
    entry("layout.argmax", Reads::Rows),
    entry("layout.select", Reads::Rows),
    entry("layout.topk", Reads::Rows),
    entry("layout.split_q_gate", Reads::Rows),
    entry("layout.split_qkv", Reads::Rows),
    entry("layout.split_rows", Reads::Rows),
    entry("layout.unpack_rows", Reads::Rows),
    entry("linear.mlp_geglu_tanh", Reads::Rows),
    entry("linear.mlp_geglu_tanh_packed", Reads::Rows),
    entry("linear.mlp_gelu_tanh", Reads::Rows),
    entry("linear.mlp_situ", Reads::Rows),
    entry("linear.mlp_swiglu", Reads::Rows),
    entry("linear.mlp_swiglu_clamp", Reads::Rows),
    entry("linear.mlp_swiglu_clamp_alpha", Reads::Rows),
    entry("linear.moe_bias_sum", Reads::Rows),
    entry("linear.moe_hash_route", Reads::Rows),
    entry("linear.moe_matmul_select", Reads::Rows),
    entry("linear.moe_matmul_select_bias", Reads::Rows),
    entry("linear.moe_matmul_select_quant", Reads::Rows),
    entry("linear.moe_sigmoid_gate_add", Reads::Rows),
    entry("linear.moe_topk_sigmoid", Reads::Rows),
    entry("linear.moe_topk_sigmoid_sink", Reads::Rows),
    entry("linear.moe_topk_softmax", Reads::Rows),
    entry("linear.moe_topk_softmax_scaled", Reads::Rows),
    entry("linear.moe_topk_sqrt_softplus", Reads::Rows),
    entry("linear.moe_weighted_sum", Reads::Rows),
    entry("linear.rel_bias", Reads::Rows),
];

#[must_use]
pub fn reads(op: &str) -> Reads {
    ENTRIES
        .iter()
        .find(|entry| entry.name == op)
        .map_or(Reads::Nothing, |entry| entry.reads)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seat_every_case() {
        names_are_unique();
        lookup_reads_the_table();
    }

    fn names_are_unique() {
        for (i, a) in ENTRIES.iter().enumerate() {
            assert!(
                ENTRIES[i + 1..].iter().all(|b| b.name != a.name),
                "`{}` is declared twice",
                a.name
            );
        }
    }

    fn lookup_reads_the_table() {
        assert_eq!(reads("attention.ssm_kda_chunked"), Reads::RowsAndLanes);
        assert_eq!(reads("elementwise.rmsnorm"), Reads::Rows);
        assert_eq!(reads("linear.matmul"), Reads::Nothing);
    }
}
