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
    entry("attn.decode", Reads::RowsAndLanes),
    entry("fused.lse_sink", Reads::RowsAndLanes),
    entry("fused.pool_attention", Reads::Rows),
    entry("attn.prefill", Reads::RowsAndLanes),
    entry("attn.ragged", Reads::RowsAndLanes),
    entry("attn.plan_decode", Reads::Nothing),
    entry("attn.plan_prefill", Reads::Nothing),
    entry("attn.index_layernorm_rope", Reads::Rows),
    entry("attn.index_rope", Reads::Rows),
    entry("attn.index_block_mean", Reads::Rows),
    entry("attn.index_topk", Reads::Rows),
    entry("attn.kv_append", Reads::Rows),
    entry("attn.merge_lse", Reads::Rows),
    entry("attn.mla_decode", Reads::Rows),
    entry("attn.mla_latents", Reads::Rows),
    entry("attn.mla_latents_rope", Reads::Rows),
    entry("fused.mla_prefill_expanded", Reads::Rows),
    entry("attn.mla_prefill", Reads::Rows),
    entry("attn.mla_split_q_b", Reads::Rows),
    entry("gather.ngram_ids", Reads::Rows),
    entry("attn.pool_lse", Reads::Rows),
    entry("attn.pool_lse", Reads::Rows),
    entry("ssm.short_conv", Reads::RowsAndLanes),
    entry("attn.sink", Reads::Rows),
    entry("ssm.causal_conv1d", Reads::RowsAndLanes),
    entry("ssm.gated_delta", Reads::RowsAndLanes),
    entry("ssm.gdn_gates", Reads::Rows),
    entry("ssm.kda", Reads::Rows),
    entry("ssm.mamba2", Reads::Rows),
    entry("gather.ngram_ids_chunked", Reads::RowsAndLanes),
    entry("ssm.short_conv_chunked", Reads::RowsAndLanes),
    entry("ssm.causal_conv1d_chunked", Reads::RowsAndLanes),
    entry("ssm.gated_delta_chunked", Reads::RowsAndLanes),
    entry("ssm.kda_chunked", Reads::RowsAndLanes),
    entry("ssm.mamba2_chunked", Reads::RowsAndLanes),
    entry("elemwise.act", Reads::Rows),
    entry("elemwise.binary", Reads::Rows),
    entry("elemwise.add_bias", Reads::Rows),
    entry("elemwise.clamp", Reads::Rows),
    entry("elemwise.gate_sigmoid_mul", Reads::Rows),
    entry("elemwise.gate_sigmoid_mul_heads", Reads::Rows),
    entry("elemwise.gated_residual_add", Reads::Rows),
    entry("fused.gated_residual_norm_modulate", Reads::Rows),
    entry("elemwise.hc_expand", Reads::Rows),
    entry("elemwise.hc_fold", Reads::Rows),
    entry("elemwise.hc_gates", Reads::Rows),
    entry("elemwise.hc_inject", Reads::Rows),
    entry("elemwise.hc_mix", Reads::Rows),
    entry("elemwise.hc_rmsnorm_f32", Reads::Rows),
    entry("fused.hc_prenorm_project", Reads::Rows),
    entry("fused.hc_gates_norm", Reads::Rows),
    entry("fused.res_blend_rmsnorm", Reads::Rows),
    entry("fused.hc_pre", Reads::Rows),
    entry("elemwise.layernorm", Reads::Rows),
    entry("elemwise.norm_gate", Reads::Rows),
    entry("elemwise.modulate", Reads::Rows),
    entry("fused.norm_modulate", Reads::Rows),
    entry("elemwise.ple_gate", Reads::Rows),
    entry("elemwise.relative_bucket_bias", Reads::Nothing),
    entry("elemwise.residual_add", Reads::Rows),
    entry("fused.residual_add_rmsnorm", Reads::Rows),
    entry("elemwise.rmsnorm", Reads::Rows),
    entry("elemwise.rmsnorm_gated", Reads::Rows),
    entry("elemwise.rmsnorm_gated_by", Reads::Rows),
    entry("elemwise.rope", Reads::Rows),
    entry("elemwise.rope_axes", Reads::Rows),
    entry("fused.rope_yarn_append", Reads::Rows),
    entry("elemwise.sinusoid", Reads::Rows),
    entry("gather.embed", Reads::Rows),
    entry("gather.pack_rows", Reads::Rows),
    entry("gather.scatter_live_rows", Reads::Rows),
    entry("gather.select", Reads::Rows),
    entry("gather.split_q_gate", Reads::Rows),
    entry("gather.split_qkv", Reads::Rows),
    entry("gather.split_rows", Reads::Rows),
    entry("gather.unpack_rows", Reads::Rows),
    entry("sampling.argmax", Reads::Rows),
    entry("sampling.top_k", Reads::Rows),
    entry("linear.glu", Reads::Rows),
    entry("linear.gelu_tanh", Reads::Rows),
    entry("linear.moe_combine", Reads::Rows),
    entry("linear.moe_route", Reads::Rows),
    entry("linear.moe_select", Reads::Rows),
    entry("linear.relu2", Reads::Rows),
    entry("fused.mlp_swiglu", Reads::Rows),
    entry("linear.moe_bias_sum", Reads::Rows),
    entry("fused.moe_experts", Reads::Rows),
    entry("linear.lora_correct", Reads::Rows),
    entry("linear.moe_sigmoid_gate_add", Reads::Rows),
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
        assert_eq!(reads("ssm.kda_chunked"), Reads::RowsAndLanes);
        assert_eq!(reads("elemwise.rmsnorm"), Reads::Rows);
        assert_eq!(reads("linear.matmul"), Reads::Nothing);
    }
}
