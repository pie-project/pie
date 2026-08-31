use checkpoint::contract::{Expr, ModelContract, TensorType};
use model_dsl::Dtype;

use super::model::{Mlp, Model};
use checkpoint_dsl::{Builder, Error, encoding};

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, Error> {
        if src.get("model.embed_tokens.weight").is_some() {
            return self.import_from_huggingface(src);
        }
        if src.get("token_embd.weight").is_some() {
            return self.import_from_gguf(src);
        }
        Err(Error::Illegible {
            name: "dsv4".to_string(),
            detail: "neither `model.embed_tokens.weight` (huggingface) nor \
                     `token_embd.weight` (gguf) names a tensor here"
                .to_string(),
        })
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp);
        b.read(&self.embed, "model.embed_tokens.weight")?;
        b.read(&self.final_norm, "model.norm.weight")?;

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("model.layers.{l}.{s}");
            let at = &w.attn;

            b.read(&w.attn_mix.scale, n("hc_attn_scale"))?;
            b.read(&w.attn_mix.base, n("hc_attn_base"))?;
            b.read(&w.mlp_mix.scale, n("hc_mlp_scale"))?;
            b.read(&w.mlp_mix.base, n("hc_mlp_base"))?;

            b.read(&at.q_down, n("self_attn.q_a_proj.weight"))?;
            b.read(&at.q_norm, n("self_attn.q_a_layernorm.weight"))?;
            b.read(&at.q_up, n("self_attn.q_b_proj.weight"))?;
            b.read(&at.kv_down, n("self_attn.kv_a_proj_with_mqa.weight"))?;
            b.read(&at.kv_norm, n("self_attn.kv_a_layernorm.weight"))?;
            b.read(&at.o_down, n("self_attn.o_a_proj.weight"))?;
            b.read(&at.o_up, n("self_attn.o_b_proj.weight"))?;

            b.read(&at.sink, n("self_attn.sinks"))?;

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    b.read_concat(gate_up, [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")])?;
                    b.read(down, n("mlp.down_proj.weight"))?;
                }
                Mlp::Routed {
                    router,
                    bias,
                    gate_up,
                    down,
                    experts,
                    ..
                } => {
                    b.read(router, n("mlp.gate.weight"))?;

                    b.read(bias, n("mlp.gate.e_score_correction_bias"))?;

                    let inter = gate_up.dim(1) / 2;
                    let hidden = gate_up.dim(2);
                    let pair = |e: u32| {
                        let leg = |half: &str| {
                            one_bank_row(
                                gate_up.dtype,
                                n(&format!("mlp.experts.{e}.{half}.weight")),
                                inter,
                                hidden,
                            )
                        };
                        Expr::concat(1, vec![leg("gate_proj"), leg("up_proj")])
                    };
                    b.read_expr(gate_up, Expr::concat(0, (0..*experts).map(pair).collect()))?;

                    let slab = |e: u32| {
                        one_bank_row(
                            down.dtype,
                            n(&format!("mlp.experts.{e}.down_proj.weight")),
                            down.dim(1),
                            down.dim(2),
                        )
                    };
                    b.read_expr(down, Expr::concat(0, (0..*experts).map(slab).collect()))?;
                }
            }
        }

        Ok(b.build())
    }

    pub fn import_from_gguf(&self, src: &ztensor::Source) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp);
        b.read(&self.embed, "token_embd.weight")?;
        b.read(&self.final_norm, "output_norm.weight")?;

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("blk.{l}.{s}");
            let at = &w.attn;

            b.read(&w.attn_mix.scale, n("hc_attn_scale.weight"))?;
            b.read(&w.attn_mix.base, n("hc_attn_base.weight"))?;
            b.read(&w.mlp_mix.scale, n("hc_mlp_scale.weight"))?;
            b.read(&w.mlp_mix.base, n("hc_mlp_base.weight"))?;

            b.read(&at.q_down, n("attn_q_a.weight"))?;
            b.read(&at.q_norm, n("attn_q_a_norm.weight"))?;
            b.read(&at.q_up, n("attn_q_b.weight"))?;
            b.read(&at.kv_down, n("attn_kv_a_mqa.weight"))?;
            b.read(&at.kv_norm, n("attn_kv_a_norm.weight"))?;
            b.read(&at.o_down, n("attn_o_a.weight"))?;
            b.read(&at.o_up, n("attn_o_b.weight"))?;

            b.read(&at.sink, n("attn_sinks"))?;

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    b.read_concat(gate_up, [n("ffn_gate.weight"), n("ffn_up.weight")])?;
                    b.read(down, n("ffn_down.weight"))?;
                }
                Mlp::Routed {
                    router,
                    bias,
                    gate_up,
                    down,
                    ..
                } => {
                    b.read(router, n("ffn_gate_inp.weight"))?;

                    b.read(bias, n("exp_probs_b.bias"))?;

                    b.read_concat(gate_up, [n("ffn_gate_exps.weight"), n("ffn_up_exps.weight")])?;

                    b.read(down, n("ffn_down_exps.weight"))?;
                }
            }
        }

        Ok(b.build())
    }
}

fn one_bank_row(dtype: Dtype, from: String, rows: u64, cols: u64) -> Expr {
    let extent = |e: u64| i64::try_from(e).expect("an extent no i64 holds");
    Expr::src(from).transmute(TensorType::new(
        vec![1, extent(rows), extent(cols)],
        encoding(dtype),
    ))
}
