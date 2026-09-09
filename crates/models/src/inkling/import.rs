use checkpoint::contract::{Expr, ModelContract, TensorType};
use checkpoint_dsl::{Builder, Error, extents};
use model_dsl::Platform;

use super::model::{Mlp, Model};

const TRUNK: &str = "model.llm.";

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        self.import_from_huggingface(src, platform)
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let at = |leaf: &str| format!("{TRUNK}{leaf}");
        let mut b = Builder::new(src, self.tp, platform);
        b.read(&self.embed, at("embed.weight"))?;
        b.read(&self.embed_norm, at("embed_norm.weight"))?;
        b.read(&self.final_norm, at("norm.weight"))?;
        b.read_expr(
            &self.unembed,
            Expr::src(at("unembed.weight")).slice(0, 0, i64::from(self.head_rows)),
        )?;

        for (l, w) in self.layers.iter().enumerate() {
            let n = |leaf: &str| format!("{TRUNK}layers.{l}.{leaf}");
            b.read(&w.attn_norm, n("attn_norm.weight"))?;
            b.read(&w.mlp_norm, n("mlp_norm.weight"))?;
            b.read(&w.q_proj, n("attn.wq_du.weight"))?;
            b.read(&w.k_proj, n("attn.wk_dv.weight"))?;
            b.read(&w.v_proj, n("attn.wv_dv.weight"))?;
            b.read(&w.r_proj, n("attn.wr_du.weight"))?;
            b.read(&w.o_proj, n("attn.wo_ud.weight"))?;
            b.read(&w.q_norm, n("attn.q_norm.weight"))?;
            b.read(&w.k_norm, n("attn.k_norm.weight"))?;
            b.read(&w.rel_proj, n("attn.rel_logits_proj.proj"))?;
            for (weight, leaf) in [
                (&w.k_conv, "attn.k_sconv.weight"),
                (&w.v_conv, "attn.v_sconv.weight"),
                (&w.attn_conv, "attn_sconv.weight"),
                (&w.mlp_conv, "mlp_sconv.weight"),
            ] {
                b.read_expr(weight, flattened(src, n(leaf), extents(weight))?)?;
            }
            match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    inter,
                    down,
                    scale,
                } => {
                    let inter = i64::from(*inter);
                    let w13 = Expr::src(n("mlp.w13_dn.weight"));
                    b.read_expr(
                        gate_up,
                        Expr::concat(
                            0,
                            vec![
                                w13.clone().stride(0, 0, inter, 2),
                                w13.stride(0, 1, inter, 2),
                            ],
                        ),
                    )?;
                    b.read(down, n("mlp.w2_md.weight"))?;
                    b.read(scale, n("mlp.global_scale"))?;
                }
                Mlp::Routed {
                    router,
                    bias,
                    scale,
                    gate_up,
                    down,
                    inter,
                    ..
                } => {
                    b.read(router, n("mlp.gate.weight"))?;
                    b.read(bias, n("mlp.gate.bias"))?;
                    b.read(scale, n("mlp.gate.global_scale"))?;
                    let inter = i64::from(*inter);
                    let combed = |name: String| {
                        let w13 = Expr::src(name);
                        Expr::concat(
                            1,
                            vec![
                                w13.clone().stride(1, 0, inter, 2),
                                w13.stride(1, 1, inter, 2),
                            ],
                        )
                    };
                    b.read_expr(
                        gate_up,
                        Expr::concat(
                            0,
                            vec![
                                combed(n("mlp.experts.w13_weight")),
                                combed(n("mlp.shared_experts.shared_w13_weight")),
                            ],
                        ),
                    )?;
                    b.read_expr(
                        down,
                        Expr::concat(
                            0,
                            vec![
                                Expr::src(n("mlp.experts.w2_weight")),
                                Expr::src(n("mlp.shared_experts.shared_w2_weight")),
                            ],
                        ),
                    )?;
                }
            }
        }

        Ok(b.build())
    }
}

fn flattened(src: &ztensor::Source, from: String, want: Vec<i64>) -> Result<Expr, Error> {
    let Some(tensor) = src.get(&from) else {
        return Err(Error::Missing(from));
    };
    let illegible = |why: &dyn std::fmt::Display| Error::Illegible {
        name: from.clone(),
        detail: why.to_string(),
    };
    let shape = tensor.shape();
    let stored: i128 = shape.iter().map(|&n| i128::from(n)).product();
    let asked: i128 = want.iter().map(|&n| i128::from(n)).product();
    if stored != asked {
        return Err(illegible(&format!(
            "is stored {shape:?} ({stored} elements) and the plan reads it as \
             {want:?} ({asked} elements)"
        )));
    }
    let encoding = checkpoint::file::encoding_of(&tensor).map_err(|why| illegible(&why))?;
    Ok(Expr::src(from).transmute(TensorType::new(want, encoding)))
}
