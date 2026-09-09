use checkpoint::contract::{Expr, ModelContract, TensorType};
use model_dsl::{Dtype, Weight};

use super::model::{Gate, GateUp, Layer, Mlp, Model};
use checkpoint_dsl::{Builder, Error, encoding};
use model_dsl::Platform;

type ImportArm<S> = fn(&S, &ztensor::Source, Platform) -> Result<ModelContract, Error>;

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let mut refusals: Vec<String> = Vec::new();
        let arms: [(&str, ImportArm<Self>); 4] = [
            (
                "an artifact with an `--aux` overlay",
                Self::import_from_own_with_aux,
            ),
            ("flash mlx", Self::import_from_mlx),
            ("huggingface", Self::import_from_huggingface),
            ("gguf", Self::import_from_gguf),
        ];
        for (what, arm) in arms {
            match arm(self, src, platform) {
                Ok(contract) => return Ok(contract),
                Err(why) => refusals.push(format!("as {what}, {why}")),
            }
        }
        Err(Error::Illegible {
            name: "dsv4".to_string(),
            detail: format!(
                "no reading of this file lands every plane this family \
                 declares — {}",
                refusals.join("; "),
            ),
        })
    }

    pub fn import_from_own_with_aux(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        if self.mtp.is_none() {
            return Err(Error::Illegible {
                name: "mtp".to_string(),
                detail: "this row declares no draft head, so there is no overlay to land \
                         on an artifact"
                    .to_string(),
            });
        }
        let is_aux = |name: &str| name.starts_with("aux.");
        let mut b = Builder::new(src, self.tp, platform);
        for read in self.mlx_reads() {
            match read {
                Read::One(w, name) if is_aux(&name) => b.read(w, name)?,
                Read::Concat(w, _, names) if names.iter().all(|n| is_aux(n)) && affine(w.dtype) => {
                    b.read_concat(w, names)?;
                }
                Read::Concat(w, axis, names) if names.iter().all(|n| is_aux(n)) => {
                    let hidden = i64::from(self.hidden);
                    let parts = names
                        .into_iter()
                        .map(|name| {
                            if w.shape.len() == 3 {
                                slab(Expr::src(name), vec![-1, -1, hidden], encoding(w.dtype))
                            } else {
                                Expr::src(name)
                            }
                        })
                        .collect();
                    b.read_expr(w, Expr::concat(axis, parts))?;
                }
                Read::One(w, _) | Read::Concat(w, _, _) => b.read_own(w)?,
            }
        }
        Ok(b.build())
    }

    pub fn import_from_mlx(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp, platform);
        for read in self.mlx_reads() {
            match read {
                Read::One(w, name) => b.read(w, name)?,
                Read::Concat(w, _, names) if affine(w.dtype) => {
                    b.read_concat(w, names)?;
                }
                Read::Concat(w, axis, names) => {
                    let hidden = i64::from(self.hidden);
                    let parts = names
                        .into_iter()
                        .map(|name| {
                            if w.shape.len() == 3 {
                                slab(Expr::src(name), vec![-1, -1, hidden], encoding(w.dtype))
                            } else {
                                Expr::src(name)
                            }
                        })
                        .collect();
                    b.read_expr(w, Expr::concat(axis, parts))?;
                }
            }
        }
        Ok(b.build())
    }

    #[must_use]
    pub fn mlx_source_names(&self) -> Vec<String> {
        self.mlx_reads()
            .into_iter()
            .flat_map(|r| match r {
                Read::One(_, name) => vec![name],
                Read::Concat(_, _, names) => names,
            })
            .collect()
    }

    #[must_use]
    pub fn mlx_planes(&self) -> Vec<(&Weight, String)> {
        self.mlx_reads()
            .into_iter()
            .flat_map(|r| match r {
                Read::One(w, name) => vec![(w, name)],
                Read::Concat(w, _, names) => names.into_iter().map(|n| (w, n)).collect(),
            })
            .collect()
    }

    fn mlx_reads(&self) -> Vec<Read<'_>> {
        let mut reads = Vec::new();
        reads.push(Read::One(&self.embed, "model.embed_tokens.weight".into()));
        if let Some(head) = &self.head {
            reads.push(Read::One(head, "lm_head.weight".into()));
        }
        reads.push(Read::One(&self.final_norm, "model.norm.weight".into()));
        if let Some(hc) = &self.hc_head {
            reads.push(Read::One(&hc.base, "model.hc_head.base".into()));
            reads.push(Read::One(&hc.dynamic, "model.hc_head.fn".into()));
            reads.push(Read::One(&hc.scale, "model.hc_head.scale".into()));
        }

        for (l, w) in self.layers.iter().enumerate() {
            layer_reads(w, &|s: &str| format!("model.layers.{l}.{s}"), &mut reads);
        }
        if let Some(mtp) = &self.mtp {
            reads.push(Read::One(&mtp.enorm, "aux.enorm.weight".into()));
            reads.push(Read::One(&mtp.hnorm, "aux.hnorm.weight".into()));
            reads.push(Read::One(&mtp.e_proj, "aux.e_proj.weight".into()));
            reads.push(Read::Concat(
                &mtp.h_proj,
                0,
                (0..self.hyper.streams)
                    .map(|_| "aux.h_proj.weight".to_string())
                    .collect(),
            ));
            layer_reads(
                &mtp.block,
                &|s: &str| format!("aux.decoder.{s}"),
                &mut reads,
            );
            reads.push(Read::One(&mtp.hc_head.base, "aux.hc_head.base".into()));
            reads.push(Read::One(&mtp.hc_head.dynamic, "aux.hc_head.fn".into()));
            reads.push(Read::One(&mtp.hc_head.scale, "aux.hc_head.scale".into()));
            reads.push(Read::One(&mtp.norm, "aux.norm.weight".into()));
        }
        reads
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        if self.mtp.is_some() {
            return Err(Error::Illegible {
                name: "mtp".to_string(),
                detail: "this SKU declares a draft head, which only the flash mlx reading \
                         (with an `--aux` overlay) lands"
                    .to_string(),
            });
        }
        let mut b = Builder::new(src, self.tp, platform);
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
                    b.read_concat(
                        gate_up,
                        [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
                    )?;
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
                Mlp::MoeFlash { .. } => {
                    return Err(Error::Illegible {
                        name: "dsv4".to_string(),
                        detail: "a flash SKU cannot read the deepseek-v3 huggingface layout; \
                                 its artifact is the mlx one (`model.hc_head.base`)"
                            .to_string(),
                    });
                }
            }
        }

        Ok(b.build())
    }

    pub fn import_from_gguf(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        if self.mtp.is_some() {
            return Err(Error::Illegible {
                name: "mtp".to_string(),
                detail: "this SKU declares a draft head and no gguf spelling of one is settled"
                    .to_string(),
            });
        }
        let mut b = Builder::new(src, self.tp, platform);
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

                    b.read_concat(
                        gate_up,
                        [n("ffn_gate_exps.weight"), n("ffn_up_exps.weight")],
                    )?;

                    b.read(down, n("ffn_down_exps.weight"))?;
                }
                Mlp::MoeFlash { .. } => {
                    return Err(Error::Illegible {
                        name: "dsv4".to_string(),
                        detail: "a flash SKU has no gguf layout".to_string(),
                    });
                }
            }
        }

        Ok(b.build())
    }
}

enum Read<'w> {
    One(&'w Weight, String),
    Concat(&'w Weight, u8, Vec<String>),
}

fn layer_reads<'w>(w: &'w Layer, n: &dyn Fn(&str) -> String, reads: &mut Vec<Read<'w>>) {
    for (mix, tag) in [(&w.attn_mix, "attn_hc"), (&w.mlp_mix, "ffn_hc")] {
        reads.push(Read::One(&mix.scale, n(&format!("{tag}.scale"))));
        reads.push(Read::One(&mix.base, n(&format!("{tag}.base"))));
        if let Some(dynamic) = &mix.dynamic {
            reads.push(Read::One(dynamic, n(&format!("{tag}.fn"))));
        }
    }
    if let Some(norm) = &w.attn_norm {
        reads.push(Read::One(norm, n("attn_norm.weight")));
    }
    if let Some(norm) = &w.mlp_norm {
        reads.push(Read::One(norm, n("ffn_norm.weight")));
    }

    let at = &w.attn;
    reads.push(Read::One(&at.q_down, n("attn.wq_a.weight")));
    reads.push(Read::One(&at.q_norm, n("attn.q_norm.weight")));
    reads.push(Read::One(&at.q_up, n("attn.wq_b.weight")));
    reads.push(Read::One(&at.kv_down, n("attn.wkv.weight")));
    reads.push(Read::One(&at.kv_norm, n("attn.kv_norm.weight")));
    reads.push(Read::One(&at.o_down, n("attn.wo_a.weight")));
    reads.push(Read::One(&at.o_up, n("attn.wo_b.weight")));
    reads.push(Read::One(&at.sink, n("attn.attn_sink")));
    if let Some(pool) = &at.pool
        && let Some(c) = &pool.compressor
    {
        reads.push(Read::One(&c.wkv, n("attn.compressor.wkv.weight")));
        reads.push(Read::One(&c.wgate, n("attn.compressor.wgate.weight")));
        reads.push(Read::One(&c.ape, n("attn.compressor.ape")));
        reads.push(Read::One(&c.norm, n("attn.compressor.norm.weight")));
    }
    if let Some(ix) = &at.indexer {
        reads.push(Read::One(&ix.wq_b, n("attn.indexer.wq_b.weight")));
        reads.push(Read::One(
            &ix.weights_proj,
            n("attn.indexer.weights_proj.weight"),
        ));
        reads.push(Read::One(
            &ix.compressor.wkv,
            n("attn.indexer.compressor.wkv.weight"),
        ));
        reads.push(Read::One(
            &ix.compressor.wgate,
            n("attn.indexer.compressor.wgate.weight"),
        ));
        reads.push(Read::One(
            &ix.compressor.ape,
            n("attn.indexer.compressor.ape"),
        ));
        reads.push(Read::One(
            &ix.compressor.norm,
            n("attn.indexer.compressor.norm.weight"),
        ));
    }

    if let Mlp::MoeFlash {
        router,
        gate,
        gate_up,
        down,
        shared_gate_up,
        shared_down,
        ..
    } = &w.mlp
    {
        match gate {
            Gate::Hash { tid2eid } => {
                reads.push(Read::One(router, n("ffn.gate.weight")));
                reads.push(Read::One(tid2eid, n("ffn.gate.tid2eid")));
            }
            Gate::Bias { bias } => {
                reads.push(Read::One(router, n("ffn.gate.weight")));
                reads.push(Read::One(bias, n("ffn.gate.e_score_correction_bias")));
            }
        }
        match gate_up {
            GateUp::Fused(bank) => reads.push(Read::Concat(
                bank,
                1,
                vec![
                    n("ffn.switch_mlp.gate_proj.weight"),
                    n("ffn.switch_mlp.up_proj.weight"),
                ],
            )),
            GateUp::Split { gate, up } => {
                reads.push(Read::One(gate, n("ffn.switch_mlp.gate_proj.weight")));
                reads.push(Read::One(up, n("ffn.switch_mlp.up_proj.weight")));
            }
        }
        reads.push(Read::One(down, n("ffn.switch_mlp.down_proj.weight")));
        reads.push(Read::Concat(
            shared_gate_up,
            0,
            vec![
                n("ffn.shared_experts.gate_proj.weight"),
                n("ffn.shared_experts.up_proj.weight"),
            ],
        ));
        reads.push(Read::One(
            shared_down,
            n("ffn.shared_experts.down_proj.weight"),
        ));
    }
}

fn affine(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::U4g64 | Dtype::U8g64 | Dtype::U4g32 | Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128
    )
}

fn one_bank_row(dtype: Dtype, from: String, rows: u64, cols: u64) -> Expr {
    let extent = |e: u64| i64::try_from(e).expect("an extent no i64 holds");
    Expr::src(from).transmute(TensorType::new(
        vec![1, extent(rows), extent(cols)],
        encoding(dtype),
    ))
}

fn slab(expr: Expr, shape: Vec<i64>, encoding: checkpoint::types::Encoding) -> Expr {
    expr.transmute(TensorType::new(shape, encoding))
}
