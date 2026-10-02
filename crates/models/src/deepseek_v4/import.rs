use checkpoint::contract::{Expr, ModelContract, TensorContract, TensorType};
use model_dsl::{Dtype, Weight};

use super::model::{Gate, GateUp, Layer, Mlp, Model};
use checkpoint_dsl::{Builder, Error, encoding, extents, scaling};
use model_dsl::Platform;

type ImportArm<S> = fn(&S, &ztensor::Source, Platform) -> Result<ModelContract, Error>;

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        let mut refusals: Vec<String> = Vec::new();
        let arms: &[(&str, ImportArm<Self>)] = if self.hyper.single_pass {
            &[
                ("a DeepSeek-V4.1 checkpoint", Self::import_from_v41),
                ("a DeepSeek-V4.1 MLX checkpoint", Self::import_from_v41_mlx),
            ]
        } else {
            &[
                (
                    "an artifact with an `--aux` overlay",
                    Self::import_from_own_with_aux,
                ),
                ("flash mlx", Self::import_from_mlx),
                ("huggingface", Self::import_from_huggingface),
                ("gguf", Self::import_from_gguf),
            ]
        };
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

    /// DeepSeek-V4.1-Flash in its per-expert release layout: the trunk
    /// under `layers.{l}.` with `embed.weight`, `head.weight` and `norm.weight`
    /// at the top, routed experts as per-expert `ffn.experts.{e}.w1/w3/w2`
    /// planes. Every plane is read at whatever representation the file
    /// stores — bf16, MLX affine codes, or MXFP4 codes beside e8m0 `.scale`
    /// exponents for the experts — into the row's declared one. Engram needs
    /// `engram.token_map`, the tokenizer-compressed id of every token, which
    /// `scripts/bench/engram_token_map.py` writes beside the checkpoint.
    pub fn import_from_v41(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        self.import_from_v41_layout(src, platform, false)
    }

    pub fn import_from_v41_mlx(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        if src
            .get("model.layers.0.ffn.switch_mlp.gate_proj.weight")
            .is_none()
        {
            return Err(Error::Illegible {
                name: "dsv4".to_string(),
                detail: "no stacked MLX V4.1 expert bank in model.layers".to_string(),
            });
        }
        self.import_from_v41_layout(src, platform, true)
    }

    fn import_from_v41_layout(
        &self,
        src: &ztensor::Source,
        platform: Platform,
        mlx: bool,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp, platform);
        b.read(
            &self.embed,
            if mlx {
                "model.embed_tokens.weight"
            } else {
                "embed.weight"
            },
        )?;
        if let Some(head) = &self.head {
            b.read(head, if mlx { "lm_head.weight" } else { "head.weight" })?;
        }
        b.read(
            &self.final_norm,
            if mlx {
                "model.norm.weight"
            } else {
                "norm.weight"
            },
        )?;
        if let Some(map) = &self.token_map {
            if src.get("engram.token_map").is_none() {
                return Err(Error::Illegible {
                    name: map.name.clone(),
                    detail: "Engram hashes tokenizer-compressed ids, and this checkpoint \
                             carries no `engram.token_map`; write it beside the weights \
                             with `scripts/bench/engram_token_map.py` first"
                        .to_string(),
                });
            }
            b.read(map, "engram.token_map")?;
        }

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| {
                if mlx {
                    format!("model.layers.{l}.{s}")
                } else {
                    format!("layers.{l}.{s}")
                }
            };
            let at = &w.attn;

            for (mix, tag) in [(&w.attn_mix, "attn"), (&w.mlp_mix, "ffn")] {
                let name = |plane: &str| {
                    if mlx {
                        n(&format!("{tag}_hc.{plane}"))
                    } else {
                        n(&format!("hc_{tag}_{plane}"))
                    }
                };
                b.read(&mix.scale, name("scale"))?;
                b.read(&mix.base, name("base"))?;
                if let Some(dynamic) = &mix.dynamic {
                    b.read(dynamic, name("fn"))?;
                }
            }
            if let Some(norm) = &w.attn_norm {
                b.read(norm, n("attn_norm.weight"))?;
            }
            if let Some(norm) = &w.mlp_norm {
                b.read(norm, n("ffn_norm.weight"))?;
            }

            b.read(&at.q_down, n("attn.wq_a.weight"))?;
            b.read(&at.q_norm, n("attn.q_norm.weight"))?;
            b.read(&at.q_up, n("attn.wq_b.weight"))?;
            b.read(&at.kv_down, n("attn.wkv.weight"))?;
            b.read(&at.kv_norm, n("attn.kv_norm.weight"))?;
            b.read(&at.o_down, n("attn.wo_a.weight"))?;
            b.read(&at.o_up, n("attn.wo_b.weight"))?;
            b.read(&at.sink, n("attn.attn_sink"))?;

            if let Some(p) = &at.pool
                && let Some(c) = &p.compressor
            {
                b.read(&c.wkv, n("attn.compressor.wkv.weight"))?;
                if let Some(wgate) = &c.wgate {
                    b.read(wgate, n("attn.compressor.wgate.weight"))?;
                }
                b.read(&c.norm, n("attn.compressor.norm.weight"))?;
            }
            if let Some(ix) = &at.indexer {
                b.read(&ix.wq_b, n("attn.indexer.wq_b.weight"))?;
                b.read(&ix.weights_proj, n("attn.indexer.weights_proj.weight"))?;
                if let Some(wk) = &ix.wk {
                    b.read(wk, n("attn.indexer.wk.weight"))?;
                }
                if let Some(k_norm) = &ix.k_norm {
                    b.read(k_norm, n("attn.indexer.k_norm.weight"))?;
                }
            }

            let Mlp::MoeFlash {
                router,
                gate,
                gate_up,
                down,
                shared_gate_up,
                shared_down,
                experts,
                ..
            } = &w.mlp
            else {
                return Err(Error::Illegible {
                    name: "dsv4".to_string(),
                    detail: "every V4.1 layer is a DeepSeekMoE block".to_string(),
                });
            };
            b.read(router, n("ffn.gate.weight"))?;
            match gate {
                Gate::Bias { bias } => b.read(
                    bias,
                    n(if mlx {
                        "ffn.gate.e_score_correction_bias"
                    } else {
                        "ffn.gate.bias"
                    }),
                )?,
                Gate::Hash { .. } => {
                    return Err(Error::Illegible {
                        name: "dsv4".to_string(),
                        detail: "V4.1 routes by bias-corrected scores on every layer".to_string(),
                    });
                }
            }
            let expert = |e: u32, leg: &str| n(&format!("ffn.experts.{e}.{leg}.weight"));
            match gate_up {
                GateUp::Split { gate, up } => {
                    if mlx {
                        b.read(gate, n("ffn.switch_mlp.gate_proj.weight"))?;
                        b.read(up, n("ffn.switch_mlp.up_proj.weight"))?;
                    } else {
                        read_bank(&mut b, src, gate, (0..*experts).map(|e| expert(e, "w1")))?;
                        read_bank(&mut b, src, up, (0..*experts).map(|e| expert(e, "w3")))?;
                    }
                }
                GateUp::Fused(_) => {
                    return Err(Error::Illegible {
                        name: "dsv4".to_string(),
                        detail: "V4.1 keeps its routed gate and up legs apart".to_string(),
                    });
                }
            }
            if mlx {
                b.read(down, n("ffn.switch_mlp.down_proj.weight"))?;
            } else {
                read_bank(&mut b, src, down, (0..*experts).map(|e| expert(e, "w2")))?;
            }
            b.read_concat(
                shared_gate_up,
                [
                    n(if mlx {
                        "ffn.shared_experts.gate_proj.weight"
                    } else {
                        "ffn.shared_experts.w1.weight"
                    }),
                    n(if mlx {
                        "ffn.shared_experts.up_proj.weight"
                    } else {
                        "ffn.shared_experts.w3.weight"
                    }),
                ],
            )?;
            b.read(
                shared_down,
                n(if mlx {
                    "ffn.shared_experts.down_proj.weight"
                } else {
                    "ffn.shared_experts.w2.weight"
                }),
            )?;

            if let Some(e) = &w.engram {
                b.read(&e.table, n("engram.embed.weight"))?;
                b.read(&e.wkv, n("engram.wkv.weight"))?;
                // The gate's normalisations carry `weight + 1`.
                b.read_over(&e.q_weight, n("engram.q_weight"), |x| x.bias(-1.0))?;
                b.read_over(&e.k_weight, n("engram.k_weight"), |x| x.bias(-1.0))?;
            }
        }

        Ok(b.build())
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

/// A routed bank `[experts, rows, cols]` from one plane per expert, at
/// whatever representation the file stores them: bf16 rows are stacked,
/// MLX affine trios are stacked by the affine reader, and MXFP4 codes
/// (`w.weight` u8 nibble pairs beside `w.scale` e8m0 exponents, as DeepSeek
/// stores its fp4 experts) are stacked with their exponents.
fn read_bank(
    b: &mut Builder<'_>,
    src: &ztensor::Source,
    w: &Weight,
    parts: impl Iterator<Item = String>,
) -> Result<(), Error> {
    let parts: Vec<String> = parts.collect();
    match w.dtype {
        Dtype::Mxfp4 => {
            let rows = i64::try_from(w.dim(1)).expect("a bank row count inside i64");
            let cols = i64::try_from(w.dim(2)).expect("a bank column count inside i64");
            let mut codes = Vec::with_capacity(parts.len());
            let mut scales = Vec::with_capacity(parts.len());
            for part in &parts {
                let stem = part.strip_suffix(".weight").unwrap_or(part);
                let scale = format!("{stem}.scale");
                if src.get(&scale).is_none() {
                    return Err(Error::Illegible {
                        name: w.name.clone(),
                        detail: format!(
                            "`{part}` is read as MXFP4 codes, whose exponents are stored \
                             beside it as `{scale}`, and the checkpoint holds none"
                        ),
                    });
                }
                codes.push(
                    Expr::src(part.clone())
                        .transmute(TensorType::new(vec![1, rows, cols], encoding(Dtype::Mxfp4))),
                );
                scales.push(Expr::src(scale).transmute(TensorType::new(
                    vec![1, rows, cols / 32],
                    encoding(Dtype::E8m0),
                )));
            }
            let pairing = scaling(w);
            let counted = checkpoint_dsl::divided(
                &extents(w),
                pairing.channel_axis,
                pairing.group_size,
                &w.name,
            );
            b.extend([
                TensorContract::inferred(
                    w.name.clone(),
                    Expr::concat(0, codes),
                    encoding(Dtype::Mxfp4),
                ),
                TensorContract::new(
                    model_dsl::scales_name(&w.name),
                    Expr::concat(0, scales),
                    counted,
                    encoding(Dtype::E8m0),
                )
                .scaling(pairing),
            ]);
            Ok(())
        }
        Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128 => b.read_stack(w, parts.into_iter().map(|part| vec![part])),
        _ => {
            let rows = w.dim(1);
            let cols = w.dim(2);
            b.read_expr(
                w,
                Expr::concat(
                    0,
                    parts
                        .into_iter()
                        .map(|part| one_bank_row(w.dtype, part, rows, cols))
                        .collect(),
                ),
            )
        }
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
        if let Some(wgate) = &c.wgate {
            reads.push(Read::One(wgate, n("attn.compressor.wgate.weight")));
        }
        if let Some(ape) = &c.ape {
            reads.push(Read::One(ape, n("attn.compressor.ape")));
        }
        reads.push(Read::One(&c.norm, n("attn.compressor.norm.weight")));
    }
    if let Some(ix) = &at.indexer {
        reads.push(Read::One(&ix.wq_b, n("attn.indexer.wq_b.weight")));
        reads.push(Read::One(
            &ix.weights_proj,
            n("attn.indexer.weights_proj.weight"),
        ));
        if let Some(c) = &ix.compressor {
            reads.push(Read::One(&c.wkv, n("attn.indexer.compressor.wkv.weight")));
            if let Some(wgate) = &c.wgate {
                reads.push(Read::One(wgate, n("attn.indexer.compressor.wgate.weight")));
            }
            if let Some(ape) = &c.ape {
                reads.push(Read::One(ape, n("attn.indexer.compressor.ape")));
            }
            reads.push(Read::One(&c.norm, n("attn.indexer.compressor.norm.weight")));
        }
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

#[cfg(test)]
mod tests {
    use std::fs::File;

    use ztensor::provide::{Catalog, Entry, Location, Store, StoreId};

    use super::*;
    use crate::deepseek_v4::model::Routed;

    #[test]
    fn single_pass_import_reads_stacked_mlx_experts() {
        let mut model = Model::flash41_mini(
            Dtype::Bf16,
            Routed::split(Dtype::Bf16),
            Dtype::Bf16,
            Dtype::Bf16,
            1,
        );
        model.layers.truncate(1);
        model.token_map = None;

        let path =
            std::env::temp_dir().join(format!("dsv41_mlx_import_{}.bin", std::process::id()));
        let mut catalog = Catalog::new();
        let mut offset = 0;
        for (weight, name) in model.mlx_planes() {
            let mut shape = weight.shape.clone();
            if name.contains("ffn.shared_experts.gate_proj.")
                || name.contains("ffn.shared_experts.up_proj.")
            {
                shape[0] /= 2;
            }
            let width = if weight.dtype == Dtype::F32 { 4 } else { 2 };
            let len = shape.iter().product::<u64>() * width;
            catalog.insert(
                name,
                Entry::leaf(
                    shape,
                    if weight.dtype == Dtype::F32 {
                        ztensor::Leaf::F32
                    } else {
                        ztensor::Leaf::BF16
                    },
                    Location {
                        store: StoreId(0),
                        offset,
                        len,
                    },
                ),
            );
            offset += len;
        }
        let file = File::create(&path).unwrap();
        file.set_len(offset).unwrap();
        drop(file);
        let store = Store::index(&path, "safetensors").unwrap();
        let src = ztensor::Source::from_parts(vec![store], catalog).unwrap();

        let contract = model.import(&src, Platform::Cuda).unwrap();
        assert!(contract.tensors.iter().any(|tensor| {
            tensor.name == model.layers[0].mlp_mix.base.name
                && matches!(&tensor.expr, Expr::Src(name) if name == "model.layers.0.ffn_hc.base")
        }));
        assert!(contract.tensors.iter().any(|tensor| {
            matches!(&tensor.expr, Expr::Src(name) if name == "model.layers.0.ffn.switch_mlp.down_proj.weight")
        }));
        std::fs::remove_file(path).unwrap();
    }
}
