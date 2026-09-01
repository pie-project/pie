use checkpoint::contract::{Expr, ModelContract, TensorType};
use model_dsl::{Dtype, Weight};

use super::model::{Gate, GateUp, Mlp, Model};
use checkpoint_dsl::{Builder, Error, encoding};

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, Error> {
        // **THE NATIVE DOOR, ASKED BEFORE THE WITNESS SNIFF** (§M-4a). A file
        // holding every plane this contract declares, under this contract's
        // names, is an artifact `pie model import` wrote out of this very
        // text, and [`Model::load`] is its reader: `read_own` throughout, no
        // transform at all. `load` failing is what says the file is foreign,
        // and it fails on the first plane it cannot find. The argument in full
        // is at `qwen_3::Model::import`.
        if let Ok(native) = self.load(src) {
            return Ok(native);
        }
        // **AND THE ARM IS CHOSEN BY BUILDING IT, NOT BY SNIFFING A NAME.**
        // The witness this used to look for — the embedding, spelled the way
        // each layout spells it — is one of the planes a promotion MOVES, so
        // an artifact this build wrote could satisfy neither door. The
        // argument in full, and the file it was measured on, is at
        // `qwen_3::Model::import`.
        //
        // **AND THE ORDER STILL MATTERS HERE, FOR THE SAME REASON IT DID.**
        // The flash artifact (mlx-community DeepSeek-V4-Flash) is the one that
        // ships the trunk hyper head and is otherwise huggingface-named, so
        // the mlx arm is tried BEFORE the deepseek-v3-style one that would
        // otherwise claim it. What changed is only how a reading is rejected:
        // by failing to build rather than by a missing name.
        let mut refusals: Vec<String> = Vec::new();
        let arms: [(&str, fn(&Self, &ztensor::Source) -> Result<ModelContract, Error>); 3] = [
            ("flash mlx", Self::import_from_mlx),
            ("huggingface", Self::import_from_huggingface),
            ("gguf", Self::import_from_gguf),
        ];
        for (what, arm) in arms {
            match arm(self, src) {
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

    /// **THE FLASH ARTIFACT'S OWN NAMES** (mlx-community DeepSeek-V4-Flash).
    ///
    /// One list of reads drives both this import and [`Model::mlx_source_names`],
    /// so the name census the bijection gate holds against cannot drift from the
    /// names this arm actually reads.
    pub fn import_from_mlx(&self, src: &ztensor::Source) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp);
        for read in self.mlx_reads() {
            match read {
                Read::One(w, name) => b.read(w, name)?,
                // **A PACKED BANK'S HALVES JOIN THROUGH THE VERB THAT KNOWS
                // ABOUT COMPANIONS.** `read_expr` states one expression over
                // one plane, which is the whole story for a bf16 fusion and
                // two thirds of it for an affine one: each half of a
                // quantized pair carries its own `.scales`/`.biases`, and
                // they join at the same seams the codes do. `read_concat`
                // says exactly that; the arm below is the dense reading.
                Read::Concat(w, _, names) if affine(w.dtype) => {
                    b.read_concat(w, names)?;
                }
                // A name the census counts and no contract states.
                Read::Named(_) => {}
                Read::Concat(w, axis, names) => {
                    let hidden = i64::from(self.hidden);
                    // Stacked MoE banks concat along the intermediate axis of a
                    // rank-3 `[experts, inter, hidden]` tensor; the shared /
                    // dense fusions concat two rank-2 `[inter, hidden]` halves.
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

    /// Every mlx source name this family's flash arm reads, in read order —
    /// the census the plan-to-checkpoint bijection is held against.
    #[must_use]
    pub fn mlx_source_names(&self) -> Vec<String> {
        self.mlx_reads()
            .into_iter()
            .flat_map(|r| match r {
                Read::One(_, name) | Read::Named(name) => vec![name],
                Read::Concat(_, _, names) => names,
            })
            .collect()
    }

    /// The same census with the WEIGHT beside each name — what the flash arm
    /// reads, and what it declares the bytes it reads as.
    ///
    /// [`mlx_source_names`](Model::mlx_source_names) is this list's name half,
    /// and this is the whole of it: a per-tensor quantization mix is a claim
    /// about the pairing, so the pairing is what a census has to be able to
    /// read. A fused bank appears once per stored part, because both parts land
    /// in the same declaration.
    #[must_use]
    pub fn mlx_planes(&self) -> Vec<(&Weight, String)> {
        self.mlx_reads()
            .into_iter()
            .flat_map(|r| match r {
                Read::One(w, name) => vec![(w, name)],
                Read::Concat(w, _, names) => names.into_iter().map(|n| (w, n)).collect(),
                // A name with no plane behind it has no pairing to report.
                Read::Named(_) => Vec::new(),
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
            let n = |s: &str| format!("model.layers.{l}.{s}");

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
            if let Some(pool) = &at.pool {
                if let Some(c) = &pool.compressor {
                    reads.push(Read::One(&c.wkv, n("attn.compressor.wkv.weight")));
                    reads.push(Read::One(&c.wgate, n("attn.compressor.wgate.weight")));
                    reads.push(Read::One(&c.ape, n("attn.compressor.ape")));
                    reads.push(Read::One(&c.norm, n("attn.compressor.norm.weight")));
                }
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
                reads.push(Read::One(&ix.compressor.ape, n("attn.indexer.compressor.ape")));
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
                    // **THE TABLE IS READ AND THE ROUTER IS THE NAME NOW.**
                    // `linear.moe_hash_route` landed, so `tid2eid` is a plane
                    // the forward fires and the contract states. Its layer's
                    // `ffn.gate.weight` is what the artifact still ships and
                    // this text no longer reads — a lookup gate computes no
                    // logits — so the census counts that name and the
                    // contract omits it, which is the sentence this pair used
                    // to say the other way round (see [`Read::Named`]).
                    Gate::Hash { tid2eid } => {
                        reads.push(Read::Named(n("ffn.gate.weight")));
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
                    // **THE UNFUSED PAIR IS TWO PLAIN READS**, which is the
                    // whole of what the form buys: each half is one stored
                    // triplet read at its OWN affine point, and nothing has to
                    // join two `.scales` rectangles that do not join.
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
                reads.push(Read::One(shared_down, n("ffn.shared_experts.down_proj.weight")));
            }
        }
        reads
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
                // The v3-style huggingface layout never names a flash MoE — the
                // flash artifact is the mlx one, read above.
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

/// One read of the flash arm: a plain plane, a fused bank concatenated from
/// several stored planes along `axis`, or a name this text does not read.
enum Read<'w> {
    One(&'w Weight, String),
    Concat(&'w Weight, u8, Vec<String>),
    /// **A NAME AND NOT A READ.**
    ///
    /// A load contract that PUBLISHES a plane no plan names is a bijection
    /// `Shell::load` refuses, by name, before a byte moves — while a name the
    /// artifact ships and the text does not read is simply a fact about the
    /// artifact. So this variant is how the census counts such a name without
    /// the contract stating it.
    ///
    /// **WHAT SITS HERE TODAY IS THE HASH LAYERS' ROUTER**
    /// (`ffn.gate.weight` for `layer < num_hash_layers`). Those layers route
    /// by `ffn.gate.tid2eid` — a lookup keyed by token identity — so they
    /// compute no logits and read no router. The snapshot ships the plane all
    /// the same, and a gate that pretended otherwise would be lying about the
    /// artifact.
    ///
    /// It used to be `tid2eid` sitting here, for the opposite reason: the
    /// table was read by the import and by nothing else, because
    /// `linear.moe_hash_route` had not landed. It has, so the two names
    /// traded places.
    Named(String),
}

/// Whether a bank stores MLX affine codes — the rows whose planes are a
/// `.weight`/`.scales`/`.biases` triplet rather than one rectangle.
fn affine(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::U4g64
            | Dtype::U8g64
            | Dtype::U4g32
            | Dtype::U2g32
            | Dtype::U2g64
            | Dtype::U2g128
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
