use model_loader::contract::{Expr, ModelContract, TensorType};

use super::model::{Head, Mixer, Mlp, Model};
use crate::contract::{ALIGNMENT, ModelError, copy, declare, extents, fused};

impl Model {
    pub fn import(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        assert!(
            self.tp == 1,
            "an import states the whole checkpoint; build the model at tp = 1"
        );
        let huggingface = "model.language_model.embed_tokens.weight";
        let gguf = "token_embd.weight";
        if src.get(huggingface).is_some() {
            return self.import_from_huggingface(src);
        }
        if src.get(gguf).is_some() {
            return self.import_from_gguf(src);
        }
        Err(ModelError::Illegible {
            name: "qwen_3".to_string(),
            detail: format!(
                "it holds neither `{huggingface}` nor `{gguf}`, so it is written \
                 in neither format this family reads"
            ),
        })
    }

    pub fn import_from_huggingface(
        &self,
        src: &ztensor::Source,
    ) -> Result<ModelContract, ModelError> {
        let mut tensors = vec![
            copy(src, &self.embed, "model.language_model.embed_tokens.weight")?,
            copy(src, &self.final_norm, "model.language_model.norm.weight")?,
        ];

        if let Head::Bank(head) = &self.head {
            tensors.push(copy(src, head, "lm_head.weight")?);
        }

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("model.language_model.layers.{l}.{s}");

            tensors.push(copy(src, &w.mixer_norm, n("input_layernorm.weight"))?);
            tensors.push(copy(
                src,
                &w.mlp_norm,
                n("post_attention_layernorm.weight"),
            )?);

            match &w.mixer {
                Mixer::Attn(a) => {
                    tensors.push(copy(src, &a.qg_proj, n("self_attn.q_proj.weight"))?);
                    tensors.push(copy(src, &a.k_proj, n("self_attn.k_proj.weight"))?);
                    tensors.push(copy(src, &a.v_proj, n("self_attn.v_proj.weight"))?);
                    tensors.push(copy(src, &a.o_proj, n("self_attn.o_proj.weight"))?);
                    tensors.push(copy(src, &a.q_norm, n("self_attn.q_norm.weight"))?);
                    tensors.push(copy(src, &a.k_norm, n("self_attn.k_norm.weight"))?);
                }
                Mixer::Gdn(g) => {
                    tensors.push(fused(
                        src,
                        &g.in_qkvz,
                        [
                            n("linear_attn.in_proj_qkv.weight"),
                            n("linear_attn.in_proj_z.weight"),
                        ],
                    )?);

                    tensors.push(fused(
                        src,
                        &g.in_ba,
                        [
                            n("linear_attn.in_proj_b.weight"),
                            n("linear_attn.in_proj_a.weight"),
                        ],
                    )?);

                    tensors.push(declare(
                        src,
                        &g.conv,
                        squeezed(src, n("linear_attn.conv1d.weight"))?,
                    )?);

                    tensors.push(copy(src, &g.dt_bias, n("linear_attn.dt_bias"))?);
                    tensors.push(copy(src, &g.a_log, n("linear_attn.A_log"))?);
                    tensors.push(copy(src, &g.norm, n("linear_attn.norm.weight"))?);
                    tensors.push(copy(src, &g.out_proj, n("linear_attn.out_proj.weight"))?);
                }
            }

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    tensors.push(fused(
                        src,
                        gate_up,
                        [n("mlp.gate_proj.weight"), n("mlp.up_proj.weight")],
                    )?);
                    tensors.push(copy(src, down, n("mlp.down_proj.weight"))?);
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared_gate_up,
                    shared_down,
                    shared_gate,
                    ..
                } => {
                    tensors.push(copy(src, router, n("mlp.gate.weight"))?);

                    tensors.push(copy(src, gate_up, n("mlp.experts.gate_up_proj"))?);
                    tensors.push(copy(src, down, n("mlp.experts.down_proj"))?);
                    tensors.push(fused(
                        src,
                        shared_gate_up,
                        [
                            n("mlp.shared_expert.gate_proj.weight"),
                            n("mlp.shared_expert.up_proj.weight"),
                        ],
                    )?);
                    tensors.push(copy(
                        src,
                        shared_down,
                        n("mlp.shared_expert.down_proj.weight"),
                    )?);
                    tensors.push(copy(src, shared_gate, n("mlp.shared_expert_gate.weight"))?);
                }
            }
        }

        // **THE DRAFT HEAD, VERIFIED BYTE-FOR-BYTE AGAINST THE CACHED
        // CHECKPOINT INDEX.** Fifteen tensors under `mtp.*` in
        // `models--Qwen--Qwen3.6-27B` at snapshot `6a9e13bd`, read out of
        // `model.safetensors.index.json` and the shard headers it names:
        //
        // ```text
        // mtp.fc.weight                                 BF16 [5120, 10240]
        // mtp.pre_fc_norm_embedding.weight              BF16 [5120]
        // mtp.pre_fc_norm_hidden.weight                 BF16 [5120]
        // mtp.layers.0.input_layernorm.weight           BF16 [5120]
        // mtp.layers.0.self_attn.q_proj.weight          BF16 [12288, 5120]
        // mtp.layers.0.self_attn.k_proj.weight          BF16 [1024, 5120]
        // mtp.layers.0.self_attn.v_proj.weight          BF16 [1024, 5120]
        // mtp.layers.0.self_attn.o_proj.weight          BF16 [5120, 6144]
        // mtp.layers.0.self_attn.q_norm.weight          BF16 [256]
        // mtp.layers.0.self_attn.k_norm.weight          BF16 [256]
        // mtp.layers.0.post_attention_layernorm.weight  BF16 [5120]
        // mtp.layers.0.mlp.gate_proj.weight             BF16 [17408, 5120]
        // mtp.layers.0.mlp.up_proj.weight               BF16 [17408, 5120]
        // mtp.layers.0.mlp.down_proj.weight             BF16 [5120, 17408]
        // mtp.norm.weight                               BF16 [5120]
        // ```
        //
        // Every attention and mlp shape is a trunk attention layer's, tensor
        // for tensor (compare `model.language_model.layers.3.self_attn.*` and
        // `layers.0.mlp.*`), which is what makes `Mtp` reuse `gated_attn` and
        // `dense_mlp` rather than restate them.
        //
        // **NO `mtp.lm_head` AND NO `mtp.embed_tokens`**, and the config says
        // so before the index does: `mtp_use_dedicated_embeddings: false`. The
        // draft readout goes through `lm_head.weight` and the draft embedding
        // through `model.language_model.embed_tokens.weight` — the base
        // planes, already claimed above, interned once by the recorder and
        // read by both heads.
        if let Some(mtp) = &self.mtp {
            tensors.push(copy(
                src,
                &mtp.pre_fc_norm_embedding,
                "mtp.pre_fc_norm_embedding.weight",
            )?);
            tensors.push(copy(
                src,
                &mtp.pre_fc_norm_hidden,
                "mtp.pre_fc_norm_hidden.weight",
            )?);

            // THE ONE STORED BANK, CUT AT ITS OWN SEAM. `mtp.fc.weight` is
            // `[hidden, 2·hidden]` and multiplies `[normed embedding | normed
            // hidden]`, so its columns `0..hidden` are the embedding half and
            // `hidden..2·hidden` the hidden half — the order dev concatenates
            // in (`launch_concat_bf16_rows(ws.q, ws.y, ...)`, `ws.q` holding
            // `rms(embed(tok))` and `ws.y` holding `rms(hidden)`). The slice
            // is the whole of the claim: no cast, no transpose, two contiguous
            // column bands of one tensor.
            let half = extents(&mtp.fc_embed)[1];
            tensors.push(declare(
                src,
                &mtp.fc_embed,
                Expr::src("mtp.fc.weight").slice(1, 0, half),
            )?);
            tensors.push(declare(
                src,
                &mtp.fc_hidden,
                Expr::src("mtp.fc.weight").slice(1, half, half),
            )?);

            let a = &mtp.attn;
            tensors.push(copy(
                src,
                &mtp.mixer_norm,
                "mtp.layers.0.input_layernorm.weight",
            )?);
            tensors.push(copy(
                src,
                &a.qg_proj,
                "mtp.layers.0.self_attn.q_proj.weight",
            )?);
            tensors.push(copy(
                src,
                &a.k_proj,
                "mtp.layers.0.self_attn.k_proj.weight",
            )?);
            tensors.push(copy(
                src,
                &a.v_proj,
                "mtp.layers.0.self_attn.v_proj.weight",
            )?);
            tensors.push(copy(
                src,
                &a.o_proj,
                "mtp.layers.0.self_attn.o_proj.weight",
            )?);
            tensors.push(copy(
                src,
                &a.q_norm,
                "mtp.layers.0.self_attn.q_norm.weight",
            )?);
            tensors.push(copy(
                src,
                &a.k_norm,
                "mtp.layers.0.self_attn.k_norm.weight",
            )?);
            tensors.push(copy(
                src,
                &mtp.mlp_norm,
                "mtp.layers.0.post_attention_layernorm.weight",
            )?);
            match &mtp.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    tensors.push(fused(
                        src,
                        gate_up,
                        [
                            "mtp.layers.0.mlp.gate_proj.weight".to_string(),
                            "mtp.layers.0.mlp.up_proj.weight".to_string(),
                        ],
                    )?);
                    tensors.push(copy(src, down, "mtp.layers.0.mlp.down_proj.weight")?);
                }
                Mlp::Routed { .. } => {
                    return Err(ModelError::Illegible {
                        name: "mtp.layers.0.mlp".to_string(),
                        detail: "a draft head is one block and routes to no experts".to_string(),
                    });
                }
            }
            tensors.push(copy(src, &mtp.norm, "mtp.norm.weight")?);
        }

        Ok(ModelContract {
            alignment: ALIGNMENT,
            tensors,

            groups: Vec::new(),
        })
    }

    pub fn import_from_gguf(&self, src: &ztensor::Source) -> Result<ModelContract, ModelError> {
        // **A NAMED REFUSAL, NOT A SILENT HALF-LOAD.** GGUF has no settled
        // spelling for this family's draft head — nothing in the cached
        // artifacts states one, and inventing `blk.*.nextn.*` here would
        // publish a contract whose names no converter writes and whose first
        // symptom is a load that lands fourteen planes and zeroes the
        // fifteenth. A SKU that declares a head and reads a file that cannot
        // state one is refused at the door, by name.
        if self.mtp.is_some() {
            return Err(ModelError::Illegible {
                name: "mtp".to_string(),
                detail: "this SKU declares an MTP draft head and no GGUF \
                         spelling of one is settled; import it from the \
                         safetensors checkpoint"
                    .to_string(),
            });
        }
        let mut tensors = vec![
            copy(src, &self.embed, "token_embd.weight")?,
            copy(src, &self.final_norm, "output_norm.weight")?,
        ];

        if let Head::Bank(head) = &self.head {
            tensors.push(copy(src, head, "output.weight")?);
        }

        for (l, w) in self.layers.iter().enumerate() {
            let n = |s: &str| format!("blk.{l}.{s}");

            tensors.push(copy(src, &w.mixer_norm, n("attn_norm.weight"))?);
            tensors.push(copy(src, &w.mlp_norm, n("ffn_norm.weight"))?);

            match &w.mixer {
                Mixer::Attn(a) => {
                    tensors.push(copy(src, &a.qg_proj, n("attn_q.weight"))?);
                    tensors.push(copy(src, &a.k_proj, n("attn_k.weight"))?);
                    tensors.push(copy(src, &a.v_proj, n("attn_v.weight"))?);
                    tensors.push(copy(src, &a.o_proj, n("attn_output.weight"))?);
                    tensors.push(copy(src, &a.q_norm, n("attn_q_norm.weight"))?);
                    tensors.push(copy(src, &a.k_norm, n("attn_k_norm.weight"))?);
                }
                Mixer::Gdn(g) => {
                    tensors.push(copy(src, &g.in_qkvz, n("ssm_in.weight"))?);
                    tensors.push(copy(src, &g.in_ba, n("ssm_beta_alpha.weight"))?);
                    tensors.push(copy(src, &g.conv, n("ssm_conv1d.weight"))?);
                    tensors.push(copy(src, &g.dt_bias, n("ssm_dt.bias"))?);
                    tensors.push(copy(src, &g.a_log, n("ssm_a"))?);
                    tensors.push(copy(src, &g.norm, n("ssm_norm.weight"))?);
                    tensors.push(copy(src, &g.out_proj, n("ssm_out.weight"))?);
                }
            }

            match &w.mlp {
                Mlp::Dense { gate_up, down, .. } => {
                    tensors.push(fused(
                        src,
                        gate_up,
                        [n("ffn_gate.weight"), n("ffn_up.weight")],
                    )?);
                    tensors.push(copy(src, down, n("ffn_down.weight"))?);
                }
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared_gate_up,
                    shared_down,
                    shared_gate,
                    ..
                } => {
                    tensors.push(copy(src, router, n("ffn_gate_inp.weight"))?);

                    tensors.push(fused(
                        src,
                        gate_up,
                        [n("ffn_gate_exps.weight"), n("ffn_up_exps.weight")],
                    )?);
                    tensors.push(copy(src, down, n("ffn_down_exps.weight"))?);
                    tensors.push(fused(
                        src,
                        shared_gate_up,
                        [n("ffn_gate_shexp.weight"), n("ffn_up_shexp.weight")],
                    )?);
                    tensors.push(copy(src, shared_down, n("ffn_down_shexp.weight"))?);
                    tensors.push(copy(src, shared_gate, n("ffn_gate_inp_shexp.weight"))?);
                }
            }
        }

        Ok(ModelContract {
            alignment: ALIGNMENT,
            tensors,

            groups: Vec::new(),
        })
    }
}

fn squeezed(src: &ztensor::Source, from: String) -> Result<Expr, ModelError> {
    let Some(tensor) = src.get(&from) else {
        return Err(ModelError::Missing(from));
    };
    let illegible = |why: &dyn std::fmt::Display| ModelError::Illegible {
        name: from.clone(),
        detail: why.to_string(),
    };
    let shape = tensor.shape();
    let [channels, 1, kernel] = *shape else {
        return Err(illegible(&format!(
            "a depthwise convolution bank is stored [channels, 1, kernel] and \
             this one is stored {shape:?}"
        )));
    };
    let part = tensor.part("data").map_err(|why| illegible(&why))?;
    let stored =
        model_loader::checkpoint::encoding_of(&tensor, &part).map_err(|why| illegible(&why))?;
    Ok(Expr::src(from).transmute(TensorType::new(
        vec![extent(channels), extent(kernel)],
        stored,
    )))
}

fn extent(of: u64) -> i64 {
    i64::try_from(of).expect("an extent no i64 holds")
}
