//! The qwen4 import contracts: foreign checkpoints of `Qwen3.8-Flash-Next`,
//! read into the declaration `model.rs` states.
//!
//! Two layouts, `qwen_3::import`'s own pair: transformers
//! (`model.language_model.*`, `lm_head.weight`) and `mlx_lm`
//! (`language_model.model.*`, `language_model.lm_head.weight`). The MLX
//! spelling folds `+1.0` into every plain-RMSNorm plane — measured for this
//! family the way it was for qwen_3: the 4-bit conversion's `hc_norm` and
//! `q_norm` planes cluster around one where transformers' initialization
//! centres them at zero — and splits the fused expert banks at the seam
//! `read_concat` rejoins. What no layout of this family does is fuse the
//! GDN in-projections: both publish `in_proj_qkv | z | b | a` as four
//! tensors, so both spellings concatenate.
//!
//! **EVERY PROJECTION READS AS STORED** — the mixed-4/8 file's eight-bit
//! triplets land as the `U8g64` planes `model.rs` declares, the four-bit
//! banks land as theirs, and the one weight transform this import states is
//! the norm fold above, taken back out. The dequantizing landing this file
//! shipped first (`Builder::read_dequant`, still the door for a scheme with
//! no kernel) was retired when the affine gemm point arrived: a text
//! declares what the file holds, and the device reads it there.
//!
//! **WHAT IS DELIBERATELY NOT READ**: `visual.*` (this SKU is text-only; a
//! vision row is a second declaration when a deployment wants one),
//! `mtp.*` (a draft arm this text does not yet carry), the
//! `self_attn.indexer.*` planes (the QSA cut — `model::Mixer::Attn`'s doc
//! names it), and the PLE's three hash buffers, which are derived and verified
//! rather than read (`model::Ple`'s doc says why).

use checkpoint::contract::{Expr, ModelContract};

use super::model::{Mixer, Mlp, Model};
use model_dsl::Weight;
use crate::qwen_3::import::squeezed;
use checkpoint_dsl::{Builder, Error};

#[derive(Clone, Copy)]
enum Layout {
    /// `model.language_model.*` + `lm_head.weight` — transformers.
    Transformers,
    /// `language_model.model.*` + `language_model.lm_head.weight` —
    /// `mlx_lm`. Whether the file is the mixed-4/8 stack or a plain bf16
    /// re-spelling is not asked here: each catalog row declares its own
    /// widths, a read that misses its triplet is a miss, and `identify`'s
    /// ladder hands the file to the row whose declaration it matches.
    Mlx,
}

impl Layout {
    fn embed(self) -> &'static str {
        match self {
            Self::Transformers => "model.language_model.embed_tokens.weight",
            Self::Mlx => "language_model.model.embed_tokens.weight",
        }
    }

    fn trunk(self, tail: &str) -> String {
        match self {
            Self::Transformers => format!("model.language_model.{tail}"),
            Self::Mlx => format!("language_model.model.{tail}"),
        }
    }

    fn lm_head(self) -> &'static str {
        match self {
            Self::Transformers => "lm_head.weight",
            Self::Mlx => "language_model.lm_head.weight",
        }
    }

    /// `mlx_lm`'s `sanitize` bakes the `+1.0` of every plain RMSNorm into
    /// the stored plane — `qwen_3::import`'s measurement, repeated on this
    /// family's own file.
    fn folds_the_norm_one(self) -> bool {
        match self {
            Self::Transformers => false,
            Self::Mlx => true,
        }
    }
}

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
        let mut refusals: Vec<String> = Vec::new();
        for (what, layout) in [
            ("transformers", Layout::Transformers),
            ("mlx_lm", Layout::Mlx),
        ] {
            match self.import_from_safetensors(src, layout) {
                Ok(contract) => return Ok(contract),
                Err(why) => refusals.push(format!("as {what}, {why}")),
            }
        }
        Err(Error::Illegible {
            name: "qwen4".to_string(),
            detail: format!(
                "no reading of this file lands every plane this family \
                 declares — {}",
                refusals.join("; "),
            ),
        })
    }

    fn import_from_safetensors(
        &self,
        src: &ztensor::Source,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, self.tp);
        for read in self.reads(layout) {
            match read {
                Read::One(w, name) => b.read(w, name)?,
                // **A JOINED BANK GOES THROUGH THE VERB THAT KNOWS ABOUT
                // COMPANIONS.** `read_concat` joins each part's
                // `.scales`/`.biases` at the same seams its codes join at,
                // which is the whole story for an affine pair and the identity
                // for a dense one.
                Read::Concat(w, names) => b.read_concat(w, names)?,
                // Undo the MLX fold on the planes the text scales by
                // `weight + 1`.
                Read::Norm(w, name) => {
                    let e = Expr::src(name);
                    let e = if layout.folds_the_norm_one() {
                        e.bias(-1.0)
                    } else {
                        e
                    };
                    b.read_expr(w, e)?;
                }
                Read::Squeeze(w, name) => b.read_derived(w, || squeezed(src, name))?,
            }
        }
        Ok(b.build())
    }

    /// **EVERY SOURCE NAME THIS IMPORT READS, IN READ ORDER, WITH THE PLANE
    /// IT LANDS IN BESIDE IT.**
    ///
    /// One list drives both [`import_from_safetensors`](Model::import_from_safetensors)
    /// and the censuses below, so a census cannot drift from the names the
    /// import actually reads — `deepseek_v4::import`'s rule, held here for the
    /// same reason: a per-tensor quantization mix is a claim about the
    /// PAIRING of a stored triplet with a declared width, so the pairing is
    /// what a census has to be able to read. A joined bank appears once per
    /// stored part, because every part lands in the one declaration.
    #[must_use]
    pub fn mlx_planes(&self) -> Vec<(&Weight, String)> {
        self.reads(Layout::Mlx)
            .into_iter()
            .flat_map(|r| match r {
                Read::One(w, name) | Read::Norm(w, name) | Read::Squeeze(w, name) => {
                    vec![(w, name)]
                }
                Read::Concat(w, names) => names.into_iter().map(|n| (w, n)).collect(),
            })
            .collect()
    }

    /// The same census with the planes dropped — the name half, which is what
    /// a bijection against a snapshot's `weight_map` is held against.
    #[must_use]
    pub fn mlx_source_names(&self) -> Vec<String> {
        self.mlx_planes().into_iter().map(|(_, name)| name).collect()
    }

    fn reads(&self, layout: Layout) -> Vec<Read<'_>> {
        let mut reads = Vec::new();
        reads.push(Read::One(&self.embed, layout.embed().to_string()));
        reads.push(Read::One(&self.head, layout.lm_head().to_string()));

        for (l, w) in self.layers.iter().enumerate() {
            let n = |tail: &str| layout.trunk(&format!("layers.{l}.{tail}"));

            match &w.mixer {
                Mixer::Attn(a) => {
                    // The stored `q_proj` is the fused query|gate bank —
                    // `[2 · q_heads · head_dim, hidden]`, `attn_output_gate`.
                    reads.push(Read::One(&a.qg_proj, n("self_attn.q_proj.weight")));
                    reads.push(Read::One(&a.k_proj, n("self_attn.k_proj.weight")));
                    reads.push(Read::One(&a.v_proj, n("self_attn.v_proj.weight")));
                    reads.push(Read::One(&a.o_proj, n("self_attn.o_proj.weight")));
                    reads.push(Read::Norm(&a.q_norm, n("self_attn.q_norm.weight")));
                    reads.push(Read::Norm(&a.k_norm, n("self_attn.k_norm.weight")));
                }
                Mixer::Gdn(g) => {
                    reads.push(Read::Concat(
                        &g.in_qkvz,
                        vec![
                            n("linear_attn.in_proj_qkv.weight"),
                            n("linear_attn.in_proj_z.weight"),
                        ],
                    ));
                    reads.push(Read::Concat(
                        &g.in_ba,
                        vec![
                            n("linear_attn.in_proj_b.weight"),
                            n("linear_attn.in_proj_a.weight"),
                        ],
                    ));
                    reads.push(Read::Squeeze(&g.conv, n("linear_attn.conv1d.weight")));
                    reads.push(Read::One(&g.dt_bias, n("linear_attn.dt_bias")));
                    reads.push(Read::One(&g.a_log, n("linear_attn.A_log")));
                    // The gated norm's weight is NOT plus-one-scaled, and MLX
                    // does not fold it — `qwen_3::import`'s ruling, held.
                    reads.push(Read::One(&g.norm, n("linear_attn.norm.weight")));
                    reads.push(Read::One(&g.out_proj, n("linear_attn.out_proj.weight")));
                }
            }

            let sites = [
                (&w.attn_res, "attn_hyper_connection"),
                (&w.mlp_res, "mlp_hyper_connection"),
            ];
            for (res, site) in sites {
                reads.push(Read::Norm(&res.norm, n(&format!("{site}.hc_norm.weight"))));
                reads.push(Read::One(
                    &res.down,
                    n(&format!("{site}.input_mix_weight_down.weight")),
                ));
                reads.push(Read::One(
                    &res.up,
                    n(&format!("{site}.input_mix_weight_up.weight")),
                ));
                if let Some(inject) = &res.inject {
                    reads.push(Read::One(
                        inject,
                        n(&format!("{site}.block_inject_weight.weight")),
                    ));
                }
            }

            match &w.mlp {
                Mlp::Dense { .. } => unreachable!("every qwen4 layer routes"),
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared_gate_up,
                    shared_down,
                    shared_gate,
                    ..
                } => {
                    reads.push(Read::One(router, n("mlp.gate.weight")));
                    // The expert banks part the way qwen_3's do: transformers
                    // fuses gate|up, `mlx_lm` splits them into `switch_mlp`
                    // and `read_concat` rejoins the pair at the bank seam.
                    match layout {
                        Layout::Transformers => {
                            reads.push(Read::One(gate_up, n("mlp.experts.gate_up_proj")));
                            reads.push(Read::One(down, n("mlp.experts.down_proj")));
                        }
                        Layout::Mlx => {
                            reads.push(Read::Concat(
                                gate_up,
                                vec![
                                    n("mlp.switch_mlp.gate_proj.weight"),
                                    n("mlp.switch_mlp.up_proj.weight"),
                                ],
                            ));
                            reads.push(Read::One(down, n("mlp.switch_mlp.down_proj.weight")));
                        }
                    }
                    reads.push(Read::Concat(
                        shared_gate_up,
                        vec![
                            n("mlp.shared_expert.gate_proj.weight"),
                            n("mlp.shared_expert.up_proj.weight"),
                        ],
                    ));
                    reads.push(Read::One(
                        shared_down,
                        n("mlp.shared_expert.down_proj.weight"),
                    ));
                    reads.push(Read::One(shared_gate, n("mlp.shared_expert_gate.weight")));
                }
            }
        }

        if let Some(p) = &self.ple {
            let n = |tail: &str| layout.trunk(&format!("layers.{}.ple.{tail}", p.layer));
            // The table: `split_ngram_parts` shards, rejoined at the seams
            // the declaration banded — one row space, the hashed offsets'
            // own.
            let shards: Vec<String> = (0..shard_count(p))
                .map(|i| n(&format!("ple_embedding.ngram_embedding.shard_{i}.weight")))
                .collect();
            reads.push(Read::Concat(&p.table, shards));
            reads.push(Read::One(&p.key_proj, n("key_proj.weight")));
            reads.push(Read::One(&p.value_proj, n("value_proj.weight")));
            reads.push(Read::Norm(&p.norm_key, n("norm_key.weight")));
            reads.push(Read::Norm(&p.norm_query, n("norm_query.weight")));
            reads.push(Read::Norm(&p.norm_conv, n("norm_conv.weight")));
            reads.push(Read::Squeeze(&p.conv, n("conv1d.weight")));
        }

        reads.push(Read::Norm(
            &self.mixer.norm,
            layout.trunk("hyper_connection_mixer.hc_norm.weight"),
        ));
        reads.push(Read::One(
            &self.mixer.down,
            layout.trunk("hyper_connection_mixer.input_mix_weight_down.weight"),
        ));
        reads.push(Read::One(
            &self.mixer.up,
            layout.trunk("hyper_connection_mixer.input_mix_weight_up.weight"),
        ));
        reads
    }
}

/// One read of this import: the plane it lands in, and the source name (or
/// names) it lands from.
enum Read<'a> {
    /// A plane read verbatim under its own name.
    One(&'a Weight, String),
    /// A bank joined from stored parts at the seams the declaration bands.
    Concat(&'a Weight, Vec<String>),
    /// A plain-RMSNorm plane, whose `+1` the MLX spelling folds in and this
    /// import takes back out.
    Norm(&'a Weight, String),
    /// A depthwise convolution stored with a unit axis the read squeezes.
    Squeeze(&'a Weight, String),
}

/// How many shards the table's band list declares — the weight's own seams,
/// counted rather than restated.
fn shard_count(p: &super::model::Ple) -> usize {
    match &p.table.shard {
        model_dsl::Shard::Cut { segments, .. } => segments.len(),
        model_dsl::Shard::Replicated => 1,
    }
}
