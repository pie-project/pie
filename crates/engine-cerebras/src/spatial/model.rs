//! The model layer: what the spatial kernels need to know of a model,
//! read off its poem trace ([`ModelSpec::read`]), and the load-time check of
//! a compiled program against it ([`ModelSpec::check_program`]).
//!
//! [`ModelSpec::read`] walks the trace a load hands the engine and keeps
//! what the hand-written kernels depend on: the dims, the structure flags and
//! the dtypes. It accepts only the structure the kernels implement
//! ("Qwen3-dense-like": pre-RMSNorm layers of full causal GQA attention, q
//! and k RMS-normed per head, RoPE over the whole head in neox halves, a
//! SwiGLU MLP, a final norm and a tied or untied lm_head) and refuses
//! anything else with the first op or field that differs.

use poem_ir::{
    Attn, CacheRow, Def, Dtype, Elemwise, Gather, GluAct, Linear, NormAxis, NormWeight, Operands,
    Operation, Param, RopeFreqs, Trace, Turn, ValueId,
};

use super::plan::ROPE_POSITIONS;

/// Why the spatial engine cannot run a model, or cannot place it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refused {}

pub(super) fn refuse<T>(why: impl Into<String>) -> Result<T, Refused> {
    Err(Refused(why.into()))
}

/// What the spatial kernels need to know of a model, read off its trace.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    /// The trace's name (the deployment: `qwen3-8b-bf16-kv-bf16`).
    pub name: String,
    pub layers: usize,
    pub hidden: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub vocab: usize,
    /// The lm_head reads the embedding table.
    pub tied: bool,
    /// q and k are RMS-normed per head before RoPE.
    pub qk_norm: bool,
    pub rope_theta: f32,
    /// The frequencies the trace states outright (llama3 or YaRN scaling),
    /// with YaRN's cos/sin scale; `None`: `theta ^ (-2i / head_dim)`.
    pub rope_table: Option<RopeFreqs>,
    pub rms_eps: f32,
    /// The attention's softmax scale (`1/sqrt(head_dim)` for every model
    /// read so far); the placement folds it into the q norm's weights.
    pub sm_scale: f32,
    /// The projections' dtype as deployed.
    pub weights: Dtype,
    /// The KV cache's dtype as deployed.
    pub kv: Dtype,
}

/// A difference from the kernels as built that the plan can absorb at small
/// cost, but that nothing implements yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Need {
    /// No q/k RMSNorm (Llama). The paged kernels always scale q and k by
    /// their per-head 1/rms (the sums of squares ride the scores' segment
    /// all-reduce); a build flag that skips both scales and the two sums
    /// would make the norm the identity. Kernel change, not built.
    IdentityQkNorm,
    /// The trace states RoPE's frequencies (llama3 or YaRN scaling). The
    /// kernels build cos/sin from per-column frequency tables the host
    /// writes (`rot_c`/`rot_s`), so this is placement only: write the stated
    /// table instead of theta's, and fold YaRN's cos/sin scale into q's and
    /// k's norm weights (scores scale by its square). `serve2d::place`
    /// takes a theta today.
    RopeTable,
    /// An untied lm_head (Qwen3-8B and up). The kernels' heads hold a
    /// separate embedding table (head2d `UNTIED` 1), but the engine's
    /// placement (`wafer::place`, `serve2d::place`) reads
    /// `model.embed_tokens.weight` for the logits: [`ModelSpec::check_program`]
    /// refuses the model on those programs. serve8b places `lm_head.weight`
    /// (`serve8b::place`).
    LmHeadPlacement,
}

/// The RMSNorm epsilon the CSL programs carry as a literal (the qwen2d
/// kernels', qwen2d_paged's and paged8b's `EPS`).
pub const KERNEL_EPS: f32 = 1e-6;

/// The per-layer facts the structural check gathers.
#[derive(Debug, Clone, Default, PartialEq)]
struct Layer {
    row_norms: Vec<(ValueId, f32)>,
    head_norms: usize,
    ropes: usize,
    appends: usize,
    attends: usize,
    glus: usize,
    residuals: usize,
    /// (act, w, y) of each projection, in trace order.
    matmuls: Vec<(ValueId, ValueId, ValueId)>,
    glu_input: Vec<ValueId>,
    head_dim: Option<u32>,
    kv_heads: Option<u32>,
    q_width: Option<u32>,
    inter: Option<u32>,
    theta: Option<f32>,
    table: Option<RopeFreqs>,
    sm_scale: Option<f32>,
}

/// What a family of ops is, for a refusal.
fn what(name: &str) -> &'static str {
    if name.starts_with("linear.moe")
        || name == "linear.group_routes"
        || name == "linear.matmul_grouped"
    {
        "a mixture of experts (the kernels hold one dense SwiGLU MLP a layer)"
    } else if name.starts_with("ssm.") {
        "a recurrent / gated-delta layer (the kernels attend over a KV cache in every layer)"
    } else if name.starts_with("attn.mla") {
        "multi-head latent attention (the kernels cache per-head K and V)"
    } else if name == "elemwise.layernorm" {
        "LayerNorm (the kernels RMS-normalize)"
    } else if name == "elemwise.add_bias" {
        "projection biases (the kernels' matvecs carry none)"
    } else if name == "elemwise.softcap" {
        "logit soft-capping"
    } else if name.starts_with("elemwise.hc_") {
        "hyper-connections (the kernels keep one residual stream)"
    } else if name == "attn.sink" {
        "attention sinks (the kernels' softmax has no sink term)"
    } else {
        "an op the spatial kernels do not implement"
    }
}

impl ModelSpec {
    /// Reads the spec off `trace`, or refuses: the first op or field that
    /// the spatial kernels do not implement, with its layer.
    pub fn read(trace: &Trace) -> Result<ModelSpec, Refused> {
        let param = |v: ValueId| -> Option<&Param> {
            match trace.values.get(v.0 as usize)?.def {
                Def::Weight(i) => trace.params.get(i as usize),
                _ => None,
            }
        };
        let read: std::collections::HashSet<ValueId> = trace
            .nodes
            .iter()
            .flat_map(|n| {
                let mut v = Vec::new();
                n.op.inputs(&mut v);
                v
            })
            .collect();
        let feeds_an_op = |op: &Operation| {
            let mut v = Vec::new();
            op.outputs(&mut v);
            v.iter().any(|y| read.contains(y))
        };
        // Whole structures first: a model with experts or recurrent layers
        // is refused for those, not for the first field that differs.
        for node in &trace.nodes {
            let name = node.op.name();
            let why = what(name);
            if why != what("") {
                let layer = node.layer.map_or(String::new(), |l| format!("layer {l}: "));
                return refuse(format!("{}: {layer}{name}: {why}", trace.name));
            }
        }
        let mut layers: Vec<Layer> = Vec::new();
        let (mut vocab, mut embed, mut q_heads) = (None, None, None);
        let (mut final_norms, mut eps, mut head) = (0usize, None, None);
        for node in &trace.nodes {
            let name = node.op.name();
            let at = |why: &str| match node.layer {
                Some(l) => format!("{}: layer {l}: {name}: {why}", trace.name),
                None => format!("{}: {name}: {why}", trace.name),
            };
            let l = node.layer.map(|l| l as usize);
            if let Some(l) = l
                && layers.len() <= l
            {
                layers.resize(l + 1, Layer::default());
            }
            let layer = l.map(|l| &mut layers[l]);
            match (&node.op, layer) {
                (
                    Operation::Gather(Gather::Embed {
                        table,
                        vocab: v,
                        weights: None,
                        scales: None,
                        concat: false,
                        ..
                    }),
                    None,
                ) => {
                    vocab = Some(*v as usize);
                    embed = Some(*table);
                }
                (Operation::Gather(Gather::Rows { .. }), None) => {}
                // A sample no op reads is only an output (the self-drafting
                // `mtp.drafts`); the wafer program samples on its own.
                (Operation::Sampling(_), None) if !feeds_an_op(&node.op) => {}
                (
                    Operation::Elemwise(Elemwise::Rmsnorm {
                        weight,
                        axis,
                        eps: e,
                        x,
                        y,
                    }),
                    layer,
                ) => {
                    if !matches!(weight, NormWeight::Scale(_)) {
                        return refuse(at(&format!(
                            "norm weight {weight:?} (the kernels scale by the weight itself; a 1 + w \
                             weight would be the host adding 1, not done)"
                        )));
                    }
                    if eps.is_some_and(|eps| eps != *e) {
                        return refuse(at("RMSNorm epsilons differ between norms"));
                    }
                    eps = Some(*e);
                    match (axis, layer) {
                        (NormAxis::Row, Some(layer)) => layer.row_norms.push((*y, *e)),
                        (NormAxis::Row, None) => {
                            final_norms += 1;
                            let _ = x;
                        }
                        (NormAxis::Head(hd), Some(layer)) => {
                            layer.head_norms += 1;
                            layer.head_dim.get_or_insert(*hd);
                        }
                        (axis, _) => return refuse(at(&format!("a norm over {axis:?}"))),
                    }
                }
                (
                    Operation::Elemwise(Elemwise::Rope {
                        k: Some(_),
                        head_dim,
                        theta,
                        turn,
                        freqs,
                        ..
                    }),
                    Some(layer),
                ) => {
                    match turn {
                        Turn::Full { interleaved: false } => {}
                        Turn::Full { interleaved: true } => {
                            return refuse(at(
                                "interleaved RoPE pairs (the kernels rotate neox halves; permuting q's \
                                 and k's projection rows on the host would map it, not done)",
                            ));
                        }
                        other => {
                            return refuse(at(&format!(
                                "RoPE over part of the head ({other:?}; the kernels rotate all of it)"
                            )));
                        }
                    }
                    layer.ropes += 1;
                    layer.head_dim.get_or_insert(*head_dim);
                    layer.theta = Some(*theta);
                    layer.table = freqs.clone();
                }
                (Operation::Elemwise(Elemwise::ResidualAdd { gate: None, .. }), Some(layer)) => {
                    layer.residuals += 1;
                }
                (Operation::Gather(Gather::SplitQkv { q_width, .. }), Some(layer)) => {
                    layer.q_width = Some(*q_width);
                }
                (
                    Operation::Attn(
                        Attn::PlanDecode {
                            q_heads: q,
                            window: None,
                            ..
                        }
                        | Attn::PlanPrefill {
                            q_heads: q,
                            window: None,
                            ..
                        },
                    ),
                    None,
                ) => {
                    q_heads = Some(*q as usize);
                }
                (Operation::Attn(Attn::KvAppend { v: Some(_), .. }), Some(layer)) => {
                    layer.appends += 1
                }
                (
                    Operation::Attn(Attn::Decode {
                        window,
                        selection,
                        bias,
                        selected,
                        rel,
                        head_dim,
                        sm_scale,
                        ..
                    }),
                    Some(layer),
                )
                | (
                    Operation::Attn(Attn::Prefill {
                        window,
                        selection,
                        bias,
                        selected,
                        rel,
                        head_dim,
                        sm_scale,
                        ..
                    }),
                    Some(layer),
                ) => {
                    if let Some(w) = window {
                        return refuse(at(&format!(
                            "sliding-window attention (window {w}; the kernels attend over every \
                             visible slot; a sink + window bitmap is expressible per page, but the \
                             pool would still hold the whole context)"
                        )));
                    }
                    if selection.is_some() || selected.is_some() || bias.is_some() || rel.is_some()
                    {
                        return refuse(at("sparse or biased attention"));
                    }
                    if let Operation::Attn(Attn::Prefill { causal: false, .. }) = &node.op {
                        return refuse(at("bidirectional attention"));
                    }
                    if let Operation::Attn(Attn::Prefill { kv_heads, .. }) = &node.op {
                        layer.kv_heads = Some(*kv_heads);
                    }
                    layer.attends += 1;
                    layer.head_dim.get_or_insert(*head_dim);
                    layer.sm_scale = Some(*sm_scale);
                }
                // Observation and adapters: the engine's business, not the
                // kernels' (it refuses score reads and LoRA on its own).
                (Operation::Attn(Attn::ScoreCapture { .. }), Some(_))
                | (Operation::Linear(Linear::LoraCorrect { .. }), Some(_))
                | (Operation::Attn(Attn::PlanDecode { .. } | Attn::PlanPrefill { .. }), None)
                    if !matches!(
                        &node.op,
                        Operation::Attn(
                            Attn::PlanDecode {
                                window: Some(_),
                                ..
                            } | Attn::PlanPrefill {
                                window: Some(_),
                                ..
                            }
                        )
                    ) => {}
                (
                    Operation::Linear(Linear::Matmul {
                        act,
                        w,
                        y,
                        quant: None,
                        input_scales: None,
                        scales: None,
                        logits,
                    }),
                    layer,
                ) => match (logits, layer) {
                    (false, Some(layer)) => layer.matmuls.push((*act, *w, *y)),
                    (true, None) => head = Some(*w),
                    _ => return refuse(at("a projection outside the decoder layers")),
                },
                (
                    Operation::Linear(Linear::Glu {
                        packed,
                        gate,
                        up,
                        weights: None,
                        intermediate,
                        act: GluAct::Swiglu,
                        limit: None,
                        ..
                    }),
                    Some(layer),
                ) => {
                    layer.glus += 1;
                    layer.inter = Some(*intermediate);
                    layer
                        .glu_input
                        .extend(packed.iter().chain(gate).chain(up).copied());
                }
                (Operation::Linear(Linear::Glu { act, .. }), Some(_)) if *act != GluAct::Swiglu => {
                    return refuse(at(&format!("a {act:?} MLP (the kernels run SwiGLU)")));
                }
                (op, _) => {
                    return refuse(at(&format!("{} ({op:?})", what(name))
                        .chars()
                        .take(400)
                        .collect::<String>()));
                }
            }
        }

        let name = &trace.name;
        let Some(first) = layers.first().cloned() else {
            return refuse(format!("{name}: no decoder layers"));
        };
        let (Some(vocab), Some(embed), Some(q_heads), Some(head)) = (vocab, embed, q_heads, head)
        else {
            return refuse(format!(
                "{name}: no token embedding, attention plan or lm_head"
            ));
        };
        if final_norms != 1 {
            return refuse(format!(
                "{name}: {final_norms} norms outside the layers (the head wants one final RMSNorm)"
            ));
        }
        let hidden = param(embed)
            .and_then(|p| p.shape.get(1).copied())
            .ok_or_else(|| Refused(format!("{name}: the embedding table is not a weight")))?
            as usize;
        let (head_dim, kv_heads, inter) = match (first.head_dim, first.kv_heads, first.inter) {
            (Some(hd), Some(kv), Some(i)) => (hd as usize, kv as usize, i as usize),
            _ => {
                return refuse(format!(
                    "{name}: layer 0 states no head_dim, kv heads or MLP width"
                ));
            }
        };
        let qk_norm = first.head_norms == 2;
        let mut weights = None;
        for (l, layer) in layers.iter().enumerate() {
            let bad = |why: String| refuse(format!("{name}: layer {l}: {why}"));
            if (
                layer.head_dim,
                layer.kv_heads,
                layer.inter,
                layer.theta,
                &layer.table,
            ) != (
                first.head_dim,
                first.kv_heads,
                first.inter,
                first.theta,
                &first.table,
            ) || layer.head_norms != first.head_norms
            {
                return bad(
                    "its attention or MLP differs from layer 0's (the kernels repeat one layer)"
                        .into(),
                );
            }
            if layer.row_norms.len() != 2 || !matches!(layer.head_norms, 0 | 2) {
                return bad(format!(
                    "{} row norms and {} head norms (wanted 2 pre-norms, and q/k norms or none)",
                    layer.row_norms.len(),
                    layer.head_norms
                ));
            }
            if layer.ropes != 1
                || layer.appends != 1
                || layer.attends == 0
                || layer.glus != 1
                || layer.residuals != 2
            {
                return bad(format!(
                    "{} RoPE, {} KV appends, {} attention reads, {} GLUs, {} residual adds \
                     (wanted one attention block and one SwiGLU MLP, each pre-normed with a residual)",
                    layer.ropes, layer.appends, layer.attends, layer.glus, layer.residuals
                ));
            }
            if layer
                .q_width
                .is_some_and(|q| q as usize != q_heads * head_dim)
            {
                return bad(format!(
                    "q is {:?} wide, not {q_heads} heads of {head_dim}",
                    layer.q_width
                ));
            }
            // Pre-norm: the first projection reads the first norm, the MLP's
            // reads the second.
            let (attn_x, mlp_x) = (layer.row_norms[0].0, layer.row_norms[1].0);
            let reads = |x: ValueId| layer.matmuls.iter().filter(move |m| m.0 == x);
            let first_reads_norm = layer.matmuls.first().is_some_and(|m| m.0 == attn_x);
            let mlp_reads_norm =
                reads(mlp_x).count() > 0 && reads(mlp_x).all(|m| layer.glu_input.contains(&m.2));
            if !first_reads_norm || !mlp_reads_norm {
                return bad("the attention and the MLP do not both read a pre-norm".into());
            }
            for (_, w, _) in &layer.matmuls {
                let dtype = param(*w).map(|p| p.dtype);
                if weights.is_some_and(|d| Some(d) != dtype) {
                    return bad("projections of different dtypes".into());
                }
                weights = dtype;
            }
        }
        let weights = weights.ok_or_else(|| Refused(format!("{name}: no projection weights")))?;
        if !matches!(weights, Dtype::Bf16 | Dtype::F32) {
            return refuse(format!(
                "{name}: {weights:?} projections (the kernels hold f32 or bf16 weights; plan the \
                 model's bf16 deployment)"
            ));
        }
        let kv = match trace.caches.first() {
            Some(CacheRow::Kv {
                dtype,
                window: None,
                rows: None,
                ..
            }) => *dtype,
            other => {
                return refuse(format!(
                    "{name}: a KV cache the kernels do not keep: {other:?}"
                ));
            }
        };
        if !matches!(kv, Dtype::Bf16 | Dtype::F32) {
            return refuse(format!(
                "{name}: {kv:?} KV (the kernels store f32 or bf16 keys and values)"
            ));
        }
        if !q_heads.is_multiple_of(kv_heads) {
            return refuse(format!(
                "{name}: {q_heads} q heads over {kv_heads} kv heads"
            ));
        }
        let tied = head == embed || param(head).map(|p| &p.name) == param(embed).map(|p| &p.name);
        Ok(ModelSpec {
            name: name.clone(),
            layers: layers.len(),
            hidden,
            q_heads,
            kv_heads,
            head_dim,
            inter,
            vocab,
            tied,
            qk_norm,
            rope_theta: first.theta.unwrap_or(0.0),
            rope_table: first.table,
            rms_eps: eps.unwrap_or(0.0),
            sm_scale: first.sm_scale.unwrap_or(0.0),
            weights,
            kv,
        })
    }

    /// The differences from the kernels as built that a plan carries.
    pub fn needs(&self) -> Vec<Need> {
        let mut needs = Vec::new();
        if !self.qk_norm {
            needs.push(Need::IdentityQkNorm);
        }
        if self.rope_table.is_some() {
            needs.push(Need::RopeTable);
        }
        if !self.tied {
            needs.push(Need::LmHeadPlacement);
        }
        needs
    }

    /// Query heads a kv head.
    pub fn group(&self) -> usize {
        self.q_heads / self.kv_heads
    }
}

/// A compiled program's params, read from its `out.json` (`params`: numbers
/// or numeric strings, as the compiler writes them).
pub fn out_json_params(out_json: &serde_json::Value) -> impl Fn(&str) -> Option<usize> + '_ {
    move |k| {
        let x = &out_json["params"][k];
        x.as_u64()
            .or_else(|| x.as_str().and_then(|s| s.trim().parse().ok()))
            .map(|n| n as usize)
    }
}

/// The load-time check: the model the trace describes against the wafer
/// program a load names (its `out.json`). Refuses a structure the kernels
/// don't implement, a program built for other dims, and what the engine's
/// placement can't do yet.
pub fn check(trace: &Trace, out_json: &serde_json::Value) -> Result<ModelSpec, Refused> {
    let spec = ModelSpec::read(trace)?;
    spec.check_program(out_json_params(out_json))?;
    Ok(spec)
}

impl ModelSpec {
    /// Checks a compiled program's params against this model: the grid
    /// builds' vocabulary (`HR`, `Qh`, `PX`, ...), with its divisibility, and the facts the
    /// programs and the placement carry as constants (eps, q/k norms, the
    /// RoPE frequencies from theta, 1/sqrt(head_dim), a tied head).
    pub fn check_program(&self, get: impl Fn(&str) -> Option<usize>) -> Result<(), Refused> {
        let name = &self.name;
        fn no<T>(name: &str, why: String) -> Result<T, Refused> {
            refuse(format!("{name}: the wafer program: {why}"))
        }
        // What every program and today's placement assume.
        if self.rms_eps != KERNEL_EPS {
            return no(
                name,
                format!(
                    "RMSNorm eps {} (the CSL's literal is {KERNEL_EPS})",
                    self.rms_eps
                ),
            );
        }
        if !self.qk_norm {
            return no(
                name,
                "no q/k RMSNorm (the kernels always norm q and k; Need::IdentityQkNorm)".into(),
            );
        }
        if self.rope_table.is_some() {
            return no(
                name,
                "scaled RoPE (the placement builds frequencies from theta; Need::RopeTable)".into(),
            );
        }
        let scale = (self.head_dim as f32).sqrt().recip();
        if (self.sm_scale - scale).abs() > 1e-6 * scale {
            return no(
                name,
                format!(
                    "softmax scale {} (the placement folds 1/sqrt(head_dim) = {scale} into q_norm)",
                    self.sm_scale
                ),
            );
        }
        // serve8b (`GS` among its params) places lm_head apart from the
        // embedding; the other programs read the embedding for the logits.
        let untied_head = get("GS").or(get("RPX"));
        if !self.tied && untied_head.is_none() {
            return no(
                name,
                "an untied lm_head (the placement reads embed_tokens for the logits and would \
                 give wrong ones; Need::LmHeadPlacement)"
                    .into(),
            );
        }
        if self.tied && untied_head.is_some() {
            return no(
                name,
                "a tied head on serve8b, which places lm_head.weight (untied models only)".into(),
            );
        }
        let p = |k: &str| {
            get(k).ok_or_else(|| Refused(format!("{name}: the wafer program has no param {k}")))
        };
        let eq = |what: &str, have: usize, want: usize| {
            if have == want {
                Ok(())
            } else {
                no(name, format!("{what} is {have}; the model's is {want}"))
            }
        };
        let divides = |what: &str, n: usize, d: usize| {
            if d != 0 && n.is_multiple_of(d) {
                Ok(n / d)
            } else {
                no(name, format!("{what}: {n} over {d} does not divide"))
            }
        };
        if get("HEADMODE").is_some() {
            // serve2d / serve2d_pipe / the 2-D layer builds.
            let (px, py, hc, dc, gs) = (p("PX")?, p("PY")?, p("Hc")?, p("Dc")?, p("Gs")?);
            eq("HR (hidden)", p("HR")?, self.hidden)?;
            eq("Dc x Gs (head_dim)", dc * gs, self.head_dim)?;
            eq("Qh (q heads a kv head)", p("Qh")?, self.group())?;
            eq("vocab", p("vocab")?, self.vocab)?;
            let region = get("RPX").is_some();
            let (gw, gh) = if region { (0, 0) } else { (p("GW")?, p("GH")?) };
            match untied_head {
                // serve8b_region: the band, the regions' layers, the grid column.
                Some(_) if region => {
                    let band = p("RC")? * p("RPX")? / px + 1;
                    eq(
                        "the units' layers",
                        band + p("RC")? * p("RR")? * p("RNL")? + p("NCOL")?,
                        self.layers,
                    )?;
                }
                // Strips past the first GS skip a slot.
                Some(gs) if gs <= gh && gw > 0 => eq(
                    "GS GW + (GH - GS)(GW - 1) (layers)",
                    gs * gw + (gh - gs) * (gw - 1),
                    self.layers,
                )?,
                Some(gs) => return no(name, format!("GS {gs} full strips of GH {gh}")),
                None => eq("GW x GH (layers)", gw * gh, self.layers)?,
            }
            let kv_cols = divides("PX over a kv head's Gs columns", px, gs)?;
            if kv_cols < self.kv_heads {
                return no(
                    name,
                    format!(
                        "{kv_cols} kv heads' columns for the model's {}",
                        self.kv_heads
                    ),
                );
            }
            if py * hc < self.hidden || px * p("Ic")? < self.inter {
                return no(
                    name,
                    format!(
                        "PY x Hc {} or PX x Ic {} is short of hidden {} / inter {}",
                        py * hc,
                        px * p("Ic")?,
                        self.hidden,
                        self.inter
                    ),
                );
            }
            if get("GRID") == Some(1) {
                let (ps, e) = (p("PS")?, p("E")?);
                if e * py * ps > ROPE_POSITIONS {
                    tracing::debug!(
                        "{name}: lanes reach {} positions; RoPE's tables stop at {ROPE_POSITIONS}",
                        e * py * ps
                    );
                }
            }
        } else {
            return no(
                name,
                "no HEADMODE: not one of the grid programs (the strip programs, serve28*, are \
                 no longer served)"
                    .into(),
            );
        }
        Ok(())
    }
}
