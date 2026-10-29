//! The spatial plan: from a model's poem trace to the wafer program's
//! compile parameters (`PLAN.md` beside this file).
//!
//! [`ModelSpec::read`] walks the trace a load hands the engine and keeps
//! what the hand-written kernels depend on: the dims, the structure flags and
//! the dtypes. It accepts only the structure the kernels implement
//! ("Qwen3-dense-like": pre-RMSNorm layers of full causal GQA attention, q
//! and k RMS-normed per head, RoPE over the whole head in neox halves, a
//! SwiGLU MLP, a final norm and a tied or untied lm_head) and refuses
//! anything else with the first op or field that differs.
//!
//! [`SpatialPlan::derive`] then picks a kernel family and computes every
//! parameter of its build from the model's dims and the wafer: the grid
//! shape and per-PE splits (`PX PY Hc Dc Gs Qh Ic`), the batch and pool
//! (`B NB MC PS E`), the grids' placement (`GW GH`) and the head's layout
//! (`HH Vr F Gx Gy`). Per-PE memory is an estimate fitted to the image ends
//! the builds reported (the kernels' READMEs); the link stays the final
//! word. For Qwen3-0.6B and Qwen3-8B it gives the parameters of the
//! requests the bring-up runs (0105-0108, 0102).

use poem_ir::{
    Attn, CacheRow, Def, Dtype, Elemwise, Gather, GluAct, Linear, NormAxis, NormWeight,
    Operands, Operation, Param, RopeFreqs, Trace, Turn, ValueId,
};

/// Why the spatial engine cannot run a model, or cannot place it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused(pub String);

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Refused {}

fn refuse<T>(why: impl Into<String>) -> Result<T, Refused> {
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
    /// `model.embed_tokens.weight` for the logits: until it places
    /// `lm_head.weight`, [`ModelSpec::check_program`] refuses the model.
    LmHeadPlacement,
}

/// The RMSNorm epsilon the CSL programs carry as a literal (serve28s3's
/// attn/mlp/head PEs, qwen2d_paged's and paged8b's `EPS`).
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
    if name.starts_with("linear.moe") || name == "linear.group_routes" || name == "linear.matmul_grouped" {
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
                (Operation::Gather(Gather::Embed { table, vocab: v, weights: None, scales: None, concat: false, .. }), None) => {
                    vocab = Some(*v as usize);
                    embed = Some(*table);
                }
                (Operation::Gather(Gather::Rows { .. }), None) => {}
                (Operation::Elemwise(Elemwise::Rmsnorm { weight, axis, eps: e, x, y }), layer) => {
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
                (Operation::Elemwise(Elemwise::Rope { k: Some(_), head_dim, theta, turn, freqs, .. }), Some(layer)) => {
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
                (Operation::Attn(Attn::PlanDecode { q_heads: q, window: None, .. } | Attn::PlanPrefill { q_heads: q, window: None, .. }), None) => {
                    q_heads = Some(*q as usize);
                }
                (Operation::Attn(Attn::KvAppend { v: Some(_), .. }), Some(layer)) => layer.appends += 1,
                (Operation::Attn(Attn::Decode { window, selection, bias, selected, rel, head_dim, sm_scale, .. }), Some(layer))
                | (Operation::Attn(Attn::Prefill { window, selection, bias, selected, rel, head_dim, sm_scale, .. }), Some(layer)) => {
                    if let Some(w) = window {
                        return refuse(at(&format!(
                            "sliding-window attention (window {w}; the kernels attend over every \
                             visible slot; a sink + window bitmap is expressible per page, but the \
                             pool would still hold the whole context)"
                        )));
                    }
                    if selection.is_some() || selected.is_some() || bias.is_some() || rel.is_some() {
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
                    if !matches!(&node.op, Operation::Attn(Attn::PlanDecode { window: Some(_), .. } | Attn::PlanPrefill { window: Some(_), .. })) => {}
                (Operation::Linear(Linear::Matmul { act, w, y, quant: None, input_scales: None, scales: None, logits }), layer) => {
                    match (logits, layer) {
                        (false, Some(layer)) => layer.matmuls.push((*act, *w, *y)),
                        (true, None) => head = Some(*w),
                        _ => return refuse(at("a projection outside the decoder layers")),
                    }
                }
                (Operation::Linear(Linear::Glu { packed, gate, up, weights: None, intermediate, act: GluAct::Swiglu, limit: None, .. }), Some(layer)) => {
                    layer.glus += 1;
                    layer.inter = Some(*intermediate);
                    layer.glu_input.extend(packed.iter().chain(gate).chain(up).copied());
                }
                (Operation::Linear(Linear::Glu { act, .. }), Some(_)) if *act != GluAct::Swiglu => {
                    return refuse(at(&format!("a {act:?} MLP (the kernels run SwiGLU)")));
                }
                (op, _) => {
                    return refuse(at(&format!("{} ({op:?})", what(name)).chars().take(400).collect::<String>()));
                }
            }
        }

        let name = &trace.name;
        let Some(first) = layers.first().cloned() else {
            return refuse(format!("{name}: no decoder layers"));
        };
        let (Some(vocab), Some(embed), Some(q_heads), Some(head)) = (vocab, embed, q_heads, head) else {
            return refuse(format!("{name}: no token embedding, attention plan or lm_head"));
        };
        if final_norms != 1 {
            return refuse(format!("{name}: {final_norms} norms outside the layers (the head wants one final RMSNorm)"));
        }
        let hidden = param(embed)
            .and_then(|p| p.shape.get(1).copied())
            .ok_or_else(|| Refused(format!("{name}: the embedding table is not a weight")))? as usize;
        let (head_dim, kv_heads, inter) = match (first.head_dim, first.kv_heads, first.inter) {
            (Some(hd), Some(kv), Some(i)) => (hd as usize, kv as usize, i as usize),
            _ => return refuse(format!("{name}: layer 0 states no head_dim, kv heads or MLP width")),
        };
        let qk_norm = first.head_norms == 2;
        let mut weights = None;
        for (l, layer) in layers.iter().enumerate() {
            let bad = |why: String| refuse(format!("{name}: layer {l}: {why}"));
            if (layer.head_dim, layer.kv_heads, layer.inter, layer.theta, &layer.table)
                != (first.head_dim, first.kv_heads, first.inter, first.theta, &first.table)
                || layer.head_norms != first.head_norms
            {
                return bad("its attention or MLP differs from layer 0's (the kernels repeat one layer)".into());
            }
            if layer.row_norms.len() != 2 || !matches!(layer.head_norms, 0 | 2) {
                return bad(format!(
                    "{} row norms and {} head norms (wanted 2 pre-norms, and q/k norms or none)",
                    layer.row_norms.len(),
                    layer.head_norms
                ));
            }
            if layer.ropes != 1 || layer.appends != 1 || layer.attends == 0 || layer.glus != 1 || layer.residuals != 2 {
                return bad(format!(
                    "{} RoPE, {} KV appends, {} attention reads, {} GLUs, {} residual adds \
                     (wanted one attention block and one SwiGLU MLP, each pre-normed with a residual)",
                    layer.ropes, layer.appends, layer.attends, layer.glus, layer.residuals
                ));
            }
            if layer.q_width.is_some_and(|q| q as usize != q_heads * head_dim) {
                return bad(format!("q is {:?} wide, not {q_heads} heads of {head_dim}", layer.q_width));
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
            Some(CacheRow::Kv { dtype, window: None, rows: None, .. }) => *dtype,
            other => return refuse(format!("{name}: a KV cache the kernels do not keep: {other:?}")),
        };
        if !matches!(kv, Dtype::Bf16 | Dtype::F32) {
            return refuse(format!("{name}: {kv:?} KV (the kernels store f32 or bf16 keys and values)"));
        }
        if !q_heads.is_multiple_of(kv_heads) {
            return refuse(format!("{name}: {q_heads} q heads over {kv_heads} kv heads"));
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

/// The fabric a program is laid out on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaferDims {
    /// The usable rectangle (the fabric less the memcpy frame: 7 columns, 2 rows).
    pub width: usize,
    pub height: usize,
    /// A PE's memory: the image must end below it.
    pub pe_bytes: usize,
}

impl WaferDims {
    /// CS-3: a 762 x 1172 fabric at offsets (4, 1); 48 KB a PE.
    pub const CS3: WaferDims = WaferDims {
        width: 755,
        height: 1170,
        pe_bytes: 0xc000,
    };
}

/// What a build is for; the rest follows from the model and the wafer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Goals {
    /// Lanes a pass (`B`).
    pub lanes: usize,
    /// The lanes the grid's shape must hold (rows are added until it does):
    /// one grid shape serves every build up to it.
    pub grid_lanes: usize,
    /// KV page size (`PS`): PIE's page size.
    pub page_size: usize,
    /// Pool blocks a PE (`NB`), pinned; `None`: the largest power of two up
    /// to `max_blocks` that fits.
    pub blocks: Option<usize>,
    pub max_blocks: usize,
    /// Lanes an MLP chunk (`MC`), pinned; `None`: two where the batch
    /// splits evenly in f32 builds (serve2d's choice), one in bf16 ones (its
    /// buffer costs about two lanes there).
    pub mlp_chunk: Option<usize>,
    /// Forced/output tokens a lane a launch (`CAP`); `None`: serve2d's 64,
    /// halved until the head's token buffers fit, or 16 in a pipeline.
    pub cap: Option<usize>,
    /// Pipeline the layers (serve2d_pipe): micro-batches of B go round.
    pub pipeline: bool,
    /// Copy pages on the host (`KVCP` 0): no copy kernel, ~0.9 KB a PE back.
    pub host_copy: bool,
}

impl Default for Goals {
    fn default() -> Goals {
        Goals {
            lanes: 4,
            grid_lanes: 8,
            page_size: 16,
            blocks: None,
            max_blocks: 8,
            mlp_chunk: None,
            cap: None,
            pipeline: false,
            host_copy: false,
        }
    }
}

impl Goals {
    pub fn lanes(lanes: usize) -> Goals {
        Goals {
            lanes,
            grid_lanes: lanes.max(8),
            ..Goals::default()
        }
    }
}

/// The kernel family a plan builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    /// `serve2d` GRID 1: Fork 8's `qwen2d_paged` grids (f32, square, two
    /// query heads a kv head, hidden unpadded) and head2d beside them.
    Serve2d,
    /// `serve2d_pipe`: the same grids as `qwen2d_pring` ring stages.
    Serve2dPipe,
    /// `qwen2d_paged8b`: one layer grid, bf16 weights and KV, any GQA ratio,
    /// rectangular and hidden-padded. No full program places it yet.
    Paged8b,
}

impl Family {
    /// The bench-job payload that builds it.
    pub fn kernel(self) -> &'static str {
        match self {
            Family::Serve2d => "serve2d",
            Family::Serve2dPipe => "serve2d_pipe",
            Family::Paged8b => "qwen2d_paged8b",
        }
    }
}

/// One layer's grid: PX columns (kv dims, MLP rows) x PY rows (hidden).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grid {
    pub px: usize,
    pub py: usize,
    /// Hidden dims a row (`hidden` zero-padded to `py * hc`).
    pub hc: usize,
    pub hidden: usize,
    /// Head dims a column, `gs` columns a kv head.
    pub dc: usize,
    pub gs: usize,
    pub qh: usize,
    /// MLP rows a column (`inter` zero-padded to `px * ic`).
    pub ic: usize,
    pub lanes: usize,
    pub page_size: usize,
    pub blocks: usize,
    pub entries: usize,
    pub mlp_chunk: usize,
    /// bf16 weights (`W16`) and KV (`KV16`).
    pub w16: bool,
    pub kv16: bool,
    pub host_copy: bool,
}

impl Grid {
    /// The KV pool a layer: pages of `page_size`.
    pub fn pages(&self) -> usize {
        self.blocks * self.py
    }

    /// A lane's most positions: E pages a row, below the RoPE tables' 4096.
    pub fn max_context(&self) -> usize {
        (self.entries * self.py * self.page_size).min(ROPE_POSITIONS)
    }
}

/// The LM head (head2d): `blocks` blocks of `c` chunk columns (plus the VM
/// column each) and `hh` rows of `vr` vocabulary rows, feeder row `hh / 2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Head {
    pub c: usize,
    pub hc: usize,
    pub hh: usize,
    pub vr: usize,
    pub gx: usize,
    pub gy: usize,
    pub gb: usize,
    pub blocks: usize,
    pub vmc: usize,
    pub untied: bool,
    /// f32 tiles, exact (0.6B); else bf16 tiles, one pass, packed-key argmax (8B).
    pub f32: bool,
}

impl Head {
    pub fn width(&self) -> usize {
        self.blocks * (self.c + self.vmc)
    }
}

/// The full program: the grids' placement and the head beside them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Program {
    /// Grids across and down, serpentine; layer 0 top right.
    pub gw: usize,
    pub gh: usize,
    pub cap: usize,
    /// serve2d_pipe: micro-batches a launch and lanes a head step.
    pub mmax: Option<usize>,
    pub hb: Option<usize>,
    /// The rectangle (grids + head).
    pub width: usize,
    pub height: usize,
}

/// A model on the wafer: the family, its grid, the head, the program when
/// one places the grids, and why not when none does.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialPlan {
    pub spec: ModelSpec,
    pub family: Family,
    pub layers: usize,
    pub vocab: usize,
    pub grid: Grid,
    pub head: Head,
    pub program: Option<Program>,
    /// A grid PE's estimated bytes, of `pe_bytes`.
    pub grid_bytes: usize,
    pub pe_bytes: usize,
    pub needs: Vec<Need>,
    /// Why no full program places the model (empty when `program` is set).
    pub blockers: Vec<String>,
}

// ---- What the kernels fix ---------------------------------------------------

/// Head dims a column: RoPE pairs Dc/2 dims from each half of a head.
const DC: usize = 8;
/// One page entry a lane a row (every build so far).
const ENTRIES: usize = 1;
/// RoPE angles from three 16-entry tables: positions below 16^3.
const ROPE_POSITIONS: usize = 4096;
/// `Tc` (a single sequence's KV slots a row): unused by GRID 1 builds,
/// carried at serve2d's value.
const TC: usize = 32;
/// head2d's vocabulary rows a PE are a multiple of 32 (the VM's exact sums,
/// request 0079).
const VR_STEP: usize = 32;
/// The bf16 head (8B, head2d v10): 256 region rows, 32 hidden dims a PE, at
/// most ~19.5 KB of tables a PE beside its buffers and code.
const HEAD16_ROWS: usize = 256;
const HEAD16_HC: usize = 32;
const HEAD16_TABLES: usize = 19_500;
/// The f32 one-block head beside serve2d's grids: tables a PE, at most.
const HEAD32_TABLES: usize = 16_384;
/// serve2d: `in_tokens` + `out_tokens` at the root take B x CAP words each,
/// 4 KB together at most (B 16 at CAP 64 ran out); the pipe's take MMAX x B
/// x CAP, 1024 words at most. Pipelines run up to 8 micro-batches, head
/// steps of up to 8 lanes.
const TOKENS_SERVE: usize = 512;
const TOKENS_PIPE: usize = 1024;
const MMAX: usize = 8;
const HEAD_BATCH: usize = 8;

// ---- Per-PE memory, fitted to the builds' image ends ------------------------
//
// A grid PE holds code and runtime, its weight tiles, the norm/RoPE tables,
// the KV pool (NB blocks of Dc x PS keys and values), a lane's buffers per
// lane and the MLP chunk's buffers. The weights, tables and pool are exact;
// the code and the lane's scratch are fitted:
// - serve2d (f32): image ends B 8 NB 8 MC 2 = 0xb180, B 16 NB 4 MC 2 = 0xb790
//   (serve2d/README.md), B 16 NB 8 does not link (model: 51.6 KB);
// - qwen2d_paged8b (bf16): B 1 NB 4 = 0xb490 (paged8b/README.md); the model
//   is within 0.1 KB of every other build in that table and fails exactly
//   the two that did not link (B 8 NB 2, B 9 NB 1; with KVCP 0, B 8 NB 4).

const SERVE2D_FIXED: usize = 24_984;
const SERVE2D_LANE_MISC: usize = 50;
const PAGED8B_FIXED: usize = 21_404;
/// The page-copy kernel (paged8b: 18.4 KB of code with it, 17.5 KB without).
const COPY_CODE: usize = 922;
/// A ring stage's handoff code (qwen2d_pring over qwen2d_paged at B 8:
/// 0x3e0 more, of which the received `[x | tab]` is 416 B).
const RING_CODE: usize = 576;

impl Grid {
    fn weight_bytes(&self) -> usize {
        // wqkv Hc x (Qh + 2) Dc, wo Qh Dc x Hc, wgu Hc x 2 Ic, wd Ic x Hc.
        let elems = self.hc * ((2 * self.qh + 2) * self.dc + 3 * self.ic);
        elems * if self.w16 { 2 } else { 4 }
    }

    fn table_bytes(&self) -> usize {
        // gamma1/2 [Hc], qkn_w [2][Qh + 1][DH], rot_c/rot_s [3][16][DH].
        let dh = self.dc / 2;
        4 * (2 * self.hc + 2 * (self.qh + 1) * dh + 2 * 3 * 16 * dh)
    }

    fn block_bytes(&self) -> usize {
        2 * self.dc * self.page_size * if self.kv16 { 2 } else { 4 }
    }

    fn lane_bytes(&self) -> usize {
        let (tab, qkv) = (2 + 3 * self.entries, (self.qh + 2) * self.dc + 1);
        let scores = self.qh * self.entries * self.page_size;
        if self.w16 {
            // paged8b: x, its table, QKV, scores (buffers shared by phase).
            4 * (self.hc + tab + qkv + scores)
        } else {
            // qwen2d_paged in serve2d: x and the handoff's x, the table, QKV,
            // scores and their scratch, RoPE rows, softmax and misc.
            let rope = 2 * (self.qh + 1) * (self.dc / 2);
            4 * (2 * self.hc + tab + qkv + 2 * scores + rope + SERVE2D_LANE_MISC)
        }
    }

    /// A PE's estimated bytes.
    pub fn bytes(&self, ring: bool) -> usize {
        let fixed = if self.w16 { PAGED8B_FIXED } else { SERVE2D_FIXED };
        let copy = if self.host_copy { COPY_CODE } else { 0 };
        let ring = if ring {
            RING_CODE + 4 * self.lanes * (self.hc + 2 + 3 * self.entries)
        } else {
            0
        };
        fixed - copy
            + ring
            + self.weight_bytes()
            + self.table_bytes()
            + self.blocks * self.block_bytes()
            + self.lanes * self.lane_bytes()
            + 4 * self.mlp_chunk * (3 * self.ic + 1)
    }
}

/// The largest even divisor of `n` at most `cap` (a two-phase all-reduce's
/// phase-1 group: 8 and 16 cost about the same, 32 more; request 0058).
fn group(n: usize, cap: usize) -> Option<usize> {
    (2..=cap.min(n)).rev().find(|g| g % 2 == 0 && n.is_multiple_of(*g))
}

impl SpatialPlan {
    /// The plan for `spec` on `wafer`, or why no grid holds it.
    pub fn derive(spec: &ModelSpec, wafer: WaferDims, goals: Goals) -> Result<SpatialPlan, Refused> {
        let name = &spec.name;
        if !spec.head_dim.is_multiple_of(DC) || !(spec.head_dim / DC).is_multiple_of(2) {
            return refuse(format!(
                "{name}: head_dim {} does not split into an even number of {DC}-dim column slices",
                spec.head_dim
            ));
        }
        if goals.lanes == 0 || goals.page_size == 0 {
            return refuse("a plan wants at least one lane and a page size");
        }
        let gs = spec.head_dim / DC;
        let px = spec.kv_heads * gs;
        if px > wafer.width {
            return refuse(format!("{name}: {px} columns a grid (kv heads x {gs}) exceed the wafer"));
        }
        let qh = spec.group();
        let mut why_not_serve2d = Vec::new();
        if qh != 2 {
            why_not_serve2d.push(format!("{qh} query heads a kv head (qwen2d_paged runs 2)"));
        }
        if !spec.hidden.is_multiple_of(px) || !spec.inter.is_multiple_of(px) {
            why_not_serve2d.push(format!(
                "hidden {} or inter {} does not split over {px} square rows/columns",
                spec.hidden, spec.inter
            ));
        }
        let pipeline = goals.pipeline;
        if why_not_serve2d.is_empty() {
            let grid = Self::grid(spec, px, px, false, &goals);
            let bytes = grid.bytes(pipeline);
            if bytes <= wafer.pe_bytes {
                return Self::serve2d(spec, wafer, goals, grid);
            }
            why_not_serve2d.push(format!(
                "the f32 grid wants {bytes} bytes a PE (of {}) at B {} NB {}",
                wafer.pe_bytes, grid.lanes, grid.blocks
            ));
        }
        let mut plan = Self::paged8b(spec, wafer, goals, px)?;
        plan.blockers.insert(
            0,
            format!(
                "serve2d (the only full program) wants square f32 grids: {}",
                why_not_serve2d.join("; ")
            ),
        );
        Ok(plan)
    }

    /// A grid of `px` x `py` at `goals`' batch, its pool deepest that fits
    /// unless pinned.
    fn grid(spec: &ModelSpec, px: usize, py: usize, w16: bool, goals: &Goals) -> Grid {
        let lanes = goals.lanes;
        let mlp_chunk = goals.mlp_chunk.unwrap_or(if !w16 && lanes.is_multiple_of(2) { 2 } else { 1 });
        let mut grid = Grid {
            px,
            py,
            hc: spec.hidden.div_ceil(py),
            hidden: spec.hidden,
            dc: DC,
            gs: spec.head_dim / DC,
            qh: spec.group(),
            ic: spec.inter.div_ceil(px),
            lanes,
            page_size: goals.page_size,
            blocks: goals.blocks.unwrap_or(1),
            entries: ENTRIES,
            mlp_chunk,
            w16,
            kv16: w16,
            host_copy: goals.host_copy,
        };
        if goals.blocks.is_none() {
            let mut nb = 1;
            while nb * 2 <= goals.max_blocks
                && (Grid { blocks: nb * 2, ..grid }).bytes(goals.pipeline) <= WaferDims::CS3.pe_bytes
            {
                nb *= 2;
            }
            grid.blocks = nb;
        }
        grid
    }

    /// serve2d / serve2d_pipe: the grids serpentine, head2d east of them.
    fn serve2d(spec: &ModelSpec, wafer: WaferDims, goals: Goals, grid: Grid) -> Result<SpatialPlan, Refused> {
        let (p, layers, vmc) = (grid.py, spec.layers, 1);
        let mut blockers = Vec::new();
        let mut best: Option<(Program, Head)> = None;
        // Every split of the layers into GW x GH (GW >= 2; the feeder row
        // between layer 0's next strip and the last: GH >= 4) whose grids
        // and one-block head fit; the smallest rectangle wins.
        for gw in 2..=layers {
            if layers % gw != 0 {
                continue;
            }
            let gh = layers / gw;
            let (width, height) = (gw * p + p + vmc, gh * p);
            if gh < 4 || width > wafer.width || height > wafer.height {
                continue;
            }
            let Some(head) = Self::head32(spec, &grid, height, gh) else {
                continue;
            };
            if best.is_none_or(|(b, _)| width * height < b.width * b.height) {
                let program = Program {
                    gw,
                    gh,
                    cap: 0,
                    mmax: None,
                    hb: None,
                    width,
                    height,
                };
                best = Some((program, head));
            }
        }
        let head = match best {
            Some((_, head)) => head,
            None => {
                blockers.push(format!(
                    "no GW x GH = {layers} split of {p} x {p} grids (GW >= 2, GH >= 4) with the \
                     head beside them fits {} x {}",
                    wafer.width, wafer.height
                ));
                Self::head32(spec, &grid, usize::MAX, usize::MAX).unwrap_or(Head {
                    c: p,
                    hc: grid.hc,
                    hh: 0,
                    vr: 0,
                    gx: 0,
                    gy: 0,
                    gb: 1,
                    blocks: 1,
                    vmc,
                    untied: !spec.tied,
                    f32: true,
                })
            }
        };
        let b = grid.lanes;
        let program = best.map(|(mut program, _)| {
            if goals.pipeline {
                let cap = goals.cap.unwrap_or(16);
                program.cap = cap;
                program.mmax = Some((TOKENS_PIPE / (b * cap)).clamp(1, MMAX));
                program.hb = Some((1..=HEAD_BATCH.min(b)).rev().find(|h| b.is_multiple_of(*h)).unwrap_or(1));
            } else {
                let mut cap = goals.cap.unwrap_or(64);
                while b * cap > TOKENS_SERVE && cap > 2 {
                    cap /= 2;
                }
                program.cap = cap;
            }
            program
        });
        Ok(SpatialPlan {
            spec: spec.clone(),
            family: if goals.pipeline { Family::Serve2dPipe } else { Family::Serve2d },
            layers,
            vocab: spec.vocab,
            grid,
            head,
            program,
            grid_bytes: grid.bytes(goals.pipeline),
            pe_bytes: wafer.pe_bytes,
            needs: spec.needs(),
            blockers,
        })
    }

    /// serve2d's head: one block of C = PY chunk columns, f32 tiles, no
    /// taller than the grids, its feeder row (HH / 2) below layer 0's strip
    /// and the next and above the last. The fewest vocabulary rows a PE
    /// (multiples of 32) that fit.
    fn head32(spec: &ModelSpec, grid: &Grid, height: usize, gh: usize) -> Option<Head> {
        let p = grid.py;
        let tables = if spec.tied { 1 } else { 2 };
        (1..)
            .map(|k| k * VR_STEP)
            .take_while(|vr| tables * 4 * grid.hc * vr <= HEAD32_TABLES)
            .find_map(|vr| {
                let hh = spec.vocab.div_ceil(vr).next_multiple_of(2);
                let f = hh / 2;
                let fits = hh < height && f > 2 * p - 1 && f < (gh.saturating_sub(1)).saturating_mul(p);
                let (gx, gy) = (group(p, 16)?, group(f, 18)?);
                fits.then_some(Head {
                    c: p,
                    hc: grid.hc,
                    hh,
                    vr,
                    gx,
                    gy,
                    gb: 1,
                    blocks: 1,
                    vmc: 1,
                    untied: !spec.tied,
                    f32: true,
                })
            })
    }

    /// qwen2d_paged8b: rows added (in steps of 16) until `grid_lanes` lanes
    /// with one pool block fit a PE; then the build's batch and pool.
    fn paged8b(spec: &ModelSpec, wafer: WaferDims, goals: Goals, px: usize) -> Result<SpatialPlan, Refused> {
        let name = &spec.name;
        let sizing = Goals {
            lanes: goals.grid_lanes.max(goals.lanes),
            blocks: Some(1),
            ..goals
        };
        let py = (1..)
            .map(|k| 16 * k)
            .take_while(|&py| py <= wafer.height)
            .find(|&py| Self::grid(spec, px, py, true, &sizing).bytes(false) <= wafer.pe_bytes)
            .ok_or_else(|| {
                let tall = Self::grid(spec, px, wafer.height / 16 * 16, true, &sizing);
                Refused(format!(
                    "{name}: no {px}-column bf16 grid holds {} lanes: even {} rows want {} bytes a PE (of {})",
                    sizing.lanes,
                    tall.py,
                    tall.bytes(false),
                    wafer.pe_bytes
                ))
            })?;
        let grid = Self::grid(spec, px, py, true, &goals);
        let bytes = grid.bytes(false);
        if bytes > wafer.pe_bytes {
            return refuse(format!(
                "{name}: the {px} x {py} grid at B {} NB {} wants {bytes} bytes a PE (of {})",
                grid.lanes, grid.blocks, wafer.pe_bytes
            ));
        }
        // The bf16 head (head2d v10): 32 dims a PE, 256 rows, the fewest
        // blocks (even) whose tables fit.
        let c = spec.hidden.div_ceil(HEAD16_HC);
        let tables = if spec.tied { 1 } else { 2 };
        let head = (1..)
            .map(|k| 2 * k)
            .take_while(|nb| nb * (c + 1) <= wafer.width)
            .find_map(|nb| {
                let vr = spec.vocab.div_ceil(nb * HEAD16_ROWS);
                (tables * 2 * HEAD16_HC * vr <= HEAD16_TABLES).then(|| Head {
                    c,
                    hc: HEAD16_HC,
                    hh: HEAD16_ROWS,
                    vr,
                    gx: group(c, 16).unwrap_or(2),
                    gy: group(HEAD16_ROWS / 2, 18).unwrap_or(2),
                    gb: group(nb, 4).unwrap_or(2),
                    blocks: nb,
                    vmc: 1,
                    untied: !spec.tied,
                    f32: false,
                })
            })
            .ok_or_else(|| Refused(format!("{name}: no head2d shape holds a {}-row vocabulary", spec.vocab)))?;
        // Where its grids would go: whole grids on the wafer beside the head.
        let (across, down) = (wafer.width / px, wafer.height / py);
        let mut blockers = vec![format!(
            "no full program places {px} x {py} grids yet (serve2d's diagonal relay also wants PX >= PY)"
        )];
        let beside = (wafer.width - head.width().min(wafer.width)) / px * down;
        if across * down < spec.layers + 1 {
            blockers.push(format!(
                "{} grids of {px} x {py} ({} PEs) and the head ({} PEs) on {} x {}: {across} x {down} = {} \
                 grids fit the whole wafer, {beside} beside a {} x {} head",
                spec.layers,
                spec.layers * px * py,
                head.width() * (head.hh + 1),
                wafer.width,
                wafer.height,
                across * down,
                head.width(),
                head.hh + 1
            ));
        }
        Ok(SpatialPlan {
            spec: spec.clone(),
            family: Family::Paged8b,
            layers: spec.layers,
            vocab: spec.vocab,
            grid,
            head,
            program: None,
            grid_bytes: bytes,
            pe_bytes: wafer.pe_bytes,
            needs: spec.needs(),
            blockers,
        })
    }

    /// The build's compile parameters, in the order the requests list them
    /// (`run.py --params`/the request's `extra`).
    pub fn params(&self) -> Vec<(&'static str, usize)> {
        let (g, h) = (&self.grid, &self.head);
        let b = |x: bool| usize::from(x);
        let mut out = vec![
            ("PX", g.px),
            ("PY", g.py),
            ("Hc", g.hc),
            ("HR", g.hidden),
            ("Dc", g.dc),
            ("Gs", g.gs),
            ("Qh", g.qh),
            ("Ic", g.ic),
        ];
        match self.family {
            Family::Paged8b => {
                out.extend([
                    ("PS", g.page_size),
                    ("E", g.entries),
                    ("W16", b(g.w16)),
                    ("NP", 3),
                    ("KV16", b(g.kv16)),
                    ("MC", g.mlp_chunk),
                    ("PROF", 0),
                    ("iters", 1),
                    ("NB", g.blocks),
                    ("B", g.lanes),
                ]);
                if g.host_copy {
                    out.push(("KVCP", 0));
                }
            }
            Family::Serve2d | Family::Serve2dPipe => {
                let pipe = self.family == Family::Serve2dPipe;
                let p = self.program.unwrap_or(Program {
                    gw: 0,
                    gh: 0,
                    cap: 0,
                    mmax: None,
                    hb: None,
                    width: 0,
                    height: 0,
                });
                out.extend([("Tc", TC), ("W16", 0)]);
                if !pipe {
                    out.push(("NP", 3));
                }
                out.push(("KV16", 0));
                if !pipe {
                    out.push(("PROF", 0));
                }
                out.push(("GRID", 1));
                if pipe {
                    out.push(("HEADMODE", 2));
                }
                out.extend([
                    ("B", g.lanes),
                    ("PS", g.page_size),
                    ("NB", g.blocks),
                    ("E", g.entries),
                    ("MC", g.mlp_chunk),
                    ("GW", p.gw),
                    ("GH", p.gh),
                ]);
                if !pipe {
                    out.push(("HEADMODE", 2));
                }
                out.extend([
                    ("F", h.hh / 2),
                    ("HH", h.hh),
                    ("Vr", h.vr),
                    ("Gx", h.gx),
                    ("Gy", h.gy),
                    ("Gb", h.gb),
                    ("vocab", self.vocab),
                    ("PASSES", 3),
                    ("UNTIED", b(h.untied)),
                    ("CERT", 0),
                    ("MODE", 0),
                    ("OVL", 0),
                    ("VMC", h.vmc),
                    ("WF32", 1),
                ]);
                if !pipe {
                    out.push(("CAP", p.cap));
                }
                out.extend([("DBG", 0), ("iters", 1)]);
                if pipe {
                    out.extend([("MMAX", p.mmax.unwrap_or(1)), ("CAP", p.cap), ("HB", p.hb.unwrap_or(1))]);
                }
                if g.host_copy {
                    out.push(("KVCP", 0));
                }
            }
        }
        out
    }

    /// The standalone bf16 head's parameters (`head2d_v10`; the 8B head).
    pub fn head_params(&self) -> Vec<(&'static str, usize)> {
        let h = &self.head;
        vec![
            ("W", h.width()),
            ("H", h.hh),
            ("C", h.c),
            ("Hc", h.hc),
            ("Vr", h.vr),
            ("Gx", h.gx),
            ("Gy", h.gy),
            ("Gb", h.gb),
            ("vocab", self.vocab),
            ("PASSES", if h.f32 { 3 } else { 1 }),
            ("UNTIED", usize::from(h.untied)),
            ("CERT", 0),
            ("MODE", if h.f32 { 0 } else { 2 }),
            ("OVL", 0),
            ("VMC", h.vmc),
            ("WF32", usize::from(h.f32)),
            ("iters", 1),
            ("SCHAIN", 0),
        ]
    }

    /// `params` as a request's `extra` string (`K:V,...`).
    pub fn extra(params: &[(&'static str, usize)]) -> String {
        params.iter().map(|(k, v)| format!("{k}:{v}")).collect::<Vec<_>>().join(",")
    }

    /// Checks a compiled program's params (its `out.json`) against this
    /// plan's structural ones: a program built for another model's dims is
    /// refused before its weights are placed.
    pub fn check_program(&self, get: impl Fn(&str) -> Option<usize>) -> Result<(), Refused> {
        let want = self.params();
        for key in ["PX", "PY", "Hc", "HR", "Dc", "Gs", "Qh", "Ic", "vocab", "UNTIED"] {
            let Some(&(_, w)) = want.iter().find(|(k, _)| *k == key) else {
                continue;
            };
            match get(key) {
                Some(v) if v == w => {}
                v => {
                    return refuse(format!(
                        "the wafer program's {key} is {v:?}; the model wants {w} ({})",
                        Self::extra(&want)
                    ));
                }
            }
        }
        if let (Some(p), Some(gw), Some(gh)) = (self.program, get("GW"), get("GH"))
            && gw * gh != p.gw * p.gh
        {
            return refuse(format!(
                "the wafer program holds {} layers; the model has {}",
                gw * gh,
                self.layers
            ));
        }
        Ok(())
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
    /// Checks a compiled program's params against this model: serve28's
    /// vocabulary (`hidden`, `heads`, `kernel_cols`, ...) or the 2-D builds'
    /// (`HR`, `Qh`, `PX`, ...), with their divisibility, and the facts the
    /// programs and the placement carry as constants (eps, q/k norms, the
    /// RoPE frequencies from theta, 1/sqrt(head_dim), a tied head).
    pub fn check_program(&self, get: impl Fn(&str) -> Option<usize>) -> Result<(), Refused> {
        let name = &self.name;
        fn no<T>(name: &str, why: String) -> Result<T, Refused> {
            refuse(format!("{name}: the wafer program: {why}"))
        }
        // What every program and today's placement assume.
        if self.rms_eps != KERNEL_EPS {
            return no(name, format!("RMSNorm eps {} (the CSL's literal is {KERNEL_EPS})", self.rms_eps));
        }
        if !self.qk_norm {
            return no(name, "no q/k RMSNorm (the kernels always norm q and k; Need::IdentityQkNorm)".into());
        }
        if self.rope_table.is_some() {
            return no(name, "scaled RoPE (the placement builds frequencies from theta; Need::RopeTable)".into());
        }
        let scale = (self.head_dim as f32).sqrt().recip();
        if (self.sm_scale - scale).abs() > 1e-6 * scale {
            return no(name, format!(
                "softmax scale {} (the placement folds 1/sqrt(head_dim) = {scale} into q_norm)",
                self.sm_scale
            ));
        }
        if !self.tied {
            return no(
                name,
                "an untied lm_head (the placement reads embed_tokens for the logits and would \
                 give wrong ones; Need::LmHeadPlacement)"
                    .into(),
            );
        }
        let p = |k: &str| get(k).ok_or_else(|| Refused(format!("{name}: the wafer program has no param {k}")));
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
            eq("GW x GH (layers)", p("GW")? * p("GH")?, self.layers)?;
            let kv_cols = divides("PX over a kv head's Gs columns", px, gs)?;
            if kv_cols < self.kv_heads {
                return no(name, format!("{kv_cols} kv heads' columns for the model's {}", self.kv_heads));
            }
            if py * hc < self.hidden || px * p("Ic")? < self.inter {
                return no(name, format!(
                    "PY x Hc {} or PX x Ic {} is short of hidden {} / inter {}",
                    py * hc,
                    px * p("Ic")?,
                    self.hidden,
                    self.inter
                ));
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
            // serve28 (serve28s/s3): attention rows of kv-head bands.
            let (cols, rows, band, mlp) = (p("kernel_cols")?, p("attn_rows")?, p("band_rows")?, p("mlp_rows")?);
            let (layers, lps) = (p("layers")?, p("lps")?);
            eq("hidden", p("hidden")?, self.hidden)?;
            eq("head_dim", p("head_dim")?, self.head_dim)?;
            eq("inter", p("inter")?, self.inter)?;
            eq("vocab", p("vocab")?, self.vocab)?;
            eq("layers", layers, self.layers)?;
            let bands = divides("attn_rows over band_rows (kv heads)", rows, band)?;
            eq("attn_rows / band_rows (kv heads)", bands, self.kv_heads)?;
            eq("heads (q heads a kv head)", p("heads")?, self.group())?;
            divides("layers over lps", layers, lps)?;
            divides("hidden over kernel_cols", self.hidden, cols)?;
            divides("head_dim over band_rows", self.head_dim, band)?;
            divides("inter over mlp_rows", self.inter, mlp)?;
            divides("positions over kernel_cols", p("positions")?, cols)?;
        }
        Ok(())
    }
}

impl SpatialPlan {
    /// [`ModelSpec::check_program`] on a program's `out.json`, then the
    /// plan's own structural params (a 2-D build for these dims).
    pub fn check(&self, out_json: &serde_json::Value) -> Result<(), Refused> {
        let get = out_json_params(out_json);
        self.spec.check_program(&get)?;
        if get("HEADMODE").is_some() {
            self.check_program(&get)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn spec(name: &str) -> Result<ModelSpec, Refused> {
        let catalog = poem_compiler::catalog::repository();
        let deployment = catalog.parse(name).unwrap_or_else(|| panic!("{name} parses"));
        ModelSpec::read(&deployment.trace(poem_ir::Platform::Cerebras))
    }

    fn map(extra: &str) -> BTreeMap<String, usize> {
        extra
            .split(',')
            .map(|kv| {
                let (k, v) = kv.split_once(':').expect("k:v");
                (k.to_string(), v.parse().expect("a number"))
            })
            .collect()
    }

    fn plan(spec: &ModelSpec, goals: Goals) -> SpatialPlan {
        SpatialPlan::derive(spec, WaferDims::CS3, goals).expect("a plan")
    }

    /// Asserts the plan's params equal a request's `extra`, key for key.
    fn same(params: &[(&'static str, usize)], extra: &str) {
        let ours = map(&SpatialPlan::extra(params));
        assert_eq!(ours, map(extra), "\nours:  {}\ntoday: {extra}", SpatialPlan::extra(params));
    }

    #[test]
    fn qwen3_dims_come_from_the_trace() {
        let s = spec("qwen3-0.6b-bf16-kv-bf16").expect("qwen3-0.6b reads");
        assert_eq!(
            (s.layers, s.hidden, s.q_heads, s.kv_heads, s.head_dim, s.inter, s.vocab),
            (28, 1024, 16, 8, 128, 3072, 151_936)
        );
        assert!(s.tied && s.qk_norm && s.rope_table.is_none());
        assert_eq!((s.rope_theta, s.rms_eps, s.weights, s.kv), (1e6, 1e-6, Dtype::Bf16, Dtype::Bf16));
        assert!((s.sm_scale - 128f32.sqrt().recip()).abs() < 1e-7);
        let s = spec("qwen3-8b-bf16-kv-bf16").expect("qwen3-8b reads");
        assert_eq!(
            (s.layers, s.hidden, s.q_heads, s.kv_heads, s.inter, s.tied),
            (36, 4096, 32, 8, 12288, false)
        );
        let s = spec("qwen3-0.6b-bf16-kv-bf16-yarn4").expect("a stretched qwen3 reads");
        assert_eq!(s.needs(), vec![Need::RopeTable]);
    }

    /// Qwen3-0.6B: serve2d's batched builds (0105, 0106) and the pipeline's
    /// (0108), from the model's dims and the batch alone.
    #[test]
    fn qwen3_0_6b_plans_todays_serve2d_builds() {
        let s = spec("qwen3-0.6b-bf16-kv-bf16").expect("qwen3-0.6b reads");
        let base = "PX:128,PY:128,Hc:8,HR:1024,Dc:8,Gs:16,Qh:2,Ic:24,Tc:32,W16:0,NP:3,KV16:0,PROF:0,\
                    GRID:1,PS:16,E:1,GW:4,GH:7,HEADMODE:2,F:396,HH:792,Vr:192,Gx:16,Gy:18,Gb:1,\
                    vocab:151936,PASSES:3,UNTIED:0,CERT:0,MODE:0,OVL:0,VMC:1,WF32:1,DBG:0,iters:1";
        // 0105's four builds: B, NB, MC, CAP.
        for (b, nb, mc, cap) in [(4, 8, 2, 64), (1, 8, 1, 64), (8, 8, 2, 64), (16, 4, 2, 32)] {
            let p = plan(&s, Goals::lanes(b));
            assert_eq!(p.family, Family::Serve2d);
            assert!(p.blockers.is_empty() && p.needs.is_empty(), "{p:?}");
            same(&p.params(), &format!("{base},B:{b},NB:{nb},MC:{mc},CAP:{cap}"));
            let prm: BTreeMap<_, _> = p.params().into_iter().collect();
            let lay = super::super::serve2d::Layout2d::of(|k| prm.get(k).copied())
                .expect("serve2d params")
                .expect("a serve2d layout");
            assert_eq!((lay.layers(), lay.head_dim()), (28, 128));
            assert!(p.check_program(|k| prm.get(k).copied()).is_ok());
        }
        let p = plan(&s, Goals::lanes(16));
        assert_eq!((p.grid.pages(), p.grid.max_context()), (512, 2048));
        assert_eq!(p.program.map(|p| (p.width, p.height)), Some((641, 896)));

        // 0108's pipelines: B, NB, MMAX, CAP, HB.
        let base = "PX:128,PY:128,Hc:8,HR:1024,Dc:8,Gs:16,Qh:2,Ic:24,Tc:32,W16:0,KV16:0,GRID:1,\
                    HEADMODE:2,PS:16,E:1,MC:2,GW:4,GH:7,F:396,HH:792,Vr:192,Gx:16,Gy:18,Gb:1,\
                    vocab:151936,PASSES:3,UNTIED:0,CERT:0,MODE:0,OVL:0,VMC:1,WF32:1,DBG:0,iters:1";
        for (b, nb, mmax, cap, hb) in [(4, 8, 8, 16, 4), (4, 8, 8, 32, 4), (8, 8, 8, 16, 8), (16, 4, 4, 16, 8)] {
            let goals = Goals {
                pipeline: true,
                cap: Some(cap),
                ..Goals::lanes(b)
            };
            let p = plan(&s, goals);
            assert_eq!(p.family, Family::Serve2dPipe);
            same(&p.params(), &format!("{base},B:{b},NB:{nb},MMAX:{mmax},CAP:{cap},HB:{hb}"));
        }
    }

    /// Qwen3-8B: 0107's paged8b layer builds (128 x 160) and 0102's head.
    #[test]
    fn qwen3_8b_plans_todays_paged8b_builds() {
        let s = spec("qwen3-8b-bf16-kv-bf16").expect("qwen3-8b reads");
        let base = "PX:128,PY:160,Hc:26,HR:4096,Dc:8,Gs:16,Qh:4,Ic:96,PS:16,E:1,W16:1,NP:3,KV16:1,\
                    MC:1,PROF:0,iters:1";
        // (B, NB pinned or not, KVCP 0): today's NB 4 at B 1 and 2 was a
        // test matrix's choice; the deepest pool would be 8 there.
        for (b, nb, auto, host) in [
            (1, 4, false, false),
            (2, 4, false, false),
            (4, 4, true, false),
            (8, 1, true, false),
            (8, 3, false, true),
        ] {
            let goals = Goals {
                blocks: (!auto).then_some(nb),
                host_copy: host,
                ..Goals::lanes(b)
            };
            let p = plan(&s, goals);
            assert_eq!(p.family, Family::Paged8b);
            assert!(p.program.is_none() && p.grid_bytes <= p.pe_bytes, "{p:?}");
            let kvcp = if host { ",KVCP:0" } else { "" };
            same(&p.params(), &format!("{base},NB:{nb},B:{b}{kvcp}"));
        }
        // Builds that did not link don't fit the model either.
        for (b, nb, host) in [(8, 2, false), (8, 4, true)] {
            let goals = Goals {
                blocks: Some(nb),
                host_copy: host,
                ..Goals::lanes(b)
            };
            assert!(SpatialPlan::derive(&s, WaferDims::CS3, goals).is_err(), "B {b} NB {nb}");
        }
        // B 9 (did not link at 160 rows) takes a taller grid.
        assert_eq!(plan(&s, Goals::lanes(9)).grid.py, 176);
        let p = plan(&s, Goals::lanes(1));
        assert_eq!(p.grid.blocks, 8);
        same(
            &p.head_params(),
            "W:516,H:256,C:128,Hc:32,Vr:149,Gx:16,Gy:16,Gb:4,vocab:151936,PASSES:1,UNTIED:1,\
             CERT:0,MODE:2,OVL:0,VMC:1,WF32:0,iters:1,SCHAIN:0",
        );
        // No full program: serve2d wants Qh 2, and 36 grids don't fit.
        assert!(p.blockers[0].contains("4 query heads a kv head"), "{:?}", p.blockers);
        assert!(p.blockers.iter().any(|b| b.contains("5 x 7 = 35 grids")), "{:?}", p.blockers);
    }

    /// Qwen3-14B: a layer grid fits a PE; 40 of them don't fit the wafer.
    #[test]
    fn qwen3_14b_plans_a_grid_that_does_not_fit_the_wafer() {
        let s = spec("qwen3-14b-bf16-kv-bf16").expect("qwen3-14b reads");
        assert_eq!((s.hidden, s.q_heads, s.kv_heads, s.inter, s.layers), (5120, 40, 8, 17408, 40));
        let p = plan(&s, Goals::lanes(4));
        assert_eq!(p.family, Family::Paged8b);
        assert_eq!((p.grid.px, p.grid.qh, p.grid.ic), (128, 5, 136));
        assert!(p.grid_bytes <= p.pe_bytes);
        assert!(p.program.is_none());
        let wafer = p.blockers.iter().find(|b| b.contains("40 grids")).expect("a wafer blocker");
        assert!(wafer.contains(&format!("{} x {}", p.grid.px, p.grid.py)), "{wafer}");
    }

    /// Llama 3.1 8B reads, without q/k norms and with llama3-scaled RoPE: it
    /// plans as a paged8b grid that needs the identity-norm variant.
    #[test]
    fn llama_3_plans_with_the_identity_qk_norm() {
        let s = spec("llama31-8b-bf16-kv-bf16").expect("llama 3.1 8b reads");
        assert!(!s.qk_norm && !s.tied);
        assert_eq!((s.hidden, s.inter, s.vocab, s.rope_theta), (4096, 14336, 128_256, 500_000.0));
        let p = plan(&s, Goals::lanes(4));
        assert_eq!(p.needs, vec![Need::IdentityQkNorm, Need::RopeTable, Need::LmHeadPlacement]);
        assert_eq!((p.family, p.grid.ic, p.grid.qh), (Family::Paged8b, 112, 4));
    }

    /// Models whose structure the kernels don't implement are refused by name.
    #[test]
    fn other_structures_are_refused() {
        for (name, why) in [
            ("qwen3-30b-a3b-bf16-kv-bf16", "mixture of experts"),
            ("qwen35-d0.8b-bf16-kv-bf16", "gated-delta"),
            ("gemma4-12b-bf16-kv-bf16", ""),
            ("qwen3-8b-u4g64-kv-bf16", "U4g64 projections"),
            ("qwen3-0.6b-bf16-kv-e4m3", "E4m3 KV"),
        ] {
            let err = spec(name).expect_err(name);
            assert!(err.0.contains(why), "{name}: {err}");
        }
    }

    fn out_json(extra: &str) -> serde_json::Value {
        let params: serde_json::Map<_, _> =
            map(extra).into_iter().map(|(k, v)| (k, serde_json::json!(v.to_string()))).collect();
        serde_json::json!({ "params": params })
    }

    /// The load check: a program for the model's dims passes, one for other
    /// dims, a broken serve28 layout or an untied model is refused.
    #[test]
    fn programs_are_checked_against_the_trace() {
        let catalog = poem_compiler::catalog::repository();
        let trace = |name: &str| catalog.parse(name).expect(name).trace(poem_ir::Platform::Cerebras);
        let small = trace("qwen3-0.6b-bf16-kv-bf16");
        let s = ModelSpec::read(&small).expect("qwen3-0.6b reads");
        let p = plan(&s, Goals::lanes(4));
        let serve2d = out_json(&SpatialPlan::extra(&p.params()));
        assert!(check(&small, &serve2d).is_ok());
        assert!(p.check(&serve2d).is_ok());
        let wrong = out_json(&SpatialPlan::extra(&p.params()).replace("vocab:151936", "vocab:100"));
        assert!(check(&small, &wrong).expect_err("vocab").0.contains("vocab is 100"));
        let serve28 = "kernel_cols:64,hidden:1024,heads:2,head_dim:128,inter:3072,positions:4096,\
                       attn_rows:128,mlp_rows:128,band_rows:16,seg_rows:32,head_rows:128,vocab_rows:1200,\
                       prompt_max:64,vocab:151936,layers:28,lps:7,iters:64";
        assert!(check(&small, &out_json(serve28)).is_ok());
        let err = check(&small, &out_json(&serve28.replace("lps:7", "lps:0"))).expect_err("lps 0");
        assert!(err.0.contains("layers over lps"), "{err}");
        let err = check(&small, &out_json(&serve28.replace("band_rows:16", "band_rows:32"))).expect_err("bands");
        assert!(err.0.contains("kv heads"), "{err}");
        // Qwen3-8B's lm_head is its own: today's placement would read the
        // embedding for it.
        let big = trace("qwen3-8b-bf16-kv-bf16");
        let err = check(&big, &serve2d).expect_err("untied");
        assert!(err.0.contains("untied lm_head"), "{err}");
    }

    /// Prints every model's plan (`--nocapture`): the table in PLAN.md.
    #[test]
    fn plans_print() {
        for (name, b) in [
            ("qwen3-0.6b-bf16-kv-bf16", 4),
            ("qwen3-8b-bf16-kv-bf16", 4),
            ("qwen3-14b-bf16-kv-bf16", 4),
            ("qwen3-32b-bf16-kv-bf16", 4),
            ("llama31-8b-bf16-kv-bf16", 4),
        ] {
            let s = spec(name).expect(name);
            match SpatialPlan::derive(&s, WaferDims::CS3, Goals::lanes(b)) {
                Ok(p) => println!(
                    "{name}: {:?} {} bytes {}/{} needs {:?}\n  head {}\n  blockers {:?}",
                    p.family,
                    SpatialPlan::extra(&p.params()),
                    p.grid_bytes,
                    p.pe_bytes,
                    p.needs,
                    SpatialPlan::extra(&p.head_params()),
                    p.blockers
                ),
                Err(e) => println!("{name}: refused: {e}"),
            }
        }
    }
}
