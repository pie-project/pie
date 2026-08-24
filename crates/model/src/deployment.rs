//! What a *driver* must know about a SKU before it can serve one: how many
//! layers of what geometry, which pools to lay out, and how wide the rows a
//! planner budgets for are.
//!
//! # There is ONE catalog, and this reads it
//!
//! `model-legacy` carried a `deployment` module of the same name whose
//! numbers were PROJECTED from a `config.json` through a per-family
//! `project.rs`. `driver-cuda` sized its KV pages, its recurrent slabs and
//! its advertised capabilities from that, while firing a `Program` traced
//! out of THIS crate — two accounts of one checkpoint, joined by a
//! `Geometry::agrees_with` check whose whole job was to catch them drifting.
//!
//! R3 deletes the second account. Every number below is read off the
//! [`Plan`] the program is built from, so the pool and the program cannot
//! disagree: the disagreement has no second source to come from. What the
//! plan does not carry — an architecture label and a context ceiling, which
//! are deployment facts and not computation facts — is stated on the
//! catalog [`Row`](crate::serve::Row) and handed in.
//!
//! # What a refusal means here
//!
//! [`Deployment::of`] refuses a plan whose pools this shape cannot describe,
//! and names what it found. That is not a claim the SKU is broken: MLA's
//! `[1, kv_lora + rope]` latent row is real, traced, and needs a pool this
//! driver does not build. The refusal is where that fact is said out loud, at
//! load, with the row printed.
//!
//! GEMMA-4'S TWO KV WIDTHS WERE ON THAT LIST AND ARE NOT ANY MORE, and what
//! moved was this function rather than the driver. The pool it feeds is
//! per-layer (`KvCacheLayout::PerLayer` carries a head-width and a kv-head
//! vector, `plan_slot` shapes each layer from its own, `layer_view` reports
//! an aliased layer's as its source's) and what it was handed was one number
//! repeated `layers` times. So the refusal was this reading, and the fix is
//! to read what the plan states: the widths are on the cache ROWS, and the
//! layer that reads a row is the layer whose statement NAMES it.

use std::collections::BTreeMap;

use model_ir::plan::{CacheRow, Plan};

/// One layer's attention geometry.
///
/// PER LAYER BECAUSE A TOWER MAY DISAGREE WITH ITSELF. gemma-4 alternates a
/// sliding-window kind with a full-attention one and the two do not share a
/// head width — e4b reads 256-wide heads through a 512-token window on 35
/// layers and 512-wide heads unwindowed on 7 — so a `[2, kv_heads *
/// head_dim]` pool row is one width on some layers and another on the rest.
/// The pool has always been per-layer (`KvCacheLayout::PerLayer`); what was
/// missing was this reading of the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerAttention {
    /// The width of one head's k/v plane.
    pub head_dim: u32,
    /// How many kv heads this layer's plane carries — the row it reads,
    /// divided by [`head_dim`](Self::head_dim).
    pub kv_heads: u32,
    /// Which layer's KV pages this layer reads. `l` for a layer that owns
    /// its own; a family that shares (gemma-4's trailing layers project no
    /// KV) names its source.
    pub kv_source: u32,
}

/// Which store this checkpoint's attention wants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KvStyle {
    /// The `[2, kv_heads * head_dim]` k/v pair a paged cache holds.
    Paged,
    /// A single-plane latent row — MLA's `[1, kv_lora + qk_rope]`, DSv4's
    /// compressed planes. Carried whole rather than parsed, because nothing
    /// in this build provisions one and a parse would be a shape nobody
    /// reads.
    Latent { row: Vec<u64> },
}

/// The recurrent (gated-delta / KDA) slabs a hybrid needs per slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentShape {
    /// Which layers carry a recurrent mixer, ascending.
    pub linear_layers: Vec<u32>,
    /// ELEMENTS in one layer's convolution window, `conv_k * conv_dim`.
    ///
    /// # The suffix is load-bearing
    ///
    /// These two were `conv_stride`/`state_stride`, and the bare name cost a
    /// measured bug: `driver-cuda/src/layout/model_costs.rs` summed them raw
    /// into a BYTE budget while `recurrent_of` filled them with ELEMENTS, so
    /// the planner reserved half the recurrent pool it then advertised slots
    /// for. `fire/launch.rs` read the same fields as elements and was right,
    /// which is why the error was invisible to every decode. A stride has no
    /// unit of its own — the slab does — so the name has to carry one, and
    /// `_elems` is what the two readers that were already correct spelled
    /// (`bind::GdnState::conv_stride_elems`, `serve::GdnState`).
    pub conv_stride_elems: usize,
    /// ELEMENTS in one layer's recurrent state, `v_h * k_d * v_d`.
    pub state_stride_elems: usize,
    /// gdn key heads.
    pub k_h: i32,
    /// gdn value heads.
    pub v_h: i32,
    /// gdn key head dim.
    pub k_d: i32,
    /// gdn value head dim.
    pub v_d: i32,
    /// The convolution's channel count.
    pub conv_dim: i32,
    /// The convolution's width, in tokens.
    pub conv_k: i32,
}

/// The widths every pool, workspace and advertised capability is sized from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
///
/// # These are the tower's WIDEST, and nothing sizes a pool from them
///
/// `head_dim`/`kv_heads` were the tower's ONE geometry when a tower could
/// only have one. A two-kind tower has two, they live per layer on
/// [`Deployment::attention`], and every pool and every attention schedule
/// reads them there. What is left for a scalar is what a scalar is for: a
/// capability to advertise, a profile key to hash, and the fallback a lane
/// that states no attention at all gets planned at. The widest is the honest
/// scalar for all three, because each of them is a BOUND.
pub struct Geometry {
    pub hidden: u32,
    pub q_heads: u32,
    /// The kv heads on the widest layer's plane — see the type's own note.
    pub kv_heads: u32,
    /// The widest head any layer attends at — see the type's own note.
    pub head_dim: u32,
    /// What a kernel allocates a head at, which is `head_dim` rounded up to
    /// a width some attention kernel instantiates.
    pub head_dim_kernel: u32,
    /// The widest feed-forward any layer asks for — a mixture's experts can
    /// be wider than a dense intermediate, and one workspace holds both.
    pub widest_mlp: u32,
    /// The LOGITS width. `embed`'s leading extent, which is the dim the
    /// sampler operates on.
    pub vocab: u32,
}

/// The head widths some attention kernel in this tree instantiates.
///
/// PRIVATE, and it is the shape of the answer that says so: nothing outside
/// this module asked which widths exist, only what a given head rounds up
/// to, and a `pub` list is an invitation to re-derive that rounding
/// somewhere else.
const ATTN_HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

fn round_up_attn_head_dim(head_dim: u32) -> u32 {
    ATTN_HEAD_DIMS
        .iter()
        .copied()
        .filter(|&d| d >= head_dim)
        .min()
        .unwrap_or(head_dim)
}

impl Geometry {
    #[must_use]
    pub const fn gqa_group(&self) -> u32 {
        match self.q_heads.checked_div(self.kv_heads) {
            Some(group) => group,
            None => 0,
        }
    }
}

/// What a deployment advertises about itself that no computation states.
///
/// Both fields are DEPLOYMENT facts rather than model facts, which is why
/// they are stated on the catalog row and not read off the plan: a trace
/// says what a layer computes, not what a scheduler may admit or what label
/// a control plane files the model under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Advertised {
    pub arch: &'static str,
    pub max_model_len: u32,
}

/// One SKU, as the thing that lays out pools sees it.
#[derive(Debug, Clone, PartialEq)]
pub struct Deployment {
    pub layers: u32,
    pub shape: Geometry,
    /// One entry per layer, in layer order.
    pub attention: Vec<LayerAttention>,
    pub kv: KvStyle,
    pub recurrent: Option<RecurrentShape>,
    pub advertised: Advertised,
}

impl Deployment {
    /// The empty deployment a shell holds between `create` and the load that
    /// fills it.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            layers: 0,
            shape: Geometry::default(),
            attention: Vec::new(),
            kv: KvStyle::Paged,
            recurrent: None,
            advertised: Advertised::default(),
        }
    }

    /// Read the deployment off `plan`, with the two stated facts `advertised`
    /// carries.
    ///
    /// # Errors
    ///
    /// A plan whose pools this shape cannot describe — every refusal names
    /// the row or the statement it found.
    pub fn of(plan: &Plan, advertised: Advertised) -> Result<Self, Refusal> {
        let params: BTreeMap<&str, &[u64]> = plan
            .params
            .iter()
            .map(|p| (p.name.as_str(), p.shape.as_slice()))
            .collect();

        let embed = params
            .get("embed")
            .ok_or(Refusal::Malformed("the plan states no `embed` weight"))?;
        let [vocab, hidden] = embed else {
            return Err(Refusal::Malformed(
                "`embed` is not the `[vocab, hidden]` table a logits width is read from",
            ));
        };

        let layers = layer_count(plan);

        // ── the KV plane ────────────────────────────────────────────────
        let mut kv_rows: BTreeMap<u32, &[u64]> = BTreeMap::new();
        let mut state_rows: BTreeMap<(&str, u32), &[u64]> = BTreeMap::new();
        for row in &plan.caches {
            match row {
                CacheRow::Kv { name, row } => {
                    let at = layer_suffix(name).ok_or(Refusal::Malformed(
                        "a kv cache row is not named `kv.<layer>`, so no layer owns it",
                    ))?;
                    kv_rows.insert(at, row.as_slice());
                }
                CacheRow::State { name, slab } => {
                    let (kind, at) = name
                        .split_once('.')
                        .and_then(|(k, _)| layer_suffix(name).map(|at| (k, at)))
                        .ok_or(Refusal::Malformed(
                            "a state cache row is not named `<kind>.<layer>`",
                        ))?;
                    state_rows.insert((kind, at), slab.as_slice());
                }
            }
        }
        if kv_rows.is_empty() {
            return Err(Refusal::Malformed(
                "the plan declares no kv cache row, so there is no attention pool to size",
            ));
        }
        if kv_rows.values().any(|row| row.first() != Some(&2)) {
            return Err(Refusal::Unsupported(
                "this checkpoint attends through a single-plane latent row \
                 (MLA's compressed kv, or a compressed KV plane), and this \
                 build provisions no store for one",
            ));
        }
        if kv_rows.len() as u32 != layers && kv_rows.keys().copied().max() >= Some(layers) {
            return Err(Refusal::Malformed(
                "a kv cache row names a layer past the end of the tower",
            ));
        }

        // ── the attention statements, LAYER BY LAYER ────────────────────
        //
        // TWO WIDTHS WAS A REFUSAL HERE, and the refusal was about this
        // function and not about the driver: the pool has been per-layer
        // since it was written (`KvCacheLayout::PerLayer` carries a
        // `head_dim` and a `kv_heads` vector and every accessor reads them),
        // and what it was handed was one number repeated. So a tower that
        // states two attention kinds — gemma-4's sliding 256 beside its
        // global 512 — is read the way it is stated.
        //
        // EVERY NUMBER OFF THE PLAN, and each off the only place that states
        // it: the head width off the statement's own params, the pool row off
        // the row the statement NAMES (which is how a shared-KV layer says
        // whose pages it reads), the kv heads off dividing the one by the
        // other, and the query heads off the landing weight's own extent.
        //
        // THE WIDEST COMES BACK FROM THE WALK that found it. This used to be
        // a second `max_by_key` over the vector `layer_attention` returns,
        // under a refusal that could not fire — a plan with no stamped
        // attention statement has already been refused inside, because the
        // unstated layers are filled FROM the widest and there is nothing to
        // fill them with.
        let (attention, widest) = layer_attention(plan, &kv_rows, layers)?;
        let (head_dim, kv_heads) = (widest.head_dim, widest.kv_heads);

        // ONE QUERY HEAD COUNT, and a tower that states two is refused rather
        // than served at the widest. The count is what a schedule's GQA group
        // is computed from and what the sampler's landing is cut at, and no
        // family in this catalog varies it per layer — gemma-4 keeps 8 across
        // both its kinds and pays for the wider head in the weight instead.
        let q_heads = q_head_count(plan, &params, &attention)?;

        // ── the recurrent slabs ─────────────────────────────────────────
        let recurrent = recurrent_of(plan, &state_rows)?;

        // ── the widest feed-forward ─────────────────────────────────────
        // Every family lands its MLP through a `down` (dense),
        // `experts_down` (routed) or `shared_down` (a mixture's always-on
        // expert) weight, and the width is that weight's INPUT extent.
        let widest_mlp = params
            .iter()
            .filter(|(name, _)| {
                let leaf = name.rsplit('.').next().unwrap_or(name);
                matches!(leaf, "down" | "shared_down")
            })
            .filter_map(|(_, shape)| shape.get(1).copied())
            .chain(
                params
                    .iter()
                    .filter(|(name, _)| name.ends_with("experts_down"))
                    .filter_map(|(_, shape)| shape.get(2).copied()),
            )
            .max()
            .unwrap_or(0);

        Ok(Deployment {
            layers,
            shape: Geometry {
                hidden: u32::try_from(*hidden)
                    .map_err(|_| Refusal::Malformed("a hidden width no u32 holds"))?,
                q_heads,
                kv_heads,
                head_dim,
                head_dim_kernel: round_up_attn_head_dim(head_dim),
                widest_mlp: u32::try_from(widest_mlp)
                    .map_err(|_| Refusal::Malformed("an mlp width no u32 holds"))?,
                vocab: u32::try_from(*vocab)
                    .map_err(|_| Refusal::Malformed("a vocab no u32 holds"))?,
            },
            attention,
            kv: KvStyle::Paged,
            recurrent,
            advertised,
        })
    }

    /// Refuse a GQA ratio no decode kernel in this build instantiates.
    ///
    /// EVERY LAYER'S, not the shape's: a tower whose two kinds disagree about
    /// the kv head count asks for two group sizes, and a check that read the
    /// widest layer's would pass a stack half of whose decodes have no kernel
    /// (gemma-4-31b is that stack — 32 q heads over 16 kv on its sliding
    /// layers and over 4 on its global ones).
    ///
    /// # Errors
    ///
    /// A fractional ratio (malformed) or an uninstantiated one (unsupported)
    /// — distinguished, because they are different faults.
    pub fn servable_by(&self, groups: &[u32]) -> Result<(), Refusal> {
        let q = self.shape.q_heads;
        for at in &self.attention {
            if at.kv_heads == 0 || !q.is_multiple_of(at.kv_heads) {
                return Err(Refusal::Unsupported(
                    "the query heads do not divide the kv heads, so this stack \
                     asks for a fractional GQA group no build instantiates",
                ));
            }
            if !groups.contains(&(q / at.kv_heads)) {
                return Err(Refusal::Unsupported(
                    "this build's decode does not instantiate the GQA group size \
                     this stack asks for",
                ));
            }
        }
        Ok(())
    }
}

/// The points that READ a kv pool row, which is what states a layer's
/// attention geometry.
///
/// Not `attention.kv_append`, which states no width; not the mla/index
/// family, whose latent rows are refused before this runs.
const ATTENDS: &[&str] = &[
    "attention.decode",
    "attention.decode_lse",
    "attention.prefill",
    "attention.prefill_lse",
    "attention.masked",
];

/// One [`LayerAttention`] per layer of the tower, read off the plan.
///
/// A layer's geometry is stated by every attention statement that stands at
/// it, and they must agree: a text whose decode and prefill arms disagreed
/// about the head width would be two models with one pool, so that is a
/// refusal rather than a first-wins.
///
/// A LAYER WITH NO ATTENDING STATEMENT keeps the tower's own geometry and
/// owns its pages. That is the hybrid case — a recurrent layer has no
/// attention and no kv row, and the legacy catalog handed the pager a plane
/// for it anyway (`qwen_3_5/project.rs` built its vector over `0..layers`
/// unconditionally), so this is the same pool rather than a new sizing
/// decision.
///
/// THE WIDEST STATED LAYER COMES BACK BESIDE THE VECTOR, because this is
/// where it is found: it is what an unstated layer is filled from, so a
/// tower with none has already been refused here and a caller re-deriving
/// it would be walking the filled vector to recover the number that filled
/// it. `Geometry`'s scalars are its two widths.
fn layer_attention(
    plan: &Plan,
    kv_rows: &BTreeMap<u32, &[u64]>,
    layers: u32,
) -> Result<(Vec<LayerAttention>, LayerAttention), Refusal> {
    let mut stated: BTreeMap<u32, LayerAttention> = BTreeMap::new();
    for op in &plan.ops {
        if !ATTENDS.contains(&op.kernel.as_str()) {
            continue;
        }
        let Some(at) = op.layer else { continue };
        let head_dim = u32::try_from(*op.params.get(1).ok_or(Refusal::Malformed(
            "an attention statement states no head_dim param",
        ))?)
        .map_err(|_| Refusal::Malformed("an attention statement states a head_dim no u32 holds"))?;
        if head_dim == 0 {
            return Err(Refusal::Malformed(
                "an attention statement states head_dim 0",
            ));
        }
        let name = op.cache.as_deref().ok_or(Refusal::Malformed(
            "an attention statement names no kv cache row, so whose pages it \
             reads is stated nowhere",
        ))?;
        let kv_source = layer_suffix(name).ok_or(Refusal::Malformed(
            "an attention statement names a kv row that is not `kv.<layer>`",
        ))?;
        let plane = *kv_rows
            .get(&kv_source)
            .ok_or(Refusal::Malformed(
                "an attention statement names a kv row the plan does not declare",
            ))?
            .get(1)
            .ok_or(Refusal::Malformed("a kv cache row states no plane width"))?;
        let kv_heads = u32::try_from(plane / u64::from(head_dim))
            .map_err(|_| Refusal::Malformed("a kv plane no u32 holds"))?;
        if kv_heads == 0 || plane % u64::from(head_dim) != 0 {
            return Err(Refusal::Malformed(
                "the kv plane is not a whole number of `head_dim`-wide heads",
            ));
        }
        let want = LayerAttention {
            head_dim,
            kv_heads,
            kv_source,
        };
        match stated.get(&at) {
            Some(held) if *held != want => {
                return Err(Refusal::Malformed(
                    "two attention statements at one layer disagree about its \
                     head width or whose pages it reads",
                ));
            }
            _ => {
                stated.insert(at, want);
            }
        }
    }
    let fallback =
        stated
            .values()
            .max_by_key(|a| a.head_dim)
            .copied()
            .ok_or(Refusal::Unsupported(
                "the plan makes no `attention.decode` statement, so this build \
             has no decode schedule to raise for it",
            ))?;
    let per_layer = (0..layers)
        .map(|l| {
            stated.get(&l).copied().unwrap_or(LayerAttention {
                kv_source: l,
                ..fallback
            })
        })
        .collect();
    Ok((per_layer, fallback))
}

/// How many query heads every layer attends with.
///
/// `gemm.attention_landing` binds `o_proj`, whose `[hidden, q_heads *
/// head_dim]` is the only place the count is stated as a width rather than
/// assumed — and the head width it divides by is THAT LAYER'S, which is why
/// this cannot read one landing and stop.
fn q_head_count(
    plan: &Plan,
    params: &BTreeMap<&str, &[u64]>,
    attention: &[LayerAttention],
) -> Result<u32, Refusal> {
    let mut held: Option<u32> = None;
    for op in plan
        .ops
        .iter()
        .filter(|o| o.kernel == "gemm.attention_landing")
    {
        let plane = *op
            .weights
            .first()
            .and_then(|w| params.get(w.as_str()))
            .ok_or(Refusal::Unsupported(
                "the plan makes no `gemm.attention_landing` statement with a \
                 weight, so the query head count is stated nowhere",
            ))?
            .get(1)
            .ok_or(Refusal::Malformed(
                "the attention landing weight is not `[hidden, q_heads * head_dim]`",
            ))?;
        let head_dim = op
            .layer
            .and_then(|l| attention.get(l as usize))
            .map_or(0, |a| a.head_dim);
        if head_dim == 0 {
            return Err(Refusal::Malformed(
                "an attention landing stands at a layer that attends nowhere",
            ));
        }
        let q_heads = u32::try_from(plane / u64::from(head_dim))
            .map_err(|_| Refusal::Malformed("a query plane no u32 holds"))?;
        if q_heads == 0 || plane % u64::from(head_dim) != 0 {
            return Err(Refusal::Malformed(
                "the attention landing is not a whole number of `head_dim`-wide heads",
            ));
        }
        match held {
            Some(was) if was != q_heads => {
                return Err(Refusal::Unsupported(
                    "this checkpoint attends with a different number of query \
                     heads on different layers, and this build states one",
                ));
            }
            _ => held = Some(q_heads),
        }
    }
    held.ok_or(Refusal::Unsupported(
        "the plan makes no `gemm.attention_landing` statement with a weight, \
         so the query head count is stated nowhere",
    ))
}

/// How deep the tower is, read off the statements' own layer column.
///
/// THE COLUMN, AND NOTHING ELSE. This used to scan `layer.39.o_proj` and
/// `kv.39` and `conv.38` for a number, because the recorder set `Op::layer`
/// on one point only and a max over it would have reported a shallow tower.
/// The recorder fills it on every statement a text's layer loop makes now,
/// so the deepest layer any statement stands at IS the tower's depth, and a
/// weight NAME is back to being a name.
///
/// A plan whose statements all stand outside a layer answers 0, and that is
/// not a silent zero: `of` refuses it two checks later, because a plan with
/// no tower still declares kv rows and every one of them then names a layer
/// past the end of it.
fn layer_count(plan: &Plan) -> u32 {
    plan.ops
        .iter()
        .filter_map(|o| o.layer)
        .max()
        .map_or(0, |top| top + 1)
}

/// The layer index a `<kind>.<layer>` cache name ends in.
///
/// THE ONE LAYER STILL READ OFF A NAME, and it is a different fact from the
/// one `Op::layer` carries. That column says where a STATEMENT stands;
/// this says which layer OWNS a declared pool row, which is a fact about
/// `caches()` and not about any statement — gemma is the proof, its
/// kv-sharing layers state `kv.{source}` and compute at a layer the row does
/// not belong to. `CacheRow` has no column for it, so the row's name is
/// where it is stated; giving the declaration one is the sibling of the
/// change that filled `Op::layer`, and it is not this one.
fn layer_suffix(name: &str) -> Option<u32> {
    name.rsplit('.').next()?.parse().ok()
}

/// The recurrent slabs, read off the state cache rows.
///
/// `conv.<l>` is `[conv_k, conv_dim]` and `delta.<l>` is
/// `[v_heads, k_dim, v_dim]` — each the rectangle its kernel indexes, slowest
/// axis first. The conv slab holds the last `conv_k` input rows oldest-first
/// (`kernels-cuda/kernels/ssm/causal_conv1d.cuh:137-139`), of which `K - 1`
/// are live between fires; the pool allocates the rectangle and not the live
/// window, so `conv_stride_elems` is the slab's own product and nothing here
/// has to add a column back.
///
/// BOTH STRIDES ARE ELEMENTS, which is what their names say and what nothing
/// here could convert anyway: a plan states shapes, not storage dtypes, so
/// the width belongs to whoever allocates the slab. `driver-cuda`'s
/// `RecurrentStateLayout` is that place — u16 for the conv window, and
/// whatever `allocate_bf16_recurrent` forces for the state.
///
/// `conv_k` IS THEREFORE STATED TWICE — once as the slab's row count, once as
/// `ssm.causal_conv1d`'s window param — and the two are held against each
/// other rather than one being derived from the other. The slab is where the
/// pool reads it and the param is where the launch reads it: a plan that
/// spelled them apart would allocate a window at one width and convolve at
/// another.
fn recurrent_of(
    plan: &Plan,
    state_rows: &BTreeMap<(&str, u32), &[u64]>,
) -> Result<Option<RecurrentShape>, Refusal> {
    let mut linear_layers: Vec<u32> = state_rows
        .keys()
        .filter(|(kind, _)| *kind == "conv")
        .map(|(_, at)| *at)
        .collect();
    if linear_layers.is_empty() {
        return Ok(None);
    }
    linear_layers.sort_unstable();

    let conv = state_rows
        .iter()
        .find(|((kind, _), _)| *kind == "conv")
        .map(|(_, slab)| *slab)
        .expect("a conv layer was just counted");
    let delta = state_rows
        .iter()
        .find(|((kind, _), _)| *kind == "delta")
        .map(|(_, slab)| *slab)
        .ok_or(Refusal::Malformed(
            "the plan declares a conv state with no recurrent state beside it",
        ))?;
    let [conv_rows, conv_dim] = conv else {
        return Err(Refusal::Malformed(
            "a conv state slab is not `[conv_k, conv_dim]`",
        ));
    };
    let [v_h, k_d, v_d] = delta else {
        return Err(Refusal::Malformed(
            "a recurrent state slab is not `[v_heads, k_dim, v_dim]`",
        ));
    };

    let conv_k = plan
        .ops
        .iter()
        .find(|o| o.kernel.starts_with("ssm.causal_conv1d"))
        .and_then(|o| o.params.first().copied())
        .ok_or(Refusal::Unsupported(
            "the plan declares a conv state but makes no `ssm.causal_conv1d` \
             statement, so its window width is stated nowhere",
        ))?;
    if conv_k != *conv_rows {
        return Err(Refusal::Malformed(
            "`ssm.causal_conv1d`'s window and the conv slab's own row count \
             disagree about how wide the convolution is",
        ));
    }
    let gd = plan
        .ops
        .iter()
        .find(|o| o.kernel.starts_with("ssm.gated_delta") || o.kernel.starts_with("ssm.kda"))
        .ok_or(Refusal::Unsupported(
            "the plan declares a recurrent state but makes no scan statement, \
             so its head counts are stated nowhere",
        ))?;
    let k_h = *gd.params.first().ok_or(Refusal::Malformed(
        "the recurrent scan states no key-head param",
    ))?;

    let num = |what: u64| -> Result<i32, Refusal> {
        i32::try_from(what).map_err(|_| Refusal::Malformed("a recurrent extent no i32 holds"))
    };
    Ok(Some(RecurrentShape {
        linear_layers,
        conv_stride_elems: (conv_k * conv_dim) as usize,
        state_stride_elems: (v_h * k_d * v_d) as usize,
        k_h: num(k_h)?,
        v_h: num(*v_h)?,
        k_d: num(*k_d)?,
        v_d: num(*v_d)?,
        conv_dim: num(*conv_dim)?,
        conv_k: num(conv_k)?,
    }))
}

/// Why a SKU cannot be deployed here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    Unsupported(&'static str),
    Malformed(&'static str),
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(what) => write!(f, "this build cannot serve it: {what}"),
            Self::Malformed(why) => write!(f, "the checkpoint contradicts its own type: {why}"),
        }
    }
}

impl std::error::Error for Refusal {}
