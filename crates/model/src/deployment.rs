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
//! `[1, kv_lora + rope]` latent row and gemma-4's two KV widths over a
//! subset of its layers are both real, both traced, and both need a pool
//! this driver does not build. The refusal is where that fact is said out
//! loud, at load, with the row printed.

use std::collections::BTreeMap;

use model_ir::plan::{CacheRow, Plan};

/// One layer's attention geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerAttention {
    /// The width of one head's k/v plane.
    pub head_dim: u32,
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

impl KvStyle {
    /// Why this build cannot provision the store, or `None` if it can.
    #[must_use]
    pub fn store_refusal(&self) -> Option<Refusal> {
        match self {
            Self::Paged => None,
            Self::Latent { .. } => Some(Refusal::Unsupported(
                "this checkpoint attends through a single-plane latent row \
                 (MLA's compressed kv, or a compressed KV plane), and this \
                 build provisions no store for one — the k/v pair the pager \
                 allocates does not fit it",
            )),
        }
    }
}

/// The recurrent (gated-delta / KDA) slabs a hybrid needs per slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecurrentShape {
    /// Which layers carry a recurrent mixer, ascending.
    pub linear_layers: Vec<u32>,
    /// Bytes-worth of elements in one layer's convolution window.
    pub conv_stride: usize,
    /// Elements in one layer's recurrent state.
    pub state_stride: usize,
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
pub struct Geometry {
    pub hidden: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
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

pub const ATTN_HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

#[must_use]
pub fn round_up_attn_head_dim(head_dim: u32) -> u32 {
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
                    let (kind, at) = name.split_once('.').and_then(|(k, _)| {
                        layer_suffix(name).map(|at| (k, at))
                    }).ok_or(Refusal::Malformed(
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
        let widths: Vec<&[u64]> = {
            let mut w: Vec<&[u64]> = kv_rows.values().copied().collect();
            w.sort_unstable();
            w.dedup();
            w
        };
        if widths.iter().any(|row| row.first() != Some(&2)) {
            return Err(Refusal::Unsupported(
                "this checkpoint attends through a single-plane latent row \
                 (MLA's compressed kv, or a compressed KV plane), and this \
                 build provisions no store for one",
            ));
        }
        if widths.len() != 1 {
            return Err(Refusal::Unsupported(
                "this checkpoint states more than one kv plane width across \
                 its layers, and this build lays out one",
            ));
        }
        if kv_rows.len() as u32 != layers && kv_rows.keys().copied().max() >= Some(layers) {
            return Err(Refusal::Malformed(
                "a kv cache row names a layer past the end of the tower",
            ));
        }

        // ── the attention statement ─────────────────────────────────────
        let decode = plan
            .ops
            .iter()
            .find(|o| o.kernel == "attention.decode" || o.kernel == "attention.decode_lse")
            .ok_or(Refusal::Unsupported(
                "the plan makes no `attention.decode` statement, so this \
                 build has no decode schedule to raise for it",
            ))?;
        let head_dim = u32::try_from(*decode.params.get(1).ok_or(Refusal::Malformed(
            "`attention.decode` states no head_dim param",
        ))?)
        .map_err(|_| Refusal::Malformed("`attention.decode` states a head_dim no u32 holds"))?;
        if head_dim == 0 {
            return Err(Refusal::Malformed("`attention.decode` states head_dim 0"));
        }
        let plane = widths[0][1];
        let kv_heads = u32::try_from(plane / u64::from(head_dim))
            .map_err(|_| Refusal::Malformed("a kv plane no u32 holds"))?;
        if kv_heads == 0 || plane % u64::from(head_dim) != 0 {
            return Err(Refusal::Malformed(
                "the kv plane is not a whole number of `head_dim`-wide heads",
            ));
        }

        // `gemm.attention_landing` binds `o_proj`, whose `[hidden, q_heads *
        // head_dim]` is the only place the query head count is stated as a
        // width rather than assumed.
        let landing = plan
            .ops
            .iter()
            .find(|o| o.kernel == "gemm.attention_landing")
            .and_then(|o| o.weights.first())
            .and_then(|w| params.get(w.as_str()))
            .ok_or(Refusal::Unsupported(
                "the plan makes no `gemm.attention_landing` statement with a \
                 weight, so the query head count is stated nowhere",
            ))?;
        let q_plane = *landing.get(1).ok_or(Refusal::Malformed(
            "the attention landing weight is not `[hidden, q_heads * head_dim]`",
        ))?;
        let q_heads = u32::try_from(q_plane / u64::from(head_dim))
            .map_err(|_| Refusal::Malformed("a query plane no u32 holds"))?;

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
            // ONE PLANE PER LAYER, each its own source. A hybrid's recurrent
            // layers get a plane they never read, which is what the legacy
            // catalog also handed the pager (`qwen_3_5/project.rs` built its
            // `LayerAttention` vector over `0..layers` unconditionally) — so
            // this is the same pool, not a new sizing decision. The layers
            // that would let it shrink are exactly the ones a shared-KV
            // family needs `kv_source` for, and that is the same edit.
            attention: (0..layers)
                .map(|l| LayerAttention {
                    head_dim,
                    kv_source: l,
                })
                .collect(),
            kv: KvStyle::Paged,
            recurrent,
            advertised,
        })
    }

    /// Refuse a GQA ratio no decode kernel in this build instantiates.
    ///
    /// # Errors
    ///
    /// A fractional ratio (malformed) or an uninstantiated one (unsupported)
    /// — distinguished, because they are different faults.
    pub fn servable_by(&self, groups: &[u32]) -> Result<(), Refusal> {
        let (q, kv) = (self.shape.q_heads, self.shape.kv_heads);
        if kv == 0 || q % kv != 0 {
            return Err(Refusal::Unsupported(
                "the query heads do not divide the kv heads, so this stack \
                 asks for a fractional GQA group no build instantiates",
            ));
        }
        if groups.contains(&self.shape.gqa_group()) {
            Ok(())
        } else {
            Err(Refusal::Unsupported(
                "this build's decode does not instantiate the GQA group size \
                 this stack asks for",
            ))
        }
    }
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
/// `conv.<l>` is `[conv_dim, conv_k - 1]` as the text declares it and
/// `delta.<l>` is `[v_heads, k_dim, v_dim]`. The conv WINDOW the driver
/// allocates is `conv_k * conv_dim` and not `(conv_k - 1) * conv_dim`,
/// because the kernel indexes `K * C` — the declaration/kernel disagreement
/// the baker backlog names as "conv slab shape". `conv_k` therefore comes
/// off `ssm.causal_conv1d`'s own param rather than off the slab, and the
/// slab's second extent is asserted against it.
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
    let [conv_dim, conv_tail] = conv else {
        return Err(Refusal::Malformed(
            "a conv state slab is not `[conv_dim, conv_k - 1]`",
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
    if conv_k != conv_tail + 1 {
        return Err(Refusal::Malformed(
            "`ssm.causal_conv1d`'s window and the conv slab's own tail \
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
        conv_stride: (conv_k * conv_dim) as usize,
        state_stride: (v_h * k_d * v_d) as usize,
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
