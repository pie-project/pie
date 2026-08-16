//! The decode geometry: the numbers a checkpoint's shape decides, in one
//! value.
//!
//! Every consumer of the batch layer sizes something from these — the heap
//! regions, the scratch slots, the launch grids, the kernel names — and the
//! C++ (`model/qwen3_5/geometry.hpp`, generic despite its path) keeps them
//! in one struct because scattering them was the defect: two numbers that
//! travel separately can be half-supplied. The stories are carried on the
//! fields they belong to.
//!
//! [`AffineFormat`] is the sharpest of them: the affine width and group are
//! one fact ("g64/b8 and g128/b4 pack to identical shapes, so the
//! checkpoint's config is the only source"), and a pipeline built for the
//! wrong pair does not fail — it reads the scales against the wrong weights
//! and returns fluent nonsense. Observed: a g64 pipeline over a g32
//! checkpoint answers token 3504, repeated. When width and group were
//! adjacent defaulted parameters, call sites passed one and let the other
//! default — twice — which compiled, bound, dispatched, and lied.
//!
//! # Where the numbers come from now
//!
//! [`geometry_from_deployment`] is the one way to build one, and it takes a
//! [`model::deployment::Deployment`] — a value with NO FAMILY NAME IN IT,
//! projected once from the `model::catalog` row the checkpoint's tensors
//! matched. What it replaces is `geometry_from_facts`, an 888-line ladder
//! over a private `ModelFacts` that a private reader had split out of a
//! `pie.model/1` descriptor into four family-prefixed blocks (`ll_*`,
//! `go_*`, `g4_*`, `q35_*`) which this file then merged back by asking
//! which block had been filled. Two halves of one document, split by family
//! and rejoined by guessing, in a crate that is forbidden to know what a
//! family is.
//!
//! The refusals survive the move because they were never about the
//! descriptor. **Every one of them is a Metal limit** — a simdgroup reduces
//! a GDN head across 32 lanes, the router ranks at most 1024 experts in one
//! threadgroup, `affine_qmv_fast` is stamped over a fixed `(group × bits)`
//! grid — and a limit does not stop being a limit because the number
//! reaching it now comes from a `const` row instead of from JSON. What
//! changed is the sentence: a refusal used to name a config key nobody had
//! written, and names a ROW's own arithmetic instead.

use model::catalog::LoadShape;
use model::deployment::{Deployment, KvStyle, RopeScaling};

/// The affine quantization's width and group: one fact, never half of it.
///
/// [`kernel_suffix`](Self::kernel_suffix) is the trailing segment shared by
/// every quantized kernel name, spelled here once instead of at every place
/// that builds one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AffineFormat {
    /// Bits per weight.
    pub bits: u32,
    /// Weights per scale/zero group.
    pub group: u32,
}

impl AffineFormat {
    /// The shipped body format: 4-bit, group 64.
    pub const G64_B4: AffineFormat = AffineFormat { bits: 4, group: 64 };

    /// `_bfloat16_gs_<group>_b_<bits>` — bf16 is not an axis; it is the one
    /// activation dtype this driver instantiates.
    #[must_use]
    pub fn kernel_suffix(self) -> String {
        format!("_bfloat16_gs_{}_b_{}", self.group, self.bits)
    }

    /// Whether the format names anything at all — the `alt_quant` absence
    /// test.
    #[must_use]
    pub const fn is_set(self) -> bool {
        self.bits != 0 && self.group != 0
    }

    /// Whether any Metal kernel is compiled to read this format.
    ///
    /// # Why the shader census is asked rather than a list kept here
    ///
    /// `quantized_qmv.metal` stamps one template over
    /// `(dtype × group × bits)`, so a format is readable exactly when the
    /// entrypoint carrying its suffix was instantiated. Asking
    /// `kernels_metal::entrypoints()` makes that a fact of the shader tree —
    /// a point added or dropped there moves this answer with it, where a list
    /// here would drift and answer for a shader that no longer exists.
    ///
    /// It reads the census rather than `KERNELS` because `quant`'s rows are
    /// RETIRED: every symbol in that family crosses, the driver resolves it
    /// through the stem the routine registry states, and the row that used to
    /// generate these names is an empty slice. `entrypoints()` is rows plus
    /// [`kernels_metal::RETIRED`], so it answers the same question across the
    /// crossing.
    ///
    /// The C++ shell refused an unreadable scheme by name at load
    /// (`heap_bind.cpp:845-890`, *"no metal kernel here reads `'<name>'`"*) and
    /// nothing did after the port. Without it the failure moves to the first
    /// fire, as the runtime compiler declining a symbol — which is loud, but
    /// arrives after the weights are staged and names a mangled entrypoint
    /// instead of the config key that chose it.
    #[must_use]
    pub fn is_readable(self) -> bool {
        if !self.is_set() {
            return false;
        }
        let suffix = self.kernel_suffix();
        // The DENSE projection, which every text names for every layer. A
        // format it cannot read is a format this driver cannot serve, whatever
        // else happens to be instantiated.
        //
        // The prefix is matched with the same stem rule the dispatcher uses --
        // the name is `affine_qmv_fast` or continues with `_` -- so a sibling
        // symbol that merely begins with those letters cannot answer for it.
        kernels_metal::entrypoints()
            .iter()
            .filter(|e| {
                e.strip_prefix("affine_qmv_fast")
                    .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
            })
            .any(|e| e.ends_with(&suffix))
    }

    /// Every format some Metal kernel reads, for a refusal that can say what
    /// the alternatives were.
    #[must_use]
    pub fn readable() -> Vec<AffineFormat> {
        let mut out = Vec::new();
        for group in [32u32, 64, 128] {
            for bits in [4u32, 8] {
                let f = AffineFormat { bits, group };
                if f.is_readable() {
                    out.push(f);
                }
            }
        }
        out
    }
}

/// The checkpoint-decided shape of the decode step.
///
/// Field meanings and defaults follow the C++; the ones that carry a story
/// keep it.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeGeometry {
    /// The model width.
    pub hidden: u32,
    /// Decoder layers.
    pub n_layers: u32,
    /// The vocabulary width the head projects onto.
    pub vocab: u32,
    /// The RMS-norm epsilon.
    pub eps: f32,
    /// The embedding and the head are one tensor.
    pub tied_embeddings: bool,
    /// Attention query heads.
    pub n_q_heads: u32,
    /// Attention key/value heads.
    pub n_kv_heads: u32,
    /// Per-head width.
    pub head_dim: u32,
    /// The body's affine format. See [`AffineFormat`].
    pub quant: AffineFormat,
    /// The format the ROUTING projections are in when it is not the body's.
    ///
    /// mlx_lm quantizes per tensor and spares the two that decide where a
    /// token goes: `mlp.gate` and `mlp.shared_expert_gate` are 8-bit inside
    /// a 4-bit checkpoint. Read as 4-bit they still produce finite,
    /// plausible numbers — the router's logits came out at cosine 0.84 to
    /// the reference and the shared gate's at cosine 1.0 with 0.56 of the
    /// magnitude: a mixture routing to almost the right experts and
    /// weighting them wrongly. Unset ([`AffineFormat::is_set`] false) when
    /// the body format covers everything.
    pub alt_quant: AffineFormat,
    /// How many leading dims of each head rotate.
    pub rotary_dims: u32,
    /// The rope base.
    pub rope_theta: f32,
    /// How often a FULL-attention layer appears in a stack that otherwise
    /// slides, or zero for a stack that does not alternate.
    pub full_attn_every: u32,
    /// The window a sliding layer attends, or zero for none.
    pub sliding_window: u32,
    /// The per-head width the FULL-attention layers use, or zero for a stack
    /// whose layers all share [`Self::head_dim`].
    ///
    /// gemma-4's `global_head_dim`. Measured on the 31b's own tensors: layer
    /// 0 (sliding) has `q_norm [256]`, layer 5 (full) has `q_norm [512]`.
    pub global_head_dim: u32,
    /// The key/value head count the FULL-attention layers use, or zero for
    /// one shape everywhere. Four on the 31b against sixteen sliding, two on
    /// the 26b against eight. See [`Self::global_head_dim`].
    pub global_kv_heads: u32,
    /// What fraction of each FULL-attention head the rotation covers, or zero
    /// for a deployment that rotates the whole head.
    ///
    /// gemma-4's `partial_rotary_factor: 0.25`. The extent reaches the GRID
    /// rather than the kernel — `Rule::Rope` launches half of it — through
    /// the rope rows' `grid_param`.
    pub full_partial_rotary: f32,
    /// The rotary base a SLIDING layer takes, when the config states a second
    /// one, or zero for a stack whose layers all share [`Self::rope_theta`].
    ///
    /// gemma-4 states both, and it is not a corner case: gemma-4-31b slides
    /// fifty of its sixty layers, so reading one base was wrong on 83% of the
    /// stack — 1e6 where the config says 1e4.
    pub rope_theta_sliding: f32,
    /// How many layers share their neighbour's KV pages
    /// (`num_kv_shared_layers`), or zero for a stack where every layer writes
    /// its own. Zero for gemma-4-31b.
    pub kv_shared_layers: u32,
    /// Whether the norm gains are stored as an OFFSET FROM ONE, so the fold
    /// is `(1 + w)` rather than `w`.
    ///
    /// gemma-1, -2 and -3 store the offset; **gemma-4 stores the multiplier**
    /// and MLX reads it with a plain `nn.RMSNorm` where the earlier three use
    /// their own `1.0 + self.weight`. `crates/model/src/gemma_4`'s text says
    /// the same thing for CUDA and has since it was written: "PLAIN, despite
    /// the family name."
    ///
    /// A FACT OF ITS OWN because it was read off the norm PLACEMENT, and
    /// `text.rs` said in as many words that this was an observation rather
    /// than a law -- "a stack that published one without the other would
    /// break here, loudly." gemma-4 is that stack, and it did not break
    /// loudly: `(1 + w)/w` is 1.002 where `w` is 444 and 1.38 where `w` is
    /// 2.6, so the norm's LARGEST value agreed with MLX to three digits
    /// while its ordinary ones were off by a third.
    pub norm_unit_offset: bool,
    /// Whether V is RMS-normed per head before it is written to the pool.
    ///
    /// gemma-4 does; nothing else here does, **including gemma-3**, which
    /// carries the per-head `q_norm`/`k_norm` and no V norm at all. So this is
    /// not implied by the QK norm and cannot be read off it.
    ///
    /// A FACT RATHER THAN A PROBE for a reason no other norm here has: MLX's
    /// `v_norm` is `RMSNormNoScale`, a module with no parameter, so the
    /// checkpoint contains nothing to ask about. `has_tensor` answers no,
    /// correctly, for a model that does it and a model that does not.
    ///
    /// Kept separate from [`Self::norm_unit_offset`] though both are read off
    /// the same gemma-4 marker: one says how a gain is stored and the other
    /// says whether a tensor is normed, and folding two statements into one
    /// flag because they agree today is how the fold got read off the
    /// sandwich.
    pub v_norm: bool,
    /// The rope RESCALING, when the row states one, or `None` for a plain
    /// geometric ladder.
    ///
    /// No `rope_theta` expresses a rescaled ladder, which is why the driver
    /// derives a TABLE and answers it as `Source::Named(<keys::RopeFrequencies as keys::Fact>::KEY)`; this is
    /// the input to that derivation and [`crate::model::rope::frequencies`]
    /// is the whole of it.
    ///
    /// It was FOUR SCALARS -- `factor`, `low`, `high`, `original_max` --
    /// which is llama-3's recipe flattened, and llama-3's alone. The row has
    /// stated a `RopeScaling` ENUM the entire time, so this field and the row
    /// held two readings of one fact and only one of them could represent a
    /// YaRN checkpoint. The other reading answered `Some(Yarn)` by refusing
    /// the load: honest, and it meant every gpt-oss reached the last reader
    /// of its config and was turned away there. The enum is the row's
    /// reading and both kinds land in it.
    ///
    /// YaRN's `attention_factor` is NOT here, for the reason
    /// [`crate::model::rope::Rescale`] gives: it scales the attention
    /// logits, not the ladder, and travels as a fact of the TEXT.
    pub rope_rescale: Option<crate::model::rope::Rescale>,
    /// GDN key heads.
    pub gdn_k_heads: u32,
    /// GDN value heads.
    pub gdn_v_heads: u32,
    /// GDN key width per head.
    pub gdn_k_dim: u32,
    /// GDN value width per head.
    pub gdn_v_dim: u32,
    /// GDN convolution taps.
    pub gdn_conv_k: u32,
    /// GDN convolution channels.
    pub gdn_conv_dim: u32,
    /// GDN value channels in total.
    pub gdn_v_total: u32,
    /// The dense FFN width.
    pub intermediate: u32,
    /// Routed experts; zero is a dense FFN. The difference between a dense
    /// and a routed decoder is these fields, not a different family.
    pub n_experts: u32,
    /// Experts each token routes to.
    pub experts_per_token: u32,
    /// One routed expert's FFN width.
    pub moe_intermediate: u32,
    /// The dense FFN every routed member runs beside the bank, under a
    /// one-scalar-per-token sigmoid gate. Zero only for a routing that has
    /// none.
    pub shared_intermediate: u32,
    /// Full attention every N layers; an interval of one (or less) makes
    /// every layer qualify. Runtime rather than constant, because the
    /// interval is a property of the checkpoint and this driver is no
    /// longer built around exactly one of them.
    pub full_attn_interval: u32,
}

impl Default for DecodeGeometry {
    /// The C++ defaults: the qwen3.5 dense shape at M=1.
    fn default() -> Self {
        DecodeGeometry {
            hidden: 1024,
            n_layers: 24,
            vocab: 248_320,
            eps: 1e-6,
            tied_embeddings: true,
            n_q_heads: 8,
            n_kv_heads: 2,
            head_dim: 256,
            quant: AffineFormat::G64_B4,
            alt_quant: AffineFormat { bits: 0, group: 0 },
            rotary_dims: 64,
            rope_theta: 1e7,
            full_attn_every: 0,
            sliding_window: 0,
            global_head_dim: 0,
            global_kv_heads: 0,
            full_partial_rotary: 0.0,
            rope_theta_sliding: 0.0,
            kv_shared_layers: 0,
            norm_unit_offset: true,
            v_norm: false,
            rope_rescale: None,
            gdn_k_heads: 16,
            gdn_v_heads: 16,
            gdn_k_dim: 128,
            gdn_v_dim: 128,
            gdn_conv_k: 4,
            gdn_conv_dim: 6144,
            gdn_v_total: 2048,
            intermediate: 3584,
            n_experts: 0,
            experts_per_token: 0,
            moe_intermediate: 0,
            shared_intermediate: 0,
            full_attn_interval: 4,
        }
    }
}

impl DecodeGeometry {
    /// Whether `layer` uses full attention rather than the linear path.
    #[must_use]
    pub fn is_full_attn(&self, layer: u32) -> bool {
        self.full_attn_interval <= 1
            || layer % self.full_attn_interval == self.full_attn_interval - 1
    }

    /// How many layers use full attention.
    #[must_use]
    pub fn full_attn_layers(&self) -> u32 {
        (0..self.n_layers)
            .filter(|&layer| self.is_full_attn(layer))
            .count() as u32
    }

    /// One GDN slot's convolution-state stride, in bytes (fp32 state).
    #[must_use]
    pub fn gdn_conv_stride_bytes(&self) -> u64 {
        u64::from(self.gdn_conv_dim) * u64::from(self.gdn_conv_k) * 4
    }

    /// One GDN slot's recurrent-state stride, in bytes (fp32 state).
    #[must_use]
    pub fn gdn_recurrent_stride_bytes(&self) -> u64 {
        u64::from(self.gdn_v_heads) * u64::from(self.gdn_v_dim) * u64::from(self.gdn_k_dim) * 4
    }

    /// Whether the FFN is a routed mixture.
    #[must_use]
    pub const fn is_moe(&self) -> bool {
        self.n_experts > 0 && self.experts_per_token > 0
    }

    /// Whether a shared expert runs beside the bank.
    #[must_use]
    pub const fn has_shared_expert(&self) -> bool {
        self.is_moe() && self.shared_intermediate > 0
    }

    /// The width one expert's gate/up produce, or the dense width.
    #[must_use]
    pub const fn ffn_width(&self) -> u32 {
        if self.is_moe() {
            self.moe_intermediate
        } else {
            self.intermediate
        }
    }

    /// Whether the routing projections live in a second affine format.
    #[must_use]
    pub const fn has_alt_quant(&self) -> bool {
        self.alt_quant.is_set()
    }
}

/// The router kernel's two hard bounds (`moe_route.metal`), mirrored so
/// the geometry that refuses an oversized stack and the launch shape read
/// the same number. The kernel clamps; a host that also clamped would
/// route with fewer experts than the row asked for and say nothing, so
/// this refuses instead.
pub const ROUTER_MAX_TOP_K: u32 = 16;
/// One lane per expert; see [`ROUTER_MAX_TOP_K`].
pub const ROUTER_MAX_EXPERTS: u32 = 1024;

/// Why a deployment did not make a geometry. The message is the whole
/// diagnosis; nothing branches on which refusal it was.
///
/// The prefix was `qwen3.5 geometry:` and was stamped on every refusal this
/// module raises. Once the schedule read stopped being gated on a family, a
/// gemma-4 checkpoint with an irregular stack was refused -- correctly, and
/// under another architecture's name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeometryRefused(pub String);

impl std::fmt::Display for GeometryRefused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "decode geometry: {}", self.0)
    }
}

impl std::error::Error for GeometryRefused {}

fn positive(v: i32) -> Option<u32> {
    u32::try_from(v).ok().filter(|&v| v > 0)
}

/// Build the geometry a catalog row's [`Deployment`] describes, or say why
/// it cannot be built.
///
/// Fills the SHAPE — all of it. `geometry_is_stated` holds this function
/// to the struct, so a field added here without a source is a failing
/// test rather than a `Default` a kernel reads as a measurement.
///
/// This doc used to say the caller set the capacity fields as well
/// (`max_tokens`, `max_requests`, `max_slots`, a page size, a page
/// count, a paged-KV flag) "because those are the OPERATOR's numbers".
/// No caller set them and nothing read them: they were one end of a
/// contract with no other end, and they are gone. The operator's real
/// numbers travel as `DriverCapabilities` and the pool's own counts,
/// which is where a scheduler already reads them.
///
/// ONE field remains unfilled — `alt_quant` — and it is not a gap for
/// want of an answer: `LoadPlan::affine_points` solves it and
/// `serve/load.rs` refuses a checkpoint that arrives at two points by
/// name. Filling the field would need a SECOND kernel set to be worth
/// anything, and `binding::observed` builds one. `geometry_is_stated`
/// names it and says so, so the gap is a sentence somebody wrote rather
/// than a default somebody inherited.
///
/// `mxfp4_experts` used to sit beside it, and it was the same slip
/// `alt_quant`'s doc was: it claimed to be "solved from the staged
/// tensors" from a struct **whose constructor is never handed them**.
/// The three arguments below are a deployment, a load shape and an
/// affine point; no tensor reaches here. The fact was already solved on
/// the other side of that line — `Loaded::mxfp4` reads the load plan,
/// `binding::observed` turns it into `MetalBinding::moe_mxfp4`, and the
/// text reads THAT to emit `WeightRepr::Mxfp4Marlin`. Two names for one
/// question, one of which could never answer it. The field is gone.
///
/// # Why three arguments and not one
///
/// `load` and `quant` are here because a `Deployment` deliberately does not
/// state them, and pretending otherwise would put a guess where a fact
/// belongs:
///
///   * [`LoadShape`] is the row's LOAD-time answer — `tied_embeddings`,
///     `kv_shared_layers`, `n_experts` — and a `Deployment` is the FIRE-time
///     one. The catalog splits them on purpose: whether the head shares the
///     embedding table changes which tensors exist, not which kernels run.
///   * `quant` is the CHECKPOINT's affine point, read from its own
///     `config.json` at load. It cannot be a row's constant because one row
///     serves the 4-bit and the 8-bit publication of the same weights —
///     `mlx-community` ships both — and those differ in nothing else.
///
/// # What this refuses that it did not have to
///
/// ONE shape is refused here that `geometry_from_facts` served, and the
/// reason is the one this file refuses for: **the fact is not in a
/// `Deployment` and this file will not invent it.**
///
///   * An IRREGULAR linear/full interleave. This driver places one full
///     layer every fixed interval and will not round an irregular stack to
///     a regular one. Reachable, and
///     `an_irregular_interleave_is_refused_rather_than_rounded` fires it.
///
/// # The two that were lifted, and how
///
/// This list said THREE until each missing fact got stated. Both lifts
/// happened the same way and neither is a relaxation — the refusals
/// named what they were missing, and the named thing arrived:
///
///   * A ROUTED stack was refused because "`Deployment` carries
///     `moe_intermediate` and `LoadShape` carries `n_experts`, and
///     neither carries the top-k." Every routed row's own facts knew it —
///     gpt-oss's `top_k`, gemma-4's `experts_per_token`, the `MoeFacts`
///     five families share. [`Geometry::experts_per_token`] is that
///     number reaching a driver, and the routed block above READS it,
///     then bounds it three ways (against zero, against the router's
///     lanes, against the expert count).
///   * TWO head dims was refused because `LayerAttention` stated a
///     per-layer `head_dim` and no per-layer KV head count, so gemma-4's
///     four-against-sixteen was unreachable. `LayerAttention::kv_heads`
///     is that count; `full_attention_shape()` reads both halves.
///
/// # What lifting the routed refusal exposed
///
/// That refusal's own text named the failure it was standing next to —
/// "the same class of failure as reading the router's 8-bit gate as
/// 4-bit ... which produced cosine 0.84 logits rather than an error."
/// The refusal is gone and THAT failure was not — routed stacks now
/// reach this driver, [`DecodeGeometry::alt_quant`] is the field the
/// second width would ride, and nothing fills it. `binding::observed`
/// builds ONE kernel set from `quant.group`/`quant.bits` for the whole
/// checkpoint, so a row publishing its gate at another width was read
/// at the stack's.
///
/// It is a REFUSAL now, and the thing that made it possible is that the
/// gap was never where it was written down. "No load path solves the
/// format from the staged tensors" was wrong: `QuantSpec` carries a
/// `group_size` and a `bits_per_element` for EVERY tensor and nothing
/// was asking. `LoadPlan::affine_points` asks, and `serve/load.rs`
/// refuses a checkpoint arriving at two points rather than reading one
/// of them at the other's width. `alt_quant` stays unset because a
/// second point needs a second kernel set to be worth filling, and
/// `a_routed_row_reads_one_width_for_every_tensor` below holds that
/// state rather than assuming it.
///
/// # Errors
///
/// [`GeometryRefused`] naming the missing or inconsistent fact.
#[allow(clippy::too_many_lines)] // one refusal ladder; splitting hides the order
pub fn geometry_from_deployment(
    d: &Deployment,
    load: LoadShape,
    quant: AffineFormat,
) -> Result<DecodeGeometry, GeometryRefused> {
    let refuse = |why: &str| Err(GeometryRefused(why.to_string()));

    // ── the decoder's own numbers ──
    //
    // One read, off one value. This used to be four family-prefixed blocks
    // (`ll_*`, `go_*`, `g4_*`, `q35_*`) copied into a fifth, each guarded by
    // "did another block already fill it" — the same question asked four
    // times because the reader below it had split one document four ways
    // behind four `model_type` tests. There is one block now and therefore
    // no order for the blocks to be wrong in.
    let Some(n_layers) = positive(i32::try_from(d.layers).unwrap_or(i32::MAX)) else {
        return refuse("the row carried no decoder shape");
    };
    let Some(hidden) = positive(i32::try_from(d.shape.hidden).unwrap_or(i32::MAX)) else {
        return refuse("layers and hidden must both be positive");
    };
    let (Some(n_q_heads), Some(n_kv_heads)) = (
        positive(i32::try_from(d.shape.q_heads).unwrap_or(i32::MAX)),
        positive(i32::try_from(d.shape.kv_heads).unwrap_or(i32::MAX)),
    ) else {
        return refuse("attention head counts must be positive");
    };
    if n_q_heads % n_kv_heads != 0 {
        return Err(GeometryRefused(format!(
            "q_heads {n_q_heads} is not a multiple of kv_heads {n_kv_heads}, \
             which GQA requires"
        )));
    }
    let Some(vocab) = positive(i32::try_from(d.shape.vocab).unwrap_or(i32::MAX)) else {
        return refuse("vocab must be positive");
    };
    // The width to ALLOCATE, which is the kernel's when one was instantiated
    // wider than the checkpoint's: phi-3's 96-wide heads run on the 128-wide
    // kernel, so a buffer sized `heads * head_dim` is short by a third.
    // `Geometry::head_dim_alloc` is the row's own answer to that and this
    // takes it rather than re-deriving a rounding rule the catalog already
    // owns.
    let head_dim = match positive(i32::try_from(d.shape.head_dim_alloc()).unwrap_or(i32::MAX)) {
        Some(head_dim) => head_dim,
        None => {
            let derived = hidden / n_q_heads;
            if derived == 0 {
                return refuse("the row states no head dim and hidden/q_heads is not positive");
            }
            derived
        }
    };

    // The per-layer table is what every question below reads, so a stack
    // whose table is a different length than its layer count is refused
    // before anything indexes it. A `Deployment` is projected by hand, once
    // per row, and a row that builds its table with the wrong bound would
    // otherwise be discovered by whichever consumer indexed furthest.
    if d.attention.len() != n_layers as usize {
        return Err(GeometryRefused(format!(
            "the row states {n_layers} layers and {} per-layer attention \
             rows; every fact below is read off that table",
            d.attention.len()
        )));
    }

    // ── the pool this build has ──
    //
    // An `unimplemented` ARM rather than a silent success. `KvStyle` is an
    // enum precisely so a shape with no pool behind it is a compile-time
    // hole in a match: the MLA lineage used to register in `FACTS_ROWS`,
    // answer its derivation happily, and have no forward path at all — it
    // loaded and died at its first fire.
    match d.kv {
        KvStyle::Paged => {}
        KvStyle::Mla { .. } | KvStyle::CompressedPlane { .. } => {
            return refuse(
                "this row needs a LATENT KV pool and this driver provisions \
                 paged K/V pages only; the absorbed projections have no metal \
                 kernel here",
            );
        }
    }

    // ── the linear-attention block ──
    //
    // A stack with no recurrent slab has no linear layers, so there is no
    // block to state and demanding one refuses every llama-like row for
    // lacking a thing it correctly does not have. The refusals below still
    // apply to a stack that DOES interleave — which is the distinction that
    // matters: absent because there is none, versus absent because the row
    // is short.
    let (gdn_k_heads, gdn_v_heads, gdn_k_dim, gdn_v_dim, gdn_conv_k) = match &d.recurrent {
        Some(r) => {
            let (Some(k_heads), Some(v_heads), Some(k_dim), Some(v_dim)) = (
                positive(r.k_h),
                positive(r.v_h),
                positive(r.k_d),
                positive(r.v_d),
            ) else {
                return refuse(
                    "the recurrent slab needs positive k_h, v_h, k_d and v_d; \
                     the conv and state strides are computed from them and a \
                     wrong stride reads one head's state as another's",
                );
            };
            let Some(conv_k) = positive(r.conv_k) else {
                return refuse("the recurrent slab needs a positive conv_k");
            };
            (k_heads, v_heads, k_dim, v_dim, conv_k)
        }
        None => (1, 1, 32, 32, 1),
    };
    if gdn_v_heads % gdn_k_heads != 0 {
        return Err(GeometryRefused(format!(
            "linear v_h {gdn_v_heads} is not a multiple of k_h {gdn_k_heads}, \
             which the GDN's head repeat requires"
        )));
    }
    if gdn_k_dim % 32 != 0 {
        return Err(GeometryRefused(format!(
            "linear k_d {gdn_k_dim} is not a multiple of 32; the GDN core \
             reduces a head across one simdgroup's lanes"
        )));
    }
    if gdn_k_dim / 32 > 8 {
        return Err(GeometryRefused(format!(
            "linear k_d {gdn_k_dim} exceeds the 256 the GDN core's per-lane \
             registers hold"
        )));
    }
    // DERIVED here and STATED by the row, which is why they are compared
    // rather than one of them taken: the convolution runs over q, k and v
    // concatenated, so `conv_dim` is `2*k_h*k_d + v_h*v_d` and nothing else.
    // A row whose stated channel count disagrees with its own head counts is
    // a row that has been edited on one line, and the conv stride computed
    // from either number would be plausible.
    let gdn_conv_dim = 2 * gdn_k_heads * gdn_k_dim + gdn_v_heads * gdn_v_dim;
    if let Some(r) = &d.recurrent
        && positive(r.conv_dim) != Some(gdn_conv_dim)
    {
        return Err(GeometryRefused(format!(
            "the recurrent slab states conv_dim {} and its own head counts \
             give 2*{gdn_k_heads}*{gdn_k_dim} + {gdn_v_heads}*{gdn_v_dim} = \
             {gdn_conv_dim}; the conv stride is computed from this and either \
             number produces a plausible one",
            r.conv_dim
        )));
    }

    // WHICH layers take the linear path, as a period rather than a list.
    //
    // The row states the list (`RecurrentShape::linear_layers`) and this
    // driver's kernels take an interval, so the list is reduced to one and
    // an irregular list is REFUSED rather than rounded. A rounded interval
    // would fire the linear kernel on a full-attention layer's weights,
    // which binds, dispatches, and answers.
    let full_attn_interval = match &d.recurrent {
        None => 1,
        Some(r) => {
            let linear: std::collections::BTreeSet<u32> = r.linear_layers.iter().copied().collect();
            if linear.is_empty() {
                return refuse(
                    "the row states a recurrent slab and no linear layers; \
                     which layers take the linear path cannot be guessed",
                );
            }
            // The complement is the full-attention set, and this driver
            // places one at the END of every period — `is_full_attn` reads
            // `layer % interval == interval - 1`.
            let full: Vec<u32> = (0..n_layers).filter(|l| !linear.contains(l)).collect();
            let Some(&first) = full.first() else {
                return refuse(
                    "every layer of this row takes the linear path; this \
                     driver's decode has no all-recurrent stack",
                );
            };
            let interval = first + 1;
            let regular = n_layers % interval == 0
                && full.len() as u32 == n_layers / interval
                && full.iter().all(|l| (l + 1) % interval == 0);
            if !regular {
                return Err(GeometryRefused(format!(
                    "the row's linear layers place full attention at {full:?}, \
                     which is not one every fixed interval; this driver will \
                     not round an irregular stack to a regular one"
                )));
            }
            interval
        }
    };

    // ── the sliding schedule ──
    //
    // `full_attn_every` and `full_attn_interval` were ONE field fed from one
    // read, and the comment that made them one said they "differ only in
    // what the OTHER layers do — a window here, the linear path there". They
    // have separate evidence now and are derived separately: the interval
    // above comes from the recurrent slab's layer list, and the period below
    // comes from the per-layer WINDOWS. A stack can have both (a hybrid that
    // also slides) and could not say so before.
    let windowed: Vec<u32> = (0..n_layers)
        .filter(|&l| d.attention[l as usize].window >= 0)
        .collect();
    let (full_attn_every, sliding_window) = if windowed.is_empty() {
        (0, 0)
    } else {
        let full: Vec<u32> = (0..n_layers)
            .filter(|&l| d.attention[l as usize].window < 0)
            .collect();
        let Some(&first) = full.first() else {
            return refuse(
                "every layer of this row slides and none attends the whole \
                 context; this driver's pool sizes the full layers first",
            );
        };
        let every = first + 1;
        let regular = n_layers % every == 0
            && full.len() as u32 == n_layers / every
            && full.iter().all(|l| (l + 1) % every == 0);
        if !regular {
            return Err(GeometryRefused(format!(
                "the row's windows place full attention at {full:?}, which is \
                 not one every fixed period; the KV pool sizes a page from \
                 this and an irregular schedule pages the wrong layers"
            )));
        }
        // One window, because one number reaches the kernel. A stack whose
        // sliding layers disagree about how far back they see is refused
        // rather than served at the first one's width.
        let widths: std::collections::BTreeSet<i32> = windowed
            .iter()
            .map(|&l| d.attention[l as usize].window)
            .collect();
        if widths.len() > 1 {
            return Err(GeometryRefused(format!(
                "the row's sliding layers state {} distinct windows ({widths:?}); \
                 one width reaches the kernel",
                widths.len()
            )));
        }
        let window = u32::try_from(d.attention[windowed[0] as usize].window).unwrap_or(0);
        (every, window)
    };

    // ── the rotary bases ──
    //
    // At most two: the full layers' and the sliding layers'. gemma-4-31b
    // slides fifty of its sixty layers, so reading ONE base was wrong on 83%
    // of the stack — 1e6 where the config says 1e4 — and nothing failed,
    // because a wrong theta is exactly right at position zero and drifts
    // from there. A stack that states three is refused rather than served at
    // two of them.
    let theta_at = |l: u32| d.attention[l as usize].rope_theta;
    let full_theta = (0..n_layers)
        .find(|&l| d.attention[l as usize].window < 0)
        .map_or_else(|| theta_at(0), theta_at);
    let sliding_theta = windowed.first().map_or(0.0, |&l| theta_at(l));
    let thetas: std::collections::BTreeSet<u32> =
        (0..n_layers).map(|l| theta_at(l).to_bits()).collect();
    if thetas.len() > 2 {
        return Err(GeometryRefused(format!(
            "the row states {} distinct rope bases across its layers; this \
             driver carries the full layers' and the sliding layers' and has \
             nowhere to put a third",
            thetas.len()
        )));
    }

    // ── the mixture, and the bounds its router has ──
    //
    // Ordered so the two ROUTER bounds stay reachable: a row that exceeds
    // them is refused by the bound it exceeds and not by the shape check
    // below, because the bound is the more specific diagnosis and because a
    // limit that is never reported is a limit nobody can act on.
    let routed = load.n_experts > 0 || d.shape.moe_intermediate > 0;
    if routed {
        if load.n_experts > ROUTER_MAX_EXPERTS {
            return Err(GeometryRefused(format!(
                "n_experts {} exceeds the {ROUTER_MAX_EXPERTS} a single \
                 threadgroup can rank",
                load.n_experts
            )));
        }
        if load.n_experts == 0 || d.shape.moe_intermediate == 0 {
            return Err(GeometryRefused(format!(
                "this row states n_experts {} and moe_intermediate {}; a \
                 routed FFN is both numbers or neither",
                load.n_experts, d.shape.moe_intermediate
            )));
        }
        // The top-k used to be refused here, and the refusal named the
        // field it was missing: "`LoadShape` counts the experts and
        // `Geometry` gives one expert's width, but how many of them a
        // token visits is stated by no value a driver receives." Every
        // routed row's own facts knew it -- gpt-oss's `top_k`, gemma-4's
        // `experts_per_token`, the `MoeFacts` five families share -- and
        // none of it reached a driver. `Geometry::experts_per_token` is
        // that statement; this reads it rather than guessing.
        if d.shape.experts_per_token == 0 {
            return Err(GeometryRefused(format!(
                "this row states {} experts and a top-k of zero; a router \
                 selecting nothing produces no MLP at all",
                load.n_experts
            )));
        }
        if d.shape.experts_per_token > ROUTER_MAX_TOP_K {
            return Err(GeometryRefused(format!(
                "top-k {} exceeds the {ROUTER_MAX_TOP_K} lanes the router \
                 ranks in one pass",
                d.shape.experts_per_token
            )));
        }
        if d.shape.experts_per_token > load.n_experts {
            return Err(GeometryRefused(format!(
                "this row routes each token to {} of {} experts, which is \
                 more experts than it has",
                d.shape.experts_per_token, load.n_experts
            )));
        }
    } else if d.shape.intermediate == 0 {
        return refuse("a dense FFN needs an intermediate width");
    }

    // ── the second attention shape ──
    //
    // Measured on `mlx-community/gemma-4-31b-it-4bit`'s own tensors, layer 0
    // (sliding) against layer 5 (full):
    //
    // | | sliding | full |
    // |---|---|---|
    // | `q_norm` | `[256]` | `[512]` |
    // | `q_proj` | `[8192, ...]` = 32x256 | `[16384, ...]` = 32x512 |
    // | `k_proj` | `[4096, ...]` = 16x256 | `[2048, ...]` = 4x512 |
    //
    // BOTH halves are needed and `Deployment` used to state one. This was a
    // REFUSAL of the whole gemma-4 generation, and it named its own missing
    // piece: `LayerAttention` carried a per-layer `head_dim`, so the 512 was
    // reachable, and there was no per-layer kv-head count anywhere, so the
    // 4-against-16 was not. A pool sizes a page from `(kv_heads, head_dim)`
    // per layer, and taking the sliding layers' sixteen for a full layer
    // reads three quarters past the end of its K — not a crash, a fluent
    // model reading another layer's memory. `LayerAttention::kv_heads` is
    // that count, so the shape is now READ rather than declined.
    //
    // Zero in both fields means "one shape everywhere", which is what
    // `head_dim_at`/`kv_heads_at` already read them as, so every row whose
    // layers agree passes through unchanged.
    let (global_head_dim, global_kv_heads, full_partial_rotary) = match d.full_attention_shape() {
        Some((hd, kv, rot)) if hd > 0 => (
            hd,
            kv,
            // A FRACTION here and an extent there, because the grid
            // launches half of it. `Rule::Rope`'s own derivation is
            // `max(2, 2 * int(0.5 * f * d))`, so handing back the ratio
            // the row spelled as an extent round-trips exactly.
            rot as f32 / hd as f32,
        ),
        _ => (0, 0, 0.0),
    };
    if global_kv_heads > 0 && n_q_heads % global_kv_heads != 0 {
        return Err(GeometryRefused(format!(
            "the full-attention layers state {global_kv_heads} kv heads, which does not \
             divide {n_q_heads} q heads; GQA has no grouping for that"
        )));
    }

    // Can any Metal kernel READ this checkpoint's weights?
    //
    // The C++ shell asked this at load and refused by name
    // (`heap_bind.cpp:845-890`: *"no metal kernel here reads '<name>'"*).
    // Nothing asked after the port, so an unreadable scheme travelled all the
    // way to the first fire and surfaced as the runtime compiler declining a
    // mangled symbol -- loud, but after the weights are staged, and naming
    // `affine_qmv_fast_bfloat16_gs_128_b_8` rather than the two config keys
    // that chose it.
    //
    // Asked of the TABLE rather than a list here, so a point added or dropped
    // in `kernels-metal` moves this answer with it.
    //
    // Scope, measured: `affine_qmv_fast` is stamped over the whole
    // `(group x bits)` grid, so this does not catch a narrow kernel table --
    // it catches a checkpoint whose numbers are off the axes ENTIRELY, a
    // group or bit width nothing was ever stamped for. That is the case the
    // C++ refusal existed for.
    if quant.is_set() && !quant.is_readable() {
        return Err(GeometryRefused(format!(
            "this checkpoint states group_size {} at {} bits and no metal \
             kernel here reads it -- `affine_qmv_fast` is instantiated at {}. \
             Refused at the geometry rather than at the first fire, where it \
             would surface as a missing symbol after the weights are staged",
            quant.group,
            quant.bits,
            AffineFormat::readable()
                .iter()
                .map(|f| format!("g{}/b{}", f.group, f.bits))
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }

    // ── the rope RESCALING ──
    //
    // Four numbers this driver's decode ladder needs and no `rope_theta`
    // expresses. They used to arrive on the `pie.model/1` descriptor as
    // `ll_rope_*`; when the descriptor went, `DecodeGeometry` kept the fields
    // and nothing filled them, so every llama-3 would have run with a factor
    // of zero — which this file reads as "no rescaling". A llama-3 does not
    // FAIL that way. It attends past its trained 8192 with the wrong
    // wavelengths and degrades fluently, which is the defect the catalog
    // exists to make unrepresentable. The row states it now.
    let rope_rescale = match d.rope_scaling {
        Some(RopeScaling::Piecewise {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position,
        }) => Some(crate::model::rope::Rescale::Piecewise {
            factor,
            low: low_freq_factor,
            high: high_freq_factor,
            original_max: original_max_position as f32,
        }),
        // A TRANSLATION now, and it was a refusal.
        //
        // The refusal was right for as long as it stood: the reader before
        // it asked `kind == "llama3"` and gave everything else a factor of
        // zero, which is indistinguishable from a model that rescales
        // nothing, so a YaRN checkpoint was served silently wrong. Declining
        // is better than that. But it named exactly one missing thing -- the
        // YaRN derivation -- and `model::rope::Rescale::Yarn` is now it, so
        // there is nothing left to decline.
        //
        // `attention_factor` is stated by the row and read nowhere here. It
        // is not a property of the ladder; see the field's doc.
        Some(RopeScaling::Yarn {
            factor,
            beta_fast,
            beta_slow,
            original_max_position,
            truncate,
            attention_factor: _,
        }) => Some(crate::model::rope::Rescale::Yarn {
            factor,
            beta_fast,
            beta_slow,
            original_max: original_max_position as f32,
            truncate,
        }),
        None => None,
    };

    Ok(DecodeGeometry {
        n_layers,
        hidden,
        vocab,
        eps: d.norm_eps,
        n_q_heads,
        n_kv_heads,
        head_dim,
        tied_embeddings: load.tied_embeddings,
        // Zero means the checkpoint declared no quantization, which is a
        // dense checkpoint; `DecodeGeometry::default`'s G64_B4 stands there
        // rather than a `gs_0_b_0` symbol no shader exports.
        quant: if quant.is_set() {
            quant
        } else {
            DecodeGeometry::default().quant
        },
        // How many leading dims of each head rotate. Zero in a
        // `LayerAttention` means "the whole head", and zero here means the
        // same, so the row's convention passes through unchanged.
        rotary_dims: d.attention[0].rotary_dim,
        rope_theta: full_theta,
        rope_theta_sliding: sliding_theta,
        rope_rescale,
        full_attn_every,
        sliding_window,
        full_attn_interval,
        // STATED by the row, not defaulted. `DecodeGeometry::default`
        // carries `true` — the answer for gemma-1, -2 and -3 — so letting
        // this fall through the struct-update below would have served
        // gemma-4 with a `(1 + w)` fold its checkpoint never asked for.
        norm_unit_offset: d.norm_unit_offset,
        // STATED for the same reason and unaskable of the tensors:
        // a weightless norm leaves nothing in the checkpoint.
        v_norm: d.v_norm,
        kv_shared_layers: load.kv_shared_layers,
        intermediate: d.shape.intermediate,
        gdn_k_heads,
        gdn_v_heads,
        gdn_k_dim,
        gdn_v_dim,
        gdn_conv_k,
        gdn_conv_dim,
        // DERIVED, not read: the value total is what the out projection
        // consumes, so neither a row nor this driver can state it
        // inconsistently with the head counts.
        gdn_v_total: gdn_v_heads * gdn_v_dim,
        // READ, now that the row states the top-k. These were zeroed with
        // a comment saying the mixture "is refused above" — true while it
        // was, and the whole reason a fallthrough default is dangerous:
        // `is_moe()` reads both counts, so a zero here does not fail, it
        // silently serves a mixture as a dense stack.
        n_experts: load.n_experts,
        experts_per_token: d.shape.experts_per_token,
        moe_intermediate: d.shape.moe_intermediate,
        // A shared expert is a whole FFN every token runs beside the bank,
        // so a wrong width here is not a slowdown — it is an addend that
        // is either missing or misread. The row states it; nothing about
        // it is derivable, which is why the two proxies a driver could
        // have reached for (`moe_intermediate`, or zero) are both wrong
        // for some shipped row.
        shared_intermediate: d.shape.shared_intermediate,
        // READ from the per-layer table above, and zero when it is uniform.
        global_head_dim,
        global_kv_heads,
        full_partial_rotary,
        // The MLP gate is deliberately NOT copied here. It reaches the
        // text as the kernel the PLAN names — `g4_geglu` is a gemma-4 op,
        // not a flag a shared emitter branches on — so a `gelu_gate: bool`
        // beside it was a second copy of a live number with no reader.
        // `geometry_is_stated` is what said so, the moment the load-path
        // probe that had been its only consumer went to `crates/model`.
        ..DecodeGeometry::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use model::deployment::{
        Advertised, AttnOutput, Geometry, LayerAttention, NormPlacement, PrefillStyle,
        RecurrentShape, Towers,
    };

    /// A dense llama-shaped stack, the case every other fixture edits.
    ///
    /// Built by hand rather than taken from a catalog row on purpose: a
    /// refusal test whose input comes from the catalog can only reach the
    /// refusals some published checkpoint happens to trip, and every bound
    /// below is a METAL limit that no shipped model reaches — which is
    /// exactly why they were never caught by a checkpoint and had to be
    /// stated as tests.
    fn dense(layers: u32) -> Deployment {
        Deployment {
            layers,
            norm_eps: 1e-5,
            mlp_gate: model::deployment::MlpGate::Silu,
            // A dense fixture: no router, so nothing reads it. It is on
            // the deployment for `driver-cuda`'s launch, which is the
            // only backend that takes the routing convention through
            // this struct — a Metal text carries it as a word of its own
            // `RouterParams`.
            norm_topk_prob: true,
            routed_scaling: 1.0,
            shape: Geometry {
                hidden: 2048,
                q_heads: 32,
                kv_heads: 8,
                head_dim: 64,
                head_dim_kernel: 64,
                intermediate: 8192,
                moe_intermediate: 0,
                experts_per_token: 0,
                shared_intermediate: 0,
                vocab: 128_256,
            },
            attention: (0..layers)
                .map(|l| LayerAttention {
                    head_dim: 64,
                    kv_heads: 8,
                    window: -1,
                    kv_source: l,
                    sm_scale: 0.125,
                    rope_theta: 500_000.0,
                    rotary_dim: 0,
                    q_gate: false,
                })
                .collect(),
            kv: KvStyle::Paged,
            recurrent: None,
            prefill: PrefillStyle::Planned,
            attn_output: AttnOutput::StatedArgs,
            logit_softcap: 0.0,
            // Nor an attention cap: gemma-2 is the one family that
            // states one and no Metal projection serves it.
            attn_logit_softcap: 0.0,
            ple_dim: 0,
            norm: NormPlacement::Pre,
            norm_unit_offset: false,
            v_norm: false,
            scales: std::collections::BTreeMap::new(),
            advertised: Advertised::default(),
            rope_scaling: None,
            towers: Towers::default(),
        }
    }

    /// The slab a hybrid stack states, sized so every bound below passes.
    fn slab(linear_layers: Vec<u32>) -> RecurrentShape {
        RecurrentShape {
            linear_layers,
            conv_stride: 0,
            state_stride: 0,
            state_elem: 4,
            k_h: 16,
            v_h: 32,
            k_d: 128,
            v_d: 128,
            conv_dim: 2 * 16 * 128 + 32 * 128,
            conv_k: 4,
            // A gated-delta fixture; B/C groups are mamba's.
            n_groups: 0,
        }
    }

    fn shape(layers: u32) -> LoadShape {
        LoadShape::dense(layers, 64, true)
    }

    fn why(d: &Deployment, load: LoadShape) -> String {
        geometry_from_deployment(d, load, AffineFormat::G64_B4)
            .expect_err("this fixture is built to be refused")
            .0
    }

    #[test]
    fn a_dense_stack_projects_every_number_the_kernels_size_from() {
        let g = geometry_from_deployment(&dense(16), shape(16), AffineFormat::G64_B4)
            .expect("a llama-shaped row is the ordinary case");
        assert_eq!((g.n_layers, g.hidden, g.vocab), (16, 2048, 128_256));
        assert_eq!((g.n_q_heads, g.n_kv_heads, g.head_dim), (32, 8, 64));
        assert!(
            g.tied_embeddings,
            "`LoadShape` states the tie, not `Deployment`"
        );
        assert_eq!(g.intermediate, 8192);
        // The rope base, which is the number nothing read for the whole life
        // of `DecodeGeometry::default`: a llama-3.2 config says 500000 and
        // the geometry answered 1e7. Position zero is right either way,
        // which is why the reference gate agreed while it was broken.
        assert_eq!(g.rope_theta, 500_000.0);
        assert_eq!(g.rope_theta_sliding, 0.0, "one base means no second one");
        // No slab and no windows: the two schedules are both "every layer".
        assert_eq!((g.full_attn_interval, g.full_attn_every), (1, 0));
        assert_eq!(g.full_attn_layers(), 16);
        assert!(!g.is_moe());
        assert_eq!(g.ffn_width(), 8192);
    }

    /// The kernel's width, not the checkpoint's, when they differ.
    #[test]
    fn a_padded_head_dim_allocates_at_the_kernels_width() {
        let mut d = dense(4);
        // phi-3: 96-wide heads on the 128-wide kernel. A buffer sized at 96
        // is short by a third of every head.
        d.shape.head_dim = 96;
        d.shape.head_dim_kernel = 128;
        let g = geometry_from_deployment(&d, shape(4), AffineFormat::G64_B4)
            .expect("a padded head is served, not refused");
        assert_eq!(g.head_dim, 128);
    }

    #[test]
    fn a_stack_with_no_layers_is_refused_before_anything_indexes_it() {
        assert!(why(&dense(0), shape(0)).contains("no decoder shape"));
    }

    #[test]
    fn the_positivity_refusals_name_the_number_that_was_zero() {
        let mut d = dense(4);
        d.shape.hidden = 0;
        assert!(why(&d, shape(4)).contains("layers and hidden"));

        let mut d = dense(4);
        d.shape.q_heads = 0;
        assert!(why(&d, shape(4)).contains("attention head counts"));

        let mut d = dense(4);
        d.shape.kv_heads = 0;
        assert!(why(&d, shape(4)).contains("attention head counts"));

        let mut d = dense(4);
        d.shape.vocab = 0;
        assert!(why(&d, shape(4)).contains("vocab must be positive"));
    }

    #[test]
    fn a_head_count_that_gqa_cannot_group_is_refused() {
        let mut d = dense(4);
        d.shape.q_heads = 12;
        d.shape.kv_heads = 8;
        let why = why(&d, shape(4));
        assert!(
            why.contains("not a multiple") && why.contains("GQA"),
            "{why}"
        );
    }

    /// A row that states no head dim at all falls back to `hidden/q_heads`,
    /// and a row that cannot even do that is refused rather than divided.
    #[test]
    fn a_stack_with_no_head_dim_derives_one_or_is_refused() {
        let mut d = dense(4);
        d.shape.head_dim = 0;
        d.shape.head_dim_kernel = 0;
        let g = geometry_from_deployment(&d, shape(4), AffineFormat::G64_B4)
            .expect("hidden/q_heads is a head dim");
        assert_eq!(g.head_dim, 2048 / 32);

        d.shape.q_heads = 4096;
        d.shape.kv_heads = 4096;
        assert!(why(&d, shape(4)).contains("no head dim"));
    }

    /// The per-layer table's length is the bound every read below uses.
    #[test]
    fn a_short_per_layer_table_is_refused_rather_than_indexed() {
        let mut d = dense(8);
        d.attention.truncate(5);
        let why = why(&d, shape(8));
        assert!(why.contains("8 layers and 5 per-layer"), "{why}");
    }

    /// A latent KV shape has no pool here, and the match is exhaustive so a
    /// new `KvStyle` cannot be added without answering this.
    #[test]
    fn a_latent_kv_shape_is_refused_because_this_driver_pages_k_and_v() {
        let mut d = dense(4);
        d.kv = KvStyle::Mla {
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
        };
        assert!(why(&d, shape(4)).contains("LATENT KV pool"));

        let mut d = dense(4);
        d.kv = KvStyle::CompressedPlane {
            ratios: vec![-1, 2, 2, 2],
        };
        assert!(why(&d, shape(4)).contains("LATENT KV pool"));
    }

    #[test]
    fn a_hybrid_stack_projects_its_slab_and_its_interval() {
        let mut d = dense(8);
        // Full attention at 3 and 7, linear everywhere else: interval 4.
        d.recurrent = Some(slab(vec![0, 1, 2, 4, 5, 6]));
        let g = geometry_from_deployment(&d, shape(8), AffineFormat::G64_B4)
            .expect("a regular interleave is the shape this driver serves");
        assert_eq!(g.full_attn_interval, 4);
        assert!(g.is_full_attn(3) && g.is_full_attn(7));
        assert!(!g.is_full_attn(0) && !g.is_full_attn(4));
        assert_eq!((g.gdn_k_heads, g.gdn_v_heads), (16, 32));
        assert_eq!((g.gdn_k_dim, g.gdn_v_dim, g.gdn_conv_k), (128, 128, 4));
        assert_eq!(g.gdn_v_total, 32 * 128);
        assert_eq!(g.gdn_conv_dim, 2 * 16 * 128 + 32 * 128);
        assert_eq!(g.gdn_conv_stride_bytes(), u64::from(g.gdn_conv_dim) * 4 * 4);
        assert_eq!(g.gdn_recurrent_stride_bytes(), 32 * 128 * 128 * 4);
    }

    /// Every number the conv and state strides are computed from, refused
    /// one at a time. A wrong stride here reads one head's state as
    /// another's — a fluent model with the wrong memory, not a crash.
    #[test]
    fn a_slab_missing_any_of_its_four_head_numbers_is_refused() {
        for edit in [
            |s: &mut RecurrentShape| s.k_h = 0,
            |s: &mut RecurrentShape| s.v_h = 0,
            |s: &mut RecurrentShape| s.k_d = 0,
            |s: &mut RecurrentShape| s.v_d = -1,
        ] {
            let mut d = dense(8);
            let mut s = slab(vec![0, 1, 2, 4, 5, 6]);
            edit(&mut s);
            d.recurrent = Some(s);
            assert!(why(&d, shape(8)).contains("positive k_h, v_h, k_d and v_d"));
        }

        let mut d = dense(8);
        let mut s = slab(vec![0, 1, 2, 4, 5, 6]);
        s.conv_k = 0;
        d.recurrent = Some(s);
        assert!(why(&d, shape(8)).contains("positive conv_k"));
    }

    #[test]
    fn the_gdn_head_repeat_and_the_simdgroup_bounds_are_refusals_not_clamps() {
        let hybrid = |edit: fn(&mut RecurrentShape)| {
            let mut d = dense(8);
            let mut s = slab(vec![0, 1, 2, 4, 5, 6]);
            edit(&mut s);
            d.recurrent = Some(s);
            d
        };

        // The value heads repeat over the key heads, so a non-multiple has
        // no repeat factor at all.
        let d = hybrid(|s| {
            s.v_h = 33;
            s.conv_dim = 2 * 16 * 128 + 33 * 128;
        });
        assert!(why(&d, shape(8)).contains("head repeat"));

        // One simdgroup reduces one head: 32 lanes, so a width off that
        // multiple leaves lanes holding nothing.
        let d = hybrid(|s| {
            s.k_d = 96 + 16;
            s.conv_dim = 2 * 16 * 112 + 32 * 128;
        });
        assert!(why(&d, shape(8)).contains("multiple of 32"));

        // Eight simdgroups' worth of per-lane registers is the ceiling.
        let d = hybrid(|s| {
            s.k_d = 288;
            s.conv_dim = 2 * 16 * 288 + 32 * 128;
        });
        assert!(why(&d, shape(8)).contains("per-lane registers"));
    }

    /// The row states the conv channel count AND the head counts it is
    /// computed from; a row that disagrees with itself is refused, because
    /// either number gives a plausible stride.
    #[test]
    fn a_slab_whose_conv_dim_contradicts_its_head_counts_is_refused() {
        let mut d = dense(8);
        let mut s = slab(vec![0, 1, 2, 4, 5, 6]);
        s.conv_dim = 4096;
        d.recurrent = Some(s);
        let why = why(&d, shape(8));
        assert!(
            why.contains("conv_dim 4096") && why.contains("plausible one"),
            "{why}"
        );
    }

    #[test]
    fn an_irregular_interleave_is_refused_rather_than_rounded() {
        let mut d = dense(8);
        // Full at 3 and 6 — not one every fixed interval.
        d.recurrent = Some(slab(vec![0, 1, 2, 4, 5, 7]));
        let why = why(&d, shape(8));
        assert!(why.contains("not one every fixed interval"), "{why}");
    }

    #[test]
    fn a_slab_that_names_no_linear_layers_is_refused_rather_than_guessed() {
        let mut d = dense(8);
        d.recurrent = Some(slab(Vec::new()));
        assert!(why(&d, shape(8)).contains("cannot be guessed"));
    }

    #[test]
    fn an_all_recurrent_stack_has_no_decode_path_here() {
        let mut d = dense(4);
        d.recurrent = Some(slab(vec![0, 1, 2, 3]));
        assert!(why(&d, shape(4)).contains("all-recurrent"));
    }

    /// The sliding schedule, derived from the WINDOWS rather than shared
    /// with the linear interval. A stack can now have both and could not
    /// say so when one field carried both readings.
    #[test]
    fn a_sliding_stack_states_its_period_its_window_and_its_second_base() {
        let mut d = dense(12);
        for l in 0..12u32 {
            let full = (l + 1).is_multiple_of(6);
            d.attention[l as usize].window = if full { -1 } else { 1024 };
            d.attention[l as usize].rope_theta = if full { 1e6 } else { 1e4 };
        }
        let g = geometry_from_deployment(&d, shape(12), AffineFormat::G64_B4)
            .expect("an alternating stack is served");
        assert_eq!(g.full_attn_every, 6);
        assert_eq!(g.sliding_window, 1024);
        // gemma-4-31b slides fifty of its sixty layers, so one base was
        // wrong on 83% of the stack.
        assert_eq!(g.rope_theta, 1e6);
        assert_eq!(g.rope_theta_sliding, 1e4);
        // The linear interval is untouched: there is no slab.
        assert_eq!(g.full_attn_interval, 1);
    }

    #[test]
    fn an_irregular_window_schedule_is_refused() {
        let mut d = dense(8);
        for l in [0usize, 1, 2, 4, 5, 7] {
            d.attention[l].window = 512;
        }
        let why = why(&d, shape(8));
        assert!(why.contains("not one every fixed period"), "{why}");
    }

    #[test]
    fn a_stack_where_every_layer_slides_is_refused() {
        let mut d = dense(4);
        for a in &mut d.attention {
            a.window = 256;
        }
        assert!(why(&d, shape(4)).contains("none attends the whole context"));
    }

    #[test]
    fn two_sliding_widths_are_refused_because_one_reaches_the_kernel() {
        let mut d = dense(8);
        for l in 0..8usize {
            d.attention[l].window = if (l + 1).is_multiple_of(4) { -1 } else { 512 };
        }
        d.attention[1].window = 256;
        assert!(why(&d, shape(8)).contains("distinct windows"));
    }

    #[test]
    fn a_third_rope_base_has_nowhere_to_go() {
        let mut d = dense(8);
        for l in 0..8usize {
            d.attention[l].window = if (l + 1).is_multiple_of(4) { -1 } else { 512 };
            d.attention[l].rope_theta = 1e4;
        }
        d.attention[3].rope_theta = 1e6;
        d.attention[7].rope_theta = 1e7;
        assert!(why(&d, shape(8)).contains("distinct rope bases"));
    }

    /// The router's two bounds are refusals and are reached BEFORE the
    /// top-k gap, so a limit that a row exceeds is reported as that limit.
    #[test]
    fn the_router_bounds_are_reachable_and_ordered_before_the_top_k_gap() {
        let mut d = dense(4);
        d.shape.moe_intermediate = 768;
        let mut load = shape(4);
        load.n_experts = ROUTER_MAX_EXPERTS + 1;
        let why = why(&d, load);
        assert!(why.contains("1025 exceeds the 1024"), "{why}");
    }

    /// Half a mixture is refused as half a mixture, not as a dense stack.
    #[test]
    fn a_mixture_is_both_numbers_or_neither_at_the_deployment_too() {
        let mut load = shape(4);
        load.n_experts = 128;
        let half_load = why(&dense(4), load);
        assert!(half_load.contains("both numbers or neither"), "{half_load}");

        let mut d = dense(4);
        d.shape.moe_intermediate = 768;
        let half_row = why(&d, shape(4));
        assert!(half_row.contains("both numbers or neither"), "{half_row}");
    }

    /// The top-k is READ, and the three ways it can be wrong are refused.
    ///
    /// This test used to assert the gap itself — "a `Deployment` states no
    /// top-k" — which was true of the migration and never true of the
    /// models: gpt-oss's facts called it `top_k`, gemma-4's mixture called
    /// it `experts_per_token`, and five families shared a `MoeFacts` that
    /// had it. None of it reached a driver. `Geometry::experts_per_token`
    /// is that statement, so the refusal became a derivation and what is
    /// left to refuse is a top-k that cannot be served.
    #[test]
    fn a_routed_mixture_reads_its_top_k_and_refuses_the_ones_it_cannot_route() {
        let routed = |top_k: u32| {
            let mut d = dense(4);
            d.shape.moe_intermediate = 768;
            d.shape.experts_per_token = top_k;
            let mut load = shape(4);
            load.n_experts = 128;
            (d, load)
        };

        let (d, load) = routed(8);
        let g = geometry_from_deployment(&d, load, AffineFormat::G64_B4)
            .expect("a routed row this driver can serve");
        assert!(g.is_moe(), "both counts reach the geometry");
        assert_eq!(g.experts_per_token, 8);
        assert_eq!(g.n_experts, 128);
        assert_eq!(g.moe_intermediate, 768);

        let (d, load) = routed(0);
        assert!(why(&d, load).contains("router selecting nothing"));

        let (d, load) = routed(ROUTER_MAX_TOP_K + 1);
        let over = why(&d, load);
        assert!(over.contains("ROUTER_MAX_TOP_K") || over.contains(&ROUTER_MAX_TOP_K.to_string()));

        // More experts visited than exist is its own diagnosis, because
        // "16 of 8" is a stated contradiction and not a bound.
        let mut d = dense(4);
        d.shape.moe_intermediate = 768;
        d.shape.experts_per_token = 12;
        let mut load = shape(4);
        load.n_experts = 8;
        assert!(why(&d, load).contains("more experts than it has"));
    }

    /// A routed row is served, and every tensor in it is read at ONE
    /// width.
    ///
    /// The measurement the header promises. Lifting the routed refusal
    /// let a mixture through, and the refusal's own text had named what
    /// it was standing next to: an mlx_lm routed checkpoint publishes
    /// its router gate at 8 bits inside a 4-bit stack, and reading the
    /// gate at the stack's width produced cosine 0.84 logits — a fluent
    /// model routing to almost the right experts, not an error.
    ///
    /// [`DecodeGeometry::alt_quant`] is the field that second width
    /// would ride. This asserts it stays UNSET through the routed path,
    /// which is the honest state: the geometry is built from ONE stated
    /// point, so `has_alt_quant()` is `false` and every tensor is read
    /// at `quant`. What keeps that from being a wrong answer is not this
    /// file — it is `serve/load.rs`, which asks `LoadPlan::affine_points`
    /// and refuses a checkpoint that arrives at more than one.
    ///
    /// # Why an assertion and not a `TODO`
    ///
    /// A gap nobody can trip over is a comment. This one is a test, so
    /// the day a load path DOES solve the second format, this fails and
    /// says where — rather than the field quietly starting to work
    /// while three docs still call it NOT WIRED.
    #[test]
    fn a_routed_row_reads_one_width_for_every_tensor() {
        let mut d = dense(4);
        d.shape.moe_intermediate = 768;
        d.shape.experts_per_token = 8;
        let mut load = shape(4);
        load.n_experts = 128;

        let g = geometry_from_deployment(&d, load, AffineFormat::G64_B4)
            .expect("the routed refusal is lifted");
        assert!(g.is_moe(), "the row this is measuring is routed");
        assert_eq!(
            g.quant,
            AffineFormat::G64_B4,
            "the stack's one point, which every tensor is read at"
        );
        assert!(
            !g.has_alt_quant(),
            "if a second format now reaches the geometry, the routed path \
             has a width to choose per tensor and `binding::observed` \
             still builds one kernel set from `quant` alone — wire the \
             choice before letting this pass"
        );
    }

    #[test]
    fn a_dense_stack_with_no_ffn_width_is_refused() {
        let mut d = dense(4);
        d.shape.intermediate = 0;
        assert!(why(&d, shape(4)).contains("dense FFN needs an intermediate width"));
    }

    /// gemma-4's two attention shapes, READ off the per-layer table.
    ///
    /// This used to be a refusal, and the refusal named the half
    /// `Deployment` could not state: a per-layer kv-head count. It has
    /// one now, so the case it declined is the case it projects, and
    /// both halves of the second shape have to arrive — a `head_dim`
    /// that crossed while the count did not would page a full layer's K
    /// at the sliding width and read past the end of it.
    #[test]
    fn two_head_dims_in_one_stack_are_read_and_not_refused() {
        let mut d = dense(12);
        // The stack-wide `Geometry` carries the SLIDING shape, which is
        // what gemma-4's own row states; the per-layer table carries the
        // full layers' departure from it.
        d.shape.head_dim = 256;
        d.shape.head_dim_kernel = 256;
        for l in 0..12usize {
            let full = (l + 1).is_multiple_of(6);
            d.attention[l].window = if full { -1 } else { 1024 };
            d.attention[l].head_dim = if full { 512 } else { 256 };
            d.attention[l].kv_heads = if full { 2 } else { 8 };
            d.attention[l].rotary_dim = if full { 128 } else { 256 };
        }
        let g = geometry_from_deployment(&d, shape(12), AffineFormat::G64_B4)
            .expect("both halves of the second shape are stated");
        assert_eq!(
            (g.head_dim, g.n_kv_heads),
            (256, 8),
            "the sliding shape is the base"
        );
        assert_eq!(
            (g.global_head_dim, g.global_kv_heads),
            (512, 2),
            "and the full layers' shape is the second one"
        );
        assert!(
            (g.full_partial_rotary - 0.25).abs() < 1e-6,
            "128 of 512 is the fraction the grid launches half of: {}",
            g.full_partial_rotary
        );
        // And the pool reads them as a per-layer PAGE, which is the
        // reader this whole pair exists for.
        let pool = crate::layout::kv::Shape {
            layers: g.n_layers,
            kv_heads: g.n_kv_heads,
            head_dim: g.head_dim,
            page_size: 16,
            pages: 4,
            element_bytes: 2,
            global_head_dim: g.global_head_dim,
            global_kv_heads: g.global_kv_heads,
            full_attn_every: g.full_attn_every,
        };
        assert_eq!(pool.heads_at(0), (8, 256));
        assert_eq!(pool.heads_at(5), (2, 512));
        assert!(!pool.is_uniform(), "two page sizes is exactly what this is");
    }

    /// A stack whose layers agree reads zeros, which is what
    /// `head_dim_at`/`kv_heads_at` take as "one shape everywhere".
    #[test]
    fn a_uniform_stack_states_no_second_shape() {
        let g = geometry_from_deployment(&dense(8), shape(8), AffineFormat::G64_B4)
            .expect("a uniform stack projects");
        assert_eq!((g.global_head_dim, g.global_kv_heads), (0, 0));
        assert_eq!((g.head_dim, g.n_kv_heads), (64, 8));
    }

    /// The second shape's count still has to group.
    #[test]
    fn a_full_layer_kv_count_that_does_not_divide_the_q_heads_is_refused() {
        let mut d = dense(12);
        d.shape.head_dim = 256;
        d.shape.head_dim_kernel = 256;
        for l in 0..12usize {
            let full = (l + 1).is_multiple_of(6);
            d.attention[l].window = if full { -1 } else { 1024 };
            d.attention[l].head_dim = if full { 512 } else { 256 };
            d.attention[l].kv_heads = if full { 7 } else { 8 };
        }
        let why = why(&d, shape(12));
        assert!(why.contains("7 kv heads"), "{why}");
        assert!(why.contains("does not divide 32"), "{why}");
    }

    /// An affine point off the instantiation grid entirely, refused at the
    /// geometry rather than at the first fire.
    #[test]
    fn an_unreadable_affine_point_is_refused_before_the_weights_are_staged() {
        let refused =
            geometry_from_deployment(&dense(4), shape(4), AffineFormat { bits: 3, group: 17 })
                .expect_err("no `affine_qmv_fast` is stamped at g17/b3");
        assert!(
            refused.0.contains("group_size 17 at 3 bits"),
            "{}",
            refused.0
        );
        // The refusal carries the grid, so the reader is told what IS
        // readable rather than only what is not.
        assert!(refused.0.contains("g64/b4"), "{}", refused.0);
        assert_eq!(
            refused.to_string(),
            format!("decode geometry: {}", refused.0),
            "the prefix names the stage, not a family"
        );
    }

    /// An unset point is an ABSENCE — a dense checkpoint — and keeps the
    /// default rather than naming a `gs_0_b_0` symbol no shader exports.
    #[test]
    fn an_unset_affine_point_is_a_dense_checkpoint_not_a_refusal() {
        let g = geometry_from_deployment(&dense(4), shape(4), AffineFormat::default())
            .expect("a bf16 checkpoint states no affine point");
        assert_eq!(g.quant, AffineFormat::G64_B4);
    }

    // ── the rope rescaling ───────────────────────────────────────────────

    /// Llama-3's four numbers reach the geometry.
    ///
    /// THE REGRESSION THIS PINS. These arrived on the `pie.model/1`
    /// descriptor as `ll_rope_*`; deleting the descriptor left
    /// `DecodeGeometry` holding four zeroes that nothing wrote, and a zero
    /// factor is read here as "no rescaling". Every Llama-3.1/3.2/3.3 would
    /// have attended past its trained 8192 with an unrescaled ladder —
    /// degrading rather than failing, so no test and no operator would have
    /// seen it. The values are the row's, not this file's.
    #[test]
    fn a_llama3_rescaled_ladder_reaches_the_geometry_whole() {
        let mut d = dense(4);
        d.rope_scaling = Some(RopeScaling::Piecewise {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position: 8_192,
        });
        let g = geometry_from_deployment(&d, shape(4), AffineFormat::default())
            .expect("a rescaled llama-3 is an ordinary stack");
        assert_eq!(
            g.rope_rescale,
            Some(crate::model::rope::Rescale::Piecewise {
                factor: 32.0,
                low: 1.0,
                high: 4.0,
                original_max: 8_192.0,
            }),
            "a `None` here is `no rescaling`"
        );
        // The base is untouched by the rescaling: the table is derived FROM
        // the ladder this builds, not instead of it.
        assert_eq!(g.rope_theta, 500_000.0);
    }

    /// A stack that states no rescaling gets `None`, and that is a
    /// STATEMENT rather than a gap.
    ///
    /// It used to be four zeroes, which is the same statement spelled so
    /// that a gap could not be told from it: a field nobody wrote and a
    /// field written with "nothing to do" both read `0.0`, and that is
    /// exactly how every llama-3 nearly shipped with an unrescaled ladder.
    /// An `Option` cannot be left unwritten.
    #[test]
    fn an_unrescaled_ladder_is_none_and_means_it() {
        let g = geometry_from_deployment(&dense(4), shape(4), AffineFormat::default())
            .expect("an unrescaled stack is ordinary");
        assert_eq!(g.rope_rescale, None);
    }

    /// YaRN reaches the geometry whole, and is not flattened to "no
    /// rescaling".
    ///
    /// THIS TEST DEMANDED A REFUSAL, and the refusal was the right answer
    /// for as long as `DecodeGeometry` held llama-3's four scalars and
    /// nothing else: the reader before that asked `kind == "llama3"` and
    /// gave every other kind a factor of zero, which this file cannot tell
    /// apart from a checkpoint that rescales nothing, so a YaRN model
    /// served silently wrong. Declining beat that.
    ///
    /// What the refusal did NOT beat is deriving the ladder, and it named
    /// the one missing piece itself. `model::rope::Rescale::Yarn` is that
    /// piece, so the claim here is the stronger half of what was being
    /// asked: the five numbers arrive, unflattened and unrounded, and the
    /// thing still forbidden is the zeroing.
    ///
    /// OLMo 3's numbers, which is the row that reaches here: dense, paged,
    /// one head dim, one window, ordinary by every other measure.
    #[test]
    fn a_yarn_rescaled_ladder_reaches_the_geometry_rather_than_flattening_to_none() {
        let mut d = dense(4);
        d.rope_scaling = Some(RopeScaling::Yarn {
            factor: 8.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: 1.207_944_2,
            original_max_position: 8_192,
            truncate: true,
        });
        let g = geometry_from_deployment(&d, shape(4), AffineFormat::default())
            .expect("a YaRN row is an ordinary stack");
        assert_eq!(
            g.rope_rescale,
            Some(crate::model::rope::Rescale::Yarn {
                factor: 8.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                original_max: 8_192.0,
                truncate: true,
            }),
            "a `None` here is the silent zeroing this refused for"
        );
        // And the ladder it builds is NOT the unrescaled one, which is the
        // defect the refusal existed to prevent. The slowest channel is the
        // one the factor divides.
        let plain = crate::model::rope::frequencies(g.head_dim, g.rope_theta, None);
        let scaled = crate::model::rope::table(&g);
        let last = plain.len() - 1;
        assert!(
            (scaled[last] - plain[last] / 8.0).abs() / (plain[last] / 8.0) < 1e-5,
            "the slow end must divide by 8: {} vs {}",
            scaled[last],
            plain[last]
        );
    }

    /// Every catalog row this driver's text serves reaches an answer, and
    /// no row reaches a panic.
    ///
    /// The claim `family_registry.rs` used to make of `MLX_ROWS` — that a
    /// registry's rows and a driver's paths agree — asked of the catalog,
    /// which is the registry now. A row either projects a geometry or is
    /// refused BY NAME; what this forbids is the third outcome, which is an
    /// index off the end of a per-layer table.
    #[test]
    fn every_catalog_row_either_projects_a_geometry_or_refuses_by_name() {
        use model::catalog::{self, Deployed};

        let mut seen = 0;
        for row in catalog::catalog() {
            let Ok(d) = row.deployment(Deployed::single()) else {
                continue;
            };
            seen += 1;
            match geometry_from_deployment(&d, row.load_shape(), AffineFormat::G64_B4) {
                Ok(g) => {
                    assert!(
                        g.n_layers > 0 && g.hidden > 0,
                        "{} projected a null shape",
                        row.id()
                    );
                    assert_eq!(g.n_layers, d.layers, "{} lost a layer", row.id());
                }
                Err(refused) => assert!(
                    !refused.0.is_empty(),
                    "{} was refused without a sentence",
                    row.id()
                ),
            }
        }
        assert!(seen > 0, "the catalog projected no deployments at all");
    }

    #[test]
    fn the_kernel_suffix_is_the_one_spelling_of_the_format() {
        assert_eq!(AffineFormat::G64_B4.kernel_suffix(), "_bfloat16_gs_64_b_4");
        assert_eq!(
            AffineFormat { bits: 8, group: 32 }.kernel_suffix(),
            "_bfloat16_gs_32_b_8"
        );
    }

    #[test]
    fn an_unset_alt_quant_is_an_absence_not_a_zero_format() {
        let mut geometry = DecodeGeometry::default();
        assert!(!geometry.has_alt_quant());
        geometry.alt_quant = AffineFormat { bits: 8, group: 64 };
        assert!(geometry.has_alt_quant());
        // Half a format is not a format: the pair travels together.
        geometry.alt_quant = AffineFormat { bits: 8, group: 0 };
        assert!(!geometry.has_alt_quant());
    }

    #[test]
    fn the_full_attention_interval_places_one_layer_per_period() {
        let geometry = DecodeGeometry::default();
        // Interval 4: layers 3, 7, 11, ... — one per period, at its end.
        assert!(!geometry.is_full_attn(0));
        assert!(geometry.is_full_attn(3));
        assert!(!geometry.is_full_attn(4));
        assert!(geometry.is_full_attn(7));
        assert_eq!(geometry.full_attn_layers(), 6, "24 layers / 4");

        // An interval of one makes every layer qualify — a family with no
        // linear attention.
        let dense = DecodeGeometry {
            full_attn_interval: 1,
            ..DecodeGeometry::default()
        };
        assert_eq!(dense.full_attn_layers(), dense.n_layers);
    }

    #[test]
    fn a_mixture_is_both_numbers_or_neither() {
        let mut geometry = DecodeGeometry::default();
        assert!(!geometry.is_moe());
        assert_eq!(geometry.ffn_width(), 3584);
        geometry.n_experts = 512;
        assert!(
            !geometry.is_moe(),
            "experts without a per-token count route nothing"
        );
        geometry.experts_per_token = 10;
        geometry.moe_intermediate = 768;
        assert!(geometry.is_moe());
        assert_eq!(geometry.ffn_width(), 768);
        assert!(!geometry.has_shared_expert());
        geometry.shared_intermediate = 512;
        assert!(geometry.has_shared_expert());
    }

    #[test]
    fn the_gdn_strides_are_the_slotted_kernels_arithmetic() {
        let geometry = DecodeGeometry::default();
        assert_eq!(geometry.gdn_conv_stride_bytes(), 6144 * 4 * 4);
        assert_eq!(geometry.gdn_recurrent_stride_bytes(), 16 * 128 * 128 * 4);
    }

    /// The `(1 + w)` fold comes from the ROW, and the default is the
    /// opposite of what a row can say.
    ///
    /// `DecodeGeometry::default()` carries `true`, so a projection that
    /// let this field fall through `..default()` would answer "folds" for
    /// every stack — including the one stack that does not. This test
    /// fails if the field is ever dropped from the construction, which is
    /// exactly the mistake the struct-update syntax makes easy.
    #[test]
    fn the_norm_fold_is_the_rows_answer_and_not_the_defaults() {
        assert!(
            DecodeGeometry::default().norm_unit_offset,
            "the default must stay `true` for this test to be able to fail"
        );

        let mut plain = dense(4);
        plain.norm_unit_offset = false;
        let g = geometry_from_deployment(&plain, shape(4), AffineFormat::G64_B4)
            .expect("a dense stack is servable");
        assert!(
            !g.norm_unit_offset,
            "a row that stores the multiplier directly must not be folded"
        );

        let mut folded = dense(4);
        folded.norm_unit_offset = true;
        let g = geometry_from_deployment(&folded, shape(4), AffineFormat::G64_B4)
            .expect("a dense stack is servable");
        assert!(g.norm_unit_offset, "a row that stores an offset must fold");
    }
}
