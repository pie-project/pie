//! **The** table: one row per model, and every aspect answered on it.
//!
//! # The three keys that were one key
//!
//! This crate used to dispatch on three strings that all came out of
//! `config.json`, through three tables that nothing held together:
//!
//! | key | table | answers |
//! |---|---|---|
//! | `model_type` | `contract::HF_ROWS` | who authors the load |
//! | `model_type` | `deployment_cuda::FACTS_ROWS` | what to deploy |
//! | `arch_name` | `instruct::create` | how to format a chat |
//!
//! Three tables existed because the identity arrived **at run time**, as
//! text, and each aspect had to re-key on it. Nothing made the answers
//! agree, and they did not: `qwen3_moe` authored as a GDN mixture and
//! deployed as a dense stack, for as long as anyone had been looking.
//!
//! Move the identity to compile time and the three answers land in one
//! ROW, where there is no seam for them to disagree across. That is the
//! whole of this module.
//!
//! # A row is reached by matching, not by parsing
//!
//! What replaces the string is [`crate::manifest`]: a row states the
//! tensors its own numbers imply, and a checkpoint is the row whose
//! manifest it satisfies. The reason is not ergonomics — it is that
//! **identification and validation become one operation**. A
//! `config.json` that lies is believed by a derivation and contradicted
//! by an assertion much later, or never; a checkpoint either publishes
//! `layer.{}.q_proj` at `[2048, 1024]` or it does not, and a variant
//! whose numbers say otherwise simply is not the one in front of you.
//!
//! It also settles what the old code did by sniffing. Four probes inside
//! one derivation asked whether the embedding table exists, whether the
//! norm is pre or post, whether the q-norm is per-head or global, and
//! whether the projections came fused. All four are manifest rows now,
//! and a checkpoint that answers them differently matches a different
//! variant rather than taking a different branch inside one.
//!
//! # No default method bodies
//!
//! Deliberate, and the reason is [`crate::deployment`]'s: the trait this
//! replaces had thirteen methods, twelve of them with default bodies,
//! and the defaults were where the exceptions hid.
//! `pins_attention_values()` returned `true` with the doc *"Only gemma-4
//! does"*; `sm_scale()` divided by `sqrt(head_dim)` because every family
//! but one does. A default body is a claim about every row that has not
//! been written yet, which is the one claim nobody is in a position to
//! make.
//!
//! So every row answers every question. The fixtures already work this
//! way — *"stated rather than defaulted because a fixture is a
//! measurement of a real checkpoint"* — and this makes the discipline
//! the type system's.
//!
//! # What a row is NOT keyed by
//!
//! **Quantization.** Qwen3-8B ships bf16, FP8, AWQ-int4 and MLX-int4;
//! splitting the row four ways would quadruple the table to record
//! something that is not about the model. A row is the LOGICAL model,
//! its manifest compares logical extents, and the observed encoding
//! flows on into [`crate::shared::policy`] — which is where a decision about how
//! to read weights belongs.
//!
//! **Packaging.** Which safetensors shard a tensor landed in gets
//! revised without the model changing, so a manifest names logical
//! tensors and never a file.
//!
//! # The cost, stated honestly
//!
//! A closed set closes new SIZES of known models, not just new models: a
//! Qwen3-72B needs a release. [`Override`] is the escape hatch, and it
//! is deliberately not a way to get a wrong answer quietly — an override
//! names a row, and the named row's manifest is still checked.

use crate::manifest::Manifest;

#[cfg(feature = "chat")]
use std::sync::Arc;

/// The shape facts an AUTHORING pass needs, and only those.
///
/// This is what is left of `ModelFacts` once the parts that were never
/// shape are taken out of it. That struct had eleven fields: one was the
/// dispatch key (`model_type` — gone, because the row IS the dispatch),
/// three were observed quantization ([`Encoding`], because an encoding
/// is a property of a FILE and not of a model), and these six are the
/// only ones a contract author actually reads.
///
/// Every one of them is here for a stated reason, and in each case the
/// reason is that no tensor extent carries it:
///
/// - `head_dim`, because tensor-parallel splits an attention projection
///   by rows and whether a row split lands on a head boundary is not a
///   question the row count can answer.
/// - `mamba_groups`, because a Mamba mixer's B and C bands are
///   `groups * state` rows of a fused tensor and the PRODUCT is all that
///   is ever stored.
/// - `kv_shared_layers`, because gemma-4's tail attends KV an earlier
///   layer wrote, so its own k/v projections are dead weight a contract
///   must not declare — and a shipped tensor cannot say "ignore me".
/// - `tied_embeddings`, because a tie is the ABSENCE of `lm_head`, and
///   the MLX authors need to know an absence was intended.
///
/// [`Encoding`]: crate::encoding::Encoding
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadShape {
    /// Decoder layers.
    pub layers: u32,
    /// One attention head's width, UNPADDED — the checkpoint's own, not
    /// the one a kernel rounds it up to.
    pub head_dim: u32,
    /// Experts in the mixture, 0 for a dense model.
    pub n_experts: u32,
    /// Mamba B/C groups, 0 for a family with no Mamba mixer.
    pub mamba_groups: u32,
    /// Trailing layers that attend an earlier layer's KV.
    pub kv_shared_layers: u32,
    /// The head shares the embedding table.
    pub tied_embeddings: bool,
}

impl LoadShape {
    /// The ordinary case: a dense transformer with no Mamba mixer and no
    /// KV sharing.
    ///
    /// Not a `Default`, and not a builder: a row states its zeros the way
    /// the fixtures do, and the two facts that are almost always zero are
    /// still named at every call site that uses this.
    #[must_use]
    pub const fn dense(layers: u32, head_dim: u32, tied_embeddings: bool) -> Self {
        Self {
            layers,
            head_dim,
            n_experts: 0,
            mamba_groups: 0,
            kv_shared_layers: 0,
            tied_embeddings,
        }
    }

    /// A mixture, otherwise ordinary.
    #[must_use]
    pub const fn mixture(
        layers: u32,
        head_dim: u32,
        n_experts: u32,
        tied_embeddings: bool,
    ) -> Self {
        Self {
            layers,
            head_dim,
            n_experts,
            mamba_groups: 0,
            kv_shared_layers: 0,
            tied_embeddings,
        }
    }
}

/// What a Metal load observed that no row can state.
///
/// Every field here is about the BYTES this checkpoint shipped or about
/// the kernels this driver built — never about the model. Its
/// counterpart on the CUDA side is `LlamaLikeCudaFacts`, which is the
/// same eight-ish questions asked of a different backend, and the reason
/// both are short is that the model half of each text's input is the
/// row's and gets projected rather than observed.
///
/// The two encoding fields are the load-time half of the module doc's
/// rule that an encoding is a policy and not an identity: the same row
/// serves bf16 and int4, and which one arrived is exactly the kind of
/// thing a table of models must not try to hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalBinding {
    /// The affine quantisation group width the staged tensors carry.
    ///
    /// Asked of the load, which reads it off the TENSORS. This said it was
    /// asked of the load "rather than inferred from a tensor's shape, because
    /// g64/b8 and g128/b4 pack to identical extents", and they do not: for
    /// 5760 columns those two are `scales` of 90 against 45 and packed
    /// `weight` of 1440 `u32` against 720. Both extents distinguish them,
    /// twice over, and `LoadPlan::affine_points` performs exactly those
    /// divisions. What the load must not do is believe `config.json`, whose
    /// block is a DEFAULT its per-tensor overrides may supersede for every
    /// tensor in the file — `gpt-oss-20b-MXFP4-Q4` declares g32/b4 and holds
    /// not one tensor at it.
    pub quant_group: u32,
    /// The affine quantisation bit width the staged tensors carry.
    pub quant_bits: u32,
    /// The ROUTER GATE's own affine point, `(0, 0)` when it has none.
    ///
    /// `mlx_lm` publishes gpt-oss's gate at 8 bits inside a 4-bit stack,
    /// deliberately: it is a tiny `[hidden, n_experts]` matrix whose error
    /// the whole mixture inherits. Read at the stack's width the model stays
    /// fluent and routes every token to almost the right experts — cosine
    /// 0.84 against the reference logits, and not one NaN to notice it by.
    ///
    /// `(0, 0)` means "the same as [`Self::quant_group`]", which is every
    /// other checkpoint and every non-MoE one. Asked of the load for the
    /// reason [`Self::moe_mxfp4`] is: it is the CHECKPOINT's and not the
    /// row's.
    pub router_quant_group: u32,
    /// The router gate's affine bit width; see [`Self::router_quant_group`].
    pub router_quant_bits: u32,
    /// Whether the expert bank reached the device still in MXFP4.
    ///
    /// A fact about what the loader did — it transcodes to affine when
    /// this driver has no native routed kernel — and so it can only be
    /// known after staging, which is why it lives here and not in the
    /// row.
    pub moe_mxfp4: bool,
    /// Whether this build folds the residual add into the decode GEMV.
    pub fuse_residual_gemv: bool,
    /// Whether this build's paged attention takes more than one row.
    pub paged_multi_batch: bool,
    /// Whether this build's quantised matmul takes more than one row.
    pub qmm_multi_batch: bool,
    /// Whether this build can launch `norm::add_bias`, and so whether the
    /// text may state the Qwen-2 family's q/k/v projection biases.
    ///
    /// See [`LlamaLikeMetalFacts::add_bias`], which this becomes: whether the
    /// biases exist is the ROW's `qkv_bias`, and this is the other half of the
    /// question -- the half that is about this build.
    ///
    /// [`LlamaLikeMetalFacts::add_bias`]:
    ///     crate::shared::llama_like::forward::facts::LlamaLikeMetalFacts::add_bias
    pub add_bias: bool,
}

/// Which driver is asking for the text.
///
/// A row states one model and every backend's reading of it, so the
/// backend is a parameter of the question rather than a second table of
/// answers. Before this existed, `driver-metal` kept an eleven-entry
/// table of architecture STRINGS and rebuilt the model's own facts from
/// nine tensor probes — the third dispatch key this catalog exists to
/// delete.
///
/// `Metal` carries its binding because a Metal text needs both halves;
/// `Cuda` carries none because that backend's binding is a projection of
/// the row and the tp width alone.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Backend<'a> {
    /// The CUDA driver, whose binding facts derive from the row.
    #[default]
    Cuda,
    /// The Metal driver, with what its load observed.
    Metal(&'a MetalBinding),
}

/// What the LOAD knows that a row cannot.
///
/// Three values, and the shortness of this list is the point. Its
/// predecessor was a `Checkpoint<'_>` whose first field was the whole
/// parsed `config.json` — 228 of 244 field reads went there — which is
/// how "what is this model" stayed a question the driver re-asked. A row
/// answers that; what is left here is genuinely about this load and not
/// about this model.
#[derive(Clone, Copy, Debug, Default)]
pub struct Deployed<'a> {
    /// Which driver is asking, and what its load observed.
    ///
    /// Defaults to [`Backend::Cuda`], which is not a preference but the
    /// shape of the fleet: one backend derives its binding and the other
    /// observes it, so only the observing one has to say so.
    pub backend: Backend<'a>,
    /// The tensor-parallel group this rank's weights were sharded for.
    ///
    /// A fact about how the checkpoint was SPREAD, which no row can
    /// state because the same model serves at any width.
    pub tp_size: u32,
    /// Per-layer scalars a family reads to host once at load.
    ///
    /// Gemma-4's `layer_scalar` and nothing else. Empty everywhere else,
    /// which is not a defaulting rule but the truth: no other row ships
    /// one.
    pub layer_scalars: &'a [f32],
}

impl<'a> Deployed<'a> {
    /// One rank, no host scalars — what a single-GPU boot passes.
    #[must_use]
    pub fn single() -> Self {
        Self {
            backend: Backend::Cuda,
            tp_size: 1,
            layer_scalars: &[],
        }
    }

    /// One rank on Metal, with what that load observed.
    #[must_use]
    pub fn metal(binding: &'a MetalBinding) -> Self {
        Self {
            backend: Backend::Metal(binding),
            tp_size: 1,
            layer_scalars: &[],
        }
    }
}

/// One model, and every aspect of it.
///
/// Implemented by a generation's own shape struct — the row IS the
/// facts, not a key that points at them — so a number cannot be stated
/// in the table and restated in the fixture that traces against it.
pub trait Variant: Sync + Send + 'static {
    /// The stable name a boundary carries and an operator types.
    ///
    /// Lowercase, hyphenated, vendor-first: `qwen3-0.6b`, `gemma-4-e4b`.
    /// It names a MODEL and never an encoding of one, so the FP8 and
    /// bf16 builds of a checkpoint share an id — see the module doc.
    fn id(&self) -> &'static str;

    /// The tensors this variant's numbers imply.
    ///
    /// A projection of the row, never a second statement of it: a
    /// hidden size of 1024 with 16 heads of 128 IS the claim that
    /// `layer.{}.q_proj` is `[2048, 1024]`.
    fn manifest(&self) -> Manifest;

    /// The six shape facts an authoring pass reads.
    ///
    /// A separate question from [`Self::manifest`] because it is a
    /// different KIND of question: a manifest says what tensors exist,
    /// and this says the things no tensor's extents can say. `head_dim`
    /// is the sharpest case — `[2048, 1024]` is 16 heads of 128 or 32 of
    /// 64 and the tensor cannot tell you which, but a TP row split that
    /// guesses wrong cuts a head in half.
    fn load_shape(&self) -> LoadShape;

    /// Everything a driver needs to serve this, with no family name in
    /// it.
    ///
    /// A PROJECTION of the row rather than a derivation from a config —
    /// which is the difference the whole table exists to make. The old
    /// path read eleven per-family functions over a parsed
    /// `config.json`; the numbers were always these numbers.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unsupported`] when this build has no forward text for
    /// the row — which a row states for itself rather than being absent
    /// from a second table. That is where `unbuilt_kv_store()` went: it
    /// existed because a family could hold a facts row, load happily
    /// and die at its first fire, and a refusal here happens at the
    /// door.
    ///
    /// [`Refusal`]: crate::deployment::Refusal
    fn deployment(
        &self,
        load: Deployed<'_>,
    ) -> Result<crate::deployment::Deployment, crate::deployment::Refusal>;

    /// Author this variant's load contract.
    ///
    /// The row calls its own family's authoring pass, so the N:1 reuse
    /// that `HF_ROWS` spelled as a table column is spelled as a call.
    ///
    /// # Errors
    ///
    /// The checkpoint contradicts a shape the author asserts.
    #[cfg(feature = "contract")]
    fn author(
        &self,
        builder: &mut crate::shared::builder::Builder<'_>,
    ) -> Result<(), model_loader::error::Error>;

    /// This variant's forward text, traced for one fire class.
    ///
    /// Backend-agnostic: `load.backend` says which driver is asking, and
    /// a row that has a text for one backend and not the other refuses
    /// the other HERE rather than being absent from that driver's own
    /// table of names. That absence is what `driver-metal` used to
    /// spell, and spelling it twice is how `gemma4` came to be listed as
    /// served by a driver that then refused it on a different ground.
    ///
    /// # Errors
    ///
    /// [`Refusal::Unsupported`](crate::deployment::Refusal) when this
    /// build has no text for the row, or none for the backend asking.
    fn trace(
        &self,
        class: model_ir::trace::FireClass,
        load: Deployed<'_>,
    ) -> Result<model_ir::trace::ForwardPlan, crate::deployment::Refusal>;

    /// The chat template that formats for this variant.
    ///
    /// Required, which is the repair: `instruct::create` ended in
    /// `_ => QwenInstruct`, so a model with no row got ChatML — and
    /// gemma-4 generated fluently, ending turns it was not having with
    /// an `<|im_end|>` its vocabulary does not contain. A total function
    /// has nowhere to put that.
    #[cfg(feature = "chat")]
    fn chat(&self, tokenizer: Arc<tokenizer::Tokenizer>) -> Arc<dyn crate::instruct::Instruct>;
}

/// Every variant this build serves.
///
/// A function rather than a `const` because a const `&[&dyn Variant]`
/// can only be written by naming every row twice — once in its
/// generation's typed table and once here, by index. The ROWS stay
/// const; only the flattening is deferred, and it happens once.
#[must_use]
pub fn catalog() -> &'static [&'static dyn Variant] {
    static CATALOG: std::sync::OnceLock<Vec<&'static dyn Variant>> = std::sync::OnceLock::new();
    CATALOG.get_or_init(|| {
        let mut rows: Vec<&'static dyn Variant> = Vec::new();
        for generation in GENERATIONS {
            rows.extend_from_slice(generation());
        }
        rows
    })
}

/// Each generation's own table, in the order the module list declares
/// them.
///
/// A generation owns its rows — that is the isolation rule
/// (`tests/sibling_isolation.rs`) applied to the catalog — and this is
/// the one place they are gathered.
type Rows = fn() -> &'static [&'static dyn Variant];

/// A generation's `rows()`, which is the same four lines everywhere.
///
/// Nineteen generations wrote this function and all nineteen bodies
/// hashed to one value — the flattening of a `const VARIANTS: &[T]` into
/// the `&[&dyn Variant]` this table gathers. There is nothing in it a
/// generation could get right or wrong, which is the test for whether a
/// repetition is knowledge: `manifest()` differs everywhere and belongs
/// to each family, and this differed nowhere.
///
/// It stays a per-generation `rows()` rather than becoming a blanket
/// impl, because [`GENERATIONS`] gathers FUNCTIONS and a generation
/// owning its own door is the isolation rule applied to the catalog.
/// What is removed is the copy, not the ownership.
#[macro_export]
macro_rules! rows_of {
    ($row:ty) => {
        /// This generation's contribution to [`crate::catalog::catalog`].
        #[must_use]
        pub fn rows() -> &'static [&'static dyn $crate::catalog::Variant] {
            static ROWS: std::sync::OnceLock<Vec<&'static dyn $crate::catalog::Variant>> =
                std::sync::OnceLock::new();
            ROWS.get_or_init(|| {
                VARIANTS
                    .iter()
                    .map(|v| v as &'static dyn $crate::catalog::Variant)
                    .collect()
            })
        }
    };
}

const GENERATIONS: &[Rows] = &[
    crate::llama_3::rows,
    crate::qwen_2::rows,
    crate::qwen_3::rows,
    crate::qwen_3_5::rows,
    crate::gemma_2::rows,
    crate::gemma_3::rows,
    crate::gemma_3n::rows,
    crate::gemma_4::rows,
    crate::glm_5::rows,
    crate::gpt_oss::rows,
    crate::kimi_k2::rows,
    crate::kimi_k3::rows,
    crate::deepseek_v4::rows,
    crate::nemotron_h::rows,
    crate::olmo_2::rows,
    crate::olmo_3::rows,
    crate::phi_3::rows,
    crate::mistral_3::rows,
    crate::csm::rows,
    // Last, and only when asked for: a row that is not a model. See
    // `test_rows` for why the closed set needs a door and why this is not
    // one — and `a_shipped_catalog_has_no_test_rows` for the guarantee that
    // it stays shut.
    #[cfg(feature = "test-rows")]
    crate::test_rows::rows,
];

/// The variant with this id, or `None`.
#[must_use]
pub fn find(id: &str) -> Option<&'static dyn Variant> {
    catalog().iter().copied().find(|row| row.id() == id)
}

/// Every id this build serves, in catalog order.
#[must_use]
pub fn ids() -> Vec<&'static str> {
    catalog().iter().map(|row| row.id()).collect()
}

/// Every FAMILY LABEL this build advertises, deduplicated and sorted.
///
/// Coarser than [`ids`] by design: `qwen3-0.6b` and `qwen3-32b` are two
/// rows and one arch, because an arch is what a guest program matches
/// on when it asks "is this a gemma" — see
/// [`Advertised::arch`](crate::deployment::Advertised::arch).
///
/// It exists so that a label can be CHECKED rather than derived. The
/// worker used to produce one by lowercasing `architectures[0]` and
/// stripping a task suffix, and `driver-metal` had a second copy of the
/// same heuristic that stripped a shorter list — so
/// `Gemma4ForConditionalGeneration` became `gemma4` on one side and
/// `gemma4forconditionalgeneration` on the other, and the second
/// matched no chat row and fell through to ChatML. Two spellings of one
/// derivation is exactly what a catalog exists to end; a derived label
/// that no row advertises is now a stateable finding.
///
/// Rows whose [`Variant::deployment`] refuses are SKIPPED rather than
/// panicking: a row this build cannot serve advertises nothing, which is
/// the truthful answer for a label list.
#[must_use]
pub fn arches() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = catalog()
        .iter()
        .filter_map(|row| row.deployment(Deployed::single()).ok())
        .map(|d| d.advertised.arch)
        .filter(|a| !a.is_empty())
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The ids closest to a string, for a refusal that suggests rather than
/// merely declines.
///
/// The metric is a plain edit distance over the id text. Not a fuzzy
/// matcher and not a prefix trie: the catalog is a hundred rows, this
/// runs once on a path that is already about to fail, and the only
/// requirement is that `qwen3-0.6` puts `qwen3-0.6b` first.
#[must_use]
pub fn nearest_ids(id: &str, take: usize) -> Vec<&'static str> {
    let mut scored: Vec<(usize, &'static str)> = ids()
        .into_iter()
        .map(|k| (edit_distance(id, k), k))
        .collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
    scored.into_iter().take(take).map(|(_, k)| k).collect()
}

/// Levenshtein distance, two rows at a time.
fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, ca) in a.chars().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Why a checkpoint is not any variant in this build.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unmatched {
    /// Nothing matched, and here is how the nearest rows differed.
    ///
    /// The DIFF rather than a bare "unsupported", because the useful
    /// question after a refusal is which row was close and by how much.
    NoRow {
        nearest: Vec<(&'static str, String)>,
    },
    /// More than one row matched, which is a defect in the TABLE — two
    /// variants that no checkpoint can tell apart are one variant.
    Ambiguous { ids: Vec<&'static str> },
    /// An override, or a boundary, named an id this build has no row
    /// for.
    ///
    /// Carries the nearest ids because the overwhelmingly likely cause
    /// is a typo — `qwen3-0.6` for `qwen3-0.6b` — and a refusal that
    /// makes the operator go and read a list is a refusal that could
    /// have just told them.
    NoSuchId {
        id: String,
        nearest: Vec<&'static str>,
    },
}

impl std::fmt::Display for Unmatched {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoRow { nearest } => {
                write!(f, "this checkpoint matches no model this build serves")?;
                for (id, why) in nearest {
                    write!(f, "\n  {id}: {why}")?;
                }
                Ok(())
            }
            Self::Ambiguous { ids } => write!(
                f,
                "this checkpoint matches {} models equally well ({}); two rows \
                 no checkpoint can tell apart are one row",
                ids.len(),
                ids.join(", "),
            ),
            Self::NoSuchId { id, nearest } if nearest.is_empty() => write!(
                f,
                "no model named '{id}' in this build; `pie model list` prints the \
                 ids this binary serves",
            ),
            Self::NoSuchId { id, nearest } => write!(
                f,
                "no model named '{id}' in this build; did you mean {}?",
                nearest.join(", "),
            ),
        }
    }
}

impl std::error::Error for Unmatched {}

/// An operator's answer to "which model is this", when they have one.
///
/// It names a ROW and does not replace the check: an override that
/// skipped the manifest would be a way to load a checkpoint as something
/// it is not, which is the failure this whole arrangement exists to
/// prevent. What it buys is the case where a checkpoint is genuinely a
/// known model under an unknown name — a fine-tune, a re-upload, a
/// mirror that renamed the directory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Override {
    /// Match on the manifest, as usual.
    #[default]
    None,
    /// Use this row, still holding the checkpoint to its manifest.
    Id(String),
}

/// Rows that no checkpoint can tell apart, declared on purpose.
///
/// [`identify`] treats two matching rows as [`Unmatched::Ambiguous`], and
/// that doc calls it "a defect in the TABLE, because two rows nothing can
/// distinguish should have been one". That rule is right about accidents
/// — a generation that copies a neighbour's `manifest()` and forgets to
/// change a number must fail loudly — and wrong about one real case.
///
/// Llama-3.1-70B and Llama-3.3-70B are 3.1's geometry exactly, retrained.
/// Every number a manifest can read is equal; what differs is the
/// WEIGHTS, which no shape check reaches. They are not one row, because a
/// guest naming `llama-3.3-70b` means a different model than one naming
/// `llama-3.1-70b`. And they cannot be separated by inspecting harder.
///
/// So the ambiguity is stated here rather than argued with. Two things
/// follow, and both matter: a checkpoint matching a declared set is still
/// REFUSED rather than guessed at — [`Override::Id`] is how a caller
/// says which twin it has, and it still holds the checkpoint to the
/// manifest — and an ambiguity that is NOT in this table is still the
/// defect the test says it is.
pub const GEOMETRIC_TWINS: &[&[&str]] = &[&["llama-3.1-70b", "llama-3.3-70b"]];

/// Whether these ids are a declared [`GEOMETRIC_TWINS`] set.
///
/// Order-insensitive and exact: a subset does not count, because a set
/// that shrank means a row learned to distinguish itself and the
/// declaration is now stale.
#[must_use]
pub fn are_declared_twins(ids: &[&str]) -> bool {
    GEOMETRIC_TWINS
        .iter()
        .any(|set| set.len() == ids.len() && set.iter().all(|id| ids.contains(id)))
}

#[cfg(feature = "contract")]
mod identify {
    use super::{Override, Unmatched, Variant, catalog, find, nearest_ids};
    use crate::manifest::Observed;
    use model_loader::checkpoint::CheckpointMetadata;

    /// Which variant this checkpoint is.
    ///
    /// One pass over the table, and the answer is the row whose
    /// manifest the checkpoint satisfies. Exactly one must, and both
    /// other outcomes are reported rather than resolved: zero is a
    /// checkpoint this build does not serve, and more than one is
    /// USUALLY a defect in the TABLE, because two rows nothing can
    /// distinguish are one row.
    ///
    /// Usually, and not always: see [`GEOMETRIC_TWINS`] for the case
    /// where two rows are one geometry under two release names, which no
    /// amount of inspecting resolves. Either way this reports rather than
    /// guesses — the caller says which one with [`Override::Id`], and the
    /// checkpoint is still held to that row's manifest.
    ///
    /// # Errors
    ///
    /// Nothing matched, more than one did, or an override named an
    /// unknown id.
    pub fn identify(
        metadata: &CheckpointMetadata,
        chosen: &Override,
    ) -> Result<&'static dyn Variant, Unmatched> {
        let observed = Observed::of(metadata);
        if let Override::Id(id) = chosen {
            let row = find(id).ok_or_else(|| Unmatched::NoSuchId {
                id: id.clone(),
                nearest: nearest_ids(id, 3),
            })?;
            return row
                .manifest()
                .check(&observed)
                .map(|()| row)
                .map_err(|why| Unmatched::NoRow {
                    nearest: vec![(row.id(), why.to_string())],
                });
        }

        let mut matched: Vec<&'static dyn Variant> = Vec::new();
        let mut misses: Vec<(&'static str, usize, String)> = Vec::new();
        for row in catalog() {
            match row.manifest().check(&observed) {
                Ok(()) => matched.push(*row),
                Err(why) => misses.push((row.id(), why.faults.len(), why.to_string())),
            }
        }
        match matched.len() {
            1 => Ok(matched[0]),
            0 => {
                misses.sort_by_key(|(_, faults, _)| *faults);
                misses.truncate(3);
                Err(Unmatched::NoRow {
                    nearest: misses.into_iter().map(|(id, _, why)| (id, why)).collect(),
                })
            }
            _ => Err(Unmatched::Ambiguous {
                ids: matched.iter().map(|row| row.id()).collect(),
            }),
        }
    }
}

#[cfg(feature = "contract")]
pub use identify::identify;

#[cfg(test)]
mod tests {
    use super::*;

    /// EVERY GENERATION DIRECTORY IS EITHER IN `GENERATIONS` OR STATED.
    ///
    /// The refactor's premise is that a model's identity is enumerated.
    /// A generation whose module exists and whose rows are not gathered
    /// is the hole in that premise: its chat template compiles, its
    /// `contract` compiles, and NOTHING can reach either, because the
    /// only door is a row id. That is the `_ =>` fallback's failure
    /// inverted — instead of an unknown model silently getting the wrong
    /// template, a known template silently gets no model.
    ///
    /// So the set is closed in both directions, source-level, the same
    /// way `NOT_YET_OPENABLE` and `DECLARED_BUT_UNREAD` are: a
    /// generation joins `GENERATIONS` or it joins the list below with a
    /// sentence saying why.
    #[test]
    fn a_generation_with_no_rows_says_so() {
        /// Generations that ship no catalog row, and why.
        ///
        /// Both hold only a chat template, both templates are correct,
        /// and neither is reachable — no row names them. They are kept
        /// rather than deleted because the template is the expensive
        /// part and a row is cheap: whoever adds `llama-2-7b` or
        /// `deepseek-r1` gets a tested template instead of writing one.
        /// R1's has since been moved to `shared::deepseek` and adopted
        /// by `deepseek_v4`, which is the shape a line leaves this list
        /// by wanting: the words outlive the directory.
        ///
        /// A line LEAVES here by gaining a `rows()` and an entry in
        /// `GENERATIONS`. A line JOINS only with a commit that says
        /// which rows are coming.
        const NO_ROWS_YET: &[(&str, &str)] = &[
            // `[INST] <<SYS>>` — shared with no later Llama. Llama 3's
            // `LlamaInstruct` is a DIFFERENT type with the same name,
            // which is exactly why this one cannot just be deleted and
            // pointed at that.
            (
                "llama_2",
                "the [INST] template; no Llama 2 row is transcribed",
            ),
            // R1's `<think>` turn, which now lives in
            // `shared::deepseek` — `deepseek_v4` adopted it, so the
            // template is reachable and this directory is not. What is
            // left here is a re-export and no row: R1's own geometry is
            // still untranscribed.
            (
                "deepseek_r1",
                "the <think> template; no R1 row is transcribed",
            ),
        ];

        let lib = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"),
        )
        .expect("lib.rs");
        let src = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/catalog.rs"),
        )
        .expect("catalog.rs");
        let gathered = src
            .split_once("const GENERATIONS")
            .expect("the table")
            .1
            .split_once("];")
            .expect("the table's end")
            .0
            .to_string();

        // The generation modules, read from the block `lib.rs` declares
        // them in rather than from the filesystem: a directory with no
        // `pub mod` is not part of the crate at all.
        let block = lib.split_once("── The generations ").expect("the block").1;
        let declared: Vec<&str> = block
            .lines()
            .filter_map(|l| l.trim().strip_prefix("pub mod ")?.strip_suffix(';'))
            .collect();
        assert!(
            declared.len() > 15,
            "the generation block reader found {} modules, so its shape assumption broke",
            declared.len()
        );

        let stated: std::collections::BTreeMap<&str, &str> = NO_ROWS_YET.iter().copied().collect();
        let mut rowless = Vec::new();
        for m in &declared {
            if gathered.contains(&format!("crate::{m}::rows")) {
                assert!(
                    !stated.contains_key(m),
                    "{m} is in GENERATIONS and also listed as having no rows — \
                     delete the line, it left by gaining rows"
                );
            } else {
                rowless.push(*m);
            }
        }
        let unexplained: Vec<&&str> = rowless
            .iter()
            .filter(|m| !stated.contains_key(**m))
            .collect();
        assert!(
            unexplained.is_empty(),
            "{unexplained:?} declare a generation module that `GENERATIONS` does \
             not gather. Nothing can reach them: the only door into a generation \
             is a row id. Add `rows()` and an entry in GENERATIONS, or say in \
             NO_ROWS_YET why the generation has none yet."
        );
        let vanished: Vec<&&str> = stated.keys().filter(|m| !rowless.contains(*m)).collect();
        assert!(
            vanished.is_empty(),
            "{vanished:?} are listed as having no rows but declare no module — \
             delete the line"
        );
    }

    /// An id names a model, never an encoding of one — so it carries no
    /// quantization word. A row per encoding would quadruple the table
    /// to record something [`crate::shared::policy`] already decides.
    #[test]
    fn an_id_names_a_model_and_not_an_encoding_of_one() {
        for id in ids() {
            for word in ["fp8", "int4", "int8", "awq", "gptq", "mxfp4", "bf16", "mlx"] {
                assert!(
                    !id.contains(word),
                    "'{id}' names an encoding; quantization is policy, not identity",
                );
            }
        }
    }

    /// Ids are unique, because `find` answers with the first match and
    /// a duplicate would make the second unreachable rather than loud.
    #[test]
    fn no_two_rows_share_an_id() {
        let mut seen = std::collections::BTreeSet::new();
        for id in ids() {
            assert!(seen.insert(id), "'{id}' appears twice in the catalog");
        }
    }

    /// The id spelling, held to one shape so a boundary that carries it
    /// and an operator who types it agree.
    #[test]
    fn ids_are_lowercase_and_hyphenated() {
        for id in ids() {
            assert!(
                id.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.'),
                "'{id}' is not lowercase-hyphenated",
            );
            assert!(
                !id.is_empty() && !id.starts_with('-') && !id.ends_with('-'),
                "'{id}'"
            );
        }
    }

    /// A build that did not ask for test rows has none.
    ///
    /// The closed set's guarantee is that a checkpoint is a known model or
    /// it is refused, and a row nobody trained is the one way to weaken it
    /// by accident. `crate::test_rows` exists because a test that writes
    /// real safetensors cannot afford a real row, and this is the fence
    /// around it: the feature is off by default, no other feature turns it
    /// on, and if either of those ever stopped being true this fails.
    ///
    /// It checks the PREFIX rather than the module, because the prefix is
    /// what a shipped catalog can be inspected for -- `pie model ls` shows
    /// it to an operator, and a `test-` id in that list is a bug report.
    /// `test_rows::every_test_row_says_so_in_its_id` holds the other end.
    #[test]
    #[cfg(not(feature = "test-rows"))]
    fn a_shipped_catalog_has_no_test_rows() {
        let leaked: Vec<&str> = ids()
            .into_iter()
            .filter(|id| id.starts_with("test-"))
            .collect();
        assert!(
            leaked.is_empty(),
            "{leaked:?} are in a catalog built without `test-rows`",
        );
    }

    /// And a build that DID ask has exactly what it asked for.
    ///
    /// The other half of the same claim: without this, the check above
    /// could pass because the feature does nothing.
    #[test]
    #[cfg(feature = "test-rows")]
    fn asking_for_test_rows_is_what_puts_them_there() {
        let present: Vec<&str> = ids()
            .into_iter()
            .filter(|id| id.starts_with("test-"))
            .collect();
        assert_eq!(present.len(), crate::test_rows::rows().len());
        assert!(!present.is_empty(), "`test-rows` added nothing");
    }

    /// Every row's manifest matches the row's OWN implied tensors.
    ///
    /// Trivially true when `manifest()` is a projection, and that is
    /// what it checks: a manifest written by hand beside the numbers
    /// instead of derived from them would drift, and this fails the
    /// day it does.
    #[test]
    fn every_row_satisfies_its_own_manifest() {
        for row in catalog() {
            let manifest = row.manifest();
            let implied = crate::manifest::Observed::from_pairs(
                manifest
                    .tensors
                    .iter()
                    .filter(|t| t.presence != crate::manifest::Presence::Absent)
                    .map(|t| (t.name.replace("{}", "0"), t.extents.clone())),
            );
            assert!(
                manifest.check(&implied).is_ok(),
                "{}: manifest does not describe itself",
                row.id(),
            );
        }
    }

    /// Every row states a layer count, because every per-layer answer
    /// below it is that long.
    #[test]
    fn every_row_has_layers() {
        for row in catalog() {
            assert!(row.manifest().layers > 0, "{} states no layers", row.id());
        }
    }

    /// A typo suggests the row it is a typo of.
    #[test]
    fn a_near_miss_id_suggests_the_row_it_missed() {
        assert_eq!(edit_distance("qwen3-0.6", "qwen3-0.6b"), 1);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
    }

    /// The suggestion list is bounded and ordered.
    #[test]
    fn nearest_is_bounded_and_deterministic() {
        let near = nearest_ids("qwen3-0.6", 3);
        assert!(near.len() <= 3);
        assert_eq!(near, nearest_ids("qwen3-0.6", 3), "two calls, one answer");
        if !ids().is_empty() {
            assert!(
                !near.is_empty(),
                "a non-empty table always has a nearest row"
            );
        }
    }

    /// A refusal that names nothing is worse than no refusal at all,
    /// because the operator has to go and find the list themselves.
    #[test]
    fn an_unknown_id_says_what_it_would_have_accepted() {
        let e = Unmatched::NoSuchId {
            id: "qwen3-0.6".into(),
            nearest: vec!["qwen3-0.6b", "qwen3-1.7b"],
        };
        let text = e.to_string();
        assert!(text.contains("qwen3-0.6b"), "{text}");
        assert!(text.contains("did you mean"), "{text}");

        let bare = Unmatched::NoSuchId {
            id: "nothing".into(),
            nearest: vec![],
        };
        assert!(bare.to_string().contains("pie model list"), "{bare}");
    }

    /// The two table defects have their own words, because they call for
    /// different repairs: a `NoRow` is a checkpoint this build does not
    /// serve, an `Ambiguous` is two rows that should have been one.
    #[test]
    fn the_two_table_defects_read_differently() {
        let no_row = Unmatched::NoRow {
            nearest: vec![("qwen3-0.6b", "layer.0.q_proj: extents differ".into())],
        };
        let text = no_row.to_string();
        assert!(text.contains("matches no model"), "{text}");
        assert!(
            text.contains("q_proj"),
            "the diff is the useful part: {text}"
        );

        let ambiguous = Unmatched::Ambiguous {
            ids: vec!["a", "b"],
        };
        let text = ambiguous.to_string();
        assert!(text.contains("two rows"), "{text}");
        assert!(text.contains('a') && text.contains('b'), "{text}");
    }

    /// `LoadShape`'s two constructors state their zeros.
    #[test]
    fn a_load_shape_states_its_zeros() {
        let dense = LoadShape::dense(28, 128, true);
        assert_eq!(dense.n_experts, 0);
        assert_eq!(dense.mamba_groups, 0);
        assert_eq!(dense.kv_shared_layers, 0);
        assert_eq!(dense.layers, 28);
        assert_eq!(dense.head_dim, 128);
        assert!(dense.tied_embeddings);

        let moe = LoadShape::mixture(48, 128, 128, false);
        assert_eq!(moe.n_experts, 128);
        assert_eq!(moe.mamba_groups, 0);
        assert!(!moe.tied_embeddings);
        assert_ne!(dense, moe);
    }

    /// `Deployed::single` is the one-GPU load, and it is a value rather
    /// than a `Default` so a caller cannot get it by forgetting.
    #[test]
    fn a_single_device_load_carries_no_scalars() {
        let d = Deployed::single();
        assert_eq!(d.tp_size, 1);
        assert!(d.layer_scalars.is_empty());
    }

    /// The label list is coarser than the id list, and every label is a
    /// plain lowercase family word.
    ///
    /// The property that matters: a label carries no size and no task
    /// suffix. `qwen3` and not `qwen3-8b`, `gemma4` and not
    /// `gemma4forconditionalgeneration` — the second is the exact
    /// string the deleted `arch_stem` produced when its strip list was
    /// one entry short.
    #[test]
    fn a_family_label_is_coarser_than_an_id() {
        let arches = arches();
        assert!(!arches.is_empty(), "some row must advertise a family");
        assert!(
            arches.len() < ids().len(),
            "a label per row is not a family label: {arches:?}"
        );
        for a in &arches {
            assert_eq!(*a, a.to_lowercase(), "'{a}' is not lowercase");
            assert!(!a.contains("for"), "'{a}' still carries a task suffix");
            assert!(!a.contains(' '), "'{a}' is not one word");
        }
        let mut sorted = arches.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(arches, sorted, "the list must be sorted and deduplicated");
    }

    /// EVERY SERVABLE ROW ADVERTISES A LABEL AND A CEILING.
    ///
    /// `arches()` drops an empty label, which is the right answer for
    /// the LIST and the wrong place to discover the hole: a generation
    /// that leaves `advertised` at `Default::default()` simply vanishes
    /// from it, and the first sign is the worker refusing a real
    /// checkpoint because the family it derived is not in a list that
    /// silently omits it. So the emptiness is caught here, at the table.
    ///
    /// `max_model_len` rides along because it fails the same way and
    /// worse: zero is not dropped anywhere, it is ADVERTISED, and a
    /// guest program that asks how much context it has is told none.
    #[test]
    fn a_servable_row_advertises_a_label_and_a_ceiling() {
        let mut silent = Vec::new();
        for row in catalog() {
            let Ok(d) = row.deployment(Deployed::single()) else {
                continue;
            };
            if d.advertised.arch.is_empty() {
                silent.push(format!("{}: no family label", row.id()));
            }
            if d.advertised.max_model_len == 0 {
                silent.push(format!("{}: no context ceiling", row.id()));
            }
        }
        assert!(
            silent.is_empty(),
            "a row this build serves advertises nothing, so it is invisible to \
             `arches()` and reports zero context to a guest:\n  {}",
            silent.join("\n  ")
        );
    }

    /// A row that refuses to deploy refuses to trace too.
    ///
    /// The two questions are one question — "does this build serve this
    /// row" — and a row that answered them differently would load
    /// happily and die at its first fire, which is the failure
    /// `unbuilt_kv_store()` existed to paper over and this design
    /// removes rather than papers.
    #[test]
    fn a_row_that_cannot_deploy_cannot_trace_either() {
        use model_ir::trace::FireClass;
        for row in catalog() {
            let deploys = row.deployment(Deployed::single()).is_ok();
            for class in [FireClass::Prefill, FireClass::Decode] {
                let traces = row.trace(class, Deployed::single()).is_ok();
                assert!(
                    deploys || !traces,
                    "{} refuses to deploy and then traces {class:?} anyway, so \
                     the door and the fire disagree about what this build serves",
                    row.id()
                );
            }
        }
    }
}

/// [`identify`] against the catalog itself, which is the only oracle
/// there is for it.
///
/// Decision 1 of the redesign is that identity IS validation: a
/// checkpoint is matched by its tensors, so a `config.json` has nowhere
/// left to lie. That decision is only sound if the table can actually
/// tell its rows apart, and nothing else in this crate asks whether it
/// can — `no_two_rows_share_an_id` checks the NAMES, and two rows with
/// distinct names and indistinguishable manifests is precisely the
/// defect `Unmatched::Ambiguous` exists to report.
#[cfg(all(test, feature = "contract"))]
pub(crate) mod identify_tests {
    use super::*;
    use crate::manifest::Presence;
    use model_loader::checkpoint::{CheckpointMetadata, RawTensor};
    use model_loader::types::{DType, Encoding, FileId, TensorId};

    /// The checkpoint a row DESCRIBES, as a real `CheckpointMetadata`.
    ///
    /// Built from the row's own manifest, so it is the same projection
    /// `every_row_satisfies_its_own_manifest` uses — only through the
    /// public door `identify` actually takes, which is what makes it a
    /// test of `identify` rather than of `Manifest::check`.
    ///
    /// `Absent` rows are dropped and `Optional` ones kept: an optional
    /// name matches either way, and including it exercises the arm that
    /// must not turn a match into a miss.
    pub(crate) fn checkpoint_of(row: &dyn Variant) -> CheckpointMetadata {
        let manifest = row.manifest();
        let tensors = manifest
            .tensors
            .iter()
            .filter(|t| t.presence != Presence::Absent)
            .enumerate()
            .map(|(i, t)| {
                let extents = if t.extents.is_empty() {
                    vec![1]
                } else {
                    t.extents.clone()
                };
                let elems: u64 = extents.iter().product();
                RawTensor {
                    id: TensorId(u32::try_from(i).unwrap_or(0)),
                    // `{}` survives `Observed::logical` untouched — the
                    // rewrite is idempotent on it — so a spec name is
                    // already a checkpoint name for this purpose.
                    name: t.name.clone(),
                    file_id: FileId(0),
                    file_offset: 0,
                    span_bytes: elems * 2,
                    shape: extents
                        .iter()
                        .map(|&e| i64::try_from(e).unwrap_or(0))
                        .collect(),
                    encoding: Encoding::Raw(DType::BF16),
                }
            })
            .collect();
        CheckpointMetadata {
            files: Vec::new(),
            tensors,
        }
    }

    /// NO TWO ROWS ARE INDISTINGUISHABLE BY TENSORS, except where the
    /// table says so out loud.
    ///
    /// The strongest statement this file can make, and the one the whole
    /// design rests on: hand `identify` the tensors a row implies and it
    /// must answer THAT row, not "ambiguous" and not a sibling. A
    /// generation that copies a neighbour's `manifest()` and forgets to
    /// change a number fails here, at the table, instead of on someone's
    /// checkpoint.
    ///
    /// The one exception is [`GEOMETRIC_TWINS`], and it is an exception
    /// about the MODELS rather than about the check: Llama-3.3-70B is
    /// 3.1-70B's geometry exactly, retrained, so no manifest can separate
    /// them and the honest thing is to say so instead of asserting
    /// something false. Ambiguity outside that table still fails, and an
    /// entry that stops being ambiguous fails too — a stale declaration
    /// is how an exception turns into a hole.
    #[test]
    fn every_row_is_identified_as_itself_and_not_as_a_sibling() {
        let mut collisions: Vec<String> = Vec::new();
        let mut twinned: Vec<&str> = Vec::new();
        for row in catalog() {
            let metadata = checkpoint_of(*row);
            match identify(&metadata, &Override::None) {
                Ok(found) if found.id() == row.id() => {}
                Ok(found) => collisions.push(format!("{} identified as {}", row.id(), found.id())),
                Err(Unmatched::Ambiguous { ids }) if are_declared_twins(&ids) => {
                    twinned.push(row.id());
                }
                Err(Unmatched::Ambiguous { ids }) => {
                    collisions.push(format!("{} is ambiguous with {ids:?}", row.id()));
                }
                Err(e) => collisions.push(format!("{} did not identify: {e}", row.id())),
            }
        }
        assert!(
            collisions.is_empty(),
            "identification is not one-to-one, so a checkpoint can load as a \
             model it is not — which is the one thing the manifest exists to \
             make impossible. If a pair here is genuinely one geometry under \
             two release names, declare it in `GEOMETRIC_TWINS`; otherwise a \
             manifest is wrong:\n  {}",
            collisions.join("\n  ")
        );
        for set in GEOMETRIC_TWINS {
            for id in *set {
                assert!(
                    twinned.contains(id),
                    "{id} is declared a geometric twin and identifies cleanly \
                     anyway, so the declaration is stale — drop it, or the \
                     next real collision hides behind it",
                );
            }
        }
    }

    /// A declared twin is a CHOICE the caller makes, not a guess the
    /// table makes for them.
    ///
    /// The exception above buys nothing if the twins are unloadable, so
    /// this is the other half: [`Override::Id`] resolves the ambiguity,
    /// and it does it without weakening the check — the checkpoint is
    /// still held to the named row's manifest.
    #[test]
    fn a_declared_twin_still_loads_when_the_caller_names_it() {
        for set in GEOMETRIC_TWINS {
            for id in *set {
                let row =
                    find(id).unwrap_or_else(|| panic!("{id} is declared but not in the catalog"));
                let metadata = checkpoint_of(row);
                let found = identify(&metadata, &Override::Id((*id).to_string()))
                    .unwrap_or_else(|e| panic!("{id} named explicitly and still refused: {e}"));
                assert_eq!(found.id(), *id);
            }
        }
    }

    /// A checkpoint nothing describes is REFUSED, with the near misses.
    ///
    /// The closed set's honest cost, made visible: an unknown model does
    /// not fall through to a plausible row. It is turned away, and the
    /// refusal carries a structural diff so the answer to "why not" is
    /// in the error rather than in a debugger.
    #[test]
    fn a_checkpoint_no_row_describes_is_refused_with_its_near_misses() {
        let metadata = CheckpointMetadata {
            files: Vec::new(),
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "embed_tokens".to_string(),
                file_id: FileId(0),
                file_offset: 0,
                span_bytes: 8,
                shape: vec![2, 2],
                encoding: Encoding::Raw(DType::BF16),
            }],
        };
        let Err(Unmatched::NoRow { nearest }) = identify(&metadata, &Override::None) else {
            panic!("a two-by-two embedding is no model in this catalog");
        };
        assert!(
            !nearest.is_empty(),
            "a refusal with no near miss is not a diagnosis"
        );
        assert!(nearest.len() <= 3, "the three closest, not the whole table");
        for (id, why) in &nearest {
            assert!(!id.is_empty() && !why.is_empty(), "{id}: {why}");
        }
    }

    /// An override NAMES a row; it does not skip the check.
    ///
    /// The escape hatch for a fine-tune or a renamed mirror. Honest cost
    /// 1 of the design says it must still refuse a manifest mismatch,
    /// because a hatch that skipped validation would be a way to load a
    /// checkpoint as something it is not — reintroducing, by hand,
    /// exactly the silent-wrong-answer the `config.json` path had.
    #[test]
    fn an_override_names_a_row_and_still_holds_it_to_the_manifest() {
        let row = *catalog().first().expect("the catalog is not empty");
        let chosen = Override::Id(row.id().to_string());

        let matching = checkpoint_of(row);
        assert_eq!(
            identify(&matching, &chosen)
                .map(|r| r.id())
                .unwrap_or("<refused>"),
            row.id(),
            "the named row accepts the checkpoint it describes",
        );

        // The same request against a checkpoint that is plainly not it.
        let empty = CheckpointMetadata {
            files: Vec::new(),
            tensors: Vec::new(),
        };
        let refused = identify(&empty, &chosen);
        assert!(
            matches!(refused, Err(Unmatched::NoRow { .. })),
            "an override must not turn a mismatch into a load; got {:?}",
            refused.map(|r| r.id()),
        );
    }

    /// An override naming an id nothing carries is a typo, and is
    /// answered as one.
    #[test]
    fn an_override_with_an_unknown_id_suggests_rather_than_guesses() {
        let real = catalog().first().expect("the catalog is not empty").id();
        let typo = format!("{real}x");
        let metadata = CheckpointMetadata {
            files: Vec::new(),
            tensors: Vec::new(),
        };
        let Err(Unmatched::NoSuchId { id, nearest }) =
            identify(&metadata, &Override::Id(typo.clone()))
        else {
            panic!("'{typo}' names no row");
        };
        assert_eq!(id, typo);
        assert!(
            nearest.contains(&real),
            "the nearest ids must include the one a single character away: {nearest:?}",
        );
    }

    /// `Override::None` is the default, so a caller that does not care
    /// gets the matching path rather than an empty id.
    #[test]
    fn no_override_is_the_default() {
        assert_eq!(Override::default(), Override::None);
    }
}
