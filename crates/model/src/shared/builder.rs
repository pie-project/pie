//! The toolkit a model family uses to state what a driver binds.
//!
//! The Rust port of the CUDA driver's `model/contract.hpp::ContractBuilder`,
//! kept semantically identical during the migration — same knobs, same
//! passes, same declaration order — because the migration is proved by
//! diffing the two authors' output over the same checkpoint. Divergence is
//! a bug in the port until the C++ author is deleted; after that this file
//! is the definition.
//!
//! A family's author receives this, sets the handful of knobs its layout
//! needs, runs the shared passes it wants, adds whatever is peculiar to it,
//! and finishes with [`Builder::publish_remaining`]. The order is not
//! incidental: a pass that fuses q/k/v into one buffer has to claim those
//! tensors before the generic tail publishes them separately.
//!
//! What the C++ builder read from three places arrives as four arguments:
//! the checkpoint's own tensor table, the [`LoadShape`](crate::catalog::LoadShape)
//! and [`Encoding`](crate::encoding::Encoding) that replaced the parsed
//! `ModelFacts`, and the [`StorageTarget`] + [`Policy`] pair — the device's
//! measurements and the caller's decisions, exactly the split
//! [`policy`](crate::shared::policy) argues for.

use std::collections::{HashMap, HashSet};

use model_loader::checkpoint::{CheckpointMetadata, RawTensor};
use model_loader::contract::{
    Expr, GroupContract, ModelContract, Scales, TensorContract, TensorType,
};
use model_loader::error::Error;
use model_loader::plan::StorageTarget;
use model_loader::types::{
    Axis, DType, Encoding, QuantGranularity, QuantScheme, QuantSpec, RepackLayout, ScaleForm,
    TensorId, Visibility,
};

use super::policy::{Component, Mxfp4MoePolicy, Mxfp4MoeRequest, Naming, Policy, RuntimeQuant};
use super::probe;

/// Authoring failures are contract errors: the family's model of the
/// checkpoint is wrong, and the message names the tensor.
/// Refuse, with a message naming what went wrong.
///
/// `pub` because a generation crate refuses for its own reasons — an
/// unstacked expert bank, a tensor the schema has no opinion on — and the
/// refusal has to read the same wherever it comes from.
pub fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

/// The dtype a consumer sees, whatever the storage format is.
pub(crate) fn logical_dtype(encoding: &Encoding) -> DType {
    match encoding {
        Encoding::Raw(dtype) => *dtype,
        Encoding::Quant(spec) => spec.logical_dtype,
    }
}

/// Whether an encoding is the plain, unquantized `dtype`.
///
/// `pub` because it is how a generation asks "did this checkpoint ship this
/// tensor already quantized", which decides whether its pass runs at all.
pub fn is_raw(encoding: &Encoding, dtype: DType) -> bool {
    matches!(encoding, Encoding::Raw(have) if *have == dtype)
}

/// Resolve the caller's MoE request against what the device can do.
///
/// The driver measured `native_mxfp4_moe`, so the driver is what turns
/// `Auto` into an answer: a native MXFP4 GEMM is always the better path when
/// the kernels exist, and decoding on the routed path is the fallback that
/// works everywhere.
fn resolve_mxfp4_moe(request: Mxfp4MoeRequest, native_mxfp4_moe: bool) -> Mxfp4MoePolicy {
    match request {
        Mxfp4MoeRequest::RoutedDecode => Mxfp4MoePolicy::RoutedDecode,
        Mxfp4MoeRequest::NativeGemm => Mxfp4MoePolicy::NativeGemm,
        Mxfp4MoeRequest::EagerBf16 => Mxfp4MoePolicy::EagerBf16,
        Mxfp4MoeRequest::Auto => {
            if native_mxfp4_moe {
                Mxfp4MoePolicy::NativeGemm
            } else {
                Mxfp4MoePolicy::RoutedDecode
            }
        }
    }
}

/// Which weights a runtime-quant request re-encodes, given the target scheme.
///
/// A statement about the CUDA driver's GEMMs, not about any model: the
/// allowlist is exactly the projections a quantized kernel exists for. That
/// is why it lives beside the lowering rather than in
/// [`probe`](crate::shared::probe).
fn runtime_quantizable_name(name: &str, scheme: QuantScheme) -> bool {
    if scheme == QuantScheme::Mxfp4E2M1E8M0 {
        // FP4 reaches experts only. Attention projections stay FP8-plus-scale
        // (block-scaled GEMM) because this hardware has no FP4 GEMM for them.
        return probe::is_expert_projection(name);
    }
    [
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".self_attn.q_a_proj.weight",
        ".self_attn.q_b_proj.weight",
        ".self_attn.kv_a_proj_with_mqa.weight",
        ".self_attn.kv_b_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    ]
    .iter()
    .any(|tail| name.ends_with(tail))
        || probe::is_expert_projection(name)
}

/// A dense join a family pass proposes; see [`Builder::fused_join_candidate`].
pub struct FusedCandidate<'a> {
    output_name: String,
    /// The source tensors, in join order.
    parts: Vec<&'a RawTensor>,
    cols: i64,
    bytes: u64,
}

/// Accumulates one family's declarations against one checkpoint.
pub struct Builder<'a> {
    /// Which model this is, so an error can NAME it.
    ///
    /// A row's id, not a `model_type`. The difference matters at the
    /// two `fail()` sites below: `model_type='qwen3'` named a
    /// DISPATCH KEY that a dozen checkpoints shared, and `qwen3-30b-a3b`
    /// names the model whose contract actually failed to author.
    id: &'static str,
    shape: crate::catalog::LoadShape,
    encoding: &'a crate::encoding::Encoding,
    target: &'a StorageTarget,
    policy: &'a Policy,
    mxfp4_moe: Mxfp4MoePolicy,
    tensors: Vec<&'a RawTensor>,
    by_name: HashMap<&'a str, &'a RawTensor>,
    consumed: HashSet<TensorId>,
    contract: ModelContract,

    source_prefix: String,
    decoder_layer_prefix: String,
    shard_axis_fn: fn(&str) -> Result<Option<u8>, Error>,
    extra_join_modules: Vec<String>,
    shard_embed_tokens: bool,
    replicate_lm_head: bool,
    allow_bf16_rq: bool,
    allow_mxfp4_rq: bool,
    encode_scope_allowed: bool,
}

impl<'a> Builder<'a> {
    /// A builder for one row against one checkpoint.
    ///
    /// It used to take a `&ModelFacts` — eleven fields parsed out of a
    /// `config.json`, of which the authors read five. The five are
    /// [`LoadShape`](crate::catalog::LoadShape), which comes from the
    /// ROW and therefore cannot disagree with what the same row deploys
    /// and traces; the three quantization fields are
    /// [`Encoding`](crate::encoding::Encoding), which comes from the
    /// FILE because that is what an encoding is a property of; and the
    /// other three are gone with the dispatch they keyed.
    pub fn new(
        metadata: &'a CheckpointMetadata,
        id: &'static str,
        shape: crate::catalog::LoadShape,
        encoding: &'a crate::encoding::Encoding,
        target: &'a StorageTarget,
        policy: &'a Policy,
    ) -> Self {
        // `weights()` and not the raw table: a `.zt` source carries metadata
        // objects a contract must never see, and a snapshot has none, so the
        // filter is free where it is not load-bearing.
        let tensors: Vec<&RawTensor> = metadata.weights().collect();
        let mut by_name = HashMap::with_capacity(tensors.len());
        for raw in &tensors {
            by_name.insert(raw.name.as_str(), *raw);
        }
        Self {
            id,
            shape,
            encoding,
            target,
            policy,
            mxfp4_moe: resolve_mxfp4_moe(policy.moe_request, target.native_mxfp4_moe),
            tensors,
            by_name,
            consumed: HashSet::new(),
            contract: ModelContract {
                alignment: target.preferred_alignment.max(1),
                tensors: Vec::new(),
                groups: Vec::new(),
            },
            source_prefix: String::new(),
            decoder_layer_prefix: "model.layers.".to_string(),
            shard_axis_fn: default_shard_axis,
            extra_join_modules: Vec::new(),
            shard_embed_tokens: false,
            replicate_lm_head: false,
            allow_bf16_rq: false,
            allow_mxfp4_rq: false,
            encode_scope_allowed: false,
        }
    }

    // -- what the family is authoring against --------------------------------

    /// The row's id, for a refusal that names a MODEL.
    #[must_use]
    pub fn id(&self) -> &'static str {
        self.id
    }

    /// The six shape facts no tensor extent carries.
    #[must_use]
    pub fn shape(&self) -> crate::catalog::LoadShape {
        self.shape
    }

    /// What the checkpoint's files declare about how they are stored.
    #[must_use]
    pub fn encoding(&self) -> &crate::encoding::Encoding {
        self.encoding
    }

    /// Which tensor names the driver asking for this contract binds.
    ///
    /// A ROW reads this, which is the one place the catalog is not a
    /// pure function of the model: the same Qwen3 authors differently
    /// for a driver that binds HuggingFace names and one that binds
    /// MLX's. That used to be two TABLES (`HF_ROWS` and `MLX_ROWS`,
    /// keyed on the same strings, seventeen rows of the second
    /// duplicating a subset of the thirty-five of the first), and two
    /// tables is how a model gets an HF author and no MLX one and
    /// nobody notices until a Metal boot. One row, one `match`, and the
    /// arms are exhaustive because `Naming` is an enum.
    #[must_use]
    pub fn naming(&self) -> Naming {
        self.policy.naming
    }

    pub fn target(&self) -> &StorageTarget {
        self.target
    }

    /// How this contract decided to hand MXFP4 experts to the driver.
    pub fn mxfp4_moe(&self) -> Mxfp4MoePolicy {
        self.mxfp4_moe
    }

    /// What the caller asked for, before any rule was applied. Only an
    /// author with a reason to disagree with the device rule needs this.
    pub fn mxfp4_moe_request(&self) -> Mxfp4MoeRequest {
        self.policy.moe_request
    }

    pub fn stream_routed_experts(&self) -> bool {
        self.policy.stream_routed_experts
    }

    /// The load-time requantization the caller asked for, already resolved
    /// against the device.
    ///
    /// The Builder's own `push_runtime_quant` serves the families that publish
    /// through it; the MLX lowering has its own authoring loop and reads the
    /// request here instead of growing a second policy field to mean the same
    /// thing.
    pub fn runtime_quant(&self) -> super::policy::RuntimeQuant {
        self.policy.runtime_quant
    }

    /// The per-family switches the caller resolved from its environment.
    pub fn knobs(&self) -> &super::policy::FamilyKnobs {
        &self.policy.knobs
    }

    pub fn tensors(&self) -> &[&'a RawTensor] {
        &self.tensors
    }

    // -- knobs, each one a claim about this family ---------------------------

    /// Source tensors *may* live under this prefix; it is stripped from
    /// output names. A prefix the checkpoint does not use is not a prefix —
    /// the knob asks the checkpoint rather than declaring.
    pub fn source_prefix(&mut self, prefix: &str) {
        self.source_prefix.clear();
        if self.tensors.iter().any(|raw| raw.name.starts_with(prefix)) {
            self.source_prefix = prefix.to_string();
        }
    }

    /// The name a source tensor is published under, given the prefix.
    pub fn output_name(&self, raw_name: &str) -> String {
        raw_name
            .strip_prefix(self.source_prefix.as_str())
            .unwrap_or(raw_name)
            .to_string()
    }

    /// The name a source tensor has in the checkpoint, given the prefix.
    pub fn source_name(&self, bound_name: &str) -> String {
        format!("{}{bound_name}", self.source_prefix)
    }

    /// Where the decoder's layers are named, up to the layer index.
    pub fn decoder_layer_prefix(&mut self, prefix: &str) {
        self.decoder_layer_prefix = prefix.to_string();
    }

    /// The prefix set above, for passes that filter `tensors()` by it.
    pub fn decoder_layer_prefix_value(&self) -> &str {
        &self.decoder_layer_prefix
    }

    /// Adopt whichever of `candidates` the checkpoint names layer 0 with.
    /// Order is preference: the first candidate the checkpoint uses wins.
    pub fn decoder_layer_prefix_any_of(&mut self, candidates: &[&str]) {
        for candidate in candidates {
            let layer_zero = self.source_name(&format!("{candidate}0."));
            if self
                .tensors
                .iter()
                .any(|raw| raw.name.starts_with(&layer_zero))
            {
                self.decoder_layer_prefix = candidate.to_string();
                return;
            }
        }
    }

    /// Tensor-parallel shard-axis strategy, keyed by tensor name. Defaults to
    /// the HF convention; a family whose checkpoint names the same operator
    /// differently registers its own.
    pub fn shard_axis_fn(&mut self, f: fn(&str) -> Result<Option<u8>, Error>) {
        self.shard_axis_fn = f;
    }

    /// Answer the caller's MXFP4 MoE request for this family, when the device
    /// rule does not apply to it.
    pub fn decide_mxfp4_moe(&mut self, policy: Mxfp4MoePolicy) {
        self.mxfp4_moe = policy;
    }

    /// Shard `embed_tokens` on axis 0 under TP, to save per-rank memory.
    pub fn shard_embed_tokens(&mut self) {
        self.shard_embed_tokens = true;
    }

    /// Keep `lm_head` replicated under TP.
    pub fn replicate_lm_head(&mut self) {
        self.replicate_lm_head = true;
    }

    /// BF16 -> FP8/INT8 runtime quant is wired for this family.
    pub fn allow_bf16_runtime_quant(&mut self) {
        self.allow_bf16_rq = true;
    }

    /// FP4/MXFP4 runtime quant of routed experts is wired for this family.
    pub fn allow_mxfp4_runtime_quant(&mut self) {
        self.allow_mxfp4_rq = true;
    }

    /// This family's bind path can be scoped to the multimodal towers alone.
    pub fn allow_encode_scope(&mut self) -> Result<(), Error> {
        if self.policy.component == Component::Encode && self.target.tp_size != 1 {
            return fail("encode-scoped loading does not support tensor parallelism");
        }
        self.encode_scope_allowed = true;
        Ok(())
    }

    /// Also apply [`Self::dense_fused_projection_joins`] to a module outside
    /// the decoder stack (a speculative-decoding head at its own prefix).
    pub fn also_join_module(&mut self, module_prefix: &str) {
        self.extra_join_modules.push(module_prefix.to_string());
    }

    // -- lookups -------------------------------------------------------------

    pub fn find(&self, name: &str) -> Option<&'a RawTensor> {
        self.by_name.get(name).copied()
    }

    /// Claim `id` for a pass that has published it under some other name, so
    /// the generic tail does not publish it a second time.
    pub fn consume(&mut self, id: TensorId) {
        self.consumed.insert(id);
    }

    // -- sharding arithmetic --------------------------------------------------

    /// Record that `expr` is split across ranks along `axis`. Left alone when
    /// there is nothing to split, so a single-GPU contract is identical to
    /// one authored without sharding.
    pub fn split(&self, expr: Expr, axis: u8) -> Expr {
        if self.target.tp_size <= 1 {
            expr
        } else {
            expr.shard(axis)
        }
    }

    /// This rank's share of a `full`-long axis.
    ///
    /// A declared shape, never an offset. Divisibility is not checked here:
    /// the loader rejects an indivisible `Shard` with a message that names
    /// the tensor and the axis.
    pub fn local_extent(&self, full: i64) -> i64 {
        let world = i64::from(self.target.tp_size.max(1));
        if full % world == 0 {
            full / world
        } else {
            full
        }
    }

    /// The `[start, start + len)` band of `expr`, split across ranks.
    /// Returns the expression and the extent this rank ends up holding.
    pub fn band(&self, expr: Expr, axis: u8, start: i64, len: i64) -> (Expr, i64) {
        (
            self.split(expr.slice(axis, start, len), axis),
            self.local_extent(len),
        )
    }

    /// Partition an expression and its shape across ranks along `axis`.
    pub fn shard(&self, expr: Expr, mut shape: Vec<i64>, axis: Option<u8>) -> (Expr, Vec<i64>) {
        let world = i64::from(self.target.tp_size);
        let Some(axis) = axis else {
            return (expr, shape);
        };
        if world <= 1 {
            return (expr, shape);
        }
        let index = usize::from(axis);
        if index < shape.len() && shape[index] % world == 0 {
            shape[index] /= world;
        }
        (expr.shard(axis), shape)
    }

    pub fn shard_axis(&self, name: &str) -> Result<Option<u8>, Error> {
        if self.target.tp_size <= 1 {
            return Ok(None);
        }
        // Shard embed_tokens on axis 0 to save per-rank memory.
        if self.shard_embed_tokens && name.ends_with(".embed_tokens.weight") {
            return Ok(Some(0));
        }
        if self.replicate_lm_head && name.ends_with(".lm_head.weight") {
            return Ok(None);
        }
        // A companion scale splits exactly like the weight it scales, so ask
        // about the weight — here, once, before the family's own rule is
        // consulted, which is what makes the pairing unforgettable for a
        // family that supplied its own `shard_axis_fn`.
        for suffix in [
            ".weight_scale_inv",
            ".weight_scale",
            ".weight_packed",
            ".scale",
        ] {
            if let Some(base) = name.strip_suffix(suffix) {
                return match (self.shard_axis_fn)(&format!("{base}.weight"))? {
                    Some(axis) => Ok(Some(axis)),
                    None => (self.shard_axis_fn)(base),
                };
            }
        }
        (self.shard_axis_fn)(name)
    }

    /// Demote a companion scale to "replicate" when it has nothing to split.
    ///
    /// A scale follows its weight, but only a block or per-channel scale has
    /// an axis of its own: a per-tensor scale is one number for a whole
    /// projection and follows it vacuously. A scale that *does* have the axis
    /// and does not divide is left alone on purpose — the loader rejects it
    /// by name.
    fn splittable_axis(name: &str, shape: &[i64], axis: Option<u8>) -> Option<u8> {
        let index = usize::from(axis?);
        if !probe::is_companion_scale(name) {
            return axis;
        }
        if index >= shape.len() || shape[index] <= 1 {
            return None;
        }
        axis
    }

    /// A row-parallel attention projection has to split on a head boundary.
    ///
    /// The loader asks whether `tp_size` divides the row count, and for a
    /// projection of `heads * head_dim` rows that is the weaker question.
    /// The sharper one belongs here, where `head_dim` is known and the
    /// loader's is not.
    fn check_head_granularity(
        &self,
        name: &str,
        shape: &[i64],
        axis: Option<u8>,
    ) -> Result<(), Error> {
        let d = i64::from(self.shape.head_dim);
        let world = i64::from(self.target.tp_size);
        if axis != Some(0) || world <= 1 || d <= 0 || shape.is_empty() {
            return Ok(());
        }
        if ![
            ".q_proj.weight",
            ".k_proj.weight",
            ".v_proj.weight",
            ".q_proj.bias",
            ".k_proj.bias",
            ".v_proj.bias",
        ]
        .iter()
        .any(|tail| name.ends_with(tail))
        {
            return Ok(());
        }
        // A fused or otherwise non-head-shaped projection is not this rule's
        // business; `local_range` still has the row count covered.
        if shape[0] % d != 0 {
            return Ok(());
        }
        let heads = shape[0] / d;
        if heads % world == 0 {
            return Ok(());
        }
        fail(format!(
            "{name} is {heads} head(s) of {d}, which tp_size {world} does not \
             divide; a rank cannot hold part of an attention head, so use a \
             tp_size that divides the head count or run single-GPU"
        ))
    }

    // -- publishing ----------------------------------------------------------

    /// Declare `expr` under `name`, unless this component is not supposed to
    /// hold it. Returns the entry's index only when the tensor was really
    /// published, so a later pass can attach declarations to it.
    pub fn define(
        &mut self,
        name: String,
        expr: Expr,
        encoding: Encoding,
        shape: Option<Vec<i64>>,
    ) -> Option<usize> {
        if self.policy.component == Component::Encode && !probe::is_tower_output(&name) {
            return None;
        }
        self.contract.tensors.push(TensorContract {
            name,
            expr,
            shape,
            encoding,
            scales: None,
            visibility: Default::default(),
        });
        Some(self.contract.tensors.len() - 1)
    }

    /// Publish a source tensor under `output`, sharded per its name's rule.
    pub fn push_direct(
        &mut self,
        raw: &RawTensor,
        output: String,
        axis: Option<u8>,
    ) -> Result<Option<usize>, Error> {
        let axis = Self::splittable_axis(&raw.name, &raw.shape, axis);
        self.check_head_granularity(&raw.name, &raw.shape, axis)?;
        let (expr, shape) = self.shard(Expr::src(&raw.name), raw.shape.clone(), axis);
        // A rank-0 tensor gets no shape prediction. The C ABI cannot tell
        // `Some([])` from `None` — an empty slice reads back as "no claim" —
        // so the C++ author's rank-0 `expect` lands as `None`, and matching
        // that observable is what keeps the differential byte-exact. (gemma-4
        // ships scalars; this is not hypothetical.)
        let shape = if shape.is_empty() { None } else { Some(shape) };
        Ok(self.define(output, expr, raw.encoding.clone(), shape))
    }

    /// Publish `expr` under `output` at the source's logical dtype, and claim
    /// the source.
    pub fn push_expr(&mut self, output: String, raw: &RawTensor, shape: Vec<i64>, expr: Expr) {
        self.define(
            output,
            expr,
            Encoding::Raw(logical_dtype(&raw.encoding)),
            Some(shape),
        );
        self.consumed.insert(raw.id);
    }

    /// Publish `src` under `output`, relaid out by `layout`.
    ///
    /// `src` is an expression, not a name, because the selection a kernel
    /// used to carry as integers — this rank's rows, the interleaved half,
    /// the column band — is stated in the algebra now. `consume` the
    /// checkpoint tensor separately if the generic tail should not republish
    /// it.
    pub fn push_repack(
        &mut self,
        output: String,
        src: Expr,
        layout: RepackLayout,
        encoding: Encoding,
        shape: Vec<i64>,
    ) -> Option<usize> {
        let node = src.repack(layout, TensorType::new(shape.clone(), encoding.clone()));
        self.define(output, node, encoding, Some(shape))
    }

    /// Keep a published declaration out of the driver's namespace.
    pub fn mark_internal(&mut self, index: usize) {
        self.contract.tensors[index].visibility = Visibility::Internal;
    }

    /// Declare that entry `index` holds the scales for `of`.
    pub fn set_scales(&mut self, index: usize, scales: Scales) {
        self.contract.tensors[index].scales = Some(scales);
    }

    /// Add a grid of interchangeable tensor sets, written once.
    pub fn push_group(&mut self, group: GroupContract) {
        self.contract.groups.push(group);
    }

    // -- shared passes -------------------------------------------------------

    /// Re-join each rank's gate and up bands of a pre-fused expert tensor.
    ///
    /// `gate_second` publishes each expert's halves as `[up|gate]` instead of
    /// the checkpoint's `[gate|up]` — the order flashinfer's CUTLASS MoE reads
    /// fc1's output in — and applies with or without sharding, so unlike the
    /// slicing this runs at `tp_size == 1` too.
    pub fn fused_moe_gate_up_tp_slices(&mut self, gate_second: bool) -> Result<(), Error> {
        let sharding = self.target.tp_size > 1;
        if !sharding && !gate_second {
            return Ok(());
        }
        for raw in self.tensors.clone() {
            if !self.source_name_allowed(&raw.name) {
                continue;
            }
            if !raw.name.ends_with(".experts.gate_up_proj")
                && !raw.name.ends_with(".mlp.experts.gate_up_proj")
            {
                continue;
            }
            if raw.shape.len() != 3 || raw.shape[1] % 2 != 0 {
                continue;
            }
            if !probe::is_dense_addressable(&raw.encoding) {
                return fail(format!(
                    "fused MoE gate/up '{}' has a non-affine packed encoding",
                    raw.name
                ));
            }
            let experts = raw.shape[0];
            let full_i = raw.shape[1] / 2;
            let hidden = raw.shape[2];
            // Gate rows [0, I) and up rows [I, 2I) are sharded independently
            // and re-joined, so the local halves stay adjacent per expert.
            let src = Expr::src(&raw.name);
            let (gate, local_i) = if sharding {
                self.band(src.clone(), 1, 0, full_i)
            } else {
                (src.clone().slice(1, 0, full_i), full_i)
            };
            let (up, _) = if sharding {
                self.band(src.clone(), 1, full_i, full_i)
            } else {
                (src.slice(1, full_i, full_i), full_i)
            };
            let parts = if gate_second {
                vec![up, gate]
            } else {
                vec![gate, up]
            };
            self.push_expr(
                self.output_name(&raw.name),
                raw,
                vec![experts, 2 * local_i, hidden],
                Expr::concat(1, parts),
            );
        }
        Ok(())
    }

    /// Fuse q/k/v and gate/up into one buffer each, where the GEMM wants it.
    ///
    /// A CUDA kernel decision, not a fact about any model: the same
    /// checkpoint on a backend without a fused GEMM declares the three
    /// projections separately — which is what [`Projections::InPlace`]
    /// selects, and why the whole pass consults the policy before anything
    /// else.
    ///
    /// [`Projections::InPlace`]: crate::shared::policy::Projections::InPlace
    pub fn dense_fused_projection_joins(&mut self) -> Result<(), Error> {
        use crate::shared::policy::Projections;
        if self.policy.projections == Projections::InPlace {
            return Ok(());
        }
        if self.runtime_quant_scheme()?.is_some() {
            return Ok(());
        }
        let mut qkv = Vec::new();
        let mut gate_up = Vec::new();
        let mut qkv_bytes = 0u64;
        let mut gate_up_bytes = 0u64;
        let mut modules: Vec<String> = (0..self.shape.layers)
            .map(|layer| format!("{}{layer}.", self.decoder_layer_prefix))
            .collect();
        modules.extend(self.extra_join_modules.iter().cloned());
        for p in &modules {
            let s = self.source_name(p);
            if let Some(candidate) = self.fused_join_candidate(
                format!("{p}self_attn.qkv_proj.fused.weight"),
                &[
                    format!("{s}self_attn.q_proj.weight"),
                    format!("{s}self_attn.k_proj.weight"),
                    format!("{s}self_attn.v_proj.weight"),
                ],
            ) {
                qkv_bytes += candidate.bytes;
                qkv.push(candidate);
            }
            if let Some(candidate) = self.fused_join_candidate(
                format!("{p}mlp.gate_up_proj.fused.weight"),
                &[
                    format!("{s}mlp.gate_proj.weight"),
                    format!("{s}mlp.up_proj.weight"),
                ],
            ) {
                gate_up_bytes += candidate.bytes;
                gate_up.push(candidate);
            }
        }
        if qkv.is_empty() && gate_up.is_empty() {
            return Ok(());
        }

        // Fused dense projections replace the original BF16 tensors, and the
        // unfused fallback binds non-owning views into the fused buffer, so
        // this is not a persistent duplicate-memory budget. It selects which
        // groups get a fused GEMM: all groups through 8B-class models, and
        // QKV-only above that, where gate/up fusion has regressed. Measured
        // on whole-checkpoint bytes so a model lands in the same class at
        // every `tp_size` — a proxy for model class, not for device memory.
        const BUDGET_BYTES: u64 = 10 * 1024 * 1024 * 1024;
        let mut chosen = Vec::new();
        if qkv_bytes + gate_up_bytes <= BUDGET_BYTES {
            chosen.extend(qkv);
            chosen.extend(gate_up);
        } else {
            // QKV first when the full set does not fit: it is much smaller
            // than gate/up on Qwen-style models and it is what enables the
            // fused decode postprocess, without giving up large-model KV
            // capacity.
            let mut used = 0;
            if qkv_bytes <= BUDGET_BYTES {
                used = qkv_bytes;
                chosen.extend(qkv);
            }
            if gate_up_bytes <= BUDGET_BYTES - used {
                chosen.extend(gate_up);
            }
        }
        self.publish_fused(chosen)
    }

    pub fn fused_join_candidate(
        &self,
        output_name: String,
        inputs: &[String],
    ) -> Option<FusedCandidate<'a>> {
        if self.find(&output_name).is_some() {
            return None;
        }
        let mut parts = Vec::with_capacity(inputs.len());
        let mut cols = None;
        let mut bytes = 0;
        for name in inputs {
            let raw = self.find(name)?;
            if raw.shape.len() != 2 || !is_raw(&raw.encoding, DType::BF16) {
                return None;
            }
            if cols.is_some_and(|have| raw.shape[1] != have) {
                return None;
            }
            cols = Some(raw.shape[1]);
            bytes += raw.span_bytes;
            parts.push(raw);
        }
        Some(FusedCandidate {
            output_name,
            parts,
            cols: cols.unwrap_or(0),
            bytes,
        })
    }

    pub fn publish_fused(&mut self, candidates: Vec<FusedCandidate<'a>>) -> Result<(), Error> {
        for candidate in candidates {
            // A join is only a fusion of the GEMM if every part is
            // distributed the same way. Column-parallel parts (q/k/v,
            // gate/up) each hand a rank its own band and the join of those
            // bands is this rank's fused weight. Replicated parts must stay
            // whole. A mixed group has no single fused layout, so leave its
            // parts to their own bind paths rather than inventing one.
            let mut all_sharded = true;
            let mut all_replicated = true;
            for raw in &candidate.parts {
                let axis = self.shard_axis(&raw.name)?;
                let row_sharded = axis == Some(0);
                all_sharded = all_sharded && row_sharded;
                all_replicated = all_replicated && axis.is_none();
            }
            if !all_sharded && !all_replicated {
                continue;
            }

            let mut parts = Vec::with_capacity(candidate.parts.len());
            let mut local_rows = Vec::with_capacity(candidate.parts.len());
            let mut rows = 0;
            for raw in &candidate.parts {
                // Shard each part, then join: q/k/v have different row
                // counts, so a row-shard of the *concatenation* would cut
                // across the q/k boundary and hand a rank a mix of two
                // projections.
                if all_sharded {
                    self.check_head_granularity(&raw.name, &raw.shape, Some(0))?;
                    parts.push(self.split(Expr::src(&raw.name), 0));
                    local_rows.push(self.local_extent(raw.shape[0]));
                } else {
                    parts.push(Expr::src(&raw.name));
                    local_rows.push(raw.shape[0]);
                }
                rows += *local_rows.last().expect("just pushed");
            }
            self.define(
                candidate.output_name.clone(),
                Expr::concat(0, parts),
                Encoding::Raw(DType::BF16),
                Some(vec![rows, candidate.cols]),
            );

            // Re-publish each projection as a view into the bank. A bind path
            // that reads q/k/v individually then finds them under their own
            // names at every `tp_size`, and the offset of a rank's band is
            // stated once, here, instead of being recomputed in pointer
            // arithmetic per family.
            let mut at = 0;
            for (raw, rows) in candidate.parts.iter().zip(&local_rows) {
                self.define(
                    self.output_name(&raw.name),
                    Expr::out(&candidate.output_name).slice(0, at, *rows),
                    Encoding::Raw(DType::BF16),
                    Some(vec![*rows, candidate.cols]),
                );
                self.consumed.insert(raw.id);
                at += rows;
            }
        }
        Ok(())
    }

    /// Publish a second, quantized view of a source tensor under another
    /// name, without consuming the source. Returns false when the checkpoint
    /// has no such tensor.
    pub fn quantized_view(
        &mut self,
        source: &str,
        output: String,
        scheme: QuantScheme,
    ) -> Result<bool, Error> {
        let Some(raw) = self.find(source) else {
            return Ok(false);
        };
        self.push_runtime_quant(raw, output, scheme)?;
        Ok(true)
    }

    // -- the generic tail ----------------------------------------------------

    /// Declare every source tensor no pass has claimed, under its own name.
    ///
    /// Always last. A family that publishes nothing else still gets a
    /// complete contract from this.
    ///
    /// **The dense tail, and why its order is load-bearing.** Most families
    /// end with the same three passes:
    ///
    /// ```text
    /// b.fused_moe_gate_up_tp_slices(false)?;
    /// b.dense_fused_projection_joins()?;
    /// b.publish_remaining()
    /// ```
    ///
    /// A pass that fuses tensors has to *claim* them before this one declares
    /// them one by one, so `publish_remaining` cannot move earlier. The MoE
    /// slice runs first because a family can have both a fused expert weight
    /// and fused dense projections (Gemma-4 does).
    ///
    /// This used to be bundled as `author_dense_contract` and called by six
    /// families. The bundle is gone: three lines is not worth a name, and
    /// hiding them meant six families' contracts could not be read where they
    /// live. nemotron_h, gpt-oss and Kimi-K3 already wrote the sequence out —
    /// now everyone does.
    pub fn publish_remaining(&mut self) -> Result<(), Error> {
        let scheme = self.runtime_quant_scheme()?;
        for raw in self.tensors.clone() {
            if self.consumed.contains(&raw.id) || !self.source_name_allowed(&raw.name) {
                continue;
            }
            match scheme {
                Some(scheme) if runtime_quantizable_name(&raw.name, scheme) => {
                    self.push_runtime_quant(raw, raw.name.clone(), scheme)?;
                }
                _ => {
                    let axis = self.shard_axis(&raw.name)?;
                    let defined = self.push_direct(raw, self.output_name(&raw.name), axis)?;
                    self.state_shipped_block_scales(raw, defined, scheme);
                }
            }
        }
        Ok(())
    }

    /// State the pairing for a scale the checkpoint shipped beside an FP8
    /// weight. The block size is read off the two shapes instead of assumed,
    /// and a companion that is not really FP8 is left alone rather than
    /// reinterpreted.
    fn state_shipped_block_scales(
        &mut self,
        raw: &RawTensor,
        scales: Option<usize>,
        scheme: Option<QuantScheme>,
    ) {
        let Some(scales) = scales else {
            return;
        };
        let Some(weight) = probe::companion_weight_name(&raw.name) else {
            return;
        };
        let Some(companion) = self.find(&weight) else {
            return;
        };
        if !is_raw(&companion.encoding, DType::F8E4M3) {
            return;
        }
        // A weight an earlier pass claimed — a fused QKV join, say — is not
        // published under this name, and naming it would be a contract the
        // loader rejects outright.
        if self.consumed.contains(&companion.id) || !self.source_name_allowed(&companion.name) {
            return;
        }
        // A weight the loader re-quantizes gets the scales the loader itself
        // writes, and states that pairing when it creates them.
        if scheme.is_some_and(|scheme| runtime_quantizable_name(&companion.name, scheme)) {
            return;
        }
        let (Some(weight_cols), Some(scale_cols)) =
            (companion.shape.last(), raw.shape.last().filter(|&&c| c > 0))
        else {
            return;
        };
        let block = weight_cols / scale_cols;
        if block <= 0 {
            return;
        }
        self.contract.tensors[scales].scales = Some(Scales {
            of: self.output_name(&companion.name),
            granularity: QuantGranularity::PerGroup,
            group_size: block as u32,
            channel_axis: 0,
            form: ScaleForm::F32Factors,
        });
    }

    /// Check what only the whole contract can answer, after the family is
    /// done, and hand it over.
    pub fn finish(self) -> Result<ModelContract, Error> {
        if self.policy.component == Component::Encode && !self.encode_scope_allowed {
            return fail(format!(
                "encode-scoped loading is not supported for '{}'",
                self.id
            ));
        }
        if self.contract.tensors.is_empty() && self.contract.groups.is_empty() {
            return fail(format!(
                "no contract was authored for '{}'; the driver must \
                 declare what it binds",
                self.id
            ));
        }
        Ok(self.contract)
    }

    // -- the source-prefix rule ----------------------------------------------

    fn source_name_allowed(&self, raw_name: &str) -> bool {
        self.source_prefix.is_empty() || raw_name.starts_with(&self.source_prefix)
    }

    // -- runtime quantization ------------------------------------------------

    /// The scheme this author's runtime-quant request resolves to, or none.
    ///
    /// Resolved on each call rather than once at construction, and it has to
    /// stay that way: the answer reads the allow knobs, which the *family*
    /// sets after construction.
    fn runtime_quant_scheme(&self) -> Result<Option<QuantScheme>, Error> {
        let scheme = match self.policy.runtime_quant {
            RuntimeQuant::None => return Ok(None),
            RuntimeQuant::Fp8 => QuantScheme::Fp8E4M3,
            RuntimeQuant::Int8 => QuantScheme::Int8Symmetric,
            RuntimeQuant::Mxfp4 => QuantScheme::Mxfp4E2M1E8M0,
            RuntimeQuant::Int4 => QuantScheme::MlxAffineU4,
        };
        // For FP4 a pre-quantized checkpoint is accepted (GLM-5.1 ships FP8
        // experts). For FP8/INT8 only BF16 weights are re-quantized, never an
        // already-quantized checkpoint.
        if !self.encoding.is_none() && scheme != QuantScheme::Mxfp4E2M1E8M0 {
            return Ok(None);
        }
        let allowed = if scheme == QuantScheme::Mxfp4E2M1E8M0 {
            self.allow_mxfp4_rq
        } else {
            self.allow_bf16_rq
        };
        if !allowed {
            return fail(format!(
                "runtime_quant={:?} is not supported for '{}'",
                self.policy.runtime_quant, self.id
            ));
        }
        Ok(Some(scheme))
    }

    fn push_runtime_quant(
        &mut self,
        raw: &RawTensor,
        output: String,
        scheme: QuantScheme,
    ) -> Result<(), Error> {
        if raw.shape.len() != 2 {
            return fail(format!("runtime_quant source '{}' must be 2-D", raw.name));
        }
        // Allowed sources are BF16/FP16/FP32 raw, handled by the executor's
        // bf16 cast path, and FP8 (E4M3) raw: those weights ship quantized,
        // the executor dequants them to bf16 with a sibling `_scale_inv` at
        // materialize time, then re-encodes to the target scheme.
        if !(is_raw(&raw.encoding, DType::BF16)
            || is_raw(&raw.encoding, DType::F16)
            || is_raw(&raw.encoding, DType::F32)
            || is_raw(&raw.encoding, DType::F8E4M3))
        {
            return fail(format!(
                "runtime_quant source '{}' must be BF16/FP16/FP32/F8E4M3",
                raw.name
            ));
        }
        let spec = match scheme {
            QuantScheme::Fp8E4M3 => QuantSpec {
                scheme,
                logical_dtype: DType::F8E4M3,
                bits_per_element: 8,
                group_size: 1,
                channel_axis: Some(Axis(0)),
            },
            QuantScheme::Int8Symmetric => QuantSpec {
                scheme,
                logical_dtype: DType::I8,
                bits_per_element: 8,
                group_size: 1,
                channel_axis: Some(Axis(0)),
            },
            QuantScheme::Mxfp4E2M1E8M0 => {
                // The K dimension must be a 32-multiple because an MXFP4
                // block scale covers 32 contiguous elements.
                if raw.shape[1] % 32 != 0 {
                    return fail(format!(
                        "runtime_quant Mxfp4 source '{}' cols {} must be a multiple of 32",
                        raw.name, raw.shape[1]
                    ));
                }
                QuantSpec {
                    scheme,
                    logical_dtype: DType::BF16,
                    bits_per_element: 4,
                    group_size: 32,
                    channel_axis: Some(Axis(1)),
                }
            }
            other => {
                return fail(format!("unsupported runtime_quant scheme {other:?}"));
            }
        };
        let axis = self.shard_axis(&raw.name)?;
        let (expr, shape) = self.shard(Expr::src(&raw.name), raw.shape.clone(), axis);
        let encoding = Encoding::Quant(spec);
        self.define(output, expr.cast(encoding.clone()), encoding, Some(shape));
        Ok(())
    }
}

/// The default shard-axis rule: the HF convention, which never errors.
///
/// A family rule may error — DeepSeek-V4 refuses to let an unrecognized FFN
/// weight silently replicate — so the pointer type is fallible and the
/// default wraps the infallible convention.
pub fn default_shard_axis(name: &str) -> Result<Option<u8>, Error> {
    Ok(probe::hf_shard_axis(name))
}

/// The MXFP4 encoding every GPT-OSS expert tensor is declared with.
pub fn mxfp4_encoding(channel_axis: u8) -> Encoding {
    Encoding::Quant(QuantSpec {
        scheme: QuantScheme::Mxfp4E2M1E8M0,
        logical_dtype: DType::BF16,
        bits_per_element: 4,
        // One MXFP4 block scale covers 32 contiguous elements along K.
        group_size: 32,
        channel_axis: Some(Axis(channel_axis)),
    })
}

/// The W4A16 pairing Kimi ships: 4-bit codes biased by 8, eight to a word.
pub fn int4b8_encoding(channel_axis: u8) -> Encoding {
    Encoding::Quant(QuantSpec {
        scheme: QuantScheme::Int4B8,
        logical_dtype: DType::BF16,
        bits_per_element: 4,
        group_size: 32,
        channel_axis: Some(Axis(channel_axis)),
    })
}

/// Round `value` up to a multiple of `alignment`.
pub fn align_up(value: i64, alignment: i64) -> Result<i64, Error> {
    if value < 0 || alignment <= 0 {
        return fail("contract: align_up needs a non-negative value and a positive alignment");
    }
    Ok((value + alignment - 1) / alignment * alignment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::LoadShape;
    use crate::encoding::Encoding as StoredEncoding;

    /// A checkpoint with no tensors at all.
    ///
    /// Enough for everything below: these test what the builder was
    /// TOLD, and the tensor table is what the family authors read, not
    /// what `new` decides from.
    fn empty_checkpoint() -> CheckpointMetadata {
        CheckpointMetadata {
            files: Vec::new(),
            tensors: Vec::new(),
        }
    }

    fn target() -> StorageTarget {
        StorageTarget {
            preferred_alignment: 256,
            ..StorageTarget::default()
        }
    }

    #[test]
    fn a_builder_carries_the_row_that_asked_for_it() {
        let meta = empty_checkpoint();
        let target = target();
        let policy = Policy::default();
        let enc = StoredEncoding::dense();
        let shape = LoadShape::dense(28, 128, true);
        let b = Builder::new(&meta, "qwen3-0.6b", shape, &enc, &target, &policy);

        // The id is a MODEL, not a dispatch key: the two `fail()` sites
        // in this file name it, and `model_type='qwen3'` named a string
        // a dozen checkpoints shared.
        assert_eq!(b.id(), "qwen3-0.6b");
        assert_eq!(b.shape(), shape);
        assert_eq!(b.shape().layers, 28);
        assert!(b.shape().tied_embeddings);
        assert!(b.encoding().is_none(), "an unquantized checkpoint");
        assert_eq!(b.target().preferred_alignment, 256);
    }

    /// The `HF_ROWS` / `MLX_ROWS` split, as a `match` a row makes.
    #[test]
    fn the_naming_reaches_the_row_that_has_to_branch_on_it() {
        let meta = empty_checkpoint();
        let target = target();
        let enc = StoredEncoding::dense();
        let shape = LoadShape::dense(28, 128, true);

        for naming in [Naming::Hf, Naming::Mlx] {
            let policy = Policy {
                naming,
                ..Policy::default()
            };
            let b = Builder::new(&meta, "qwen3-0.6b", shape, &enc, &target, &policy);
            assert_eq!(b.naming(), naming);
        }
    }

    /// The contract starts aligned to the target, and never to zero —
    /// an alignment of zero is a division a later stage does not guard.
    #[test]
    fn the_contract_opens_at_the_targets_alignment() {
        let meta = empty_checkpoint();
        let policy = Policy::default();
        let enc = StoredEncoding::dense();
        let shape = LoadShape::dense(1, 64, false);

        let t = StorageTarget {
            preferred_alignment: 512,
            ..StorageTarget::default()
        };
        let b = Builder::new(&meta, "x", shape, &enc, &t, &policy);
        assert_eq!(b.contract.alignment, 512);

        let t = StorageTarget {
            preferred_alignment: 0,
            ..StorageTarget::default()
        };
        let b = Builder::new(&meta, "x", shape, &enc, &t, &policy);
        assert_eq!(b.contract.alignment, 1, "zero would divide by zero later");
    }

    /// `Auto` is the ABSENCE of a policy, so the device answers it; the
    /// three stated requests are carried through unchanged.
    #[test]
    fn auto_is_answered_by_the_device_and_the_rest_are_obeyed() {
        assert_eq!(
            resolve_mxfp4_moe(Mxfp4MoeRequest::Auto, true),
            Mxfp4MoePolicy::NativeGemm
        );
        assert_eq!(
            resolve_mxfp4_moe(Mxfp4MoeRequest::Auto, false),
            Mxfp4MoePolicy::RoutedDecode
        );
        for native in [false, true] {
            assert_eq!(
                resolve_mxfp4_moe(Mxfp4MoeRequest::NativeGemm, native),
                Mxfp4MoePolicy::NativeGemm,
                "a stated request is not the device's to reconsider"
            );
            assert_eq!(
                resolve_mxfp4_moe(Mxfp4MoeRequest::RoutedDecode, native),
                Mxfp4MoePolicy::RoutedDecode
            );
            assert_eq!(
                resolve_mxfp4_moe(Mxfp4MoeRequest::EagerBf16, native),
                Mxfp4MoePolicy::EagerBf16
            );
        }
    }

    /// And the builder resolves it once, at construction, so a family
    /// that asks twice gets the same answer.
    #[test]
    fn the_moe_resolution_happens_once_at_construction() {
        let meta = empty_checkpoint();
        let enc = StoredEncoding::dense();
        let shape = LoadShape::dense(1, 64, false);
        let policy = Policy {
            moe_request: Mxfp4MoeRequest::Auto,
            ..Policy::default()
        };

        let t = StorageTarget {
            native_mxfp4_moe: true,
            ..StorageTarget::default()
        };
        let b = Builder::new(&meta, "gpt-oss-20b", shape, &enc, &t, &policy);
        assert_eq!(b.mxfp4_moe(), Mxfp4MoePolicy::NativeGemm);
        assert_eq!(b.mxfp4_moe(), b.mxfp4_moe());

        let t = StorageTarget {
            native_mxfp4_moe: false,
            ..StorageTarget::default()
        };
        let b = Builder::new(&meta, "gpt-oss-20b", shape, &enc, &t, &policy);
        assert_eq!(b.mxfp4_moe(), Mxfp4MoePolicy::RoutedDecode);
    }

    /// The encoding is the FILE's answer and the shape is the ROW's,
    /// and they are separate because one model has four downloads.
    #[test]
    fn one_row_serves_every_encoding_of_itself() {
        let meta = empty_checkpoint();
        let target = target();
        let policy = Policy::default();
        let shape = LoadShape::dense(36, 128, false);

        let bf16 = StoredEncoding::dense();
        let awq = StoredEncoding {
            method: "awq".into(),
            bits: 4,
            group_size: 128,
        };
        let mxfp4 = StoredEncoding {
            method: "mxfp4".into(),
            bits: 4,
            group_size: 32,
        };

        for enc in [&bf16, &awq, &mxfp4] {
            let b = Builder::new(&meta, "qwen3-8b", shape, enc, &target, &policy);
            assert_eq!(
                b.shape(),
                shape,
                "the shape does not move with the encoding"
            );
            assert_eq!(b.id(), "qwen3-8b", "nor does the identity");
        }
        assert!(bf16.is_none());
        assert!(!awq.is_none());
        assert!(mxfp4.is_mxfp4());
        assert!(!awq.is_mxfp4());
    }

    #[test]
    fn align_up_rounds_up_and_refuses_a_nonsense_alignment() {
        assert_eq!(align_up(0, 256).expect("zero is aligned"), 0);
        assert_eq!(align_up(1, 256).expect("rounds up"), 256);
        assert_eq!(align_up(256, 256).expect("already aligned"), 256);
        assert_eq!(align_up(257, 256).expect("rounds up"), 512);
        assert_eq!(align_up(5, 1).expect("everything is 1-aligned"), 5);
        assert!(
            align_up(-1, 256).is_err(),
            "a negative offset is not a rounding question"
        );
        assert!(align_up(256, 0).is_err(), "and neither is a zero alignment");
        assert!(align_up(256, -8).is_err());
    }

    // ── Runtime quantization ─────────────────────────────────────────────
    //
    // The whole of `runtime_quant_scheme` / `push_runtime_quant` was unrun.
    // `boot.rs` sets `RuntimeQuant::None` and nothing else in the workspace
    // sets anything else, so ~90 lines of the builder -- every refusal, every
    // QuantSpec, the shape rule -- existed only as an intention.

    /// A checkpoint from `(name, shape, encoding)` triples.
    fn checkpoint(rows: &[(&str, Vec<i64>, Encoding)]) -> CheckpointMetadata {
        CheckpointMetadata {
            files: Vec::new(),
            tensors: rows
                .iter()
                .enumerate()
                .map(|(i, (name, shape, encoding))| RawTensor {
                    id: TensorId(i as u32),
                    name: (*name).to_string(),
                    file_id: model_loader::types::FileId(0),
                    file_offset: 0,
                    span_bytes: shape.iter().product::<i64>().max(0) as u64 * 2,
                    shape: shape.clone(),
                    encoding: encoding.clone(),
                })
                .collect(),
        }
    }

    fn bf16(shape: Vec<i64>) -> (Vec<i64>, Encoding) {
        (shape, Encoding::Raw(DType::BF16))
    }

    /// Author a contract over `rows` under `policy`, running only the
    /// generic tail -- which is the pass runtime quant lives in.
    fn publish(
        rows: &[(&str, Vec<i64>, Encoding)],
        enc: &StoredEncoding,
        policy: &Policy,
        allow: impl FnOnce(&mut Builder<'_>),
    ) -> Result<ModelContract, Error> {
        let meta = checkpoint(rows);
        let target = target();
        let shape = LoadShape::dense(1, 128, false);
        let mut b = Builder::new(&meta, "test-row", shape, enc, &target, policy);
        allow(&mut b);
        b.publish_remaining()?;
        b.finish()
    }

    fn rq(kind: RuntimeQuant) -> Policy {
        Policy {
            runtime_quant: kind,
            ..Policy::default()
        }
    }

    /// A family that has not wired runtime quant refuses the request by name.
    ///
    /// The refusal is the point. Without it the knob would be silently
    /// ignored and the caller would get a bf16 contract while believing it
    /// asked for FP8 -- a difference visible only in memory use.
    #[test]
    fn a_family_that_has_not_wired_requant_refuses_it() {
        let rows = [(
            "model.layers.0.mlp.gate_proj.weight",
            bf16(vec![64, 128]).0,
            Encoding::Raw(DType::BF16),
        )];
        for kind in [RuntimeQuant::Fp8, RuntimeQuant::Int8, RuntimeQuant::Mxfp4] {
            let err = publish(&rows, &StoredEncoding::dense(), &rq(kind), |_| {})
                .expect_err("no allow_* knob was set");
            let Error::Contract(msg) = err else {
                panic!("expected a contract error for {kind:?}");
            };
            assert!(msg.contains("not supported for 'test-row'"), "{msg}");
            assert!(
                msg.contains(&format!("{kind:?}")),
                "the message names the request: {msg}"
            );
        }
    }

    /// `RuntimeQuant::None` is not a refusal, it is the ordinary path.
    #[test]
    fn no_request_publishes_the_weights_as_they_are() {
        let rows = [(
            "model.layers.0.mlp.gate_proj.weight",
            vec![64, 128],
            Encoding::Raw(DType::BF16),
        )];
        let c = publish(
            &rows,
            &StoredEncoding::dense(),
            &rq(RuntimeQuant::None),
            |b| {
                b.allow_bf16_runtime_quant();
            },
        )
        .unwrap();
        assert_eq!(c.tensors.len(), 1);
        assert!(
            matches!(c.tensors[0].encoding, Encoding::Raw(DType::BF16)),
            "an unrequested requant must not happen: {:?}",
            c.tensors[0].encoding
        );
    }

    /// FP8 and INT8 each get their own spec, and only the listed projections
    /// are touched.
    ///
    /// Both are per-output-channel with no grouping, so the two differ by
    /// exactly one field -- the logical dtype the consumer sees. A spec that
    /// copied the wrong one still produces a well-formed contract.
    #[test]
    fn bf16_requant_states_a_per_channel_spec_for_the_projections_only() {
        let rows = [
            (
                "model.layers.0.mlp.gate_proj.weight",
                vec![64, 128],
                Encoding::Raw(DType::BF16),
            ),
            (
                "model.layers.0.input_layernorm.weight",
                vec![128],
                Encoding::Raw(DType::BF16),
            ),
        ];
        for (kind, want_dtype) in [
            (RuntimeQuant::Fp8, DType::F8E4M3),
            (RuntimeQuant::Int8, DType::I8),
        ] {
            let c = publish(&rows, &StoredEncoding::dense(), &rq(kind), |b| {
                b.allow_bf16_runtime_quant();
            })
            .unwrap();
            let proj = c
                .tensors
                .iter()
                .find(|t| t.name.ends_with("gate_proj.weight"))
                .expect("the projection is published");
            let Encoding::Quant(spec) = &proj.encoding else {
                panic!("{kind:?}: the projection was not requantized");
            };
            assert_eq!(spec.logical_dtype, want_dtype, "{kind:?}");
            assert_eq!(spec.bits_per_element, 8, "{kind:?}");
            assert_eq!(spec.group_size, 1, "{kind:?} is per-channel, not grouped");
            assert_eq!(
                spec.channel_axis,
                Some(Axis(0)),
                "{kind:?} splits by output row"
            );

            let norm = c
                .tensors
                .iter()
                .find(|t| t.name.ends_with("input_layernorm.weight"))
                .expect("the norm is published");
            assert!(
                matches!(norm.encoding, Encoding::Raw(DType::BF16)),
                "{kind:?} reached a norm it has no business quantizing: {:?}",
                norm.encoding
            );
        }
    }

    /// MXFP4 reaches routed experts and nothing else.
    ///
    /// The comment on `runtime_quantizable_name` gives the reason -- this
    /// hardware has no FP4 GEMM for attention -- so a dense projection under
    /// an Mxfp4 request must come out untouched, not merely un-crashed.
    #[test]
    fn mxfp4_requant_reaches_experts_and_leaves_attention_alone() {
        let rows = [
            (
                "model.layers.0.mlp.experts.0.up_proj.weight",
                vec![64, 128],
                Encoding::Raw(DType::BF16),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                vec![64, 128],
                Encoding::Raw(DType::BF16),
            ),
        ];
        let c = publish(
            &rows,
            &StoredEncoding::dense(),
            &rq(RuntimeQuant::Mxfp4),
            |b| {
                b.allow_mxfp4_runtime_quant();
            },
        )
        .unwrap();
        let expert = c
            .tensors
            .iter()
            .find(|t| t.name.contains("experts"))
            .unwrap();
        let Encoding::Quant(spec) = &expert.encoding else {
            panic!("the expert was not requantized: {:?}", expert.encoding);
        };
        assert_eq!(spec.scheme, QuantScheme::Mxfp4E2M1E8M0);
        assert_eq!(spec.bits_per_element, 4);
        assert_eq!(
            spec.group_size, 32,
            "an MXFP4 block scale covers 32 elements"
        );
        assert_eq!(
            spec.channel_axis,
            Some(Axis(1)),
            "grouped along K, not along the output row"
        );
        assert_eq!(
            spec.logical_dtype,
            DType::BF16,
            "the consumer still sees bf16"
        );

        let attn = c
            .tensors
            .iter()
            .find(|t| t.name.contains("q_proj"))
            .unwrap();
        assert!(
            matches!(attn.encoding, Encoding::Raw(DType::BF16)),
            "FP4 has no GEMM for attention on this hardware: {:?}",
            attn.encoding
        );
    }

    /// The two allow knobs are not interchangeable.
    ///
    /// A family that wired BF16->FP8 has not thereby wired FP4 experts, and
    /// vice versa. Sharing one flag would let either request through on the
    /// strength of the other's implementation.
    #[test]
    fn the_two_requant_knobs_gate_different_requests() {
        let rows = [(
            "model.layers.0.mlp.experts.0.up_proj.weight",
            vec![64, 128],
            Encoding::Raw(DType::BF16),
        )];
        assert!(
            publish(
                &rows,
                &StoredEncoding::dense(),
                &rq(RuntimeQuant::Mxfp4),
                |b| {
                    b.allow_bf16_runtime_quant();
                }
            )
            .is_err(),
            "the bf16 knob must not admit an FP4 request"
        );
        assert!(
            publish(
                &rows,
                &StoredEncoding::dense(),
                &rq(RuntimeQuant::Fp8),
                |b| {
                    b.allow_mxfp4_runtime_quant();
                }
            )
            .is_err(),
            "the FP4 knob must not admit an FP8 request"
        );
    }

    /// An already-quantized checkpoint is re-quantized to FP4 and to nothing
    /// else.
    ///
    /// GLM-5.1 ships FP8 experts and is still asked for FP4. The same
    /// checkpoint asked for FP8 must fall through to the ordinary path
    /// rather than re-encoding what is already encoded -- and `Ok(None)`,
    /// not an error, because there is nothing wrong with the request.
    #[test]
    fn a_prequantized_checkpoint_admits_only_the_fp4_request() {
        let rows = [(
            "model.layers.0.mlp.experts.0.up_proj.weight",
            vec![64, 128],
            Encoding::Raw(DType::F8E4M3),
        )];
        let quantized = StoredEncoding {
            method: "fp8".into(),
            bits: 8,
            group_size: 0,
        };

        let c = publish(&rows, &quantized, &rq(RuntimeQuant::Fp8), |b| {
            b.allow_bf16_runtime_quant();
        })
        .expect("an FP8 request over a quantized checkpoint is declined, not refused");
        assert!(
            !matches!(&c.tensors[0].encoding, Encoding::Quant(s) if s.scheme == QuantScheme::Fp8E4M3),
            "it must not re-encode an already-encoded checkpoint"
        );

        let c = publish(&rows, &quantized, &rq(RuntimeQuant::Mxfp4), |b| {
            b.allow_mxfp4_runtime_quant();
        })
        .unwrap();
        let Encoding::Quant(spec) = &c.tensors[0].encoding else {
            panic!("FP4 over an FP8 checkpoint should still requantize");
        };
        assert_eq!(spec.scheme, QuantScheme::Mxfp4E2M1E8M0);
    }

    /// Sources that cannot be requantized are named, not skipped.
    #[test]
    fn a_source_the_requantizer_cannot_read_is_refused_by_name() {
        // Not 2-D.
        let rows = [(
            "model.layers.0.mlp.gate_proj.weight",
            vec![64],
            Encoding::Raw(DType::BF16),
        )];
        let Err(Error::Contract(msg)) = publish(
            &rows,
            &StoredEncoding::dense(),
            &rq(RuntimeQuant::Fp8),
            |b| {
                b.allow_bf16_runtime_quant();
            },
        ) else {
            panic!("a 1-D source should be refused");
        };
        assert!(msg.contains("must be 2-D"), "{msg}");
        assert!(
            msg.contains("gate_proj"),
            "the message names the tensor: {msg}"
        );

        // A dtype the executor has no cast path for.
        let rows = [(
            "model.layers.0.mlp.gate_proj.weight",
            vec![64, 128],
            Encoding::Raw(DType::I8),
        )];
        let Err(Error::Contract(msg)) = publish(
            &rows,
            &StoredEncoding::dense(),
            &rq(RuntimeQuant::Fp8),
            |b| {
                b.allow_bf16_runtime_quant();
            },
        ) else {
            panic!("an i8 source should be refused");
        };
        assert!(msg.contains("BF16/FP16/FP32/F8E4M3"), "{msg}");
    }

    /// Every dtype the doc lists as a legal requant source really is one.
    #[test]
    fn all_four_documented_source_dtypes_are_accepted() {
        for dtype in [DType::BF16, DType::F16, DType::F32, DType::F8E4M3] {
            let rows = [(
                "model.layers.0.mlp.gate_proj.weight",
                vec![64, 128],
                Encoding::Raw(dtype),
            )];
            let c = publish(
                &rows,
                &StoredEncoding::dense(),
                &rq(RuntimeQuant::Fp8),
                |b| {
                    b.allow_bf16_runtime_quant();
                },
            )
            .unwrap_or_else(|e| panic!("{dtype:?} is documented as a legal source: {e:?}"));
            assert!(
                matches!(c.tensors[0].encoding, Encoding::Quant(_)),
                "{dtype:?} was accepted but not requantized"
            );
        }
    }

    /// An MXFP4 block scale spans 32 columns, so a K that is not a multiple
    /// of 32 is refused rather than rounded.
    #[test]
    fn mxfp4_refuses_a_k_that_does_not_divide_into_blocks() {
        let rows = [(
            "model.layers.0.mlp.experts.0.up_proj.weight",
            vec![64, 130],
            Encoding::Raw(DType::BF16),
        )];
        let Err(Error::Contract(msg)) = publish(
            &rows,
            &StoredEncoding::dense(),
            &rq(RuntimeQuant::Mxfp4),
            |b| {
                b.allow_mxfp4_runtime_quant();
            },
        ) else {
            panic!("130 columns is not a whole number of 32-element blocks");
        };
        assert!(msg.contains("multiple of 32"), "{msg}");
        assert!(
            msg.contains("130"),
            "the message gives the offending width: {msg}"
        );
        // 128 is, and passes.
        let rows = [(
            "model.layers.0.mlp.experts.0.up_proj.weight",
            vec![64, 128],
            Encoding::Raw(DType::BF16),
        )];
        assert!(
            publish(
                &rows,
                &StoredEncoding::dense(),
                &rq(RuntimeQuant::Mxfp4),
                |b| {
                    b.allow_mxfp4_runtime_quant();
                }
            )
            .is_ok()
        );
    }

    /// `RuntimeQuant::Int4` resolves a scheme the CUDA push path cannot
    /// emit.
    ///
    /// `runtime_quant_scheme` maps it to `MlxAffineU4`, which then falls to
    /// `push_runtime_quant`'s catch-all. It is reachable: a family that
    /// calls `allow_bf16_runtime_quant` -- the knob whose name says FP8/INT8
    /// -- admits the Int4 request as far as the push. The failure is at
    /// least loud and names the scheme; this test exists so that if the MLX
    /// path is ever wired through the shared builder, the change is
    /// deliberate rather than a silently altered error string.
    #[test]
    fn an_int4_request_reaches_the_push_and_is_refused_there() {
        let rows = [(
            "model.layers.0.mlp.gate_proj.weight",
            vec![64, 128],
            Encoding::Raw(DType::BF16),
        )];
        let Err(Error::Contract(msg)) = publish(
            &rows,
            &StoredEncoding::dense(),
            &rq(RuntimeQuant::Int4),
            |b| {
                b.allow_bf16_runtime_quant();
            },
        ) else {
            panic!("Int4 has no CUDA push path");
        };
        assert!(msg.contains("unsupported runtime_quant scheme"), "{msg}");
        assert!(msg.contains("MlxAffineU4"), "{msg}");
    }

    /// A contract nobody authored is refused, rather than handed over empty.
    #[test]
    fn an_empty_contract_is_refused() {
        let Err(Error::Contract(msg)) =
            publish(&[], &StoredEncoding::dense(), &Policy::default(), |_| {})
        else {
            panic!("an empty contract binds nothing");
        };
        assert!(msg.contains("no contract was authored"), "{msg}");
        assert!(msg.contains("test-row"), "the message names the row: {msg}");
    }

    // ── Shipped block scales ─────────────────────────────────────────────
    //
    // `state_shipped_block_scales` runs on every tensor the generic tail
    // publishes and does nothing for almost all of them. Its whole body was
    // unrun: no test ever published a `_scale_inv` beside an FP8 weight.

    /// An FP8 weight and its shipped scale are paired, with the block size
    /// read off the two shapes.
    ///
    /// DeepSeek ships `weight` at [N, K] FP8 and `weight_scale_inv` at
    /// [N, K/128] F32. Nothing declares 128 anywhere; it is the quotient of
    /// the two trailing dims. A hard-coded block would be right for
    /// DeepSeek and wrong for the next checkpoint that picks another one.
    #[test]
    fn a_shipped_scale_is_paired_with_its_fp8_weight() {
        let rows = [
            (
                "model.layers.0.mlp.down_proj.weight",
                vec![64, 256],
                Encoding::Raw(DType::F8E4M3),
            ),
            (
                "model.layers.0.mlp.down_proj.weight_scale_inv",
                vec![64, 2],
                Encoding::Raw(DType::F32),
            ),
        ];
        let c = publish(&rows, &StoredEncoding::dense(), &Policy::default(), |_| {}).unwrap();
        let scale = c
            .tensors
            .iter()
            .find(|t| t.name.ends_with("weight_scale_inv"))
            .expect("the scale is published in its own right");
        let paired = scale
            .scales
            .as_ref()
            .expect("the scale states what it scales");
        assert_eq!(paired.of, "model.layers.0.mlp.down_proj.weight");
        assert_eq!(paired.group_size, 128, "256 columns over 2 scales");
        assert_eq!(paired.granularity, QuantGranularity::PerGroup);
        assert_eq!(paired.form, ScaleForm::F32Factors);
    }

    /// The block size is a quotient, not a constant.
    #[test]
    fn the_block_size_comes_from_the_two_shapes() {
        for (cols, n_scales, want) in [(256i64, 2i64, 128u32), (256, 4, 64), (96, 3, 32)] {
            let rows = [
                (
                    "model.layers.0.mlp.down_proj.weight",
                    vec![64, cols],
                    Encoding::Raw(DType::F8E4M3),
                ),
                (
                    "model.layers.0.mlp.down_proj.weight_scale_inv",
                    vec![64, n_scales],
                    Encoding::Raw(DType::F32),
                ),
            ];
            let c = publish(&rows, &StoredEncoding::dense(), &Policy::default(), |_| {}).unwrap();
            let scale = c
                .tensors
                .iter()
                .find(|t| t.name.ends_with("_scale_inv"))
                .unwrap();
            assert_eq!(
                scale.scales.as_ref().unwrap().group_size,
                want,
                "{cols} cols over {n_scales} scales"
            );
        }
    }

    /// A scale beside a weight that is not FP8 is left unpaired.
    ///
    /// "A companion that is not really FP8 is left alone rather than
    /// reinterpreted": a bf16 weight with something named `_scale_inv` next
    /// to it is not a block-scaled tensor, and claiming it is would have the
    /// loader divide by a factor nobody applied.
    #[test]
    fn a_scale_beside_a_bf16_weight_is_not_claimed() {
        let rows = [
            (
                "model.layers.0.mlp.down_proj.weight",
                vec![64, 256],
                Encoding::Raw(DType::BF16),
            ),
            (
                "model.layers.0.mlp.down_proj.weight_scale_inv",
                vec![64, 2],
                Encoding::Raw(DType::F32),
            ),
        ];
        let c = publish(&rows, &StoredEncoding::dense(), &Policy::default(), |_| {}).unwrap();
        let scale = c
            .tensors
            .iter()
            .find(|t| t.name.ends_with("_scale_inv"))
            .unwrap();
        assert!(scale.scales.is_none(), "a bf16 weight is not block-scaled");
    }

    /// A scale whose weight is not in the checkpoint is left unpaired
    /// rather than naming a tensor the contract never declares.
    #[test]
    fn an_orphan_scale_names_nothing() {
        let rows = [(
            "model.layers.0.mlp.down_proj.weight_scale_inv",
            vec![64, 2],
            Encoding::Raw(DType::F32),
        )];
        let c = publish(&rows, &StoredEncoding::dense(), &Policy::default(), |_| {}).unwrap();
        assert!(c.tensors[0].scales.is_none());
    }

    /// An ordinary weight is not mistaken for a scale.
    #[test]
    fn a_tensor_that_is_not_a_scale_is_passed_over() {
        let rows = [(
            "model.layers.0.mlp.down_proj.weight",
            vec![64, 256],
            Encoding::Raw(DType::F8E4M3),
        )];
        let c = publish(&rows, &StoredEncoding::dense(), &Policy::default(), |_| {}).unwrap();
        assert!(
            c.tensors[0].scales.is_none(),
            "a weight does not scale itself"
        );
    }

    /// A zero-length scale is declined rather than dividing by zero.
    #[test]
    fn a_scale_with_no_columns_does_not_divide_by_zero() {
        let rows = [
            (
                "model.layers.0.mlp.down_proj.weight",
                vec![64, 256],
                Encoding::Raw(DType::F8E4M3),
            ),
            (
                "model.layers.0.mlp.down_proj.weight_scale_inv",
                vec![64, 0],
                Encoding::Raw(DType::F32),
            ),
        ];
        let c = publish(&rows, &StoredEncoding::dense(), &Policy::default(), |_| {}).unwrap();
        let scale = c
            .tensors
            .iter()
            .find(|t| t.name.ends_with("_scale_inv"))
            .unwrap();
        assert!(scale.scales.is_none());
    }

    /// A weight the loader is about to re-quantize gets the loader's own
    /// scales, not the checkpoint's.
    ///
    /// Both would otherwise be stated, and the shipped pairing would
    /// describe a block layout the re-quantized tensor no longer has.
    #[test]
    fn a_requantized_weight_keeps_the_loaders_scales_not_the_shipped_ones() {
        let rows = [
            (
                "model.layers.0.mlp.down_proj.weight",
                vec![64, 256],
                Encoding::Raw(DType::F8E4M3),
            ),
            (
                "model.layers.0.mlp.down_proj.weight_scale_inv",
                vec![64, 2],
                Encoding::Raw(DType::F32),
            ),
        ];
        // Unrequested: the shipped pairing stands.
        let c = publish(&rows, &StoredEncoding::dense(), &Policy::default(), |_| {}).unwrap();
        let scale = c
            .tensors
            .iter()
            .find(|t| t.name.ends_with("_scale_inv"))
            .unwrap();
        assert!(scale.scales.is_some(), "the baseline pairs them");

        // Requested: `down_proj.weight` is on the requantizable list, so the
        // shipped pairing is dropped.
        let c = publish(
            &rows,
            &StoredEncoding::dense(),
            &rq(RuntimeQuant::Fp8),
            |b| {
                b.allow_bf16_runtime_quant();
            },
        )
        .unwrap();
        let scale = c
            .tensors
            .iter()
            .find(|t| t.name.ends_with("_scale_inv"))
            .unwrap();
        assert!(
            scale.scales.is_none(),
            "the loader writes its own scales for a weight it re-encodes"
        );
    }

    /// Under a source prefix, the pairing names the BOUND name.
    ///
    /// A prefixed checkpoint publishes `...down_proj.weight`, having
    /// stripped `language_model.`, but the companion lookup works on raw
    /// names. If the pairing recorded the raw name it would point at a
    /// tensor the contract does not declare, and the loader rejects that
    /// outright. The default prefix is empty, so nothing else here can
    /// tell the two apart.
    #[test]
    fn under_a_prefix_the_pairing_names_the_published_tensor() {
        let rows = [
            (
                "language_model.layers.0.mlp.down_proj.weight",
                vec![64, 256],
                Encoding::Raw(DType::F8E4M3),
            ),
            (
                "language_model.layers.0.mlp.down_proj.weight_scale_inv",
                vec![64, 2],
                Encoding::Raw(DType::F32),
            ),
        ];
        let meta = checkpoint(&rows);
        let target = target();
        let policy = Policy::default();
        let enc = StoredEncoding::dense();
        let mut b = Builder::new(
            &meta,
            "test-row",
            LoadShape::dense(1, 128, false),
            &enc,
            &target,
            &policy,
        );
        b.source_prefix("language_model.");
        b.publish_remaining().unwrap();
        let c = b.finish().unwrap();

        let scale = c
            .tensors
            .iter()
            .find(|t| t.name.ends_with("_scale_inv"))
            .unwrap();
        assert_eq!(
            scale.name, "layers.0.mlp.down_proj.weight_scale_inv",
            "the prefix is stripped"
        );
        let paired = scale.scales.as_ref().expect("still paired under a prefix");
        assert_eq!(
            paired.of, "layers.0.mlp.down_proj.weight",
            "the pairing must name a tensor this contract declares"
        );
        assert!(
            c.tensors.iter().any(|t| t.name == paired.of),
            "the named weight is in the contract"
        );
    }

    /// A tensor outside the prefix is not published at all.
    #[test]
    fn a_tensor_outside_the_prefix_is_left_to_another_scope() {
        let rows = [
            (
                "language_model.layers.0.mlp.down_proj.weight",
                vec![64, 256],
                Encoding::Raw(DType::BF16),
            ),
            (
                "vision_tower.encoder.0.weight",
                vec![8, 8],
                Encoding::Raw(DType::BF16),
            ),
        ];
        let meta = checkpoint(&rows);
        let target = target();
        let policy = Policy::default();
        let enc = StoredEncoding::dense();
        let mut b = Builder::new(
            &meta,
            "test-row",
            LoadShape::dense(1, 128, false),
            &enc,
            &target,
            &policy,
        );
        b.source_prefix("language_model.");
        b.publish_remaining().unwrap();
        let c = b.finish().unwrap();
        assert_eq!(c.tensors.len(), 1, "only the prefixed tensor is bound");
        assert_eq!(c.tensors[0].name, "layers.0.mlp.down_proj.weight");
    }
}
