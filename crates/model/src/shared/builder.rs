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
    /// Keep a published declaration out of the driver's namespace.
    ///
    /// `index` is an `Option` for the same reason [`Self::set_scales`]'s is:
    /// [`Self::define`] publishes nothing under an encode scope, so a pass
    /// that marks what it just declared may be marking nothing. Seven call
    /// sites used to write the same `if let Some` around this — and they
    /// had to, because the function INDEXES the contract and a
    /// never-handed-out index would panic rather than be skipped. The
    /// decision is stated here instead, once, where it can be read.
    pub fn mark_internal(&mut self, index: Option<usize>) {
        if let Some(index) = index {
            self.contract.tensors[index].visibility = Visibility::Internal;
        }
    }

    /// Declare that entry `index` holds the scales for `of`.
    ///
    /// `index` is an `Option` because [`Self::define`] returns one: an
    /// encode-scoped load publishes only the towers, so a pass that pairs a
    /// scale may find it published nothing. That is not an error and not a
    /// case for the caller to spell out — a pairing for a tensor nobody
    /// declared is simply nothing to state. Four passes used to write the
    /// same `if let` around this call; they now state the pairing
    /// unconditionally and read the decision here, once.
    pub fn set_scales(&mut self, index: Option<usize>, scales: Scales) {
        if let Some(index) = index {
            self.contract.tensors[index].scales = Some(scales);
        }
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
    use model_loader::types::BackendKind;

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

    /// An axis this rank cannot divide is left WHOLE, not rounded.
    ///
    /// `local_extent` states a declared shape, and the alternative to
    /// stating the full width is stating a truncated one: eight heads over
    /// three ranks would declare two apiece and the last two heads would
    /// exist in no rank's contract at all. The loader rejects the
    /// indivisible shard by name instead, which is a message; a quietly
    /// short declaration is not.
    #[test]
    fn an_axis_that_does_not_divide_is_declared_whole() {
        let meta = empty_checkpoint();
        let policy = Policy::default();
        let enc = StoredEncoding::dense();
        let shape = LoadShape::dense(2, 64, false);
        let t = |tp| StorageTarget::for_backend(BackendKind::Cuda, 0, tp);

        let two = t(2);
        let b = Builder::new(&meta, "x", shape, &enc, &two, &policy);
        assert_eq!(
            b.local_extent(8),
            4,
            "a divisible axis is this rank's share"
        );
        assert_eq!(
            b.local_extent(9),
            9,
            "an indivisible one is left for the loader to refuse"
        );

        // And a single rank holds all of everything, divisible or not.
        let one = t(1);
        let b = Builder::new(&meta, "x", shape, &enc, &one, &policy);
        assert_eq!(b.local_extent(9), 9);
    }

    /// A per-tensor scale follows its weight VACUOUSLY -- it has no axis of
    /// its own to split.
    ///
    /// One number for a whole projection, so a shard of it is either the
    /// number or nothing. Splitting it would hand rank 1 an empty scale and
    /// rank 0 a scale for the whole tensor, and the weights each rank holds
    /// would be divided by a factor meant for a different set of rows.
    #[test]
    fn a_scale_with_no_axis_of_its_own_is_not_split() {
        let axis = Some(0);
        // The weight itself splits whatever its shape.
        assert_eq!(
            Builder::splittable_axis("model.layers.0.mlp.down_proj.weight", &[8, 4], axis),
            axis
        );
        // A companion scale with a real axis splits with it.
        assert_eq!(
            Builder::splittable_axis("model.layers.0.mlp.down_proj.weight_scale", &[8, 4], axis),
            axis
        );
        // A companion scale that is ONE number does not.
        assert_eq!(
            Builder::splittable_axis("model.layers.0.mlp.down_proj.weight_scale", &[1, 4], axis),
            None
        );
        // Nor one whose shape does not reach the axis at all.
        assert_eq!(
            Builder::splittable_axis("model.layers.0.mlp.down_proj.weight_scale", &[], axis),
            None
        );
        // And nothing splits when there is no axis to begin with.
        assert_eq!(
            Builder::splittable_axis("model.layers.0.mlp.down_proj.weight_scale", &[8, 4], None),
            None
        );
    }

    /// The four reasons a fused join declines, each of which would
    /// otherwise produce a bank whose rows are not what the GEMM reads.
    #[test]
    fn a_fused_join_declines_what_it_cannot_stack() {
        let good = || vec![("q", bf16(vec![4, 8])), ("k", bf16(vec![2, 8]))];
        let policy = Policy::default();
        let enc = StoredEncoding::dense();
        let shape = LoadShape::dense(1, 8, false);
        let target = target();
        let candidate = |rows: Vec<(&str, (Vec<i64>, Encoding))>, out: &str| {
            let rows: Vec<_> = rows
                .into_iter()
                .map(|(n, (shape, enc))| (n, shape, enc))
                .collect();
            let meta = checkpoint(&rows);
            let b = Builder::new(&meta, "x", shape, &enc, &target, &policy);
            b.fused_join_candidate(out.to_string(), &["q".into(), "k".into()])
                .map(|c| (c.cols, c.parts.len()))
        };

        assert_eq!(candidate(good(), "qkv"), Some((8, 2)), "two stackable rows");

        // ONE: the checkpoint already SHIPS the fused bank, so joining would
        // declare the same output twice.
        let mut shipped = good();
        shipped.push(("qkv", bf16(vec![6, 8])));
        assert_eq!(candidate(shipped, "qkv"), None);

        // TWO: a part that is not there. A join of one half is not a join.
        assert_eq!(candidate(vec![("q", bf16(vec![4, 8]))], "qkv"), None);

        // THREE: a part that is not a matrix. Stacking rank-1 rows would
        // concatenate along the only axis there is, which is the wrong one.
        let mut flat = good();
        flat[1] = ("k", bf16(vec![16]));
        assert_eq!(candidate(flat, "qkv"), None);

        // ... or not bf16, which a stacked bank has no way to say per part.
        let mut mixed = good();
        mixed[1] = ("k", (vec![2, 8], Encoding::Raw(DType::F32)));
        assert_eq!(candidate(mixed, "qkv"), None);

        // FOUR: parts of different WIDTHS. The stack is row-wise, so a
        // narrower part would leave the bank ragged.
        let mut ragged = good();
        ragged[1] = ("k", bf16(vec![2, 4]));
        assert_eq!(candidate(ragged, "qkv"), None);
    }

    /// Marking or pairing an entry that was never published is nothing,
    /// not a panic.
    ///
    /// Both functions INDEX the contract, so before they took the `Option`
    /// each of their eleven call sites had to guard -- and a site that
    /// forgot would panic on an out-of-range index the moment an
    /// encode-scoped load reached it. The decision belongs here, where it
    /// is one statement and one test rather than eleven of each.
    #[test]
    fn marking_or_pairing_what_was_never_published_is_nothing() {
        let meta = checkpoint(&[("w", vec![4, 8], Encoding::Raw(DType::BF16))]);
        let enc = StoredEncoding::dense();
        let target = target();
        let shape = LoadShape::dense(1, 8, false);
        let pairing = || Scales {
            of: "w".into(),
            granularity: QuantGranularity::PerGroup,
            group_size: 32,
            channel_axis: 0,
            form: ScaleForm::F32Factors,
        };

        // An encode-scoped builder publishes nothing for a decoder weight,
        // so `define` hands back `None` -- the case the guards were for.
        let encode = Policy {
            component: Component::Encode,
            ..Policy::default()
        };
        let mut b = Builder::new(&meta, "x", shape, &enc, &target, &encode);
        let index = b.define(
            "w".into(),
            Expr::src("w"),
            Encoding::Raw(DType::BF16),
            Some(vec![4, 8]),
        );
        assert_eq!(index, None, "an encode scope declared a decoder weight");
        b.mark_internal(index);
        b.set_scales(index, pairing());
        assert!(b.contract.tensors.is_empty());

        // And an entry that WAS published takes both.
        let policy = Policy::default();
        let mut b = Builder::new(&meta, "x", shape, &enc, &target, &policy);
        let index = b.define(
            "w".into(),
            Expr::src("w"),
            Encoding::Raw(DType::BF16),
            Some(vec![4, 8]),
        );
        assert_eq!(index, Some(0));
        b.mark_internal(index);
        b.set_scales(index, pairing());
        assert_eq!(b.contract.tensors[0].visibility, Visibility::Internal);
        assert_eq!(
            b.contract.tensors[0].scales.as_ref().map(|s| s.group_size),
            Some(32)
        );
    }

    /// A family whose tensors live under a prefix leaves everything OUTSIDE
    /// it alone -- including the expert banks the MoE slicing walks.
    ///
    /// The prefix is how a multimodal checkpoint keeps its towers and its
    /// language model apart in one file. A pass that ignored it would slice
    /// the vision tower's banks into the text model's contract, under names
    /// the text model's binder then looks up and does not find.
    #[test]
    fn the_moe_slicing_walks_only_what_the_prefix_admits() {
        let rows = [
            (
                "language_model.layers.0.mlp.experts.gate_up_proj",
                vec![4, 8, 16],
                Encoding::Raw(DType::BF16),
            ),
            (
                "vision_tower.layers.0.mlp.experts.gate_up_proj",
                vec![4, 8, 16],
                Encoding::Raw(DType::BF16),
            ),
        ];
        let meta = checkpoint(&rows);
        let policy = Policy::default();
        let enc = StoredEncoding::dense();
        let target = target();
        let shape = LoadShape::dense(1, 16, false);
        let mut b = Builder::new(&meta, "x", shape, &enc, &target, &policy);
        b.source_prefix("language_model.");
        b.fused_moe_gate_up_tp_slices(true).unwrap();
        let names: Vec<&str> = b.contract.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.iter().any(|n| n.contains("layers.0.mlp.experts")),
            "the admitted bank was not sliced: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.contains("vision_tower")),
            "a bank outside the prefix was sliced anyway: {names:?}"
        );
    }

    /// A checkpoint the loader RE-QUANTIZES is not also fused.
    ///
    /// The join stacks two bf16 matrices into one bank. Runtime quant turns
    /// each source into a quantized tensor with its own scales, and a scale
    /// is per-tensor: stacking two of them produces a bank whose rows want
    /// two different scales and whose contract can state only one. So the
    /// pass declines rather than producing a bank that is silently wrong in
    /// half its rows.
    #[test]
    fn a_re_quantized_checkpoint_is_not_also_fused() {
        let rows = [
            (
                "model.layers.0.self_attn.q_proj.weight",
                vec![4, 8],
                Encoding::Raw(DType::BF16),
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                vec![2, 8],
                Encoding::Raw(DType::BF16),
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                vec![2, 8],
                Encoding::Raw(DType::BF16),
            ),
        ];
        let meta = checkpoint(&rows);
        let enc = StoredEncoding::dense();
        let target = target();
        let shape = LoadShape::dense(1, 8, false);
        let joined = |quant| {
            let policy = Policy {
                runtime_quant: quant,
                ..Policy::default()
            };
            let mut b = Builder::new(&meta, "x", shape, &enc, &target, &policy);
            b.allow_bf16_runtime_quant();
            b.dense_fused_projection_joins().unwrap();
            b.contract
                .tensors
                .iter()
                .any(|t| t.name.contains("qkv_proj.fused"))
        };
        assert!(
            joined(RuntimeQuant::None),
            "an unquantized checkpoint joins"
        );
        assert!(
            !joined(RuntimeQuant::Int8),
            "a re-quantized one was fused anyway"
        );
    }

    /// A join whose parts are distributed DIFFERENTLY is left unfused.
    ///
    /// A fused bank is ONE buffer, so it has one distribution. Parts that
    /// disagree -- one row-sharded, one replicated -- have no single fused
    /// layout, and picking either would give some rank rows it does not own
    /// or rows it owns twice. The parts keep their own bind paths instead,
    /// which is a slower model rather than a wrong one.
    #[test]
    fn a_join_of_differently_distributed_parts_is_declined() {
        fn all_row_parallel(_: &str) -> Result<Option<u8>, Error> {
            Ok(Some(0))
        }
        fn gate_alone(name: &str) -> Result<Option<u8>, Error> {
            Ok(if name.contains("gate_proj") {
                Some(0)
            } else {
                None
            })
        }
        let rows = [
            (
                "model.layers.0.mlp.gate_proj.weight",
                vec![8, 4],
                Encoding::Raw(DType::BF16),
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                vec![8, 4],
                Encoding::Raw(DType::BF16),
            ),
        ];
        let meta = checkpoint(&rows);
        let enc = StoredEncoding::dense();
        let policy = Policy::default();
        let shape = LoadShape::dense(1, 4, false);
        let fused_under = |tp, rule: fn(&str) -> Result<Option<u8>, Error>| {
            let target = StorageTarget::for_backend(BackendKind::Cuda, 0, tp);
            let mut b = Builder::new(&meta, "x", shape, &enc, &target, &policy);
            b.shard_axis_fn(rule);
            b.dense_fused_projection_joins().unwrap();
            b.contract
                .tensors
                .iter()
                .any(|t| t.name.contains("gate_up_proj.fused"))
        };
        assert!(
            fused_under(1, all_row_parallel),
            "one rank shards nothing, so every part is replicated and they agree"
        );
        assert!(
            fused_under(2, all_row_parallel),
            "two row-parallel parts agree"
        );
        assert!(
            !fused_under(2, gate_alone),
            "a row-sharded part was fused with a replicated one"
        );
    }

    /// A contract nobody authored is a refusal, not an empty contract.
    ///
    /// An empty contract binds nothing, so the driver would come up with no
    /// weights and the first fire would read zeros -- a model that runs and
    /// emits noise. The message names the row, because "which model" is the
    /// question a reader has at that moment.
    #[test]
    fn a_contract_nobody_authored_is_refused() {
        let meta = empty_checkpoint();
        let policy = Policy::default();
        let enc = StoredEncoding::dense();
        let target = target();
        let shape = LoadShape::dense(1, 8, false);
        let b = Builder::new(&meta, "the-row", shape, &enc, &target, &policy);
        let err = b.finish().unwrap_err().to_string();
        assert!(err.contains("the-row"), "{err}");
        assert!(err.contains("no contract"), "{err}");
    }

    /// An ENCODE-scoped load is refused unless the family said it can do
    /// one.
    ///
    /// The component is a request from the caller: bind the towers alone.
    /// A family whose bind path has no tower scope would answer it by
    /// authoring the whole model, which is not what was asked for and is
    /// the wrong shape for the buffer the caller sized. Two families opt
    /// in; every other row has to say no.
    #[test]
    fn an_encode_scoped_load_is_refused_by_a_family_that_cannot_do_one() {
        // A tower tensor, because an encode-scoped publish declares those
        // and nothing else -- so a checkpoint of decoder weights would fail
        // for the EMPTY-contract reason instead of the one under test.
        let rows = [(
            "model.vision_tower.blocks.0.w",
            vec![4, 8],
            Encoding::Raw(DType::BF16),
        )];
        let meta = checkpoint(&rows);
        let enc = StoredEncoding::dense();
        let target = target();
        let shape = LoadShape::dense(1, 8, false);
        let authored = |opt_in: bool| {
            let policy = Policy {
                component: Component::Encode,
                ..Policy::default()
            };
            let mut b = Builder::new(&meta, "text-only-row", shape, &enc, &target, &policy);
            if opt_in {
                b.allow_encode_scope().unwrap();
            }
            b.publish_remaining().unwrap();
            b.finish()
        };
        let err = authored(false).unwrap_err().to_string();
        assert!(err.contains("text-only-row"), "{err}");
        assert!(err.contains("encode-scoped"), "{err}");
        let opted_in = authored(true).expect("a family that opted in is served");
        assert_eq!(opted_in.tensors.len(), 1);
    }

    /// A view of a tensor the checkpoint does not have is `false`, not an
    /// error and not a declaration.
    ///
    /// The distinction is the whole point of the `bool`: a family asks for
    /// a runtime-quantized view of an OPTIONAL weight, and "absent" has to
    /// be answerable without the family knowing in advance which rows ship
    /// it. Declaring one anyway would name a source no file contains.
    #[test]
    fn a_quantized_view_of_a_missing_tensor_declares_nothing() {
        let meta = checkpoint(&[("present", vec![4, 8], Encoding::Raw(DType::BF16))]);
        let policy = Policy::default();
        let enc = StoredEncoding::dense();
        let target = target();
        let mut b = Builder::new(
            &meta,
            "x",
            LoadShape::dense(1, 8, false),
            &enc,
            &target,
            &policy,
        );

        let scheme = QuantScheme::Int8Symmetric;
        assert!(!b.quantized_view("absent", "out".into(), scheme).unwrap());
        assert!(
            b.contract.tensors.is_empty(),
            "a decline that still declared something"
        );
        assert!(b.quantized_view("present", "out".into(), scheme).unwrap());
        assert!(!b.contract.tensors.is_empty());
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

    /// A pair the contract never declared is never paired.
    ///
    /// An Encode-scoped load publishes the towers and drops everything
    /// else, so `push_direct` on a decoder tensor comes back with no
    /// index -- there is no `TensorContract` to hang a `scales` on. Every
    /// other test here runs the FULL scope, where a direct push always
    /// lands, so this is the one arm that can be reached only by scoping
    /// the load down.
    ///
    /// Without the check the pairing walks a tensor list position that
    /// belongs to some other tensor, or to nothing at all: an index into
    /// a vector the drop just made shorter.
    #[test]
    fn a_scale_whose_weight_was_scoped_out_is_not_paired() {
        let rows = [
            (
                "model.vision_tower.embeddings.weight",
                vec![8, 8],
                Encoding::Raw(DType::BF16),
            ),
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
        let encode = Policy {
            component: Component::Encode,
            ..Policy::default()
        };
        let c = publish(&rows, &StoredEncoding::dense(), &encode, |b| {
            b.allow_encode_scope().expect("one rank")
        })
        .unwrap();
        let published: Vec<&str> = c.tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            published,
            vec!["model.vision_tower.embeddings.weight"],
            "the decoder weight and its scale are both outside the scope"
        );
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

    // ── the three refusals nothing had ever produced ──────────────────

    fn tp(tp_size: u32) -> StorageTarget {
        StorageTarget {
            preferred_alignment: 256,
            tp_size,
            ..StorageTarget::default()
        }
    }

    /// A rank cannot hold part of an attention head.
    ///
    /// The loader asks whether `tp_size` divides the ROW COUNT, which is
    /// the weaker question: 6 heads of 128 is 768 rows, and 4 divides 768
    /// while it does not divide 6. Only this builder knows `head_dim`, so
    /// the sharper question has to be asked here or not at all — and a
    /// rank given 192 rows would hold one and a half heads, which no
    /// attention kernel can read.
    #[test]
    fn a_head_count_the_world_does_not_divide_is_refused_here_not_by_row_count() {
        let meta = checkpoint(&[(
            "model.layers.0.self_attn.q_proj.weight",
            vec![768, 128],
            Encoding::Raw(DType::BF16),
        )]);
        let target = tp(4);
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
        b.shard_axis_fn(|_| Ok(Some(0)));
        let err = b
            .publish_remaining()
            .expect_err("6 heads do not divide 4 ways");
        let msg = err.to_string();
        assert!(
            msg.contains("6 head(s) of 128") && msg.contains("tp_size 4 does not"),
            "the refusal names the arithmetic: {msg}"
        );
        // And the loader's own question passes: 768 rows over 4 ranks is
        // 192 each. That is why this check has to exist here, where
        // `head_dim` is known, rather than there.
        let (rows, world, head_dim) = (768_i64, 4_i64, 128_i64);
        assert_eq!(rows % world, 0, "the row count divides");
        assert_ne!((rows / head_dim) % world, 0, "the head count does not");
    }

    /// Four ways a projection is not this rule's business, and each
    /// leaves by a different door.
    ///
    /// The last two are the ones that made a control lie: a fused bank
    /// whose rows are not a multiple of `head_dim` exits at the NAME
    /// check long before reaching the shape escape it appears to
    /// exercise, so the escape needs a name that IS in the list. And a
    /// projection this rank does not split by row has no head boundary
    /// to respect, however its head count divides.
    #[test]
    fn a_projection_this_rule_has_no_business_with_is_passed_through() {
        for (case, name, rows, axis) in [
            (
                "8 heads over 4 ranks divides",
                "model.layers.0.self_attn.q_proj.weight",
                1024_i64,
                Some(0),
            ),
            (
                "an output projection is not in the list",
                "model.layers.0.self_attn.o_proj.weight",
                768,
                Some(0),
            ),
            (
                "nor is a fused bank",
                "model.layers.0.self_attn.qkv_proj.weight",
                768,
                Some(0),
            ),
            (
                "a listed projection whose rows are not whole heads",
                "model.layers.0.self_attn.q_proj.weight",
                750,
                Some(0),
            ),
            (
                "column-parallel, so no row is cut",
                "model.layers.0.self_attn.q_proj.weight",
                768,
                Some(1),
            ),
            (
                "replicated, so nothing is cut at all",
                "model.layers.0.self_attn.q_proj.weight",
                768,
                None,
            ),
        ] {
            let meta = checkpoint(&[(name, vec![rows, 128], Encoding::Raw(DType::BF16))]);
            let target = tp(4);
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
            match axis {
                Some(0) => b.shard_axis_fn(|_| Ok(Some(0))),
                Some(_) => b.shard_axis_fn(|_| Ok(Some(1))),
                None => b.shard_axis_fn(|_| Ok(None)),
            }
            assert!(b.publish_remaining().is_ok(), "{case}");
        }
    }

    /// Encode-scoped loading and tensor parallelism do not compose.
    ///
    /// The towers are not sharded, so a family that scopes its bind to
    /// them has nothing to say about which rank holds what. Refused when
    /// the scope is asked for rather than at the first tensor.
    #[test]
    fn an_encode_scope_across_ranks_is_refused_when_it_is_asked_for() {
        for (case, component, tp_size, ok) in [
            (
                "the towers alone across four ranks",
                Component::Encode,
                4,
                false,
            ),
            ("the towers alone on one", Component::Encode, 1, true),
            ("the whole model across four", Component::Full, 4, true),
        ] {
            let meta = empty_checkpoint();
            let target = tp(tp_size);
            let policy = Policy {
                component,
                ..Policy::default()
            };
            let enc = StoredEncoding::dense();
            let mut b = Builder::new(
                &meta,
                "test-row",
                LoadShape::dense(1, 128, false),
                &enc,
                &target,
                &policy,
            );
            assert_eq!(b.allow_encode_scope().is_ok(), ok, "{case}");
            if !ok {
                let msg = b.allow_encode_scope().unwrap_err().to_string();
                assert!(msg.contains("does not support tensor parallelism"), "{msg}");
            }
        }
    }

    /// A companion scale splits exactly like the weight it scales, and
    /// the question is asked about the WEIGHT before the family's own
    /// rule sees the scale's name.
    ///
    /// Two steps, because a family's rule may key on either spelling:
    /// `foo.weight_scale` asks about `foo.weight` first and falls back to
    /// `foo`. Without this a family that supplied its own `shard_axis_fn`
    /// would have to remember every companion suffix, and forgetting one
    /// splits a scale differently from the tensor it scales.
    #[test]
    fn a_companion_scale_is_asked_about_under_the_weights_name() {
        fn only_the_weight(name: &str) -> Result<Option<u8>, Error> {
            Ok(if name == "block.weight" {
                Some(1)
            } else {
                None
            })
        }
        fn only_the_base(name: &str) -> Result<Option<u8>, Error> {
            Ok(if name == "block" { Some(1) } else { None })
        }
        let meta = empty_checkpoint();
        let target = tp(2);
        let policy = Policy::default();
        for (case, f) in [
            (
                "the family keys on `block.weight`",
                only_the_weight as fn(&str) -> _,
            ),
            ("the family keys on `block`", only_the_base as fn(&str) -> _),
        ] {
            let enc = StoredEncoding::dense();
            let mut b = Builder::new(
                &meta,
                "test-row",
                LoadShape::dense(1, 128, false),
                &enc,
                &target,
                &policy,
            );
            b.shard_axis_fn(f);
            for suffix in [
                ".weight_scale_inv",
                ".weight_scale",
                ".weight_packed",
                ".scale",
            ] {
                assert_eq!(
                    b.shard_axis(&format!("block{suffix}")).expect("no refusal"),
                    Some(1),
                    "{case}: block{suffix} follows what it scales"
                );
            }
            assert_eq!(
                b.shard_axis("elsewhere.weight_scale").expect("no refusal"),
                None,
                "{case}: a scale whose weight replicates replicates too"
            );
        }
    }

    /// At one rank there is nothing to ask, and the family's rule is not
    /// consulted at all — which is why a rule that panics on an unknown
    /// name is still safe single-GPU.
    #[test]
    fn a_single_rank_never_asks_the_familys_rule() {
        fn never(_: &str) -> Result<Option<u8>, Error> {
            unreachable!("a single rank has nothing to split")
        }
        let meta = empty_checkpoint();
        let target = tp(1);
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
        b.shard_axis_fn(never);
        assert_eq!(b.shard_axis("anything.weight").expect("no refusal"), None);
    }

    /// The embedding table and the head, which are asked about before
    /// the family's rule and answered by the two knobs.
    #[test]
    fn the_table_and_the_head_are_answered_before_the_familys_rule() {
        fn column(_: &str) -> Result<Option<u8>, Error> {
            Ok(Some(1))
        }
        let meta = empty_checkpoint();
        let target = tp(2);
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
        b.shard_axis_fn(column);
        assert_eq!(
            b.shard_axis("model.embed_tokens.weight").expect("ok"),
            Some(1),
            "unasked, the family's rule answers"
        );
        b.shard_embed_tokens();
        b.replicate_lm_head();
        assert_eq!(
            b.shard_axis("model.embed_tokens.weight").expect("ok"),
            Some(0),
            "row-parallel to save per-rank memory, not column"
        );
        assert_eq!(
            b.shard_axis("model.lm_head.weight").expect("ok"),
            None,
            "replicated, so every rank can produce whole logits"
        );
    }

    /// A checkpoint whose experts are already fused, with the byte spans
    /// stated rather than derived.
    ///
    /// `span_bytes` is what the fusion budget reads, and a fixture that had to
    /// actually be that large could not be built.
    fn fused_experts(shape: Vec<i64>, span_bytes: u64) -> CheckpointMetadata {
        CheckpointMetadata {
            files: Vec::new(),
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "model.layers.0.mlp.experts.gate_up_proj".to_string(),
                file_id: model_loader::types::FileId(0),
                file_offset: 0,
                span_bytes,
                shape,
                encoding: Encoding::Raw(DType::BF16),
            }],
        }
    }

    fn moe_slices(
        meta: &CheckpointMetadata,
        tp_size: u32,
        gate_second: bool,
    ) -> Result<ModelContract, Error> {
        let target = tp(tp_size);
        let policy = Policy::default();
        let enc = StoredEncoding::dense();
        let mut b = Builder::new(
            meta,
            "test-row",
            LoadShape::dense(1, 128, false),
            &enc,
            &target,
            &policy,
        );
        b.fused_moe_gate_up_tp_slices(gate_second)?;
        b.publish_remaining()?;
        b.finish()
    }

    /// `gate_second` republishes each expert as `[up|gate]`, and it is the one
    /// reason this pass runs without sharding at all.
    ///
    /// The checkpoint stores `[gate|up]`; flashinfer's CUTLASS MoE reads
    /// fc1's output the other way round. Swapping the two bands is a pure
    /// reordering -- same tensor, same shape, same bytes -- so nothing
    /// downstream can tell it happened, and a model built with the halves the
    /// wrong way round applies its gate to the up projection and its up to the
    /// gate. That silu-gated product is still finite and still the right
    /// shape, so the model generates fluent nonsense rather than failing.
    ///
    /// The shape is asserted to be UNCHANGED on purpose: a reordering that
    /// altered the extents would be caught by everything, which is exactly why
    /// one that does not is worth a test.
    #[test]
    fn gate_second_reorders_the_two_bands_without_sharding_or_resizing_anything() {
        let meta = fused_experts(vec![4, 2 * 64, 32], 4 * 128 * 32 * 2);
        let name = "model.layers.0.mlp.experts.gate_up_proj";

        let c = moe_slices(&meta, 1, true).expect("world of one");
        let t = c
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} was not published"));

        // The whole expression, because the ORDER is the entire content of
        // this pass and an order is only visible against the offsets. `up` is
        // the band at [I, 2I) and it has to come first.
        assert_eq!(
            t.expr,
            Expr::concat(
                1,
                vec![
                    Expr::src(name).slice(1, 64, 64),
                    Expr::src(name).slice(1, 0, 64),
                ]
            ),
            "gate_second did not publish [up|gate]"
        );

        // And the extents are untouched: a reordering that resized something
        // would be caught by everything, which is why one that does not is
        // worth stating.
        assert_eq!(t.shape.clone().expect("a declared shape"), vec![4, 128, 32]);
    }

    /// Without `gate_second`, a world of one has nothing to do.
    ///
    /// The early return is not an optimisation: at `tp_size == 1` the two
    /// bands are `[0, I)` and `[I, 2I)` of a tensor that is already exactly
    /// those two bands concatenated, so re-publishing it would replace a
    /// direct read with a slice-slice-concat that computes the identity. The
    /// tensor should come out untouched, by the ordinary dense path.
    #[test]
    fn an_unsharded_checkpoint_in_the_checkpoints_own_order_is_left_alone() {
        let meta = fused_experts(vec![4, 128, 32], 4 * 128 * 32 * 2);
        let name = "model.layers.0.mlp.experts.gate_up_proj";
        let c = moe_slices(&meta, 1, false).expect("world of one");
        let t = c
            .tensors
            .iter()
            .find(|t| t.name == name)
            .expect("published");
        assert_eq!(
            t.expr,
            Expr::src(name),
            "the identity was rebuilt as a concat"
        );
    }

    /// Under TP the halves are sharded INDEPENDENTLY and re-joined, so each
    /// rank's gate and up stay adjacent within its own expert.
    ///
    /// Sharding the fused tensor as one band would give rank 0 the whole gate
    /// and rank 1 the whole up -- each rank holding a complete half of an
    /// operator it needs both halves of. The rejoin is what makes the local
    /// tensor a smaller version of the same thing rather than a piece of a
    /// different one.
    #[test]
    fn each_rank_keeps_its_own_gate_beside_its_own_up() {
        let meta = fused_experts(vec![4, 2 * 64, 32], 4 * 128 * 32 * 2);
        let name = "model.layers.0.mlp.experts.gate_up_proj";
        let c = moe_slices(&meta, 2, false).expect("two ranks");
        let t = c
            .tensors
            .iter()
            .find(|t| t.name == name)
            .expect("published");

        // Each band is sliced out FIRST and sharded SECOND, then the two
        // local halves are concatenated back together on the band axis.
        // Sharding the fused tensor as one band instead would give rank 0 the
        // whole gate and rank 1 the whole up -- each rank holding a complete
        // half of an operator it needs both halves of.
        assert_eq!(
            t.expr,
            Expr::concat(
                1,
                vec![
                    Expr::src(name).slice(1, 0, 64).shard(1),
                    Expr::src(name).slice(1, 64, 64).shard(1),
                ]
            ),
            "the two bands are not sharded independently and rejoined"
        );
        assert_eq!(
            t.shape.clone().expect("a declared shape"),
            vec![4, 2 * 32, 32],
            "a rank of two should hold half of each band, not one whole band"
        );
    }

    /// Three shapes and one encoding this pass declines, each by its own door.
    #[test]
    fn a_fused_expert_tensor_this_pass_cannot_split_is_declined_or_refused() {
        let name = "model.layers.0.mlp.experts.gate_up_proj";

        // Rank 2 is not an expert grid: there is no expert axis to keep.
        let meta = fused_experts(vec![128, 32], 128 * 32 * 2);
        let c = moe_slices(&meta, 2, true).expect("declined, not refused");
        let t = c
            .tensors
            .iter()
            .find(|t| t.name == name)
            .expect("published");
        assert_eq!(t.expr, Expr::src(name), "a rank-2 tensor was split anyway");

        // An odd middle extent has no halfway point to cut at.
        let meta = fused_experts(vec![4, 65, 32], 4 * 65 * 32 * 2);
        let c = moe_slices(&meta, 2, true).expect("declined, not refused");
        let t = c
            .tensors
            .iter()
            .find(|t| t.name == name)
            .expect("published");
        assert_eq!(
            t.expr,
            Expr::src(name),
            "an odd band count was halved anyway"
        );

        // A name that is neither of the two spellings.
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "model.layers.0.mlp.experts.up_gate_proj".to_string(),
                file_id: model_loader::types::FileId(0),
                file_offset: 0,
                span_bytes: 4 * 128 * 32 * 2,
                shape: vec![4, 128, 32],
                encoding: Encoding::Raw(DType::BF16),
            }],
        };
        let c = moe_slices(&meta, 2, true).expect("declined, not refused");
        let t = c
            .tensors
            .iter()
            .find(|t| t.name == "model.layers.0.mlp.experts.up_gate_proj")
            .expect("published");
        assert_eq!(t.expr, Expr::src("model.layers.0.mlp.experts.up_gate_proj"));

        // A packed encoding whose elements are not at affine byte offsets
        // cannot be sliced at all, and that one IS refused: silently
        // declining would leave the tensor unsharded while the forward pass
        // went on expecting a local band.
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: name.to_string(),
                file_id: model_loader::types::FileId(0),
                file_offset: 0,
                span_bytes: 4 * 128 * 32,
                shape: vec![4, 128, 32],
                encoding: mxfp4_encoding(2),
            }],
        };
        let e = moe_slices(&meta, 2, true).expect_err("a packed bank was sliced");
        let msg = e.to_string();
        assert!(
            msg.contains("non-affine") && msg.contains(name),
            "the refusal should name the tensor and the reason: {msg}"
        );
    }

    /// The fusion budget picks a model CLASS, and QKV wins when only one fits.
    ///
    /// Fused dense projections replace the originals and the unfused fallback
    /// binds non-owning views into the fused buffer, so this is not a memory
    /// budget in the usual sense -- nothing is duplicated. It selects which
    /// groups get a fused GEMM: everything through 8B-class models, and
    /// QKV-only above that, where gate/up fusion regressed.
    ///
    /// Two things make it worth pinning. The preference is not symmetric: when
    /// the pair does not fit, QKV goes first because it is much smaller than
    /// gate/up on Qwen-style models and it is what enables the fused decode
    /// postprocess. And it is a greedy fill rather than a choice: the second
    /// admission is charged against what the first actually SPENT, so a large
    /// QKV can exclude a gate/up that would have fit alone, while a QKV that
    /// does not fit at all spends nothing and leaves gate/up the whole budget.
    /// Writing the second test as `gate_up_bytes <= BUDGET` reads like the
    /// same rule and lets both in.
    ///
    /// The spans are stated rather than allocated; ten gigabytes of fixture is
    /// not a thing a test can build.
    #[test]
    fn when_both_fused_groups_do_not_fit_the_budget_qkv_is_the_one_that_gets_in() {
        const GB: u64 = 1024 * 1024 * 1024;

        // `span_bytes` is what the budget reads, so the shapes stay small and
        // the declared spans do the talking.
        let rows = |qkv_gb: u64, gate_up_gb: u64| {
            let one = |name: &str, span: u64| RawTensor {
                id: TensorId(0),
                name: name.to_string(),
                file_id: model_loader::types::FileId(0),
                file_offset: 0,
                span_bytes: span,
                shape: vec![64, 64],
                encoding: Encoding::Raw(DType::BF16),
            };
            let p = "model.layers.0.";
            let mut tensors = vec![
                one(&format!("{p}self_attn.q_proj.weight"), qkv_gb * GB / 3),
                one(&format!("{p}self_attn.k_proj.weight"), qkv_gb * GB / 3),
                one(&format!("{p}self_attn.v_proj.weight"), qkv_gb * GB / 3),
                one(&format!("{p}mlp.gate_proj.weight"), gate_up_gb * GB / 2),
                one(&format!("{p}mlp.up_proj.weight"), gate_up_gb * GB / 2),
            ];
            for (i, t) in tensors.iter_mut().enumerate() {
                t.id = TensorId(i as u32);
            }
            CheckpointMetadata {
                files: Vec::new(),
                tensors,
            }
        };

        let fused = |qkv_gb: u64, gate_up_gb: u64| -> (bool, bool) {
            let meta = rows(qkv_gb, gate_up_gb);
            let target = tp(1);
            let policy = Policy::default();
            let enc = StoredEncoding::dense();
            let mut b = Builder::new(
                &meta,
                "test-row",
                LoadShape::dense(1, 64, false),
                &enc,
                &target,
                &policy,
            );
            b.dense_fused_projection_joins().expect("joins");
            b.publish_remaining().expect("publish");
            let c = b.finish().expect("finish");
            let has = |n: &str| c.tensors.iter().any(|t| t.name == n);
            (
                has("model.layers.0.self_attn.qkv_proj.fused.weight"),
                has("model.layers.0.mlp.gate_up_proj.fused.weight"),
            )
        };

        // Comfortably inside: both groups fuse.
        assert_eq!(fused(1, 2), (true, true), "a small model should fuse both");

        // Over the 10 GiB line together, and each under it alone. QKV gets in
        // and gate/up does not -- and gate/up is the LARGER of the two, so a
        // rule that simply kept the bigger one would answer the other way.
        assert_eq!(
            fused(3, 9),
            (true, false),
            "over budget together, QKV is the group that should fuse"
        );

        // QKV alone exceeds the budget. It is dropped, and it spends NOTHING
        // on the way out: `used` is charged only where the group is actually
        // admitted, so gate/up then gets the whole budget rather than
        // inheriting a debt for a fusion that did not happen. This is the
        // arm that distinguishes a greedy fill in priority order from a
        // strict "QKV or nothing".
        assert_eq!(
            fused(12, 2),
            (false, true),
            "a QKV that does not fit should not spend budget on its way out"
        );
    }

    // ── The pairing a shipped FP8 scale gets, or does not ────────────

    fn raw_t(id: u32, name: &str, shape: Vec<i64>, encoding: Encoding) -> RawTensor {
        use model_loader::types::{FileId, TensorId};
        RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: 0,
            shape,
            encoding,
        }
    }

    const W: &str = "model.layers.0.self_attn.o_proj.weight";
    const S: &str = "model.layers.0.self_attn.o_proj.weight_scale";

    /// An FP8 weight and the factors the checkpoint shipped beside it.
    /// 128 columns against 4 scale columns is a block of 32.
    fn fp8_pair(weight_cols: i64, scale_cols: i64) -> Vec<RawTensor> {
        vec![
            raw_t(1, W, vec![64, weight_cols], Encoding::Raw(DType::F8E4M3)),
            raw_t(2, S, vec![64, scale_cols], Encoding::Raw(DType::F32)),
        ]
    }

    fn publish_fp8(tensors: Vec<RawTensor>) -> ModelContract {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors,
        };
        let t = target();
        let policy = Policy::default();
        let enc = StoredEncoding::dense();
        let mut b = Builder::new(
            &meta,
            "fp8-test",
            LoadShape::dense(1, 64, false),
            &enc,
            &t,
            &policy,
        );
        b.publish_remaining().expect("publish");
        b.finish().expect("finish")
    }

    fn scales_of(c: &ModelContract, name: &str) -> Option<Scales> {
        c.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name} is declared"))
            .scales
            .clone()
    }

    /// The block size is READ OFF the two shapes, never assumed.
    ///
    /// A scale that never names its weight is not a load error: the
    /// weight binds as raw FP8 and the factors sit beside it unread, so
    /// every number the kernel produces is off by whatever the factors
    /// were going to correct. That is a wrong answer, not a failure, and
    /// the only place it can be caught is here.
    #[test]
    fn a_shipped_fp8_scale_names_its_weight_and_the_block_it_covers() {
        let c = publish_fp8(fp8_pair(128, 4));
        let s = scales_of(&c, S).expect("the scale names its weight");
        assert_eq!(s.of, W, "the factors must name the tensor they scale");
        assert_eq!(s.group_size, 32, "128 columns over 4 factors is 32");
        assert_eq!(s.granularity, QuantGranularity::PerGroup);
        assert_eq!(s.form, ScaleForm::F32Factors);
    }

    /// Different shapes, different block -- it is not a constant.
    #[test]
    fn the_block_comes_from_the_shapes_and_not_from_a_constant() {
        for (cols, factors, block) in [(128i64, 4i64, 32u32), (128, 1, 128), (256, 2, 128)] {
            let c = publish_fp8(fp8_pair(cols, factors));
            assert_eq!(
                scales_of(&c, S).expect("paired").group_size,
                block,
                "{cols} columns over {factors} factors"
            );
        }
    }

    /// A companion that is not really FP8 is LEFT ALONE.
    ///
    /// The `.weight_scale` suffix is a naming convention, not a
    /// guarantee. Binding factors to a BF16 weight would tell the loader
    /// to dequantize something that was never quantized.
    #[test]
    fn factors_beside_a_weight_that_is_not_fp8_are_left_unpaired() {
        let mut ck = fp8_pair(128, 4);
        ck[0].encoding = Encoding::Raw(DType::BF16);
        assert!(
            scales_of(&publish_fp8(ck), S).is_none(),
            "a BF16 weight was told it has block factors"
        );
    }

    /// A scale whose weight is not in the checkpoint at all.
    #[test]
    fn factors_with_no_weight_beside_them_are_left_unpaired() {
        let ck = vec![raw_t(2, S, vec![64, 4], Encoding::Raw(DType::F32))];
        assert!(scales_of(&publish_fp8(ck), S).is_none());
    }

    /// A scale that states ZERO factor columns.
    ///
    /// `weight_cols / scale_cols` is the block, so a zero divides by
    /// zero and a scale with no shape at all has no last dimension to
    /// divide by. Both are left unpaired rather than panicking, because
    /// this runs over every tensor of every checkpoint.
    #[test]
    fn factors_that_cannot_state_a_block_are_left_unpaired() {
        for shape in [vec![64, 0], Vec::new()] {
            let mut ck = fp8_pair(128, 4);
            ck[1].shape = shape.clone();
            assert!(
                scales_of(&publish_fp8(ck), S).is_none(),
                "a scale shaped {shape:?} produced a block anyway"
            );
        }
    }

    /// FEWER weight columns than factors is a block of zero.
    ///
    /// Integer division truncates, so 4 columns over 8 factors is `0` --
    /// a `group_size` of zero is a division the lowering does not guard.
    /// The pass declines rather than declaring it.
    #[test]
    fn a_block_that_truncates_to_zero_is_declined_rather_than_declared() {
        assert!(
            scales_of(&publish_fp8(fp8_pair(4, 8)), S).is_none(),
            "four columns over eight factors declared a block of zero"
        );
    }

    /// A weight an earlier pass CLAIMED is not published under this name.
    ///
    /// Naming a consumed tensor would be a contract the loader rejects
    /// outright -- the `of` would point at an output that does not
    /// exist. This is the one decline that depends on what another pass
    /// did rather than on the tensors themselves.
    #[test]
    fn factors_whose_weight_an_earlier_pass_consumed_are_left_unpaired() {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: fp8_pair(128, 4),
        };
        let t = target();
        let policy = Policy::default();
        let enc = StoredEncoding::dense();
        let mut b = Builder::new(
            &meta,
            "fp8-test",
            LoadShape::dense(1, 64, false),
            &enc,
            &t,
            &policy,
        );
        let claimed = b.find(W).expect("the weight is there").id;
        b.consumed.insert(claimed);
        b.publish_remaining().expect("publish");
        let c = b.finish().expect("finish");
        assert!(
            scales_of(&c, S).is_none(),
            "the factors named a weight no pass published"
        );
        assert!(
            !c.tensors.iter().any(|x| x.name == W),
            "the fixture must actually withhold the weight, or this test \
             is asserting over a weight that IS published"
        );
    }
}
