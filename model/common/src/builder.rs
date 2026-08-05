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
//! What the C++ builder read from three places arrives as three arguments:
//! the checkpoint's own tensor table, the [`ModelFacts`] the caller parsed,
//! and the [`StorageTarget`] + [`Policy`] pair — the device's measurements
//! and the caller's decisions, exactly the split
//! [`policy`](crate::policy) argues for.

use std::collections::{HashMap, HashSet};

use pie_loader::checkpoint::{CheckpointMetadata, RawTensor};
use pie_loader::contract::{
    Expr, GroupContract, ModelContract, Scales, TensorContract, TensorType,
};
use pie_loader::error::Error;
use pie_loader::plan::StorageTarget;
use pie_loader::types::{
    Axis, DType, Encoding, QuantGranularity, QuantScheme, QuantSpec, RepackLayout, ScaleForm,
    TensorId, Visibility,
};

use super::facts::ModelFacts;
use super::policy::{Component, Mxfp4MoePolicy, Mxfp4MoeRequest, Policy, RuntimeQuant};
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
/// [`probe`](crate::probe).
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
    facts: &'a ModelFacts,
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
    pub fn new(
        metadata: &'a CheckpointMetadata,
        facts: &'a ModelFacts,
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
            facts,
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

    pub fn facts(&self) -> &ModelFacts {
        self.facts
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
        let d = i64::from(self.facts.head_dim);
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
    /// [`Projections::InPlace`]: crate::policy::Projections::InPlace
    pub fn dense_fused_projection_joins(&mut self) -> Result<(), Error> {
        use crate::policy::Projections;
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
        let mut modules: Vec<String> = (0..self.facts.num_hidden_layers)
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
                "encode-scoped loading is not supported for model_type='{}'",
                self.facts.model_type
            ));
        }
        if self.contract.tensors.is_empty() && self.contract.groups.is_empty() {
            return fail(format!(
                "no contract was authored for model_type='{}'; the driver must \
                 declare what it binds",
                self.facts.model_type
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
        if !self.facts.quant_method.is_empty() && scheme != QuantScheme::Mxfp4E2M1E8M0 {
            return Ok(None);
        }
        let allowed = if scheme == QuantScheme::Mxfp4E2M1E8M0 {
            self.allow_mxfp4_rq
        } else {
            self.allow_bf16_rq
        };
        if !allowed {
            return fail(format!(
                "runtime_quant={:?} is not supported for model_type='{}'",
                self.policy.runtime_quant, self.facts.model_type
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
