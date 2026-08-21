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

pub fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

pub(crate) fn logical_dtype(encoding: &Encoding) -> DType {
    match encoding {
        Encoding::Raw(dtype) => *dtype,
        Encoding::Quant(spec) => spec.logical_dtype,
    }
}

pub fn is_raw(encoding: &Encoding, dtype: DType) -> bool {
    matches!(encoding, Encoding::Raw(have) if *have == dtype)
}

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

fn runtime_quantizable_name(name: &str, scheme: QuantScheme) -> bool {
    if scheme == QuantScheme::Mxfp4E2M1E8M0 {

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

pub struct FusedCandidate<'a> {
    output_name: String,

    parts: Vec<&'a RawTensor>,
    cols: i64,
    bytes: u64,
}

pub struct Builder<'a> {

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

    pub fn new(
        metadata: &'a CheckpointMetadata,
        id: &'static str,
        shape: crate::catalog::LoadShape,
        encoding: &'a crate::encoding::Encoding,
        target: &'a StorageTarget,
        policy: &'a Policy,
    ) -> Self {

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

    #[must_use]
    pub fn id(&self) -> &'static str {
        self.id
    }

    #[must_use]
    pub fn shape(&self) -> crate::catalog::LoadShape {
        self.shape
    }

    #[must_use]
    pub fn encoding(&self) -> &crate::encoding::Encoding {
        self.encoding
    }

    #[must_use]
    pub fn naming(&self) -> Naming {
        self.policy.naming
    }

    pub fn target(&self) -> &StorageTarget {
        self.target
    }

    pub fn mxfp4_moe(&self) -> Mxfp4MoePolicy {
        self.mxfp4_moe
    }

    pub fn mxfp4_moe_request(&self) -> Mxfp4MoeRequest {
        self.policy.moe_request
    }

    pub fn stream_routed_experts(&self) -> bool {
        self.policy.stream_routed_experts
    }

    pub fn runtime_quant(&self) -> super::policy::RuntimeQuant {
        self.policy.runtime_quant
    }

    pub fn knobs(&self) -> &super::policy::FamilyKnobs {
        &self.policy.knobs
    }

    pub fn tensors(&self) -> &[&'a RawTensor] {
        &self.tensors
    }

    pub fn source_prefix(&mut self, prefix: &str) {
        self.source_prefix.clear();
        if self.tensors.iter().any(|raw| raw.name.starts_with(prefix)) {
            self.source_prefix = prefix.to_string();
        }
    }

    pub fn output_name(&self, raw_name: &str) -> String {
        raw_name
            .strip_prefix(self.source_prefix.as_str())
            .unwrap_or(raw_name)
            .to_string()
    }

    pub fn source_name(&self, bound_name: &str) -> String {
        format!("{}{bound_name}", self.source_prefix)
    }

    pub fn decoder_layer_prefix(&mut self, prefix: &str) {
        self.decoder_layer_prefix = prefix.to_string();
    }

    pub fn decoder_layer_prefix_value(&self) -> &str {
        &self.decoder_layer_prefix
    }

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

    pub fn shard_axis_fn(&mut self, f: fn(&str) -> Result<Option<u8>, Error>) {
        self.shard_axis_fn = f;
    }

    pub fn decide_mxfp4_moe(&mut self, policy: Mxfp4MoePolicy) {
        self.mxfp4_moe = policy;
    }

    pub fn shard_embed_tokens(&mut self) {
        self.shard_embed_tokens = true;
    }

    pub fn replicate_lm_head(&mut self) {
        self.replicate_lm_head = true;
    }

    pub fn allow_bf16_runtime_quant(&mut self) {
        self.allow_bf16_rq = true;
    }

    pub fn allow_mxfp4_runtime_quant(&mut self) {
        self.allow_mxfp4_rq = true;
    }

    pub fn allow_encode_scope(&mut self) -> Result<(), Error> {
        if self.policy.component == Component::Encode && self.target.tp_size != 1 {
            return fail("encode-scoped loading does not support tensor parallelism");
        }
        self.encode_scope_allowed = true;
        Ok(())
    }

    pub fn also_join_module(&mut self, module_prefix: &str) {
        self.extra_join_modules.push(module_prefix.to_string());
    }

    pub fn find(&self, name: &str) -> Option<&'a RawTensor> {
        self.by_name.get(name).copied()
    }

    pub fn consume(&mut self, id: TensorId) {
        self.consumed.insert(id);
    }

    pub fn split(&self, expr: Expr, axis: u8) -> Expr {
        if self.target.tp_size <= 1 {
            expr
        } else {
            expr.shard(axis)
        }
    }

    pub fn local_extent(&self, full: i64) -> i64 {
        let world = i64::from(self.target.tp_size.max(1));
        if full % world == 0 {
            full / world
        } else {
            full
        }
    }

    pub fn band(&self, expr: Expr, axis: u8, start: i64, len: i64) -> (Expr, i64) {
        (
            self.split(expr.slice(axis, start, len), axis),
            self.local_extent(len),
        )
    }

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

        if self.shard_embed_tokens && name.ends_with(".embed_tokens.weight") {
            return Ok(Some(0));
        }
        if self.replicate_lm_head && name.ends_with(".lm_head.weight") {
            return Ok(None);
        }

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

    pub fn push_direct(
        &mut self,
        raw: &RawTensor,
        output: String,
        axis: Option<u8>,
    ) -> Result<Option<usize>, Error> {
        let axis = Self::splittable_axis(&raw.name, &raw.shape, axis);
        self.check_head_granularity(&raw.name, &raw.shape, axis)?;
        let (expr, shape) = self.shard(Expr::src(&raw.name), raw.shape.clone(), axis);

        let shape = if shape.is_empty() { None } else { Some(shape) };
        Ok(self.define(output, expr, raw.encoding.clone(), shape))
    }

    pub fn push_expr(&mut self, output: String, raw: &RawTensor, shape: Vec<i64>, expr: Expr) {
        self.define(
            output,
            expr,
            Encoding::Raw(logical_dtype(&raw.encoding)),
            Some(shape),
        );
        self.consumed.insert(raw.id);
    }

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

    pub fn mark_internal(&mut self, index: Option<usize>) {
        if let Some(index) = index {
            self.contract.tensors[index].visibility = Visibility::Internal;
        }
    }

    pub fn set_scales(&mut self, index: Option<usize>, scales: Scales) {
        if let Some(index) = index {
            self.contract.tensors[index].scales = Some(scales);
        }
    }

    pub fn push_group(&mut self, group: GroupContract) {
        self.contract.groups.push(group);
    }

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

        const BUDGET_BYTES: u64 = 10 * 1024 * 1024 * 1024;
        let mut chosen = Vec::new();
        if qkv_bytes + gate_up_bytes <= BUDGET_BYTES {
            chosen.extend(qkv);
            chosen.extend(gate_up);
        } else {

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

                    if self.requant_consumes_shipped_scale(raw, scheme) {
                        continue;
                    }
                    let axis = self.shard_axis(&raw.name)?;
                    let defined = self.push_direct(raw, self.output_name(&raw.name), axis)?;
                    self.state_shipped_block_scales(raw, defined, scheme);
                }
            }
        }
        Ok(())
    }

    fn requant_consumes_shipped_scale(&self, raw: &RawTensor, scheme: Option<QuantScheme>) -> bool {
        let Some(scheme) = scheme else {
            return false;
        };
        let Some(weight) = probe::companion_weight_name(&raw.name) else {
            return false;
        };
        let Some(companion) = self.find(&weight) else {
            return false;
        };
        is_raw(&companion.encoding, DType::F8E4M3)
            && !self.consumed.contains(&companion.id)
            && self.source_name_allowed(&companion.name)
            && runtime_quantizable_name(&companion.name, scheme)
    }

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

        if self.consumed.contains(&companion.id) || !self.source_name_allowed(&companion.name) {
            return;
        }

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

    fn decode_bound_blocks(&mut self) {
        for tensor in &mut self.contract.tensors {
            if tensor.visibility != Visibility::Public {
                continue;
            }
            let Encoding::Quant(spec) = &tensor.encoding else {
                continue;
            };
            if !spec.scheme.is_self_contained() {
                continue;
            }
            let decoded = Encoding::Raw(spec.logical_dtype);
            let expr = std::mem::replace(&mut tensor.expr, Expr::Src(String::new()));
            tensor.expr = expr.cast(decoded.clone());
            tensor.encoding = decoded;
        }
    }

    pub fn finish(mut self) -> Result<ModelContract, Error> {
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
        self.decode_bound_blocks();
        Ok(self.contract)
    }

    fn source_name_allowed(&self, raw_name: &str) -> bool {
        self.source_prefix.is_empty() || raw_name.starts_with(&self.source_prefix)
    }

    fn runtime_quant_scheme(&self) -> Result<Option<QuantScheme>, Error> {
        let scheme = match self.policy.runtime_quant {
            RuntimeQuant::None => return Ok(None),
            RuntimeQuant::Fp8 => QuantScheme::Fp8E4M3,
            RuntimeQuant::Int8 => QuantScheme::Int8Symmetric,
            RuntimeQuant::Mxfp4 => QuantScheme::Mxfp4E2M1E8M0,
            RuntimeQuant::Int4 => QuantScheme::MlxAffineU4,
        };

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

pub fn default_shard_axis(name: &str) -> Result<Option<u8>, Error> {
    Ok(probe::hf_shard_axis(name))
}

pub fn mxfp4_encoding(channel_axis: u8) -> Encoding {
    Encoding::Quant(QuantSpec {
        scheme: QuantScheme::Mxfp4E2M1E8M0,
        logical_dtype: DType::BF16,
        bits_per_element: 4,

        group_size: 32,
        channel_axis: Some(Axis(channel_axis)),
    })
}

pub fn int4b8_encoding(channel_axis: u8) -> Encoding {
    Encoding::Quant(QuantSpec {
        scheme: QuantScheme::Int4B8,
        logical_dtype: DType::BF16,
        bits_per_element: 4,
        group_size: 32,
        channel_axis: Some(Axis(channel_axis)),
    })
}

pub fn align_up(value: i64, alignment: i64) -> Result<i64, Error> {
    if value < 0 || alignment <= 0 {
        return fail("contract: align_up needs a non-negative value and a positive alignment");
    }
    Ok((value + alignment - 1) / alignment * alignment)
}
