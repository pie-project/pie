//! THE RECORDER — what a declaration calls, in computation order.

use super::*;

/// Records ops as a declaration executes. The declaration calls these
/// methods in computation order; the builder assigns value ids and keeps
/// the op list flat — structure (layers) is carried on the ops themselves.
///
/// # On the visibility of the recording methods
///
/// Twelve of them (`set_layer`, `open_guard`, `push_seam`,
/// `try_fold_residual`, …) were `pub(crate)` while the authoring surface and
/// the IR were one crate, and they are `pub` now because the surface is
/// `model-dsl` and a crate boundary has no `pub(super)`.
///
/// The widening is smaller than it looks, and it is worth being precise about
/// why: these methods were never private in the sense that mattered — the
/// only caller was always the eDSL, and the only thing `pub(crate)` bought
/// was that the compiler said so. What actually keeps a declaration from
/// hand-rolling a tape is that [`Self::finish`] refuses a plan the signature
/// tables do not cover, and that check is unchanged and unavoidable. A caller
/// who reaches past `model-dsl` to push ops directly still cannot produce a
/// [`ForwardPlan`] that names a symbol no backend implements.
pub struct TraceBuilder {
    family: String,
    values: Vec<ValueInfo>,
    ops: Vec<Op>,
    layer: Option<u32>,
    /// Seam statements in text order ([`SeamStatement`]).
    seams: Vec<SeamStatement>,
    /// Open [`Self::open_guard`] depth. Nesting is part of the
    /// vocabulary since A1 (north-star-dsl.md, the class-collapse
    /// amendment): a nested guard is an ordinary op inside a region —
    /// region lengths count it and its regions, the aux wire encoding
    /// is unchanged, the walk keeps a skip stack, the emitter recurses.
    guard_depth: u32,
    /// Open VALUE-PRODUCING regions ([`Self::open_guard`] /
    /// [`Self::open_peel`] with output shapes). A launch recorded while
    /// this is non-zero is a LOWERING of the enclosing construct's
    /// output — it binds that buffer and records no SSA output of its
    /// own — and a launch recorded at zero produces its own value.
    ///
    /// That is a property of WHERE THE STATEMENT IS, not of the kernel,
    /// and encoding it in the wrapper name is why `dsl::cuda` grew ten
    /// wrappers over five kernels (`.wiki/tart/dsl.md` ②, migration
    /// step 2). Tracking it here lets one wrapper serve both positions.
    value_region_depth: u32,
    /// V2 rung ②: the depth axis, DECLARED BY THE BODY
    /// ([`Self::declare_depth_window`]) instead of painted on after the
    /// trace (the review's smell — family.rs:64-91). While set, every
    /// layer-tagged op records its [`DepthRole`] at push time: the
    /// flashinfer decode dispatch swaps to the depth prefix plan on
    /// union tail layers, everything else windows.
    depth_axis: bool,
}

impl TraceBuilder {
    pub fn new(family: impl Into<String>) -> Self {
        Self {
            family: family.into(),
            values: Vec::new(),
            ops: Vec::new(),
            layer: None,
            seams: Vec::new(),
            guard_depth: 0,
            value_region_depth: 0,
            depth_axis: false,
        }
    }

    /// V2 rung ②: the body states the depth axis (the deployment gate
    /// lives with the statement, in the declaration text). Must precede
    /// the first layer-tagged op; the plan serializes with
    /// `depth_window` set and roles assigned exactly as the retired
    /// post-trace paint-over assigned them (the goldens pin it).
    pub fn declare_depth_window(&mut self) {
        debug_assert!(
            self.ops.iter().all(|op| op.layer.is_none()),
            "depth axis declared after layer-tagged ops were recorded"
        );
        self.depth_axis = true;
    }

    /// Bracket ops that belong to layer `l`.
    pub fn layer<T>(&mut self, l: u32, f: impl FnOnce(&mut Self) -> T) -> T {
        let previous = self.layer.replace(l);
        let out = f(self);
        self.layer = previous;
        out
    }

    /// The dsl surface's per-op layer tag (`model-dsl` derives it from
    /// the handle an op touches rather than from this bracket).
    pub fn set_layer(&mut self, layer: Option<u32>) {
        self.layer = layer;
    }

    /// A value's shape, for dsl ops whose outputs mirror their inputs.
    pub fn value_shape(&self, id: ValueId) -> Shape {
        self.values[id as usize].shape.clone()
    }

    /// Open a [`OpKind::Guard`] chain: records the op with empty arms
    /// (and its output values, if any — created HERE so dataflow sees
    /// one producer whichever arm runs) and returns its index for
    /// [`Self::close_guard`] to patch once the dsl has run every region
    /// closure. Guards may NEST (A1): the inner guard op and its
    /// regions are contiguous ops inside the enclosing region, so the
    /// enclosing arm's length simply counts them.
    pub fn open_guard(&mut self, out_shapes: Vec<(Shape, DType)>) -> (usize, Vec<ValueId>) {
        self.guard_depth += 1;
        if !out_shapes.is_empty() {
            self.value_region_depth += 1;
        }
        let outs = self.push(
            OpKind::Guard {
                arms: Vec::new(),
                else_ops: 0,
            },
            vec![],
            out_shapes,
        );
        (self.ops.len() - 1, outs)
    }

    pub fn op_count_now(&self) -> usize {
        self.ops.len()
    }

    /// Is the statement being recorded a LOWERING of an enclosing
    /// construct's output rather than a producer of its own value?
    pub fn inside_value_region(&self) -> bool {
        self.value_region_depth > 0
    }

    /// Open an [`OpKind::Peel`]: records the op with empty region
    /// lengths (and its output values — created here so dataflow sees
    /// one producer, jointly lowered by both regions) and returns its
    /// index for [`Self::close_peel`]. Region ops follow consecutively,
    /// prefix first; guards may nest inside either region.
    pub fn open_peel(
        &mut self,
        out_shapes: Vec<(Shape, DType)>,
        window: PeelWindow,
    ) -> (usize, Vec<ValueId>) {
        let outs = self.push(
            OpKind::Peel {
                prefix_ops: 0,
                tail_ops: 0,
                window,
            },
            vec![],
            out_shapes,
        );
        if !outs.is_empty() {
            self.value_region_depth += 1;
        }
        (self.ops.len() - 1, outs)
    }

    pub fn close_peel(&mut self, peel_idx: usize, prefix: u32, tail: u32) {
        let OpKind::Peel {
            prefix_ops,
            tail_ops,
            ..
        } = &mut self.ops[peel_idx].kind
        else {
            panic!("close_peel: not a peel at {peel_idx}");
        };
        *prefix_ops = prefix;
        *tail_ops = tail;
        if !self.ops[peel_idx].outputs.is_empty() {
            self.value_region_depth -= 1;
        }
    }

    /// Patch a peel's AXIS after its arms have run — the axis is a
    /// consequence of the arm's row predicate (`model_dsl::RowPred`),
    /// which is only known once the arm is written.
    pub fn set_peel_window(&mut self, peel_idx: usize, w: PeelWindow) {
        let OpKind::Peel { window, .. } = &mut self.ops[peel_idx].kind else {
            panic!("set_peel_window: not a peel at {peel_idx}");
        };
        *window = w;
    }

    pub fn push_hook_site(&mut self, stage: HookStage, layer: u32, q: ValueId) {
        self.push(OpKind::HookSite { stage, layer }, vec![q], vec![]);
    }

    /// Record that the text stated a seam, with the index of the op
    /// carrying it when one does.
    pub fn push_seam(
        &mut self,
        seam: &str,
        layer: Option<u32>,
        op: Option<u32>,
        values: Vec<ValueId>,
    ) {
        self.seams.push(SeamStatement {
            seam: seam.to_string(),
            layer,
            op,
            values,
        });
    }

    pub fn close_guard(&mut self, guard_idx: usize, arms: Vec<GuardArm>, else_ops: u32) {
        let OpKind::Guard {
            arms: a,
            else_ops: e,
        } = &mut self.ops[guard_idx].kind
        else {
            panic!("close_guard: not a guard at {guard_idx}");
        };
        *a = arms;
        *e = else_ops;
        assert!(self.guard_depth > 0, "close_guard without open_guard");
        self.guard_depth -= 1;
        if !self.ops[guard_idx].outputs.is_empty() {
            self.value_region_depth -= 1;
        }
    }

    /// The `+=` fold (`model-dsl`): if `rhs` is the output of the op
    /// just recorded and that op is a plain unfused matmul, rewrite it to
    /// the `beta_one` accumulate against `residual` — id-neutral, the
    /// same op [`Self::matmul_add`] records directly. Returns false when
    /// the shape doesn't hold (rhs older than the last op, or the last op
    /// is not a plain matmul), in which case the caller lands the
    /// residual explicitly.
    pub fn try_fold_residual(&mut self, rhs: ValueId, residual: ValueId) -> bool {
        let Some(op) = self.ops.last_mut() else {
            return false;
        };
        let foldable = matches!(
            &op.kind,
            OpKind::Matmul {
                beta_one: false,
                selector: None,
                ..
            }
        ) && op.outputs == [rhs];
        if !foldable {
            return false;
        }
        let OpKind::Matmul { beta_one, .. } = &mut op.kind else {
            unreachable!("matched above");
        };
        *beta_one = true;
        op.inputs.push(residual);
        true
    }

    fn value(&mut self, shape: Shape, dtype: DType) -> ValueId {
        self.values.push(ValueInfo {
            shape,
            dtype,
            dyn_axis: None,
        });
        (self.values.len() - 1) as ValueId
    }

    /// Declare a fragment parameter: a value no op of this trace produces.
    ///
    /// Full-model traces never need this — `embed` starts them — but a
    /// traced *fragment* (`family::qwen3_5_moe_mlp_block`) takes the
    /// residual stream it lands on as a parameter, and stating that as a
    /// producer-less value keeps the dataflow honest: the composing
    /// declaration substitutes its own value where the fragment reads the
    /// parameter.
    pub fn input(&mut self, shape: Shape, dtype: DType) -> ValueId {
        self.value(shape, dtype)
    }

    fn push(
        &mut self,
        kind: OpKind,
        inputs: Vec<ValueId>,
        out_shapes: Vec<(Shape, DType)>,
    ) -> Vec<ValueId> {
        let outputs: Vec<ValueId> = out_shapes
            .into_iter()
            .map(|(shape, dtype)| self.value(shape, dtype))
            .collect();
        self.ops.push(Op {
            kind,
            inputs,
            outputs: outputs.clone(),
            layer: self.layer,
        });
        outputs
    }

    pub fn embed(&mut self, weight: &str, hidden: u32) -> ValueId {
        self.push(
            OpKind::Embed {
                weight: weight.to_string(),
            },
            vec![],
            vec![(Shape(vec![Dim::Tokens, Dim::Const(hidden)]), DType::BF16)],
        )[0]
    }

    pub fn matmul(&mut self, x: ValueId, weight: &str, out_width: u32) -> ValueId {
        self.matmul_inner(x, weight, out_width, false)
    }

    /// The residual-accumulate form: `out += x @ w^T` where `out` is the
    /// residual stream. Returns the (new SSA id of the) accumulated value.
    pub fn matmul_add(
        &mut self,
        x: ValueId,
        weight: &str,
        residual: ValueId,
        out_width: u32,
    ) -> ValueId {
        let out = self.matmul_inner(x, weight, out_width, true);
        // The residual is an input of the accumulate — record it so the
        // dataflow is honest even though the lowering is one GEMM.
        self.ops
            .last_mut()
            .expect("matmul_inner pushed")
            .inputs
            .push(residual);
        out
    }

    fn matmul_inner(
        &mut self,
        x: ValueId,
        weight: &str,
        out_width: u32,
        beta_one: bool,
    ) -> ValueId {
        let rows = self.values[x as usize].shape.0[0];
        self.push(
            OpKind::Matmul {
                weight: weight.to_string(),
                beta_one,
                selector: None,
            },
            vec![x],
            vec![(Shape(vec![rows, Dim::Const(out_width)]), DType::BF16)],
        )[0]
    }

    /// The expert-indexed matmul: `weight_template` names a weight bank
    /// (`layer.0.expert.{e}.gate_up`) and `selector` — a [`Self::topk`]
    /// index value, `[Tokens, k]` — resolves `{e}` per token. Each token
    /// row is multiplied against its k selected experts' weights, so the
    /// result is `[Tokens, k, out_width]` (the driver's route-expanded
    /// `[N*K, out]` scratch, kept factored because k is a load-time
    /// constant and Tokens is not). One op = one launch: the grouped
    /// gate_up/down GEMM of the hand-written MoE pass, whatever strategy
    /// (cuBLAS batched, aligned blocks, CUTLASS fused) the emitter picks.
    pub fn matmul_per_token(
        &mut self,
        x: ValueId,
        weight_template: &str,
        selector: ValueId,
        out_width: u32,
    ) -> ValueId {
        assert!(
            weight_template.contains("{e}"),
            "per-token matmul weight must be a template with an {{e}} slot, got {weight_template:?}"
        );
        assert_eq!(
            self.values[selector as usize].dyn_axis,
            Some(DynAxis::PerToken),
            "per-token matmul selector must be a dyn PerToken value"
        );
        let rows = self.values[x as usize].shape.0[0];
        let k = self.values[selector as usize].shape.0[1];
        self.push(
            OpKind::Matmul {
                weight: weight_template.to_string(),
                beta_one: false,
                selector: Some(selector),
            },
            // The selector is an input too — its content is consumed — and
            // by convention the last one, like matmul_add's residual.
            vec![x, selector],
            vec![(Shape(vec![rows, k, Dim::Const(out_width)]), DType::BF16)],
        )[0]
    }

    pub fn rmsnorm(&mut self, x: ValueId, weight: &str, variant: NormVariant) -> ValueId {
        let shape = self.values[x as usize].shape.clone();
        self.push(
            OpKind::Rmsnorm {
                weight: weight.to_string(),
                variant,
            },
            vec![x],
            vec![(shape, DType::BF16)],
        )[0]
    }

    pub fn add_bias(&mut self, x: ValueId, weight: &str) -> ValueId {
        let shape = self.values[x as usize].shape.clone();
        self.push(
            OpKind::AddBias {
                weight: weight.to_string(),
            },
            vec![x],
            vec![(shape, DType::BF16)],
        )[0]
    }

    pub fn rmsnorm_per_head(
        &mut self,
        x: ValueId,
        weight: &str,
        head_dim: u32,
        variant: NormVariant,
    ) -> ValueId {
        let shape = self.values[x as usize].shape.clone();
        self.push(
            OpKind::RmsnormPerHead {
                weight: weight.to_string(),
                head_dim,
                variant,
            },
            vec![x],
            vec![(shape, DType::BF16)],
        )[0]
    }

    pub fn split_qkv(
        &mut self,
        packed: ValueId,
        q_width: u32,
        kv_width: u32,
    ) -> (ValueId, ValueId, ValueId) {
        let rows = self.values[packed as usize].shape.0[0];
        let out = self.push(
            OpKind::SplitQkv { q_width, kv_width },
            vec![packed],
            vec![
                (Shape(vec![rows, Dim::Const(q_width)]), DType::BF16),
                (Shape(vec![rows, Dim::Const(kv_width)]), DType::BF16),
                (Shape(vec![rows, Dim::Const(kv_width)]), DType::BF16),
            ],
        );
        (out[0], out[1], out[2])
    }

    /// Rope mutates Q and K in place; SSA-wise it produces two new values.
    pub fn rope(&mut self, q: ValueId, k: ValueId, kind: RopeKind) -> (ValueId, ValueId) {
        self.rope_inner(q, k, kind, None)
    }

    /// The partial-rotary form: only the first `rotary_dim` channels of
    /// each head rotate (`kernels::rope::rope_partial_bf16`; qwen3.5's
    /// `partial_rotary_factor` resolved to a channel count — see
    /// [`OpKind::Rope`]).
    pub fn rope_partial(
        &mut self,
        q: ValueId,
        k: ValueId,
        kind: RopeKind,
        rotary_dim: u32,
    ) -> (ValueId, ValueId) {
        self.rope_inner(q, k, kind, Some(rotary_dim))
    }

    fn rope_inner(
        &mut self,
        q: ValueId,
        k: ValueId,
        kind: RopeKind,
        partial: Option<u32>,
    ) -> (ValueId, ValueId) {
        let q_shape = self.values[q as usize].shape.clone();
        let k_shape = self.values[k as usize].shape.clone();
        let out = self.push(
            OpKind::Rope { kind, partial },
            vec![q, k],
            vec![(q_shape, DType::BF16), (k_shape, DType::BF16)],
        );
        (out[0], out[1])
    }

    pub fn kv_append(&mut self, layer: u32, k: ValueId, v: ValueId) {
        self.push(OpKind::KvAppend { layer }, vec![k, v], vec![]);
    }

    pub fn attention(&mut self, layer: u32, q: ValueId, q_width: u32) -> ValueId {
        self.push(
            OpKind::Attention { layer },
            vec![q],
            vec![(Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)],
        )[0]
    }

    /// A STATED kernel launch ([`OpKind::Launch`]) — the recording half of
    /// the raw kernel signatures in `model_dsl::cuda`; declarations
    /// call those, never this.
    pub fn launch(
        &mut self,
        kernel: &str,
        weights: Vec<String>,
        state: Option<StateRef>,
        inputs: Vec<ValueId>,
        out_shapes: Vec<(Shape, DType)>,
    ) -> Vec<ValueId> {
        self.launch_with_params(kernel, weights, state, Vec::new(), inputs, out_shapes)
    }

    /// [`Self::launch`], plus the scalar arguments the symbol takes
    /// that no operand shape gives — see [`OpKind::Launch::params`].
    ///
    /// A separate entry point rather than a sixth argument on `launch`,
    /// because the overwhelming majority of statements have no such
    /// scalar and a `Vec::new()` at every one of them would say
    /// nothing.
    pub fn launch_with_params(
        &mut self,
        kernel: &str,
        weights: Vec<String>,
        state: Option<StateRef>,
        params: Vec<u32>,
        inputs: Vec<ValueId>,
        out_shapes: Vec<(Shape, DType)>,
    ) -> Vec<ValueId> {
        self.launch_with_extents(
            kernel,
            weights,
            state,
            params,
            Vec::new(),
            inputs,
            out_shapes,
        )
    }

    /// [`Self::launch_with_params`], plus the scalars whose value is an
    /// extent the FIRE decides — see [`OpKind::Launch::param_extents`].
    ///
    /// The constants left in `params` at those indices are read by nothing
    /// once a lowering has run, and they are written as zero for that
    /// reason: a plausible-looking constant beside a channel that
    /// overwrites it is the state this argument exists to end.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_with_extents(
        &mut self,
        kernel: &str,
        weights: Vec<String>,
        state: Option<StateRef>,
        params: Vec<u32>,
        param_extents: Vec<(u8, Shape)>,
        inputs: Vec<ValueId>,
        out_shapes: Vec<(Shape, DType)>,
    ) -> Vec<ValueId> {
        self.push(
            OpKind::Launch {
                kernel: kernel.to_string(),
                weights,
                state,
                params,
                param_extents,
            },
            inputs,
            out_shapes,
        )
    }

    /// SwiGLU halves the trailing gate‖up dim and keeps every leading dim,
    /// so it covers both the dense `[Tokens, 2*inter]` activation and the
    /// route-expanded `[Tokens, k, 2*inter]` one (the driver's
    /// `chunked_swiglu` over `N*K` rows).
    pub fn swiglu(&mut self, packed: ValueId, inter: u32) -> ValueId {
        let mut shape = self.values[packed as usize].shape.clone();
        *shape.0.last_mut().expect("swiglu input has a trailing dim") = Dim::Const(inter);
        self.push(
            OpKind::Swiglu { inter },
            vec![packed],
            vec![(shape, DType::BF16)],
        )[0]
    }

    /// Router top-k: `(indices, weights)`, both `[Tokens, k]`. The indices
    /// are the trace's first `dyn` value ([`DynAxis::PerToken`]); the
    /// weights are already softmaxed and renormalized, because the launch
    /// this op mirrors (`kernels::moe::topk_softmax_bf16`) does all three.
    pub fn topk(&mut self, logits: ValueId, k: u32) -> (ValueId, ValueId) {
        let rows = self.values[logits as usize].shape.0[0];
        let out = self.push(
            OpKind::TopK { k },
            vec![logits],
            vec![
                (Shape(vec![rows, Dim::Const(k)]), DType::I32),
                (Shape(vec![rows, Dim::Const(k)]), DType::F32),
            ],
        );
        self.values[out[0] as usize].dyn_axis = Some(DynAxis::PerToken);
        (out[0], out[1])
    }

    /// The top-k combine: collapse `x` (`[Tokens, k, d]`) to `[Tokens, d]`
    /// under per-token `weights` (`[Tokens, k]`). Operand order: weights,
    /// then the value they weight.
    pub fn weighted_sum(&mut self, weights: ValueId, x: ValueId) -> ValueId {
        let x_shape = &self.values[x as usize].shape.0;
        let (rows, d) = (x_shape[0], x_shape[2]);
        let k = match self.values[weights as usize].shape.0[1] {
            Dim::Const(k) => k,
            other => panic!("weighted_sum weights must have a Const k dim, got {other:?}"),
        };
        self.push(
            OpKind::WeightedSum { k },
            vec![weights, x],
            vec![(Shape(vec![rows, d]), DType::BF16)],
        )[0]
    }

    /// The shared-expert landing: `base + sigmoid(gate) * x`. Operand
    /// order mirrors [`Self::residual_add`] — the fresh value first, the
    /// stream it lands on last — and the result is the (new SSA id of the)
    /// combined value.
    pub fn sigmoid_gate_add(&mut self, x: ValueId, gate: ValueId, base: ValueId) -> ValueId {
        let shape = self.values[base as usize].shape.clone();
        self.push(
            OpKind::SigmoidGateAdd,
            vec![x, gate, base],
            vec![(shape, DType::BF16)],
        )[0]
    }

    /// The two-way GDN split: packed `[rows, w0 + w1]` into `[rows, w0]`
    /// and `[rows, w1]` at `w0`.
    pub fn split_gdn(&mut self, packed: ValueId, width0: u32, width1: u32) -> (ValueId, ValueId) {
        let rows = self.values[packed as usize].shape.0[0];
        let out = self.push(
            OpKind::SplitGdn { width0, width1 },
            vec![packed],
            vec![
                (Shape(vec![rows, Dim::Const(width0)]), DType::BF16),
                (Shape(vec![rows, Dim::Const(width1)]), DType::BF16),
            ],
        );
        (out[0], out[1])
    }

    /// The interleaved per-head `[query | gate]` split of a 2×-wide gated
    /// q projection: packed `[rows, heads * 2 * head_dim]` into (q, gate),
    /// each `[rows, heads * head_dim]`. See [`OpKind::SplitQGate`] for why
    /// this is not a [`Self::split_gdn`] row split.
    pub fn split_q_gate(
        &mut self,
        packed: ValueId,
        heads: u32,
        head_dim: u32,
    ) -> (ValueId, ValueId) {
        let rows = self.values[packed as usize].shape.0[0];
        match self.values[packed as usize].shape.0[1] {
            Dim::Const(w) if w == 2 * heads * head_dim => {}
            other => panic!("split_q_gate input width {other:?} must be 2 * {heads} * {head_dim}"),
        }
        let half = Shape(vec![rows, Dim::Const(heads * head_dim)]);
        let out = self.push(
            OpKind::SplitQGate { heads, head_dim },
            vec![packed],
            vec![(half.clone(), DType::BF16), (half, DType::BF16)],
        );
        (out[0], out[1])
    }

    /// The multiply-only output gate: `out = x * sigmoid(gate)`, both
    /// operands the same shape ([`OpKind::SigmoidGateMul`] — no residual,
    /// unlike [`Self::sigmoid_gate_add`]).
    pub fn sigmoid_gate_mul(&mut self, x: ValueId, gate: ValueId) -> ValueId {
        let shape = self.values[x as usize].shape.clone();
        assert_eq!(
            shape, self.values[gate as usize].shape,
            "sigmoid_gate_mul operands must share a shape"
        );
        self.push(
            OpKind::SigmoidGateMul,
            vec![x, gate],
            vec![(shape, DType::BF16)],
        )[0]
    }

    /// Depthwise causal conv1d (+ fused SiLU) over the packed qkv, against
    /// layer `layer`'s per-request conv state. Shape-preserving.
    pub fn causal_conv1d(
        &mut self,
        layer: u32,
        qkv: ValueId,
        weight: &str,
        kernel: u32,
    ) -> ValueId {
        let shape = self.values[qkv as usize].shape.clone();
        self.push(
            OpKind::CausalConv1d {
                weight: weight.to_string(),
                layer,
                kernel,
            },
            vec![qkv],
            vec![(shape, DType::BF16)],
        )[0]
    }

    /// The post-conv GDN prep: `(q, k, v, g, beta)`, all f32, with q/k in
    /// the compact `[Tokens, key_heads, key_dim]` per-head layout and v in
    /// `[Tokens, value_heads, value_dim]`. Operand order `[qkv, a, b]` is
    /// the kernel's.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_prep(
        &mut self,
        qkv: ValueId,
        a: ValueId,
        b: ValueId,
        a_log: &str,
        dt_bias: &str,
        key_heads: u32,
        key_dim: u32,
        value_heads: u32,
        value_dim: u32,
    ) -> (ValueId, ValueId, ValueId, ValueId, ValueId) {
        let rows = self.values[qkv as usize].shape.0[0];
        let qk = Shape(vec![rows, Dim::Const(key_heads), Dim::Const(key_dim)]);
        let out = self.push(
            OpKind::GdnPrep {
                a_log: a_log.to_string(),
                dt_bias: dt_bias.to_string(),
            },
            vec![qkv, a, b],
            vec![
                (qk.clone(), DType::F32),
                (qk, DType::F32),
                (
                    Shape(vec![rows, Dim::Const(value_heads), Dim::Const(value_dim)]),
                    DType::F32,
                ),
                (Shape(vec![rows, Dim::Const(value_heads)]), DType::F32),
                (Shape(vec![rows, Dim::Const(value_heads)]), DType::F32),
            ],
        );
        (out[0], out[1], out[2], out[3], out[4])
    }

    /// The gated-delta recurrence against layer `layer`'s per-request
    /// recurrent state. The core output keeps v's `[Tokens, Vh, Vd]` shape.
    pub fn gated_delta(
        &mut self,
        layer: u32,
        q: ValueId,
        k: ValueId,
        v: ValueId,
        g: ValueId,
        beta: ValueId,
    ) -> ValueId {
        let shape = self.values[v as usize].shape.clone();
        self.push(
            OpKind::GatedDelta { layer },
            vec![q, k, v, g, beta],
            vec![(shape, DType::F32)],
        )[0]
    }

    /// The gated RMSNorm landing: per-head norm of the rank-3 f32 core
    /// output, silu-gated by `gate`, flattened to `gate`'s `[Tokens,
    /// Vh * Vd]` bf16 shape (the fused fp32→bf16 conversion).
    pub fn rmsnorm_gated(&mut self, x: ValueId, gate: ValueId, weight: &str) -> ValueId {
        let x_elems: u32 = self.values[x as usize].shape.0[1..]
            .iter()
            .map(|d| match d {
                Dim::Const(c) => *c,
                other => panic!("rmsnorm_gated x must have Const head dims, got {other:?}"),
            })
            .product();
        let gate_shape = self.values[gate as usize].shape.clone();
        match gate_shape.0[1] {
            Dim::Const(w) if w == x_elems => {}
            other => {
                panic!("rmsnorm_gated gate width {other:?} must equal x's flattened {x_elems}")
            }
        }
        self.push(
            OpKind::RmsnormGated {
                weight: weight.to_string(),
            },
            vec![x, gate],
            vec![(gate_shape, DType::BF16)],
        )[0]
    }

    /// The post-norm residual landing: `residual += x`. Operand order
    /// mirrors [`Self::matmul_add`] — the freshly computed value first,
    /// the residual stream it lands on appended — and the result is the
    /// (new SSA id of the) accumulated stream.
    pub fn residual_add(&mut self, x: ValueId, residual: ValueId) -> ValueId {
        let shape = self.values[residual as usize].shape.clone();
        self.push(
            OpKind::ResidualAdd,
            vec![x, residual],
            vec![(shape, DType::BF16)],
        )[0]
    }

    /// [`OpKind::Select`]: the window of `x` along its leading dim.
    ///
    /// The output shape is the input's minus that dim — computed here
    /// rather than passed, because it is not a choice: a window of
    /// `[k, Tokens, hidden]` at an index IS `[Tokens, hidden]`, and a
    /// caller free to say otherwise could state a value the buffer
    /// arithmetic would then disagree with.
    pub fn select(&mut self, x: ValueId, index: u32) -> ValueId {
        let shape = self.value_shape(x);
        assert!(
            shape.0.len() >= 2,
            "select: a rank-{} value has no leading dim to window",
            shape.0.len()
        );
        if let Dim::Const(k) = shape.0[0] {
            assert!(
                index < k,
                "select: index {index} is outside the leading dim's {k}"
            );
        }
        let dtype = self.values[x as usize].dtype;
        let inner = Shape(shape.0[1..].to_vec());
        self.push(OpKind::Select { index }, vec![x], vec![(inner, dtype)])[0]
    }

    pub fn lm_head(&mut self, hidden: ValueId, weight: &str, vocab: u32) -> ValueId {
        self.push(
            OpKind::LmHead {
                weight: weight.to_string(),
            },
            vec![hidden],
            vec![(Shape(vec![Dim::Requests, Dim::Const(vocab)]), DType::F32)],
        )[0]
    }

    pub fn finish(self) -> ForwardPlan {
        let plan = ForwardPlan {
            family: self.family,
            values: self.values,
            ops: self.ops,
            depth_window: self.depth_axis,
            seams: self.seams,
        };
        // ② The kernel signatures, checked (`.wiki/tart/dsl.md` ②,
        // migration step 3). A declaration is traced when the model
        // LOADS, so this is the load-time check the design asks for:
        // `whole` and the table's own coverage stop being rules a
        // reader has to know and become rules a build cannot violate.
        let mut problems = crate::kernels::check_plan(&plan);
        problems.extend(crate::seam::check_plan(&plan));
        assert!(
            problems.is_empty(),
            "signature violations in this declaration:\n  {}",
            problems.join("\n  ")
        );
        plan
    }
}
