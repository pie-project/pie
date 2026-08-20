//! Trace recorder: what a declaration calls, in computation order.

use super::*;

/// Records ops as a declaration executes; ids are assigned here and
/// structure such as layers stays on the ops.
pub struct TraceBuilder {
    family: String,
    values: Vec<ValueInfo>,
    ops: Vec<Op>,
    layer: Option<u32>,
    /// Seam statements in text order ([`SeamStatement`]).
    seams: Vec<SeamStatement>,
    /// Open [`Self::open_guard`] depth; nested guards are ordinary ops
    /// inside a region and count toward its length.
    guard_depth: u32,
    /// Open value-producing regions. Launches inside one write the
    /// region's buffer and record no SSA output of their own.
    value_region_depth: u32,
    /// Outputs owned by enclosing value regions, innermost last.
    region_dests: Vec<Vec<ValueId>>,
    /// Whether layer-tagged ops participate in the depth-window axis.
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
            region_dests: Vec::new(),
            depth_axis: false,
        }
    }

    /// Declare the depth axis before the first layer-tagged op.
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

    pub fn set_layer(&mut self, layer: Option<u32>) {
        self.layer = layer;
    }

    /// The logical shape of a TENSOR value.
    ///
    /// # Panics
    ///
    /// When `id` is a raise. A raise has no rectangle, and the empty shape
    /// stored beside it would flow into a caller as a zero-width tensor rather
    /// than as the mistake it is — so this refuses where the field cannot.
    /// `close_guard` panics for the same class of reason: a malformed trace is
    /// the builder's caller's bug, not a fire's.
    pub fn value_shape(&self, id: ValueId) -> Shape {
        let v = &self.values[id as usize];
        assert!(
            !v.is_raised(),
            "value_shape on value {id}, which is the raise {:?} and not a tensor",
            v.raised.as_deref().unwrap_or_default()
        );
        v.shape.clone()
    }

    /// Whether `id` is a raise rather than a tensor. See [`ValueInfo::raised`].
    pub fn is_raised(&self, id: ValueId) -> bool {
        self.values[id as usize].is_raised()
    }

    /// Open a [`OpKind::Guard`] chain and return the op index that
    /// [`Self::close_guard`] patches after all regions are recorded.
    pub fn open_guard(&mut self, out_shapes: Vec<(Shape, DType)>) -> (usize, Vec<ValueId>) {
        self.guard_depth += 1;
        let valued = !out_shapes.is_empty();
        if valued {
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
        // The Guard op produces the destination; it does not write into it.
        if valued {
            self.region_dests.push(outs.clone());
        }
        (self.ops.len() - 1, outs)
    }

    pub fn op_count_now(&self) -> usize {
        self.ops.len()
    }

    pub fn inside_value_region(&self) -> bool {
        self.value_region_depth > 0
    }

    /// Open an [`OpKind::Peel`]; prefix ops are recorded before tail ops.
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
            // Prefix and tail both lower into the same peel output buffer.
            self.region_dests.push(outs.clone());
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
            self.region_dests.pop();
        }
    }

    /// Patch a peel's axis after the row predicate is known.
    pub fn set_peel_window(&mut self, peel_idx: usize, w: PeelWindow) {
        let OpKind::Peel { window, .. } = &mut self.ops[peel_idx].kind else {
            panic!("set_peel_window: not a peel at {peel_idx}");
        };
        *window = w;
    }

    pub fn push_hook_site(&mut self, stage: HookStage, layer: u32, q: ValueId) {
        self.push(OpKind::HookSite { stage, layer }, vec![q], vec![]);
    }

    /// Record that this fire needs a host-side preparation raised, and return
    /// the value it publishes.
    ///
    /// No inputs: what a prep reads is the fire's own page geometry, which is
    /// not an SSA value. It is stated once per fire rather than per layer — a
    /// schedule is a property of the batch.
    ///
    /// **One output, and it used to have none.** What the prep publishes was
    /// held by the backend under a name and recovered by the statements that
    /// execute it — from a family string, and for decode from a guess. The
    /// value is that edge, written down. See [`OpKind::Prep`].
    ///
    /// Built here rather than through `push`, which mints TENSOR outputs from
    /// shapes. The `dest` is empty for the reason `push` would also leave it
    /// empty: an op that produces a value of its own does not write an
    /// enclosing region's buffer.
    pub fn push_prep(&mut self, prep: PrepKind) -> ValueId {
        let raised = self.values.len() as ValueId;
        self.values.push(ValueInfo::raise(prep.key()));
        self.ops.push(Op {
            kind: OpKind::Prep { prep },
            inputs: Vec::new(),
            outputs: vec![raised],
            layer: self.layer,
            dest: Vec::new(),
        });
        raised
    }

    /// Record that the text stated a seam.
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
            self.region_dests.pop();
        }
    }

    /// Fold `rhs += residual` into the just-recorded plain matmul when
    /// that matmul produced `rhs`.
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

    /// Mint one TENSOR value. A raise is minted by [`Self::push_prep`], which
    /// is the only op that publishes one.
    fn value(&mut self, shape: Shape, dtype: DType) -> ValueId {
        self.values.push(ValueInfo {
            shape,
            dtype,
            dyn_axis: None,
            raised: None,
        });
        (self.values.len() - 1) as ValueId
    }

    /// Declare a fragment parameter: a value no op of this trace produces.
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
        // Inside a value region, empty-output statements write the enclosing
        // destination instead of producing a new SSA value. Region-opening ops
        // call `push` before growing the stack, so they are not inside it.
        let dest = if outputs.is_empty() {
            self.region_dests.last().cloned().unwrap_or_default()
        } else {
            Vec::new()
        };
        self.ops.push(Op {
            kind,
            inputs,
            outputs: outputs.clone(),
            layer: self.layer,
            dest,
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

    /// Residual-accumulate matmul: `out += x @ w^T`.
    pub fn matmul_add(
        &mut self,
        x: ValueId,
        weight: &str,
        residual: ValueId,
        out_width: u32,
    ) -> ValueId {
        let out = self.matmul_inner(x, weight, out_width, true);
        // The residual is an input even though lowering is one GEMM.
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

    /// Expert-indexed matmul. `selector` is a [`Self::topk`] value
    /// `[Tokens, k]` choosing `{e}` per token; output is
    /// `[Tokens, k, out_width]`.
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
            // Selector content is consumed, and is the last input by convention.
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

    /// Partial rotary: only the first `rotary_dim` channels of each head rotate.
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

    /// Record a stated kernel launch ([`OpKind::Launch`]).
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

    /// [`Self::launch`], plus scalar parameters not derivable from operand shapes.
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

    /// [`Self::launch_with_params`], plus fire-decided scalar extents.
    /// Constants at those `params` indices are placeholders and written as zero.
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

    /// Halve the trailing gate‖up dim; leading dims are preserved.
    pub fn swiglu(&mut self, packed: ValueId, inter: u32) -> ValueId {
        let mut shape = self.values[packed as usize].shape.clone();
        *shape.0.last_mut().expect("swiglu input has a trailing dim") = Dim::Const(inter);
        self.push(
            OpKind::Swiglu { inter },
            vec![packed],
            vec![(shape, DType::BF16)],
        )[0]
    }

    /// Router top-k returns `(indices, weights)`, both `[Tokens, k]`;
    /// indices are [`DynAxis::PerToken`].
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

    /// Collapse `x` `[Tokens, k, d]` with weights `[Tokens, k]`.
    /// Operand order: weights, then values.
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

    /// Shared-expert landing. Operand order: fresh value, gate, base stream.
    pub fn sigmoid_gate_add(&mut self, x: ValueId, gate: ValueId, base: ValueId) -> ValueId {
        let shape = self.values[base as usize].shape.clone();
        self.push(
            OpKind::SigmoidGateAdd,
            vec![x, gate, base],
            vec![(shape, DType::BF16)],
        )[0]
    }

    /// Split packed `[rows, w0 + w1]` at `w0`.
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

    /// Interleaved per-head `[query | gate]` split; not a row split.
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

    /// Shape-preserving depthwise causal conv1d against per-request state.
    pub fn causal_conv1d(
        &mut self,
        layer: u32,
        qkv: ValueId,
        weight: &str,
        bias: Option<&str>,
        kernel: u32,
    ) -> ValueId {
        let shape = self.values[qkv as usize].shape.clone();
        self.push(
            OpKind::CausalConv1d {
                weight: weight.to_string(),
                bias: bias.map(str::to_string),
                layer,
                kernel,
            },
            vec![qkv],
            vec![(shape, DType::BF16)],
        )[0]
    }

    /// Post-conv GDN prep outputs `(q, k, v, g, beta)` in that order.
    /// Inputs are `[qkv, a, b]`; q/k are `[Tokens, key_heads, key_dim]`.
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

    /// Per-head f32 core norm, gated and flattened to gate's bf16 shape.
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

    /// Post-norm residual landing. Operand order: fresh value, residual stream.
    pub fn residual_add(&mut self, x: ValueId, residual: ValueId) -> ValueId {
        let shape = self.values[residual as usize].shape.clone();
        self.push(
            OpKind::ResidualAdd,
            vec![x, residual],
            vec![(shape, DType::BF16)],
        )[0]
    }

    /// Window `x` along its leading dim; output shape drops that dim.
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
        // Load-time signature and seam checks.
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

#[cfg(test)]
mod raises {
    use super::*;

    /// A prep publishes one value, and the op names it.
    ///
    /// The edge this design exists to create: before, `outputs` was empty and
    /// the statements that execute what a prep raised had to find it by other
    /// means. See [`OpKind::Prep`].
    #[test]
    fn a_prep_publishes_the_object_it_raises() {
        let mut b = TraceBuilder::new("fixture.cuda");
        let plan = b.push_prep(PrepKind::PrefillAttention { head_dim: 128 });

        let plan_op = b.ops.last().expect("the prep was recorded");
        assert!(matches!(plan_op.kind, OpKind::Prep { .. }));
        assert_eq!(plan_op.outputs, vec![plan], "the op names what it raised");
        assert!(plan_op.inputs.is_empty(), "a prep reads no SSA value");
        assert!(b.is_raised(plan));
    }

    /// The word comes from `raise!` and is not spelled twice.
    ///
    /// Asserted against `kernels-cuda`'s declaration rather than a literal, so
    /// a drift between the two crates fails HERE rather than at a fire, where
    /// it would surface as `Refusal::Unstated` on a key nothing answers.
    #[test]
    fn the_key_is_the_one_the_declaration_wrote() {
        use kernels::raises::Raise;

        let mut b = TraceBuilder::new("fixture.cuda");
        let prefill = b.push_prep(PrepKind::PrefillAttention { head_dim: 128 });
        let decode = b.push_prep(PrepKind::DecodeAttention {
            head_dim: 128,
            full_attention: true,
        });

        let plan = b.finish();
        assert_eq!(
            plan.values[prefill as usize].raised.as_deref(),
            Some(kernels_cuda::raises::Fa2Prefill::KEY)
        );
        assert_eq!(
            plan.values[decode as usize].raised.as_deref(),
            Some(kernels_cuda::raises::Fa2Decode::KEY)
        );
    }

    /// A raise survives the wire, which is what a trace is on disk.
    #[test]
    fn a_raise_round_trips_through_the_serialized_plan() {
        let mut b = TraceBuilder::new("fixture.cuda");
        let plan = b.push_prep(PrepKind::PrefillAttention { head_dim: 128 });
        let tensor = b.embed("tok_embeddings", 1024);

        let before = b.finish();
        let json = serde_json::to_string(&before).expect("a plan serializes");
        let after: ForwardPlan = serde_json::from_str(&json).expect("and comes back");

        assert_eq!(before, after);
        assert!(after.values[plan as usize].is_raised());
        assert_eq!(
            after.values[plan as usize].raised.as_deref(),
            Some(PrepKind::PrefillAttention { head_dim: 128 }.key())
        );
        // AND THE FIELD IS ABSENT ON A TENSOR, not `null`: `skip_serializing_if`
        // is what keeps a golden plan's every other value byte-identical to
        // what it was before this field existed.
        assert!(!after.values[tensor as usize].is_raised());
        assert!(!json.contains("\"raised\":null"));
    }

    /// `value_shape` refuses a raise rather than handing back the empty shape.
    #[test]
    #[should_panic(expected = "which is the raise")]
    fn a_raise_has_no_shape_to_ask_for() {
        let mut b = TraceBuilder::new("fixture.cuda");
        let plan = b.push_prep(PrepKind::PrefillAttention { head_dim: 128 });
        let _ = b.value_shape(plan);
    }

    /// And an ordinary value still answers, which is the other half.
    #[test]
    fn a_tensor_still_has_one() {
        let mut b = TraceBuilder::new("fixture.cuda");
        let _ = b.push_prep(PrepKind::PrefillAttention { head_dim: 128 });
        let x = b.embed("tok_embeddings", 1024);
        assert_eq!(
            b.value_shape(x),
            Shape(vec![Dim::Tokens, Dim::Const(1024)]),
            "the raise minted before it did not disturb this"
        );
    }
}
