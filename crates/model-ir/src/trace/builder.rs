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
    /// Runtime values this trace names, in mint order ([`RuntimeBinding`]).
    runtime: Vec<RuntimeBinding>,
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
            runtime: Vec::new(),
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

    /// The family this trace is being built for — what tier-1 resolution
    /// reads the backend off.
    pub fn family(&self) -> &str {
        &self.family
    }

    /// One value's dtype, for a statement whose result mirrors its operand.
    pub fn value_dtype(&self, id: ValueId) -> DType {
        self.values[id as usize].dtype
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

    /// `residual += rhs` folded into the producing GEMM when the last op is a
    /// `from` launch producing `rhs`: the kernel swaps to `to` (the beta-one
    /// symbol) and takes the residual as an extra operand. The generalised
    /// form of what `OpKind::Matmul::beta_one` said before the retirement.
    pub fn try_fold_beta(
        &mut self,
        rhs: ValueId,
        residual: ValueId,
        from: &str,
        to: &str,
    ) -> bool {
        if self.value_region_depth > 0 {
            return false;
        }
        let Some(op) = self.ops.last_mut() else {
            return false;
        };
        if op.outputs != vec![rhs] {
            return false;
        }
        let OpKind::Launch { kernel, .. } = &mut op.kind else {
            return false;
        };
        if kernel != from {
            return false;
        }
        *kernel = to.to_string();
        op.inputs.push(residual);
        true
    }

    /// Mint the value of one runtime OBJECT — a driver-owned view out of
    /// `kernels::runtime`'s vocabulary (`"kv_cache"`, `"recurrent_state"`).
    ///
    /// Raise-shaped: no rectangle, no dtype, resolved by NAME at bind. The
    /// statements that read it take it as an operand, positionally, exactly
    /// as they take what a [`Self::push_prep`] raised — this is the resident
    /// half of the same channel.
    pub fn runtime_object(&mut self, name: &str, layer: Option<u32>) -> ValueId {
        if let Some(b) = self.runtime.iter().find(|b| b.name == name && b.layer == layer) {
            return b.value;
        }
        let id = self.values.len() as ValueId;
        self.values.push(ValueInfo::raise(name));
        self.runtime.push(RuntimeBinding { name: name.to_string(), layer, value: id });
        id
    }

    /// Mint one runtime TENSOR — a per-fire stream the driver stages
    /// (`"positions"`, `"qo_indptr"`). A real tensor with rows and a dtype;
    /// the lowering leaves it backend-bound (`Buffers::NAMED`) and the
    /// driver answers the name.
    pub fn runtime_tensor(
        &mut self,
        name: &str,
        layer: Option<u32>,
        shape: Shape,
        dtype: DType,
    ) -> ValueId {
        if let Some(b) = self.runtime.iter().find(|b| b.name == name && b.layer == layer) {
            return b.value;
        }
        let id = self.value(shape, dtype);
        self.runtime.push(RuntimeBinding { name: name.to_string(), layer, value: id });
        id
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
        self.launch_devwin(
            kernel,
            weights,
            state,
            params,
            param_extents,
            None,
            inputs,
            out_shapes,
        )
    }

    /// [`Self::launch_with_extents`], plus the peel-window slot pair the
    /// lowering fills with this launch's own rectangle. See
    /// [`OpKind::Launch::peel_slots`].
    #[allow(clippy::too_many_arguments)]
    pub fn launch_devwin(
        &mut self,
        kernel: &str,
        weights: Vec<String>,
        state: Option<StateRef>,
        params: Vec<u32>,
        param_extents: Vec<(u8, Shape)>,
        peel_slots: Option<(u8, u8)>,
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
                peel_slots,
            },
            inputs,
            out_shapes,
        )
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
            runtime: self.runtime,
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
