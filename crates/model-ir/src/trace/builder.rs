use super::*;

pub struct TraceBuilder {
    family: String,
    values: Vec<ValueInfo>,
    ops: Vec<Op>,
    layer: Option<u32>,

    seams: Vec<SeamStatement>,

    guard_depth: u32,

    value_region_depth: u32,

    region_dests: Vec<Vec<ValueId>>,

    depth_axis: bool,

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

    pub fn declare_depth_window(&mut self) {
        debug_assert!(
            self.ops.iter().all(|op| op.layer.is_none()),
            "depth axis declared after layer-tagged ops were recorded"
        );
        self.depth_axis = true;
    }

    pub fn layer<T>(&mut self, l: u32, f: impl FnOnce(&mut Self) -> T) -> T {
        let previous = self.layer.replace(l);
        let out = f(self);
        self.layer = previous;
        out
    }

    pub fn set_layer(&mut self, layer: Option<u32>) {
        self.layer = layer;
    }

    pub fn value_shape(&self, id: ValueId) -> Shape {
        let v = &self.values[id as usize];
        assert!(
            !v.is_raised(),
            "value_shape on value {id}, which is the raise {:?} and not a tensor",
            v.raised.as_deref().unwrap_or_default()
        );
        v.shape.clone()
    }

    pub fn family(&self) -> &str {
        &self.family
    }

    pub fn value_dtype(&self, id: ValueId) -> DType {
        self.values[id as usize].dtype
    }

    pub fn is_raised(&self, id: ValueId) -> bool {
        self.values[id as usize].is_raised()
    }

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

    pub fn set_peel_window(&mut self, peel_idx: usize, w: PeelWindow) {
        let OpKind::Peel { window, .. } = &mut self.ops[peel_idx].kind else {
            panic!("set_peel_window: not a peel at {peel_idx}");
        };
        *window = w;
    }

    pub fn push_hook_site(&mut self, stage: HookStage, layer: u32, q: ValueId) {
        self.push(OpKind::HookSite { stage, layer }, vec![q], vec![]);
    }

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

    fn value(&mut self, shape: Shape, dtype: DType) -> ValueId {
        self.values.push(ValueInfo {
            shape,
            dtype,
            dyn_axis: None,
            raised: None,
        });
        (self.values.len() - 1) as ValueId
    }

    pub fn input(&mut self, shape: Shape, dtype: DType) -> ValueId {
        self.value(shape, dtype)
    }

    pub fn try_fold_beta(&mut self, rhs: ValueId, residual: ValueId, from: &str, to: &str) -> bool {
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

    pub fn runtime_object(&mut self, name: &str, layer: Option<u32>) -> ValueId {
        if let Some(b) = self
            .runtime
            .iter()
            .find(|b| b.name == name && b.layer == layer)
        {
            return b.value;
        }
        let id = self.values.len() as ValueId;
        self.values.push(ValueInfo::raise(name));
        self.runtime.push(RuntimeBinding {
            name: name.to_string(),
            layer,
            value: id,
        });
        id
    }

    pub fn runtime_tensor(
        &mut self,
        name: &str,
        layer: Option<u32>,
        shape: Shape,
        dtype: DType,
    ) -> ValueId {
        if let Some(b) = self
            .runtime
            .iter()
            .find(|b| b.name == name && b.layer == layer)
        {
            return b.value;
        }
        let id = self.value(shape, dtype);
        self.runtime.push(RuntimeBinding {
            name: name.to_string(),
            layer,
            value: id,
        });
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
        let sampled = self.runtime_tensor(
            "sampling_indices",
            None,
            Shape(vec![Dim::Requests]),
            DType::I32,
        );
        self.push(
            OpKind::LmHead {
                weight: weight.to_string(),
            },
            vec![hidden, sampled],
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

        // TWO TRACE-TIME CHECKS STOOD HERE — `kernels::check_plan`'s arity
        // walk (a statement's operand count against the routine's signature
        // column) and `seam::check_plan`'s ordering walk. Both read the
        // legacy `KernelSig` columns and both are deleted; R4e's census
        // found that the last thing to BUILD a `TraceBuilder` was
        // `model-dsl-legacy`'s `Trace`, which R3 deleted, so nothing had
        // reached either walk since.
        plan
    }
}
