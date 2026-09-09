use std::collections::HashMap;

use crate::dispatch::{
    DispatchAttention, DispatchCollective, DispatchCustomCuda, DispatchElementwise, DispatchLayout,
    DispatchLinear, DispatchSpatial,
};
use crate::error::KernelError;
use model_compiler::{Lowering, Region};
use model_ir::ops::{Attention, Collective, Elementwise};
use model_ir::{
    CacheRow, CustomCuda, Def, Dim, Dtype, Guard, Layout, Linear, Node, Operands, Operation,
    Platform, RuntimeInput, Seam, Spatial, StructKind, Trace, Ty, ValueDecl, ValueId,
};

use crate::fire::sink::{EventId, Sink};

pub(crate) struct Build {
    pub(crate) trace: Trace,
    inputs: u32,
}

pub(crate) fn act(width: u64) -> Ty {
    Ty::Tensor {
        shape: vec![Dim::Tokens, Dim::Const(width)],
        dtype: Dtype::Bf16,
    }
}

pub(crate) fn fact(bit: u8) -> Guard {
    Guard::Fact(bit)
}

impl Build {
    pub(crate) fn new() -> Build {
        Build {
            trace: Trace {
                name: "hand-built".to_string(),
                platform: Platform::Cuda,
                params: Vec::new(),
                caches: vec![CacheRow::State {
                    name: "state".to_string(),
                    slab: vec![1],
                    dtype: Dtype::Bf16,
                }],
                values: Vec::new(),
                nodes: Vec::new(),
                seams: Vec::new(),
                drafter: None,
            },
            inputs: 0,
        }
    }

    pub(crate) fn value(&mut self, def: Def, ty: Ty) -> ValueId {
        self.trace.values.push(ValueDecl { def, ty });
        ValueId((self.trace.values.len() - 1) as u32)
    }

    pub(crate) fn input(&mut self, width: u64) -> ValueId {
        self.inputs += 1;
        let which = RuntimeInput::Mask {
            space: self.inputs - 1,
        };
        self.value(Def::Input(which), act(width))
    }

    pub(crate) fn cache(&mut self) -> ValueId {
        self.value(Def::Cache(0), act(1))
    }

    pub(crate) fn op(&mut self, x: ValueId, width: u64, guard: Guard) -> ValueId {
        let node = self.trace.nodes.len() as u32;
        let y = self.value(Def::Op(node), act(width));
        self.push(
            Elementwise::RmsnormNoScale {
                x,
                head_dim: 1,
                eps: 1e-6,
                y,
            }
            .into(),
            guard,
        );
        y
    }

    pub(crate) fn prepare(&mut self, guard: Guard) -> ValueId {
        let kv_indptr = self.input(1);
        let kv_indices = self.input(1);
        let last_page_len = self.input(1);
        let kv_len = self.input(1);
        let node = self.trace.nodes.len() as u32;
        let plan = self.value(Def::Op(node), Ty::Struct(StructKind::AttnDecodePlan));
        self.push(
            Attention::PlanDecode {
                kv_indptr,
                kv_indices,
                last_page_len,
                kv_len,
                q_heads: 1,
                kv_heads: 1,
                head_dim: 4,
                window: None,
                plan,
            }
            .into(),
            guard,
        );
        plan
    }

    pub(crate) fn decode(&mut self, q: ValueId, plan: ValueId, guard: Guard) -> ValueId {
        let cache = self.cache();
        let node = self.trace.nodes.len() as u32;
        let o = self.value(Def::Op(node), act(4));
        self.push(
            Attention::Decode {
                q,
                plan,
                cache,
                window: None,
                head_dim: 4,
                sm_scale: 1.0,
                o,
            }
            .into(),
            guard,
        );
        o
    }

    pub(crate) fn merge(&mut self, arms: &[(ValueId, Guard)], width: u64) -> ValueId {
        self.value(Def::Merge(arms.to_vec()), act(width))
    }

    pub(crate) fn out(&mut self, v: ValueId) -> &mut Build {
        self.trace.seams.push(Seam {
            seam: "out".to_string(),
            values: vec![v],
            layer: None,
        });
        self
    }

    fn push(&mut self, op: Operation, guard: Guard) {
        self.trace.nodes.push(Node {
            op,
            guard,
            layer: None,
        });
    }
}

pub(crate) struct MockDispatch<'p> {
    at: HashMap<usize, u32>,
    pub(crate) seen: Vec<(u32, &'static str)>,
    pub(crate) refuse: Option<&'static str>,
    pub(crate) copies: bool,
    pub(crate) moved: Vec<(u32, &'static str)>,
    trace: &'p Trace,
}

impl<'p> MockDispatch<'p> {
    pub(crate) fn new(trace: &'p Trace) -> MockDispatch<'p> {
        let at = trace
            .nodes
            .iter()
            .enumerate()
            .map(|(j, node)| (payload(&node.op), j as u32))
            .collect();
        MockDispatch {
            at,
            seen: Vec::new(),
            refuse: None,
            copies: false,
            moved: Vec::new(),
            trace,
        }
    }

    pub(crate) fn nodes(&self) -> Vec<u32> {
        self.seen.iter().map(|(node, _)| *node).collect()
    }

    pub(crate) fn names(&self) -> Vec<&'static str> {
        self.seen.iter().map(|(_, name)| *name).collect()
    }

    fn note<T: Operands>(&mut self, op: &T) -> Result<(), KernelError> {
        if self.refuse == Some(op.name()) {
            return Err(KernelError::Unsupported { op: op.name() });
        }
        let address = address(op);
        let node = *self
            .at
            .get(&address)
            .expect("every dispatched op is a node of the plan the mock was built from");
        assert!(
            (node as usize) < self.trace.nodes.len(),
            "the mock's map is built from the plan's own nodes",
        );
        self.seen.push((node, op.name()));
        Ok(())
    }
}

fn address<T>(value: &T) -> usize {
    std::ptr::from_ref(value).cast::<()>() as usize
}

fn payload(op: &Operation) -> usize {
    match op {
        Operation::Attention(op) => address(op),
        Operation::Linear(op) => address(op),
        Operation::Elementwise(op) => address(op),
        Operation::Layout(op) => address(op),
        Operation::Collective(op) => address(op),
        Operation::CustomCuda(op) => address(op),
        Operation::Spatial(op) => address(op),
    }
}

impl DispatchAttention for MockDispatch<'_> {
    fn dispatch(&mut self, op: &Attention) -> Result<(), KernelError> {
        self.note(op)
    }
}

impl DispatchLinear for MockDispatch<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.note(op)
    }
}

impl DispatchElementwise for MockDispatch<'_> {
    fn dispatch(&mut self, op: &Elementwise) -> Result<(), KernelError> {
        self.note(op)
    }
}

impl DispatchLayout for MockDispatch<'_> {
    fn dispatch(&mut self, op: &Layout) -> Result<(), KernelError> {
        self.note(op)
    }
}

impl DispatchCollective for MockDispatch<'_> {
    fn dispatch(&mut self, op: &Collective) -> Result<(), KernelError> {
        self.note(op)
    }
}

impl DispatchCustomCuda for MockDispatch<'_> {
    fn dispatch(&mut self, op: &CustomCuda) -> Result<(), KernelError> {
        self.note(op)
    }
}

impl DispatchSpatial for MockDispatch<'_> {
    fn dispatch(&mut self, op: &Spatial) -> Result<(), KernelError> {
        self.note(op)
    }
}

impl crate::fire::fallback::Serve for MockDispatch<'_> {
    fn copies(&self, _region: &model_compiler::Region) -> bool {
        self.copies
    }

    fn gather(&mut self, region: &model_compiler::Region) -> Result<(), KernelError> {
        self.moved.push((region.nodes.start, "gather"));
        Ok(())
    }

    fn scatter(&mut self, region: &model_compiler::Region) -> Result<(), KernelError> {
        self.moved.push((region.nodes.start, "scatter"));
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Event {
    Begin(u32),
    End(u32),
    Run(u32, u32),
    CondBegin,
    CondArm(u8),
    CondEnd,
    Fork(u32),
    Join(u32),
}

#[derive(Debug, Default)]
pub(crate) struct Recorder {
    pub(crate) events: Vec<Event>,
}

impl Sink for Recorder {
    fn region_begin(&mut self, region: &Region) {
        self.events.push(Event::Begin(region.nodes.start));
    }
    fn region_end(&mut self, region: &Region) {
        self.events.push(Event::End(region.nodes.start));
    }
    fn run(&mut self, run: u32, runs: u32) {
        if runs > 1 {
            self.events.push(Event::Run(run, runs));
        }
    }
    fn cond_begin(&mut self, _lowering: &Lowering) {
        self.events.push(Event::CondBegin);
    }
    fn cond_arm(&mut self, arm: u8) {
        self.events.push(Event::CondArm(arm));
    }
    fn cond_end(&mut self) {
        self.events.push(Event::CondEnd);
    }
    fn fork(&mut self, event: EventId) {
        self.events.push(Event::Fork(event.0));
    }
    fn join(&mut self, event: EventId) {
        self.events.push(Event::Join(event.0));
    }
}

impl crate::dispatch::DispatchProbe for MockDispatch<'_> {}
