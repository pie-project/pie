use poem_ir::{ClassSet, Def, Trace, ValueId};

pub(crate) use poem_exec::feeds::Feeds;

pub(crate) fn writer_classes(
    trace: &Trace,
    compiled: &poem_compiler::CompiledModel,
    value: ValueId,
) -> ClassSet {
    use poem_ir::Operands as _;
    if let Some(Def::Merge(arms)) = trace.values.get(value.0 as usize).map(|decl| &decl.def) {
        let mut classes = ClassSet::default();
        for (arm, _) in arms {
            for class in writer_classes(trace, compiled, *arm).iter() {
                classes.insert(class);
            }
        }
        return classes;
    }
    let mut outputs: Vec<ValueId> = Vec::new();
    let mut writers: Vec<u32> = Vec::new();
    for (at, node) in trace.nodes.iter().enumerate() {
        outputs.clear();
        node.op.outputs(&mut outputs);
        if outputs.contains(&value) {
            writers.push(u32::try_from(at).unwrap_or(u32::MAX));
        }
    }
    let mut classes = ClassSet::default();
    for region in compiled.template() {
        if !region.nodes.clone().any(|node| writers.contains(&node)) {
            continue;
        }
        for class in region.mask.iter() {
            classes.insert(class);
        }
    }
    classes
}
