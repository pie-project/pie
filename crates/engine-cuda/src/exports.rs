use poem_compiler::CompiledModel;
use poem_ir::{Operands, Trace, ValueId};

use crate::error::{Fault, Result};
use poem_exec::feeds::classes_running;

use poem_exec::feeds::ClassOutput;

/// The outputs a load's classes write.
#[derive(Debug, Clone)]
pub(crate) struct Exports {
    pub(crate) outputs: Vec<ClassOutput>,
}

impl Exports {
    pub(crate) fn of(trace: &Trace, compiled: &CompiledModel) -> Result<Exports> {
        let outputs = poem_exec::feeds::class_outputs(trace, compiled).map_err(Fault::Recipe)?;
        if outputs.is_empty() {
            return Err(Fault::Unbound {
                what: "no output, so a fire would compute nothing a reader can take".to_string(),
            });
        }
        Ok(Exports { outputs })
    }

    /// The outputs class `class` writes.
    pub(crate) fn of_class(&self, class: usize) -> impl Iterator<Item = &ClassOutput> {
        self.outputs
            .iter()
            .filter(move |out| out.classes.contains(class))
    }

    /// What a lane of `class` is read back through on the host: the last
    /// float row or voxel output it writes.
    #[must_use]
    pub(crate) fn readback(&self, class: usize) -> Option<&ClassOutput> {
        self.of_class(class)
            .filter(|out| {
                out.decl.axis != poem_ir::InputAxis::Lane
                    && matches!(out.decl.dtype, poem_ir::Dtype::Bf16 | poem_ir::Dtype::F32)
            })
            .last()
    }

    /// Whether some output of the load is read out at the readout rows.
    #[must_use]
    pub(crate) fn reads_readout_rows(&self) -> bool {
        self.outputs.iter().any(|out| out.decl.readout)
    }

    /// The widest row of an output on the readout rows, in f32.
    #[must_use]
    pub(crate) fn readout_row_bytes(&self) -> u64 {
        self.outputs
            .iter()
            .filter(|out| out.decl.readout)
            .map(|out| u64::from(out.decl.width) * 4)
            .max()
            .unwrap_or(0)
    }
}

#[must_use]
pub(crate) fn masked_classes(trace: &Trace, compiled: &CompiledModel) -> poem_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            poem_ir::Operation::Attn(
                poem_ir::Attn::Prefill {
                    mask: Some(_),
                    selection: None,
                    bias: None,
                    lse: None,
                    ..
                } | poem_ir::Attn::Prefill {
                    mask: Some(_),
                    selection: None,
                    bias: None,
                    lse: Some(_),
                    ..
                }
            )
        )
    })
}

#[must_use]
pub(crate) fn wide_classes(trace: &Trace, compiled: &CompiledModel) -> poem_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            poem_ir::Operation::Attn(poem_ir::Attn::Prefill { causal: false, .. })
        )
    })
}

/// Mask bits are staged once per lane with causality baked in, unclipped in a wide
/// class: a causal masked prefill there would see future keys.
pub(crate) fn one_direction_per_mask(trace: &Trace, compiled: &CompiledModel) -> Result<()> {
    let causal_masked = classes_running(trace, compiled, |op| {
        matches!(
            op,
            poem_ir::Operation::Attn(poem_ir::Attn::Prefill {
                mask: Some(_),
                causal: true,
                ..
            })
        )
    });
    match wide_classes(trace, compiled)
        .iter()
        .find(|&class| causal_masked.contains(class))
    {
        Some(class) => Err(Fault::Recipe(format!(
            "class {class} attends both ways under one mask"
        ))),
        None => Ok(()),
    }
}

#[must_use]
pub(crate) fn corrected_classes(trace: &Trace, compiled: &CompiledModel) -> poem_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            poem_ir::Operation::Linear(poem_ir::Linear::LoraCorrect { .. })
        )
    })
}

#[must_use]
pub(crate) fn media_classes(trace: &Trace, compiled: &CompiledModel) -> poem_ir::ClassSet {
    classes_running(trace, compiled, |op| {
        matches!(
            op,
            poem_ir::Operation::Gather(
                poem_ir::Gather::ScatterRows { .. } | poem_ir::Gather::ScatterLiveRows { .. }
            )
        )
    })
}

/// Each class's requests, the plainest first.
#[must_use]
pub(crate) fn landing_requests(
    facts: &poem_ir::Facts,
    classes: &poem_ir::ClassTable,
) -> Vec<Vec<poem_ir::Request>> {
    let mut landing = vec![Vec::new(); classes.classes.len()];
    for request in facts.requests() {
        let word = facts.word(&request) & classes.mask;
        if let Some(class) = classes.class_of(word) {
            landing[class].push(request);
        }
    }
    landing
}

#[must_use]
pub(crate) fn plain_of(landing: &[Vec<poem_ir::Request>]) -> poem_ir::ClassSet {
    poem_ir::ClassSet::of(
        landing
            .iter()
            .enumerate()
            .filter(|(_, requests)| requests.iter().any(poem_ir::Request::plain))
            .map(|(class, _)| class),
    )
}

#[must_use]
pub(crate) fn decoding_of(landing: &[Vec<poem_ir::Request>]) -> poem_ir::ClassSet {
    poem_ir::ClassSet::of(
        landing
            .iter()
            .enumerate()
            .filter(|(_, requests)| {
                !requests.is_empty() && requests.iter().all(|request| request.query_len() == 1)
            })
            .map(|(class, _)| class),
    )
}

#[must_use]
pub(crate) fn regions_shifting(trace: &Trace, compiled: &CompiledModel) -> Vec<bool> {
    compiled
        .template()
        .iter()
        .map(|region| {
            region.nodes.clone().all(|node| {
                trace.nodes.get(node as usize).is_some_and(|node| {
                    let name = poem_ir::Operands::name(&node.op);
                    crate::shifted(&node.op) || crate::PLANNED.contains(&name)
                })
            })
        })
        .collect()
}

/// Regions holding DeepSeek's attention over quantized rows: at prefill width they
/// stage bf16 planes sized by the fire's page tables, so a graph replaying them
/// against a later fire's tables would read past what it staged.
#[must_use]
pub(crate) fn regions_staging(trace: &Trace, compiled: &CompiledModel) -> Vec<bool> {
    let quantized = |attn: &poem_ir::Attn| match attn {
        poem_ir::Attn::Prefill {
            cache,
            mask: None,
            selection: None,
            bias: None,
            lse: Some(_),
            ..
        } => matches!(
            trace.values.get(cache.0 as usize).map(|decl| &decl.def),
            Some(poem_ir::Def::Cache(row)) if matches!(
                trace.caches.get(*row as usize),
                Some(poem_ir::CacheRow::Kv { dtype: poem_ir::Dtype::U8, .. })
            )
        ),
        _ => false,
    };
    compiled
        .template()
        .iter()
        .map(|region| {
            region.nodes.clone().any(|node| {
                trace
                    .nodes
                    .get(node as usize)
                    .is_some_and(|node| match &node.op {
                        poem_ir::Operation::Fused(
                            poem_ir::Fused::PoolAttention { window, .. }
                            | poem_ir::Fused::LseSink { attn: window, .. },
                        ) => quantized(window),
                        _ => false,
                    })
            })
        })
        .collect()
}

#[must_use]
pub(crate) fn regions_lane_shifting(trace: &Trace, compiled: &CompiledModel) -> Vec<bool> {
    compiled
        .template()
        .iter()
        .map(|region| {
            region
                .nodes
                .clone()
                .all(|node| lane_shifting_node(trace, node))
        })
        .collect()
}

fn lane_shifting_node(trace: &Trace, node: u32) -> bool {
    let Some(node) = trace.nodes.get(node as usize) else {
        return false;
    };
    let name = Operands::name(&node.op);
    if crate::lane_shifted(&node.op) || crate::PLANNED.contains(&name) {
        return true;
    }
    let mut operands: Vec<ValueId> = Vec::new();
    node.op.inputs(&mut operands);
    node.op.outputs(&mut operands);
    operands.iter().all(|id| {
        let Some(decl) = trace.values.get(id.0 as usize) else {
            return false;
        };
        if matches!(&decl.def, poem_ir::Def::Cache(_)) {
            return false;
        }
        let poem_ir::Ty::Tensor { shape, .. } = &decl.ty else {
            return true;
        };
        !matches!(
            shape.first(),
            Some(
                poem_ir::Dim::Lanes
                    | poem_ir::Dim::LanesPlus(_)
                    | poem_ir::Dim::Images
                    | poem_ir::Dim::ImagesPlus(_)
            )
        )
    })
}

pub(crate) use poem_exec::feeds::Feeds;

/// For each value, whether it is a schedule read only in regions every class
/// of which is `decoding` (one query row a lane): a grouped decode read
/// through the prefill kernel, whose fires carry at most a row a lane.
#[must_use]
pub(crate) fn read_only_by_decoders(
    trace: &Trace,
    compiled: &CompiledModel,
    decoding: &poem_ir::ClassSet,
) -> Vec<bool> {
    // None: unread; Some(true): every reading region decodes.
    let mut read: Vec<Option<bool>> = vec![None; trace.values.len()];
    let mut inputs: Vec<ValueId> = Vec::new();
    for region in compiled.template() {
        let decodes = !region.mask.is_empty() && region.mask.iter().all(|c| decoding.contains(c));
        for node in region.nodes.clone() {
            let Some(node) = trace.nodes.get(node as usize) else {
                continue;
            };
            inputs.clear();
            node.op.inputs(&mut inputs);
            for id in &inputs {
                let at = id.0 as usize;
                if !trace
                    .values
                    .get(at)
                    .is_some_and(|decl| matches!(decl.ty, poem_ir::Ty::Struct(_)))
                {
                    continue;
                }
                let seen = read[at].get_or_insert(true);
                *seen &= decodes;
            }
        }
    }
    read.into_iter().map(|seen| seen == Some(true)).collect()
}

#[must_use]
pub(crate) fn schedule_streams(trace: &Trace, compiled: &CompiledModel) -> Vec<Option<u32>> {
    let mut streams: Vec<Option<Vec<u32>>> = vec![None; trace.values.len()];
    let mut inputs: Vec<ValueId> = Vec::new();
    for region in compiled.template() {
        for node in region.nodes.clone() {
            let Some(node) = trace.nodes.get(node as usize) else {
                continue;
            };
            inputs.clear();
            node.op.inputs(&mut inputs);
            for id in &inputs {
                let at = id.0 as usize;
                if !trace
                    .values
                    .get(at)
                    .is_some_and(|decl| matches!(decl.ty, poem_ir::Ty::Struct(_)))
                {
                    continue;
                }
                let readers = streams[at].get_or_insert_with(Vec::new);
                if !readers.contains(&region.stream) {
                    readers.push(region.stream);
                }
            }
        }
    }
    streams
        .into_iter()
        .map(|readers| match readers.as_deref() {
            Some([stream]) => Some(*stream),
            _ => None,
        })
        .collect()
}

#[must_use]
pub(crate) fn regions_launching_schedules(
    trace: &Trace,
    compiled: &CompiledModel,
) -> Vec<Option<u32>> {
    let mut out: Vec<Option<u32>> = vec![None; trace.values.len()];
    let mut claimed: Vec<bool> = vec![false; trace.values.len()];
    let mut inputs: Vec<ValueId> = Vec::new();
    for (at, region) in compiled.template().iter().enumerate() {
        let here = u32::try_from(at).unwrap_or(u32::MAX);
        for node in region.nodes.clone() {
            let Some(node) = trace.nodes.get(node as usize) else {
                continue;
            };
            inputs.clear();
            node.op.inputs(&mut inputs);
            for id in &inputs {
                let at = id.0 as usize;
                if !trace
                    .values
                    .get(at)
                    .is_some_and(|decl| matches!(decl.ty, poem_ir::Ty::Struct(_)))
                {
                    continue;
                }
                let Some(slot) = out.get_mut(at) else {
                    continue;
                };
                if !claimed[at] {
                    claimed[at] = true;
                    *slot = Some(here);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_class_attending_both_ways_under_one_mask_is_refused() {
        let mut trace = poem_compiler::catalog::deployment("qwen35-d9b-dflash-u4g64-kv-bf16")
            .expect("the catalog ships the drafter row")
            .trace(poem::Platform::Cuda);
        let compile = |trace: &Trace| {
            poem_compiler::compile_axes(
                trace,
                &poem_compiler::Budgets {
                    tokens: poem_compiler::Budget::new(4, 256),
                    patches: None,
                    voxels: None,
                },
                &poem_compiler::DeviceProfile::default(),
            )
            .expect("the plan bakes")
        };
        one_direction_per_mask(&trace, &compile(&trace)).expect("the row loads");
        let causal = trace
            .nodes
            .iter_mut()
            .find_map(|node| match &mut node.op {
                poem_ir::Operation::Attn(poem_ir::Attn::Prefill {
                    mask: Some(_),
                    causal,
                    ..
                }) if *causal => Some(causal),
                _ => None,
            })
            .expect("the trunk runs a causal masked prefill");
        *causal = false;
        let refusal = one_direction_per_mask(&trace, &compile(&trace))
            .expect_err("one class now attends both ways under one mask");
        assert!(
            refusal
                .to_string()
                .ends_with("attends both ways under one mask"),
            "{refusal}"
        );
    }

    #[test]
    fn every_switch_a_request_carries_lands_in_its_class() {
        let mut facts = poem_ir::Facts::default();
        facts.declare("mode", &["plain", "wide"]).unwrap();
        for builtin in [
            poem_ir::Builtin::Drafts,
            poem_ir::Builtin::Scores,
            poem_ir::Builtin::Encode,
        ] {
            let _ = facts.builtin(builtin);
        }
        let _ = facts.switch("mode", "wide");
        let bits = facts.table.len();
        let classes = poem_ir::ClassTable {
            classes: (0..bits)
                .map(|bit| poem_ir::Class {
                    words: vec![1 << bit],
                    live: Vec::new(),
                })
                .collect(),
            node_mask: Vec::new(),
            merges: Vec::new(),
            merge_arm: Vec::new(),
            mask: (1 << bits) - 1,
            dead: Vec::new(),
        };
        let landing = landing_requests(&facts, &classes);
        for (class, requests) in landing.iter().enumerate() {
            assert!(
                requests
                    .iter()
                    .any(|request| facts.word(request) == 1 << class),
                "no request lands in class {class}"
            );
        }
    }
}
