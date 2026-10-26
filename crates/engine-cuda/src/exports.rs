use poem_compiler::CompiledModel;
use poem_ir::{Operands, Trace, ValueId};

use crate::error::{Fault, Result};
use poem_exec::feeds::{classes_running, writer_classes};

pub(crate) const OUT_SEAM: &str = poem_compiler::EXPORT_SEAMS[0];
pub(crate) const MTP_SEAM: &str = poem_compiler::EXPORT_SEAMS[1];
pub(crate) const SCORES_SEAM: &str = poem_compiler::EXPORT_SEAMS[2];
pub(crate) const FLOAT_READOUT_SEAMS: [&str; 3] = poem_compiler::FLOAT_READOUT_SEAMS;
pub(crate) const PIXELS_SEAM: &str = poem_compiler::EXPORT_SEAMS[6];
pub(crate) const DRAFTS_SEAM: &str = poem_compiler::EXPORT_SEAMS[3];

#[derive(Debug, Clone)]
pub struct Export {
    pub value: ValueId,
    pub layer: u32,
    pub classes: poem_ir::ClassSet,
}

#[derive(Debug, Clone)]
pub(crate) struct Exports {
    pub(crate) out_classes: poem_ir::ClassSet,
    pub(crate) out: Option<ValueId>,
    pub(crate) mtp: Option<Export>,
    pub(crate) drafts: Option<Export>,
    pub(crate) drafts_depth: u32,
    pub(crate) scores: Vec<Export>,
    pub(crate) capturing: poem_ir::ClassSet,
    pub(crate) velocity: Option<Export>,
    pub(crate) hidden: Vec<Export>,
    pub(crate) pixels: Vec<(Export, ValueId)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReadoutSeam {
    pub(crate) seam: engine::fire::ReadoutSeam,
    pub(crate) value: ValueId,
}

impl Exports {
    #[must_use]
    pub(crate) fn pixels_for(&self, class: Option<usize>) -> Option<(ValueId, ValueId)> {
        class
            .and_then(|class| {
                self.pixels
                    .iter()
                    .find(|(export, _)| export.classes.contains(class))
            })
            .or_else(|| (self.pixels.len() == 1).then(|| &self.pixels[0]))
            .map(|(export, grid)| (export.value, *grid))
    }

    pub(crate) fn writes_pixels(&self, class: usize) -> bool {
        self.pixels
            .iter()
            .any(|(export, _)| export.classes.contains(class))
    }

    pub(crate) fn on_voxels(&self, trace: &Trace, value: ValueId) -> bool {
        matches!(
            &trace.values[value.0 as usize].ty,
            poem_ir::Ty::Tensor { shape, .. }
                if matches!(shape.first(), Some(poem_ir::Dim::Voxels | poem_ir::Dim::VoxelsTimes(_)))
        )
    }

    #[must_use]
    pub(crate) fn readout_for(&self, class: usize) -> Option<ReadoutSeam> {
        use engine::fire::ReadoutSeam as Seam;
        if let Some(out) = self.out
            && self.out_classes.contains(class)
        {
            return Some(ReadoutSeam {
                seam: Seam::Logits,
                value: out,
            });
        }
        if let Some(velocity) = self.velocity_for(class) {
            return Some(ReadoutSeam {
                seam: Seam::Velocity,
                value: velocity.value,
            });
        }
        if let Some(hidden) = self.hidden_for(class) {
            return Some(ReadoutSeam {
                seam: Seam::Hidden,
                value: hidden.value,
            });
        }
        self.readout()
    }

    #[must_use]
    pub(crate) fn velocity_for(&self, class: usize) -> Option<&Export> {
        self.velocity
            .as_ref()
            .filter(|export| export.classes.contains(class))
    }

    #[must_use]
    pub(crate) fn hidden_for(&self, class: usize) -> Option<&Export> {
        self.hidden
            .iter()
            .rev()
            .find(|export| export.classes.contains(class))
    }

    #[must_use]
    pub(crate) fn readout(&self) -> Option<ReadoutSeam> {
        use engine::fire::ReadoutSeam as Seam;
        if let Some(out) = self.out {
            return Some(ReadoutSeam {
                seam: Seam::Logits,
                value: out,
            });
        }
        if let Some(velocity) = &self.velocity {
            return Some(ReadoutSeam {
                seam: Seam::Velocity,
                value: velocity.value,
            });
        }
        self.hidden.last().map(|hidden| ReadoutSeam {
            seam: Seam::Hidden,
            value: hidden.value,
        })
    }
}

impl Exports {
    pub(crate) fn of(trace: &Trace, compiled: &CompiledModel) -> Result<Exports> {
        let out = trace
            .seams
            .iter()
            .find(|seam| seam.seam == OUT_SEAM)
            .and_then(|seam| seam.values.first().copied());
        let float_readout = trace.seams.iter().any(|seam| {
            FLOAT_READOUT_SEAMS.contains(&seam.seam.as_str()) && !seam.values.is_empty()
        });
        if out.is_none() && !float_readout {
            return Err(Fault::Unbound {
                what: format!(
                    "no `{OUT_SEAM}` seam and no float readout ({}), so a fire would compute \
                     nothing a reader can take",
                    FLOAT_READOUT_SEAMS.join(", ")
                ),
            });
        }
        let named = |name: &str| -> Vec<Export> {
            trace
                .seams
                .iter()
                .filter(|seam| seam.seam == name)
                .flat_map(|seam| {
                    let layer = seam.layer.unwrap_or(0);
                    seam.values.iter().map(move |value| (layer, *value))
                })
                .map(|(layer, value)| Export {
                    value,
                    layer,
                    classes: writer_classes(trace, compiled, value),
                })
                .collect()
        };
        let scores = named(SCORES_SEAM);
        let drafts = named(DRAFTS_SEAM).into_iter().next();
        let drafts_depth = match &drafts {
            Some(export) => {
                let width = poem_exec::store::kv::width_of(trace, export.value).map_err(|why| {
                    Fault::Unbound {
                        what: format!("the `{DRAFTS_SEAM}` export's width: {why}"),
                    }
                })?;
                match u32::try_from(width) {
                    Ok(depth) if depth > 0 => depth,
                    _ => {
                        return Err(Fault::Unbound {
                            what: format!("a `{DRAFTS_SEAM}` export {width} wide drafts nothing"),
                        });
                    }
                }
            }
            None => 0,
        };
        let mut capturing = poem_ir::ClassSet::default();
        for export in &scores {
            for class in export.classes.iter() {
                capturing.insert(class);
            }
        }
        let pixels = trace
            .seams
            .iter()
            .filter(|seam| seam.seam == PIXELS_SEAM)
            .filter_map(|seam| match seam.values.as_slice() {
                [plane, grid, ..] => Some((
                    Export {
                        value: *plane,
                        layer: seam.layer.unwrap_or(0),
                        classes: writer_classes(trace, compiled, *plane),
                    },
                    *grid,
                )),
                _ => None,
            })
            .collect();
        Ok(Exports {
            out_classes: out.map_or_else(poem_ir::ClassSet::default, |out| {
                writer_classes(trace, compiled, out)
            }),
            out,
            mtp: named(MTP_SEAM).into_iter().next(),
            drafts,
            drafts_depth,
            scores,
            capturing,
            velocity: named(FLOAT_READOUT_SEAMS[0]).into_iter().next(),
            hidden: named(FLOAT_READOUT_SEAMS[1]),
            pixels,
        })
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

/// The classes one body per bucket lays out whole ([`crate::record::Tiers::whole`]),
/// in the order a fire lays their rows out: each a request lands in, that
/// stages no media and reads no guest input, so a synthetic lane stands in
/// for it when the body is armed.
#[must_use]
pub(crate) fn whole_of(
    landing: &[Vec<poem_ir::Request>],
    compiled: &CompiledModel,
    feeds: &poem_exec::feeds::Feeds,
    media: &poem_ir::ClassSet,
) -> Box<[u32]> {
    let every = poem_ir::ClassSet::of(
        landing
            .iter()
            .enumerate()
            .filter(|(class, requests)| {
                !requests.is_empty()
                    && !media.contains(*class)
                    && !feeds
                        .inputs
                        .iter()
                        .any(|(_, readers)| readers.contains(*class))
            })
            .map(|(class, _)| class),
    );
    let order = compiled.order_for(poem_ir::RowAxis::Tokens).map_or_else(
        || every.iter().map(|class| class as u8).collect(),
        |order| order.class_order(&every),
    );
    order.into_iter().map(u32::from).collect()
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
