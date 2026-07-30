//! Groups: one plan, compiled once, run `arity` times.
//!
//! A [`GroupContract`](crate::contract::GroupContract) declares a set of
//! tensors that differ from each other only in which bytes they read -- routed
//! experts being the case that motivated it, but nothing here knows that word.
//! The loader's job is to prove that claim and to say what "which bytes"
//! resolves to for each index; what the driver then *does* with a group (stack
//! every instance into an arena, or page instances through a fixed set of
//! slots) is policy, and policy is not in the plan.
//!
//! The proof is the compile. Every instance is compiled independently and then
//! required to be identical to instance 0 in every field except the three that
//! locate bytes in a checkpoint: `file_id`, `tensor_id`, `file_offset`. If two
//! instances disagree about anything else -- a buffer size, a transform, an
//! instruction count -- they are not interchangeable, and the group is
//! rejected here rather than producing slots that quietly differ in layout.
//!
//! That equality is also the compression. The plan carries one instance's
//! instructions plus an `arity`-long table of source rebindings, instead of
//! `arity` copies of a plan that agree everywhere but three integers.

use std::collections::HashMap;

use crate::checkpoint::CheckpointMetadata;
use crate::contract::{GroupContract, ModelContract};
use crate::error::{Error, Result};
use crate::plan::{GroupPlan, LoadPlan, SourceBinding, StorageInstr, StorageTarget};

/// Compile every group in `contract`.
pub(crate) fn compile_all(
    metadata: &CheckpointMetadata,
    contract: &ModelContract,
    target: &StorageTarget,
) -> Result<Vec<GroupPlan>> {
    let mut seen: HashMap<&str, ()> = HashMap::new();
    let mut plans = Vec::with_capacity(contract.groups.len());
    for group in &contract.groups {
        if seen.insert(group.name.as_str(), ()).is_some() {
            return Err(Error::Contract(format!(
                "the contract declares group '{}' twice",
                group.name
            )));
        }
        plans.push(compile_one(metadata, contract, group, target)?);
    }
    Ok(plans)
}

fn compile_one(
    metadata: &CheckpointMetadata,
    contract: &ModelContract,
    group: &GroupContract,
    target: &StorageTarget,
) -> Result<GroupPlan> {
    if group.arity == 0 {
        return Err(Error::Contract(format!(
            "group '{}' has arity 0; a group with no instances declares nothing",
            group.name
        )));
    }
    if group.tensors.is_empty() {
        return Err(Error::Contract(format!(
            "group '{}' declares no tensors",
            group.name
        )));
    }

    // A group is compiled as a contract of its own. It has to be: the whole
    // point is that its instructions are a self-contained program the driver
    // can run against a destination of its choosing, which it would not be if
    // its buffers were numbered inside the resident plan.
    let sub = ModelContract {
        alignment: contract.alignment,
        tensors: group.tensors.clone(),
        groups: Vec::new(),
    };

    let template = compile_instance(metadata, &sub, target, 0, group)?;
    let mut bindings = Vec::with_capacity(group.arity as usize);
    bindings.push(read_bindings(&template));

    for index in 1..group.arity {
        let instance = compile_instance(metadata, &sub, target, index, group)?;
        let binding = read_bindings(&instance);
        assert_interchangeable(&template, &instance, &binding, group, index)?;
        bindings.push(binding);
    }

    Ok(GroupPlan {
        name: group.name.clone(),
        arity: group.arity,
        plan: template,
        bindings,
    })
}

fn compile_instance(
    metadata: &CheckpointMetadata,
    sub: &ModelContract,
    target: &StorageTarget,
    index: u32,
    group: &GroupContract,
) -> Result<LoadPlan> {
    let rewritten = crate::contract::rewrite::coalesce_direct_row_shards(sub, metadata, target)?;
    let mut plan =
        crate::plan::build::build_instance(metadata, &rewritten, target.clone(), Some(index))
            .map_err(|err| at_index(err, group, index))?;
    plan.passes = crate::plan::pass::run_all(&mut plan)?;
    crate::plan::passes::tile::lower(&mut plan);
    Ok(plan)
}

/// Name the instance in a group's diagnostics.
///
/// Without this a failure reports a checkpoint tensor nobody wrote down --
/// the substituted name -- with no way back to the template that produced it.
fn at_index(err: Error, group: &GroupContract, index: u32) -> Error {
    Error::Contract(format!("group '{}' index {index}: {err}", group.name))
}

/// Every source an instance reads, in instruction order.
fn read_bindings(plan: &LoadPlan) -> Vec<SourceBinding> {
    plan.instrs
        .iter()
        .filter_map(|instr| {
            let (id, source) = match instr {
                StorageInstr::ExtentWrite { id, source, .. }
                | StorageInstr::BulkExtentWrite { id, source, .. } => (*id, source),
                StorageInstr::TileMap {
                    id,
                    source: Some(source),
                    ..
                } => (*id, source),
                _ => return None,
            };
            Some(SourceBinding {
                instr: id,
                file_id: source.file_id,
                tensor_id: source.tensor_id,
                file_offset: source.file_offset,
            })
        })
        .collect()
}

/// Prove that `instance` differs from `template` only where a binding says it
/// may.
///
/// Implemented by rewriting the instance back onto the template's sources and
/// demanding exact equality: it is the cheap direction of the same statement,
/// and it cannot miss a field the way an enumerated comparison would drift as
/// [`StorageInstr`] grows.
fn assert_interchangeable(
    template: &LoadPlan,
    instance: &LoadPlan,
    binding: &[SourceBinding],
    group: &GroupContract,
    index: u32,
) -> Result<()> {
    let mut rebound = instance.clone();
    let template_bindings = read_bindings(template);
    if binding.len() != template_bindings.len() {
        return Err(mismatch(
            group,
            index,
            "reads a different number of sources",
        ));
    }
    let mut want = template_bindings.iter();
    for instr in &mut rebound.instrs {
        let source = match instr {
            StorageInstr::ExtentWrite { source, .. }
            | StorageInstr::BulkExtentWrite { source, .. } => source,
            StorageInstr::TileMap {
                source: Some(source),
                ..
            } => source,
            _ => continue,
        };
        let Some(want) = want.next() else {
            return Err(mismatch(
                group,
                index,
                "reads a different number of sources",
            ));
        };
        source.file_id = want.file_id;
        source.tensor_id = want.tensor_id;
        source.file_offset = want.file_offset;
    }

    if rebound.instrs != template.instrs {
        return Err(mismatch(
            group,
            index,
            "compiles to different instructions, so its bytes are not \
             interchangeable with index 0's",
        ));
    }
    if rebound.buffers != template.buffers {
        return Err(mismatch(group, index, "allocates different buffers"));
    }
    if rebound.tensors != template.tensors {
        return Err(mismatch(group, index, "publishes different tensors"));
    }
    if rebound.schedule != template.schedule {
        return Err(mismatch(
            group,
            index,
            "runs its instructions in a different order",
        ));
    }
    if rebound.memory != template.memory {
        return Err(mismatch(group, index, "needs a different amount of memory"));
    }
    if rebound.attachments != template.attachments {
        return Err(mismatch(group, index, "attaches quantization differently"));
    }
    Ok(())
}

fn mismatch(group: &GroupContract, index: u32, what: &str) -> Error {
    Error::Contract(format!(
        "group '{}' is not uniform: index {index} {what}. Every instance of a \
         group must differ from index 0 only in which bytes it reads.",
        group.name
    ))
}
