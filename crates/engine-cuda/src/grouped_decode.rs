//! Single-token rows of a wide head group read the cache through the
//! prefill kernel, where the profile says so
//! (`kernel.sdpa.decode.grouped_prefill_from`, on for sm_100 from four
//! query heads a KV head): the prefill kernel's tensor cores take the
//! group's rows as one tile, where the decode kernel walks them on CUDA
//! cores. The trace is rewritten before anything reads its plans' kinds, so
//! the schedules, the workspaces and the dispatch all see prefill plans.
//! At 0 (the default, and every other chip's profile) the trace is returned
//! untouched.

use std::collections::BTreeMap;

use poem_ir::{Attention, Operands, Operation, StructKind, Trace, Ty, ValueId};

/// `trace` with every decode plan of `from` or more query heads a KV head
/// that only plain decodes read planned and read as a prefill.
#[must_use]
pub fn rewrite(mut trace: Trace, from: u32) -> Trace {
    if from == 0 {
        return trace;
    }
    // Decode plans wide enough, by plan value: their KV heads.
    let mut wide: BTreeMap<ValueId, u32> = trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Attention(Attention::PlanDecode {
                q_heads,
                kv_heads,
                plan,
                ..
            }) if *kv_heads > 0 && q_heads / kv_heads >= from => Some((*plan, *kv_heads)),
            _ => None,
        })
        .collect();
    // A plan another op reads (a log-sum-exp decode, a biased one, a
    // selection) stays a decode plan.
    let mut inputs = Vec::new();
    for node in &trace.nodes {
        if matches!(
            node.op,
            Operation::Attention(Attention::PlanDecode { .. } | Attention::Decode { .. })
        ) {
            continue;
        }
        inputs.clear();
        node.op.inputs(&mut inputs);
        for v in &inputs {
            wide.remove(v);
        }
    }
    for node in &mut trace.nodes {
        let Operation::Attention(op) = &mut node.op else {
            continue;
        };
        *op = match op.clone() {
            Attention::PlanDecode {
                kv_indptr,
                kv_indices,
                last_page_len,
                kv_len,
                q_heads,
                kv_heads,
                head_dim,
                window,
                plan,
            } if wide.contains_key(&plan) => Attention::PlanPrefill {
                kv_indptr,
                kv_indices,
                last_page_len,
                kv_len,
                q_heads,
                kv_heads,
                head_dim,
                window,
                plan,
            },
            Attention::Decode {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
            } if wide.contains_key(&plan) => Attention::Prefill {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads: wide[&plan],
                sm_scale,
                o,
            },
            other => other,
        };
    }
    for plan in wide.keys() {
        trace.values[plan.0 as usize].ty = Ty::Struct(StructKind::AttnPrefillPlan);
    }
    trace
}
