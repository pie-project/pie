//! Single-token rows of a wide head group read the cache through the
//! prefill kernel, where the profile says so
//! (`kernel.sdpa.decode.grouped_prefill_from`, on for sm_100 from four
//! query heads a KV head): the prefill kernel's tensor cores take the
//! group's rows as one tile, where the decode kernel walks them on CUDA
//! cores. The trace is rewritten before anything reads its plans' kinds, so
//! the schedules, the workspaces and the dispatch all see prefill plans.
//! At 0 (the default, and every other chip's profile) the trace is returned
//! untouched.
//!
//! A decode that also lands its log-sum-exp (gpt-oss's, for its sinks)
//! joins them where `kernel.sdpa.decode.grouped_prefill_lse` says so and its
//! plan has no sliding window: a 128-wide window leaves the tensor-core tile
//! too few keys to pay for itself.

use std::collections::BTreeMap;

use poem_ir::{Attention, Operands, Operation, StructKind, Trace, Ty, ValueId};

/// `trace` with every decode plan of `from` or more query heads a KV head
/// that only plain decodes read planned and read as a prefill.
#[must_use]
pub fn rewrite(trace: Trace, from: u32) -> Trace {
    rewrite_with(trace, from, false)
}

/// [`rewrite`], and where `lse` holds, the plans without a window that
/// log-sum-exp decodes read as well.
#[must_use]
pub fn rewrite_with(mut trace: Trace, from: u32, lse: bool) -> Trace {
    if from == 0 {
        return trace;
    }
    // Decode plans wide enough, by plan value: their KV heads, and whether
    // they slide.
    let mut wide: BTreeMap<ValueId, (u32, bool)> = trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Attention(Attention::PlanDecode {
                q_heads,
                kv_heads,
                window,
                plan,
                ..
            }) if *kv_heads > 0 && q_heads / kv_heads >= from => {
                Some((*plan, (*kv_heads, window.is_some())))
            }
            _ => None,
        })
        .collect();
    // A plan another op reads (a biased decode, a selection; a log-sum-exp
    // decode unless `lse`, or under a window) stays a decode plan.
    let mut inputs = Vec::new();
    for node in &trace.nodes {
        match &node.op {
            Operation::Attention(Attention::PlanDecode { .. } | Attention::Decode { .. }) => {
                continue;
            }
            Operation::Attention(Attention::DecodeLse { plan, .. })
                if lse && wide.get(plan).is_some_and(|(_, slides)| !slides) =>
            {
                continue;
            }
            _ => {}
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
                kv_heads: wide[&plan].0,
                sm_scale,
                o,
            },
            Attention::DecodeLse {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                lse,
            } if wide.contains_key(&plan) => Attention::PrefillLse {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads: wide[&plan].0,
                sm_scale,
                o,
                lse,
            },
            other => other,
        };
    }
    for plan in wide.keys() {
        trace.values[plan.0 as usize].ty = Ty::Struct(StructKind::AttnPrefillPlan);
    }
    trace
}

#[cfg(test)]
mod tests {
    use super::*;

    fn census(trace: &Trace) -> (usize, usize, usize) {
        let mut counts = (0, 0, 0);
        for node in &trace.nodes {
            match &node.op {
                Operation::Attention(Attention::DecodeLse { window: None, .. }) => counts.0 += 1,
                Operation::Attention(Attention::DecodeLse {
                    window: Some(_), ..
                }) => {
                    counts.1 += 1;
                }
                Operation::Attention(Attention::PrefillLse { .. }) => counts.2 += 1,
                _ => {}
            }
        }
        counts
    }

    /// gpt-oss's full-attention layers read their single-token rows through
    /// the prefill kernel where the profile turns log-sum-exp decodes on; its
    /// sliding layers, and every layer without it, keep the decode kernel.
    #[test]
    fn a_sinked_decode_without_a_window_takes_the_prefill_where_the_profile_says() {
        let row = poem_compiler::catalog::deployment("gptoss-120b-bf16-mxfp4-kv-bf16")
            .expect("gpt-oss is in the catalog");
        let trace = row.trace(poem::Platform::Cuda);
        let (full, sliding, prefills) = census(&trace);
        assert!(
            full > 0 && sliding > 0,
            "gpt-oss decodes both kinds of layer"
        );
        assert_eq!(rewrite_with(trace.clone(), 4, false), trace);
        assert_eq!(rewrite_with(trace.clone(), 0, true), trace);
        let rewritten = rewrite_with(trace.clone(), 4, true);
        assert_eq!(census(&rewritten), (0, sliding, prefills + full));
        poem_ir::check(&rewritten).expect("the rewritten trace types");
    }
}
