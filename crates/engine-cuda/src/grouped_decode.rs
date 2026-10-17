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
//! too few keys to pay for itself. A plan the tensor-core decode reads across
//! a cluster (`kernel.sdpa.decode.xqa_cluster`) stays a decode plan: the
//! cluster folds its own split, which the prefill's split leaves to a launch.

use std::collections::BTreeMap;

use kernels_cuda::attn::xqa::Gate;
use poem_ir::{Attention, Fused, Operands, Operation, StructKind, Trace, Ty, ValueId};

/// `trace` with every decode plan of `from` or more query heads a KV head
/// that only plain decodes read planned and read as a prefill.
#[must_use]
pub fn rewrite(trace: Trace, from: u32) -> Trace {
    rewrite_with(trace, from, false, Gate::default())
}

/// [`rewrite`], and where `lse` holds, the plans without a window that
/// log-sum-exp decodes read as well, but for those `xqa` reads across a
/// cluster.
#[must_use]
pub fn rewrite_with(mut trace: Trace, from: u32, lse: bool, xqa: Gate) -> Trace {
    if from == 0 {
        return trace;
    }
    // Decode plans wide enough, by plan value: their KV heads, and whether
    // they slide.
    let mut wide: BTreeMap<ValueId, (u32, bool, bool)> = trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Attention(Attention::PlanDecode {
                q_heads,
                kv_heads,
                head_dim,
                window,
                plan,
                ..
            }) if *kv_heads > 0 && q_heads / kv_heads >= from => {
                let clustered = window.is_none()
                    && xqa.ranks(*head_dim, q_heads / kv_heads, 1, *kv_heads, -1, u32::MAX) > 0;
                Some((*plan, (*kv_heads, window.is_some(), clustered)))
            }
            _ => None,
        })
        .collect();
    // A plan another op reads (a biased decode, a selection; a log-sum-exp
    // decode unless `lse`, or under a window) stays a decode plan.
    let mut inputs = Vec::new();
    for node in &trace.nodes {
        let attention = match &node.op {
            Operation::Attention(op) => Some(op),
            Operation::Fused(Fused::LseSink { attn, .. }) => Some(&**attn),
            _ => None,
        };
        match attention {
            Some(Attention::PlanDecode { .. } | Attention::Decode { .. }) => continue,
            Some(Attention::DecodeLse { plan, .. })
                if lse
                    && wide
                        .get(plan)
                        .is_some_and(|(_, slides, clustered)| !slides && !clustered) =>
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
        let op = match &mut node.op {
            Operation::Attention(op) => op,
            Operation::Fused(Fused::LseSink { attn, .. }) => &mut **attn,
            _ => continue,
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
                floor: None,
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
            let op = match &node.op {
                Operation::Attention(op) => op,
                Operation::Fused(Fused::LseSink { attn, .. }) => &**attn,
                _ => continue,
            };
            match op {
                Attention::DecodeLse { window: None, .. } => counts.0 += 1,
                Attention::DecodeLse {
                    window: Some(_), ..
                } => counts.1 += 1,
                Attention::PrefillLse { .. } => counts.2 += 1,
                _ => {}
            }
        }
        counts
    }

    /// gpt-oss's full-attention layers read their single-token rows through
    /// the prefill kernel where the profile turns log-sum-exp decodes on; its
    /// sliding layers, and every layer without it, keep the decode kernel.
    /// So with its sinks fused onto the attentions, and without.
    #[test]
    fn a_sinked_decode_without_a_window_takes_the_prefill_where_the_profile_says() {
        let row = poem_compiler::catalog::deployment("gptoss-120b-bf16-mxfp4-kv-bf16")
            .expect("gpt-oss is in the catalog");
        let plain = row.trace(poem::Platform::Cuda);
        let fused = poem_compiler::fuse::fuse(plain.clone(), &["attention.lse_sink"]);
        assert!(
            fused
                .nodes
                .iter()
                .any(|node| matches!(node.op, Operation::Fused(Fused::LseSink { .. }))),
            "gpt-oss's sinks fuse onto its attentions"
        );
        for trace in [plain, fused] {
            regroups(&trace);
        }
    }

    fn regroups(trace: &Trace) {
        let trace = trace.clone();
        let (full, sliding, prefills) = census(&trace);
        assert!(
            full > 0 && sliding > 0,
            "gpt-oss decodes both kinds of layer"
        );
        let off = Gate::default();
        assert_eq!(rewrite_with(trace.clone(), 4, false, off), trace);
        assert_eq!(rewrite_with(trace.clone(), 0, true, off), trace);
        let rewritten = rewrite_with(trace.clone(), 4, true, off);
        assert_eq!(census(&rewritten), (0, sliding, prefills + full));
        // Rows the tensor-core decode reads across a cluster stay decodes.
        let clustered = Gate {
            on: true,
            cluster: 8,
            ..off
        };
        assert_eq!(rewrite_with(trace.clone(), 4, true, clustered), trace);
        poem_ir::check(&rewritten).expect("the rewritten trace types");
    }
}
