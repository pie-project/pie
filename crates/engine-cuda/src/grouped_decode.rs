//! Single-token rows of a wide head group read the cache through the prefill
//! kernel where the profile says (`kernel.sdpa.decode.grouped_prefill_from`):
//! its tensor cores take the group's rows as one tile. The trace is rewritten
//! before anything reads its plans' kinds; at 0 (the default) only a group the
//! decode kernel is not stamped at (`fa2::DECODE_GROUPS`; Nemotron-3 Ultra's
//! 32) is rewritten.
//!
//! A decode landing its log-sum-exp (gpt-oss's sinks) joins under
//! `kernel.sdpa.decode.grouped_prefill_lse` unless windowed (too few keys a
//! tile). A plan read across a cluster (`kernel.sdpa.decode.gqa_cluster`)
//! stays a decode: the cluster folds its own split.

use std::collections::BTreeMap;

use kernels_cuda::attn::decode_gqa::Gate;
use poem_ir::{Attn, Fused, Operands, Operation, StructKind, Trace, Ty, ValueId};

/// `trace` with every decode plan of `from` or more query heads a KV head
/// that only plain decodes read planned and read as a prefill.
#[must_use]
pub fn rewrite(trace: Trace, from: u32) -> Trace {
    rewrite_with(trace, from, false, false, Gate::default())
}

/// [`rewrite`], and where `lse` holds, the plans without a window that
/// log-sum-exp decodes read as well, but for those the GQA decode reads across a
/// cluster.
#[must_use]
pub fn rewrite_with(mut trace: Trace, from: u32, lse: bool, lse_window: bool, gate: Gate) -> Trace {
    let unstamped = |group: u32| !kernels_cuda::attn::fa2::DECODE_GROUPS.contains(&group);
    // Decode plans wide enough, by plan value: their KV heads, and whether
    // they slide.
    let mut wide: BTreeMap<ValueId, (u32, bool)> = trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            Operation::Attn(Attn::PlanDecode {
                q_heads,
                kv_heads,
                head_dim,
                window,
                plan,
                ..
            }) if *kv_heads > 0
                && ((from > 0 && q_heads / kv_heads >= from) || unstamped(q_heads / kv_heads)) =>
            {
                // A plan a cluster reads stays a decode: the cluster folds
                // its own slices, with no split to merge.
                let left = window.map_or(-1, |w| i32::try_from(w).unwrap_or(i32::MAX) - 1);
                let clustered =
                    gate.ranks(*head_dim, q_heads / kv_heads, 1, *kv_heads, left, u32::MAX) > 0;
                (!clustered).then_some((*plan, (*kv_heads, window.is_some())))
            }
            _ => None,
        })
        .collect();
    // A plan another op reads (a biased decode, a selection; a log-sum-exp
    // decode unless `lse`, or under a window) stays a decode plan.
    let mut inputs = Vec::new();
    for node in &trace.nodes {
        let attention = match &node.op {
            Operation::Attn(op) => Some(op),
            Operation::Fused(Fused::LseSink { attn, .. }) => Some(&**attn),
            _ => None,
        };
        match attention {
            Some(
                Attn::PlanDecode { .. }
                | Attn::Decode {
                    selection: None,
                    bias: None,
                    lse: None,
                    ..
                },
            ) => continue,
            Some(Attn::Decode {
                plan,
                selection: None,
                bias: None,
                lse: Some(_),
                ..
            }) if lse
                && wide
                    .get(plan)
                    .is_some_and(|(_, slides)| !slides || lse_window) =>
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
            Operation::Attn(op) => op,
            Operation::Fused(Fused::LseSink { attn, .. }) => &mut **attn,
            _ => continue,
        };
        *op = match op.clone() {
            Attn::PlanDecode {
                kv_indptr,
                kv_indices,
                last_page_len,
                kv_len,
                q_heads,
                kv_heads,
                head_dim,
                window,
                plan,
            } if wide.contains_key(&plan) => Attn::PlanPrefill {
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
            Attn::Decode {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                selection: None,
                bias: None,
                selected: None,
                rel: None,
                lse: None,
            } if wide.contains_key(&plan) => Attn::Prefill {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads: wide[&plan].0,
                sm_scale,
                o,
                mask: None,
                selection: None,
                bias: None,
                floor: None,
                causal: true,
                selected: None,
                rel: None,
                lse: None,
            },
            Attn::Decode {
                q,
                plan,
                cache,
                window,
                head_dim,
                sm_scale,
                o,
                lse: Some(lse),
                selection: None,
                bias: None,
                selected: None,
                rel: None,
            } if wide.contains_key(&plan) => Attn::Prefill {
                q,
                plan,
                cache,
                window,
                head_dim,
                kv_heads: wide[&plan].0,
                sm_scale,
                o,
                lse: Some(lse),
                floor: None,
                mask: None,
                selection: None,
                bias: None,
                causal: true,
                selected: None,
                rel: None,
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
                Operation::Attn(op) => op,
                Operation::Fused(Fused::LseSink { attn, .. }) => &**attn,
                _ => continue,
            };
            match op {
                Attn::Decode {
                    window: None,
                    selection: None,
                    bias: None,
                    lse: Some(_),
                    ..
                } => counts.0 += 1,
                Attn::Decode {
                    window: Some(_),
                    selection: None,
                    bias: None,
                    lse: Some(_),
                    ..
                } => counts.1 += 1,
                Attn::Prefill {
                    mask: None,
                    selection: None,
                    bias: None,
                    lse: Some(_),
                    ..
                } => counts.2 += 1,
                _ => {}
            }
        }
        counts
    }

    /// gpt-oss's full-attention layers read single-token rows through the prefill
    /// kernel when log-sum-exp decodes are on; sliding layers keep the decode
    /// kernel. With its sinks fused onto the attentions, and without.
    #[test]
    fn a_sinked_decode_without_a_window_takes_the_prefill_where_the_profile_says() {
        let row = poem_compiler::catalog::deployment("gptoss-120b-bf16-mxfp4-kv-bf16")
            .expect("gpt-oss is in the catalog");
        let plain = row.trace(poem::Platform::Cuda);
        let fused = poem_compiler::fuse::fuse(
            plain.clone(),
            &poem_compiler::fuse::rules::forming(&["fused.lse_sink"]),
        );
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

    /// Nemotron-3 Ultra's 32 query heads a KV head are past the decode
    /// kernel's stamps: its attention reads as a prefill with the profile
    /// silent, and Super's 16 stays a decode.
    #[test]
    fn a_group_the_decode_kernel_is_not_stamped_at_takes_the_prefill() {
        let plans = |id: &str| {
            let trace = poem_compiler::catalog::deployment(id)
                .expect("nemotron is in the catalog")
                .trace(poem::Platform::Cuda);
            let rewritten = rewrite(trace, 0);
            poem_ir::check(&rewritten).expect("the rewritten trace types");
            rewritten
                .nodes
                .iter()
                .fold((0, 0), |(decodes, prefills), node| match &node.op {
                    Operation::Attn(Attn::PlanDecode { .. }) => (decodes + 1, prefills),
                    Operation::Attn(Attn::PlanPrefill { .. }) => (decodes, prefills + 1),
                    _ => (decodes, prefills),
                })
        };
        let (decodes, _) = plans("nemotron3-ultra-mini-bf16-kv-bf16");
        assert_eq!(decodes, 0, "no group-32 plan stays a decode");
        let (decodes, _) = plans("nemotron3-super-mini-bf16-nvfp4a4-e4m3tile128a8-kv-bf16");
        assert!(decodes > 0, "Super's group of 16 stays a decode");
    }

    fn regroups(trace: &Trace) {
        let trace = trace.clone();
        let (full, sliding, prefills) = census(&trace);
        assert!(
            full > 0 && sliding > 0,
            "gpt-oss decodes both kinds of layer"
        );
        let off = Gate::default();
        assert_eq!(rewrite_with(trace.clone(), 4, false, false, off), trace);
        assert_eq!(rewrite_with(trace.clone(), 0, true, false, off), trace);
        let rewritten = rewrite_with(trace.clone(), 4, true, false, off);
        assert_eq!(census(&rewritten), (0, sliding, prefills + full));
        // Rows the tensor-core decode reads across a cluster stay decodes.
        let clustered = Gate {
            on: true,
            cluster: 8,
            ..off
        };
        assert_eq!(
            rewrite_with(trace.clone(), 4, true, false, clustered),
            trace
        );
        poem_ir::check(&rewritten).expect("the rewritten trace types");
    }
}
