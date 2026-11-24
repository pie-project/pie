use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use eta_dsl::builder::Builder;
use eta_dsl::channel::PutValue;
use eta_dsl::eta::registry::{Axis, OutputDecl};
use eta_dsl::value::ConstData;
use eta_dsl::{Channel as DslChannel, IntoConst, IntoPut, IntoShape, Port, Shape, Stage, Tensor};

use crate::pie::inferlet::channel as wit_channel;
use crate::pie::inferlet::forward as wit_forward;
use crate::pie::inferlet::pipeline as wit_pipeline;
use crate::pie::inferlet::types::Dtype as WitDtype;
use crate::pie::inferlet::working_set as wit_ws;
/// Who a working set shares with ([`WorkingSet::share`]).
pub use crate::pie::inferlet::working_set::ShareScope;

pub use eta_dsl::intrinsics;

pub use eta_dsl::Dtype;
pub use eta_dsl::{
    abs, add, and, broadcast, cast, causal_mask, cos, cummass_le, cumprod, cumsum, div, dtype,
    entropy, entropy_from_logprobs, eq, exp, gather, gather_row, ge, gt, gumbel, gumbel_argmax,
    gumbel_max, indptr, iota, l2norm, le, log, log_softmax, lt, mask_apply, mask_apply_packed,
    masked_argmax, matmul, max_elem, min_elem, mul, ne, neg, normal, not, nucleus_sample, or,
    pivot_threshold, prob_ge, rank_le, recip, reduce_argmax, reduce_max, reduce_min, reduce_sum,
    rem, reshape, rng, row_membership, rsqrt, scalar_gather, scatter_add, scatter_set, select,
    sign, sin, sink_window_mask, sliding_window_mask, softmax, sort_desc, sqrt, sub, top_k,
    transpose,
};

mod channel;
mod forward;
mod pipeline;
mod working_set;

pub mod adapter;
pub mod diffusion;

pub use channel::*;
pub use forward::*;
pub use pipeline::*;
pub use working_set::*;

pub mod shared_prelude {
    pub use super::{
        Channel, Fold, Pipeline, ShareScope, StateGeometry, TOKEN_PAD, WorkingSet, channel_capacity,
        frame_size, kv_page_size, readout_spans,
    };
    pub use crate::{Context, Result, model};
    pub use eta_dsl::Stage;
    pub use eta_dsl::dtype;
    pub use eta_dsl::intrinsics;
    pub use eta_dsl::value::{
        Tensor, abs, and, broadcast, cast, causal_mask, cos, cummass_le, cumprod, cumsum, entropy,
        entropy_from_logprobs, eq, exp, gather, gather_row, ge, gt, gumbel, gumbel_argmax,
        gumbel_max, indptr, iota, l2norm, le, log, log_softmax, lt, mask_apply, mask_apply_packed,
        masked_argmax, matmul, max_elem, min_elem, ne, normal, not, nucleus_sample, or,
        pivot_threshold, prob_ge, rank_le, recip, reduce_argmax, reduce_max, reduce_min,
        reduce_sum, reshape, rng, row_membership, rsqrt, scalar_gather, scatter_add, scatter_set,
        select, sign, sin, sink_window_mask, sliding_window_mask, softmax, sort_desc, sqrt, top_k,
        transpose,
    };
    pub use std::ops::ControlFlow;
}

pub mod prelude {
    pub use super::{ForwardPass, run_ahead, submit_frame};
    pub use crate::eta::shared_prelude::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A seed the host refused is no panic: the channel's next fallible use
    /// (a pass binding it, a submit tracing it, a host take) returns the
    /// host's error, and only the first refusal is kept.
    #[test]
    fn a_refused_seed_is_returned_by_the_next_use() {
        let gid = u64::MAX;
        refuse(gid, "channel seed: channel is closed".to_string());
        refuse(gid, "a later refusal".to_string());
        assert_eq!(
            channel_handle(gid).err().as_deref(),
            Some("channel seed: channel is closed")
        );
        assert!(channel_handle(u64::MAX - 1).is_err(), "never declared");
    }

    fn ready<F: std::future::Future>(f: F) -> F::Output {
        let mut f = std::pin::pin!(f);
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match f.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(out) => out,
            std::task::Poll::Pending => panic!("the driver never waits here"),
        }
    }

    /// A pass is final unless the program states it retractable: the SDK
    /// states its rows final then, and only then, and a refusal names the
    /// opt-out.
    #[test]
    fn a_pass_is_final_unless_it_is_retractable() {
        use std::cell::Cell;
        let stated = Cell::new(0);
        state_final(false, || {
            stated.set(stated.get() + 1);
            Ok(())
        })
        .unwrap();
        assert_eq!(stated.get(), 1, "a default pass states its rows final");
        state_final(true, || {
            stated.set(stated.get() + 1);
            Ok(())
        })
        .unwrap();
        assert_eq!(stated.get(), 1, "a retractable pass states nothing");
        let refused = state_final(false, || Err("final rows: a beam".to_string())).unwrap_err();
        assert!(refused.starts_with("final rows: a beam"), "{refused}");
        assert!(refused.contains("ForwardPass::retractable"), "{refused}");
    }

    /// A run ahead never submits past its budget, whatever its frame, its
    /// window, or where the program breaks; a break takes back only the
    /// fires past the one read (none where nothing ran past it).
    #[test]
    fn a_run_ahead_never_samples_past_its_budget() {
        use std::cell::Cell;
        for r in 1..=4 {
            for window_frames in 1..=4 {
                for budget in 0..=13 {
                    for stop in (0..=budget).map(Some).chain([None]) {
                        let (fired, taken_back, reads) = (Cell::new(0), Cell::new(0), Cell::new(0));
                        let driver = Driver {
                            r,
                            window_frames,
                            budget,
                            submit: |live: usize| {
                                fired.set(fired.get() + live);
                                assert!(fired.get() <= budget, "submitted past the budget");
                                assert!(fired.get() - reads.get() <= window_frames * r);
                                Ok(())
                            },
                            close: || {},
                            retract: |fires: u32| {
                                taken_back.set(fires as usize);
                                Ok(())
                            },
                        };
                        let kept = if budget == 0 {
                            0
                        } else {
                            ready(run_ahead_driven(driver, async || {
                                reads.set(reads.get() + 1);
                                Ok(if Some(reads.get()) == stop {
                                    std::ops::ControlFlow::Break(())
                                } else {
                                    std::ops::ControlFlow::Continue(())
                                })
                            }))
                            .unwrap()
                        };
                        let what = format!("r {r} window {window_frames} budget {budget} stop {stop:?}");
                        assert!(fired.get() <= budget, "{what}");
                        match stop.filter(|&s| s >= 1 && budget > 0) {
                            Some(s) => {
                                assert_eq!(kept, s, "{what}");
                                assert_eq!(taken_back.get(), fired.get() - s, "{what}");
                            }
                            None => {
                                assert_eq!(kept, budget, "{what}");
                                assert_eq!(fired.get(), budget, "{what}");
                                assert_eq!(taken_back.get(), 0, "{what}");
                            }
                        }
                    }
                }
            }
        }
    }
}
