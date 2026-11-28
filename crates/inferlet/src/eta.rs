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
        Channel, Fold, Pipeline, ShareScope, StateGeometry, TOKEN_PAD, WorkingSet,
        channel_capacity, frame_size, kv_page_size, readout_spans,
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
}
