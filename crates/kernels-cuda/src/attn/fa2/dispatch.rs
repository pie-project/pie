use super::geometry::Device as FaDevice;
use crate::attn::fa2::params::{PrefillPagedParams, make_prefill_params};
use crate::attn::fa2::{PrefillArm, PrefillPoint};

use super::plan::{DecodePlanCache, PrefillPlanCache};

use crate::attn::fa2::params::{Buffers, DecodePlan, Partials, PrefillPlan};

#[must_use]
#[derive(Clone, Copy, Debug)]
pub enum Fired<D> {
    Whole(D),
    Split(D, Partials),
    Declined(Decline),
}

impl<D> Fired<D> {
    pub fn dispatch(&self) -> Option<&D> {
        match self {
            Self::Whole(d) | Self::Split(d, _) => Some(d),
            Self::Declined(_) => None,
        }
    }

    #[must_use]
    pub fn partials(&self) -> Option<Partials> {
        match *self {
            Self::Split(_, p) => Some(p),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decline {
    Unplanned,
    CaptureVariantUnsupported,
    CaptureSinkMissing,
    Sm90Unported,
}

impl core::fmt::Display for Decline {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            Self::Unplanned => {
                write!(
                    f,
                    "flashinfer fa2 dispatch: the plan cache is empty; plan before firing"
                )
            }
            Self::CaptureVariantUnsupported => write!(
                f,
                "flashinfer fa2 score capture: not instantiated with a logits soft cap \
                 or a sliding window"
            ),
            Self::CaptureSinkMissing => write!(
                f,
                "flashinfer fa2 score capture: requires score_out, score_indptr and a \
                 non-zero window"
            ),
            Self::Sm90Unported => write!(
                f,
                "flashinfer fa2 prefill: this plan is an SM90 plan and the SM90 launcher \
                 is not part of this lattice"
            ),
        }
    }
}


#[must_use]
pub fn decode_plan_of(cache: &DecodePlanCache, device: FaDevice) -> DecodePlan {
    DecodePlan {
        info: cache.plan_info,
        device,
        num_requests: cache.num_requests,
        num_q_heads: cache.num_q_heads,
        num_kv_heads: cache.num_kv_heads,
        head_dim: cache.head_dim,
        page_size: cache.page_size,
        int_base_bytes: cache.int_base_bytes as u64,
        hnd_layout: cache.hnd_layout,
        full_attention_variant: cache.full_attention_variant,
        valid: cache.valid,
    }
}

#[must_use]
pub fn prefill_plan_of(cache: &PrefillPlanCache, device: FaDevice) -> PrefillPlan {
    PrefillPlan {
        info: cache.plan_info,
        device,
        num_requests: cache.num_requests,
        num_q_heads: cache.num_q_heads,
        num_kv_heads: cache.num_kv_heads,
        head_dim: cache.head_dim,
        page_size: cache.page_size,
        cta_tile_q: cache.cta_tile_q,
        window_left: cache.window_left,
        hnd_layout: cache.hnd_layout,
        full_attention_variant: cache.full_attention_variant,
        causal_mask: cache.causal_mask,
        use_sm90: cache.use_sm90,
        valid: cache.valid,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrefillDispatch<P = PrefillPagedParams> {
    pub at: PrefillPoint,
    pub params: P,
}

pub fn prefill(
    cache: &PrefillPlanCache,
    bufs: &Buffers,
    device: FaDevice,
    arm: PrefillArm,
    logits_soft_cap: f32,
    sm_scale: f32,
) -> Fired<PrefillDispatch> {
    fn prefill_plan_usable(cache: &PrefillPlanCache) -> Result<(), Decline> {
        if !cache.valid {
            return Err(Decline::Unplanned);
        }
        if cache.use_sm90 {
            return Err(Decline::Sm90Unported);
        }
        Ok(())
    }

    if let Err(why) = prefill_plan_usable(cache) {
        return Fired::Declined(why);
    }
    let plan = prefill_plan_of(cache, device);

    let (params, partials) = make_prefill_params(&plan, bufs, logits_soft_cap, sm_scale);

    let ready = PrefillDispatch {
        at: super::prefill_at(&plan, arm, params.padded_batch_size),
        params,
    };
    if cache.plan_info.split_kv {
        Fired::Split(ready, partials)
    } else {
        Fired::Whole(ready)
    }
}
