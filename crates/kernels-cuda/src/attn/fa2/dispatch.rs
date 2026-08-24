use super::geometry::Device as FaDevice;

use super::plan::{DecodePlanCache, PrefillPlanCache};

use crate::attn::fa2::params::{DecodePlan, PrefillPlan};

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
