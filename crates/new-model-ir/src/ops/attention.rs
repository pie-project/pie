use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

/// Paged attention. Plans are explicit ops: `PlanDecode`/`PlanPrefill` define
/// `Struct` values from declared geometry inputs, and every variant that walks
/// the cache takes the plan it was built from — `cache` is the pool pointer,
/// nothing more. The append ops carry their write geometry the same way.
#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Attention {
    /// Defines `Struct(AttnDecodePlan)`. Host work; runs in the prepare phase.
    PlanDecode {
        kv_indptr: ValueId,
        kv_indices: ValueId,
        last_page_len: ValueId,
        #[out]
        plan: ValueId,
    },
    /// Defines `Struct(AttnPrefillPlan)`.
    PlanPrefill {
        kv_indptr: ValueId,
        kv_indices: ValueId,
        last_page_len: ValueId,
        #[out]
        plan: ValueId,
    },
    Decode {
        q: ValueId,
        plan: ValueId,
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
        #[out]
        o: ValueId,
    },
    Prefill {
        q: ValueId,
        plan: ValueId,
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        #[out]
        o: ValueId,
    },
    /// Prefill against a query-provided mask instead of the causal one.
    Masked {
        q: ValueId,
        plan: ValueId,
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
        #[out]
        o: ValueId,
    },
    DecodeLse {
        q: ValueId,
        plan: ValueId,
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        sm_scale: f32,
        #[out]
        o: ValueId,
        #[out]
        lse: ValueId,
    },
    PrefillLse {
        q: ValueId,
        plan: ValueId,
        cache: ValueId,
        window: Option<u32>,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        #[out]
        o: ValueId,
        #[out]
        lse: ValueId,
    },
    /// Folds attention-sink mass into `o` using its log-sum-exp.
    Sink {
        o: ValueId,
        lse: ValueId,
        sink: ValueId,
        head_dim: u32,
        #[out(alias = o)]
        o_out: ValueId,
    },
    MergeLse {
        o1: ValueId,
        lse1: ValueId,
        o2: ValueId,
        lse2: ValueId,
        heads: u32,
        head_dim: u32,
        #[out]
        o: ValueId,
        #[out]
        lse: ValueId,
    },
    LogitSoftcap {
        x: ValueId,
        cap: f32,
        #[out(alias = x)]
        x_out: ValueId,
    },
    KvAppend {
        k: ValueId,
        v: ValueId,
        cache: ValueId,
        kv_indices: ValueId,
        positions: ValueId,
    },
    /// Appends one plane shared as both k and v.
    KvAppendShared {
        plane: ValueId,
        cache: ValueId,
        kv_indices: ValueId,
        positions: ValueId,
    },
}
