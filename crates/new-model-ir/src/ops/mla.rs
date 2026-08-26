use serde::{Deserialize, Serialize};

use crate::{Operands, ValueId};

/// Multi-head latent attention. Same plan discipline as `Attention`: one
/// `Plan` op defines the struct, the four cache-walking variants take it, and
/// `KvAppend` carries its write geometry. The absorb/split variants are pure
/// math and take nothing but tensors.
#[derive(Operands, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Mla {
    /// Defines `Struct(MlaPlan)`, shared by decode and prefill.
    Plan {
        kv_indptr: ValueId,
        kv_indices: ValueId,
        last_page_len: ValueId,
        #[out]
        plan: ValueId,
    },
    /// Splits `kv_a` into the rmsnormed compressed latent and the rope plane.
    Latents {
        kv_a: ValueId,
        weight: ValueId,
        eps: f32,
        kv_lora_rank: u32,
        #[out]
        kv_c: ValueId,
        #[out]
        k_pe: ValueId,
    },
    LatentsRope {
        kv_a: ValueId,
        positions: ValueId,
        weight: ValueId,
        eps: f32,
        kv_lora_rank: u32,
        rope_dim: u32,
        theta: f32,
        #[out]
        kv_c: ValueId,
        #[out]
        k_pe: ValueId,
    },
    SplitQB {
        q_b: ValueId,
        heads: u32,
        nope_dim: u32,
        rope_dim: u32,
        #[out]
        q_nope: ValueId,
        #[out]
        q_pe: ValueId,
    },
    /// Absorbs `kv_b`'s up-projection into q, mapping heads into latent space.
    AbsorbQ {
        q_nope: ValueId,
        kv_b: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        nope_dim: u32,
        v_head_dim: u32,
        #[out]
        q_latent: ValueId,
    },
    AbsorbOut {
        latent: ValueId,
        kv_b: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        v_head_dim: u32,
        nope_dim: u32,
        #[out]
        o: ValueId,
    },
    KvAppend {
        kv_c: ValueId,
        k_pe: ValueId,
        cache: ValueId,
        kv_indices: ValueId,
        positions: ValueId,
    },
    AttentionDecode {
        q: ValueId,
        plan: ValueId,
        q_pe: ValueId,
        cache: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        #[out]
        o: ValueId,
    },
    AttentionPrefill {
        q: ValueId,
        plan: ValueId,
        q_pe: ValueId,
        cache: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        #[out]
        o: ValueId,
    },
    /// Decode over the sparse `selection` produced by `Index::Topk`.
    AttentionDecodeSelected {
        q: ValueId,
        plan: ValueId,
        q_pe: ValueId,
        selection: ValueId,
        cache: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        #[out]
        o: ValueId,
    },
    AttentionPrefillSelected {
        q: ValueId,
        plan: ValueId,
        q_pe: ValueId,
        selection: ValueId,
        cache: ValueId,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        #[out]
        o: ValueId,
    },
}
