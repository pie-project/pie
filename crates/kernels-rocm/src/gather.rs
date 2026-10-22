//! The gather group's surface, as kernels-cuda spells it. The modules behind
//! it keep kernels-rocm's layout until its kernel tree takes the convention.

pub use crate::attn_ple::{ngram_ids, ngram_ids_chunked};
pub use crate::layout::{
    embed, embed_scale_add, embed_scale_add_select, embed_vocab_shard, gather_rows, pack_rows,
    scatter_rows, select, split_q_gate, split_qkv, split_qkv_padded, split_rows, unpack_rows,
};
pub use crate::layout_concat_cols::concat_cols;
pub use crate::layout_embed_concat::{embed_concat, embed_concat_mlxu4, embed_mlx_affine};
pub use crate::layout_embed_e4m3::embed_concat_mxfp8;
pub use crate::layout_embed_weighted::embed_weighted;
pub use crate::layout_fold::{merge_rows, pool_rows};
pub use crate::layout_pixels::{grid_taps, pixels};
pub use crate::layout_scatter_live::scatter_live_rows;
