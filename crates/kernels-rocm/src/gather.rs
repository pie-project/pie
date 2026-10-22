//! The gather group: embedding lookups, splits and concatenations, row
//! gathers, scatters and permutations, and the n-gram ids they read.

mod concat_cols;
mod embed_concat;
mod embed_e4m3;
mod embed_weighted;
mod fold;
mod ngram;
mod pixels;
pub(crate) mod rows;
mod scatter_live;

pub use concat_cols::concat_cols;
pub use embed_concat::{embed_concat, embed_concat_mlxu4, embed_mlx_affine};
pub use embed_e4m3::embed_concat_mxfp8;
pub use embed_weighted::embed_weighted;
pub use fold::{merge_rows, pool_rows};
pub use ngram::{ngram_ids, ngram_ids_chunked};
pub use pixels::{grid_taps, pixels};
pub use rows::{
    embed, embed_scale_add, embed_scale_add_select, embed_vocab_shard, gather_rows, pack_rows,
    scatter_rows, select, split_q_gate, split_qkv, split_qkv_padded, split_rows, unpack_rows,
};
pub use scatter_live::scatter_live_rows;
