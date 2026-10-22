mod embed;
mod ngram;
mod pixels;
mod rows;

pub(crate) use embed::comm_rank;
pub use embed::{
    embed, embed_concat, embed_concat_e4m3, embed_concat_mlxu4, embed_concat_mxfp8,
    embed_mlx_affine, embed_scale_add, embed_scale_add_select, embed_vocab_shard, embed_weighted,
};
pub use ngram::{ngram_ids, ngram_ids_chunked};
pub use pixels::{grid_taps, pixels};
#[cfg(feature = "cuda")]
pub(crate) use rows::gather_width_concat;
pub use rows::{
    concat_cols, gather_rows, merge_rows, pack_rows, pool_rows, scatter_live_rows, scatter_rows,
    select, split_q_gate, split_qkv, split_qkv_padded, split_rows, unpack_rows,
};
