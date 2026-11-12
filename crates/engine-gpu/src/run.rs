use kernels_slang::attn::mla::MlaPlan;
use kernels_slang::attn::sdpa::Tables;
use kernels_slang::{Bank, KvPool, Tensor};

use crate::store::RecurrentPool;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRow {
    Dense(Tensor),

    Planes(Bank),
}

#[derive(Clone, Debug, Default)]
pub struct WeightTable(pub Vec<Option<WeightRow>>);

#[derive(Clone, Debug, Default, Hash)]
pub struct SlotTable(pub Vec<Option<Tensor>>);

#[derive(Clone, Copy, Debug, Hash)]
pub enum CachePool {
    Kv(KvPool),

    Recurrent(RecurrentPool),
}

#[derive(Clone, Debug, Default, Hash)]
pub struct CacheTable(pub Vec<CachePool>);

#[derive(Clone, Copy, Debug, Default, Hash)]
pub struct CacheGeometry {
    pub indptr: Option<Tensor>,

    pub indices: Option<Tensor>,

    pub seq_lens: Option<Tensor>,

    pub last_page_len: Option<Tensor>,

    pub kv_len: Option<Tensor>,

    pub row_valid: Option<Tensor>,

    pub request_of_token: Option<Tensor>,

    pub write_page: Option<Tensor>,

    pub write_offset: Option<Tensor>,
}

#[derive(Clone, Copy, Debug)]
pub struct FireTables {
    pub request_of_token: Tensor,

    pub mask: Tensor,

    pub mask_enabled: Tensor,

    pub mask_stride: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct PoolSlabs {
    pub state_kv: Tensor,

    pub state_score: Tensor,
}

#[derive(Clone, Copy, Debug)]
pub enum StructSlot {
    Decode(Tables),

    Prefill(Tables),

    Mla(MlaPlan),
}
