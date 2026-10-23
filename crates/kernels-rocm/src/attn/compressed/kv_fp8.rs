use crate::error::Error;
use crate::jit::Ctx;
use crate::tensor::{KvPool, Tensor};

pub fn append(
    _ctx: &Ctx,
    _rows: Tensor,
    _pool: &KvPool,
    _write_page: Tensor,
    _write_offset: Tensor,
) -> Result<(), Error> {
    Err(Error::Unsupported {
        op: "attn.kv_append_shared",
    })
}

pub fn store_entries(
    _ctx: &Ctx,
    _entries: Tensor,
    _boundary_pos: Tensor,
    _boundary_req: Tensor,
    _pool: &KvPool,
) -> Result<(), Error> {
    Err(Error::Unsupported {
        op: "attn.pool_kv_append",
    })
}
