use crate::error::Error;
use crate::jit::Ctx;
use crate::tensor::Tensor;

const OP: &str = "attention.ragged";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaggedMask {
    None,
    ReferenceSelfOnly {
        ref_start: Tensor,
    },
    ReferenceTags {
        q_tags: Tensor,
        kv_tags: Tensor,
    },
    /// `q_classes[q] < 0 || kv_classes[kv] < 0 || table[q_class * count + kv_class] != 0`.
    ClassTable {
        q_classes: Tensor,
        kv_classes: Tensor,
        table: Tensor,
        count: u32,
    },
    RelativeBias {
        table: Tensor,
        max_len: u32,
    },
}

#[must_use]
const fn cta_tile_q(head_dim: u32) -> u32 {
    if head_dim >= 256 { 64 } else { 128 }
}

#[must_use]
pub fn padded_work_items(rows: u32, groups: u32, group_size: u32, head_dim: u32) -> u64 {
    let tile = u64::from(cta_tile_q(head_dim));
    let packed = u64::from(rows) * u64::from(group_size.max(1));
    (packed + u64::from(groups) * (tile - 1))
        .div_ceil(tile)
        .max(1)
}

/// Ragged (unpaged) attention over packed q/k/v, the vision towers' form.
/// Not built for AMD yet: it answers `Unsupported` until its kernel lands.
#[allow(clippy::too_many_arguments)]
pub fn ragged(
    ctx: &Ctx,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_indptr: Tensor,
    kv_indptr: Tensor,
    head_dim: u32,
    sm_scale: f32,
    mask: RaggedMask,
    o: &mut Tensor,
) -> Result<(), Error> {
    let _ = (ctx, q, k, v, q_indptr, kv_indptr, head_dim, sm_scale, mask, o);
    Err(Error::Unsupported { op: OP })
}
