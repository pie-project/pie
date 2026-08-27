//! The `CustomCuda` family: the escape hatch for a plane-specific mega-kernel
//! that no portable decomposition covers.

use super::*;

/// Splits packed qkv, head-norms q and k, ropes them, norms v, and appends
/// k/v in one pass; `q` is the only tensor left over. `positions` feeds
/// the rope math; `write_page`/`write_offset` address the append.
pub fn qkv_fused_qknorm_rope_vnorm_write(
    packed: &Value,
    q_norm: &Weight,
    q_norm_eps: f32,
    k_norm: &Weight,
    k_norm_eps: f32,
    kv_heads: u32,
    head_dim: u32,
    pages: ValueId,
    write_page: &Value,
    write_offset: &Value,
    theta: f32,
    positions: &Value,
) -> Value {
    let r = packed.rec();
    let q_width = packed.width() - 2 * u64::from(kv_heads) * u64::from(head_dim);
    let q = r.fresh(tensor(packed.rows(), q_width, packed.dtype()));
    r.push(
        CustomCuda::QkvFusedQknormRopeVnormWrite {
            packed: packed.id(),
            positions: positions.id(),
            q_norm_weight: r.weight(q_norm),
            q_norm_eps,
            k_norm_weight: r.weight(k_norm),
            k_norm_eps,
            cache: pages,
            write_page: write_page.id(),
            write_offset: write_offset.id(),
            kv_heads,
            head_dim,
            theta,
            q: q.id(),
        },
        &[packed, positions, write_page, write_offset],
    );
    q
}
