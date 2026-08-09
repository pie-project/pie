use kernels::LaunchRule as Rule;

/// The fire-time quantities a CUDA launch rule may read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dims {
    /// Rows the rectangle covers — tokens, requests or routed slots,
    pub rows: u32,
    /// Elements per row of the launch's last widthed operand — its output.
    pub width: u32,
    /// Elements per row of its first widthed operand — its input. Read by
    pub in_width: u32,
    /// Query heads — the head count of the tensor a per-head launch reads
    pub q_heads: u32,
    /// Key/value heads — and, for the rules whose operand is not q, THE HEAD
    pub kv_heads: u32,
    /// Elements per head — `head_dim`, `V_d`, `K_d`, or a norm group's width,
    pub head_dim: u32,
    /// The per-head width THE STATEMENT NAMED, and **zero means it named
    pub stated_head_dim: u32,
    /// Channels a partial rope rotates.
    pub rotary_dims: u32,
    /// Experts the router scores. Read by [`Rule::RouterSort`], which sizes
    pub n_experts: u32,
    /// Experts each token routes to. Read by [`Rule::RoutedQmv`],
    pub experts_per_token: u32,
    /// Requests the fire covers — the CSR's `R`, the extent
    pub requests: u32,
    /// AltUp residual streams — `K`, the rank of the parallel residual the
    pub altup_streams: u32,
}

/// A launch, in CUDA's spelling: blocks, threads per block, dynamic shared
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// Blocks per axis.
    pub grid: [u32; 3],
    /// Threads per block per axis.
    pub block: [u32; 3],
    /// Dynamic shared memory, in bytes.
    pub smem: u32,
}

/// Why a rule could not produce a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ungeometric {
    /// The row states no rule, so nothing can be dispatched from it. Drift,
    Unstated,
    /// The rule is real and this backend has not ported its arithmetic.
    Unported(Rule),
    /// A launch over an empty extent.
    Empty,
}

/// Threads per block for the pointwise passes.
const BLOCK: u32 = 256;

/// The widest block CUDA will launch.
const MAX_BLOCK: u32 = 1024;

/// Threads per warp — the unit `block_sum`'s shared scratch is counted in.
const WARP: u32 = 32;

/// One `float`, in bytes. Spelled once because six of the rules below turn a
const FLOAT: u32 = 4;

/// The pad/strip block, `attn/head_dim_pad.cuh`'s `kPadBlock`.
const PAD_BLOCK: u32 = 128;

/// The narrowest and widest block `attn_sink_rescale`'s launcher will build
const SINK_BLOCK_MIN: u32 = WARP;
/// See [`SINK_BLOCK_MIN`].
const SINK_BLOCK_MAX: u32 = 128;

/// The counting sort's block, `moe/moe_dispatch.cu`'s `BS`. One block, and
const SORT_BLOCK: u32 = MAX_BLOCK;

/// The softmax router's block, `moe/topk_softmax.cuh`'s `kSoftmaxBlock`.
const ROUTER_BLOCK: u32 = 64;

/// The largest half-head `rope/rope.cu` will cache sin/cos pairs for —
const ROPE_MAX_CACHED_PAIRS: u32 = 4096;

/// The recurrence's block, `ssm/gated_delta_net.cu`'s `constexpr int BLOCK`.
const SCAN_BLOCK: u32 = 128;

/// The prefill convolution's block, `ssm/causal_conv1d.cu`'s
const CONV_BLOCK: u32 = 64;

/// Elements one `bf16_to_narrow` load moves — `quant/dequant_wna16.cu`'s
const SLAB_VEC: u32 = 8;

/// The grid cap [`slab`] launches under — `std::min<long long>(..., 1024)`.
const SLAB_GRID_MAX: u32 = 1024;

/// The square tile [`tile16`] walks a rectangle in — `dim3 B2(16,16)`, which
const TILE: u32 = 16;

/// Warps a [`warp_tiled_scan`] block splits the value channels over —
const SCAN_WARPS: u32 = 4;

/// The block [`per_row_narrow`] launches — `vision/gemma4_audio.cu`'s literal
const LAYERNORM_BLOCK: u32 = 128;

/// The reference paged attention's block — `attn/attention_naive_paged.cu`'s
const PAGED_BLOCK: u32 = 128;

/// AltUp's block — the deleted `csrc/src/norm/altup.cu`'s
const ALTUP_BLOCK: u32 = 128;

/// One block per row, [`BLOCK`] wide, with scratch for the warp combine.
fn rms(rows: u32) -> Launch {
    Launch {
        grid: [rows, 1, 1],
        block: [BLOCK, 1, 1],
        smem: (BLOCK / WARP) * 4,
    }
}

/// One block per row PER HEAD — [`rms`]' grid with the per-head reading of
fn rows_per_head(rows: u32, width: u32, stated_head_dim: u32) -> Result<Launch, Ungeometric> {
    let blocks = if stated_head_dim == 0 {
        rows
    } else {
        if width == 0 || !width.is_multiple_of(stated_head_dim) {
            return Err(Ungeometric::Empty);
        }
        rows.checked_mul(width / stated_head_dim).ok_or(Ungeometric::Empty)?
    };
    Ok(Launch { grid: [blocks, 1, 1], block: [BLOCK, 1, 1], smem: 0 })
}

/// Flat pointwise: `n` elements, [`BLOCK`] per block, rounded up.
fn elementwise(n: u32) -> Launch {
    Launch {
        grid: [n.div_ceil(BLOCK), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// Flat pointwise over what the launch READS — `rows * in_width` elements,
fn elementwise_in(n: u32) -> Launch {
    Launch {
        grid: [n.div_ceil(BLOCK), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// `ceil(rows / 256)` blocks of [`BLOCK`] — ONE THREAD per row.
fn rows_flat(rows: u32) -> Launch {
    Launch {
        grid: [rows.div_ceil(BLOCK), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// A capped grid-stride slab — `min(ceil(units / 256), 1024)` blocks of
fn slab(n: u32) -> Launch {
    let units = if n >= SLAB_VEC { n / SLAB_VEC } else { n };
    Launch {
        grid: [units.div_ceil(BLOCK).clamp(1, SLAB_GRID_MAX), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// Pointwise with the row on its own grid axis.
fn elementwise_rows(rows: u32, width: u32) -> Launch {
    Launch {
        grid: [rows, width.div_ceil(BLOCK), 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// One block per row, as wide as the row, rounded up to a warp and capped.
fn route_rows(rows: u32, width: u32) -> Launch {
    Launch {
        grid: [rows, 1, 1],
        block: [width.div_ceil(WARP).max(1).saturating_mul(WARP).min(MAX_BLOCK), 1, 1],
        smem: 0,
    }
}

/// One block per row, a fixed [`BLOCK`] wide, nothing shared — the scatter
fn per_row(rows: u32) -> Launch {
    Launch { grid: [rows, 1, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// One block per row, [`LAYERNORM_BLOCK`] wide, nothing shared — [`per_row`]'s
fn per_row_narrow(rows: u32) -> Launch {
    Launch { grid: [rows, 1, 1], block: [LAYERNORM_BLOCK, 1, 1], smem: 0 }
}

/// **One block**, [`BLOCK`] wide, nothing shared — the grid is a literal.
fn single() -> Launch {
    Launch { grid: [1, 1, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// [`single`] at ONE WARP — `<<<1, 32>>>`.
fn single_warp() -> Launch {
    Launch { grid: [1, 1, 1], block: [WARP, 1, 1], smem: 0 }
}

/// One block per REQUEST, [`BLOCK`] wide, nothing shared.
fn per_request(requests: u32) -> Launch {
    Launch { grid: [requests, 1, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// One block per row, [`BLOCK`] wide, with one float of shared scratch per
fn row_scores(rows: u32) -> Result<Launch, Ungeometric> {
    let smem = rows.checked_mul(FLOAT).ok_or(Ungeometric::Empty)?;
    Ok(Launch { grid: [rows, 1, 1], block: [BLOCK, 1, 1], smem })
}

/// One block per COLUMN, [`CONV_BLOCK`] wide, the rows walked inside the
fn per_channel(width: u32) -> Launch {
    Launch { grid: [width, 1, 1], block: [CONV_BLOCK, 1, 1], smem: 0 }
}

/// One block per (head, row), 128 threads — the head-dim pad and its inverse.
fn per_head(rows: u32, heads: u32) -> Launch {
    Launch { grid: [heads, rows, 1], block: [PAD_BLOCK, 1, 1], smem: 0 }
}

/// One block per (row, head), as wide as a head — the per-head pointwise
fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    Launch {
        grid: [rows, heads, 1],
        block: [head_dim.clamp(SINK_BLOCK_MIN, SINK_BLOCK_MAX), 1, 1],
        smem: 0,
    }
}

/// One block per (row, head), 256 threads — the gated and per-head norms.
fn gated_rms(rows: u32, heads: u32) -> Launch {
    Launch { grid: [rows, heads, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// One block per (row, value head), [`SCAN_BLOCK`] threads, with two key
fn recurrent_scan(rows: u32, heads: u32, head_dim: u32) -> Result<Launch, Ungeometric> {
    let smem = head_dim
        .checked_mul(2)
        .and_then(|floats| floats.checked_mul(FLOAT))
        .ok_or(Ungeometric::Empty)?;
    Ok(Launch { grid: [rows, heads, 1], block: [SCAN_BLOCK, 1, 1], smem })
}

/// The recurrence tiled by warps over the VALUE width — `dim3(rows, heads,
fn warp_tiled_scan(rows: u32, heads: u32, value_width: u32) -> Launch {
    Launch {
        grid: [rows, heads, value_width.div_ceil(SCAN_WARPS)],
        block: [SCAN_WARPS * WARP, 1, 1],
        smem: 0,
    }
}

/// One block per (query head, row), 256 threads, with the KV extent in
fn sdpa_vector(rows: u32, q_heads: u32) -> Result<Launch, Ungeometric> {
    let smem = rows
        .checked_add(BLOCK)
        .and_then(|floats| floats.checked_mul(FLOAT))
        .ok_or(Ungeometric::Empty)?;
    Ok(Launch { grid: [q_heads, rows, 1], block: [BLOCK, 1, 1], smem })
}

/// Pointwise over the launch's INPUT width with the row on `grid.y` — one
fn split_packed(rows: u32, in_width: u32) -> Launch {
    Launch { grid: [in_width.div_ceil(BLOCK), rows, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// The rotation: rows on `grid.x`, packed heads on `grid.y`, a cached
fn rope(rows: u32, q_heads: u32, kv_heads: u32, head_dim: u32) -> Launch {
    let half = head_dim / 2;
    let heads_per_block = if half >= BLOCK { 1 } else { BLOCK / half };
    let cache_pairs = if half <= ROPE_MAX_CACHED_PAIRS { half } else { 0 };
    Launch {
        grid: [rows, (q_heads + kv_heads).div_ceil(heads_per_block), 1],
        block: [BLOCK, 1, 1],
        smem: cache_pairs * 2 * FLOAT,
    }
}

/// One block per row, [`ROUTER_BLOCK`] wide — the router's top-k.
fn router_lane(rows: u32) -> Launch {
    Launch { grid: [rows, 1, 1], block: [ROUTER_BLOCK, 1, 1], smem: 0 }
}

/// ONE block, whatever the rows, with the sort's counters in shared memory.
fn router_sort(n_experts: u32) -> Result<Launch, Ungeometric> {
    let words = n_experts
        .checked_mul(3)
        .and_then(|counters| counters.checked_add(34))
        .ok_or(Ungeometric::Empty)?;
    let smem = words.checked_mul(FLOAT).ok_or(Ungeometric::Empty)?;
    Ok(Launch { grid: [1, 1, 1], block: [SORT_BLOCK, 1, 1], smem })
}

/// One block per (route, warp-tile of the output width) — `dim3(rows *
fn routed_qmv(routes: u32, width: u32) -> Launch {
    Launch {
        grid: [routes, width.div_ceil(BLOCK / WARP), 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// The MXFP4 decode block — `quant/dequant_fp4.cu:39`'s
const MXFP4_DECODE_BLOCK: u32 = 128;

/// Output rows one WARP of the MXFP4 decode GEMVs owns — the template
const MXFP4_ROWS_PER_WARP: u32 = 4;

/// [`routed_qmv`]'s axes at a QUAD tile — `dim3(routes, ceil(width / 16))` at
fn routed_qmv_quad(routes: u32, width: u32) -> Launch {
    let tile = (MXFP4_DECODE_BLOCK / WARP) * MXFP4_ROWS_PER_WARP;
    Launch {
        grid: [routes, width.div_ceil(tile), 1],
        block: [MXFP4_DECODE_BLOCK, 1, 1],
        smem: 0,
    }
}

/// [`routed_qmv`]'s two axes swapped — `dim3(ceil(width / 8), routes)` at
fn routed_qmv_transposed(routes: u32, width: u32) -> Launch {
    Launch {
        grid: [width.div_ceil(BLOCK / WARP), routes, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    }
}

/// A [`TILE`]-square block over a rectangle — `ceil(width / 16)` by
fn tile16(rows: u32, width: u32) -> Launch {
    Launch {
        grid: [width.div_ceil(TILE), rows.div_ceil(TILE), 1],
        block: [TILE, TILE, 1],
        smem: 0,
    }
}

/// One WARP per (head, row), heads on `grid.y` and rows on `grid.z` —
fn axial_rope(rows: u32, heads: u32) -> Launch {
    Launch { grid: [1, heads, rows], block: [WARP, 1, 1], smem: 0 }
}

/// The reference paged attention's PREFILL launch — `dim3(requests, rows,
fn paged_scores(requests: u32, rows: u32, q_heads: u32, head_dim: u32) -> Launch {
    Launch {
        grid: [requests, rows, q_heads],
        block: [PAGED_BLOCK, 1, 1],
        smem: (head_dim + PAGED_BLOCK) * FLOAT,
    }
}

/// The same family's DECODE launch — `dim3(rows, q_heads)` at
fn paged_scores_decode(rows: u32, q_heads: u32, head_dim: u32) -> Launch {
    Launch {
        grid: [rows, q_heads, 1],
        block: [PAGED_BLOCK, 1, 1],
        smem: (head_dim + PAGED_BLOCK) * FLOAT,
    }
}

/// MLA's fused prepare — `dim3(rows, 1 + ceil(q_heads / heads_per_block))` at
fn mla_prepare(rows: u32, q_heads: u32, rotary_dims: u32) -> Launch {
    let half = rotary_dims / 2;
    let heads_per_block = if half >= BLOCK { 1 } else { BLOCK / half };
    let q_blocks = q_heads.div_ceil(heads_per_block);
    Launch { grid: [rows, 1 + q_blocks, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// One block per (row, packed head) — `dim3(rows, q_heads + kv_heads)` at
fn rows_packed_heads(rows: u32, packed_heads: u32) -> Launch {
    Launch { grid: [rows, packed_heads, 1], block: [BLOCK, 1, 1], smem: 0 }
}

/// [`rows_packed_heads`] at [`SCAN_BLOCK`] threads — the DECODE form.
fn rows_packed_heads_narrow(rows: u32, packed_heads: u32) -> Launch {
    Launch { grid: [rows, packed_heads, 1], block: [SCAN_BLOCK, 1, 1], smem: 0 }
}

/// One WARP per (row, packed head), flattened — `ceil(rows * packed_heads /
fn warp_packed_heads(rows: u32, packed_heads: u32) -> Result<Launch, Ungeometric> {
    let units = rows.checked_mul(packed_heads).ok_or(Ungeometric::Empty)?;
    Ok(Launch {
        grid: [units.div_ceil(BLOCK / WARP), 1, 1],
        block: [BLOCK, 1, 1],
        smem: 0,
    })
}

/// A third grid axis over an ALTUP STREAM count — `dim3(rows, streams,
fn altup_streams(rows: u32, streams: u32, hidden: u32) -> Launch {
    Launch {
        grid: [rows, streams, hidden.div_ceil(ALTUP_BLOCK)],
        block: [ALTUP_BLOCK, 1, 1],
        smem: 0,
    }
}

/// A head geometry, checked.
const fn headed(heads: u32, head_dim: u32) -> Result<(), Ungeometric> {
    if heads == 0 || head_dim == 0 {
        return Err(Ungeometric::Empty);
    }
    Ok(())
}

/// The launch `rule` produces for `dims`.
pub fn eval(rule: Rule, dims: Dims) -> Result<Launch, Ungeometric> {
    if dims.rows == 0 {
        return Err(Ungeometric::Empty);
    }
    Ok(match rule {
        Rule::Unstated => return Err(Ungeometric::Unstated),
        Rule::Rms => rms(dims.rows),
        Rule::Elementwise => {
            let n = dims.rows.checked_mul(dims.width).ok_or(Ungeometric::Empty)?;
            if n == 0 {
                return Err(Ungeometric::Empty);
            }
            elementwise(n)
        }
        Rule::ElementwiseRows => {
            if dims.width == 0 {
                return Err(Ungeometric::Empty);
            }
            elementwise_rows(dims.rows, dims.width)
        }
        Rule::RouteRows => {
            if dims.width == 0 {
                return Err(Ungeometric::Empty);
            }
            route_rows(dims.rows, dims.width)
        }
        Rule::PerHead => {
            headed(dims.kv_heads, dims.head_dim)?;
            per_head(dims.rows, dims.kv_heads)
        }
        Rule::PerHeadElementwise => {
            headed(dims.q_heads, dims.head_dim)?;
            per_head_elementwise(dims.rows, dims.q_heads, dims.head_dim)
        }
        Rule::GatedRms => {
            headed(dims.kv_heads, dims.head_dim)?;
            gated_rms(dims.rows, dims.kv_heads)
        }
        Rule::SdpaVector => {
            if dims.q_heads == 0 {
                return Err(Ungeometric::Empty);
            }
            sdpa_vector(dims.rows, dims.q_heads)?
        }
        Rule::SplitPacked => {
            if dims.in_width == 0 {
                return Err(Ungeometric::Empty);
            }
            split_packed(dims.rows, dims.in_width)
        }
        Rule::Rope => {
            headed(dims.q_heads + dims.kv_heads, dims.head_dim)?;
            if dims.head_dim < 2 {
                return Err(Ungeometric::Empty);
            }
            rope(dims.rows, dims.q_heads, dims.kv_heads, dims.head_dim)
        }
        Rule::RouterLane => router_lane(dims.rows),
        Rule::RouterSort => {
            if dims.n_experts == 0 {
                return Err(Ungeometric::Empty);
            }
            router_sort(dims.n_experts)?
        }
        Rule::RecurrentScan => {
            headed(dims.kv_heads, dims.head_dim)?;
            recurrent_scan(dims.rows, dims.kv_heads, dims.head_dim)?
        }
        Rule::PerRow => per_row(dims.rows),
        Rule::PerRowNarrow => per_row_narrow(dims.rows),
        Rule::Single => single(),
        Rule::SingleWarp => single_warp(),
        Rule::PerRequest => {
            if dims.requests == 0 {
                return Err(Ungeometric::Empty);
            }
            per_request(dims.requests)
        }
        Rule::PerChannel => {
            if dims.width == 0 {
                return Err(Ungeometric::Empty);
            }
            per_channel(dims.width)
        }
        Rule::ElementwiseIn => {
            let n = dims.rows.checked_mul(dims.in_width).ok_or(Ungeometric::Empty)?;
            if n == 0 {
                return Err(Ungeometric::Empty);
            }
            elementwise_in(n)
        }
        Rule::RowScores => row_scores(dims.rows)?,
        Rule::RowsPerHead => rows_per_head(dims.rows, dims.width, dims.stated_head_dim)?,
        Rule::RowsFlat => rows_flat(dims.rows),
        Rule::Slab => {
            let n = dims.rows.checked_mul(dims.width).ok_or(Ungeometric::Empty)?;
            if n == 0 {
                return Err(Ungeometric::Empty);
            }
            slab(n)
        }
        Rule::RoutedQmv => {
            if dims.width == 0 || dims.experts_per_token == 0 {
                return Err(Ungeometric::Empty);
            }
            let routes =
                dims.rows.checked_mul(dims.experts_per_token).ok_or(Ungeometric::Empty)?;
            routed_qmv(routes, dims.width)
        }
        Rule::RoutedQmvQuad => {
            if dims.width == 0 || dims.experts_per_token == 0 {
                return Err(Ungeometric::Empty);
            }
            if dims.width % dims.experts_per_token != 0 {
                return Err(Ungeometric::Empty);
            }
            let routes =
                dims.rows.checked_mul(dims.experts_per_token).ok_or(Ungeometric::Empty)?;
            routed_qmv_quad(routes, dims.width / dims.experts_per_token)
        }
        Rule::Tile16 => {
            if dims.width == 0 {
                return Err(Ungeometric::Empty);
            }
            tile16(dims.rows, dims.width)
        }
        Rule::AxialRope => {
            headed(dims.kv_heads, dims.head_dim)?;
            axial_rope(dims.rows, dims.kv_heads)
        }
        Rule::WarpTiledScan => {
            headed(dims.kv_heads, dims.head_dim)?;
            if dims.width == 0 || !dims.width.is_multiple_of(dims.kv_heads) {
                return Err(Ungeometric::Empty);
            }
            warp_tiled_scan(dims.rows, dims.kv_heads, dims.width / dims.kv_heads)
        }
        Rule::PagedScores => {
            headed(dims.q_heads, dims.head_dim)?;
            if dims.requests == 0 {
                return Err(Ungeometric::Empty);
            }
            paged_scores(dims.requests, dims.rows, dims.q_heads, dims.head_dim)
        }
        Rule::PagedScoresDecode => {
            headed(dims.q_heads, dims.head_dim)?;
            paged_scores_decode(dims.rows, dims.q_heads, dims.head_dim)
        }
        Rule::MlaPrepare => {
            if dims.q_heads == 0 || dims.rotary_dims < 2 || !dims.rotary_dims.is_multiple_of(2)
            {
                return Err(Ungeometric::Empty);
            }
            mla_prepare(dims.rows, dims.q_heads, dims.rotary_dims)
        }
        Rule::RowsPackedHeads => {
            headed(dims.q_heads, dims.head_dim)?;
            rows_packed_heads(dims.rows, packed_heads(&dims)?)
        }
        Rule::RowsPackedHeadsNarrow => {
            headed(dims.q_heads, dims.head_dim)?;
            rows_packed_heads_narrow(dims.rows, packed_heads(&dims)?)
        }
        Rule::WarpPackedHeads => {
            headed(dims.q_heads, dims.head_dim)?;
            warp_packed_heads(dims.rows, packed_heads(&dims)?)?
        }
        Rule::RoutedQmvTransposed => {
            if dims.width == 0 || dims.experts_per_token == 0 {
                return Err(Ungeometric::Empty);
            }
            let routes =
                dims.rows.checked_mul(dims.experts_per_token).ok_or(Ungeometric::Empty)?;
            routed_qmv_transposed(routes, dims.width)
        }
        Rule::AltUpStreams => {
            if dims.altup_streams == 0
                || dims.width == 0
                || !dims.width.is_multiple_of(dims.altup_streams)
            {
                return Err(Ungeometric::Empty);
            }
            altup_streams(dims.rows, dims.altup_streams, dims.width / dims.altup_streams)
        }
        other => return Err(Ungeometric::Unported(other)),
    })
}

/// `q_heads + kv_heads`, checked — the axis the three fused QKV rules open.
fn packed_heads(dims: &Dims) -> Result<u32, Ungeometric> {
    if dims.q_heads == 0 || dims.kv_heads == 0 {
        return Err(Ungeometric::Empty);
    }
    dims.q_heads.checked_add(dims.kv_heads).ok_or(Ungeometric::Empty)
}

#[cfg(test)]
mod tests {
    use super::{Dims, Launch, Rule, Ungeometric, eval};

    /// gemma-3n's shape: four AltUp streams, 2048 hidden, sixteen tokens.
    const T: u32 = 16;
    const H: u32 = 2048;
    const K: u32 = 4;

    /// The rules this backend has written the arithmetic for.
    const PORTED: &[Rule] = &[
        Rule::Rms,
        Rule::Elementwise,
        Rule::ElementwiseRows,
        Rule::RouteRows,
        Rule::PerHead,
        Rule::PerHeadElementwise,
        Rule::GatedRms,
        Rule::SdpaVector,
        Rule::SplitPacked,
        Rule::Rope,
        Rule::RouterLane,
        Rule::RouterSort,
        Rule::RecurrentScan,
        Rule::PerRow,
        Rule::PerChannel,
        Rule::ElementwiseIn,
        Rule::RowScores,
        Rule::RowsPerHead,
        Rule::RowsFlat,
        Rule::Slab,
        Rule::RoutedQmv,
        Rule::Tile16,
        Rule::AxialRope,
        Rule::WarpTiledScan,
        Rule::PerRowNarrow,
        Rule::PagedScores,
        Rule::PagedScoresDecode,
        Rule::MlaPrepare,
        Rule::RowsPackedHeads,
        Rule::RowsPackedHeadsNarrow,
        Rule::WarpPackedHeads,
        Rule::RoutedQmvTransposed,
        Rule::AltUpStreams,
        Rule::RoutedQmvQuad,
        Rule::Single,
        Rule::SingleWarp,
        Rule::PerRequest,
    ];

    /// A rectangle with every field filled, because a head-shaped rule
    fn dims(rows: u32, width: u32) -> Dims {
        Dims {
            rows,
            width,
            in_width: width,
            q_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            stated_head_dim: 0,
            rotary_dims: 64,
            n_experts: 128,
            experts_per_token: 8,
            requests: 4,
            altup_streams: K,
        }
    }

    /// Every rule the JIT's rows state evaluates. The rows and the arithmetic
    #[test]
    fn every_stated_rule_is_ported() {
        for k in crate::unit::rows() {
            let d = dims(T, H);
            assert!(
                !matches!(eval(k.sig.launch, d), Err(Ungeometric::Unported(_))),
                "{} states {:?}, which this runtime has not ported",
                k.sig.symbol,
                k.sig.launch
            );
        }
    }

    /// **Every variant of the vocabulary is answered, not just the ones some
    #[test]
    fn every_rule_in_the_vocabulary_is_answered() {
        for &rule in Rule::ALL {
            match eval(rule, dims(T, H)) {
                Ok(_) => assert!(
                    PORTED.contains(&rule),
                    "{rule:?} evaluates and is not on PORTED — either the \
                     arithmetic was added without saying so, or it is a guess"
                ),
                Err(Ungeometric::Unported(named)) => {
                    assert_eq!(named, rule, "a refusal must name the rule it refused");
                    assert!(
                        !PORTED.contains(&rule),
                        "{rule:?} is on PORTED and refuses; the list is stale"
                    );
                }
                Err(Ungeometric::Unstated) => {
                    assert_eq!(rule, Rule::Unstated, "only the empty row is unstated");
                }
                Err(Ungeometric::Empty) => panic!(
                    "{rule:?} calls a {T}x{H} rectangle empty, which no rule may: \
                     `Empty` is for a rectangle that collapsed, not for a shape \
                     a rule dislikes"
                ),
            }
        }
    }

    /// The reduction pair reproduces `compute_rms_bf16`'s launcher exactly:
    #[test]
    fn rms_reproduces_the_cpp_launcher() {
        assert_eq!(
            eval(Rule::Rms, dims(T, H)),
            Ok(Launch { grid: [T, 1, 1], block: [256, 1, 1], smem: 32 })
        );
    }

    /// `tanh_kernel<<<(numel + 255) / 256, 256, 0, stream>>>`, where the
    #[test]
    fn elementwise_reproduces_the_cpp_launcher() {
        let numel = T * H;
        assert_eq!(
            eval(Rule::Elementwise, dims(T, H)),
            Ok(Launch { grid: [numel.div_ceil(256), 1, 1], block: [256, 1, 1], smem: 0 })
        );
    }

    /// `mean_streams_bf16` launched `dim3(T, (H + 127) / 128)` with 128
    #[test]
    fn elementwise_rows_covers_every_channel() {
        let l = eval(Rule::ElementwiseRows, dims(T, H)).expect("rule evaluates");
        assert_eq!(l.grid[0], T, "one block per row");
        assert!(
            l.grid[1] * l.block[0] >= H,
            "grid.y ({}) x block ({}) must cover H ({H})",
            l.grid[1],
            l.block[0]
        );
        assert!((l.grid[1] - 1) * l.block[0] < H);
    }

    /// `unpack_predict_coefs_kernel<<<T, K * K>>>` — one block per row, as
    #[test]
    fn route_rows_covers_the_row_it_is_given() {
        let predict = eval(Rule::RouteRows, dims(T, K * K)).expect("rule evaluates");
        assert_eq!(predict.grid, [T, 1, 1]);
        assert!(predict.block[0] >= K * K);
        assert_eq!(predict.block[0] % 32, 0, "a partial warp is a wasted scheduler slot");

        let wide = eval(Rule::RouteRows, dims(T, 4096)).expect("rule evaluates");
        assert_eq!(wide.block, [1024, 1, 1]);
    }

    /// A collapsed rectangle is refused, not floored.
    #[test]
    fn an_empty_extent_is_refused() {
        assert_eq!(eval(Rule::Rms, dims(0, H)), Err(Ungeometric::Empty));
        assert_eq!(eval(Rule::ElementwiseRows, dims(T, 0)), Err(Ungeometric::Empty));
        assert_eq!(eval(Rule::RouteRows, dims(T, 0)), Err(Ungeometric::Empty));
        assert_eq!(eval(Rule::Elementwise, dims(T, 0)), Err(Ungeometric::Empty));
    }

    /// An unported rule says which one, and an unstated row says neither.
    #[test]
    fn the_two_refusals_are_different_sentences() {
        assert_eq!(eval(Rule::Unstated, dims(T, H)), Err(Ungeometric::Unstated));
        assert_eq!(eval(Rule::Qmv, dims(T, H)), Err(Ungeometric::Unported(Rule::Qmv)));
    }

    /// **A head-shaped rule refuses a rectangle with no heads in it.**
    #[test]
    fn a_rule_that_reads_a_head_count_refuses_a_rectangle_without_one() {
        let headless = Dims { q_heads: 0, kv_heads: 0, ..dims(T, H) };
        let flat = Dims { head_dim: 0, ..dims(T, H) };
        for rule in [Rule::PerHead, Rule::PerHeadElementwise, Rule::GatedRms, Rule::Rope] {
            assert_eq!(eval(rule, headless), Err(Ungeometric::Empty), "{rule:?} with no heads");
            assert_eq!(eval(rule, flat), Err(Ungeometric::Empty), "{rule:?} with no channels");
        }
        assert_eq!(eval(Rule::SdpaVector, headless), Err(Ungeometric::Empty));
        assert_eq!(
            eval(Rule::RouterSort, Dims { n_experts: 0, ..dims(T, H) }),
            Err(Ungeometric::Empty),
            "a sort over no experts allocates no counters and scans them"
        );
        assert_eq!(eval(Rule::SplitPacked, Dims { in_width: 0, ..dims(T, H) }), Err(Ungeometric::Empty));
    }

    /// **The one zero that is a value**, and the reason [`Dims`] has ten
    #[test]
    fn a_zero_stated_head_is_the_absent_arm_and_not_a_refusal() {
        let absent = dims(T, H);
        assert_eq!(absent.stated_head_dim, 0);
        assert_ne!(absent.head_dim, 0, "the fire still has an attention head width");
        assert_eq!(
            eval(Rule::RowsPerHead, absent).map(|l| l.grid),
            Ok([T, 1, 1]),
            "one block per row: the statement named no head, so `hidden` is the whole row"
        );
        let stated = Dims { stated_head_dim: 128, ..absent };
        assert_eq!(
            eval(Rule::RowsPerHead, stated).map(|l| l.grid),
            Ok([T * (H / 128), 1, 1]),
            "one block per (row, head) once a head was named"
        );
        assert_eq!(
            eval(Rule::RowsPerHead, Dims { stated_head_dim: 96, ..absent }),
            Err(Ungeometric::Empty),
            "a named head the row's width does not divide is refused, not rounded"
        );
    }

    /// **A rule reads only the dims it names**, which is what makes [`Dims`]
    #[test]
    fn a_new_field_cannot_move_an_old_rule() {
        let d = dims(T, H);
        let other = Dims { q_heads: 7, kv_heads: 3, head_dim: 64, n_experts: 4, ..d };
        for rule in [Rule::Rms, Rule::Elementwise, Rule::ElementwiseRows, Rule::RouteRows] {
            assert!(rule_eq(rule, d, other), "{rule:?} read a field it does not name");
        }
        assert_ne!(
            eval(Rule::PerHead, d),
            eval(Rule::PerHead, other),
            "and a rule that DOES name one must move"
        );

        let stated = Dims { stated_head_dim: 64, ..d };
        for &rule in PORTED {
            if rule == Rule::RowsPerHead {
                continue;
            }
            assert!(
                rule_eq(rule, d, stated),
                "{rule:?} moved when the STATEMENT's head width changed — only RowsPerHead reads it"
            );
        }
        assert_ne!(
            eval(Rule::RowsPerHead, d),
            eval(Rule::RowsPerHead, stated),
            "and the one rule that DOES name it must move"
        );

        let refired = Dims { head_dim: 64, ..stated };
        assert_eq!(
            eval(Rule::RowsPerHead, stated),
            eval(Rule::RowsPerHead, refired),
            "RowsPerHead read the FIRE's head width — it must read only the statement's"
        );
    }

    fn rule_eq(rule: Rule, a: Dims, b: Dims) -> bool {
        eval(rule, a) == eval(rule, b)
    }
}
