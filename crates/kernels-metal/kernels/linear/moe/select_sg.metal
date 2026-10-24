#include <metal_stdlib>
#include "../register_mma.h"
#include "../../prim/register_table.h"
using namespace metal;

// Up to eight tokens' routed experts on the register tile
// (linear/qmm_sg.metal), as Splash runs its Apple9 expert tiles: the
// routes grouped by expert into 8-row tiles, an expert's rows in one tile (a
// token routes to an expert at most once, so eight tokens fill at most one),
// with the shared expert, where one rides along, as tile 0 over every token;
// each tile's rows laid as the register tile's input table; each tile
// multiplied by its own expert's codes, which it streams once for all its
// rows, where the per-route kernel streams them once a route.
//
// `tiles` holds the live tile count, then each tile's expert (`experts` for
// the shared one); `rows` each tile row's token, -1 past its routes; `inv`
// each route's tile row. A pass's grid holds tile slots, each taking every
// slots-th live tile, as no grid can be sized by the live count it reads. A
// tile's table is `moe_sg::table_width` bf16 wide: its rows' inputs and their
// sums as `qmm_sg_table` lays them.

namespace moe_sg {

inline ulong table_width(uint width) { return 8 * ulong(width) + width / 2; }

// acc[f] += the 64-code groups [0, groups) of column `column[f]` of a 4-bit,
// 64-group affine bank times the tile's rows, read from its table `xt` and
// sums: the register tile's chains (`affine_qmm_sg`), each fragment its own
// column, two groups a step with the next two's words, scales and biases in
// flight.
template <int NF>
__attribute__((always_inline)) inline void chains(thread float2 (&acc)[NF], const device uint *codes,
                                                  const device bfloat *scales, const device bfloat *biases,
                                                  thread const ulong (&column)[NF], uint c, ulong words,
                                                  ulong groups, const device vec<bfloat, 16> *xt,
                                                  const device float *sums, uint fn) {
  using namespace register_mma;
  constexpr int U = 2;
  const device uint2 *w[NF];
  const device bfloat *s[NF], *b[NF];
#pragma unroll
  for (int f = 0; f < NF; ++f) {
    w[f] = reinterpret_cast<const device uint2 *>(codes + column[f] * words) + c;
    s[f] = scales + column[f] * groups;
    b[f] = biases + column[f] * groups;
    acc[f] = float2(0.0f);
  }
  uint2 q[U][NF];
  float qs[U][NF], qb[U][NF];
#pragma unroll
  for (int u = 0; u < U; ++u)
#pragma unroll
    for (int f = 0; f < NF; ++f) {
      q[u][f] = w[f][u * 4];
      qs[u][f] = float(s[f][u]);
      qb[u][f] = float(b[f][u]);
    }
  for (int h0 = 0; h0 < int(groups); h0 += U) {
    uint2 cw[U][NF];
    float2 ck[U][NF];
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        cw[u][f] = q[u][f];
        ck[u][f] = float2(qs[u][f], fma(-128.0f, qs[u][f], qb[u][f]));
      }
    if (h0 + U < int(groups)) {
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int f = 0; f < NF; ++f) {
          q[u][f] = w[f][ulong(h0 + U + u) * 4];
          qs[u][f] = float(s[f][h0 + U + u]);
          qb[u][f] = float(b[f][h0 + U + u]);
        }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const int h = h0 + u;
      const vec<bfloat, 16> x = xt[ulong(h) * 32];
      float2 d[NF];
#pragma unroll
      for (int f = 0; f < NF; ++f)
        d[f] = float2(0.0f);
#pragma unroll
      for (int j = 0; j < 2; ++j)
#pragma unroll
        for (uint p = 0; p < 4; ++p) {
          const bfloat2 bx = bfloat2(x[8 * j + 2 * p], x[8 * j + 2 * p + 1]);
#pragma unroll
          for (int f = 0; f < NF; ++f)
            mma(d[f], as_type<bfloat2>(((cw[u][f][j] >> (4 * p)) & 0x000F000Fu) | 0x43004300u), bx);
        }
      const float2 sm = *reinterpret_cast<const device float2 *>(sums + 2 * ulong(h) * 8 + fn) +
                        *reinterpret_cast<const device float2 *>(sums + (2 * ulong(h) + 1) * 8 + fn);
#pragma unroll
      for (int f = 0; f < NF; ++f) {
        acc[f] = fma(d[f], ck[u][f].x, acc[f]);
        acc[f] = fma(sm, ck[u][f].y, acc[f]);
      }
    }
  }
}

// A lane's (fm, fn) and the word pair c it reads, as `affine_qmm_sg` maps them.
struct Lane {
  uint fm, fn, c;
};

inline Lane lane_of(uint lane) {
  const uint qid = lane >> 2;
  const uint fm = (qid & 4) | ((lane >> 1) & 3);
  const uint fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  return {fm, fn, fn >> 1};
}

// Input `k` of a threadgroup row that holds the inputs from `by` on, for
// `register_table::half_sum`, which indexes a row from its first input.
struct Shifted {
  const threadgroup bfloat *at;
  int by;
  bfloat operator[](uint k) const { return at[int(k) - by]; }
};

} // namespace moe_sg

// The most routes a fire groups: eight tokens of a 16-wide fan-out.
constant constexpr uint kGroupRoutes = 128;

// The most experts a fire groups: a bit each in 32 words.
constant constexpr uint kGroupExperts = 1024;

// The shared expert's scalar gate of row `row` of `x`: its dot with the gate's
// one column (an affine bank of 64-code groups, `bits` 4 or 8), summed by the
// threadgroup's simdgroups in turn, landed in bf16 as the gate's matmul lands
// it, then its sigmoid, as the gated add takes it.
inline float shared_gate(const device bfloat *x, const device uchar *codes, const device bfloat *scales,
                         const device bfloat *biases, uint bits, uint width, uint row, uint lid,
                         uint threads, uint lane, uint simd, threadgroup float *parts) {
  float acc = 0.0f;
  for (uint k = lid; k < width; k += threads) {
    const uint q = bits == 8 ? uint(codes[k]) : (uint(codes[k / 2]) >> (4 * (k & 1))) & 15u;
    const float w = fma(float(scales[k / 64]), float(q), float(biases[k / 64]));
    acc = fma(float(x[ulong(row) * width + k]), w, acc);
  }
  acc = simd_sum(acc);
  if (lane == 0)
    parts[simd] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float dot = 0.0f;
  for (uint s = 0; s < threads / 32; ++s)
    dot += parts[s];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float g = float(bfloat(dot));
  return 1.0f / (1.0f + metal::exp(-g));
}

// Where `moe_sg_combine` reads the shared expert's gates in `mix`, after
// the routes' weights.
constant constexpr uint kMixGates = kGroupRoutes;

// Groups `tokens` x `top_k` routes (at most kGroupRoutes, of at most
// kGroupExperts experts) by expert and lays each live tile's input table from
// its rows' tokens in `x`, in one pass: every threadgroup groups the routes
// for itself (64 routes are cheaper to group again than to wait a launch for)
// and takes its tile, the tiles in expert order after the shared expert's
// where `shared`, each expert's rows in route order; the first of a tile's
// threadgroups records its expert, rows and routes, the first of tile 0's the
// live count and the expert passes' `grids` over the live tiles alone. Then
// `qmm_sg_table` over the tile's rows: a simdgroup per
// 64-input group (grid y) and tile (grid z), a lane per slot. Where
// `gate_bits` is not 0, the shared tile's threadgroups also land what
// `moe_sg_combine` reads in `mix`: the routes' `weights`, and each row's
// sigmoid of the shared expert's scalar gate (a threadgroup a row in turn),
// so the gate's own matmul runs nothing.
[[kernel]] void moe_sg_group_gather(
    const device bfloat *x [[buffer(0)]], const device int *routes [[buffer(1)]],
    device bfloat *tables [[buffer(2)]], device int *tiles [[buffer(3)]], device int *rows [[buffer(4)]],
    device int *inv [[buffer(5)]], constant uint &width [[buffer(6)]], constant uint &tokens [[buffer(7)]],
    constant uint &top_k [[buffer(8)]], constant uint &experts [[buffer(9)]],
    constant uint &shared [[buffer(10)]], const device float *weights [[buffer(11)]],
    const device uchar *gate_codes [[buffer(12)]], const device bfloat *gate_scales [[buffer(13)]],
    const device bfloat *gate_biases [[buffer(14)]], device float *mix [[buffer(15)]],
    constant uint &gate_bits [[buffer(16)]], device uint *grids [[buffer(17)]],
    constant uint &glu_groups [[buffer(18)]], constant uint &down_groups [[buffer(19)]],
    uint3 at [[thread_position_in_grid]],
    uint3 group [[threadgroup_position_in_grid]], uint3 groups [[threadgroups_per_grid]],
    uint lid [[thread_index_in_threadgroup]], uint3 extent [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  const uint threads = extent.x * extent.y * extent.z;
  threadgroup int staged[kGroupRoutes];
  threadgroup atomic_uint present[kGroupExperts / 32];
  threadgroup int expert_of, count_of, token_of[8];
  threadgroup uint hits[32];
  const uint n = min(tokens * top_k, kGroupRoutes), tile = at.z;
  for (uint w = lid; w < kGroupExperts / 32; w += threads)
    atomic_store_explicit(&present[w], 0u, memory_order_relaxed);
  for (uint r = lid; r < n; r += threads)
    staged[r] = routes[r];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = lid; r < n; r += threads) {
    const uint e = uint(staged[r]);
    if (e < min(experts, kGroupExperts))
      atomic_fetch_or_explicit(&present[e >> 5], 1u << (e & 31), memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // The k-th expert present, in expert order: lane w holds word w's.
  if (simd == 0) {
    const uint word = atomic_load_explicit(&present[lane], memory_order_relaxed);
    const uint held = popcount(word), before = simd_prefix_exclusive_sum(held);
    const uint total = simd_sum(held);
    const int k = int(tile) - int(shared);
    if (lane == 0) {
      count_of = int(shared + total);
      expert_of = shared != 0 && tile == 0 ? int(experts) : -1;
    }
    if (k >= 0 && uint(k) >= before && uint(k) < before + held) {
      uint left = word;
      for (uint skip = uint(k) - before; skip > 0; --skip)
        left &= left - 1;
      expert_of = int(32 * lane + ctz(left));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(tile) >= count_of)
    return;
  const int expert = expert_of;
  const bool first = group.y == 0;
  // Route `lid`'s slot in its expert's tile: the expert's routes before it
  // (a token routes to an expert at most once, so at most eight), counted
  // across the simdgroups; the threadgroup's threads cover every route.
  const bool hit = expert != int(experts) && lid < n && staged[lid] == expert;
  const uint before = simd_prefix_exclusive_sum(uint(hit)), own = simd_sum(uint(hit));
  if (lane == 0)
    hits[simd] = own;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint held = 0, slot_of = before;
  for (uint s = 0; s < threads / 32; ++s) {
    slot_of += s < simd ? hits[s] : 0;
    held += hits[s];
  }
  if (hit && slot_of < 8) {
    token_of[slot_of] = int(lid / top_k);
    if (first)
      inv[lid] = int(tile * 8 + slot_of);
  }
  if (lid < 8) {
    if (expert == int(experts))
      token_of[lid] = lid < tokens ? int(lid) : -1;
    else if (lid >= held)
      token_of[lid] = -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (first && lid < 8) {
    rows[tile * 8 + lid] = token_of[lid];
    if (lid == 0) {
      tiles[1 + tile] = expert;
      if (tile == 0) {
        tiles[0] = count_of;
        // The gate-up's and the down's grids: their column threadgroups by
        // the live tiles.
        grids[0] = glu_groups, grids[1] = uint(count_of), grids[2] = 1;
        grids[4] = down_groups, grids[5] = uint(count_of), grids[6] = 1;
      }
    }
  }
  const uint slot = at.x, h = at.y;
  device bfloat *table = tables + tile * moe_sg::table_width(width);
  const uint fm = slot >> 2, c = slot & 3;
  const uint cp = fm >> 1, i = fm & 1;
  const int token[2] = {token_of[2 * c], token_of[2 * c + 1]};
  vec<bfloat, 16> v;
  float sum[2] = {0.0f, 0.0f};
#pragma unroll
  for (int j = 0; j < 2; ++j)
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const uint k = 64 * h + 8 * (2 * cp + j) + p + 4 * i;
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const bfloat value = token[r] >= 0 ? x[ulong(token[r]) * width + k] : bfloat(0.0f);
        v[8 * j + 2 * p + r] = value;
        sum[r] += float(value);
      }
    }
  reinterpret_cast<device vec<bfloat, 16> *>(table)[ulong(h) * 32 + slot] = v;
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    sum[r] += simd_shuffle_xor(sum[r], 4u);
    sum[r] += simd_shuffle_xor(sum[r], 8u);
  }
  if ((fm & 3) == 0) {
    device float *sums = register_table::sums(table, width);
#pragma unroll
    for (int r = 0; r < 2; ++r)
      sums[(2 * ulong(h) + (fm >> 2)) * 8 + 2 * c + r] = sum[r];
  }
  if (gate_bits == 0 || tile != 0)
    return;
  threadgroup float parts[32];
  if (group.y == 0)
    for (uint r = lid; r < n; r += threads)
      mix[r] = weights[r];
  for (uint row = group.y; row < tokens; row += groups.y) {
    const float g = shared_gate(x, gate_codes, gate_scales, gate_biases, gate_bits, width, row, lid,
                                threads, lane, simd, parts);
    if (lid == 0)
      mix[kMixGates + row] = g;
  }
}

// Each token's row of the routed MLP and its gated shared expert, from the
// register tiles' down rows: the routes' rows weighed by their weights, in
// route order, then the shared expert's row (tile 0's row of the token) by
// its gate's sigmoid, as `moe_sg_group_gather` landed both in `mix`. Four
// columns a thread, a token a grid row.
[[kernel]] void moe_sg_combine(const device bfloat *down [[buffer(0)]], const device int *inv [[buffer(1)]],
                               const device float *mix [[buffer(2)]], device bfloat *out [[buffer(3)]],
                               constant uint &width [[buffer(4)]], constant uint &top_k [[buffer(5)]],
                               uint2 at [[thread_position_in_grid]]) {
  const uint c = 4 * at.x, row = at.y;
  if (c >= width)
    return;
  float4 acc = float4(0.0f);
  for (uint k = 0; k < top_k; ++k) {
    const uint route = row * top_k + k;
    acc = fma(float4(*reinterpret_cast<const device vec<bfloat, 4> *>(down + ulong(inv[route]) * width + c)),
              mix[route], acc);
  }
  acc = fma(float4(*reinterpret_cast<const device vec<bfloat, 4> *>(down + ulong(row) * width + c)),
            mix[kMixGates + row], acc);
  *reinterpret_cast<device vec<bfloat, 4> *>(out + ulong(row) * width + c) = vec<bfloat, 4>(acc);
}

// A routed gate-up and its SwiGLU: each of the threadgroup's SG simdgroups
// takes HF 8-column fragments of the intermediate and the gate and up columns
// of each (rows c and `inter` + c of the expert's packed rows), and lands
// silu(gate) * up, both rounded to bf16 first as the packed rows would hold
// them, straight into the down's input table of the tile; the threadgroup's
// 8 HF SG columns are whole halves of 64-input groups, whose sums it adds as
// `qmm_sg_table` does. The grid's y extent is its tile slots: slot y takes
// tiles y, y + slots, ... of the live count; launched on the grid the
// grouping wrote, a slot a live tile.
// Compiled for its SG simdgroups a threadgroup: left to itself a compiler
// may fit fewer (the M1 Max's, 384 threads), and a launch past a pipeline's
// most threads is undefined.
template <int HF, int SG>
[[kernel]] [[max_total_threads_per_threadgroup(32 * SG)]] void affine_qmm_sg_glu_routed(
    const device uint *codes [[buffer(0)]], const device bfloat *scales [[buffer(1)]],
    const device bfloat *biases [[buffer(2)]], const device uint *shared_codes [[buffer(3)]],
    const device bfloat *shared_scales [[buffer(4)]], const device bfloat *shared_biases [[buffer(5)]],
    const device bfloat *tables [[buffer(6)]], device bfloat *hidden [[buffer(7)]],
    const device int *tiles [[buffer(8)]], constant int &width [[buffer(9)]],
    constant int &inter [[buffer(10)]], constant int &experts [[buffer(11)]],
    uint2 position [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]],
    uint lid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr int NF = 2 * HF;
  constexpr uint COLUMNS = 8 * HF * SG, HALVES = COLUMNS / 32;
  static_assert(COLUMNS % 32 == 0, "a threadgroup's columns are whole halves of 64-input groups");
  threadgroup bfloat laid[8][COLUMNS];
  const moe_sg::Lane l = moe_sg::lane_of(lane);
  const uint local = simd * 8 * HF + l.fm;
  const uint at = position.x * COLUMNS + local;
  const ulong words = ulong(width) / 8, groups = ulong(width) / 64;
  const int count = tiles[0];
  for (uint tile = position.y; int(tile) < count; tile += grid.y) {
    const int expert = tiles[1 + tile];
    const bool own = expert == experts;
    const ulong first = own ? 0 : ulong(expert) * 2 * ulong(inter);
    ulong column[NF];
#pragma unroll
    for (int f = 0; f < HF; ++f) {
      column[f] = first + at + 8 * f;
      column[HF + f] = first + inter + at + 8 * f;
    }
    const device bfloat *table = tables + tile * moe_sg::table_width(width);
    float2 acc[NF];
    moe_sg::chains<NF>(acc, own ? shared_codes : codes, own ? shared_scales : scales,
                       own ? shared_biases : biases, column, l.c, words, groups,
                       reinterpret_cast<const device vec<bfloat, 16> *>(table) + l.fm * 4 + l.c,
                       register_table::sums(const_cast<device bfloat *>(table), width), l.fn);
    device bfloat *out = hidden + tile * moe_sg::table_width(inter);
    // The last tile's sums are read out of `laid` before it is written again.
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
    for (int f = 0; f < HF; ++f)
#pragma unroll
      for (uint i = 0; i < 2; ++i) {
        const uint row = l.fn + i;
        const float g = float(bfloat(acc[f][i]));
        const float u = float(bfloat(acc[HF + f][i]));
        const bfloat h = bfloat((g / (1.0f + metal::exp(-g))) * u);
        out[register_table::slot(at + 8 * f, row)] = h;
        laid[row][local + 8 * f] = h;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Rows x halves: a thread each.
    if (lid < 8 * HALVES) {
      const uint row = lid & 7, part = lid >> 3;
      const uint n = position.x * HALVES + part; // the 32-input half of the down's input
      const moe_sg::Shifted src{&laid[row][0], int(position.x * COLUMNS)};
      register_table::sums(out, inter)[ulong(n) * 8 + row] =
          register_table::half_sum(src, n >> 1, n & 1);
    }
  }
}

#define PIE_MOE_SG_GLU(hf, sg)                                                                     \
  template [[host_name("affine_qmm_sg_glu_routed_h" #hf "_s" #sg)]] [[kernel]] void               \
  affine_qmm_sg_glu_routed<hf, sg>(                                                                \
      const device uint *, const device bfloat *, const device bfloat *, const device uint *,      \
      const device bfloat *, const device bfloat *, const device bfloat *, device bfloat *,        \
      const device int *, constant int &, constant int &, constant int &, uint2, uint2, uint,      \
      uint, uint);
PIE_MOE_SG_GLU(1, 4)
PIE_MOE_SG_GLU(1, 8)
PIE_MOE_SG_GLU(1, 16)
PIE_MOE_SG_GLU(2, 4)
PIE_MOE_SG_GLU(2, 8)

// A routed projection of each live tile's table by its expert's columns, SG
// simdgroups of 16 a threadgroup, into the tile's rows of `out` that hold a
// route (the rest are left as they were, read by nothing); the grid's y
// extent is its tile slots, as the gate-up's; compiled for its SG simdgroups
// as the gate-up is (the M1 Max's compiler fits 448 threads of it otherwise).
template <int SG>
[[kernel]] [[max_total_threads_per_threadgroup(32 * SG)]] void affine_qmm_sg_routed(
    const device uint *codes [[buffer(0)]], const device bfloat *scales [[buffer(1)]],
    const device bfloat *biases [[buffer(2)]], const device uint *shared_codes [[buffer(3)]],
    const device bfloat *shared_scales [[buffer(4)]], const device bfloat *shared_biases [[buffer(5)]],
    const device bfloat *tables [[buffer(6)]], device bfloat *out [[buffer(7)]],
    const device int *tiles [[buffer(8)]], constant int &width [[buffer(9)]],
    constant int &columns [[buffer(10)]], constant int &experts [[buffer(11)]],
    const device int *rows [[buffer(12)]], uint2 position [[threadgroup_position_in_grid]],
    uint2 grid [[threadgroups_per_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr int NF = 2;
  const moe_sg::Lane l = moe_sg::lane_of(lane);
  const uint at = position.x * 8 * NF * SG + simd * 8 * NF + l.fm;
  const ulong words = ulong(width) / 8, groups = ulong(width) / 64;
  const int count = tiles[0];
  for (uint tile = position.y; int(tile) < count; tile += grid.y) {
    const int expert = tiles[1 + tile];
    const bool own = expert == experts;
    const ulong first = own ? 0 : ulong(expert) * ulong(columns);
    ulong column[NF];
#pragma unroll
    for (int f = 0; f < NF; ++f)
      column[f] = first + at + 8 * f;
    const device bfloat *table = tables + tile * moe_sg::table_width(width);
    float2 acc[NF];
    moe_sg::chains<NF>(acc, own ? shared_codes : codes, own ? shared_scales : scales,
                       own ? shared_biases : biases, column, l.c, words, groups,
                       reinterpret_cast<const device vec<bfloat, 16> *>(table) + l.fm * 4 + l.c,
                       register_table::sums(const_cast<device bfloat *>(table), width), l.fn);
#pragma unroll
    for (uint i = 0; i < 2; ++i) {
      const ulong row = ulong(tile) * 8 + l.fn + i;
      if (rows[row] < 0)
        continue;
#pragma unroll
      for (int f = 0; f < NF; ++f)
        out[row * columns + at + 8 * f] = bfloat(acc[f][i]);
    }
  }
}

#define PIE_MOE_SG_DOWN(sg)                                                                        \
  template [[host_name("affine_qmm_sg_routed_s" #sg)]] [[kernel]] void affine_qmm_sg_routed<sg>(   \
      const device uint *, const device bfloat *, const device bfloat *, const device uint *,      \
      const device bfloat *, const device bfloat *, const device bfloat *, device bfloat *,        \
      const device int *, constant int &, constant int &, constant int &, const device int *,      \
      uint2, uint2, uint, uint);
PIE_MOE_SG_DOWN(4)
PIE_MOE_SG_DOWN(8)
PIE_MOE_SG_DOWN(16)
