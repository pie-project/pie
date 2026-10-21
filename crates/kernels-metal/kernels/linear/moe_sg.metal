#include <metal_stdlib>
#include "../common/register_mma.metal"
#include "../common/register_table.metal"
using namespace metal;

// Up to eight tokens' routed experts on the register tile
// (linear/quant_qmm_sg.metal), as Splash runs its Apple9 expert tiles: the
// routes grouped by expert into 8-row tiles, an expert's rows in one tile (a
// token routes to an expert at most once, so eight tokens fill at most one),
// with the shared expert, where one rides along, as tile 0 over every token;
// each tile's rows laid as the register tile's input table; each tile
// multiplied by its own expert's codes, which it streams once for all its
// rows, where the per-route kernel streams them once a route.
//
// `tiles` holds the live tile count, then each tile's expert (`experts` for
// the shared one); `rows` each tile row's token, -1 past its routes; `inv`
// each route's tile row. A pass launches every tile a fire could fill and
// the tiles past the count return. A tile's table is `moe_sg::table_width`
// bf16 wide: its rows' inputs and their sums as `qmm_sg_table` lays them.

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
  using namespace qmm_sg;
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

// Row `k - by` of a threadgroup row, for `register_table::half_sum` over a
// row that holds one half of a 64-input group.
struct Shifted {
  const threadgroup bfloat *at;
  int by;
  bfloat operator[](uint k) const { return at[int(k) - by]; }
};

} // namespace moe_sg

// Groups `tokens` x `top_k` routes by expert: one threadgroup, a thread per
// expert (a multiple of 32 threads, at most 1024), the tiles in expert order,
// after the shared expert's where `shared`, each expert's rows in route order.
[[kernel]] void moe_sg_group(const device int *routes [[buffer(0)]], device int *tiles [[buffer(1)]],
                             device int *rows [[buffer(2)]], device int *inv [[buffer(3)]],
                             constant uint &tokens [[buffer(4)]], constant uint &top_k [[buffer(5)]],
                             constant uint &experts [[buffer(6)]], constant uint &shared [[buffer(7)]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint lanes [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]]) {
  threadgroup uint before[32];
  const uint n = tokens * top_k;
  uint count = 0;
  if (lid < experts)
    for (uint r = 0; r < n; ++r)
      count += uint(routes[r] == int(lid));
  const uint present = count > 0 ? 1u : 0u;
  const uint within = simd_prefix_exclusive_sum(present);
  if (lane == 31)
    before[simd] = within + present;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint at = shared + within;
  for (uint s = 0; s < simd; ++s)
    at += before[s];
  if (present) {
    tiles[1 + at] = int(lid);
    uint slot = 0;
    for (uint r = 0; r < n && slot < 8; ++r)
      if (routes[r] == int(lid)) {
        rows[at * 8 + slot] = int(r / top_k);
        inv[r] = int(at * 8 + slot);
        ++slot;
      }
    for (; slot < 8; ++slot)
      rows[at * 8 + slot] = -1;
  }
  if (lid == lanes - 1)
    tiles[0] = int(at + present);
  if (shared != 0 && lid < 8)
    rows[lid] = lid < tokens ? int(lid) : -1;
  if (shared != 0 && lid == 0)
    tiles[1] = int(experts);
}

// Each live tile's input table, from its rows' tokens in `x`: `qmm_sg_table`
// over the tile's rows, a simdgroup per 64-input group (grid y) and tile
// (grid z), a lane per slot.
[[kernel]] void moe_sg_gather(const device bfloat *x [[buffer(0)]], device bfloat *tables [[buffer(1)]],
                              const device int *tiles [[buffer(2)]], const device int *rows [[buffer(3)]],
                              constant uint &width [[buffer(4)]],
                              uint3 at [[thread_position_in_grid]]) {
  const uint slot = at.x, h = at.y, tile = at.z;
  if (int(tile) >= tiles[0])
    return;
  device bfloat *table = tables + tile * moe_sg::table_width(width);
  const uint fm = slot >> 2, c = slot & 3;
  const uint cp = fm >> 1, i = fm & 1;
  const int token[2] = {rows[tile * 8 + 2 * c], rows[tile * 8 + 2 * c + 1]};
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
}

// A routed gate-up and its SwiGLU: each of the threadgroup's SG simdgroups
// takes HF 8-column fragments of the intermediate and the gate and up columns
// of each (rows c and `inter` + c of the expert's packed rows), and lands
// silu(gate) * up, both rounded to bf16 first as the packed rows would hold
// them, straight into the down's input table of the tile; the threadgroup's
// 8 HF SG columns are whole halves of 64-input groups, whose sums it adds as
// `qmm_sg_table` does. A threadgroup of more simdgroups is fewer for the GPU
// to dispatch, of which the tiles past the count are pure cost.
template <int HF, int SG>
[[kernel]] void affine_qmm_sg_glu_routed(
    const device uint *codes [[buffer(0)]], const device bfloat *scales [[buffer(1)]],
    const device bfloat *biases [[buffer(2)]], const device uint *shared_codes [[buffer(3)]],
    const device bfloat *shared_scales [[buffer(4)]], const device bfloat *shared_biases [[buffer(5)]],
    const device bfloat *tables [[buffer(6)]], device bfloat *hidden [[buffer(7)]],
    const device int *tiles [[buffer(8)]], constant int &width [[buffer(9)]],
    constant int &inter [[buffer(10)]], constant int &experts [[buffer(11)]],
    uint2 position [[threadgroup_position_in_grid]], uint lid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr int NF = 2 * HF;
  constexpr uint COLUMNS = 8 * HF * SG, HALVES = COLUMNS / 32;
  static_assert(COLUMNS % 32 == 0, "a threadgroup's columns are whole halves of 64-input groups");
  threadgroup bfloat laid[8][COLUMNS];
  const uint tile = position.y;
  if (int(tile) >= tiles[0])
    return;
  const int expert = tiles[1 + tile];
  const bool own = expert == experts;
  const ulong first = own ? 0 : ulong(expert) * 2 * ulong(inter);
  const moe_sg::Lane l = moe_sg::lane_of(lane);
  const uint local = simd * 8 * HF + l.fm;
  const uint at = position.x * COLUMNS + local;
  ulong column[NF];
#pragma unroll
  for (int f = 0; f < HF; ++f) {
    column[f] = first + at + 8 * f;
    column[HF + f] = first + inter + at + 8 * f;
  }
  const ulong words = ulong(width) / 8, groups = ulong(width) / 64;
  const device bfloat *table = tables + tile * moe_sg::table_width(width);
  float2 acc[NF];
  moe_sg::chains<NF>(acc, own ? shared_codes : codes, own ? shared_scales : scales,
                     own ? shared_biases : biases, column, l.c, words, groups,
                     reinterpret_cast<const device vec<bfloat, 16> *>(table) + l.fm * 4 + l.c,
                     register_table::sums(const_cast<device bfloat *>(table), width), l.fn);
  device bfloat *out = hidden + tile * moe_sg::table_width(inter);
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

#define PIE_MOE_SG_GLU(hf, sg)                                                                     \
  template [[host_name("affine_qmm_sg_glu_routed_h" #hf "_s" #sg)]] [[kernel]] void               \
  affine_qmm_sg_glu_routed<hf, sg>(                                                                \
      const device uint *, const device bfloat *, const device bfloat *, const device uint *,      \
      const device bfloat *, const device bfloat *, const device bfloat *, device bfloat *,        \
      const device int *, constant int &, constant int &, constant int &, uint2, uint, uint, uint);
PIE_MOE_SG_GLU(1, 4)
PIE_MOE_SG_GLU(1, 8)
PIE_MOE_SG_GLU(1, 16)
PIE_MOE_SG_GLU(2, 4)
PIE_MOE_SG_GLU(2, 8)

// A routed projection of each live tile's table by its expert's columns, SG
// simdgroups of 16 a threadgroup, into the tile's eight rows of `out` (rows
// past its routes are zero).
template <int SG>
[[kernel]] void affine_qmm_sg_routed(
    const device uint *codes [[buffer(0)]], const device bfloat *scales [[buffer(1)]],
    const device bfloat *biases [[buffer(2)]], const device uint *shared_codes [[buffer(3)]],
    const device bfloat *shared_scales [[buffer(4)]], const device bfloat *shared_biases [[buffer(5)]],
    const device bfloat *tables [[buffer(6)]], device bfloat *out [[buffer(7)]],
    const device int *tiles [[buffer(8)]], constant int &width [[buffer(9)]],
    constant int &columns [[buffer(10)]], constant int &experts [[buffer(11)]],
    uint2 position [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
  constexpr int NF = 2;
  const uint tile = position.y;
  if (int(tile) >= tiles[0])
    return;
  const int expert = tiles[1 + tile];
  const bool own = expert == experts;
  const ulong first = own ? 0 : ulong(expert) * ulong(columns);
  const moe_sg::Lane l = moe_sg::lane_of(lane);
  const uint at = position.x * 8 * NF * SG + simd * 8 * NF + l.fm;
  ulong column[NF];
#pragma unroll
  for (int f = 0; f < NF; ++f)
    column[f] = first + at + 8 * f;
  const ulong words = ulong(width) / 8, groups = ulong(width) / 64;
  const device bfloat *table = tables + tile * moe_sg::table_width(width);
  float2 acc[NF];
  moe_sg::chains<NF>(acc, own ? shared_codes : codes, own ? shared_scales : scales,
                     own ? shared_biases : biases, column, l.c, words, groups,
                     reinterpret_cast<const device vec<bfloat, 16> *>(table) + l.fm * 4 + l.c,
                     register_table::sums(const_cast<device bfloat *>(table), width), l.fn);
#pragma unroll
  for (uint i = 0; i < 2; ++i) {
    const ulong row = ulong(tile) * 8 + l.fn + i;
#pragma unroll
    for (int f = 0; f < NF; ++f)
      out[row * columns + at + 8 * f] = bfloat(acc[f][i]);
  }
}

#define PIE_MOE_SG_DOWN(sg)                                                                        \
  template [[host_name("affine_qmm_sg_routed_s" #sg)]] [[kernel]] void affine_qmm_sg_routed<sg>(   \
      const device uint *, const device bfloat *, const device bfloat *, const device uint *,      \
      const device bfloat *, const device bfloat *, const device bfloat *, device bfloat *,        \
      const device int *, constant int &, constant int &, constant int &, uint2, uint, uint);
PIE_MOE_SG_DOWN(4)
PIE_MOE_SG_DOWN(8)
PIE_MOE_SG_DOWN(16)
