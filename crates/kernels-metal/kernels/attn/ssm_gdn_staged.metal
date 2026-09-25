#include "ssm_gdn_scan.metal"

template <int BLOCK>
[[kernel]] void
gdn_staged(const device bfloat *qkv [[buffer(0)]], const device int *indptr [[buffer(1)]],
           const device float *gates [[buffer(2)]], device float *state [[buffer(3)]],
           const device uint *slots [[buffer(4)]], device float *y [[buffer(5)]],
           constant int &kh [[buffer(6)]], constant int &vh [[buffer(7)]],
           constant int &dk [[buffer(8)]], constant int &dv [[buffer(9)]],
           uint3 tile [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr int D = 128, ROWS = 32;
  const int req = tile.z / vh, hv = tile.z % vh, hk = hv / (vh / kh);
  const int begin = indptr[req], end = indptr[req + 1];
  if (end <= begin)
    return;
  const int packed_lane = tid % 8, value_base = tile.y * ROWS + (tid / 8) * 2;
  const size_t keys = size_t(kh) * dk, pitch = 2 * keys + size_t(vh) * dv;
  const float scale = 1.0f / metal::sqrt(float(dk));
  device float *cells = gdn_cells<2, 16>(state, int(slots[begin]), hv, vh, value_base, dk, dv);
  float st[2][16];
  gdn_load<2, 16>(st, cells, packed_lane, dk);
  threadgroup float queries[BLOCK * D], key_data[BLOCK * D], values[BLOCK * ROWS],
      gate_data[BLOCK * 2];
  for (int start = begin; start < end; start += BLOCK) {
    const int count = min(BLOCK, end - start);
    for (int b = sg; b < count; b += 4) {
      const size_t row = size_t(start + b) * pitch;
      float qr[4], kr[4], qs = 0, ks = 0;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        qr[j] = float(qkv[row + size_t(hk) * dk + lane * 4 + j]);
        kr[j] = float(qkv[row + keys + size_t(hk) * dk + lane * 4 + j]);
        qs += qr[j] * qr[j];
        ks += kr[j] * kr[j];
      }
      const float qi = metal::rsqrt(gdn_row_sum<32>(qs) + 1e-6f) * scale;
      const float ki = metal::rsqrt(gdn_row_sum<32>(ks) + 1e-6f);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int d = lane * 4 + j, at = b * D + (d % 16) * 8 + d / 16;
        queries[at] = qr[j] * qi;
        key_data[at] = kr[j] * ki;
      }
      if (lane == 0) {
        const size_t g = size_t(start + b) * 2 * vh + hv;
        gate_data[b] = metal::exp(gates[g]);
        gate_data[BLOCK + b] = gates[g + vh];
      }
    }
    for (int i = tid; i < count * ROWS; i += 128) {
      const int b = i / ROWS, v = i % ROWS;
      values[i] =
          float(qkv[size_t(start + b) * pitch + 2 * keys + size_t(hv) * dv + tile.y * ROWS + v]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int b = 0; b < count; ++b) {
      const float decay = gate_data[b], beta = gate_data[BLOCK + b];
      {
        float k[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
          k[j] = key_data[b * D + j * 8 + packed_lane];
#pragma unroll
        for (int v = 0; v < 2; ++v) {
#pragma unroll
          for (int j = 0; j < 16; ++j)
            st[v][j] *= decay;
          const float mem = gdn_row_sum<8>(gdn_packed_dot(st[v], k));
          const float delta = (values[b * ROWS + (tid / 8) * 2 + v] - mem) * beta;
#pragma unroll
          for (int j = 0; j < 16; ++j)
            st[v][j] += k[j] * delta;
        }
      }
      {
        float q[16];
#pragma unroll
        for (int j = 0; j < 16; ++j)
          q[j] = queries[b * D + j * 8 + packed_lane];
#pragma unroll
        for (int v = 0; v < 2; ++v) {
          const float out = gdn_row_sum<8>(gdn_packed_dot(st[v], q));
          if (packed_lane == 0)
            y[(size_t(start + b) * vh + hv) * dv + value_base + v] = out;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  gdn_store<2, 16>(cells, st, packed_lane, dk);
}
#define PIE_GDN_STAGED(name, block)                                                                \
  template [[host_name(name)]] [[kernel]] void gdn_staged<block>(                                  \
      const device bfloat *, const device int *, const device float *, device float *,             \
      const device uint *, device float *, constant int &, constant int &, constant int &,         \
      constant int &, uint3, uint, uint, uint);
