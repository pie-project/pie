#include <metal_stdlib>

using namespace metal;

template <typename T>
[[kernel]] void kv_append(
    const device T* k_new   [[buffer(0)]],
    const device T* v_new   [[buffer(1)]],
    device T* k_cache       [[buffer(2)]],
    device T* v_cache       [[buffer(3)]],
    const device int* pos                [[buffer(4)]],
    const constant int& head_dim         [[buffer(5)]],
    const constant size_t& k_head_stride [[buffer(6)]],
    const constant size_t& k_seq_stride  [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int d = int(tid.x);
  const int h = int(tid.y);
  if (d >= head_dim) return;

  const size_t dst = h * k_head_stride + size_t(pos[0]) * k_seq_stride + d;
  const int src = h * head_dim + d;
  k_cache[dst] = k_new[src];
  v_cache[dst] = v_new[src];
}

#define instantiate_kv_append(name, itype)                        \
  template [[host_name("kv_append_" #name)]]                      \
  [[kernel]] void kv_append<itype>(                               \
      const device itype*, const device itype*, device itype*,    \
      device itype*, const device int*, const constant int&,      \
      const constant size_t&, const constant size_t&, uint2);

instantiate_kv_append(bfloat16, bfloat)

template <typename T>
[[kernel]] void kv_append_paged(
    const device T* k_new   [[buffer(0)]],
    const device T* v_new   [[buffer(1)]],
    device T* k_pages       [[buffer(2)]],
    device T* v_pages       [[buffer(3)]],
    const constant int& head_dim         [[buffer(5)]],

    const constant int& page_size        [[buffer(10)]],
    const constant int& n_kv_heads       [[buffer(12)]],
    const device uint* w_page            [[buffer(13)]],
    const device uint* w_off             [[buffer(14)]],

    const constant int& src_row_stride   [[buffer(15)]],
    uint3 tid [[thread_position_in_grid]]) {
  const int d = int(tid.x);
  const int h = int(tid.y);
  const int i = int(tid.z);
  if (d >= head_dim) return;

  const uint page = w_page[i];
  const size_t slot = size_t(page) * size_t(page_size) + size_t(w_off[i]);

  const size_t row_stride = size_t(n_kv_heads) * size_t(head_dim);
  const size_t dst = slot * row_stride + size_t(h) * size_t(head_dim) + size_t(d);
  const size_t src_row = src_row_stride > 0 ? size_t(src_row_stride) : row_stride;
  const size_t src = size_t(i) * src_row + size_t(h) * size_t(head_dim) + size_t(d);

  k_pages[dst] = k_new[src];
  v_pages[dst] = v_new[src];
}

#define instantiate_kv_append_paged(name, itype)                  \
  template [[host_name("kv_append_paged_" #name)]]                \
  [[kernel]] void kv_append_paged<itype>(                         \
      const device itype*, const device itype*, device itype*,    \
      device itype*, const constant int&, const constant int&,    \
      const constant int&, const device uint*,                    \
      const device uint*, const constant int&, uint3);

instantiate_kv_append_paged(bfloat16, bfloat)

// C2b — the QUANTIZING paged KV write. Where `kv_append_paged` copies one
// element per thread into a bf16 page, this packs one whole head (a 256-block)
// per threadgroup into the LOCKED v1 KV format: 4-bit symmetric absmax, one
// inline fp16 scale, 130 bytes per head, offset-binary nibbles (store q+8).
//
// PRODUCTION SCALE REQUIREMENT (self-consistent): the fp16 scale is rounded
// FIRST and the codes are then quantized against that SAME fp16 scale, so the
// scale encode uses and the scale decode reads are identical — no systematic
// encode/decode mismatch. This is the one behavioural difference from the C2a
// pack kernel, which quantizes against the full-precision scale.
//
// One threadgroup per (kv head, token); the head_dim (== 256) threads of the
// group reduce the block's absmax together, then the low 128 threads each write
// one packed byte and thread 0 writes the fp16 scale. The launch MUST pin the
// x-extent (== head_dim) to 256 so every thread reaches each barrier — a head
// that is not a whole 256-block cannot pack and is refused on the host side.
#define KVP_BLOCK 256u
#define KVP_NIB_BYTES 128u
#define KVP_BYTES 130u
#define KVP_LIM 7.0f

template <typename T>
[[kernel]] void kv_append_paged_pack_sym4(
    const device T* k_new   [[buffer(0)]],
    const device T* v_new   [[buffer(1)]],
    device uchar* k_pages   [[buffer(2)]],
    device uchar* v_pages   [[buffer(3)]],
    const constant int& head_dim         [[buffer(5)]],

    const constant int& page_size        [[buffer(10)]],
    const constant int& n_kv_heads       [[buffer(12)]],
    const device uint* w_page            [[buffer(13)]],
    const device uint* w_off             [[buffer(14)]],

    const constant int& src_row_stride   [[buffer(15)]],
    uint3 tid [[thread_position_in_grid]]) {
  const uint d = tid.x;  // element within the head (== the 256-block), [0, 256)
  const uint h = tid.y;  // kv head
  const uint i = tid.z;  // token

  threadgroup float ak[KVP_BLOCK];
  threadgroup float av[KVP_BLOCK];
  threadgroup uchar nk[KVP_BLOCK];
  threadgroup uchar nv[KVP_BLOCK];

  const uint hd = uint(head_dim);
  const uint src_row =
      src_row_stride > 0 ? uint(src_row_stride) : uint(n_kv_heads) * hd;
  const uint src = i * src_row + h * hd + d;
  const float kx = float(k_new[src]);
  const float vx = float(v_new[src]);

  // Per-block absmax over the 256 threads (tree reduction).
  ak[d] = fabs(kx);
  av[d] = fabs(vx);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = KVP_BLOCK / 2; stride > 0; stride >>= 1) {
    if (d < stride) {
      ak[d] = max(ak[d], ak[d + stride]);
      av[d] = max(av[d], av[d + stride]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float absmax_k = ak[0];
  const float absmax_v = av[0];

  // Self-consistent scale: round to fp16 first, quantize against that fp16.
  const half hsk = half(absmax_k / KVP_LIM);
  const half hsv = half(absmax_v / KVP_LIM);
  const float sck = float(hsk);
  const float scv = float(hsv);
  const int qk = absmax_k > 0.0f ? int(round(clamp(kx / sck, -KVP_LIM, KVP_LIM))) : 0;
  const int qv = absmax_v > 0.0f ? int(round(clamp(vx / scv, -KVP_LIM, KVP_LIM))) : 0;
  nk[d] = uchar(qk + 8);  // offset-binary [1, 15]
  nv[d] = uchar(qv + 8);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Destination: one 130-byte block per (slot, head); a slot's heads are
  // contiguous, so the whole slot is n_kv_heads * 130 bytes.
  const uint slot = w_page[i] * uint(page_size) + w_off[i];
  const uint base = (slot * uint(n_kv_heads) + h) * KVP_BYTES;
  if (d < KVP_NIB_BYTES) {
    k_pages[base + d] = uchar((uint(nk[2 * d + 1]) << 4) | uint(nk[2 * d]));
    v_pages[base + d] = uchar((uint(nv[2 * d + 1]) << 4) | uint(nv[2 * d]));
  }
  if (d == 0) {
    const ushort bk = as_type<ushort>(hsk);
    const ushort bv = as_type<ushort>(hsv);
    k_pages[base + KVP_NIB_BYTES] = uchar(bk & 0xff);
    k_pages[base + KVP_NIB_BYTES + 1] = uchar((bk >> 8) & 0xff);
    v_pages[base + KVP_NIB_BYTES] = uchar(bv & 0xff);
    v_pages[base + KVP_NIB_BYTES + 1] = uchar((bv >> 8) & 0xff);
  }
}

#define instantiate_kv_append_paged_pack(name, itype)             \
  template [[host_name("kv_append_paged_pack_sym4_" #name)]]      \
  [[kernel]] void kv_append_paged_pack_sym4<itype>(               \
      const device itype*, const device itype*, device uchar*,    \
      device uchar*, const constant int&, const constant int&,    \
      const constant int&, const device uint*,                    \
      const device uint*, const constant int&, uint3);

instantiate_kv_append_paged_pack(bfloat16, bfloat)
