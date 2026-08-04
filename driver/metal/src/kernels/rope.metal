// Raw-Metal port of NEOX RoPE for Phase-0 decode (single token, in-place).
//
// Source semantics: pie_metal_driver::ops::rope (driver/metal/src/ops/rope.cpp)
//   non-YaRN fast path == MLX fast::rope(traditional=false, dims=rope_dims),
//   which dispatches mlx .../kernels/rope.metal `rope` (rope_impl, NEOX).
// Port notes (qwen3.6: head_dim=256, rope_dims=64 partial, theta=1e7):
//   * Single decode token (seq len 1) -> pos.y(seq)=0, one position per token.
//   * NEOX rotate-half over the first rope_dims channels: pair (i, i+half),
//     half=rope_dims/2. Channels [rope_dims, head_dim) are pass-through (untouched).
//   * position IS the per-token IO scalar -> read from a *buffer* (offset[0]) per
//     decode_abi I1, never setBytes; CB stays byte-identical.
//   * IN-PLACE: each thread owns a disjoint (i, i+half) pair, so reading then
//     writing the same buffer is hazard-free. Matches the qk-norm->rope fold
//     (rope runs on the resident q/k projection slot, no separate out copy).
//   * inv_freq[i] = exp2(-(i/half) * base), base = log2(theta); matches MLX.
//   * scale = 1/scaling_factor (1.0 for qwen3.6 default).
// Launch: dispatchThreads grid=(half, n_head, 1), tg=(half,1,1). bfloat native.

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void rope_neox_decode(
    device T* x                       [[buffer(0)]],  // in-place [n_head, head_dim]
    const device int* position        [[buffer(1)]],  // IO scalar (I1): position[0]
    const constant float& scale       [[buffer(2)]],
    const constant float& base        [[buffer(3)]],  // log2(theta)
    const constant int& head_dim      [[buffer(4)]],
    uint2 pos  [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  const int i = int(pos.x);            // freq index, 0..half-1
  const int h = int(pos.y);            // head index
  const int half_rd = int(grid.x);        // rope_dims / 2

  float d = static_cast<float>(i) / static_cast<float>(half_rd);
  float inv_freq = exp2(-d * base);
  float L = scale * static_cast<float>(position[0]);
  float theta = L * inv_freq;
  float costheta = fast::cos(theta);
  float sintheta = fast::sin(theta);

  const int i1 = h * head_dim + i;
  const int i2 = i1 + half_rd;
  float x1 = static_cast<float>(x[i1]);
  float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_neox(name, itype)                       \
  template [[host_name("rope_neox_decode_" #name)]]              \
  [[kernel]] void rope_neox_decode<itype>(                       \
      device itype*, const device int*, const constant float&,   \
      const constant float&, const constant int&, uint2, uint2);

instantiate_rope_neox(bfloat16, bfloat)

// ── M>1 batched NEOX RoPE (multi-batch lane). ────────────────────────────────
// Adds a token-row dim (pos.z = m); token m rotates by its own position[m] (per-row
// IO read — the M>1 relaxation of the sealed position[0]). x is token-major
// [N, n_head, head_dim]; per-token stride = n_head*head_dim = grid.y*head_dim.
// In-place + hazard-free (disjoint (m,h,i/i+half) pairs). Reduces to
// rope_neox_decode at N=1 (pos.z=0, position[0]). Launch grid=(half, n_head, N).
template <typename T>
[[kernel]] void rope_neox_mb(
    device T* x                       [[buffer(0)]],  // [N, n_head, head_dim]
    const device int* position        [[buffer(1)]],  // [N] per-token positions
    const constant float& scale       [[buffer(2)]],
    const constant float& base        [[buffer(3)]],
    const constant int& head_dim      [[buffer(4)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);            // freq index, 0..half-1
  const int h = int(pos.y);            // head index
  const int m = int(pos.z);            // token row
  const int half_rd = int(grid.x);
  const int n_head  = int(grid.y);

  float d = static_cast<float>(i) / static_cast<float>(half_rd);
  float inv_freq = exp2(-d * base);
  float L = scale * static_cast<float>(position[m]);
  float theta = L * inv_freq;
  float costheta = fast::cos(theta);
  float sintheta = fast::sin(theta);

  const int base_x = m * (n_head * head_dim);
  const int i1 = base_x + h * head_dim + i;
  const int i2 = i1 + half_rd;
  float x1 = static_cast<float>(x[i1]);
  float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_neox_mb(name, itype)                    \
  template [[host_name("rope_neox_mb_" #name)]]                  \
  [[kernel]] void rope_neox_mb<itype>(                           \
      device itype*, const device int*, const constant float&,   \
      const constant float&, const constant int&, uint3, uint3);

instantiate_rope_neox_mb(bfloat16, bfloat)

// ── Proportional (gemma4) partial NEOX RoPE ──────────────────────────────────
// `rope_neox_decode` infers two things from the launch: the pair offset and the
// frequency divisor are both `grid.x` = rotary_dims/2. That is MLX's
// `fast::rope(x, dims=rotary_dims)`, which is what qwen3.6 wants -- it rotates
// the first `rotary_dims` channels, pairing (i, i+rotary_dims/2).
//
// gemma4's full-attention layers do NOT. `ProportionalRoPE` calls
// `fast::rope(x, dims=HEAD_DIM, freqs=...)` with the tail of `freqs` set to
// infinity, so the pairing is over the whole head -- (i, i+head_dim/2) -- and
// the exponent divides by head_dim. Verified against mlx_lm at head_dim=512,
// rotary=128: the channels that move are [0,63] and [256,319], not [0,127].
//
// Both attention types run this one kernel: on a sliding layer rotary_dims ==
// head_dim and it reduces exactly to `rope_neox_decode`. The rotated pair count
// is grid.x, so no extra buffer is needed -- only which value indexes what.
template <typename T>
[[kernel]] void rope_neox_prop_decode(
    device T* x                       [[buffer(0)]],
    const device int* position        [[buffer(1)]],
    const constant float& scale       [[buffer(2)]],
    const constant float& base        [[buffer(3)]],  // log2(theta)
    const constant int& head_dim      [[buffer(4)]],
    uint2 pos  [[thread_position_in_grid]]) {
  const int i = int(pos.x);            // rotated pair index, 0..rotary_dims/2-1
  const int h = int(pos.y);
  const int half_hd = head_dim / 2;    // NEOX pair offset, over the FULL head

  float d = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
  float inv_freq = exp2(-d * base);
  float theta = scale * static_cast<float>(position[0]) * inv_freq;
  float costheta = fast::cos(theta);
  float sintheta = fast::sin(theta);

  const int i1 = h * head_dim + i;
  const int i2 = i1 + half_hd;
  float x1 = static_cast<float>(x[i1]);
  float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_prop(name, itype)                       \
  template [[host_name("rope_neox_prop_decode_" #name)]]         \
  [[kernel]] void rope_neox_prop_decode<itype>(                  \
      device itype*, const device int*, const constant float&,   \
      const constant float&, const constant int&, uint2);

instantiate_rope_prop(bfloat16, bfloat)

// M>1 counterpart: token m rotates by its own position[m]. x is token-major
// [N, n_head, head_dim]. Reduces to rope_neox_prop_decode at N=1.
template <typename T>
[[kernel]] void rope_neox_prop_mb(
    device T* x                       [[buffer(0)]],
    const device int* position        [[buffer(1)]],
    const constant float& scale       [[buffer(2)]],
    const constant float& base        [[buffer(3)]],
    const constant int& head_dim      [[buffer(4)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int m = int(pos.z);
  const int n_head = int(grid.y);
  const int half_hd = head_dim / 2;

  float d = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
  float inv_freq = exp2(-d * base);
  float theta = scale * static_cast<float>(position[m]) * inv_freq;
  float costheta = fast::cos(theta);
  float sintheta = fast::sin(theta);

  const int i1 = (m * n_head + h) * head_dim + i;
  const int i2 = i1 + half_hd;
  float x1 = static_cast<float>(x[i1]);
  float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_prop_mb(name, itype)                    \
  template [[host_name("rope_neox_prop_mb_" #name)]]             \
  [[kernel]] void rope_neox_prop_mb<itype>(                      \
      device itype*, const device int*, const constant float&,   \
      const constant float&, const constant int&, uint3, uint3);

instantiate_rope_prop_mb(bfloat16, bfloat)

// ── RoPE from a supplied frequency table (gpt-oss / YaRN) ───────────────────
//
// gpt-oss scales its rotary frequencies by YaRN: a per-dimension interpolation
// between the original and extended context, parameterised by beta_fast,
// beta_slow, a factor and the original window. That is a closed form over
// `head_dim/2` values -- 32 of them here -- and it does not depend on the
// position, so it is arithmetic the host does once at setup rather than
// arithmetic every head redoes every token.
//
// So this variant reads `inv_freq[i]` instead of deriving it from a base. It is
// also the general shape: any rope whose frequencies are a table rather than a
// geometric series (YaRN, llama3, longrope) binds here without a new kernel.
template <typename T>
[[kernel]] void rope_neox_freqs_decode(
    device T* x                       [[buffer(0)]],  // in-place [n_head, head_dim]
    const device int* position        [[buffer(1)]],
    const constant float& scale       [[buffer(2)]],
    const device float* inv_freq      [[buffer(3)]],  // [rotary_dims/2]
    const constant int& head_dim      [[buffer(4)]],
    // YaRN's `mscale`: an attention-temperature correction that scales q and k
    // by `0.1*log(factor)+1` -- 1.3466 at factor 32. It rides here rather than
    // in a dispatch of its own because rotation is linear, so scaling before
    // the rotation and scaling after it are the same thing.
    const constant float& mscale      [[buffer(5)]],
    uint2 pos  [[thread_position_in_grid]],
    uint2 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int half_rd = int(grid.x);

  const float theta = scale * static_cast<float>(position[0]) * inv_freq[i];
  const float costheta = fast::cos(theta);
  const float sintheta = fast::sin(theta);

  const int i1 = h * head_dim + i;
  const int i2 = i1 + half_rd;
  const float x1 = static_cast<float>(x[i1]);
  const float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(mscale * (x1 * costheta - x2 * sintheta));
  x[i2] = static_cast<T>(mscale * (x1 * sintheta + x2 * costheta));
}

// The same rotation over N token rows.
//
// `pos.z` is the row: it reads `position[row]` and strides `x` by the row's
// whole width, which is `n_heads * head_dim` and NOT derivable from this
// kernel's grid -- q and k have different head counts and share the kernel.
template <typename T>
[[kernel]] void rope_neox_freqs_mb(
    device T* x                       [[buffer(0)]],  // [N, n_head, head_dim]
    const device int* position        [[buffer(1)]],  // [N]
    const constant float& scale       [[buffer(2)]],
    const device float* inv_freq      [[buffer(3)]],  // [rotary_dims/2]
    const constant int& head_dim      [[buffer(4)]],
    const constant float& mscale      [[buffer(5)]],
    const constant int& row_stride    [[buffer(6)]],  // n_heads * head_dim
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int row = int(pos.z);
  const int half_rd = int(grid.x);

  const float theta = scale * static_cast<float>(position[row]) * inv_freq[i];
  const float costheta = fast::cos(theta);
  const float sintheta = fast::sin(theta);

  device T* xr = x + size_t(row) * size_t(row_stride);
  const int i1 = h * head_dim + i;
  const int i2 = i1 + half_rd;
  const float x1 = static_cast<float>(xr[i1]);
  const float x2 = static_cast<float>(xr[i2]);
  xr[i1] = static_cast<T>(mscale * (x1 * costheta - x2 * sintheta));
  xr[i2] = static_cast<T>(mscale * (x1 * sintheta + x2 * costheta));
}

#define instantiate_rope_freqs(name, itype)                        \
  template [[host_name("rope_neox_freqs_decode_" #name)]]          \
  [[kernel]] void rope_neox_freqs_decode<itype>(                   \
      device itype*, const device int*, const constant float&,     \
      const device float*, const constant int&, const constant float&, uint2, uint2); \
  template [[host_name("rope_neox_freqs_mb_" #name)]]              \
  [[kernel]] void rope_neox_freqs_mb<itype>(                       \
      device itype*, const device int*, const constant float&,     \
      const device float*, const constant int&, const constant float&, \
      const constant int&, uint3, uint3);

instantiate_rope_freqs(bfloat16, bfloat)
