#include <metal_stdlib>
using namespace metal;

/// **MULTIMODAL ROTARY: ONE ROW, THREE POSITIONS** — the Metal mirror of
/// `kernels-cuda`'s `elemwise/rope_mrope.cuh` (`.wiki/alto/multimodal.md`
/// §2's second op).
///
/// `neox_prop_mb`'s arithmetic, unchanged, over a position that is a TRIPLE
/// instead of a scalar. An image lane's row does not sit at one place in one
/// sequence; it sits at `(t, h, w)` — time, and the patch's row and column in
/// its grid — and the sections say which of the three each frequency pair
/// turns by. Everything else is the scalar kernel next door: the same
/// `exp2(-d * base)` frequency over the FULL head width, the same
/// `(i, i + head_dim / 2)` pair, so a row whose triple is `(p, p, p)` comes
/// out where `neox_prop_mb` at `p` would have put it, to the last bit the two
/// expressions can share.
///
/// **The split is INTERLEAVED.** Frequency pairs alternate `t, h, w, t, h,
/// w, ...` for as far as the sections reach: pair `i` turns by `h` when
/// `i % 3 == 1` and `i < 3 * s1`, by `w` when `i % 3 == 2` and `i < 3 * s2`,
/// and by `t` otherwise — so a checkpoint stating `[11, 11, 10]` over a
/// 64-wide head gets exactly 11 `h` pairs, 10 `w` pairs, and the remaining 11
/// of the interleaved prefix plus every pair above it turning by `t`. `s0` is
/// read by the shape of that "otherwise" and never by name; it is taken
/// anyway so the trace's three numbers arrive as three numbers, and so this
/// entry's argument list can be read against its CUDA twin's.
///
/// **The rotated prefix is the GRID, not a branch.** The scalar partial
/// kernel walks every pair of the head and skips the ones at or above
/// `rotary_dim / 2`; this plane's rope family sizes the launch to the rotated
/// pairs instead (`rope::rope_grid`), so the tail a checkpoint does not
/// rotate is never dispatched rather than dispatched and dropped.
///
/// **One tensor per launch, which is the plane's shape and not the twin's.**
/// The CUDA kernel rotates q and k in one grid over `num_q_heads +
/// num_kv_heads`; every arm of `elemwise/rope_neox.metal` takes one tensor
/// and the entry fires twice. Following the neighbour keeps `num_kv_heads`
/// out of the shader entirely — a `k`-shaped absence is a launch not made,
/// rather than a zero the kernel has to mean something by.
template <typename T>
[[kernel]] void rope_mrope_interleaved(
    device T* x                       [[buffer(0)]],
    const device int* positions       [[buffer(1)]],
    const constant float& base        [[buffer(2)]],
    const constant int& head_dim      [[buffer(3)]],
    const constant int& s0            [[buffer(4)]],
    const constant int& s1            [[buffer(5)]],
    const constant int& s2            [[buffer(6)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int m = int(pos.z);
  const int n_head = int(grid.y);
  const int half_hd = head_dim / 2;

  const int pos_t = positions[3 * m + 0];
  const int pos_h = positions[3 * m + 1];
  const int pos_w = positions[3 * m + 2];
  (void)s0;

  int axis_pos;
  const int r = i % 3;
  if (r == 1 && i < 3 * s1) {
    axis_pos = pos_h;
  } else if (r == 2 && i < 3 * s2) {
    axis_pos = pos_w;
  } else {
    axis_pos = pos_t;
  }

  const float d = 2.0f * static_cast<float>(i) / static_cast<float>(head_dim);
  const float inv_freq = exp2(-d * base);
  const float theta = static_cast<float>(axis_pos) * inv_freq;
  const float costheta = fast::cos(theta);
  const float sintheta = fast::sin(theta);

  const size_t i1 =
      (size_t(m) * size_t(n_head) + size_t(h)) * size_t(head_dim) + size_t(i);
  const size_t i2 = i1 + size_t(half_hd);
  const float x1 = static_cast<float>(x[i1]);
  const float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_mrope(name, itype)                                \
  template [[host_name("rope_mrope_interleaved_" #name)]]                  \
  [[kernel]] void rope_mrope_interleaved<itype>(                           \
      device itype*, const device int*, const constant float&,             \
      const constant int&, const constant int&, const constant int&,       \
      const constant int&, uint3, uint3);

instantiate_rope_mrope(bfloat16, bfloat)

/// **THE TOWER'S ROTATION: CONTIGUOUS SECTIONS, AND EACH RESTARTS THE
/// LADDER** — the Metal mirror of `kernels-cuda`'s `rope_mrope_blocked`
/// (`.wiki/alto/multimodal.md` §6.3).
///
/// The same pairing as `rope_mrope_interleaved` above — `(i, i + head_dim/2)`,
/// which is `rotate_half` — over the same `[rows, 3]` position stream. What
/// differs is which pair takes which axis, and at what frequency:
///
/// - the sections are CONTIGUOUS BLOCKS. Pairs `[0, s0)` turn by `t`,
///   `[s0, s0+s1)` by `h`, `[s0+s1, s0+s1+s2)` by `w`. The tower states
///   `[0, head_dim/4, head_dim/4]`, so it turns by `(h, w)` and reads no `t`
///   at all — `s0 == 0` is how a two-axis rotation is spelled here, rather
///   than by a second position shape.
/// - and each block RESTARTS the frequency ladder. The `i`-th pair OF ITS
///   BLOCK turns at `theta^(-2i / total)` where `total = s0 + s1 + s2`.
///
/// That second half is the part nobody would guess and the part a wrong
/// kernel still looks plausible under.
/// `Qwen3_5VisionRotaryEmbedding(head_dim / 2)` builds `head_dim/4`
/// frequencies over a `head_dim/2`-wide ladder, and `freqs[pos_ids].flatten(1)`
/// indexes that ONE ladder once per axis before concatenating — so the
/// exponent's numerator counts WITHIN the block and its denominator is the
/// ladder's width, which is `total` exactly when the sections tile the
/// rotated pairs, as the tower's do.
///
/// **THE FREQUENCY IS `exp2(-d * base)` AND NOT `powf`**, which is this
/// plane's spelling of the twin's `powf(theta, ...)` — `base = log2(theta)`
/// arrives from the entry, exactly as it does for the interleaved arm above
/// and for every arm of `rope_neox.metal`. Note that `d` is
/// `2 * within / total` here where the interleaved arm's is
/// `2 * i / head_dim`: the blocked ladder's denominator is the SECTIONS' own
/// width and not the head's, which is §6.3's whole point.
///
/// **PAIRS PAST `total` ARE NOT DISPATCHED RATHER THAN DISPATCHED AND
/// DROPPED.** The twin walks every pair of the head and `continue`s past
/// both `rotary_dim / 2` and `total`; this plane's rope family sizes the
/// launch to the pairs that turn, so the entry hands this shader
/// `min(rotary_dim / 2, total)` lanes on `x` and the tail is never a thread.
/// Same pairs rotated, same pairs left alone.
template <typename T>
[[kernel]] void rope_mrope_blocked(
    device T* x                       [[buffer(0)]],
    const device int* positions       [[buffer(1)]],
    const constant float& base        [[buffer(2)]],
    const constant int& head_dim      [[buffer(3)]],
    const constant int& s0            [[buffer(4)]],
    const constant int& s1            [[buffer(5)]],
    const constant int& s2            [[buffer(6)]],
    uint3 pos  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const int i = int(pos.x);
  const int h = int(pos.y);
  const int m = int(pos.z);
  const int n_head = int(grid.y);
  const int half_hd = head_dim / 2;
  const int total = s0 + s1 + s2;

  const int axis_of[3] = {positions[3 * m + 0],
                          positions[3 * m + 1],
                          positions[3 * m + 2]};

  int axis;
  int within;
  if (i < s0) {
    axis = 0;
    within = i;
  } else if (i < s0 + s1) {
    axis = 1;
    within = i - s0;
  } else {
    axis = 2;
    within = i - s0 - s1;
  }

  const float d = 2.0f * static_cast<float>(within) / static_cast<float>(total);
  const float inv_freq = exp2(-d * base);
  const float theta = static_cast<float>(axis_of[axis]) * inv_freq;
  const float costheta = fast::cos(theta);
  const float sintheta = fast::sin(theta);

  const size_t i1 =
      (size_t(m) * size_t(n_head) + size_t(h)) * size_t(head_dim) + size_t(i);
  const size_t i2 = i1 + size_t(half_hd);
  const float x1 = static_cast<float>(x[i1]);
  const float x2 = static_cast<float>(x[i2]);
  x[i1] = static_cast<T>(x1 * costheta - x2 * sintheta);
  x[i2] = static_cast<T>(x1 * sintheta + x2 * costheta);
}

#define instantiate_rope_mrope_blocked(name, itype)                        \
  template [[host_name("rope_mrope_blocked_" #name)]]                      \
  [[kernel]] void rope_mrope_blocked<itype>(                               \
      device itype*, const device int*, const constant float&,             \
      const constant int&, const constant int&, const constant int&,       \
      const constant int&, uint3, uint3);

instantiate_rope_mrope_blocked(bfloat16, bfloat)
