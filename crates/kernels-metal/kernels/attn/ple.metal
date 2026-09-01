#include <metal_stdlib>
using namespace metal;

// The PLE n-gram hasher (qwen4), ported organ for organ from
// `kernels-cuda/kernels/attn/ple.cuh`. Every token's hashed n-gram table rows,
// one column per head. The hash is the reference's own — token ids multiplied
// by seed-derived odd constants, xor-folded, reduced modulo a per-head prime
// plus a per-head offset.
//
// **THE CONSTANTS ARRIVE AS A BUFFER AND NOT AS AN AGGREGATE.** The CUDA
// sibling hands `PleHash` across the launch ABI by value (`ArgValue::Bytes`);
// this plane's `ArgValue` has no by-value blob seat and growing one would have
// to cross the ICB's eight-byte scalar arena and the recording's `Copy` `Arg`.
// So the same numbers ride ONE `ulong` plane, laid down and written once at
// load by `engine_metal::scratch`, in the order
//
//     [ mults[0..ngram] ][ primes[0..heads] ][ offsets[0..heads] ]
//
// which is the field order of `PleHash` with its three fixed-size arrays cut
// to the lengths the node actually states. `ngram`, `heads`, `heads_per_ngram`
// and `eos` stay scalars, as they are on the CUDA plane.
//
// The window cache is a per-lane state slab of `ngram - 1` i32 cells storing
// PREVIOUS token ids as `id + 1`, so a zeroed slot reads as "no history" and
// the reference's eos padding falls out of the sentinel rather than out of a
// separate reset. `engine_metal::store`'s `clear` is what zeroes a slot when a
// sequence opens, which is the same discipline the recurrent conv keeps.
//
// **THE STAGED-GEOMETRY `win` SEAT IS NOT PORTED.** The CUDA kernels grew a
// `const u32* win` for graph-bucket replay; the null path is the semantics and
// this plane has no recording to replay, so every index below is the fire's
// own — `hc.metal`'s argument, one family over.

constant constexpr int PLE_MAX_NGRAM = 4;
constant constexpr int PLE_MAX_HEADS = 32;

/// Apply the eos-segmentation rule to the raw window: a previous id is
/// replaced by eos when a NEARER previous id is eos (the window crossed a
/// sequence boundary).
inline void ple_mask_window(thread int* window, int ngram, int eos) {
  bool crossed = false;
  for (int p = 1; p < ngram; ++p) {
    if (crossed) {
      window[p] = eos;
    }
    if (window[p] == eos) {
      crossed = true;
    }
  }
}

/// Hash the window `[t, p1, p2, ...]` (newest first) for every head.
///
/// The products are taken in `ulong`, which is what lets this agree with
/// torch's `long`: the multipliers are odd and bounded by
/// `i64::MAX / vocab`, so `id * mult` cannot overflow sixty-four bits.
inline void ple_hash_row(
    const device ulong* hash,
    thread const int* window,
    int ngram,
    int heads,
    int heads_per_ngram,
    thread int* out) {
  const device ulong* mults = hash;
  const device ulong* primes = hash + ngram;
  const device ulong* offsets = hash + ngram + heads;
  for (int order = 2; order <= ngram; ++order) {
    ulong mixed = ulong(window[0]) * mults[0];
    for (int p = 1; p < order; ++p) {
      mixed ^= ulong(window[p]) * mults[p];
    }
    const int base = (order - 2) * heads_per_ngram;
    for (int k = 0; k < heads_per_ngram; ++k) {
      const int head = base + k;
      out[head] = int(mixed % primes[head] + offsets[head]);
    }
  }
}

/// Decode form: one thread per lane row. Reads the lane's window state, hashes
/// the one new token, shifts the window.
[[kernel]] void ple_ngram_ids_update(
    const device int* ids                   [[buffer(0)]],
    device int* state                       [[buffer(1)]],
    const device uint* slots                [[buffer(2)]],
    const device ulong* hash                [[buffer(3)]],
    device int* ngram_ids                   [[buffer(4)]],
    const constant int& ngram               [[buffer(5)]],
    const constant int& heads               [[buffer(6)]],
    const constant int& heads_per_ngram     [[buffer(7)]],
    const constant int& eos                 [[buffer(8)]],
    uint pos [[thread_position_in_grid]]) {
  const int r = int(pos);
  const int span = ngram - 1;
  const size_t slab = size_t(slots[r]) * size_t(span);

  int window[PLE_MAX_NGRAM];
  const int fresh = ids[r];
  window[0] = fresh;
  for (int p = 1; p <= span; ++p) {
    const int cell = state[slab + size_t(span - p)];
    window[p] = cell == 0 ? eos : cell - 1;
  }
  ple_mask_window(window, ngram, eos);

  int out[PLE_MAX_HEADS];
  ple_hash_row(hash, window, ngram, heads, heads_per_ngram, out);
  for (int k = 0; k < heads; ++k) {
    ngram_ids[size_t(r) * size_t(heads) + size_t(k)] = out[k];
  }

  // The shift is ascending and in place, which is what makes one seat enough:
  // cell `p` is read before it is written and cell `p + 1` is written after it
  // is read.
  for (int p = 0; p + 1 < span; ++p) {
    state[slab + size_t(p)] = state[slab + size_t(p + 1)];
  }
  state[slab + size_t(span - 1)] = fresh + 1;
}

/// Prefill form: one thread per request, walking that request's tokens in
/// order — `ssm_causal_conv1d.metal`'s chunked shape, for its reason. Every
/// token past the first `span` of a segment reads its window out of the fire's
/// own rows; only the leading ones reach into the lane's state.
[[kernel]] void ple_ngram_ids_chunked(
    const device int* ids                   [[buffer(0)]],
    const device int* indptr                [[buffer(1)]],
    device int* state                       [[buffer(2)]],
    const device uint* slots                [[buffer(3)]],
    const device ulong* hash                [[buffer(4)]],
    device int* ngram_ids                   [[buffer(5)]],
    const constant int& ngram               [[buffer(6)]],
    const constant int& heads               [[buffer(7)]],
    const constant int& heads_per_ngram     [[buffer(8)]],
    const constant int& eos                 [[buffer(9)]],
    uint pos [[thread_position_in_grid]]) {
  const int r = int(pos);
  const int begin = indptr[r];
  const int end = indptr[r + 1];
  if (end <= begin) {
    return;
  }
  const int rows = end - begin;
  const int span = ngram - 1;
  const size_t slab = size_t(slots[begin]) * size_t(span);

  for (int t = 0; t < rows; ++t) {
    int window[PLE_MAX_NGRAM];
    window[0] = ids[begin + t];
    for (int p = 1; p <= span; ++p) {
      if (t - p >= 0) {
        window[p] = ids[begin + t - p];
      } else {
        const int cell = state[slab + size_t(span - (p - t))];
        window[p] = cell == 0 ? eos : cell - 1;
      }
    }
    ple_mask_window(window, ngram, eos);

    int out[PLE_MAX_HEADS];
    ple_hash_row(hash, window, ngram, heads, heads_per_ngram, out);
    for (int k = 0; k < heads; ++k) {
      ngram_ids[size_t(begin + t) * size_t(heads) + size_t(k)] = out[k];
    }
  }

  // The new window: the last `span` ids of (state ++ segment), staged before
  // any of it is written back because a short segment reads cells this loop
  // also lands on.
  int next[PLE_MAX_NGRAM];
  for (int p = 0; p < span; ++p) {
    const int src = rows - span + p;
    next[p] = src >= 0 ? ids[begin + src] + 1 : state[slab + size_t(p + rows)];
  }
  for (int p = 0; p < span; ++p) {
    state[slab + size_t(p)] = next[p];
  }
}
