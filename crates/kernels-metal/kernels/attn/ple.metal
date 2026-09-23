#include <metal_stdlib>
using namespace metal;


constant constexpr int PLE_MAX_NGRAM = 4;
constant constexpr int PLE_MAX_HEADS = 32;

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

// Engram hashes tokenizer-compressed ids: the window holds raw ids (so the
// state and the eos barrier stay in the tokenizer's id space) and is mapped
// after masking, so the pre-start padding maps like every other id.
inline void ple_map_window(const device int* map, int has_map, thread int* window, int ngram) {
  if (!has_map) return;
  for (int p = 0; p < ngram; ++p) window[p] = map[window[p]];
}

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
    const device int* map                   [[buffer(9)]],
    const constant int& has_map             [[buffer(10)]],
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
  ple_map_window(map, has_map, window, ngram);

  int out[PLE_MAX_HEADS];
  ple_hash_row(hash, window, ngram, heads, heads_per_ngram, out);
  for (int k = 0; k < heads; ++k) {
    ngram_ids[size_t(r) * size_t(heads) + size_t(k)] = out[k];
  }

  for (int p = 0; p + 1 < span; ++p) {
    state[slab + size_t(p)] = state[slab + size_t(p + 1)];
  }
  state[slab + size_t(span - 1)] = fresh + 1;
}

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
    const device int* map                   [[buffer(10)]],
    const constant int& has_map             [[buffer(11)]],
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
    ple_map_window(map, has_map, window, ngram);

    int out[PLE_MAX_HEADS];
    ple_hash_row(hash, window, ngram, heads, heads_per_ngram, out);
    for (int k = 0; k < heads; ++k) {
      ngram_ids[size_t(begin + t) * size_t(heads) + size_t(k)] = out[k];
    }
  }

  int next[PLE_MAX_NGRAM];
  for (int p = 0; p < span; ++p) {
    const int src = rows - span + p;
    next[p] = src >= 0 ? ids[begin + src] + 1 : state[slab + size_t(p + rows)];
  }
  for (int p = 0; p < span; ++p) {
    state[slab + size_t(p)] = next[p];
  }
}

[[kernel]] void ple_ngram_ids_committed(
    const device int* ids                   [[buffer(0)]],
    const device int* indptr                [[buffer(1)]],
    const device int* replay                [[buffer(2)]],
    const device int* commit                [[buffer(3)]],
    const device int* slots                 [[buffer(4)]],
    const constant int& lane0               [[buffer(5)]],
    device int* state                       [[buffer(6)]],
    const device ulong* hash                [[buffer(7)]],
    device int* ngram_ids                   [[buffer(8)]],
    const constant int& ngram               [[buffer(9)]],
    const constant int& heads               [[buffer(10)]],
    const constant int& heads_per_ngram     [[buffer(11)]],
    const constant int& eos                 [[buffer(12)]],
    const device int* map                   [[buffer(13)]],
    const constant int& has_map             [[buffer(14)]],
    uint pos [[thread_position_in_grid]]) {
  const int r = int(pos);
  int begin = indptr[r];
  for (int j = 0; j < r; ++j) {
    begin += replay[lane0 + j];
  }
  const int rows = (indptr[r + 1] - indptr[r]) + replay[lane0 + r];
  if (rows <= 0) {
    return;
  }
  const int slot = slots[lane0 + r];
  if (slot < 0) {
    return;
  }
  const int span = ngram - 1;
  const size_t slab = size_t(slot) * size_t(span);

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
    ple_map_window(map, has_map, window, ngram);

    const int replayed = replay[lane0 + r];
    if (t < replayed) {
      continue;
    }
    int out[PLE_MAX_HEADS];
    ple_hash_row(hash, window, ngram, heads, heads_per_ngram, out);
    const size_t own = size_t(indptr[r] + (t - replayed));
    for (int k = 0; k < heads; ++k) {
      ngram_ids[own * size_t(heads) + size_t(k)] = out[k];
    }
  }

  int keep = commit[lane0 + r];
  if (keep > rows) {
    keep = rows;
  }
  if (keep <= 0) {
    return;
  }
  int next[PLE_MAX_NGRAM];
  for (int p = 0; p < span; ++p) {
    const int src = keep - span + p;
    next[p] = src >= 0 ? ids[begin + src] + 1 : state[slab + size_t(p + keep)];
  }
  for (int p = 0; p < span; ++p) {
    state[slab + size_t(p)] = next[p];
  }
}
