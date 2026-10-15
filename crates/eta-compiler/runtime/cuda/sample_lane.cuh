
// ------------------------------------------------------------ lane sampler --
//
// A decode lane's sampler — a softmax over the vocabulary, a top-p
// (`PivotThreshold(CummassLe)`) or top-k (`RankLe`) cut of it, the cut logits
// masked, Gumbel noise added, an argmax — on several blocks a lane instead of
// one. Each block walks its stretch of the row; the reductions and every pass
// of the cut meet through the lane's temporary arena, the blocks waiting for
// one another at a barrier (a lane's blocks are launched together, a grid the
// device holds at once). Every merge runs in block order and every decision
// after it is a function of the merged results alone, so the blocks decide
// alike and the verdicts are the one-block cut's: the sum follows the same
// width-32 tree (`canonical_reduce`), and the cut is `m1_cut_row` /
// `m1_rank_row` over the group's hooks.
//
// The arena, per block (`temporary_stride` apart): a barrier word (block 0's
// is the lane's; zeroed before the launch), then two exchange buffers, used by
// alternate merges (a block is at most one merge ahead of the slowest).

#define M1_LANE_HALF 18432u
#define M1_LANE_HEAD 64u
#define M1_LANE_MAX_SLICES 64u

struct M1Lane {
  m1_u32 lo;
  m1_u32 hi;
  m1_u32 slice;
  m1_u32 slices;
  m1_u8* arena;
  m1_u32 stride;
  m1_u32 parity;
  m1_u32 arrivals;

  __device__ __forceinline__ void row(m1_u32 len) {
    // Stretches of whole 32-element chunks (the sum's tree), 16-byte aligned.
    const m1_u32 chunks = (len + 31u) >> 5;
    const m1_u32 per = (chunks + slices - 1u) / slices;
    lo = slice * per * 32u;
    hi = lo + per * 32u;
    if (lo > len) lo = len;
    if (hi > len) hi = len;
  }

  __device__ __forceinline__ m1_u8* buffer(m1_u32 block) const {
    return arena + (m1_u64)block * stride + M1_LANE_HEAD + parity * M1_LANE_HALF;
  }

  // The exchange words are written and read at device scope (each block's
  // cache stays its own: no write-back or invalidation of it at a barrier).
  __device__ __forceinline__ void put32(m1_u8* at, m1_u32 index, m1_u32 v) const {
    M1_LANE_PUT32(reinterpret_cast<m1_u32*>(at) + index, v);
  }
  __device__ __forceinline__ void put64(m1_u8* at, m1_u32 index, m1_u64 v) const {
    M1_LANE_PUT64(reinterpret_cast<m1_u64*>(at) + index, v);
  }
  __device__ __forceinline__ m1_u32 get32(m1_u32 block, m1_u32 byte, m1_u32 index) const {
    return M1_LANE_GET32(reinterpret_cast<m1_u32*>(buffer(block) + byte) + index);
  }
  __device__ __forceinline__ m1_u64 get64(m1_u32 block, m1_u32 byte, m1_u32 index) const {
    return M1_LANE_GET64(reinterpret_cast<m1_u64*>(buffer(block) + byte) + index);
  }

  // Every block of the lane here before any goes on; what each put before
  // is then what all get.
  __device__ __forceinline__ void barrier() {
    M1_LANE_RELEASE();
    __syncthreads();
    if (threadIdx.x == 0u) {
      m1_u32* word = reinterpret_cast<m1_u32*>(arena);
      arrivals += 1u;
      atomicAdd(word, 1u);
      const m1_u32 target = arrivals * slices;
      while (M1_LANE_LOAD(word) < target) M1_LANE_PAUSE();
      M1_LANE_ACQUIRE();
    }
    __syncthreads();
  }

  // After a pass: this block's results out (put before), everyone's in.
  __device__ __forceinline__ void publish() { barrier(); }
  __device__ __forceinline__ void done() {
    __syncthreads();
    parity ^= 1u;
  }

  // The merges read every block's buffer at once (one load a thread a
  // block-and-item, not a block after another) into LDS atomics: the sums,
  // minima and unions are integer, so their order is not their result.

  __device__ __forceinline__ void entry(M1CutShared& cut) {
    m1_u8* out = buffer(slice);
    if (threadIdx.x == 0u) {
      put32(out, 0u, cut.first_nan);
      put32(out, 1u, cut.negative);
      put32(out, 2u, cut.wide);
      put32(out, 3u, cut.key_least);
      put32(out, 4u, cut.key_most);
    }
    publish();
    if (threadIdx.x == 0u) {
      cut.first_nan = 0xFFFFFFFFu;
      cut.negative = 0u;
      cut.wide = 0u;
      cut.key_least = 0xFFFFFFFFu;
      cut.key_most = 0u;
    }
    __syncthreads();
    if (threadIdx.x < slices) {
      const m1_u32 b = threadIdx.x;
      const m1_u32 v0 = get32(b, 0u, 0u), v1 = get32(b, 0u, 1u), v2 = get32(b, 0u, 2u);
      const m1_u32 v3 = get32(b, 0u, 3u), v4 = get32(b, 0u, 4u);
      atomicMin(&cut.first_nan, v0);
      atomicOr(&cut.negative, v1);
      atomicOr(&cut.wide, v2);
      atomicMin(&cut.key_least, v3);
      atomicMax(&cut.key_most, v4);
    }
    done();
  }

  // A select pass: the bucket weights (and least indices, and gathered tie
  // values, where the pass keeps them). Layout: 256 u64 weights, 256 u32
  // least indices, the tie slots and their overflow word.
  __device__ __forceinline__ void select(M1CutShared& cut, bool least, bool ties) {
    m1_u8* out = buffer(slice);
    for (m1_u32 b = threadIdx.x; b < 256u; b += blockDim.x) {
      put64(out, b, cut.weight[b]);
      if (least) put32(out + 2048u, b, cut.least[b]);
    }
    if (ties) {
      for (m1_u32 t = threadIdx.x; t < M1_CUT_TIE_SLOTS; t += blockDim.x)
        put32(out + 3072u, t, cut.tie_slot[t]);
      if (threadIdx.x == 0u) put32(out + 3072u, M1_CUT_TIE_SLOTS, cut.overflow);
    }
    publish();
    for (m1_u32 b = threadIdx.x; b < 256u; b += blockDim.x) {
      cut.weight[b] = 0ull;
      if (least) cut.least[b] = 0xFFFFFFFFu;
    }
    if (ties) {
      for (m1_u32 t = threadIdx.x; t < M1_CUT_TIE_SLOTS; t += blockDim.x) cut.tie_slot[t] = 0u;
      if (threadIdx.x == 0u) {
        cut.ties = 0u;
        cut.overflow = 0u;
      }
    }
    __syncthreads();
    for (m1_u32 at = threadIdx.x; at < 256u * slices; at += 4u * blockDim.x) {
      m1_u64 w[4];
      m1_u32 l[4];
#pragma unroll
      for (m1_u32 u = 0u; u < 4u; ++u) {
        const m1_u32 i = at + u * blockDim.x;
        const bool live = i < 256u * slices;
        w[u] = live ? get64(i >> 8, 0u, i & 255u) : 0ull;
        l[u] = live && least ? get32(i >> 8, 2048u, i & 255u) : 0xFFFFFFFFu;
      }
#pragma unroll
      for (m1_u32 u = 0u; u < 4u; ++u) {
        const m1_u32 i = at + u * blockDim.x;
        if (w[u] != 0ull) atomicAdd(&cut.weight[i & 255u], w[u]);
        if (l[u] != 0xFFFFFFFFu) atomicMin(&cut.least[i & 255u], l[u]);
      }
    }
    if (ties) {
      if (threadIdx.x < slices && get32(threadIdx.x, 3072u, M1_CUT_TIE_SLOTS) != 0u)
        cut.overflow = 1u;
      __syncthreads();
      // The union of the blocks' tie values, into the fresh hash set.
      for (m1_u32 at = threadIdx.x; at < M1_CUT_TIE_SLOTS * slices; at += blockDim.x) {
        const m1_u32 key = get32(at / M1_CUT_TIE_SLOTS, 3072u, at % M1_CUT_TIE_SLOTS);
        if (key != 0u) m1_cut_tie(key, cut);
      }
    }
    done();
  }

  __device__ __forceinline__ void weights(M1CutShared& cut) {
    m1_u8* out = buffer(slice);
    for (m1_u32 b = threadIdx.x; b < 256u; b += blockDim.x) put64(out, b, cut.weight[b]);
    publish();
    for (m1_u32 b = threadIdx.x; b < 256u; b += blockDim.x) cut.weight[b] = 0ull;
    __syncthreads();
    for (m1_u32 at = threadIdx.x; at < 256u * slices; at += 4u * blockDim.x) {
      m1_u64 w[4];
#pragma unroll
      for (m1_u32 u = 0u; u < 4u; ++u) {
        const m1_u32 i = at + u * blockDim.x;
        w[u] = i < 256u * slices ? get64(i >> 8, 0u, i & 255u) : 0ull;
      }
#pragma unroll
      for (m1_u32 u = 0u; u < 4u; ++u)
        if (w[u] != 0ull) atomicAdd(&cut.weight[(at + u * blockDim.x) & 255u], w[u]);
    }
    done();
  }

  // A gather: the blocks' elements, concatenated in block order (the walk
  // sorts them).
  __device__ __forceinline__ void chunk(M1CutShared& cut) {
    __shared__ m1_u32 starts[M1_LANE_MAX_SLICES + 1u];
    m1_u8* out = buffer(slice);
    const m1_u32 mine = cut.count < M1_CUT_CHUNK ? cut.count : M1_CUT_CHUNK;
    if (threadIdx.x == 0u) put64(out, 0u, mine);
    for (m1_u32 n = threadIdx.x; n < mine; n += blockDim.x) put64(out, 1u + n, cut.chunk[n]);
    publish();
    if (threadIdx.x < slices) starts[threadIdx.x + 1u] = (m1_u32)get64(threadIdx.x, 0u, 0u);
    __syncthreads();
    if (threadIdx.x == 0u) {
      starts[0] = 0u;
      for (m1_u32 s = 1u; s <= slices; ++s) starts[s] += starts[s - 1u];
      cut.count = starts[slices];
    }
    __syncthreads();
    const m1_u32 total = starts[slices] < M1_CUT_CHUNK ? starts[slices] : M1_CUT_CHUNK;
    for (m1_u32 at = threadIdx.x; at < total; at += blockDim.x) {
      m1_u32 s = 0u;
      while (s + 1u < slices && starts[s + 1u] <= at) ++s;
      cut.chunk[at] = get64(s, 0u, 1u + at - starts[s]);
    }
    done();
  }

  // The tie gaps: ties + 1 u64 gap weights, then ties u32 run counts.
  __device__ __forceinline__ void gaps(M1CutShared& cut, m1_u32 ties) {
    m1_u8* out = buffer(slice);
    const m1_u32 counts = (M1_CUT_TIES + 1u) * 8u;
    for (m1_u32 g = threadIdx.x; g <= ties; g += blockDim.x) {
      put64(out, g, cut.gap[g]);
      if (g < ties) put32(out + counts, g, cut.tie_count[g]);
    }
    publish();
    for (m1_u32 g = threadIdx.x; g <= ties; g += blockDim.x) {
      cut.gap[g] = 0ull;
      if (g < ties) cut.tie_count[g] = 0u;
    }
    __syncthreads();
    const m1_u32 per = ties + 1u;
    for (m1_u32 at = threadIdx.x; at < per * slices; at += blockDim.x) {
      const m1_u32 s = at / per, g = at % per;
      const m1_u64 w = get64(s, 0u, g);
      const m1_u32 c = g < ties ? get32(s, counts, g) : 0u;
      if (w != 0ull) atomicAdd(&cut.gap[g], w);
      if (c != 0u) atomicAdd(&cut.tie_count[g], c);
    }
    done();
  }

  __device__ __forceinline__ void sum_u32(m1_u32* values, m1_u32 n) {
    m1_u8* out = buffer(slice);
    for (m1_u32 b = threadIdx.x; b < n; b += blockDim.x) put32(out, b, values[b]);
    publish();
    for (m1_u32 b = threadIdx.x; b < n; b += blockDim.x) values[b] = 0u;
    __syncthreads();
    for (m1_u32 at = threadIdx.x; at < n * slices; at += 4u * blockDim.x) {
      m1_u32 w[4];
#pragma unroll
      for (m1_u32 u = 0u; u < 4u; ++u) {
        const m1_u32 i = at + u * blockDim.x;
        w[u] = i < n * slices ? get32(i / n, 0u, i % n) : 0u;
      }
#pragma unroll
      for (m1_u32 u = 0u; u < 4u; ++u)
        if (w[u] != 0u) atomicAdd(&values[(at + u * blockDim.x) % n], w[u]);
    }
    done();
  }

  // A value every block holds a share of, merged by `combine` in block order
  // (moved as 32-bit words).
  template <class T, class F>
  __device__ __forceinline__ T merge(T mine, F combine) {
    static_assert(sizeof(T) % 4u == 0u, "a share is whole words");
    constexpr m1_u32 WORDS = sizeof(T) / 4u;
    __shared__ T shares[M1_LANE_MAX_SLICES];
    __shared__ T merged;
    m1_u8* out = buffer(slice);
    if (threadIdx.x < WORDS) put32(out, threadIdx.x, reinterpret_cast<const m1_u32*>(&mine)[threadIdx.x]);
    publish();
    for (m1_u32 at = threadIdx.x; at < slices * WORDS; at += blockDim.x)
      reinterpret_cast<m1_u32*>(shares)[at] = get32(at / WORDS, 0u, at % WORDS);
    __syncthreads();
    if (threadIdx.x == 0u) {
      T value = shares[0];
      for (m1_u32 s = 1u; s < slices; ++s) value = combine(value, shares[s]);
      merged = value;
    }
    done();
    return merged;
  }
};

// The canonical width-32 tree over `count` values (identity past the end),
// every block alike: `canonical_reduce`. `work` holds two levels of at most
// 1024 values (a first level of up to 32768).
template <class F>
__device__ __forceinline__ float m1_lane_tree(
    const float* values, m1_u32 count, float identity, float* work, F combine) {
  __shared__ float tree_total;
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 warp = threadIdx.x >> 5u;
  const m1_u32 warps = blockDim.x >> 5u;
  if (count == 0u) return identity;
  const float* level = values;
  m1_u32 n = count;
  m1_u32 which = 0u;
  while (n > 1u) {
    const m1_u32 chunks = (n + 31u) >> 5;
    float* out = work + which * 1024u;
    for (m1_u32 c = warp; c < chunks; c += warps) {
      const m1_u32 at = c * 32u + lane;
      float v = at < n ? (level == values ? __uint_as_float(M1_LANE_GET32(reinterpret_cast<m1_u32*>(const_cast<float*>(level)) + at)) : level[at]) : identity;
      for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
        const float other = m1_chunk_down(v, offset);
        if (lane < offset) v = combine(v, other);
      }
      if (lane == 0u) out[c] = v;
    }
    __syncthreads();
    level = out;
    which ^= 1u;
    n = chunks;
  }
  if (threadIdx.x == 0u) tree_total = level[0];
  __syncthreads();
  const float total = tree_total;
  __syncthreads();
  return total;
}

// The sampler of each of a lane's rows (`rows` of `len`), on the lane's
// blocks: `logits` f32, `threshold` the top-p (f32) or top-k (i32) a row or
// one for all, `state` the Gumbel key, `probs` and `keep` the planes the
// probabilities and the cut land in, `token` the i32 out. RANK picks the
// top-k cut.
template <bool RANK>
__device__ void ptir_lane_sample(
    M1Lane& g,
    const m1_u8* logits,
    const m1_u8* threshold,
    const M1ValueDesc threshold_desc,
    const m1_u8* state,
    const M1ValueDesc state_desc,
    m1_u8* probs,
    m1_u8* keep,
    m1_u8* token,
    m1_u32 rows,
    m1_u32 len) {
  __shared__ M1CutShared cut;
  __shared__ float tree_work[2048];
  __shared__ float block_max[32];
  __shared__ M1ArgmaxCandidate block_best[32];
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 warp = threadIdx.x >> 5u;
  const m1_u32 warps = blockDim.x >> 5u;
  const float* x = reinterpret_cast<const float*>(logits);
  float* p = reinterpret_cast<float*>(probs);
  const m1_u32 key = m1_load_u(state, 0u, state_desc.dtype);
  const m1_u32 counter = state_desc.len > 1u ? m1_load_u(state, 1u, state_desc.dtype) : 0u;
  const m1_u64 seed = ptir_rng_keyed_seed(key, counter);
  g.row(len);
  for (m1_u32 row = 0u; row < rows; ++row) {
    const m1_u32 base = row * len;
    // The row's maximum (order-free: the canonical max is).
    float m = m1_neg_inf();
    for (m1_u32 i = g.lo + threadIdx.x; i < g.hi; i += blockDim.x)
      m = m1_canonical_max(m, x[base + i]);
    for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u)
      m = m1_canonical_max(m, m1_chunk_down(m, offset));
    if (lane == 0u) block_max[warp] = m;
    __syncthreads();
    if (threadIdx.x == 0u) {
      float t = m1_neg_inf();
      for (m1_u32 w = 0u; w < warps; ++w) t = m1_canonical_max(t, block_max[w]);
      block_max[0] = t;
    }
    __syncthreads();
    const float maximum = g.merge(block_max[0], [](float a, float b) { return m1_canonical_max(a, b); });
    // The exponentials, and their width-32 chunk sums (the tree's first
    // level) into the keep plane, which the cut has not reached yet.
    float* chunk_sums = reinterpret_cast<float*>(keep + ((base + 3u) & ~3u));
    const m1_u32 first_chunk = g.lo >> 5;
    const m1_u32 end_chunk = (g.hi + 31u) >> 5;
    for (m1_u32 c = first_chunk + warp; c < end_chunk; c += warps) {
      const m1_u32 i = c * 32u + lane;
      float e = 0.0f;
      if (i < len) {
        e = expf(x[base + i] - maximum);
        p[base + i] = e;
      }
      for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
        const float other = m1_chunk_down(e, offset);
        if (lane < offset) e = e + other;
      }
      if (lane == 0u) M1_LANE_PUT32(reinterpret_cast<m1_u32*>(chunk_sums) + c, __float_as_uint(e));
    }
    g.barrier();
    const float sum = m1_lane_tree(chunk_sums, (len + 31u) >> 5, 0.0f, tree_work,
                                   [](float a, float b) { return a + b; });
    // Every block has read the chunk sums before any overwrites them (the
    // cut writes the keep plane).
    g.barrier();
    for (m1_u32 i = g.lo + threadIdx.x; i < g.hi; i += blockDim.x) p[base + i] = p[base + i] / sum;
    g.barrier();
    // The cut.
    if constexpr (RANK) {
      m1_u32 k;
      if (threshold_desc.dtype == 1u) {
        const int signed_k = m1_load_i(threshold, m1_pick(threshold_desc.len, row), threshold_desc.dtype);
        k = signed_k <= 0 ? 0u : (m1_u32)signed_k;
      } else {
        k = m1_load_u(threshold, m1_pick(threshold_desc.len, row), threshold_desc.dtype);
      }
      if (k > len) k = len;
      if (k == 0u) {
        for (m1_u32 i = g.lo + threadIdx.x; i < g.hi; i += blockDim.x) m1_store_b(keep, base + i, false);
        __syncthreads();
      } else {
        m1_rank_row(probs, base, len, 0u, k, keep, g);
      }
    } else {
      const float cutoff = m1_load_f(threshold, m1_pick(threshold_desc.len, row), threshold_desc.dtype);
      m1_cut_row(probs, base, len, 0u, cutoff, keep, cut, g);
    }
    // The cut logits, Gumbel noise, the argmax.
    M1ArgmaxCandidate best{m1_neg_inf(), 0u, 0u, 0u};
    for (m1_u32 i = g.lo + threadIdx.x; i < g.hi; i += blockDim.x) {
      const float masked = m1_load_b(keep, base + i, 3u) ? x[base + i] : m1_neg_inf();
      const float uniform = ptir_rng_hash_uniform(seed, base + i);
      const float value = masked + (-logf(-logf(uniform)));
      const M1ArgmaxCandidate next{value, i, m1_isnan(value) ? 0u : 1u, 0u};
      best = m1_argmax_combine(best, next);
    }
    best = m1_argmax_warp_reduce(best, lane);
    if (lane == 0u) block_best[warp] = best;
    __syncthreads();
    if (threadIdx.x == 0u) {
      M1ArgmaxCandidate t{m1_neg_inf(), 0u, 0u, 0u};
      for (m1_u32 w = 0u; w < warps; ++w) t = m1_argmax_combine(t, block_best[w]);
      block_best[0] = t;
    }
    __syncthreads();
    const M1ArgmaxCandidate chosen =
        g.merge(block_best[0], [](M1ArgmaxCandidate a, M1ArgmaxCandidate b) { return m1_argmax_combine(a, b); });
    if (g.slice == 0u && threadIdx.x == 0u) m1_store_i(token, row, (int)chosen.index);
    __syncthreads();
  }
}

// A lone top-p or top-k cut (`PivotThreshold`) of each row on the lane's
// blocks: the verdicts land in `output` as the one-block cut lands them.
template <bool RANK>
__device__ void ptir_lane_cut(
    M1Lane& g,
    const m1_u8* input,
    const M1ValueDesc input_desc,
    const m1_u8* threshold,
    const M1ValueDesc threshold_desc,
    m1_u8* output) {
  __shared__ M1CutShared cut;
  const m1_u32 len = input_desc.last;
  const m1_u32 rows = input_desc.rows;
  const m1_u32 dtype = input_desc.dtype;
  if (len == 0u) return;
  g.row(len);
  for (m1_u32 row = 0u; row < rows; ++row) {
    const m1_u32 base = row * len;
    if constexpr (RANK) {
      m1_u32 k;
      if (threshold_desc.dtype == 1u) {
        const int signed_k = m1_load_i(threshold, m1_pick(threshold_desc.len, row), threshold_desc.dtype);
        k = signed_k <= 0 ? 0u : (m1_u32)signed_k;
      } else {
        k = m1_load_u(threshold, m1_pick(threshold_desc.len, row), threshold_desc.dtype);
      }
      if (k > len) k = len;
      if (k == 0u) {
        for (m1_u32 i = g.lo + threadIdx.x; i < g.hi; i += blockDim.x) m1_store_b(output, base + i, false);
        __syncthreads();
      } else {
        m1_rank_row(input, base, len, dtype, k, output, g);
      }
    } else {
      const float cutoff = m1_load_f(threshold, m1_pick(threshold_desc.len, row), threshold_desc.dtype);
      m1_cut_row(input, base, len, dtype, cutoff, output, cut, g);
    }
  }
}
