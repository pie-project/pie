
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
// The arena, per block (`temporary_stride` apart): a 64-byte line (block 0's
// holds the lane's barrier words, one a block; zeroed before the launch),
// two exchange buffers, used by alternate merges (a block is at most one
// merge ahead of the slowest), and two buffers of gathered elements.

#define M1_LANE_HALF 24576u
#define M1_LANE_HEAD 64u
#define M1_LANE_MAX_SLICES 64u
// The blocks' arenas a lane has (the launcher's LANE_SAMPLER_BLOCKS), and
// the gathered elements each holds (two buffers of them, after the halves).
#define M1_LANE_ARENAS 16u
#define M1_FAST_GATHER 512u

// ---------------------------------------------------------- the fast cut --
//
// A monotone row's top-p cut in a few passes instead of a pass a radix digit:
// one pass counts the live elements over M1_FAST_BUCKETS buckets of the row's
// key range (value-ordered); from the counts and each bucket's value bounds
// the blocks bound the running mass at every bucket edge and plan each
// bucket: whole (it sits inside one binade of the running sum, below the
// cutoff: a second pass sums its elements' units at that binade, an add that
// is order-free), taken (its elements are gathered and walked one by one),
// or past (the walk will have stopped before it). The walk then goes bucket
// by bucket in every block alike: a whole bucket adds its units (if the
// running sum is in the planned binade, the bucket holds no element half a
// unit off a whole, and the sum stays in the binade below the cutoff), a run
// of taken ones is sorted and walked 32 elements a step, an element's units
// at a time (`m1_fast_walk`). A bucket the plan misjudged ends the round; the
// next plans again from its edge with the sum the walk reached, and after
// M1_FAST_ROUNDS the cut's own passes take over from there.

#define M1_FAST_BUCKETS 2048u
#define M1_FAST_ROUNDS 4u
#define M1_FAST_EMPTY ((short)32767)
#define M1_FAST_TAKE ((short)32766)
#define M1_FAST_PAST ((short)32765)
#define M1_FAST_LATE ((short)32764)
// Taken past what one sort holds: gathered with the rest, sorted and walked
// when the walk reaches it.
#define M1_FAST_DEFER ((short)32763)
// A whole bucket's flag (a byte a bucket, four to a word; the first pass
// keeps its low-bit mask there): an element half a unit off a whole.
#define M1_FAST_TIE 1u

struct M1Fast {
  // A whole bucket's units (during the plan: the low bound on the mass
  // before it, as f32 bits; while a taken bucket's elements are placed: the
  // next slot of its run), and right after them (merged with them) the
  // buckets' flags.
  m1_u32 weight[M1_FAST_BUCKETS];
  m1_u32 flags[M1_FAST_BUCKETS / 4u];
  m1_u32 count[M1_FAST_BUCKETS];
  // A bucket's plan (during the first pass: how many of its sampled
  // elements its sum holds, two buckets a word).
  short plan[M1_FAST_BUCKETS];
  m1_u32 starts[M1_LANE_ARENAS + 1u];
  m1_u32 mains[M1_LANE_ARENAS + 1u];
  m1_u32 late;
  m1_u32 taken;
  m1_u32 overflow;
  m1_u32 outcome;
  m1_u32 ended;
  m1_u32 walked;
  m1_u64 pivot;
  float mass;
};

// Where an element's lowest set mantissa bit lies (the hidden bit counted),
// a bit of eight: bit k for lowest bits 3k..3k+2 (bit 7 from 21 up). An
// element is exactly half a unit off a whole at 2^eu iff its lowest set bit
// is eu - ey - 1, so a bucket whose elements' bits miss that one holds no
// tie there. Bits 0 and 1 (lowest bits below 6, most elements') are taken
// as set: a whole bucket that close to its binade's units is taken anyway.
__device__ __forceinline__ m1_u32 m1_fast_low_bit(float y) {
  const m1_u32 bits = __float_as_uint(y) & 0x7FFFFFFFu;
  const m1_u32 mantissa = (bits >> 23) == 0u ? (bits & 0x7FFFFFu) : ((bits & 0x7FFFFFu) | 0x800000u);
  if (mantissa == 0u) return 0u;
  const m1_u32 low = (m1_u32)(__ffs((int)mantissa) - 1);
  return low < 6u ? 0u : 1u << (low >= 21u ? 7u : low / 3u);
}

// Whether a bucket of values from `bottom` to `top` may hold an element
// half a unit off a whole at 2^eu, by its low-bit mask.
__device__ __forceinline__ bool m1_fast_may_tie(m1_u32 mask, float bottom, float top, int eu) {
  mask |= 3u;
  const m1_u32 hi = __float_as_uint(top) >> 23;
  const m1_u32 lo = __float_as_uint(bottom) >> 23;
  if (hi - lo > 1u) return true;
  for (m1_u32 biased = lo; biased <= hi; ++biased) {
    const int ey = biased == 0u ? -149 : (int)biased - 150;
    const int shift = eu - ey;
    if (shift < 1) return true;
    if (shift > 24) continue;
    const m1_u32 low = (m1_u32)(shift - 1);
    if ((mask >> (low >= 21u ? 7u : low / 3u)) & 1u) return true;
  }
  return false;
}

__device__ __forceinline__ m1_u32 m1_fast_flags(const M1Fast& f, m1_u32 b) {
  return (f.flags[b >> 2] >> ((b & 3u) * 8u)) & 0xFFu;
}

// The block's fast-cut state (one, whoever fills it: the cut's own first
// pass, or the sampler's probability pass).
__device__ __forceinline__ M1Fast& m1_fast_state() {
  __shared__ M1Fast f;
  return f;
}

// The buckets over keys [least, most]: each 2^stretch keys wide.
__device__ __forceinline__ int m1_fast_stretch(m1_u32 least, m1_u32 most) {
  const m1_u32 span = most - least;
  int stretch = 0;
  while (stretch < 32 && (span >> stretch) >= M1_FAST_BUCKETS) ++stretch;
  return stretch;
}

// A key's bucket, held to the range (a key past either edge lands in the
// bucket at that edge, so the buckets keep the keys' order).
__device__ __forceinline__ m1_u32 m1_fast_bucket(m1_u32 key, m1_u32 least, int stretch) {
  if (key < least) return 0u;
  const m1_u32 b = (key - least) >> stretch;
  return b < M1_FAST_BUCKETS ? b : M1_FAST_BUCKETS - 1u;
}

// The first pass's state cleared, an element noted in it (its bucket's
// count, low-bit mask, and for indices a multiple of eight its f32 sum and
// how many it holds, two buckets a word in `plan`).
__device__ __forceinline__ void m1_fast_clear(M1Fast& f) {
  for (m1_u32 b = threadIdx.x; b < M1_FAST_BUCKETS; b += blockDim.x) {
    f.count[b] = 0u;
    f.weight[b] = 0u;
    f.plan[b] = 0;
    if (b < M1_FAST_BUCKETS / 4u) f.flags[b] = 0u;
  }
}

__device__ __forceinline__ void m1_fast_note(M1Fast& f, m1_u32 b, m1_u32 i, float y) {
  atomicAdd(&f.count[b], 1u);
  const m1_u32 low = m1_fast_low_bit(y);
  if (low != 0u) atomicOr(&f.flags[b >> 2], low << ((b & 3u) * 8u));
  if ((i & 7u) == 0u) {
    M1_LANE_FADD(reinterpret_cast<float*>(&f.weight[b]), y);
    atomicAdd(reinterpret_cast<m1_u32*>(f.plan) + (b >> 1), 1u << ((b & 1u) * 16u));
  }
}

// The least whole number of units 2^eu at or past the cutoff, held to the
// binade's top (2^24): ceil(cutoff / 2^eu) from the cutoff's own bits.
__device__ __forceinline__ m1_u32 m1_fast_limit(float cutoff, int eu) {
  const m1_u64 top = 1ull << 24;
  m1_u64 limit = top;
  const m1_u32 bits = __float_as_uint(cutoff);
  const m1_u32 cb = bits >> 23;
  if (cb < 255u) {
    const m1_u64 cm = cb == 0u ? (m1_u64)(bits & 0x7FFFFFu)
                               : (m1_u64)((bits & 0x7FFFFFu) | 0x800000u);
    const int ec = cb == 0u ? -149 : (int)cb - 150;
    const int up = ec - eu;
    if (up >= 0) {
      if (up < 26) limit = cm << up;
    } else if (-up < 40) {
      const int down = -up;
      limit = (cm + (1ull << down) - 1ull) >> down;
    } else {
      limit = 1ull;
    }
    if (limit > top) limit = top;
  }
  return (m1_u32)limit;
}

// The binade 2^eu of a nonnegative running sum and the units it holds.
__device__ __forceinline__ int m1_fast_binade(float mass, m1_u32* units) {
  const m1_u32 bits = __float_as_uint(mass);
  const m1_u32 biased = bits >> 23;
  *units = biased == 0u ? (bits & 0x7FFFFFu) : ((bits & 0x7FFFFFu) | 0x800000u);
  return biased == 0u ? -149 : (int)biased - 150;
}

// What `y` adds to a sum of units 2^eu: its units (saturated at 2^25 for a
// value at or past 2^(eu+23), which leaves any sum's binade), or `tie` for
// one exactly half a unit off a whole (its add depends on the sum's parity).
__device__ __forceinline__ m1_u32 m1_fast_units(float y, int eu, bool* tie) {
  *tie = false;
  if (eu >= -126 && eu <= 126) {
    // y / 2^eu is exact in f32 (a power-of-two scale) wherever it is near a
    // half: its whole and its rest are then exact too.
    const float z = fabsf(y) * __uint_as_float((m1_u32)(127 - eu) << 23);
    if (!(z < 8388608.0f)) return 1u << 25;
    const float whole = floorf(z);
    const float rest = z - whole;
    *tie = rest == 0.5f;
    return (m1_u32)whole + (rest > 0.5f ? 1u : 0u);
  }
  const m1_u32 bits = __float_as_uint(y) & 0x7FFFFFFFu;
  if (bits == 0u) return 0u;
  const m1_u32 biased = bits >> 23;
  const int ey = biased == 0u ? -149 : (int)biased - 150;
  if (ey >= eu) return 1u << 25;
  m1_u32 whole;
  const m1_u32 rest = m1_cut_units(y, eu, &whole);
  *tie = rest == 1u;
  return whole + (rest == 2u ? 1u : 0u);
}

// Exclusive prefix sums, in place, of `n` values in LDS, every block alike:
// each thread sums a run of consecutive values, the runs scan across their
// 32-lane chunk, the chunks' totals in order.
template <class T>
__device__ void m1_fast_scan(T* values, m1_u32 n) {
  __shared__ T chunk_total[32];
  const m1_u32 per = (n + blockDim.x - 1u) / blockDim.x;
  const m1_u32 from = threadIdx.x * per;
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 chunk = threadIdx.x >> 5;
  T run = T(0);
  for (m1_u32 k = 0u; k < per; ++k)
    if (from + k < n) run += values[from + k];
  T inclusive = run;
  for (m1_u32 offset = 1u; offset < 32u; offset <<= 1) {
    const T other = m1_chunk_up(inclusive, offset);
    if (lane >= offset) inclusive += other;
  }
  if (lane == 31u) chunk_total[chunk] = inclusive;
  __syncthreads();
  if (threadIdx.x == 0u) {
    T total = T(0);
    for (m1_u32 c = 0u; c < (blockDim.x >> 5); ++c) {
      const T t = chunk_total[c];
      chunk_total[c] = total;
      total += t;
    }
  }
  __syncthreads();
  T prefix = chunk_total[chunk] + inclusive - run;
  for (m1_u32 k = 0u; k < per; ++k) {
    if (from + k < n) {
      const T v = values[from + k];
      values[from + k] = prefix;
      prefix += v;
    }
  }
  __syncthreads();
}

// The f32 holding `units` (below 2^24) of 2^eu: exact, as the binade's bits.
__device__ __forceinline__ float m1_fast_mass(m1_u32 units, int eu) {
  return __uint_as_float(((m1_u32)(eu + 149) << 23) + units);
}

// The walk of `n` sorted elements (composites, descending value) from the
// running sum `mass`, on one 32-lane chunk: a step takes the next 128, four
// consecutive ones a lane, each its units at the sum's binade, and adds the
// run before the first that leaves the binade, reaches the cutoff's units
// or is a tie; that one is added in f32 as the reference adds it. True (and
// `pivot` the last kept) at the first add whose sum reaches the cutoff.
__device__ __forceinline__ bool m1_fast_walk(
    const m1_u64* items, m1_u32 n, float cutoff, float& mass, m1_u64& pivot) {
  const m1_u32 lane = threadIdx.x & 31u;
  m1_u32 at = 0u;
  while (at < n) {
    m1_u32 units;
    const int eu = m1_fast_binade(mass, &units);
    const m1_u32 limit = m1_fast_limit(cutoff, eu);
    float y[4];
    m1_u32 r[4];
    bool special[4];
    m1_u32 own = 0u;
#pragma unroll
    for (m1_u32 j = 0u; j < 4u; ++j) {
      const m1_u32 i = at + lane * 4u + j;
      y[j] = 0.0f;
      r[j] = 0u;
      special[j] = false;
      if (i < n) {
        y[j] = m1_cut_value((m1_u32)(items[i] >> 32));
        bool tie;
        r[j] = m1_fast_units(y[j], eu, &tie);
        special[j] = tie || r[j] >= (1u << 25);
        if (special[j]) r[j] = 0u;
      }
      own += r[j];
    }
    const m1_u32 sum = m1_chunk_scan(own);
    // This lane's first crossing element (4: none), and the units before it.
    m1_u32 before = sum - own;
    m1_u32 first = 4u;
#pragma unroll
    for (m1_u32 j = 0u; j < 4u; ++j) {
      const bool live = at + lane * 4u + j < n;
      if (first == 4u && live && (special[j] || units + before + r[j] >= limit)) first = j;
      if (first == 4u) before += r[j];
    }
    const m1_u32 ballot = m1_chunk_ballot(first != 4u);
    if (ballot == 0u) {
      units += m1_lead_from(sum, 31u);
      mass = m1_fast_mass(units, eu);
      at += 128u;
      continue;
    }
    const m1_u32 k = (m1_u32)(__ffs((int)ballot) - 1);
    const m1_u32 j = m1_lead_from(first, k);
    const m1_u32 prior = m1_lead_from(before, k);
    float crossing = y[0];
#pragma unroll
    for (m1_u32 q = 1u; q < 4u; ++q)
      if (q == j) crossing = y[q];
    crossing = m1_lead_from(crossing, k);
    mass = m1_fast_mass(units + prior, eu);
    mass = mass + crossing;
    at += k * 4u + j + 1u;
    if (!(mass < cutoff)) {
      pivot = items[at - 1u];
      return true;
    }
  }
  return false;
}

#define M1_FAST_MISS 0u
#define M1_FAST_STOPPED 1u
#define M1_FAST_ALL 2u
#define M1_FAST_FAILED 3u

// The walk over the buckets from `first` (f.walked of the sorted taken
// elements `items` behind it, the sum f.mass), on one 32-lane chunk: a step
// looks at 32 buckets, adds the whole ones' units up to the first that is
// not a whole bucket of the sum's binade (or whose add would leave it or
// reach the cutoff's units), and walks a run of taken buckets' elements
// where one starts. Lands the outcome, the bucket it ended at (a miss: the
// bucket the plan misjudged, the sum at its edge), the sum, the elements
// walked and the pivot in `f`.
__device__ __forceinline__ void m1_fast_route(
    M1Fast& f, const m1_u64* items, m1_u32 gathered, m1_u32 first, float cutoff) {
  const m1_u32 lane = threadIdx.x & 31u;
  float mass = f.mass;
  m1_u64 pivot = 0ull;
  m1_u32 b = first;
  m1_u32 walked = f.walked;
  m1_u32 outcome = M1_FAST_ALL;
  while (b < M1_FAST_BUCKETS) {
    // A step: 128 buckets, four consecutive ones a lane.
    m1_u32 units;
    const int eu = m1_fast_binade(mass, &units);
    const m1_u32 limit = m1_fast_limit(cutoff, eu);
    m1_u64 w[4];
    m1_u32 halt = 4u;
    m1_u64 own = 0ull;
#pragma unroll
    for (m1_u32 j = 0u; j < 4u; ++j) {
      const m1_u32 at = b + lane * 4u + j;
      const short plan = at < M1_FAST_BUCKETS ? f.plan[at] : M1_FAST_EMPTY;
      const bool whole = plan != M1_FAST_EMPTY && plan != M1_FAST_TAKE && plan != M1_FAST_PAST;
      const bool good = whole && (int)plan == eu && m1_fast_flags(f, at) == 0u;
      if (halt == 4u && plan != M1_FAST_EMPTY && !good) halt = j;
      w[j] = halt == 4u && good ? (m1_u64)f.weight[at] : 0ull;
      own += w[j];
    }
    const m1_u64 sum = m1_chunk_scan(own);
    const m1_u32 halts = m1_chunk_ballot(halt != 4u);
    const m1_u32 stop = halts == 0u ? 32u : (m1_u32)(__ffs((int)halts) - 1);
    // The first whole bucket whose add reaches the limit, and the units
    // before it.
    m1_u64 run = sum - own;
    m1_u32 crossing = 4u;
    m1_u64 prior = 0ull;
#pragma unroll
    for (m1_u32 j = 0u; j < 4u; ++j) {
      if (crossing == 4u && w[j] != 0ull) {
        if ((m1_u64)units + run + w[j] >= limit) {
          crossing = j;
          prior = run;
        }
        run += w[j];
      }
    }
    const m1_u32 over = m1_chunk_ballot(lane <= stop && crossing != 4u);
    if (over != 0u) {
      const m1_u32 k = (m1_u32)(__ffs((int)over) - 1);
      const m1_u32 j = m1_lead_from(crossing, k);
      mass = m1_fast_mass(units + (m1_u32)m1_lead_from(prior, k), eu);
      b += k * 4u + j;
      outcome = M1_FAST_MISS;
      break;
    }
    if (stop == 32u) {
      mass = m1_fast_mass(units + (m1_u32)m1_lead_from(sum, 31u), eu);
      b += 128u;
      continue;
    }
    mass = m1_fast_mass(units + (m1_u32)m1_lead_from(sum, stop), eu);
    b += stop * 4u + m1_lead_from(halt, stop);
    if (f.plan[b] != M1_FAST_TAKE) {
      outcome = M1_FAST_MISS;
      break;
    }
    // A run of taken buckets (and empty ones) from b: its elements are the
    // next of the sorted ones.
    m1_u32 elements = 0u;
    m1_u32 end = b;
    while (end < M1_FAST_BUCKETS) {
      m1_u32 out = 4u;
      m1_u32 c = 0u;
#pragma unroll
      for (m1_u32 j = 0u; j < 4u; ++j) {
        const m1_u32 q = end + lane * 4u + j;
        const short kind = q < M1_FAST_BUCKETS ? f.plan[q] : M1_FAST_PAST;
        if (out == 4u && kind != M1_FAST_TAKE && kind != M1_FAST_EMPTY) out = j;
        if (out == 4u && kind == M1_FAST_TAKE) c += f.count[q];
      }
      const m1_u32 outs = m1_chunk_ballot(out != 4u);
      const m1_u32 counted = m1_chunk_scan(c);
      if (outs == 0u) {
        elements += m1_lead_from(counted, 31u);
        end += 128u;
        continue;
      }
      const m1_u32 k = (m1_u32)(__ffs((int)outs) - 1);
      elements += m1_lead_from(counted, k);
      end += k * 4u + m1_lead_from(out, k);
      break;
    }
    if (end > M1_FAST_BUCKETS) end = M1_FAST_BUCKETS;
    if (walked + elements > gathered) {
      outcome = M1_FAST_FAILED;
      break;
    }
    if (m1_fast_walk(items + walked, elements, cutoff, mass, pivot)) {
      outcome = M1_FAST_STOPPED;
      break;
    }
    walked += elements;
    b = end;
  }
  if (b > M1_FAST_BUCKETS) b = M1_FAST_BUCKETS;
  if (outcome == M1_FAST_ALL && walked != gathered) outcome = M1_FAST_FAILED;
  if (lane == 0u) {
    f.mass = mass;
    f.pivot = pivot;
    f.outcome = outcome;
    f.ended = b;
    f.walked = walked;
  }
}

struct M1Lane {
  m1_u32 lo;
  m1_u32 hi;
  m1_u32 slice;
  m1_u32 slices;
  m1_u8* arena;
  m1_u32 stride;
  m1_u32 parity;
  m1_u32 arrivals;
  // Whether the fast cut's first pass is done already (the sampler's
  // probability pass took it).
  bool counted;

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
    // Each block stores how many barriers it reached in its own word of the
    // lane's first line (no read-modify-write the blocks would queue on),
    // and a lane of the first chunk watches each block's word.
    M1_LANE_RELEASE();
    __syncthreads();
    arrivals += 1u;
    if (threadIdx.x < 32u) {
      m1_u32* words = reinterpret_cast<m1_u32*>(arena);
      if (threadIdx.x == 0u) M1_LANE_PUT32(words + slice, arrivals);
      bool ready = threadIdx.x >= slices;
      for (;;) {
        if (!ready) ready = M1_LANE_LOAD(words + threadIdx.x) >= arrivals;
        if (m1_chunk_ballot(!ready) == 0u) break;
        M1_LANE_PAUSE();
      }
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

  // Every block's `n` u32 words (`mine`, LDS) combined over the lane's
  // blocks into `mine` by `combine(a, b, word)` (integer, so in any order),
  // and each block's `tail` word into `tails[block]` (if any). With four
  // blocks or more each combines a share of the words and puts it, then
  // every block gets every share: two meets, but a block reads 2n words, not
  // n a block.
  template <class F>
  __device__ __forceinline__ void combine_u32(
      m1_u32* mine, m1_u32 n, m1_u32 tail, m1_u32* tails, F combine) {
    m1_u8* out = buffer(slice);
    const bool scatter = slices >= 4u && n != 0u;
    const m1_u32 share = (n + slices - 1u) / slices;
    const m1_u32 tail_at = 4u * n + (scatter ? 4u * share : 0u);
    for (m1_u32 b = threadIdx.x; b < n; b += blockDim.x) put32(out, b, mine[b]);
    if (threadIdx.x == 0u) put32(out + tail_at, 0u, tail);
    publish();
    if (tails != nullptr && threadIdx.x < slices) tails[threadIdx.x] = get32(threadIdx.x, tail_at, 0u);
    const m1_u32 from = scatter ? slice * share : 0u;
    const m1_u32 to = scatter ? (from + share < n ? from + share : n) : n;
    for (m1_u32 b = from + threadIdx.x; b < to; b += blockDim.x) {
      m1_u32 got[M1_LANE_ARENAS];
#pragma unroll
      for (m1_u32 s = 0u; s < M1_LANE_ARENAS; ++s) got[s] = s < slices ? get32(s, 0u, b) : 0u;
      m1_u32 v = got[0];
#pragma unroll
      for (m1_u32 s = 1u; s < M1_LANE_ARENAS; ++s)
        if (s < slices) v = combine(v, got[s], b);
      if (scatter) put32(out + 4u * n, b - from, v);
      else mine[b] = v;
    }
    if (scatter) {
      barrier();
      for (m1_u32 b = threadIdx.x; b < n; b += blockDim.x) {
        const m1_u32 s = b / share;
        mine[b] = get32(s, 4u * n, b - s * share);
      }
    }
    done();
  }

  __device__ __forceinline__ void sum_u32(m1_u32* values, m1_u32 n) {
    combine_u32(values, n, 0u, nullptr, [](m1_u32 a, m1_u32 b, m1_u32) { return a + b; });
  }

  // Where a block's `n`-th gathered element goes: its own arena's gather
  // buffer, then those of the arenas past the lane's blocks (block + slices,
  // + 2 slices, ...), as many as the lane has.
  __device__ __forceinline__ m1_u64* gather_slot(m1_u32 block, m1_u32 n, m1_u32 which) const {
    const m1_u32 home = block + slices * (n / M1_FAST_GATHER);
    return reinterpret_cast<m1_u64*>(
               arena + (m1_u64)home * stride + M1_LANE_HEAD + 2u * M1_LANE_HALF +
               which * M1_FAST_GATHER * 8u) +
           (n % M1_FAST_GATHER);
  }
  __device__ __forceinline__ m1_u32 gather_room(m1_u32 block) const {
    return block >= M1_LANE_ARENAS ? 0u
                                   : M1_FAST_GATHER * ((M1_LANE_ARENAS - 1u - block) / slices + 1u);
  }

  // The fast cut's first pass merged: counts and masks and sampled sums
  // (the sums added in block order, every block alike).
  __device__ __forceinline__ void fast_counted(M1Fast& f) {
  combine_u32(f.weight, M1_FAST_BUCKETS * 2u + M1_FAST_BUCKETS / 4u + M1_FAST_BUCKETS / 2u, 0u,
                nullptr, [](m1_u32 a, m1_u32 b, m1_u32 word) {
                  if (word < M1_FAST_BUCKETS) return __float_as_uint(__uint_as_float(a) + __uint_as_float(b));
                  return word < M1_FAST_BUCKETS + M1_FAST_BUCKETS / 4u ? a | b : a + b;
                });
  }

  // The fast cut's second pass: the whole buckets' units summed (held to 32
  // bits), their flags joined, each block's gathered count (into `starts`,
  // its prefix, held to the block's room) and overflow.
  __device__ __forceinline__ void fast_merge(M1Fast& f, bool units) {
    const m1_u32 tail = (f.taken & 0x7FFFFFFFu) | (f.overflow != 0u ? 0x80000000u : 0u);
    combine_u32(f.weight, units ? M1_FAST_BUCKETS + M1_FAST_BUCKETS / 4u : 0u, tail, f.starts + 1u,
                [](m1_u32 a, m1_u32 b, m1_u32 word) {
                  if (word >= M1_FAST_BUCKETS) return a | b;
                  const m1_u32 sum = a + b;
                  return sum < a ? 0xFFFFFFFFu : sum;
                });
    __syncthreads();
    if (threadIdx.x == 0u) {
      f.starts[0] = 0u;
      for (m1_u32 s = 1u; s <= slices; ++s) {
        const m1_u32 word = f.starts[s];
        if ((word & 0x80000000u) != 0u) f.overflow = 1u;
        const m1_u32 room = gather_room(s - 1u);
        const m1_u32 held = word & 0x7FFFFFFFu;
        f.starts[s] = f.starts[s - 1u] + (held < room ? held : room);
      }
    }
    __syncthreads();
  }

  __device__ __forceinline__ void fast(const m1_u8* input, m1_u32 base, m1_u32 len, m1_u32 dtype,
                       float cutoff, M1CutShared& cut);

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

// The fast cut (see above) of the row at `base`: leaves the cut stopped
// (pivot), all kept, or still running from the cursor and mass it reached.
__device__ __forceinline__ void M1Lane::fast(const m1_u8* input, m1_u32 base, m1_u32 len, m1_u32 dtype,
                             float cutoff, M1CutShared& cut) {
  M1Fast& f = m1_fast_state();
  const m1_u32 least = cut.key_least;
  const m1_u32 most = cut.key_most;
  const bool noted = counted;
  counted = false;
  if (least > most) return;
  const int stretch = m1_fast_stretch(least, most);
  auto bucket_of = [&](m1_u32 key) { return m1_fast_bucket(key, least, stretch); };
  // The edges of a bucket's keys: its largest value and its least.
  auto top_value = [&](m1_u32 b) {
    return (double)m1_cut_value(least + (m1_u32)((m1_u64)b << stretch));
  };
  auto bottom_value = [&](m1_u32 b) {
    const m1_u64 last = (m1_u64)least + (((m1_u64)b + 1ull) << stretch) - 1ull;
    return (double)m1_cut_value(last > most ? most : (m1_u32)last);
  };
  // The first pass (unless the caller's took it): each bucket's count, its
  // elements' low-bit mask (in `flags`, for the first round's plan), and
  // the f32 sum (in `weight`) and count (in `plan`) of its elements at
  // indices a multiple of eight, from which the first round's plan reckons
  // its mass.
  if (!noted) {
    m1_fast_clear(f);
    __syncthreads();
    m1_cut_scan_range(input, base, lo, hi, dtype, [&](m1_u32 i, float y) {
      if (!m1_isnan(y)) m1_fast_note(f, bucket_of(m1_desc_key(y)), i, y);
    });
    __syncthreads();
    fast_counted(f);
  }
  // A bucket's mass, reckoned: its count times its sampled elements' mean
  // (its middle value if none was sampled).
  auto reckoned = [&](m1_u32 b, m1_u32 c) {
    const m1_u32 n = (m1_u32)(unsigned short)f.plan[b];
    return n != 0u ? (double)__uint_as_float(f.weight[b]) * (double)c / (double)n
                   : (double)c * 0.5 * (top_value(b) + bottom_value(b));
  };
  float mass = 0.0f;
  m1_u32 first = 0u;
  bool settled = false;
  m1_u32 gathers = 0u;
  // A round gathers what one sort holds, and at most three quarters of the
  // least gather room of the lane's blocks; a repair, half of that.
  const m1_u32 room = gather_room(slices - 1u) * slices / 4u * 3u;
  const m1_u32 planned = room < M1_CUT_CHUNK ? room : M1_CUT_CHUNK;
  const m1_u32 deferred = room < 2u * M1_CUT_CHUNK ? room : 2u * M1_CUT_CHUNK;
  const m1_u32 mending = planned / 2u;
  double* bound = reinterpret_cast<double*>(cut.chunk);
  m1_u32* takes = reinterpret_cast<m1_u32*>(cut.before);
  m1_u64* mends = reinterpret_cast<m1_u64*>(cut.before);
  const m1_u32 mine = gather_room(slice);
  for (m1_u32 round = 0u; round < M1_FAST_ROUNDS && !settled; ++round) {
    // A later round's gathers reuse buffers the last one's reads may still
    // be in: every block past those first.
    if (round != 0u) barrier();
    // The plan, from the running mass expected at each bucket's edges: the
    // first round's from the buckets' sums, a later one's from bounds by the
    // buckets' edge values (the low bound before each into `weight`, as f32,
    // then the high after it). The f32 walk strays from the exact sum by
    // far less than the margin.
    const bool summed = round == 0u;
    for (m1_u32 b = threadIdx.x; b < M1_FAST_BUCKETS; b += blockDim.x) {
      const m1_u32 c = b >= first ? f.count[b] : 0u;
      bound[b] = c == 0u ? 0.0 : summed ? reckoned(b, c) : (double)c * bottom_value(b);
    }
    __syncthreads();
    m1_fast_scan(bound, M1_FAST_BUCKETS);
    // (A summed round's `bound` then holds the mass through each bucket.)
    for (m1_u32 b = threadIdx.x; b < M1_FAST_BUCKETS; b += blockDim.x) {
      const m1_u32 c = b >= first ? f.count[b] : 0u;
      const double low = ((double)mass + bound[b]) * (1.0 - 1.0 / 4096.0);
      const double own = summed && c != 0u ? reckoned(b, c) : 0.0;
      f.weight[b] = __float_as_uint((float)low);
      bound[b] = summed ? bound[b] + own : (c == 0u ? 0.0 : (double)c * top_value(b));
    }
    __syncthreads();
    if (!summed) m1_fast_scan(bound, M1_FAST_BUCKETS);
    for (m1_u32 b = threadIdx.x; b < M1_FAST_BUCKETS; b += blockDim.x) {
      const m1_u32 c = f.count[b];
      short plan = M1_FAST_EMPTY;
      if (b >= first && c != 0u) {
        const double lower = (double)__uint_as_float(f.weight[b]);
        const double upper =
            ((double)mass + bound[b] + (summed ? 0.0 : (double)c * top_value(b))) *
            (1.0 + 1.0 / 4096.0);
        if (lower >= (double)cutoff || c > M1_CUT_CHUNK) {
          plan = M1_FAST_PAST;
        } else if (!summed && m1_fast_flags(f, b) != 0u) {
          plan = M1_FAST_TAKE;
        } else {
          int el, eh;
          frexp(lower, &el);
          frexp(upper, &eh);
          plan = upper < (double)cutoff && lower >= 1.1754943508222875e-38 && el == eh
                     ? (short)(el - 24)
                     : M1_FAST_TAKE;
          if (summed && plan != M1_FAST_TAKE &&
              m1_fast_may_tie(m1_fast_flags(f, b), (float)bottom_value(b), (float)top_value(b), plan))
            plan = M1_FAST_TAKE;
          // A whole bucket's units must sum in 32 bits: its count times its
          // largest element's units (2^25 saturated) stays below 2^32.
          if (plan != M1_FAST_TAKE) {
            const double most_units = ldexp(top_value(b), -(int)plan);
            const double each = most_units >= 8388608.0 ? 33554432.0 : most_units + 1.0;
            if ((double)c * each >= 4294967296.0) plan = M1_FAST_TAKE;
          }
        }
      }
      f.plan[b] = plan;
      takes[b] = plan == M1_FAST_TAKE ? c : 0u;
    }
    __syncthreads();
    m1_fast_scan(takes, M1_FAST_BUCKETS);
    for (m1_u32 b = threadIdx.x; b < M1_FAST_BUCKETS; b += blockDim.x) {
      f.weight[b] = 0u;
      if (summed && b < M1_FAST_BUCKETS / 4u) f.flags[b] = 0u;
      if (f.plan[b] == M1_FAST_TAKE && takes[b] + f.count[b] > planned)
        f.plan[b] = takes[b] + f.count[b] > deferred ? M1_FAST_PAST : M1_FAST_DEFER;
    }
    if (threadIdx.x == 0u) {
      f.taken = 0u;
      f.overflow = 0u;
    }
    __syncthreads();
    // The second pass: whole buckets' units, taken buckets' elements.
    const m1_u32 which = gathers++ & 1u;
    m1_cut_scan_range(input, base, lo, hi, dtype, [&](m1_u32 i, float y) {
      if (m1_isnan(y)) return;
      const m1_u32 key = m1_desc_key(y);
      const m1_u32 b = bucket_of(key);
      const short plan = f.plan[b];
      if (plan == M1_FAST_EMPTY || plan == M1_FAST_PAST) return;
      if (plan == M1_FAST_TAKE || plan == M1_FAST_DEFER) {
        const m1_u32 n = atomicAdd(&f.taken, 1u);
        if (n < mine) M1_LANE_PUT64(gather_slot(slice, n, which), m1_cut_composite(key, i));
        else f.overflow = 1u;
        return;
      }
      bool tie;
      const m1_u32 units = m1_fast_units(y, plan, &tie);
      if (tie) {
        atomicOr(&f.flags[b >> 2], M1_FAST_TIE << ((b & 3u) * 8u));
      } else if (units != 0u) {
        atomicAdd(&f.weight[b], units);
      }
    });
    __syncthreads();
    fast_merge(f, true);
    if (f.overflow != 0u) break;
    // Whole buckets the second pass flagged (a tie the plan did not foresee,
    // whose add depends on the order before it) are taken too, in a pass of
    // their own, if the sort holds them.
    if (threadIdx.x == 0u) {
      f.late = 0u;
      f.walked = 0u;
    }
    for (m1_u32 s = threadIdx.x; s <= slices; s += blockDim.x) f.mains[s] = f.starts[s];
    __syncthreads();
    for (m1_u32 b = threadIdx.x; b < M1_FAST_BUCKETS; b += blockDim.x) {
      const short plan = f.plan[b];
      if (plan != M1_FAST_EMPTY && plan != M1_FAST_TAKE && plan != M1_FAST_PAST &&
          plan != M1_FAST_DEFER && m1_fast_flags(f, b) != 0u)
        atomicAdd(&f.late, f.count[b]);
      if (plan == M1_FAST_TAKE) atomicAdd(&f.walked, f.count[b]);
    }
    __syncthreads();
    // The main gather holds the taken and the deferred; the sort, the taken.
    const m1_u32 main_all = f.mains[slices];
    const m1_u32 main_gathered = f.walked;
    const bool late = f.late != 0u && main_gathered + f.late <= M1_CUT_CHUNK &&
                      f.late <= gather_room(slices - 1u) * slices / 4u * 3u;
    m1_u32 late_gathered = 0u;
    const m1_u32 late_which = gathers & 1u;
    if (late) {
      ++gathers;
      for (m1_u32 b = threadIdx.x; b < M1_FAST_BUCKETS; b += blockDim.x) {
        const short plan = f.plan[b];
        const bool flagged = plan != M1_FAST_EMPTY && plan != M1_FAST_TAKE &&
                             plan != M1_FAST_PAST && plan != M1_FAST_DEFER &&
                             m1_fast_flags(f, b) != 0u;
        if (flagged) f.plan[b] = M1_FAST_LATE;
      }
      if (threadIdx.x == 0u) {
        f.taken = 0u;
        f.overflow = 0u;
      }
      __syncthreads();
      m1_cut_scan_range(input, base, lo, hi, dtype, [&](m1_u32 i, float y) {
        if (m1_isnan(y)) return;
        const m1_u32 key = m1_desc_key(y);
        const bool take = f.plan[bucket_of(key)] == M1_FAST_LATE;
        if (!take) return;
        const m1_u32 n = atomicAdd(&f.taken, 1u);
        if (n < mine) M1_LANE_PUT64(gather_slot(slice, n, late_which), m1_cut_composite(key, i));
        else f.overflow = 1u;
      });
      __syncthreads();
      fast_merge(f, false);
      if (f.overflow != 0u) break;
      late_gathered = f.starts[slices];
      for (m1_u32 b = threadIdx.x; b < M1_FAST_BUCKETS; b += blockDim.x)
        if (f.plan[b] == M1_FAST_LATE) f.plan[b] = M1_FAST_TAKE;
      __syncthreads();
    }
    // Every taken element in (the main gather's, not its deferred ones, and
    // the late pass's), sorted.
    const m1_u32 gathered = main_gathered + late_gathered;
    if (gathered > M1_CUT_CHUNK) break;
    if (threadIdx.x == 0u) f.taken = 0u;
    __syncthreads();
    for (m1_u32 at = threadIdx.x; at < main_all + late_gathered; at += blockDim.x) {
      const bool from_late = at >= main_all;
      const m1_u32 index = from_late ? at - main_all : at;
      const m1_u32* starts = from_late ? f.starts : f.mains;
      m1_u32 s = 0u;
      while (s + 1u < slices && starts[s + 1u] <= index) ++s;
      const m1_u64 composite =
          M1_LANE_GET64(gather_slot(s, index - starts[s], from_late ? late_which : which));
      if (!from_late && f.plan[bucket_of((m1_u32)(composite >> 32))] != M1_FAST_TAKE) continue;
      const m1_u32 slot = atomicAdd(&f.taken, 1u);
      if (slot < M1_CUT_CHUNK) cut.chunk[slot] = composite;
    }
    __syncthreads();
    if (f.taken != gathered) {
      first = 0u;
      mass = 0.0f;
      break;
    }
    m1_cut_sort(cut.chunk, gathered);
    if (threadIdx.x == 0u) {
      f.mass = mass;
      f.walked = 0u;
    }
    __syncthreads();
    // The walk, on one chunk (every block's alike), and repairs: a bucket
    // the plan misjudged, and the past ones after it (as many as half a
    // round holds), gathered and walked in f32 as the reference walks them;
    // or a run of deferred buckets, from the main gather.
    m1_u32 from = first;
    m1_u32 repairs = 0u;
    for (;;) {
      if (threadIdx.x < 32u) m1_fast_route(f, cut.chunk, gathered, from, cutoff);
      __syncthreads();
      const m1_u32 outcome = f.outcome;
      if (outcome == M1_FAST_STOPPED || outcome == M1_FAST_ALL) {
        mass = f.mass;
        if (threadIdx.x == 0u) {
          cut.stage = outcome == M1_FAST_STOPPED ? M1_CUT_STOPPED : M1_CUT_ALL;
          cut.pivot = f.pivot;
          cut.nan_kept = mass < cutoff ? 1u : 0u;
          cut.mass = mass;
        }
        settled = true;
        break;
      }
      if (outcome == M1_FAST_FAILED) {
        first = 0u;
        mass = 0.0f;
        round = M1_FAST_ROUNDS;
        break;
      }
      const m1_u32 missed = f.ended;
      mass = f.mass;
      first = missed;
      // Once every taken element is walked the repair takes the round's
      // sort; before, the other half of the cut's buffers.
      const bool spent = f.walked == gathered;
      m1_u64* mended = spent ? cut.chunk : mends;
      const m1_u32 holds = spent ? planned : mending;
      const bool defer = f.plan[missed] == M1_FAST_DEFER;
      m1_u32 end = missed + 1u;
      m1_u32 held = f.count[missed];
      while (end < M1_FAST_BUCKETS &&
             (f.plan[end] == (defer ? M1_FAST_DEFER : M1_FAST_PAST) || f.plan[end] == M1_FAST_EMPTY) &&
             held + f.count[end] <= holds) {
        held += f.count[end];
        ++end;
      }
      if (repairs == M1_FAST_ROUNDS * 2u || held > holds) break;
      ++repairs;
      m1_u32 count = 0u;
      if (defer) {
        // Deferred buckets: their elements are in the main gather.
        if (threadIdx.x == 0u) f.taken = 0u;
        __syncthreads();
        for (m1_u32 at = threadIdx.x; at < main_all; at += blockDim.x) {
          m1_u32 s = 0u;
          while (s + 1u < slices && f.mains[s + 1u] <= at) ++s;
          const m1_u64 composite = M1_LANE_GET64(gather_slot(s, at - f.mains[s], which));
          const m1_u32 b = bucket_of((m1_u32)(composite >> 32));
          if (b < missed || b >= end) continue;
          const m1_u32 slot = atomicAdd(&f.taken, 1u);
          if (slot < holds) mended[slot] = composite;
        }
        __syncthreads();
        count = f.taken;
        if (count != held) {
          round = M1_FAST_ROUNDS;
          break;
        }
      } else {
        // The repair pass: the elements of [missed, end), into the gather
        // buffer the main gather left (every block past the last reads of it
        // first).
        if (threadIdx.x == 0u) {
          f.taken = 0u;
          f.overflow = 0u;
        }
        barrier();
        const m1_u32 mend = which ^ 1u;
        m1_cut_scan_range(input, base, lo, hi, dtype, [&](m1_u32 i, float y) {
          if (m1_isnan(y)) return;
          const m1_u32 key = m1_desc_key(y);
          const m1_u32 b = bucket_of(key);
          const bool take = b >= missed && b < end;
          if (!take) return;
          const m1_u32 n = atomicAdd(&f.taken, 1u);
          if (n < mine) M1_LANE_PUT64(gather_slot(slice, n, mend), m1_cut_composite(key, i));
          else f.overflow = 1u;
        });
        __syncthreads();
        fast_merge(f, false);
        count = f.starts[slices];
        if (f.overflow != 0u || count != held) {
          round = M1_FAST_ROUNDS;
          break;
        }
        for (m1_u32 at = threadIdx.x; at < count; at += blockDim.x) {
          m1_u32 s = 0u;
          while (s + 1u < slices && f.starts[s + 1u] <= at) ++s;
          mended[at] = M1_LANE_GET64(gather_slot(s, at - f.starts[s], mend));
        }
        __syncthreads();
      }
      m1_cut_sort(mended, count);
      if (threadIdx.x < 32u) {
        float sum = mass;
        m1_u64 pivot = 0ull;
        const bool stopped = m1_fast_walk(mended, count, cutoff, sum, pivot);
        if (threadIdx.x == 0u) {
          f.mass = sum;
          f.pivot = pivot;
          f.outcome = stopped ? M1_FAST_STOPPED : M1_FAST_MISS;
        }
      }
      __syncthreads();
      mass = f.mass;
      if (f.outcome == M1_FAST_STOPPED) {
        if (threadIdx.x == 0u) {
          cut.stage = M1_CUT_STOPPED;
          cut.pivot = f.pivot;
          cut.mass = mass;
        }
        settled = true;
        break;
      }
      from = end;
      first = end;
      if (end == M1_FAST_BUCKETS) {
        if (threadIdx.x == 0u) {
          cut.stage = M1_CUT_ALL;
          cut.nan_kept = mass < cutoff ? 1u : 0u;
          cut.mass = mass;
        }
        settled = true;
        break;
      }
    }
  }
  if (!settled && threadIdx.x == 0u && first != 0u) {
    cut.cursor = m1_cut_composite(least + (m1_u32)((m1_u64)first << stretch) - 1u, 0xFFFFFFFFu);
    cut.mass = mass;
  }
  __syncthreads();
}

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

// A row's maximum (the canonical max), least value, and whether all are
// finite.
struct M1Range {
  float maximum;
  float least;
  m1_u32 finite;
};

__device__ __forceinline__ M1Range m1_range_combine(M1Range a, M1Range b) {
  return M1Range{m1_canonical_max(a.maximum, b.maximum), fminf(a.least, b.least), a.finite & b.finite};
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
  float* tree_work = cut.before;
  __shared__ M1Range block_range[32];
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
    // The row's maximum (order-free: the canonical max is), and its least
    // value and whether every value is finite (the top-p cut's first pass
    // then rides the probability pass).
    float m = m1_neg_inf();
    float least_x = __int_as_float(0x7F800000);
    m1_u32 finite = 1u;
    for (m1_u32 i = g.lo + threadIdx.x; i < g.hi; i += blockDim.x) {
      const float v = x[base + i];
      m = m1_canonical_max(m, v);
      least_x = fminf(least_x, v);
      if (!(fabsf(v) < __int_as_float(0x7F800000))) finite = 0u;
    }
    for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
      m = m1_canonical_max(m, m1_chunk_down(m, offset));
      least_x = fminf(least_x, m1_chunk_down(least_x, offset));
      finite &= m1_chunk_down(finite, offset);
    }
    if (lane == 0u) block_range[warp] = M1Range{m, least_x, finite};
    __syncthreads();
    if (threadIdx.x == 0u) {
      M1Range t{m1_neg_inf(), __int_as_float(0x7F800000), 1u};
      for (m1_u32 w = 0u; w < warps; ++w) t = m1_range_combine(t, block_range[w]);
      block_range[0] = t;
    }
    __syncthreads();
    const M1Range range = g.merge(block_range[0], m1_range_combine);
    const float maximum = range.maximum;
    // The exponentials, and their width-32 chunk sums (the tree's first
    // level) into the keep plane, which the cut has not reached yet.
    float* chunk_sums = reinterpret_cast<float*>(keep + ((base + 3u) & ~3u));
    const m1_u32 first_chunk = g.lo >> 5;
    const m1_u32 end_chunk = (g.hi + 31u) >> 5;
    for (m1_u32 c = first_chunk + warp; c < end_chunk; c += warps) {
      const m1_u32 i = c * 32u + lane;
      float e = i < len ? expf(x[base + i] - maximum) : 0.0f;
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
    // The probabilities (the exponentials again, as the reference's), the
    // top-p cut's notes taken on the way. Over finite logits they are known
    // before they are made: none negative, wide or NaN, the largest the
    // maximum's 1 / sum, the least the least logit's; the fast cut's first
    // pass rides along.
    if constexpr (!RANK) m1_cut_reset(cut);
    const bool counted = !RANK && range.finite != 0u;
    m1_u32 least = 0xFFFFFFFFu;
    m1_u32 most = 0u;
    int stretch = 0;
    M1Fast& fs = m1_fast_state();
    if (counted) {
      least = m1_desc_key(expf(maximum - maximum) / sum);
      most = m1_desc_key(expf(range.least - maximum) / sum);
      if (most < least) most = least;
      stretch = m1_fast_stretch(least, most);
      m1_fast_clear(fs);
    }
    __syncthreads();
    for (m1_u32 i = g.lo + threadIdx.x; i < g.hi; i += blockDim.x) {
      const float q = expf(x[base + i] - maximum) / sum;
      p[base + i] = q;
      if constexpr (!RANK) {
        if (counted) m1_fast_note(fs, m1_fast_bucket(m1_desc_key(q), least, stretch), i, q);
        else m1_cut_note(cut, i, q, least, most);
      }
    }
    // The cut; where its verdicts follow from its outcome (top-k's key, a
    // top-p walk's pivot), the argmax pass takes them from it.
    bool marked = false;
    m1_u32 rank_key = 0u;
    m1_u32 rank_k = 0u;
    if constexpr (RANK) {
      m1_u32 k;
      if (threshold_desc.dtype == 1u) {
        const int signed_k = m1_load_i(threshold, m1_pick(threshold_desc.len, row), threshold_desc.dtype);
        k = signed_k <= 0 ? 0u : (m1_u32)signed_k;
      } else {
        k = m1_load_u(threshold, m1_pick(threshold_desc.len, row), threshold_desc.dtype);
      }
      if (k > len) k = len;
      __syncthreads();
      rank_k = k;
      if (k != 0u) rank_key = m1_rank_row(probs, base, len, 0u, k, keep, g, false);
    } else {
      const float cutoff = m1_load_f(threshold, m1_pick(threshold_desc.len, row), threshold_desc.dtype);
      if (counted) {
        __syncthreads();
        g.fast_counted(fs);
        if (threadIdx.x == 0u) {
          cut.key_least = least;
          cut.key_most = most;
          const m1_u32 differ = least ^ most;
          m1_u32 skip = 0u;
          while (skip < 3u && (differ >> (24u - 8u * skip)) == 0u) ++skip;
          cut.key_skip = skip;
        }
        __syncthreads();
        g.counted = true;
      } else {
        m1_cut_enter(cut, g, least, most);
      }
      m1_cut_walk(probs, base, len, 0u, cutoff, keep, cut, g);
      if (cut.negative != 0u) {
        m1_cut_mark(probs, base, 0u, keep, cut, g);
        marked = true;
      }
    }
    // The cut logits, Gumbel noise, the argmax.
    M1ArgmaxCandidate best{m1_neg_inf(), 0u, 0u, 0u};
    for (m1_u32 i = g.lo + threadIdx.x; i < g.hi; i += blockDim.x) {
      bool kept;
      if (marked) {
        kept = m1_load_b(keep, base + i, 3u);
      } else {
        const float q = p[base + i];
        if constexpr (RANK) kept = rank_k != 0u && !m1_isnan(q) && m1_desc_key(q) <= rank_key;
        else kept = m1_cut_kept(cut, i, q);
      }
      // A cut element's value is -inf whatever its noise (the uniform draw
      // stays inside (0, 1), so the noise is finite): it is not drawn.
      float value = m1_neg_inf();
      if (kept) {
        const float uniform = ptir_rng_hash_uniform(seed, base + i);
        value = x[base + i] + (-logf(-logf(uniform)));
      }
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
