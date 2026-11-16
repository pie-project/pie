// The closed set of calls a stepwise guest stage dispatches
// (`eta_compiler::codegen::steps`), spelled for NVRTC and hiprtc: the WGSL
// runtime's steps (`runtime/wgsl/ptir_runtime.wgsl`) over the same tables,
// and the edges the engine runs on the device here (a channel's read and
// put, an output's row). A call names no program: it reads its node, rung
// and selector from its arguments (the step's uniform) and finds the node's
// operands through the stage's tables, which a bind lays
// (`eta_exec::tables`). One launch runs a step for every lane of a fire:
// `groups` blocks a lane, each lane over its own scratch.

typedef unsigned char ps_u8;
typedef unsigned short ps_u16;
typedef unsigned int ps_u32;
typedef unsigned long long ps_u64;

#define PTIR_WG 256u
#define DESC_WORDS 9u
#define PARAM_WORDS 16u

#define DT_F32 0u
#define DT_I32 1u
#define DT_U32 2u
#define DT_BOOL 3u

#define RED_SUM 0u
#define RED_MAX 1u
#define RED_MIN 2u
#define RED_ARGMAX 3u
#define PTIR_REDUCE_LEVELS 7u
#define PIVOT_UNROLL 32u

#define SEL_CAP 1024u
#define SEL_GROUPS 32u
#define SEL_WORDS 10320u
#define SEL_MODE 0u
#define SEL_ACTIVE 1u
#define SEL_WANT 2u
#define SEL_PREFIX 3u
#define SEL_REMAINING 4u
#define SEL_TOTAL_LT 5u
#define SEL_KEEP 6u
#define SEL_RUNS 0u
#define SEL_DECIDED 1u
#define SEL_LADDER 2u
#define SEL_KEEP_CUT 0u
#define SEL_KEEP_ALL 1u
#define SEL_KEEP_NONE 2u

struct PtirLaneTableHeader {
  ps_u32 abi_version;
  ps_u32 lane_count;
  ps_u32 channel_slots_per_lane;
  ps_u32 flags;
};

struct PtirLaneRecord {
  ps_u64 logits_base;
  ps_u32 logits_row_offset;
  ps_u32 logits_row_count;
  ps_u32 row_count;
  ps_u32 token_count;
  ps_u32 sampled_rows;
  ps_u32 query_len;
  ps_u32 key_len;
  ps_u32 channel_slot_offset;
  ps_u64 rng_state;
  ps_u64 commit_slot;
  ps_u64 active_row_mask;
  ps_u64 sample_output_channel_mask;
  ps_u64 row_valid;
  ps_u32 row_valid_offset;
  ps_u32 reserved0;
  ps_u64 attn_score_base;
  ps_u32 attn_score_row_stride;
  ps_u32 reserved1;
  ps_u64 mtp_drafts_base;
  ps_u32 mtp_drafts_depth;
  ps_u32 reserved2;
};

struct PtirLaneChannelSlot {
  ps_u64 committed_cell;
  ps_u64 pending_cell;
  ps_u64 expected_head;
  ps_u64 expected_tail;
};

struct PsCfg {
  ps_u32 value_count;
  ps_u32 temporary;
  ps_u32 op_count;
  ps_u32 lane;
};

static_assert(sizeof(PtirLaneTableHeader) == 16, "lane header ABI");
static_assert(sizeof(PtirLaneRecord) == 120, "lane record ABI");
static_assert(sizeof(PtirLaneChannelSlot) == 32, "lane channel ABI");
static_assert(sizeof(PsCfg) == 16, "cfg ABI");

// A block's own: whether its lane's fire runs, and the one-row select's.
__shared__ ps_u32 ps_go;
__shared__ ps_u32 sel_wg_hist[256];
__shared__ ps_u32 sel_wg_fill[2];
__shared__ ps_u32 sel_wg_key[1024];
__shared__ ps_u32 sel_wg_idx[1024];
__shared__ ps_u32 sel_wg_skip;

__device__ __forceinline__ float ps_f(ps_u32 bits) { return __uint_as_float(bits); }
__device__ __forceinline__ ps_u32 ps_bits(float x) { return __float_as_uint(x); }
__device__ __forceinline__ ps_u32 ps_maxu(ps_u32 a, ps_u32 b) { return a > b ? a : b; }
__device__ __forceinline__ ps_u32 ps_minu(ps_u32 a, ps_u32 b) { return a < b ? a : b; }
__device__ __forceinline__ int ps_maxi(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ int ps_mini(int a, int b) { return a < b ? a : b; }
// WGSL's conversions from f32 saturate, NaN to 0.
__device__ __forceinline__ int ps_f2i(float x) { return __float2int_rz(x); }
__device__ __forceinline__ ps_u32 ps_f2u(float x) { return __float2uint_rz(x); }

__device__ __forceinline__ ps_u64 ps_splitmix64(ps_u64 x) {
  x = (x ^ (x >> 27)) * 0x3C79AC492BA7B653ull;
  x = (x ^ (x >> 33)) * 0x1C69B3F74AC4AE35ull;
  return x ^ (x >> 27);
}
#define PS_RNG_STRIDE 0x9E3779B97F4A7C15ull

__device__ __forceinline__ ps_u64 ps_rng_seed_eff_stream(ps_u32 seed, ps_u32 stream) {
  return ((ps_u64)seed ^ 0xA5A5A5A5ull) ^ ps_splitmix64((ps_u64)stream * PS_RNG_STRIDE);
}

__device__ __forceinline__ ps_u64 ps_rng_keyed_seed(ps_u32 key, ps_u32 ctr) {
  return ps_splitmix64(((ps_u64)key << 32) | (ps_u64)ctr);
}

__device__ __forceinline__ float ps_rng_hash_uniform(ps_u64 seed_eff, ps_u32 index) {
  const ps_u64 x = seed_eff + PS_RNG_STRIDE * (ps_u64)(ps_u32)(index + 1u);
  const ps_u32 bits = (ps_u32)(ps_splitmix64(x) >> 40);
  const float raw = ((float)bits + 0.5f) * (1.0f / 16777216.0f);
  if (raw < 0.99999994f) return raw;
  return 0.99999994f;
}

__device__ __forceinline__ float ps_rng_hash_normal(ps_u64 seed_eff, ps_u32 index) {
  const ps_u32 lane = index * 2u;
  const float u0 = ps_rng_hash_uniform(seed_eff, lane);
  const float u1 = ps_rng_hash_uniform(seed_eff, lane + 1u);
  const float radius = sqrtf(-2.0f * logf(u0));
  return radius * cosf(6.2831855f * u1);
}

__device__ __forceinline__ ps_u32 ps_sort_key(float x) {
  if (x != x) return 0xFFFFFFFFu;
  const ps_u32 b = x == 0.0f ? 0u : ps_bits(x);
  ps_u32 asc;
  if ((b & 0x80000000u) != 0u) asc = ~b; else asc = b | 0x80000000u;
  return 0xFFFFFFFFu - asc;
}

__device__ __forceinline__ bool ps_key_less(ps_u32 ah, ps_u32 al, ps_u32 bh, ps_u32 bl) {
  if (ah != bh) return ah < bh;
  return al < bl;
}

__device__ __forceinline__ bool ps_sign_bit(float x) { return (ps_bits(x) & 0x80000000u) != 0u; }
__device__ __forceinline__ float ps_pos_inf() { return ps_f(0x7F800000u); }
__device__ __forceinline__ float ps_neg_inf() { return ps_f(0xFF800000u); }
__device__ __forceinline__ float ps_neg_zero() { return ps_f(0x80000000u); }

__device__ __forceinline__ float ps_cmax(float l, float r) {
  const bool ln = l != l;
  const bool rn = r != r;
  if (ln && rn) return ps_neg_inf();
  if (ln) return r;
  if (rn) return l;
  if (l == 0.0f && r == 0.0f) {
    if (ps_sign_bit(l) && ps_sign_bit(r)) return ps_neg_zero();
    return 0.0f;
  }
  if (l > r) return l;
  return r;
}

__device__ __forceinline__ float ps_cmin(float l, float r) {
  const bool ln = l != l;
  const bool rn = r != r;
  if (ln && rn) return ps_pos_inf();
  if (ln) return r;
  if (rn) return l;
  if (l == 0.0f && r == 0.0f) {
    if (ps_sign_bit(l) || ps_sign_bit(r)) return ps_neg_zero();
    return 0.0f;
  }
  if (l < r) return l;
  return r;
}

__device__ __forceinline__ float ps_red_identity(ps_u32 kind) {
  if (kind == RED_SUM) return 0.0f;
  if (kind == RED_MIN) return ps_pos_inf();
  return ps_neg_inf();
}

__device__ __forceinline__ float ps_red_combine(ps_u32 kind, float l, float r) {
  if (kind == RED_SUM) return l + r;
  if (kind == RED_MAX) return ps_cmax(l, r);
  return ps_cmin(l, r);
}

__device__ __forceinline__ bool ps_arg_takes(
    float lv, ps_u32 li, float rv, ps_u32 ri, ps_u32 count, ps_u32 lane, ps_u32 off) {
  if (lane + off >= count) return false;
  const bool lnan = lv != lv;
  const bool rnan = rv != rv;
  if (rnan) return false;
  if (lnan) return true;
  return rv > lv || (rv == lv && ri < li);
}

__device__ __forceinline__ ps_u32 ps_pick(ps_u32 n, ps_u32 i) { return n == 1u ? 0u : i; }

__device__ __forceinline__ float ps_arith_f(ps_u32 which, float x, float y) {
  switch (which) {
    case 0u: return x + y;
    case 1u: return x - y;
    case 2u: return x * y;
    case 3u: return x / y;
    case 4u: return fmaxf(x, y);
    case 5u: return fminf(x, y);
    default: return fmodf(x, y);
  }
}

__device__ __forceinline__ int ps_arith_i(ps_u32 which, int x, int y) {
  switch (which) {
    case 0u: return (int)((ps_u32)x + (ps_u32)y);
    case 1u: return (int)((ps_u32)x - (ps_u32)y);
    case 2u: return (int)((ps_u32)x * (ps_u32)y);
    case 3u: if (y == 0) return 0; if (y == -1) return (int)(0u - (ps_u32)x); return x / y;
    case 4u: return ps_maxi(x, y);
    case 5u: return ps_mini(x, y);
    default: if (y == 0 || y == -1) return 0; return x % y;
  }
}

__device__ __forceinline__ ps_u32 ps_arith_u(ps_u32 which, ps_u32 x, ps_u32 y) {
  switch (which) {
    case 0u: return x + y;
    case 1u: return x - y;
    case 2u: return x * y;
    case 3u: if (y == 0u) return 0u; return x / y;
    case 4u: return ps_maxu(x, y);
    case 5u: return ps_minu(x, y);
    default: if (y == 0u) return 0u; return x % y;
  }
}

// One lane's view of its stage: the tables and its scratch, as the WGSL
// runtime's bindings and private variables hold them.
struct Ps {
  const ps_u32* descs;
  const ps_u32* params;
  const ps_u32* offs;
  ps_u32* heap;
  ps_u32 temporary;
  ps_u32 tid;
  ps_u32 lanes;
  ps_u32 sel_lt;
  ps_u32 sel_group;
  ps_u32 sel_groups;

  __device__ ps_u32 d_len(ps_u32 v) { return descs[v * DESC_WORDS + 0u]; }
  __device__ ps_u32 d_rows(ps_u32 v) { return descs[v * DESC_WORDS + 1u]; }
  __device__ ps_u32 d_last(ps_u32 v) { return descs[v * DESC_WORDS + 2u]; }
  __device__ ps_u32 d_rank(ps_u32 v) { return descs[v * DESC_WORDS + 3u]; }
  __device__ ps_u32 d_dtype(ps_u32 v) { return descs[v * DESC_WORDS + 4u]; }
  __device__ ps_u32 d_dim(ps_u32 v, ps_u32 k) { return descs[v * DESC_WORDS + 5u + k]; }

  __device__ ps_u32 base(ps_u32 v) { return offs[v] >> 2u; }
  __device__ ps_u8* bytes(ps_u32 v) { return reinterpret_cast<ps_u8*>(heap) + offs[v]; }

  __device__ ps_u32 rows_of(ps_u32 v) {
    const ps_u32 rank = d_rank(v);
    if (rank < 2u) return 1u;
    ps_u32 n = 1u;
    for (ps_u32 k = 0u; k + 1u < rank; ++k) n = n * d_dim(v, k);
    return n;
  }

  __device__ ps_u32 p_at(ps_u32 p, ps_u32 k) { return params[p * PARAM_WORDS + k]; }
  __device__ ps_u32 p_tag(ps_u32 p) { return p_at(p, 0u); }
  __device__ ps_u32 p_a0(ps_u32 p) { return p_at(p, 1u); }
  __device__ ps_u32 p_a1(ps_u32 p) { return p_at(p, 2u); }
  __device__ ps_u32 p_a2(ps_u32 p) { return p_at(p, 3u); }
  __device__ ps_u32 p_o0(ps_u32 p) { return p_at(p, 4u); }
  __device__ ps_u32 p_o1(ps_u32 p) { return p_at(p, 5u); }
  __device__ ps_u32 p_imm(ps_u32 p) { return p_at(p, 6u); }
  __device__ ps_u32 p_imm2(ps_u32 p) { return p_at(p, 7u); }
  __device__ ps_u32 p_imm3(ps_u32 p) { return p_at(p, 8u); }
  __device__ ps_u32 p_kind(ps_u32 p) { return p_at(p, 9u); }
  __device__ ps_u32 p_pred_tag(ps_u32 p) { return p_at(p, 10u); }
  __device__ ps_u32 p_lit_dtype(ps_u32 p) { return p_at(p, 11u); }
  __device__ ps_u32 p_lit_bits(ps_u32 p) { return p_at(p, 12u); }
  __device__ ps_u32 p_channel(ps_u32 p) { return p_at(p, 13u); }
  __device__ ps_u32 p_intr(ps_u32 p) { return p_at(p, 14u); }
  __device__ ps_u32 p_sink_bytes(ps_u32 p) { return p_at(p, 15u); }
  __device__ ps_u32 p_pred_payload(ps_u32 p) { return p_a1(p); }

  __device__ ps_u32 ld_raw(ps_u32 v, ps_u32 i) { return heap[base(v) + i]; }

  __device__ ps_u32 ld_byte(ps_u32 v, ps_u32 i) {
    const ps_u32 w = heap[base(v) + (i >> 2u)];
    return (w >> ((i & 3u) * 8u)) & 0xFFu;
  }

  __device__ float ld_f(ps_u32 v, ps_u32 i) {
    const ps_u32 dt = d_dtype(v);
    if (dt == DT_F32) return ps_f(ld_raw(v, i));
    if (dt == DT_I32) return (float)(int)ld_raw(v, i);
    if (dt == DT_U32) return (float)ld_raw(v, i);
    return ld_byte(v, i) != 0u ? 1.0f : 0.0f;
  }

  __device__ int ld_i(ps_u32 v, ps_u32 i) {
    const ps_u32 dt = d_dtype(v);
    if (dt == DT_F32) return ps_f2i(ps_f(ld_raw(v, i)));
    if (dt == DT_I32) return (int)ld_raw(v, i);
    if (dt == DT_U32) return (int)ld_raw(v, i);
    return ld_byte(v, i) != 0u ? 1 : 0;
  }

  __device__ ps_u32 ld_u(ps_u32 v, ps_u32 i) {
    const ps_u32 dt = d_dtype(v);
    if (dt == DT_F32) return ps_f2u(ps_f(ld_raw(v, i)));
    if (dt == DT_I32) return ld_raw(v, i);
    if (dt == DT_U32) return ld_raw(v, i);
    return ld_byte(v, i) != 0u ? 1u : 0u;
  }

  __device__ bool ld_b(ps_u32 v, ps_u32 i) {
    const ps_u32 dt = d_dtype(v);
    if (dt == DT_BOOL) return ld_byte(v, i) != 0u;
    return ld_raw(v, i) != 0u;
  }

  __device__ void st_raw(ps_u32 v, ps_u32 i, ps_u32 w) { heap[base(v) + i] = w; }
  __device__ void st_f(ps_u32 v, ps_u32 i, float x) { st_raw(v, i, ps_bits(x)); }
  __device__ void st_i(ps_u32 v, ps_u32 i, int x) { st_raw(v, i, (ps_u32)x); }
  __device__ void st_u(ps_u32 v, ps_u32 i, ps_u32 x) { st_raw(v, i, x); }

  __device__ void st_b_word(ps_u32 v, ps_u32 w, ps_u32 n, const bool* b) {
    ps_u32 packed = 0u;
    if (b[0] && (w * 4u + 0u) < n) packed |= 0x000000FFu;
    if (b[1] && (w * 4u + 1u) < n) packed |= 0x0000FF00u;
    if (b[2] && (w * 4u + 2u) < n) packed |= 0x00FF0000u;
    if (b[3] && (w * 4u + 3u) < n) packed |= 0xFF000000u;
    heap[base(v) + w] = packed;
  }

  __device__ void st_as_f(ps_u32 v, ps_u32 i, float x) {
    const ps_u32 dt = d_dtype(v);
    if (dt == DT_F32) st_f(v, i, x);
    else if (dt == DT_I32) st_i(v, i, ps_f2i(x));
    else if (dt == DT_U32) st_u(v, i, ps_f2u(x));
  }

  __device__ void st_as_i(ps_u32 v, ps_u32 i, int x) {
    const ps_u32 dt = d_dtype(v);
    if (dt == DT_F32) st_f(v, i, (float)x);
    else if (dt == DT_I32) st_i(v, i, x);
    else if (dt == DT_U32) st_u(v, i, (ps_u32)x);
  }

  // ---- elementwise ----

  __device__ float unary_f(ps_u32 which, ps_u32 a, ps_u32 j) {
    const float x = ld_f(a, j);
    switch (which) {
      case 0u: return expf(x);
      case 1u: return logf(x);
      case 2u: return -x;
      case 3u: return 1.0f / x;
      case 4u: return fabsf(x);
      case 5u:
        if (x > 0.0f) return 1.0f;
        if (x < 0.0f) return -1.0f;
        return 0.0f;
      case 7u: return sinf(x);
      case 8u: return cosf(x);
      case 9u: return sqrtf(x);
      case 10u: return 1.0f / sqrtf(x);
      default: return x;
    }
  }

  __device__ int unary_i(ps_u32 which, ps_u32 a, ps_u32 j) {
    switch (which) {
      case 0u: return ps_f2i(expf(ld_f(a, j)));
      case 1u: return ps_f2i(logf(ld_f(a, j)));
      case 3u: return ps_f2i(1.0f / ld_f(a, j));
      case 2u: return (int)(0u - (ps_u32)ld_i(a, j));
      case 4u: {
        const int v = ld_i(a, j);
        if (v == -2147483647 - 1) return v;
        return v < 0 ? -v : v;
      }
      case 5u: {
        const int v = ld_i(a, j);
        return v > 0 ? 1 : (v < 0 ? -1 : 0);
      }
      case 7u: return ps_f2i(sinf(ld_f(a, j)));
      case 8u: return ps_f2i(cosf(ld_f(a, j)));
      case 9u: return ps_f2i(sqrtf(ld_f(a, j)));
      case 10u: return ps_f2i(1.0f / sqrtf(ld_f(a, j)));
      default: return ld_i(a, j);
    }
  }

  __device__ ps_u32 unary_u(ps_u32 which, ps_u32 a, ps_u32 j) {
    switch (which) {
      case 0u: return ps_f2u(expf(ld_f(a, j)));
      case 1u: return ps_f2u(logf(ld_f(a, j)));
      case 3u: return ps_f2u(1.0f / ld_f(a, j));
      case 2u: return 0u - ld_u(a, j);
      case 4u: return ld_u(a, j);
      case 5u: return ld_u(a, j) != 0u ? 1u : 0u;
      case 7u: return ps_f2u(sinf(ld_f(a, j)));
      case 8u: return ps_f2u(cosf(ld_f(a, j)));
      case 9u: return ps_f2u(sqrtf(ld_f(a, j)));
      case 10u: return ps_f2u(1.0f / sqrtf(ld_f(a, j)));
      default: return ld_u(a, j);
    }
  }

  __device__ bool unary_b(ps_u32 which, ps_u32 a, ps_u32 j) {
    (void)which;
    return ld_f(a, j) != 0.0f;
  }

  __device__ void map_unary(ps_u32 p, ps_u32 which) {
    const ps_u32 a = p_a0(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(o);
    const ps_u32 odt = d_dtype(o);
    if (odt == DT_BOOL) {
      const ps_u32 words = (n + 3u) / 4u;
      for (ps_u32 w = tid; w < words; w += lanes) {
        bool b[4];
        for (ps_u32 k = 0u; k < 4u; ++k) {
          const ps_u32 i = w * 4u + k;
          b[k] = i < n ? unary_b(which, a, ps_pick(d_len(a), i)) : false;
        }
        st_b_word(o, w, n, b);
      }
      return;
    }
    for (ps_u32 i = tid; i < n; i += lanes) {
      const ps_u32 j = ps_pick(d_len(a), i);
      if (odt == DT_F32) st_f(o, i, unary_f(which, a, j));
      else if (odt == DT_I32) st_i(o, i, unary_i(which, a, j));
      else st_u(o, i, unary_u(which, a, j));
    }
  }

  __device__ void bin_arith(ps_u32 p, ps_u32 which) {
    const ps_u32 a = p_a0(p);
    const ps_u32 b = p_a1(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(o);
    const ps_u32 adt = d_dtype(a);
    const ps_u32 na = d_len(a);
    const ps_u32 nb = d_len(b);
    if (adt == DT_F32) {
      for (ps_u32 i = tid; i < n; i += lanes)
        st_as_f(o, i, ps_arith_f(which, ld_f(a, ps_pick(na, i)), ld_f(b, ps_pick(nb, i))));
      return;
    }
    if (adt == DT_U32) {
      for (ps_u32 i = tid; i < n; i += lanes) {
        const ps_u32 r = ps_arith_u(which, ld_u(a, ps_pick(na, i)), ld_u(b, ps_pick(nb, i)));
        if (d_dtype(o) == DT_F32) st_f(o, i, (float)r); else st_u(o, i, r);
      }
      return;
    }
    for (ps_u32 i = tid; i < n; i += lanes)
      st_as_i(o, i, ps_arith_i(which, ld_i(a, ps_pick(na, i)), ld_i(b, ps_pick(nb, i))));
  }

  __device__ bool cmp_lane(ps_u32 which, ps_u32 a, ps_u32 b, ps_u32 i) {
    const ps_u32 ia = ps_pick(d_len(a), i);
    const ps_u32 ib = ps_pick(d_len(b), i);
    const ps_u32 adt = d_dtype(a);
    if (adt == DT_F32) {
      const float x = ld_f(a, ia);
      const float y = ld_f(b, ib);
      switch (which) {
        case 0u: return x > y;
        case 1u: return x >= y;
        case 2u: return x == y;
        case 3u: return x != y;
        case 4u: return x < y;
        default: return x <= y;
      }
    }
    if (adt == DT_U32) {
      const ps_u32 x = ld_u(a, ia);
      const ps_u32 y = ld_u(b, ib);
      switch (which) {
        case 0u: return x > y;
        case 1u: return x >= y;
        case 2u: return x == y;
        case 3u: return x != y;
        case 4u: return x < y;
        default: return x <= y;
      }
    }
    const int x = ld_i(a, ia);
    const int y = ld_i(b, ib);
    switch (which) {
      case 0u: return x > y;
      case 1u: return x >= y;
      case 2u: return x == y;
      case 3u: return x != y;
      case 4u: return x < y;
      default: return x <= y;
    }
  }

  __device__ void cmp_op(ps_u32 p, ps_u32 which) {
    const ps_u32 a = p_a0(p);
    const ps_u32 b = p_a1(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(o);
    const ps_u32 words = (n + 3u) / 4u;
    for (ps_u32 w = tid; w < words; w += lanes) {
      bool bits[4];
      for (ps_u32 k = 0u; k < 4u; ++k) {
        const ps_u32 i = w * 4u + k;
        bits[k] = i < n ? cmp_lane(which, a, b, i) : false;
      }
      st_b_word(o, w, n, bits);
    }
  }

  __device__ void logic_op(ps_u32 p, ps_u32 which) {
    const ps_u32 a = p_a0(p);
    const ps_u32 b = p_a1(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(o);
    const ps_u32 words = (n + 3u) / 4u;
    for (ps_u32 w = tid; w < words; w += lanes) {
      bool bits[4];
      for (ps_u32 k = 0u; k < 4u; ++k) {
        const ps_u32 i = w * 4u + k;
        bool v = false;
        if (i < n) {
          const bool x = ld_b(a, ps_pick(d_len(a), i));
          if (which == 2u) {
            v = !x;
          } else {
            const bool y = ld_b(b, ps_pick(d_len(b), i));
            v = which == 0u ? (x && y) : (x || y);
          }
        }
        bits[k] = v;
      }
      st_b_word(o, w, n, bits);
    }
  }

  __device__ void op_select(ps_u32 p) {
    const ps_u32 c = p_a0(p);
    const ps_u32 x = p_a1(p);
    const ps_u32 y = p_a2(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(o);
    const ps_u32 odt = d_dtype(o);
    if (odt == DT_BOOL) {
      const ps_u32 words = (n + 3u) / 4u;
      for (ps_u32 w = tid; w < words; w += lanes) {
        bool bits[4];
        for (ps_u32 k = 0u; k < 4u; ++k) {
          const ps_u32 i = w * 4u + k;
          bool v = false;
          if (i < n) {
            if (ld_b(c, ps_pick(d_len(c), i))) v = ld_b(x, ps_pick(d_len(x), i));
            else v = ld_b(y, ps_pick(d_len(y), i));
          }
          bits[k] = v;
        }
        st_b_word(o, w, n, bits);
      }
      return;
    }
    for (ps_u32 i = tid; i < n; i += lanes) {
      const bool take = ld_b(c, ps_pick(d_len(c), i));
      if (odt == DT_F32) {
        st_f(o, i, take ? ld_f(x, ps_pick(d_len(x), i)) : ld_f(y, ps_pick(d_len(y), i)));
      } else if (odt == DT_I32) {
        st_i(o, i, take ? ld_i(x, ps_pick(d_len(x), i)) : ld_i(y, ps_pick(d_len(y), i)));
      } else {
        st_u(o, i, take ? ld_u(x, ps_pick(d_len(x), i)) : ld_u(y, ps_pick(d_len(y), i)));
      }
    }
  }

  // ---- reductions ----

  __device__ ps_u32 tmp_v(ps_u32 row, ps_u32 stride, ps_u32 c, ps_u32 buf, ps_u32 span) {
    return temporary + (buf * span + row * stride + c) * 2u;
  }

  __device__ void op_reduce_level(ps_u32 p, ps_u32 kind, ps_u32 level) {
    const ps_u32 a = p_a0(p);
    const ps_u32 rows = rows_of(a);
    const ps_u32 len = d_len(a) / ps_maxu(rows, 1u);
    const ps_u32 stride = (len + 31u) / 32u;
    ps_u32 n_in = len;
    for (ps_u32 l = 0u; l < level; ++l) n_in = (n_in + 31u) / 32u;
    if (n_in <= 1u) return;
    const ps_u32 n_out = (n_in + 31u) / 32u;
    const ps_u32 total = rows * n_out;
    for (ps_u32 job = tid; job < total; job += lanes) {
      const ps_u32 row = job / n_out;
      const ps_u32 c = job % n_out;
      float vs[32];
      ps_u32 ix[32];
      const ps_u32 count = ps_minu(32u, n_in - c * 32u);
      for (ps_u32 k = 0u; k < 32u; ++k) {
        vs[k] = ps_red_identity(kind);
        ix[k] = 0u;
        if (k < count) {
          const ps_u32 at = c * 32u + k;
          if (level == 0u) {
            vs[k] = ld_f(a, row * len + at);
            ix[k] = at;
          } else {
            const ps_u32 cell = tmp_v(row, stride, at, level % 2u, rows * stride);
            vs[k] = ps_f(heap[cell]);
            ix[k] = heap[cell + 1u];
          }
        }
      }
      for (ps_u32 off = 16u; off >= 1u; off = off / 2u) {
        for (ps_u32 lane = 0u; lane < off; ++lane) {
          if (kind == RED_ARGMAX) {
            if (ps_arg_takes(vs[lane], ix[lane], vs[lane + off], ix[lane + off], count, lane, off)) {
              vs[lane] = vs[lane + off];
              ix[lane] = ix[lane + off];
            }
          } else {
            vs[lane] = ps_red_combine(kind, vs[lane], vs[lane + off]);
          }
        }
        if (off == 1u) break;
      }
      const ps_u32 cell = tmp_v(row, stride, c, (level + 1u) % 2u, rows * stride);
      heap[cell] = ps_bits(vs[0]);
      heap[cell + 1u] = ix[0];
    }
  }

  __device__ void op_reduce_finish(ps_u32 p, ps_u32 kind) {
    const ps_u32 a = p_a0(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 rows = rows_of(a);
    const ps_u32 len = d_len(a) / ps_maxu(rows, 1u);
    const ps_u32 stride = (len + 31u) / 32u;
    for (ps_u32 row = tid; row < rows; row += lanes) {
      float v;
      ps_u32 idx;
      if (len == 0u) {
        v = ps_red_identity(kind);
        idx = 0u;
      } else if (len == 1u) {
        v = ld_f(a, row * len);
        idx = 0u;
      } else {
        ps_u32 folded = len;
        ps_u32 last = 0u;
        for (ps_u32 l = 0u; l < PTIR_REDUCE_LEVELS; ++l) {
          if (folded <= 1u) break;
          last = l;
          folded = (folded + 31u) / 32u;
        }
        const ps_u32 cell = tmp_v(row, stride, 0u, (last + 1u) % 2u, rows * stride);
        v = ps_f(heap[cell]);
        idx = heap[cell + 1u];
      }
      if (kind == RED_ARGMAX) st_i(o, row, (int)idx);
      else st_as_f(o, row, v);
    }
  }

  __device__ void op_cumulative(ps_u32 p, bool is_prod) {
    const ps_u32 a = p_a0(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 rows = rows_of(a);
    const ps_u32 len = d_len(a) / ps_maxu(rows, 1u);
    for (ps_u32 row = tid; row < rows; row += lanes) {
      const ps_u32 rowb = row * len;
      float acc = is_prod ? 1.0f : 0.0f;
      for (ps_u32 j = 0u; j < len; ++j) {
        const float x = ld_f(a, rowb + j);
        if (is_prod) acc = acc * x; else acc = acc + x;
        st_as_f(o, rowb + j, acc);
      }
    }
  }

  // ---- data movement ----

  __device__ void op_copy(ps_u32 p) {
    const ps_u32 a = p_a0(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(o);
    const ps_u32 na = d_len(a);
    const ps_u32 odt = d_dtype(o);
    if (odt == DT_BOOL) {
      const ps_u32 words = (n + 3u) / 4u;
      for (ps_u32 w = tid; w < words; w += lanes) {
        bool b[4];
        for (ps_u32 k = 0u; k < 4u; ++k) {
          const ps_u32 i = w * 4u + k;
          b[k] = i < n ? ld_b(a, ps_pick(na, i)) : false;
        }
        st_b_word(o, w, n, b);
      }
      return;
    }
    for (ps_u32 i = tid; i < n; i += lanes) {
      const ps_u32 j = ps_pick(na, i);
      if (odt == DT_F32) st_f(o, i, ld_f(a, j));
      else if (odt == DT_I32) st_i(o, i, ld_i(a, j));
      else st_u(o, i, ld_u(a, j));
    }
  }

  __device__ ps_u32 bcast_src(ps_u32 a, ps_u32 o, ps_u32 linear) {
    const ps_u32 orank = d_rank(o);
    const ps_u32 arank = d_rank(a);
    ps_u32 rem = linear;
    ps_u32 src = 0u;
    ps_u32 sstride[4] = {1u, 1u, 1u, 1u};
    for (int dim = (int)orank - 2; dim >= 0; --dim) {
      const ps_u32 next = (ps_u32)dim + 1u;
      ps_u32 sd = 1u;
      if (next < arank) sd = d_dim(a, next);
      sstride[dim] = sstride[dim + 1] * sd;
    }
    for (ps_u32 dim = 0u; dim < orank; ++dim) {
      ps_u32 stride = 1u;
      for (ps_u32 next = dim + 1u; next < orank; ++next) stride = stride * d_dim(o, next);
      if (stride == 0u) stride = 1u;
      const ps_u32 coord = rem / stride;
      rem = rem % stride;
      ps_u32 sd = 1u;
      if (dim < arank) sd = d_dim(a, dim);
      if (sd != 1u) src = src + coord * sstride[dim];
    }
    return src;
  }

  __device__ void op_broadcast(ps_u32 p) {
    const ps_u32 a = p_a0(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(o);
    const ps_u32 odt = d_dtype(o);
    if (odt == DT_BOOL) {
      const ps_u32 words = (n + 3u) / 4u;
      for (ps_u32 w = tid; w < words; w += lanes) {
        bool b[4];
        for (ps_u32 k = 0u; k < 4u; ++k) {
          const ps_u32 i = w * 4u + k;
          b[k] = i < n ? ld_b(a, bcast_src(a, o, i)) : false;
        }
        st_b_word(o, w, n, b);
      }
      return;
    }
    for (ps_u32 i = tid; i < n; i += lanes) {
      const ps_u32 j = bcast_src(a, o, i);
      if (odt == DT_F32) st_f(o, i, ld_f(a, j));
      else if (odt == DT_I32) st_i(o, i, ld_i(a, j));
      else st_u(o, i, ld_u(a, j));
    }
  }

  __device__ void op_transpose(ps_u32 p) {
    const ps_u32 a = p_a0(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 m = d_dim(a, 0u);
    const ps_u32 n2 = d_dim(a, 1u);
    const ps_u32 n = m * n2;
    const ps_u32 odt = d_dtype(o);
    if (odt == DT_BOOL) {
      const ps_u32 words = (n + 3u) / 4u;
      for (ps_u32 w = tid; w < words; w += lanes) {
        bool b[4];
        for (ps_u32 k = 0u; k < 4u; ++k) {
          const ps_u32 at = w * 4u + k;
          b[k] = at < n ? ld_b(a, (at % m) * n2 + at / m) : false;
        }
        st_b_word(o, w, n, b);
      }
      return;
    }
    for (ps_u32 at = tid; at < n; at += lanes) {
      const ps_u32 src = (at % m) * n2 + at / m;
      if (odt == DT_F32) st_f(o, at, ld_f(a, src));
      else if (odt == DT_I32) st_i(o, at, ld_i(a, src));
      else if (odt == DT_U32) st_u(o, at, ld_u(a, src));
    }
  }

  __device__ ps_u32 gather_rest(ps_u32 a) {
    const ps_u32 rank = d_rank(a);
    if (rank < 2u) return 1u;
    ps_u32 r = 1u;
    for (ps_u32 k = 1u; k < rank; ++k) r = r * d_dim(a, k);
    return ps_maxu(r, 1u);
  }

  __device__ ps_u32 gather_first(ps_u32 a) {
    if (d_rank(a) == 0u) return 1u;
    return d_dim(a, 0u);
  }

  __device__ void op_gather(ps_u32 p) {
    const ps_u32 a = p_a0(p);
    const ps_u32 ix = p_a1(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(o);
    const ps_u32 rest = gather_rest(a);
    const ps_u32 n0 = gather_first(a);
    const ps_u32 odt = d_dtype(o);
    if (odt == DT_BOOL) {
      const ps_u32 words = (n + 3u) / 4u;
      for (ps_u32 w = tid; w < words; w += lanes) {
        bool b[4];
        for (ps_u32 k = 0u; k < 4u; ++k) {
          const ps_u32 at = w * 4u + k;
          bool v = false;
          if (at < n) {
            const int want = ld_i(ix, ps_pick(d_len(ix), at / rest));
            if (want >= 0 && (ps_u32)want < n0) v = ld_b(a, (ps_u32)want * rest + at % rest);
          }
          b[k] = v;
        }
        st_b_word(o, w, n, b);
      }
      return;
    }
    for (ps_u32 at = tid; at < n; at += lanes) {
      const ps_u32 slot = at / rest;
      const ps_u32 within = at % rest;
      const int want = ld_i(ix, ps_pick(d_len(ix), slot));
      const bool ok = want >= 0 && (ps_u32)want < n0;
      const ps_u32 j = ok ? (ps_u32)want * rest + within : 0u;
      if (odt == DT_F32) st_f(o, at, ok ? ld_f(a, j) : 0.0f);
      else if (odt == DT_I32) st_i(o, at, ok ? ld_i(a, j) : 0);
      else if (odt == DT_U32) st_u(o, at, ok ? ld_u(a, j) : 0u);
    }
  }

  __device__ void op_gather_row(ps_u32 p) {
    const ps_u32 a = p_a0(p);
    const ps_u32 ix = p_a1(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 m = d_dim(a, 0u);
    const ps_u32 n2 = d_dim(a, 1u);
    const ps_u32 odt = d_dtype(o);
    if (odt == DT_BOOL) {
      const ps_u32 words = (m + 3u) / 4u;
      for (ps_u32 w = tid; w < words; w += lanes) {
        bool b[4];
        for (ps_u32 k = 0u; k < 4u; ++k) {
          const ps_u32 i = w * 4u + k;
          bool v = false;
          if (i < m) {
            const int c = ld_i(ix, ps_pick(d_len(ix), i));
            if (c >= 0 && (ps_u32)c < n2) v = ld_b(a, i * n2 + (ps_u32)c);
          }
          b[k] = v;
        }
        st_b_word(o, w, m, b);
      }
      return;
    }
    for (ps_u32 i = tid; i < m; i += lanes) {
      const int c = ld_i(ix, ps_pick(d_len(ix), i));
      const bool ok = c >= 0 && (ps_u32)c < n2;
      const ps_u32 j = ok ? i * n2 + (ps_u32)c : 0u;
      if (odt == DT_F32) st_f(o, i, ok ? ld_f(a, j) : 0.0f);
      else if (odt == DT_I32) st_i(o, i, ok ? ld_i(a, j) : 0);
      else if (odt == DT_U32) st_u(o, i, ok ? ld_u(a, j) : 0u);
    }
  }

  __device__ void op_scatter(ps_u32 p, bool is_add) {
    const ps_u32 a = p_a0(p);
    const ps_u32 ix = p_a1(p);
    const ps_u32 src = p_a2(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(o);
    const ps_u32 rest = gather_rest(a);
    const ps_u32 n0 = gather_first(a);
    const ps_u32 slots = d_len(ix);
    const bool scalar = d_len(src) == 1u && slots * rest != 1u;
    const ps_u32 odt = d_dtype(o);
    if (odt == DT_BOOL) {
      const ps_u32 words = (n + 3u) / 4u;
      for (ps_u32 w = tid; w < words; w += lanes) {
        bool b[4];
        for (ps_u32 k = 0u; k < 4u; ++k) {
          const ps_u32 at = w * 4u + k;
          bool acc = false;
          if (at < n) {
            acc = ld_b(a, at);
            const ps_u32 dest = at / rest;
            const ps_u32 within = at % rest;
            for (ps_u32 s = 0u; s < slots; ++s) {
              const int want = ld_i(ix, s);
              if (want < 0 || (ps_u32)want >= n0 || (ps_u32)want != dest) continue;
              const bool v = ld_b(src, scalar ? 0u : s * rest + within);
              acc = is_add ? (acc || v) : v;
            }
          }
          b[k] = acc;
        }
        st_b_word(o, w, n, b);
      }
      return;
    }
    for (ps_u32 at = tid; at < n; at += lanes) {
      float acc_f = ld_f(a, at);
      int acc_i = ld_i(a, at);
      const ps_u32 dest = at / rest;
      const ps_u32 within = at % rest;
      for (ps_u32 k = 0u; k < slots; ++k) {
        const int want = ld_i(ix, k);
        if (want < 0 || (ps_u32)want >= n0) continue;
        if ((ps_u32)want != dest) continue;
        ps_u32 lane = k * rest + within;
        if (scalar) lane = 0u;
        if (odt == DT_F32) {
          const float v = ld_f(src, lane);
          if (is_add) acc_f = acc_f + v; else acc_f = v;
        } else {
          const int v = ld_i(src, lane);
          if (is_add) acc_i = (int)((ps_u32)acc_i + (ps_u32)v); else acc_i = v;
        }
      }
      if (odt == DT_F32) st_f(o, at, acc_f); else st_as_i(o, at, acc_i);
    }
  }

  __device__ void op_iota(ps_u32 p) {
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(o);
    const ps_u32 odt = d_dtype(o);
    for (ps_u32 i = tid; i < n; i += lanes) {
      if (odt == DT_F32) st_f(o, i, (float)i);
      else if (odt == DT_I32) st_i(o, i, (int)i);
      else st_u(o, i, i);
    }
  }

  __device__ void op_const(ps_u32 p) {
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(o);
    const ps_u32 dt = p_lit_dtype(p);
    const ps_u32 bits = p_lit_bits(p);
    if (d_dtype(o) == DT_BOOL) {
      const ps_u32 words = (n + 3u) / 4u;
      const bool on = bits != 0u;
      const bool b[4] = {on, on, on, on};
      for (ps_u32 w = tid; w < words; w += lanes) st_b_word(o, w, n, b);
      return;
    }
    for (ps_u32 i = tid; i < n; i += lanes) {
      if (dt == DT_F32) st_as_f(o, i, ps_f(bits));
      else st_as_i(o, i, (int)bits);
    }
  }

  __device__ void op_mask_apply(ps_u32 p) {
    const ps_u32 a = p_a0(p);
    const ps_u32 m = p_a1(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(o);
    const ps_u32 rank = d_rank(a);
    ps_u32 width = 1u;
    if (rank > 0u) width = d_dim(a, rank - 1u);
    const ps_u32 mw = d_len(m);
    for (ps_u32 j = tid; j < n; j += lanes) {
      const ps_u32 c = j % ps_maxu(width, 1u);
      const ps_u32 w = c >> 5u;
      ps_u32 word = 0u;
      if (w < mw) word = ld_u(m, w);
      if (((word >> (c & 31u)) & 1u) != 0u) st_f(o, j, ld_f(a, j));
      else st_f(o, j, ps_neg_inf());
    }
  }

  __device__ void op_struct_mask(ps_u32 p, ps_u32 which) {
    const ps_u32 pos = p_a0(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 keys = p_imm(p);
    ps_u32 window = p_imm3(p);
    if (which == 1u) window = p_imm2(p);
    const ps_u32 sink = p_imm2(p);
    const ps_u32 n = d_len(o);
    const ps_u32 words = (n + 3u) / 4u;
    for (ps_u32 w = tid; w < words; w += lanes) {
      bool bits[4];
      for (ps_u32 k = 0u; k < 4u; ++k) {
        const ps_u32 at = w * 4u + k;
        bool v = false;
        if (at < n && keys != 0u) {
          const ps_u32 position = ld_u(pos, at / keys);
          const ps_u32 key = at % keys;
          bool allowed = key <= position;
          if (allowed && which != 0u) {
            ps_u32 sum = key + window;
            if (sum < key) sum = 0xFFFFFFFFu;
            const bool recent = sum > position;
            if (which == 1u) allowed = recent;
            else allowed = key < sink || recent;
          }
          v = allowed;
        }
        bits[k] = v;
      }
      st_b_word(o, w, n, bits);
    }
  }

  __device__ void op_rng(ps_u32 p, bool keyed) {
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(o);
    ps_u64 seed;
    if (keyed) {
      const ps_u32 st = p_a0(p);
      const ps_u32 key = ld_u(st, 0u);
      ps_u32 ctr = 0u;
      if (d_len(st) > 1u) ctr = ld_u(st, 1u);
      seed = ps_rng_keyed_seed(key, ctr);
    } else {
      seed = ps_rng_seed_eff_stream(0u, p_imm(p));
    }
    const ps_u32 kind = p_kind(p);
    for (ps_u32 i = tid; i < n; i += lanes) {
      if (kind == 2u) {
        st_f(o, i, ps_rng_hash_normal(seed, i));
        continue;
      }
      const float u = ps_rng_hash_uniform(seed, i);
      if (kind == 1u) st_f(o, i, -logf(-logf(u))); else st_f(o, i, u);
    }
  }

  // ---- the sort ladder ----

  __device__ ps_u32 sort_cell(ps_u32 buf, ps_u32 n, ps_u32 c) {
    return temporary + ((buf * n) + c) * 2u;
  }

  __device__ ps_u32 sort_rows_of(ps_u32 p, ps_u32 a) {
    if (p_tag(p) == 0x50u) return 1u;
    return rows_of(a);
  }

  __device__ void op_sort_seed(ps_u32 p) {
    const ps_u32 a = p_a0(p);
    const ps_u32 n = d_len(a);
    const ps_u32 rows = sort_rows_of(p, a);
    const ps_u32 len = n / ps_maxu(rows, 1u);
    for (ps_u32 i = tid; i < n; i += lanes) {
      const ps_u32 c = sort_cell(0u, n, i);
      heap[c] = ps_sort_key(ld_f(a, i));
      heap[c + 1u] = i - (i / ps_maxu(len, 1u)) * ps_maxu(len, 1u);
    }
  }

  __device__ void op_sort_round(ps_u32 p, ps_u32 r) {
    const ps_u32 a = p_a0(p);
    const ps_u32 n = d_len(a);
    const ps_u32 rows = sort_rows_of(p, a);
    const ps_u32 len = n / ps_maxu(rows, 1u);
    const ps_u32 src = r & 1u;
    const ps_u32 dst = 1u - src;
    const ps_u32 run = 1u << r;
    if (run >= len) {
      for (ps_u32 i = tid; i < n; i += lanes) {
        const ps_u32 s = sort_cell(src, n, i);
        const ps_u32 d = sort_cell(dst, n, i);
        heap[d] = heap[s];
        heap[d + 1u] = heap[s + 1u];
      }
      return;
    }
    const ps_u32 span = run * 2u;
    for (ps_u32 i = tid; i < n; i += lanes) {
      const ps_u32 row = i / ps_maxu(len, 1u);
      const ps_u32 rowb = row * len;
      const ps_u32 at = i - rowb;
      const ps_u32 blk = (at / span) * span;
      const ps_u32 mid = ps_minu(blk + run, len);
      const ps_u32 end = ps_minu(blk + span, len);
      const ps_u32 me = sort_cell(src, n, i);
      const ps_u32 mh = heap[me];
      const ps_u32 ml = heap[me + 1u];
      ps_u32 lo = 0u;
      ps_u32 hi = 0u;
      ps_u32 own = 0u;
      if (at < mid) {
        hi = end - mid;
        own = at - blk;
      } else {
        hi = mid - blk;
        own = at - mid;
      }
      while (lo < hi) {
        const ps_u32 m = lo + (hi - lo) / 2u;
        const ps_u32 c = at < mid ? sort_cell(src, n, rowb + mid + m) : sort_cell(src, n, rowb + blk + m);
        if (ps_key_less(heap[c], heap[c + 1u], mh, ml)) lo = m + 1u; else hi = m;
      }
      const ps_u32 d = sort_cell(dst, n, rowb + blk + own + lo);
      heap[d] = mh;
      heap[d + 1u] = ml;
    }
  }

  __device__ ps_u32 pivot_flag(ps_u32 n, ps_u32 t) { (void)n; return temporary + t; }
  __device__ ps_u32 pivot_meta(ps_u32 n, ps_u32 row) { return temporary + n + row; }
  __device__ ps_u32 pivot_val(ps_u32 n, ps_u32 t) { return temporary + 2u * n + t; }
  __device__ ps_u32 pivot_pos(ps_u32 n, ps_u32 i) { return temporary + 3u * n + i; }

  __device__ void op_sort_pre(ps_u32 p) {
    if (p_tag(p) != 0x58u) return;
    if (p_pred_tag(p) != 1u) return;
    const ps_u32 a = p_a0(p);
    const ps_u32 n = d_len(a);
    const ps_u32 rows = rows_of(a);
    const ps_u32 len = ps_maxu(n / ps_maxu(rows, 1u), 1u);
    for (ps_u32 c = tid; c < n; c += lanes) {
      const ps_u32 rowb = (c / len) * len;
      const ps_u32 idx = heap[sort_cell(0u, n, c) + 1u];
      heap[pivot_val(n, c)] = ps_bits(ld_f(a, rowb + idx));
      heap[pivot_pos(n, rowb + idx)] = c - rowb;
    }
  }

  __device__ void op_pivot_pack(ps_u32 p) {
    if (p_tag(p) != 0x58u) return;
    if (p_pred_tag(p) != 1u) return;
    const ps_u32 a = p_a0(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(a);
    const ps_u32 rows = rows_of(a);
    const ps_u32 len = ps_maxu(n / ps_maxu(rows, 1u), 1u);
    const ps_u32 words = (n + 3u) / 4u;
    for (ps_u32 w = tid; w < words; w += lanes) {
      bool b[4];
      for (ps_u32 k = 0u; k < 4u; ++k) {
        const ps_u32 i = w * 4u + k;
        bool on = false;
        if (i < n) {
          const ps_u32 row = i / len;
          const ps_u32 t = heap[pivot_pos(n, i)];
          on = t < heap[pivot_meta(n, row)] && heap[pivot_flag(n, row * len + t)] != 0u;
        }
        b[k] = on;
      }
      st_b_word(o, w, n, b);
    }
  }

  __device__ ps_u32 count_key_below(ps_u32 n, ps_u32 rowb, ps_u32 len, ps_u32 kh) {
    ps_u32 lo = 0u;
    ps_u32 hi = len;
    while (lo < hi) {
      const ps_u32 m = lo + (hi - lo) / 2u;
      const ps_u32 c = sort_cell(0u, n, rowb + m);
      if (heap[c] < kh) lo = m + 1u; else hi = m;
    }
    return lo;
  }

  __device__ void op_sort_desc_finish(ps_u32 p) {
    const ps_u32 a = p_a0(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 o1 = p_o1(p);
    const ps_u32 n = d_len(a);
    for (ps_u32 t = tid; t < n; t += lanes) {
      const ps_u32 idx = heap[sort_cell(0u, n, t) + 1u];
      st_as_f(o, t, ld_f(a, idx));
      st_u(o1, t, idx);
    }
  }

  __device__ void op_top_k_finish(ps_u32 p) {
    const ps_u32 a = p_a0(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 o1 = p_o1(p);
    const ps_u32 n = d_len(a);
    const ps_u32 rows = rows_of(a);
    const ps_u32 len = n / ps_maxu(rows, 1u);
    const ps_u32 want = ps_minu(p_imm(p), len);
    const ps_u32 total = rows * want;
    for (ps_u32 j = tid; j < total; j += lanes) {
      const ps_u32 row = j / ps_maxu(want, 1u);
      const ps_u32 t = j - row * want;
      const ps_u32 idx = heap[sort_cell(0u, n, row * len + t) + 1u];
      st_as_f(o, j, ld_f(a, row * len + idx));
      st_u(o1, j, idx);
    }
  }

  __device__ void op_pivot_finish(ps_u32 p) {
    const ps_u32 a = p_a0(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 payload = p_pred_payload(p);
    const ps_u32 pred = p_pred_tag(p);
    const ps_u32 n = d_len(a);
    const ps_u32 rows = rows_of(a);
    const ps_u32 len = n / ps_maxu(rows, 1u);
    const ps_u32 np = d_len(payload);

    if (pred == 1u) {
      for (ps_u32 row = tid; row < rows; row += lanes) {
        if (len == 0u) continue;
        const ps_u32 rowb = row * len;
        const float cut = ld_f(payload, ps_pick(np, row));
        const ps_u32 vals = pivot_val(n, rowb);
        const ps_u32 flags = pivot_flag(n, rowb);
        const float last = ps_f(heap[vals + len - 1u]);
        const bool monotone = !(last < 0.0f) && last == last;
        float excl = 0.0f;
        ps_u32 t = 0u;
        for (;;) {
          if (t + PIVOT_UNROLL > len) break;
          float v[PIVOT_UNROLL];
#pragma unroll
          for (ps_u32 k = 0u; k < PIVOT_UNROLL; ++k) v[k] = ps_f(heap[vals + t + k]);
#pragma unroll
          for (ps_u32 k = 0u; k < PIVOT_UNROLL; ++k) {
            heap[flags + t + k] = excl < cut ? 1u : 0u;
            excl = excl + v[k];
          }
          t = t + PIVOT_UNROLL;
          if (!(excl < cut) && monotone) break;
        }
        for (;;) {
          if (t >= len) break;
          heap[flags + t] = excl < cut ? 1u : 0u;
          excl = excl + ps_f(heap[vals + t]);
          t = t + 1u;
          if (!(excl < cut) && monotone) break;
        }
        heap[pivot_meta(n, row)] = t;
      }
      return;
    }

    const ps_u32 words = (n + 3u) / 4u;
    for (ps_u32 w = tid; w < words; w += lanes) {
      bool b[4];
      for (ps_u32 k = 0u; k < 4u; ++k) {
        const ps_u32 i = w * 4u + k;
        bool on = false;
        if (i < n) {
          const ps_u32 row = i / ps_maxu(len, 1u);
          const ps_u32 rowb = row * len;
          const float x = ld_f(a, i);
          if (pred == 0u) {
            const int want = ps_mini(ps_maxi(ld_i(payload, ps_pick(np, row)), 0), (int)len);
            if (!(x != x)) on = (int)count_key_below(n, rowb, len, ps_sort_key(x)) < want;
          } else {
            on = x >= ld_f(payload, ps_pick(np, row));
          }
        }
        b[k] = on;
      }
      st_b_word(o, w, n, b);
    }
  }

  // ---- the one-row select ----

  __device__ ps_u32 sel_at(ps_u32 k) { return temporary - SEL_WORDS + k; }
  __device__ ps_u32 sel_part(ps_u32 g, ps_u32 b) { return sel_at(16u + g * 256u + b); }
  __device__ ps_u32 sel_count(ps_u32 g, ps_u32 which) {
    return sel_at(16u + SEL_GROUPS * 256u + g * 2u + which);
  }
  __device__ ps_u32 sel_cand_key(ps_u32 i) { return sel_at(16u + SEL_GROUPS * 258u + i); }
  __device__ ps_u32 sel_cand_idx(ps_u32 i) { return sel_at(16u + SEL_GROUPS * 258u + SEL_CAP + i); }

  // Whether this block sits the step out; uniform across it.
  __device__ bool sel_skips(bool only_first) {
    if (sel_lt == 0u) {
      const ps_u32 busy = heap[sel_at(SEL_ACTIVE)];
      const bool idle = only_first ? sel_group != 0u : sel_group >= busy;
      sel_wg_skip = (heap[sel_at(SEL_MODE)] != SEL_RUNS || idle) ? 1u : 0u;
    }
    __syncthreads();
    return sel_wg_skip != 0u;
  }

  __device__ uint2 sel_chunk(ps_u32 n) {
    const ps_u32 busy = heap[sel_at(SEL_ACTIVE)];
    const ps_u32 chunk = (n + busy - 1u) / busy;
    const ps_u32 lo = ps_minu(sel_group * chunk, n);
    return make_uint2(lo, ps_minu(lo + chunk, n));
  }

  __device__ bool sel_ladder_idle() { return heap[sel_at(SEL_MODE)] != SEL_LADDER; }

  __device__ void sel_init(ps_u32 p) {
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(p_a0(p));
    const ps_u32 words = (n + 3u) / 4u;
    for (ps_u32 w = tid; w < words; w += lanes) heap[base(o) + w] = 0u;
    if (tid == 0u) {
      heap[sel_at(SEL_MODE)] = SEL_RUNS;
      heap[sel_at(SEL_ACTIVE)] = ps_maxu(ps_minu(SEL_GROUPS, sel_groups), 1u);
      heap[sel_at(SEL_PREFIX)] = 0u;
      heap[sel_at(SEL_KEEP)] = SEL_KEEP_CUT;
    }
  }

  __device__ void sel_hist(ps_u32 p, ps_u32 sweep) {
    if (sel_skips(false)) return;
    const ps_u32 a = p_a0(p);
    const uint2 span = sel_chunk(d_len(a));
    const ps_u32 shift = 24u - 8u * sweep;
    const ps_u32 prefix = heap[sel_at(SEL_PREFIX)];
    sel_wg_hist[sel_lt] = 0u;
    __syncthreads();
    for (ps_u32 i = span.x + sel_lt; i < span.y; i += PTIR_WG) {
      const float x = ld_f(a, i);
      if (x != x) continue;
      const ps_u32 key = ps_sort_key(x);
      if (sweep == 0u || (key >> (shift + 8u)) == (prefix >> (shift + 8u)))
        atomicAdd(&sel_wg_hist[(key >> shift) & 255u], 1u);
    }
    __syncthreads();
    heap[sel_part(sel_group, sel_lt)] = sel_wg_hist[sel_lt];
  }

  __device__ void sel_pick(ps_u32 p, ps_u32 sweep) {
    if (sel_skips(true)) return;
    const ps_u32 busy = heap[sel_at(SEL_ACTIVE)];
    ps_u32 here = 0u;
    for (ps_u32 g = 0u; g < busy; ++g) here = here + heap[sel_part(g, sel_lt)];
    sel_wg_key[sel_lt] = here;
    __syncthreads();
    if (sel_lt != 0u) return;
    ps_u32 remaining = heap[sel_at(SEL_REMAINING)];
    if (sweep == 0u) {
      const ps_u32 n = d_len(p_a0(p));
      ps_u32 count = 0u;
      ps_u32 negative = 0u;
      for (ps_u32 b = 0u; b < 256u; ++b) {
        count = count + sel_wg_key[b];
        if (b >= 128u) negative = negative + sel_wg_key[b];
      }
      const ps_u32 payload = p_pred_payload(p);
      ps_u32 want = 0u;
      if (p_pred_tag(p) == 0u) {
        const int k = ld_i(payload, 0u);
        if (k <= 0) {
          heap[sel_at(SEL_KEEP)] = SEL_KEEP_NONE;
          heap[sel_at(SEL_MODE)] = SEL_DECIDED;
          return;
        }
        if ((ps_u32)k >= count) {
          heap[sel_at(SEL_KEEP)] = SEL_KEEP_ALL;
          heap[sel_at(SEL_MODE)] = SEL_DECIDED;
          return;
        }
        want = (ps_u32)k;
      } else {
        const float cut = ld_f(payload, 0u);
        if (!(0.0f < cut)) {
          heap[sel_at(SEL_MODE)] = SEL_DECIDED;
          return;
        }
        if (negative != 0u || count != n || count == 0u) {
          heap[sel_at(SEL_MODE)] = SEL_LADDER;
          return;
        }
        want = ps_minu(SEL_CAP, count);
      }
      heap[sel_at(SEL_WANT)] = want;
      remaining = want;
    }
    const ps_u32 shift = 24u - 8u * sweep;
    ps_u32 before = 0u;
    ps_u32 digit = 0u;
    for (;;) {
      if (digit + 1u >= 256u) break;
      const ps_u32 count = sel_wg_key[digit];
      if (before + count >= remaining) break;
      before = before + count;
      digit = digit + 1u;
    }
    heap[sel_at(SEL_PREFIX)] = heap[sel_at(SEL_PREFIX)] | (digit << shift);
    heap[sel_at(SEL_REMAINING)] = remaining - before;
    if (sweep == 3u) heap[sel_at(SEL_TOTAL_LT)] = heap[sel_at(SEL_WANT)] - (remaining - before);
  }

  __device__ void sel_rank_mask(ps_u32 p) {
    const ps_u32 a = p_a0(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 n = d_len(a);
    const ps_u32 keep = heap[sel_at(SEL_KEEP)];
    const ps_u32 cut = heap[sel_at(SEL_PREFIX)];
    const ps_u32 words = (n + 3u) / 4u;
    for (ps_u32 w = tid; w < words; w += lanes) {
      bool b[4];
      for (ps_u32 k = 0u; k < 4u; ++k) {
        const ps_u32 i = w * 4u + k;
        bool on = false;
        if (i < n && keep != SEL_KEEP_NONE) {
          const float x = ld_f(a, i);
          on = !(x != x) && (keep == SEL_KEEP_ALL || ps_sort_key(x) <= cut);
        }
        b[k] = on;
      }
      st_b_word(o, w, n, b);
    }
  }

  __device__ void sel_tally(ps_u32 p) {
    if (sel_skips(false)) return;
    const ps_u32 a = p_a0(p);
    const uint2 span = sel_chunk(d_len(a));
    const ps_u32 cut = heap[sel_at(SEL_PREFIX)];
    if (sel_lt < 2u) sel_wg_fill[sel_lt] = 0u;
    __syncthreads();
    for (ps_u32 i = span.x + sel_lt; i < span.y; i += PTIR_WG) {
      const ps_u32 key = ps_sort_key(ld_f(a, i));
      if (key < cut) atomicAdd(&sel_wg_fill[0], 1u);
      else if (key == cut) atomicAdd(&sel_wg_fill[1], 1u);
    }
    __syncthreads();
    if (sel_lt < 2u) heap[sel_count(sel_group, sel_lt)] = sel_wg_fill[sel_lt];
  }

  __device__ void sel_place(ps_u32 p) {
    if (sel_skips(false)) return;
    const ps_u32 a = p_a0(p);
    const uint2 span = sel_chunk(d_len(a));
    const ps_u32 cut = heap[sel_at(SEL_PREFIX)];
    const ps_u32 below = heap[sel_at(SEL_TOTAL_LT)];
    ps_u32 lt_at = 0u;
    ps_u32 eq_at = 0u;
    for (ps_u32 g = 0u; g < sel_group; ++g) {
      lt_at = lt_at + heap[sel_count(g, 0u)];
      eq_at = eq_at + heap[sel_count(g, 1u)];
    }
    if (sel_lt < 2u) sel_wg_fill[sel_lt] = 0u;
    __syncthreads();
    for (ps_u32 i = span.x + sel_lt; i < span.y; i += PTIR_WG) {
      const ps_u32 key = ps_sort_key(ld_f(a, i));
      if (key < cut) {
        const ps_u32 slot = lt_at + atomicAdd(&sel_wg_fill[0], 1u);
        heap[sel_cand_key(slot)] = key;
        heap[sel_cand_idx(slot)] = i;
      } else if (key == cut) {
        const ps_u32 slot = below + eq_at + atomicAdd(&sel_wg_fill[1], 1u);
        if (slot < SEL_CAP) {
          heap[sel_cand_key(slot)] = key;
          heap[sel_cand_idx(slot)] = i;
        }
      }
    }
  }

  __device__ void st_b_on(ps_u32 v, ps_u32 i) {
    const ps_u32 at = base(v) + i / 4u;
    heap[at] = heap[at] | (0xFFu << (8u * (i % 4u)));
  }

  __device__ void sel_walk(ps_u32 p) {
    if (sel_skips(true)) return;
    const ps_u32 a = p_a0(p);
    const ps_u32 o = p_o0(p);
    const ps_u32 busy = heap[sel_at(SEL_ACTIVE)];
    ps_u32 ties = 0u;
    for (ps_u32 g = 0u; g < busy; ++g) ties = ties + heap[sel_count(g, 1u)];
    const ps_u32 below = heap[sel_at(SEL_TOTAL_LT)];
    // Ties past the cap landed in no particular order, so the walk trusts
    // only the keys below the cut then.
    const ps_u32 held = below + ties <= SEL_CAP ? below + ties : below;
    for (ps_u32 i = sel_lt; i < SEL_CAP; i += PTIR_WG) {
      if (i < held) {
        sel_wg_key[i] = heap[sel_cand_key(i)];
        sel_wg_idx[i] = heap[sel_cand_idx(i)];
      } else {
        sel_wg_key[i] = 0xFFFFFFFFu;
        sel_wg_idx[i] = 0xFFFFFFFFu;
      }
    }
    __syncthreads();
    for (ps_u32 size = 2u; size <= SEL_CAP; size = size * 2u) {
      for (ps_u32 stride = size / 2u; stride > 0u; stride = stride / 2u) {
        for (ps_u32 t = sel_lt; t < SEL_CAP / 2u; t += PTIR_WG) {
          const ps_u32 lo = 2u * stride * (t / stride) + (t % stride);
          const ps_u32 hi = lo + stride;
          const bool ascending = (lo & size) == 0u;
          const ps_u32 klo = sel_wg_key[lo];
          const ps_u32 khi = sel_wg_key[hi];
          const ps_u32 ilo = sel_wg_idx[lo];
          const ps_u32 ihi = sel_wg_idx[hi];
          const bool lo_greater = klo > khi || (klo == khi && ilo > ihi);
          if (lo_greater == ascending) {
            sel_wg_key[lo] = khi;
            sel_wg_key[hi] = klo;
            sel_wg_idx[lo] = ihi;
            sel_wg_idx[hi] = ilo;
          }
        }
        __syncthreads();
      }
    }
    if (sel_lt != 0u) return;
    const float cut = ld_f(p_pred_payload(p), 0u);
    float excl = 0.0f;
    bool reached = false;
    for (ps_u32 t = 0u; t < held; ++t) {
      if (!(excl < cut)) {
        reached = true;
        break;
      }
      const ps_u32 index = sel_wg_idx[t];
      st_b_on(o, index);
      excl = excl + ld_f(a, index);
    }
    const bool whole = held == d_len(a);
    heap[sel_at(SEL_MODE)] = (reached || !(excl < cut) || whole) ? SEL_DECIDED : SEL_LADDER;
  }

  __device__ void op_matmul(ps_u32 p) {
    const ps_u32 a = p_a0(p);
    const ps_u32 b = p_a1(p);
    const ps_u32 o = p_o0(p);
    if (d_rank(a) != 2u || d_rank(b) != 2u) return;
    const ps_u32 m = d_dim(a, 0u);
    const ps_u32 kk = d_dim(a, 1u);
    const ps_u32 n = d_dim(b, 1u);
    const ps_u32 total = m * n;
    for (ps_u32 q = tid; q < total; q += lanes) {
      const ps_u32 i = q / ps_maxu(n, 1u);
      const ps_u32 j = q - i * n;
      float acc = 0.0f;
      for (ps_u32 l = 0u; l < kk; ++l) {
        const float xv = ld_f(a, i * kk + l);
        if (xv == 0.0f) continue;
        acc = acc + xv * ld_f(b, l * n + j);
      }
      st_as_f(o, q, acc);
    }
  }

  __device__ void op_finish(ps_u32 p, ps_u32 finish) {
    if (finish == 0u) op_sort_desc_finish(p);
    else if (finish == 1u) op_top_k_finish(p);
    else op_pivot_finish(p);
  }
};

// ---- the edges the engine runs here ----

__device__ __forceinline__ float ps_intrinsic_load(const ps_u8* input, ps_u64 index, ps_u32 mode) {
  if (mode == 0u) return reinterpret_cast<const float*>(input)[index];
  return __uint_as_float((ps_u32)reinterpret_cast<const ps_u16*>(input)[index] << 16);
}

__device__ __forceinline__ float ps_intrinsic_row_load(
    const ps_u8* input, ps_u64 row, ps_u32 column, ps_u32 stride, ps_u32 mode) {
  if (mode != 2u) return ps_intrinsic_load(input, row * (ps_u64)stride + column, mode);
  const ps_u64 row_address = reinterpret_cast<const ps_u64*>(input)[row];
  const ps_u16 value = reinterpret_cast<const ps_u16*>(row_address)[column];
  return __uint_as_float((ps_u32)value << 16);
}

// The arguments every call takes: the fire's lane table, the stage's tables,
// the lanes' scratch, their pending flags and output bindings, then the
// launch's shape and the step's uniform (node, rung, selector).
#define PS_ARGS                                                                \
  const PtirLaneTableHeader *header, const PtirLaneRecord *lane_records,       \
      const PtirLaneChannelSlot *channels, const ps_u32 *descs,                \
      const ps_u32 *params, const ps_u32 *offs, const PsCfg *cfg,              \
      ps_u8 *all_scratch, ps_u32 scratch_stride, ps_u8 *pending_flags,         \
      const ps_u64 *intrinsic_bases, const ps_u32 *intrinsic_modes,            \
      const ps_u32 *intrinsic_widths, const ps_u32 *intrinsic_strides,         \
      const ps_u32 *intrinsic_offsets, ps_u32 pitch, ps_u32 groups,            \
      ps_u32 node, ps_u32 rung, ps_u32 selector

// A block finds its lane and group, and runs only where its lane's fire was
// admitted (the commit word the pull set) and still stands.
#define PS_ENTER                                                               \
  const ps_u32 ps_lane = blockIdx.x / groups;                                  \
  const ps_u32 ps_group = blockIdx.x % groups;                                 \
  if (header == nullptr || ps_lane >= header->lane_count) return;              \
  const PtirLaneRecord& lane = lane_records[ps_lane];                          \
  ps_u32* commit = reinterpret_cast<ps_u32*>(lane.commit_slot);                \
  if (threadIdx.x == 0u) ps_go = commit == nullptr ? 0u : *commit;             \
  __syncthreads();                                                             \
  if (ps_go == 0u) return;                                                     \
  Ps c;                                                                        \
  c.descs = descs;                                                             \
  c.params = params;                                                           \
  c.offs = offs;                                                               \
  c.heap = reinterpret_cast<ps_u32*>(all_scratch +                             \
                                     (ps_u64)ps_lane * scratch_stride);        \
  c.temporary = cfg->temporary;                                                \
  c.tid = ps_group * PTIR_WG + threadIdx.x;                                    \
  c.lanes = groups * PTIR_WG;                                                  \
  c.sel_lt = threadIdx.x;                                                      \
  c.sel_group = ps_group;                                                      \
  c.sel_groups = groups;                                                       \
  (void)channels; (void)pending_flags; (void)intrinsic_bases;                  \
  (void)intrinsic_modes; (void)intrinsic_widths; (void)intrinsic_strides;      \
  (void)intrinsic_offsets; (void)pitch; (void)rung; (void)selector;            \
  (void)lane; (void)commit

#define PS_GUARD if (c.sel_ladder_idle()) return

// A channel read or take: the cell this fire reads (the pending one where
// this stage put it already) into the value's bytes.
__device__ __forceinline__ void ps_edge_read(
    Ps& c, const PtirLaneRecord& lane, const PtirLaneChannelSlot* channels,
    const ps_u8* pending_flags, ps_u32 p) {
  const ps_u32 o = c.p_o0(p);
  const ps_u32 index = lane.channel_slot_offset + c.p_channel(p);
  const PtirLaneChannelSlot slot = channels[index];
  const ps_u8* input = reinterpret_cast<const ps_u8*>(
      pending_flags[index] != 0u ? slot.pending_cell : slot.committed_cell);
  ps_u8* output = c.bytes(o);
  const ps_u32 bytes = c.d_dtype(o) == DT_BOOL ? c.d_len(o) : c.d_len(o) * 4u;
  for (ps_u32 at = c.tid; at < bytes; at += c.lanes) output[at] = input[at];
}

// A put: the value's bytes into the pending cell, the cell's bytes past it
// zeroed, a row the lane does not keep left as committed (or the sampled
// output's 0xff), and the channel marked pending for the stages after.
__device__ __forceinline__ void ps_edge_put(
    Ps& c, const PtirLaneRecord& lane, const PtirLaneChannelSlot* channels,
    ps_u8* pending_flags, ps_u32* commit, ps_u32 p) {
  const ps_u32 a = c.p_a0(p);
  const ps_u32 chan = c.p_channel(p);
  const ps_u32 index = lane.channel_slot_offset + chan;
  const PtirLaneChannelSlot slot = channels[index];
  const ps_u32 sink = c.p_sink_bytes(p);
  const ps_u32 len = c.d_len(a);
  const ps_u32 dtype = c.d_dtype(a);
  const ps_u32 logical = dtype == DT_BOOL ? len : len * 4u;
  if (logical > sink) {
    if (c.tid == 0u) *commit = 0u;
    return;
  }
  const ps_u8* input = c.bytes(a);
  ps_u8* output = reinterpret_cast<ps_u8*>(slot.pending_cell);
  const ps_u8* committed = reinterpret_cast<const ps_u8*>(slot.committed_cell);
  const ps_u8* row_valid = reinterpret_cast<const ps_u8*>(lane.row_valid);
  const bool lane_active = row_valid == nullptr || row_valid[lane.row_valid_offset] != 0u;
  const bool sample_output = (lane.sample_output_channel_mask & (1ull << chan)) != 0ull;
  const ps_u32 element_bytes = dtype == DT_BOOL ? 1u : 4u;
  ps_u32 per_row = 0u;
  if (lane.token_count != 0u) {
    if (c.d_rows(a) == lane.token_count) per_row = c.d_last(a);
    else if (len == lane.token_count) per_row = 1u;
  }
  for (ps_u32 byte = c.tid; byte < sink; byte += c.lanes) {
    if (byte >= logical) {
      output[byte] = 0u;
      continue;
    }
    bool row_active = lane_active;
    if (row_valid != nullptr && per_row != 0u) {
      const ps_u32 row = (byte / element_bytes) / per_row;
      if (row < lane.token_count) row_active = row_valid[lane.row_valid_offset + row] != 0u;
    }
    if (row_active) output[byte] = input[byte];
    else if (sample_output) output[byte] = 0xffu;
    else if (committed != nullptr) output[byte] = committed[byte];
    else *commit = 0u;
  }
  if (c.tid == 0u) pending_flags[index] = 1u;
}

// An output: the lane's bound rows of it (f32, bf16 or rows of bf16 by
// pointer; ids as stored) into the value.
__device__ __forceinline__ void ps_edge_output(
    Ps& c, ps_u32 lane_index, const ps_u64* intrinsic_bases, const ps_u32* intrinsic_modes,
    const ps_u32* intrinsic_widths, const ps_u32* intrinsic_strides,
    const ps_u32* intrinsic_offsets, ps_u32 pitch, ps_u32* commit, ps_u32 p) {
  const ps_u32 o = c.p_o0(p);
  const ps_u32 at = lane_index * pitch + c.p_intr(p);
  const ps_u8* input = reinterpret_cast<const ps_u8*>(intrinsic_bases[at]);
  const ps_u32 mode = intrinsic_modes[at];
  const ps_u32 last = c.d_last(o);
  const ps_u32 width = last == 0u ? intrinsic_widths[at] : last;
  const ps_u32 stride = intrinsic_strides[at] == 0u ? width : intrinsic_strides[at];
  const ps_u64 first_row = (ps_u64)intrinsic_offsets[at] + (ps_u64)c.p_imm2(p);
  const ps_u32 n = c.d_len(o);
  if (input == nullptr || width == 0u || stride < width) {
    if (c.tid == 0u) *commit = 0u;
    return;
  }
  if (c.d_dtype(o) == DT_I32) {
    if (mode != 3u) {
      if (c.tid == 0u) *commit = 0u;
      return;
    }
    for (ps_u32 i = c.tid; i < n; i += c.lanes)
      c.st_i(o, i, reinterpret_cast<const int*>(input)[(first_row + i / width) * stride + i % width]);
    return;
  }
  if (c.d_dtype(o) != DT_F32 || mode > 2u) {
    if (c.tid == 0u) *commit = 0u;
    return;
  }
  for (ps_u32 i = c.tid; i < n; i += c.lanes)
    c.st_f(o, i, ps_intrinsic_row_load(input, first_row + i / width, i % width, stride, mode));
}

// ---- the entry points ----
