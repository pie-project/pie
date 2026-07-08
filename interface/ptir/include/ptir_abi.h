// ptir_abi.h — GENERATED from pie-sampling-ir `src/ptir/{op,registry}.rs`.
// DO NOT EDIT. Regenerate: PTIR_REGEN=1 cargo test -p pie-sampling-ir ptir_header
// Container layout: interface/sampling-ir/PTIR-CONTAINER.md
#pragma once
#include <stdint.h>

#define PTIR_MAGIC "PTIR"
#define PTIR_VERSION 1
#define PTIB_MAGIC "PTIB" // bound-trace typed sidecar (PTIR-CONTAINER.md section 7)
#define PTIB_VERSION 1
// v1.1 extern channels (PTIR-CONTAINER.md section 6b): wire-version 2 iff externs
#define PTIR_VERSION_EXTERN 2
enum PtirExternDir : uint8_t { PTIR_EXTERN_IMPORT = 0, PTIR_EXTERN_EXPORT = 1 };

// ── op tags (X-macro: name, tag, value-operands, results; 0xFF = variadic) ──
#define PTIR_OP_LIST(X) \
  X(exp, 0x01, 1, 1) \
  X(log, 0x02, 1, 1) \
  X(neg, 0x03, 1, 1) \
  X(recip, 0x04, 1, 1) \
  X(abs, 0x05, 1, 1) \
  X(sign, 0x06, 1, 1) \
  X(cast, 0x07, 1, 1) \
  X(add, 0x10, 2, 1) \
  X(sub, 0x11, 2, 1) \
  X(mul, 0x12, 2, 1) \
  X(div, 0x13, 2, 1) \
  X(max_elem, 0x14, 2, 1) \
  X(min_elem, 0x15, 2, 1) \
  X(gt, 0x16, 2, 1) \
  X(ge, 0x17, 2, 1) \
  X(eq, 0x18, 2, 1) \
  X(ne, 0x19, 2, 1) \
  X(lt, 0x1A, 2, 1) \
  X(le, 0x1B, 2, 1) \
  X(and, 0x1C, 2, 1) \
  X(or, 0x1D, 2, 1) \
  X(not, 0x1E, 1, 1) \
  X(rem, 0x1F, 2, 1) \
  X(select, 0x20, 3, 1) \
  X(reduce_sum, 0x30, 1, 1) \
  X(reduce_max, 0x31, 1, 1) \
  X(reduce_min, 0x32, 1, 1) \
  X(reduce_argmax, 0x33, 1, 1) \
  X(broadcast, 0x38, 1, 1) \
  X(reshape, 0x39, 1, 1) \
  X(transpose, 0x3A, 1, 1) \
  X(cumsum, 0x40, 1, 1) \
  X(cumprod, 0x41, 1, 1) \
  X(sort_desc, 0x50, 1, 2) \
  X(top_k, 0x51, 1, 2) \
  X(matmul, 0x55, 2, 1) \
  X(pivot_threshold, 0x58, 2, 1) \
  X(gather, 0x60, 2, 1) \
  X(gather_row, 0x61, 2, 1) \
  X(scatter_add, 0x62, 3, 1) \
  X(scatter_set, 0x63, 3, 1) \
  X(iota, 0x64, 0, 1) \
  X(mask_apply_packed, 0x65, 2, 1) \
  X(rng, 0x70, 0, 1) \
  X(rng_keyed, 0x71, 1, 1) \
  X(const, 0x81, 0, 1) \
  X(chan_take, 0x90, 0, 1) \
  X(chan_read, 0x91, 0, 1) \
  X(chan_put, 0x92, 1, 0) \
  X(intrinsic_val, 0xA0, 0, 1) \
  X(kernel_call, 0xA1, 0xFF, 1) \
  X(sink_call, 0xA2, 0xFF, 0)

enum PtirOpTag : uint8_t {
  PTIR_OP_EXP = 0x01,
  PTIR_OP_LOG = 0x02,
  PTIR_OP_NEG = 0x03,
  PTIR_OP_RECIP = 0x04,
  PTIR_OP_ABS = 0x05,
  PTIR_OP_SIGN = 0x06,
  PTIR_OP_CAST = 0x07,
  PTIR_OP_ADD = 0x10,
  PTIR_OP_SUB = 0x11,
  PTIR_OP_MUL = 0x12,
  PTIR_OP_DIV = 0x13,
  PTIR_OP_MAX_ELEM = 0x14,
  PTIR_OP_MIN_ELEM = 0x15,
  PTIR_OP_GT = 0x16,
  PTIR_OP_GE = 0x17,
  PTIR_OP_EQ = 0x18,
  PTIR_OP_NE = 0x19,
  PTIR_OP_LT = 0x1A,
  PTIR_OP_LE = 0x1B,
  PTIR_OP_AND = 0x1C,
  PTIR_OP_OR = 0x1D,
  PTIR_OP_NOT = 0x1E,
  PTIR_OP_REM = 0x1F,
  PTIR_OP_SELECT = 0x20,
  PTIR_OP_REDUCE_SUM = 0x30,
  PTIR_OP_REDUCE_MAX = 0x31,
  PTIR_OP_REDUCE_MIN = 0x32,
  PTIR_OP_REDUCE_ARGMAX = 0x33,
  PTIR_OP_BROADCAST = 0x38,
  PTIR_OP_RESHAPE = 0x39,
  PTIR_OP_TRANSPOSE = 0x3A,
  PTIR_OP_CUMSUM = 0x40,
  PTIR_OP_CUMPROD = 0x41,
  PTIR_OP_SORT_DESC = 0x50,
  PTIR_OP_TOP_K = 0x51,
  PTIR_OP_MATMUL = 0x55,
  PTIR_OP_PIVOT_THRESHOLD = 0x58,
  PTIR_OP_GATHER = 0x60,
  PTIR_OP_GATHER_ROW = 0x61,
  PTIR_OP_SCATTER_ADD = 0x62,
  PTIR_OP_SCATTER_SET = 0x63,
  PTIR_OP_IOTA = 0x64,
  PTIR_OP_MASK_APPLY_PACKED = 0x65,
  PTIR_OP_RNG = 0x70,
  PTIR_OP_RNG_KEYED = 0x71,
  PTIR_OP_CONST = 0x81,
  PTIR_OP_CHAN_TAKE = 0x90,
  PTIR_OP_CHAN_READ = 0x91,
  PTIR_OP_CHAN_PUT = 0x92,
  PTIR_OP_INTRINSIC_VAL = 0xA0,
  PTIR_OP_KERNEL_CALL = 0xA1,
  PTIR_OP_SINK_CALL = 0xA2,
};

// ── dtypes (channel decls may also carry PTIR_DT_ACT = late-bound activation) ──
enum PtirDType : uint8_t {
  PTIR_DT_F32 = 0,
  PTIR_DT_I32 = 1,
  PTIR_DT_U32 = 2,
  PTIR_DT_BOOL = 3,
  PTIR_DT_ACT = 4,
};

// ── stages (per-layer taps: ON_ATTN_PROJ, ON_ATTN) ──
enum PtirStage : uint8_t {
  PTIR_STAGE_PROLOGUE = 0,
  PTIR_STAGE_ON_ATTN_PROJ = 1,
  PTIR_STAGE_ON_ATTN = 2,
  PTIR_STAGE_EPILOGUE = 3,
};
// readiness-table phase tag for the descriptor (not a program stage)
#define PTIR_PHASE_DESCRIPTOR 0xFF

// ── descriptor ports (token family CONSUMES, geometry/masks PEEK) ──
enum PtirPort : uint8_t {
  PTIR_PORT_EMBED_TOKENS = 0,
  PTIR_PORT_EMBED_INDPTR = 1,
  PTIR_PORT_POSITIONS = 2,
  PTIR_PORT_PAGES = 3,
  PTIR_PORT_PAGE_INDPTR = 4,
  PTIR_PORT_KV_LEN = 5,
  PTIR_PORT_W_SLOT = 6,
  PTIR_PORT_W_OFF = 7,
  PTIR_PORT_READOUT = 8,
  PTIR_PORT_ATTN_MASK = 9,
};

// ── first-party value intrinsics (op 0xA0 payload) ──
enum PtirIntrinsic : uint16_t {
  PTIR_INTR_LOGITS = 0,
  PTIR_INTR_MTP_LOGITS = 1,
  PTIR_INTR_HIDDEN = 2,
  PTIR_INTR_QUERY = 3,
  PTIR_INTR_VALUE_HEAD = 4,
  PTIR_INTR_LAYER = 5,
  PTIR_INTR_MTP_DRAFTS = 6,
};

// ── channel host roles / readiness direction / lowering classes ──
enum PtirHostRole : uint8_t { PTIR_HOST_NONE = 0, PTIR_HOST_WRITER = 1, PTIR_HOST_READER = 2 };
enum PtirDirection : uint8_t { PTIR_NEEDS_FULL = 0, PTIR_NEEDS_EMPTY = 1 };
enum PtirChannelClass : uint8_t { PTIR_CHAN_FULL_RING = 0, PTIR_CHAN_IN_PLACE = 1, PTIR_CHAN_IN_PLACE_UNDO = 2 };

// ── well-known first-party sink names (scope: 0 = pass-wide/prologue-only, 1 = attention) ──
#define PTIR_SINK_ATTN_PAGE_MASK "attn_page_mask" // scope 1
#define PTIR_SINK_LORA "lora" // scope 0
#define PTIR_SINK_MINFERENCE_SPARSE "minference_sparse" // scope 0

// ── numeric contract (T8 replay determinism; golden interp is normative) ──
// argmax: lower index wins ties; NaN never selected (all-NaN row -> index 0).
// sort_desc/top_k: descending, ties -> lower original index first; NaN sorts below -inf.
// rank_le(k): #strictly-greater < k (ties may admit > k elements at the boundary).
// cummass_le(p): inclusive nucleus (keep while exclusive prefix mass < p). prob_ge: >=.
// rng_keyed(state=[key,ctr]): seed64 = splitmix64((key<<32)|ctr); u(j) = hash_uniform(seed64, j)
//   with splitmix64/hash_uniform exactly as BYTECODE.md §5 / eval.rs; gumbel = -log(-log(u)).
