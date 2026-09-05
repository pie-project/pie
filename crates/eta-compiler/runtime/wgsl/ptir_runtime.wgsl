// PTIR device runtime, WGSL. The value store an emitted guest pass reads and
// writes, and one function per ETA op tag over it.
//
// The shape follows `runtime/cuda/ptir_m1_runtime_body.cuh` and answers what
// `eta_exec::op::eval_op` answers, because that host interpreter is the oracle
// a device pass is diffed against. Where the host's order is load-bearing it is
// reproduced here rather than improved on: `reduce_sum` folds in the 32-wide
// bracketing `canonical_reduce` uses, so the two agree bit for bit, while
// `reduce_argmax` is free to scan because it selects under a total order.
//
// ## The value store
//
// A value lives at a byte offset in one scratch allocation, 256-byte aligned
// (`eta_exec::scratch::layout`), so every value starts on a word boundary and
// the buffer is bound as `array<u32>` indexed by word. An `f32` or `i32` lane
// is one word read through `bitcast`; a `bool` lane is one BYTE, which is why
// every bool-producing op writes whole words at a time — WGSL has no
// byte-addressable store and a read-modify-write would race its neighbours.
//
// ## Parallelism
//
// **NOTHING HERE IS TIED TO ONE WORKGROUP.** No body declares `var<workgroup>`
// and none carries a barrier: every op reads and writes `heap`, and lanes never
// speak to each other except through it. A body stripes its work from `tid` by
// `lanes`, and the ENTRY POINT decides what those two mean — one workgroup's
// local id and `PTIR_WG` for a region sequenced by in-shader barriers, or the
// global id and the whole grid's width for a region sequenced by dispatch
// boundaries. The bodies do not know which they are in.
//
// Sequencing between ops is the entry point's business either way, because the
// only ordering WGSL offers inside a shader is a workgroup barrier, and that
// orders nothing between workgroups. A region that wants more than one
// workgroup therefore wants one dispatch per step, not a wider launch.
//
// `PTIR_WG` remains the workgroup SIZE — how many invocations a workgroup has,
// which is a property of the `@workgroup_size` attribute — and is no longer the
// stride any loop walks by.

const PTIR_WG : u32 = 256u;

// Descriptor words per value, matching `M1ValueDesc`.
const DESC_WORDS : u32 = 9u;
// Parameter words per op, matching `M1OpParams`.
const PARAM_WORDS : u32 = 16u;

// `eta_ir::types::WIRE_ORDER`.
const DT_F32 : u32 = 0u;
const DT_I32 : u32 = 1u;
const DT_U32 : u32 = 2u;
const DT_BOOL : u32 = 3u;

struct Cfg {
  // Values in the stage; the stride of one lane's descriptor block.
  value_count : u32,
  // Word offset of the scratch temporary region.
  temporary : u32,
  // Ops in this region.
  op_count : u32,
  // Lane this dispatch runs, for the descriptor and scratch strides.
  lane : u32,
}

@group(0) @binding(0) var<storage, read_write> status  : array<u32>;
@group(0) @binding(1) var<storage, read>       descs   : array<u32>;
@group(0) @binding(2) var<storage, read>       params  : array<u32>;
@group(0) @binding(3) var<storage, read>       offs    : array<u32>;
@group(0) @binding(4) var<storage, read_write> heap    : array<u32>;
@group(0) @binding(5) var<uniform>             cfg     : Cfg;

var<private> tid : u32;

/// Invocations the whole dispatch has, and so the stride every body walks its
/// work by. The entry point sets it: `PTIR_WG` when one workgroup runs the
/// region, `num_workgroups.x * PTIR_WG` when the grid does.
var<private> lanes : u32;

// ---- descriptors ---------------------------------------------------------

fn d_len(v : u32) -> u32 { return descs[v * DESC_WORDS + 0u]; }
fn d_rows(v : u32) -> u32 { return descs[v * DESC_WORDS + 1u]; }
fn d_rank(v : u32) -> u32 { return descs[v * DESC_WORDS + 3u]; }
fn d_dtype(v : u32) -> u32 { return descs[v * DESC_WORDS + 4u]; }
fn d_dim(v : u32, k : u32) -> u32 { return descs[v * DESC_WORDS + 5u + k]; }

/// Word index of a value's first lane.
fn base(v : u32) -> u32 { return offs[v] >> 2u; }

/// Rows a reduction folds over: everything but the trailing dimension, and
/// one row for a rank below two (`eta_exec::op::canonical_rows`).
fn rows_of(v : u32) -> u32 {
  let rank = d_rank(v);
  if (rank < 2u) { return 1u; }
  var n : u32 = 1u;
  for (var k : u32 = 0u; k + 1u < rank; k = k + 1u) { n = n * d_dim(v, k); }
  return n;
}

// ---- parameters ----------------------------------------------------------

fn p_at(p : u32, k : u32) -> u32 { return params[p * PARAM_WORDS + k]; }
fn p_tag(p : u32) -> u32 { return p_at(p, 0u); }
fn p_a0(p : u32) -> u32 { return p_at(p, 1u); }
fn p_a1(p : u32) -> u32 { return p_at(p, 2u); }
fn p_a2(p : u32) -> u32 { return p_at(p, 3u); }
fn p_o0(p : u32) -> u32 { return p_at(p, 4u); }
fn p_o1(p : u32) -> u32 { return p_at(p, 5u); }
fn p_imm(p : u32) -> u32 { return p_at(p, 6u); }
fn p_imm2(p : u32) -> u32 { return p_at(p, 7u); }
fn p_imm3(p : u32) -> u32 { return p_at(p, 8u); }
fn p_kind(p : u32) -> u32 { return p_at(p, 9u); }
fn p_pred_tag(p : u32) -> u32 { return p_at(p, 10u); }
fn p_lit_dtype(p : u32) -> u32 { return p_at(p, 11u); }
fn p_lit_bits(p : u32) -> u32 { return p_at(p, 12u); }
fn p_pred_payload(p : u32) -> u32 { return p_at(p, 13u); }
fn p_dtype(p : u32) -> u32 { return p_at(p, 14u); }

// ---- lanes ---------------------------------------------------------------
//
// A load answers in the asked-for class whatever the value holds, the way the
// host's `lanes_f32` / `lanes_i64` do. A `bool` lane is a byte inside a word.

fn ld_raw(v : u32, i : u32) -> u32 { return heap[base(v) + i]; }

fn ld_byte(v : u32, i : u32) -> u32 {
  let w = heap[base(v) + (i >> 2u)];
  return (w >> ((i & 3u) * 8u)) & 0xFFu;
}

fn ld_f(v : u32, i : u32) -> f32 {
  let dt = d_dtype(v);
  if (dt == DT_F32) { return bitcast<f32>(ld_raw(v, i)); }
  if (dt == DT_I32) { return f32(bitcast<i32>(ld_raw(v, i))); }
  if (dt == DT_U32) { return f32(ld_raw(v, i)); }
  return select(0.0, 1.0, ld_byte(v, i) != 0u);
}

fn ld_i(v : u32, i : u32) -> i32 {
  let dt = d_dtype(v);
  if (dt == DT_F32) { return i32(bitcast<f32>(ld_raw(v, i))); }
  if (dt == DT_I32) { return bitcast<i32>(ld_raw(v, i)); }
  if (dt == DT_U32) { return i32(ld_raw(v, i)); }
  return select(0, 1, ld_byte(v, i) != 0u);
}

fn ld_u(v : u32, i : u32) -> u32 {
  let dt = d_dtype(v);
  if (dt == DT_F32) { return u32(bitcast<f32>(ld_raw(v, i))); }
  if (dt == DT_I32) { return bitcast<u32>(bitcast<i32>(ld_raw(v, i))); }
  if (dt == DT_U32) { return ld_raw(v, i); }
  return select(0u, 1u, ld_byte(v, i) != 0u);
}

fn ld_b(v : u32, i : u32) -> bool {
  let dt = d_dtype(v);
  if (dt == DT_BOOL) { return ld_byte(v, i) != 0u; }
  return ld_raw(v, i) != 0u;
}

fn st_raw(v : u32, i : u32, w : u32) { heap[base(v) + i] = w; }
fn st_f(v : u32, i : u32, x : f32) { st_raw(v, i, bitcast<u32>(x)); }
fn st_i(v : u32, i : u32, x : i32) { st_raw(v, i, bitcast<u32>(x)); }
fn st_u(v : u32, i : u32, x : u32) { st_raw(v, i, x); }

/// Stores one WORD of a bool value — four lanes at once, so no two
/// invocations write the same word. `w` is the word index, `n` the lane count.
fn st_b_word(v : u32, w : u32, n : u32, b0 : bool, b1 : bool, b2 : bool, b3 : bool) {
  var packed : u32 = 0u;
  if (b0 && (w * 4u + 0u) < n) { packed = packed | 0x000000FFu; }
  if (b1 && (w * 4u + 1u) < n) { packed = packed | 0x0000FF00u; }
  if (b2 && (w * 4u + 2u) < n) { packed = packed | 0x00FF0000u; }
  if (b3 && (w * 4u + 3u) < n) { packed = packed | 0xFF000000u; }
  heap[base(v) + w] = packed;
}

/// A value written in the class its descriptor names, from an `f32` lane.
fn st_as_f(v : u32, i : u32, x : f32) {
  let dt = d_dtype(v);
  if (dt == DT_F32) { st_f(v, i, x); }
  else if (dt == DT_I32) { st_i(v, i, i32(x)); }
  else if (dt == DT_U32) { st_u(v, i, u32(x)); }
}

/// A value written from an `i32` lane, the host's `Value::from_i64` narrowing.
fn st_as_i(v : u32, i : u32, x : i32) {
  let dt = d_dtype(v);
  if (dt == DT_F32) { st_f(v, i, f32(x)); }
  else if (dt == DT_I32) { st_i(v, i, x); }
  else if (dt == DT_U32) { st_u(v, i, bitcast<u32>(x)); }
}

/// The host's `pick`: a one-lane operand broadcasts, anything else is indexed
/// straight (`eta_exec::value::pick`).
fn pick(n : u32, i : u32) -> u32 {
  if (n == 1u) { return 0u; }
  return i;
}

// ---- the map family ------------------------------------------------------
//
// One invocation per lane, striped. The output's own dtype decides the class
// written, so a `cast` is the same loop with a different destination class.

fn map_unary(p : u32, which : u32) {
  let a = p_a0(p);
  let o = p_o0(p);
  let n = d_len(o);
  let odt = d_dtype(o);
  if (odt == DT_BOOL) {
    let words = (n + 3u) / 4u;
    for (var w : u32 = tid; w < words; w = w + lanes) {
      var b : array<bool, 4>;
      for (var k : u32 = 0u; k < 4u; k = k + 1u) {
        let i = w * 4u + k;
        var v = false;
        if (i < n) { v = unary_b(which, a, pick(d_len(a), i)); }
        b[k] = v;
      }
      st_b_word(o, w, n, b[0], b[1], b[2], b[3]);
    }
    return;
  }
  for (var i : u32 = tid; i < n; i = i + lanes) {
    let j = pick(d_len(a), i);
    if (odt == DT_F32) { st_f(o, i, unary_f(which, a, j)); }
    else if (odt == DT_I32) { st_i(o, i, unary_i(which, a, j)); }
    else { st_u(o, i, unary_u(which, a, j)); }
  }
}

fn unary_f(which : u32, a : u32, j : u32) -> f32 {
  let x = ld_f(a, j);
  switch (which) {
    case 0u: { return exp(x); }
    case 1u: { return log(x); }
    case 2u: { return -x; }
    case 3u: { return 1.0 / x; }
    case 4u: { return abs(x); }
    case 5u: {
      if (x > 0.0) { return 1.0; }
      if (x < 0.0) { return -1.0; }
      return 0.0;
    }
    default: { return x; }
  }
}

fn unary_i(which : u32, a : u32, j : u32) -> i32 {
  // `exp`/`log`/`recip` on an integer value go through f32 the way the host's
  // `map_f32` does, then narrow.
  switch (which) {
    case 0u: { return i32(exp(ld_f(a, j))); }
    case 1u: { return i32(log(ld_f(a, j))); }
    case 3u: { return i32(1.0 / ld_f(a, j)); }
    case 2u: { return -ld_i(a, j); }
    case 4u: {
      let v = ld_i(a, j);
      if (v == -2147483647 - 1) { return v; }
      return abs(v);
    }
    case 5u: { return sign(ld_i(a, j)); }
    default: { return ld_i(a, j); }
  }
}

fn unary_u(which : u32, a : u32, j : u32) -> u32 {
  switch (which) {
    case 0u: { return u32(exp(ld_f(a, j))); }
    case 1u: { return u32(log(ld_f(a, j))); }
    case 3u: { return u32(1.0 / ld_f(a, j)); }
    case 2u: { return 0u - ld_u(a, j); }
    case 4u: { return ld_u(a, j); }
    case 5u: { return select(0u, 1u, ld_u(a, j) != 0u); }
    default: { return ld_u(a, j); }
  }
}

fn unary_b(which : u32, a : u32, j : u32) -> bool {
  // The only unary reaching a bool destination is `cast`, whose rule is
  // "nonzero", read through the f32 class as the host does.
  return ld_f(a, j) != 0.0;
}

// ---- the binary arithmetic family ---------------------------------------
//
// `bin_arith` computes in f32 when the FIRST OPERAND's dtype is f32 and in
// i64 otherwise, then narrows to the output. i64 is unavailable here, so an
// integer op runs in the operand's own signedness: for every value a lane can
// hold, 32-bit wrapping agrees with "widen, operate, truncate", and division
// is the one that would not, which is why it is split by class rather than
// done signed for both.

fn bin_arith(p : u32, which : u32) {
  let a = p_a0(p);
  let b = p_a1(p);
  let o = p_o0(p);
  let n = d_len(o);
  let adt = d_dtype(a);
  let na = d_len(a);
  let nb = d_len(b);
  if (adt == DT_F32) {
    for (var i : u32 = tid; i < n; i = i + lanes) {
      st_as_f(o, i, arith_f(which, ld_f(a, pick(na, i)), ld_f(b, pick(nb, i))));
    }
    return;
  }
  if (adt == DT_U32) {
    for (var i : u32 = tid; i < n; i = i + lanes) {
      let r = arith_u(which, ld_u(a, pick(na, i)), ld_u(b, pick(nb, i)));
      if (d_dtype(o) == DT_F32) { st_f(o, i, f32(r)); } else { st_u(o, i, r); }
    }
    return;
  }
  for (var i : u32 = tid; i < n; i = i + lanes) {
    st_as_i(o, i, arith_i(which, ld_i(a, pick(na, i)), ld_i(b, pick(nb, i))));
  }
}

fn arith_f(which : u32, x : f32, y : f32) -> f32 {
  switch (which) {
    case 0u: { return x + y; }
    case 1u: { return x - y; }
    case 2u: { return x * y; }
    case 3u: { return x / y; }
    case 4u: { return max(x, y); }
    case 5u: { return min(x, y); }
    default: { return x % y; }
  }
}

fn arith_i(which : u32, x : i32, y : i32) -> i32 {
  switch (which) {
    case 0u: { return x + y; }
    case 1u: { return x - y; }
    case 2u: { return x * y; }
    case 3u: { if (y == 0) { return 0; } return x / y; }
    case 4u: { return max(x, y); }
    case 5u: { return min(x, y); }
    default: { if (y == 0) { return 0; } return x % y; }
  }
}

fn arith_u(which : u32, x : u32, y : u32) -> u32 {
  switch (which) {
    case 0u: { return x + y; }
    case 1u: { return x - y; }
    case 2u: { return x * y; }
    case 3u: { if (y == 0u) { return 0u; } return x / y; }
    case 4u: { return max(x, y); }
    case 5u: { return min(x, y); }
    default: { if (y == 0u) { return 0u; } return x % y; }
  }
}

// ---- comparisons and logic ----------------------------------------------

fn cmp_op(p : u32, which : u32) {
  let a = p_a0(p);
  let b = p_a1(p);
  let o = p_o0(p);
  let n = d_len(o);
  let words = (n + 3u) / 4u;
  for (var w : u32 = tid; w < words; w = w + lanes) {
    var bits : array<bool, 4>;
    for (var k : u32 = 0u; k < 4u; k = k + 1u) {
      let i = w * 4u + k;
      var v = false;
      if (i < n) { v = cmp_lane(which, a, b, i); }
      bits[k] = v;
    }
    st_b_word(o, w, n, bits[0], bits[1], bits[2], bits[3]);
  }
}

fn cmp_lane(which : u32, a : u32, b : u32, i : u32) -> bool {
  let ia = pick(d_len(a), i);
  let ib = pick(d_len(b), i);
  let adt = d_dtype(a);
  if (adt == DT_F32) {
    let x = ld_f(a, ia);
    let y = ld_f(b, ib);
    switch (which) {
      case 0u: { return x > y; }
      case 1u: { return x >= y; }
      case 2u: { return x == y; }
      case 3u: { return x != y; }
      case 4u: { return x < y; }
      default: { return x <= y; }
    }
  }
  if (adt == DT_U32) {
    let x = ld_u(a, ia);
    let y = ld_u(b, ib);
    switch (which) {
      case 0u: { return x > y; }
      case 1u: { return x >= y; }
      case 2u: { return x == y; }
      case 3u: { return x != y; }
      case 4u: { return x < y; }
      default: { return x <= y; }
    }
  }
  let x = ld_i(a, ia);
  let y = ld_i(b, ib);
  switch (which) {
    case 0u: { return x > y; }
    case 1u: { return x >= y; }
    case 2u: { return x == y; }
    case 3u: { return x != y; }
    case 4u: { return x < y; }
    default: { return x <= y; }
  }
}

/// `and` / `or` / `not`: bool in, bool out.
fn logic_op(p : u32, which : u32) {
  let a = p_a0(p);
  let b = p_a1(p);
  let o = p_o0(p);
  let n = d_len(o);
  let words = (n + 3u) / 4u;
  for (var w : u32 = tid; w < words; w = w + lanes) {
    var bits : array<bool, 4>;
    for (var k : u32 = 0u; k < 4u; k = k + 1u) {
      let i = w * 4u + k;
      var v = false;
      if (i < n) {
        let x = ld_b(a, pick(d_len(a), i));
        if (which == 2u) { v = !x; }
        else {
          let y = ld_b(b, pick(d_len(b), i));
          if (which == 0u) { v = x && y; } else { v = x || y; }
        }
      }
      bits[k] = v;
    }
    st_b_word(o, w, n, bits[0], bits[1], bits[2], bits[3]);
  }
}

// ---- select --------------------------------------------------------------

fn op_select(p : u32) {
  let c = p_a0(p);
  let x = p_a1(p);
  let y = p_a2(p);
  let o = p_o0(p);
  let n = d_len(o);
  let odt = d_dtype(o);
  if (odt == DT_BOOL) {
    let words = (n + 3u) / 4u;
    for (var w : u32 = tid; w < words; w = w + lanes) {
      var bits : array<bool, 4>;
      for (var k : u32 = 0u; k < 4u; k = k + 1u) {
        let i = w * 4u + k;
        var v = false;
        if (i < n) {
          if (ld_b(c, pick(d_len(c), i))) { v = ld_b(x, pick(d_len(x), i)); }
          else { v = ld_b(y, pick(d_len(y), i)); }
        }
        bits[k] = v;
      }
      st_b_word(o, w, n, bits[0], bits[1], bits[2], bits[3]);
    }
    return;
  }
  for (var i : u32 = tid; i < n; i = i + lanes) {
    let take = ld_b(c, pick(d_len(c), i));
    if (odt == DT_F32) {
      var v = ld_f(y, pick(d_len(y), i));
      if (take) { v = ld_f(x, pick(d_len(x), i)); }
      st_f(o, i, v);
    } else if (odt == DT_I32) {
      var v = ld_i(y, pick(d_len(y), i));
      if (take) { v = ld_i(x, pick(d_len(x), i)); }
      st_i(o, i, v);
    } else {
      var v = ld_u(y, pick(d_len(y), i));
      if (take) { v = ld_u(x, pick(d_len(x), i)); }
      st_u(o, i, v);
    }
  }
}

// ---- reductions ----------------------------------------------------------
//
// `canonical_reduce` folds a row in chunks of 32 with the ladder 16/8/4/2/1,
// then folds the chunk results the same way, until one is left. A sum in any
// other bracketing is a different float, so that shape is reproduced rather
// than replaced. Each level is its own dispatch step, because the level after
// reads what the level before wrote and the barrier between them has to sit
// in uniform control flow — `execute` emits `REDUCE_LEVELS` calls and the
// barriers between them, and a level past the row's height does nothing.
//
// 32^7 exceeds any row a value can hold, so seven levels always finish.

const REDUCE_LEVELS : u32 = 7u;

// Which fold a reduce level is running.
const RED_SUM : u32 = 0u;
const RED_MAX : u32 = 1u;
const RED_MIN : u32 = 2u;
const RED_ARGMAX : u32 = 3u;

fn POS_INF() -> f32 { return bitcast<f32>(0x7F800000u); }
fn NEG_INF() -> f32 { return bitcast<f32>(0xFF800000u); }

/// `eta_exec::op::canonical_max`.
fn cmax(l : f32, r : f32) -> f32 {
  let ln = l != l;
  let rn = r != r;
  if (ln && rn) { return NEG_INF(); }
  if (ln) { return r; }
  if (rn) { return l; }
  if (l == 0.0 && r == 0.0) {
    if (sign_bit(l) && sign_bit(r)) { return -0.0; }
    return 0.0;
  }
  if (l > r) { return l; }
  return r;
}

/// `eta_exec::op::canonical_min`.
fn cmin(l : f32, r : f32) -> f32 {
  let ln = l != l;
  let rn = r != r;
  if (ln && rn) { return POS_INF(); }
  if (ln) { return r; }
  if (rn) { return l; }
  if (l == 0.0 && r == 0.0) {
    if (sign_bit(l) || sign_bit(r)) { return -0.0; }
    return 0.0;
  }
  if (l < r) { return l; }
  return r;
}

fn sign_bit(x : f32) -> bool { return (bitcast<u32>(x) & 0x80000000u) != 0u; }

/// What a padding lane holds. **`RED_ARGMAX` IS A MAX**, so its identity is
/// negative infinity like `RED_MAX`'s and not positive infinity like
/// `RED_MIN`'s — a fall-through that gave it `RED_MIN`'s identity handed every
/// padded lane a value that beats every real one.
fn red_identity(kind : u32) -> f32 {
  if (kind == RED_SUM) { return 0.0; }
  if (kind == RED_MIN) { return POS_INF(); }
  return NEG_INF();
}

fn red_combine(kind : u32, l : f32, r : f32) -> f32 {
  if (kind == RED_SUM) { return l + r; }
  if (kind == RED_MAX) { return cmax(l, r); }
  return cmin(l, r);
}

/// How many rungs a reduce ladder has. `32^7` is past `u32::MAX`, so seven
/// folds any row a descriptor can name. The emitter spells exactly this many
/// calls, and `the_runtime_and_the_emitter_agree_on_the_ladders_height` reads
/// this line back to keep the two numbers one number.
const PTIR_REDUCE_LEVELS : u32 = 7u;

/// Word stride of one row inside the scratch temporary, in the two-word
/// (value, index) cells a level writes.
fn red_stride(len : u32) -> u32 { return (len + 31u) / 32u; }

/// **THE LADDER PING-PONGS, AND IT HAS TO.**
///
/// A level's jobs run concurrently across the workgroup: job `c` writes cell
/// `c` while job `0` is still reading cells `0..31`, which includes cell `c`.
/// One buffer would therefore have a level racing itself, and the race is not
/// theoretical — it is why a fold over more than 32 * 32 lanes answered lane
/// zero. So a level READS one half of the temporary and WRITES the other, the
/// same shape the sort ladder's buffers already have.
///
/// `span` is one buffer's cell count, `rows * stride`. The scratch reserves
/// four words per element and the two buffers together come to an eighth of
/// one, so this costs nothing anybody has to budget for.
fn tmp_v(row : u32, stride : u32, c : u32, buf : u32, span : u32) -> u32 {
  return cfg.temporary + (buf * span + row * stride + c) * 2u;
}

/// Which half the level BEFORE `level` wrote, and so which half `level` reads.
fn red_read_buf(level : u32) -> u32 { return level % 2u; }

/// Which half `level` writes.
fn red_write_buf(level : u32) -> u32 { return (level + 1u) % 2u; }

/// One level of the fold. `level` 0 reads the source value, later levels read
/// what the level before wrote. Does nothing once the row is down to one cell.
fn op_reduce_level(p : u32, kind : u32, level : u32) {
  let a = p_a0(p);
  let rows = rows_of(a);
  let len = d_len(a) / max(rows, 1u);
  let stride = red_stride(len);
  var n_in = len;
  for (var l : u32 = 0u; l < level; l = l + 1u) { n_in = (n_in + 31u) / 32u; }
  if (n_in <= 1u) { return; }
  let n_out = (n_in + 31u) / 32u;
  let total = rows * n_out;
  for (var job : u32 = tid; job < total; job = job + lanes) {
    let row = job / n_out;
    let c = job % n_out;
    var vs : array<f32, 32>;
    var ix : array<u32, 32>;
    let count = min(32u, n_in - c * 32u);
    for (var k : u32 = 0u; k < 32u; k = k + 1u) {
      vs[k] = red_identity(kind);
      ix[k] = 0u;
      if (k < count) {
        let at = c * 32u + k;
        if (level == 0u) {
          vs[k] = ld_f(a, row * len + at);
          ix[k] = at;
        } else {
          let cell = tmp_v(row, stride, at, red_read_buf(level), rows * stride);
          vs[k] = bitcast<f32>(heap[cell]);
          ix[k] = heap[cell + 1u];
        }
      }
    }
    // The ladder, in the host's order.
    for (var off : u32 = 16u; off >= 1u; off = off / 2u) {
      for (var lane : u32 = 0u; lane < off; lane = lane + 1u) {
        if (kind == RED_ARGMAX) {
          let takes = arg_takes(vs[lane], ix[lane], vs[lane + off], ix[lane + off], level, count, lane, off);
          if (takes) { vs[lane] = vs[lane + off]; ix[lane] = ix[lane + off]; }
        } else {
          vs[lane] = red_combine(kind, vs[lane], vs[lane + off]);
        }
      }
      if (off == 1u) { break; }
    }
    let cell = tmp_v(row, stride, c, red_write_buf(level), rows * stride);
    heap[cell] = bitcast<u32>(vs[0]);
    heap[cell + 1u] = ix[0];
  }
}

/// Argmax's selection: skip NaN, and take the right only when it is strictly
/// greater, so the earliest index wins a tie. A candidate that never held a
/// non-NaN lane loses to one that did.
///
/// **THE PADDING GUARD IS NOT A LEVEL-ZERO GUARD.** A chunk holds `count` real
/// lanes and up to `32 - count` padded ones, and that is true at every rung of
/// the ladder, not only the first: a row of 32000 folds to 1000 and then to
/// 32, and the last of those 32 chunks carries eight real lanes and
/// twenty-four padded. Reading a padded lane as a candidate is how a fold over
/// more than 1024 lanes came back as index zero.
fn arg_takes(lv : f32, li : u32, rv : f32, ri : u32, level : u32, count : u32, lane : u32, off : u32) -> bool {
  if (lane + off >= count) { return false; }
  let lnan = lv != lv;
  let rnan = rv != rv;
  if (rnan) { return false; }
  if (lnan) { return true; }
  return rv > lv;
}

/// Writes the folded row out. `reduce_argmax` answers the index as `i32`.
fn op_reduce_finish(p : u32, kind : u32) {
  let a = p_a0(p);
  let o = p_o0(p);
  let rows = rows_of(a);
  let len = d_len(a) / max(rows, 1u);
  let stride = red_stride(len);
  for (var row : u32 = tid; row < rows; row = row + lanes) {
    var v : f32;
    var idx : u32;
    if (len == 0u) {
      v = red_identity(kind);
      idx = 0u;
    } else if (len == 1u) {
      // `canonical_reduce_by` returns a single element untouched rather than
      // combining it with the identity.
      v = ld_f(a, row * len);
      idx = 0u;
    } else {
      // **WHICH HALF THE LAST RUNG LEFT ITS ANSWER IN.** A level does
      // nothing once the row is down to one cell, so the ladder stops early
      // and which buffer holds the answer depends on how many rungs it
      // actually climbed. Walked here with the same recurrence the levels
      // use, rather than assumed.
      var folded : u32 = len;
      var last : u32 = 0u;
      for (var l : u32 = 0u; l < PTIR_REDUCE_LEVELS; l = l + 1u) {
        if (folded <= 1u) { break; }
        last = l;
        folded = (folded + 31u) / 32u;
      }
      let cell = tmp_v(row, stride, 0u, red_write_buf(last), rows * stride);
      v = bitcast<f32>(heap[cell]);
      idx = heap[cell + 1u];
    }
    if (kind == RED_ARGMAX) { st_i(o, row, bitcast<i32>(idx)); }
    else { st_as_f(o, row, v); }
  }
}

/// `cumsum` and `cumprod`: a running total per canonical row, one invocation
/// walking the row in ascending order.
///
/// The walk is the point. The host's is `acc = acc OP x[j]` down the row, so a
/// parallel scan would reassociate into different floats — the same reason
/// `op_matmul` sums its contraction in order and `op_pivot_finish`'s top-`p`
/// arm walks rather than folds. Parallelism across rows is kept, since rows are
/// independent and write disjoint words; within a row it is given up to keep
/// the bits.
///
/// Two details of the host's loop are load-bearing and easy to lose. The
/// accumulator starts at the IDENTITY and is combined with the first element
/// rather than seeded from it, so `cumsum` of a leading `-0.0` answers `+0.0`
/// (`0.0 + -0.0`) while `cumprod` answers `-0.0` (`1.0 * -0.0`). And the
/// accumulator RESETS per row, which is what makes `rows_of` the row count here
/// and not something a caller passes.
///
/// `len` is `d_len(a) / rows`, so a row count that does not divide the value
/// leaves the tail lanes untouched — exactly the `rows * len` elements the
/// host's `Vec` holds.
fn op_cumulative(p : u32, is_prod : bool) {
  let a = p_a0(p);
  let o = p_o0(p);
  let rows = rows_of(a);
  let len = d_len(a) / max(rows, 1u);
  for (var row : u32 = tid; row < rows; row = row + lanes) {
    let rowb = row * len;
    var acc : f32 = select(0.0, 1.0, is_prod);
    for (var j : u32 = 0u; j < len; j = j + 1u) {
      let x = ld_f(a, rowb + j);
      if (is_prod) { acc = acc * x; } else { acc = acc + x; }
      st_as_f(o, rowb + j, acc);
    }
  }
}

// ---- shape ---------------------------------------------------------------

/// `broadcast` and `reshape`: the lanes are the source's, repeated when the
/// source is one lane, which is the host's `broadcast_value` for the shapes a
/// pass builds.
fn op_copy(p : u32) {
  let a = p_a0(p);
  let o = p_o0(p);
  let n = d_len(o);
  let na = d_len(a);
  let odt = d_dtype(o);
  if (odt == DT_BOOL) {
    let words = (n + 3u) / 4u;
    for (var w : u32 = tid; w < words; w = w + lanes) {
      var b : array<bool, 4>;
      for (var k : u32 = 0u; k < 4u; k = k + 1u) {
        let i = w * 4u + k;
        var v = false;
        if (i < n) { v = ld_b(a, pick(na, i)); }
        b[k] = v;
      }
      st_b_word(o, w, n, b[0], b[1], b[2], b[3]);
    }
    return;
  }
  for (var i : u32 = tid; i < n; i = i + lanes) {
    let j = pick(na, i);
    if (odt == DT_F32) { st_f(o, i, ld_f(a, j)); }
    else if (odt == DT_I32) { st_i(o, i, ld_i(a, j)); }
    else { st_u(o, i, ld_u(a, j)); }
  }
}

/// `transpose` of a rank-2 value.
fn op_transpose(p : u32) {
  let a = p_a0(p);
  let o = p_o0(p);
  let m = d_dim(a, 0u);
  let n2 = d_dim(a, 1u);
  let n = m * n2;
  let odt = d_dtype(o);
  for (var at : u32 = tid; at < n; at = at + lanes) {
    let src = (at % m) * n2 + at / m;
    if (odt == DT_F32) { st_f(o, at, ld_f(a, src)); }
    else if (odt == DT_I32) { st_i(o, at, ld_i(a, src)); }
    else if (odt == DT_U32) { st_u(o, at, ld_u(a, src)); }
  }
}

// ---- index ---------------------------------------------------------------

/// `gather`: one ROW of the source per index — `rest` is the product of the
/// dimensions after the first, so a rank-1 source gathers single lanes and a
/// rank-2 one gathers whole rows. An out-of-range index answers zero, which is
/// the host's `None` slot in `gather_flat`.
fn op_gather(p : u32) {
  let a = p_a0(p);
  let ix = p_a1(p);
  let o = p_o0(p);
  let n = d_len(o);
  let rest = gather_rest(a);
  let n0 = gather_first(a);
  let odt = d_dtype(o);
  for (var at : u32 = tid; at < n; at = at + lanes) {
    let slot = at / rest;
    let within = at % rest;
    let want = ld_i(ix, pick(d_len(ix), slot));
    let ok = want >= 0 && u32(want) < n0;
    let j = select(0u, u32(want) * rest + within, ok);
    if (odt == DT_F32) { st_f(o, at, select(0.0, ld_f(a, j), ok)); }
    else if (odt == DT_I32) { st_i(o, at, select(0, ld_i(a, j), ok)); }
    else if (odt == DT_U32) { st_u(o, at, select(0u, ld_u(a, j), ok)); }
  }
}

/// Lanes per index slot: the source's dimensions after the first.
fn gather_rest(a : u32) -> u32 {
  let rank = d_rank(a);
  if (rank < 2u) { return 1u; }
  var r : u32 = 1u;
  for (var k : u32 = 1u; k < rank; k = k + 1u) { r = r * d_dim(a, k); }
  return max(r, 1u);
}

fn gather_first(a : u32) -> u32 {
  if (d_rank(a) == 0u) { return 1u; }
  return d_dim(a, 0u);
}

/// `gather_row`: one lane per row of a rank-2 source, the column named by the
/// index. An out-of-range column answers zero.
fn op_gather_row(p : u32) {
  let a = p_a0(p);
  let ix = p_a1(p);
  let o = p_o0(p);
  let m = d_dim(a, 0u);
  let n2 = d_dim(a, 1u);
  let odt = d_dtype(o);
  for (var i : u32 = tid; i < m; i = i + lanes) {
    let c = ld_i(ix, pick(d_len(ix), i));
    let ok = c >= 0 && u32(c) < n2;
    let j = select(0u, i * n2 + u32(c), ok);
    if (odt == DT_F32) { st_f(o, i, select(0.0, ld_f(a, j), ok)); }
    else if (odt == DT_I32) { st_i(o, i, select(0, ld_i(a, j), ok)); }
    else if (odt == DT_U32) { st_u(o, i, select(0u, ld_u(a, j), ok)); }
  }
}

/// `scatter_set` / `scatter_add`: the source copied through, then the named
/// slots overwritten or accumulated. One invocation owns one destination slot
/// so two indices naming it are applied in index order, as the host applies
/// them.
fn op_scatter(p : u32, is_add : bool) {
  let a = p_a0(p);
  let ix = p_a1(p);
  let src = p_a2(p);
  let o = p_o0(p);
  let n = d_len(o);
  let rest = gather_rest(a);
  let n0 = gather_first(a);
  let slots = d_len(ix);
  let scalar = d_len(src) == 1u && slots * rest != 1u;
  let odt = d_dtype(o);
  for (var at : u32 = tid; at < n; at = at + lanes) {
    var acc_f = ld_f(a, at);
    var acc_i = ld_i(a, at);
    let dest = at / rest;
    let within = at % rest;
    for (var k : u32 = 0u; k < slots; k = k + 1u) {
      let want = ld_i(ix, k);
      if (want < 0 || u32(want) >= n0) { continue; }
      if (u32(want) != dest) { continue; }
      var lane = k * rest + within;
      if (scalar) { lane = 0u; }
      if (odt == DT_F32) {
        let v = ld_f(src, lane);
        if (is_add) { acc_f = acc_f + v; } else { acc_f = v; }
      } else {
        let v = ld_i(src, lane);
        if (is_add) { acc_i = acc_i + v; } else { acc_i = v; }
      }
    }
    if (odt == DT_F32) { st_f(o, at, acc_f); } else { st_as_i(o, at, acc_i); }
  }
}

/// `iota`: the lane's own index.
fn op_iota(p : u32) {
  let o = p_o0(p);
  let n = d_len(o);
  let odt = d_dtype(o);
  for (var i : u32 = tid; i < n; i = i + lanes) {
    if (odt == DT_F32) { st_f(o, i, f32(i)); }
    else if (odt == DT_I32) { st_i(o, i, i32(i)); }
    else { st_u(o, i, i); }
  }
}

/// `const`: one literal in every lane.
fn op_const(p : u32) {
  let o = p_o0(p);
  let n = d_len(o);
  let dt = p_lit_dtype(p);
  let bits = p_lit_bits(p);
  if (d_dtype(o) == DT_BOOL) {
    let words = (n + 3u) / 4u;
    let on = bits != 0u;
    for (var w : u32 = tid; w < words; w = w + lanes) {
      st_b_word(o, w, n, on, on, on, on);
    }
    return;
  }
  for (var i : u32 = tid; i < n; i = i + lanes) {
    if (dt == DT_F32) { st_as_f(o, i, bitcast<f32>(bits)); }
    else { st_as_i(o, i, bitcast<i32>(bits)); }
  }
}

// ---- masks ---------------------------------------------------------------

/// `mask_apply_packed`: a bit per key in a `u32` plane, a cleared bit sending
/// the lane to negative infinity.
fn op_mask_apply(p : u32) {
  let a = p_a0(p);
  let m = p_a1(p);
  let o = p_o0(p);
  let n = d_len(o);
  let rank = d_rank(a);
  var width : u32 = 1u;
  if (rank > 0u) { width = d_dim(a, rank - 1u); }
  let mw = d_len(m);
  for (var j : u32 = tid; j < n; j = j + lanes) {
    let c = j % max(width, 1u);
    let w = c >> 5u;
    var word : u32 = 0u;
    if (w < mw) { word = ld_u(m, w); }
    if (((word >> (c & 31u)) & 1u) != 0u) { st_f(o, j, ld_f(a, j)); }
    else { st_f(o, j, NEG_INF()); }
  }
}

/// `causal_mask` / `sliding_window_mask` / `sink_window_mask`: one bool per
/// (position, key). `which` is 0 causal, 1 sliding, 2 sink.
fn op_struct_mask(p : u32, which : u32) {
  let pos = p_a0(p);
  let o = p_o0(p);
  let keys = p_imm(p);
  var window = p_imm3(p);
  if (which == 1u) { window = p_imm2(p); }
  let sink = p_imm2(p);
  let n = d_len(o);
  let words = (n + 3u) / 4u;
  for (var w : u32 = tid; w < words; w = w + lanes) {
    var bits : array<bool, 4>;
    for (var k : u32 = 0u; k < 4u; k = k + 1u) {
      let at = w * 4u + k;
      var v = false;
      if (at < n && keys != 0u) {
        let position = ld_u(pos, at / keys);
        let key = at % keys;
        var allowed = key <= position;
        if (allowed && which != 0u) {
          // `saturating_add`, then a strict compare.
          var sum = key + window;
          if (sum < key) { sum = 0xFFFFFFFFu; }
          let recent = sum > position;
          if (which == 1u) { allowed = recent; }
          else { allowed = key < sink || recent; }
        }
        v = allowed;
      }
      bits[k] = v;
    }
    st_b_word(o, w, n, bits[0], bits[1], bits[2], bits[3]);
  }
}

// ---- the rng -------------------------------------------------------------
//
// `eta_ir::rng::RNG_FORMULA` in 32-bit halves, because WGSL has no 64-bit
// integer. The constants are ABI: a backend that reproduces the ops but not
// these numbers samples a different token from the same seed.

struct U64 { lo : u32, hi : u32 }

fn u64_mul32(a : u32, b : u32) -> U64 {
  let al = a & 0xFFFFu;
  let ah = a >> 16u;
  let bl = b & 0xFFFFu;
  let bh = b >> 16u;
  let p0 = al * bl;
  let p1 = al * bh;
  let p2 = ah * bl;
  let p3 = ah * bh;
  let mid = (p0 >> 16u) + (p1 & 0xFFFFu) + (p2 & 0xFFFFu);
  let lo = (p0 & 0xFFFFu) | (mid << 16u);
  let hi = p3 + (p1 >> 16u) + (p2 >> 16u) + (mid >> 16u);
  return U64(lo, hi);
}

fn u64_mul(a : U64, b : U64) -> U64 {
  let low = u64_mul32(a.lo, b.lo);
  return U64(low.lo, low.hi + a.lo * b.hi + a.hi * b.lo);
}

fn u64_xor(a : U64, b : U64) -> U64 { return U64(a.lo ^ b.lo, a.hi ^ b.hi); }

fn u64_add(a : U64, b : U64) -> U64 {
  let lo = a.lo + b.lo;
  let carry = select(0u, 1u, lo < a.lo);
  return U64(lo, a.hi + b.hi + carry);
}

/// `x >> n` for `n` in 0..63.
fn u64_shr(x : U64, n : u32) -> U64 {
  if (n == 0u) { return x; }
  if (n < 32u) { return U64((x.lo >> n) | (x.hi << (32u - n)), x.hi >> n); }
  return U64(x.hi >> (n - 32u), 0u);
}

fn splitmix64(x0 : U64) -> U64 {
  var x = x0;
  x = u64_mul(u64_xor(x, u64_shr(x, 27u)), U64(0x2BA7B653u, 0x3C79AC49u));
  x = u64_mul(u64_xor(x, u64_shr(x, 33u)), U64(0x4AC4AE35u, 0x1C69B3F7u));
  return u64_xor(x, u64_shr(x, 27u));
}

/// `RNG_FORMULA.lane_stride`, `0x9E37_79B9_7F4A_7C15`.
fn rng_lane_stride() -> U64 { return U64(0x7F4A7C15u, 0x9E3779B9u); }

/// `RNG_FORMULA.ambient_seed_xor`, `0xA5A5_A5A5`.
fn rng_seed_eff(seed : u32) -> U64 {
  return u64_xor(U64(seed, 0u), U64(0xA5A5A5A5u, 0u));
}

fn rng_stream_salt(stream : u32) -> U64 {
  return splitmix64(u64_mul(U64(stream, 0u), rng_lane_stride()));
}

fn rng_seed_eff_stream(seed : u32, stream : u32) -> U64 {
  return u64_xor(rng_seed_eff(seed), rng_stream_salt(stream));
}

fn rng_keyed_seed(key : u32, ctr : u32) -> U64 {
  return splitmix64(U64(ctr, key));
}

/// `hash_uniform`: a draw in `(0, 1)`, never exactly either end.
fn rng_hash_uniform(seed_eff : U64, index : u32) -> f32 {
  let x = u64_add(seed_eff, u64_mul(rng_lane_stride(), U64(index + 1u, 0u)));
  let mixed = splitmix64(x);
  let bits = u64_shr(mixed, 40u).lo;
  let raw = (f32(bits) + 0.5) * (1.0 / 16777216.0);
  if (raw < 0.99999994) { return raw; }
  return 0.99999994;
}

fn op_rng(p : u32, keyed : bool) {
  let o = p_o0(p);
  let n = d_len(o);
  var seed : U64;
  if (keyed) {
    let st = p_a0(p);
    let key = ld_u(st, 0u);
    var ctr : u32 = 0u;
    if (d_len(st) > 1u) { ctr = ld_u(st, 1u); }
    seed = rng_keyed_seed(key, ctr);
  } else {
    seed = rng_seed_eff_stream(0u, p_imm(p));
  }
  let gumbel = p_kind(p) == 1u;
  for (var i : u32 = tid; i < n; i = i + lanes) {
    let u = rng_hash_uniform(seed, i);
    if (gumbel) { st_f(o, i, -log(-log(u))); } else { st_f(o, i, u); }
  }
}

// ---- order ---------------------------------------------------------------
//
// `sort_desc`, `top_k` and `pivot_threshold` all want a row in the host's
// descending order (`eta_exec::op::sort_desc_order`): NaN after every number,
// two NaNs and two equal values by ascending index. That comparator is a TOTAL
// order on indices, so the sorted permutation is UNIQUE and any correct sort
// reproduces the host's exactly — unlike a sum, whose bracketing is
// load-bearing and which is why the reduce ladder above copies the host's.
//
// It is built as a key sort. `sort_key` maps a lane to a `u32` that ascends as
// the value descends, with NaN above every number; the lane's own index is the
// second half of the key, so ties break by index for free. Sorting
// (key, index) ascending is then the host's order, by construction.
//
// The sort is a merge ladder, one round per emitted call: the round after
// reads what the round before wrote, and the barrier between them has to sit
// in uniform control flow, which inside `ptir_step`'s switch it cannot (the
// same reason the reduce is spelled as levels). A round whose runs already
// span the row copies instead of merging, so the ping-pong parity holds and
// the answer always lands in buffer 0 — `SORT_ROUNDS` is even for that.
//
// Each element finds its own destination by binary-searching the other run:
// its rank is its offset within its own run plus the count of the other run
// ordering below it. O(log) per element, fully parallel, and no tie-breaking
// between runs because the keys are unique.

/// Merge rounds. `2^28` exceeds any row the scratch bound (`MAX_BYTES`) can
/// hold, and an even count lands the answer in buffer 0.
const SORT_ROUNDS : u32 = 28u;

/// The host's descending order as one ascending `u32`.
fn sort_key(x : f32) -> u32 {
  if (x != x) { return 0xFFFFFFFFu; }
  // `-0.0` and `+0.0` compare equal on the host, so they must share a key and
  // let the index break the tie; their bit patterns do not, so normalise.
  let b = select(bitcast<u32>(x), 0u, x == 0.0);
  var asc : u32;
  if ((b & 0x80000000u) != 0u) { asc = ~b; } else { asc = b | 0x80000000u; }
  // `asc` ascends with the value and the key must descend. The smallest value,
  // `-inf`, gives `asc = 0x007FFFFF` and so a key of `0xFF800000`, which stays
  // below the NaN key rather than colliding with it.
  return 0xFFFFFFFFu - asc;
}

fn key_less(ah : u32, al : u32, bh : u32, bl : u32) -> bool {
  if (ah != bh) { return ah < bh; }
  return al < bl;
}

/// Word index of cell `c` of ping-pong buffer `buf`, for a value of `n` lanes.
/// Two buffers of two words each is `4n`, exactly what `scratch::layout`
/// reserves (`TEMPORARIES_PER_ELEMENT`).
fn sort_cell(buf : u32, n : u32, c : u32) -> u32 {
  return cfg.temporary + ((buf * n) + c) * 2u;
}

/// Rows and row length a sorting op reads. `sort_desc` orders the whole value
/// as one row, the way `eval_op` hands `sort_desc_order` the flat lanes;
/// `top_k` and `pivot_threshold` order each canonical row.
fn sort_rows_of(p : u32, a : u32) -> u32 {
  if (p_tag(p) == 0x50u) { return 1u; }
  return rows_of(a);
}

fn op_sort_seed(p : u32) {
  let a = p_a0(p);
  let n = d_len(a);
  let rows = sort_rows_of(p, a);
  let len = n / max(rows, 1u);
  for (var i : u32 = tid; i < n; i = i + lanes) {
    let c = sort_cell(0u, n, i);
    heap[c] = sort_key(ld_f(a, i));
    // The index within the row, which is both the host's tiebreak and what
    // `top_k` publishes.
    heap[c + 1u] = i - (i / max(len, 1u)) * max(len, 1u);
  }
}

fn op_sort_round(p : u32, r : u32) {
  let a = p_a0(p);
  let n = d_len(a);
  let rows = sort_rows_of(p, a);
  let len = n / max(rows, 1u);
  let src = r & 1u;
  let dst = 1u - src;
  let run = 1u << r;
  if (run >= len) {
    // The runs already span the row. Copying rather than returning is what
    // keeps the parity, so the answer lands in buffer 0 whatever the length.
    for (var i : u32 = tid; i < n; i = i + lanes) {
      let s = sort_cell(src, n, i);
      let d = sort_cell(dst, n, i);
      heap[d] = heap[s];
      heap[d + 1u] = heap[s + 1u];
    }
    return;
  }
  let span = run * 2u;
  for (var i : u32 = tid; i < n; i = i + lanes) {
    let row = i / max(len, 1u);
    let rowb = row * len;
    let at = i - rowb;
    let blk = (at / span) * span;
    let mid = min(blk + run, len);
    let end = min(blk + span, len);
    let me = sort_cell(src, n, i);
    let mh = heap[me];
    let ml = heap[me + 1u];
    var lo : u32 = 0u;
    var hi : u32 = 0u;
    var own : u32 = 0u;
    if (at < mid) {
      hi = end - mid;
      own = at - blk;
    } else {
      hi = mid - blk;
      own = at - mid;
    }
    loop {
      if (lo >= hi) { break; }
      let m = lo + (hi - lo) / 2u;
      var c : u32 = 0u;
      if (at < mid) { c = sort_cell(src, n, rowb + mid + m); }
      else { c = sort_cell(src, n, rowb + blk + m); }
      if (key_less(heap[c], heap[c + 1u], mh, ml)) { lo = m + 1u; } else { hi = m; }
    }
    let d = sort_cell(dst, n, rowb + blk + own + lo);
    heap[d] = mh;
    heap[d + 1u] = ml;
  }
}

/// **THE SCRATCH `pivot_threshold`'s TOP-`p` WALK RUNS ON.**
///
/// The walk is one invocation adding a row in order, and what it costs is not
/// the adds — it is one round trip to memory per lane, because a single
/// invocation has nothing else to run while a load is outstanding. Everything
/// here exists so that its accesses are CONTIGUOUS and INDEPENDENT, and a
/// block of them can be in flight at once. Measured on an L40S over 262144
/// lanes, that is the difference between 35.3 ms and 2.8 ms.
///
/// **THE FOUR ARRAYS ARE FLAT, NOT CELLS.** The ladder's ping-pong addresses
/// scratch as `n` cells of two words, so consecutive positions are two words
/// apart; a stride-two walk is not a contiguous one and gets none of the
/// above. The ladder is finished by the time these are read, so the same
/// `4n` words are re-cut as four flat arrays of `n`:
///
/// | array         | words                     | written by     | read by   |
/// |---------------|---------------------------|----------------|-----------|
/// | `pivot_flag`  | `temporary + 0n .. 1n`    | the walk       | the pack  |
/// | `pivot_meta`  | `temporary + 1n .. 1n+rows` | the walk     | the pack  |
/// | `pivot_val`   | `temporary + 2n .. 3n`    | `op_sort_pre`  | the walk  |
/// | `pivot_pos`   | `temporary + 3n .. 4n`    | `op_sort_pre`  | the pack  |
///
/// The cut is what keeps the passes off each other. `op_sort_pre` READS the
/// ladder's buffer 0, which is the first `2n` words, and writes only into the
/// second `2n`; the walk writes only the first, and by then every read of the
/// ladder's output is behind a dispatch boundary. Nothing is both read and
/// written in one pass.
///
/// `pivot_flag` is indexed by SORTED POSITION and `pivot_pos` by ELEMENT
/// INDEX, which is the pairing that leaves neither pass with a scattered
/// store: the walk writes flags in the order it visits them, and the pack
/// looks up each element's position rather than scattering back through the
/// ladder's order.
fn pivot_flag(n : u32, t : u32) -> u32 { return cfg.temporary + t; }
fn pivot_meta(n : u32, row : u32) -> u32 { return cfg.temporary + n + row; }
fn pivot_val(n : u32, t : u32) -> u32 { return cfg.temporary + 2u * n + t; }
fn pivot_pos(n : u32, i : u32) -> u32 { return cfg.temporary + 3u * n + i; }

/// Lanes the walk loads before it adds any of them. The adds stay in order —
/// the sum is still the host's sum, bracketed the host's way — but the loads
/// that feed them are issued together, which is the whole point.
const PIVOT_UNROLL : u32 = 32u;

/// Stages the row the top-`p` walk streams: its values in sorted order, and
/// the inverse of that order. A no-op for every other tag and predicate.
fn op_sort_pre(p : u32) {
  if (p_tag(p) != 0x58u) { return; }
  if (p_pred_tag(p) != 1u) { return; }
  let a = p_a0(p);
  let n = d_len(a);
  let rows = rows_of(a);
  let len = max(n / max(rows, 1u), 1u);
  for (var c : u32 = tid; c < n; c = c + lanes) {
    let rowb = (c / len) * len;
    let idx = heap[sort_cell(0u, n, c) + 1u];
    heap[pivot_val(n, c)] = bitcast<u32>(ld_f(a, rowb + idx));
    // `idx` is a permutation of the row, so each position is written once and
    // this scatter races nothing.
    heap[pivot_pos(n, rowb + idx)] = c - rowb;
  }
}

/// Packs the top-`p` walk's keep flags into the bool output.
///
/// The walk cannot write the output itself: a bool lane is one byte, four
/// share a word, and a read-modify-write from the one invocation that owns a
/// row would race the invocations owning its neighbours. So the walk writes
/// one word per sorted position and this pass, after the dispatch boundary
/// that orders them, folds four elements into a word — reading each element's
/// position out of `pivot_pos` rather than scattering back through the order.
///
/// A position at or past the row's `stop` was never written, and on a row the
/// walk was allowed to leave early everything past `stop` is unkept, so the
/// bound is the answer rather than a guard on one.
fn op_pivot_pack(p : u32) {
  if (p_tag(p) != 0x58u) { return; }
  if (p_pred_tag(p) != 1u) { return; }
  let a = p_a0(p);
  let o = p_o0(p);
  let n = d_len(a);
  let rows = rows_of(a);
  let len = max(n / max(rows, 1u), 1u);
  let words = (n + 3u) / 4u;
  for (var w : u32 = tid; w < words; w = w + lanes) {
    var b : array<bool, 4>;
    for (var k : u32 = 0u; k < 4u; k = k + 1u) {
      let i = w * 4u + k;
      var on = false;
      if (i < n) {
        let row = i / len;
        let t = heap[pivot_pos(n, i)];
        on = t < heap[pivot_meta(n, row)] && heap[pivot_flag(n, row * len + t)] != 0u;
      }
      b[k] = on;
    }
    st_b_word(o, w, n, b[0], b[1], b[2], b[3]);
  }
}

/// Cells of the sorted row whose key orders strictly below `kh` — the host's
/// `row.iter().filter(|&&y| !y.is_nan() && y > row[i]).count()`. A NaN carries
/// the largest key, so it never counts as greater; equal values share a key
/// after the `-0.0` normalisation, so they do not either.
fn count_key_below(n : u32, rowb : u32, len : u32, kh : u32) -> u32 {
  var lo : u32 = 0u;
  var hi : u32 = len;
  loop {
    if (lo >= hi) { break; }
    let m = lo + (hi - lo) / 2u;
    let c = sort_cell(0u, n, rowb + m);
    if (heap[c] < kh) { lo = m + 1u; } else { hi = m; }
  }
  return lo;
}

fn op_sort_desc_finish(p : u32) {
  let a = p_a0(p);
  let o = p_o0(p);
  let o1 = p_o1(p);
  let n = d_len(a);
  for (var t : u32 = tid; t < n; t = t + lanes) {
    let idx = heap[sort_cell(0u, n, t) + 1u];
    st_as_f(o, t, ld_f(a, idx));
    st_u(o1, t, idx);
  }
}

fn op_top_k_finish(p : u32) {
  let a = p_a0(p);
  let o = p_o0(p);
  let o1 = p_o1(p);
  let n = d_len(a);
  let rows = rows_of(a);
  let len = n / max(rows, 1u);
  // `order.iter().take(k)` yields the row when `k` runs past it.
  let want = min(p_imm(p), len);
  let total = rows * want;
  for (var j : u32 = tid; j < total; j = j + lanes) {
    let row = j / max(want, 1u);
    let t = j - row * want;
    let idx = heap[sort_cell(0u, n, row * len + t) + 1u];
    st_as_f(o, j, ld_f(a, row * len + idx));
    st_u(o1, j, idx);
  }
}

/// `pivot_threshold`. The two parallel arms write every word of the output
/// themselves; the top-`p` arm writes the keep flags `op_pivot_pack` folds
/// into it, which is why the output needs no clearing pass any more.
///
/// The top-`p` arm is the ordered walk `eval_op` takes. Its accumulator is
/// sequential and a differently bracketed sum is a different float, so the
/// fold stays serial — `crates/eta-compiler/src/codegen/cuda/scan.rs` states
/// the same constraint and settles it the same way: **parallelism is one row
/// per invocation, and within a row the walk is ordered.**
///
/// **WHAT IS SLOW ABOUT SUCH A WALK IS NOT THE ADDS.** One invocation with one
/// outstanding load has nothing to overlap it with, so it pays full memory
/// latency per lane. Two things answer that, and neither touches the order the
/// adds happen in: `op_sort_pre` staged the row so every access here is
/// contiguous, and the bulk is unrolled `PIVOT_UNROLL` wide so that many loads
/// are in flight before the first add consumes one. Measured on an L40S over
/// 262144 lanes, the walk went from 35.3 ms to 2.8 ms.
///
/// It stops as soon as the mass clears the cut, and only where that is exact.
/// `excl` can fall back under the cut only if some later lane is negative, and
/// descending order puts the negatives and the NaNs LAST — so the whole
/// question is answered by the row's final key, and testing the lane just
/// added instead would be wrong in both directions. `eta_exec`'s own bounded
/// pass defers on the same condition. A row that holds a negative is walked to
/// its end, which is what the host's fallback does.
fn op_pivot_finish(p : u32) {
  let a = p_a0(p);
  let o = p_o0(p);
  let payload = p_pred_payload(p);
  let pred = p_pred_tag(p);
  let n = d_len(a);
  let rows = rows_of(a);
  let len = n / max(rows, 1u);
  let np = d_len(payload);

  if (pred == 1u) {
    for (var row : u32 = tid; row < rows; row = row + lanes) {
      if (len == 0u) { continue; }
      let rowb = row * len;
      let cut = ld_f(payload, pick(np, row));
      let vals = pivot_val(n, rowb);
      let flags = pivot_flag(n, rowb);
      // The row's SMALLEST value, because the order is descending — so this
      // one lane answers "no lane is negative", which is the guard the break
      // needs, and it reads an array the walk never writes.
      let last = bitcast<f32>(heap[vals + len - 1u]);
      let monotone = !(last < 0.0) && last == last;
      var excl : f32 = 0.0;
      var t : u32 = 0u;
      loop {
        if (t + PIVOT_UNROLL > len) { break; }
        let v0 = bitcast<f32>(heap[vals + t + 0u]);
        let v1 = bitcast<f32>(heap[vals + t + 1u]);
        let v2 = bitcast<f32>(heap[vals + t + 2u]);
        let v3 = bitcast<f32>(heap[vals + t + 3u]);
        let v4 = bitcast<f32>(heap[vals + t + 4u]);
        let v5 = bitcast<f32>(heap[vals + t + 5u]);
        let v6 = bitcast<f32>(heap[vals + t + 6u]);
        let v7 = bitcast<f32>(heap[vals + t + 7u]);
        let v8 = bitcast<f32>(heap[vals + t + 8u]);
        let v9 = bitcast<f32>(heap[vals + t + 9u]);
        let v10 = bitcast<f32>(heap[vals + t + 10u]);
        let v11 = bitcast<f32>(heap[vals + t + 11u]);
        let v12 = bitcast<f32>(heap[vals + t + 12u]);
        let v13 = bitcast<f32>(heap[vals + t + 13u]);
        let v14 = bitcast<f32>(heap[vals + t + 14u]);
        let v15 = bitcast<f32>(heap[vals + t + 15u]);
        let v16 = bitcast<f32>(heap[vals + t + 16u]);
        let v17 = bitcast<f32>(heap[vals + t + 17u]);
        let v18 = bitcast<f32>(heap[vals + t + 18u]);
        let v19 = bitcast<f32>(heap[vals + t + 19u]);
        let v20 = bitcast<f32>(heap[vals + t + 20u]);
        let v21 = bitcast<f32>(heap[vals + t + 21u]);
        let v22 = bitcast<f32>(heap[vals + t + 22u]);
        let v23 = bitcast<f32>(heap[vals + t + 23u]);
        let v24 = bitcast<f32>(heap[vals + t + 24u]);
        let v25 = bitcast<f32>(heap[vals + t + 25u]);
        let v26 = bitcast<f32>(heap[vals + t + 26u]);
        let v27 = bitcast<f32>(heap[vals + t + 27u]);
        let v28 = bitcast<f32>(heap[vals + t + 28u]);
        let v29 = bitcast<f32>(heap[vals + t + 29u]);
        let v30 = bitcast<f32>(heap[vals + t + 30u]);
        let v31 = bitcast<f32>(heap[vals + t + 31u]);
        heap[flags + t + 0u] = select(0u, 1u, excl < cut);
        excl = excl + v0;
        heap[flags + t + 1u] = select(0u, 1u, excl < cut);
        excl = excl + v1;
        heap[flags + t + 2u] = select(0u, 1u, excl < cut);
        excl = excl + v2;
        heap[flags + t + 3u] = select(0u, 1u, excl < cut);
        excl = excl + v3;
        heap[flags + t + 4u] = select(0u, 1u, excl < cut);
        excl = excl + v4;
        heap[flags + t + 5u] = select(0u, 1u, excl < cut);
        excl = excl + v5;
        heap[flags + t + 6u] = select(0u, 1u, excl < cut);
        excl = excl + v6;
        heap[flags + t + 7u] = select(0u, 1u, excl < cut);
        excl = excl + v7;
        heap[flags + t + 8u] = select(0u, 1u, excl < cut);
        excl = excl + v8;
        heap[flags + t + 9u] = select(0u, 1u, excl < cut);
        excl = excl + v9;
        heap[flags + t + 10u] = select(0u, 1u, excl < cut);
        excl = excl + v10;
        heap[flags + t + 11u] = select(0u, 1u, excl < cut);
        excl = excl + v11;
        heap[flags + t + 12u] = select(0u, 1u, excl < cut);
        excl = excl + v12;
        heap[flags + t + 13u] = select(0u, 1u, excl < cut);
        excl = excl + v13;
        heap[flags + t + 14u] = select(0u, 1u, excl < cut);
        excl = excl + v14;
        heap[flags + t + 15u] = select(0u, 1u, excl < cut);
        excl = excl + v15;
        heap[flags + t + 16u] = select(0u, 1u, excl < cut);
        excl = excl + v16;
        heap[flags + t + 17u] = select(0u, 1u, excl < cut);
        excl = excl + v17;
        heap[flags + t + 18u] = select(0u, 1u, excl < cut);
        excl = excl + v18;
        heap[flags + t + 19u] = select(0u, 1u, excl < cut);
        excl = excl + v19;
        heap[flags + t + 20u] = select(0u, 1u, excl < cut);
        excl = excl + v20;
        heap[flags + t + 21u] = select(0u, 1u, excl < cut);
        excl = excl + v21;
        heap[flags + t + 22u] = select(0u, 1u, excl < cut);
        excl = excl + v22;
        heap[flags + t + 23u] = select(0u, 1u, excl < cut);
        excl = excl + v23;
        heap[flags + t + 24u] = select(0u, 1u, excl < cut);
        excl = excl + v24;
        heap[flags + t + 25u] = select(0u, 1u, excl < cut);
        excl = excl + v25;
        heap[flags + t + 26u] = select(0u, 1u, excl < cut);
        excl = excl + v26;
        heap[flags + t + 27u] = select(0u, 1u, excl < cut);
        excl = excl + v27;
        heap[flags + t + 28u] = select(0u, 1u, excl < cut);
        excl = excl + v28;
        heap[flags + t + 29u] = select(0u, 1u, excl < cut);
        excl = excl + v29;
        heap[flags + t + 30u] = select(0u, 1u, excl < cut);
        excl = excl + v30;
        heap[flags + t + 31u] = select(0u, 1u, excl < cut);
        excl = excl + v31;
        t = t + PIVOT_UNROLL;
        if (!(excl < cut) && monotone) { break; }
      }
      // The tail, and the whole of a row shorter than one block.
      loop {
        if (t >= len) { break; }
        heap[flags + t] = select(0u, 1u, excl < cut);
        excl = excl + bitcast<f32>(heap[vals + t]);
        t = t + 1u;
        if (!(excl < cut) && monotone) { break; }
      }
      // Where this row's answer ends. Positions at and past it were never
      // written, and on a row the walk was free to leave early they are all
      // unkept, so the bound IS the answer rather than a guard on one.
      heap[pivot_meta(n, row)] = t;
    }
    return;
  }

  // The two parallel arms, which write whole words because four bool lanes
  // share one.
  let words = (n + 3u) / 4u;
  for (var w : u32 = tid; w < words; w = w + lanes) {
    var b : array<bool, 4>;
    for (var k : u32 = 0u; k < 4u; k = k + 1u) {
      let i = w * 4u + k;
      var on = false;
      if (i < n) {
        let row = i / max(len, 1u);
        let rowb = row * len;
        let x = ld_f(a, i);
        if (pred == 0u) {
          let want = clamp(ld_i(payload, pick(np, row)), 0, i32(len));
          // The host leaves a NaN lane unkept rather than counting for it.
          if (!(x != x)) {
            on = i32(count_key_below(n, rowb, len, sort_key(x))) < want;
          }
        } else {
          on = x >= ld_f(payload, pick(np, row));
        }
      }
      b[k] = on;
    }
    st_b_word(o, w, n, b[0], b[1], b[2], b[3]);
  }
}

/// `matmul`. One invocation per output cell, summing the contraction in the
/// host's order — ascending `l`, skipping a zero left operand. Both are
/// load-bearing: a reduction tree over `l` is a different float, and the skip
/// matters because `0.0 * inf` is NaN and `-0.0 + 0.0` is `0.0`.
fn op_matmul(p : u32) {
  let a = p_a0(p);
  let b = p_a1(p);
  let o = p_o0(p);
  // The host faults on any rank but two. There is no fault channel here yet,
  // so a plan that reaches this with another rank is one the shell owes a
  // rejection; leaving the output untouched is the most this can say.
  if (d_rank(a) != 2u || d_rank(b) != 2u) { return; }
  let m = d_dim(a, 0u);
  let kk = d_dim(a, 1u);
  let n = d_dim(b, 1u);
  let total = m * n;
  for (var q : u32 = tid; q < total; q = q + lanes) {
    let i = q / max(n, 1u);
    let j = q - i * n;
    var acc : f32 = 0.0;
    for (var l : u32 = 0u; l < kk; l = l + 1u) {
      let xv = ld_f(a, i * kk + l);
      if (xv == 0.0) { continue; }
      acc = acc + xv * ld_f(b, l * n + j);
    }
    st_as_f(o, q, acc);
  }
}

// ---- dispatch ------------------------------------------------------------
//
// One call per op, in plan order. A reduce is several calls with the barriers
// between them supplied by the caller, which is why it is spelled out here
// rather than hidden inside the op: a barrier reached by only some invocations
// is undefined, and the emitted body puts every barrier at this level.

fn ptir_step(p : u32) {
  let tag = p_tag(p);
  switch (tag) {
    // map
    case 0x01u: { map_unary(p, 0u); }
    case 0x02u: { map_unary(p, 1u); }
    case 0x03u: { map_unary(p, 2u); }
    case 0x04u: { map_unary(p, 3u); }
    case 0x05u: { map_unary(p, 4u); }
    case 0x06u: { map_unary(p, 5u); }
    case 0x07u: { map_unary(p, 6u); }
    // arithmetic
    case 0x10u: { bin_arith(p, 0u); }
    case 0x11u: { bin_arith(p, 1u); }
    case 0x12u: { bin_arith(p, 2u); }
    case 0x13u: { bin_arith(p, 3u); }
    case 0x14u: { bin_arith(p, 4u); }
    case 0x15u: { bin_arith(p, 5u); }
    case 0x1Fu: { bin_arith(p, 6u); }
    // comparisons
    case 0x16u: { cmp_op(p, 0u); }
    case 0x17u: { cmp_op(p, 1u); }
    case 0x18u: { cmp_op(p, 2u); }
    case 0x19u: { cmp_op(p, 3u); }
    case 0x1Au: { cmp_op(p, 4u); }
    case 0x1Bu: { cmp_op(p, 5u); }
    // logic
    case 0x1Cu: { logic_op(p, 0u); }
    case 0x1Du: { logic_op(p, 1u); }
    case 0x1Eu: { logic_op(p, 2u); }
    // choice
    case 0x20u: { op_select(p); }
    // scan — one pass, not a ladder: the row is walked in order
    case 0x40u: { op_cumulative(p, false); }
    case 0x41u: { op_cumulative(p, true); }
    // shape
    case 0x38u: { op_copy(p); }
    case 0x39u: { op_copy(p); }
    case 0x3Au: { op_transpose(p); }
    // index
    case 0x60u: { op_gather(p); }
    case 0x61u: { op_gather_row(p); }
    case 0x62u: { op_scatter(p, true); }
    case 0x63u: { op_scatter(p, false); }
    case 0x64u: { op_iota(p); }
    // masks
    case 0x65u: { op_mask_apply(p); }
    case 0x66u: { op_struct_mask(p, 0u); }
    case 0x67u: { op_struct_mask(p, 1u); }
    case 0x68u: { op_struct_mask(p, 2u); }
    // sampling
    case 0x70u: { op_rng(p, false); }
    case 0x71u: { op_rng(p, true); }
    // order — the finishing pass of a sort ladder, whose rounds the caller
    // has already run and separated with barriers
    case 0x50u: { op_sort_desc_finish(p); }
    case 0x51u: { op_top_k_finish(p); }
    case 0x58u: { op_pivot_finish(p); }
    // linear
    case 0x55u: { op_matmul(p); }
    // leaf
    case 0x81u: { op_const(p); }
    // The Metal identity boundary: `kernel_call` copies, `sink_call` does
    // nothing, as `eta_exec::op::eval_op` has them.
    case 0xA1u: { op_copy(p); }
    case 0xA2u: {}
    default: {}
  }
}

/// The reduce family, whose levels the caller separates with barriers.
fn ptir_reduce_level(p : u32, level : u32) {
  let tag = p_tag(p);
  switch (tag) {
    case 0x30u: { op_reduce_level(p, RED_SUM, level); }
    case 0x31u: { op_reduce_level(p, RED_MAX, level); }
    case 0x32u: { op_reduce_level(p, RED_MIN, level); }
    case 0x33u: { op_reduce_level(p, RED_ARGMAX, level); }
    default: {}
  }
}

fn ptir_reduce_finish(p : u32) {
  let tag = p_tag(p);
  switch (tag) {
    case 0x30u: { op_reduce_finish(p, RED_SUM); }
    case 0x31u: { op_reduce_finish(p, RED_MAX); }
    case 0x32u: { op_reduce_finish(p, RED_MIN); }
    case 0x33u: { op_reduce_finish(p, RED_ARGMAX); }
    default: {}
  }
}

/// The sort family's ladder. `seed` fills buffer 0 from the value, each round
/// doubles the merged run, `pre` stages what the finishing pass reads, and
/// for `pivot_threshold` alone `pack` folds what it wrote — with the caller
/// supplying a barrier after each, for the same reason the reduce's levels
/// are spelled out: the pass after reads what the pass before wrote.
fn ptir_sort_seed(p : u32) {
  if (ptir_is_sort(p)) { op_sort_seed(p); }
}

fn ptir_sort_round(p : u32, r : u32) {
  if (ptir_is_sort(p)) { op_sort_round(p, r); }
}

fn ptir_sort_pre(p : u32) {
  if (ptir_is_sort(p)) { op_sort_pre(p); }
}

/// The ladder's last pass, and the only one that is not shared: it exists
/// because `pivot_threshold`'s top-`p` walk owns a row rather than a word,
/// and four bool lanes share a word. A no-op for every other tag and for
/// `pivot_threshold`'s two parallel arms, which write their words directly.
fn ptir_pivot_pack(p : u32) {
  if (ptir_is_sort(p)) { op_pivot_pack(p); }
}

/// True for the tags whose answer is read off an ordered row.
fn ptir_is_sort(p : u32) -> bool {
  let tag = p_tag(p);
  return tag == 0x50u || tag == 0x51u || tag == 0x58u;
}

/// True for the tags whose work is a reduce ladder rather than one pass.
fn ptir_is_reduce(p : u32) -> bool {
  let tag = p_tag(p);
  return tag >= 0x30u && tag <= 0x33u;
}
