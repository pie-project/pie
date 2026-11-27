

const PTIR_WG : u32 = 256u;

const DESC_WORDS : u32 = 9u;

const PARAM_WORDS : u32 = 16u;

const DT_F32 : u32 = 0u;
const DT_I32 : u32 = 1u;
const DT_U32 : u32 = 2u;
const DT_BOOL : u32 = 3u;
// bf16 (eta_ir::types::WIRE_BF16): two elements a word, element 2w in the
// low half of word w. Every op computes in f32; a bf16 result is rounded to
// nearest even on its store (bf16_round), and an invocation stores whole
// words of it, two adjacent elements, so no two invocations ever write one
// word. A value starts on a word, so the high half of an odd-length value's
// last word is its own padding.
const DT_BF16 : u32 = 5u;

// A stage's config table, by word: its values, its heap's words (no access
// reaches past them), its ops and its lane.
const CFG_HEAP_WORDS : u32 = 1u;

// The fault classes this runtime writes (eta-compiler codegen::fault).
const OUT_OF_BOUNDS : u32 = 0xB5u;

// What a stage reads and writes, in two pool segments a backend binds whole:
// the segment holding the instance's slots (its status word, read as
// atomics, and its tables: descriptors, offsets, op params and config, each
// at a base the step's record states) and the segment holding its heap. So
// one bind group serves every instance whose slots and heap lie in the same
// two segments, each step dispatched at its record's offset.
//
// Isolation, by three facts a bind and this runtime hold together:
//   1. The only words a step writes are heap[heap_base + w], w checked on
//      the device against the stage's heap words (hst), and
//      status[status_base + k] with k a constant (pie_fault). The naga test
//      of the step modules holds that no other store exists, and none to
//      the step's record.
//   2. Every table index starts from words the host wrote and no guest
//      store reaches (a value id from an op's params, an op from the step's
//      record, a constant word), checked at registration against the
//      stage's own counts (engine_gpu's plan_tables_valid), so a table read
//      stays in the instance's own table.
//   3. The bases come from the step's record, which the host writes: a bind
//      refuses a base whose table, status or heap runs outside the
//      instance's own slots or state, or its segment, and a heap that
//      overlaps a table where both lie in one segment.
@group(0) @binding(0) var<storage, read_write> status  : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read>       tables  : array<u32>;
@group(0) @binding(4) var<storage, read_write> heap    : array<u32>;

var<private> tid : u32;

var<private> lanes : u32;

// The invocation within its workgroup, the workgroup, and how many run.
var<private> sel_lt : u32;
var<private> sel_group : u32;
var<private> sel_groups : u32;


// The stage's heap bound, read once at the step's entry: every heap access
// is checked against it (the tables need no device check; see d_word). A heap
// index is relative to the stage's heap, [0, lim_heap), at its base in the
// segment the heap binding spans: never checked against the binding's size.
var<private> lim_heap : u32;

// Where the step's tables, status and heap start, in words of their
// segments, and the first heap word of its node's temporary area: read once
// at the step's entry from its record.
var<private> b_status : u32;
var<private> b_descs : u32;
var<private> b_params : u32;
var<private> b_offs : u32;
var<private> b_heap : u32;
var<private> tmp_base : u32;

// A fault in the runtimes' status layout (eta_exec::Status): the state
// Fault (3), its code, and the index and limit an access was refused at.
// The first fault of a stage wins: of the threads that fault, the one that
// swaps the state from unset writes the code, index and limit, so the three
// are one fault's.
fn pie_fault(code : u32, at : u32, limit : u32) {
  if (atomicExchange(&status[b_status], 3u) != 0u) { return; }
  atomicStore(&status[b_status + 1u], code);
  atomicStore(&status[b_status + 2u], at);
  atomicStore(&status[b_status + 3u], limit);
}

// The heap's words, each access refused past the stage's heap: a read
// reads zero, a write writes nothing, and the stage faults.
// The first heap access this invocation refused, written as the step's
// fault when it ends (heap_flush): a refused read reads zero with no branch,
// a refused store stores nothing.
var<private> heap_bad : bool;
var<private> heap_at : u32;

fn heap_refuse(w : u32, ok : bool) {
  heap_at = select(heap_at, w, !ok && !heap_bad);
  heap_bad = heap_bad || !ok;
}

fn hld(w : u32) -> u32 {
  let ok = w < lim_heap;
  heap_refuse(w, ok);
  return select(0u, heap[b_heap + min(w, lim_heap - 1u)], ok);
}

fn hst(w : u32, x : u32) {
  let ok = w < lim_heap;
  heap_refuse(w, ok);
  if (ok) { heap[b_heap + w] = x; }
}

fn heap_flush() {
  if (heap_bad) { pie_fault(OUT_OF_BOUNDS, heap_at, lim_heap); }
}

// A value's descriptor word `k`. The descriptors, offsets and op params
// are read at indices with no device check, by facts 2 and 3 above: every
// index is host-written and checked at registration against the stage's own
// counts, each table's base is checked at the bind, and no store of this
// runtime reaches the tables (it declares them read-only; a backend that
// binds them read_write beside the status, wgpu, still stores to them
// nowhere, as the step modules' test holds). Only the heap, whose indices
// come from guest data, is checked on the device (hld / hst).
fn d_word(v : u32, k : u32) -> u32 {
  return tables[b_descs + v * DESC_WORDS + k];
}

fn d_len(v : u32) -> u32 { return d_word(v, 0u); }
fn d_rows(v : u32) -> u32 { return d_word(v, 1u); }
fn d_rank(v : u32) -> u32 { return d_word(v, 3u); }
fn d_dtype(v : u32) -> u32 { return d_word(v, 4u); }
fn d_dim(v : u32, k : u32) -> u32 { return d_word(v, 5u + k); }

fn base(v : u32) -> u32 { return tables[b_offs + v] >> 2u; }

fn rows_of(v : u32) -> u32 {
  let rank = d_rank(v);
  if (rank < 2u) { return 1u; }
  var n : u32 = 1u;
  for (var k : u32 = 0u; k + 1u < rank; k = k + 1u) { n = n * d_dim(v, k); }
  return n;
}


fn p_at(p : u32, k : u32) -> u32 { return tables[b_params + p * PARAM_WORDS + k]; }
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
fn p_pred_payload(p : u32) -> u32 { return p_a1(p); }
fn p_dtype(p : u32) -> u32 { return p_at(p, 14u); }


fn ld_raw(v : u32, i : u32) -> u32 { return hld(base(v) + i); }

fn ld_byte(v : u32, i : u32) -> u32 {
  let w = hld(base(v) + (i >> 2u));
  return (w >> ((i & 3u) * 8u)) & 0xFFu;
}

// Element i of a bf16 value: its word, the half i names, widened exactly.
fn ld_h_bits(v : u32, i : u32) -> u32 {
  let w = hld(base(v) + (i >> 1u));
  return (w >> ((i & 1u) * 16u)) & 0xFFFFu;
}

fn ld_h(v : u32, i : u32) -> f32 { return bitcast<f32>(ld_h_bits(v, i) << 16u); }

fn is_float_dt(dt : u32) -> bool { return dt == DT_F32 || dt == DT_BF16; }

fn ld_f(v : u32, i : u32) -> f32 {
  let dt = d_dtype(v);
  if (dt == DT_F32) { return bitcast<f32>(ld_raw(v, i)); }
  if (dt == DT_BF16) { return ld_h(v, i); }
  if (dt == DT_I32) { return f32(bitcast<i32>(ld_raw(v, i))); }
  if (dt == DT_U32) { return f32(ld_raw(v, i)); }
  return select(0.0, 1.0, ld_byte(v, i) != 0u);
}

fn ld_i(v : u32, i : u32) -> i32 {
  let dt = d_dtype(v);
  if (dt == DT_F32) { return i32(bitcast<f32>(ld_raw(v, i))); }
  if (dt == DT_BF16) { return i32(ld_h(v, i)); }
  if (dt == DT_I32) { return bitcast<i32>(ld_raw(v, i)); }
  if (dt == DT_U32) { return i32(ld_raw(v, i)); }
  return select(0, 1, ld_byte(v, i) != 0u);
}

fn ld_u(v : u32, i : u32) -> u32 {
  let dt = d_dtype(v);
  if (dt == DT_F32) { return u32(bitcast<f32>(ld_raw(v, i))); }
  if (dt == DT_BF16) { return u32(ld_h(v, i)); }
  if (dt == DT_I32) { return bitcast<u32>(bitcast<i32>(ld_raw(v, i))); }
  if (dt == DT_U32) { return ld_raw(v, i); }
  return select(0u, 1u, ld_byte(v, i) != 0u);
}

fn ld_b(v : u32, i : u32) -> bool {
  let dt = d_dtype(v);
  if (dt == DT_BOOL) { return ld_byte(v, i) != 0u; }
  if (dt == DT_BF16) { return ld_h_bits(v, i) != 0u; }
  return ld_raw(v, i) != 0u;
}

fn st_raw(v : u32, i : u32, w : u32) { hst(base(v) + i, w); }
fn st_f(v : u32, i : u32, x : f32) { st_raw(v, i, bitcast<u32>(x)); }
fn st_i(v : u32, i : u32, x : i32) { st_raw(v, i, bitcast<u32>(x)); }
fn st_u(v : u32, i : u32, x : u32) { st_raw(v, i, x); }

fn st_b_word(v : u32, w : u32, n : u32, b0 : bool, b1 : bool, b2 : bool, b3 : bool) {
  var packed : u32 = 0u;
  if (b0 && (w * 4u + 0u) < n) { packed = packed | 0x000000FFu; }
  if (b1 && (w * 4u + 1u) < n) { packed = packed | 0x0000FF00u; }
  if (b2 && (w * 4u + 2u) < n) { packed = packed | 0x00FF0000u; }
  if (b3 && (w * 4u + 3u) < n) { packed = packed | 0xFF000000u; }
  hst(base(v) + w, packed);
}

// x rounded to bf16, round-to-nearest-even, as its 16 bits; a NaN stays a
// quiet NaN (eta_ir::types::bf16_bits).
fn bf16_round(x : f32) -> u32 {
  let b = bitcast<u32>(x);
  if ((b & 0x7FFFFFFFu) > 0x7F800000u) { return ((b >> 16u) | 0x40u) & 0xFFFFu; }
  return (b + 0x7FFFu + ((b >> 16u) & 1u)) >> 16u;
}

// Word w of a bf16 value of n elements: elements 2w and 2w + 1, x0 and x1,
// the high half the value's padding (zero) where 2w + 1 is past its end.
// The one bf16 store: an invocation owns the word it writes.
fn st_h_word(v : u32, w : u32, n : u32, x0 : f32, x1 : f32) {
  var packed = bf16_round(x0);
  if (w * 2u + 1u < n) { packed = packed | (bf16_round(x1) << 16u); }
  hst(base(v) + w, packed);
}

fn h_words(n : u32) -> u32 { return (n + 1u) / 2u; }

// A store of one element at its dtype. Never a bf16 one: a bf16 result is
// stored a word at a time (st_h_word).
fn st_as_f(v : u32, i : u32, x : f32) {
  let dt = d_dtype(v);
  if (dt == DT_F32) { st_f(v, i, x); }
  else if (dt == DT_I32) { st_i(v, i, i32(x)); }
  else if (dt == DT_U32) { st_u(v, i, u32(x)); }
}

fn st_as_i(v : u32, i : u32, x : i32) {
  let dt = d_dtype(v);
  if (dt == DT_F32) { st_f(v, i, f32(x)); }
  else if (dt == DT_I32) { st_i(v, i, x); }
  else if (dt == DT_U32) { st_u(v, i, bitcast<u32>(x)); }
}

fn pick(n : u32, i : u32) -> u32 {
  if (n == 1u) { return 0u; }
  return i;
}


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
  if (odt == DT_BF16) {
    let na = d_len(a);
    for (var w : u32 = tid; w < h_words(n); w = w + lanes) {
      let i = w * 2u;
      var x1 = 0.0;
      if (i + 1u < n) { x1 = unary_f(which, a, pick(na, i + 1u)); }
      st_h_word(o, w, n, unary_f(which, a, pick(na, i)), x1);
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
    case 7u: { return sin(x); }
    case 8u: { return cos(x); }
    case 9u: { return sqrt(x); }
    case 10u: { return 1.0 / sqrt(x); }
    default: {}
  }
  return x;
}

fn unary_i(which : u32, a : u32, j : u32) -> i32 {

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
    case 7u: { return i32(sin(ld_f(a, j))); }
    case 8u: { return i32(cos(ld_f(a, j))); }
    case 9u: { return i32(sqrt(ld_f(a, j))); }
    case 10u: { return i32(1.0 / sqrt(ld_f(a, j))); }
    default: {}
  }
  return ld_i(a, j);
}

fn unary_u(which : u32, a : u32, j : u32) -> u32 {
  switch (which) {
    case 0u: { return u32(exp(ld_f(a, j))); }
    case 1u: { return u32(log(ld_f(a, j))); }
    case 3u: { return u32(1.0 / ld_f(a, j)); }
    case 2u: { return 0u - ld_u(a, j); }
    case 4u: { return ld_u(a, j); }
    case 5u: { return select(0u, 1u, ld_u(a, j) != 0u); }
    case 7u: { return u32(sin(ld_f(a, j))); }
    case 8u: { return u32(cos(ld_f(a, j))); }
    case 9u: { return u32(sqrt(ld_f(a, j))); }
    case 10u: { return u32(1.0 / sqrt(ld_f(a, j))); }
    default: {}
  }
  return ld_u(a, j);
}

fn unary_b(which : u32, a : u32, j : u32) -> bool {

  return ld_f(a, j) != 0.0;
}


fn bin_arith(p : u32, which : u32) {
  let a = p_a0(p);
  let b = p_a1(p);
  let o = p_o0(p);
  let n = d_len(o);
  let adt = d_dtype(a);
  let na = d_len(a);
  let nb = d_len(b);
  if (d_dtype(o) == DT_BF16) {
    for (var w : u32 = tid; w < h_words(n); w = w + lanes) {
      let i = w * 2u;
      var x1 = 0.0;
      if (i + 1u < n) { x1 = arith_f(which, ld_f(a, pick(na, i + 1u)), ld_f(b, pick(nb, i + 1u))); }
      st_h_word(o, w, n, arith_f(which, ld_f(a, pick(na, i)), ld_f(b, pick(nb, i))), x1);
    }
    return;
  }
  if (is_float_dt(adt) || is_float_dt(d_dtype(b))) {
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
    default: {}
  }
  return x % y;
}

fn arith_i(which : u32, x : i32, y : i32) -> i32 {
  switch (which) {
    case 0u: { return x + y; }
    case 1u: { return x - y; }
    case 2u: { return x * y; }
    case 3u: { if (y == 0) { return 0; } return x / y; }
    case 4u: { return max(x, y); }
    case 5u: { return min(x, y); }
    default: {}
  }
  if (y == 0) { return 0; } return x % y;
}

fn arith_u(which : u32, x : u32, y : u32) -> u32 {
  switch (which) {
    case 0u: { return x + y; }
    case 1u: { return x - y; }
    case 2u: { return x * y; }
    case 3u: { if (y == 0u) { return 0u; } return x / y; }
    case 4u: { return max(x, y); }
    case 5u: { return min(x, y); }
    default: {}
  }
  if (y == 0u) { return 0u; } return x % y;
}


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
  if (is_float_dt(adt) || is_float_dt(d_dtype(b))) {
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
    default: {}
  }
  return x <= y;
}

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
  if (odt == DT_BF16) {
    for (var w : u32 = tid; w < h_words(n); w = w + lanes) {
      var xs = array<f32, 2>(0.0, 0.0);
      for (var k : u32 = 0u; k < 2u; k = k + 1u) {
        let i = w * 2u + k;
        if (i < n) {
          if (ld_b(c, pick(d_len(c), i))) { xs[k] = ld_f(x, pick(d_len(x), i)); }
          else { xs[k] = ld_f(y, pick(d_len(y), i)); }
        }
      }
      st_h_word(o, w, n, xs[0], xs[1]);
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


const REDUCE_LEVELS : u32 = 7u;

const RED_SUM : u32 = 0u;
const RED_MAX : u32 = 1u;
const RED_MIN : u32 = 2u;
const RED_ARGMAX : u32 = 3u;
// gumbel_argmax (tag 0x34): an argmax over lanes logits[i] * scale +
// gumbel(state, i), -inf where keep is false, each made in f32 in the first
// level and never stored (eta_ir::types: a draw is never a bf16 row). Its
// operands: a0 the logits, a1 keep, a2 the scale, imm the rng state.
const RED_GUMBEL : u32 = 4u;

fn gumbel_seed(p : u32) -> U64 {
  let st = p_imm(p);
  var ctr : u32 = 0u;
  if (d_len(st) > 1u) { ctr = ld_u(st, 1u); }
  return rng_keyed_seed(ld_u(st, 0u), ctr);
}

// Lane i of a gumbel_argmax: what `select(keep, logits * scale, -inf) +
// rng_keyed(state)` holds at i, in the same f32 operations.
fn gumbel_lane(p : u32, seed : U64, i : u32) -> f32 {
  let kp = p_a1(p);
  if (!ld_b(kp, pick(d_len(kp), i))) { return NEG_INF(); }
  let u = rng_hash_uniform(seed, i);
  let noise = -log(-log(u));
  return ld_f(p_a0(p), i) * ld_f(p_a2(p), 0u) + noise;
}

fn POS_INF() -> f32 { return bitcast<f32>(0x7F800000u); }
fn NEG_INF() -> f32 { return bitcast<f32>(0xFF800000u); }
// The literal `-0.0` folds to `+0.0` under Tint (Chrome), where a runtime
// negation keeps its sign; the bits are read through a `var<private>` so no
// compiler folds them (the same route `for_tint` takes for the infinities).
var<private> pie_neg_zero_bits : u32 = 0x80000000u;
fn NEG_ZERO() -> f32 { return bitcast<f32>(pie_neg_zero_bits); }

fn cmax(l : f32, r : f32) -> f32 {
  let ln = f32_is_nan(l);
  let rn = f32_is_nan(r);
  if (ln && rn) { return NEG_INF(); }
  if (ln) { return r; }
  if (rn) { return l; }
  if (l == 0.0 && r == 0.0) {
    if (sign_bit(l) && sign_bit(r)) { return NEG_ZERO(); }
    return 0.0;
  }
  if (l > r) { return l; }
  return r;
}

fn cmin(l : f32, r : f32) -> f32 {
  let ln = f32_is_nan(l);
  let rn = f32_is_nan(r);
  if (ln && rn) { return POS_INF(); }
  if (ln) { return r; }
  if (rn) { return l; }
  if (l == 0.0 && r == 0.0) {
    if (sign_bit(l) || sign_bit(r)) { return NEG_ZERO(); }
    return 0.0;
  }
  if (l < r) { return l; }
  return r;
}

fn sign_bit(x : f32) -> bool { return (bitcast<u32>(x) & 0x80000000u) != 0u; }

// A NaN read off its bits: WGSL compilers fold a self-compare (`x != x`)
// to false, so a NaN test written as one never fires.
fn f32_is_nan(x : f32) -> bool {
  let b = bitcast<u32>(x);
  return (b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u;
}

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

const PTIR_REDUCE_LEVELS : u32 = 7u;

// The lanes a job of a reduction's first level folds: 32, but 8 for a
// gumbel_argmax, whose lanes each hash the rng and take two logs, so a row
// spreads over four times the invocations (eta_compiler::codegen::steps::
// first_fan). Every level above folds 32.
fn red_fan0(kind : u32) -> u32 { return select(32u, 8u, kind == RED_GUMBEL); }

// The groups of a row the first level writes, which the levels above read:
// the reduction's temporary cells a row (wgsl::temporary_bytes).
fn red_stride_k(len : u32, kind : u32) -> u32 {
  let fan = red_fan0(kind);
  return (len + fan - 1u) / fan;
}

// The values a row has left before level `level`.
fn red_n_in(len : u32, kind : u32, level : u32) -> u32 {
  var n = len;
  for (var l : u32 = 0u; l < level; l = l + 1u) {
    let fan = select(32u, red_fan0(kind), l == 0u);
    n = (n + fan - 1u) / fan;
  }
  return n;
}

fn tmp_v(row : u32, stride : u32, c : u32, buf : u32, span : u32) -> u32 {
  return tmp_base + (buf * span + row * stride + c) * 2u;
}

fn red_read_buf(level : u32) -> u32 { return level % 2u; }

fn red_write_buf(level : u32) -> u32 { return (level + 1u) % 2u; }

fn op_reduce_level(p : u32, kind : u32, level : u32) {
  let a = p_a0(p);
  let rows = rows_of(a);
  let len = d_len(a) / max(rows, 1u);
  let stride = red_stride_k(len, kind);
  let n_in = red_n_in(len, kind, level);
  if (n_in <= 1u) { return; }
  var seed : U64;
  if (kind == RED_GUMBEL && level == 0u) { seed = gumbel_seed(p); }
  let fan = select(32u, red_fan0(kind), level == 0u);
  let n_out = (n_in + fan - 1u) / fan;
  let total = rows * n_out;
  for (var job : u32 = tid; job < total; job = job + lanes) {
    let row = job / n_out;
    let c = job % n_out;
    var vs : array<f32, 32>;
    var ix : array<u32, 32>;
    let count = min(fan, n_in - c * fan);
    for (var k : u32 = 0u; k < 32u; k = k + 1u) {
      vs[k] = red_identity(kind);
      ix[k] = 0u;
      if (k < count) {
        let at = c * fan + k;
        if (level == 0u && kind == RED_GUMBEL) {
          vs[k] = gumbel_lane(p, seed, row * len + at);
          ix[k] = at;
        } else if (level == 0u) {
          vs[k] = ld_f(a, row * len + at);
          ix[k] = at;
        } else {
          let cell = tmp_v(row, stride, at, red_read_buf(level), rows * stride);
          vs[k] = bitcast<f32>(hld(cell));
          ix[k] = hld(cell + 1u);
        }
      }
    }

    for (var off : u32 = 16u; off >= 1u; off = off / 2u) {
      for (var lane : u32 = 0u; lane < off; lane = lane + 1u) {
        if (kind == RED_ARGMAX || kind == RED_GUMBEL) {
          let takes = arg_takes(vs[lane], ix[lane], vs[lane + off], ix[lane + off], level, count, lane, off);
          if (takes) { vs[lane] = vs[lane + off]; ix[lane] = ix[lane + off]; }
        } else {
          vs[lane] = red_combine(kind, vs[lane], vs[lane + off]);
        }
      }
      if (off == 1u) { break; }
    }
    let cell = tmp_v(row, stride, c, red_write_buf(level), rows * stride);
    hst(cell, bitcast<u32>(vs[0]));
    hst(cell + 1u, ix[0]);
  }
}

fn arg_takes(lv : f32, li : u32, rv : f32, ri : u32, level : u32, count : u32, lane : u32, off : u32) -> bool {
  if (lane + off >= count) { return false; }
  let lnan = f32_is_nan(lv);
  let rnan = f32_is_nan(rv);
  if (rnan) { return false; }
  if (lnan) { return true; }
  return rv > lv || (rv == lv && ri < li);
}

fn op_reduce_finish(p : u32, kind : u32) {
  let a = p_a0(p);
  let o = p_o0(p);
  let rows = rows_of(a);
  let len = d_len(a) / max(rows, 1u);
  let stride = red_stride_k(len, kind);
  for (var row : u32 = tid; row < rows; row = row + lanes) {
    var v : f32;
    var idx : u32;
    if (len == 0u) {
      v = red_identity(kind);
      idx = 0u;
    } else if (len == 1u) {

      v = ld_f(a, row * len);
      idx = 0u;
    } else {

      var last : u32 = 0u;
      for (var l : u32 = 0u; l < PTIR_REDUCE_LEVELS; l = l + 1u) {
        if (red_n_in(len, kind, l) <= 1u) { break; }
        last = l;
      }
      let cell = tmp_v(row, stride, 0u, red_write_buf(last), rows * stride);
      v = bitcast<f32>(hld(cell));
      idx = hld(cell + 1u);
    }
    if (kind == RED_ARGMAX || kind == RED_GUMBEL) { st_i(o, row, bitcast<i32>(idx)); }
    else { st_as_f(o, row, v); }
  }
}

fn op_cumulative(p : u32, is_prod : bool) {
  let a = p_a0(p);
  let o = p_o0(p);
  let rows = rows_of(a);
  let len = d_len(a) / max(rows, 1u);
  if (d_dtype(o) == DT_BF16) {
    // An invocation scans a whole number of words: a row where a row is
    // whole words, two rows where it is not (an even span, from an even
    // element), so a word two rows share is one invocation's. Each prefix
    // is the f32 running value, rounded once on its store.
    if (len == 0u) { return; }
    let group = select(2u, 1u, (len & 1u) == 0u);
    let total = rows * len;
    let groups = (rows + group - 1u) / group;
    for (var g : u32 = tid; g < groups; g = g + lanes) {
      let first = g * group * len;
      let last = min(first + group * len, total);
      var acc : f32 = 0.0;
      var lo : f32 = 0.0;
      for (var at : u32 = first; at < last; at = at + 1u) {
        if ((at - first) % len == 0u) { acc = select(0.0, 1.0, is_prod); }
        let x = ld_f(a, at);
        if (is_prod) { acc = acc * x; } else { acc = acc + x; }
        if ((at & 1u) == 0u) {
          lo = acc;
          if (at + 1u == last) { st_h_word(o, at >> 1u, total, lo, 0.0); }
        } else {
          st_h_word(o, at >> 1u, total, lo, acc);
        }
      }
    }
    return;
  }
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
  if (odt == DT_BF16) {
    for (var w : u32 = tid; w < h_words(n); w = w + lanes) {
      let i = w * 2u;
      var x1 = 0.0;
      if (i + 1u < n) { x1 = ld_f(a, pick(na, i + 1u)); }
      st_h_word(o, w, n, ld_f(a, pick(na, i)), x1);
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

fn bcast_src(a : u32, o : u32, linear : u32) -> u32 {
  let orank = d_rank(o);
  let arank = d_rank(a);
  var rem : u32 = linear;
  var src : u32 = 0u;
  var sstride : array<u32, 4> = array<u32, 4>(1u, 1u, 1u, 1u);
  for (var dim : i32 = i32(orank) - 2; dim >= 0; dim = dim - 1) {
    let next = u32(dim) + 1u;
    var sd : u32 = 1u;
    if (next < arank) { sd = d_dim(a, next); }
    sstride[dim] = sstride[dim + 1] * sd;
  }
  for (var dim : u32 = 0u; dim < orank; dim = dim + 1u) {
    var stride : u32 = 1u;
    for (var next : u32 = dim + 1u; next < orank; next = next + 1u) { stride = stride * d_dim(o, next); }
    if (stride == 0u) { stride = 1u; }
    let coord = rem / stride;
    rem = rem % stride;
    var sd : u32 = 1u;
    if (dim < arank) { sd = d_dim(a, dim); }
    if (sd != 1u) { src = src + coord * sstride[dim]; }
  }
  return src;
}

fn op_broadcast(p : u32) {
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
        if (i < n) { v = ld_b(a, bcast_src(a, o, i)); }
        b[k] = v;
      }
      st_b_word(o, w, n, b[0], b[1], b[2], b[3]);
    }
    return;
  }
  if (odt == DT_BF16) {
    for (var w : u32 = tid; w < h_words(n); w = w + lanes) {
      let i = w * 2u;
      var x1 = 0.0;
      if (i + 1u < n) { x1 = ld_f(a, bcast_src(a, o, i + 1u)); }
      st_h_word(o, w, n, ld_f(a, bcast_src(a, o, i)), x1);
    }
    return;
  }
  for (var i : u32 = tid; i < n; i = i + lanes) {
    let j = bcast_src(a, o, i);
    if (odt == DT_F32) { st_f(o, i, ld_f(a, j)); }
    else if (odt == DT_I32) { st_i(o, i, ld_i(a, j)); }
    else { st_u(o, i, ld_u(a, j)); }
  }
}

fn op_transpose(p : u32) {
  let a = p_a0(p);
  let o = p_o0(p);
  let m = d_dim(a, 0u);
  let n2 = d_dim(a, 1u);
  let n = m * n2;
  let odt = d_dtype(o);
  if (odt == DT_BOOL) {
    let words = (n + 3u) / 4u;
    for (var w : u32 = tid; w < words; w = w + lanes) {
      var b : array<bool, 4>;
      for (var k : u32 = 0u; k < 4u; k = k + 1u) {
        let at = w * 4u + k;
        var v = false;
        if (at < n) { v = ld_b(a, (at % m) * n2 + at / m); }
        b[k] = v;
      }
      st_b_word(o, w, n, b[0], b[1], b[2], b[3]);
    }
    return;
  }
  if (odt == DT_BF16) {
    for (var w : u32 = tid; w < h_words(n); w = w + lanes) {
      let at = w * 2u;
      var x1 = 0.0;
      if (at + 1u < n) { x1 = ld_f(a, ((at + 1u) % m) * n2 + (at + 1u) / m); }
      st_h_word(o, w, n, ld_f(a, (at % m) * n2 + at / m), x1);
    }
    return;
  }
  for (var at : u32 = tid; at < n; at = at + lanes) {
    let src = (at % m) * n2 + at / m;
    if (odt == DT_F32) { st_f(o, at, ld_f(a, src)); }
    else if (odt == DT_I32) { st_i(o, at, ld_i(a, src)); }
    else if (odt == DT_U32) { st_u(o, at, ld_u(a, src)); }
  }
}


fn op_gather(p : u32) {
  let a = p_a0(p);
  let ix = p_a1(p);
  let o = p_o0(p);
  let n = d_len(o);
  let rest = gather_rest(a);
  let n0 = gather_first(a);
  let odt = d_dtype(o);
  if (odt == DT_BOOL) {
    let words = (n + 3u) / 4u;
    for (var w : u32 = tid; w < words; w = w + lanes) {
      var b : array<bool, 4>;
      for (var k : u32 = 0u; k < 4u; k = k + 1u) {
        let at = w * 4u + k;
        var v = false;
        if (at < n) {
          let want = ld_i(ix, pick(d_len(ix), at / rest));
          if (want >= 0 && u32(want) < n0) { v = ld_b(a, u32(want) * rest + at % rest); }
        }
        b[k] = v;
      }
      st_b_word(o, w, n, b[0], b[1], b[2], b[3]);
    }
    return;
  }
  if (odt == DT_BF16) {
    for (var w : u32 = tid; w < h_words(n); w = w + lanes) {
      var xs = array<f32, 2>(0.0, 0.0);
      for (var k : u32 = 0u; k < 2u; k = k + 1u) {
        let at = w * 2u + k;
        if (at < n) {
          let want = ld_i(ix, pick(d_len(ix), at / rest));
          if (want >= 0 && u32(want) < n0) { xs[k] = ld_f(a, u32(want) * rest + at % rest); }
        }
      }
      st_h_word(o, w, n, xs[0], xs[1]);
    }
    return;
  }
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

fn op_gather_row(p : u32) {
  let a = p_a0(p);
  let ix = p_a1(p);
  let o = p_o0(p);
  let m = d_dim(a, 0u);
  let n2 = d_dim(a, 1u);
  let odt = d_dtype(o);
  if (odt == DT_BOOL) {
    let words = (m + 3u) / 4u;
    for (var w : u32 = tid; w < words; w = w + lanes) {
      var b : array<bool, 4>;
      for (var k : u32 = 0u; k < 4u; k = k + 1u) {
        let i = w * 4u + k;
        var v = false;
        if (i < m) {
          let c = ld_i(ix, pick(d_len(ix), i));
          if (c >= 0 && u32(c) < n2) { v = ld_b(a, i * n2 + u32(c)); }
        }
        b[k] = v;
      }
      st_b_word(o, w, m, b[0], b[1], b[2], b[3]);
    }
    return;
  }
  if (odt == DT_BF16) {
    for (var w : u32 = tid; w < h_words(m); w = w + lanes) {
      var xs = array<f32, 2>(0.0, 0.0);
      for (var k : u32 = 0u; k < 2u; k = k + 1u) {
        let i = w * 2u + k;
        if (i < m) {
          let c = ld_i(ix, pick(d_len(ix), i));
          if (c >= 0 && u32(c) < n2) { xs[k] = ld_f(a, i * n2 + u32(c)); }
        }
      }
      st_h_word(o, w, m, xs[0], xs[1]);
    }
    return;
  }
  for (var i : u32 = tid; i < m; i = i + lanes) {
    let c = ld_i(ix, pick(d_len(ix), i));
    let ok = c >= 0 && u32(c) < n2;
    let j = select(0u, i * n2 + u32(c), ok);
    if (odt == DT_F32) { st_f(o, i, select(0.0, ld_f(a, j), ok)); }
    else if (odt == DT_I32) { st_i(o, i, select(0, ld_i(a, j), ok)); }
    else if (odt == DT_U32) { st_u(o, i, select(0u, ld_u(a, j), ok)); }
  }
}

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
  if (odt == DT_BOOL) {
    // Four cells a word: a thread owns the word it writes. An add is an or.
    let words = (n + 3u) / 4u;
    for (var w : u32 = tid; w < words; w = w + lanes) {
      var b : array<bool, 4>;
      for (var q : u32 = 0u; q < 4u; q = q + 1u) {
        let at = w * 4u + q;
        var v = false;
        if (at < n) {
          v = ld_b(a, at);
          let dest = at / rest;
          let within = at % rest;
          for (var k : u32 = 0u; k < slots; k = k + 1u) {
            let want = ld_i(ix, k);
            if (want < 0 || u32(want) >= n0 || u32(want) != dest) { continue; }
            var lane = k * rest + within;
            if (scalar) { lane = 0u; }
            let x = ld_b(src, lane);
            if (is_add) { v = v || x; } else { v = x; }
          }
        }
        b[q] = v;
      }
      st_b_word(o, w, n, b[0], b[1], b[2], b[3]);
    }
    return;
  }
  if (odt == DT_BF16) {
    // Two cells a word, each summed in f32 and rounded once.
    for (var w : u32 = tid; w < h_words(n); w = w + lanes) {
      var xs = array<f32, 2>(0.0, 0.0);
      for (var q : u32 = 0u; q < 2u; q = q + 1u) {
        let at = w * 2u + q;
        if (at >= n) { continue; }
        var acc = ld_f(a, at);
        let dest = at / rest;
        let within = at % rest;
        for (var k : u32 = 0u; k < slots; k = k + 1u) {
          let want = ld_i(ix, k);
          if (want < 0 || u32(want) >= n0 || u32(want) != dest) { continue; }
          var lane = k * rest + within;
          if (scalar) { lane = 0u; }
          let x = ld_f(src, lane);
          if (is_add) { acc = acc + x; } else { acc = x; }
        }
        xs[q] = acc;
      }
      st_h_word(o, w, n, xs[0], xs[1]);
    }
    return;
  }
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


fn op_mask_apply(p : u32) {
  let a = p_a0(p);
  let m = p_a1(p);
  let o = p_o0(p);
  let n = d_len(o);
  let rank = d_rank(a);
  var width : u32 = 1u;
  if (rank > 0u) { width = d_dim(a, rank - 1u); }
  let mw = d_len(m);
  if (d_dtype(o) == DT_BF16) {
    for (var w2 : u32 = tid; w2 < h_words(n); w2 = w2 + lanes) {
      var xs = array<f32, 2>(0.0, 0.0);
      for (var k : u32 = 0u; k < 2u; k = k + 1u) {
        let j = w2 * 2u + k;
        if (j >= n) { continue; }
        let c = j % max(width, 1u);
        let w = c >> 5u;
        var word : u32 = 0u;
        if (w < mw) { word = ld_u(m, w); }
        xs[k] = select(NEG_INF(), ld_f(a, j), ((word >> (c & 31u)) & 1u) != 0u);
      }
      st_h_word(o, w2, n, xs[0], xs[1]);
    }
    return;
  }
  for (var j : u32 = tid; j < n; j = j + lanes) {
    let c = j % max(width, 1u);
    let w = c >> 5u;
    var word : u32 = 0u;
    if (w < mw) { word = ld_u(m, w); }
    if (((word >> (c & 31u)) & 1u) != 0u) { st_f(o, j, ld_f(a, j)); }
    else { st_f(o, j, NEG_INF()); }
  }
}

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

fn rng_lane_stride() -> U64 { return U64(0x7F4A7C15u, 0x9E3779B9u); }

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

fn rng_hash_uniform(seed_eff : U64, index : u32) -> f32 {
  let x = u64_add(seed_eff, u64_mul(rng_lane_stride(), U64(index + 1u, 0u)));
  let mixed = splitmix64(x);
  let bits = u64_shr(mixed, 40u).lo;
  let raw = (f32(bits) + 0.5) * (1.0 / 16777216.0);
  if (raw < 0.99999994) { return raw; }
  return 0.99999994;
}

fn rng_hash_normal(seed_eff : U64, index : u32) -> f32 {
  let lane = index * 2u;
  let u0 = rng_hash_uniform(seed_eff, lane);
  let u1 = rng_hash_uniform(seed_eff, lane + 1u);
  let radius = sqrt(-2.0 * log(u0));
  return radius * cos(6.2831855 * u1);
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
  let kind = p_kind(p);
  for (var i : u32 = tid; i < n; i = i + lanes) {
    if (kind == 2u) { st_f(o, i, rng_hash_normal(seed, i)); continue; }
    let u = rng_hash_uniform(seed, i);
    if (kind == 1u) { st_f(o, i, -log(-log(u))); } else { st_f(o, i, u); }
  }
}


const SORT_ROUNDS : u32 = 28u;

fn sort_key(x : f32) -> u32 {
  if (f32_is_nan(x)) { return 0xFFFFFFFFu; }

  let b = select(bitcast<u32>(x), 0u, x == 0.0);
  var asc : u32;
  if ((b & 0x80000000u) != 0u) { asc = ~b; } else { asc = b | 0x80000000u; }

  return 0xFFFFFFFFu - asc;
}

fn key_less(ah : u32, al : u32, bh : u32, bl : u32) -> bool {
  if (ah != bh) { return ah < bh; }
  return al < bl;
}

fn sort_cell(buf : u32, n : u32, c : u32) -> u32 {
  return tmp_base + ((buf * n) + c) * 2u;
}

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
    hst(c, sort_key(ld_f(a, i)));

    hst(c + 1u, i - (i / max(len, 1u)) * max(len, 1u));
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

    for (var i : u32 = tid; i < n; i = i + lanes) {
      let s = sort_cell(src, n, i);
      let d = sort_cell(dst, n, i);
      hst(d, hld(s));
      hst(d + 1u, hld(s + 1u));
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
    let mh = hld(me);
    let ml = hld(me + 1u);
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
      if (key_less(hld(c), hld(c + 1u), mh, ml)) { lo = m + 1u; } else { hi = m; }
    }
    let d = sort_cell(dst, n, rowb + blk + own + lo);
    hst(d, mh);
    hst(d + 1u, ml);
  }
}

fn pivot_flag(n : u32, t : u32) -> u32 { return tmp_base + t; }
fn pivot_meta(n : u32, row : u32) -> u32 { return tmp_base + n + row; }
fn pivot_val(n : u32, t : u32) -> u32 { return tmp_base + 2u * n + t; }
fn pivot_pos(n : u32, i : u32) -> u32 { return tmp_base + 3u * n + i; }

const PIVOT_UNROLL : u32 = 32u;

fn op_sort_pre(p : u32) {
  if (p_tag(p) != 0x58u) { return; }
  if (p_pred_tag(p) != 1u) { return; }
  let a = p_a0(p);
  let n = d_len(a);
  let rows = rows_of(a);
  let len = max(n / max(rows, 1u), 1u);
  for (var c : u32 = tid; c < n; c = c + lanes) {
    let rowb = (c / len) * len;
    let idx = hld(sort_cell(0u, n, c) + 1u);
    if (idx >= len) { pie_fault(OUT_OF_BOUNDS, idx, len); continue; }
    hst(pivot_val(n, c), bitcast<u32>(ld_f(a, rowb + idx)));

    hst(pivot_pos(n, rowb + idx), c - rowb);
  }
}

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
        let t = hld(pivot_pos(n, i));
        on = t < hld(pivot_meta(n, row)) && hld(pivot_flag(n, row * len + t)) != 0u;
      }
      b[k] = on;
    }
    st_b_word(o, w, n, b[0], b[1], b[2], b[3]);
  }
}

fn count_key_below(n : u32, rowb : u32, len : u32, kh : u32) -> u32 {
  var lo : u32 = 0u;
  var hi : u32 = len;
  loop {
    if (lo >= hi) { break; }
    let m = lo + (hi - lo) / 2u;
    let c = sort_cell(0u, n, rowb + m);
    if (hld(c) < kh) { lo = m + 1u; } else { hi = m; }
  }
  return lo;
}

fn op_sort_desc_finish(p : u32) {
  let a = p_a0(p);
  let o = p_o0(p);
  let o1 = p_o1(p);
  let n = d_len(a);
  if (d_dtype(o) == DT_BF16) {
    for (var w : u32 = tid; w < h_words(n); w = w + lanes) {
      var xs = array<f32, 2>(0.0, 0.0);
      for (var k : u32 = 0u; k < 2u; k = k + 1u) {
        let t = w * 2u + k;
        if (t >= n) { continue; }
        let idx = hld(sort_cell(0u, n, t) + 1u);
        if (idx >= n) { pie_fault(OUT_OF_BOUNDS, idx, n); continue; }
        xs[k] = ld_f(a, idx);
        st_u(o1, t, idx);
      }
      st_h_word(o, w, n, xs[0], xs[1]);
    }
    return;
  }
  for (var t : u32 = tid; t < n; t = t + lanes) {
    let idx = hld(sort_cell(0u, n, t) + 1u);
    if (idx >= n) { pie_fault(OUT_OF_BOUNDS, idx, n); continue; }
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

  let want = min(p_imm(p), len);
  let total = rows * want;
  if (d_dtype(o) == DT_BF16) {
    for (var w : u32 = tid; w < h_words(total); w = w + lanes) {
      var xs = array<f32, 2>(0.0, 0.0);
      for (var k : u32 = 0u; k < 2u; k = k + 1u) {
        let j = w * 2u + k;
        if (j >= total) { continue; }
        let row = j / max(want, 1u);
        let t = j - row * want;
        let idx = hld(sort_cell(0u, n, row * len + t) + 1u);
        if (idx >= len) { pie_fault(OUT_OF_BOUNDS, idx, len); continue; }
        xs[k] = ld_f(a, row * len + idx);
        st_u(o1, j, idx);
      }
      st_h_word(o, w, total, xs[0], xs[1]);
    }
    return;
  }
  for (var j : u32 = tid; j < total; j = j + lanes) {
    let row = j / max(want, 1u);
    let t = j - row * want;
    let idx = hld(sort_cell(0u, n, row * len + t) + 1u);
    if (idx >= len) { pie_fault(OUT_OF_BOUNDS, idx, len); continue; }
    st_as_f(o, j, ld_f(a, row * len + idx));
    st_u(o1, j, idx);
  }
}

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

      let last = bitcast<f32>(hld(vals + len - 1u));
      let monotone = !(last < 0.0) && !f32_is_nan(last);
      var excl : f32 = 0.0;
      var t : u32 = 0u;
      loop {
        if (t + PIVOT_UNROLL > len) { break; }
        // Rolled: SwiftShader in Chrome grew past 11 GiB compiling this as 32
        // loads then 32 stores spelled out; each element still adds in turn.
        for (var k : u32 = 0u; k < PIVOT_UNROLL; k = k + 1u) {
          let v = bitcast<f32>(hld(vals + t + k));
          hst(flags + t + k, select(0u, 1u, excl < cut));
          excl = excl + v;
        }
        t = t + PIVOT_UNROLL;
        if (!(excl < cut) && monotone) { break; }
      }

      loop {
        if (t >= len) { break; }
        hst(flags + t, select(0u, 1u, excl < cut));
        excl = excl + bitcast<f32>(hld(vals + t));
        t = t + 1u;
        if (!(excl < cut) && monotone) { break; }
      }

      hst(pivot_meta(n, row), t);
    }
    return;
  }

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

          if (!f32_is_nan(x)) {
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

// A one-row PivotThreshold by radix select rather than a sort of the row.
// Four 8-bit histogram passes find the want-th smallest sort key: for RankLe
// that key is the cut itself (keep x when key(x) <= it, ties and all); for
// CummassLe it bounds the SEL_CAP heaviest candidates, which one workgroup
// sorts and walks in the reference's order. A row whose mass those do not
// reach, or that holds a negative or NaN, takes the sort ladder after all.
// The state sits in the SEL_WORDS just below the temporary area, which the
// ladder overwrites; eta-exec's scratch layout reserves them.

const SEL_CAP : u32 = 1024u;
const SEL_GROUPS : u32 = 32u;
const SEL_WORDS : u32 = 10320u;

const SEL_MODE : u32 = 0u;
const SEL_ACTIVE : u32 = 1u;
const SEL_WANT : u32 = 2u;
const SEL_PREFIX : u32 = 3u;
const SEL_REMAINING : u32 = 4u;
const SEL_TOTAL_LT : u32 = 5u;
const SEL_KEEP : u32 = 6u;

// SEL_MODE: the select runs, has decided the row, or hands it to the ladder.
const SEL_RUNS : u32 = 0u;
const SEL_DECIDED : u32 = 1u;
const SEL_LADDER : u32 = 2u;

// SEL_KEEP for RankLe: the cut decides, or every non-NaN lane is kept, or none.
const SEL_KEEP_CUT : u32 = 0u;
const SEL_KEEP_ALL : u32 = 1u;
const SEL_KEEP_NONE : u32 = 2u;

var<workgroup> sel_wg_hist : array<atomic<u32>, 256>;
var<workgroup> sel_wg_fill : array<atomic<u32>, 2>;
var<workgroup> sel_wg_key : array<u32, 1024>;
var<workgroup> sel_wg_idx : array<u32, 1024>;
var<workgroup> sel_wg_skip : u32;

fn sel_at(k : u32) -> u32 { return tmp_base - SEL_WORDS + k; }
fn sel_part(g : u32, b : u32) -> u32 { return sel_at(16u + g * 256u + b); }
fn sel_count(g : u32, which : u32) -> u32 { return sel_at(16u + SEL_GROUPS * 256u + g * 2u + which); }
fn sel_cand_key(i : u32) -> u32 { return sel_at(16u + SEL_GROUPS * 258u + i); }
fn sel_cand_idx(i : u32) -> u32 { return sel_at(16u + SEL_GROUPS * 258u + SEL_CAP + i); }

// Whether this workgroup sits the step out; uniform across it.
fn sel_skips(only_first : bool) -> bool {
  if (sel_lt == 0u) {
    let busy = hld(sel_at(SEL_ACTIVE));
    let idle = select(sel_group >= busy, sel_group != 0u, only_first);
    sel_wg_skip = select(0u, 1u, hld(sel_at(SEL_MODE)) != SEL_RUNS || idle);
  }
  return workgroupUniformLoad(&sel_wg_skip) != 0u;
}

fn sel_chunk(n : u32) -> vec2<u32> {
  let busy = hld(sel_at(SEL_ACTIVE));
  let chunk = (n + busy - 1u) / busy;
  let lo = min(sel_group * chunk, n);
  return vec2<u32>(lo, min(lo + chunk, n));
}

// The ladder's steps run only for a row the select handed over.
fn sel_ladder_idle() -> bool {
  return hld(sel_at(SEL_MODE)) != SEL_LADDER;
}

fn sel_init(p : u32) {
  let o = p_o0(p);
  let n = d_len(p_a0(p));
  // A pivot's mask starts clear; a top-k writes each of its results.
  if (p_tag(p) != 0x51u) {
    let words = (n + 3u) / 4u;
    for (var w : u32 = tid; w < words; w = w + lanes) { hst(base(o) + w, 0u); }
  }
  if (tid == 0u) {
    hst(sel_at(SEL_MODE), SEL_RUNS);
    hst(sel_at(SEL_ACTIVE), max(min(SEL_GROUPS, sel_groups), 1u));
    hst(sel_at(SEL_PREFIX), 0u);
    hst(sel_at(SEL_KEEP), SEL_KEEP_CUT);
  }
}

fn sel_hist(p : u32, sweep : u32) {
  if (sel_skips(false)) { return; }
  let a = p_a0(p);
  let span = sel_chunk(d_len(a));
  let shift = 24u - 8u * sweep;
  let prefix = hld(sel_at(SEL_PREFIX));
  atomicStore(&sel_wg_hist[sel_lt], 0u);
  workgroupBarrier();
  for (var i : u32 = span.x + sel_lt; i < span.y; i = i + PTIR_WG) {
    let x = ld_f(a, i);
    if (f32_is_nan(x)) { continue; }
    let key = sort_key(x);
    if (sweep == 0u || (key >> (shift + 8u)) == (prefix >> (shift + 8u))) {
      atomicAdd(&sel_wg_hist[(key >> shift) & 255u], 1u);
    }
  }
  workgroupBarrier();
  hst(sel_part(sel_group, sel_lt), atomicLoad(&sel_wg_hist[sel_lt]));
}

fn sel_pick(p : u32, sweep : u32) {
  if (sel_skips(true)) { return; }
  let busy = hld(sel_at(SEL_ACTIVE));
  var here : u32 = 0u;
  for (var g : u32 = 0u; g < busy; g = g + 1u) { here = here + hld(sel_part(g, sel_lt)); }
  sel_wg_key[sel_lt] = here;
  workgroupBarrier();
  if (sel_lt != 0u) { return; }
  var remaining = hld(sel_at(SEL_REMAINING));
  if (sweep == 0u) {
    let n = d_len(p_a0(p));
    var count : u32 = 0u;
    var negative : u32 = 0u;
    for (var b : u32 = 0u; b < 256u; b = b + 1u) {
      count = count + sel_wg_key[b];
      if (b >= 128u) { negative = negative + sel_wg_key[b]; }
    }
    let payload = p_pred_payload(p);
    var want : u32 = 0u;
    if (p_tag(p) == 0x51u) {
      // A top-k: its `k` from the node. None, or more than the row's
      // non-NaN keys or the candidates hold, is the sort's to take.
      let k = p_imm(p);
      if (k == 0u || k >= count || k > SEL_CAP) {
        hst(sel_at(SEL_MODE), SEL_LADDER);
        return;
      }
      want = k;
    } else if (p_pred_tag(p) == 0u) {
      let k = ld_i(payload, 0u);
      if (k <= 0) {
        hst(sel_at(SEL_KEEP), SEL_KEEP_NONE);
        hst(sel_at(SEL_MODE), SEL_DECIDED);
        return;
      }
      if (u32(k) >= count) {
        hst(sel_at(SEL_KEEP), SEL_KEEP_ALL);
        hst(sel_at(SEL_MODE), SEL_DECIDED);
        return;
      }
      want = u32(k);
    } else {
      let cut = ld_f(payload, 0u);
      if (!(0.0 < cut)) {
        hst(sel_at(SEL_MODE), SEL_DECIDED);
        return;
      }
      if (negative != 0u || count != n || count == 0u) {
        hst(sel_at(SEL_MODE), SEL_LADDER);
        return;
      }
      want = min(SEL_CAP, count);
    }
    hst(sel_at(SEL_WANT), want);
    remaining = want;
  }
  let shift = 24u - 8u * sweep;
  var before : u32 = 0u;
  var digit : u32 = 0u;
  loop {
    if (digit + 1u >= 256u) { break; }
    let count = sel_wg_key[digit];
    if (before + count >= remaining) { break; }
    before = before + count;
    digit = digit + 1u;
  }
  hst(sel_at(SEL_PREFIX), hld(sel_at(SEL_PREFIX)) | (digit << shift));
  hst(sel_at(SEL_REMAINING), remaining - before);
  if (sweep == 3u) {
    hst(sel_at(SEL_TOTAL_LT), hld(sel_at(SEL_WANT)) - (remaining - before));
  }
}

fn sel_rank_mask(p : u32) {
  let a = p_a0(p);
  let o = p_o0(p);
  let n = d_len(a);
  let keep = hld(sel_at(SEL_KEEP));
  let cut = hld(sel_at(SEL_PREFIX));
  let words = (n + 3u) / 4u;
  for (var w : u32 = tid; w < words; w = w + lanes) {
    var b : array<bool, 4>;
    for (var k : u32 = 0u; k < 4u; k = k + 1u) {
      let i = w * 4u + k;
      var on = false;
      if (i < n && keep != SEL_KEEP_NONE) {
        let x = ld_f(a, i);
        on = !f32_is_nan(x) && (keep == SEL_KEEP_ALL || sort_key(x) <= cut);
      }
      b[k] = on;
    }
    st_b_word(o, w, n, b[0], b[1], b[2], b[3]);
  }
}

// Counts each workgroup's candidates: keys below the cut, and keys at it.
fn sel_tally(p : u32) {
  if (sel_skips(false)) { return; }
  let a = p_a0(p);
  let span = sel_chunk(d_len(a));
  let cut = hld(sel_at(SEL_PREFIX));
  if (sel_lt < 2u) { atomicStore(&sel_wg_fill[sel_lt], 0u); }
  workgroupBarrier();
  for (var i : u32 = span.x + sel_lt; i < span.y; i = i + PTIR_WG) {
    let key = sort_key(ld_f(a, i));
    if (key < cut) { atomicAdd(&sel_wg_fill[0], 1u); }
    else if (key == cut) { atomicAdd(&sel_wg_fill[1], 1u); }
  }
  workgroupBarrier();
  if (sel_lt < 2u) { hst(sel_count(sel_group, sel_lt), atomicLoad(&sel_wg_fill[sel_lt])); }
}

// Writes the candidates out: those below the cut first, then the ties the
// cap leaves room for.
fn sel_place(p : u32) {
  if (sel_skips(false)) { return; }
  let a = p_a0(p);
  let span = sel_chunk(d_len(a));
  let cut = hld(sel_at(SEL_PREFIX));
  let below = hld(sel_at(SEL_TOTAL_LT));
  var lt_at : u32 = 0u;
  var eq_at : u32 = 0u;
  for (var g : u32 = 0u; g < sel_group; g = g + 1u) {
    lt_at = lt_at + hld(sel_count(g, 0u));
    eq_at = eq_at + hld(sel_count(g, 1u));
  }
  if (sel_lt < 2u) { atomicStore(&sel_wg_fill[sel_lt], 0u); }
  workgroupBarrier();
  for (var i : u32 = span.x + sel_lt; i < span.y; i = i + PTIR_WG) {
    let key = sort_key(ld_f(a, i));
    if (key < cut) {
      let slot = lt_at + atomicAdd(&sel_wg_fill[0], 1u);
      if (slot >= SEL_CAP) { pie_fault(OUT_OF_BOUNDS, slot, SEL_CAP); continue; }
      hst(sel_cand_key(slot), key);
      hst(sel_cand_idx(slot), i);
    } else if (key == cut) {
      let slot = below + eq_at + atomicAdd(&sel_wg_fill[1], 1u);
      if (slot < SEL_CAP) {
        hst(sel_cand_key(slot), key);
        hst(sel_cand_idx(slot), i);
      }
    }
  }
}

fn st_b_on(v : u32, i : u32) {
  let at = base(v) + i / 4u;
  hst(at, hld(at) | (0xFFu << (8u * (i % 4u))));
}

// Sorts the candidates by (key, index) and walks them as the reference does.
fn sel_walk(p : u32) {
  if (sel_skips(true)) { return; }
  let a = p_a0(p);
  let o = p_o0(p);
  let busy = hld(sel_at(SEL_ACTIVE));
  var ties : u32 = 0u;
  for (var g : u32 = 0u; g < busy; g = g + 1u) { ties = ties + hld(sel_count(g, 1u)); }
  let below = hld(sel_at(SEL_TOTAL_LT));
  // Ties past the cap landed in no particular order, so the walk trusts
  // only the keys below the cut then.
  let held = select(below, below + ties, below + ties <= SEL_CAP);
  for (var i : u32 = sel_lt; i < SEL_CAP; i = i + PTIR_WG) {
    if (i < held) {
      sel_wg_key[i] = hld(sel_cand_key(i));
      sel_wg_idx[i] = hld(sel_cand_idx(i));
    } else {
      sel_wg_key[i] = 0xFFFFFFFFu;
      sel_wg_idx[i] = 0xFFFFFFFFu;
    }
  }
  workgroupBarrier();
  for (var size : u32 = 2u; size <= SEL_CAP; size = size * 2u) {
    for (var stride : u32 = size / 2u; stride > 0u; stride = stride / 2u) {
      for (var t : u32 = sel_lt; t < SEL_CAP / 2u; t = t + PTIR_WG) {
        let lo = 2u * stride * (t / stride) + (t % stride);
        let hi = lo + stride;
        let ascending = (lo & size) == 0u;
        let klo = sel_wg_key[lo];
        let khi = sel_wg_key[hi];
        let ilo = sel_wg_idx[lo];
        let ihi = sel_wg_idx[hi];
        let lo_greater = klo > khi || (klo == khi && ilo > ihi);
        if (lo_greater == ascending) {
          sel_wg_key[lo] = khi;
          sel_wg_key[hi] = klo;
          sel_wg_idx[lo] = ihi;
          sel_wg_idx[hi] = ilo;
        }
      }
      workgroupBarrier();
    }
  }
  if (sel_lt != 0u) { return; }
  let cut = ld_f(p_pred_payload(p), 0u);
  var excl : f32 = 0.0;
  var reached = false;
  for (var t : u32 = 0u; t < held; t = t + 1u) {
    if (!(excl < cut)) { reached = true; break; }
    let index = sel_wg_idx[t];
    if (index >= d_len(o) || index >= d_len(a)) {
      pie_fault(OUT_OF_BOUNDS, index, min(d_len(o), d_len(a)));
      break;
    }
    st_b_on(o, index);
    excl = excl + ld_f(a, index);
  }
  let whole = held == d_len(a);
  hst(sel_at(SEL_MODE), select(SEL_LADDER, SEL_DECIDED, reached || !(excl < cut) || whole));
}

// A top-k's candidates sorted by (key, index), the sort's order, its first
// `k` written as its values and indices. Ties at the cut past the cap
// landed in no particular order, so the row is the sort's then.
fn sel_top_k(p : u32) {
  if (sel_skips(true)) { return; }
  let a = p_a0(p);
  let o = p_o0(p);
  let o1 = p_o1(p);
  let busy = hld(sel_at(SEL_ACTIVE));
  var ties : u32 = 0u;
  for (var g : u32 = 0u; g < busy; g = g + 1u) { ties = ties + hld(sel_count(g, 1u)); }
  let below = hld(sel_at(SEL_TOTAL_LT));
  let held = below + ties;
  if (sel_lt == 0u) {
    sel_wg_skip = select(0u, 1u, held > SEL_CAP);
    if (held > SEL_CAP) { hst(sel_at(SEL_MODE), SEL_LADDER); }
  }
  if (workgroupUniformLoad(&sel_wg_skip) != 0u) { return; }
  for (var i : u32 = sel_lt; i < SEL_CAP; i = i + PTIR_WG) {
    if (i < held) {
      sel_wg_key[i] = hld(sel_cand_key(i));
      sel_wg_idx[i] = hld(sel_cand_idx(i));
    } else {
      sel_wg_key[i] = 0xFFFFFFFFu;
      sel_wg_idx[i] = 0xFFFFFFFFu;
    }
  }
  workgroupBarrier();
  for (var size : u32 = 2u; size <= SEL_CAP; size = size * 2u) {
    for (var stride : u32 = size / 2u; stride > 0u; stride = stride / 2u) {
      for (var t : u32 = sel_lt; t < SEL_CAP / 2u; t = t + PTIR_WG) {
        let lo = 2u * stride * (t / stride) + (t % stride);
        let hi = lo + stride;
        let ascending = (lo & size) == 0u;
        let klo = sel_wg_key[lo];
        let khi = sel_wg_key[hi];
        let ilo = sel_wg_idx[lo];
        let ihi = sel_wg_idx[hi];
        let lo_greater = klo > khi || (klo == khi && ilo > ihi);
        if (lo_greater == ascending) {
          sel_wg_key[lo] = khi;
          sel_wg_key[hi] = klo;
          sel_wg_idx[lo] = ihi;
          sel_wg_idx[hi] = ilo;
        }
      }
      workgroupBarrier();
    }
  }
  let want = hld(sel_at(SEL_WANT));
  let len = d_len(a);
  if (d_dtype(o) == DT_BF16) {
    for (var w : u32 = sel_lt; w < h_words(want); w = w + PTIR_WG) {
      var xs = array<f32, 2>(0.0, 0.0);
      for (var k : u32 = 0u; k < 2u; k = k + 1u) {
        let j = w * 2u + k;
        if (j >= want) { continue; }
        let idx = sel_wg_idx[j];
        if (idx >= len) { pie_fault(OUT_OF_BOUNDS, idx, len); continue; }
        xs[k] = ld_f(a, idx);
        st_u(o1, j, idx);
      }
      st_h_word(o, w, want, xs[0], xs[1]);
    }
    if (sel_lt == 0u) { hst(sel_at(SEL_MODE), SEL_DECIDED); }
    return;
  }
  for (var j : u32 = sel_lt; j < want; j = j + PTIR_WG) {
    let idx = sel_wg_idx[j];
    if (idx >= len) { pie_fault(OUT_OF_BOUNDS, idx, len); continue; }
    st_as_f(o, j, ld_f(a, idx));
    st_u(o1, j, idx);
  }
  if (sel_lt == 0u) { hst(sel_at(SEL_MODE), SEL_DECIDED); }
}

fn op_matmul(p : u32) {
  let a = p_a0(p);
  let b = p_a1(p);
  let o = p_o0(p);

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


fn ptir_step(p : u32) {
  let tag = p_tag(p);
  switch (tag) {

    case 0x01u: { map_unary(p, 0u); }
    case 0x02u: { map_unary(p, 1u); }
    case 0x03u: { map_unary(p, 2u); }
    case 0x04u: { map_unary(p, 3u); }
    case 0x05u: { map_unary(p, 4u); }
    case 0x06u: { map_unary(p, 5u); }
    case 0x07u: { map_unary(p, 6u); }
    case 0x08u: { map_unary(p, 7u); }
    case 0x09u: { map_unary(p, 8u); }
    case 0x0Au: { map_unary(p, 9u); }
    case 0x0Bu: { map_unary(p, 10u); }

    case 0x10u: { bin_arith(p, 0u); }
    case 0x11u: { bin_arith(p, 1u); }
    case 0x12u: { bin_arith(p, 2u); }
    case 0x13u: { bin_arith(p, 3u); }
    case 0x14u: { bin_arith(p, 4u); }
    case 0x15u: { bin_arith(p, 5u); }
    case 0x1Fu: { bin_arith(p, 6u); }

    case 0x16u: { cmp_op(p, 0u); }
    case 0x17u: { cmp_op(p, 1u); }
    case 0x18u: { cmp_op(p, 2u); }
    case 0x19u: { cmp_op(p, 3u); }
    case 0x1Au: { cmp_op(p, 4u); }
    case 0x1Bu: { cmp_op(p, 5u); }

    case 0x1Cu: { logic_op(p, 0u); }
    case 0x1Du: { logic_op(p, 1u); }
    case 0x1Eu: { logic_op(p, 2u); }

    case 0x20u: { op_select(p); }

    case 0x40u: { op_cumulative(p, false); }
    case 0x41u: { op_cumulative(p, true); }

    case 0x38u: { op_broadcast(p); }
    case 0x39u: { op_copy(p); }
    case 0x3Au: { op_transpose(p); }

    case 0x60u: { op_gather(p); }
    case 0x61u: { op_gather_row(p); }
    case 0x62u: { op_scatter(p, true); }
    case 0x63u: { op_scatter(p, false); }
    case 0x64u: { op_iota(p); }

    case 0x65u: { op_mask_apply(p); }
    case 0x66u: { op_struct_mask(p, 0u); }
    case 0x67u: { op_struct_mask(p, 1u); }
    case 0x68u: { op_struct_mask(p, 2u); }

    case 0x70u: { op_rng(p, false); }
    case 0x71u: { op_rng(p, true); }

    case 0x50u: { op_sort_desc_finish(p); }
    case 0x51u: { op_top_k_finish(p); }
    case 0x58u: { op_pivot_finish(p); }

    case 0x55u: { op_matmul(p); }

    case 0x81u: { op_const(p); }

    case 0xA1u: { op_copy(p); }
    case 0xA2u: {}
    default: {}
  }
}

fn ptir_reduce_level(p : u32, level : u32) {
  let tag = p_tag(p);
  switch (tag) {
    case 0x30u: { op_reduce_level(p, RED_SUM, level); }
    case 0x31u: { op_reduce_level(p, RED_MAX, level); }
    case 0x32u: { op_reduce_level(p, RED_MIN, level); }
    case 0x33u: { op_reduce_level(p, RED_ARGMAX, level); }
    case 0x34u: { op_reduce_level(p, RED_GUMBEL, level); }
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
    case 0x34u: { op_reduce_finish(p, RED_GUMBEL); }
    default: {}
  }
}

fn ptir_sort_seed(p : u32) {
  if (ptir_is_sort(p)) { op_sort_seed(p); }
}

fn ptir_sort_round(p : u32, r : u32) {
  if (ptir_is_sort(p)) { op_sort_round(p, r); }
}

fn ptir_sort_pre(p : u32) {
  if (ptir_is_sort(p)) { op_sort_pre(p); }
}

fn ptir_pivot_pack(p : u32) {
  if (ptir_is_sort(p)) { op_pivot_pack(p); }
}

fn ptir_is_sort(p : u32) -> bool {
  let tag = p_tag(p);
  return tag == 0x50u || tag == 0x51u || tag == 0x58u;
}

fn ptir_is_reduce(p : u32) -> bool {
  let tag = p_tag(p);
  return tag >= 0x30u && tag <= 0x34u;
}
