

const PTIR_WG : u32 = 256u;

const DESC_WORDS : u32 = 9u;

const PARAM_WORDS : u32 = 16u;

const DT_F32 : u32 = 0u;
const DT_I32 : u32 = 1u;
const DT_U32 : u32 = 2u;
const DT_BOOL : u32 = 3u;

struct Cfg {

  value_count : u32,

  temporary : u32,

  op_count : u32,

  lane : u32,
}

@group(0) @binding(0) var<storage, read_write> status  : array<u32>;
@group(0) @binding(1) var<storage, read>       descs   : array<u32>;
@group(0) @binding(2) var<storage, read>       params  : array<u32>;
@group(0) @binding(3) var<storage, read>       offs    : array<u32>;
@group(0) @binding(4) var<storage, read_write> heap    : array<u32>;
@group(0) @binding(5) var<uniform>             cfg     : Cfg;

var<private> tid : u32;

var<private> lanes : u32;


fn d_len(v : u32) -> u32 { return descs[v * DESC_WORDS + 0u]; }
fn d_rows(v : u32) -> u32 { return descs[v * DESC_WORDS + 1u]; }
fn d_rank(v : u32) -> u32 { return descs[v * DESC_WORDS + 3u]; }
fn d_dtype(v : u32) -> u32 { return descs[v * DESC_WORDS + 4u]; }
fn d_dim(v : u32, k : u32) -> u32 { return descs[v * DESC_WORDS + 5u + k]; }

fn base(v : u32) -> u32 { return offs[v] >> 2u; }

fn rows_of(v : u32) -> u32 {
  let rank = d_rank(v);
  if (rank < 2u) { return 1u; }
  var n : u32 = 1u;
  for (var k : u32 = 0u; k + 1u < rank; k = k + 1u) { n = n * d_dim(v, k); }
  return n;
}


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
fn p_pred_payload(p : u32) -> u32 { return p_a1(p); }
fn p_dtype(p : u32) -> u32 { return p_at(p, 14u); }


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

fn st_b_word(v : u32, w : u32, n : u32, b0 : bool, b1 : bool, b2 : bool, b3 : bool) {
  var packed : u32 = 0u;
  if (b0 && (w * 4u + 0u) < n) { packed = packed | 0x000000FFu; }
  if (b1 && (w * 4u + 1u) < n) { packed = packed | 0x0000FF00u; }
  if (b2 && (w * 4u + 2u) < n) { packed = packed | 0x00FF0000u; }
  if (b3 && (w * 4u + 3u) < n) { packed = packed | 0xFF000000u; }
  heap[base(v) + w] = packed;
}

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
    default: { return x; }
  }
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
    case 7u: { return u32(sin(ld_f(a, j))); }
    case 8u: { return u32(cos(ld_f(a, j))); }
    case 9u: { return u32(sqrt(ld_f(a, j))); }
    case 10u: { return u32(1.0 / sqrt(ld_f(a, j))); }
    default: { return ld_u(a, j); }
  }
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

fn POS_INF() -> f32 { return bitcast<f32>(0x7F800000u); }
fn NEG_INF() -> f32 { return bitcast<f32>(0xFF800000u); }
// The literal `-0.0` folds to `+0.0` under Tint (Chrome), where a runtime
// negation keeps its sign; the bits are read through a `var<private>` so no
// compiler folds them (the same route `for_tint` takes for the infinities).
var<private> pie_neg_zero_bits : u32 = 0x80000000u;
fn NEG_ZERO() -> f32 { return bitcast<f32>(pie_neg_zero_bits); }

fn cmax(l : f32, r : f32) -> f32 {
  let ln = l != l;
  let rn = r != r;
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
  let ln = l != l;
  let rn = r != r;
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

fn red_stride(len : u32) -> u32 { return (len + 31u) / 32u; }

fn tmp_v(row : u32, stride : u32, c : u32, buf : u32, span : u32) -> u32 {
  return cfg.temporary + (buf * span + row * stride + c) * 2u;
}

fn red_read_buf(level : u32) -> u32 { return level % 2u; }

fn red_write_buf(level : u32) -> u32 { return (level + 1u) % 2u; }

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

fn arg_takes(lv : f32, li : u32, rv : f32, ri : u32, level : u32, count : u32, lane : u32, off : u32) -> bool {
  if (lane + off >= count) { return false; }
  let lnan = lv != lv;
  let rnan = rv != rv;
  if (rnan) { return false; }
  if (lnan) { return true; }
  return rv > lv || (rv == lv && ri < li);
}

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

      v = ld_f(a, row * len);
      idx = 0u;
    } else {

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
  if (x != x) { return 0xFFFFFFFFu; }

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
  return cfg.temporary + ((buf * n) + c) * 2u;
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
    heap[c] = sort_key(ld_f(a, i));

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

fn pivot_flag(n : u32, t : u32) -> u32 { return cfg.temporary + t; }
fn pivot_meta(n : u32, row : u32) -> u32 { return cfg.temporary + n + row; }
fn pivot_val(n : u32, t : u32) -> u32 { return cfg.temporary + 2u * n + t; }
fn pivot_pos(n : u32, i : u32) -> u32 { return cfg.temporary + 3u * n + i; }

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
    let idx = heap[sort_cell(0u, n, c) + 1u];
    heap[pivot_val(n, c)] = bitcast<u32>(ld_f(a, rowb + idx));

    heap[pivot_pos(n, rowb + idx)] = c - rowb;
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
        let t = heap[pivot_pos(n, i)];
        on = t < heap[pivot_meta(n, row)] && heap[pivot_flag(n, row * len + t)] != 0u;
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

      loop {
        if (t >= len) { break; }
        heap[flags + t] = select(0u, 1u, excl < cut);
        excl = excl + bitcast<f32>(heap[vals + t]);
        t = t + 1u;
        if (!(excl < cut) && monotone) { break; }
      }

      heap[pivot_meta(n, row)] = t;
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
  return tag >= 0x30u && tag <= 0x33u;
}
