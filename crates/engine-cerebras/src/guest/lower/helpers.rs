//! The CSL helper functions a lowered stage calls (`g_*`), by block: a
//! block is declared once in the PE program when any op needs it.

use super::*;

/// The CSL source of a helper block (several functions that call each
/// other travel together).
pub(super) fn helper_source(name: &str) -> String {
    let mut s = String::new();
    match name {
        "g_copy" => {
            s.push_str(
                r"// A word copy through a DSD move: a copy loop would become a
// `memmove` call the PE has no library for.
var g_dummy_u32 = @zeros([1]u32);
const g_base_u32 = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> g_dummy_u32[i] });
var g_dummy_u16 = @zeros([1]u16);
const g_base_u16 = @get_dsd(mem1d_dsd, .{ .tensor_access = |i|{1} -> g_dummy_u16[i] });
fn g_copy16(dst: [*]u16, doff: i32, src: [*]u16, soff: i32, n: i32) void {
  if (n <= 0) { return; }
  var d = @set_dsd_base_addr(g_base_u16, dst);
  d = @set_dsd_length(d, @bitcast(u16, @as(i16, n)));
  d = @increment_dsd_offset(d, @as(i16, doff), u16);
  var s = @set_dsd_base_addr(g_base_u16, src);
  s = @set_dsd_length(s, @bitcast(u16, @as(i16, n)));
  s = @increment_dsd_offset(s, @as(i16, soff), u16);
  @mov16(d, s);
}
fn g_copy32(dst: [*]u32, doff: i32, src: [*]u32, soff: i32, n: i32) void {
  if (n <= 0) { return; }
  var d = @set_dsd_base_addr(g_base_u32, dst);
  d = @set_dsd_length(d, @bitcast(u16, @as(i16, n)));
  d = @increment_dsd_offset(d, @as(i16, doff), u32);
  var s = @set_dsd_base_addr(g_base_u32, src);
  s = @set_dsd_length(s, @bitcast(u16, @as(i16, n)));
  s = @increment_dsd_offset(s, @as(i16, soff), u32);
  @mov32(d, s);
}
",
            );
        }
        "g_cast" => {
            let _ = write!(
                s,
                r"// `as i32`: toward zero, saturating, NaN to 0.
fn g_f2i(x: f32) i32 {{
  if (math.isNaN_f32(x)) {{ return 0; }}
  if (x <= {lo}) {{ return {min}; }}
  if (x >= {hi}) {{ return {max}; }}
  return @as(i32, x);
}}
// `as u32`: toward zero, saturating at 0 and u32::MAX, NaN to 0 (the
// compiler's own conversion saturates at 2^31).
fn g_f2u(x: f32) u32 {{
  if (math.isNaN_f32(x)) {{ return 0; }}
  if (x <= {zero}) {{ return 0; }}
  if (x >= {top}) {{ return @as(u32, 0xFFFFFFFF); }}
  if (x >= {hi}) {{ return @as(u32, @as(i32, x - {hi})) + @as(u32, 0x80000000); }}
  return @as(u32, @as(i32, x));
}}
",
                lo = flit(-2_147_483_648.0),
                hi = flit(2_147_483_648.0),
                top = flit(4_294_967_296.0),
                zero = flit(0.0),
                min = ilit(i32::MIN),
                max = ilit(i32::MAX),
            );
        }
        "g_int" => {
            s.push_str(
                r"// The interpreter's i64 division, truncated back: by zero is 0,
// i32::MIN / -1 wraps and i32::MIN % -1 is 0.
fn g_idiv(x: i32, y: i32) i32 {
  if (y == 0) { return 0; }
  if (y == -1) { return @as(i32, 0) - x; }
  return x / y;
}
// The signed remainder goes through the unsigned one on the magnitudes
// (the compiler's own gave 4 % -5 = -4; Rust: 4); the sign is the
// dividend's, as C and Rust truncate.
fn g_irem(x: i32, y: i32) i32 {
  if (y == 0) { return 0; }
  if (y == -1) { return 0; }
  var ax: u32 = @bitcast(u32, x); if (x < 0) { ax = @as(u32, 0) - ax; }
  var ay: u32 = @bitcast(u32, y); if (y < 0) { ay = @as(u32, 0) - ay; }
  var r: u32 = ax % ay;
  if (x < 0) { return @as(i32, 0) - @bitcast(i32, r); }
  return @bitcast(i32, r);
}
fn g_udiv(x: u32, y: u32) u32 {
  if (y == 0) { return 0; }
  return x / y;
}
fn g_urem(x: u32, y: u32) u32 {
  if (y == 0) { return 0; }
  return x % y;
}
fn g_imax(x: i32, y: i32) i32 { if (x > y) { return x; } return y; }
fn g_imin(x: i32, y: i32) i32 { if (x < y) { return x; } return y; }
fn g_umax(x: u32, y: u32) u32 { if (x > y) { return x; } return y; }
fn g_umin(x: u32, y: u32) u32 { if (x < y) { return x; } return y; }
fn g_iclamp(x: i32, lo: i32, hi: i32) i32 {
  if (x < lo) { return lo; }
  if (x > hi) { return hi; }
  return x;
}
fn g_uclamp(x: u32, hi: i32) i32 {
  if (x > @as(u32, hi)) { return hi; }
  return @as(i32, x);
}
",
            );
        }
        "g_fmod" => {
            s.push_str(
                r"// `x % y` as Rust computes it for f32 (C `fmod`: exact, the sign of `x`).
fn g_fmod(x: f32, y: f32) f32 {
  if (math.isNaN_f32(y) or math.isNaN_f32(x)) { return x + y; }
  var ax: f32 = math.abs_f32(x);
  var ay: f32 = math.abs_f32(y);
  if (ay == 0.0 or math.isInf_f32(x)) { return @bitcast(f32, @as(u32, 0x7FC00000)); }
  if (math.isInf_f32(y) or ax < ay) { return x; }
  // Subtract y's largest power-of-two multiple at or below the rest.
  var r: f32 = ax;
  while (r >= ay) {
    var er: i32 = @as(i32, (@bitcast(u32, r) >> 23) & 0xFF);
    var ed: i32 = @as(i32, (@bitcast(u32, ay) >> 23) & 0xFF);
    var t: f32 = ay;
    var e: i32 = er - ed;
    while (e > 0) : (e -= 1) { t = t * 2.0; }
    if (t > r) { t = t * 0.5; }
    r = r - t;
  }
  return @bitcast(f32, (@bitcast(u32, r) & 0x7FFFFFFF) | (@bitcast(u32, x) & 0x80000000));
}
",
            );
        }
        "g_tree" => {
            s.push_str(
                r"// The interpreter's canonical float max/min: a NaN loses to the other,
// two NaNs give the identity, zeros keep the sign the fold states.
fn g_cmax(l: f32, r: f32) f32 {
  var ln: bool = math.isNaN_f32(l); var rn: bool = math.isNaN_f32(r);
  if (ln and rn) { return @bitcast(f32, @as(u32, 0xFF800000)); }
  if (ln) { return r; }
  if (rn) { return l; }
  if (l == 0.0 and r == 0.0) {
    if ((@bitcast(u32, l) & 0x80000000) != 0 and (@bitcast(u32, r) & 0x80000000) != 0) { return @bitcast(f32, @as(u32, 0x80000000)); }
    return 0.0;
  }
  if (l > r) { return l; }
  return r;
}
fn g_cmin(l: f32, r: f32) f32 {
  var ln: bool = math.isNaN_f32(l); var rn: bool = math.isNaN_f32(r);
  if (ln and rn) { return @bitcast(f32, @as(u32, 0x7F800000)); }
  if (ln) { return r; }
  if (rn) { return l; }
  if (l == 0.0 and r == 0.0) {
    if ((@bitcast(u32, l) & 0x80000000) != 0 or (@bitcast(u32, r) & 0x80000000) != 0) { return @bitcast(f32, @as(u32, 0x80000000)); }
    return 0.0;
  }
  if (l < r) { return l; }
  return r;
}
// `f32::max` / `f32::min`: a NaN operand yields the other one.
fn g_fmax(a: f32, b: f32) f32 { if (math.isNaN_f32(a)) { return b; } if (math.isNaN_f32(b)) { return a; } if (a > b) { return a; } return b; }
fn g_fmin(a: f32, b: f32) f32 { if (math.isNaN_f32(a)) { return b; } if (math.isNaN_f32(b)) { return a; } if (a < b) { return a; } return b; }
// The interpreter's 32-lane tree fold over row `off..off+n` of `src`:
// blocks of 32 lanes fold by halving (16, 8, 4, 2, 1), their results fold
// the same way level by level. `kind`: 0 sum, 1 max, 2 min.
var g_lanes = @zeros([32]f32);
fn g_fold32(kind: i32) void {
  var offset: i32 = 16;
  while (offset >= 1) : (offset = offset / 2) {
    var lane: i32 = 0;
    while (lane < offset) : (lane += 1) {
      var l: f32 = g_lanes[lane];
      var r: f32 = g_lanes[lane + offset];
      if (kind == 0) { g_lanes[lane] = l + r; }
      else if (kind == 1) { g_lanes[lane] = g_cmax(l, r); }
      else { g_lanes[lane] = g_cmin(l, r); }
    }
  }
}
fn g_tfold(src: [*]f32, off: i32, n: i32, lvl: [*]f32, kind: i32) f32 {
  var ident: f32 = 0.0;
  if (kind == 1) { ident = @bitcast(f32, @as(u32, 0xFF800000)); }
  if (kind == 2) { ident = @bitcast(f32, @as(u32, 0x7F800000)); }
  if (n == 0) { return ident; }
  if (n == 1) { return src[off]; }
  var count: i32 = 0;
  var base: i32 = 0;
  while (base < n) : (base += 32) {
    var c: i32 = 32; if (n - base < c) { c = n - base; }
    var q: i32 = 0;
    while (q < 32) : (q += 1) { if (q < c) { g_lanes[q] = src[off + base + q]; } else { g_lanes[q] = ident; } }
    g_fold32(kind);
    lvl[count] = g_lanes[0];
    count = count + 1;
  }
  while (count > 1) {
    var next: i32 = 0;
    var b2: i32 = 0;
    while (b2 < count) : (b2 += 32) {
      var c: i32 = 32; if (count - b2 < c) { c = count - b2; }
      var q: i32 = 0;
      while (q < 32) : (q += 1) { if (q < c) { g_lanes[q] = lvl[b2 + q]; } else { g_lanes[q] = ident; } }
      g_fold32(kind);
      lvl[next] = g_lanes[0];
      next = next + 1;
    }
    count = next;
  }
  return lvl[0];
}
fn g_tsum(src: [*]f32, off: i32, n: i32, lvl: [*]f32) f32 { return g_tfold(src, off, n, lvl, 0); }
fn g_tmax(src: [*]f32, off: i32, n: i32, lvl: [*]f32) f32 { return g_tfold(src, off, n, lvl, 1); }
fn g_tmin(src: [*]f32, off: i32, n: i32, lvl: [*]f32) f32 { return g_tfold(src, off, n, lvl, 2); }
",
            );
        }
        "g_sort" => {
            s.push_str(
                r"// An i32 whose order is the interpreter's descending-sort order of an
// f32: monotone in the value, -0 and +0 one key, NaN below -inf.
fn g_okey(x: f32) i32 {
  if (math.isNaN_f32(x)) { return @bitcast(i32, @as(u32, 0x80000000)); }
  var v: f32 = x;
  if (v == 0.0) { v = 0.0; }
  var bits: i32 = @bitcast(i32, v);
  if (bits < 0) { return bits ^ 0x7FFFFFFF; }
  return bits;
}
// Row `off..off+cap` of `src` as order keys with their source indices;
// lanes past `n` (a block's beyond this PE's share) key as NaN: last.
fn g_keys(src: [*]f32, off: i32, n: i32, cap: i32, key: [*]i32, idx: [*]u32) void {
  var j: i32 = 0;
  while (j < cap) : (j += 1) {
    if (j < n) { key[j] = g_okey(src[off + j]); } else { key[j] = @bitcast(i32, @as(u32, 0x80000000)); }
    idx[j] = @as(u32, j);
  }
}
// Keys at or above `t` (strictly above for `g_count_gt`) in the
// descending-sorted row `off..off+n`.
fn g_count_ge(key: [*]i32, off: i32, n: i32, t: i32) i32 {
  var lo: i32 = 0; var hi: i32 = n;
  while (lo < hi) { var mid: i32 = lo + (hi - lo) / 2; if (key[off + mid] >= t) { lo = mid + 1; } else { hi = mid; } }
  return lo;
}
fn g_count_gt(key: [*]i32, off: i32, n: i32, t: i32) i32 {
  var lo: i32 = 0; var hi: i32 = n;
  while (lo < hi) { var mid: i32 = lo + (hi - lo) / 2; if (key[off + mid] > t) { lo = mid + 1; } else { hi = mid; } }
  return lo;
}
// The value whose order key is `k` (`g_okey`'s inverse; NaN's key gives
// a NaN).
fn g_unkey(k: i32) f32 {
  if (k < 0) { return @bitcast(f32, k ^ 0x7FFFFFFF); }
  return @bitcast(f32, k);
}
// The greatest last-offered key among the PEs whose `m` candidates are
// all taken (row `rr` of `recv`, `per_pe` words a PE, candidate values):
// a pick at or below it might have been beaten by a key that PE did not
// offer.
fn g_exhausted_value(recv: [*]u32, per_pe: i32, m: i32, rr: i32, taken: [*]u16, cols: i32) i32 {
  var best: i32 = @bitcast(i32, @as(u32, 0x80000000)); var any: bool = false;
  var c: i32 = 0;
  while (c < cols) : (c += 1) {
    if (@as(i32, taken[c]) < m) { continue; }
    var kk: i32 = g_okey(@bitcast(f32, recv[c * per_pe + rr * m + m - 1]));
    if (!any or kk > best) { best = kk; any = true; }
  }
  return best;
}
// A stable merge sort of (key, idx) by key descending: equal keys keep
// their index order. `tk`/`ti` are scratch of n; the result lands in
// `key`/`idx`.
fn g_msort(key: [*]i32, idx: [*]u32, n: i32, tk: [*]i32, ti: [*]u32) void {
  var width: i32 = 1;
  var in_tmp: bool = false;
  while (width < n) : (width = width * 2) {
    var lo: i32 = 0;
    while (lo < n) : (lo += width * 2) {
      var mid: i32 = lo + width; if (mid > n) { mid = n; }
      var hi: i32 = lo + width * 2; if (hi > n) { hi = n; }
      var a: i32 = lo; var b: i32 = mid; var o: i32 = lo;
      while (o < hi) : (o += 1) {
        var take_left: bool = false;
        if (a < mid and b < hi) {
          if (in_tmp) { take_left = tk[a] >= tk[b]; } else { take_left = key[a] >= key[b]; }
        } else { take_left = a < mid; }
        if (take_left) {
          if (in_tmp) { key[o] = tk[a]; idx[o] = ti[a]; } else { tk[o] = key[a]; ti[o] = idx[a]; }
          a = a + 1;
        } else {
          if (in_tmp) { key[o] = tk[b]; idx[o] = ti[b]; } else { tk[o] = key[b]; ti[o] = idx[b]; }
          b = b + 1;
        }
      }
    }
    in_tmp = !in_tmp;
  }
  if (in_tmp) {
    var j: i32 = 0;
    while (j < n) : (j += 1) { key[j] = tk[j]; idx[j] = ti[j]; }
  }
}
",
            );
        }
        "g_rng" => {
            let f = rng::RNG_FORMULA;
            let mut rounds = String::new();
            for round in f.splitmix64_rounds {
                let _ = writeln!(rounds, "  x = x ^ (x >> {});", round.xor_shift);
                if let Some(m) = round.multiplier {
                    let _ = writeln!(rounds, "  x = x * {};", u64lit(m));
                }
            }
            let _ = write!(
                s,
                r"// splitmix64, and the uniform draw `hash_uniform(seed, j)`, bit for bit.
fn g_splitmix(v: u64) u64 {{
  var x: u64 = v;
{rounds}  return x;
}}
fn g_uniform(seed: u64, j: u32) f32 {{
  var x: u64 = seed + {stride} * (@as(u64, j) + {bias});
  var bits: u32 = @as(u32, g_splitmix(x) >> {shift});
  var raw: f32 = (@as(f32, bits) + {mid}) * {unit};
  if (raw < {cap}) {{ return raw; }}
  return {cap};
}}
",
                stride = u64lit(f.lane_stride),
                bias = u64lit(f.lane_index_bias),
                shift = f.uniform_mantissa_shift,
                mid = flit(f.uniform_midpoint),
                unit = flit(1.0 / (1u32 << f.uniform_mantissa_bits) as f32),
                cap = flit(rng::UNIFORM_MAX),
            );
        }
        other => panic!("no guest helper {other}"),
    }
    s
}
