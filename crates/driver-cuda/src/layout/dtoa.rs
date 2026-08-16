//! `nlohmann::json`'s double formatter, digit for digit.
//!
//! Not `format!("{v}")`: Rust emits the shortest round-tripping decimal while
//! `nlohmann` uses Grisu2, and the two disagree on ~0.1% of values. Both
//! round-trip, but the profile cache is a file the C++ and this crate both
//! read-merge-rewrite, so a disagreeing formatter would rewrite untouched
//! entries and make every write a spurious whole-file diff. Kept recognisable
//! against `nlohmann/json.hpp`'s `dtoa_impl` (Grisu2); unsigned ops are
//! `wrapping_*` because the rounding loop relies on C++ unsigned wraparound
//! (Rust would panic in debug).

use std::fmt::Write as _;

/// A 64-bit "do-it-yourself floating point" value, `f * 2^e`.
#[derive(Clone, Copy, Debug)]
struct DiyFp {
    f: u64,
    e: i32,
}

impl DiyFp {
    /// `x - y`. Requires `x.e == y.e` and `x.f >= y.f`.
    fn sub(x: Self, y: Self) -> Self {
        Self {
            f: x.f.wrapping_sub(y.f),
            e: x.e,
        }
    }

    /// `x * y`, keeping the upper 64 bits, rounded with ties away from zero.
    fn mul(x: Self, y: Self) -> Self {
        let u_lo = x.f & 0xFFFF_FFFF;
        let u_hi = x.f >> 32;
        let v_lo = y.f & 0xFFFF_FFFF;
        let v_hi = y.f >> 32;

        let p0 = u_lo * v_lo;
        let p1 = u_lo * v_hi;
        let p2 = u_hi * v_lo;
        let p3 = u_hi * v_hi;

        let p0_hi = p0 >> 32;
        let p1_lo = p1 & 0xFFFF_FFFF;
        let p1_hi = p1 >> 32;
        let p2_lo = p2 & 0xFFFF_FFFF;
        let p2_hi = p2 >> 32;

        let mut q = p0_hi + p1_lo + p2_lo;
        // Round, ties up.
        q += 1u64 << 31;

        let h = p3
            .wrapping_add(p2_hi)
            .wrapping_add(p1_hi)
            .wrapping_add(q >> 32);
        Self {
            f: h,
            e: x.e + y.e + 64,
        }
    }

    /// Shift left until the significand's top bit is set. Requires `f != 0`.
    fn normalize(mut x: Self) -> Self {
        while (x.f >> 63) == 0 {
            x.f <<= 1;
            x.e -= 1;
        }
        x
    }

    /// Shift left so the exponent becomes `target`. Requires `target <= x.e`.
    fn normalize_to(x: Self, target: i32) -> Self {
        let delta = x.e - target;
        Self {
            f: x.f << delta,
            e: target,
        }
    }
}

/// `v` normalised, plus the two boundaries `m-` and `m+` between which every
/// real number rounds to `v`.
struct Boundaries {
    w: DiyFp,
    minus: DiyFp,
    plus: DiyFp,
}

/// IEEE-754 binary64 constants.
const PRECISION: i32 = 53;
/// `max_exponent - 1 + (PRECISION - 1)` = `1024 - 1 + 52`.
const BIAS: i32 = 1075;
/// The exponent of a subnormal, `1 - BIAS`.
const MIN_EXP: i32 = 1 - BIAS;
const HIDDEN_BIT: u64 = 1u64 << (PRECISION - 1);

/// Requires `value` finite and strictly positive.
fn compute_boundaries(value: f64) -> Boundaries {
    let bits = value.to_bits();
    let e_field = bits >> (PRECISION - 1);
    let f_field = bits & (HIDDEN_BIT - 1);

    let v = if e_field == 0 {
        DiyFp {
            f: f_field,
            e: MIN_EXP,
        }
    } else {
        DiyFp {
            f: f_field + HIDDEN_BIT,
            // The exponent field of an f64 is 11 bits, so this always fits.
            e: e_field as i32 - BIAS,
        }
    };

    // A power of two that is not the smallest normal has a closer predecessor
    // than successor, so its lower boundary is half the usual distance away.
    let lower_boundary_is_closer = f_field == 0 && e_field > 1;

    let m_plus = DiyFp {
        f: 2 * v.f + 1,
        e: v.e - 1,
    };
    let m_minus = if lower_boundary_is_closer {
        DiyFp {
            f: 4 * v.f - 1,
            e: v.e - 2,
        }
    } else {
        DiyFp {
            f: 2 * v.f - 1,
            e: v.e - 1,
        }
    };

    let w_plus = DiyFp::normalize(m_plus);
    let w_minus = DiyFp::normalize_to(m_minus, w_plus.e);
    Boundaries {
        w: DiyFp::normalize(v),
        minus: w_minus,
        plus: w_plus,
    }
}

const ALPHA: i32 = -60;

/// A cached power of ten, `f * 2^e ~= 10^k`.
#[derive(Clone, Copy)]
struct CachedPower {
    f: u64,
    e: i32,
    k: i32,
}

const CACHED_POWERS_MIN_DEC_EXP: i32 = -300;
const CACHED_POWERS_DEC_STEP: i32 = 8;

/// Every eighth power of ten from `10^-300` to `10^324`, as normalised
/// `DiyFp`s. Eight is the widest spacing for which some entry always lands the
/// product's exponent inside `[ALPHA, GAMMA]`.
#[rustfmt::skip]
const CACHED_POWERS: [CachedPower; 79] = {
    macro_rules! p {
        ($($f:expr, $e:expr, $k:expr;)*) => { [$(CachedPower { f: $f, e: $e, k: $k },)*] };
    }
    p![
        0xAB70_FE17_C79A_C6CA, -1060, -300; 0xFF77_B1FC_BEBC_DC4F, -1034, -292;
        0xBE56_91EF_416B_D60C, -1007, -284; 0x8DD0_1FAD_907F_FC3C,  -980, -276;
        0xD351_5C28_3155_9A83,  -954, -268; 0x9D71_AC8F_ADA6_C9B5,  -927, -260;
        0xEA9C_2277_23EE_8BCB,  -901, -252; 0xAECC_4991_4078_536D,  -874, -244;
        0x823C_1279_5DB6_CE57,  -847, -236; 0xC210_9436_4DFB_5637,  -821, -228;
        0x9096_EA6F_3848_984F,  -794, -220; 0xD774_85CB_2582_3AC7,  -768, -212;
        0xA086_CFCD_97BF_97F4,  -741, -204; 0xEF34_0A98_172A_ACE5,  -715, -196;
        0xB238_67FB_2A35_B28E,  -688, -188; 0x84C8_D4DF_D2C6_3F3B,  -661, -180;
        0xC5DD_4427_1AD3_CDBA,  -635, -172; 0x936B_9FCE_BB25_C996,  -608, -164;
        0xDBAC_6C24_7D62_A584,  -582, -156; 0xA3AB_6658_0D5F_DAF6,  -555, -148;
        0xF3E2_F893_DEC3_F126,  -529, -140; 0xB5B5_ADA8_AAFF_80B8,  -502, -132;
        0x8762_5F05_6C7C_4A8B,  -475, -124; 0xC9BC_FF60_34C1_3053,  -449, -116;
        0x964E_858C_91BA_2655,  -422, -108; 0xDFF9_7724_7029_7EBD,  -396, -100;
        0xA6DF_BD9F_B8E5_B88F,  -369,  -92; 0xF8A9_5FCF_8874_7D94,  -343,  -84;
        0xB944_7093_8FA8_9BCF,  -316,  -76; 0x8A08_F0F8_BF0F_156B,  -289,  -68;
        0xCDB0_2555_6531_31B6,  -263,  -60; 0x993F_E2C6_D07B_7FAC,  -236,  -52;
        0xE45C_10C4_2A2B_3B06,  -210,  -44; 0xAA24_2499_6973_92D3,  -183,  -36;
        0xFD87_B5F2_8300_CA0E,  -157,  -28; 0xBCE5_0864_9211_1AEB,  -130,  -20;
        0x8CBC_CC09_6F50_88CC,  -103,  -12; 0xD1B7_1758_E219_652C,   -77,   -4;
        0x9C40_0000_0000_0000,   -50,    4; 0xE8D4_A510_0000_0000,   -24,   12;
        0xAD78_EBC5_AC62_0000,     3,   20; 0x813F_3978_F894_0984,    30,   28;
        0xC097_CE7B_C907_15B3,    56,   36; 0x8F7E_32CE_7BEA_5C70,    83,   44;
        0xD5D2_38A4_ABE9_8068,   109,   52; 0x9F4F_2726_179A_2245,   136,   60;
        0xED63_A231_D4C4_FB27,   162,   68; 0xB0DE_6538_8CC8_ADA8,   189,   76;
        0x83C7_088E_1AAB_65DB,   216,   84; 0xC45D_1DF9_4271_1D9A,   242,   92;
        0x924D_692C_A61B_E758,   269,  100; 0xDA01_EE64_1A70_8DEA,   295,  108;
        0xA26D_A399_9AEF_774A,   322,  116; 0xF209_787B_B47D_6B85,   348,  124;
        0xB454_E4A1_79DD_1877,   375,  132; 0x865B_8692_5B9B_C5C2,   402,  140;
        0xC835_53C5_C896_5D3D,   428,  148; 0x952A_B45C_FA97_A0B3,   455,  156;
        0xDE46_9FBD_99A0_5FE3,   481,  164; 0xA59B_C234_DB39_8C25,   508,  172;
        0xF6C6_9A72_A398_9F5C,   534,  180; 0xB7DC_BF53_54E9_BECE,   561,  188;
        0x88FC_F317_F222_41E2,   588,  196; 0xCC20_CE9B_D35C_78A5,   614,  204;
        0x9816_5AF3_7B21_53DF,   641,  212; 0xE2A0_B5DC_971F_303A,   667,  220;
        0xA8D9_D153_5CE3_B396,   694,  228; 0xFB9B_7CD9_A4A7_443C,   720,  236;
        0xBB76_4C4C_A7A4_4410,   747,  244; 0x8BAB_8EEF_B640_9C1A,   774,  252;
        0xD01F_EF10_A657_842C,   800,  260; 0x9B10_A4E5_E991_3129,   827,  268;
        0xE710_9BFB_A19C_0C9D,   853,  276; 0xAC28_20D9_623B_F429,   880,  284;
        0x8044_4B5E_7AA7_CF85,   907,  292; 0xBF21_E440_03AC_DD2D,   933,  300;
        0x8E67_9C2F_5E44_FF8F,   960,  308; 0xD433_179D_9C8C_B841,   986,  316;
        0x9E19_DB92_B4E3_1BA9,  1013,  324;
    ]
};

/// The cached power whose product with a `DiyFp` of binary exponent `e` lands
/// in `[ALPHA, GAMMA]`.
fn cached_power_for_binary_exponent(e: i32) -> CachedPower {
    // k = ceil((ALPHA - e - 1) * log10(2)), with 78913/2^18 standing in for
    // log10(2) and the `+1` supplying the ceiling for positive f.
    let f = ALPHA - e - 1;
    let k = (f * 78913) / (1 << 18) + i32::from(f > 0);
    let index =
        (-CACHED_POWERS_MIN_DEC_EXP + k + (CACHED_POWERS_DEC_STEP - 1)) / CACHED_POWERS_DEC_STEP;
    #[expect(
        clippy::cast_sign_loss,
        reason = "the binary exponent of a finite f64 keeps this in 0..79"
    )]
    CACHED_POWERS[index as usize]
}

/// For `n != 0`, the `pow10` and `k` with `10^(k-1) = pow10 <= n < 10^k`.
/// For `n == 0`, `(1, 1)`.
fn find_largest_pow10(n: u32) -> (u32, i32) {
    if n >= 1_000_000_000 {
        (1_000_000_000, 10)
    } else if n >= 100_000_000 {
        (100_000_000, 9)
    } else if n >= 10_000_000 {
        (10_000_000, 8)
    } else if n >= 1_000_000 {
        (1_000_000, 7)
    } else if n >= 100_000 {
        (100_000, 6)
    } else if n >= 10_000 {
        (10_000, 5)
    } else if n >= 1_000 {
        (1_000, 4)
    } else if n >= 100 {
        (100, 3)
    } else if n >= 10 {
        (10, 2)
    } else {
        (1, 1)
    }
}

/// Walk the last digit down while doing so moves the result closer to `w`
/// without leaving the rounding interval.
fn grisu2_round(buf: &mut [u8], len: usize, dist: u64, delta: u64, rest: u64, ten_k: u64) {
    let mut rest = rest;
    while rest < dist
        && delta.wrapping_sub(rest) >= ten_k
        && (rest.wrapping_add(ten_k) < dist
            || dist.wrapping_sub(rest) > rest.wrapping_add(ten_k).wrapping_sub(dist))
    {
        buf[len - 1] -= 1;
        rest = rest.wrapping_add(ten_k);
    }
}

/// Generate `V = buf * 10^decimal_exponent` with `M- <= V <= M+`.
fn grisu2_digit_gen(
    buf: &mut [u8],
    len: &mut usize,
    decimal_exponent: &mut i32,
    m_minus: DiyFp,
    w: DiyFp,
    m_plus: DiyFp,
) {
    let mut delta = DiyFp::sub(m_plus, m_minus).f;
    let mut dist = DiyFp::sub(m_plus, w).f;

    let one = DiyFp {
        f: 1u64 << -m_plus.e,
        e: m_plus.e,
    };
    let shift = (-one.e) as u32;

    #[expect(
        clippy::cast_possible_truncation,
        reason = "-one.e >= 32, so the high half is what remains"
    )]
    let mut p1 = (m_plus.f >> shift) as u32;
    let mut p2 = m_plus.f & (one.f - 1);

    // Integral part: emit digits from the most significant down.
    let (mut pow10, k) = find_largest_pow10(p1);
    let mut n = k;
    while n > 0 {
        let d = p1 / pow10;
        let r = p1 % pow10;
        buf[*len] = b'0' + u8::try_from(d).unwrap_or(b'0');
        *len += 1;
        p1 = r;
        n -= 1;

        let rest = (u64::from(p1) << shift).wrapping_add(p2);
        if rest <= delta {
            *decimal_exponent += n;
            let ten_n = u64::from(pow10) << shift;
            grisu2_round(buf, *len, dist, delta, rest, ten_n);
            return;
        }
        pow10 /= 10;
    }

    // Fractional part: multiply out one digit at a time until what remains is
    // smaller than the width of the rounding interval.
    let mut m = 0i32;
    loop {
        p2 = p2.wrapping_mul(10);
        let d = p2 >> shift;
        let r = p2 & (one.f - 1);
        buf[*len] = b'0' + u8::try_from(d).unwrap_or(b'0');
        *len += 1;
        p2 = r;
        m += 1;
        delta = delta.wrapping_mul(10);
        dist = dist.wrapping_mul(10);
        if p2 <= delta {
            break;
        }
    }
    *decimal_exponent -= m;
    let ten_m = one.f;
    grisu2_round(buf, *len, dist, delta, p2, ten_m);
}

/// The shortest-ish digits and decimal exponent for a finite, positive `value`.
fn grisu2(buf: &mut [u8], value: f64) -> (usize, i32) {
    let w = compute_boundaries(value);
    let mut len = 0usize;
    let mut decimal_exponent;

    let cached = cached_power_for_binary_exponent(w.plus.e);
    let c_minus_k = DiyFp {
        f: cached.f,
        e: cached.e,
    };

    let big_w = DiyFp::mul(w.w, c_minus_k);
    let w_minus = DiyFp::mul(w.minus, c_minus_k);
    let w_plus = DiyFp::mul(w.plus, c_minus_k);

    // Shrink the interval by one ulp on each side, so that a value generated
    // inside it is inside the *true* interval despite the rounding above.
    let m_minus = DiyFp {
        f: w_minus.f + 1,
        e: w_minus.e,
    };
    let m_plus = DiyFp {
        f: w_plus.f - 1,
        e: w_plus.e,
    };

    decimal_exponent = -cached.k;
    grisu2_digit_gen(buf, &mut len, &mut decimal_exponent, m_minus, big_w, m_plus);
    (len, decimal_exponent)
}

/// Fixed-point below `10^MAX_FMT_EXP`, scientific above. `digits10` for
/// binary64, exactly as `to_chars` passes it.
const MAX_FMT_EXP: i32 = 15;
/// Fixed-point down to `10^MIN_FMT_EXP`, scientific below.
const MIN_FMT_EXP: i32 = -4;

/// Lay the digits out the way `format_buffer` does.
fn format_buffer(out: &mut String, digits: &[u8], decimal_exponent: i32) {
    let k = i32::try_from(digits.len()).unwrap_or(i32::MAX);
    let n = k + decimal_exponent;
    let s = |d: &[u8]| String::from_utf8_lossy(d).into_owned();

    if k <= n && n <= MAX_FMT_EXP {
        // digits[000].0 — integral values still get a `.0` to read back as double.
        out.push_str(&s(digits));
        #[expect(clippy::cast_sign_loss, reason = "guarded by k <= n")]
        for _ in 0..((n - k) as usize) {
            out.push('0');
        }
        out.push_str(".0");
        return;
    }
    if 0 < n && n <= MAX_FMT_EXP {
        // digits[.]digits
        #[expect(clippy::cast_sign_loss, reason = "guarded by 0 < n < k")]
        let split = n as usize;
        out.push_str(&s(&digits[..split]));
        out.push('.');
        out.push_str(&s(&digits[split..]));
        return;
    }
    if MIN_FMT_EXP < n && n <= 0 {
        // 0.[000]digits
        out.push_str("0.");
        #[expect(clippy::cast_sign_loss, reason = "guarded by n <= 0")]
        for _ in 0..((-n) as usize) {
            out.push('0');
        }
        out.push_str(&s(digits));
        return;
    }
    // d.dddde+XX
    out.push(char::from(digits[0]));
    if digits.len() > 1 {
        out.push('.');
        out.push_str(&s(&digits[1..]));
    }
    out.push('e');
    append_exponent(out, n - 1);
}

/// The exponent, signed, with at least two digits.
fn append_exponent(out: &mut String, e: i32) {
    if e < 0 {
        out.push('-');
    } else {
        out.push('+');
    }
    let k = e.unsigned_abs();
    if k < 10 {
        let _ = write!(out, "0{k}");
    } else {
        let _ = write!(out, "{k}");
    }
}

/// Append `v` exactly as `nlohmann::json::dump()` would.
///
/// Non-finite values become `null`, as `dump` emits (JSON cannot spell
/// infinity, and `nlohmann` chose the lossy encoding over throwing).
pub fn write_f64(out: &mut String, v: f64) {
    if !v.is_finite() {
        out.push_str("null");
        return;
    }
    let value = if v.is_sign_negative() {
        out.push('-');
        -v
    } else {
        v
    };
    // Both zeroes land here, and both print `0.0` after the sign above.
    if value == 0.0 {
        out.push_str("0.0");
        return;
    }
    let mut buf = [0u8; 32];
    let (len, decimal_exponent) = grisu2(&mut buf, value);
    format_buffer(out, &buf[..len], decimal_exponent);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(v: f64) -> String {
        let mut s = String::new();
        write_f64(&mut s, v);
        s
    }

    #[test]
    fn integral_values_keep_a_fractional_part() {
        // `1` would read back as an integer; the schema's readers are type-strict.
        assert_eq!(f(0.0), "0.0");
        assert_eq!(f(1.0), "1.0");
        assert_eq!(f(100.0), "100.0");
        assert_eq!(f(12345.0), "12345.0");
    }

    #[test]
    fn negative_zero_keeps_its_sign() {
        // Sign from the sign bit, not a comparison, since `-0.0 == 0.0`.
        assert_eq!(f(-0.0), "-0.0");
        assert_ne!(f(-0.0), f(0.0));
    }

    #[test]
    fn the_fixed_point_window_matches_nlohmanns() {
        // 10^15 is the first scientific value and 10^-4 the last fixed-point one.
        assert_eq!(f(1e14), "100000000000000.0");
        assert_eq!(f(1e15), "1e+15");
        assert_eq!(f(1e-4), "0.0001");
        assert_eq!(f(1e-5), "1e-05");
    }

    #[test]
    fn exponents_are_at_least_two_digits() {
        assert_eq!(f(1e-5), "1e-05");
        assert_eq!(f(1e21), "1e+21");
        assert_eq!(f(1e100), "1e+100");
    }

    #[test]
    fn subnormals_and_extremes_survive() {
        assert_eq!(f(5e-324), "5e-324");
        assert_eq!(f(f64::MAX), "1.7976931348623157e+308");
        assert_eq!(f(f64::MIN_POSITIVE), "2.2250738585072014e-308");
    }

    #[test]
    fn non_finite_values_become_null() {
        // JSON cannot spell these, and `dump()` does not throw.
        assert_eq!(f(f64::NAN), "null");
        assert_eq!(f(f64::INFINITY), "null");
        assert_eq!(f(f64::NEG_INFINITY), "null");
    }

    #[test]
    #[expect(
        clippy::excessive_precision,
        reason = "the extra digits are the subject"
    )]
    fn grisu2_is_reproduced_where_it_differs_from_shortest() {
        // The values from the module docs, where Rust's formatter disagrees
        // with Grisu2. Do not "tidy" these literals: the excess digits are the
        // subject, not a mistake.
        assert_eq!(f(46934.815584012416), "46934.815584012416");
        assert_eq!(f(72972.67707126706), "72972.67707126706");
        assert_eq!(f(27453.918300648482), "27453.918300648482");
        assert_eq!(f(3.4110366750178187e-295), "3.4110366750178187e-295");
    }

    #[test]
    fn everything_written_parses_back_to_the_same_bits() {
        // Round-tripping is the property Grisu2 actually guarantees; the
        // digits it picks are one valid choice among several.
        let mut x = 1u64;
        for _ in 0..20000 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = f64::from_bits(x);
            if !v.is_finite() {
                continue;
            }
            let s = f(v);
            let back: f64 = s.parse().expect("emitted a parseable number");
            assert_eq!(back.to_bits(), v.to_bits(), "round trip failed for {s}");
        }
    }
}
