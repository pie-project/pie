#pragma once

// Non-finite floats read from private words: WGSL refuses a non-finite
// constant, and a literal bit pattern folds into one.
static uint ieee_neg_inf = 0xff800000u;
static uint ieee_pos_inf = 0x7f800000u;
static uint ieee_nan = 0x7fc00000u;

float neg_inf() { return asfloat(ieee_neg_inf); }
float pos_inf() { return asfloat(ieee_pos_inf); }
float quiet_nan() { return asfloat(ieee_nan); }

// NaN and finiteness read off the bits: a self-compare (`x != x`) folds to
// false under WGSL's compilers, and a driver may fold `isnan` where it
// assumes no NaN; an exponent and mantissa test folds under neither.
bool ieee_is_nan(float x) {
    uint bits = asuint(x);
    return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u;
}
bool ieee_finite(float x) { return (asuint(x) & 0x7f800000u) != 0x7f800000u; }
