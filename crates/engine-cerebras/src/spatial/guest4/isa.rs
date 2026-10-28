//! The guest4 ISA (spatial/bench-job/payload/guest4/pe.csl): six-word
//! instructions `[op, fl, d, a, b, c]`. `fl`: bits 0-1 the dtype (f32, i32,
//! u32, bool), bit 2 `a` broadcast (a pool scalar), bit 3 `b` broadcast (or
//! a scatter's add), bit 4 `c` broadcast (or the f32 add / one-word state),
//! bits 8.. the word count.

pub const HALT: u32 = 0;
pub const WBIN: u32 = 1;
pub const WCMP: u32 = 2;
pub const WUN: u32 = 3;
pub const WSEL: u32 = 4;
pub const WFILL: u32 = 5;
pub const WIOTA: u32 = 6;
pub const WRNG: u32 = 7;
pub const WMOV: u32 = 8;
pub const RRED: u32 = 10;
pub const WGATHER: u32 = 11;
pub const WSCATTER: u32 = 12;
pub const TOPK: u32 = 13;
pub const PIVRANK: u32 = 14;
pub const PIVMASS: u32 = 15;
pub const QBIN: u32 = 20;
pub const QCMP: u32 = 21;
pub const QUN: u32 = 22;
pub const QSEL: u32 = 23;
pub const QIOTA: u32 = 24;
pub const QCONST: u32 = 25;
pub const QMOV: u32 = 26;
pub const QGATHER: u32 = 27;
pub const QSCATTER: u32 = 28;
pub const QCUM: u32 = 29;
pub const QRED: u32 = 30;
pub const QRNG: u32 = 31;
pub const OUT: u32 = 40;


// Binary ops (WBIN/QBIN `c`).
pub const B_ADD: u32 = 0;
pub const B_SUB: u32 = 1;
pub const B_MUL: u32 = 2;
pub const B_DIV: u32 = 3;
pub const B_MAX: u32 = 4;
pub const B_MIN: u32 = 5;
pub const B_REM: u32 = 6;
// Unary ops (WUN/QUN `c`; 16 + dtype: a cast to it).
pub const U_NEG: u32 = 0;
pub const U_ABS: u32 = 1;
pub const U_EXP: u32 = 2;
pub const U_LOG: u32 = 3;
pub const U_SQRT: u32 = 4;
pub const U_NOT: u32 = 5;
pub const U_TRUNC: u32 = 6;
pub const U_NONZERO: u32 = 7;
pub const U_RECIP: u32 = 8;
pub const U_SIGN: u32 = 9;
pub const U_RSQRT: u32 = 10;
pub const U_CAST: u32 = 16;

pub const T_F32: u32 = 0;
pub const T_I32: u32 = 1;
pub const T_U32: u32 = 2;
pub const T_BOOL: u32 = 3;
