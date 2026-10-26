//! The skeleton: what a kernel runs over, as typed primitives. A backend's
//! skeleton provides each over its threads and planes, and its printer names
//! them; the interpreter's skeleton ([`crate::interp::OneRow`]) provides them
//! over one row on the host.
//!
//! A thread holds elements `0..N` of the row (which columns is the skeleton's);
//! a kernel names an element by that index alone.

use dtype::Dtype;

use crate::region::Reduce;

/// The primitives a kernel's body calls.
pub trait Skeleton {
    /// The elements of the row a thread holds.
    fn elements(&self) -> usize;
    /// Element `e` lies inside the row (one past it reads garbage, and a
    /// reduction leaves it out).
    fn valid(&self, e: usize) -> bool;
    /// Element `e` of row parameter `k`, as f32.
    fn load(&self, k: u32, e: usize) -> f32;
    /// Scalar parameter `k`, one value a row, as f32.
    fn one(&self, k: u32) -> f32;
    /// Writes element `e` of result `k`; returns `v` as its plane holds it
    /// (rounded to the plane's dtype), what every later read sees.
    fn store(&mut self, k: u32, e: usize, v: f32) -> f32;
    /// Result `k` is complete.
    fn flush(&mut self, k: u32);
    /// A thread's partial combined over the row's threads, the total to
    /// every one, as f32. A device skeleton's partials are f32 in its own
    /// order; the interpreter's, f64.
    fn reduce(&mut self, op: Reduce, partial: f64) -> f32;
    /// `v` rounded to `dtype` and back: a value the region rounds where it
    /// is defined.
    fn round(&self, dtype: Dtype, v: f32) -> f32;
    /// Element `e`'s column in the row.
    fn column(&self, e: usize) -> u32;
    /// Each of a thread's vectors' partials (`partials[e / 8]`) combined
    /// over its head of `seg` vectors, in place: lane `i` joined with lane
    /// `i ^ off`, `off` halving from `seg / 2` (a butterfly).
    fn reduce_heads(&mut self, op: Reduce, seg: u32, partials: &mut [f64]);
    /// Element `e` of the row a pass shares.
    fn share(&mut self, e: usize, v: f32);
    /// The shared row at column `c` (column 0 past the row).
    fn peer(&self, c: u32) -> f32;
    /// Value `i` of small-vector parameter `k`, one a row.
    fn vector(&self, k: u32, i: usize) -> f32;
    /// Writes value `i` of small-vector result `k`, one a row.
    fn store_vector(&mut self, k: u32, i: usize, v: f32);
}

/// `x` rounded to bf16 (nearest, ties to even) and widened back.
#[must_use]
pub fn round_bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}

/// `x` rounded to e4m3 (the nearest, ties to even; saturating at ±448, as
/// CUDA's `satfinite` conversion lands it), as an f32.
#[must_use]
pub fn round_e4m3(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    const MAX: f32 = 448.0;
    let a = x.abs();
    // The quantum of a's binade: 2^(e - 3), or the subnormals' 2^-9.
    let exponent = ((a.to_bits() >> 23) & 0xff) as i32 - 127;
    let quantum = 2f32.powi(exponent.max(-6) - 3);
    let q = ((a / quantum).round_ties_even() * quantum).min(MAX);
    q.copysign(x)
}
