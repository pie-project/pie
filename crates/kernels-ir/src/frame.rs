//! The frame: what a kernel runs over, as typed primitives. A backend's
//! frame provides each over its threads and planes, and its printer names
//! them; the interpreter's frame ([`crate::interp::OneRow`]) provides them
//! over one row on the host.
//!
//! A thread holds elements `0..N` of the row (which columns is the frame's);
//! a kernel names an element by that index alone.

use dtype::Dtype;

use crate::region::Reduce;

/// The primitives a kernel's body calls.
pub trait Frame {
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
    /// every one.
    fn reduce(&mut self, op: Reduce, partial: f32) -> f32;
    /// `v` rounded to `dtype` and back: a value the region rounds where it
    /// is defined.
    fn round(&self, dtype: Dtype, v: f32) -> f32;
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
