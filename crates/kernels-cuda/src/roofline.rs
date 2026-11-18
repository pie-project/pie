//! The device's roofline as a dispatch choice between kernels reads it: the
//! engine's (`engine::roofline::Constants`, the profile's `[rates]`), handed
//! to a context at load ([`crate::Ctx::with_roofline`]). A choice that reads
//! it on a context that holds none is refused by name.

/// The costs a choice between kernels weighs, in ns.
pub trait Roofline: Sync {
    /// A few-rows product of `flops` (over the rows its kernel's tile
    /// computes) moving `bytes`, dispatched as `workgroups` blocks over
    /// `cores` multiprocessors: its launch and the longer of its bytes at
    /// `hbm_gbps` and its compute at `few_row_tflops` under the wave rule
    /// (`engine::roofline::Constants::few_row_dispatch_ns`); the `[rates]`
    /// key it lacks where the profile states none.
    fn few_rows_ns(
        &self,
        flops: f64,
        bytes: f64,
        workgroups: u32,
        cores: u32,
    ) -> Result<f64, &'static str>;

    /// A product of `m x n x k` reading `bytes` on cuBLASLt's algorithm:
    /// `engine::roofline::Constants::gemm_ns`.
    fn gemm_ns(&self, m: u32, n: u32, k: u32, bytes: f64) -> f64;

    /// A per-row launch moving `bytes`: its launch and the bytes at
    /// `elementwise_gbps`.
    fn per_row_ns(&self, bytes: f64) -> f64;
}
