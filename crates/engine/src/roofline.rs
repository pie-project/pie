//! The roofline every engine weighs a fire by: a dispatch costs its launch
//! and the longer of its compute and its memory traffic,
//!
//! `t = launch_ns + max(flops / rate, bytes / bandwidth)`,
//!
//! over the device's measured rates (the profile's `[rates]`:
//! `profile::fire::Rates`). The compute rates are measured curves, not
//! peaks: a matmul's rate by its rows (measured at
//! [`crate::rates::GEMM_WIDTH`] columns and depth, read at a plane's rows
//! whatever its width) and the attention's by its keys, each
//! as the backend's own kernels run it, so a kernel's tiles, waves and
//! residency are in the rate and nothing models them. The per-row ops run at
//! their own measured bandwidth.

use crate::rates::{AGREEMENT, GEMM_WIDTH};

/// The device's rates as the roofline reads them (flops, bytes and
/// elements a nanosecond).
#[derive(Debug, Clone, PartialEq)]
pub struct Constants {
    pub launch_ns: f64,
    /// Matmul flops a ns by rows at [`GEMM_WIDTH`] columns and depth.
    pub gemm_by_rows: Vec<(u32, f64)>,
    /// Attention flops a ns (the rectangle's) by keys.
    pub attn_by_keys: Vec<(u32, f64)>,
    pub bytes_per_ns: f64,
    pub elementwise_bytes_per_ns: f64,
    pub scan_elements_per_ns: f64,
}

impl Constants {
    /// The constants `rates` states, or the first key unstated of those
    /// every model needs (the matmul curve, the bandwidths); the attention
    /// curve and the scan rate are read as unbounded where unstated, and the
    /// estimate asks for them only where the model attends or scans.
    pub fn of(rates: &profile::fire::Rates) -> Result<Constants, &'static str> {
        let flops = |curve: &Vec<(u32, f64)>| -> Vec<(u32, f64)> {
            curve.iter().map(|&(x, t)| (x, t * 1e3)).collect()
        };
        Ok(Constants {
            launch_ns: rates.launch_ns.unwrap_or(0) as f64,
            gemm_by_rows: flops(
                rates
                    .gemm_tflops_by_rows
                    .as_ref()
                    .ok_or("gemm_tflops_by_rows")?,
            ),
            attn_by_keys: rates
                .attn_tflops_by_keys
                .as_ref()
                .map(flops)
                .unwrap_or_default(),
            bytes_per_ns: rates.hbm_gbps.ok_or("hbm_gbps")?,
            elementwise_bytes_per_ns: rates.elementwise_gbps.ok_or("elementwise_gbps")?,
            scan_elements_per_ns: rates.scan_gelems.unwrap_or(f64::INFINITY),
        })
    }

    /// The matmul rate at `m` rows: the curve there.
    #[must_use]
    pub fn gemm_rate(&self, m: u32) -> f64 {
        profile::fire::interpolate(&self.gemm_by_rows, f64::from(m)).unwrap_or(f64::INFINITY)
    }

    /// The attention rate at `keys` keys a row.
    #[must_use]
    pub fn attn_rate(&self, keys: u32) -> f64 {
        profile::fire::interpolate(&self.attn_by_keys, f64::from(keys)).unwrap_or(f64::INFINITY)
    }

    /// A dispatch's cost: its launch and the longer of `flops` over `rate`
    /// and `bytes` over the bandwidth.
    #[must_use]
    pub fn dispatch_ns(&self, flops: f64, rate: f64, bytes: f64) -> f64 {
        let over = |work: f64, rate: f64| if work > 0.0 { work / rate } else { 0.0 };
        self.launch_ns + over(flops, rate).max(over(bytes, self.bytes_per_ns))
    }

    /// A matmul of `m x n x k` reading `bytes` (its weight plane).
    #[must_use]
    pub fn gemm_ns(&self, m: u32, n: u32, k: u32, bytes: f64) -> f64 {
        let flops = 2.0 * f64::from(m) * f64::from(n) * f64::from(k);
        self.dispatch_ns(flops, self.gemm_rate(m), bytes)
    }
}

/// The rows past which no rung can be chosen, by the roofline alone: of the
/// matmul curve's rows, ascending, the first `p` past which a doubling to
/// `2p` rows of a [`GEMM_WIDTH`]-square plane (its launch, and its bf16
/// read or its compute, [`Constants::gemm_ns`]) costs a row no more than
/// [`AGREEMENT`] less; the curve's widest row where every doubling still
/// gains. A top rung is the argmax of efficiency times live capacity, and
/// live capacity only falls as rows grow, so no rung past it can win. On
/// `lattice` (ascending), its first row at or past it.
#[must_use]
pub fn fire_top(lattice: &[u32], constants: &Constants) -> u32 {
    let w = GEMM_WIDTH;
    let plane = 2.0 * f64::from(w) * f64::from(w);
    let per_row = |m: u32| constants.gemm_ns(m, w, w, plane) / f64::from(m.max(1));
    let rows: Vec<u32> = constants.gemm_by_rows.iter().map(|&(m, _)| m).collect();
    let top = rows
        .iter()
        .copied()
        .find(|&p| per_row(p.saturating_mul(2)) > (1.0 - AGREEMENT) * per_row(p))
        .or(rows.last().copied())
        .unwrap_or(1);
    lattice
        .iter()
        .copied()
        .find(|&m| m >= top)
        .or(lattice.last().copied())
        .unwrap_or(top)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constants() -> Constants {
        Constants {
            launch_ns: 900.0,
            // 80 TFLOP/s at 256 rows rising to 200 at 4096 and flat past.
            gemm_by_rows: vec![
                (256, 80_000.0),
                (512, 120_000.0),
                (1024, 160_000.0),
                (2048, 192_000.0),
                (4096, 200_000.0),
                (8192, 202_000.0),
            ],
            attn_by_keys: vec![(512, 50_000.0), (8192, 400_000.0)],
            bytes_per_ns: 700.0,
            elementwise_bytes_per_ns: 400.0,
            scan_elements_per_ns: 2_000.0,
        }
    }

    #[test]
    fn a_matmul_reads_the_curve_at_its_rows() {
        let c = constants();
        assert_eq!(c.gemm_rate(1024), 160_000.0);
        // Past the curve's ends: clamped.
        assert_eq!(c.gemm_rate(1), 80_000.0);
        assert_eq!(c.gemm_rate(65536), 202_000.0);
        // Halfway in log rows.
        assert!((c.gemm_rate(724) - 140_000.0).abs() < 200.0);
        assert!((c.attn_rate(2048) - 225_000.0).abs() < 1e-6);
    }

    #[test]
    fn a_dispatch_is_its_launch_and_the_longer_of_compute_and_traffic() {
        let c = constants();
        let plane = 2.0 * 4096.0 * 4096.0;
        // A decode row: the 32 MiB read binds.
        assert_eq!(c.gemm_ns(1, 4096, 4096, plane), 900.0 + plane / 700.0);
        // 8192 rows: the compute binds, at the curve's 8192.
        let flops = 2.0 * 8192.0 * 4096.0 * 4096.0;
        assert_eq!(
            c.gemm_ns(8192, 4096, 4096, plane),
            900.0 + flops / 202_000.0
        );
        assert_eq!(c.dispatch_ns(0.0, 1.0, 0.0), 900.0);
    }

    #[test]
    fn the_fire_top_is_where_a_doubling_stops_paying() {
        let lattice: Vec<u32> = (0..=14).map(|i| 1u32 << i).chain([2234, 2436]).collect();
        let mut lattice = lattice;
        lattice.sort_unstable();
        // 2048 to 4096 gains 4% (192 to 200): not more than AGREEMENT, so
        // 2048 is the top.
        assert_eq!(fire_top(&lattice, &constants()), 2048);
        // A curve still rising 10% each doubling to its end: its widest row.
        let rising = Constants {
            gemm_by_rows: (8..=13)
                .map(|i| (1u32 << i, 1000.0 * 1.1f64.powi(i)))
                .collect(),
            ..constants()
        };
        assert_eq!(fire_top(&lattice, &rising), 8192);
    }
}
