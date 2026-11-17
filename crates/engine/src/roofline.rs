//! The roofline every engine weighs a fire by: a dispatch costs its launch
//! and the longer of its compute and its memory traffic,
//!
//! `t = launch_ns + max(flops / rate, bytes / bandwidth)`,
//!
//! over the device's measured rates (the profile's `[rates]`:
//! `profile::fire::Rates`). The compute rates are measured curves, not
//! peaks: a matmul's rate by its rows at a few widths (columns and depth;
//! a plane of `n x k` is read at its square width `sqrt(n k)`, in `log
//! width` between the widths measured, at the octave of rows at or below its
//! own) and the attention's by its keys, each as the backend's own kernels
//! run it, so a kernel's waves and residency at those points are in the
//! rate. Between octaves a matmul pays for the rows its kernel pads to its
//! tile (`ceil(m / TM) · TM`, `TM` the tile rows of the algorithm the
//! backend runs at that shape: [`RowTile`]). The per-row ops run at their
//! own measured bandwidth.

use std::sync::Arc;

/// The rows of the tile a backend's matmul kernel pads `m` rows to at
/// `(m, n, k)` (1 where it pads nothing).
pub type RowTile = Arc<dyn Fn(u32, u32, u32) -> u32 + Send + Sync>;

/// The device's rates as the roofline reads them (flops, bytes and
/// elements a nanosecond), and the backend's matmul tile rows.
#[derive(Clone)]
pub struct Constants {
    pub launch_ns: f64,
    /// Matmul flops a ns by rows at each width (columns and depth both),
    /// ascending in width.
    pub gemm_by_rows: Vec<(u32, Vec<(u32, f64)>)>,
    /// Attention flops a ns (the rectangle's) by keys.
    pub attn_by_keys: Vec<(u32, f64)>,
    pub bytes_per_ns: f64,
    pub elementwise_bytes_per_ns: f64,
    pub scan_elements_per_ns: f64,
    /// The rows a matmul pads to at a shape.
    pub row_tile: RowTile,
}

impl std::fmt::Debug for Constants {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constants")
            .field("launch_ns", &self.launch_ns)
            .field("gemm_by_rows", &self.gemm_by_rows)
            .field("attn_by_keys", &self.attn_by_keys)
            .field("bytes_per_ns", &self.bytes_per_ns)
            .field("elementwise_bytes_per_ns", &self.elementwise_bytes_per_ns)
            .field("scan_elements_per_ns", &self.scan_elements_per_ns)
            .finish_non_exhaustive()
    }
}

impl PartialEq for Constants {
    fn eq(&self, other: &Constants) -> bool {
        self.launch_ns == other.launch_ns
            && self.gemm_by_rows == other.gemm_by_rows
            && self.attn_by_keys == other.attn_by_keys
            && self.bytes_per_ns == other.bytes_per_ns
            && self.elementwise_bytes_per_ns == other.elementwise_bytes_per_ns
            && self.scan_elements_per_ns == other.scan_elements_per_ns
            && Arc::ptr_eq(&self.row_tile, &other.row_tile)
    }
}

impl Constants {
    /// The constants `rates` states, or the first key unstated of those
    /// every model needs (the matmul curves, the bandwidths); the attention
    /// curve and the scan rate are read as unbounded where unstated, and the
    /// estimate asks for them only where the model attends or scans. A
    /// matmul pads nothing until [`Constants::with_row_tile`] states its
    /// backend's tile.
    pub fn of(rates: &profile::fire::Rates) -> Result<Constants, &'static str> {
        let flops = |curve: &Vec<(u32, f64)>| -> Vec<(u32, f64)> {
            curve.iter().map(|&(x, t)| (x, t * 1e3)).collect()
        };
        Ok(Constants {
            launch_ns: rates.launch_ns.unwrap_or(0) as f64,
            gemm_by_rows: rates
                .gemm_tflops_by_rows
                .as_ref()
                .ok_or("gemm_tflops_by_rows")?
                .iter()
                .map(|(w, c)| (*w, flops(c)))
                .collect(),
            attn_by_keys: rates
                .attn_tflops_by_keys
                .as_ref()
                .map(flops)
                .unwrap_or_default(),
            bytes_per_ns: rates.hbm_gbps.ok_or("hbm_gbps")?,
            elementwise_bytes_per_ns: rates.elementwise_gbps.ok_or("elementwise_gbps")?,
            scan_elements_per_ns: rates.scan_gelems.unwrap_or(f64::INFINITY),
            row_tile: Arc::new(|_, _, _| 1),
        })
    }

    /// These constants with the backend's matmul tile rows.
    #[must_use]
    pub fn with_row_tile(self, row_tile: RowTile) -> Constants {
        Constants { row_tile, ..self }
    }

    /// The matmul rate at `m` rows of an `n x k` plane: each width's curve
    /// at the octave of rows at or below `m` (its first below that), in
    /// `log width` at `sqrt(n k)` between the widths, clamped past them.
    #[must_use]
    pub fn gemm_rate(&self, m: u32, n: u32, k: u32) -> f64 {
        let at = |curve: &[(u32, f64)]| -> f64 {
            curve
                .iter()
                .rev()
                .find(|&&(rows, _)| rows <= m)
                .or(curve.first())
                .map_or(f64::INFINITY, |&(_, r)| r)
        };
        let width: Vec<(u32, f64)> = self.gemm_by_rows.iter().map(|(w, c)| (*w, at(c))).collect();
        let square = (f64::from(n) * f64::from(k)).sqrt();
        // Past the widest width, its last span's slope in log width carries
        // on (a wider plane fills more tiles); never below the widest's rate.
        if let [.., (w0, r0), (w1, r1)] = width[..]
            && square > f64::from(w1)
        {
            let slope = (r1 - r0) / (f64::from(w1).ln() - f64::from(w0).ln());
            return (r1 + slope * (square.ln() - f64::from(w1).ln())).max(r1);
        }
        profile::fire::interpolate(&width, square).unwrap_or(f64::INFINITY)
    }

    /// The rows a matmul of `m x n x k` computes: `m` padded to its
    /// kernel's tile.
    #[must_use]
    pub fn padded_rows(&self, m: u32, n: u32, k: u32) -> u32 {
        let tile = (self.row_tile)(m, n, k).max(1);
        m.div_ceil(tile) * tile
    }

    /// The flops a matmul of `m x n x k` takes as its kernel runs it: its
    /// rows padded to its tile.
    #[must_use]
    pub fn gemm_flops(&self, m: u32, n: u32, k: u32) -> f64 {
        2.0 * f64::from(self.padded_rows(m, n, k)) * f64::from(n) * f64::from(k)
    }

    /// A matmul's compute at `m x n x k`, in ns.
    #[must_use]
    pub fn gemm_compute_ns(&self, m: u32, n: u32, k: u32) -> f64 {
        let flops = self.gemm_flops(m, n, k);
        if flops > 0.0 {
            flops / self.gemm_rate(m, n, k)
        } else {
            0.0
        }
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
        let read = if bytes > 0.0 {
            bytes / self.bytes_per_ns
        } else {
            0.0
        };
        self.launch_ns + self.gemm_compute_ns(m, n, k).max(read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(scale: f64) -> Vec<(u32, f64)> {
        // 80 TFLOP/s at 256 rows rising to 200 at 4096 and flat past.
        [
            (256, 80_000.0),
            (512, 120_000.0),
            (1024, 160_000.0),
            (2048, 192_000.0),
            (4096, 200_000.0),
            (8192, 202_000.0),
        ]
        .iter()
        .map(|&(m, r)| (m, r * scale))
        .collect()
    }

    fn constants() -> Constants {
        Constants {
            launch_ns: 900.0,
            gemm_by_rows: vec![(1024, curve(0.5)), (4096, curve(1.0))],
            attn_by_keys: vec![(512, 50_000.0), (8192, 400_000.0)],
            bytes_per_ns: 700.0,
            elementwise_bytes_per_ns: 400.0,
            scan_elements_per_ns: 2_000.0,
            row_tile: Arc::new(|_, _, _| 1),
        }
    }

    #[test]
    fn a_matmul_reads_its_width_at_the_octave_below_its_rows() {
        let c = constants();
        assert_eq!(c.gemm_rate(1024, 4096, 4096), 160_000.0);
        // Between octaves: the octave below.
        assert_eq!(c.gemm_rate(1500, 4096, 4096), 160_000.0);
        // Past the curve's ends: its first row's, its last's.
        assert_eq!(c.gemm_rate(1, 4096, 4096), 80_000.0);
        assert_eq!(c.gemm_rate(65536, 4096, 4096), 202_000.0);
        // A 1024-square plane reads the 1024 curve; 2048 square halfway in
        // log width; a 1024 x 4096 plane is 2048 square.
        assert_eq!(c.gemm_rate(1024, 1024, 1024), 80_000.0);
        assert!((c.gemm_rate(1024, 2048, 2048) - 120_000.0).abs() < 1e-6);
        assert!((c.gemm_rate(1024, 1024, 4096) - 120_000.0).abs() < 1e-6);
        // Past the widest width: its last span's slope (1024 to 4096 at 1024
        // rows, 80 to 160) carries on: 16384 square two more octaves, 240.
        assert!((c.gemm_rate(1024, 16384, 16384) - 240_000.0).abs() < 1e-6);
        assert!((c.attn_rate(2048) - 225_000.0).abs() < 1e-6);
    }

    #[test]
    fn a_matmul_pays_for_the_rows_its_tile_pads() {
        // A 256-row tile: 1100 rows run as 1280, at the 1024 octave's rate.
        let c = constants().with_row_tile(Arc::new(|_, _, _| 256));
        let ns = c.gemm_compute_ns(1100, 4096, 4096);
        assert_eq!(ns, 2.0 * 1280.0 * 4096.0 * 4096.0 / 160_000.0);
        // At an octave the tile divides the rows: the curve itself.
        assert_eq!(
            c.gemm_compute_ns(1024, 4096, 4096),
            2.0 * 1024.0 * 4096.0 * 4096.0 / 160_000.0
        );
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
}
