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
//!
//! Two device facts weigh the choice between kernels of one op. A matmul
//! of decode-shaped rows (4 to 256) runs at `few_row_tflops`, the rate of
//! the kernel serving picks there ([`Constants::few_row_ns`]). A dispatch
//! of `W` workgroups fills the device's `S = workgroups_per_core · cores`
//! slots only in part where `W < S`: its compute takes `work / rate ·
//! max(1, S / W)` ([`Constants::wave_ns`]), the one wave-fill rule every
//! choice reads (the prefill attention's per-row against tiled kernel now;
//! the few-row matmul's selection and CUDA's wave rule next). Neither is
//! read as unbounded where unstated: a choice that needs one and finds it
//! unstated is refused by its key.

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
    /// `few_row_tflops` in flops a ns, where stated.
    pub few_row_flops_per_ns: Option<f64>,
    /// `workgroups_per_core`, where stated.
    pub workgroups_per_core: Option<f64>,
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
            .field("few_row_flops_per_ns", &self.few_row_flops_per_ns)
            .field("workgroups_per_core", &self.workgroups_per_core)
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
            && self.few_row_flops_per_ns == other.few_row_flops_per_ns
            && self.workgroups_per_core == other.workgroups_per_core
            && Arc::ptr_eq(&self.row_tile, &other.row_tile)
    }
}

impl Constants {
    /// The constants `rates` states, or the first key unstated of those
    /// every model needs (the matmul curves, the bandwidths); the attention
    /// curve and the scan rate are read as unbounded where unstated, and the
    /// estimate asks for them only where the model attends or scans. A
    /// matmul pads nothing until [`Constants::with_row_tile`] states its
    /// backend's tile. A rate stated at or below zero, or not finite, is
    /// refused by its key as an unstated one is
    /// (`crate::rates::unmeasured`): nothing divides by it.
    pub fn of(rates: &profile::fire::Rates) -> Result<Constants, &'static str> {
        if let Some((key, _)) = crate::rates::unmeasured(rates) {
            return Err(key);
        }
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
            few_row_flops_per_ns: rates.few_row_tflops.map(|t| t * 1e3),
            workgroups_per_core: rates.workgroups_per_core,
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

    /// The few-row matmul's rate (`few_row_tflops`) in flops a ns, or its
    /// key where the profile states none.
    pub fn few_row_rate(&self) -> Result<f64, &'static str> {
        self.few_row_flops_per_ns.ok_or("few_row_tflops")
    }

    /// A matmul of decode-shaped `m x n x k` (4 to 256 rows) reading
    /// `bytes`, on the kernel serving picks there: its launch and the
    /// longer of its weight reads and its flops (its rows padded to its
    /// kernel's tile, [`Constants::gemm_flops`]) at `few_row_tflops`; the
    /// key where the profile states none.
    pub fn few_row_ns(&self, m: u32, n: u32, k: u32, bytes: f64) -> Result<f64, &'static str> {
        let rate = self.few_row_rate()?;
        Ok(self.dispatch_ns(self.gemm_flops(m, n, k), rate, bytes))
    }

    /// The workgroups the device holds at once for a kernel's full rate
    /// over `cores` compute units, `S = workgroups_per_core · cores`; the
    /// key where the profile states none.
    pub fn full_rate_workgroups(&self, cores: u32) -> Result<f64, &'static str> {
        let per = self.workgroups_per_core.ok_or("workgroups_per_core")?;
        Ok(per * f64::from(cores.max(1)))
    }

    /// The compute of `work` at `rate` (work a ns, the kernel's full rate)
    /// as a dispatch of `workgroups` over `cores` units runs it: `work /
    /// rate · max(1, S / workgroups)`, `S` the device's
    /// [`Constants::full_rate_workgroups`]. The one wave-fill rule every
    /// choice between kernels reads.
    pub fn wave_ns(
        &self,
        work: f64,
        rate: f64,
        workgroups: u32,
        cores: u32,
    ) -> Result<f64, &'static str> {
        let full = self.full_rate_workgroups(cores)?;
        let fill = (full / f64::from(workgroups.max(1))).max(1.0);
        Ok(if work > 0.0 { work / rate * fill } else { 0.0 })
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
            few_row_flops_per_ns: Some(4_000.0),
            workgroups_per_core: Some(3.3),
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

    #[test]
    fn a_few_row_matmul_runs_at_its_own_rate_past_its_weight_reads() {
        let c = constants();
        let plane = 2.0 * 1024.0 * 1024.0;
        // 8 rows of a 1024 plane: 16.8 MFLOP at 4 TFLOP/s (4.19 us) against
        // 2 MiB at 700 GB/s (3.0 us): the compute binds.
        let flops = 2.0 * 8.0 * 1024.0 * 1024.0;
        assert_eq!(
            c.few_row_ns(8, 1024, 1024, plane),
            Ok(900.0 + flops / 4_000.0)
        );
        // One row: the read binds.
        assert_eq!(
            c.few_row_ns(1, 1024, 1024, plane),
            Ok(900.0 + plane / 700.0)
        );
        let unstated = Constants {
            few_row_flops_per_ns: None,
            ..constants()
        };
        assert_eq!(
            unstated.few_row_ns(8, 1024, 1024, plane),
            Err("few_row_tflops")
        );
    }

    #[test]
    fn a_dispatch_short_of_the_device_s_workgroups_pays_for_the_slots_it_leaves() {
        let c = constants();
        // 80 units at 3.3 a unit: 264 workgroups fill the device.
        assert!((c.full_rate_workgroups(80).unwrap() - 264.0).abs() < 1e-9);
        // 264 or more: the work at its rate.
        assert_eq!(c.wave_ns(1e6, 1e3, 264, 80), Ok(1e3));
        assert_eq!(c.wave_ns(1e6, 1e3, 1000, 80), Ok(1e3));
        // 66, a quarter: four times as long.
        assert!((c.wave_ns(1e6, 1e3, 66, 80).unwrap() - 4e3).abs() < 1e-6);
        let unstated = Constants {
            workgroups_per_core: None,
            ..constants()
        };
        assert_eq!(
            unstated.wave_ns(1e6, 1e3, 66, 80),
            Err("workgroups_per_core")
        );
    }

    #[test]
    fn a_rate_stated_at_nothing_is_refused_by_its_key() {
        let all = profile::fire::Rates {
            launch_ns: Some(900),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 100.0)])]),
            attn_tflops_by_keys: Some(vec![(512, 50.0)]),
            hbm_gbps: Some(700.0),
            elementwise_gbps: Some(500.0),
            scan_gelems: Some(2000.0),
            few_row_tflops: Some(4.0),
            workgroups_per_core: Some(3.3),
        };
        assert!(Constants::of(&all).is_ok());
        for (bad, key) in [
            (
                profile::fire::Rates {
                    few_row_tflops: Some(0.0),
                    ..all.clone()
                },
                "few_row_tflops",
            ),
            (
                profile::fire::Rates {
                    workgroups_per_core: Some(f64::NAN),
                    ..all.clone()
                },
                "workgroups_per_core",
            ),
            (
                profile::fire::Rates {
                    attn_tflops_by_keys: Some(vec![(512, 0.0)]),
                    ..all.clone()
                },
                "attn_tflops_by_keys",
            ),
            (
                profile::fire::Rates {
                    hbm_gbps: Some(-1.0),
                    ..all.clone()
                },
                "hbm_gbps",
            ),
        ] {
            assert_eq!(Constants::of(&bad).err(), Some(key));
        }
    }
}
