//! The roofline every engine weighs a fire by: a dispatch costs its launch
//! and the longer of its compute and its memory traffic,
//!
//! `t = launch_ns + max(flops / rate, bytes / bandwidth)`,
//!
//! over the device's measured rates (the profile's `[rates]`:
//! `profile::fire::Rates`). The compute rates are measured curves, not
//! peaks: a matmul's rate by its rows at a few widths (columns and depth;
//! a plane of `n x k` is read at its square width `sqrt(n k)`, in `log
//! width` between the widths measured, at the timed rows (half octaves) at or
//! below its own) and the attention's by its keys, each as the backend's own kernels
//! run it, so a kernel's waves and residency at those points are in the
//! rate. Between timed rows a matmul pays for the rows its kernel pads to its
//! tile (`ceil(m / TM) · TM`, `TM` the tile rows of the algorithm the
//! backend runs at that shape: [`RowTile`]). The per-row ops run at their
//! own measured bandwidth.
//!
//! Two device facts weigh the choice between kernels of one op. A matmul
//! of decode-shaped rows (4 to 256) runs at `few_row_tflops`, the rate of
//! the kernel serving picks there, under the wave rule over that kernel's
//! grid ([`Constants::with_few_rows`], [`Constants::matmul_compute_ns`]).
//! A dispatch of `W` workgroups fills the device's `S =
//! workgroups_per_core · cores` slots only in part where `W < S`: its
//! compute takes `work / rate · max(1, S / W)` ([`Constants::wave_ns`]),
//! the one wave-fill rule every choice reads. A matmul on the vendor
//! library's algorithm, where the library states the waves its algorithm
//! fills at the shape, is priced by those waves rounded up to whole ones
//! ([`wave_fill`], [`Constants::with_gemm_waves`]) at the curve's full
//! rate; where it states none, at the curve as timed.
//! Neither key is read as unbounded where unstated: a choice that needs one
//! and finds it unstated is refused by its key.

use std::sync::Arc;

/// The rows of the tile a backend's matmul kernel pads `m` rows to at
/// `(m, n, k)` (1 where it pads nothing).
pub type RowTile = Arc<dyn Fn(u32, u32, u32) -> u32 + Send + Sync>;

/// The workgroups of a backend's few-rows kernel at `(m, n, k)` where it
/// serves that matmul (its grid), `None` where the kernel the matmul curve
/// times does.
pub type FewRowGrid = Arc<dyn Fn(u32, u32, u32) -> Option<u32> + Send + Sync>;

/// The waves the kernel the matmul curve times (the vendor library's
/// algorithm) fills at `(m, n, k)` on the device: its workgroups over those
/// the device holds at once of it, as the library states them for the
/// algorithm it picks there. `None` where it states none.
pub type GemmWaves = Arc<dyn Fn(u32, u32, u32) -> Option<f64> + Send + Sync>;

/// A backend's few-rows kernel as the roofline weighs it: its grid at a
/// shape, its rate (`few_row_tflops`, flops a ns) and the workgroups the
/// device holds at once for its full rate (`workgroups_per_core` times the
/// compute units).
#[derive(Clone)]
pub struct FewRows {
    pub grid: FewRowGrid,
    pub rate: f64,
    pub full: f64,
}

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
    /// The backend's few-rows kernel, where it states one
    /// ([`Constants::with_few_rows`]): the matmuls it serves are weighed at
    /// its rate under the wave rule, not the curve's.
    pub few_rows: Option<FewRows>,
    /// The waves the curve's kernel fills at a shape, where the backend
    /// states them ([`Constants::with_gemm_waves`]).
    pub gemm_waves: Option<GemmWaves>,
    /// `gemm_by_rows` at its full rate: each point times the fill of the
    /// waves it filled where it was timed, where the backend states them
    /// ([`Constants::with_gemm_waves`]). `gemm_by_rows` stays as measured.
    pub gemm_full_by_rows: Option<Vec<(u32, Vec<(u32, f64)>)>>,
}

impl std::fmt::Debug for Constants {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constants")
            .field("launch_ns", &self.launch_ns)
            .field("gemm_by_rows", &self.gemm_by_rows)
            .field("gemm_full_by_rows", &self.gemm_full_by_rows)
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
            && self.gemm_full_by_rows == other.gemm_full_by_rows
            && self.attn_by_keys == other.attn_by_keys
            && self.bytes_per_ns == other.bytes_per_ns
            && self.elementwise_bytes_per_ns == other.elementwise_bytes_per_ns
            && self.scan_elements_per_ns == other.scan_elements_per_ns
            && self.few_row_flops_per_ns == other.few_row_flops_per_ns
            && self.workgroups_per_core == other.workgroups_per_core
            && Arc::ptr_eq(&self.row_tile, &other.row_tile)
            && match (&self.few_rows, &other.few_rows) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    Arc::ptr_eq(&a.grid, &b.grid) && a.rate == b.rate && a.full == b.full
                }
                _ => false,
            }
            && match (&self.gemm_waves, &other.gemm_waves) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
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
            few_rows: None,
            gemm_waves: None,
            gemm_full_by_rows: None,
        })
    }

    /// These constants with the backend's matmul tile rows.
    #[must_use]
    pub fn with_row_tile(self, row_tile: RowTile) -> Constants {
        Constants { row_tile, ..self }
    }

    /// These constants with the backend's few-rows kernel: its `grid` over
    /// `cores` compute units, at `few_row_tflops` under the wave rule
    /// ([`Constants::wave_ns`]); the key where the profile states either
    /// rate it reads.
    pub fn with_few_rows(self, grid: FewRowGrid, cores: u32) -> Result<Constants, &'static str> {
        let rate = self.few_row_rate()?;
        let full = self.full_rate_workgroups(cores)?;
        Ok(Constants {
            few_rows: Some(FewRows { grid, rate, full }),
            ..self
        })
    }

    /// These constants with the waves the curve's kernel fills at a shape
    /// (`waves`, as the vendor library states them for the algorithm it
    /// picks): the curve, timed at `m x w x w`, also stated at its full
    /// rate (`gemm_full_by_rows`: each point times [`wave_fill`] of the
    /// waves it filled there, where stated), so a matmul whose waves are
    /// stated costs its flops at the full rate times the fill of its own
    /// waves ([`Constants::gemm_compute_ns`]): whole waves run at the full
    /// rate, a tile past them pays a wave. `gemm_by_rows` stays as
    /// measured, so laying the waves twice is laying them once.
    #[must_use]
    pub fn with_gemm_waves(self, waves: GemmWaves) -> Constants {
        let full = self
            .gemm_by_rows
            .iter()
            .map(|(w, curve)| {
                let full = curve
                    .iter()
                    .map(|&(m, r)| (m, r * waves(m, *w, *w).map_or(1.0, wave_fill)))
                    .collect();
                (*w, full)
            })
            .collect();
        Constants {
            gemm_full_by_rows: Some(full),
            gemm_waves: Some(waves),
            ..self
        }
    }

    /// The compute of a matmul of `m x n x k` taking `flops` (its rows
    /// padded to its tile), in ns: on the backend's few-rows kernel where
    /// it serves the shape, `flops` at `few_row_tflops` under the wave rule
    /// over its grid ([`Constants::wave_ns`]); else on the curve's kernel
    /// ([`Constants::gemm_compute_ns`]).
    #[must_use]
    pub fn matmul_compute_ns(&self, m: u32, n: u32, k: u32, flops: f64) -> f64 {
        if flops <= 0.0 {
            return 0.0;
        }
        match &self.few_rows {
            Some(few) => match (few.grid)(m, n, k) {
                Some(workgroups) => wave(flops, few.rate, workgroups, few.full),
                None => self.curve_compute_ns(m, n, k, flops),
            },
            None => self.curve_compute_ns(m, n, k, flops),
        }
    }

    /// `flops` of `m x n x k` on the curve's kernel: where the backend
    /// states the waves it fills there, at the curve's full rate times
    /// their [`wave_fill`]; else at the curve's rate as timed.
    fn curve_compute_ns(&self, m: u32, n: u32, k: u32, flops: f64) -> f64 {
        let stated = self
            .gemm_waves
            .as_ref()
            .and_then(|waves| waves(m, n, k))
            .zip(self.gemm_full_by_rows.as_deref());
        match stated {
            Some((waves, full)) => {
                flops / profile::fire::gemm_rate(full, m, n, k) * wave_fill(waves)
            }
            None => flops / self.gemm_rate(m, n, k),
        }
    }

    /// The matmul rate at `m` rows of an `n x k` plane, as the curve was
    /// timed ([`profile::fire::gemm_rate`]).
    #[must_use]
    pub fn gemm_rate(&self, m: u32, n: u32, k: u32) -> f64 {
        profile::fire::gemm_rate(&self.gemm_by_rows, m, n, k)
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
            self.curve_compute_ns(m, n, k, flops)
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
        Ok(wave(work, rate, workgroups, self.full_rate_workgroups(cores)?))
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

/// `work` at `rate` as a dispatch of `workgroups` runs it on a device
/// needing `full` at once for its full rate: `work / rate · max(1, full /
/// workgroups)` ([`Constants::wave_ns`]).
fn wave(work: f64, rate: f64, workgroups: u32, full: f64) -> f64 {
    let fill = (full / f64::from(workgroups.max(1))).max(1.0);
    if work > 0.0 { work / rate * fill } else { 0.0 }
}

/// The time a dispatch of `waves` waves, as the vendor library states
/// them for its algorithm, takes over the time its work takes at the full
/// rate: its waves rounded up to whole ones (at least one) over `waves`.
/// A dispatch short of a wave leaves the rest of the device idle (`1 /
/// waves`); one a tile past whole waves runs the last wave nearly empty.
/// 1 at whole waves, and where `waves` is nothing. Only stated waves are
/// rounded: [`Constants::wave_ns`]'s `S` is a knee, not a residency.
#[must_use]
pub fn wave_fill(waves: f64) -> f64 {
    if waves.is_finite() && waves > 0.0 {
        waves.ceil().max(1.0) / waves
    } else {
        1.0
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
            few_rows: None,
            gemm_waves: None,
            gemm_full_by_rows: None,
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

    /// The few-rows kernel's matmuls at its rate under the wave rule; the
    /// rest at the curve's.
    #[test]
    fn a_matmul_the_few_rows_kernel_serves_is_weighed_at_its_rate_and_grid() {
        let grid: FewRowGrid = Arc::new(|m, n, _| (m <= 16).then_some(n / 4));
        let c = constants().with_few_rows(grid.clone(), 80).unwrap();
        let flops = 2.0 * 8.0 * 4096.0 * 1024.0;
        // 1024 workgroups fill the 264: the flops at 4 TFLOP/s.
        assert!((c.matmul_compute_ns(8, 4096, 1024, flops) - flops / 4_000.0).abs() < 1e-6);
        // 64 workgroups, a quarter short of 264: 264 / 64 the time.
        let narrow = c.matmul_compute_ns(8, 256, 1024, flops);
        assert!((narrow - flops / 4_000.0 * 264.0 / 64.0).abs() < 1e-6);
        // Past its rows, the curve's rate.
        assert_eq!(
            c.matmul_compute_ns(32, 4096, 1024, flops),
            flops / c.gemm_rate(32, 4096, 1024)
        );
        let unstated = Constants {
            few_row_flops_per_ns: None,
            ..constants()
        };
        assert_eq!(
            unstated.with_few_rows(grid, 80).err(),
            Some("few_row_tflops")
        );
    }

    /// The curve kept as measured and stated beside it at its full rate;
    /// a shape a tile past whole waves pays the last wave, one whose waves
    /// the library states none of runs at the curve as timed.
    #[test]
    fn a_matmul_on_the_library_s_kernel_pays_for_the_waves_it_fills() {
        assert_eq!(wave_fill(3.0), 1.0);
        assert!((wave_fill(0.5) - 2.0).abs() < 1e-12);
        assert!((wave_fill(2.25) - 3.0 / 2.25).abs() < 1e-12);
        // Every curve point (n = k, a width it times) fills 1.5 waves
        // (fill 4/3); the 3000-column shape at hand 2 or 2.25; a
        // 5000-column one none stated.
        let at = |w: f64| -> GemmWaves {
            Arc::new(move |_, n, _| match n {
                3000 => Some(w),
                5000 => None,
                _ => Some(1.5),
            })
        };
        let plain = constants();
        let c = plain.clone().with_gemm_waves(at(2.0));
        assert_eq!(c.gemm_by_rows, plain.gemm_by_rows, "the curve stays as measured");
        assert_eq!(c.clone().with_gemm_waves(at(2.0)).gemm_full_by_rows, c.gemm_full_by_rows);
        let rate = plain.gemm_rate(2048, 3000, 4096);
        let flops = c.gemm_flops(2048, 3000, 4096);
        let whole = c.gemm_compute_ns(2048, 3000, 4096);
        assert!((whole - flops / (rate * 4.0 / 3.0)).abs() < 1e-6);
        let past = plain.clone().with_gemm_waves(at(2.25)).gemm_compute_ns(2048, 3000, 4096);
        assert!((past / whole - 3.0 / 2.25).abs() < 1e-9);
        assert_eq!(
            c.gemm_compute_ns(2048, 5000, 4096),
            plain.gemm_compute_ns(2048, 5000, 4096)
        );
    }

    #[test]
    fn a_rate_stated_at_nothing_is_refused_by_its_key() {
        let all = profile::fire::Rates {
            launch_ns: Some(900),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 100.0)])]),
            attn_tflops_by_keys: Some(vec![(512, 50.0)]),
            hbm_gbps: Some(700.0),
            cached_gbps: None,
            elementwise_gbps: Some(500.0),
            scan_gelems: Some(2000.0),
            few_row_tflops: Some(4.0),
            workgroups_per_core: Some(3.3),
            submit_worst_ratio: None,
            resident_workgroups: None,
            attn_tflops_by_rows_keys: None,
            product_ns: None,
            kernel_tflops_by_shape: None,
            dense_settled: None,
            measured_on: None,
            measured_by: None,
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
                    cached_gbps: None,
                    ..all.clone()
                },
                "hbm_gbps",
            ),
        ] {
            assert_eq!(Constants::of(&bad).err(), Some(key));
        }
    }
}
