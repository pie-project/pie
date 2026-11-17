//! The roofline every engine weighs a fire by: a dispatch costs its launch
//! and the longer of its compute and its memory traffic,
//!
//! `t = launch_ns + max(flops' / rate, bytes / bandwidth)`,
//!
//! over the device's few measured constants (the profile's `[rates]`:
//! `profile::fire::Rates`) and what the device and its kernels are, read at
//! run time ([`Machine`]): its SMs (CUs on ROCm) and the tile each kernel
//! takes. `flops'` is the work as the device runs it, wave quantized: a
//! kernel of `g` work-groups (a matmul's output tiles, times its K splits;
//! an attention's heads times its query tiles) runs `ceil(g / SMs)` waves,
//! each SM a work-group a wave, so the work is `waves · SMs` work-groups'
//! (a matmul's each its whole tile, padding and all; an attention's slots
//! are its kernel's CTAs an SM, and where its query tiles leave slots idle
//! the kernel splits the keys to fill them, as a split-KV prefill does). Nothing in it is fitted
//! to a shape: the tiles are the kernels' own and the constants the
//! device's.

use std::sync::Arc;

/// A matmul kernel's work-group as the backend runs it at a shape: `rows`
/// of the product's rows and `cols` of its columns, its depth cut in
/// `splits` (K splits; 1 where it does not split).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    pub rows: u32,
    pub cols: u32,
    pub splits: u32,
}

impl Tile {
    /// A tile of `rows x cols`, K whole.
    #[must_use]
    pub const fn of(rows: u32, cols: u32) -> Tile {
        Tile {
            rows,
            cols,
            splits: 1,
        }
    }
}

/// The matmul tile a backend's kernel takes at `(m, n, k)`.
pub type TileOf = Arc<dyn Fn(u32, u32, u32) -> Tile + Send + Sync>;

/// What the device is, read from it at run time, and the work-groups its
/// kernels take: never a profile's.
#[derive(Clone)]
pub struct Machine {
    /// Its streaming multiprocessors (compute units).
    pub sms: u32,
    /// The matmul kernel's tile at a shape (cuBLASLt's chosen algorithm's,
    /// or the engine's own kernel's constant).
    pub gemm_tile: TileOf,
    /// The prefill attention kernel's query rows a work-group, its heads'
    /// rows packed (a grouped head's queries share its keys).
    pub attn_rows: u32,
    /// The attention kernel's work-groups an SM holds at once.
    pub attn_ctas_per_sm: u32,
}

impl std::fmt::Debug for Machine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Machine")
            .field("sms", &self.sms)
            .field("attn_rows", &self.attn_rows)
            .field("attn_ctas_per_sm", &self.attn_ctas_per_sm)
            .finish_non_exhaustive()
    }
}

impl PartialEq for Machine {
    fn eq(&self, other: &Machine) -> bool {
        self.sms == other.sms
            && self.attn_rows == other.attn_rows
            && self.attn_ctas_per_sm == other.attn_ctas_per_sm
            && Arc::ptr_eq(&self.gemm_tile, &other.gemm_tile)
    }
}

impl Machine {
    /// A device of `sms` whose matmul takes one `tile` at every shape.
    #[must_use]
    pub fn fixed(sms: u32, tile: Tile, attn_rows: u32) -> Machine {
        Machine {
            sms,
            gemm_tile: Arc::new(move |_, _, _| tile),
            attn_rows,
            attn_ctas_per_sm: 1,
        }
    }

    /// `waves · SMs / groups`: what `groups` work-groups cost against their
    /// own work, each SM one a wave (1 where they fill every wave).
    #[must_use]
    pub fn wave_factor(&self, groups: u64) -> f64 {
        if groups == 0 {
            return 1.0;
        }
        let sms = u64::from(self.sms.max(1));
        (groups.div_ceil(sms) * sms) as f64 / groups as f64
    }

    /// The flops a matmul of `m x n x k` takes as the device runs it: its
    /// tiles' (each whole, padding and all) over its waves.
    #[must_use]
    pub fn gemm_flops(&self, m: u32, n: u32, k: u32) -> f64 {
        if m == 0 || n == 0 || k == 0 {
            return 0.0;
        }
        let t = (self.gemm_tile)(m, n, k);
        let (rows, cols, splits) = (t.rows.max(1), t.cols.max(1), t.splits.max(1));
        let groups = u64::from(m.div_ceil(rows)) * u64::from(n.div_ceil(cols)) * u64::from(splits);
        let sms = u64::from(self.sms.max(1));
        let waves = groups.div_ceil(sms);
        // Each wave as long as one work-group: its tile over its share of K.
        let group = 2.0 * f64::from(rows) * f64::from(cols) * f64::from(k.div_ceil(splits));
        (waves * sms) as f64 * group
    }

    /// What an attention of `lanes` lanes of `rows` query rows over `heads`
    /// query heads costs against its own work: its work-groups (a lane's
    /// packed rows a query tile each) over the slots the device holds at
    /// once (`attn_ctas_per_sm` an SM), the keys split to fill the slots
    /// where the tiles leave some idle.
    #[must_use]
    pub fn attn_wave_factor(&self, lanes: u32, rows: u32, heads: u32) -> f64 {
        let packed = u64::from(rows) * u64::from(heads.max(1));
        let groups = u64::from(lanes) * packed.div_ceil(u64::from(self.attn_rows.max(1)));
        if groups == 0 {
            return 1.0;
        }
        let slots = u64::from(self.sms.max(1)) * u64::from(self.attn_ctas_per_sm.max(1));
        let splits = (slots / groups).max(1);
        let groups = groups * splits;
        (groups.div_ceil(slots) * slots) as f64 / groups as f64
    }
}

/// The device's constants as the roofline reads them (flops, bytes and
/// elements a nanosecond).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Constants {
    pub launch_ns: f64,
    pub gemm_flops_per_ns: f64,
    pub attn_flops_per_ns: f64,
    pub bytes_per_ns: f64,
    pub scan_elements_per_ns: f64,
}

impl Constants {
    /// A dispatch's cost: its launch and the longer of `flops` (as the
    /// device runs them) over `rate` and `bytes` over the bandwidth.
    #[must_use]
    pub fn dispatch_ns(&self, flops: f64, rate: f64, bytes: f64) -> f64 {
        let over = |work: f64, rate: f64| if work > 0.0 { work / rate } else { 0.0 };
        self.launch_ns + over(flops, rate).max(over(bytes, self.bytes_per_ns))
    }

    /// A matmul of `m x n x k` reading `bytes` (its weight plane; its
    /// activations beside it are the rows' own traffic), on `machine`.
    #[must_use]
    pub fn gemm_ns(&self, machine: &Machine, m: u32, n: u32, k: u32, bytes: f64) -> f64 {
        self.dispatch_ns(machine.gemm_flops(m, n, k), self.gemm_flops_per_ns, bytes)
    }
}

/// The rows past which no rung can be chosen, by the roofline alone: the
/// last row of `lattice` (ascending) where some matmul shape of
/// `crate::rates::DEPTHS` squared (its bf16 plane read and its compute,
/// [`Constants::gemm_ns`]) costs a row more than [`crate::rates::AGREEMENT`]
/// less than the least it cost at any row before ([`crate::rates::saturate`],
/// the rule a measured grid saturated by, over the model's costs instead).
/// A top rung is the argmax of efficiency times live capacity, and live
/// capacity only falls as rows grow, so no rung past it can win.
#[must_use]
pub fn fire_top(lattice: &[u32], constants: &Constants, machine: &Machine) -> u32 {
    let shapes: Vec<(u32, u32)> = crate::rates::DEPTHS
        .iter()
        .flat_map(|&k| crate::rates::DEPTHS.iter().map(move |&n| (n, k)))
        .collect();
    let (_, saturation) = crate::rates::saturate(lattice, shapes.len(), |at, row, _| {
        let (n, k) = shapes[at];
        let m = lattice[row];
        let plane = 2.0 * f64::from(n) * f64::from(k);
        let cost = constants.gemm_ns(machine, m, n, k, plane) / f64::from(m.max(1));
        Ok::<_, ()>((0.0, cost))
    })
    .expect("the model fails nothing");
    saturation.rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l40s() -> Machine {
        Machine::fixed(142, Tile::of(128, 128), 128)
    }

    fn constants() -> Constants {
        Constants {
            launch_ns: 900.0,
            gemm_flops_per_ns: 180_000.0,
            attn_flops_per_ns: 120_000.0,
            bytes_per_ns: 700.0,
            scan_elements_per_ns: 2_000.0,
        }
    }

    #[test]
    fn a_matmul_pays_for_its_waves() {
        let machine = l40s();
        // 1024 x 1024: 8 x 8 = 64 tiles, one wave of 142 SMs: each SM a
        // tile's time, as long as 142 tiles' work.
        let flops = machine.gemm_flops(1024, 1024, 1024);
        assert_eq!(flops, 142.0 * 2.0 * 128.0 * 128.0 * 1024.0);
        // 142 tiles fill the wave exactly: no more time than 64 took.
        assert_eq!(machine.gemm_flops(142 * 128, 128, 1024), flops);
        // 143 tiles take a second wave: twice the time.
        assert_eq!(machine.gemm_flops(143 * 128, 128, 1024), 2.0 * flops);
        // A row past a tile pads a whole tile row: 129 rows cost 256's.
        assert_eq!(
            machine.gemm_flops(129, 128, 1024),
            machine.gemm_flops(256, 128, 1024)
        );
        // Many waves: the quantization fades (4096^3: 1024 tiles, 8 waves
        // of 142, 1136 tiles' work).
        let big = machine.gemm_flops(4096, 4096, 4096);
        assert!((big / (2.0 * 4096f64.powi(3)) - 1136.0 / 1024.0).abs() < 1e-12);
        // K split in 4: four times the groups, each a quarter of K.
        let split = Machine::fixed(
            142,
            Tile {
                rows: 128,
                cols: 128,
                splits: 4,
            },
            128,
        );
        // 256 groups: two waves of 142, each a quarter of K.
        assert_eq!(
            split.gemm_flops(1024, 1024, 1024),
            284.0 * 2.0 * 128.0 * 128.0 * 256.0
        );
        assert_eq!(machine.wave_factor(142), 1.0);
        assert_eq!(machine.wave_factor(143), 284.0 / 143.0);
    }

    #[test]
    fn an_attention_pays_for_its_waves_and_splits_its_keys_to_fill_them() {
        // 142 SMs, two CTAs each: 284 slots; 128 packed rows a work-group.
        let machine = Machine {
            attn_ctas_per_sm: 2,
            ..l40s()
        };
        // 2048 rows of 16 heads: 256 tiles, under 284 slots, no split fits
        // a second copy: one wave, 284 slots' time for 256.
        assert_eq!(machine.attn_wave_factor(1, 2048, 16), 284.0 / 256.0);
        // 4096 rows: 512 tiles, two waves (568 slots' time).
        assert_eq!(machine.attn_wave_factor(1, 4096, 16), 568.0 / 512.0);
        // 512 rows: 64 tiles, the keys split in 4 to 256: one wave.
        assert_eq!(machine.attn_wave_factor(1, 512, 16), 284.0 / 256.0);
        // Many waves: the quantization fades.
        assert!(machine.attn_wave_factor(1, 65536, 16) < 1.01);
    }

    #[test]
    fn a_dispatch_is_its_launch_and_the_longer_of_compute_and_traffic() {
        let c = constants();
        let machine = l40s();
        // A decode row over a 4096 x 4096 plane on a 16 x 64 tile split in
        // 4: 64 tiles x 4 splits, two waves of 16 x 64 x 1024: the 32 MiB
        // read binds.
        let plane = 2.0 * 4096.0 * 4096.0;
        let skinny = Machine::fixed(
            142,
            Tile {
                rows: 16,
                cols: 64,
                splits: 4,
            },
            128,
        );
        assert_eq!(
            skinny.gemm_flops(1, 4096, 4096),
            284.0 * 2.0 * 16.0 * 64.0 * 1024.0
        );
        assert_eq!(
            c.gemm_ns(&skinny, 1, 4096, 4096, plane),
            900.0 + plane / 700.0
        );
        // 8192 rows: the compute binds.
        let ns = c.gemm_ns(&machine, 8192, 4096, 4096, plane);
        assert_eq!(ns, 900.0 + machine.gemm_flops(8192, 4096, 4096) / 180_000.0);
        assert_eq!(c.dispatch_ns(0.0, 1.0, 0.0), 900.0);
    }

    #[test]
    fn the_fire_top_is_where_the_cost_a_row_stops_falling() {
        // The L40S's lattice to 19484: the narrowest shapes (n 1024) fill
        // their second wave at 4467 rows (35 x 8 = 280 of 284 tiles, from
        // 4096's 256: 9% a row cheaper) and gain nothing past it.
        let lattice: Vec<u32> = (0..=14)
            .flat_map(|i| {
                let p = 1u32 << i;
                [p, p + p / 11]
            })
            .collect();
        let mut lattice: Vec<u32> = lattice.into_iter().filter(|&m| m >= 1).collect();
        lattice.sort_unstable();
        lattice.dedup();
        assert!(lattice.contains(&4096) && lattice.contains(&4468));
        let top = fire_top(&lattice, &constants(), &l40s());
        assert_eq!(top, 4468);
        // A device of one SM has no waves to fill: past its ridge point
        // (the read a row against the compute) nothing gains.
        let one = Machine::fixed(1, Tile::of(1, 1), 1);
        assert!(fire_top(&lattice, &constants(), &one) <= 1024);
    }
}
