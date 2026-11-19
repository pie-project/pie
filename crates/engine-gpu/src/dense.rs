//! The dense matmul's kernel: the vector kernels or the tiled ones,
//! whichever the estimate (`engine::roofline`) weighs faster over the
//! model's dense planes, from the device's measured rates (the vector
//! kernels at `few_row_tflops`, the tiled ones on the `gemm_tflops_by_rows`
//! curve, each plane's rows padded to its tile), each kernel reading its
//! plane once a group of rows that share a read
//! (`gemm::Kernel::rows_a_read`): no cache is assumed to hold it.
//!
//! A backend that records its fires (one recording a rung) takes one kernel
//! a rung, whatever rows a fire of it lands: the one whose worse cost over
//! the rows the rung serves, `(rung / 2, rung]`, lies nearer that row
//! count's best (the least regret at the rung's bottom and its top). A
//! backend that fires eagerly takes the faster at each fire's own rows.

use engine::load::Counts;
use engine::roofline::Constants;
use kernels_slang::Target;
use kernels_slang::linear::gemm::{self, Tactic};

/// The tactic a matmul of some rows runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dense {
    /// `(rows, tactic)`, ascending: a rung's top and its tactic where fires
    /// are recorded a rung, else every row count to the top.
    by_rows: Vec<(u32, Tactic)>,
}

impl Dense {
    /// The tactic a matmul of `rows` runs: that of the fewest rows stated
    /// that hold them (its rung's, or its own), the most past them all.
    #[must_use]
    pub fn at(&self, rows: u32) -> Tactic {
        self.by_rows
            .iter()
            .find(|&&(at, _)| at >= rows)
            .or(self.by_rows.last())
            .map(|&(_, tactic)| tactic)
            .expect("a load has one rung at least")
    }
}

/// The two kernels' costs at some rows, in ns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cost {
    pub rows: u32,
    pub vector_ns: f64,
    pub tiled_ns: f64,
}

impl Cost {
    fn best(&self) -> f64 {
        self.vector_ns.min(self.tiled_ns)
    }
}

/// One rung weighed both ways at its bottom and top rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weighed {
    pub bottom: Cost,
    pub top: Cost,
}

impl Weighed {
    /// The vector kernels' worse ratio to the best over the rung's rows
    /// lies strictly under the tiled kernels'.
    #[must_use]
    pub fn vector(&self) -> bool {
        let regret = |of: fn(&Cost) -> f64| {
            [self.bottom, self.top]
                .iter()
                .map(|c| of(c) / c.best().max(f64::MIN_POSITIVE))
                .fold(1.0, f64::max)
        };
        regret(|c| c.vector_ns) < regret(|c| c.tiled_ns)
    }
}

impl std::fmt::Display for Weighed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kernel = if self.vector() { "gemv" } else { "tiled" };
        let at = |c: &Cost| {
            format!(
                "{} rows {:.1} µs gemv vs {:.1} µs tiled",
                c.rows,
                c.vector_ns / 1e3,
                c.tiled_ns / 1e3
            )
        };
        write!(
            f,
            "rung {}: {kernel} by the estimate ({}",
            self.top.rows,
            at(&self.top)
        )?;
        if self.bottom.rows != self.top.rows {
            write!(f, "; {}", at(&self.bottom))?;
        }
        write!(f, ")")
    }
}

/// The dense tactic over `rungs` (ascending rows) on `target`, the model's
/// dense planes (`counts`: every dense shape and the output head, each a
/// dispatch reading its plane) weighed over `rates`; `tiled` is the
/// backend's tiled tactic at some rows. One a rung where the backend
/// `records` its fires, else one a row count to the top rung. Each rung's
/// weighing beside it; the `[rates]` key it lacks where the profile states
/// none of one it reads.
pub fn choose(
    target: Target,
    rungs: &[u32],
    tiled: impl Fn(u32) -> Tactic,
    rates: &profile::fire::Rates,
    counts: &Counts,
    records: bool,
) -> Result<(Dense, Vec<Weighed>), &'static str> {
    let c = Constants::of(rates)?;
    let few = c.few_row_rate()?;
    // `(n, k, planes, bytes a plane)` of every dense plane a row reads.
    let mut planes: Vec<(u32, u32, u32, f64)> = counts
        .gemm_shapes
        .iter()
        .filter(|s| !s.routed && s.planes > 0)
        .map(|s| (s.n, s.k, s.planes, s.bytes as f64 / f64::from(s.planes)))
        .collect();
    if counts.readout_flops > 0 {
        let depth = counts.readout_depth.max(1);
        let vocab =
            u32::try_from(counts.readout_flops / (2 * u64::from(depth))).unwrap_or(u32::MAX);
        planes.push((vocab, depth, 1, counts.head_bytes as f64));
    }
    // The two tactics at `rows` (the tiled one the backend's at `kernel_rows`,
    // the vector one carrying that many rows a workgroup) and their costs.
    let cost = |rows: u32, kernel_rows: u32| -> (Tactic, Tactic, Cost) {
        let tactic = tiled(kernel_rows);
        let vector = Tactic {
            vector: gemm::vector_rows(kernel_rows),
            ..tactic
        };
        let (mut vector_ns, mut tiled_ns) = (0.0, 0.0);
        for &(n, k, count, bytes) in &planes {
            let flops = |rows: u32| 2.0 * f64::from(rows) * f64::from(n) * f64::from(k);
            // Each kernel reads the plane once its rows sharing a read.
            let read = |kernel: gemm::Kernel| {
                f64::from(rows.div_ceil(kernel.rows_a_read().max(1))) * bytes
            };
            let tiles = gemm::tile_for(target, tactic, rows, n, k);
            let tile = tiles.row_tile().max(1);
            let padded = rows.div_ceil(tile) * tile;
            let count = f64::from(count);
            tiled_ns += count * c.dispatch_ns(flops(padded), c.gemm_rate(rows, n, k), read(tiles));
            let vectors = gemm::tile_for(target, vector, rows, n, k);
            vector_ns += count * c.dispatch_ns(flops(rows), few, read(vectors));
        }
        (
            tactic,
            vector,
            Cost {
                rows,
                vector_ns,
                tiled_ns,
            },
        )
    };
    let pick = |tactic: Tactic, vector: Tactic, takes_vector: bool| {
        if takes_vector { vector } else { tactic }
    };
    let mut by_rows = Vec::new();
    let mut weighed = Vec::with_capacity(rungs.len());
    let mut below = 0;
    for &rung in rungs {
        let bottom_rows = (below + 1).min(rung);
        let (tactic, vector, top) = cost(rung, rung);
        let (_, _, bottom) = cost(bottom_rows, rung);
        let w = Weighed { bottom, top };
        if records {
            by_rows.push((rung, pick(tactic, vector, w.vector())));
        } else {
            for rows in bottom_rows..=rung {
                let (tactic, vector, at) = cost(rows, rows);
                by_rows.push((rows, pick(tactic, vector, at.vector_ns < at.tiled_ns)));
            }
        }
        weighed.push(w);
        below = rung;
    }
    Ok((Dense { by_rows }, weighed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels_slang::Tier;

    fn counts() -> Counts {
        Counts {
            weight_bytes: 1 << 30,
            expert_bytes: 0,
            experts: None,
            // Thirty-two 4096-square bf16 planes.
            gemm_shapes: vec![engine::load::GemmShape {
                planes: 32,
                ..engine::load::GemmShape::dense(4096, 4096, 32 << 25, 32 << 25)
            }],
            head_bytes: 0,
            attn_flops_per_row_key: 0,
            kv_bytes_per_key: 0,
            attn_head_dim: 128,
            readout_flops: 0,
            readout_depth: 4096,
            state_elements_per_row: 0,
            elementwise_ops: vec![],
            attn_dispatches: 0,
            scan_dispatches: 0,
            width: 4096,
        }
    }

    fn rates(few: Option<f64>) -> profile::fire::Rates {
        profile::fire::Rates {
            launch_ns: Some(3000),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 10.0), (2048, 30.0)])]),
            hbm_gbps: Some(650.0),
            elementwise_gbps: Some(700.0),
            few_row_tflops: few,
            ..Default::default()
        }
    }

    fn tiled(_: u32) -> Tactic {
        Tactic {
            tier: Tier::Coopmat,
            vector: 0,
        }
    }

    /// A rung takes the vector kernels where the tiled one's tile pads its
    /// rows past what the vector rate costs, and the tiled ones where the
    /// curve outruns the vector rate; a matmul runs its rung's.
    #[test]
    fn each_rung_takes_the_kernel_the_estimate_weighs_faster() {
        let rungs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
        let (dense, weighed) = choose(
            Target::Spirv,
            &rungs,
            tiled,
            &rates(Some(5.0)),
            &counts(),
            true,
        )
        .unwrap();
        let vector: Vec<bool> = weighed.iter().map(Weighed::vector).collect();
        // To 4 rows the vector kernel reads the plane once, as the 64-row
        // tile does; from 256 rows its 64 reads and 5 TFLOP/s lose.
        assert!(vector[..4].iter().all(|&v| v), "{weighed:?}");
        assert!(vector[8..].iter().all(|&v| !v), "{weighed:?}");
        for (i, &rows) in rungs.iter().enumerate() {
            assert_eq!(dense.at(rows).vector > 0, vector[i]);
            assert_eq!(
                dense.at(rows.div_ceil(2) + 1).vector > 0,
                vector[i],
                "{rows}"
            );
        }
        assert_eq!(dense.at(3).vector, 4);
        assert_eq!(dense.at(12).vector, if vector[4] { 16 } else { 0 });
        assert_eq!(dense.at(5000).vector > 0, vector[rungs.len() - 1]);
        assert!(
            weighed[2]
                .to_string()
                .starts_with("rung 4: gemv by the estimate (")
        );
    }

    /// A profile stating no vector rate is refused by its key.
    #[test]
    fn no_vector_rate_is_refused_by_its_key() {
        let refused = choose(Target::Spirv, &[1, 2], tiled, &rates(None), &counts(), true);
        assert_eq!(refused.unwrap_err(), "few_row_tflops");
    }

    /// A fire fired eagerly takes the faster kernel at its own rows; a
    /// recorded rung one kernel for all its rows, the faster at its top where
    /// it is also the faster at its bottom.
    #[test]
    fn an_eager_fire_takes_the_faster_at_its_own_rows() {
        let rungs = engine::memory::rungs(1, 2048);
        let (eager, _) = choose(
            Target::Spirv,
            &rungs,
            tiled,
            &rates(Some(5.0)),
            &counts(),
            false,
        )
        .unwrap();
        let (recorded, weighed) = choose(
            Target::Spirv,
            &rungs,
            tiled,
            &rates(Some(5.0)),
            &counts(),
            true,
        )
        .unwrap();
        for rows in 1..=2048 {
            let (_, w) = choose(
                Target::Spirv,
                &[rows],
                tiled,
                &rates(Some(5.0)),
                &counts(),
                true,
            )
            .unwrap();
            assert_eq!(
                eager.at(rows).vector > 0,
                w[0].top.vector_ns < w[0].top.tiled_ns,
                "{rows}"
            );
        }
        for w in &weighed {
            let (top, bottom) = (
                w.top.vector_ns < w.top.tiled_ns,
                w.bottom.vector_ns < w.bottom.tiled_ns,
            );
            if top == bottom {
                assert_eq!(recorded.at(w.top.rows).vector > 0, top, "{w}");
            }
        }
    }

    /// The tiled kernels' cost never falls as a rung's rows grow, on every
    /// target and tier, at one rate (a flat curve, so only the padding
    /// moves).
    #[test]
    fn the_tiled_cost_grows_with_the_rows() {
        let rungs: Vec<u32> = engine::memory::rungs(1, 8192);
        let flat = profile::fire::Rates {
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 20.0)])]),
            ..rates(Some(5.0))
        };
        for target in [Target::Spirv, Target::Wgsl] {
            for tier in [Tier::Baseline, Tier::Coopmat] {
                let tiled = |_| Tactic { tier, vector: 0 };
                let (_, weighed) = choose(target, &rungs, tiled, &flat, &counts(), true).unwrap();
                for pair in weighed.windows(2) {
                    assert!(
                        pair[1].top.tiled_ns >= pair[0].top.tiled_ns
                            && pair[1].top.vector_ns >= pair[0].top.vector_ns,
                        "{target:?} {tier:?}: {pair:?}"
                    );
                }
            }
        }
    }
}
