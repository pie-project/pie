//! The dense matmul's kernel: the vector kernels or one of the tiled ones,
//! whichever the estimate (`engine::roofline`) weighs fastest over the
//! model's dense planes, from the device's measured rates: the vector
//! kernels at `few_row_tflops`; each tiled kernel at its own measured rate
//! at the rows it lands (`tile_tflops_by_rows`) below the matmul curve's
//! first row count, and on the `gemm_tflops_by_rows` curve, its rows padded
//! to its tile, from there. Each kernel reads its plane once a group of
//! rows that share a read (`gemm::Kernel::rows_a_read`): no cache is assumed
//! to hold it.
//!
//! A backend that records its fires (one recording a rung) takes one kernel
//! a rung, whatever rows a fire of it lands: the one whose worse cost over
//! the rows the rung serves, `(rung / 2, rung]`, lies nearest that row
//! count's best (the least regret at the rung's bottom and its top). A
//! backend that fires eagerly takes the fastest at each fire's own rows.

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

/// The name a kernel is said by.
fn name(tactic: &Tactic) -> &'static str {
    match (tactic.vector > 0, tactic.tier, tactic.wide) {
        (true, ..) => "gemv",
        (false, kernels_slang::Tier::Coopmat, _) => "coopmat",
        (false, _, false) => "tiled",
        (false, _, true) => "wide tiled",
    }
}

/// One rung weighed on each candidate kernel at its bottom and top rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Weighed {
    pub bottom: u32,
    pub top: u32,
    /// Each candidate and its cost at the bottom and at the top, in ns.
    pub costs: Vec<(Tactic, f64, f64)>,
    /// The candidate taken (least regret).
    pub taken: usize,
}

impl Weighed {
    /// Whether the rung takes the vector kernels.
    #[must_use]
    pub fn vector(&self) -> bool {
        self.costs[self.taken].0.vector > 0
    }

    /// The candidate whose worse ratio to the best at the rung's bottom and
    /// top is least (the first so where they tie).
    fn least_regret(costs: &[(Tactic, f64, f64)]) -> usize {
        let best = |of: fn(&(Tactic, f64, f64)) -> f64| {
            costs
                .iter()
                .map(of)
                .fold(f64::INFINITY, f64::min)
                .max(f64::MIN_POSITIVE)
        };
        let (bottom, top) = (best(|c| c.1), best(|c| c.2));
        let regret = |c: &(Tactic, f64, f64)| (c.1 / bottom).max(c.2 / top);
        (0..costs.len())
            .min_by(|&a, &b| regret(&costs[a]).total_cmp(&regret(&costs[b])))
            .unwrap_or(0)
    }
}

impl std::fmt::Display for Weighed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rung {}: {} by the estimate (",
            self.top,
            name(&self.costs[self.taken].0)
        )?;
        let at = |pick: fn(&(Tactic, f64, f64)) -> f64| {
            self.costs
                .iter()
                .map(|c| format!("{} {:.1} µs", name(&c.0), pick(c) / 1e3))
                .collect::<Vec<_>>()
                .join(" vs ")
        };
        write!(f, "{} rows: {}", self.top, at(|c| c.2))?;
        if self.bottom != self.top {
            write!(f, "; {} rows: {}", self.bottom, at(|c| c.1))?;
        }
        write!(f, ")")
    }
}

/// The tiled kernels a backend can take under its tiled tactic `base`: one
/// under SPIR-V's cooperative-matrix tier, else the narrow and the wide.
#[must_use]
pub fn tiled_kernels(target: Target, base: Tactic) -> Vec<Tactic> {
    if target == Target::Spirv && base.tier == kernels_slang::Tier::Coopmat {
        vec![Tactic { vector: 0, ..base }]
    } else {
        [false, true]
            .map(|wide| Tactic {
                vector: 0,
                wide,
                ..base
            })
            .to_vec()
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
    let tiles = rates
        .tile_tflops_by_rows
        .as_ref()
        .ok_or("tile_tflops_by_rows")?;
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
    // The candidates at `kernel_rows` (a rung's, or a fire's own): the
    // vector kernel carrying that many rows a workgroup, and each tiled one.
    let candidates = |kernel_rows: u32| -> Vec<Tactic> {
        let base = tiled(kernel_rows);
        let mut all = vec![Tactic {
            vector: gemm::vector_rows(kernel_rows),
            ..base
        }];
        all.extend(tiled_kernels(target, base));
        all
    };
    // `tactic`'s cost over every plane at `rows`, in ns.
    let cost = |tactic: Tactic, rows: u32| -> f64 {
        let mut ns = 0.0;
        for &(n, k, count, bytes) in &planes {
            let flops = |rows: u32| 2.0 * f64::from(rows) * f64::from(n) * f64::from(k);
            let kernel = gemm::tile_for(target, tactic, n, k);
            let read = f64::from(rows.div_ceil(kernel.rows_a_read().max(1))) * bytes;
            let compute = if tactic.vector > 0 {
                flops(rows) / few
            } else {
                let tile = kernel.row_tile().max(1);
                // Below the curve's first row count, the kernel's own rate
                // at the rows it lands; from there the curve's, its rows
                // padded to its tile.
                match tiles.iter().find(|(at, _)| *at == tile) {
                    Some((_, curve)) if curve.last().is_some_and(|&(most, _)| rows <= most) => {
                        let rate = profile::fire::interpolate(curve, f64::from(rows))
                            .unwrap_or(f64::INFINITY)
                            * 1e3;
                        flops(rows) / rate
                    }
                    _ => flops(rows.div_ceil(tile) * tile) / c.gemm_rate(rows, n, k),
                }
            };
            ns += f64::from(count) * (c.launch_ns + compute.max(read / c.bytes_per_ns));
        }
        ns
    };
    let mut by_rows = Vec::new();
    let mut weighed = Vec::with_capacity(rungs.len());
    let mut below = 0;
    for &rung in rungs {
        let bottom = (below + 1).min(rung);
        let costs: Vec<(Tactic, f64, f64)> = candidates(rung)
            .into_iter()
            .map(|t| (t, cost(t, bottom), cost(t, rung)))
            .collect();
        let taken = Weighed::least_regret(&costs);
        if records {
            by_rows.push((rung, costs[taken].0));
        } else {
            for rows in bottom..=rung {
                let fastest = candidates(rows)
                    .into_iter()
                    .map(|t| (t, cost(t, rows)))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(t, _)| t)
                    .expect("one candidate at least");
                by_rows.push((rows, fastest));
            }
        }
        weighed.push(Weighed {
            bottom,
            top: rung,
            costs,
            taken,
        });
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

    /// Rates whose tiled kernels run at `tiles` below the curve.
    fn rates(few: Option<f64>, tiles: Vec<(u32, Vec<(u32, f64)>)>) -> profile::fire::Rates {
        profile::fire::Rates {
            launch_ns: Some(3000),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 10.0), (2048, 30.0)])]),
            hbm_gbps: Some(650.0),
            elementwise_gbps: Some(700.0),
            few_row_tflops: few,
            tile_tflops_by_rows: Some(tiles),
            ..Default::default()
        }
    }

    /// A coopmat tile measured at a sixtieth of the curve's rate at 4 rows,
    /// rising to it at 256.
    fn coopmat_tiles() -> Vec<(u32, Vec<(u32, f64)>)> {
        vec![(64, vec![(4, 0.16), (64, 2.5), (256, 10.0)])]
    }

    fn coopmat(_: u32) -> Tactic {
        Tactic {
            tier: Tier::Coopmat,
            vector: 0,
            wide: true,
        }
    }

    /// A rung takes the vector kernels where the tiled one's measured rate
    /// at its rows costs more, and the tiled ones where the curve outruns
    /// the vector rate; a matmul runs its rung's.
    #[test]
    fn each_rung_takes_the_kernel_the_estimate_weighs_faster() {
        let rungs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
        let r = rates(Some(5.0), coopmat_tiles());
        let (dense, weighed) = choose(Target::Spirv, &rungs, coopmat, &r, &counts(), true).unwrap();
        let vector: Vec<bool> = weighed.iter().map(Weighed::vector).collect();
        assert!(vector[..4].iter().all(|&v| v), "{weighed:?}");
        assert!(vector[9..].iter().all(|&v| !v), "{weighed:?}");
        for (i, &rows) in rungs.iter().enumerate() {
            assert_eq!(dense.at(rows).vector > 0, vector[i]);
            assert_eq!(
                dense.at(rows.div_ceil(2) + 1).vector > 0,
                vector[i],
                "{rows}"
            );
        }
        assert_eq!(dense.at(3).vector, 4);
        assert!(
            weighed[2]
                .to_string()
                .starts_with("rung 4: gemv by the estimate ("),
            "{}",
            weighed[2]
        );
    }

    /// Without coopmat both tiled kernels are candidates, each priced at its
    /// own measured rate: a wide tile measured faster at partial fill is
    /// taken there though it pads more rows.
    #[test]
    fn the_wide_tile_is_taken_where_it_measures_faster() {
        let baseline = |_| Tactic {
            tier: Tier::Baseline,
            vector: 0,
            wide: true,
        };
        // The 32-row tile slow, the 256-row register tile fast below 256.
        let tiles = vec![
            (32, vec![(4, 0.1), (256, 1.0)]),
            (256, vec![(4, 0.1), (128, 6.0), (256, 10.0)]),
        ];
        let r = rates(Some(0.5), tiles);
        let (dense, weighed) =
            choose(Target::Spirv, &[128, 256], baseline, &r, &counts(), true).unwrap();
        assert!(
            dense.at(200).wide && dense.at(200).vector == 0,
            "{weighed:?}"
        );
        assert_eq!(weighed[1].costs.len(), 3);
    }

    /// A profile stating no vector rate, or no tiled kernel's, is refused by
    /// its key.
    #[test]
    fn a_missing_rate_is_refused_by_its_key() {
        let refused = choose(
            Target::Spirv,
            &[1, 2],
            coopmat,
            &rates(None, coopmat_tiles()),
            &counts(),
            true,
        );
        assert_eq!(refused.unwrap_err(), "few_row_tflops");
        let mut no_tiles = rates(Some(5.0), coopmat_tiles());
        no_tiles.tile_tflops_by_rows = None;
        let refused = choose(Target::Spirv, &[1, 2], coopmat, &no_tiles, &counts(), true);
        assert_eq!(refused.unwrap_err(), "tile_tflops_by_rows");
    }

    /// A fire fired eagerly takes the fastest kernel at its own rows.
    #[test]
    fn an_eager_fire_takes_the_fastest_at_its_own_rows() {
        let r = rates(Some(5.0), coopmat_tiles());
        let rungs = engine::memory::rungs(1, 2048);
        let (eager, _) = choose(Target::Spirv, &rungs, coopmat, &r, &counts(), false).unwrap();
        for rows in [1, 3, 5, 17, 100, 129, 200, 256, 300, 2048] {
            let (_, w) = choose(Target::Spirv, &[rows], coopmat, &r, &counts(), true).unwrap();
            let fastest = w[0]
                .costs
                .iter()
                .min_by(|a, b| a.2.total_cmp(&b.2))
                .map(|c| c.0)
                .unwrap();
            assert_eq!(eager.at(rows), fastest, "{rows}");
        }
    }

    /// Every tiled kernel's cost never falls as a rung's rows grow, on every
    /// target and tier, at one rate.
    #[test]
    fn the_cost_grows_with_the_rows() {
        let rungs: Vec<u32> = engine::memory::rungs(1, 8192);
        let flat_tiles: Vec<(u32, Vec<(u32, f64)>)> = [16, 32, 64, 256]
            .map(|t| (t, vec![(4, 20.0), (256, 20.0)]))
            .to_vec();
        let mut flat = rates(Some(5.0), flat_tiles);
        flat.gemm_tflops_by_rows = Some(vec![(4096, vec![(256, 20.0)])]);
        for target in [Target::Spirv, Target::Wgsl] {
            for tier in [Tier::Baseline, Tier::Coopmat] {
                let tiled = |_| Tactic {
                    tier,
                    vector: 0,
                    wide: true,
                };
                let (_, weighed) = choose(target, &rungs, tiled, &flat, &counts(), true).unwrap();
                for pair in weighed.windows(2) {
                    for (a, b) in pair[0].costs.iter().zip(&pair[1].costs).skip(1) {
                        assert!(b.2 >= a.2, "{target:?} {tier:?}: {pair:?}");
                    }
                }
            }
        }
    }
}
