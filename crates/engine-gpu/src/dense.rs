//! The dense matmul's kernel: the vector kernels or one of the tiled ones,
//! whichever the estimate (`engine::roofline`) weighs fastest over the
//! model's dense planes, from the device's measured rates: each kernel at
//! each of the model's own plane shapes at its rate measured there, at the
//! rows it lands (`kernel_tflops_by_shape`, timed once the model's shapes
//! are known, `crate::rates::measure_shapes`). A measured rate holds every
//! weight pass the kernel makes and the occupancy it reaches at that plane.
//! No kernel is weighed past the rows it was timed at: it is no candidate
//! there. A plane no kernel was timed at to a rung's rows (every curve at it
//! stopped short, by the bound or the device's largest buffer) is not
//! weighed there, nothing telling the kernels apart at it; a rung past every
//! curve at every plane keeps the rung below's kernel.
//!
//! A backend that records its fires (one recording a rung) takes one kernel
//! a rung, whatever rows a fire of it lands: the one whose worse cost over
//! the rows the rung serves, `(rung / 2, rung]`, lies nearest that row
//! count's best (the least regret at the rung's bottom and its top). A
//! backend that fires eagerly takes the fastest at each fire's own rows.
//! Candidates that tie are told apart by the rows a workgroup carries, the
//! fewest taken.

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
    /// The tactic each run of row counts takes, said compactly: `rows-rows
    /// name` for every run of one tactic, ascending.
    #[must_use]
    pub fn runs(&self) -> String {
        let mut out: Vec<String> = Vec::new();
        let mut from = 1;
        for (i, &(rows, tactic)) in self.by_rows.iter().enumerate() {
            let last = self
                .by_rows
                .get(i + 1)
                .is_none_or(|&(_, next)| next != tactic);
            if last {
                out.push(format!("{from}-{rows} {}", name(&tactic)));
                from = rows + 1;
            }
        }
        out.join(", ")
    }

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
fn name(tactic: &Tactic) -> String {
    match (tactic.vector, tactic.tier, tactic.wide) {
        (0, kernels_slang::Tier::Coopmat, _) => "coopmat".to_owned(),
        (0, _, false) => "tiled".to_owned(),
        (0, _, true) => "wide tiled".to_owned(),
        (rows, ..) => format!("gemv x{rows}"),
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
    /// Whether the rung keeps the rung below's kernel, no kernel timed at
    /// its rows at any of the model's planes (`costs` holds it alone,
    /// unpriced).
    pub held: bool,
}

impl Weighed {
    /// Whether the rung takes the vector kernels.
    #[must_use]
    pub fn vector(&self) -> bool {
        self.costs[self.taken].0.vector > 0
    }

    /// The candidate whose worse ratio to the best at the rung's bottom and
    /// top is least; where candidates tie, the one whose workgroups carry
    /// the fewest rows (`rows`, [`rows_a_workgroup`]).
    fn least_regret(costs: &[(Tactic, f64, f64)], rows: &dyn Fn(&Tactic) -> u32) -> usize {
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
            .min_by(|&a, &b| {
                regret(&costs[a])
                    .total_cmp(&regret(&costs[b]))
                    .then(rows(&costs[a].0).cmp(&rows(&costs[b].0)))
            })
            .unwrap_or(0)
    }
}

impl std::fmt::Display for Weighed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.held {
            return write!(
                f,
                "rung {}: {} held from the rung below (no kernel was timed at these rows at any \
                 of the model's planes)",
                self.top,
                name(&self.costs[self.taken].0)
            );
        }
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

/// The rows one workgroup of `tactic`'s kernel carries on `target` (a
/// vector kernel's rows, a tiled kernel's row tile): what a tie between
/// candidates is broken by, the fewest first.
fn rows_a_workgroup(target: Target, tactic: &Tactic) -> u32 {
    gemm::tile_for(target, *tactic, 4096, 4096).rows_a_read()
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

/// What a load is refused by where no kernel was timed at its first rung's
/// rows at any of its planes: the key whose curves it lacks.
pub const UNREACHED: &str = "kernel_tflops_by_shape";

/// The `(n, k)` of every weight a dense matmul of `trace` multiplies by
/// (`Linear::Matmul`, the output head's among them): the planes the dense
/// kernel runs over, of the params [`Counts`] holds.
#[must_use]
pub fn multiplied(trace: &poem_ir::Trace) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = trace
        .nodes
        .iter()
        .filter_map(|node| match &node.op {
            poem_ir::Operation::Linear(poem_ir::Linear::Matmul { w, .. }) => {
                match trace.values.get(w.0 as usize).map(|decl| &decl.def) {
                    Some(poem_ir::Def::Weight(at)) => trace.params.get(*at as usize),
                    _ => None,
                }
            }
            _ => None,
        })
        .map(|param| {
            let (n, k) = engine::load::rectangle(&param.shape);
            let to = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
            (to(n), to(k))
        })
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// `(n, k, planes, bytes a plane)` of every dense plane a row of the model
/// `counts` describes reads: each dense shape a matmul multiplies by (of
/// `multiplied`, [`multiplied`]; every one where `None`), and the output
/// head.
#[must_use]
pub fn planes(counts: &Counts, multiplied: Option<&[(u32, u32)]>) -> Vec<(u32, u32, u32, f64)> {
    let mut planes: Vec<(u32, u32, u32, f64)> = counts
        .gemm_shapes
        .iter()
        .filter(|s| !s.routed && s.planes > 0)
        .filter(|s| multiplied.is_none_or(|m| m.contains(&(s.n, s.k))))
        .map(|s| (s.n, s.k, s.planes, s.bytes as f64 / f64::from(s.planes)))
        .collect();
    if counts.readout_flops > 0 {
        let depth = counts.readout_depth.max(1);
        let vocab =
            u32::try_from(counts.readout_flops / (2 * u64::from(depth))).unwrap_or(u32::MAX);
        planes.push((vocab, depth, 1, counts.head_bytes as f64));
    }
    planes
}

/// The dense tactic over `rungs` (ascending rows) on `target`, the model's
/// dense planes (`planes`, [`planes`]: each a dispatch reading its plane)
/// weighed over `rates`; `tiled` is the
/// backend's tiled tactic at some rows. One a rung where the backend
/// `records` its fires, else one a row count to the top rung. Each rung's
/// weighing beside it; the `[rates]` key it lacks where the profile states
/// none of one it reads.
pub fn choose(
    target: Target,
    rungs: &[u32],
    tiled: impl Fn(u32) -> Tactic,
    rates: &profile::fire::Rates,
    planes: &[(u32, u32, u32, f64)],
    records: bool,
) -> Result<(Dense, Vec<Weighed>), &'static str> {
    let c = Constants::of(rates)?;
    let curves = rates.kernel_tflops_by_shape.as_deref().ok_or(UNREACHED)?;
    // The candidates at `kernel_rows` (a rung's, or a fire's own): each
    // vector kernel, and each tiled one.
    let candidates = |kernel_rows: u32| -> Vec<Tactic> {
        let base = tiled(kernel_rows);
        let mut all: Vec<Tactic> = gemm::VECTOR_ROWS
            .iter()
            .map(|&rows| Tactic {
                vector: rows,
                ..base
            })
            .collect();
        all.extend(tiled_kernels(target, base));
        all
    };
    // `tactic`'s time at an `n x k` plane at `rows`, in ns, where its curve
    // there reaches the rows: the flops over the rate at the rows, between
    // measured row counts in log rows; below the first row count timed, that
    // count's time (a kernel takes no less for fewer rows than it was timed
    // at, so its rate there is not held).
    let time_at = |tactic: Tactic, n: u32, k: u32, rows: u32| -> Option<f64> {
        let (kind, key) = crate::rates::curve_key(target, tactic, n, k);
        let curve = &curves.iter().find(|c| c.id() == (kind, key, n, k))?.curve;
        let (&(first, _), &(most, _)) = (curve.first()?, curve.last()?);
        if rows > most {
            return None;
        }
        let at = rows.max(first);
        let flops = 2.0 * f64::from(at) * f64::from(n) * f64::from(k);
        profile::fire::interpolate(curve, f64::from(at)).map(|t| flops / (t * 1e3))
    };
    // `tactic`'s cost at `rows` over every plane some candidate of `among`
    // was timed at to the rows, in ns: each at the kernel's rate measured
    // there (every weight pass it makes and whatever a cache spares it in the
    // time) and a launch. `None` where its curve at one of them does not
    // reach the rows, or no plane was timed to them.
    let cost = |among: &[Tactic], tactic: Tactic, rows: u32| -> Option<f64> {
        let mut ns = 0.0;
        let mut weighed = false;
        for &(n, k, count, _) in planes {
            if among.iter().all(|&t| time_at(t, n, k, rows).is_none()) {
                continue;
            }
            ns += f64::from(count) * (c.launch_ns + time_at(tactic, n, k, rows)?);
            weighed = true;
        }
        weighed.then_some(ns)
    };

    let mut by_rows: Vec<(u32, Tactic)> = Vec::new();
    let mut weighed = Vec::with_capacity(rungs.len());
    let mut below = 0;
    for &rung in rungs {
        let bottom = (below + 1).min(rung);
        let among = candidates(rung);
        // The candidates timed over the rung's rows, bottom to top.
        let costs: Vec<(Tactic, f64, f64)> = among
            .iter()
            .filter_map(|&t| Some((t, cost(&among, t, bottom)?, cost(&among, t, rung)?)))
            .collect();
        if costs.is_empty() {
            // Past every curve at every plane: the rung below's kernel.
            let &(_, kept) = by_rows.last().ok_or(UNREACHED)?;
            by_rows.push((rung, kept));
            weighed.push(Weighed {
                bottom,
                top: rung,
                costs: vec![(kept, f64::NAN, f64::NAN)],
                taken: 0,
                held: true,
            });
            below = rung;
            continue;
        }
        let rows_of = |t: &Tactic| rows_a_workgroup(target, t);
        let taken = Weighed::least_regret(&costs, &rows_of);
        if records {
            by_rows.push((rung, costs[taken].0));
        } else {
            for rows in bottom..=rung {
                let among = candidates(rows);
                let fastest = among
                    .iter()
                    .filter_map(|&t| Some((t, cost(&among, t, rows)?)))
                    .min_by(|a, b| a.1.total_cmp(&b.1).then(rows_of(&a.0).cmp(&rows_of(&b.0))))
                    .map(|(t, _)| t)
                    .or_else(|| by_rows.last().map(|&(_, t)| t))
                    .ok_or(UNREACHED)?;
                by_rows.push((rows, fastest));
            }
        }
        weighed.push(Weighed {
            bottom,
            top: rung,
            costs,
            taken,
            held: false,
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

    /// A kernel's curve at the 4096-square planes of [`counts`].
    fn at_4096(kind: u32, key: u32, curve: Vec<(u32, f64)>) -> profile::fire::KernelCurve {
        profile::fire::KernelCurve {
            kind,
            key,
            n: 4096,
            k: 4096,
            curve,
            stop: None,
        }
    }

    /// `r` with its vector kernels' curves at 4096 square `vectors`, `(rows a
    /// workgroup carries, curve)`.
    fn set_vectors(r: &mut profile::fire::Rates, vectors: Vec<(u32, Vec<(u32, f64)>)>) {
        let curves = r.kernel_tflops_by_shape.get_or_insert_with(Vec::new);
        curves.retain(|c| c.kind != profile::fire::KernelCurve::VECTOR);
        curves.extend(
            vectors
                .into_iter()
                .map(|(rows, curve)| at_4096(profile::fire::KernelCurve::VECTOR, rows, curve)),
        );
    }

    /// Rates whose vector kernels run at `few` TFLOP/s at every rows (none
    /// stated where `None`) and tiled kernels at `tiles` (by row tile), all
    /// at [`counts`]' 4096-square planes.
    fn rates(few: Option<f64>, tiles: Vec<(u32, Vec<(u32, f64)>)>) -> profile::fire::Rates {
        let mut r = profile::fire::Rates {
            launch_ns: Some(3000),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 10.0), (2048, 30.0)])]),
            hbm_gbps: Some(650.0),
            cached_gbps: None,
            elementwise_gbps: Some(700.0),
            kernel_tflops_by_shape: Some(
                tiles
                    .into_iter()
                    .map(|(tile, curve)| at_4096(profile::fire::KernelCurve::TILE, tile, curve))
                    .collect(),
            ),
            ..Default::default()
        };
        if let Some(t) = few {
            set_vectors(
                &mut r,
                gemm::VECTOR_ROWS
                    .map(|rows| (rows, vec![(1, t), (8192, t)]))
                    .to_vec(),
            );
        }
        r
    }

    /// A coopmat tile measured at a sixtieth of the curve's rate at 4 rows,
    /// rising to it at 256 and on to 8192.
    fn coopmat_tiles() -> Vec<(u32, Vec<(u32, f64)>)> {
        vec![(
            64,
            vec![
                (4, 0.16),
                (64, 2.5),
                (256, 10.0),
                (2048, 30.0),
                (8192, 30.0),
            ],
        )]
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
        let (dense, weighed) = choose(
            Target::Spirv,
            &rungs,
            coopmat,
            &r,
            &planes(&counts(), None),
            true,
        )
        .unwrap();
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
        assert!(dense.at(3).vector > 0);
        assert!(
            weighed[2].to_string().starts_with("rung 4: gemv x"),
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
        let (dense, weighed) = choose(
            Target::Spirv,
            &[128, 256],
            baseline,
            &r,
            &planes(&counts(), None),
            true,
        )
        .unwrap();
        assert!(
            dense.at(200).wide && dense.at(200).vector == 0,
            "{weighed:?}"
        );
        assert_eq!(weighed[1].costs.len(), gemm::VECTOR_ROWS.len() + 2);
    }

    /// A profile stating no kernel's curve, or none at the model's planes,
    /// is refused by the key; one stating the tiled kernel's alone picks it.
    #[test]
    fn a_missing_rate_is_refused_by_its_key() {
        let mut none = rates(Some(5.0), coopmat_tiles());
        none.kernel_tflops_by_shape = None;
        let refused = choose(
            Target::Spirv,
            &[1, 2],
            coopmat,
            &none,
            &planes(&counts(), None),
            true,
        );
        assert_eq!(refused.unwrap_err(), "kernel_tflops_by_shape");
        let mut elsewhere = rates(Some(5.0), coopmat_tiles());
        for c in elsewhere.kernel_tflops_by_shape.iter_mut().flatten() {
            c.n = 1024;
        }
        let refused = choose(
            Target::Spirv,
            &[1, 2],
            coopmat,
            &elsewhere,
            &planes(&counts(), None),
            true,
        );
        assert_eq!(refused.unwrap_err(), UNREACHED);
        let (dense, _) = choose(
            Target::Spirv,
            &[1, 2],
            coopmat,
            &rates(None, coopmat_tiles()),
            &planes(&counts(), None),
            true,
        )
        .unwrap();
        assert_eq!(dense.at(1).vector, 0);
    }

    /// Each kernel is weighed at the model's own planes: where the narrow
    /// tile measures faster at a 1024-wide plane than the wide one, a model
    /// of such planes takes it, whatever the kernels measure at planes the
    /// model does not hold (Qwen3-0.6B on the L40S's wgpu: the wide tile 2.5x
    /// faster at 4096 square, the narrow faster at its own planes).
    #[test]
    fn each_kernel_is_weighed_at_the_models_own_planes() {
        let tiled = |_| Tactic {
            tier: Tier::Subgroup,
            vector: 0,
            wide: false,
        };
        let shape = |n, k| engine::load::GemmShape {
            planes: 28,
            ..engine::load::GemmShape::dense(n, k, 2 * u64::from(n * k), 2 * u64::from(n * k))
        };
        let model = Counts {
            gemm_shapes: vec![shape(1024, 3072), shape(3072, 1024)],
            ..counts()
        };
        let curve = |kind, key, n, k, rate: f64| profile::fire::KernelCurve {
            kind,
            key,
            n,
            k,
            curve: vec![(1, rate / 100.0), (512, rate)],
            stop: None,
        };
        let tile =
            |wide, n, k| crate::rates::curve_key(Target::Wgsl, Tactic { wide, ..tiled(0) }, n, k).1;
        let tag = profile::fire::KernelCurve::TILE;
        let mut curves = Vec::new();
        for (n, k) in [(1024, 3072), (3072, 1024)] {
            curves.push(curve(tag, tile(false, n, k), n, k, 9.0));
            curves.push(curve(tag, tile(true, n, k), n, k, 6.0));
            for rows in gemm::VECTOR_ROWS {
                curves.push(curve(profile::fire::KernelCurve::VECTOR, rows, n, k, 0.5));
            }
        }
        // At 4096 square, not a plane of the model: the wide tile far ahead.
        curves.push(curve(tag, tile(true, 4096, 4096), 4096, 4096, 16.0));
        curves.push(curve(tag, tile(false, 4096, 4096), 4096, 4096, 6.4));
        let r = profile::fire::Rates {
            kernel_tflops_by_shape: Some(curves),
            ..rates(None, Vec::new())
        };
        let (dense, weighed) =
            choose(Target::Wgsl, &[512], tiled, &r, &planes(&model, None), true).unwrap();
        let at = dense.at(512);
        assert!(at.vector == 0 && !at.wide, "{}", weighed[0]);
    }

    /// A fire fired eagerly takes the fastest kernel at its own rows.
    #[test]
    fn an_eager_fire_takes_the_fastest_at_its_own_rows() {
        let r = rates(Some(5.0), coopmat_tiles());
        let rungs = engine::memory::rungs(1, 2048);
        let (eager, _) = choose(
            Target::Spirv,
            &rungs,
            coopmat,
            &r,
            &planes(&counts(), None),
            false,
        )
        .unwrap();
        for rows in [1, 3, 5, 17, 100, 129, 200, 256, 300, 2048] {
            let (_, w) = choose(
                Target::Spirv,
                &[rows],
                coopmat,
                &r,
                &planes(&counts(), None),
                true,
            )
            .unwrap();
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
            .map(|t| (t, vec![(4, 20.0), (8192, 20.0)]))
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
                let (_, weighed) =
                    choose(target, &rungs, tiled, &flat, &planes(&counts(), None), true).unwrap();
                for pair in weighed.windows(2) {
                    for (a, b) in pair[0].costs.iter().zip(&pair[1].costs) {
                        assert!(b.2 >= a.2, "{target:?} {tier:?}: {pair:?}");
                    }
                }
            }
        }
    }

    /// A measured rate holds the kernel's weight passes: a one-row-a-workgroup
    /// vector kernel measured faster at 3 rows (its re-reads cached) is taken
    /// there, though it makes three passes.
    #[test]
    fn a_measured_rate_holds_the_passes_a_cache_spares() {
        let mut r = rates(Some(1.0), coopmat_tiles());
        set_vectors(
            &mut r,
            vec![
                (1, vec![(1, 1.0), (4, 4.0), (256, 4.0)]),
                (4, vec![(1, 1.0), (4, 3.0), (256, 12.0)]),
            ],
        );
        let (eager, _) = choose(
            Target::Wgsl,
            &[1, 2, 4],
            coopmat,
            &r,
            &planes(&counts(), None),
            false,
        )
        .unwrap();
        assert_eq!(eager.at(3).vector, 1);
    }

    /// Below the first row count a kernel was timed at it takes that count's
    /// time, not its rate over fewer flops: a tile timed from 4 rows is not
    /// priced at a quarter of that at 1 row.
    #[test]
    fn below_its_first_timed_rows_a_kernel_takes_that_time() {
        let baseline = |_| Tactic {
            tier: Tier::Baseline,
            vector: 0,
            wide: false,
        };
        let tiles = vec![
            (32, vec![(4, 4.0), (256, 8.0)]),
            (256, vec![(4, 4.0), (256, 8.0)]),
        ];
        let r = rates(Some(1.0), tiles);
        let (_, one) = choose(
            Target::Spirv,
            &[1],
            baseline,
            &r,
            &planes(&counts(), None),
            true,
        )
        .unwrap();
        let (_, four) = choose(
            Target::Spirv,
            &[4],
            baseline,
            &r,
            &planes(&counts(), None),
            true,
        )
        .unwrap();
        let tiled = |w: &[Weighed]| w[0].costs.iter().find(|c| c.0.vector == 0).unwrap().2;
        assert_eq!(tiled(&one), tiled(&four));
    }

    /// A tiled kernel whose curve stops short of a rung is no candidate
    /// there (no pick from a rate it was never timed at); a rung past every
    /// curve keeps the rung below's kernel, and a first rung past them all
    /// is refused by name.
    #[test]
    fn a_curve_that_stops_short_is_no_candidate_past_it() {
        let baseline = |_| Tactic {
            tier: Tier::Baseline,
            vector: 0,
            wide: true,
        };
        // The 32-row tile measured fast but only to 256 rows; the register
        // tile measured to 4096, slower.
        let tiles = vec![
            (32, vec![(1, 0.5), (256, 50.0)]),
            (256, vec![(1, 0.1), (256, 5.0), (4096, 8.0)]),
        ];
        let mut r = rates(Some(0.01), tiles);
        set_vectors(&mut r, vec![(4, vec![(1, 0.01), (4096, 0.01)])]);
        let (dense, weighed) = choose(
            Target::Spirv,
            &[256, 512, 1024],
            baseline,
            &r,
            &planes(&counts(), None),
            true,
        )
        .unwrap();
        assert!(!dense.at(256).wide, "{weighed:?}");
        for rows in [512, 1024] {
            assert!(dense.at(rows).wide, "{rows}: {weighed:?}");
        }
        assert!(weighed[1].costs.iter().all(|c| c.0.wide || c.0.vector > 0));
        // No kernel timed at 8192 rows: the rung keeps 4096's; alone, it is
        // refused by name.
        let (dense, weighed) = choose(
            Target::Spirv,
            &[4096, 8192],
            baseline,
            &r,
            &planes(&counts(), None),
            true,
        )
        .unwrap();
        assert_eq!(dense.at(8192), dense.at(4096));
        assert!(
            weighed[1].held && weighed[1].to_string().contains("held"),
            "{}",
            weighed[1]
        );
        let refused = choose(
            Target::Spirv,
            &[8192],
            baseline,
            &r,
            &planes(&counts(), None),
            true,
        );
        assert_eq!(refused.unwrap_err(), UNREACHED);
    }
}
