//! The dense matmul's kernel: the vector kernels or one of the tiled ones,
//! whichever the estimate (`engine::roofline`) weighs fastest over the
//! model's dense planes, from the device's measured rates: each kernel at
//! each of the model's own plane shapes at its rate measured there, at the
//! rows it lands (`kernel_curves`, timed once the model's shapes
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

/// The tactic a matmul of some rows runs: at each of the model's planes its
/// own pick, else the pick weighed over the model's planes together.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dense {
    /// `(rows, tactic)`, ascending: a rung's top and its tactic where fires
    /// are recorded a rung, else every row count to the top; weighed over
    /// every plane together (a plane of none of `planes`' shapes).
    by_rows: Vec<(u32, Tactic)>,
    /// Each plane's own `(n, k)` and its `by_rows`, weighed at it alone.
    planes: Vec<((u32, u32), Vec<(u32, Tactic)>)>,
}

/// The tactic of `by_rows` (ascending) a matmul of `rows` runs: that of the
/// fewest rows stated that hold them, the most past them all.
fn at_rows(by_rows: &[(u32, Tactic)], rows: u32) -> Tactic {
    by_rows
        .iter()
        .find(|&&(at, _)| at >= rows)
        .or(by_rows.last())
        .map(|&(_, tactic)| tactic)
        .expect("a load has one rung at least")
}

/// `by_rows` said compactly: `rows-rows name` for every run of one tactic.
fn runs_of(by_rows: &[(u32, Tactic)]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut from = 1;
    for (i, &(rows, tactic)) in by_rows.iter().enumerate() {
        if by_rows.get(i + 1).is_none_or(|&(_, next)| next != tactic) {
            out.push(format!("{from}-{rows} {}", name(&tactic)));
            from = rows + 1;
        }
    }
    out.join(", ")
}

impl Dense {
    /// The profile path's kernel at each of `rungs` (`tiled`), every plane
    /// alike: what a load picks before its curves are timed.
    #[must_use]
    pub fn path(rungs: &[u32], tiled: impl Fn(u32) -> Tactic) -> Dense {
        Dense {
            by_rows: rungs.iter().map(|&rows| (rows, tiled(rows))).collect(),
            planes: Vec::new(),
        }
    }

    /// The tactic each run of row counts takes at each plane, a line a
    /// plane: `n x k: rows-rows name, ...`.
    #[must_use]
    pub fn runs(&self) -> Vec<String> {
        self.planes
            .iter()
            .map(|((n, k), by_rows)| format!("{n}x{k}: {}", runs_of(by_rows)))
            .collect()
    }

    /// The tactic a matmul of `rows` over an `n x k` plane runs: the
    /// plane's own pick, else the pick over the model's planes together.
    #[must_use]
    pub fn at(&self, rows: u32, n: u32, k: u32) -> Tactic {
        match self.planes.iter().find(|(at, _)| *at == (n, k)) {
            Some((_, by_rows)) => at_rows(by_rows, rows),
            None => at_rows(&self.by_rows, rows),
        }
    }

    /// This choice with `tactic` taken by every plane over rows `bottom..=
    /// top` (a rung's rows, or a row count's on an eager backend).
    #[must_use]
    pub fn with_rows(&self, bottom: u32, top: u32, tactic: Tactic) -> Dense {
        let set = |by_rows: &[(u32, Tactic)]| -> Vec<(u32, Tactic)> {
            by_rows
                .iter()
                .map(|&(at, t)| {
                    (
                        at,
                        if (bottom..=top).contains(&at) {
                            tactic
                        } else {
                            t
                        },
                    )
                })
                .collect()
        };
        Dense {
            by_rows: set(&self.by_rows),
            planes: self.planes.iter().map(|(at, by)| (*at, set(by))).collect(),
        }
    }

    /// The kernel each plane takes at `rows`, named.
    #[must_use]
    pub fn runs_at(&self, rows: u32) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for (_, by) in &self.planes {
            let n = name(&at_rows(by, rows));
            if !out.contains(&n) {
                out.push(n);
            }
        }
        if out.is_empty() {
            out.push(name(&at_rows(&self.by_rows, rows)));
        }
        out
    }

    /// Every vector kernel this choice takes at any rows, over every plane.
    #[must_use]
    pub fn vector_kernels(&self) -> Vec<Tactic> {
        let mut all: Vec<Tactic> = Vec::new();
        let every = self
            .by_rows
            .iter()
            .chain(self.planes.iter().flat_map(|(_, by)| by.iter()));
        for &(_, t) in every {
            if t.vector > 0 && !all.contains(&t) {
                all.push(t);
            }
        }
        all
    }

    /// The rows the choice has an entry at: a recording backend's rung tops,
    /// an eager one's row counts.
    #[must_use]
    pub fn entries(&self) -> Vec<u32> {
        self.by_rows.iter().map(|&(at, _)| at).collect()
    }

    /// Every tactic a fire of `rows` runs at some plane: what its rows are
    /// padded for (the most any of them pads).
    #[must_use]
    pub fn tactics_at(&self, rows: u32) -> Vec<Tactic> {
        let mut all = vec![at_rows(&self.by_rows, rows)];
        for (_, by_rows) in &self.planes {
            let tactic = at_rows(by_rows, rows);
            if !all.contains(&tactic) {
                all.push(tactic);
            }
        }
        all
    }
}

/// A tier as a number, as a settle's rows state it.
fn tier_code(tier: kernels_slang::Tier) -> u32 {
    match tier {
        kernels_slang::Tier::Baseline => 0,
        kernels_slang::Tier::Subgroup => 1,
        kernels_slang::Tier::Fp16 => 2,
        kernels_slang::Tier::Coopmat => 3,
    }
}

impl Dense {
    /// This choice in canonical words: each run of rows and its kernel as
    /// numbers, the planes' own picks sorted by plane. Content alone, so a
    /// key built from it is the same whatever order the choice was built in.
    #[must_use]
    pub fn canonical(&self) -> String {
        let rows = |by: &[(u32, Tactic)]| {
            by.iter()
                .map(|&(r, t)| format!("{r}:{}.{}.{}", t.vector, u8::from(t.wide), tier_code(t.tier)))
                .collect::<Vec<_>>()
                .join(",")
        };
        let mut planes: Vec<&((u32, u32), Vec<(u32, Tactic)>)> = self.planes.iter().collect();
        planes.sort_by_key(|(at, _)| *at);
        let mut out = format!("all[{}]", rows(&self.by_rows));
        for ((n, k), by) in planes {
            out.push_str(&format!(" {n}x{k}[{}]", rows(by)));
        }
        out
    }
}

/// Which model's settle a stored one is (`profile::curves::DenseSettled::of`):
/// a fingerprint of the model's dense planes, the load's lanes, its top rows
/// and the target, in canonical words (each list sorted, every number as it
/// reads back from a profile). One entry a model a device: a model's settle
/// stored again replaces its own, never another's, so no entry outlives
/// its model's last settle. No curve keys it: the curves decide a settle's
/// outcome only through the kernel each range takes, which its `code` holds
/// ([`settle_code`]), so another model's curves, or a curve's points that
/// move no pick, leave it.
#[must_use]
pub fn settle_of(planes: &[(u32, u32, u32, f64)], lanes: u32, top: u32, target: Target) -> u32 {
    let mut planes: Vec<String> = planes
        .iter()
        .map(|&(n, k, count, bytes)| format!("{n}x{k}*{count}:{bytes}"))
        .collect();
    planes.sort();
    let target = match target {
        Target::Spirv => "spirv",
        Target::Wgsl => "wgsl",
    };
    profile::rates::fingerprint(
        format!("planes {}|lanes {lanes}|top {top}|{target}", planes.join(" ")).as_bytes(),
    )
}

/// What a stored settle's ranges were weighed against
/// (`profile::curves::DenseSettled::code`): the code of every kernel the
/// load's fires can ask for (`kernels`, `crate::stored::code_of_all`) and
/// the kernel the curves picked at each range ([`Dense::canonical`]). A
/// change of either settles the model again.
#[must_use]
pub fn settle_code(kernels: u32, dense: &Dense) -> u32 {
    profile::rates::fingerprint(format!("kernels {kernels:#010x}|picks {}", dense.canonical()).as_bytes())
}

/// Which of a settle's candidates' timings (`samples`, each candidate's
/// fires, the incumbent first) decides its range: the one whose median
/// widened by its spread (`median · (1 + spread)`) lies under every other's
/// median narrowed by its own (`median · (1 − spread)`), so no repeat's noise
/// could turn the two around; `None` where no candidate separates so (a tie
/// inside the spreads: the incumbent stands). A candidate with no timing
/// takes no part. Derived from the measured spreads alone, no margin chosen.
#[must_use]
pub fn settle_pick(samples: &[Vec<f64>]) -> Option<usize> {
    let timed: Vec<(usize, f64, f64)> = samples
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.is_empty())
        .map(|(i, s)| {
            let (median, spread) = crate::rates::median_spread(s);
            (i, median, spread)
        })
        .collect();
    let &(winner, median, spread) = timed.iter().min_by(|a, b| a.1.total_cmp(&b.1))?;
    let high = median * (1.0 + spread);
    timed
        .iter()
        .filter(|&&(i, ..)| i != winner)
        .all(|&(_, m, sp)| high < m * (1.0 - sp))
        .then_some(winner)
}

/// The name a kernel is said by.
pub(crate) fn name(tactic: &Tactic) -> String {
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
                "rung {}: {} held from the rung below, or the profile's path's below none (no \
                 kernel was timed at these rows at any of the model's planes)",
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

/// What a load is refused by where its profile states no kernel's curve at
/// all: the key it lacks.
pub const UNREACHED: &str = profile::rates::KERNEL_CURVES;

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

/// `tactic`'s time, ns, at `rows` of one `n x k` plane by its curve there
/// (`rates`): past the curve's last point at that point's rate, never
/// below it. `None` where no point was timed at the plane.
#[must_use]
pub fn plane_ns(
    target: Target,
    rates: &profile::rates::Rates,
    tactic: Tactic,
    (n, k): (u32, u32),
    rows: u32,
) -> Option<f64> {
    let curves = rates.kernel_curves.as_deref()?;
    let (kind, key) = crate::rates::curve_key(target, tactic, n, k);
    let found = curves.iter().find(|c| c.id() == (kind, key, n, k))?;
    let (&(first, _), &(most, last)) = (found.curve.first()?, found.curve.last()?);
    let flops = |at: u32| 2.0 * f64::from(at) * f64::from(n) * f64::from(k);
    if rows > most {
        return (last > 0.0).then(|| flops(rows) / (last * 1e3));
    }
    let at = rows.max(first);
    profile::curves::interpolate(&found.curve, f64::from(at)).map(|t| flops(at) / (t * 1e3))
}

/// How many times `standing`'s dense matmul time at `rows` over `planes`
/// `option`'s takes, by the kernels' curves (`rates`): the most it could
/// lengthen a step `standing` is estimated at. `None` where a curve does
/// not price a kernel either takes at a plane.
#[must_use]
pub fn slower_by(
    target: Target,
    rates: &profile::rates::Rates,
    planes: &[(u32, u32, u32, f64)],
    standing: &Dense,
    option: &Dense,
    rows: u32,
) -> Option<f64> {
    let curves = rates.kernel_curves.as_deref()?;
    let time = |tactic: Tactic, n: u32, k: u32| -> Option<f64> {
        let (kind, key) = crate::rates::curve_key(target, tactic, n, k);
        let found = curves.iter().find(|c| c.id() == (kind, key, n, k))?;
        let (&(first, _), &(most, _)) = (found.curve.first()?, found.curve.last()?);
        if rows > most {
            return None;
        }
        let at = rows.max(first);
        let flops = 2.0 * f64::from(at) * f64::from(n) * f64::from(k);
        profile::curves::interpolate(&found.curve, f64::from(at)).map(|t| flops / (t * 1e3))
    };
    let (mut was, mut is) = (0.0, 0.0);
    for &(n, k, count, _) in planes {
        was += f64::from(count) * time(standing.at(rows, n, k), n, k)?;
        is += f64::from(count) * time(option.at(rows, n, k), n, k)?;
    }
    (was > 0.0).then(|| is / was)
}

/// The dense tactic over `rungs` (ascending rows) on `target`, the model's
/// dense planes (`planes`, [`planes`]: each a dispatch reading its plane)
/// weighed over `rates`; `full` the rows the chunker cuts a prompt by, whose
/// rung (where every full chunk lands, at its top) is picked at its top rows
/// alone; `tiled` is the
/// backend's tiled tactic at some rows. One a rung where the backend
/// `records` its fires, else one a row count to the top rung. Each rung's
/// weighing beside it; the `[rates]` key it lacks where the profile states
/// none of one it reads.
pub fn choose(
    target: Target,
    rungs: &[u32],
    tiled: impl Fn(u32) -> Tactic,
    rates: &profile::rates::Rates,
    planes: &[(u32, u32, u32, f64)],
    records: bool,
    full: Option<u32>,
) -> Result<(Dense, Vec<Weighed>), &'static str> {
    let launch_ns = Constants::of(rates)?.launch_ns;
    weigh(target, rungs, &tiled, rates, launch_ns, planes, records, full)
}

/// [`choose`] where no roofline is stated, under a load that skips
/// measuring (`engine::rates::Measuring::Skip`): the kernels are weighed by
/// the curves stated (a committed profile's, or the user's) as [`choose`]
/// weighs them, each plane's launch at `[rates] launch_ns` where stated,
/// else none, alike for every kernel a plane takes (a launch a plane, the
/// same whichever kernel runs it). [`UNREACHED`] where no curve is stated.
pub fn choose_unrated(
    target: Target,
    rungs: &[u32],
    tiled: impl Fn(u32) -> Tactic,
    rates: &profile::rates::Rates,
    planes: &[(u32, u32, u32, f64)],
    records: bool,
    full: Option<u32>,
) -> Result<(Dense, Vec<Weighed>), &'static str> {
    let launch_ns = rates.launch_ns.map_or(0.0, |ns| ns as f64);
    weigh(target, rungs, &tiled, rates, launch_ns, planes, records, full)
}

/// [`choose`] at `launch_ns` a plane.
#[allow(clippy::too_many_arguments)]
fn weigh(
    target: Target,
    rungs: &[u32],
    tiled: &dyn Fn(u32) -> Tactic,
    rates: &profile::rates::Rates,
    launch_ns: f64,
    planes: &[(u32, u32, u32, f64)],
    records: bool,
    full: Option<u32>,
) -> Result<(Dense, Vec<Weighed>), &'static str> {
    let (by_rows, weighed) = pick(
        target, rungs, tiled, rates, launch_ns, planes, records, full, None,
    )?;
    // What each plane's own pick must beat, and what a plane no kernel was
    // timed at keeps: the one kernel a rung (or a row count) weighed over
    // every plane together, as a load picked before planes picked their own.
    let together = |rows: u32| at_rows(&by_rows, rows);
    // Each plane its own pick, by its own curves (a kernel beaten at one
    // plane, an output head's, may win another): a plane no kernel was timed
    // at takes the pick over every plane together.
    let mut own: Vec<((u32, u32), Vec<(u32, Tactic)>)> = Vec::new();
    for plane @ &(n, k, ..) in planes {
        if own.iter().any(|(at, _)| *at == (n, k)) {
            continue;
        }
        if let Ok((by_rows, _)) = pick(
            target,
            rungs,
            tiled,
            rates,
            launch_ns,
            &[*plane],
            records,
            full,
            Some(&together),
        ) {
            own.push(((n, k), by_rows));
        }
    }
    Ok((
        Dense {
            by_rows,
            planes: own,
        },
        weighed,
    ))
}

/// [`choose`] over `planes` together: one tactic a rung (or a row count)
/// for all of them, and each rung's weighing. With an `incumbent` (the pick
/// over every plane together, for one plane's own), the pick replaces it
/// only past the two kernels' measured spread, and a rung no kernel was
/// timed at keeps it.
#[allow(clippy::too_many_arguments)]
fn pick(
    target: Target,
    rungs: &[u32],
    tiled: &dyn Fn(u32) -> Tactic,
    rates: &profile::rates::Rates,
    launch_ns: f64,
    planes: &[(u32, u32, u32, f64)],
    records: bool,
    full: Option<u32>,
    incumbent: Option<&dyn Fn(u32) -> Tactic>,
) -> Result<(Vec<(u32, Tactic)>, Vec<Weighed>), &'static str> {
    let curves = rates.kernel_curves.as_deref().ok_or(UNREACHED)?;
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
    // `tactic`'s time at an `n x k` plane at `rows`, ns, and whether it is
    // measured there (else carried past the curve's last point): within the
    // curve, its rate there; past it (the curve stopped short, by the
    // bound, the buffer, a beating or no measurement), its last point's
    // rate held, so the time only grows with the rows from the last point's
    // (never below it scaled by rows: a kernel past its curve is never
    // cheaper than where it was last timed). `None` where no point was
    // timed at the plane.
    let timed_at = |tactic: Tactic, n: u32, k: u32, rows: u32| -> Option<(f64, bool)> {
        let (kind, key) = crate::rates::curve_key(target, tactic, n, k);
        let found = curves.iter().find(|c| c.id() == (kind, key, n, k))?;
        let curve = &found.curve;
        let (&(first, _), &(most, last)) = (curve.first()?, curve.last()?);
        let flops = |at: u32| 2.0 * f64::from(at) * f64::from(n) * f64::from(k);
        if rows > most {
            return (last > 0.0).then(|| (flops(rows) / (last * 1e3), false));
        }
        // Below its first point, a kernel takes that point's time; and never
        // less than at any point of fewer rows (more rows are never less
        // work: a curve whose rate rises faster than its rows between two
        // noisy points does not make the rows between them cheaper).
        let at = rows.max(first);
        let floor = curve
            .iter()
            .filter(|&&(r, _)| r <= at)
            .map(|&(r, t)| flops(r) / (t * 1e3))
            .fold(0.0, f64::max);
        profile::curves::interpolate(curve, f64::from(at))
            .map(|t| ((flops(at) / (t * 1e3)).max(floor), true))
    };
    let time_at =
        |tactic: Tactic, n: u32, k: u32, rows: u32| timed_at(tactic, n, k, rows).map(|(ns, _)| ns);
    // `tactic`'s cost at `rows` over every plane some candidate of `among`
    // was timed at to the rows, in ns: each at the kernel's rate measured
    // there (every weight pass it makes and whatever a cache spares it in the
    // time) and a launch. `None` where its curve at one of them does not
    // reach the rows, or no plane was timed to them.
    // The planes `among` are weighed at, at `rows`: those every candidate
    // timed at some plane to the rows was timed at (a plane some of them
    // were not tells nothing between them; its own pick sees it).
    let weighed_at = |among: &[Tactic], n: u32, k: u32, rows: u32| -> bool {
        let priced = |t: Tactic| time_at(t, n, k, rows).is_some();
        let timed_somewhere = |t: Tactic| {
            planes
                .iter()
                .any(|&(pn, pk, ..)| time_at(t, pn, pk, rows).is_some())
        };
        among.iter().any(|&t| priced(t)) && among.iter().all(|&t| priced(t) || !timed_somewhere(t))
    };
    // Whether `tactic`'s curve at some plane stopped as beaten short of
    // `rows` (`KernelCurve::BEATEN`): shown slower there than a tile.
    let beaten_by = |tactic: Tactic, rows: u32| -> bool {
        planes.iter().any(|&(n, k, ..)| {
            let (kind, key) = crate::rates::curve_key(target, tactic, n, k);
            curves.iter().any(|c| {
                c.id() == (kind, key, n, k)
                    && matches!(c.stop, Some((at, why))
                        if why == profile::curves::KernelCurve::BEATEN && rows >= at)
            })
        })
    };
    // `tactic`'s measured spread at an `n x k` plane at `rows` (its points'
    // repeats, `(max - min) / median`, between its points in log rows): 0
    // where none was measured.
    let spread_at = |tactic: Tactic, n: u32, k: u32, rows: u32| -> f64 {
        let (kind, key) = crate::rates::curve_key(target, tactic, n, k);
        curves
            .iter()
            .find(|c| c.id() == (kind, key, n, k))
            .and_then(|c| profile::curves::interpolate(&c.spread, f64::from(rows)))
            .unwrap_or(0.0)
    };
    // `tactic`'s spread at `rows` over every plane some candidate of
    // `among` was timed at to the rows: the widest of them.
    let noise = |among: &[Tactic], tactic: Tactic, rows: u32| -> f64 {
        planes
            .iter()
            .filter(|&&(n, k, ..)| weighed_at(among, n, k, rows))
            .map(|&(n, k, ..)| spread_at(tactic, n, k, rows))
            .fold(0.0, f64::max)
    };
    let cost = |among: &[Tactic], tactic: Tactic, rows: u32| -> Option<f64> {
        let mut ns = 0.0;
        let mut weighed = false;
        for &(n, k, count, _) in planes {
            if !weighed_at(among, n, k, rows) {
                continue;
            }
            ns += f64::from(count) * (launch_ns + time_at(tactic, n, k, rows)?);
            weighed = true;
        }
        weighed.then_some(ns)
    };
    // Whether `tactic`'s cost at `rows` over the planes `among` is weighed
    // at is measured at every one (none carried past a curve's last point).
    let measured = |among: &[Tactic], tactic: Tactic, rows: u32| -> bool {
        planes.iter().all(|&(n, k, ..)| {
            !weighed_at(among, n, k, rows)
                || timed_at(tactic, n, k, rows).is_some_and(|(_, measured)| measured)
        })
    };
    // The candidates that may be taken at `bottom..=top` rows: those
    // measured at every weighed plane at both. A kernel carried past its
    // curve's last point at a plane still weighs that plane for the others
    // (it is not dropped from the sum, so no estimate falls as the rows
    // grow), but it never wins by the carried cost; where no kernel is
    // measured, the rung keeps the kernel below it.
    let takeable = |among: &[Tactic], t: Tactic, bottom: u32, top: u32| -> bool {
        measured(among, t, bottom) && measured(among, t, top)
    };

    let mut by_rows: Vec<(u32, Tactic)> = Vec::new();
    let mut weighed = Vec::with_capacity(rungs.len());
    let mut below = 0;
    for &rung in rungs {
        // The rung full prompt chunks land on is picked at its top rows,
        // where they land; every other rung over the rows it serves.
        let bottom = if full == Some(rung) {
            rung
        } else {
            (below + 1).min(rung)
        };
        let among = candidates(rung);
        // The candidates timed over the rung's rows, bottom to top.
        let costs: Vec<(Tactic, f64, f64)> = among
            .iter()
            .filter(|&&t| takeable(&among, t, bottom, rung))
            .filter_map(|&t| Some((t, cost(&among, t, bottom)?, cost(&among, t, rung)?)))
            .collect();
        if costs.is_empty() {
            // Past every curve at every plane: the incumbent's kernel, else
            // the rung below's, else (no kernel timed at any rows, every
            // curve stopped at its first by the bound or the buffer) the
            // profile's path's own.
            let kept = incumbent
                .map(|of| of(rung))
                .unwrap_or_else(|| by_rows.last().map_or_else(|| tiled(rung), |&(_, t)| t));
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
        let mut taken = Weighed::least_regret(&costs, &rows_of);
        // The incumbent keeps the rung unless the pick beats it by more than
        // the two kernels' measured spread there: a gap narrower than their
        // repeats spread over is no difference.
        let path = incumbent.map(|of| of(rung));
        // An incumbent no curve at the plane prices keeps the rung, unless
        // its curve there stopped as beaten.
        let unpriced = path.filter(|&path| {
            costs.iter().all(|c| c.0 != path) && !beaten_by(path, bottom) && !beaten_by(path, rung)
        });
        if let Some((path, kept)) =
            path.and_then(|path| Some((path, costs.iter().position(|c| c.0 == path)?)))
        {
            let best = |of: fn(&(Tactic, f64, f64)) -> f64| {
                costs
                    .iter()
                    .map(of)
                    .fold(f64::INFINITY, f64::min)
                    .max(f64::MIN_POSITIVE)
            };
            let (low, high) = (best(|c| c.1), best(|c| c.2));
            let regret = |c: &(Tactic, f64, f64)| (c.1 / low).max(c.2 / high);
            let spread = |t: Tactic| noise(&among, t, bottom).max(noise(&among, t, rung));
            let gap = regret(&costs[kept]) - regret(&costs[taken]);
            if taken != kept && gap <= spread(costs[taken].0) + spread(path) {
                taken = kept;
            }
        }
        if records {
            by_rows.push((rung, unpriced.unwrap_or(costs[taken].0)));
        } else {
            for rows in bottom..=rung {
                let among = candidates(rows);
                let priced: Vec<(Tactic, f64)> = among
                    .iter()
                    .filter(|&&t| takeable(&among, t, rows, rows))
                    .filter_map(|&t| Some((t, cost(&among, t, rows)?)))
                    .collect();
                let fastest = priced
                    .iter()
                    .copied()
                    .min_by(|a, b| a.1.total_cmp(&b.1).then(rows_of(&a.0).cmp(&rows_of(&b.0))));
                let path = incumbent.map(|of| of(rows));
                // The incumbent unless the fastest beats it by more than
                // their spreads at these rows.
                let priced_path = path.and_then(|path| priced.iter().find(|p| p.0 == path));
                let pick = match (fastest, priced_path) {
                    (Some((t, ns)), Some(&(path, path_ns))) => {
                        let spread = noise(&among, t, rows) + noise(&among, path, rows);
                        if (path_ns - ns) / path_ns.max(f64::MIN_POSITIVE) > spread {
                            Some(t)
                        } else {
                            Some(path)
                        }
                    }
                    // An incumbent no curve here prices keeps the rows.
                    (fastest, None) => path
                        .filter(|&path| !beaten_by(path, rows))
                        .or(fastest.map(|(t, _)| t)),
                    (None, Some(&(path, _))) => Some(path),
                };
                // The kernel the rows below took keeps these unless the pick
                // beats it by more than their spreads: one fire's noise
                // does not flip the kernel row to row.
                let before = by_rows.last().map(|&(_, t)| t);
                let pick = match (pick, before) {
                    (Some(t), Some(before)) if t != before => {
                        let ns = |of: Tactic| priced.iter().find(|p| p.0 == of).map(|p| p.1);
                        match (ns(t), ns(before)) {
                            (Some(ns), Some(before_ns))
                                if (before_ns - ns) / before_ns.max(f64::MIN_POSITIVE)
                                    <= noise(&among, t, rows) + noise(&among, before, rows) =>
                            {
                                Some(before)
                            }
                            _ => Some(t),
                        }
                    }
                    (pick, _) => pick,
                };
                let pick = pick.or(path).or(before).unwrap_or_else(|| tiled(rows));
                by_rows.push((rows, pick));
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
    Ok((by_rows, weighed))
}

/// The work a kernel does over the model's dense planes at some rows, from
/// its shape alone ([`gemm::tile_for`]): the weight bytes it reads (a plane
/// once a workgroup's rows, [`gemm::Kernel::rows_a_read`]) and the flops it
/// computes (the rows padded to its row tile, [`gemm::Kernel::row_tile`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shaped {
    pub bytes: f64,
    pub flops: f64,
}

impl Shaped {
    /// `tactic`'s work at `rows` over `planes` on `target`.
    #[must_use]
    pub fn of(target: Target, tactic: Tactic, planes: &[(u32, u32, u32, f64)], rows: u32) -> Shaped {
        let mut work = Shaped {
            bytes: 0.0,
            flops: 0.0,
        };
        for &(n, k, count, bytes) in planes {
            let kernel = gemm::tile_for(target, tactic, n, k);
            let reads = rows.max(1).div_ceil(kernel.rows_a_read().max(1));
            let tile = kernel.row_tile().max(1);
            let computed = rows.max(1).div_ceil(tile) * tile;
            work.bytes += f64::from(count) * f64::from(reads) * bytes;
            work.flops += f64::from(count) * 2.0 * f64::from(computed) * f64::from(n) * f64::from(k);
        }
        work
    }

    /// No more of either than `other`.
    fn within(self, other: Shaped) -> bool {
        self.bytes <= other.bytes && self.flops <= other.flops
    }
}

/// [`choose`] where no kernel's curve is stated, under a load that skips
/// measuring (`engine::rates::Measuring::Skip`): each rung's kernel
/// estimated from the work's shape ([`Shaped`]) at its bottom and top rows
/// (each row count's own on an eager backend).
///
/// - With the device's roofline stated (`roofline`): each candidate's cost
///   a plane its launch and the longer of its weight reads at the stated
///   bandwidth and its padded flops at the stated matmul rate at the rows
///   it computes, the least regret taken as [`choose`] takes it.
/// - With none: the candidate whose reads and flops are each no more than
///   every other's at both ends (by the roofline, `max(bytes / bandwidth,
///   flops / rate)`, it is no slower on any device); ties to the fewest
///   rows a workgroup. A vector kernel reads each plane once for the rows
///   one workgroup carries ([`gemm::VECTOR_ROWS`]) and pads none, so it is
///   that kernel up to the most rows a vector workgroup carries. Past them
///   it reads more often and a tile pads more: which is faster turns on the
///   bandwidth and flops no layer states, and the rung keeps the profile
///   path's kernel (`tiled`).
///
/// Each rung's estimate, in words, beside the choice.
pub fn by_shape(
    target: Target,
    rungs: &[u32],
    tiled: impl Fn(u32) -> Tactic,
    roofline: Option<&Constants>,
    planes: &[(u32, u32, u32, f64)],
    records: bool,
    full: Option<u32>,
) -> (Dense, Vec<String>) {
    let candidates = |rows: u32| -> Vec<Tactic> {
        let base = tiled(rows);
        let mut all: Vec<Tactic> = gemm::VECTOR_ROWS
            .iter()
            .map(|&vector| Tactic { vector, ..base })
            .collect();
        all.extend(tiled_kernels(target, base));
        all
    };
    let rows_of = |t: &Tactic| rows_a_workgroup(target, t);
    // The roofline's cost of `t` at `rows`, ns: a launch a plane and the
    // longer of the plane's reads and its padded compute.
    let cost = |c: &Constants, t: Tactic, rows: u32| -> f64 {
        planes
            .iter()
            .map(|&(n, k, count, bytes)| {
                let one = Shaped::of(target, t, &[(n, k, 1, bytes)], rows);
                let computed = (one.flops / (2.0 * f64::from(n) * f64::from(k))) as u32;
                let read = one.bytes / c.bytes_per_ns.max(f64::MIN_POSITIVE);
                let compute = one.flops / c.gemm_rate(computed.max(1), n, k).max(f64::MIN_POSITIVE);
                f64::from(count) * (c.launch_ns + read.max(compute))
            })
            .sum()
    };
    // The kernel taken over `bottom..=top` rows, and why.
    let decide = |bottom: u32, top: u32| -> (Tactic, String) {
        let among = candidates(top);
        match roofline {
            Some(c) => {
                let costs: Vec<(Tactic, f64, f64)> = among
                    .iter()
                    .map(|&t| (t, cost(c, t, bottom), cost(c, t, top)))
                    .collect();
                let taken = Weighed::least_regret(&costs, &rows_of);
                let at = |pick: fn(&(Tactic, f64, f64)) -> f64| {
                    costs
                        .iter()
                        .map(|c| format!("{} {:.1} µs", name(&c.0), pick(c) / 1e3))
                        .collect::<Vec<_>>()
                        .join(" vs ")
                };
                (
                    costs[taken].0,
                    format!(
                        "{} by the stated roofline over the work's shape ({top} rows: {}; \
                         {bottom} rows: {})",
                        name(&costs[taken].0),
                        at(|c| c.2),
                        at(|c| c.1)
                    ),
                )
            }
            None => {
                let work = |t: Tactic| {
                    (
                        Shaped::of(target, t, planes, bottom),
                        Shaped::of(target, t, planes, top),
                    )
                };
                let no_worse = |t: Tactic| {
                    let (b, p) = work(t);
                    among.iter().all(|&o| {
                        let (ob, op) = work(o);
                        b.within(ob) && p.within(op)
                    })
                };
                let said = |t: Tactic| {
                    let (b, p) = work(t);
                    format!(
                        "{} {:.1} MB read, {:.2} GFLOP at {top} rows ({:.1} MB, {:.2} GFLOP at \
                         {bottom})",
                        name(&t),
                        p.bytes / 1e6,
                        p.flops / 1e9,
                        b.bytes / 1e6,
                        b.flops / 1e9
                    )
                };
                match among
                    .iter()
                    .copied()
                    .filter(|&t| no_worse(t))
                    .min_by_key(|t| rows_of(t))
                {
                    Some(t) => (
                        t,
                        format!(
                            "{} by the work's shape: it reads no more weight and computes no \
                             more rows than any other kernel, so no bandwidth or flops makes \
                             another faster ({})",
                            name(&t),
                            said(t)
                        ),
                    ),
                    None => {
                        let path = tiled(top);
                        (
                            path,
                            format!(
                                "{} the profile path's: no kernel reads and computes no more \
                                 than every other (fewer weight reads trade against fewer \
                                 padded rows, which the device's bandwidth and flops weigh, and \
                                 no layer states them: {})",
                                name(&path),
                                among.iter().map(|&t| said(t)).collect::<Vec<_>>().join("; ")
                            ),
                        )
                    }
                }
            }
        }
    };
    let mut by_rows: Vec<(u32, Tactic)> = Vec::new();
    let mut said = Vec::with_capacity(rungs.len());
    let mut below = 0;
    for &rung in rungs {
        let bottom = if full == Some(rung) {
            rung
        } else {
            (below + 1).min(rung)
        };
        if records {
            let (t, why) = decide(bottom, rung);
            by_rows.push((rung, t));
            said.push(format!("rung {rung} (rows {bottom}-{rung}): {why}"));
        } else {
            for rows in bottom..=rung {
                by_rows.push((rows, decide(rows, rows).0));
            }
            let (_, why) = decide(rung, rung);
            said.push(format!("rows {bottom}-{rung}: at {rung} rows, {why}"));
        }
        below = rung;
    }
    (
        Dense {
            by_rows,
            planes: Vec::new(),
        },
        said,
    )
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
    fn at_4096(kind: u32, key: u32, curve: Vec<(u32, f64)>) -> profile::curves::KernelCurve {
        profile::curves::KernelCurve {
            by: profile::rates::MeasuredBy::default(),
            kind,
            key,
            n: 4096,
            k: 4096,
            curve,
            stop: None,
            spread: Vec::new(),
        }
    }

    /// `r` with its vector kernels' curves at 4096 square `vectors`, `(rows a
    /// workgroup carries, curve)`.
    fn set_vectors(r: &mut profile::rates::Rates, vectors: Vec<(u32, Vec<(u32, f64)>)>) {
        let curves = r.kernel_curves.get_or_insert_with(Vec::new);
        curves.retain(|c| c.kind != profile::curves::KernelCurve::VECTOR);
        curves.extend(
            vectors
                .into_iter()
                .map(|(rows, curve)| at_4096(profile::curves::KernelCurve::VECTOR, rows, curve)),
        );
    }

    /// Rates whose vector kernels run at `few` TFLOP/s at every rows (none
    /// stated where `None`) and tiled kernels at `tiles` (by row tile), all
    /// at [`counts`]' 4096-square planes.
    fn rates(few: Option<f64>, tiles: Vec<(u32, Vec<(u32, f64)>)>) -> profile::rates::Rates {
        let mut r = profile::rates::Rates {
            launch_ns: Some(3000),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 10.0), (2048, 30.0)])]),
            hbm_gbps: Some(650.0),
            cached_gbps: None,
            elementwise_gbps: Some(700.0),
            kernel_curves: Some(
                tiles
                    .into_iter()
                    .map(|(tile, curve)| at_4096(profile::curves::KernelCurve::TILE, tile, curve))
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

    /// Under skip with no curve and no roofline stated, each rung's kernel
    /// is read off the work's shape: a vector kernel reads each plane once
    /// for every rung up to the most rows a vector workgroup carries, pads
    /// none, and is taken there (the fewest rows a workgroup among those
    /// that tie); past them no kernel reads and computes no more than every
    /// other, and the rung keeps the profile path's tile. The same on an
    /// eager backend, row count by row count, on WGSL's kernels too.
    #[test]
    fn without_curves_or_roofline_the_work_shape_takes_the_vector_kernel_to_its_rows() {
        let rungs = [1, 2, 4, 8, 16, 32, 64, 128];
        let most = gemm::VECTOR_ROWS[gemm::VECTOR_ROWS.len() - 1];
        let (dense, said) = by_shape(
            Target::Spirv,
            &rungs,
            coopmat,
            None,
            &planes(&counts(), None),
            true,
            Some(128),
        );
        assert_eq!(said.len(), rungs.len());
        for &rows in &rungs {
            let t = dense.at(rows, 4096, 4096);
            if rows <= most {
                assert_eq!(t.vector, gemm::vector_rows(rows), "{rows}: {said:#?}");
            } else {
                assert_eq!(t, coopmat(rows), "{rows}: {said:#?}");
            }
        }
        assert!(said[0].contains("gemv x1 by the work's shape"), "{}", said[0]);
        assert!(said[5].contains("the profile path's"), "{}", said[5]);
        let wgsl = |_| Tactic {
            tier: Tier::Subgroup,
            vector: 0,
            wide: true,
        };
        let (eager, _) = by_shape(
            Target::Wgsl,
            &rungs,
            wgsl,
            None,
            &planes(&counts(), None),
            false,
            Some(128),
        );
        for rows in 1..=128 {
            let t = eager.at(rows, 4096, 4096);
            if rows <= most {
                assert_eq!(t.vector, gemm::vector_rows(rows), "{rows}");
            } else {
                assert_eq!(t.vector, 0, "{rows}");
            }
        }
    }

    /// With the roofline stated and no curve, each rung weighs the work's
    /// shape at the stated bandwidth and matmul rate: the vector kernel
    /// where a step reads its weight (few rows), a tile where the rows make
    /// it compute-bound.
    #[test]
    fn without_curves_the_stated_roofline_weighs_the_work_shape() {
        let rungs = [1, 4, 16, 64, 256, 1024, 2048];
        let mut r = rates(None, Vec::new());
        r.kernel_curves = None;
        let c = Constants::of(&r).unwrap();
        let (dense, said) = by_shape(
            Target::Spirv,
            &rungs,
            coopmat,
            Some(&c),
            &planes(&counts(), None),
            true,
            None,
        );
        assert!(dense.at(1, 4096, 4096).vector > 0, "{said:#?}");
        assert!(dense.at(4, 4096, 4096).vector > 0, "{said:#?}");
        assert_eq!(dense.at(2048, 4096, 4096), coopmat(2048), "{said:#?}");
        assert!(said[0].contains("by the stated roofline"), "{}", said[0]);
    }

    /// Under skip with curves stated but no roofline, the curves weigh the
    /// kernels as a measured load's do; [`choose`] refuses the same profile
    /// by the roofline's key.
    #[test]
    fn stated_curves_weigh_the_kernels_without_a_roofline() {
        let rungs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
        let mut r = rates(Some(5.0), coopmat_tiles());
        r.launch_ns = None;
        r.hbm_gbps = None;
        r.gemm_tflops_by_rows = None;
        assert!(choose(
            Target::Spirv,
            &rungs,
            coopmat,
            &r,
            &planes(&counts(), None),
            true,
            None
        )
        .is_err());
        let (dense, weighed) = choose_unrated(
            Target::Spirv,
            &rungs,
            coopmat,
            &r,
            &planes(&counts(), None),
            true,
            None,
        )
        .unwrap();
        assert!(dense.at(1, 4096, 4096).vector > 0, "{weighed:?}");
        assert_eq!(dense.at(2048, 4096, 4096).vector, 0, "{weighed:?}");
        r.kernel_curves = None;
        assert_eq!(
            choose_unrated(
                Target::Spirv,
                &rungs,
                coopmat,
                &r,
                &planes(&counts(), None),
                true,
                None
            )
            .unwrap_err(),
            UNREACHED
        );
    }

    /// A settle's pick: candidates whose timings overlap inside their
    /// spreads tie (the incumbent stands); separated ones take the faster;
    /// a noise-swamped trial (every spread wide) replaces nothing.
    #[test]
    fn a_settle_picks_only_past_the_spreads() {
        // Overlapping: 100 ±10% against 105 ±10%.
        assert_eq!(settle_pick(&[vec![95.0, 100.0, 105.0], vec![100.0, 105.0, 110.0]]), None);
        // Separated: the second clearly faster.
        assert_eq!(settle_pick(&[vec![100.0, 101.0, 102.0], vec![80.0, 81.0, 82.0]]), Some(1));
        // The incumbent clearly fastest: decided, kept.
        assert_eq!(settle_pick(&[vec![50.0, 51.0, 52.0], vec![80.0, 81.0, 82.0]]), Some(0));
        // Windows' B trial: every candidate ~25% spread, a 20% median gap.
        assert_eq!(
            settle_pick(&[
                vec![32000.0, 36000.0, 40000.0],
                vec![29000.0, 32272.0, 37000.0],
                vec![34000.0, 38873.0, 44000.0],
            ]),
            None,
            "a noise-swamped trial replaces nothing"
        );
        // An untimed candidate takes no part.
        assert_eq!(settle_pick(&[vec![100.0, 101.0], Vec::new(), vec![60.0, 61.0]]), Some(2));
    }

    /// A settle is keyed by its model alone and checked by the picks it was
    /// weighed against: a second model measured on the same device (its
    /// curves joined to the first's, at its planes or the first's) leaves
    /// the first model's key; the picks key its code, the same under any
    /// order of the curves and once they are written to a profile and read
    /// back, anew where a pick moves.
    #[test]
    fn a_models_settle_key_is_its_own_and_its_code_its_picks() {
        let r = rates(Some(5.0), coopmat_tiles());
        let planes = planes(&counts(), None);
        let rungs = [1, 2, 4, 8, 16, 32, 64];
        let pick = |r: &profile::rates::Rates| {
            choose(Target::Spirv, &rungs, coopmat, r, &planes, true, None).unwrap().0
        };
        let a = settle_of(&planes, 4, 64, Target::Spirv);
        let code = |r: &profile::rates::Rates| settle_code(7, &pick(r));
        let first = code(&r);
        // Model B measured after A: its curves join A's (another plane, and
        // A's own plane timed again to more rows at the same rates).
        let mut joined = r.clone();
        let mut elsewhere = joined.kernel_curves.as_ref().unwrap()[0].clone();
        elsewhere.n = 777;
        joined.kernel_curves.as_mut().unwrap().push(elsewhere);
        let longer = joined.kernel_curves.as_mut().unwrap();
        let last = *longer[0].curve.last().unwrap();
        longer[0].curve.push((last.0 * 2, last.1));
        assert_eq!(settle_of(&planes, 4, 64, Target::Spirv), a, "B leaves A's key");
        assert_eq!(code(&joined), first, "B's curves move none of A's picks");
        let mut reversed = r.clone();
        reversed.kernel_curves.as_mut().unwrap().reverse();
        assert_eq!(code(&reversed), first, "the curves' order keys nothing");
        let registry = profile::Registry::new("test", &[], &[]);
        let read = registry
            .load_user(&format!("api = 1\n{}", r.toml()), "round trip")
            .expect("the curves read back");
        assert_eq!(code(&read.rates()), first, "written and read back, the same");
        let mut slower = r.clone();
        for c in slower.kernel_curves.as_mut().unwrap() {
            if c.kind == profile::curves::KernelCurve::VECTOR {
                for p in &mut c.curve {
                    p.1 /= 1000.0;
                }
            }
        }
        assert_ne!(code(&slower), first, "a moved pick checks the settle anew");
        assert_ne!(settle_of(&planes, 8, 64, Target::Spirv), a, "other lanes, another load");
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
            None,
        )
        .unwrap();
        let vector: Vec<bool> = weighed.iter().map(Weighed::vector).collect();
        assert!(vector[..4].iter().all(|&v| v), "{weighed:?}");
        assert!(vector[9..].iter().all(|&v| !v), "{weighed:?}");
        for (i, &rows) in rungs.iter().enumerate() {
            assert_eq!(dense.at(rows, 4096, 4096).vector > 0, vector[i]);
            assert_eq!(
                dense.at(rows.div_ceil(2) + 1, 4096, 4096).vector > 0,
                vector[i],
                "{rows}"
            );
        }
        assert!(dense.at(3, 4096, 4096).vector > 0);
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
            None,
        )
        .unwrap();
        assert!(
            dense.at(200, 4096, 4096).wide && dense.at(200, 4096, 4096).vector == 0,
            "{weighed:?}"
        );
        assert_eq!(weighed[1].costs.len(), gemm::VECTOR_ROWS.len() + 2);
    }

    /// A profile stating no kernel's curve is refused by the key; one stating
    /// none at the model's planes takes the profile's path's own tactic; one
    /// stating the tiled kernel's alone picks it.
    #[test]
    fn a_missing_rate_is_refused_by_its_key() {
        let mut none = rates(Some(5.0), coopmat_tiles());
        none.kernel_curves = None;
        let refused = choose(
            Target::Spirv,
            &[1, 2],
            coopmat,
            &none,
            &planes(&counts(), None),
            true,
            None,
        );
        assert_eq!(refused.unwrap_err(), profile::rates::KERNEL_CURVES);
        let mut elsewhere = rates(Some(5.0), coopmat_tiles());
        for c in elsewhere.kernel_curves.iter_mut().flatten() {
            c.n = 1024;
        }
        let refused = choose(
            Target::Spirv,
            &[1, 2],
            coopmat,
            &elsewhere,
            &planes(&counts(), None),
            true,
            None,
        );
        // No kernel timed at the model's planes: the profile's path's own,
        // said as held, never a refused load.
        let (dense, weighed) = refused.unwrap();
        assert_eq!(dense.at(1, 4096, 4096), coopmat(1));
        assert!(weighed.iter().all(|w| w.held), "{weighed:?}");
        let (dense, _) = choose(
            Target::Spirv,
            &[1, 2],
            coopmat,
            &rates(None, coopmat_tiles()),
            &planes(&counts(), None),
            true,
            None,
        )
        .unwrap();
        assert_eq!(dense.at(1, 4096, 4096).vector, 0);
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
        let curve = |kind, key, n, k, rate: f64| profile::curves::KernelCurve {
            by: profile::rates::MeasuredBy::default(),
            kind,
            key,
            n,
            k,
            curve: vec![(1, rate / 100.0), (512, rate)],
            stop: None,
            spread: Vec::new(),
        };
        let tile =
            |wide, n, k| crate::rates::curve_key(Target::Wgsl, Tactic { wide, ..tiled(0) }, n, k).1;
        let tag = profile::curves::KernelCurve::TILE;
        let mut curves = Vec::new();
        for (n, k) in [(1024, 3072), (3072, 1024)] {
            curves.push(curve(tag, tile(false, n, k), n, k, 9.0));
            curves.push(curve(tag, tile(true, n, k), n, k, 6.0));
            for rows in gemm::VECTOR_ROWS {
                curves.push(curve(profile::curves::KernelCurve::VECTOR, rows, n, k, 0.5));
            }
        }
        // At 4096 square, not a plane of the model: the wide tile far ahead.
        curves.push(curve(tag, tile(true, 4096, 4096), 4096, 4096, 16.0));
        curves.push(curve(tag, tile(false, 4096, 4096), 4096, 4096, 6.4));
        let r = profile::rates::Rates {
            kernel_curves: Some(curves),
            ..rates(None, Vec::new())
        };
        let (dense, weighed) = choose(
            Target::Wgsl,
            &[512],
            tiled,
            &r,
            &planes(&model, None),
            true,
            None,
        )
        .unwrap();
        let at = dense.at(512, 1024, 3072);
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
            None,
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
                None,
            )
            .unwrap();
            let fastest = w[0]
                .costs
                .iter()
                .min_by(|a, b| a.2.total_cmp(&b.2))
                .map(|c| c.0)
                .unwrap();
            assert_eq!(eager.at(rows, 4096, 4096), fastest, "{rows}");
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
                let (_, weighed) = choose(
                    target,
                    &rungs,
                    tiled,
                    &flat,
                    &planes(&counts(), None),
                    true,
                    None,
                )
                .unwrap();
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
            None,
        )
        .unwrap();
        assert_eq!(eager.at(3, 4096, 4096).vector, 1);
    }

    /// The Adreno 840's curves for Qwen3-0.6B (measured native), some
    /// stopped short (beaten, or by the buffer at the head): each rung's
    /// estimate for the kernel it takes only grows with the rows, and is the
    /// model's planes' flops over that kernel's measured rates (within a
    /// factor of two: launches and the spread aside). Weighing only the
    /// planes every kernel reached dropped all but one plane: rung 1024 was
    /// estimated at 256 ms where its flops at the curves' rates take ~2.6 s,
    /// and rung 512 below rung 256.
    #[test]
    fn the_phones_rung_estimates_grow_with_the_rows_and_are_the_curves_rates() {
        let tiled = |_| Tactic {
            tier: Tier::Subgroup,
            vector: 0,
            wide: true,
        };
        type Curve = (
            u32,
            u32,
            u32,
            u32,
            Option<(u32, u32)>,
            Vec<(u32, f64)>,
            Vec<(u32, f64)>,
        );
        let phone: Vec<Curve> = vec![
            (
                1,
                1,
                1024,
                2048,
                Some((4096, 3)),
                vec![
                    (1, 0.004),
                    (4, 0.016),
                    (16, 0.041),
                    (64, 0.09),
                    (256, 0.117),
                    (1024, 0.153),
                ],
                vec![
                    (1, 0.44),
                    (4, 0.52),
                    (16, 0.04),
                    (64, 0.14),
                    (256, 0.07),
                    (1024, 0.01),
                ],
            ),
            (
                1,
                4,
                1024,
                2048,
                None,
                vec![
                    (1, 0.006),
                    (4, 0.023),
                    (16, 0.085),
                    (64, 0.151),
                    (256, 0.368),
                    (1024, 0.433),
                    (4096, 0.565),
                    (10240, 0.574),
                ],
                vec![
                    (1, 0.1),
                    (4, 0.18),
                    (16, 0.27),
                    (64, 0.03),
                    (256, 0.15),
                    (1024, 0.04),
                    (4096, 0.0),
                    (10240, 0.11),
                ],
            ),
            (
                1,
                8,
                1024,
                2048,
                None,
                vec![
                    (1, 0.006),
                    (4, 0.026),
                    (16, 0.094),
                    (64, 0.183),
                    (256, 0.375),
                    (1024, 0.47),
                    (4096, 0.627),
                    (10240, 0.611),
                ],
                vec![
                    (1, 0.16),
                    (4, 0.09),
                    (16, 0.16),
                    (64, 0.45),
                    (256, 0.11),
                    (1024, 0.03),
                    (4096, 0.03),
                    (10240, 0.01),
                ],
            ),
            (
                1,
                16,
                1024,
                2048,
                None,
                vec![
                    (1, 0.006),
                    (4, 0.026),
                    (16, 0.101),
                    (64, 0.183),
                    (256, 0.21),
                    (1024, 0.378),
                    (4096, 0.427),
                    (10240, 0.451),
                ],
                vec![
                    (1, 0.09),
                    (4, 0.06),
                    (16, 0.22),
                    (64, 0.5),
                    (256, 0.11),
                    (1024, 0.05),
                    (4096, 0.01),
                    (10240, 0.01),
                ],
            ),
            (
                2,
                32,
                1024,
                2048,
                None,
                vec![
                    (1, 0.005),
                    (4, 0.019),
                    (16, 0.068),
                    (64, 0.148),
                    (256, 0.189),
                    (1024, 0.326),
                    (4096, 0.376),
                    (10240, 0.391),
                ],
                vec![
                    (1, 0.02),
                    (4, 0.01),
                    (16, 0.01),
                    (64, 0.23),
                    (256, 0.01),
                    (1024, 0.02),
                    (4096, 0.01),
                    (10240, 0.0),
                ],
            ),
            (
                2,
                256,
                1024,
                2048,
                None,
                vec![
                    (1, 0.002),
                    (4, 0.007),
                    (16, 0.021),
                    (64, 0.1),
                    (256, 0.221),
                    (1024, 0.459),
                    (4096, 0.671),
                    (10240, 0.762),
                ],
                vec![
                    (1, 0.01),
                    (4, 0.11),
                    (16, 0.15),
                    (64, 0.21),
                    (256, 0.02),
                    (1024, 0.13),
                    (4096, 0.03),
                    (10240, 0.02),
                ],
            ),
            (
                1,
                1,
                1024,
                3072,
                Some((256, 3)),
                vec![(1, 0.006), (4, 0.017), (16, 0.048), (64, 0.071)],
                vec![(1, 0.58), (4, 0.27), (16, 0.29), (64, 0.18)],
            ),
            (
                1,
                4,
                1024,
                3072,
                None,
                vec![
                    (1, 0.008),
                    (4, 0.027),
                    (16, 0.073),
                    (64, 0.173),
                    (256, 0.283),
                    (1024, 0.472),
                    (4096, 0.581),
                    (10240, 0.55),
                ],
                vec![
                    (1, 0.08),
                    (4, 0.08),
                    (16, 0.16),
                    (64, 0.23),
                    (256, 0.13),
                    (1024, 0.0),
                    (4096, 0.01),
                    (10240, 0.01),
                ],
            ),
            (
                1,
                8,
                1024,
                3072,
                None,
                vec![
                    (1, 0.008),
                    (4, 0.033),
                    (16, 0.115),
                    (64, 0.248),
                    (256, 0.303),
                    (1024, 0.554),
                    (4096, 0.635),
                    (10240, 0.638),
                ],
                vec![
                    (1, 0.09),
                    (4, 0.16),
                    (16, 0.27),
                    (64, 0.55),
                    (256, 0.18),
                    (1024, 0.05),
                    (4096, 0.03),
                    (10240, 0.02),
                ],
            ),
            (
                1,
                16,
                1024,
                3072,
                None,
                vec![
                    (1, 0.009),
                    (4, 0.032),
                    (16, 0.12),
                    (64, 0.264),
                    (256, 0.258),
                    (1024, 0.438),
                    (4096, 0.494),
                    (10240, 0.481),
                ],
                vec![
                    (1, 0.13),
                    (4, 0.11),
                    (16, 0.42),
                    (64, 0.52),
                    (256, 0.17),
                    (1024, 0.07),
                    (4096, 0.01),
                    (10240, 0.02),
                ],
            ),
            (
                2,
                32,
                1024,
                3072,
                None,
                vec![
                    (1, 0.005),
                    (4, 0.02),
                    (16, 0.073),
                    (64, 0.174),
                    (256, 0.217),
                    (1024, 0.329),
                    (4096, 0.385),
                    (10240, 0.385),
                ],
                vec![
                    (1, 0.05),
                    (4, 0.03),
                    (16, 0.06),
                    (64, 0.29),
                    (256, 0.11),
                    (1024, 0.04),
                    (4096, 0.03),
                    (10240, 0.01),
                ],
            ),
            (
                2,
                256,
                1024,
                3072,
                None,
                vec![
                    (1, 0.001),
                    (4, 0.006),
                    (16, 0.02),
                    (64, 0.079),
                    (256, 0.269),
                    (1024, 0.526),
                    (4096, 0.711),
                    (10240, 0.799),
                ],
                vec![
                    (1, 0.18),
                    (4, 0.24),
                    (16, 0.25),
                    (64, 0.29),
                    (256, 0.1),
                    (1024, 0.06),
                    (4096, 0.1),
                    (10240, 0.11),
                ],
            ),
            (
                1,
                1,
                4096,
                1024,
                Some((256, 3)),
                vec![(1, 0.007), (4, 0.027), (16, 0.066), (64, 0.087)],
                vec![(1, 0.45), (4, 0.53), (16, 0.09), (64, 0.12)],
            ),
            (
                1,
                4,
                4096,
                1024,
                None,
                vec![
                    (1, 0.012),
                    (4, 0.037),
                    (16, 0.106),
                    (64, 0.241),
                    (256, 0.288),
                    (1024, 0.405),
                    (4096, 0.414),
                    (10240, 0.424),
                ],
                vec![
                    (1, 0.13),
                    (4, 0.12),
                    (16, 0.47),
                    (64, 0.09),
                    (256, 0.07),
                    (1024, 0.05),
                    (4096, 0.05),
                    (10240, 0.01),
                ],
            ),
            (
                1,
                8,
                4096,
                1024,
                None,
                vec![
                    (1, 0.011),
                    (4, 0.04),
                    (16, 0.12),
                    (64, 0.247),
                    (256, 0.302),
                    (1024, 0.434),
                    (4096, 0.449),
                    (10240, 0.449),
                ],
                vec![
                    (1, 0.11),
                    (4, 0.15),
                    (16, 0.63),
                    (64, 0.13),
                    (256, 0.1),
                    (1024, 0.01),
                    (4096, 0.01),
                    (10240, 0.01),
                ],
            ),
            (
                1,
                16,
                4096,
                1024,
                Some((4096, 3)),
                vec![
                    (1, 0.011),
                    (4, 0.045),
                    (16, 0.121),
                    (64, 0.173),
                    (256, 0.251),
                    (1024, 0.31),
                ],
                vec![
                    (1, 0.16),
                    (4, 0.23),
                    (16, 0.27),
                    (64, 0.09),
                    (256, 0.06),
                    (1024, 0.03),
                ],
            ),
            (
                2,
                32,
                4096,
                1024,
                None,
                vec![
                    (1, 0.007),
                    (4, 0.027),
                    (16, 0.103),
                    (64, 0.182),
                    (256, 0.279),
                    (1024, 0.36),
                    (4096, 0.39),
                    (10240, 0.392),
                ],
                vec![
                    (1, 0.02),
                    (4, 0.06),
                    (16, 0.09),
                    (64, 0.18),
                    (256, 0.1),
                    (1024, 0.05),
                    (4096, 0.01),
                    (10240, 0.01),
                ],
            ),
            (
                2,
                256,
                4096,
                1024,
                None,
                vec![
                    (1, 0.002),
                    (4, 0.007),
                    (16, 0.025),
                    (64, 0.101),
                    (256, 0.393),
                    (1024, 0.638),
                    (4096, 0.776),
                    (10240, 0.774),
                ],
                vec![
                    (1, 0.34),
                    (4, 0.2),
                    (16, 0.03),
                    (64, 0.17),
                    (256, 0.14),
                    (1024, 0.06),
                    (4096, 0.02),
                    (10240, 0.03),
                ],
            ),
            (
                1,
                1,
                6144,
                1024,
                Some((1024, 3)),
                vec![
                    (1, 0.012),
                    (4, 0.038),
                    (16, 0.066),
                    (64, 0.092),
                    (256, 0.13),
                ],
                vec![(1, 0.84), (4, 0.48), (16, 0.15), (64, 0.13), (256, 0.01)],
            ),
            (
                1,
                4,
                6144,
                1024,
                None,
                vec![
                    (1, 0.015),
                    (4, 0.047),
                    (16, 0.144),
                    (64, 0.237),
                    (256, 0.347),
                    (1024, 0.417),
                    (4096, 0.401),
                    (10240, 0.415),
                ],
                vec![
                    (1, 0.16),
                    (4, 0.53),
                    (16, 0.55),
                    (64, 0.1),
                    (256, 0.13),
                    (1024, 0.02),
                    (4096, 0.01),
                    (10240, 0.01),
                ],
            ),
            (
                1,
                8,
                6144,
                1024,
                None,
                vec![
                    (1, 0.016),
                    (4, 0.055),
                    (16, 0.186),
                    (64, 0.249),
                    (256, 0.37),
                    (1024, 0.438),
                    (4096, 0.442),
                    (10240, 0.44),
                ],
                vec![
                    (1, 0.13),
                    (4, 0.04),
                    (16, 0.36),
                    (64, 0.16),
                    (256, 0.07),
                    (1024, 0.1),
                    (4096, 0.02),
                    (10240, 0.02),
                ],
            ),
            (
                1,
                16,
                6144,
                1024,
                Some((4096, 3)),
                vec![
                    (1, 0.014),
                    (4, 0.056),
                    (16, 0.138),
                    (64, 0.144),
                    (256, 0.26),
                    (1024, 0.314),
                ],
                vec![
                    (1, 0.18),
                    (4, 0.14),
                    (16, 0.47),
                    (64, 0.29),
                    (256, 0.06),
                    (1024, 0.02),
                ],
            ),
            (
                2,
                32,
                6144,
                1024,
                None,
                vec![
                    (1, 0.007),
                    (4, 0.027),
                    (16, 0.088),
                    (64, 0.154),
                    (256, 0.281),
                    (1024, 0.365),
                    (4096, 0.382),
                    (10240, 0.391),
                ],
                vec![
                    (1, 0.07),
                    (4, 0.05),
                    (16, 0.21),
                    (64, 0.16),
                    (256, 0.01),
                    (1024, 0.0),
                    (4096, 0.03),
                    (10240, 0.01),
                ],
            ),
            (
                2,
                256,
                6144,
                1024,
                None,
                vec![
                    (1, 0.002),
                    (4, 0.009),
                    (16, 0.031),
                    (64, 0.124),
                    (256, 0.466),
                    (1024, 0.716),
                    (4096, 0.745),
                    (10240, 0.795),
                ],
                vec![
                    (1, 0.14),
                    (4, 0.17),
                    (16, 0.1),
                    (64, 0.18),
                    (256, 0.1),
                    (1024, 0.03),
                    (4096, 0.05),
                    (10240, 0.09),
                ],
            ),
            (
                1,
                1,
                151936,
                1024,
                Some((256, 3)),
                vec![(1, 0.029), (4, 0.062), (16, 0.116), (64, 0.138)],
                vec![(1, 0.06), (4, 0.26), (16, 0.11), (64, 0.03)],
            ),
            (
                1,
                4,
                151936,
                1024,
                Some((4096, 2)),
                vec![
                    (1, 0.029),
                    (4, 0.102),
                    (16, 0.282),
                    (64, 0.385),
                    (256, 0.412),
                    (1024, 0.42),
                ],
                vec![
                    (1, 0.11),
                    (4, 0.15),
                    (16, 0.11),
                    (64, 0.04),
                    (256, 0.03),
                    (1024, 0.07),
                ],
            ),
            (
                1,
                8,
                151936,
                1024,
                Some((4096, 2)),
                vec![
                    (1, 0.029),
                    (4, 0.104),
                    (16, 0.289),
                    (64, 0.407),
                    (256, 0.439),
                    (1024, 0.42),
                ],
                vec![
                    (1, 0.08),
                    (4, 0.13),
                    (16, 0.14),
                    (64, 0.01),
                    (256, 0.01),
                    (1024, 0.08),
                ],
            ),
            (
                1,
                16,
                151936,
                1024,
                Some((1024, 3)),
                vec![
                    (1, 0.028),
                    (4, 0.09),
                    (16, 0.204),
                    (64, 0.293),
                    (256, 0.298),
                ],
                vec![(1, 0.04), (4, 0.14), (16, 0.12), (64, 0.03), (256, 0.01)],
            ),
            (
                2,
                32,
                151936,
                1024,
                Some((4096, 2)),
                vec![
                    (1, 0.01),
                    (4, 0.035),
                    (16, 0.136),
                    (64, 0.329),
                    (256, 0.352),
                    (1024, 0.334),
                ],
                vec![
                    (1, 0.04),
                    (4, 0.21),
                    (16, 0.15),
                    (64, 0.01),
                    (256, 0.02),
                    (1024, 0.08),
                ],
            ),
            (
                2,
                256,
                151936,
                1024,
                Some((4096, 2)),
                vec![
                    (1, 0.003),
                    (4, 0.011),
                    (16, 0.047),
                    (64, 0.207),
                    (256, 0.698),
                    (1024, 0.69),
                ],
                vec![
                    (1, 0.02),
                    (4, 0.08),
                    (16, 0.12),
                    (64, 0.03),
                    (256, 0.02),
                    (1024, 0.07),
                ],
            ),
        ];
        let mut curves: Vec<profile::curves::KernelCurve> = phone
            .into_iter()
            .map(
                |(kind, key, n, k, stop, curve, spread)| profile::curves::KernelCurve {
                    by: profile::rates::MeasuredBy::default(),
                    kind,
                    key,
                    n,
                    k,
                    curve,
                    stop,
                    spread,
                },
            )
            .collect();
        curves.sort_by_key(profile::curves::KernelCurve::id);
        let r = profile::rates::Rates {
            kernel_curves: Some(curves.clone()),
            ..rates(None, Vec::new())
        };
        let shape = |n: u32, k: u32, planes| engine::load::GemmShape {
            planes,
            ..engine::load::GemmShape::dense(
                n,
                k,
                2 * u64::from(n) * u64::from(k),
                2 * u64::from(n) * u64::from(k),
            )
        };
        let model = Counts {
            gemm_shapes: vec![
                shape(1024, 2048, 28),
                shape(1024, 3072, 28),
                shape(4096, 1024, 28),
                shape(6144, 1024, 28),
                shape(151_936, 1024, 1),
            ],
            ..counts()
        };
        let planes = planes(&model, None);
        let rungs: Vec<u32> = (0..=10).map(|p| 1 << p).collect();
        let (_, weighed) =
            choose(Target::Spirv, &rungs, tiled, &r, &planes, true, Some(1024)).unwrap();
        // The model's flops at `rows` over `tactic`'s rate at each plane.
        let at_rates = |tactic: Tactic, rows: u32| -> f64 {
            planes
                .iter()
                .map(|&(n, k, count, _)| {
                    let (kind, key) = crate::rates::curve_key(Target::Spirv, tactic, n, k);
                    let c = curves.iter().find(|c| c.id() == (kind, key, n, k)).unwrap();
                    let last = c.curve.last().unwrap();
                    let rate = if rows > last.0 {
                        last.1
                    } else {
                        profile::curves::interpolate(&c.curve, f64::from(rows)).unwrap()
                    };
                    f64::from(count) * 2.0 * f64::from(rows) * f64::from(n) * f64::from(k)
                        / (rate * 1e3)
                })
                .sum()
        };
        // Each kernel's estimate at a rung's top only grows with the rungs.
        for pair in weighed.windows(2) {
            for &(tactic, _, top) in &pair[1].costs {
                if let Some(&(_, _, below)) = pair[0].costs.iter().find(|c| c.0 == tactic) {
                    assert!(
                        top >= below,
                        "{tactic:?}: {below} ns at {} rows, {top} ns at {}",
                        pair[0].top,
                        pair[1].top
                    );
                }
            }
        }
        for w in &weighed {
            let (tactic, _, top) = w.costs[w.taken];
            let flops = at_rates(tactic, w.top);
            assert!(
                top >= flops / 2.0 && top <= flops * 2.0,
                "rung {}: estimated {:.1} ms, its flops at the curves' rates {:.1} ms",
                w.top,
                top / 1e6,
                flops / 1e6
            );
        }
    }

    /// A curve stopped short of a rung's rows (the phone's head, its vector
    /// curve beaten at 512 rows) keeps weighing its plane: no candidate's
    /// estimate falls as the rungs grow, and the kernel measured at every
    /// plane is taken over one whose cost there would be carried past its
    /// curve. Dropping the head from the sum instead had a rung of 512 rows
    /// estimated below one of 256 and the vector kernel taken at 512 and
    /// 1024 by the layers alone.
    #[test]
    fn a_curve_stopped_short_neither_lowers_an_estimate_nor_wins_past_it() {
        let tiled = |_| Tactic {
            tier: Tier::Subgroup,
            vector: 0,
            wide: false,
        };
        let shape = |n: u32, k: u32, planes| engine::load::GemmShape {
            planes,
            ..engine::load::GemmShape::dense(
                n,
                k,
                2 * u64::from(n) * u64::from(k),
                2 * u64::from(n) * u64::from(k),
            )
        };
        let (head_n, head_k) = (151_936, 1024);
        let model = Counts {
            gemm_shapes: vec![shape(1024, 1024, 28), shape(head_n, head_k, 1)],
            ..counts()
        };
        let curve = |kind, key, n, k, points: Vec<(u32, f64)>, stop| profile::curves::KernelCurve {
            by: profile::rates::MeasuredBy::default(),
            kind,
            key,
            n,
            k,
            curve: points,
            stop,
            spread: Vec::new(),
        };
        let tile = |n, k| crate::rates::curve_key(Target::Wgsl, tiled(0), n, k).1;
        let (v, t) = (
            profile::curves::KernelCurve::VECTOR,
            profile::curves::KernelCurve::TILE,
        );
        let mut curves = vec![
            // The layers: the vector kernel ahead of the tile to 1024 rows.
            curve(
                t,
                tile(1024, 1024),
                1024,
                1024,
                vec![(1, 0.01), (1024, 1.0)],
                None,
            ),
            curve(v, 8, 1024, 1024, vec![(1, 0.05), (1024, 2.0)], None),
            // The head: the tile timed to 1024; the vector kernel beaten at
            // 512, its last point at 256.
            curve(
                t,
                tile(head_n, head_k),
                head_n,
                head_k,
                vec![(1, 0.02), (1024, 8.0)],
                None,
            ),
            curve(
                v,
                8,
                head_n,
                head_k,
                vec![(1, 1.0), (256, 1.5)],
                Some((512, profile::curves::KernelCurve::BEATEN)),
            ),
        ];
        curves.sort_by_key(profile::curves::KernelCurve::id);
        let r = profile::rates::Rates {
            kernel_curves: Some(curves),
            ..rates(None, Vec::new())
        };
        let rungs = [256, 512, 1024];
        let (dense, weighed) = choose(
            Target::Wgsl,
            &rungs,
            tiled,
            &r,
            &planes(&model, None),
            true,
            None,
        )
        .unwrap();
        for rows in [512, 1024] {
            assert_eq!(
                dense.at(rows, head_n, head_k).vector,
                0,
                "{rows}: {:?}",
                dense.runs()
            );
            assert!(!weighed[1].costs.is_empty());
        }
        // Each kernel's estimate at a rung's top only grows with the rungs.
        for pair in weighed.windows(2) {
            for &(tactic, _, top) in &pair[1].costs {
                if let Some(&(_, _, below)) = pair[0].costs.iter().find(|c| c.0 == tactic) {
                    assert!(
                        top >= below,
                        "{tactic:?}: {below} at {} then {top} at {}",
                        pair[0].top,
                        pair[1].top
                    );
                }
            }
        }
    }

    /// Each plane takes its own kernel at a rung: the vector kernel the
    /// layers' planes it wins, the tile the head where the vector kernel was
    /// beaten, or stopped short by the bound (unmeasured there).
    #[test]
    fn each_plane_takes_its_own_kernel_at_one_rung() {
        let tiled = |_| Tactic {
            tier: Tier::Subgroup,
            vector: 0,
            wide: false,
        };
        let shape = |n, k, planes| engine::load::GemmShape {
            planes,
            ..engine::load::GemmShape::dense(n, k, 2 * u64::from(n * k), 2 * u64::from(n * k))
        };
        // Many layer planes and one wide head.
        let model = Counts {
            gemm_shapes: vec![shape(1024, 1024, 28), shape(8192, 1024, 1)],
            ..counts()
        };
        let curve = |kind, key, n, k, points: Vec<(u32, f64)>, stop| profile::curves::KernelCurve {
            by: profile::rates::MeasuredBy::default(),
            kind,
            key,
            n,
            k,
            curve: points,
            stop,
            spread: Vec::new(),
        };
        let tile = |n, k| crate::rates::curve_key(Target::Wgsl, tiled(0), n, k).1;
        let (v, t) = (
            profile::curves::KernelCurve::VECTOR,
            profile::curves::KernelCurve::TILE,
        );
        let mut curves = vec![
            // The layer planes: the vector kernel 4x the tile to 256 rows.
            curve(
                t,
                tile(1024, 1024),
                1024,
                1024,
                vec![(1, 0.01), (256, 1.0)],
                None,
            ),
            curve(v, 16, 1024, 1024, vec![(1, 0.04), (256, 4.0)], None),
            // The head: the vector kernel beaten at 16 rows.
            curve(
                t,
                tile(8192, 1024),
                8192,
                1024,
                vec![(1, 0.05), (256, 8.0)],
                None,
            ),
            curve(
                v,
                16,
                8192,
                1024,
                vec![(1, 0.02), (16, 2.0)],
                Some((64, profile::curves::KernelCurve::BEATEN)),
            ),
        ];
        let r = profile::rates::Rates {
            kernel_curves: Some(curves.clone()),
            ..rates(None, Vec::new())
        };
        let (dense, weighed) = choose(
            Target::Wgsl,
            &[128],
            tiled,
            &r,
            &planes(&model, None),
            true,
            None,
        )
        .unwrap();
        // At one rung the layers' planes take the vector kernel and the head,
        // where it was beaten, the tile.
        assert_eq!(dense.at(128, 1024, 1024).vector, 16, "{:?}", dense.runs());
        assert_eq!(dense.at(128, 8192, 1024).vector, 0, "{:?}", dense.runs());
        let _ = weighed;
        // Stopped by the bound at the head instead: not shown beaten.
        curves[3].stop = Some((64, profile::curves::KernelCurve::BOUND));
        let r = profile::rates::Rates {
            kernel_curves: Some(curves),
            ..rates(None, Vec::new())
        };
        let (dense, weighed) = choose(
            Target::Wgsl,
            &[128],
            tiled,
            &r,
            &planes(&model, None),
            true,
            None,
        )
        .unwrap();
        // Stopped by the bound at 64 rows, x16 is not measured at the head at
        // 128: the measured tile takes it (a cost carried past a curve never
        // wins against a measured one); the layers keep x16.
        assert_eq!(dense.at(128, 1024, 1024).vector, 16, "{}", weighed[0]);
        assert_eq!(
            dense.at(128, 8192, 1024).vector,
            0,
            "{} {:?}",
            weighed[0],
            dense.runs()
        );
    }

    /// The rung full prompt chunks land on is picked at its top rows: on the
    /// L40S's native curves at the o_proj plane, gemv x16 is least regret
    /// over (512, 1024] (it wins at 513 rows) but coopmat wins at 1024, where
    /// every full chunk of 1024 rows lands.
    #[test]
    fn the_full_chunk_rung_is_picked_at_its_top_rows() {
        let (n, k) = (1024, 2048);
        let tile = crate::rates::curve_key(Target::Spirv, coopmat(0), n, k).1;
        let curve = |kind, key, points: Vec<(u32, f64)>| profile::curves::KernelCurve {
            by: profile::rates::MeasuredBy::default(),
            kind,
            key,
            n,
            k,
            curve: points,
            stop: None,
            spread: Vec::new(),
        };
        let r = profile::rates::Rates {
            kernel_curves: Some(vec![
                curve(
                    profile::curves::KernelCurve::TILE,
                    tile,
                    vec![(1, 0.02), (256, 3.8), (1024, 15.4)],
                ),
                curve(
                    profile::curves::KernelCurve::VECTOR,
                    16,
                    vec![(1, 0.6), (256, 11.2), (1024, 12.6)],
                ),
            ]),
            ..rates(None, Vec::new())
        };
        let shape = engine::load::GemmShape {
            planes: 28,
            ..engine::load::GemmShape::dense(n, k, 2 * u64::from(n * k), 2 * u64::from(n * k))
        };
        let model = Counts {
            gemm_shapes: vec![shape],
            ..counts()
        };
        let rungs = [512, 1024];
        let spread = choose(
            Target::Spirv,
            &rungs,
            coopmat,
            &r,
            &planes(&model, None),
            true,
            None,
        )
        .unwrap()
        .0;
        assert_eq!(spread.at(1024, n, k).vector, 16, "{:?}", spread.runs());
        let full = choose(
            Target::Spirv,
            &rungs,
            coopmat,
            &r,
            &planes(&model, None),
            true,
            Some(1024),
        )
        .unwrap()
        .0;
        assert_eq!(full.at(1024, n, k).vector, 0, "{:?}", full.runs());
        // Only that rung: the one below still weighs the rows it serves.
        assert_eq!(full.at(512, n, k), spread.at(512, n, k));
    }

    /// A plane's own pick replaces the pick over every plane together only
    /// past their measured spread: at a plane where the wide tile runs 5%
    /// faster than the narrow one the model's other plane takes, it is taken
    /// at a 1% spread and not at 10%.
    #[test]
    fn a_planes_pick_replaces_the_models_only_past_their_spread() {
        let narrow = |_| Tactic {
            tier: Tier::Subgroup,
            vector: 0,
            wide: false,
        };
        let key = |wide, n, k| {
            crate::rates::curve_key(Target::Wgsl, Tactic { wide, ..narrow(0) }, n, k).1
        };
        let with = |spread: f64| {
            let curve = |wide, n, k, rate: f64| profile::curves::KernelCurve {
                by: profile::rates::MeasuredBy::default(),
                kind: profile::curves::KernelCurve::TILE,
                key: key(wide, n, k),
                n,
                k,
                curve: vec![(1, rate / 100.0), (256, rate)],
                stop: None,
                spread: vec![(1, spread), (256, spread)],
            };
            profile::rates::Rates {
                kernel_curves: Some(vec![
                    // The plane the model's pick follows: narrow far ahead.
                    curve(false, 4096, 4096, 10.0),
                    curve(true, 4096, 4096, 5.0),
                    // A small plane: wide 5% ahead.
                    curve(false, 1024, 1024, 10.0),
                    curve(true, 1024, 1024, 10.5),
                ]),
                ..rates(None, Vec::new())
            }
        };
        let shape = |n: u32, k: u32| engine::load::GemmShape {
            planes: 28,
            ..engine::load::GemmShape::dense(n, k, 2 * u64::from(n * k), 2 * u64::from(n * k))
        };
        let model = Counts {
            gemm_shapes: vec![shape(4096, 4096), shape(1024, 1024)],
            ..counts()
        };
        let planes = planes(&model, None);
        for records in [true, false] {
            let wide = |spread| {
                choose(
                    Target::Wgsl,
                    &[256],
                    narrow,
                    &with(spread),
                    &planes,
                    records,
                    None,
                )
                .unwrap()
                .0
                .at(256, 1024, 1024)
                .wide
            };
            assert!(wide(0.01), "records {records}");
            assert!(!wide(0.10), "records {records}");
        }
    }

    /// Eager (a pick each row count), two vector kernels a few percent
    /// apart, trading the lead row to row inside their 5% spread, take one
    /// kernel over every row: noise does not flip the kernel.
    #[test]
    fn an_eager_pick_does_not_alternate_inside_the_spread() {
        let tiled = |_| Tactic {
            tier: Tier::Subgroup,
            vector: 0,
            wide: false,
        };
        let rows: Vec<u32> = (0..7).map(|i| 1 << i).collect();
        let curve = |key: u32, lead: bool| profile::curves::KernelCurve {
            by: profile::rates::MeasuredBy::default(),
            kind: profile::curves::KernelCurve::VECTOR,
            key,
            n: 4096,
            k: 4096,
            curve: rows
                .iter()
                .enumerate()
                .map(|(i, &r)| {
                    let up = (i % 2 == 0) == lead;
                    (r, f64::from(r) * if up { 0.102 } else { 0.1 })
                })
                .collect(),
            stop: None,
            spread: rows.iter().map(|&r| (r, 0.05)).collect(),
        };
        let mut curves = vec![curve(4, true), curve(8, false)];
        curves.sort_by_key(profile::curves::KernelCurve::id);
        let r = profile::rates::Rates {
            kernel_curves: Some(curves),
            ..rates(None, Vec::new())
        };
        let (dense, _) = choose(
            Target::Wgsl,
            &[64],
            tiled,
            &r,
            &planes(&counts(), None),
            false,
            None,
        )
        .unwrap();
        let first = dense.at(1, 4096, 4096);
        assert!(first.vector > 0, "{:?}", dense.runs());
        for rows in 2..=64 {
            assert_eq!(dense.at(rows, 4096, 4096), first, "{:?}", dense.runs());
        }
    }

    /// A head-shaped plane whose only timed kernel is the tile, stopped at 4
    /// rows by the bound, keeps the pick over every plane together (a vector
    /// kernel at decode rows) at 1 row and past its curves: a kernel no
    /// curve there prices is not shown beaten, and a tile at one row of a
    /// 152k-wide head is a hundred milliseconds a step.
    #[test]
    fn a_head_stopped_at_four_rows_keeps_the_models_kernel() {
        let wide = |_| Tactic {
            tier: Tier::Subgroup,
            vector: 0,
            wide: true,
        };
        let curve = |kind, key, n, k, points: Vec<(u32, f64)>, stop| profile::curves::KernelCurve {
            by: profile::rates::MeasuredBy::default(),
            kind,
            key,
            n,
            k,
            curve: points,
            stop,
            spread: Vec::new(),
        };
        let (v, t) = (
            profile::curves::KernelCurve::VECTOR,
            profile::curves::KernelCurve::TILE,
        );
        let tile = |n, k| crate::rates::curve_key(Target::Wgsl, wide(0), n, k).1;
        let (head_n, head_k) = (151_936, 1024);
        let bound = Some((4, profile::curves::KernelCurve::BOUND));
        let mut curves = vec![
            // The layers: the vector kernel well ahead of the tile to 64 rows.
            curve(
                t,
                tile(1024, 1024),
                1024,
                1024,
                vec![(1, 0.01), (64, 0.6)],
                None,
            ),
            curve(v, 8, 1024, 1024, vec![(1, 0.5), (64, 6.0)], None),
            // The head: the tile alone timed, stopped at 4 rows (the vector
            // kernels' curves there stopped at their first rows).
            curve(
                t,
                tile(head_n, head_k),
                head_n,
                head_k,
                vec![(1, 0.01), (2, 0.02)],
                bound,
            ),
        ];
        curves.sort_by_key(profile::curves::KernelCurve::id);
        let r = profile::rates::Rates {
            kernel_curves: Some(curves),
            ..rates(None, Vec::new())
        };
        let shape = |n: u32, k: u32, planes| engine::load::GemmShape {
            planes,
            ..engine::load::GemmShape::dense(
                n,
                k,
                2 * u64::from(n) * u64::from(k),
                2 * u64::from(n) * u64::from(k),
            )
        };
        let model = Counts {
            gemm_shapes: vec![shape(1024, 1024, 28), shape(head_n, head_k, 1)],
            ..counts()
        };
        let rungs = [1, 2, 4, 8, 16, 32, 64];
        for records in [true, false] {
            let (dense, _) = choose(
                Target::Wgsl,
                &rungs,
                wide,
                &r,
                &planes(&model, None),
                records,
                None,
            )
            .unwrap();
            for rows in [1, 4, 16, 64] {
                assert!(
                    dense.at(rows, head_n, head_k).vector > 0,
                    "records {records}, {rows} rows: {:?}",
                    dense.runs()
                );
            }
        }
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
            None,
        )
        .unwrap();
        let (_, four) = choose(
            Target::Spirv,
            &[4],
            baseline,
            &r,
            &planes(&counts(), None),
            true,
            None,
        )
        .unwrap();
        let tiled = |w: &[Weighed]| w[0].costs.iter().find(|c| c.0.vector == 0).unwrap().2;
        assert_eq!(tiled(&one), tiled(&four));
    }

    /// A tiled kernel whose curve stops short of a rung is no candidate
    /// there (no pick from a rate it was never timed at); a rung past every
    /// curve keeps the rung below's kernel, and a first rung past them all
    /// the profile's path's.
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
            None,
        )
        .unwrap();
        assert!(!dense.at(256, 4096, 4096).wide, "{weighed:?}");
        for rows in [512, 1024] {
            assert!(dense.at(rows, 4096, 4096).wide, "{rows}: {weighed:?}");
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
            None,
        )
        .unwrap();
        assert_eq!(dense.at(8192, 4096, 4096), dense.at(4096, 4096, 4096));
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
            None,
        );
        // Alone, no rung below: the profile's path's own tactic.
        let (dense, weighed) = refused.unwrap();
        assert_eq!(dense.at(8192, 4096, 4096), baseline(8192));
        assert!(weighed[0].held);
    }
}
