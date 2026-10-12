//! From measurements to a ladder: what `pie tune` writes for a decision.
//!
//! Times are weighed relative to the fastest candidate at each point, so a
//! decode row and a long prefill count by their weight in the workload mix,
//! not by how long they take.

use std::fmt::Write as _;

use crate::spec::{Spec, Var};

/// One thing a rung can pick, as the TOML body of the rung: `use =
/// "tiled", crossover_tg = 96`, or `value = 32`.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub body: String,
    /// Whether it may end a ladder: a partial alternative may not.
    pub terminal: bool,
}

/// One grid point: its coordinates in the decision's variable order, its
/// weight, and each candidate's time there. `None` where the candidate does
/// not run at this point: it does not fit, it fell through to another
/// kernel, or it answered wrongly.
#[derive(Clone, Debug)]
pub struct Sample {
    pub at: Vec<i64>,
    pub weight: f64,
    pub times: Vec<Option<f64>>,
}

impl Sample {
    fn best(&self) -> Option<f64> {
        self.times
            .iter()
            .flatten()
            .copied()
            .fold(None, |b, t| Some(b.map_or(t, |b: f64| b.min(t))))
    }
}

/// One condition on one variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Atom {
    AtLeast(usize, i64),
    Below(usize, i64),
    Is(usize, i64),
}

impl Atom {
    fn holds(self, at: &[i64]) -> bool {
        match self {
            Atom::AtLeast(v, t) => at[v] >= t,
            Atom::Below(v, t) => at[v] < t,
            Atom::Is(v, t) => at[v] == t,
        }
    }

    fn render(self, vars: &[Var]) -> String {
        match self {
            Atom::AtLeast(v, t) => format!("{} >= {t}", vars[v].name),
            Atom::Below(v, t) => format!("{} < {t}", vars[v].name),
            Atom::Is(v, t) => format!("{} == {}", vars[v].name, vars[v].domain.render(t)),
        }
    }
}

/// A fitted rung: where it holds (every atom), and what it picks.
#[derive(Clone, Debug, PartialEq)]
pub struct FitRung {
    pub when: Vec<Atom>,
    pub candidate: usize,
}

/// A fitted ladder and what it costs over the grid.
#[derive(Clone, Debug)]
pub struct Fit {
    pub rungs: Vec<FitRung>,
    /// Weighted relative time of the fitted ladder: `Σ w · t / best`.
    pub cost: f64,
    /// The same for a perfect per-point choice: `Σ w`.
    pub floor: f64,
    /// The same for the baseline the caller passed, if any.
    pub baseline: Option<f64>,
}

impl Fit {
    /// The ladder as the TOML value of the decision's key.
    #[must_use]
    pub fn toml(&self, spec: &Spec, candidates: &[Candidate]) -> String {
        let mut out = String::from("[\n");
        for rung in &self.rungs {
            let body = &candidates[rung.candidate].body;
            if rung.when.is_empty() {
                let _ = writeln!(out, "  {{ {body} }},");
            } else {
                let when: Vec<String> = rung.when.iter().map(|a| a.render(spec.vars)).collect();
                let _ = writeln!(out, "  {{ when = \"{}\", {body} }},", when.join(" and "));
            }
        }
        out.push(']');
        out
    }
}

/// What the fit may spend to describe the data.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// A rung is added only if it saves this fraction of the floor.
    pub min_gain: f64,
    pub max_rungs: usize,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            min_gain: 0.01,
            max_rungs: 8,
        }
    }
}

/// Why a decision could not be fitted.
#[derive(Clone, Debug, PartialEq)]
pub enum Refusal {
    NoSamples,
    /// No terminal candidate runs at every point.
    NoTerminal,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::NoSamples => write!(f, "no point was measured"),
            Refusal::NoTerminal => {
                write!(f, "no candidate that may end a ladder ran at every point")
            }
        }
    }
}

/// The first rung of `rungs` that holds at `s` and whose candidate ran
/// there.
fn lands(rungs: &[FitRung], s: &Sample) -> Option<usize> {
    rungs
        .iter()
        .find(|r| r.when.iter().all(|a| a.holds(&s.at)) && s.times[r.candidate].is_some())
        .map(|r| r.candidate)
}

fn relative(s: &Sample, c: usize) -> f64 {
    match (s.times[c], s.best()) {
        (Some(t), Some(b)) if b > 0.0 => s.weight * t / b,
        _ => f64::INFINITY,
    }
}

/// The weighted relative time of `rungs` over `samples`.
fn cost(rungs: &[FitRung], samples: &[Sample]) -> f64 {
    samples
        .iter()
        .map(|s| lands(rungs, s).map_or(f64::INFINITY, |c| relative(s, c)))
        .sum()
}

/// The weighted relative time of a per-point choice, such as the ladder a
/// profile holds today: `None` where it picked nothing measured.
fn cost_of(choice: &[Option<usize>], samples: &[Sample]) -> f64 {
    samples
        .iter()
        .zip(choice)
        .map(|(s, c)| c.map_or(f64::INFINITY, |c| relative(s, c)))
        .sum()
}

const THRESHOLDS: usize = 16;

/// One atom, or two on different variables, by index into the atoms.
type Condition = (usize, Option<usize>);

/// A ladder's rungs before its tail, as conditions and candidates.
type Rungs = Vec<(Condition, usize)>;

/// Every single-variable condition the data can tell apart.
fn atoms(spec: &Spec, samples: &[Sample]) -> Vec<Atom> {
    let mut out = Vec::new();
    for (v, var) in spec.vars.iter().enumerate() {
        let mut values: Vec<i64> = samples.iter().map(|s| s.at[v]).collect();
        values.sort_unstable();
        values.dedup();
        if values.len() < 2 {
            continue;
        }
        if var.domain.is_ordered() {
            // At most `THRESHOLDS` cuts per variable, at quantiles of the
            // values seen: enough to place a crossover, few enough to keep
            // the search over pairs small.
            let cuts: Vec<i64> = if values.len() - 1 <= THRESHOLDS {
                values[1..].to_vec()
            } else {
                let mut c: Vec<i64> = (1..=THRESHOLDS)
                    .map(|i| values[i * (values.len() - 1) / THRESHOLDS])
                    .collect();
                c.dedup();
                c
            };
            for &t in &cuts {
                out.push(Atom::AtLeast(v, t));
                out.push(Atom::Below(v, t));
            }
        } else {
            out.extend(values.iter().map(|&t| Atom::Is(v, t)));
        }
    }
    out
}

/// One local search: take the best move (replace, insert or delete a rung,
/// or change the tail) while `cost + lambda · rungs` falls; `insert` allows
/// new rungs.
#[allow(clippy::too_many_arguments)]
fn search(
    rel: &[Vec<f64>],
    holds: &[Vec<bool>],
    conditions: &[Condition],
    terminals: &[usize],
    mut ladder: Rungs,
    mut tail: usize,
    options: Options,
    lambda: f64,
    insert: bool,
) -> (Rungs, usize) {
    let inside = |(i, j): Condition, s: usize| holds[i][s] && j.is_none_or(|j| holds[j][s]);
    let points = rel.len();
    // Where each point lands on `ladder[from..]` then `tail`.
    let land_from = |ladder: &[(Condition, usize)], from: usize, tail: usize, s: usize| {
        ladder[from..]
            .iter()
            .find(|&&(c, k)| inside(c, s) && rel[s][k].is_finite())
            .map_or(tail, |&(_, k)| k)
    };
    let score = |ladder: &[(Condition, usize)], tail: usize| -> f64 {
        (0..points)
            .map(|s| rel[s][land_from(ladder, 0, tail, s)])
            .sum::<f64>()
            + lambda * ladder.len() as f64
    };
    let mut current = score(&ladder, tail);
    for _ in 0..64 {
        let mut best: Option<(f64, Rungs, usize)> = None;
        let mut offer = |value: f64, ladder: Rungs, tail: usize| {
            if value < current - 1e-12 && best.as_ref().is_none_or(|(b, ..)| value < *b) {
                best = Some((value, ladder, tail));
            }
        };
        // The tail alone.
        for &t in terminals {
            if t != tail {
                offer(score(&ladder, t), ladder.clone(), t);
            }
        }
        // Deleting a rung.
        for at in 0..ladder.len() {
            let mut shorter = ladder.clone();
            shorter.remove(at);
            offer(score(&shorter, tail), shorter, tail);
        }
        // Replacing the rung at `at`, or inserting one before it: only the
        // points that reach `at` change, and they fall to what follows.
        for at in 0..=ladder.len() {
            let reach: Vec<usize> = (0..points)
                .filter(|&s| {
                    !ladder[..at]
                        .iter()
                        .any(|&(c, k)| inside(c, s) && rel[s][k].is_finite())
                })
                .collect();
            let settled: f64 = (0..points)
                .filter(|s| !reach.contains(s))
                .map(|s| rel[s][land_from(&ladder, 0, tail, s)])
                .sum();
            for replace in [false, true] {
                if replace && at == ladder.len() {
                    continue;
                }
                if !replace && (!insert || ladder.len() + 1 >= options.max_rungs) {
                    continue;
                }
                let next = if replace { at + 1 } else { at };
                let after: Vec<usize> = reach
                    .iter()
                    .map(|&s| land_from(&ladder, next, tail, s))
                    .collect();
                let rungs = ladder.len() - usize::from(replace) + 1;
                for &cond in conditions {
                    let mask: Vec<bool> = reach.iter().map(|&s| inside(cond, s)).collect();
                    if !mask.iter().any(|m| *m) {
                        continue;
                    }
                    for k in 0..rel.first().map_or(0, Vec::len) {
                        let mut value = settled + lambda * rungs as f64;
                        for (r, &s) in reach.iter().enumerate() {
                            value += if mask[r] && rel[s][k].is_finite() {
                                rel[s][k]
                            } else {
                                rel[s][after[r]]
                            };
                        }
                        if value < current - 1e-12 && best.as_ref().is_none_or(|(b, ..)| value < *b)
                        {
                            let mut changed = ladder.clone();
                            if replace {
                                changed[at] = (cond, k);
                            } else {
                                changed.insert(at, (cond, k));
                            }
                            best = Some((value, changed, tail));
                        }
                    }
                }
            }
        }
        match best {
            Some((value, changed, t)) => {
                ladder = changed;
                tail = t;
                current = value;
            }
            None => break,
        }
    }
    (ladder, tail)
}

/// Fit an ordered ladder to the measurements.
///
/// `baseline` is the candidate today's profile runs at each point, for the
/// report; pass an empty slice for none.
pub fn fit(
    spec: &Spec,
    candidates: &[Candidate],
    samples: &[Sample],
    baseline: &[Option<usize>],
    options: Options,
) -> Result<Fit, Refusal> {
    let kept: Vec<usize> = (0..samples.len())
        .filter(|&i| samples[i].weight > 0.0 && samples[i].best().is_some())
        .collect();
    let baseline: Option<Vec<Option<usize>>> =
        (!baseline.is_empty()).then(|| kept.iter().map(|&i| baseline[i]).collect());
    let samples: Vec<Sample> = kept.iter().map(|&i| samples[i].clone()).collect();
    if samples.is_empty() {
        return Err(Refusal::NoSamples);
    }
    let floor: f64 = samples.iter().map(|s| s.weight).sum();
    // rel[s][c]: the weighted relative time, infinite where c did not run.
    let rel: Vec<Vec<f64>> = samples
        .iter()
        .map(|s| (0..candidates.len()).map(|c| relative(s, c)).collect())
        .collect();
    let terminals: Vec<usize> = (0..candidates.len())
        .filter(|&c| candidates[c].terminal)
        .collect();
    // Greedy first: from the cheapest terminal over every point, add the
    // rung that saves the most while it saves `min_gain` of the floor.
    let (mut terminal, mut total) = terminals
        .iter()
        .map(|&c| (c, rel.iter().map(|r| r[c]).sum::<f64>()))
        .filter(|(_, cost)| cost.is_finite())
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or(Refusal::NoTerminal)?;
    let mut chosen: Rungs = Vec::new();
    // Where each point lands so far: `None` is the tail.
    let mut landed: Vec<Option<usize>> = vec![None; samples.len()];

    let singles = atoms(spec, &samples);
    let holds: Vec<Vec<bool>> = singles
        .iter()
        .map(|a| samples.iter().map(|s| a.holds(&s.at)).collect())
        .collect();
    let var = |x: &Atom| match *x {
        Atom::AtLeast(v, _) | Atom::Below(v, _) | Atom::Is(v, _) => v,
    };
    let mut conditions: Vec<Condition> = (0..singles.len()).map(|i| (i, None)).collect();
    // Pairs join two variables, at least one of them dynamic: a crossover
    // moves along rows, and a pair says where it moves with something else.
    for i in 0..singles.len() {
        for j in i + 1..singles.len() {
            let (a, b) = (var(&singles[i]), var(&singles[j]));
            if a != b && (spec.vars[a].dynamic || spec.vars[b].dynamic) {
                conditions.push((i, Some(j)));
            }
        }
    }

    while chosen.len() + 1 < options.max_rungs {
        let settled: f64 = (0..samples.len())
            .filter_map(|s| landed[s].map(|c| rel[s][c]))
            .sum();
        let reach: Vec<usize> = (0..samples.len())
            .filter(|&s| landed[s].is_none())
            .collect();
        // Each terminal's cost over the reach: (finite sum, points it
        // cannot run).
        let reach_cost: Vec<(f64, usize)> = terminals
            .iter()
            .map(|&t| {
                reach.iter().fold((0.0, 0usize), |(sum, inf), &s| {
                    if rel[s][t].is_finite() {
                        (sum + rel[s][t], inf)
                    } else {
                        (sum, inf + 1)
                    }
                })
            })
            .collect();
        // (cost, condition, candidate, terminal) of the best rung so far.
        let mut best: Option<(f64, Condition, usize, usize)> = None;
        for &(i, j) in &conditions {
            let inside: Vec<usize> = reach
                .iter()
                .copied()
                .filter(|&s| holds[i][s] && j.is_none_or(|j| holds[j][s]))
                .collect();
            if inside.is_empty() {
                continue;
            }
            for c in 0..candidates.len() {
                let taken_at: Vec<usize> = inside
                    .iter()
                    .copied()
                    .filter(|&s| rel[s][c].is_finite())
                    .collect();
                if taken_at.is_empty() {
                    continue;
                }
                let taken: f64 = taken_at.iter().map(|&s| rel[s][c]).sum();
                // The tail over what is left: today's reach minus what this
                // rung takes, per terminal, counting points it cannot run.
                let mut tail_pick: Option<(usize, f64)> = None;
                for (ti, &t) in terminals.iter().enumerate() {
                    let (mut sum, mut inf) = reach_cost[ti];
                    for &s in &taken_at {
                        if rel[s][t].is_finite() {
                            sum -= rel[s][t];
                        } else {
                            inf -= 1;
                        }
                    }
                    if inf == 0 && tail_pick.is_none_or(|(_, b)| sum < b) {
                        tail_pick = Some((t, sum.max(0.0)));
                    }
                }
                let Some((t, tail_cost)) = tail_pick else {
                    continue;
                };
                let new = settled + taken + tail_cost;
                if best.as_ref().is_none_or(|(b, ..)| new < *b) {
                    best = Some((new, (i, j), c, t));
                }
            }
        }
        match best {
            Some((new, (i, j), c, t)) if total - new >= options.min_gain * floor => {
                for s in 0..samples.len() {
                    if landed[s].is_none()
                        && holds[i][s]
                        && j.is_none_or(|j| holds[j][s])
                        && rel[s][c].is_finite()
                    {
                        landed[s] = Some(c);
                    }
                }
                chosen.push(((i, j), c));
                terminal = t;
                total = new;
            }
            _ => break,
        }
    }
    // The greedy pass cannot revisit a rung, so an early broad one can hide
    // a crossover that moves with size: search locally for any move that
    // lowers the cost, then for parsimony, with each rung charged the least
    // gain it must earn, so rungs that do not earn their place go.
    let (ladder, tail) = search(
        &rel,
        &holds,
        &conditions,
        &terminals,
        chosen,
        terminal,
        options,
        0.0,
        true,
    );
    let lambda = options.min_gain * floor;
    let (ladder, tail) = search(
        &rel,
        &holds,
        &conditions,
        &terminals,
        ladder,
        tail,
        options,
        lambda,
        false,
    );
    let mut rungs: Vec<FitRung> = ladder
        .iter()
        .map(|&((i, j), c)| {
            let mut when = vec![singles[i]];
            when.extend(j.map(|j| singles[j]));
            FitRung { when, candidate: c }
        })
        .collect();
    rungs.push(FitRung {
        when: Vec::new(),
        candidate: tail,
    });
    let cost = cost(&rungs, &samples);
    Ok(Fit {
        rungs,
        cost,
        floor,
        baseline: baseline.map(|b| cost_of(&b, &samples)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{Domain, Shape, Tune};

    static VARS: [Var; 2] = [
        Var::dynamic("rows", Domain::COUNT),
        Var::fixed("bits", Domain::Ints(&[2, 4, 8])),
    ];
    static SPEC: Spec = Spec {
        id: "kernel.test",
        doc: "",
        shape: Shape::Flag,
        vars: &VARS,
        default: "",
        tune: Tune::Search,
    };

    fn candidates() -> Vec<Candidate> {
        vec![
            Candidate {
                body: "use = \"vector\"".into(),
                terminal: true,
            },
            Candidate {
                body: "use = \"tiled\"".into(),
                terminal: false,
            },
        ]
    }

    /// The vector kernel costs `rows`; the tiled one 7, and does not run
    /// below 4 rows.
    fn samples() -> Vec<Sample> {
        let mut out = Vec::new();
        for bits in [2, 4, 8] {
            for rows in [1, 2, 4, 6, 8, 12, 16, 32] {
                out.push(Sample {
                    at: vec![rows, bits],
                    weight: 1.0,
                    times: vec![Some(rows as f64), (rows >= 4).then_some(7.0)],
                });
            }
        }
        out
    }

    #[test]
    fn the_crossover_lands_where_the_tiled_kernel_starts_to_win() {
        let fit = fit(&SPEC, &candidates(), &samples(), &[], Options::default()).unwrap();
        assert_eq!(
            fit.rungs,
            vec![
                FitRung {
                    when: vec![Atom::AtLeast(0, 8)],
                    candidate: 1
                },
                FitRung {
                    when: vec![],
                    candidate: 0
                },
            ]
        );
        assert_eq!(
            fit.toml(&SPEC, &candidates()),
            "[\n  { when = \"rows >= 8\", use = \"tiled\" },\n  { use = \"vector\" },\n]"
        );
        assert!((fit.cost - fit.floor).abs() < 1e-9, "{fit:?}");
    }

    #[test]
    fn a_rung_that_saves_too_little_is_left_out() {
        let mut samples = samples();
        for s in &mut samples {
            // The tiled kernel now only wins by a hair at 32 rows.
            s.times[1] = (s.at[0] >= 4).then_some(if s.at[0] == 32 { 31.9 } else { 99.0 });
        }
        let fit = fit(&SPEC, &candidates(), &samples, &[], Options::default()).unwrap();
        assert_eq!(fit.rungs.len(), 1, "{fit:?}");
    }

    #[test]
    fn a_ladder_never_ends_on_a_kernel_that_does_not_run_everywhere() {
        let mut c = candidates();
        c[0].terminal = false;
        assert_eq!(
            fit(&SPEC, &c, &samples(), &[], Options::default()).unwrap_err(),
            Refusal::NoTerminal
        );
    }

    #[test]
    fn the_baseline_is_priced_on_the_same_points() {
        let base: Vec<Option<usize>> = samples().iter().map(|_| Some(0)).collect();
        let fit = fit(&SPEC, &candidates(), &samples(), &base, Options::default()).unwrap();
        assert!(fit.baseline.unwrap() > fit.cost);
    }

    /// A large bank turns tiled at fewer rows than a small one: the fit
    /// says so with a size condition, rather than splitting the difference.
    #[test]
    fn a_crossover_that_moves_with_size_gets_a_size_condition() {
        static SIZED: [Var; 2] = [
            Var::dynamic("rows", Domain::COUNT),
            Var::fixed("n", Domain::COUNT),
        ];
        static SPEC: Spec = Spec {
            id: "kernel.sized",
            doc: "",
            shape: Shape::Flag,
            vars: &SIZED,
            default: "",
            tune: Tune::Search,
        };
        let mut samples = Vec::new();
        for n in [4096, 34816] {
            for rows in [1, 2, 4, 6, 8, 10, 12, 16, 32] {
                // Tiled costs a flat tile; on the big bank the tile is
                // cheaper per row, so it wins from 8 rows, not 12.
                let tile = if n > 8192 { 7.5 } else { 11.5 };
                samples.push(Sample {
                    at: vec![rows, n],
                    weight: 1.0,
                    times: vec![Some(rows as f64), (rows >= 4).then_some(tile)],
                });
            }
        }
        let fit = fit(&SPEC, &candidates(), &samples, &[], Options::default()).unwrap();
        assert!(
            (fit.cost - fit.floor).abs() < 1e-9,
            "{}",
            fit.toml(&SPEC, &candidates())
        );
        let toml = fit.toml(&SPEC, &candidates());
        assert!(
            toml.contains("n >= 34816") || toml.contains("n < 34816"),
            "{toml}"
        );
    }
}
