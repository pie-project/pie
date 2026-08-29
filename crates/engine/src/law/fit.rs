//! The fitting engine: exact integer linear algebra over recorded samples.
//!
//! **THERE IS NO DEVICE IN THIS FILE AND THERE never was.** It moved here out
//! of `engine_metal::abi` unchanged, because everything it does is arithmetic
//! over `i128` — a two-point exact-rational slope read off a probe ladder, a
//! bounded search for a tiling law's divisor by interval arithmetic, a
//! Cramer's-rule inversion of the probe basis over the integers — and a shell
//! is only ever the thing that supplies the samples. Both planes' recorders
//! can call it; neither can be the only one that owns it.
//!
//! Every answer here is EXACT or refused. Nothing rounds: a slope that is not
//! a whole multiple, a basis whose inverse is not integral, a ceiling whose
//! offset interval is empty — each is a named refusal
//! ([`Refusal`](super::Refusal)) rather than a nearest fit, because the
//! consumer is an artifact that will be replayed for the life of a load and a
//! law that is nearly right is a wrong number forever.

use super::{Axis, Law, Recipe, Refusal, Refuse};

/// The largest divisor a tiling law is searched for.
///
/// A tile is a compile-time constant of a kernel entry and the ones that
/// exist are 32 (`SDPA_TILE`, `TILE_M`, `TILE_N`) and 128 (`VECTOR_GROUP`).
/// The ceiling is stated so the search is bounded and so a tile past it is a
/// named miss rather than a silent one.
pub const MAX_TILE: i128 = 512;

/// The largest multiplier on the window's rows a tiling law's numerator is
/// searched for.
pub const MAX_NUMERATOR_SCALE: i128 = 32;

/// One component of one arm: a constant, a law, or a refusal.
///
/// `observed` is `(coords, window rows, value)` at every sample of the arm.
///
/// # Errors
///
/// [`Refuse::Unaffine`] naming what the line said and what the tiling search
/// said, for a component neither form predicts.
pub fn component(
    axes: &[Axis],
    observed: &[(Vec<i128>, i128, i128)],
) -> Result<Option<Law>, Refusal> {
    let first = observed[0].2;
    // **A CONSTANT IS A CLAIM AND IT GETS CHECKED**, over every sample of the
    // arm rather than over the ones a bump happened to move. Build log 30's
    // bug was exactly here: a grid axis written `rows.div_ceil(32)` is flat
    // across every step small enough to stay inside one tile, and a fit that
    // only verified the components it had already decided were variable would
    // call it a constant and encode the wrong grid into the ICB once, at
    // load.
    if observed.iter().all(|(_, _, v)| *v == first) {
        return Ok(None);
    }
    let points: Vec<(Vec<i128>, i128)> = observed
        .iter()
        .map(|(coords, _, value)| (coords.clone(), *value))
        .collect();
    match affine(axes, &points) {
        Ok(law) => Ok(Some(law)),
        Err(affine_why) => match ceiling(observed) {
            Some(law) => Ok(Some(law)),
            None => Err(Refusal::new(
                Refuse::Unaffine,
                format!(
                    "{}; and no tiling law \
                     `mul·ceil((α·rows + β)/div)` with div ≤ {MAX_TILE} and α ≤ \
                     {MAX_NUMERATOR_SCALE} fits it either — the samples catch the staircase \
                     crossing {} time(s), and two is what pins the period",
                    affine_why.why,
                    crossings(observed)
                ),
            )),
        },
    }
}

/// `base + Σ slope·coord`, fitted exactly and verified at every sample.
///
/// # Errors
///
/// [`Refuse::Unaffine`] for a direction no two samples witness, a step that
/// is not a whole multiple, two disagreeing slopes along one direction, or a
/// line that does not reproduce every sample it was fitted from.
pub fn affine(axes: &[Axis], points: &[(Vec<i128>, i128)]) -> Result<Law, Refusal> {
    let refuse = |why: String| Refusal::new(Refuse::Unaffine, why);
    let (here, value) = &points[0];
    let mut slope = vec![0i128; axes.len()];
    for k in 0..axes.len() {
        // A pair of samples that differ ONLY in direction k. A probe's base
        // and its own ladder are exactly that; anything else the harness
        // supplies is a bonus.
        let mut seen: Option<i128> = None;
        for (a, (xa, va)) in points.iter().enumerate() {
            for (xb, vb) in points.iter().skip(a + 1) {
                let moved: Vec<usize> = xa
                    .iter()
                    .zip(xb)
                    .enumerate()
                    .filter_map(|(at, (p, q))| (p != q).then_some(at))
                    .collect();
                if moved != vec![k] {
                    continue;
                }
                let run = xb[k] - xa[k];
                let rise = vb - va;
                if rise % run != 0 {
                    return Err(refuse(format!(
                        "stepping `{}` by {run} moved it by {rise}, which is not a whole \
                         multiple",
                        axes[k]
                    )));
                }
                let b = rise / run;
                match seen {
                    None => seen = Some(b),
                    Some(had) if had == b => {}
                    Some(had) => {
                        return Err(refuse(format!(
                            "`{}` moves it by {had} per step at one place and {b} at another",
                            axes[k]
                        )));
                    }
                }
            }
        }
        match seen {
            Some(b) => slope[k] = b,
            None => {
                return Err(refuse(format!(
                    "no two samples of this arm differ only in `{}`, so its slope is \
                     unwitnessed",
                    axes[k]
                )));
            }
        }
    }
    // The value at the ZERO of the coordinates, not at this arm's first
    // sample: a law has to be evaluable anywhere, including at the
    // compositions the probes could not visit (an empty class is one).
    let base = value - slope.iter().zip(here).map(|(b, x)| b * x).sum::<i128>();
    let law = Law::Affine { base, slope };
    for (coords, want) in points {
        let got = law.at(coords, 0).expect("an affine law is total");
        if got != *want {
            return Err(refuse(format!(
                "the line fitted from the ladders predicts {got} at {coords:?} and the walk \
                 produced {want}"
            )));
        }
    }
    Ok(law)
}

/// `mul·⌈(α·rows + β)/div⌉`, solved by interval arithmetic over the samples.
///
/// **THE SEARCH IS BOUNDED AND THE ANSWER IS CANONICAL.** For a candidate
/// `(mul, α, div)` the offset β is not searched at all: every sample says
/// `div·(w−1) < α·rows + β ≤ div·w`, so β lies in one half-open interval and
/// the intersection over the samples is one interval or empty. The smallest
/// `mul`, then the smallest `div`, then the smallest `α` that leaves a
/// non-empty interval wins — smallest because `⌈n/32⌉` and `⌈2n/64⌉` are the
/// same function and one of them is the one a reader can check against
/// `SDPA_TILE`.
#[must_use]
pub fn ceiling(observed: &[(Vec<i128>, i128, i128)]) -> Option<Law> {
    if crossings(observed) < 2 {
        return None;
    }
    let mut common: i128 = 0;
    for (_, _, v) in observed {
        common = gcd(common, v.abs());
    }
    if common == 0 {
        return None;
    }
    let mut scales: Vec<i128> = (1..=common).filter(|m| common % m == 0).collect();
    scales.sort_unstable();
    for mul in scales {
        for div in 2..=MAX_TILE {
            for alpha in 1..=MAX_NUMERATOR_SCALE {
                let mut lo = i128::MIN;
                let mut hi = i128::MAX;
                let mut fits = true;
                for (_, rows, value) in observed {
                    if value % mul != 0 {
                        fits = false;
                        break;
                    }
                    let w = value / mul;
                    lo = lo.max(div * (w - 1) - alpha * rows);
                    hi = hi.min(div * w - alpha * rows);
                }
                if !fits || lo >= hi {
                    continue;
                }
                let law = Law::Ceil {
                    mul,
                    alpha,
                    beta: hi,
                    div,
                };
                if observed
                    .iter()
                    .all(|(coords, rows, value)| law.at(coords, *rows) == Some(*value))
                {
                    return Some(law);
                }
            }
        }
    }
    None
}

/// How many times the samples catch the staircase in the act: a pair of
/// window row counts `r` and `r+1` whose values differ.
///
/// **TWO, OR THE PERIOD IS NOT PINNED.** One crossing says only that the
/// value stepped somewhere between two rows, and a ceiling with any divisor
/// wide enough to hold the sampled range explains that — over rows 16..47,
/// `⌈(r−15)/17⌉` is `⌈r/32⌉` exactly, and the search would answer 17 because
/// 17 is smaller. Two crossings say how far apart the steps are, which is the
/// divisor. This is the fitter's third-point discipline, in the form the
/// tiling law needs it: sample points straddling a multiple of the tile, and
/// then straddling the next one.
#[must_use]
pub fn crossings(observed: &[(Vec<i128>, i128, i128)]) -> usize {
    let mut by_rows: std::collections::BTreeMap<i128, i128> = std::collections::BTreeMap::new();
    for (_, rows, value) in observed {
        by_rows.insert(*rows, *value);
    }
    by_rows
        .iter()
        .filter(|(rows, value)| {
            by_rows
                .get(&(**rows + 1))
                .is_some_and(|next| next != *value)
        })
        .count()
}

fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// One sample the basis inversion is verified against: a class table, and
/// where the harness placed that walk in the probe basis.
pub type Site<'a> = (&'a [(u32, u32)], &'a [i128]);

/// Invert the basis: one linear functional per direction over the class
/// table's own numbers, solved exactly and verified at every probe.
///
/// **THIS IS WHAT MAKES THE TABLE READABLE BY SOMETHING THAT DID NOT WALK.**
/// The laws are written in a basis of reachable directions; a fire carries a
/// class table. `step` says what one unit of each direction does to that
/// table, which is a `2·classes × directions` matrix; a square subsystem of
/// it that inverts over the integers is the recipe. There may be several and
/// they agree — the verification over every probe is what says so.
///
/// # Errors
///
/// [`Refuse::Unstructured`] for a direction whose step is stated over the
/// wrong number of classes, for a basis no square subsystem inverts over the
/// integers, and for a recipe that reads some sample's class table back into
/// coordinates the harness did not place it at.
pub fn invert(
    axes: &[Axis],
    origin_classes: &[(u32, u32)],
    origin: &[i128],
    every: &[Site<'_>],
) -> Result<Vec<Recipe>, Refusal> {
    let refuse = |why: String| Refusal::new(Refuse::Unstructured, why);
    let classes = origin_classes.len();
    let k = axes.len();
    for axis in axes {
        if axis.step.len() != classes {
            return Err(refuse(format!(
                "direction `{axis}` states a step over {} classes and the base \
                 composition has {classes}",
                axis.step.len()
            )));
        }
    }
    // The full `2·classes × k` step matrix, row `2c` = class c's rows, row
    // `2c+1` = its lanes.
    let column = |row: usize, axis: usize| -> i128 {
        let (rows, lanes) = axes[axis].step[row / 2];
        i128::from(if row.is_multiple_of(2) { rows } else { lanes })
    };
    let width = 2 * classes;
    // Every choice of `k` rows, in index order, until one inverts.
    let mut pivots: Vec<usize> = (0..k).collect();
    loop {
        if let Some(recipe) = try_pivots(&pivots, k, classes, origin_classes, origin, &column) {
            // The recipe is a claim about every probe, not only about the
            // rows it was solved from.
            for (walk_classes, walk_coords) in every {
                let got: Vec<i128> = recipe.iter().map(|row| row.at(walk_classes)).collect();
                if got != *walk_coords {
                    return Err(refuse(format!(
                        "the basis inverts to a recipe that reads {got:?} out of the \
                         class table {walk_classes:?}, and the harness placed that walk at \
                         {walk_coords:?} — the directions are not independent in the \
                         descriptor's own numbers"
                    )));
                }
            }
            return Ok(recipe);
        }
        // Next combination of `k` rows out of `width`.
        let mut at = k;
        loop {
            if at == 0 {
                return Err(refuse(format!(
                    "no {k} of the class table's {width} numbers invert this basis over \
                     the integers, so a fire's composition cannot be read back into the \
                     coordinates the laws are written in"
                )));
            }
            at -= 1;
            if pivots[at] < width - (k - at) {
                pivots[at] += 1;
                for next in at + 1..k {
                    pivots[next] = pivots[next - 1] + 1;
                }
                break;
            }
        }
    }
}

/// One choice of pivot rows, inverted by Cramer's rule over the integers.
// The two `0..k` loops index three parallel things at once — the identity
// column, the transposed matrix's row, and the origin's coordinate — so the
// index is the subject and not an accident of the iteration.
#[allow(clippy::needless_range_loop)]
fn try_pivots(
    pivots: &[usize],
    k: usize,
    classes: usize,
    origin_classes: &[(u32, u32)],
    origin: &[i128],
    column: &dyn Fn(usize, usize) -> i128,
) -> Option<Vec<Recipe>> {
    let a: Vec<Vec<i128>> = pivots
        .iter()
        .map(|row| (0..k).map(|axis| column(*row, axis)).collect())
        .collect();
    let det = determinant(&a);
    if det == 0 {
        return None;
    }
    // Row `k` of A⁻¹: solve `xᵀ·A = e_k`, i.e. `Aᵀ·x = e_k`, by Cramer.
    let mut recipe = Vec::with_capacity(k);
    for axis in 0..k {
        let mut coefficients = vec![0i128; k];
        for pivot in 0..k {
            let mut m = transpose(&a);
            for row in 0..k {
                m[row][pivot] = i128::from(row == axis);
            }
            let numerator = determinant(&m);
            if numerator % det != 0 {
                return None;
            }
            coefficients[pivot] = numerator / det;
        }
        let mut rows = vec![0i128; classes];
        let mut lanes = vec![0i128; classes];
        let mut konst = origin[axis];
        for (pivot, weight) in pivots.iter().zip(&coefficients) {
            let class = pivot / 2;
            let (r, l) = origin_classes[class];
            if pivot % 2 == 0 {
                rows[class] += *weight;
                konst -= *weight * i128::from(r);
            } else {
                lanes[class] += *weight;
                konst -= *weight * i128::from(l);
            }
        }
        recipe.push(Recipe {
            konst,
            rows,
            lanes,
        });
    }
    Some(recipe)
}

fn transpose(a: &[Vec<i128>]) -> Vec<Vec<i128>> {
    let n = a.len();
    (0..n)
        .map(|row| (0..n).map(|col| a[col][row]).collect())
        .collect()
}

/// Laplace expansion. The matrices here are `k × k` with `k` the number of
/// probe directions — three today, and a basis with more than a handful of
/// directions is a harness that has lost the plot.
fn determinant(a: &[Vec<i128>]) -> i128 {
    let n = a.len();
    match n {
        0 => 1,
        1 => a[0][0],
        2 => a[0][0] * a[1][1] - a[0][1] * a[1][0],
        _ => {
            let mut sum = 0;
            for col in 0..n {
                let minor: Vec<Vec<i128>> = a[1..]
                    .iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .filter_map(|(at, v)| (at != col).then_some(*v))
                            .collect()
                    })
                    .collect();
                let sign = if col % 2 == 0 { 1 } else { -1 };
                sum += sign * a[0][col] * determinant(&minor);
            }
            sum
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axes() -> Vec<Axis> {
        vec![
            Axis::new("a decode lane", vec![(1, 1), (0, 0)]),
            Axis::new("a prefill lane of 8 tokens", vec![(0, 0), (8, 1)]),
            Axis::new("one more prefill token", vec![(0, 0), (1, 0)]),
        ]
    }

    /// The basis the live Metal harness uses, inverted: two classes, three
    /// directions, and the answer is the one a reader can check by hand.
    #[test]
    fn the_probe_basis_inverts_into_a_reading_of_the_class_table() {
        let origin_classes = vec![(2u32, 2u32), (16, 2)];
        let coords = [0i128, 0, 0];
        let recipe = invert(
            &axes(),
            &origin_classes,
            &coords,
            &[(origin_classes.as_slice(), coords.as_slice())],
        )
        .expect("inverts");
        // d = decode rows − 2; p = prefill lanes − 2; t = prefill rows − 8·prefill lanes.
        assert_eq!(recipe[0].at(&[(5, 5), (16, 2)]), 3);
        assert_eq!(recipe[1].at(&[(2, 2), (24, 3)]), 1);
        assert_eq!(recipe[2].at(&[(2, 2), (17, 2)]), 1);
        // AND THE ALL-DECODE COMPOSITION, which is the one the probes cannot
        // visit and the ICB has to serve: every prefill number is zero.
        assert_eq!(recipe[1].at(&[(4, 4), (0, 0)]), -2);
        assert_eq!(recipe[2].at(&[(4, 4), (0, 0)]), 0);
    }

    /// A basis whose inverse is not integral is refused by name rather than
    /// rounded — and the refusal is [`Refuse::Unstructured`], not a fit.
    #[test]
    fn a_basis_that_does_not_invert_over_the_integers_is_refused_by_name() {
        let origin_classes = vec![(0u32, 0u32)];
        let coords = [0i128, 0];
        let basis = vec![
            Axis::new("two rows at a time", vec![(2, 0)]),
            Axis::new("four rows at a time", vec![(4, 0)]),
        ];
        let refused = invert(
            &basis,
            &origin_classes,
            &coords,
            &[(origin_classes.as_slice(), coords.as_slice())],
        )
        .expect_err("two parallel directions cannot both be read back");
        assert_eq!(refused.reason, Refuse::Unstructured);
    }

    /// `⌈rows/32⌉` is refused as affine and fitted as a tiling law, and the
    /// answer is the constant a reader can find in `kernels_metal::attn`.
    #[test]
    fn a_div_ceil_grid_axis_fits_the_tiling_law_and_names_the_tile() {
        let observed: Vec<(Vec<i128>, i128, i128)> = (16..80)
            .map(|rows: i128| {
                (
                    vec![0, 0, rows - 16],
                    rows,
                    rows.div_euclid(32) + i128::from(rows % 32 != 0),
                )
            })
            .collect();
        let law = ceiling(&observed).expect("the tiling law fits");
        assert_eq!(
            law,
            Law::Ceil {
                mul: 1,
                alpha: 1,
                beta: 0,
                div: 32
            }
        );
        for rows in 1..200i128 {
            let want = rows.div_euclid(32) + i128::from(rows % 32 != 0);
            assert_eq!(law.at(&[], rows), Some(want), "at {rows} rows");
        }
    }

    /// `2·⌈rows/32⌉` — `linear::gemm`'s tile arm — needs the multiplier, and
    /// the fit finds it rather than being told.
    #[test]
    fn a_scaled_tiling_law_finds_its_multiplier() {
        let observed: Vec<(Vec<i128>, i128, i128)> = (32..112)
            .map(|rows: i128| {
                (
                    vec![0, 0, rows],
                    rows,
                    2 * (rows.div_euclid(32) + i128::from(rows % 32 != 0)),
                )
            })
            .collect();
        let law = ceiling(&observed).expect("the scaled tiling law fits");
        assert_eq!(
            law,
            Law::Ceil {
                mul: 2,
                alpha: 1,
                beta: 0,
                div: 32
            }
        );
    }

    /// A staircase no ceiling explains is still a refusal.
    #[test]
    fn a_component_that_is_neither_law_is_still_refused() {
        let observed: Vec<(Vec<i128>, i128, i128)> = (1..40)
            .map(|rows: i128| (vec![rows], rows, rows * rows))
            .collect();
        assert_eq!(ceiling(&observed), None);
        let one = vec![Axis::new("one more row", vec![(1, 0)])];
        let refused = component(&one, &observed).expect_err("neither law predicts a square");
        assert_eq!(refused.reason, Refuse::Unaffine);
    }

    /// A component that never moves across the samples is not a law at all —
    /// it is a constant the encode states once.
    #[test]
    fn a_component_that_never_moves_is_no_law_and_no_refusal() {
        let one = vec![Axis::new("one more row", vec![(1, 0)])];
        let observed: Vec<(Vec<i128>, i128, i128)> =
            (1..8i128).map(|rows| (vec![rows], rows, 4096)).collect();
        assert_eq!(component(&one, &observed).expect("a constant is not a refusal"), None);
    }

    /// The line is fitted off the ladder and VERIFIED off the lattice: the
    /// base is the value at the zero of the coordinates, which is a point no
    /// sample visited.
    #[test]
    fn an_affine_law_extrapolates_off_the_probed_box() {
        let basis = axes();
        let points: Vec<(Vec<i128>, i128)> = vec![
            (vec![0, 0, 0], 100),
            (vec![1, 0, 0], 107),
            (vec![0, 1, 0], 100 + 12288),
            (vec![0, 0, 1], 101),
        ];
        let law = affine(&basis, &points).expect("one slope per direction is witnessed");
        assert_eq!(
            law,
            Law::Affine {
                base: 100,
                slope: vec![7, 12288, 1]
            }
        );
        assert_eq!(law.at(&[-2, 0, 0], 0), Some(86), "a law evaluates outside the box");
        assert_eq!(law.reads(), vec![0, 1, 2]);
    }

    /// A direction no pair of samples steps alone leaves its slope
    /// unwitnessed, and an unwitnessed slope is refused rather than assumed
    /// to be zero.
    #[test]
    fn a_direction_no_ladder_stepped_is_refused_rather_than_assumed_flat() {
        let basis = axes();
        let points: Vec<(Vec<i128>, i128)> = vec![(vec![0, 0, 0], 1), (vec![1, 0, 0], 2)];
        let refused = affine(&basis, &points).expect_err("two directions are unwitnessed");
        assert_eq!(refused.reason, Refuse::Unaffine);
        assert!(refused.why.contains("unwitnessed"), "{refused}");
    }

    /// A slot law is the one form the fit never produces and the one form
    /// the coordinates cannot evaluate.
    #[test]
    fn a_slot_law_reads_the_descriptor_and_not_the_coordinates() {
        let law = Law::Slot(super::super::SlotId(2));
        assert_eq!(law.at(&[1, 2, 3], 8), None);
        assert_eq!(law.at_in(&[1, 2, 3], 8, &[10, 11, 12]), Some(12));
        assert_eq!(law.at_in(&[], 0, &[10]), None);
        assert!(law.varies());
        assert_eq!(law.kind(), "slot");
    }
}
