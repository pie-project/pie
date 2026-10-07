use super::{Axis, Law, Recipe, Refusal, Refuse};

pub const MAX_TILE: i128 = 512;

pub const MAX_NUMERATOR_SCALE: i128 = 32;

pub fn component(
    axes: &[Axis],
    observed: &[(Vec<i128>, i128, i128)],
) -> Result<Option<Law>, Refusal> {
    let first = observed[0].2;
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

pub fn affine(axes: &[Axis], points: &[(Vec<i128>, i128)]) -> Result<Law, Refusal> {
    let refuse = |why: String| Refusal::new(Refuse::Unaffine, why);
    let (here, value) = &points[0];
    let mut slope = vec![0i128; axes.len()];
    for k in 0..axes.len() {
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

pub type Site<'a> = (&'a [(u32, u32)], &'a [i128]);

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
    let column = |row: usize, axis: usize| -> i128 {
        let (rows, lanes) = axes[axis].step[row / 2];
        i128::from(if row.is_multiple_of(2) { rows } else { lanes })
    };
    let width = 2 * classes;
    let mut pivots: Vec<usize> = (0..k).collect();
    loop {
        if let Some(recipe) = try_pivots(&pivots, k, classes, origin_classes, origin, &column) {
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
        recipe.push(Recipe { konst, rows, lanes });
    }
    Some(recipe)
}

fn transpose(a: &[Vec<i128>]) -> Vec<Vec<i128>> {
    let n = a.len();
    (0..n)
        .map(|row| (0..n).map(|col| a[col][row]).collect())
        .collect()
}

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

    #[test]
    fn fit_every_case() {
        a_div_ceil_grid_axis_fits_the_tiling_law_and_names_the_tile();
        an_affine_law_extrapolates_off_the_probed_box();
        a_slot_law_reads_the_descriptor_and_not_the_coordinates();
    }

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
        assert_eq!(
            law.at(&[-2, 0, 0], 0),
            Some(86),
            "a law evaluates outside the box"
        );
        assert_eq!(law.reads(), vec![0, 1, 2]);
    }

    fn a_slot_law_reads_the_descriptor_and_not_the_coordinates() {
        let law = Law::Slot(super::super::SlotId(2));
        assert_eq!(law.at(&[1, 2, 3], 8), None);
        assert_eq!(law.at_in(&[1, 2, 3], 8, &[10, 11, 12]), Some(12));
        assert_eq!(law.at_in(&[], 0, &[10]), None);
        assert!(law.varies());
        assert_eq!(law.kind(), "slot");
    }
}
