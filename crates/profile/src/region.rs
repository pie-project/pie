//! Sets of shape points.
//!
//! Every `when` normalises to a [`Region`]: a union of boxes, a box holding
//! one [`Set`] per variable. Integer variables, finite integer sets,
//! enumerations (by value index) and booleans (0 and 1) all live on the same
//! integer line, so one set type serves them all.

/// A union of disjoint, non-adjacent, inclusive integer intervals, kept sorted.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Set(Vec<(i64, i64)>);

impl Set {
    #[must_use]
    pub fn empty() -> Set {
        Set(Vec::new())
    }

    #[must_use]
    pub fn span(lo: i64, hi: i64) -> Set {
        if lo > hi {
            Set::empty()
        } else {
            Set(vec![(lo, hi)])
        }
    }

    #[must_use]
    pub fn point(v: i64) -> Set {
        Set(vec![(v, v)])
    }

    #[must_use]
    pub fn points(values: &[i64]) -> Set {
        let mut set = Set::empty();
        for &v in values {
            set = set.union(&Set::point(v));
        }
        set
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[must_use]
    pub fn contains(&self, v: i64) -> bool {
        self.0.iter().any(|&(lo, hi)| lo <= v && v <= hi)
    }

    #[must_use]
    pub fn intervals(&self) -> &[(i64, i64)] {
        &self.0
    }

    #[must_use]
    pub fn min(&self) -> Option<i64> {
        self.0.first().map(|&(lo, _)| lo)
    }

    #[must_use]
    pub fn max(&self) -> Option<i64> {
        self.0.last().map(|&(_, hi)| hi)
    }

    #[must_use]
    pub fn is_subset(&self, of: &Set) -> bool {
        self.minus(of).is_empty()
    }

    #[must_use]
    pub fn intersect(&self, other: &Set) -> Set {
        let mut out = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < self.0.len() && j < other.0.len() {
            let (a_lo, a_hi) = self.0[i];
            let (b_lo, b_hi) = other.0[j];
            let lo = a_lo.max(b_lo);
            let hi = a_hi.min(b_hi);
            if lo <= hi {
                out.push((lo, hi));
            }
            if a_hi < b_hi {
                i += 1;
            } else {
                j += 1;
            }
        }
        Set(out)
    }

    #[must_use]
    pub fn union(&self, other: &Set) -> Set {
        let mut all: Vec<(i64, i64)> = self.0.iter().chain(&other.0).copied().collect();
        all.sort_unstable();
        let mut out: Vec<(i64, i64)> = Vec::with_capacity(all.len());
        for (lo, hi) in all {
            match out.last_mut() {
                Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
                _ => out.push((lo, hi)),
            }
        }
        Set(out)
    }

    /// The members of `self` that are not in `other`.
    #[must_use]
    pub fn minus(&self, other: &Set) -> Set {
        let mut out = Vec::new();
        for &(lo, hi) in &self.0 {
            let mut cursor = lo;
            let mut done = false;
            for &(b_lo, b_hi) in &other.0 {
                if b_hi < cursor || b_lo > hi {
                    continue;
                }
                if b_lo > cursor {
                    out.push((cursor, b_lo - 1));
                }
                if b_hi >= hi {
                    done = true;
                    break;
                }
                cursor = b_hi + 1;
            }
            if !done {
                out.push((cursor, hi));
            }
        }
        Set(out)
    }
}

/// A union of boxes over a fixed list of variables. A box with an empty
/// coordinate is dropped as soon as it appears, so an empty region has no
/// boxes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    boxes: Vec<Vec<Set>>,
}

impl Region {
    #[must_use]
    pub fn full(space: &[Set]) -> Region {
        Region::of_box(space.to_vec())
    }

    #[must_use]
    pub fn nothing() -> Region {
        Region { boxes: Vec::new() }
    }

    /// The points of `space` whose `var` lies in `set`.
    #[must_use]
    pub fn on(space: &[Set], var: usize, set: &Set) -> Region {
        let mut b = space.to_vec();
        b[var] = b[var].intersect(set);
        Region::of_box(b)
    }

    fn of_box(b: Vec<Set>) -> Region {
        if b.iter().any(Set::is_empty) {
            Region::nothing()
        } else {
            Region { boxes: vec![b] }
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.boxes.is_empty()
    }

    #[must_use]
    pub fn boxes(&self) -> &[Vec<Set>] {
        &self.boxes
    }

    #[must_use]
    pub fn contains(&self, at: &[i64]) -> bool {
        self.boxes
            .iter()
            .any(|b| b.iter().zip(at).all(|(set, &v)| set.contains(v)))
    }

    #[must_use]
    pub fn union(&self, other: &Region) -> Region {
        let mut boxes = self.boxes.clone();
        boxes.extend(other.boxes.iter().cloned());
        Region { boxes }.simplified()
    }

    #[must_use]
    pub fn intersect(&self, other: &Region) -> Region {
        let mut boxes = Vec::new();
        for a in &self.boxes {
            for b in &other.boxes {
                let c: Vec<Set> = a.iter().zip(b).map(|(x, y)| x.intersect(y)).collect();
                if !c.iter().any(Set::is_empty) {
                    boxes.push(c);
                }
            }
        }
        Region { boxes }.simplified()
    }

    #[must_use]
    pub fn complement(&self, space: &[Set]) -> Region {
        let mut out = Region::full(space);
        for b in &self.boxes {
            let mut outside = Region::nothing();
            for (var, set) in b.iter().enumerate() {
                outside = outside.union(&Region::on(space, var, &space[var].minus(set)));
            }
            out = out.intersect(&outside);
            if out.is_empty() {
                break;
            }
        }
        out
    }

    #[must_use]
    pub fn minus(&self, other: &Region, space: &[Set]) -> Region {
        self.intersect(&other.complement(space))
    }

    /// Whether every point of `self` lies in `of`.
    #[must_use]
    pub fn is_subset(&self, of: &Region, space: &[Set]) -> bool {
        self.minus(of, space).is_empty()
    }

    /// The region with `var` pinned to `value`, or nothing when no box holds
    /// it. This is partial evaluation: a static variable substituted.
    #[must_use]
    pub fn pinned(&self, var: usize, value: i64) -> Region {
        let boxes = self
            .boxes
            .iter()
            .filter(|b| b[var].contains(value))
            .map(|b| {
                let mut b = b.clone();
                b[var] = Set::point(value);
                b
            })
            .collect();
        Region { boxes }.simplified()
    }

    /// The region over the variables in `keep` alone: a point is in it when
    /// some point of `self` agrees with it there.
    #[must_use]
    pub fn project(&self, keep: std::ops::Range<usize>) -> Region {
        let boxes = self
            .boxes
            .iter()
            .map(|b| b[keep.clone()].to_vec())
            .collect();
        Region { boxes }.simplified()
    }

    /// The region with the variables of `extra` appended, unconstrained.
    #[must_use]
    pub fn lifted(&self, extra: &[Set]) -> Region {
        let boxes = self
            .boxes
            .iter()
            .map(|b| b.iter().chain(extra).cloned().collect())
            .collect();
        Region { boxes }
    }

    /// Drop boxes another box already holds.
    fn simplified(mut self) -> Region {
        let mut keep: Vec<Vec<Set>> = Vec::with_capacity(self.boxes.len());
        self.boxes
            .sort_by_key(|b| std::cmp::Reverse(b.iter().map(|s| s.0.len()).sum::<usize>()));
        'next: for b in self.boxes {
            for k in &keep {
                if b.iter().zip(k).all(|(x, y)| x.is_subset(y)) {
                    continue 'next;
                }
            }
            keep.retain(|k| !k.iter().zip(&b).all(|(x, y)| x.is_subset(y)));
            keep.push(b);
        }
        Region { boxes: keep }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_set_unions_adjacent_spans_into_one() {
        assert_eq!(Set::span(0, 4).union(&Set::span(5, 9)), Set::span(0, 9));
        assert_eq!(
            Set::points(&[1, 2, 4]).intervals(),
            &[(1, 2), (4, 4)],
            "points that do not touch stay apart"
        );
    }

    #[test]
    fn a_set_minus_cuts_holes_and_ends() {
        let s = Set::span(0, 100).minus(&Set::span(10, 20));
        assert_eq!(s.intervals(), &[(0, 9), (21, 100)]);
        assert!(Set::span(0, 5).minus(&Set::span(0, 5)).is_empty());
        assert_eq!(Set::span(0, 5).minus(&Set::span(3, 9)), Set::span(0, 2));
        assert_eq!(Set::span(0, 5).minus(&Set::span(-9, 2)), Set::span(3, 5));
        assert_eq!(
            Set::span(i64::MIN, i64::MAX)
                .minus(&Set::point(0))
                .intervals(),
            &[(i64::MIN, -1), (1, i64::MAX)],
            "the ends of the line do not overflow"
        );
    }

    #[test]
    fn a_complement_of_a_complement_is_the_region() {
        let space = vec![Set::span(0, 100), Set::points(&[2, 4, 8])];
        let r = Region::on(&space, 0, &Set::span(0, 9)).intersect(&Region::on(
            &space,
            1,
            &Set::point(4),
        ));
        let back = r.complement(&space).complement(&space);
        assert!(back.is_subset(&r, &space) && r.is_subset(&back, &space));
        assert!(r.contains(&[3, 4]));
        assert!(!r.contains(&[3, 2]));
        assert!(!r.contains(&[10, 4]));
    }

    #[test]
    fn a_region_and_its_complement_cover_the_space() {
        let space = vec![Set::span(0, 1000), Set::span(0, 1000)];
        let r = Region::on(&space, 0, &Set::span(0, 4)).union(&Region::on(
            &space,
            1,
            &Set::span(500, 1000),
        ));
        let all = r.union(&r.complement(&space));
        assert!(Region::full(&space).is_subset(&all, &space));
        assert!(r.intersect(&r.complement(&space)).is_empty());
    }

    #[test]
    fn pinning_a_variable_keeps_only_boxes_that_hold_it() {
        let space = vec![Set::span(0, 100), Set::points(&[2, 4, 8])];
        let r = Region::on(&space, 1, &Set::point(4)).intersect(&Region::on(
            &space,
            0,
            &Set::span(0, 9),
        ));
        assert!(r.pinned(1, 2).is_empty());
        assert!(!r.pinned(1, 4).is_empty());
    }
}
