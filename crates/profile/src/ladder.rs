//! Rungs, ladders, and the trait a decision's Rust type implements.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::region::{Region, Set};
use crate::spec::{Spec, Value, space};
use crate::when::When;

/// What a rung picks, before it is typed.
#[derive(Clone, Debug, PartialEq)]
pub enum Pick {
    /// A choice's alternative (index into [`Spec::alternatives`]) with every
    /// constant filled, in declaration order.
    Alternative { index: usize, constants: Vec<Value> },
    /// A constant decision's value.
    Value(Value),
    /// A flag that holds.
    On,
}

/// One rung, untyped: where it holds and what it picks.
#[derive(Clone, Debug, PartialEq)]
pub struct Rung {
    pub when: Option<When>,
    pub region: Region,
    pub pick: Pick,
}

/// The shape a decision is evaluated at: one coordinate per declared
/// variable, in declaration order. Enumerations are their value's index;
/// `None` is a value this point does not know.
pub trait Point {
    fn coord(&self, var: usize) -> Option<i64>;
}

/// A decision with no shape variables.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Anywhere;

impl Point for Anywhere {
    fn coord(&self, var: usize) -> Option<i64> {
        unreachable!("a decision with no variables asked for variable {var}")
    }
}

/// A decision's Rust type. `#[derive(Decision)]` writes this.
pub trait Decision: Sized + Clone + Send + Sync + 'static {
    /// The shape a rung is chosen at.
    type At: Point;

    fn spec() -> &'static Spec;

    /// The typed value of a rung the registry has already checked.
    fn build(pick: &Pick) -> Self;

    /// Whether this value can serve `at`; only a partial alternative may
    /// say no.
    fn fits(&self, _at: &Self::At) -> bool {
        true
    }
}

/// A typed ladder: first rung whose region holds the point and whose value
/// fits it wins.
#[derive(Clone, Debug)]
pub struct Ladder<T> {
    rungs: Vec<(Region, T)>,
    /// Per rung, whether its value is a partial alternative: one that may
    /// not fit a shape its region holds, which then falls to the rungs
    /// after it.
    partial: Vec<bool>,
    /// Raised when the ladder is evaluated at a point (`Profile::read`).
    read: Arc<AtomicBool>,
}

impl<T: Decision> Ladder<T> {
    #[must_use]
    pub fn of(rungs: &[Rung]) -> Ladder<T> {
        Ladder {
            rungs: rungs
                .iter()
                .map(|r| (r.region.clone(), T::build(&r.pick)))
                .collect(),
            partial: rungs
                .iter()
                .map(|r| match (&r.pick, T::spec().shape) {
                    (Pick::Alternative { index, .. }, crate::spec::Shape::Choice(alts)) => {
                        alts[*index].partial
                    }
                    _ => false,
                })
                .collect(),
            read: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn read_flag(&self) -> Arc<AtomicBool> {
        self.read.clone()
    }

    fn mark(&self) {
        if !self.read.load(Ordering::Relaxed) {
            self.read.store(true, Ordering::Relaxed);
        }
    }

    /// The value for `at`, or `None` where nothing holds: a flag that is off.
    /// A checked choice or constant ladder always answers.
    #[must_use]
    pub fn at(&self, at: &T::At) -> Option<&T> {
        self.candidates(at).next()
    }

    /// Every rung whose region holds `at` and whose value fits it, in
    /// order: for code that tries each until one can run. Where `at` does
    /// not know a variable, a rung holds only if it holds whatever the
    /// value, so a `when` that compares it does not.
    pub fn candidates<'a>(&'a self, at: &T::At) -> impl Iterator<Item = &'a T> {
        self.mark();
        let vars = T::spec().vars;
        // Where a variable is unknown, every point `at` may be.
        let point = (0..vars.len())
            .map(|i| at.coord(i))
            .collect::<Option<Vec<i64>>>()
            .ok_or_else(|| {
                let space = space(vars);
                let cell: Vec<Set> = space
                    .iter()
                    .enumerate()
                    .map(|(i, s)| at.coord(i).map_or_else(|| s.clone(), Set::point))
                    .collect();
                (Region::full(&cell), space)
            });
        self.rungs
            .iter()
            .filter(move |(region, value)| {
                let holds = match &point {
                    Ok(coords) => region.contains(coords),
                    Err((cell, space)) => cell.is_subset(region, space),
                };
                holds && value.fits(at)
            })
            .map(|(_, value)| value)
    }

    /// Every rung whose region meets `cell`, a set per declared variable,
    /// in order and before its value is asked whether it fits: the rungs
    /// some shape of the cell can reach, for code that must know every
    /// value a range of shapes picks before any of them is stated.
    pub fn meeting<'a>(&'a self, cell: &[Set]) -> impl Iterator<Item = &'a T> {
        self.mark();
        let cell = Region::full(cell);
        self.rungs
            .iter()
            .filter(move |(region, _)| !region.intersect(&cell).is_empty())
            .map(|(_, value)| value)
    }

    /// Every rung some shape of `cell` picks, in order: one whose region
    /// meets what the rungs before it leave of the cell. A rung on a
    /// partial alternative leaves its region to the rungs after it, since
    /// it may not fit a shape there. For code that must name every value a
    /// range of shapes picks, and none it never picks.
    pub fn reached(&self, cell: &[Set]) -> Vec<&T> {
        self.mark();
        let space = space(T::spec().vars);
        let mut left = Region::full(cell);
        let mut reached = Vec::new();
        for ((region, value), partial) in self.rungs.iter().zip(&self.partial) {
            if left.is_empty() {
                break;
            }
            if region.intersect(&left).is_empty() {
                continue;
            }
            reached.push(value);
            if !partial {
                left = left.minus(region, &space);
            }
        }
        reached
    }

    /// Every value the ladder can pick, for sizing an envelope.
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.rungs.iter().map(|(_, v)| v)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.rungs.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rungs.is_empty()
    }
}
