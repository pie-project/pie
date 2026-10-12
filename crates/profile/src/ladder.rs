//! Rungs, ladders, and the trait a decision's Rust type implements.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::region::Region;
use crate::spec::{Spec, Value};
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
/// variable, in declaration order. Enumerations are their value's index.
pub trait Point {
    fn coord(&self, var: usize) -> i64;
}

/// A decision with no shape variables.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Anywhere;

impl Point for Anywhere {
    fn coord(&self, var: usize) -> i64 {
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
    /// Raised the first time the ladder is evaluated at a point: whether a
    /// load ever asked this decision anything (`Profile::read`).
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
        self.mark();
        let n = T::spec().vars.len();
        self.rungs
            .iter()
            .find(|(region, value)| {
                region
                    .boxes()
                    .iter()
                    .any(|b| (0..n).all(|i| b[i].contains(at.coord(i))))
                    && value.fits(at)
            })
            .map(|(_, value)| value)
    }

    /// Every rung whose region holds `at` and whose value fits it, in
    /// order: for code that tries each until one can run (a runtime
    /// condition, such as scratch room, can still turn one down).
    pub fn candidates<'a>(&'a self, at: &'a T::At) -> impl Iterator<Item = &'a T> + 'a {
        self.mark();
        let n = T::spec().vars.len();
        self.rungs
            .iter()
            .filter(move |(region, value)| {
                region.boxes().iter().any(|b| (0..n).all(|i| b[i].contains(at.coord(i))))
                    && value.fits(at)
            })
            .map(|(_, value)| value)
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
