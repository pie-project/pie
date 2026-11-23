//! The choice between the kernels that can serve a decode product, by their
//! times measured in place (`[rates] product_ns`): no threshold stands
//! where the device can be asked. Shared by every backend that measures its
//! products in their bodies; a backend names its kernels by codes
//! (`[rates] product_ns`'s `kernel`) and its candidates for a product.
//!
//! A product takes its stated kernel (the first candidate) unless another
//! candidate is measured faster than it by more than the two measurements'
//! spread: equal within that noise, or unmeasured, the stated kernel serves.
//! The choice is a bank's at its rows and K, whatever the product does
//! around it: a plain projection and its fused epilogue take the same
//! kernel, so they land the same bits; the time is the kind a body serves
//! (the first a load met). An unmeasured shape is noted with the weight it
//! met and its candidates ([`Routes::wanted`]), so a load can measure
//! exactly the shapes its bodies hold, over the weights they read, save
//! them, and arm again.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;

/// What a product does around its weight: the product's kind in
/// `[rates] product_ns`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    /// `y = act · wᵀ`, landed bf16.
    Plain = 0,
    /// A packed `[gate | up]` bank, its gated activation landed.
    Gated = 1,
    /// Rows normed (a residual add and RMS norm) where the kernel stages them.
    Normed = 2,
    /// [`Kind::Normed`] over a packed `[gate | up]` bank.
    NormedGated = 3,
    /// `y = act · wᵀ`, landed f32 (unrounded).
    PlainF32 = 4,
}

impl Kind {
    pub const ALL: [Kind; 5] = [Kind::Plain, Kind::Gated, Kind::Normed, Kind::NormedGated, Kind::PlainF32];
}

/// A product's shape as the table keys it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Shape {
    pub kind: Kind,
    /// Bank rows: `2n` for a packed `[gate | up]` bank.
    pub rows: u32,
    pub k: u32,
    pub m: u32,
}

/// A shape no measurement answered: the weights it met, in the order it met
/// them (a body's layers), and the kernel codes that can serve it, the
/// stated one first: what a load measures it over and under.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Wanted {
    pub weights: Vec<u64>,
    pub candidates: Vec<u32>,
}

/// A device's measured product times and the shapes they lacked.
#[derive(Debug, Default)]
pub struct Routes {
    times: HashMap<((u32, u32, u32), u32), (Kind, f64, f64)>,
    wanted: Mutex<BTreeMap<Shape, Wanted>>,
    /// Every shape answered by one kernel: a test's table ([`Routes::all`]).
    all: Option<u32>,
    /// Banks answered by one kernel whatever is measured: a probe's table
    /// while it times a body under each choice ([`Routes::forcing`]).
    forced: HashMap<(u32, u32, u32), u32>,
}

impl Routes {
    /// The table `[rates] product_ns` states (entries of a kind this crate
    /// does not know are not read).
    #[must_use]
    pub fn of(times: &[crate::fire::ProductTime]) -> Self {
        let mut table = HashMap::new();
        for t in times {
            if let Some(kind) = Kind::ALL.into_iter().find(|k| *k as u32 == t.kind) {
                table.entry(((t.rows, t.k, t.m), t.kernel)).or_insert((kind, t.ns, t.spread));
            }
        }
        Self { times: table, ..Self::default() }
    }

    /// The table that answers `kernel` for every shape: what a probe (or a
    /// test) runs a product under to time (or check) it on that kernel,
    /// through the path a body takes.
    #[must_use]
    pub fn all(kernel: u32) -> Self {
        Self { all: Some(kernel), ..Self::default() }
    }

    /// The kernel for `shape` among `candidates` (the stated kernel first):
    /// the candidate measured fastest where it beats the stated kernel by
    /// more than either measurement's spread, else the stated kernel. Where
    /// a candidate is unmeasured, `weight` is noted for [`Routes::wanted`]
    /// and the stated kernel serves. A forced or test table answers its
    /// kernel where it is a candidate.
    pub fn pick(&self, shape: Shape, weight: u64, candidates: &[u32]) -> u32 {
        let Some(&stated) = candidates.first() else {
            return 0;
        };
        if let Some(kernel) = self.all.filter(|k| candidates.contains(k)) {
            return kernel;
        }
        if let Some(&kernel) = self
            .forced
            .get(&(shape.rows, shape.k, shape.m))
            .filter(|k| candidates.contains(k))
        {
            return kernel;
        }
        let time = |kernel| self.times.get(&((shape.rows, shape.k, shape.m), kernel)).copied();
        let measured: Option<Vec<(u32, f64, f64)>> = candidates
            .iter()
            .map(|&kernel| time(kernel).map(|(_, ns, spread)| (kernel, ns, spread)))
            .collect();
        let Some(measured) = measured else {
            let mut wanted = self.wanted.lock().expect("routes lock poisoned");
            // One note a bank shape, under the kind first met.
            let at = wanted
                .keys()
                .find(|s| (s.rows, s.k, s.m) == (shape.rows, shape.k, shape.m))
                .copied()
                .unwrap_or(shape);
            let noted = wanted.entry(at).or_default();
            if !noted.weights.contains(&weight) {
                noted.weights.push(weight);
            }
            if noted.candidates.is_empty() {
                noted.candidates = candidates.to_vec();
            }
            return stated;
        };
        let (_, stated_ns, stated_spread) = measured[0];
        measured
            .iter()
            .skip(1)
            .filter(|&&(_, ns, spread)| stated_ns - ns > spread.max(stated_spread))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(stated, |&(kernel, ..)| kernel)
    }

    /// A fingerprint of what every [`Routes::pick`] answers by: each
    /// measured time (its shape, kernel, kind, time and spread, in a fixed
    /// order), the one kernel a test's table answers and every forced bank.
    /// Two tables with the same fingerprint pick alike.
    #[must_use]
    pub fn fingerprint(&self) -> u32 {
        let mut times: Vec<_> = self
            .times
            .iter()
            .map(|(&(shape, kernel), &(kind, ns, spread))| {
                (shape, kernel, kind as u32, ns.to_bits(), spread.to_bits())
            })
            .collect();
        times.sort_unstable();
        let mut forced: Vec<_> = self.forced.iter().map(|(&shape, &k)| (shape, k)).collect();
        forced.sort_unstable();
        crate::fire::fingerprint(format!("{times:?}|{:?}|{forced:?}", self.all).as_bytes())
    }

    /// The shapes [`Routes::pick`] met unmeasured since the last call, with
    /// their weights and candidates; the notes are cleared.
    pub fn wanted(&self) -> BTreeMap<Shape, Wanted> {
        std::mem::take(&mut *self.wanted.lock().expect("routes lock poisoned"))
    }

    /// The table with `measured` laid over it.
    #[must_use]
    pub fn with(&self, measured: &[(Shape, u32, f64, f64)]) -> Self {
        let mut times = self.times.clone();
        for &(shape, kernel, ns, spread) in measured {
            times.insert(((shape.rows, shape.k, shape.m), kernel), (shape.kind, ns, spread));
        }
        Self { times, ..Self::default() }
    }

    /// The table with `shape`'s bank answered by `kernel`, whatever is
    /// measured: what a probe arms a body under to time that choice.
    #[must_use]
    pub fn forcing(&self, shape: Shape, kernel: u32) -> Self {
        let mut forced = self.forced.clone();
        forced.insert((shape.rows, shape.k, shape.m), kernel);
        Self { times: self.times.clone(), forced, ..Self::default() }
    }
}

/// `measured` as `[rates] product_ns` entries.
#[must_use]
pub fn product_times(measured: &[(Shape, u32, f64, f64)]) -> Vec<crate::fire::ProductTime> {
    measured
        .iter()
        .map(|&(s, kernel, ns, spread)| crate::fire::ProductTime {
            kind: s.kind as u32,
            rows: s.rows,
            k: s.k,
            m: s.m,
            kernel,
            ns,
            spread,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATE_UP: Shape = Shape { kind: Kind::NormedGated, rows: 6144, k: 1024, m: 4 };
    const STATED: u32 = 0;
    const OTHER: u32 = 1;
    const THIRD: u32 = 0xdead_beef;

    #[test]
    fn another_kernel_serves_only_where_it_is_faster_past_the_noise() {
        let routes = |other: f64, stated: f64| {
            Routes::default().with(&[(GATE_UP, OTHER, other, 100.0), (GATE_UP, STATED, stated, 60.0)])
        };
        assert_eq!(routes(9000.0, 9500.0).pick(GATE_UP, 1, &[STATED, OTHER]), OTHER);
        // Within the larger spread: the stated kernel, by rule.
        assert_eq!(routes(9000.0, 9080.0).pick(GATE_UP, 1, &[STATED, OTHER]), STATED);
        assert_eq!(routes(9500.0, 9000.0).pick(GATE_UP, 1, &[STATED, OTHER]), STATED);
    }

    #[test]
    fn of_several_faster_kernels_the_fastest_serves() {
        let routes = Routes::default().with(&[
            (GATE_UP, STATED, 9500.0, 10.0),
            (GATE_UP, OTHER, 9000.0, 10.0),
            (GATE_UP, THIRD, 8800.0, 10.0),
        ]);
        assert_eq!(routes.pick(GATE_UP, 1, &[STATED, OTHER, THIRD]), THIRD);
    }

    #[test]
    fn a_bank_takes_one_kernel_whatever_the_product_does_around_it() {
        let routes =
            Routes::default().with(&[(GATE_UP, OTHER, 9000.0, 50.0), (GATE_UP, STATED, 9500.0, 50.0)]);
        let plain = Shape { kind: Kind::Plain, ..GATE_UP };
        assert_eq!(routes.pick(plain, 1, &[STATED, OTHER]), OTHER);
    }

    #[test]
    fn an_unmeasured_shape_takes_the_stated_kernel_and_is_noted_with_its_weights() {
        let routes = Routes::default();
        for weight in [0x1000, 0x2000, 0x1000] {
            assert_eq!(routes.pick(GATE_UP, weight, &[STATED, OTHER]), STATED);
        }
        let wanted = routes.wanted();
        assert_eq!(wanted[&GATE_UP].weights, [0x1000, 0x2000]);
        assert_eq!(wanted[&GATE_UP].candidates, [STATED, OTHER]);
        assert!(routes.wanted().is_empty(), "the notes are taken once");
    }

    #[test]
    fn a_forced_bank_answers_its_kernel() {
        let routes = Routes::default().forcing(GATE_UP, OTHER);
        assert_eq!(routes.pick(GATE_UP, 1, &[STATED, OTHER]), OTHER);
    }
}
