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
    pub const ALL: [Kind; 5] = [
        Kind::Plain,
        Kind::Gated,
        Kind::Normed,
        Kind::NormedGated,
        Kind::PlainF32,
    ];
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
    /// The entries of `times` that are estimates: figures stored under a
    /// stale key (`profile::fire::Rates::stale`), served as they stand
    /// ([`Routes::estimating`]).
    estimated: std::collections::HashSet<((u32, u32, u32), u32)>,
    /// The shapes [`Routes::pick`] answered by an estimate since the last
    /// call of [`Routes::behind`].
    behind: Mutex<BTreeMap<Shape, Wanted>>,
    /// Each bank's kernel as [`Routes::pick`] last answered it, shared with
    /// the tables a load's measuring lays over this one ([`Routes::with`]),
    /// not with a probe's ([`Routes::forcing`]): what the load's bodies
    /// were armed under ([`Routes::fingerprint`]).
    picked: std::sync::Arc<Mutex<BTreeMap<(u32, u32, u32), u32>>>,
}

impl Routes {
    /// The table `[rates] product_ns` states (entries of a kind this crate
    /// does not know are not read).
    #[must_use]
    pub fn of(times: &[crate::fire::ProductTime]) -> Self {
        let mut table = HashMap::new();
        for t in times {
            if let Some(kind) = Kind::ALL.into_iter().find(|k| *k as u32 == t.kind) {
                table
                    .entry(((t.rows, t.k, t.m), t.kernel))
                    .or_insert((kind, t.ns, t.spread));
            }
        }
        Self {
            times: table,
            ..Self::default()
        }
    }

    /// The table `measured` states, each entry it lacks answered by
    /// `estimates` (figures stored under a stale key): served as they
    /// stand, and noted where a pick reads one ([`Routes::behind`]) so the
    /// load says which banks serve by them.
    #[must_use]
    pub fn estimating(
        measured: &[crate::fire::ProductTime],
        estimates: &[crate::fire::ProductTime],
    ) -> Self {
        let mut routes = Self::of(measured);
        for (at, time) in Self::of(estimates).times {
            if let std::collections::hash_map::Entry::Vacant(entry) = routes.times.entry(at) {
                entry.insert(time);
                routes.estimated.insert(at);
            }
        }
        routes
    }

    /// How many entries of the table are estimates.
    #[must_use]
    pub fn estimates(&self) -> usize {
        self.estimated.len()
    }

    /// The table that answers `kernel` for every shape: what a probe (or a
    /// test) runs a product under to time (or check) it on that kernel,
    /// through the path a body takes.
    #[must_use]
    pub fn all(kernel: u32) -> Self {
        Self {
            all: Some(kernel),
            ..Self::default()
        }
    }

    /// The kernel for `shape` among `candidates` (the stated kernel first):
    /// the candidate measured fastest where it beats the stated kernel by
    /// more than either measurement's spread, else the stated kernel. Where
    /// a candidate is unmeasured, `weight` is noted for [`Routes::wanted`]
    /// and the stated kernel serves; unless the bank holds an estimate
    /// ([`Routes::estimating`]): then the candidates stated serve by the
    /// same rule, and `weight` is noted for [`Routes::behind`]. A forced or
    /// test table answers its
    /// kernel where it is a candidate.
    pub fn pick(&self, shape: Shape, weight: u64, candidates: &[u32]) -> u32 {
        let kernel = self.choose(shape, weight, candidates);
        self.picked
            .lock()
            .expect("routes lock poisoned")
            .insert((shape.rows, shape.k, shape.m), kernel);
        kernel
    }

    fn choose(&self, shape: Shape, weight: u64, candidates: &[u32]) -> u32 {
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
        let bank = (shape.rows, shape.k, shape.m);
        let time = |kernel| self.times.get(&(bank, kernel)).copied();
        let known: Vec<(u32, f64, f64)> = candidates
            .iter()
            .filter_map(|&kernel| time(kernel).map(|(_, ns, spread)| (kernel, ns, spread)))
            .collect();
        let estimated = candidates
            .iter()
            .any(|&kernel| self.estimated.contains(&(bank, kernel)));
        if known.len() < candidates.len() && !estimated {
            note(&self.wanted, shape, weight, candidates);
            return stated;
        }
        // Every candidate measured; or the bank has estimates (figures
        // stored under a stale key), which serve as they stand, a candidate
        // none states (its code changed) not taken, and the bank is noted
        // as served by stale figures.
        if estimated {
            note(&self.behind, shape, weight, candidates);
        }
        let Some(&(_, stated_ns, stated_spread)) =
            known.iter().find(|&&(kernel, ..)| kernel == stated)
        else {
            return stated;
        };
        known
            .iter()
            .filter(|&&(kernel, ns, spread)| {
                kernel != stated && stated_ns - ns > spread.max(stated_spread)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(stated, |&(kernel, ..)| kernel)
    }

    /// A fingerprint of the kernel each bank was picked (the last
    /// [`Routes::pick`] of each, over this table and the tables it was laid
    /// over by measuring), with the one kernel a test's table answers: two
    /// loads that armed their bodies under the same picks share it, whatever
    /// figures (measured or estimated) chose them.
    #[must_use]
    pub fn fingerprint(&self) -> u32 {
        let picked = self.picked.lock().expect("routes lock poisoned").clone();
        crate::fire::fingerprint(format!("{picked:?}|{:?}", self.all).as_bytes())
    }

    /// The shapes [`Routes::pick`] met unmeasured since the last call, with
    /// their weights and candidates; the notes are cleared.
    pub fn wanted(&self) -> BTreeMap<Shape, Wanted> {
        std::mem::take(&mut *self.wanted.lock().expect("routes lock poisoned"))
    }

    /// The shapes [`Routes::pick`] answered by an estimate since the last
    /// call, with their weights and candidates: the banks a load serves by
    /// stale figures. The notes are cleared.
    pub fn behind(&self) -> BTreeMap<Shape, Wanted> {
        std::mem::take(&mut *self.behind.lock().expect("routes lock poisoned"))
    }

    /// The table with `measured` laid over it (each measured entry no
    /// longer an estimate).
    #[must_use]
    pub fn with(&self, measured: &[(Shape, u32, f64, f64)]) -> Self {
        let mut times = self.times.clone();
        let mut estimated = self.estimated.clone();
        for &(shape, kernel, ns, spread) in measured {
            let at = ((shape.rows, shape.k, shape.m), kernel);
            times.insert(at, (shape.kind, ns, spread));
            estimated.remove(&at);
        }
        Self {
            times,
            estimated,
            picked: self.picked.clone(),
            ..Self::default()
        }
    }

    /// The table with `shape`'s bank answered by `kernel`, whatever is
    /// measured: what a probe arms a body under to time that choice.
    #[must_use]
    pub fn forcing(&self, shape: Shape, kernel: u32) -> Self {
        let mut forced = self.forced.clone();
        forced.insert((shape.rows, shape.k, shape.m), kernel);
        Self {
            times: self.times.clone(),
            forced,
            estimated: self.estimated.clone(),
            ..Self::default()
        }
    }
}

/// `shape` met with `weight` under `candidates`, noted in `notes` once a
/// bank shape, under the kind first met.
fn note(notes: &Mutex<BTreeMap<Shape, Wanted>>, shape: Shape, weight: u64, candidates: &[u32]) {
    let mut notes = notes.lock().expect("routes lock poisoned");
    let at = notes
        .keys()
        .find(|s| (s.rows, s.k, s.m) == (shape.rows, shape.k, shape.m))
        .copied()
        .unwrap_or(shape);
    let noted = notes.entry(at).or_default();
    if !noted.weights.contains(&weight) {
        noted.weights.push(weight);
    }
    if noted.candidates.is_empty() {
        noted.candidates = candidates.to_vec();
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

    const GATE_UP: Shape = Shape {
        kind: Kind::NormedGated,
        rows: 6144,
        k: 1024,
        m: 4,
    };
    const STATED: u32 = 0;
    const OTHER: u32 = 1;
    const THIRD: u32 = 0xdead_beef;

    #[test]
    fn another_kernel_serves_only_where_it_is_faster_past_the_noise() {
        let routes = |other: f64, stated: f64| {
            Routes::default().with(&[
                (GATE_UP, OTHER, other, 100.0),
                (GATE_UP, STATED, stated, 60.0),
            ])
        };
        assert_eq!(
            routes(9000.0, 9500.0).pick(GATE_UP, 1, &[STATED, OTHER]),
            OTHER
        );
        // Within the larger spread: the stated kernel, by rule.
        assert_eq!(
            routes(9000.0, 9080.0).pick(GATE_UP, 1, &[STATED, OTHER]),
            STATED
        );
        assert_eq!(
            routes(9500.0, 9000.0).pick(GATE_UP, 1, &[STATED, OTHER]),
            STATED
        );
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
        let routes = Routes::default().with(&[
            (GATE_UP, OTHER, 9000.0, 50.0),
            (GATE_UP, STATED, 9500.0, 50.0),
        ]);
        let plain = Shape {
            kind: Kind::Plain,
            ..GATE_UP
        };
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

    /// A shape the measured table lacks is answered by its stale figures
    /// (the same pick they made), noted as stale, not
    /// wanted before ready; once timed it is measured, no longer noted.
    #[test]
    fn a_stale_figure_answers_and_is_noted_behind() {
        let time = |kernel, ns| crate::fire::ProductTime {
            kind: Kind::NormedGated as u32,
            rows: GATE_UP.rows,
            k: GATE_UP.k,
            m: GATE_UP.m,
            kernel,
            ns,
            spread: 10.0,
        };
        let routes = Routes::estimating(&[], &[time(STATED, 9500.0), time(OTHER, 9000.0)]);
        assert_eq!(routes.pick(GATE_UP, 1, &[STATED, OTHER]), OTHER);
        assert!(routes.wanted().is_empty(), "nothing before ready");
        assert_eq!(routes.behind()[&GATE_UP].candidates, [STATED, OTHER]);
        let timed = routes.with(&[
            (GATE_UP, STATED, 9400.0, 10.0),
            (GATE_UP, OTHER, 9100.0, 10.0),
        ]);
        assert_eq!(timed.pick(GATE_UP, 1, &[STATED, OTHER]), OTHER);
        assert!(timed.behind().is_empty(), "measured now");
        // A candidate whose code changed (none states it) is not taken; the
        // bank serves by its estimates, noted behind.
        assert_eq!(routes.pick(GATE_UP, 1, &[STATED, THIRD, OTHER]), OTHER);
        assert!(routes.wanted().is_empty());
        assert_eq!(routes.behind()[&GATE_UP].candidates, [STATED, THIRD, OTHER]);
    }

    /// Two tables whose figures differ but pick alike share a fingerprint;
    /// a probe's forced picks never reach the table it was forced off.
    #[test]
    fn the_fingerprint_is_the_picks() {
        let a = Routes::default().with(&[
            (GATE_UP, OTHER, 9000.0, 10.0),
            (GATE_UP, STATED, 9500.0, 10.0),
        ]);
        let b = Routes::default().with(&[
            (GATE_UP, OTHER, 8000.0, 10.0),
            (GATE_UP, STATED, 9600.0, 10.0),
        ]);
        a.pick(GATE_UP, 1, &[STATED, OTHER]);
        b.pick(GATE_UP, 1, &[STATED, OTHER]);
        assert_eq!(a.fingerprint(), b.fingerprint());
        a.forcing(GATE_UP, STATED)
            .pick(GATE_UP, 1, &[STATED, OTHER]);
        assert_eq!(
            a.fingerprint(),
            b.fingerprint(),
            "a probe's pick is its own"
        );
    }

    #[test]
    fn a_forced_bank_answers_its_kernel() {
        let routes = Routes::default().forcing(GATE_UP, OTHER);
        assert_eq!(routes.pick(GATE_UP, 1, &[STATED, OTHER]), OTHER);
    }
}
