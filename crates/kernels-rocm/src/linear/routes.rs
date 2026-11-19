//! The choice between the kernels that can serve a decode product, by their
//! times measured in place (`[rates] product_ns`): no threshold stands
//! where the device can be asked.
//!
//! A product the turned kernel and the skinny kernel both cover takes the
//! turned kernel only where the table measures it faster by more than the
//! two measurements' spread; equal within that noise, or unmeasured, it
//! takes the skinny kernel, by stated rule. The choice is a bank's at its
//! rows and K, whatever the product does around it: a plain projection and
//! its fused GLU or folded norm take the same kernel, so they land the same
//! bits; the time is the kind a body serves (the first a load met). An unmeasured shape is noted
//! with the weight it met ([`Routes::wanted`]), so a load can measure
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

/// A kernel that can serve a product: its code in `[rates] product_ns`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kernel {
    /// The skinny kernel (`skinny_bf16`): a weight row a wavefront.
    Skinny = 0,
    /// The turned kernel (`skinnyt_bf16`): 16-row tiles through the matrix core.
    Turned = 1,
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

/// A shape no measurement answered, and the weights it met, in the order it
/// met them (a body's layers): what a load measures it over.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Wanted {
    pub weights: Vec<u64>,
}

/// A device's measured product times and the shapes they lacked.
#[derive(Debug, Default)]
pub struct Routes {
    times: HashMap<((u32, u32, u32), Kernel), (Kind, f64, f64)>,
    wanted: Mutex<BTreeMap<Shape, Wanted>>,
    /// Every shape answered by one kernel: a test's table ([`Routes::all`]).
    all: Option<Kernel>,
    /// Banks answered by one kernel whatever is measured: a probe's table
    /// while it times a body under each choice ([`Routes::forcing`]).
    forced: HashMap<(u32, u32, u32), Kernel>,
}

impl Routes {
    /// The table `[rates] product_ns` states (entries of a kind or kernel
    /// this crate does not know are not read).
    #[must_use]
    pub fn of(times: &[profile::fire::ProductTime]) -> Self {
        let kind = |c: u32| {
            [Kind::Plain, Kind::Gated, Kind::Normed, Kind::NormedGated, Kind::PlainF32]
                .into_iter()
                .find(|k| *k as u32 == c)
        };
        let kernel = |c: u32| [Kernel::Skinny, Kernel::Turned].into_iter().find(|k| *k as u32 == c);
        let mut table = HashMap::new();
        for t in times {
            if let (Some(kind), Some(kernel)) = (kind(t.kind), kernel(t.kernel)) {
                table.entry(((t.rows, t.k, t.m), kernel)).or_insert((kind, t.ns, t.spread));
            }
        }
        Self { times: table, ..Self::default() }
    }

    /// The table that answers `kernel` for every shape both kernels cover:
    /// what a probe (or a test) runs a product under to time (or check) it
    /// on that kernel, through the path a body takes.
    #[must_use]
    pub fn all(kernel: Kernel) -> Self {
        Self { all: Some(kernel), ..Self::default() }
    }

    /// The kernel for `shape` where both cover it: the turned kernel where
    /// it is measured faster than the skinny kernel by more than either
    /// measurement's spread, else the skinny kernel. Unmeasured, `weight`
    /// is noted for [`Routes::wanted`] and the skinny kernel serves.
    pub fn pick(&self, shape: Shape, weight: u64) -> Kernel {
        if let Some(kernel) = self.all {
            return kernel;
        }
        if let Some(&kernel) = self.forced.get(&(shape.rows, shape.k, shape.m)) {
            return kernel;
        }
        let time = |kernel| self.times.get(&((shape.rows, shape.k, shape.m), kernel)).copied();
        match (time(Kernel::Turned), time(Kernel::Skinny)) {
            (Some((_, turned, t_spread)), Some((_, skinny, s_spread))) => {
                if skinny - turned > t_spread.max(s_spread) {
                    Kernel::Turned
                } else {
                    Kernel::Skinny
                }
            }
            _ => {
                let mut wanted = self.wanted.lock().expect("routes lock poisoned");
                // One note a bank shape, under the kind first met.
                let at = wanted
                    .keys()
                    .find(|s| (s.rows, s.k, s.m) == (shape.rows, shape.k, shape.m))
                    .copied()
                    .unwrap_or(shape);
                let weights = &mut wanted.entry(at).or_default().weights;
                if !weights.contains(&weight) {
                    weights.push(weight);
                }
                Kernel::Skinny
            }
        }
    }

    /// The shapes [`Routes::pick`] met unmeasured since the last call, with
    /// their weights; the notes are cleared.
    pub fn wanted(&self) -> BTreeMap<Shape, Wanted> {
        std::mem::take(&mut *self.wanted.lock().expect("routes lock poisoned"))
    }

    /// The table with `measured` laid over it.
    #[must_use]
    pub fn with(&self, measured: &[(Shape, Kernel, f64, f64)]) -> Self {
        let mut times = self.times.clone();
        for &(shape, kernel, ns, spread) in measured {
            times.insert(((shape.rows, shape.k, shape.m), kernel), (shape.kind, ns, spread));
        }
        Self { times, ..Self::default() }
    }

    /// The table with `shape`'s bank answered by `kernel`, whatever is
    /// measured: what a probe arms a body under to time that choice.
    #[must_use]
    pub fn forcing(&self, shape: Shape, kernel: Kernel) -> Self {
        let mut forced = self.forced.clone();
        forced.insert((shape.rows, shape.k, shape.m), kernel);
        Self { times: self.times.clone(), forced, ..Self::default() }
    }
}

/// `measured` as `[rates] product_ns` entries.
#[must_use]
pub fn product_times(measured: &[(Shape, Kernel, f64, f64)]) -> Vec<profile::fire::ProductTime> {
    measured
        .iter()
        .map(|&(s, kernel, ns, spread)| profile::fire::ProductTime {
            kind: s.kind as u32,
            rows: s.rows,
            k: s.k,
            m: s.m,
            kernel: kernel as u32,
            ns,
            spread,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATE_UP: Shape = Shape { kind: Kind::NormedGated, rows: 6144, k: 1024, m: 4 };

    #[test]
    fn the_turned_kernel_serves_only_where_it_is_faster_past_the_noise() {
        let routes = |turned: f64, skinny: f64| {
            Routes::default().with(&[
                (GATE_UP, Kernel::Turned, turned, 100.0),
                (GATE_UP, Kernel::Skinny, skinny, 60.0),
            ])
        };
        assert_eq!(routes(9000.0, 9500.0).pick(GATE_UP, 1), Kernel::Turned);
        // Within the larger spread: the skinny kernel, by rule.
        assert_eq!(routes(9000.0, 9080.0).pick(GATE_UP, 1), Kernel::Skinny);
        assert_eq!(routes(9500.0, 9000.0).pick(GATE_UP, 1), Kernel::Skinny);
    }

    #[test]
    fn a_bank_takes_one_kernel_whatever_the_product_does_around_it() {
        let routes = Routes::default().with(&[
            (GATE_UP, Kernel::Turned, 9000.0, 50.0),
            (GATE_UP, Kernel::Skinny, 9500.0, 50.0),
        ]);
        let plain = Shape { kind: Kind::Plain, ..GATE_UP };
        assert_eq!(routes.pick(plain, 1), Kernel::Turned);
    }

    #[test]
    fn an_unmeasured_shape_takes_the_skinny_kernel_and_is_noted_with_its_weights() {
        let routes = Routes::default();
        for weight in [0x1000, 0x2000, 0x1000] {
            assert_eq!(routes.pick(GATE_UP, weight), Kernel::Skinny);
        }
        let wanted = routes.wanted();
        assert_eq!(wanted[&GATE_UP].weights, [0x1000, 0x2000]);
        assert!(routes.wanted().is_empty(), "the notes are taken once");
    }
}
