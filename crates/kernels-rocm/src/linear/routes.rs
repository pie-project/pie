//! The choice between the kernels that can serve a decode product, by their
//! times measured in place (`[rates] product_ns`): the shared table
//! ([`profile::routes`]), with this backend's two candidates.
//!
//! A product the turned kernel and the skinny kernel both cover takes the
//! turned kernel only where the table measures it faster by more than the
//! two measurements' spread; equal within that noise, or unmeasured, it
//! takes the skinny kernel, by stated rule.

pub use profile::routes::{Kind, Routes, Shape, Wanted, product_times};

/// A kernel that can serve a product: its code in `[rates] product_ns`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kernel {
    /// The skinny kernel (`skinny_bf16`): a weight row a wavefront.
    Skinny = 0,
    /// The turned kernel (`skinnyt_bf16`): 16-row tiles through the matrix core.
    Turned = 1,
}

impl Kernel {
    /// The candidates of a product both cover, the stated one first.
    pub const CANDIDATES: [u32; 2] = [Kernel::Skinny as u32, Kernel::Turned as u32];

    /// The kernel coded `code`, where it is one of this backend's.
    #[must_use]
    pub fn of(code: u32) -> Option<Kernel> {
        [Kernel::Skinny, Kernel::Turned].into_iter().find(|k| *k as u32 == code)
    }
}

/// The kernel `routes` picks for `shape` (its weight `weight`), between the
/// skinny kernel (stated) and the turned kernel.
#[must_use]
pub fn pick(routes: &Routes, shape: Shape, weight: u64) -> Kernel {
    Kernel::of(routes.pick(shape, weight, &Kernel::CANDIDATES)).unwrap_or(Kernel::Skinny)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GATE_UP: Shape = Shape { kind: Kind::NormedGated, rows: 6144, k: 1024, m: 4 };

    #[test]
    fn the_turned_kernel_serves_only_where_it_is_faster_past_the_noise() {
        let routes = |turned: f64, skinny: f64| {
            Routes::default().with(&[
                (GATE_UP, Kernel::Turned as u32, turned, 100.0),
                (GATE_UP, Kernel::Skinny as u32, skinny, 60.0),
            ])
        };
        assert_eq!(pick(&routes(9000.0, 9500.0), GATE_UP, 1), Kernel::Turned);
        assert_eq!(pick(&routes(9000.0, 9080.0), GATE_UP, 1), Kernel::Skinny);
        assert_eq!(pick(&routes(9500.0, 9000.0), GATE_UP, 1), Kernel::Skinny);
    }

    #[test]
    fn an_unmeasured_shape_takes_the_skinny_kernel() {
        assert_eq!(pick(&Routes::default(), GATE_UP, 1), Kernel::Skinny);
    }
}
