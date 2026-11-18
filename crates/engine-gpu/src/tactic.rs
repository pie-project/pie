//! Which kernel a dense matmul of a fire's rows takes: the vector kernel or
//! the tile the profile's path names, whichever the device's roofline
//! prices cheaper at the fire's shape. No row count decides it.
//!
//! - The vector kernel reads its weight plane once a block of rows
//!   (`kernels_slang::linear::gemm::passes`), at `[rates] hbm_gbps`, and
//!   computes the fire's own rows at `[rates] few_row_tflops` (timed on it).
//! - A tile reads the plane once a tile of rows and computes its rows padded
//!   to its tile, at the matmul curve's rate there (`[rates]
//!   gemm_tflops_by_rows`, `engine::roofline::Constants::gemm_rate`).
//!
//! Each costs its launch and the longer of its reads and its compute; the
//! cheaper is taken. A recorded fire's rows are fixed, so a recording
//! replays the kernel its rows chose. A rate the choice reads that the
//! profile leaves unstated is refused by its key: no row count stands in.

use engine::roofline::Constants;
use kernels_slang::Target;
use kernels_slang::linear::gemm::{Kernel, Tactic, passes, tile_for};

/// Each kernel's cost at a shape, in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Costs {
    pub vector: f64,
    pub tiled: f64,
}

/// The vector kernel's and `tiled`'s (the profile's path, its `vector`
/// unset) costs for a bf16 matmul of `m x n x k` on `target`, as `rates`
/// price them; the key of the first rate they need that is unstated.
pub fn costs(
    target: Target,
    tiled: Tactic,
    m: u32,
    n: u32,
    k: u32,
    rates: &Constants,
) -> Result<Costs, &'static str> {
    let plane = 2.0 * f64::from(n) * f64::from(k);
    let flops = |rows: u32| 2.0 * f64::from(rows) * f64::from(n) * f64::from(k);
    let vector_kind = tile_for(
        target,
        Tactic {
            vector: true,
            ..tiled
        },
        m,
        n,
        k,
    );
    let vector = rates.dispatch_ns(
        flops(m),
        rates.few_row_rate()?,
        f64::from(passes(vector_kind, m)) * plane,
    );
    let tile_kind = tile_for(
        target,
        Tactic {
            vector: false,
            ..tiled
        },
        m,
        n,
        k,
    );
    let tile = tile_kind.row_tile().max(1);
    let padded = m.div_ceil(tile) * tile;
    let tiled = rates.dispatch_ns(
        flops(padded),
        rates.gemm_rate(m, n, k),
        f64::from(passes(tile_kind, m)) * plane,
    );
    Ok(Costs { vector, tiled })
}

/// The tactic a bf16 matmul of `m x n x k` on `target` takes: `tiled` (the
/// profile's path at those rows) with the vector kernel where [`costs`]
/// prices it cheaper; the key of a rate the choice needs and the profile
/// leaves unstated. The one rule every caller reads (a dispatch, the
/// estimate's row tile, a load recording its rungs).
pub fn gemm_at(
    target: Target,
    tiled: Tactic,
    m: u32,
    n: u32,
    k: u32,
    rates: &Constants,
) -> Result<Tactic, &'static str> {
    let Costs {
        vector,
        tiled: tile,
    } = costs(target, tiled, m, n, k, rates)?;
    Ok(Tactic {
        vector: vector <= tile,
        ..tiled
    })
}

/// Whether `kernel` is a vector kernel.
#[must_use]
pub fn is_vector(kernel: Kernel) -> bool {
    matches!(kernel, Kernel::GemvV | Kernel::Gemv | Kernel::PairsGemv)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An RTX PRO 4500's Vulkan rates: the vector kernel's few-row rate,
    /// the matmul curve at 4096, the copy's bandwidth.
    fn rates() -> Constants {
        Constants::of(&profile::fire::Rates {
            launch_ns: Some(2188),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 18.6), (8192, 25.4)])]),
            attn_tflops_by_keys: Some(vec![(8192, 6.9)]),
            hbm_gbps: Some(790.0),
            elementwise_gbps: Some(826.0),
            scan_gelems: Some(137.0),
            few_row_tflops: Some(4.5),
            workgroups_per_core: Some(4.0),
        })
        .unwrap()
    }

    fn coopmat() -> Tactic {
        Tactic {
            tier: kernels_slang::Tier::Coopmat,
            vector: false,
        }
    }

    /// The vector kernel reads the plane once a block of 4 rows; the
    /// 64-row cooperative tile computes 64 rows at the curve's rate, 2.7x
    /// the plane's read at these rates: the vector kernel through 8 rows (2
    /// reads), the tile from 12 (3 reads past its 2.7), wherever the two
    /// estimates cross, not at a row count.
    #[test]
    fn the_vector_kernel_serves_until_its_reads_pass_the_tiles_compute() {
        let r = rates();
        for m in [1, 3, 4, 8] {
            let t = gemm_at(Target::Spirv, coopmat(), m, 4096, 4096, &r).unwrap();
            assert!(t.vector, "{m} rows take the vector kernel");
        }
        for m in [12, 16, 64, 256] {
            let t = gemm_at(Target::Spirv, coopmat(), m, 4096, 4096, &r).unwrap();
            assert!(!t.vector, "{m} rows take the tile");
        }
        // At the crossing the two estimates cross: below it the vector
        // kernel is the cheaper, past it the tile.
        let c = |m| costs(Target::Spirv, coopmat(), m, 4096, 4096, &r).unwrap();
        assert!(c(8).vector < c(8).tiled && c(12).vector > c(12).tiled);
    }

    /// Without the vector kernel's rate the choice is refused by its key.
    #[test]
    fn an_unpriced_choice_is_refused_by_the_key_it_lacks() {
        let r = Constants {
            few_row_flops_per_ns: None,
            ..rates()
        };
        assert_eq!(
            gemm_at(Target::Spirv, coopmat(), 4, 4096, 4096, &r),
            Err("few_row_tflops")
        );
    }
}
