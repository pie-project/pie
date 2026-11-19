//! The dense matmul's kernel at each rung: the vector kernels or the tiled
//! ones, whichever the estimate (`engine::roofline`) weighs faster over the
//! model's dense planes at the rung's rows, from the device's measured
//! rates (the vector kernels at `few_row_tflops`, the tiled ones on the
//! `gemm_tflops_by_rows` curve, each plane's rows padded to its tile). A
//! matmul runs its rung's kernel whatever rows it lands, so a rung's
//! kernels are one set: a fire of fewer rows than its rung pays the rung's
//! kernel.

use engine::load::Counts;
use engine::roofline::Constants;
use kernels_slang::Target;
use kernels_slang::linear::gemm::{self, Tactic};

/// Each rung's rows and the tactic its dense matmuls run, ascending.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dense {
    rungs: Vec<(u32, Tactic)>,
}

impl Dense {
    /// The tactic a matmul of `rows` runs: its rung's, the narrowest rung
    /// that holds its rows (the widest past them all).
    #[must_use]
    pub fn at(&self, rows: u32) -> Tactic {
        self.rungs
            .iter()
            .find(|&&(rung, _)| rung >= rows)
            .or(self.rungs.last())
            .map(|&(_, tactic)| tactic)
            .expect("a load has one rung at least")
    }
}

/// One rung's dense matmuls weighed both ways by the estimate, in ns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Weighed {
    pub rows: u32,
    pub vector_ns: f64,
    pub tiled_ns: f64,
}

impl Weighed {
    /// The vector kernels weigh strictly less.
    #[must_use]
    pub fn vector(&self) -> bool {
        self.vector_ns < self.tiled_ns
    }
}

impl std::fmt::Display for Weighed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kernel, ns, other) = if self.vector() {
            ("gemv", self.vector_ns, self.tiled_ns)
        } else {
            ("tiled", self.tiled_ns, self.vector_ns)
        };
        write!(
            f,
            "rung {}: {kernel} by the estimate ({:.1} µs vs {:.1} µs)",
            self.rows,
            ns / 1e3,
            other / 1e3
        )
    }
}

/// The tactic of each of `rungs` (ascending rows) on `target`: `tiled` at
/// the rung's rows where the estimate over `rates` weighs the model's
/// dense planes (`counts`: every dense shape and the output head, each a
/// dispatch reading its plane) faster on it, else that tactic on the vector
/// kernels; and each rung's weighing. The `[rates]` key it lacks where the
/// profile states none of one it reads.
pub fn choose(
    target: Target,
    rungs: &[u32],
    tiled: impl Fn(u32) -> Tactic,
    rates: &profile::fire::Rates,
    counts: &Counts,
) -> Result<(Dense, Vec<Weighed>), &'static str> {
    let c = Constants::of(rates)?;
    let few = c.few_row_rate()?;
    // `(n, k, planes, bytes a plane)` of every dense plane a row reads.
    let mut planes: Vec<(u32, u32, u32, f64)> = counts
        .gemm_shapes
        .iter()
        .filter(|s| !s.routed && s.planes > 0)
        .map(|s| (s.n, s.k, s.planes, s.bytes as f64 / f64::from(s.planes)))
        .collect();
    if counts.readout_flops > 0 {
        let depth = counts.readout_depth.max(1);
        let vocab = u32::try_from(counts.readout_flops / (2 * u64::from(depth))).unwrap_or(u32::MAX);
        planes.push((vocab, depth, 1, counts.head_bytes as f64));
    }
    let mut chosen = Vec::with_capacity(rungs.len());
    let mut weighed = Vec::with_capacity(rungs.len());
    for &rows in rungs {
        let tactic = tiled(rows);
        let (mut vector_ns, mut tiled_ns) = (0.0, 0.0);
        for &(n, k, count, bytes) in &planes {
            let flops = |rows: u32| 2.0 * f64::from(rows) * f64::from(n) * f64::from(k);
            let tile = gemm::tile_for(target, tactic, rows, n, k).row_tile().max(1);
            let padded = rows.div_ceil(tile) * tile;
            let count = f64::from(count);
            tiled_ns += count * c.dispatch_ns(flops(padded), c.gemm_rate(rows, n, k), bytes);
            vector_ns += count * c.dispatch_ns(flops(rows), few, bytes);
        }
        let w = Weighed {
            rows,
            vector_ns,
            tiled_ns,
        };
        chosen.push((
            rows,
            Tactic {
                vector: w.vector(),
                ..tactic
            },
        ));
        weighed.push(w);
    }
    Ok((Dense { rungs: chosen }, weighed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels_slang::Tier;

    fn counts() -> Counts {
        Counts {
            weight_bytes: 1 << 30,
            expert_bytes: 0,
            experts: None,
            // Thirty-two 4096-square bf16 planes.
            gemm_shapes: vec![engine::load::GemmShape {
                planes: 32,
                ..engine::load::GemmShape::dense(4096, 4096, 32 << 25, 32 << 25)
            }],
            head_bytes: 0,
            attn_flops_per_row_key: 0,
            kv_bytes_per_key: 0,
            attn_head_dim: 128,
            readout_flops: 0,
            readout_depth: 4096,
            state_elements_per_row: 0,
            elementwise_ops: vec![],
            attn_dispatches: 0,
            scan_dispatches: 0,
            width: 4096,
        }
    }

    fn rates(few: Option<f64>) -> profile::fire::Rates {
        profile::fire::Rates {
            launch_ns: Some(3000),
            gemm_tflops_by_rows: Some(vec![(4096, vec![(256, 10.0), (2048, 30.0)])]),
            hbm_gbps: Some(650.0),
            elementwise_gbps: Some(700.0),
            few_row_tflops: few,
            ..Default::default()
        }
    }

    fn tiled(_: u32) -> Tactic {
        Tactic {
            tier: Tier::Coopmat,
            vector: false,
        }
    }

    /// A rung takes the vector kernels where the tiled one's tile pads its
    /// rows past what the vector rate costs, and the tiled ones where the
    /// curve outruns the vector rate; a matmul runs its rung's.
    #[test]
    fn each_rung_takes_the_kernel_the_estimate_weighs_faster() {
        let rungs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
        let (dense, weighed) =
            choose(Target::Spirv, &rungs, tiled, &rates(Some(5.0)), &counts()).unwrap();
        let vector: Vec<bool> = weighed.iter().map(Weighed::vector).collect();
        // Below the 64-row tile the vector kernels read the plane once at
        // its bandwidth; from 256 rows the curve outruns 5 TFLOP/s.
        assert!(vector[..4].iter().all(|&v| v), "{weighed:?}");
        assert!(vector[8..].iter().all(|&v| !v), "{weighed:?}");
        for (i, &rows) in rungs.iter().enumerate() {
            assert_eq!(dense.at(rows).vector, vector[i]);
            assert_eq!(dense.at(rows.div_ceil(2) + 1).vector, vector[i], "{rows}");
        }
        assert_eq!(dense.at(3).vector, vector[2]);
        assert_eq!(dense.at(5000).vector, vector[rungs.len() - 1]);
        assert!(weighed[2].to_string().starts_with("rung 4: gemv by the estimate ("));
    }

    /// A profile stating no vector rate is refused by its key.
    #[test]
    fn no_vector_rate_is_refused_by_its_key() {
        let refused = choose(Target::Spirv, &[1, 2], tiled, &rates(None), &counts());
        assert_eq!(refused.unwrap_err(), "few_row_tflops");
    }
}
