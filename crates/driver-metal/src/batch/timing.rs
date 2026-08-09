//! GPU-exec attribution: which dispatches a step actually spends on.
//!
//! Once a step is correct, the question is which of its ~400 dispatches
//! dominate GPU time, because that is what decides what to fuse or cut. The
//! encoder marks a timestamp boundary before every dispatch plus one after
//! the last; diffing consecutive boundaries yields each dispatch's GPU
//! time, and this module rolls that up per kind and per layer. Ported from
//! `decode_timing.cpp`.
//!
//! ## What changes
//!
//! * `attribute_step` answered a boundary/DAG mismatch with a struct whose
//!   `valid` flag the caller had to remember to read — the same
//!   did-the-validator-run shape this subsystem keeps regrowing. It is an
//!   [`Err`] carrying both counts.
//! * The ablation table read `PIE_METAL_ABLATE` inside the query function
//!   behind a `static` initialiser. [`Ablation::parse`] takes the spec as a
//!   value (testable), returns the tokens that matched nothing — the C++
//!   learned the hard way that a typo'd token "ablates NOTHING and this run
//!   will report the baseline", after a whole session of no-op readings —
//!   and [`Ablation::from_env`] is the one place the environment is read.
//!   Matching is exact per comma token via [`Kernel::from_name`], which the
//!   C++ approximated with a substring walk and boundary checks.
//! * `print_attribution` wrote to a `FILE*`; this crate denies stdout and
//!   stderr by policy, so [`StepAttribution::report`] returns the text and
//!   the caller owns where it goes.

use std::fmt::Write as _;

use crate::lowering::abi::Kernel;

/// One dispatch's identity, as the attribution needs it.
///
/// The full family `Dispatch` carries launch geometry and fusion knobs;
/// this is the slice of it the timing rollup reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DispatchInfo {
    /// The kernel kind.
    pub kind: Kernel,
    /// The dispatch's flat DAG ordinal.
    pub ordinal: u32,
    /// The model layer, or [`None`] for the layer-less head and tail.
    pub layer: Option<u32>,
}

/// One dispatch's attributed GPU time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DispatchAttribution {
    /// The dispatch's identity.
    pub dispatch: DispatchInfo,
    /// Its share of the step, in milliseconds.
    pub gpu_ms: f64,
}

/// A step's GPU time, attributed per dispatch and rolled up per kind.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepAttribution {
    /// Per-dispatch times, in DAG order.
    pub per_dispatch: Vec<DispatchAttribution>,
    /// Summed milliseconds per kernel kind.
    pub by_kind: Vec<f64>,
    /// Dispatch count per kernel kind.
    pub count_kind: Vec<u32>,
    /// The whole step's GPU time.
    pub total_gpu_ms: f64,
}

/// The boundary array does not describe the DAG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundaryMismatch {
    /// Timestamps supplied.
    pub boundaries: usize,
    /// Dispatches in the DAG (one more boundary than these is needed).
    pub dispatches: usize,
}

/// Attribute a step from resolved timestamp boundaries.
///
/// `boundary_ticks[i]` is the GPU timestamp at which the walker reached
/// dispatch `i`, with one final entry after the last dispatch;
/// `ns_per_tick` converts the GPU's tick domain (1.0 on the calibrated
/// box). A non-increasing pair — a clock wrap or reorder — attributes zero
/// rather than a negative share.
///
/// # Errors
///
/// [`BoundaryMismatch`] when the counts disagree. The C++ returned a
/// zeroed struct with `valid = false` for the caller to remember to check.
pub fn attribute_step(
    dag: &[DispatchInfo],
    boundary_ticks: &[u64],
    ns_per_tick: f64,
) -> Result<StepAttribution, BoundaryMismatch> {
    if boundary_ticks.len() != dag.len() + 1 {
        return Err(BoundaryMismatch {
            boundaries: boundary_ticks.len(),
            dispatches: dag.len(),
        });
    }
    let mut attribution = StepAttribution {
        per_dispatch: Vec::with_capacity(dag.len()),
        by_kind: vec![0.0; Kernel::COUNT],
        count_kind: vec![0; Kernel::COUNT],
        total_gpu_ms: 0.0,
    };
    for (index, dispatch) in dag.iter().enumerate() {
        let t0 = boundary_ticks[index];
        let t1 = boundary_ticks[index + 1];
        let gpu_ms = if t1 > t0 {
            (t1 - t0) as f64 * ns_per_tick / 1e6
        } else {
            0.0
        };
        attribution.per_dispatch.push(DispatchAttribution {
            dispatch: *dispatch,
            gpu_ms,
        });
        attribution.by_kind[dispatch.kind.index()] += gpu_ms;
        attribution.count_kind[dispatch.kind.index()] += 1;
        attribution.total_gpu_ms += gpu_ms;
    }
    Ok(attribution)
}

impl StepAttribution {
    /// The fusion/cut-oriented report: per-kind totals sorted descending,
    /// then the `top_n` hottest individual dispatches.
    #[must_use]
    pub fn report(&self, title: &str, top_n: usize) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "==== GPU-exec attribution: {title} ====");
        let _ = writeln!(
            out,
            "step gpu-exec total = {:.4} ms  ({} dispatches)",
            self.total_gpu_ms,
            self.per_dispatch.len()
        );

        let mut kinds: Vec<Kernel> = Kernel::ALL
            .into_iter()
            .filter(|kind| self.count_kind[kind.index()] > 0)
            .collect();
        kinds.sort_by(|a, b| self.by_kind[b.index()].total_cmp(&self.by_kind[a.index()]));
        let _ = writeln!(out, "-- per kernel-kind (sorted by total gpu-exec) --");
        for kind in kinds {
            let ms = self.by_kind[kind.index()];
            let n = self.count_kind[kind.index()];
            let pct = if self.total_gpu_ms > 0.0 {
                100.0 * ms / self.total_gpu_ms
            } else {
                0.0
            };
            let _ = writeln!(
                out,
                "  {:<20} {ms:8.4} ms  n={n:<4} {:9.5} ms/disp {pct:5.1}%",
                kind.name(),
                ms / f64::from(n.max(1)),
            );
        }

        let mut hottest: Vec<&DispatchAttribution> = self.per_dispatch.iter().collect();
        hottest.sort_by(|a, b| b.gpu_ms.total_cmp(&a.gpu_ms));
        let shown = top_n.min(hottest.len());
        let _ = writeln!(out, "-- top {shown} hottest dispatches --");
        for entry in &hottest[..shown] {
            let _ = writeln!(
                out,
                "  ord {:>5} {:<20} layer {:>3} {:9.5} ms",
                entry.dispatch.ordinal,
                entry.dispatch.kind.name(),
                entry
                    .dispatch
                    .layer
                    .map_or_else(|| "-".to_owned(), |layer| layer.to_string()),
                entry.gpu_ms,
            );
        }
        out
    }
}

/// Which kernel kinds a benchmarking run skips.
///
/// The one tool that prices a kernel correctly: the dispatch trace bills
/// overlapping kernels for the same wall time, so their shares sum to more
/// than the fire costs; ablation drops the dispatch, keeps the wall clock,
/// and the difference is the kind's price — at the cost of a wrong answer,
/// which is why it is a knob and not a mode.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ablation {
    ablated: Vec<bool>,
}

impl Ablation {
    /// Parse a comma-separated list of kind names.
    ///
    /// Returns the table and every token that named no kind. The unmatched
    /// list is the point: a typo'd token ablates nothing and the run
    /// reports the baseline while looking armed — the C++ lost a session
    /// of readings to exactly that before it learned to say so. Matching
    /// is exact per token; the trace's pipeline host names do not match.
    #[must_use]
    pub fn parse(spec: &str) -> (Ablation, Vec<String>) {
        let mut ablated = vec![false; Kernel::COUNT];
        let mut unmatched = Vec::new();
        for token in spec.split(',').filter(|token| !token.is_empty()) {
            match Kernel::from_name(token) {
                Some(kind) => ablated[kind.index()] = true,
                None => unmatched.push(token.to_owned()),
            }
        }
        (Ablation { ablated }, unmatched)
    }

    /// [`parse`](Self::parse) from `PIE_METAL_ABLATE` — the one place the
    /// environment is read.
    #[must_use]
    pub fn from_env() -> (Ablation, Vec<String>) {
        match std::env::var("PIE_METAL_ABLATE") {
            Ok(spec) => Self::parse(&spec),
            Err(_) => (Ablation::default(), Vec::new()),
        }
    }

    /// Whether the encoder should skip this kind.
    #[must_use]
    pub fn ablated(&self, kind: Kernel) -> bool {
        self.ablated.get(kind.index()).copied().unwrap_or(false)
    }

    /// Whether anything is ablated at all — the banner condition.
    #[must_use]
    pub fn any(&self) -> bool {
        self.ablated.iter().any(|&on| on)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dag() -> Vec<DispatchInfo> {
        vec![
            DispatchInfo {
                kind: Kernel::Rms,
                ordinal: 0,
                layer: Some(0),
            },
            DispatchInfo {
                kind: Kernel::QmvQ,
                ordinal: 1,
                layer: Some(0),
            },
            DispatchInfo {
                kind: Kernel::Rms,
                ordinal: 2,
                layer: Some(1),
            },
            DispatchInfo {
                kind: Kernel::QmvLmHead,
                ordinal: 3,
                layer: None,
            },
        ]
    }

    #[test]
    fn boundaries_diff_into_per_dispatch_and_per_kind_shares() {
        // 1ms, 3ms, 2ms, 4ms in nanosecond ticks.
        let ticks = [0, 1_000_000, 4_000_000, 6_000_000, 10_000_000];
        let a = attribute_step(&dag(), &ticks, 1.0).expect("attributes");
        assert_eq!(a.per_dispatch[1].gpu_ms, 3.0);
        assert_eq!(a.total_gpu_ms, 10.0);
        assert_eq!(a.by_kind[Kernel::Rms.index()], 3.0, "two rms dispatches");
        assert_eq!(a.count_kind[Kernel::Rms.index()], 2);
        assert_eq!(a.by_kind[Kernel::QmvLmHead.index()], 4.0);
        // The tick scale is honoured.
        let half = attribute_step(&dag(), &ticks, 0.5).expect("attributes");
        assert_eq!(half.total_gpu_ms, 5.0);
    }

    #[test]
    fn a_non_increasing_boundary_pair_attributes_zero_not_negative() {
        let ticks = [5_000_000, 5_000_000, 4_000_000, 6_000_000, 7_000_000];
        let a = attribute_step(&dag(), &ticks, 1.0).expect("attributes");
        assert_eq!(a.per_dispatch[0].gpu_ms, 0.0, "equal pair");
        assert_eq!(a.per_dispatch[1].gpu_ms, 0.0, "the clock went backwards");
        assert!(a.total_gpu_ms > 0.0);
    }

    #[test]
    fn a_boundary_count_that_does_not_match_the_dag_is_an_error() {
        assert_eq!(
            attribute_step(&dag(), &[0, 1, 2], 1.0),
            Err(BoundaryMismatch {
                boundaries: 3,
                dispatches: 4
            }),
            "the C++ returned valid=false for the caller to remember"
        );
    }

    #[test]
    fn the_report_leads_with_the_hottest_kind() {
        let ticks = [0, 1_000_000, 4_000_000, 6_000_000, 10_000_000];
        let a = attribute_step(&dag(), &ticks, 1.0).expect("attributes");
        let report = a.report("test step", 2);
        let head = report.find("qmv_lm_head").expect("the 4ms kind is listed");
        let rms = report.find(" rms").expect("rms is listed");
        assert!(head < rms, "kinds sort by total gpu time, descending");
        assert!(report.contains("top 2 hottest"));
    }

    #[test]
    fn ablation_matches_whole_kind_names_and_reports_the_rest() {
        let (ablation, unmatched) = Ablation::parse("rms,affine_qmv_routed_bf16,sdpa_paged");
        assert!(ablation.ablated(Kernel::Rms));
        assert!(ablation.ablated(Kernel::SdpaPaged));
        assert!(
            !ablation.ablated(Kernel::FfnRms),
            "`rms` must not also ablate `ffn_rms`"
        );
        assert_eq!(
            unmatched,
            ["affine_qmv_routed_bf16"],
            "a pipeline host name is not a kind and must be called out"
        );
        assert!(ablation.any());

        let (none, unmatched) = Ablation::parse("");
        assert!(!none.any());
        assert!(unmatched.is_empty());
    }
}
