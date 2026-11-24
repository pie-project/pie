//! The self-check's decision: did the watermark register in this text?
//!
//! The detector scores each distinct context once, and under H0 (text not
//! sampled with this key) its statistic is a standard-normal z-score. The
//! watermark registered when that z reaches [`Z_THRESHOLD`] (one-sided
//! p ~ 3e-5 under H0) over at least [`MIN_CONTEXTS`] contexts. Fewer
//! contexts are too few to judge: a peaked model repeats its contexts, and
//! over a few dozen of them a real watermark's z overlaps the null's, so the
//! check neither passes nor fails such a run.

/// The z a watermarked text must reach.
pub const Z_THRESHOLD: f32 = 4.0;

/// The fewest distinct contexts the check judges over.
pub const MIN_CONTEXTS: usize = 100;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// The keyed z reached the threshold.
    Registered,
    /// The keyed z fell short of it over enough contexts: the context hash
    /// is not reaching the sampler.
    Missing,
    /// Too few contexts to judge.
    TooFew,
    /// The text was sampled without the watermark; nothing to check.
    Unwatermarked,
}

impl Verdict {
    #[must_use]
    pub fn of(watermark: bool, contexts: usize, z: f32) -> Verdict {
        if !watermark {
            Verdict::Unwatermarked
        } else if contexts < MIN_CONTEXTS {
            Verdict::TooFew
        } else if z >= Z_THRESHOLD {
            Verdict::Registered
        } else {
            Verdict::Missing
        }
    }

    #[must_use]
    pub fn describe(self, contexts: usize, z: f32) -> String {
        match self {
            Verdict::Registered => {
                format!("registered: z {z:.2} >= {Z_THRESHOLD} over {contexts} contexts")
            }
            Verdict::Missing => {
                format!("missing: z {z:.2} < {Z_THRESHOLD} over {contexts} contexts")
            }
            Verdict::TooFew => format!("too few contexts to judge: {contexts} < {MIN_CONTEXTS}"),
            Verdict::Unwatermarked => "not watermarked".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_strong_z_over_enough_contexts_registers() {
        assert_eq!(Verdict::of(true, 120, 5.8), Verdict::Registered);
        assert_eq!(
            Verdict::of(true, MIN_CONTEXTS, Z_THRESHOLD),
            Verdict::Registered
        );
    }

    #[test]
    fn a_null_z_over_enough_contexts_is_missing() {
        assert_eq!(Verdict::of(true, 120, 1.0), Verdict::Missing);
        assert_eq!(Verdict::of(true, 400, -0.5), Verdict::Missing);
    }

    /// The default run on a peaked model: ~30 contexts. Whatever its z, it
    /// is neither a pass nor a failure.
    #[test]
    fn a_few_contexts_are_too_few_to_judge() {
        assert_eq!(Verdict::of(true, 30, 0.2), Verdict::TooFew);
        assert_eq!(Verdict::of(true, 30, 9.0), Verdict::TooFew);
        assert_eq!(Verdict::of(true, MIN_CONTEXTS - 1, 9.0), Verdict::TooFew);
    }

    #[test]
    fn an_unwatermarked_run_is_not_judged() {
        assert_eq!(Verdict::of(false, 400, 0.0), Verdict::Unwatermarked);
    }
}
