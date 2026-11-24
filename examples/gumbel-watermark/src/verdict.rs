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

/// Where a keyed stream cycled: the distinct contexts it made before it
/// stopped making new ones, and the period it repeats with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cycle {
    pub distinct: usize,
    pub period: usize,
}

/// Whether the stream of per-step contexts ends in a cycle: past its last
/// new context it settles into a loop: the shortest period `p` its suffix
/// repeats with for at least two whole periods, the loop holding every
/// step past the last new context but for an entry of fewer than `p` old
/// contexts off the loop.
/// A watermark that picks each token from its context's keyed noise does
/// exactly this on a peaked model: the same context draws the same token,
/// which makes the same next context.
#[must_use]
pub fn cycle(contexts: &[u32]) -> Option<Cycle> {
    let mut seen = std::collections::HashSet::new();
    let mut last_new = None;
    for (at, &context) in contexts.iter().enumerate() {
        if seen.insert(context) {
            last_new = Some(at);
        }
    }
    let first_old = last_new? + 1;
    let len = contexts.len();
    let tail = len.saturating_sub(first_old);
    for period in 1..=tail / 2 {
        // The loop's start: the step after the last one, scanning back from
        // the end, that differs from the step a period after it.
        let start = (first_old..len - period)
            .rev()
            .find(|&at| contexts[at] != contexts[at + period])
            .map_or(first_old, |at| at + 1);
        // The loop must hold the whole tail but for an entry shorter than
        // one period, and repeat at least twice.
        if start - first_old < period && len - start >= 2 * period {
            return Some(Cycle {
                distinct: seen.len(),
                period,
            });
        }
    }
    None
}

/// The verdict's text, and why a cycled stream is too few to judge.
#[must_use]
pub fn describe(verdict: Verdict, contexts: usize, z: f32, cycled: Option<Cycle>) -> String {
    match (verdict, cycled) {
        (Verdict::TooFew, Some(cycle)) => format!(
            "too few contexts to judge: the keyed output cycled after {} distinct contexts \
             (period {}) — deterministic Gumbel repeats on peaked models; try a different \
             prompt or secret",
            cycle.distinct, cycle.period
        ),
        _ => verdict.describe(contexts, z),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 30 new contexts, then a period-4 loop for 200 more steps.
    #[test]
    fn a_stream_that_loops_is_a_cycle() {
        let mut stream: Vec<u32> = (0..30).collect();
        stream.extend((0..200).map(|k| 26 + (k % 4) as u32));
        let cycled = cycle(&stream);
        assert_eq!(
            cycled,
            Some(Cycle {
                distinct: 30,
                period: 4
            })
        );
        let text = describe(Verdict::of(true, 30, 1.0), 30, 1.0, cycled);
        assert!(text.contains("cycled after 30 distinct contexts"), "{text}");
        assert!(text.contains("try a different prompt or secret"), "{text}");
    }

    /// A short run that made a new context every step has not cycled: it
    /// is too few to judge for its length alone.
    #[test]
    fn a_short_run_is_not_a_cycle() {
        let stream: Vec<u32> = (0..32).collect();
        assert_eq!(cycle(&stream), None);
        let text = describe(Verdict::of(true, 32, 3.0), 32, 3.0, cycle(&stream));
        assert_eq!(text, "too few contexts to judge: 32 < 100");
    }

    /// A loop whose step after the last new context is an old context off
    /// the loop's first pass, and whose period holds a context twice (Mac's
    /// Qwen3.5 stream: 86 distinct, period 19): still a cycle.
    #[test]
    fn a_loop_entered_through_an_old_context_is_a_cycle() {
        // 80 new contexts, then old context 3 (off the loop), then a loop of
        // period 19 over old contexts with 40 appearing twice in it.
        let mut stream: Vec<u32> = (0..80).collect();
        stream.push(3);
        let body: Vec<u32> = (41..58).chain([40, 40]).collect();
        assert_eq!(body.len(), 19);
        stream.extend(body.iter().cycle().take(256 - stream.len()));
        let cycled = cycle(&stream);
        assert_eq!(
            cycled,
            Some(Cycle {
                distinct: 80,
                period: 19
            })
        );
        let text = describe(Verdict::of(true, 80, 1.0), 80, 1.0, cycled);
        assert!(
            text.contains("cycled after 80 distinct contexts (period 19)"),
            "{text}"
        );
    }

    /// A tail that only ends on a repeated pair is no loop.
    #[test]
    fn a_trailing_repeat_is_not_a_cycle() {
        let mut stream: Vec<u32> = (0..40).collect();
        stream.extend([3, 9, 17, 5, 5]);
        assert_eq!(cycle(&stream), None);
    }

    /// Repeats that do not settle into a loop (a context seen again, then
    /// new ones) are no cycle.
    #[test]
    fn a_repeat_before_new_contexts_is_not_a_cycle() {
        let stream = [1, 2, 3, 1, 4, 5, 6, 2];
        assert_eq!(cycle(&stream), None);
    }

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
