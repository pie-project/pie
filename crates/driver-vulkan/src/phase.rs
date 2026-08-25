//! Host-side phase timing, off unless it is asked for.
//!
//! [`crate::device::Device::timings`] answers where the DEVICE went, and the
//! answer it gives is complete: two timestamps around every dispatch account
//! for a submission end to end. Nothing accounted for the host, and the host
//! is not small -- a qwen3-0.6b decode spends about a fifth of its wall step
//! outside `run_all` entirely, in the lowering, the plan and the scalar
//! blocks, and that share does NOT fall with context because none of the work
//! is proportional to the history.
//!
//! It was measured, before this module existed, by editing timers into
//! `serve.rs` and taking them out again. That is a measurement nobody can
//! repeat and no test can hold, which is how the number in
//! `the_projections_dominate_both_steps_now_that_the_decode_splits_its_keys`
//! came to be a table in a doc comment with no code behind it.
//!
//! # What it costs when it is off
//!
//! One relaxed atomic load and a branch per span. [`span`] returns `None`
//! then, and `None`'s `Drop` does nothing. Measured against a build with the
//! calls deleted outright: no difference a decode can see.
//!
//! # Why a thread local
//!
//! A fire is one thread from `Serving::once` to `run_all`, so the totals want
//! no lock, and two threads firing at once want SEPARATE totals rather than a
//! sum nobody can attribute. A caller reads back on the thread that fired,
//! which is the thread that called `shell::Shell::step` -- and that verb went
//! with `shell`, so what a caller reads back on is now whatever thread drove
//! `serve::run`. The rule is unchanged and the name of the caller is not.

use std::cell::RefCell;
use std::time::Instant;

/// Whether `PIE_VULKAN_HOST_PHASES` was set when this process started.
///
/// Read once. An environment variable read per span would be its own
/// measurement -- `std::env::var_os` walks the environment block -- and this
/// is a tool for finding microseconds.
fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("PIE_VULKAN_HOST_PHASES").is_some())
}

thread_local! {
    /// Milliseconds and entries per name, in the order the names were first
    /// seen -- which is the order the phases run, and is what makes the
    /// printed table readable without sorting it.
    static ROWS: RefCell<Vec<(&'static str, f64, u64)>> = const { RefCell::new(Vec::new()) };
}

/// A running phase. Records into this thread's totals when it is dropped.
///
/// Nesting is allowed and is not unpicked: a parent's span includes its
/// children's, and the names are written to say so (`fire`, `fire/plan`,
/// `fire/plan/routine`). Subtracting would need a stack and would answer a
/// question nobody asked -- the interesting number is almost always "how much
/// of the parent is this", which a reader does by dividing.
pub struct Span {
    name: &'static str,
    at: Instant,
}

impl Drop for Span {
    fn drop(&mut self) {
        let ms = self.at.elapsed().as_secs_f64() * 1000.0;
        let name = self.name;
        ROWS.with(|r| {
            let mut r = r.borrow_mut();
            // BY POINTER FIRST, AND THE DIFFERENCE IS THE TOOL'S WHOLE COST.
            //
            // Every caller passes a literal, so the `&'static str` a span
            // carries is the same POINTER every time that call site runs, and
            // the row it belongs to is found by comparing addresses. `*n ==
            // name` is a length test and a `memcmp`, over a table that is
            // twenty-six rows deep by the end of a fire, for names that share
            // `fire/plan/routine/` and so disagree only in their last word --
            // paid 3,200 times a decode step, once per span.
            //
            // Measured, release, `tests/hostprof.rs`: the whole tool cost
            // 0.39 ms of the 5.11 ms step it reported, against a 4.72 ms step
            // with it off. That is a third of the 1.4 ms then believed to be
            // host -- a figure since retracted, see this crate's module doc,
            // but the tool's own cost is measured against the WALL and so
            // survives the retraction. It was attributed to the phases it was
            // measuring, and it aimed an investigation at a number that was
            // substantially the profiler's own.
            //
            // The `==` stays as a fallback because nothing GUARANTEES two
            // equal literals in different codegen units are one address; the
            // pointer test is the fast path, not the rule.
            if let Some(row) = r
                .iter_mut()
                .find(|(n, _, _)| std::ptr::eq(n.as_ptr(), name.as_ptr()) || *n == name)
            {
                row.1 += ms;
                row.2 += 1;
                return;
            }
            r.push((name, ms, 1));
        });
    }
}

/// Start timing a phase, or `None` when the tool is off.
///
/// `let _span = phase::span("fire/plan");` -- bound to a name, because
/// `let _ = ` drops it at once and would time nothing.
#[must_use]
pub fn span(name: &'static str) -> Option<Span> {
    on().then(|| Span {
        name,
        at: Instant::now(),
    })
}

/// This thread's totals so far: name, milliseconds, entries.
#[must_use]
pub fn rows() -> Vec<(&'static str, f64, u64)> {
    ROWS.with(|r| r.borrow().clone())
}

/// Forget this thread's totals, so a warm-up is not counted with the
/// measurement.
pub fn reset() {
    ROWS.with(|r| r.borrow_mut().clear());
}

#[cfg(test)]
mod tests {
    /// A span records under its own name and accumulates across entries.
    ///
    /// Runs whichever way the environment is set: when the tool is off there
    /// is nothing to record and the totals stay empty, which is the property
    /// worth checking on that path -- an "off" that still allocated would be
    /// a cost every user of this driver paid for a number none of them read.
    #[test]
    fn a_span_records_under_its_name_when_the_tool_is_on() {
        super::reset();
        for _ in 0..3 {
            let _s = super::span("a-phase-no-other-test-names");
            std::thread::sleep(std::time::Duration::from_micros(50));
        }
        let rows = super::rows();
        let found = rows
            .iter()
            .find(|(n, _, _)| *n == "a-phase-no-other-test-names");
        match (super::on(), found) {
            (true, Some((_, ms, n))) => {
                assert_eq!(*n, 3, "three entries were timed");
                assert!(*ms > 0.0, "three sleeps took no time at all");
            }
            (true, None) => panic!("the tool is on and recorded nothing"),
            (false, None) => {}
            (false, Some(_)) => panic!("the tool is off and recorded something"),
        }
        super::reset();
    }
}
