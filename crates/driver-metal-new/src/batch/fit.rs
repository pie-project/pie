//! Whether a model fits, and which of the two ways it does not.
//!
//! `fits_on_this_gpu`'s decision. The arithmetic is small; what it is worth is
//! that there are **two independent bounds** and the C++ learned the second one
//! the expensive way.
//!
//! # Why the machine is a separate bound from the device
//!
//! The device ceiling is what this GPU would hold on an idle machine. What the
//! machine will actually give right now is a second bound, and on unified
//! memory it is usually the smaller one. Checking only the first is how a
//! 14 GiB model was admitted onto a box with 18 GiB left, allocated its pools,
//! and then hung: the command buffer never signalled, the context was abandoned
//! as unsafe to release, and **the process became unkillable**. Every retry left
//! another one, so free memory fell with each attempt while the ceiling being
//! checked never moved.
//!
//! Refusing here is the only cheap moment. Afterwards there is no failure path
//! — the allocation does not fail, the dispatch does not return, and nothing
//! short of a reboot recovers the memory.
//!
//! # The refusal has to name a knob
//!
//! *"A shorter context shrinks the KV"* was the whole of the old advice, and on
//! a paged family it is wrong twice: the operator reaches for `total_pages`, the
//! number does not move, and nothing says why. It does not move because a paged
//! model allocates **both** the paged pool that `total_pages` sizes **and** the
//! M=1 contiguous ring that `max_ctx` sizes — two KV regions, one knob each.
//!
//! Naming them separately is the difference between a refusal an operator can
//! act on and one they can only read, which is why [`Breakdown`] is carried
//! into the refusal rather than logged beside it.

use core::fmt;

/// Headroom left for what the load adds beyond the plan: the mmap'd weights
/// file leaves a file-backed copy roughly the size of the model, and the kernel
/// needs room to keep running.
///
/// A flat floor rather than a fraction, deliberately, so the refusal stays
/// legible — an operator can subtract two from a number they can see.
pub const HOST_MARGIN: u64 = 2 << 30;

/// The most the load's mapped-but-not-yet-copied window is counted for.
///
/// The host bound is about the **peak**, and the peak is the heap plus one
/// copy window rather than the heap plus the whole checkpoint.
pub const COPY_WINDOW: u64 = 2 << 30;

/// What the model wants resident.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Footprint {
    /// The heap: weights plus what the family needs on top of them.
    pub heap_bytes: u64,
    /// KV, recurrent state and scratch.
    pub elastic_bytes: u64,
    /// Of the heap, the part that is weights. Reported, never shrinkable.
    pub resident_weights: u64,
}

impl Footprint {
    /// Everything that must be resident at once.
    #[must_use]
    pub const fn want(&self) -> u64 {
        self.heap_bytes + self.elastic_bytes
    }
}

/// Where the elastic bytes go, so a refusal can name the knob for each.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Breakdown {
    /// The M=1 contiguous ring — sized by `max_model_len`.
    pub kv_ring: u64,
    /// The paged pool — sized by `total_pages` × `kv_page_size`.
    pub kv_pool: u64,
    /// Recurrent state — sized by `max_forward_requests`.
    pub state: u64,
    /// Scratch.
    pub scratch: u64,
}

/// What the machine and the device will hold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MachineFacts {
    /// The device's working set. Zero means it would not say.
    pub device_limit: u64,
    /// What the host could reclaim right now. Zero means unknown.
    pub reclaimable: u64,
    /// Whether the ceiling was forced by configuration rather than measured.
    ///
    /// A forced ceiling describes a device, not this machine, so the host bound
    /// is not applied against it.
    pub ceiling_is_forced: bool,
}

/// The admission answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    /// It fits, or the device would not say what it holds.
    Fits,
    /// The GPU itself will not hold it.
    DeviceBound {
        /// Bytes wanted resident.
        want: u64,
        /// Bytes the device holds.
        limit: u64,
    },
    /// The GPU would, but the machine has not got it right now.
    HostBound {
        /// Bytes wanted resident.
        want: u64,
        /// Bytes wanted at the load's peak, including the copy window.
        peak: u64,
        /// Bytes the host could reclaim.
        reclaimable: u64,
        /// Bytes the device itself would have held.
        limit: u64,
    },
}

impl Fit {
    /// Whether the model may be admitted.
    #[must_use]
    pub const fn is_fit(&self) -> bool {
        matches!(self, Fit::Fits)
    }
}

/// Decide whether this footprint may be admitted.
///
/// `transient_copy_bytes` is the checkpoint window the load maps before copying;
/// it is counted toward the host bound only, and only up to [`COPY_WINDOW`].
///
/// A `device_limit` of zero returns [`Fit::Fits`]: the device would not say what
/// it holds, and inventing a ceiling is worse than not having one.
///
/// The device bound is checked first. A footprint that exceeds both is reported
/// as [`Fit::DeviceBound`], because shrinking to fit the machine would still not
/// fit the GPU and the operator would then be told the second thing after
/// acting on the first.
#[must_use]
pub fn fits(footprint: Footprint, machine: MachineFacts, transient_copy_bytes: u64) -> Fit {
    let limit = machine.device_limit;
    if limit == 0 {
        return Fit::Fits;
    }
    let want = footprint.want();
    if want > limit {
        return Fit::DeviceBound { want, limit };
    }

    let reclaimable = if machine.ceiling_is_forced {
        0
    } else {
        machine.reclaimable
    };
    if reclaimable == 0 {
        return Fit::Fits;
    }
    let peak = want + transient_copy_bytes.min(COPY_WINDOW);
    if peak + HOST_MARGIN > reclaimable {
        return Fit::HostBound {
            want,
            peak,
            reclaimable,
            limit,
        };
    }
    Fit::Fits
}

/// A refusal, rendered for an operator.
///
/// Separate from [`Fit`] because the numbers are the answer and the prose is a
/// presentation of it: the caller decides, and only then does anyone need the
/// sentence.
#[derive(Clone, Copy, Debug)]
pub struct Refusal {
    /// The verdict being explained. [`Fit::Fits`] renders as nothing.
    pub fit: Fit,
    /// What the model wants.
    pub footprint: Footprint,
    /// Where the elastic bytes go, if known.
    pub breakdown: Option<Breakdown>,
}

/// Bytes as GiB, to two decimals.
fn gib(bytes: u64) -> String {
    format!("{:.2}", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let needs = |f: &mut fmt::Formatter<'_>, want: u64| {
            write!(
                f,
                "it needs {} GiB resident ({} GiB of weights, {} GiB of KV, state and scratch)",
                gib(want),
                gib(self.footprint.resident_weights),
                gib(self.footprint.elastic_bytes)
            )
        };
        match self.fit {
            Fit::Fits => Ok(()),
            Fit::HostBound {
                want,
                peak,
                reclaimable,
                limit,
            } => {
                write!(
                    f,
                    "this model does not fit the memory this machine has left: "
                )?;
                needs(f, want)?;
                if peak != want {
                    write!(
                        f,
                        " and {} GiB more while it loads, for the window of the \
                         checkpoint that is mapped and not yet copied, so the peak is {} GiB",
                        gib(peak - want),
                        gib(peak)
                    )?;
                }
                write!(
                    f,
                    ", and only {} GiB is reclaimable. The GPU itself would hold {} GiB, \
                     so this is the machine, not the device: something else already has \
                     the memory. On macOS a previously wedged run is the usual cause — it \
                     survives kill -9, holds its pages, and is only cleared by a reboot.",
                    gib(reclaimable),
                    gib(limit)
                )
            }
            Fit::DeviceBound { want, limit } => {
                write!(f, "this model does not fit this GPU: ")?;
                needs(f, want)?;
                write!(f, " and the device will hold {} GiB.", gib(limit))?;
                if let Some(b) = self.breakdown {
                    write!(
                        f,
                        " Of that: {} GiB M=1 KV ring (from max_model_len), {} GiB paged \
                         KV pool (total_pages x kv_page_size), {} GiB recurrent state \
                         (max_forward_requests), {} GiB scratch.",
                        gib(b.kv_ring),
                        gib(b.kv_pool),
                        gib(b.state),
                        gib(b.scratch)
                    )?;
                }
                write!(f, " The weights do not shrink.")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn footprint(heap: u64, elastic: u64) -> Footprint {
        Footprint {
            heap_bytes: heap,
            elastic_bytes: elastic,
            resident_weights: heap,
        }
    }

    fn machine(limit: u64, reclaimable: u64) -> MachineFacts {
        MachineFacts {
            device_limit: limit,
            reclaimable,
            ceiling_is_forced: false,
        }
    }

    #[test]
    fn a_device_that_will_not_say_what_it_holds_is_not_given_an_invented_ceiling() {
        assert_eq!(
            fits(footprint(900 * GIB, 0), machine(0, 1), 0),
            Fit::Fits,
            "no limit is not a limit of zero"
        );
    }

    #[test]
    fn the_machine_is_a_second_bound_and_it_is_usually_the_smaller_one() {
        // The 14-GiB-model-on-18-GiB-free case: the device would hold it and
        // the machine would not, and admitting it wedged the box.
        let f = footprint(14 * GIB, 0);
        // The device would hold it twice over; the machine has 15 left, and
        // 14 wanted plus the 2 GiB margin does not leave enough.
        let m = machine(24 * GIB, 15 * GIB);
        assert_eq!(
            fits(f, m, 0),
            Fit::HostBound {
                want: 14 * GIB,
                peak: 14 * GIB,
                reclaimable: 15 * GIB,
                limit: 24 * GIB,
            }
        );
        // With more of the machine free, the same model is admitted.
        assert_eq!(fits(f, machine(24 * GIB, 18 * GIB), 0), Fit::Fits);
    }

    #[test]
    fn the_margin_is_what_decides_a_near_miss() {
        let m = machine(64 * GIB, 10 * GIB);
        // 8 GiB + 2 GiB margin == 10, and the test is `>`, so exactly meeting
        // the reclaimable is admitted.
        assert_eq!(fits(footprint(8 * GIB, 0), m, 0), Fit::Fits);
        // One byte more is not.
        assert!(matches!(
            fits(footprint(8 * GIB + 1, 0), m, 0),
            Fit::HostBound { .. }
        ));
    }

    #[test]
    fn the_copy_window_counts_toward_the_peak_but_is_capped() {
        let m = machine(64 * GIB, 11 * GIB);
        let f = footprint(8 * GIB, 0);
        // 8 + 2 margin = 10, under 11: fits with no window.
        assert_eq!(fits(f, m, 0), Fit::Fits);
        // A whole 40 GiB checkpoint does not count 40 — the peak is one window.
        match fits(f, m, 40 * GIB) {
            Fit::HostBound { peak, .. } => assert_eq!(peak, 8 * GIB + COPY_WINDOW),
            other => panic!("expected a host bound, got {other:?}"),
        }
    }

    #[test]
    fn a_forced_ceiling_describes_a_device_so_the_machine_is_not_consulted() {
        let m = MachineFacts {
            device_limit: 64 * GIB,
            reclaimable: GIB, // the machine is tight
            ceiling_is_forced: true,
        };
        assert_eq!(
            fits(footprint(8 * GIB, 0), m, 0),
            Fit::Fits,
            "a forced ceiling is a claim about a device, not this machine"
        );
    }

    #[test]
    fn the_device_bound_wins_when_both_are_exceeded() {
        // Otherwise the operator shrinks to fit the machine and is then told
        // the second thing after acting on the first.
        let f = footprint(40 * GIB, 0);
        let m = machine(24 * GIB, 8 * GIB);
        assert_eq!(
            fits(f, m, 0),
            Fit::DeviceBound {
                want: 40 * GIB,
                limit: 24 * GIB
            }
        );
    }

    #[test]
    fn an_unknown_reclaimable_leaves_only_the_device_bound() {
        assert_eq!(
            fits(footprint(8 * GIB, 0), machine(24 * GIB, 0), 0),
            Fit::Fits
        );
    }

    #[test]
    fn a_device_refusal_names_a_knob_per_region_because_kv_has_two() {
        // The whole reason the breakdown exists: "a shorter context shrinks the
        // KV" sends an operator to `total_pages`, which does not move the ring.
        let refusal = Refusal {
            fit: Fit::DeviceBound {
                want: 40 * GIB,
                limit: 24 * GIB,
            },
            footprint: Footprint {
                heap_bytes: 30 * GIB,
                elastic_bytes: 10 * GIB,
                resident_weights: 30 * GIB,
            },
            breakdown: Some(Breakdown {
                kv_ring: 4 * GIB,
                kv_pool: 4 * GIB,
                state: GIB,
                scratch: GIB,
            }),
        };
        let text = refusal.to_string();
        assert!(text.contains("max_model_len"), "the ring's knob");
        assert!(text.contains("total_pages"), "the pool's knob");
        assert!(text.contains("max_forward_requests"), "the state's knob");
        assert!(text.contains("The weights do not shrink."));
    }

    #[test]
    fn a_host_refusal_says_it_is_the_machine_and_names_the_usual_cause() {
        let refusal = Refusal {
            fit: Fit::HostBound {
                want: 14 * GIB,
                peak: 16 * GIB,
                reclaimable: 15 * GIB,
                limit: 24 * GIB,
            },
            footprint: footprint(14 * GIB, 0),
            breakdown: None,
        };
        let text = refusal.to_string();
        assert!(text.contains("this is the machine, not the device"));
        assert!(text.contains("kill -9"), "the wedged-run cause");
        assert!(text.contains("peak is"), "the copy window is explained");
    }

    #[test]
    fn a_fit_renders_as_nothing_because_there_is_nothing_to_explain() {
        let refusal = Refusal {
            fit: Fit::Fits,
            footprint: footprint(GIB, 0),
            breakdown: None,
        };
        assert_eq!(refusal.to_string(), "");
    }
}
