//! What the kernel left behind: the 16-byte status word and how to read it.
//!
//! Every M1, M2 and M3 dispatch writes one `M1Status` into a device buffer the
//! driver allocated and zeroed. When the command finishes, that struct is the
//! entire account of what the GPU did — there is no other channel, no log, and
//! nothing to re-run. Reading it wrongly is not a cosmetic problem: the caller
//! decides between committing the fire, retrying it, and refusing it on the
//! basis of these four words.
//!
//! ## The fault space is named, and the C++ did not use the names
//!
//! `compiler/codegen/src/fault.rs` declares every number a kernel can leave in
//! `fault`, with a test that the classes cannot alias. Its own module doc says
//! "Nothing decodes these: the drivers surface the number and a human reads
//! it." That was true, and it is the thing this module changes. The classes are
//! written down; a driver that prints `160` when the host has a constant saying
//! `FUSED_GEOMETRY_MISMATCH` is discarding information it already has.
//!
//! The constants are restated here rather than imported because this crate does
//! not depend on `tensor-compiler` at build time — the same reason
//! `identity::Versions` is a parameter. [`FAULT_CLASSES`] is the mirror and
//! [`describe_fault`] reads it. A hand-copied table that nothing checks drifts,
//! so the compiler is a *dev*-dependency and
//! [`the_mirror_still_matches_the_compilers_table`](tests) compares the two
//! entry by entry.

/// The status word one dispatch writes. `M1Status` in
/// `compiler/codegen/runtime/metal/ptir_m1_runtime.metal`, 16 bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct Status {
    /// See [`State`].
    pub state: u32,
    /// The fault code, or an op tag. See [`describe_fault`].
    pub fault: u32,
    /// The intrinsic id, for a fault raised by `m1_fault_op`.
    pub reserved0: u32,
    /// `(site << 24) | (imm & 0x00ff_ffff)`, for a fault raised by
    /// `m1_fault_op`. See [`Site`].
    pub reserved1: u32,
}

/// The size the Metal side agrees on. The C++ asserted this; so does
/// [`the_status_word_is_sixteen_bytes`](tests).
pub const STATUS_BYTES: usize = 16;

impl Status {
    /// Read a status out of a mapped buffer.
    ///
    /// # Errors
    ///
    /// `None` when there are not [`STATUS_BYTES`] to read. The C++ reached the
    /// buffer through `*static_cast<const DeviceStatus*>(contents())` with no
    /// length in sight, so a status buffer that failed to allocate — `contents`
    /// returns null — was a null dereference in the completion handler.
    #[must_use]
    pub fn read(bytes: &[u8]) -> Option<Status> {
        if bytes.len() < STATUS_BYTES {
            return None;
        }
        let word = |at: usize| -> u32 {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };
        Some(Status {
            state: word(0),
            fault: word(4),
            reserved0: word(8),
            reserved1: word(12),
        })
    }

    /// What the `state` word means, or `None` if it is not one of the five.
    #[must_use]
    pub fn state(self) -> Option<State> {
        match self.state {
            0 => Some(State::Unset),
            1 => Some(State::Running),
            2 => Some(State::Retry),
            3 => Some(State::Fault),
            4 => Some(State::Committed),
            _ => None,
        }
    }

    /// The guard site, when this fault came from `m1_fault_op`.
    #[must_use]
    pub fn site(self) -> Site {
        match self.reserved1 >> 24 {
            1 => Site::SinkNarrowerThanValue,
            2 => Site::MtpDraftsZeroRowWidth,
            3 => Site::NoArmClaimedTheTag,
            other => Site::Unknown(other),
        }
    }

    /// The `imm` the faulting op carried, when it came from `m1_fault_op`.
    #[must_use]
    pub fn immediate(self) -> u32 {
        self.reserved1 & 0x00ff_ffff
    }
}

/// The five values `M1Status::state` takes.
///
/// The C++ tested for 4 and for 2 and treated *everything else* as a fault, so
/// three distinct conditions arrived at the caller wearing the same words. See
/// [`Outcome::of`] for why that matters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// 0 — the kernel never wrote anything.
    Unset,
    /// 1 — the kernel started and did not reach its end.
    Running,
    /// 2 — the readiness guard was unmet; the fire may be re-fired later.
    Retry,
    /// 3 — a guard inside the kernel refused; `fault` says which.
    Fault,
    /// 4 — the fire ran to completion and its effects are committed.
    Committed,
}

/// Which guard in `m1_fault_op` fired.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    /// The channel's cell is narrower than the value being put into it.
    SinkNarrowerThanValue,
    /// `MtpDrafts` was reached with a row width of zero.
    MtpDraftsZeroRowWidth,
    /// The interpreter has no arm for this op tag.
    NoArmClaimedTheTag,
    /// Not a site this driver knows, including 0 — which is what a plain
    /// `m1_fault` leaves, since it does not touch `reserved1` at all.
    Unknown(u32),
}

impl Site {
    /// A phrase for a diagnostic, or `None` when the fault did not record one.
    #[must_use]
    pub fn describe(self) -> Option<&'static str> {
        match self {
            Site::SinkNarrowerThanValue => Some("channel sink narrower than the value"),
            Site::MtpDraftsZeroRowWidth => Some("MtpDrafts with a zero row width"),
            Site::NoArmClaimedTheTag => Some("no arm claimed this op tag"),
            Site::Unknown(_) => None,
        }
    }
}

/// One named region of the fault space. Mirrors `codegen::fault::FaultClass`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaultClass {
    /// The code written for channel 0.
    pub base: u32,
    /// The symbolic name, matching the host constant.
    pub name: &'static str,
    /// Whether the emitter adds the channel index to [`base`](Self::base).
    pub per_channel: bool,
}

/// Every fault class, ascending. Mirrors `codegen::fault::CLASSES`.
///
/// The per-channel classes own a run `base ..= base + MAX_CHANNELS - 1`; the
/// tightest gap between two bases is `0x80`, which is what bounds
/// [`MAX_CHANNELS`](crate::pipeline::MAX_CHANNELS). That invariant is checked
/// on the host and again in [`tests`] here, because a driver that decodes a
/// number is now a second thing that breaks when the classes alias.
pub const FAULT_CLASSES: &[FaultClass] = &[
    FaultClass {
        base: 0xA0,
        name: "FUSED_GEOMETRY_MISMATCH",
        per_channel: false,
    },
    FaultClass {
        base: 0xB3,
        name: "M3_THREADS_EXCEEDED",
        per_channel: false,
    },
    FaultClass {
        base: 0x100,
        name: "LANE_HEADER_MISMATCH",
        per_channel: false,
    },
    FaultClass {
        base: 0x200,
        name: "M1_RING_CORRUPT",
        per_channel: true,
    },
    FaultClass {
        base: 0x300,
        name: "M1_HEAD_STALE",
        per_channel: true,
    },
    FaultClass {
        base: 0x400,
        name: "M1_NOT_FULL",
        per_channel: true,
    },
    FaultClass {
        base: 0x480,
        name: "M1_NOT_EMPTY",
        per_channel: true,
    },
    FaultClass {
        base: 0x500,
        name: "M1_PUT_BLOCKED",
        per_channel: true,
    },
    FaultClass {
        base: 0x700,
        name: "M3_RING_CORRUPT",
        per_channel: true,
    },
    FaultClass {
        base: 0x780,
        name: "M3_NOT_READY",
        per_channel: true,
    },
];

/// A decoded fault code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fault {
    /// The class it falls in, when it falls in one.
    pub class: Option<&'static str>,
    /// The channel the class was written for, for a per-channel class.
    pub channel: Option<u32>,
    /// Whether the code is also a valid op tag, so the class name may be
    /// wrong.
    ///
    /// Every class below `0x100` shares numbers with the op table, and one
    /// emitted kernel both writes `FUSED_GEOMETRY_MISMATCH` (`0xA0`) and
    /// reports an unhandled op as its raw tag — and `0xA0` *is*
    /// `intrinsic_val`'s tag. The host records this as a live ambiguity rather
    /// than fixing it, because respacing rewrites every fused golden. A
    /// decoder that does not say so turns a recorded ambiguity into a
    /// confident wrong answer.
    pub ambiguous_with_op_tag: bool,
}

/// Decode a `fault` code against [`FAULT_CLASSES`].
///
/// `max_channel` is the highest channel index the program can address, which is
/// what bounds a per-channel class's run. Pass
/// [`MAX_CHANNELS`](crate::pipeline::MAX_CHANNELS)` - 1` when the program's own
/// count is not to hand.
#[must_use]
pub fn describe_fault(fault: u32, max_channel: u32) -> Fault {
    let mut found: Option<(&'static FaultClass, u32)> = None;
    for class in FAULT_CLASSES {
        let span = if class.per_channel { max_channel } else { 0 };
        if fault >= class.base && fault - class.base <= span {
            found = Some((class, fault - class.base));
        }
    }
    match found {
        Some((class, offset)) => Fault {
            class: Some(class.name),
            channel: class.per_channel.then_some(offset),
            ambiguous_with_op_tag: class.base < 0x100,
        },
        None => Fault {
            class: None,
            channel: None,
            // Below 0x100 with no class is an op tag and nothing else, which is
            // unambiguous — the ambiguity is a class colliding with a tag.
            ambiguous_with_op_tag: false,
        },
    }
}

/// What the caller does with the fire.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The fire ran; commit it.
    Committed,
    /// The fire did not run and may be re-fired.
    Retry,
    /// The fire did not run and will not.
    Failed,
}

impl Outcome {
    /// Read the status the way the caller needs it.
    ///
    /// `dispatched` is whether the command was actually encoded and submitted.
    /// The M3 reporter learned to ask this — a group prepared and never encoded
    /// leaves the buffer's zero fill, and reading that back produces "state=0
    /// op_tag=0x0" for every lane: a GPU fault report for something the GPU was
    /// never asked to do, which then replaces the executor's own account of why
    /// the forward was refused. The M1 path never learned it, so the same
    /// zero fill there became `"Metal M1 generated op fault 0"`.
    ///
    /// `State::Running` is the other one the C++ folded into the fault arm. A
    /// kernel that set `state = 1` and stopped did not fault — it was killed,
    /// or the command was aborted, or the device reset under it. Reporting that
    /// as an op fault sends the reader to the emitter for a bug that is not
    /// there.
    #[must_use]
    pub fn of(status: Status, dispatched: bool) -> (Outcome, Diagnosis) {
        if !dispatched {
            return (Outcome::Failed, Diagnosis::NeverDispatched);
        }
        match status.state() {
            Some(State::Committed) => (Outcome::Committed, Diagnosis::Committed),
            Some(State::Retry) => (Outcome::Retry, Diagnosis::ReadinessUnmet),
            Some(State::Fault) => (Outcome::Failed, Diagnosis::Faulted),
            Some(State::Unset) => (Outcome::Failed, Diagnosis::NeverWritten),
            Some(State::Running) => (Outcome::Failed, Diagnosis::NeverFinished),
            None => (Outcome::Failed, Diagnosis::UnknownState(status.state)),
        }
    }
}

/// Why the fire ended as it did — the part of the outcome that is for a human.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diagnosis {
    /// Ran to completion.
    Committed,
    /// A readiness guard in the kernel was unmet.
    ReadinessUnmet,
    /// A guard inside the kernel refused; the fault code says which.
    Faulted,
    /// The command was prepared and never submitted.
    NeverDispatched,
    /// The command was submitted and the kernel wrote nothing.
    NeverWritten,
    /// The kernel started and did not reach its end.
    NeverFinished,
    /// A `state` this driver has no name for.
    UnknownState(u32),
}

impl Diagnosis {
    /// A phrase for a diagnostic.
    #[must_use]
    pub fn describe(self) -> &'static str {
        match self {
            Diagnosis::Committed => "committed",
            Diagnosis::ReadinessUnmet => "a readiness guard in the kernel was unmet",
            Diagnosis::Faulted => "a guard inside the kernel refused",
            Diagnosis::NeverDispatched => {
                "the command was prepared but never encoded, so no kernel ran"
            }
            Diagnosis::NeverWritten => {
                "the command was submitted and the kernel wrote no status at all"
            }
            Diagnosis::NeverFinished => "the kernel started and did not reach its end",
            Diagnosis::UnknownState(_) => "the kernel wrote a status this driver cannot read",
        }
    }
}

/// The full account of one status, as a line for a log.
///
/// The C++ built this string twice and differently: the M1 path printed
/// `std::to_string(status.fault)`, in decimal, and dropped `reserved0` and
/// `reserved1` entirely; the M3 path printed the fault in hex and decoded the
/// guard site. The same kernel faulting the same way reported `160` under M1
/// and `0xa0 intr=3 guard=no arm claimed this op tag` under M3.
#[must_use]
pub fn report(status: Status, dispatched: bool, max_channel: u32) -> String {
    let (_, diagnosis) = Outcome::of(status, dispatched);
    if !dispatched {
        return diagnosis.describe().to_string();
    }
    if matches!(diagnosis, Diagnosis::Committed) {
        return "committed".to_string();
    }
    let fault = describe_fault(status.fault, max_channel);
    let mut line = format!("{}: fault 0x{:x}", diagnosis.describe(), status.fault);
    if let Some(class) = fault.class {
        line.push_str(" (");
        line.push_str(class);
        if let Some(channel) = fault.channel {
            line.push_str(&format!(" channel {channel}"));
        }
        if fault.ambiguous_with_op_tag {
            line.push_str(", or the op tag of the same value");
        }
        line.push(')');
    }
    if let Some(site) = status.site().describe() {
        line.push_str(&format!(
            " intr={} imm={} guard={site}",
            status.reserved0,
            status.immediate()
        ));
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::MAX_CHANNELS;

    const LAST_CHANNEL: u32 = MAX_CHANNELS as u32 - 1;

    #[test]
    fn the_status_word_is_sixteen_bytes() {
        assert_eq!(core::mem::size_of::<Status>(), STATUS_BYTES);
    }

    #[test]
    fn a_status_shorter_than_the_struct_is_not_read_at_all() {
        assert_eq!(Status::read(&[0u8; 15]), None);
    }

    #[test]
    fn the_four_words_are_read_little_endian_in_order() {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&1u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&0xA0u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&7u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&0x0300_0009u32.to_le_bytes());
        let status = Status::read(&bytes).expect("long enough");
        assert_eq!(status.state, 1);
        assert_eq!(status.fault, 0xA0);
        assert_eq!(status.reserved0, 7);
        assert_eq!(status.site(), Site::NoArmClaimedTheTag);
        assert_eq!(status.immediate(), 9);
    }

    #[test]
    fn a_plain_fault_records_no_guard_site() {
        // `m1_fault` never touches reserved1, so the site reads back as 0.
        let status = Status {
            state: 3,
            fault: 0x400,
            ..Status::default()
        };
        assert_eq!(status.site(), Site::Unknown(0));
        assert_eq!(status.site().describe(), None);
    }

    #[test]
    fn a_command_that_was_never_encoded_is_not_a_gpu_fault() {
        let zero_fill = Status::default();
        let (outcome, why) = Outcome::of(zero_fill, false);
        assert_eq!(outcome, Outcome::Failed);
        assert_eq!(why, Diagnosis::NeverDispatched);
        assert!(!report(zero_fill, false, LAST_CHANNEL).contains("fault"));
    }

    #[test]
    fn a_kernel_that_wrote_nothing_is_distinct_from_one_that_faulted() {
        let (_, wrote_nothing) = Outcome::of(Status::default(), true);
        let (_, faulted) = Outcome::of(
            Status {
                state: 3,
                fault: 0,
                ..Status::default()
            },
            true,
        );
        assert_eq!(wrote_nothing, Diagnosis::NeverWritten);
        assert_eq!(faulted, Diagnosis::Faulted);
        assert_ne!(wrote_nothing, faulted);
    }

    #[test]
    fn a_kernel_that_started_and_stopped_is_not_an_op_fault() {
        let (outcome, why) = Outcome::of(
            Status {
                state: 1,
                ..Status::default()
            },
            true,
        );
        assert_eq!(outcome, Outcome::Failed);
        assert_eq!(why, Diagnosis::NeverFinished);
    }

    #[test]
    fn a_state_this_driver_does_not_know_says_so() {
        let (outcome, why) = Outcome::of(
            Status {
                state: 99,
                ..Status::default()
            },
            true,
        );
        assert_eq!(outcome, Outcome::Failed);
        assert_eq!(why, Diagnosis::UnknownState(99));
    }

    #[test]
    fn committed_and_retry_are_the_two_non_failures() {
        for (state, expected) in [(4, Outcome::Committed), (2, Outcome::Retry)] {
            let (outcome, _) = Outcome::of(
                Status {
                    state,
                    ..Status::default()
                },
                true,
            );
            assert_eq!(outcome, expected);
        }
    }

    #[test]
    fn a_per_channel_fault_names_its_class_and_its_channel() {
        let fault = describe_fault(0x400 + 5, LAST_CHANNEL);
        assert_eq!(fault.class, Some("M1_NOT_FULL"));
        assert_eq!(fault.channel, Some(5));
    }

    #[test]
    fn a_code_past_a_classs_channel_run_does_not_belong_to_it() {
        // 0x400 + MAX_CHANNELS is one past M1_NOT_FULL's run and still short
        // of M1_NOT_EMPTY at 0x480.
        let fault = describe_fault(0x400 + MAX_CHANNELS as u32, LAST_CHANNEL);
        assert_eq!(fault.class, None);
    }

    #[test]
    fn a_class_that_shares_a_number_with_an_op_tag_says_it_might_be_either() {
        // 0xA0 is FUSED_GEOMETRY_MISMATCH and intrinsic_val's tag, and one
        // emitted kernel writes both.
        let fault = describe_fault(0xA0, LAST_CHANNEL);
        assert_eq!(fault.class, Some("FUSED_GEOMETRY_MISMATCH"));
        assert!(fault.ambiguous_with_op_tag);
        assert!(
            report(
                Status {
                    state: 3,
                    fault: 0xA0,
                    ..Status::default()
                },
                true,
                LAST_CHANNEL
            )
            .contains("or the op tag of the same value")
        );
    }

    #[test]
    fn an_op_tag_that_is_not_a_class_is_reported_without_a_guess() {
        let fault = describe_fault(0x42, LAST_CHANNEL);
        assert_eq!(fault.class, None);
        assert!(!fault.ambiguous_with_op_tag);
    }

    #[test]
    fn a_fault_is_reported_in_hex_because_it_is_a_tag() {
        let line = report(
            Status {
                state: 3,
                fault: 0xA0,
                reserved0: 3,
                reserved1: 0x0300_0000,
            },
            true,
            LAST_CHANNEL,
        );
        assert!(line.contains("0xa0"), "{line}");
        assert!(line.contains("no arm claimed this op tag"), "{line}");
    }

    /// The one test that makes the copy safe. Everything else here checks the
    /// decoder; this checks that the numbers it decodes against are still the
    /// numbers the emitter writes. A class added on the host without an entry
    /// here would otherwise decode as "no class" — a silently worse diagnostic
    /// than the raw number the C++ printed, because it looks like an answer.
    #[test]
    fn the_mirror_still_matches_the_compilers_table() {
        let theirs = tensor_compiler::codegen::fault::CLASSES;
        assert_eq!(
            FAULT_CLASSES.len(),
            theirs.len(),
            "the compiler declares {} fault classes and this mirror has {}",
            theirs.len(),
            FAULT_CLASSES.len()
        );
        for (mine, theirs) in FAULT_CLASSES.iter().zip(theirs) {
            assert_eq!(mine.name, theirs.name);
            assert_eq!(mine.base, theirs.base, "{} moved", theirs.name);
            assert_eq!(
                mine.per_channel, theirs.per_channel,
                "{} changed shape",
                theirs.name
            );
        }
    }

    #[test]
    fn the_classes_are_ordered_and_their_channel_runs_do_not_collide() {
        for pair in FAULT_CLASSES.windows(2) {
            let (lower, upper) = (pair[0], pair[1]);
            assert!(lower.base < upper.base, "{} is misplaced", upper.name);
            let last = if lower.per_channel {
                lower.base + LAST_CHANNEL
            } else {
                lower.base
            };
            assert!(
                last < upper.base,
                "{} runs to {last:#x} and collides with {}",
                lower.name,
                upper.name
            );
        }
    }
}
