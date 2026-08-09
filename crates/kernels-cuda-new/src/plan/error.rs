//! What a planner refuses, and why refusing is the port's job.
//!
//! # `cudaError_t` was never the right return
//!
//! Upstream returns `cudaError_t` and throws `flashinfer::Error` — two failure
//! channels for one function, and the interesting failures use the *second*
//! one. `FLASHINFER_ERROR` fires when the workspace is too small, when
//! `qo_indptr` goes backwards, when `num_qo_heads % num_kv_heads != 0`, when
//! the schedule needs more work items than the buffer holds. None of those is
//! a CUDA error and none of them is recoverable by retrying — they are all
//! "the plan you asked for does not fit", and a caller that can size a
//! workspace can act on that.
//!
//! So this port has one channel, and it carries the numbers. The workspace
//! overflow in particular names the allocation, its size and what was left,
//! because the answer to it is almost always "grant more int workspace" and the
//! difference between a 400 KiB grant and a 4 MiB one is exactly these numbers.
//!
//! # Three of these are places upstream has no behaviour at all
//!
//! [`Error::EmptyBatch`] is the sharpest: `PrefillSplitQOKVIndptr` computes
//! `sum_packed_qo_len / batch_size` and `MLAPlan` computes
//! `accum_packed_qo_len / batch_size`, so a batch of zero requests is an
//! integer division by zero — `SIGFPE` on x86-64, not an exception, not a
//! `cudaError_t`. `tests/plan.rs` runs the C++ on an empty batch in its own
//! process and records that it dies; this port returns an error instead. That
//! is a deliberate deviation and it is the only kind this port makes: where
//! upstream has undefined behaviour there is no byte sequence to be faithful
//! to.

use core::fmt;

/// Why a plan could not be produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The int or float workspace could not hold one of the plan's arrays.
    ///
    /// `FLASHINFER_ERROR("Buffer overflow when allocating memory for ...")`.
    /// The name is upstream's allocation label — `batch_prefill_merge_indptr`,
    /// `mla_q_indptr` — so a grep for it lands in the C++ this replaced.
    WorkspaceOverflow {
        /// Upstream's label for the allocation that did not fit.
        what: &'static str,
        /// Bytes requested.
        size: usize,
        /// Alignment requested (1 or 16 — no call site asks for anything else).
        alignment: usize,
        /// Bytes left in the arena, before alignment padding.
        remaining: usize,
    },
    /// An `indptr` array went backwards, so a request had a negative length.
    ///
    /// Upstream checks this per element and reports the pair; so do we, because
    /// the usual cause is a caller that built the array in device order and
    /// read it back in host order.
    NegativeSpan {
        /// `"qo_indptr"` or `"kv_indptr"`.
        array: &'static str,
        /// The index `i` for which `array[i + 1] < array[i]`.
        index: usize,
        /// `array[i]`.
        begin: i64,
        /// `array[i + 1]`.
        end: i64,
    },
    /// `num_qo_heads % num_kv_heads != 0`, so there is no GQA group size.
    HeadsNotDivisible {
        /// Query/output heads.
        num_qo_heads: u32,
        /// Key/value heads.
        num_kv_heads: u32,
    },
    /// The schedule produced more work items than the padded batch holds.
    ///
    /// Upstream's `FLASHINFER_CHECK(new_batch_size <= padded_batch_size)`,
    /// which fires when a fixed split size is combined with CUDA graphs.
    BatchExceedsPadded {
        /// Work items the split produced.
        new_batch_size: u64,
        /// Slots the plan reserved for them.
        padded_batch_size: u64,
    },
    /// The batch is empty and the planner divides by its size.
    ///
    /// Not an upstream error — upstream divides by zero. See the module note.
    EmptyBatch,
    /// An `indptr` slice was shorter than `batch_size + 1`.
    ///
    /// Upstream reads `indptr_h[batch_size]` off a raw pointer and cannot
    /// notice; a slice can, and this is the one refusal here that has no
    /// counterpart in the C++ at all.
    IndptrTooShort {
        /// `"qo_indptr"`, `"kv_indptr"` or `"kv_len_arr"`.
        array: &'static str,
        /// Elements the planner needs.
        needed: usize,
        /// Elements it was given.
        got: usize,
    },
    /// More work items than the plan's fixed-size arrays hold.
    ///
    /// MLA's `max_total_num_works = 16384`. Upstream computes the arrays,
    /// allocates for the cap, and copies the computed ones — so exceeding it
    /// writes past the allocation; this refuses instead.
    TooManyWorks {
        /// Work items the schedule produced.
        total: i64,
        /// The cap the arrays were sized for.
        max: i64,
    },
    /// MLA needed more merge CTAs than the device has SMs.
    ///
    /// Upstream's `FLASHINFER_CHECK(merge_cta_counter <= num_sm, "Internal
    /// Error ... please report this bug to the developers")`. Reachable, and
    /// the bound is a proof in a comment rather than an invariant of the
    /// inputs, so it is checked.
    MergeCtasExceedSm {
        /// Merge CTAs the schedule wanted.
        counter: i64,
        /// SMs available.
        num_sm: i64,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkspaceOverflow { what, size, alignment, remaining } => write!(
                f,
                "workspace overflow allocating {what}: {size} bytes at alignment {alignment}, \
                 {remaining} left -- grant a larger workspace"
            ),
            Self::NegativeSpan { array, index, begin, end } => write!(
                f,
                "{array}[{}] = {end} is below {array}[{index}] = {begin}: a request cannot have \
                 negative length",
                index + 1
            ),
            Self::HeadsNotDivisible { num_qo_heads, num_kv_heads } => write!(
                f,
                "num_qo_heads {num_qo_heads} is not divisible by num_kv_heads {num_kv_heads}"
            ),
            Self::BatchExceedsPadded { new_batch_size, padded_batch_size } => write!(
                f,
                "the split produced {new_batch_size} work items but the plan reserved \
                 {padded_batch_size}: with a fixed split size, disable cuda graph"
            ),
            Self::EmptyBatch => {
                f.write_str("the batch is empty, and this planner divides by the batch size")
            }
            Self::IndptrTooShort { array, needed, got } => {
                write!(f, "{array} holds {got} entries, and the plan needs {needed}")
            }
            Self::TooManyWorks { total, max } => {
                write!(f, "the schedule produced {total} work items, above the cap of {max}")
            }
            Self::MergeCtasExceedSm { counter, num_sm } => write!(
                f,
                "the schedule wants {counter} merge CTAs on a device with {num_sm} SMs"
            ),
        }
    }
}

impl std::error::Error for Error {}
