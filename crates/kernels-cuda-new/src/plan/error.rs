use core::fmt;

/// Why a plan could not be produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The int or float workspace could not hold one of the plan's arrays.
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
    BatchExceedsPadded {
        /// Work items the split produced.
        new_batch_size: u64,
        /// Slots the plan reserved for them.
        padded_batch_size: u64,
    },
    /// The batch is empty and the planner divides by its size.
    EmptyBatch,
    /// An `indptr` slice was shorter than `batch_size + 1`.
    IndptrTooShort {
        /// `"qo_indptr"`, `"kv_indptr"` or `"kv_len_arr"`.
        array: &'static str,
        /// Elements the planner needs.
        needed: usize,
        /// Elements it was given.
        got: usize,
    },
    /// More work items than the plan's fixed-size arrays hold.
    TooManyWorks {
        /// Work items the schedule produced.
        total: i64,
        /// The cap the arrays were sized for.
        max: i64,
    },
    /// MLA needed more merge CTAs than the device has SMs.
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
