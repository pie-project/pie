use core::fmt;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    WorkspaceOverflow {
        what: &'static str,
        size: usize,
        alignment: usize,
        remaining: usize,
    },
    NegativeSpan {
        array: &'static str,
        index: usize,
        begin: i64,
        end: i64,
    },
    HeadsNotDivisible {
        num_qo_heads: u32,
        num_kv_heads: u32,
    },
    BatchExceedsPadded {
        new_batch_size: u64,
        padded_batch_size: u64,
    },
    EmptyBatch,
    IndptrTooShort {
        array: &'static str,
        needed: usize,
        got: usize,
    },
    TooManyWorks {
        total: i64,
        max: i64,
    },
    MergeCtasExceedSm {
        counter: i64,
        num_sm: i64,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkspaceOverflow {
                what,
                size,
                alignment,
                remaining,
            } => write!(
                f,
                "workspace overflow allocating {what}: {size} bytes at alignment {alignment}, \
                 {remaining} left -- grant a larger workspace"
            ),
            Self::NegativeSpan {
                array,
                index,
                begin,
                end,
            } => write!(
                f,
                "{array}[{}] = {end} is below {array}[{index}] = {begin}: a request cannot have \
                 negative length",
                index + 1
            ),
            Self::HeadsNotDivisible {
                num_qo_heads,
                num_kv_heads,
            } => write!(
                f,
                "num_qo_heads {num_qo_heads} is not divisible by num_kv_heads {num_kv_heads}"
            ),
            Self::BatchExceedsPadded {
                new_batch_size,
                padded_batch_size,
            } => write!(
                f,
                "the split produced {new_batch_size} work items but the plan reserved \
                 {padded_batch_size}: with a fixed split size, disable cuda graph"
            ),
            Self::EmptyBatch => {
                f.write_str("the batch is empty, and this planner divides by the batch size")
            }
            Self::IndptrTooShort { array, needed, got } => {
                write!(
                    f,
                    "{array} holds {got} entries, and the plan needs {needed}"
                )
            }
            Self::TooManyWorks { total, max } => {
                write!(
                    f,
                    "the schedule produced {total} work items, above the cap of {max}"
                )
            }
            Self::MergeCtasExceedSm { counter, num_sm } => {
                write!(
                    f,
                    "the schedule wants {counter} merge CTAs on a device with {num_sm} SMs"
                )
            }
        }
    }
}

impl std::error::Error for Error {}
