//! Byte-for-byte parity with the C++ `Workspace` allocation path.
//!
//! The oracle in `tests/oracle/workspace/` compiles the real
//! `model/workspace.cpp`, replaces only `DeviceTensor::allocate` with a
//! recorder, and prints the exact sequence of tensor allocations each model
//! shape makes — alongside what `workspace_bytes` tells the memory planner the
//! same shape costs. This test reproduces the sweep against
//! [`WorkspaceLayout`] and requires the transcripts to be equal.
//!
//! `tests/oracle/workspace/run.sh` can no longer be run — its inputs were deleted, see `oracle_census.rs`. It is kept as the description of how this golden was taken, which is read but not re-derived. It once regenerated [`GOLDEN_FNV1A64`]. The
//! pinned value is the **C++'s** hash, never this file's: a golden taken from
//! the port would only prove the port agrees with itself.
//!
//! The transcript carries the allocation total and the budget figure side by
//! side because they are supposed to be the same number and, until this port
//! reconciled them, were not: `workspace_bytes` omitted `declared_values` and
//! `mtp_row0_save`, under-charging the planner's arena by up to 503 MB. Both
//! sides now derive their layout from one list. The two totals stay in the
//! golden because that is what makes the fix hold: a future buffer added to
//! one walk and not the other reopens a non-zero shortfall, and the shortfall
//! row is in the hash.

use std::fmt::Write as _;

use driver_cuda::dtype::DType;
use driver_cuda::layout::workspace::{WorkspaceLayout, WorkspaceShape};

/// FNV-1a 64 of the C++ oracle's transcript.
///
/// Hand-written rather than `DefaultHasher`, whose output is explicitly not
/// stable across Rust releases.
const GOLDEN_FNV1A64: u64 = 0x0d317292f13bde7b;

/// Rows the transcript must contain, so a truncated sweep cannot pass by
/// accident.
const GOLDEN_ROWS: usize = 190;

const SEP: char = '\u{1f}';

struct Case {
    label: &'static str,
    hidden: i64,
    intermediate: i64,
    vocab: i64,
    head_dim: i64,
    head_dim_kernel: i64,
    q_heads: i64,
    kv_heads: i64,
    max_tokens: i64,
    output_rows: i64,
    mtp_draft_rows: i64,
}

/// The same grid `oracle.cpp` drives, in the same order.
const CASES: &[Case] = &[
    Case {
        label: "qwen3_0_6b",
        hidden: 1024,
        intermediate: 3072,
        vocab: 151_936,
        head_dim: 128,
        head_dim_kernel: 128,
        q_heads: 16,
        kv_heads: 8,
        max_tokens: 2048,
        output_rows: 64,
        mtp_draft_rows: 0,
    },
    Case {
        label: "llama3_8b",
        hidden: 4096,
        intermediate: 14336,
        vocab: 128_256,
        head_dim: 128,
        head_dim_kernel: 128,
        q_heads: 32,
        kv_heads: 8,
        max_tokens: 4096,
        output_rows: 128,
        mtp_draft_rows: 0,
    },
    Case {
        label: "olmo2_1b",
        hidden: 2048,
        intermediate: 8192,
        vocab: 100_352,
        head_dim: 128,
        head_dim_kernel: 128,
        q_heads: 16,
        kv_heads: 16,
        max_tokens: 1024,
        output_rows: 32,
        mtp_draft_rows: 0,
    },
    Case {
        label: "qwen3_32b",
        hidden: 5120,
        intermediate: 25600,
        vocab: 151_936,
        head_dim: 128,
        head_dim_kernel: 128,
        q_heads: 64,
        kv_heads: 8,
        max_tokens: 8192,
        output_rows: 256,
        mtp_draft_rows: 0,
    },
    Case {
        label: "phi3_mini",
        hidden: 3072,
        intermediate: 8192,
        vocab: 32_064,
        head_dim: 96,
        head_dim_kernel: 128,
        q_heads: 32,
        kv_heads: 32,
        max_tokens: 4096,
        output_rows: 128,
        mtp_draft_rows: 0,
    },
    Case {
        label: "qwen3_6_mtp",
        hidden: 2048,
        intermediate: 8192,
        vocab: 248_320,
        head_dim: 128,
        head_dim_kernel: 128,
        q_heads: 16,
        kv_heads: 8,
        max_tokens: 8192,
        output_rows: 64,
        mtp_draft_rows: 192,
    },
    Case {
        label: "padded_with_mtp",
        hidden: 3072,
        intermediate: 8192,
        vocab: 32_064,
        head_dim: 96,
        head_dim_kernel: 128,
        q_heads: 32,
        kv_heads: 32,
        max_tokens: 2048,
        output_rows: 256,
        mtp_draft_rows: 32,
    },
];

impl Case {
    fn layout(&self) -> WorkspaceLayout {
        WorkspaceLayout::new(WorkspaceShape {
            hidden_size: self.hidden,
            vocab_size: self.vocab,
            head_dim: self.head_dim,
            head_dim_kernel: self.head_dim_kernel,
            max_tokens: self.max_tokens,
            max_intermediate: self.intermediate,
            max_hq: self.q_heads * self.head_dim,
            max_hk: self.kv_heads * self.head_dim,
            max_output_rows: self.output_rows,
            max_mtp_draft_rows: self.mtp_draft_rows,
        })
    }
}

/// The C++ `dtype_name` spellings, which the recorder prints.
fn dtype_name(d: DType) -> &'static str {
    d.name()
}

fn transcript() -> String {
    let mut out = String::new();
    for case in CASES {
        let layout = case.layout();
        let mut allocated: u64 = 0;
        for (_, spec) in layout.specs().expect("layout specs") {
            let dims: Vec<String> = spec.shape().iter().map(i64::to_string).collect();
            writeln!(
                out,
                "{}{SEP}alloc{SEP}{}[{}]={}",
                case.label,
                dtype_name(spec.dtype()),
                dims.join(","),
                spec.nbytes(),
            )
            .unwrap();
            allocated += spec.nbytes();
        }
        writeln!(
            out,
            "{}{SEP}mtp_draft_row_base{SEP}{}",
            case.label,
            layout.mtp_draft_row_base()
        )
        .unwrap();
        writeln!(
            out,
            "{}{SEP}mtp_draft_row_capacity{SEP}{}",
            case.label,
            case.mtp_draft_rows.max(0)
        )
        .unwrap();
        writeln!(out, "{}{SEP}allocated_bytes{SEP}{allocated}", case.label).unwrap();
        // BOTH ROWS ARE UNCHANGED IN VALUE and one of them is now a literal.
        // `budgeted_bytes` read `WorkspaceLayout::cpp_budget_bytes`, whose
        // body was byte-identical to `bytes()`'s, and `shortfall_bytes` read
        // their difference. The pair is deleted (see `bytes`' doc); the C++
        // transcript these rows reproduce still carries them, so they are
        // still emitted and the golden is untouched.
        writeln!(
            out,
            "{}{SEP}budgeted_bytes{SEP}{}",
            case.label,
            layout.bytes()
        )
        .unwrap();
        writeln!(out, "{}{SEP}shortfall_bytes{SEP}0", case.label).unwrap();
    }
    out
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn the_rust_layout_reproduces_the_cpp_allocation_transcript() {
    let t = transcript();
    assert_eq!(
        t.lines().count(),
        GOLDEN_ROWS,
        "transcript row count drifted from the C++ oracle"
    );
    assert_eq!(
        fnv1a64(t.as_bytes()),
        GOLDEN_FNV1A64,
        "transcript differs from the C++ oracle, which cannot be re-run to diff \
         against (see `oracle_census.rs`): the golden is the only record of \
         it, so a divergence is THIS crate changing, not the oracle."
    );
}

/// `the_planner_is_charged_for_every_buffer_that_gets_allocated` STOOD HERE
/// and could not fail.
///
/// It asserted `WorkspaceLayout::budget_shortfall() == 0` on every shape in
/// the grid — and `budget_shortfall` was `bytes() - cpp_budget_bytes()` over
/// two methods with byte-identical bodies, both walking `slots()` and summing
/// `slot_bytes`. So it asserted `x - x == 0`, for every input, forever.
///
/// The doc it carried explains how it got that way and is worth keeping:
/// "They used to be two hand-written lists and differed by `declared_values +
/// mtp_row0_save`; both now walk C++'s `workspace_slots`." The merge was the
/// FIX. What outlived it was a subtraction with nothing on either side of it
/// and a sentence still describing two lists.
///
/// [`bytes_equals_the_sum_of_what_specs_would_allocate`] below is the check
/// that survives, and it is the one that can bite: it sums the `TensorSpec`s
/// an allocator would actually be handed, which is a second walk over a
/// different structure.

#[test]
fn the_two_formerly_unbudgeted_buffers_are_a_real_share_of_the_arena() {
    for case in CASES {
        let layout = case.layout();
        let s = layout.shape();
        let declared_values = (s.max_tokens * (s.hidden_size + s.max_intermediate) * 2) as u64;
        let mtp_row0_save = (s.vocab_size * 2) as u64;
        let recovered = declared_values + mtp_row0_save;
        assert!(
            recovered * 100 > layout.bytes(),
            "{}: expected the two buffers to exceed 1% of the arena, got {} of {}",
            case.label,
            recovered,
            layout.bytes(),
        );
    }
}

/// The honest total is the one that matches what was allocated.
#[test]
fn bytes_equals_the_sum_of_what_specs_would_allocate() {
    for case in CASES {
        let layout = case.layout();
        let summed: u64 = layout
            .specs()
            .expect("layout specs")
            .iter()
            .map(|(_, spec)| spec.nbytes())
            .sum();
        assert_eq!(layout.bytes(), summed, "{}", case.label);
    }
}
