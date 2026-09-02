//! **A GGUF K-QUANT WEIGHT SERVES AS STORED, END TO END** (QNF wave, alto
//! `next.md` §J5) — the connective tissue between a model text that declares
//! the stored super-block variants and `kernels_cuda::linear::kquant`, held against the
//! tree's own host decode of the same bytes.
//!
//! ```text
//! cargo test -p engine-cuda --features cuda \
//!   --test a_stored_k_quant_row_serves_as_stored -- --nocapture
//! ```
//!
//! # What is under test, and what already was
//!
//! The kernels are golden (`kernels-cuda/tests/kquant_matmul.rs`, all five
//! schemes against a transcribed block decode). What had never run is
//! everything between a declaration and a launch:
//!
//! ```text
//! Dtype::U4g32k   the text's word
//!   -> Weight::planes            one plane, [n, Dtype::row_bytes(k)] BYTES
//!   -> checkpoint_dsl::encoding  -> qnf::scheme_of_sig -> GgufQ4K
//!   -> the ladder's identity rung (stored == wanted: no cast, no decode)
//!   -> weights::plane_bytes      the rectangle IS the byte count
//!   -> WeightRow::Dense at the DECLARED dtype
//!   -> Run::maybe_stored         re-badged as the U8 byte rectangle
//!   -> linear::kquant::{matmul, lm_head}
//! ```
//!
//! # THE ORACLE IS THE SAME FILE, READ THE OTHER WAY
//!
//! One checkpoint, two texts. The stored text declares its projection `q4_k`
//! and its head `q6_k` and serves them as they lie; the reference text
//! declares both `Bf16` over the SAME tensors, and the checkpoint's ladder
//! takes its `Quant -> Raw` rung — the dequant door — decoding each block on
//! the host at load and handing cuBLAS a dense rectangle.
//!
//! That is what makes this a gate rather than a self-comparison. The
//! reference decode is `checkpoint::executor::walk`'s, which carries its own
//! bit-identity evidence against the `gguf` package; nothing in this file
//! decodes anything, so a wrong super-block width, a wrong plane rectangle or
//! a weight bound at the wrong address fails here and cannot be papered over
//! by a matching mistake in an oracle written beside it.
//!
//! **THE SAME NUMBERS, NOT THE SAME BITS.** The two arms round differently on
//! purpose: the host decode lands each weight in BF16 (`logical_dtype`) and
//! the fused point decodes in f32 inside the dot, so the honest comparison is
//! against the logit row's own spread — §J4a-1's ruling, where a 13.6%
//! element turned out to be a cancellation point measured with the wrong
//! ruler.
//!
//! # Gating
//!
//! Skipped at run time with no device, as `routed_experts_stream.rs` is, and
//! for its reason: an `#[ignore]` on the one box that could run it is a test
//! nobody runs. Nothing here reads a checkpoint off disk — the container is
//! written from the trace's own params into a scratch directory.

use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Platform, Request, Value, Weight, ops,
    trace_hybrid,
};
use model_ir::Trace;

// ─────────────────────────────────────────────────────────────────────────
// The text
// ─────────────────────────────────────────────────────────────────────────

/// Rows of the head, and of the embedding table.
const VOCAB: u32 = 1024;

/// The contracted axis of both projections — two whole ggml super-blocks, so
/// every scheme's row width is `2 x its block` and the `k % 256` refusal in
/// `linear::kquant` is not what this gate measures.
const HIDDEN: u64 = 512;

/// `q4_k`, spelled. The mangled form is the format's only name and `sig` is a
/// `const fn`, so a typo here does not compile.
const Q4_K: Dtype = Dtype::U4g32k;

/// `q6_k` — what a Q4_K_M mix stores `output.weight` at, which is why the
/// head is the scheme's busiest consumer and why it is the head's row here.
const Q6_K: Dtype = Dtype::I6g16k;

/// The fact vocabulary a three-op trace needs: none.
struct NoFacts;

impl Classify for NoFacts {
    fn of(_: &Request) -> NoFacts {
        NoFacts
    }
    fn word(&self) -> u64 {
        0
    }
}

/// Which way the two banks are declared.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arm {
    /// `q4_k` and `q6_k`: served as the file stores them.
    Stored,
    /// `Bf16` over the same tensors: the checkpoint's dequant door decodes
    /// every block at load and cuBLAS folds the rectangle.
    Decoded,
}

/// **A MODEL NOBODY SHIPS, AND DELIBERATELY NOT A CATALOG ROW.** Every shipped
/// family declares its embedding table in the weight representation `w`, and
/// there is no embed-gather point over a braided K-quant block — the gather
/// twins are the affine family's. So a text that quantizes its PROJECTIONS
/// and leaves its table alone is not a parameter any catalog row takes, and
/// writing one here is smaller than teaching one to a family for a gate.
///
/// Three ops, which is every op this wave touched: a gather to make a token
/// rectangle, the `linear.matmul` arm, and the `linear.lm_head` arm.
struct Micro {
    embed: Weight,
    proj: Weight,
    head: Weight,
}

impl Micro {
    fn new(arm: Arm) -> Micro {
        let (proj, head) = match arm {
            Arm::Stored => (Q4_K, Q6_K),
            Arm::Decoded => (Dtype::Bf16, Dtype::Bf16),
        };
        Micro {
            embed: Weight::sym("embed", [u64::from(VOCAB), HIDDEN], Dtype::Bf16),
            // The logical rectangle, in both arms and at both widths. What a
            // stored block costs is `Dtype::row_bytes(k)` and `Weight::planes`
            // is the one place that folding happens.
            proj: Weight::sym("proj", [HIDDEN, HIDDEN], proj),
            head: Weight::sym("lm_head", [u64::from(VOCAB), HIDDEN], head),
        }
    }

}

impl ForwardHybrid for Micro {
    type Facts = NoFacts;

    fn caches(&self) -> HybridSpec {
        HybridSpec::new()
    }

    fn forward(&self, inputs: Input<NoFacts>) -> Value {
        let x = ops::layout::embed(&inputs.tokens(), &self.embed, VOCAB);
        let h = ops::linear::matmul(&x, &self.proj);
        ops::linear::lm_head(&h, &self.head)
    }
}

fn trace(arm: Arm) -> (Micro, Trace) {
    let m = Micro::new(arm);
    let trace = trace_hybrid("kquant-micro", &m, Platform::Cuda);
    (m, trace)
}

// ─────────────────────────────────────────────────────────────────────────
// The checkpoint
// ─────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────
// The fire
// ─────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────
// (0) the host half — no device, and it runs on every `cargo test`
// ─────────────────────────────────────────────────────────────────────────

/// **THE DECLARATION FOLDS INTO THE CONTAINER, AND THE NUMBERS ARE ggml's.**
///
/// The trace is the artifact the engine is handed, so what it says about a
/// stored block is the whole of what the shell can know: one plane, no
/// companions, and a rectangle whose width is the row's BYTES. 288 is two
/// `block_q4_K`s and 420 is two `block_q6_K`s, and neither number is written
/// anywhere in this tree — both fall out of the term.
#[test]
fn a_stored_declaration_interns_one_byte_rectangle() {
    let (_, stored) = trace(Arm::Stored);
    let plane = |name: &str| {
        stored
            .params
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("the trace interns `{name}`"))
    };
    assert_eq!(plane("proj").shape, vec![HIDDEN, 288]);
    assert_eq!(plane("proj").dtype, Q4_K);
    assert_eq!(plane("lm_head").shape, vec![u64::from(VOCAB), 420]);
    assert_eq!(plane("lm_head").dtype, Q6_K);
    // No `.scales`, no `.biases`: a braided block has no companion to name.
    assert_eq!(stored.params.len(), 3, "three planes for three weights");

    // And the reference text says the logical rectangle over the same names,
    // which is what makes the two arms one checkpoint.
    let (_, decoded) = trace(Arm::Decoded);
    assert_eq!(plane_of(&decoded, "proj").shape, vec![HIDDEN, HIDDEN]);
    assert_eq!(plane_of(&decoded, "proj").dtype, Dtype::Bf16);
}

fn plane_of<'a>(trace: &'a Trace, name: &str) -> &'a model_ir::Param {
    trace
        .params
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("the trace interns `{name}`"))
}

// ─────────────────────────────────────────────────────────────────────────
// (1) the device half
// ─────────────────────────────────────────────────────────────────────────

