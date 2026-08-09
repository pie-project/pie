//! Turning logits into tokens.
//!
//! One row per launcher symbol. The words a row is written in —
//! [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are `kernels`'.

use kernels::kernel;
use kernels::operands;
use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // Produces TOKEN IDS, not logits: a greedy-decode fast path that never
    // materializes the vocab-wide row, which is why it is its own statement
    // rather than `lm_head` followed by an argmax.
    kernel!(lm_head_gemv_argmax_int8 "sample::lm_head_gemv_argmax_int8",
        operands = operands![
            hidden_states: Buf,
            lm_head_weight: I8s,
            scale_inv: F32s,
            token_ids: I32sMut,
            num_rows: I32,
            hidden: I32,
            vocab: I32,
            stream: Stream,
        ]),
    // The plain `sample::argmax_bf16` is deliberately NOT here, though CSM's
    // backbone fires it. A row was added and `the_table_is_exactly_the_dsl_
    // surface` rejected it: this table and `dsl::cuda` are the same set, and
    // a DSL statement is something a TRACE records. CSM's backbone is a
    // hand-written forward, so nothing traces that argmax and the statement
    // would have no caller. See EXPECTED_RESIDUE in
    // scripts/kernel-vocabulary-audit.py, which excuses it for this reason.
];
