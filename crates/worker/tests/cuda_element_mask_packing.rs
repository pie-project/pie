//! A caller's mask whose numerics are causal must produce the causal answer.
//!
//! # A byte where the kernel wanted a bit
//!
//! Both CUDA attention kernels read the custom mask BIT-packed, one bit per
//! `(query, key)` pair, and they say so in the same arithmetic:
//!
//! ```text
//!   // kernels/attn/attention_naive_paged.cuh, custom_mask_allows
//!   bit  = qo_off * kv_total + kv_idx;
//!   byte = mask_indptr[request_idx] + (bit >> 3);
//!   ((mask[byte] >> (bit & 7)) & 1) != 0
//!
//!   // kernels/flashinfer/attention/variants.cuh
//!   mask &= ((custom_mask_ptr[offset / 8] >> (offset % 8)) & 1);
//! ```
//!
//! `mask_indptr` is therefore a BYTE offset, and a request has to start on a
//! byte boundary or its first row reads its neighbour's tail.
//!
//! This engine published one BYTE per pair, with the CSR counting pairs. The
//! kernel then fetched pair `8i`'s byte for all of pairs `8i..8i+8` and took
//! bit `k % 8` of a value that is only ever 0 or 1 -- so one position in eight
//! was read at all, and the other seven were forced closed. Every attention
//! that consulted a caller's mask was attending to an eighth of its context.
//!
//! # Why nothing caught it
//!
//! The causal plan is published UNCONDITIONALLY, so that the unmasked arm can
//! be recorded under `GuardMode::Union` -- which captures both arms and aborts
//! if either's mask was never built. But the unmasked form's kernel is compiled
//! without custom-mask support and never dereferences the pointer. Nothing on
//! the exercised arm read the array, and the arm that read it had no fixture
//! running against it. The module's unit tests pinned the layout the packer
//! wrote rather than the layout the kernel reads, so they agreed with the bug.
//!
//! # What this gate does
//!
//! `tart-masked` is the mask-axis parity probe, and its `bisect` knob is what
//! makes it one: at `0` it supplies a dense causal mask on every fire, at `1`
//! it drops the decode mask, and at `2` it supplies none at all. Its mask is
//! constructed to be exactly causal, which is also what attention does with no
//! mask at all, so all three arms are obliged to answer the same thing.
//!
//! Before the repack they did not. At `max_tokens: 8` on qwen-3-0.6b the
//! masked arm answered `" wore of of of.. the."` and the unmasked one
//! `"<think>\nOkay, the user is asking"`. That gap is the whole defect, stated
//! as an assertion, and it is why this file compares ANSWERS across the knob
//! rather than checking any one of them against a fixed string -- a model or
//! snapshot change moves the text, and would not move the agreement.
//!
//! Run:
//!   cargo test --release -p worker --features engine-cuda-13 \
//!     --test cuda_element_mask_packing -- --ignored --nocapture

mod common;

#[test]
#[ignore = "real-hardware: needs an RTX GPU + --features engine-cuda-13 + a local model snapshot"]
fn a_causal_user_mask_answers_what_no_mask_answers() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let _worker = common::boot_cuda().await;

        // Eight tokens is enough for the disagreement to be unmistakable -- the
        // old masked arm diverged at the FIRST one -- and short enough that
        // three runs of it stay well inside the harness timeout.
        let mut answers = Vec::new();
        for bisect in [0u32, 1, 2] {
            let input = format!(
                r#"{{"prompt":"The capital of France is","max_tokens":8,"bisect":{bisect}}}"#
            );
            let text = common::spawn_inferlet("tart-masked", &input)
                .await
                .unwrap_or_else(|e| panic!("tart-masked bisect={bisect} failed: {e:?}"));
            assert!(
                !text.trim().is_empty(),
                "tart-masked bisect={bisect} answered nothing"
            );
            eprintln!("[cuda_element_mask_packing] bisect={bisect} => {text:?}");
            answers.push((bisect, text));
        }

        // The masks differ in COVERAGE across the three arms -- all fires, decode
        // only, none -- and not in what they permit, so any difference in the
        // answer is the mask path disagreeing with itself.
        let (_, reference) = &answers[2];
        for (bisect, text) in &answers[..2] {
            assert_eq!(
                text, reference,
                "a causal mask changed the answer (bisect={bisect}): the engine's \
                 mask layout no longer matches what the kernels read"
            );
        }
    });
}
