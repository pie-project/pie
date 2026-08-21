//! The parity §5.1 asks of an adapter, on real hardware.
//!
//! "With no adapters the answer is the base model's" is the one LoRA claim
//! that can be checked without a reference implementation, because the base
//! model is sitting right there. `lora-probe` and `naive-baseline` are the
//! same inferlet apart from the adapter: same prompt, same seed, same
//! temperature, same Gumbel-max draw, same epilogue. At `adapter_scale: 0.0`
//! the fixture seeds B as all zeros, so the correction term is EXACTLY zero
//! and the two must answer the same string.
//!
//! # Why this gate took five defects to become writable
//!
//! It could not be written before, and the reason is worth keeping because it
//! is the reason a gate is worth writing at all. `lora-probe` on CUDA spent a
//! long time answering the base model's token on most runs and something else
//! on roughly one in three, in the same process at the same seed. Five
//! separate defects sat between the fixture and an answer that means
//! anything, each hidden by the one in front of it: one `Prepared` per program
//! rather than per stage, an arena-resident operand `lora_pins` could not
//! read, head counts passed where projection row strides were wanted, an
//! xA-transpose gate aliased onto the driver-pinned attention output, and a
//! GEMM autotuner that tuned inside an in-flight graph capture and consulted
//! its throttle before its disk cache. Their histories are in
//! `driver-cuda/src/fire/{launch,lora,scratch}.rs`, `program/session.rs` and
//! `kernels-cuda/src/gemm/dense.rs`.
//!
//! # Why the temperature is collapsed
//!
//! `temperature: 0.01` scales the logits up a hundredfold before the Gumbel
//! noise is added, which makes the draw an argmax in everything but name. That
//! is not a way of making the gate easier to pass; it is a way of making it
//! test the forward instead of the sampler, and the two were confused here for
//! a long time.
//!
//! At `temperature: 1.0` this fixture answers " Paris" on about fifteen fires
//! in sixteen and " Senate" or " N" on the rest, and the reason is NOT the
//! forward: hashing the whole 151936-wide logits row at the sampled position
//! showed it bit-identical on every fire, argmax 12095, including on the fires
//! that answered otherwise. Collapsing the temperature takes the drift from
//! three in thirty-two to zero in thirty-two. Whatever wanders is downstream
//! of logits that do not, which puts it in the seeded Gumbel-max draw, and
//! that is a live defect this gate deliberately does not cover -- see the
//! `cuda_canaries` census. Asserting the sampler here would make an adapter
//! gate red for a sampler bug.
//!
//! # What is NOT asserted
//!
//! That a NONZERO adapter reproduces itself. It does not: at
//! `adapter_scale: 0.5`, with the temperature collapsed and the sampler thus
//! out of the picture, three separate processes answered " a fictional series
//! of novels an", " capital of capital of capital o" and " a country that is a
//! countr". That is not a small numeric wobble, it is a different computation,
//! and it is an open defect. The gate below asserts only that a nonzero
//! adapter CHANGES the answer, which is the part that holds.

mod common;

/// Pull `"text"` out of an inferlet's JSON answer without a schema.
fn text_of(out: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(out).expect("inferlet answered non-JSON");
    v.get("text")
        .and_then(serde_json::Value::as_str)
        .expect("inferlet answered no text")
        .to_string()
}

/// The prompt and the draw both fixed, so the only variable is the adapter.
fn input(scale: f64) -> String {
    format!(
        "{{\"prompt\":\"The capital of France is\",\"max_tokens\":6,\
         \"temperature\":0.01,\"seed\":7,\"adapter_scale\":{scale}}}"
    )
}

#[test]
#[ignore = "real-hardware: needs an RTX GPU + --features driver-cuda-13 + a local model snapshot; one boot per process"]
fn cuda_zero_adapter_answers_the_base_model() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let worker = common::boot_cuda().await;
        eprintln!("[lora_parity] engine up on {}", worker.url());

        let base = common::spawn_inferlet(
            "naive-baseline",
            "{\"prompt\":\"The capital of France is\",\"max_tokens\":6,\
             \"temperature\":0.01,\"seed\":7}",
        )
        .await
        .expect("naive-baseline errored on cuda");
        let base = text_of(&base);
        eprintln!("[lora_parity] base => {base:?}");
        assert!(!base.trim().is_empty(), "the base model answered nothing");

        // THE PARITY. A zero-B adapter is a correction that is exactly zero, so
        // this is not "close enough": it is the same string.
        let zero = common::spawn_inferlet("lora-probe", &input(0.0))
            .await
            .expect("lora-probe errored on cuda at adapter_scale 0.0");
        let zero = text_of(&zero);
        eprintln!("[lora_parity] zero-adapter => {zero:?}");
        assert_eq!(
            zero, base,
            "a zero-B adapter changed the answer -- the correction term is \
             exactly zero, so the adapter path wrote something it should not \
             have, or read an operand that was not the one the lowering named"
        );

        // AND IT HOLDS ON REPEAT. The defect this gate was written after was
        // intermittent, so one agreeing fire is not evidence; the first fire of
        // a process took a different GEMM tactic from every fire after it until
        // `dense_tactic_for` was fixed, and a single-shot gate would have
        // passed throughout.
        for i in 1..4 {
            let again = common::spawn_inferlet("lora-probe", &input(0.0))
                .await
                .expect("lora-probe errored on cuda on a repeat");
            let again = text_of(&again);
            assert_eq!(
                again, base,
                "fire {i} of one process disagreed with the base model after \
                 fire 0 agreed -- the fire is not a function of its inputs"
            );
        }

        // AND THE ADAPTER IS NOT A NO-OP. Without this the gate above passes
        // just as well when the adapter is never staged at all, which is the
        // exact shape one of the five defects had: `lora_pins` answered `None`
        // and the fixture answered the base model perfectly.
        //
        // The scale is 8.0 and not 0.5 because the fixture's A and B are
        // random, so the correction at 0.5 is a real delta that is simply too
        // small to move the argmax on this prompt -- which it is entitled to
        // be. `adapter_scale: 0.5` DID move it before the projection input was
        // pinned, but only because the correction was reading the attention
        // output of the previous op, and a gate that passes on that is a gate
        // that fails when the defect is fixed.
        let live = common::spawn_inferlet("lora-probe", &input(8.0))
            .await
            .expect("lora-probe errored on cuda at adapter_scale 8.0");
        let live = text_of(&live);
        eprintln!("[lora_parity] live adapter => {live:?}");
        assert_ne!(
            live, base,
            "a nonzero adapter left the answer unchanged -- the correction is \
             not reaching the projections, and the parity above proves nothing"
        );

        // AND A NONZERO ADAPTER REPRODUCES ITSELF. This is the assertion the
        // gate could not make when it was first written: at the time, a
        // nonzero adapter answered something different on nearly every fire,
        // for two reasons that both took locating. The correction was reading
        // a projection input the arena had already recycled under it, and its
        // cuBLAS calls -- which is all of it -- were recorded onto the outer
        // capturing stream rather than the guard body's, so they ran
        // unconditionally and unordered against the region they belong to.
        //
        // Only the second of those two showed at `adapter_scale: 0.0`, and
        // only faintly: the site GEMM accumulates at beta 1.0, so even a
        // correction that adds exactly zero still reads q and writes it back,
        // and doing that at an arbitrary point relative to rope put the
        // pre-rope q back about one fire in sixteen. That is the whole of what
        // was recorded here as a wandering SAMPLER.
        for i in 1..3 {
            let again = common::spawn_inferlet("lora-probe", &input(8.0))
                .await
                .expect("lora-probe errored on cuda on a repeat at 8.0");
            let again = text_of(&again);
            assert_eq!(
                again, live,
                "fire {i} at adapter_scale 8.0 disagreed with fire 0 -- the \
                 correction is not a function of its inputs"
            );
        }
    });
}
