//! Real-hardware #4 validation -- production-inferlet canaries on `cuda_native`
//! (Lane C).
//!
//! Proves what the mock engine CANNOT: that a curated inferlet driving the
//! paged-KV / CoW forward path produces real coherent tokens on real silicon,
//! via the result-captured canary harness. (The former CAS-index dedup half of
//! this lane sampled the live CAS index through `::runtime::arena` /
//! `working_set::kv_cas`, an introspection surface the runtime no longer
//! exposes; the prefix-dedup contract is covered by the runtime's prefix-cache
//! e2e and the prefix-heavy benchmark gate.)
//!
//! Shares the `common` cuda harness (`boot_cuda` + `install_inferlet` +
//! `spawn_inferlet`). One boot per process (global runtime state). Run warm:
//!   cargo test -p worker --features engine-cuda-13 --test cuda_canaries -- --ignored --nocapture
//!
//! # The census, taken on a 4090 with CUDA 13 and qwen-3-0.6b
//!
//! This file used to name four fixtures -- `text-completion`, `best-of-n`,
//! `text-completion-spec` and `demo-persistent-kv` -- and NONE of the four is
//! a member of `tests/inferlets` any more, so it had not reached a GPU in a
//! long time. Picking replacements meant asking every plausible survivor what
//! it actually does on this engine.
//!
//! The census below is the SECOND one. The first was taken before
//! `cuda_host_writer_channels`' deadlock fix and listed four fixtures as
//! answering nothing within 180 s; two of those were the deadlock and two were
//! never wedged at all. This one covers all 37 members of
//! `tests/inferlets/Cargo.toml`, one boot, 8 tokens each, 75-second cap.
//!
//! Nineteen run and answer coherently: `asap-grammar-aligned-decoding`,
//! `cacheback-speculative-decoding`, `chat-completion`, `contrastive-decoding`,
//! `entropy-adaptive-temperature`, `eta-epsilon-sampling`,
//! `greenlist-watermarking`, `mirostat-v2-sampling`, `naive-baseline`,
//! `repetition-penalty`, `dry-repetition-penalty`, `gumbel-watermark`,
//! `synthid-tournament-sampling`, `classifier-free-guidance`,
//! `sampling-primitives`, `text-completion-bench`, `token-healing`,
//! `top-a-sampling`, `xtc-sampling`.
//!
//! The rest fall into five classes, and the class is the useful unit -- a test
//! that failed for five reasons at once would teach nothing:
//!
//! | class | fixtures | what it is |
//! |---|---|---|
//! | `fire geometry: EmbedTokens is not host-derivable` | `attention-sink`, `consensus-decoding`, `naive-masked`, `sliding-window-attention` | the pool-owned device-geometry class this engine deliberately does not claim |
//! | emitter declines a single-op library lift | `beam-search`, `locally-typical-sampling`, `tail-free-sampling` | all three fold a `top_k`, and this engine has no `top_k` kernel |
//! | a capability this engine does not claim | `tova-attention`, `trackb-h2o`, `trackb-snapkv` (`attn_score`), `mtp-speculative-decoding` (`mtp_logits`), `quest-attention` (`envelope_dot`) | five bits, all hardcoded `false` in one block of `serve/load.rs` -- see below |
//! | budget too small to finish | `constrained-speculative-decoding`, `json-schema-constrained-decoding` | eight tokens does not close a JSON object; not a defect |
//!
//! That row used to be two, and both said the wrong thing. It read
//! `model-gated intrinsic attn_score unavailable on this model` as "a MODEL
//! fact -- qwen-3-0.6b has neither", and `the backend's model profile does not
//! advertise this kernel` as something "one layer lower". Neither is about the
//! checkpoint. `ModelProfile` is built in `runtime/src/pipeline/program.rs` by
//! copying the ENGINE's `EtaCaps` field for field, and this engine states
//! `has_attn_score: false`, `has_attn_page_mask: false`, `has_kv_envelopes:
//! false`, `has_mtp_logits: false` and `has_value_head: false` as literals, in
//! one block of `serve/load.rs`, under a comment that says exactly why: each is
//! a claim a program binds against, so a false one is a silent no-op. Five
//! fixtures, one cause. Nobody needs to go looking in the checkpoint.
//!
//! Both refusals SAY that now -- the `IntrinsicUnavailable` and
//! `KernelUnavailable` that `eta-ir`'s `validate` raises (this crate does
//! not link it; the strings arrive over the wire) name the engine's
//! `EtaCaps` and add, of the intrinsic one, that it "says nothing about the
//! checkpoint". The census quotes the old wording above because that is what
//! the run printed when it was taken.
//!
//! `PIE_CUDA_KV_ENVELOPES=1` does not change it. The refusal text used to name
//! that variable and invite the attempt, which is why this was tried: all four
//! of `quest-attention`, `trackb-h2o`, `trackb-snapkv` and `tova-attention`
//! refuse identically with it set and unset. The knob is read by
//! `pools/kv_cache.rs::envelopes_requested`, where it decides whether the
//! memory planner RESERVES envelope bytes; it never reaches the advertised
//! capability, and it could not, because the capability is a literal. The
//! invitation is gone from the message -- that measurement is what removed it.
//!
//! How much stands behind the bit, for whoever picks this up: envelope
//! MAINTENANCE is real, in `kernels-cuda/src/attn.rs`, where the write-KV
//! kernels call `envelope_merge_written` and `envelope_update_appended` gated
//! on the `KvHasEnvelopes` ask. There is a hole -- `write_kv_explicit_bf16_
//! devwin` asserts `!has_envelopes` with "envelope maintenance not yet"
//! against it -- and `envelope_dot` itself is not a hand-written kernel at all
//! but a second-party region the tensor compiler generates, whitelisted in
//! `codegen/cuda/validate.rs`. So this is unfinished work with most of a floor
//! under it, not a wall. Do not flip the literal to find out; the same rule
//! applies here as to `DEVICE_GEOMETRY_PORTS`, and for the same reason.
//!
//! Four exceeded the 75-second cap, and a cap cannot tell "wedged" from
//! "slow", so all four were asked directly with the `max_tokens` bisect:
//!
//! * `prefix-tree-kv-cache` is slow and always was -- it appends a page at a
//!   time and each growth reallocates the pool.
//! * `context-aware-decoding` raised two questions and both are answered.
//!
//!   `context_shift=0.000` does NOT mean the two lanes never differ. Swept
//!   across alpha at four tokens, `mean_kl` is 0.0000, 0.0001, 0.0043, 10.1375
//!   at alpha 0, 0.5, 2 and 8, and the shift reaches 0.500 at alpha 8. The
//!   contrast is real and monotone; the alpha = 0 identity holds exactly. At
//!   the default alpha of 0.5 the contrast is simply too weak to move an argmax
//!   in four tokens, which is a fact about the fixture's alpha, not a defect.
//!
//!   The cost was a defect, and a sharp one. The two lanes have the same fire
//!   SHAPE and different addresses, so they arrived with one `BucketKey` and
//!   two `capture_digest`s in strict alternation. The digest was a validity
//!   check on the single exec a bucket could hold, so each fire found the other
//!   lane's exec, refused it, EVICTED it, and paid a 535-launch capture plus a
//!   `cudaGraphInstantiate` to install one the next fire would evict in turn.
//!   Six fires, six captures, zero hits, five mismatches -- against a counter
//!   whose own doc says a nonzero value on a steady workload is a bug.
//!
//!   The digest discriminated the lanes perfectly; it was being used to reject
//!   instead of to select. It is half the cache key now, up to
//!   `SLOTS_PER_BUCKET` address-sets per shape, and eight tokens went from 61.6
//!   to 26.9 seconds -- 7.7 to 2.4 seconds on the marginal token. This gate
//!   itself dropped from 50 seconds to 32, so the sharing was not the
//!   contrastive decoder's alone.
//!
//!   What is left is ordinary: the misses that remain are `PlanEpoch` bumps,
//!   the KV pool growing under a lengthening sequence and invalidating every
//!   recorded address. That is the cache working as designed.
//! * `tart-masked` was WEDGED, and is the gate `cuda_seeded_channel_cursors`
//!   now stands over: a seeded host-writer channel never had its cursors
//!   published into the mirror, so the guest's second `put` found the ring
//!   permanently full. It runs to any length now. The MASK disagreement that
//!   probe then exposed -- a dense causal mask answering differently from no
//!   mask at all -- was a second defect, the engine packing the custom mask one
//!   byte per pair where both kernels read one BIT per pair, and it is fixed
//!   and gated by `cuda_element_mask_packing`.
//! * `lora-probe` is the one left, and it was the most misleading of the four.
//!   It answered nothing even at ONE token while its forward pass ran to
//!   completion -- 536 launches, `ended_ok=true`, GPU at 0% afterwards --
//!   because its PROGRAM was refused at registration and the epilogue that
//!   samples was therefore never compiled, never found, and never run. The
//!   guest waited on a token nobody was going to put.
//!
//!   Three defects were in that chain, all now fixed: the shared adopter
//!   applied METAL's boundary vocabulary to every backend, so `lora`,
//!   `attn_page_mask` and `envelope_dot` were all non-executable on CUDA; the
//!   library tag on the wire was a bare enum discriminant no engine could name;
//!   and a second-party region's emitter decline -- which is correct, there is
//!   no generated kernel for a sink -- was read as a compile failure.
//!
//!   `lora-probe` then REPORTED rather than hanging, and what it reported was
//!   a real capability gap: its prologue launches a generated region of its
//!   own, so the program needs two `Prepared`s and the fire built one. The
//!   fire now builds one per launching stage and commits once for the program,
//!   which is correct because every stage's `Prepared` reads the same cursors
//!   -- nothing advances them until that single commit.
//!
//!   Three MORE defects stood between that and an adapter that does anything,
//!   each hidden by the one in front of it, and all three now fixed:
//!
//!     1. The correction's operands were read as named SSA values only, and
//!        the projection input is an ARENA slot on qwen3. One operand of three
//!        was unresolvable, so the adapter phase staged nothing and the fire
//!        ran the base model.
//!     2. The staging was handed `q_heads`/`kv_heads` -- head COUNTS -- where
//!        it wanted the projection ROW STRIDES, and correctly refused *d_out
//!        2048 != q projection width 16*. The strides are now read off the
//!        lowering's own operands, which state them once.
//!     3. The correction's xAᵀ gate was `Scratch::attn_out`, which on a
//!        `EnginePinned` family is the attention output buffer. The adapter
//!        wrote its intermediate over the attention's rows.
//!
//!   `lora-probe` now applies a real delta: `adapter_scale` 0.0 answers
//!   coherently, 0.5 drifts, 2.0 is noise, which is what a random A and a
//!   random B should do to a 0.6B model.
//!
//!   THE PARITY GATE IS NOW WRITTEN and it is `tests/cuda_lora_parity.rs`:
//!   `lora-probe` at `adapter_scale: 0.0` answers `naive-baseline` byte for
//!   byte, on four fires of one process, and an adapter at `adapter_scale:
//!   8.0` changes the answer and reproduces itself. It became writable only
//!   once the thing that had been read as a LoRA defect for four segments was
//!   correctly located, and it was not in LoRA.
//!
//!   NOTHING IS OPEN ON THIS FIXTURE ANY MORE, and the last two things that
//!   were had one cause between them.
//!
//!   They looked like two. One was that a NONZERO adapter did not reproduce
//!   itself: at `adapter_scale: 0.5`, temperature collapsed so the draw was
//!   out of the picture, three runs of one build answered " a fictional
//!   series of novels an", " capital of capital of capital o" and " a country
//!   that is a countr". The other was read for a segment as a wandering
//!   SAMPLER: at `temperature: 1.0` and `adapter_scale: 0.0` the fixture
//!   answered " Paris" about fifteen fires in sixteen and " Senate", " N" or
//!   "____" otherwise, on a logits row that hashing showed to be
//!   bit-identical on every fire including the ones that answered otherwise.
//!
//!   Two defects were found and both are fixed.
//!
//!   The first: the correction's third operand. `gemm::lora_qkv_correction`
//!   computes `scale * (x A^T) B^T`, and the seam statement names `q` and
//!   `v` but not `x`, because a statement names what the attachment rewrites.
//!   `Buffers::assign` therefore freed x's block the moment the projection
//!   consumed it and `attn::split_qkv_bf16` wrote the K projection into it.
//!   It is pinned now, for a fire that carries an adapter, and `model --test
//!   lowering the_corrections_projection_input_is_not_recycled_under_it`
//!   holds it without hardware.
//!
//!   The second, which is the one that explains both symptoms: a cuBLAS call
//!   takes a HANDLE, and the handle carries the stream. `run_captured` moves
//!   `ctx.stream` into a conditional's body per launch, which is enough for a
//!   kernel, whose stream is an argument -- and does nothing for cuBLAS,
//!   whose stream was bound once per fire by `step_impl`. Every cuBLAS call
//!   inside a guard arm was recorded onto the outer capturing stream, into
//!   the graph's main body rather than the conditional node, so it ran
//!   unconditionally and unordered against the region it belongs to. The
//!   backbone never noticed because its projections are not guarded; the one
//!   guarded region that reaches cuBLAS is the adapter's, and the whole of
//!   the correction is cuBLAS.
//!
//!   That is also why it showed at `adapter_scale: 0.0`, where the delta is
//!   exactly zero and ought to be invisible. The site GEMM accumulates at
//!   beta 1.0, so a correction adding zero still READS q and writes it back,
//!   and doing that at an arbitrary point relative to rope put the pre-rope q
//!   back roughly one fire in sixteen. Nothing was wrong with the sampler.
//!
//!   With `run_captured` rebinding the handle per region: 32 fires of 32 at
//!   `temperature: 1.0` answer " Paris", and the captured path now agrees
//!   with `PIE_CUDA_SUPERGRAPH=0` to the token at `adapter_scale: 8.0` on
//!   every fire of every process. `tests/cuda_lora_parity.rs` asserts both.
//!
//!   A third thing was found while measuring the second, and it was not a
//!   wrong answer but a wrong PRICE, sitting on top of a wrong answer nobody
//!   had triggered yet.
//!
//!   `union_eligibility` refuses to cache an ungrouped adapter's exec, and
//!   `lora-probe` uses the solo path, so this fixture missed the bucket on
//!   every fire. It missed AFTER paying for it: two warm eager passes, a
//!   535-launch capture, a `cudaGraphInstantiate` and a launch, and only then
//!   was the exec handed to `Recordings::insert` and refused. `Ineligible`'s
//!   own doc says such a fire "stays eager"; it did not. Four tokens cost 17
//!   to 20 seconds against 4.2 for the same fire under
//!   `PIE_CUDA_SUPERGRAPH=0`. The eligibility check now runs before the warm
//!   loop, and `cuda_lora_parity` went from about 180 seconds to 52.
//!
//!   Moving that check exposed the wrong answer underneath it. Every path out
//!   of `capture_or_replay` that gives up handed the fire's own `lowered` to
//!   `run` -- and when the supergraph is on, that lowering is `GuardMode::
//!   Union`, which contains BOTH arms of every guard. `run` never reads
//!   `launch.cond`. So a failed `begin_capture`, a failed `instantiate`, a
//!   failed `exec.launch` or an abandoned capture did not degrade the fire to
//!   a slower correct path; it degraded it to a fast wrong one, silently. At
//!   `adapter_scale: 8.0` the union-eager walk answers " Navigation llii"
//!   against the resolved walk's " Navigationervicesii Instructions",
//!   reproducibly. All six of those paths now go through `run_resolved`,
//!   which lowers the fire's own rows a second time under `Resolve`.
//!
//!   The obvious next question -- if the exec is good, why not KEEP it? -- was
//!   measured and the answer is no. `capture_fingerprint` already mixes every
//!   determinant of the solo body's launch shape (lane count, ranks, d_in,
//!   d_out, site bits, token spans, both adapter pointers, the staging arena's
//!   base) and the adapter's scale rides in the re-staged bf16 cast, not in a
//!   baked argument -- so lifting `Ineligible::UngroupedLora` answers
//!   CORRECTLY: " Paris. The capital" at 0.0 and 0.5, " Navigationervicesii
//!   Instructions" at 8.0, twice each. It is simply slower: 5.3 to 10.2
//!   seconds against 4.2 to 4.5 eager, because a four-token request never
//!   amortizes one capture and one instantiate. The refusal stays. If a
//!   fixture ever generates long enough for the arithmetic to turn over, this
//!   is the paragraph to reread -- the correctness question is already
//!   answered.
//!
//!   Only the adapter's region is guarded on this fixture, so only LoRA could
//!   show it -- but the fallbacks are on the path of EVERY captured fire, and
//!   any family that puts an arm under a guard inherits the same trap.
//!
//!   How wide the stream half of it actually was is worth stating, because it
//!   was read for two segments as an adapter problem. `model --test lowering
//!   every_op_that_lowers_under_a_guard_is_accounted_for` enumerates what
//!   lowers under a guard on the live deployments, and `Matmul` is on the
//!   list: on a qwen3 decode the per-layer `.qkv` projection is guarded, the
//!   fused `attn::qkv_decode_qk_norm_rope_write_kv_bf16` arm against the
//!   unfused Matmul + `SplitQkv` + rope arm that an adapter forces. `Matmul`
//!   is cuBLAS. So what was landing in the graph's main body, unconditional
//!   and unordered, was not one correction. It was every layer's projection,
//!   racing the fused arm that writes the same q, k and v.
//!
//!   One thing is worth carrying to the Metal side. `a_generation_agrees_
//!   with_mlx_token_for_token` concludes from TOKENS that an M > 1 prefill
//!   does not reproduce itself, while `one_token_at_position_zero_agrees_
//!   with_mlx`, which compares LOGITS, is five for five. Read here, that
//!   pair of facts meant a defect downstream of a deterministic forward --
//!   and the guess it prompted, that the sampler was at fault, was wrong.
//!   The forward was deterministic because the fire that produced the hashed
//!   logits and the fire that produced the wrong token were not the same
//!   fire.
//!
//! Two of the five classes are worth reading twice.
//!
//! The `EmbedTokens` class is documented at the header of
//! `tests/gpu/tests/cuda_sliding_window_attention_e2e`, which is the gate that
//! is about it. Do not flip the capability bit to make these four pass; a
//! false capability is a silent wrong answer.
//!
//! The emitter class is INTENTIONAL and is not a hole:
//! `eta-compiler/tests/cuda_every_region_runs`'s `REFUSED` table names it
//! outright. A `RegionKind::Library` wrapping a single `top_k` would fall back
//! to `ptir_m1_execute`, whose single-threaded form is O(len^2) and does not
//! return at a real vocabulary, so refusing is the honest answer. What was
//! wrong was only that the refusal did not say WHICH op -- it does now, and
//! that is how three fixtures turned out to be one cause rather than three.
//!
//! # What `prefix-tree-kv-cache` cost, and why it is the canary
//!
//! It is the only surviving fixture that drives `copy_kv`, and it found three
//! separate defects in `engine-cuda`'s control path, each hidden by the one in
//! front of it:
//!
//! 1. `copy_kv` and `resize_pool` notified their completion without publishing
//!    a terminal outcome (`engine callback published before terminal outcome
//!    settled`), and `copy_state` bound its target to `_completion` and did
//!    neither -- the 850-second-hang shape `runtime::engine::backend`'s
//!    `settle_control` documents. All four control verbs settle through
//!    `serve::settle_control` now.
//! 2. Five of `copy_kv`'s refusals returned a bare `INVALID_ARGUMENT` with
//!    nothing on stderr, so the runtime's `status -1` was the whole of what
//!    anyone got. Each names itself now.
//! 3. The refusal those then revealed: a copy whose DESTINATION is one page
//!    above the elastic pool was rejected instead of growing it -- the exact
//!    defect `engine-vulkan/tests/device.rs` and `engine-wgpu/tests/serving.rs`
//!    each carry a device test for, arrived at from the third backend.
//!
//! # The test surface was dark, and that is why the migration stayed half done
//!
//! `cargo test -p <crate>` reports a build error, not a failure count, when a
//! `#[cfg(test)]` module or a `tests/*.rs` target fails to COMPILE. In a
//! workspace sweep that reads as noise. Entire crates' worth of gates had been
//! un-run for a long time on exactly that reading: `model-ir`, `model-compiler`,
//! `engine-metal`, six of `engine-cuda`'s seven test targets, both of
//! `kernels`', and `engine-vulkan` outright (a merge left two imports behind,
//! one of them naming a type that no longer exists).
//!
//! **`cargo check --workspace --all-targets` is the knob that surfaces this.**
//! Run it. It is not the same question as `cargo test`, and it is the one that
//! was not being asked.
//!
//! What the dark surface COST is the point. The no-ask migration -- which
//! deleted the `keys::*` / `ctx.ask` vocabulary and made every routine's
//! operands derive from its own `fn` signature, bound positionally through
//! `Source::Slot` -- was landed in six places and not in six others, and every
//! one of the six omissions had a test that would have named it:
//!
//! * `TraceBuilder::finish` runs `model_ir::kernels::check_plan`, whose
//!   `OpKind::Launch` arm refused every statement of a family with no backend
//!   segment. Semantic texts state kernels now (`canon::*`, `layout::embed_bf16`),
//!   so EVERY semantic family text panicked out of its own constructor. Nothing
//!   in the tree could call one.
//! * `site_table::derive_sites` anchored on the expert-indexed grouped GEMM.
//!   The CUDA reading of a MoE block does not state one -- `moe::
//!   flashinfer_cutlass_moe_bf16` fuses the whole leg -- so a repaired walk
//!   still derived an EMPTY site table for a mixture-of-experts model. Not a
//!   panic; a fire plan quietly missing a divergence. It anchors on
//!   `moe::topk_softmax` now, which both legs state.
//! * `bind::table`'s `#[cfg(test)]` hand arms -- the second opinion the derived
//!   column is diffed against -- still bound the pre-migration lists, and four
//!   `crossed()` assertions named symbols whose dtype suffix went away with
//!   their generic instantiation. A stale symbol asserts `crossed` and gets
//!   `false`, silently.
//! * `kernels`' own `routine_derivation` fixture, the file that PROVES a table
//!   row derives from a `fn` signature, spoke the ask contract throughout.
//! * `shader::COUNT` -- a vocabulary size asserted rather than narrated,
//!   precisely so it could not drift -- had drifted, because a test that does
//!   not compile does not assert.
//!
//! The pattern to carry: a gate that cannot compile is worse than a gate that
//! does not exist, because the absence of one is visible and the silence of the
//! other is not.

mod common;

#[test]
#[ignore = "real-hardware: needs an RTX GPU + --features engine-cuda-13 + a local model snapshot; one boot per process"]
fn cuda_inferlet_canaries() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let worker = common::boot_cuda().await;
        eprintln!("[cuda_canaries] engine up on {}", worker.url());

        // Direct coherence check first: `spawn_text` RETURNS the generated text
        // (unlike the pipeline canary below, which streams via `println!` and
        // returns a status string -- so for that one `Ok` == the pipeline ran).
        let program = common::install_inferlet("text-completion-bench").await;
        let text = common::spawn_text(&program, "The capital of France is", 16).await;
        eprintln!("[cuda canary] coherence text => {text:?}");
        let text = text.expect("text-completion-bench errored on cuda");
        assert!(
            !text.trim().is_empty(),
            "cuda decode produced empty text -- generation/forward regressed"
        );

        // KV reuse across branches, which is the paged-CoW claim this file is
        // for: `prefix-tree-kv-cache` appends along a shared prefix and branches
        // off it, so the pages under the common span are read by more than one
        // context AND the branch's destination lands one page past the last
        // prefill's high-water mark. That last part is what makes it the canary:
        // it is the only surviving fixture that drives `copy_kv` at all, and it
        // found three separate defects in this engine's control path (see the
        // census above).
        let reuse = common::spawn_inferlet("prefix-tree-kv-cache", r#"{"num_tokens":32}"#).await;
        eprintln!("[cuda canary] prefix-tree-kv-cache (shared-prefix pages) => {reuse:?}");
        reuse.expect("prefix-tree-kv-cache errored on cuda");

        worker.shutdown().await;
    });
}
