//! `attn` in fn-world — §5 step 5's last family, and the largest by a wide
//! margin.
//!
//! Twenty-three `.cuh` roots, forty-one table rows and fifteen host programs
//! in `driver-cuda/src/fire/` when this file opened. **Zero rows remain**:
//! `table::attn::KERNELS` is empty, `table::ROW_TABLES` is `&[]`, and
//! `table::KERNELS` is `crate::x::SIGS` alone. That was the whole remaining
//! structural distance to north star step 6.
//!
//! The first pass took the self-contained leaves — five roots, six rows — and
//! everything since has been measured the same way and landed in the order
//! the code allowed rather than the order the plan named.
//!
//! # What crossed, and the arrangement each root got
//!
//! | root | rows declared | contracts | binds | table rows deleted |
//! |---|---|---|---|---|
//! | [`attn_sink`] | 2 | 2 | 2 | `lse_log2_to_ln`, `attention_sink_rescale` |
//! | [`attn_res`] | 1 | 1 | 1 | `attn_res_blend` |
//! | [`head_dim_pad`] | 2 | 2 | 2 | `pad_head_dim`, `strip_head_dim` |
//! | [`softcap`] | 2 | 1 | 1 | `logit_softcap` |
//! | [`split_packed`] | 2 | 0 | 0 | — |
//! | [`pack_dense_mask`] | 2 | 0 | 0 | — |
//! | [`dsa_indexer`] | 3 | 3 | 3 | `dsa_index_knorm_rope`, `dsa_index_q_rope`, `dsa_index_topk_mask` |
//! | [`mla_naive`] | 2 | 0 | 0 | — |
//! | [`kimi_mla`] | 2 | 2 | 2 | `kimi_split_kv_a_norm`, `kimi_split_q_b` |
//! | [`mla_paged`] | 2 | 0 | 0 | — |
//! | [`mla_fa2`] | 6 | 1 | 1 | `attention_mla` (unbound) |
//! | [`qkv_fused`] | 11 | 1 | 1 | `qkv_packed_post` |
//! | [`dsv4_compress`] | 10 | 3 | 1 | `dsv4_compress_gather_paged`, `dsv4_store_comp_entries` (both unbound), `combine_attn_outputs` |
//! | [`kv_paged`] | 20 | 4 | 3 | `write_kv_to_pages`, `write_kv_explicit_bf16`, `write_kv_explicit_bf16_devwin` (unbound), `dequant_kv_cache_layer_to_bf16_active` |
//! | [`page_compact`] | 2 | 2 | 2 | `count_kept`, `scan_and_scatter` |
//! | [`attention_naive`] | 3 | 2 | 2 | `mtp_shift_hidden`, `mtp_update_pending_hidden` |
//! | [`attention_flashinfer`] | 1 | 1 | 0 | `attn_score_fold_heads` |
//! | [`attention_naive_paged`] | 2 | 1 | 1 | `attention_naive_paged` |
//!
//! Plus three that are not rooted in this table at all: `split_qkv_devwin`
//! (a free `fn` beside [`attn_res_blend_bf16`], over
//! `crate::x::driver_internal::split_qkv_bf16`'s text), and the absorb pair,
//! which are `x::gemm`'s cuBLAS driver ops.
//!
//! **Eighteen roots, and NO row remains in `table/attn.rs`** — the six
//! FlashInfer launchers went in one index as driver ops, and MLA's absorb
//! pair in the next as two more, which is a different mechanism from every
//! crossing above them and is described where the contracts are declared.
//! The last was `attn::qkv_decode_qk_norm_rope_write_kv_bf16`, whose host
//! program had already moved here and which was waiting only on `Cx::q_out`.
//! Before those passes, twenty-three rows —
//! thirty-five five passes ago, `0dc8e9e9b` took
//! `attn::attention_xqa_decode_bf16_prepared` since, [`kimi_mla`] took two
//! here, [`mla_fa2`] took `attn::dispatch_attention_mla_bf16`, [`qkv_fused`]
//! took `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` and
//! [`dsv4_compress`] took three more. **Five roots moved no row at all**,
//! all five unit-only: [`split_packed`], [`pack_dense_mask`],
//! [`mla_naive`], [`mla_paged`] and now [`kv_paged`]. [`dsa_indexer`] left
//! that list when its three rows crossed.
//!
//! [`kv_paged`] is the largest root in the family by a wide margin — twenty
//! device rows, more than the next two roots together — and it crossed in
//! two named halves rather than one pass.
//!
//! **HALF A** was the unit: twenty rows, eight template families, and the
//! five `Specialisation`s that turned out to be a second copy of a decision
//! `driver-cuda` had always made in Rust. `device::SPECIALISED` is empty and
//! terminal because of it.
//!
//! **HALF B WAS THE MOVE, and the move is DONE**: seven host programs came
//! out of `driver-cuda/src/fire/kv_paged.rs` and are `pub unsafe fn`s in
//! [`kv_paged`]. None of them needed a driver RESOURCE, which is the whole
//! discriminator and is now `x/mod.rs`'s registration doc with this family
//! as its worked failure.
//!
//! (The first statement of Half A said the four rows were blocked because
//! `kernels-cuda-new` cannot call `driver-cuda`. **That is true and it is not
//! the reason** — the dependency runs the other way, and this file's
//! `x::layout` neighbours are already called from that very module. The
//! correction is at [`kv_paged`], in place.)
//!
//! **AND THE FOUR ROWS HAVE NOW CROSSED**, on four `Cx` queries that landed
//! for them: `first_token`, `num_pages_in_batch`, `w_page_d` and `w_off_d`.
//! All four had a PRODUCER in `AttnCtx` before fn-world existed — the
//! queries were the missing half, not the fill — which is what separates
//! them from `Cx::mla_layer`, and it is why the ask was worth making instead
//! of writing four `none:` arms that would each have shadowed a live
//! dispatch. `w_page_d` and `w_off_d` are NULL-CHECKED in the query, which
//! is the row's old `Source::AttnNonZero` predicate moved from the emitter
//! to the fact.
//!
//! `Cx::plan()` is exercised here for the first time in the migration and
//! came out right. It replaced six `Source::Attn` cells with one query,
//! because the CSR arrays, `row_valid` and `requests` describe one thing and
//! are read together.
//!
//! **Three of the four bind; `write_kv_explicit_bf16_devwin` is a `none:`
//! arm** and is the documented exception to `x/gemm.rs:1145`'s overlap rule,
//! safe for two measured reasons rather than one assumed one: its row stated
//! `Source::Unbound` on all nine operands so `abi.rs:810` skipped it whole
//! and no dispatch arm ever existed to shadow, and `dsl::cuda::
//! write_kv_explicit_devwin` has zero callers in the workspace. Its `win_d`
//! is missing its FILL and not merely its query.
//!
//! **Four roots moved no row at all**, all four unit-only:
//! [`split_packed`], [`pack_dense_mask`], [`mla_naive`] and [`mla_paged`].
//! [`dsa_indexer`] was the fifth and took all three of its rows since.
//!
//! **[`mla_fa2`] IS NO LONGER ONE OF THEM, and its row is the one that
//! mattered.** Crossing it deleted
//! `crates/kernels-cuda/csrc/src/attn/attention_mla.cu` — the last two
//! nvcc-compiled `<<<>>>` in the workspace — and took
//! `kernels-cuda/tests/sources.rs`' `EXPECTED` to **0**.
//!
//! It crossed **unbound**, and that is a real crossing rather than a
//! deferral: a contract with no `operands` is what `abi::emit_c_shim` reads
//! to mean "no ahead-of-time entry", so the shim entry the `.cu` was the
//! definition of stopped being emitted and the file stopped having a
//! declaration to satisfy. What the `none:` arm withholds is the FIRE, not
//! the crossing. Both host programs exist — [`mla_fa2`] here and
//! `driver-cuda/src/fire/mla_naive.rs` for sm_100 — and `Cx` states neither
//! the MLA cache layer, the plan handle, the attention workspace nor
//! `sm_scale`, which is verbatim the reason `executor_bind.rs:1519` gives for
//! the row never having been armed in the first place. **It never fired, and
//! a row that never fires while holding an nvcc translation unit hostage is
//! worth strictly less than the contract that replaces it.**
//!
//! **Eleven roots, and the eleventh is a NEW `.cuh`.**
//! `csrc/src/attn/attention_mla_fa2.cuh` is not one of the twenty-three: it
//! is an NVRTC root in `csrc/src/attn/fa2.cuh`'s idiom — includes, four
//! `using`s, one alias template, one `__device__` echo, no `__global__` of
//! ours — written so the six rows can name upstream's kernel. The root it
//! replaced is `crates/kernels-cuda/csrc/src/attn/attention_mla.cu`, which
//! was host program throughout and is now deleted.
//!
//! **Unit-only is a real arrangement and five roots now use it**, which is
//! enough to state the rule: a root whose host programs are already Rust and
//! already outside a `bind!` crosses as a `unit!` and nothing else. The device
//! text belongs where the family is; the program belongs where it already
//! runs. `x/driver_internal.rs` says the first half — *"the rows stay where
//! the device text is"* — and these five are the second.
//!
//! **The rule's consequence is a schedule — with one correction the owner
//! asked for.** The framing offered was that the rows left in
//! `attn::KERNELS` are exactly the roots with `bind!` consumers.
//! **`mla_naive` and `mla_paged` disagree, and `dsa_indexer` DID until its
//! three rows crossed**: each of the three arrived unit-only, with its host
//! programs Rust in `fire/`, and every one of its table rows STAYED. A row
//! is the trace-facing dispatch entry; it survives until a CONTRACT replaces
//! it, whatever world the host program lives in. `dsa_indexer` is the worked
//! demonstration of the second half of that sentence rather than a
//! counter-example to it: nothing about its host programs changed when the
//! rows went, except which crate they are in. **A `unit!` moves device text; only a contract retires a
//! row** — and [`mla_fa2`] is the correction to the earlier version of that
//! sentence, which said `bind!`. A contract with no `operands` retires the
//! shim entry whether or not it has a bind, which is precisely how the
//! unbindable row left.
//!
//! # §66 — the row count is the whole of what is left in the CUDA lane
//!
//! `kernels-cuda/native` is the ONLY switch over the entire nvcc and `.cpp`
//! surface in the workspace, and the only thing that turns it on is
//! `driver-cuda/bridge`, which is deletable when `ROW_TABLES` empties. So
//! nvcc-zero, `.cpp`-zero and step 6 half B are not three goals — they are
//! three consequences of `attn`'s twenty-eight, and `moe`'s four have gone.
//!
//! **THE LAST TWO `<<<>>>` IN THE TREE ARE GONE.** They were
//! `mla_naive_paged_kernel` and `mla_mma_paged_kernel` in `attention_mla.cu`,
//! behind `attn::dispatch_attention_mla_bf16`, and that row crossed here.
//! Their device text is [`mla_naive`]'s `.cuh` and their host program is
//! `driver-cuda/src/fire/mla_naive.rs`; neither moved. What moved is the FA2
//! arm, which had no Rust at all, and the row, which is what the file's
//! shim entry hung on. `kernels-cuda/csrc/CMakeLists.txt`'s
//! `PIE_CUDA_GRAPH_KERNEL_SOURCES` is now an empty list, and the only file
//! this repository still asks nvcc to compile is `moe/flashinfer_moe.cu` in
//! a different target.
//!
//! # `ArgValue::Bytes` — §5.1's standing question, ANSWERED and CLOSED, and
//! # not the way the warning predicted
//!
//! §5.1 warned eleven families that *a wrong bypass is a launch with a
//! garbage struct, not a type error*, and asked each of them to be the first
//! family-level caller. `attn` is that family, via `MLAParams`, and the
//! answer is that **the bypass was never what blocked it.**
//!
//! `by_value!`'s grammar required `tag = $tag:ident` and asserted
//! `Ty::$tag.needs_mirror()` — a closed list of six kinds that does not
//! include `MLAParams`. So **an open set of `Abi` impls was gated behind a
//! closed set of `Ty` tags**, and the obvious fix, a seventh variant, is the
//! one `x/abi.rs:415-417` argues against in its own words. The floor took the
//! other patch instead: an UNTAGGED arm, keeping every assertion and dropping
//! only the permission. [`mla_params`] is its first caller and the full
//! account is in that module's doc.
//!
//! `x/xqa.rs`'s `KvCacheList` was the tree's only `by_value!` and it worked
//! because `Ty::KvCacheLayerView` happened to already exist and happened to
//! mean roughly the right thing. That is why eleven families produced no
//! second one.
//!
//! What the measurement itself said, because it is worth the header:
//! `sizeof(MLAParams) = 288`, and a transcription would have written 248 —
//! **forty bytes short**, because `uint_fastdiv` measures twenty-four bytes
//! and not four, twice. And `PROFILER_PARAMS_DECL` sits in the middle of the
//! struct, expanding to a pointer or to nothing depending on a macro defined
//! in another file: `-DFLASHINFER_ENABLE_PROFILER=1` moves `sizeof` to 296
//! and every field after `work_indptr` by eight. That is `x/xqa.rs`'s
//! `ENABLE_4BIT_KV_CACHE` hazard in a second family, which makes it a pattern.
//!
//! `softcap` declares two device rows against one table row on purpose, and
//! §3.2's hazard is the reason — see [`softcap`].
//!
//! # What did NOT cross, and why — the useful half of this header
//!
//! **`attn/softcap.cuh` was blocked on the floor and is no longer.** Its cap
//! came from `Source::CtxNonZero("final_logit_softcap")` — `DispatchCtx`'s
//! field at `driver-cuda/src/bind/mod.rs:1161`, *"gemma's FINAL logit softcap
//! (`cap * tanh(x / cap)` over the logits)"* — and [`Facts`] had no method
//! that reached it. `Facts::final_logit_softcap() -> Option<f32>` landed in
//! `a46bdbbe0` with the driver impl `(self.ctx.final_logit_softcap > 0.0)
//! .then_some(…)`, which is `CtxNonZero`'s reading moved into the type: zero
//! is ABSENCE, not a cap of zero, and a cap of zero would scale every logit
//! to nothing. The root crossed in the same pass that asked for it.
//!
//! **A `none:` arm would have been wrong here and it is worth saying why**,
//! because the shape recurs: `none:` surfaces as
//! [`Route::Unbound`](crate::x::Route::Unbound) at model LOAD, so a `none:`
//! for a symbol that fires correctly today would have refused every Gemma-2,
//! Gemma-3 and Gemma-3n deployment at load. The row world is not a fallback
//! for a bind that cannot be written; it is where a kernel legitimately
//! stays until one can be, and asking for the accessor is what the floor's
//! *"say so with the exact patch you want"* is for.
//!
//! **`attn/split_packed.cuh` crossed as a unit first and finished later**,
//! and the paragraph that stood here is worth keeping as a retraction. It
//! said the devwin half *"**cannot** be bound and it is not a floor gap"*,
//! because *"[`Cx::arg_in`]/[`Cx::arg_out`] read `BoundLaunch::args`, which
//! `resolve_arg_windowed` has already offset by the region's first row, so a
//! bind would hand the kernel pointers it windows a second time."*
//!
//! **The premise is true of every symbol but this one.** `bind/mod.rs:3973`
//! resolves every arg of a kernel whose name ends `_devwin` at row 0 and
//! says why in the same breath — *"The `_devwin` forms are the stated
//! exception. Their contract is BASE pointers."* So `Cx` hands a bind
//! exactly the base pointers the kernel requires, and
//! `attn::split_qkv_bf16_devwin` is [`SPLIT_QKV_DEVWIN`], BOUND, with
//! [`split_qkv_bf16_devwin`] as its host program.
//!
//! The other half stays as it was: `attn::split_qkv_bf16` is
//! [`crate::x::driver_internal::split_qkv_bf16`] — the fourth arrangement,
//! no contract because no trace can state it. So the root is a unit with one
//! bind and one driver-internal firer, not a unit and nothing else.
//!
//! **`attn/pack_dense_mask.cuh` crossed the same way, and its blocker was
//! answered by `x/xqa.rs`'s precedent rather than by a floor patch.** The
//! mirror lives in [`params`], in THIS crate, for `x/xqa.rs`'s reason:
//! `unit!` has to name the type and `driver-cuda` depends on
//! `kernels-cuda-new` rather than the other way round. `driver-cuda`'s
//! `bind/abi.rs` now re-exports it, so there is one definition and one
//! measured layout. The `Abi` impl is hand-written beside the mirror and not
//! a `ptr_abi!` in `x/abi.rs`, because `ptr_abi!` is private to that module
//! and `x/abi.rs`'s own header asks for exactly this: *"adding a crossing
//! type means writing one impl, next to the kernel that needed it, and
//! nothing else in the tree changes."* **No floor patch was needed.**
//!
//! Both kernels are unit-only, like `split_packed`'s two: their
//! `table::driver_internal` rows are already gone, deleted with
//! `attn/pack_dense_mask.cu` and its `.hpp`, and
//! `driver-cuda/tests/launch_abi.rs:651-654` records why — *"Empty consumer
//! set on all five channels; not ported, per §60.1."* Nothing in the tree
//! launches either one today. So: `unit!` yes, `contract!` no, `bind!` no,
//! and a `none:` arm would be wrong for the reason §5.1 gives — it would
//! surface at model load as a refusal for a symbol no trace can state.
//!
//! # THE DEFECT THE MIRROR SURFACED: `Ty::StructuredMasks` names a type
//! # that no longer exists
//!
//! `kernels/src/lib.rs:1079` spells `Ty::StructuredMasks` as `"const
//! ::pie_cuda_driver::kernels::attn::StructuredMaskParams*"` — namespace
//! `attn`, which was `pack_dense_mask.hpp`'s. **That header is deleted**, and
//! a sweep of `*.hpp`, `*.cuh`, `*.cu` and `*.rs` finds exactly one
//! definition left: `attn::device::StructuredMaskParams`, at
//! `pack_dense_mask.cuh:136`. The host spelling names nothing.
//!
//! It is not a live break, because `emit::crossing` refuses the operand
//! before `Ty::cpp()` is ever asked for it, and the two rows that could have
//! asked are gone. It is a spelling waiting to be wrong. [`params`] states
//! the DEVICE spelling, which is the one NVRTC resolves, and records the
//! divergence rather than editing `crates/kernels` mid-sweep.
//!
//! `pack_dense_mask.cuh`'s own header claims a check that also died with the
//! `.cu`: *"`pack_dense_mask.cu` includes BOTH definitions and
//! `static_assert`s size, alignment and all three field offsets against each
//! other."* Those five `static_assert`s were the entire argument for a second
//! definition of a three-`u32` POD. The re-export removes the second
//! definition instead of restoring the check, which is the stronger answer.
//!
//! **`attn/page_compact.cuh` is a row that can cross and a program that
//! cannot yet.** Every one of its eleven operands is unsourced —
//! `scratch_counts` is a driver-owned scratch buffer and `keep_stride` comes
//! off a host CSR, and no `Source` spells either — so no `Cx` query reaches
//! them and the bind that would exist is the empty one. It is left for the
//! pass that moves `FirePageMask` with it; splitting a program from the only
//! caller that can supply its buffers would leave both halves half-done.
//!
//! **The FA2 and XQA lattices are deliberately untouched**, per §5.1: 56
//! units on one root and six on another, both already NVRTC-native, neither
//! blocking anything.
//!
//! # `Facts::plan()` and `Facts::slab()` were not exercised
//!
//! §5.1 names `attn` as where they are first used and most likely to be
//! wrong. They are not used HERE: all three roots this pass took are
//! pointwise or per-head corrections applied AFTER an attention kernel has
//! written its output, and none of them reads a page CSR or a state slab.
//! [`Cx::plan`] IS NOW EXERCISED, by [`kv_paged`], and it is right: its six
//! fields are exactly the four CSR arrays, `row_valid` and `num_requests`
//! that the two append rows source from `AttnCtx` one at a time, and
//! `bind/facts.rs:505` fills all six by direct field copy. It states no
//! `first_token`, which is the one thing those rows also need and the first
//! entry in [`kv_paged`]'s ask — a plan is what the fire's geometry is, not
//! where a partial write resumes from, so that is a separate query and not a
//! seventh field.
//! Agent `sweep-ssm` is exercising [`Cx::slab`] concurrently.
//!
//! # `ArgValue::Bytes` was not exercised either
//!
//! §5.1 names `attn` (`MLAParams`/`HopperParams`) as its first family-level
//! caller. Not in this pass: every parameter of these five kernels is a
//! pointer or a scalar, and `unit!` needs no new grammar for either. The
//! by-value aggregate is still owed a first caller — `x/xqa.rs`'s
//! `KvCacheList` is the only `by_value!` in the tree and it is agent
//! `xqa-nvrtc`'s. **The failure mode §5.1 warns about therefore remains
//! untested**: a wrong bypass is a launch with a garbage struct rather than
//! a type error, and the typecheck TU is the only thing that would catch it.
//!
//! # §3.2's two-formats-one-width hazard IS exercised, and this is the first
//!
//! `softcap` is the only place in `attn` where one template is instantiated
//! at two sixteen-bit formats, and §5.1 named it as live for this family.
//! What the hazard is, concretely: `bf16` and `f16` are both sixteen bits and
//! both `unsigned short` to any C ABI, so **the two rows the row world wrote
//! for them are byte-identical apart from the symbol string** — same
//! `LaunchRule::Elementwise`, same `in_place = &[(0, 0)]`, same three
//! operands at the same `Ty`s from the same `Source`s. The only thing that
//! told them apart was `DeviceKernel::elem`, which lives on the UNIT row and
//! not on the table row, and `Ty::BufMut` is a `void*` whose element type is
//! whatever that `elem` said. Feed a bf16 buffer to the f16 symbol and it
//! binds, launches, and computes `cap * tanh(x / cap)` over a reinterpretation
//! of the same bits — neither format has a trap representation, so the answer
//! is finite, plausible and wrong.
//!
//! In fn-world `x/abi.rs` makes them distinct unit structs, so
//! [`logit_softcap_bf16`] takes `*mut bf16`, [`logit_softcap_f16`] takes
//! `*mut f16`, and no caller can pass one for the other without writing a
//! cast. **The residue, stated rather than glossed:** what pairs a symbol
//! STRING with a type parameter is still a human decision, made in the two
//! host `fn`s below and nowhere else. The port does not eliminate it; it
//! reduces it from every call site to two adjacent lines, and it puts a
//! check on the other end — `unit!`'s `where [T = f16] "device::f16"` feeds
//! `abi::emit_device_typecheck`, which spells the parameter
//! `::pie_cuda_driver::kernels::device::f16*` against the `__global__`'s own,
//! so a row whose type disagrees with the DEVICE TEXT is a C++ compile error
//! naming the symbol. A row whose type disagrees with the host `fn`'s cast is
//! the one gap, and both are in this file, twenty lines apart.
//!
//!
//! Every launch below is a [`Launch`] literal or one of its two conveniences,
//! and every one cites the `<<<>>>` or the `LaunchRule` it came from. Four of
//! the five had **no host program at all** before this file — they were rows,
//! and the generated dispatch arm built their grid from the rule — so the
//! citation is the rule function in `runtime/launch.rs` plus the `<<<>>>` the
//! rule's own doc was checked against. Nothing here is invented.
//!
//!
//! # THE TWENTY-THREE ROWS THAT REMAIN, MEASURED — read this before the next
//! # pass, because "the FA2 lattice" is three different jobs wearing one name
//!
//! ## The lattice itself is DONE and was never the work
//!
//! `families::fa2` is **fifty-six units over one root** — twenty decode
//! points on `(head_dim, gqa_group)` and thirty-six prefill points on
//! `(head_dim, cta_tile_q, num_mma_kv)`, macro-generated by `decode_unit!`
//! and `prefill_unit!`, all sharing `attn/fa2.cuh`. Every row states no
//! `operands`; the family already carries its own resolution API
//! (`decode_unit_name`, `decode_symbol`, `prefill_unit_name`,
//! `prefill_symbol`). **There is nothing to port there.**
//!
//! One thing about it is worth a line because it contradicts a doc:
//! `Unit::name` is documented as *"its root's path under `csrc/src` without
//! the extension"*, and that is **false for exactly this family** — the
//! fifty-six are `attn/fa2_decode_hd128_g4` and its siblings and they share
//! one file. `Unit::file` is the path; `Unit::name` is a KEY. Every other
//! family's two happen to coincide, which is why the doc reads as a
//! definition rather than as the coincidence it is.
//!
//! ## The six FA2 LAUNCHERS are driver ops, and the third registration shape
//! ## does not reach them for free
//!
//! `dispatch_attention_flashinfer_decode`, `…_decode_capture`,
//! `…_prefill_bf16`, `…_prefill_capture_bf16`, `…_prefill_custom` and
//! `attention_flashinfer_prefill`. **Name the resource**, which is the test
//! `x/mod.rs` states, and it answers in one line for five of the six:
//!
//! ```text
//! cache: DecodePlanCache  <- Source::AttnPlan("decode")     five of six
//! cache: PrefillPlanCache <- Source::AttnPlan("prefill")
//! ```
//!
//! A `Box`ed, MUTABLE, cross-fire plan cache owned by `bind::DecodePlan` /
//! `bind::PrefillPlan` and planned by `Prepare::DecodePlan` /
//! `Prepare::PrefillPlan` before the fire. §3.3 keeps precisely that surface
//! out of `Cx`: a fact is a value a fire HAS, and this is state the driver
//! KEEPS. So these are driver ops by the same argument as `x::gemm`'s twelve.
//!
//! **THE SIXTH NAMES NOTHING, AND THAT IS THE FINDING OF THIS PASS.** The
//! discriminator was run across all six before any was written, which is the
//! remedy §75 asks for, and the sixth came out different rather than
//! agreeing. `attention_flashinfer_prefill` builds a `PrefillPlanCache` **on
//! its own stack**, drops it before it returns, allocates nothing that
//! outlives the call, and asks `plan_device()` — a read-only capability
//! query, not a pool, not a handle, not an allocator, not an arena. Run *name
//! the resource* and the line comes out empty.
//!
//! It is a driver op anyway, by the SECOND condition: **can a `Cx` state what
//! the body reads?** It cannot. The body walks `qo_indptr_h` and
//! `kv_page_indptr_h` — the HOST mirrors of a device CSR — on the CPU, to
//! learn `num_pages_in_batch`. No `Cx` query answers a host pointer, and
//! reading the device copy host-side is a synchronise, which §0 forbids
//! inside a fire.
//!
//! That is `split_qkv_bf16_devwin`'s shape and not `gemm`'s, and the two
//! together are now the evidence for the rule this family supplied in the
//! first place: **name the resource is NECESSARY AND NOT SUFFICIENT.** Five
//! of six pass on condition one; the sixth passes on condition two alone;
//! and a pass that had only checked the first member of the set would have
//! written the same arm for the wrong reason and left the rule looking
//! simpler than it is.
//!
//! **AND CROSSING THEM IS NOT FREE THE WAY `gemm`'S WAS.** This is the
//! finding, and it is the reason to read this paragraph rather than to start
//! typing. `x::gemm`'s twelve could take the third registration shape in one
//! commit because the driver ALREADY fired them by another route — a
//! lowering reaches `bind::quant_gemm::act_x_w` through the row's
//! `lowered_as`, so deleting the row deleted a shim entry nothing used.
//! **These six have exactly one caller and it is the GENERATED arm.**
//! `abi::emit_rust_dispatch` writes it from the row's `Source`s;
//! `Contract::sig` states no operands; so the contract that retires the row
//! also deletes the only thing that fires it, and `Route::Driver` falls
//! through to `bind/mod.rs`'s hand-written driver-op `match`, which has no
//! arm for any of them and answers `NoArm`.
//!
//! So the honest cost is six hand-written arms re-resolving thirteen to
//! nineteen operands each, including `Source::AttnPlan`, `Source::AttnWindow`
//! and `Source::Or(&Out(0), &Attn("o_out"))` — the three-arity output. That
//! is `emit_rust_dispatch`'s resolution written a second time, by hand, for
//! the six largest rows in the table, and *nothing is written twice* is the
//! sentence this migration exists to enforce. **It is a decision and not a
//! transcription, and it belongs to whoever owns the floor.**
//!
//! ### The ruling, and why it is not paying twice
//!
//! **`emit_rust_dispatch` dies with `bridge`.** That is north star step 6
//! half B, and `bridge` goes when `ROW_TABLES` empties — which is what these
//! rows are. So the six arms have to be hand-written *whatever order the
//! migration takes them in*; writing them at the crossing is paying step 6's
//! cost at the moment the information is in front of the reader rather than
//! three weeks later when it is not.
//!
//! **They are written and landed**, in `driver-cuda/src/bind/mod.rs`:
//! `fa2_decode`, `fa2_decode_capture`, `fa2_prefill`, `fa2_prefill_capture`,
//! `fa2_prefill_custom`, `fa2_prefill_planless`. Every operand came out of
//! `BoundLaunch::args`, `LaunchSpec` or `AttnCtx` — none needed a fact the
//! arm could not reach — and every guard the generated branch carried became
//! a refusal, in the generated order, because `&&` short-circuits and `a_of`
//! is an `expect`. Every one of those refusals precedes the arm's single
//! launch, which is free here and worth saying anyway: one call each, so
//! there is no partial device state a late `Declined` could misdescribe.
//!
//! ### What the six surfaced, and it is one thing worth a floor edit
//!
//! `Source::AttnPlan`'s rule is **`attn_plan`, a nested `fn` inside
//! `bind::dispatch_generated`** — reachable only from the generated `match`
//! in the same body. So are `a_of`, `kv_view`, `has_kv_layer`, `width_of` and
//! the `AlreadyConst` trait: the generated arm's whole vocabulary is private
//! to the function that dies with `bridge`.
//!
//! Five of those six are READS and a hand arm re-spells them in one line
//! each. `attn_plan` is **a decision**:
//!
//! ```text
//! decode: window_of(spec, a, layer) == -1 && !a.decode_plan_full.is_null()
//!             ? a.decode_plan_full : a.decode_plan
//! ```
//!
//! — the gemma-4 two-kind rule (512 vs 256 head dim), turning on a fact no
//! row can see. Copying it into a hand arm makes it *"a second copy of a
//! decision"*, which is the shape this family has already found twice
//! (`QKV_DECODE_BLOCK` and its six pins; the five base rows and their five
//! `Specialisation`s). So it **moved**, and not to a forwarder: `attn_plan`
//! is now module-level in `bind/mod.rs`, beside `window_of`, which it calls
//! and which was already module-level — the pair was split by nothing but
//! the order they were written in. The nested copy is **deleted**, and the
//! generated text needs no edit at all: it spells `attn_plan(a_of(attn),
//! spec, ..)` and now resolves to the one definition. A rule with one
//! spelling, reachable from both the generated arm and the hand arms, was
//! available the whole time and cost a deletion.
//!
//! ### And the wiring, which is one line each and not this file's
//!
//! The six arms are inert until `bind::dispatch`'s driver-op `match` names
//! them. **The rows must not be deleted before the arms are wired**:
//! `execution::SERVED` is what `x::route` reads to answer `Route::Driver`,
//! and a `Route::Driver` with no `match` arm is `NoArm` — every decode fire
//! in the tree. So the crossing of these six is one commit and not two, and
//! it landed as one:
//!
//! ```text
//! "attn::dispatch_attention_flashinfer_decode" =>
//!     fa2_decode(bound, spec, ctx, attn)?,
//! "attn::dispatch_attention_flashinfer_decode_capture" =>
//!     fa2_decode_capture(bound, spec, ctx, attn)?,
//! "attn::dispatch_attention_flashinfer_prefill_bf16" =>
//!     fa2_prefill(bound, spec, ctx, attn)?,
//! "attn::dispatch_attention_flashinfer_prefill_capture_bf16" =>
//!     fa2_prefill_capture(bound, spec, ctx, attn)?,
//! "attn::dispatch_attention_flashinfer_prefill_custom" =>
//!     fa2_prefill_custom(bound, spec, ctx, attn)?,
//! "attn::attention_flashinfer_prefill" =>
//!     fa2_prefill_planless(bound, spec, ctx, attn, rows)?,
//! ```
//!
//! And the six symbols left [`crate::execution`]'s `WALKED` for `SERVED`
//! with `Service::DriverOp` in the same index, because `x::route` reads
//! `SERVED` and there is no route a `Walk` produces — the test that refuses
//! a symbol which is both says why: **a walk and a service are two answers
//! to one question.** Their `refuses` lists did not travel with them and are
//! not lost: every string on them is a variant of
//! `fire::flashinfer_fa2_dispatch::Decline` now, and an enum the code
//! returns cannot go stale the way a list of strings beside it can.
//!
//! ### One more thing the six surfaced, and it is a gate
//!
//! `bind::service` was **`bridge`-gated** by `f38d199c2` — a live CI break,
//! found and fixed while these arms were being written: the module named
//! `DispatchCtx` twenty-eight times and `DispatchCtx` was
//! `#[cfg(feature = "bridge")]`. **The gate was the wrong half of an honest
//! fix.** North star §6 measured the archive's reach — one item, `abi::ffi`
//! — and `DispatchCtx`'s thirty-five fields name no archive type, so the
//! gated half was the one lying and both are `_cuda` now. The two did have
//! to agree; gating the CONSUMER is what spread the attribute to 68 items
//! that never reached the C++ at all. That module is *"the consumer that
//! makes the classification cost the C++ its
//! body"* — `bridge`'s whole subject.
//!
//! **But these six arms are not that.** They exist precisely so the six
//! symbols keep firing after `bridge` goes, and they are sitting in a module
//! that dies with it. So do the six entry points they call — which are not
//! cuBLAS bodies either: each is a thin resolution over
//! `crate::fire::flashinfer_fa2{,_dispatch}`, which is ungated already.
//!
//! The shape that fixes it is the one `dequant_kv_cache_layer_to_bf16_active`
//! and `attn_plan_for` both took: **the body moves to the surviving side.**
//! The six entry points and the six arms belong in
//! `crate::fire::flashinfer_fa2_dispatch`, leaving `bind::service` holding
//! only what `bridge` is actually about. It could not happen before the
//! crossing, because `every_rust_served_symbol_is_spelled_here` reads
//! `include_str!("service.rs")` and would fail the moment an entry point left
//! a file whose symbol is still on `RUST_SERVED` — so the move, the
//! `RUST_SERVED` removal and the row deletion are one commit, which is the
//! same commit the `match` arms landed in. **It has happened.**
//!
//! The two halves split on the gate and not on the seam, which is worth a
//! line because it is not where the plan put them. The ENTRY POINTS need
//! only `c_void`, `KvCacheLayerView`, `AttentionWorkspaceView`, `KvLayer`
//! and `merge_states`, all `_cuda`-tier, once the unused `_ctx:
//! &DispatchCtx` parameter goes — so they live in `fire`. The ARMS take
//! `&AttnCtx` and return `DispatchRefusal`, both `bridge`-gated AT THE TIME
//! (both `_cuda` since §6's re-gate), so they could not; they live in
//! `bind/mod.rs` beside the `match` that calls them and
//! beside `window_of` and `attn_plan`, which is where the plan-choosing rule
//! wanted to be anyway.
//!
//! ## `attention_naive_paged` — and a length literal of my own
//!
//! The one launcher in this block with no plan cache, no workspace and no
//! host mirror — *"Head dims flashinfer's prefill template rejects (gemma-4's
//! 512) take a naive paged kernel instead. No plan at all; fire-shaped."*
//!
//! **This section used to say "sixteen operands, fifteen of them queries" and
//! ask for the sixteenth.** The row has **fourteen**, and re-deriving them
//! against `table/attn.rs` is how the count was corrected. That is the same
//! class as `device.rs:991`'s hold, `DSV4_COMPRESS_SIGS[4]`'s sources,
//! `assert_eq!(checked, 14, "seven specialised kernels")` against five
//! entries, `RUST_SERVED`'s *"all four rows unsourced"* over a row stating
//! ten, and the devwin `Walk`'s *"fully sourced"* — **a literal that names a
//! length, written once and re-derived by nothing.** It is recorded here
//! rather than quietly fixed because this one is in the porting agent's own
//! prose, which is the strongest available evidence that the discriminator
//! has to be run across a whole set and not against the member in front of
//! you.
//!
//! All fourteen have queries today, and `Cx::window_left()` was the last:
//!
//! ```text
//! q, o                <- arg_in(0), arg_out(0)
//! kv_layer            <- kv_layer()                    seventeen fields
//! the four CSR arrays <- plan()
//! total_tokens        <- rows().count                  the row is `whole`
//! num_requests        <- plan().requests
//! num_pages_in_batch  <- num_pages_in_batch()          landed for kv_paged
//! num_q_heads         <- in_width(0) / layer.head_dim  Div(Width, KvField)
//! sm_scale            <- sm_scale()                    already implemented
//! window_left         <- window_left()                 landed, `window_of`
//! ```
//!
//! ### The blocker was on the DEVICE row, not the table row — and it is closed
//!
//! That was the finding, because the two rows disagreed about what the launch
//! takes. `attn::attention_naive_paged` was in `device::JIT_DISPATCHED`: its
//! `.cu` launcher is deleted, `emit_rust_dispatch` emitted a **JIT** arm
//! rather than a shim call, and that arm resolved `families::attn`'s device
//! sig — **twenty-three operands** — under `LaunchRule::PagedScores`. **The
//! JIT arm WAS the host program**, so a `bind!` had to satisfy the device row
//! and the fourteen-operand table row was only the trace's half of it.
//!
//! Nine of the extra nine are the layer view unpacked (`k_pages`, `v_pages`,
//! `k_scales`, `v_scales`, `num_kv_heads`, `head_dim`, `page_size`, `scheme`,
//! `storage_dtype`, `block_size` — ten, less `num_q_heads`, which the table
//! row also states) plus the two `Lit::Null` mask slots, and `Cx::kv_layer()`
//! answers every one. The two that were answered by nothing —
//! `logits_soft_cap` and `lse_out` — landed in `247e78a99`.
//!
//! **What was left was `x::Abi`, and it was not an ask.** Both kernels take
//! `device::KvScheme` and `device::KvDType` BY VALUE, each an
//! `enum class ... : ::std::uint8_t`, and no `Abi` impl marshalled a scalar
//! byte. `x/abi.rs:226` had predicted its own first caller and said where the
//! impl goes — *"an open set adds the impl with its first kernel"* — so
//! [`kv_scheme`] and [`kv_dtype`] are here, beside the kernel, and nothing in
//! `x/abi.rs` changed. Reading that note as a floor ask would have been
//! reading an instruction as an obstacle.
//!
//! And the crossing **restored a predicate the row world lost**:
//! `check_head_dim_supported`, which the deleted `.cu` made against
//! `kMaxHeadDim` and which the JIT arm could not make, because a `LaunchRule`
//! opens a grid and cannot refuse. See [`attention_naive_paged`].
//!
//! ## What is left, re-derived rather than remembered
//!
//! **NOTHING. `table::attn::KERNELS` is empty and so is
//! `table::ROW_TABLES`.** This section held a twenty-three-row list, then a
//! one-row list; every entry on both has been crossed or named as a stay, and
//! each list was replaced rather than annotated because a list of things that
//! are done is a thing that gets read as a list of things to do.
//!
//! The last was `attn::qkv_decode_qk_norm_rope_write_kv_bf16`, and what it
//! was waiting for is worth keeping because the wait produced a rule:
//!
//! * **THE ASK WAS `Cx::q_out`**, landed as `2dc9957b7`. Twenty-two of
//!   `qkv_decode_fused`'s twenty-three operands had queries; the
//!   twenty-third was
//!   `Source::Attn("q_out")` over `AttnCtx::q_out` (`bind/mod.rs:1437`), the
//!   observed-query pin the fused qkv writes and the dispatch reads. It has a
//!   producer, and the producer was the decision: `fire/launch.rs:3248` writes
//!   `core::ptr::null_mut()` when the fire pins no query, while the row
//!   grammar is plain `Source::Attn`, which ASSERTS presence. Those disagree,
//!   and `qkv_fused.cuh:177` stores through the pointer with no null test
//!   while `:182`, two arguments along, DOES test `w_page`/`w_off`. So the
//!   query is an `Option` and the host program refuses `Refusal::Absent`.
//!
//! * **AND THE RULE THAT CAME OUT OF IT.** `lse_out`'s discriminator — *"a
//!   plain `Source::Attn` asserts presence; an `Option` invents a state the
//!   row denies"* — holds only while the row and the producer AGREE. When
//!   they disagree, **the producer is the fact and the row is a claim**: a
//!   row cannot make a pointer non-null, it can only be believed or checked,
//!   and the device text says which. The grammar was a summary of the
//!   producer that happened to be accurate.
//!
//! `compact_page_csr` STAYS, by instruction, and keeps its
//! `execution::COMPOSED` entry -- that entry is a finding about the BODY (two
//! launches, one stream, the second reading the first's buffer) and the body
//! is still two ops. `gemm::mla_absorb_{q_to_latent,latent_to_v}` are
//! `x::gemm`'s cuBLAS driver ops and were never this family's rows to cross.
//! `mla_prepare_bf16` and `write_mla_to_pages` crossed as `none:` arms over
//! [`Cx::mla_layer`], which refuses because NOTHING FILLS IT -- the MLA cache
//! pool has zero callers and `serve/load.rs` refuses every MLA checkpoint at
//! load. That is a feature, not a seam, and the refusal says so.
//! `execution::COMPOSED` entry -- that entry is a finding about the BODY (two
//! launches, one stream, the second reading the first's buffer) and the body
//! is still two ops. `gemm::mla_absorb_{q_to_latent,latent_to_v}` are
//! `x::gemm`'s cuBLAS driver ops and were never this family's rows to cross.
//! `mla_prepare_bf16` and `write_mla_to_pages` crossed as `none:` arms over
//! [`Cx::mla_layer`], which refuses because NOTHING FILLS IT -- the MLA cache
//! pool has zero callers and `serve/load.rs` refuses every MLA checkpoint at
//! load. That is a feature, not a seam, and the refusal says so.
//!
//! ### The measurement that made each `none:` arm safe
//!
//! Recorded once, because it is the one thing in this migration that must be
//! taken **per symbol** and can never be inherited from a neighbour. A
//! `contract!` with a `none:` arm SHADOWS whatever the row world was doing:
//! `x::route`'s ladder consults `entry()` first, so an `Entry` that refuses
//! turns a working generated dispatch into `Route::Unbound`, which refuses
//! the model at LOAD. It is safe only where both hold:
//!
//! ```text
//! (a) the row was Source::Unbound throughout, so abi.rs:810 skipped it
//!     whole and NO ARM WAS EVER GENERATED; and
//! (b) no model reaches the `dsl::cuda` wrapper, so nothing calls it by the
//!     other road either.
//! ```
//!
//! `dsa_index_topk_mask` is the counter-example that proves the measurement
//! is not a formality: it read like its two siblings, it was **fully
//! sourced**, and a `none:` arm there would have refused a model at load. It
//! binds for real. Its two siblings are `none:` arms and were measured
//! separately to say so.
//!
//! ### And the device rows are the other half
//!
//! §60.6 splits a device row's symbol from the trace symbol, and this family
//! is where the split was found MISSING three times -- `attn_score_fold_heads`
//! most recently, whose one string was the device row's name, the table row's
//! name and `dsl::cuda`'s stated symbol at once. A `contract!` symbol may
//! never also be a unit row's symbol: `execution::a_walk_is_only_a_walk`
//! asserts it, `migration_status`' `refused_set()` reads it, and the failure
//! without them is silent -- `unit_of` answers a trace symbol and the trace
//! fires a device row. Every crossing here renames the device row `_dev` and
//! the one firer follows, which is one line when the firer resolves through
//! `unit_of` and a sweep when it does not.
//!
//! [`Facts`]: crate::x::Facts
//! [`Cx::arg_in`]: crate::x::Cx::arg_in
//! [`Cx::arg_out`]: crate::x::Cx::arg_out
//! [`crate::x::driver_internal::split_qkv_bf16`]: crate::x::driver_internal::split_qkv_bf16
//! [`Cx::plan`]: crate::x::Cx::plan
//! [`Cx::slab`]: crate::x::Cx::slab

#![allow(clippy::too_many_arguments)]

use crate::unit::Unit;
use crate::x::abi::{bf16, f16};
use crate::x::launch::Launch;
use kernels::{Cap, Prepare};

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
use crate::x::cx::{MlaLayer, Yarn};
#[cfg(feature = "_cuda")]
use core::ffi::c_void;
#[cfg(feature = "_cuda")]
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasGemmStridedBatchedEx,
    cublasOperation_t, cublasStatus_t, cudaDataType,
};

// ---------------------------------------------------------------------------
// The two scalar-byte `Abi` impls, added HERE and not in `x/abi.rs`.
// ---------------------------------------------------------------------------
//
// `x/abi.rs:226` predicted this exact caller and said where the impl goes:
//
//   > No scalar `u8`, and not an oversight: `Ty` has no general byte tag --
//   > the row world only ever crossed a scalar byte as a semantic enum
//   > (`KvScheme`, `KvDType`, both checked against `ArgValue::U8`) -- AND NO
//   > FN-WORLD KERNEL TAKES ONE. **An open set adds the impl with its first
//   > kernel**, under whichever tag is honest for it, rather than minting a
//   > near-miss here.
//
// `attention_naive_paged` is that first kernel, so these are that impl, and
// the note's placement instruction is followed rather than read as an ask:
// `Abi` is a `pub trait` over an open set, these are local types, and an
// `impl` beside the one kernel that needs it is the whole of what "with its
// first kernel" means. NOTHING IN `x/abi.rs` CHANGES.
//
// # `fp8_kind` is the precedent and the shape is copied exactly
//
// A `#[repr(transparent)]` newtype over the primitive the wire carries, with
// `CPP` naming the C++ enum and `TY` a SEMANTIC tag rather than a width tag.
// The reason two tags exist rather than one is `kernels::Ty::KvScheme`'s own
// and it is worth repeating where the values are built: the two operands are
// ADJACENT in both parameter lists and the same width, so one shared kind
// would make the SWAP type-check on every side this crate can check. Two
// kinds put the check where the C++ can make it.
//
// # The narrowing is real and this is the one place it happens
//
// `x::cx`'s [`KvScheme`] and [`KvDType`] are `#[repr(i32)]`; the device
// mirrors at `attention_naive_paged.cuh:187` and `:198` are
// `enum class ... : ::std::uint8_t`. **Four bytes on one side, one on the
// other**, which is §3.2's hazard rotated ninety degrees -- not two formats
// at one width but one format at two widths. The `as u8` below is exact
// because every enumerator of both mirrors is in `0..=11` and
// `driver-cuda/tests/enum_mirrors.rs` asserts every one of them against the
// `.cuh`; it is written ONCE, here, so no host program can spell it a
// second way.
//
// The `from_*` constructors are how a body gets one, and they take the `Cx`
// enum rather than a `u8`, so a caller cannot hand over a number it computed.

/// `attn::device::KvScheme` — how a paged KV bank is quantised, as the device
/// text spells it.
///
/// [`crate::x::cx::KvScheme`] is the same five enumerators on the host side.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct kv_scheme(pub u8);

impl kv_scheme {
    /// The device spelling of a [`Cx`](crate::x::Cx)-stated scheme.
    #[must_use]
    pub const fn of(scheme: crate::x::cx::KvScheme) -> Self {
        Self(scheme as i32 as u8)
    }
}

impl crate::x::Abi for kv_scheme {
    const CPP: &'static str = "::pie_cuda_driver::kernels::attn::device::KvScheme";
    const TY: kernels::Ty = kernels::Ty::KvScheme;
    #[cfg(feature = "_cuda")]
    fn arg(&self) -> crate::runtime::ArgValue {
        crate::runtime::ArgValue::U8(self.0)
    }
}

/// `attn::device::KvDType` — what a page element actually is.
///
/// [`crate::x::cx::KvDType`] is the host side's five. The device mirror
/// carries twelve enumerators because `attention_naive_paged.cuh:198` states
/// the rule *"a partial mirror is a renumbering waiting to happen"*; the
/// five the `Cx` mirror carries are the five a KV page can hold, and
/// `Cx::kv_layer()` returns `None` rather than widening if a producer
/// reaches one of the other seven.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct kv_dtype(pub u8);

impl kv_dtype {
    /// The device spelling of a [`Cx`](crate::x::Cx)-stated storage dtype.
    #[must_use]
    pub const fn of(dtype: crate::x::cx::KvDType) -> Self {
        Self(dtype as i32 as u8)
    }
}

impl crate::x::Abi for kv_dtype {
    const CPP: &'static str = "::pie_cuda_driver::kernels::attn::device::KvDType";
    const TY: kernels::Ty = kernels::Ty::KvDType;
    #[cfg(feature = "_cuda")]
    fn arg(&self) -> crate::runtime::ArgValue {
        crate::runtime::ArgValue::U8(self.0)
    }
}

// ---------------------------------------------------------------------------
// Truth one, declared: the device text and its instantiations.
//
// THREE `unit!` INVOCATIONS CANNOT SHARE A SCOPE — each emits `UNITS`, `ROWS`,
// `PARAMS` and `raw` at module scope. Each root gets a module and the family
// re-exports the three below, which is `x/layout.rs`' arrangement for five and
// `x/mlp.rs`' for two.
// ---------------------------------------------------------------------------

/// `attn`'s `#[repr(C)]` mirrors of C++ aggregates, and their measured
/// layouts.
///
/// One struct today. It is a module rather than three items at file scope
/// because `x/xqa.rs` is the precedent and its shape is worth matching
/// exactly: the mirror, its [`Abi`](crate::x::Abi) impl, and a [`Layout`]
/// carrying numbers **measured out of NVRTC's PTX** with the probe that
/// measured them named beside them. `MLAParams` and `HopperParams` land here
/// when `mla_*` crosses; nothing is written for them in advance, because
/// §0's rule is data only for what has a reading consumer.
///
/// [`Layout`]: crate::x::Layout
pub mod params {
    use core::ffi::c_void;

    use kernels::Ty;

    /// One lane's structured-mask descriptor, as
    /// `attn::pack_structured_mask` reads it.
    ///
    /// Mirrors `csrc/src/attn/pack_dense_mask.cuh:136`.
    ///
    /// # Where the numbers came from
    ///
    /// **Not from reading the header, and not from the driver's mirror.**
    /// Measured by `nvrtc-probes/attn_structured_mask.py`, whose method is
    /// `nvrtc-probes/params_layout.py`'s: emit `(unsigned)((char*)&((P*)0)->f
    /// - (char*)(P*)0)` into a `__constant__` array and read the initialiser
    /// back out of the PTX. `offsetof` and `__builtin_offsetof` are both
    /// unavailable under NVRTC; only the pointer DIFFERENCE folds.
    ///
    /// Measured, `rc=0`, NVRTC 13.0, `compute_89`, `-std=c++17
    /// -default-device`, against `csrc/src/attn/pack_dense_mask.cuh` under
    /// `-I csrc/{shim,vendor,src}`:
    ///
    /// ```text
    /// attn::device::StructuredMaskParams: sizeof=12  alignof=4
    ///     0  kind
    ///     4  window
    ///     8  sink
    /// ```
    ///
    /// This is the second time in the migration that reading and measuring
    /// agreed, and the reason is `x/xqa.rs`'s: **no nested aggregate.** Three
    /// `unsigned int`s, and every header set in the tree agrees that an
    /// `unsigned int` is four bytes. The traps `params_layout.txt` records —
    /// `uint_fastdiv` at 24 bytes not 4, CuTe's `dA` at 8 not 24, two
    /// `paged_kv_t`s with equal `sizeof` and different interiors — are all
    /// nested aggregates whose size disagrees between `csrc/shim` and CCCL,
    /// and none of them can reach a struct with no members but scalars. Worth
    /// writing down for the same reason `xqa` wrote it down: the property is
    /// this struct's, not the technique's.
    ///
    /// # There was supposed to be an oracle, and it is gone
    ///
    /// `pack_dense_mask.cuh:29-50` still argues that two definitions of this
    /// POD are acceptable because *"`pack_dense_mask.cu` includes BOTH
    /// definitions and `static_assert`s size, alignment and all three field
    /// offsets against each other. A field added, reordered or widened on
    /// either side fails the ahead-of-time build with the two spellings named
    /// in the message."* **Those five `static_assert`s no longer exist.**
    /// `attn/pack_dense_mask.cu` and `attn/pack_dense_mask.hpp` are both
    /// deleted — `driver-cuda/tests/launch_abi.rs:651-654` records the
    /// deletion — so from that day the two mirrors agreed by luck.
    ///
    /// They did agree: `driver-cuda/src/bind/abi.rs`'s three `u32`s in this
    /// order match the measurement field for field. That is the answer to
    /// "check yours against the oracle" — the oracle was right and unchecked,
    /// which is the worse of the two ways to be right. It is now a re-export
    /// of this type, so there is one definition and the question cannot
    /// recur.
    ///
    /// # `kind` is a number, and there are two numberings
    ///
    /// The `__global__` reads it as a literal — `descriptor.kind == 1` causal,
    /// `== 2` sliding window, `== 3` sink (`pack_dense_mask.cuh:236-240`). The
    /// host enum it mirrored, `attn::StructuredMaskKind`, went with the
    /// `.hpp`; this doc and the kernel body are the only surviving record of
    /// the numbering.
    ///
    /// **`ptir/tier0.cuh:613`'s `Tier0StructuredMaskKind` is a DIFFERENT
    /// numbering** — `Causal = 0`, `SlidingWindow = 1`, `SinkWindow = 2`, the
    /// default discriminants of an `enum class : uint8_t`. It describes the
    /// same three kinds one lower. Filling this field from that enum yields
    /// sliding-window where causal was meant, on every lane, with no
    /// diagnostic anywhere: `kind == 0` matches none of the three arms, so
    /// every bit falls to the `causal &&` conjunction alone and the mask is
    /// silently a plain causal one. No enum is minted here to close that,
    /// because nothing on this side reads one — §0's placement rule — and an
    /// enum whose only consumer is its own definition is the data this
    /// migration is removing.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    #[repr(C)]
    pub struct StructuredMaskParams {
        /// `kind` — 1 causal, 2 sliding window, 3 sink. See the type's doc
        /// for the other numbering and what it costs.
        pub kind: u32,
        /// `window` — the sliding window's extent in keys, for kinds 2 and 3.
        ///
        /// The kernel saturates `key + window` at `0xffffffff` rather than
        /// letting it wrap, because a wrapped sum reads as a CLOSED window
        /// and masks out exactly the tokens the open one admits
        /// (`pack_dense_mask.cuh:227-233`).
        pub window: u32,
        /// `sink` — the attention-sink width in keys, for kind 3: every key
        /// below it is admitted regardless of the window.
        pub sink: u32,
    }

    /// How C++ spells the struct itself, for the `static_assert`s
    /// [`typecheck_tu`](crate::x::abi::typecheck_tu) emits from [`LAYOUTS`].
    ///
    /// **Not [`Abi::CPP`](crate::x::Abi::CPP), and the difference is the
    /// point.** `Layout::cpp`'s doc says "the same string as `Abi::CPP`",
    /// which holds for a by-value aggregate like `x::xqa::KvCacheList` and
    /// cannot hold here: this struct crosses as a POINTER, so `Abi::CPP` is
    /// `const T*` while `sizeof`, `alignof` and `offsetof` all need the bare
    /// `T`. Two constants, one suffix apart, so neither can be used where the
    /// other belongs.
    const STRUCTURED_MASK_PARAMS: &str =
        "::pie_cuda_driver::kernels::attn::device::StructuredMaskParams";

    /// The array of descriptors, one per lane, as `pack_structured_mask`
    /// takes it.
    ///
    /// # Why this impl is written by hand
    ///
    /// [`ptr_abi!`] is a plain `macro_rules!` private to `x/abi.rs`, and
    /// `x/abi.rs` is not this sweep's to edit. It does not need to be:
    /// `x/abi.rs`'s own header states the rule — *"adding a crossing type
    /// means writing one impl, next to the kernel that needed it, and nothing
    /// else in the tree changes"* — and that is exactly what this is. Only
    /// the `*const` half is written, because every use is `const`; a `*mut`
    /// impl would be a spelling with no `__global__` behind it.
    ///
    /// # The namespace is `attn::device` and NOT `attn`
    ///
    /// `kernels/src/lib.rs:1079` spells `Ty::StructuredMasks` in the `attn`
    /// namespace, which was `pack_dense_mask.hpp`'s and is now nothing's.
    /// This string is the one NVRTC resolves, which is what the typecheck TU
    /// compiles. See this module's file header for the full account; the
    /// divergence is deliberate and recorded rather than papered over by
    /// matching a spelling that names no type.
    ///
    /// [`ptr_abi!`]: crate::x::abi
    impl crate::x::Abi for *const StructuredMaskParams {
        const CPP: &'static str =
            "const ::pie_cuda_driver::kernels::attn::device::StructuredMaskParams*";
        const TY: Ty = Ty::StructuredMasks;
        #[cfg(feature = "_cuda")]
        fn arg(&self) -> crate::runtime::ArgValue {
            crate::runtime::ArgValue::Ptr(*self as *mut c_void)
        }
    }

    // The same three numbers `by_value!` would have asserted, written out
    // because `by_value!` is the wrong macro for a pointer crossing: its
    // `Abi::arg` is `ArgValue::Bytes`, which would pass twelve bytes of
    // descriptor where the kernel wants an eight-byte address to an ARRAY of
    // them. The assertions are the half that does apply, so they are kept.
    const _: () = assert!(
        ::core::mem::size_of::<StructuredMaskParams>() == 12,
        "StructuredMaskParams: sizeof disagrees with the measured \
         attn::device::StructuredMaskParams; re-run nvrtc-probes/attn_structured_mask.py",
    );
    const _: () = assert!(
        ::core::mem::align_of::<StructuredMaskParams>() == 4,
        "StructuredMaskParams: alignof disagrees with the measured \
         attn::device::StructuredMaskParams; re-run nvrtc-probes/attn_structured_mask.py",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, kind) == 0,
        "StructuredMaskParams.kind: offset disagrees with the measured \
         attn::device::StructuredMaskParams::kind",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, window) == 4,
        "StructuredMaskParams.window: offset disagrees with the measured \
         attn::device::StructuredMaskParams::window",
    );
    const _: () = assert!(
        ::core::mem::offset_of!(StructuredMaskParams, sink) == 8,
        "StructuredMaskParams.sink: offset disagrees with the measured \
         attn::device::StructuredMaskParams::sink",
    );

    /// The measured layout, as C++ `static_assert`s.
    ///
    /// Handed to [`typecheck_tu`](crate::x::abi::typecheck_tu) with
    /// [`pack_dense_mask::PARAMS`](super::pack_dense_mask::PARAMS). The Rust
    /// assertions above catch a drifted mirror; these catch a drifted header,
    /// and only both together catch a field that moved in the header while
    /// someone updated the mirror to the wrong numbers.
    ///
    /// A `LAYOUTS` for a POINTER parameter is not what `x/xqa.rs`'s is for —
    /// a pointer crossing is checked by `is_same_v` on the whole parameter
    /// list whether or not its pointee's layout is asserted. It is here
    /// because this pointee has a mirror, and a mirror with no assertion is
    /// the shape that was already wrong once in this file's history: see the
    /// deleted `.cu` above.
    pub static LAYOUTS: &[crate::x::Layout] = &[crate::x::Layout {
        cpp: STRUCTURED_MASK_PARAMS,
        size: 12,
        align: 4,
        fields: &[("kind", 0), ("window", 4), ("sink", 8)],
        probe: "nvrtc-probes/attn_structured_mask.py",
    }];
}

/// `attn/attn_sink.cuh` — gpt-oss's sink correction and the LSE rebase it
/// reads.
pub mod attn_sink {
    use super::bf16;

    unit! {
        /// The attention-sink pair, both rows: the log2→ln rebase and the
        /// per-head rescale that reads it.
        ///
        /// Both are corrections applied AFTER an attention kernel has
        /// already written its output, which is why they are separate
        /// launches and not a fused epilogue — the attention kernel is
        /// flashinfer's and cannot be edited.
        unit ATTN_SINK = "attn/attn_sink",
            text = include_str!("../../csrc/src/attn/attn_sink.cuh"),
            file = "attn/attn_sink.cuh";

        /// `attn_sink.cuh:74` — flashinfer publishes its LSE in log2 and the
        /// sink correction works in ln.
        ///
        /// A unit conversion, stated so a reader never has to guess which
        /// base an LSE is in — and the drift it prevents is measured:
        /// without it the sigmoid argument was off by 0.693, which matched
        /// HF's top-1 on most prompts and then degenerated greedy decoding
        /// after a few steps.
        ///
        /// **`elem` is `attn::device::f32` and not `device::bf16`'s
        /// sibling.** The prelude has no `device::f32` alias and `Elem` has
        /// no `float` specialisation to hang one on, so the alias is declared
        /// in the `.cuh` beside the kernel that is the only thing asking for
        /// it.
        ///
        /// `n` is `usize` where the sink twin's `N` is `i32`, because the
        /// kernel's parameter is `device::usize` — the twin's `int` was the
        /// launcher's signature, not the kernel's.
        fn lse_log2_to_ln = "attn::device::lse_log2_to_ln" <T> (
            lse: *mut T,
            n: usize,
        ) where *mut T {
            "attn::lse_log2_to_ln" => where [T = f32] "attn::device::f32",
        }

        /// `attn_sink.cuh:93` — `o[t, h, :] *= sigmoid(ln_lse[t, h] -
        /// sink[h])`, in place.
        ///
        /// GPT-OSS learns a per-head sink scalar and extends the softmax
        /// denominator with `exp(sink)`; this is that correction, applied to
        /// the attention OUTPUT.
        ///
        /// **`N` and `num_q_heads` stay operands though the rule recovers
        /// both.** They are the kernel's own `if (t >= N || h >=
        /// num_q_heads) return;` and its row stride `num_q_heads *
        /// head_dim`; an operand list shorter than the `__global__`'s
        /// parameter list is a `void**` the driver reads past. What left is
        /// the stream, which was never one.
        fn attn_sink_rescale = "attn::device::attn_sink_rescale" <T> (
            o: *mut T,
            lse: *const f32,
            sinks: *const T,
            n: i32,
            num_q_heads: i32,
            head_dim: i32,
        ) where *const T, *mut T {
            "attn::attention_sink_rescale_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `attn/attn_res.cuh` — K3's residual-block blend.
pub mod attn_res {
    use super::bf16;

    unit! {
        /// K3's residual-block blend, at bf16.
        ///
        /// K3's residual stream is not a single vector: a prefix and `B`
        /// candidate blocks compete, and the layer picks a convex
        /// combination of them. The score of each candidate is an
        /// RMS-normalised projection — normalise the row, dot it against
        /// `norm_weight * proj_weight`, softmax the `B + 1` scores, blend.
        /// Fusing it is not an optimisation but a memory decision.
        unit ATTN_RES = "attn/attn_res",
            text = include_str!("../../csrc/src/attn/attn_res.cuh"),
            file = "attn/attn_res.cuh";

        /// `attn_res.cuh:99` — one block per token, 256 threads.
        ///
        /// **`T` is gone from the operand list where the deleted twin stated
        /// it.** It did two jobs: a bound check, which is now the grid's
        /// promise, and a block stride, which survives as `block_rows`. The
        /// launcher's `block_rows > 0 ? block_rows : T` default is the row
        /// count, which is the value that ternary produced on every call
        /// site that existed.
        fn attn_res_blend = "attn::device::attn_res_blend" <T> (
            prefix: *const T,
            blocks: *const T,
            norm_weight: *const T,
            proj_weight: *const T,
            out: *mut T,
            b: i32,
            h: i32,
            block_rows: i32,
            eps: f32,
        ) where *const T, *mut T {
            "attn::attn_res_blend_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `attn/head_dim_pad.cuh` — flashinfer's supported head widths, reached by
/// padding and reached back out of by stripping.
pub mod head_dim_pad {
    use super::bf16;

    unit! {
        /// The pad and the strip, at bf16.
        ///
        /// flashinfer compiles its attention kernels for a fixed set of head
        /// widths — 64, 128, 256, 512 — and a model whose `head_dim` is none
        /// of them cannot use them at all. Phi-3-mini ships 96. These two
        /// kernels buy that model the dense path: pad every head out to the
        /// next supported width on the way in, strip the padding on the way
        /// out. The zero pad is not arbitrary: `q_e . k_e = q[:d] . k[:d]`
        /// when both are zero above `d`.
        ///
        /// A unit that could not exist until [`LaunchRule::PerHead`] did.
        /// Both kernels were extracted, proved NVRTC-clean and left rowless,
        /// and a rowless unit is refused rather than compiled:
        /// `every_unit_compiles_and_every_row_resolves` asserts
        /// `!unit.rows.is_empty()`, because a cubin nothing can fire is one
        /// cached per architecture for nobody.
        ///
        /// [`LaunchRule::PerHead`]: crate::runtime::LaunchRule::PerHead
        unit HEAD_DIM_PAD = "attn/head_dim_pad",
            text = include_str!("../../csrc/src/attn/head_dim_pad.cuh"),
            file = "attn/head_dim_pad.cuh";

        /// `head_dim_pad.cuh:73` — copy `head_dim` values per (token, head)
        /// and zero the trailing columns.
        ///
        /// Threads stride over the PADDED extent so every thread executes
        /// exactly one store — a copy or a zero — rather than one branch
        /// executing and the other stalling. Same instruction count either
        /// side of the boundary.
        ///
        /// `num_tokens` and the stream are not operands: the first is
        /// `grid.y` and the second never was one. Everything the
        /// `__global__` declares stays, including `num_heads` — the geometry
        /// puts the count on an axis the kernel does not read it back from,
        /// so a row without it is a `void**` one entry short.
        fn pad_head_dim = "attn::device::pad_head_dim" <T> (
            packed: *const T,
            padded: *mut T,
            num_heads: i32,
            head_dim: i32,
            head_dim_padded: i32,
        ) where *const T, *mut T {
            "attn::pad_head_dim_bf16" => where [T = bf16] "device::bf16",
        }

        /// `head_dim_pad.cuh:92` — the inverse, and the same five operands
        /// with the two buffers swapped.
        fn strip_head_dim = "attn::device::strip_head_dim" <T> (
            padded: *const T,
            packed: *mut T,
            num_heads: i32,
            head_dim: i32,
            head_dim_padded: i32,
        ) where *const T, *mut T {
            "attn::strip_head_dim_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `attn/softcap.cuh` — the logit cap, at both numeric formats.
///
/// **The two-formats-one-width root**, and the only one in `attn`. The header
/// argues the hazard; this module is where it is answered, by two rows on one
/// `fn` declaration whose type binding is the whole of the difference between
/// them.
pub mod softcap {
    use super::{bf16, f16};

    unit! {
        /// One `__global__` template and nothing else — no host function, no
        /// `<<<>>>`, no entry point, which is what `softcap.cuh`'s own header
        /// says about itself.
        ///
        /// # Why it is a template when the original was not
        ///
        /// The original was `_bf16` and only `_bf16`, because an
        /// ahead-of-time build has to choose its instantiations and nobody
        /// spends a translation unit on a second one. Under a JIT the element
        /// type is the row's, so a capped fp16 logit row costs **the line
        /// below** rather than a `cicc` invocation. That is the measurement
        /// `norm/elementwise.cuh` made first with its fp16 `residual_add`,
        /// and it is the reason this migration was worth making.
        unit SOFTCAP = "attn/softcap",
            text = include_str!("../../csrc/src/attn/softcap.cuh"),
            file = "attn/softcap.cuh";

        /// `softcap.cuh:67` — `x = cap * tanh(x / cap)`, elementwise and in
        /// place.
        ///
        /// The cap is a SATURATION and not a clamp: the tail is squashed
        /// smoothly, which is what gemma-2/3 and grok train against, and a
        /// hard `min`/`max` here changes the distribution the sampler then
        /// reads.
        ///
        /// **`x` is the only buffer and there is no destination.** That is
        /// what the deleted row's `in_place = &[(0, 0)]` said and what
        /// `Buffers::assign` was already relying on — *"the logit softcap
        /// accumulates into the logits it was handed"*, where it widens a
        /// seam's pin over an alias set. While the row said nothing the set
        /// had one member and the widening reached nothing: the head wrote
        /// the logits into the arena, the cap ran over `ws.logits`, and the
        /// sampler read an uncapped previous fire. [`super::LOGIT_SOFTCAP`]
        /// carries the `in_place` forward.
        ///
        /// **`n` is `usize` and not `i32`**, because the kernel's parameter
        /// is `device::usize`. There is no bound check against a row count
        /// and no `rows` argument: the grid covers `n` and `if (i >= n)` at
        /// `softcap.cuh:69` is the only guard there has ever been.
        ///
        /// **The reciprocal is not a parameter.** `attn_softcap.cu`'s
        /// launcher passed BOTH `1.f / cap` and `cap`, because a `<<<>>>` is
        /// the only place a host can do arithmetic on the way to a kernel;
        /// `softcap.cuh:70` does the division on the device instead and it is
        /// the same bits. This crate compiles every unit with
        /// `--prec-div=true`, so `1.f / cap` is the correctly-rounded fp32
        /// quotient on the device exactly as it was on the host, and
        /// `--fmad=false` keeps the multiply from being contracted into
        /// something else — two IEEE-754 operations either way, same
        /// rounding, same result. **A `fn` does not restore it**: doing the
        /// division here would put a second implementation of the same
        /// expression on the host, and §0's *"nothing is written twice"* is
        /// exactly that.
        ///
        /// # Two rows, and the second one has no consumer on purpose
        ///
        /// `attn::logit_softcap_f16` is *"the row the ahead-of-time build
        /// never had"* — `families/attn.rs`' header names it that. It has no
        /// table row, no [`contract`](super::LOGIT_SOFTCAP) and no trace
        /// spelling, and it keeps all three of those absences here: a
        /// contract is what a trace MAY say, and no trace says this. What it
        /// has is device text that compiles and a typechecked signature, so
        /// the day a head lands in fp16 the cost is a contract rather than a
        /// port.
        fn logit_softcap = "attn::device::logit_softcap" <T> (
            x: *mut T,
            cap: f32,
            n: usize,
        ) where *mut T {
            "attn::logit_softcap_bf16" => where [T = bf16] "device::bf16",
            "attn::logit_softcap_f16" => where [T = f16] "device::f16",
        }
    }
}

/// `attn/split_packed.cuh` — the fused QKV product cut into three operands.
///
/// **A ROOT WHOSE TWO KERNELS HAVE TWO DIFFERENT ARRANGEMENTS**, which is
/// what makes it worth reading. Neither gets a `fn` inside this module; one
/// is fired by `x::driver_internal` and the other by a bind over a `fn` in
/// this module's parent:
///
/// * `attn::split_qkv_bf16` — the non-devwin half — **is already in
///   fn-world**, as [`crate::x::driver_internal::split_qkv_bf16`. It is the
///   fourth arrangement: no unit, no contract, no bind, absent from
///   `FAMILIES` and from `SIGS`, because no trace can state it.
///   `model-compiler`'s `lower.rs::semantic()` DOES name a symbol for
///   `SplitQkv` at `:1545-1548` — but it names this DEVICE row, and
///   `lowered.kernels` (`lower.rs:1095-1096`) is written only from the
///   launch-emitting path, which a driver-internal symbol was never on. So
///   `x::route` never sees it, and `driver_internal`'s six functions fire it
///   BY SYMBOL through [`crate::x::fire::fire`]. What it does: one pass over
///   packed memory, pure copy, no compute — the fused matmul writes one
///   row-major `[N, q_dim + 2 * kv_dim]` tensor and everything downstream
///   addresses Q, K and V as its own `[N, dim]` buffer, because their widths
///   differ under GQA and a single stride cannot describe all three.
/// * `attn::split_qkv_bf16_devwin` — the table symbol, whose device row is
///   spelled `attn::split_qkv_devwin` — **is bound**, as
///   [`SPLIT_QKV_DEVWIN`], over [`split_qkv_bf16_devwin`] in this module's
///   parent. Its row and `driver-cuda/src/fire/split_packed.rs` are both
///   gone. The paragraph that stood here said it could not be, and the
///   module header carries the retraction.
///
/// So this module is the unit and only the unit, which is what
/// `x/driver_internal.rs` says the destination is: *"the rows stay where the
/// device text is — in `families::{attn,norm,layout,mlp}` today, and in
/// `x::{attn,norm,layout,mlp}` when those families land"*. The root moves; the
/// two programs do not move with it, because they are already where they go.
///
/// [`crate::x::driver_internal::split_qkv_bf16`]: crate::x::driver_internal::split_qkv_bf16
pub mod split_packed {
    use super::bf16;

    unit! {
        /// Two `__global__` templates, no host code.
        ///
        /// # `LaunchRule::SplitPacked` is not why either row exists here
        ///
        /// Both launchers were `<<<dim3(ceil(max(q_dim, kv_dim) / 256), n),
        /// 256>>>`. `SplitPacked` is the rule with that shape and its grid
        /// over the INPUT width (`q_dim + 2 * kv_dim`) is WIDER than the
        /// launcher's over `max(q_dim, kv_dim)` — the outputs are identical
        /// either way, because *"every loop strides by `blockDim.x *
        /// gridDim.x` and bounds itself on its own output width, so extra
        /// blocks contribute nothing but a shorter loop"*. Wider is safe in
        /// this direction and only this one; a grid narrower than an output
        /// leaves the tail of every row unwritten. In fn-world neither row
        /// carries a rule at all: a `Launch` is written by whoever fires it,
        /// and both firers write it from the numbers they were handed.
        unit SPLIT_PACKED = "attn/split_packed",
            text = include_str!("../../csrc/src/attn/split_packed.cuh"),
            file = "attn/split_packed.cuh";

        /// `split_packed.cuh:74` — the host-window form, over
        /// already-offset pointers.
        ///
        /// Six parameters where the launcher had eight: `n_tokens` is the
        /// grid's second axis and never reaches the kernel, and the stream
        /// was never an operand. The two widths come off what is WRITTEN and
        /// not off the packed operand — a `[N, q + 2 * kv]` row cannot say
        /// where the cut falls, and both results can.
        ///
        /// Fired by [`crate::x::driver_internal::split_qkv_bf16`], whose
        /// `Launch` is the launcher's own literal: `grid [ceil(max(q_dim,
        /// kv_dim) / 256), n_tokens, 1]`, `block [256, 1, 1]`, no shared
        /// memory.
        ///
        /// [`crate::x::driver_internal::split_qkv_bf16`]: crate::x::driver_internal::split_qkv_bf16
        fn split_qkv = "attn::device::split_qkv" <T> (
            src: *const T,
            q_out: *mut T,
            k_out: *mut T,
            v_out: *mut T,
            q_dim: i32,
            kv_dim: i32,
        ) where *const T, *mut T {
            "attn::split_qkv_bf16" => where [T = bf16] "device::bf16",
        }

        /// `split_packed.cuh:111` — the device-window form, over BASE
        /// pointers.
        ///
        /// **A second kernel and not a flag**, because the pointers the two
        /// are handed mean different things: base pointers here,
        /// already-offset pointers there, and a null check cannot reconcile
        /// that. The grid spans the full lane count and rows outside
        /// `[win[0], win[0] + win[1])` return before touching anything, which
        /// is what lets a captured graph replay across different row splits
        /// without re-recording — the window changes in a buffer, not in a
        /// launch.
        ///
        /// **`win` is the seventh parameter and `n_max` is not a parameter
        /// at all**: it is the grid's second axis. `split_packed.cu:45-46`
        /// — `dim3 grid(xblocks, n_max)` then `device::split_qkv_devwin<bf16>
        /// <<<grid, BLOCK, 0, stream>>>(packed, q_out, k_out, v_out, win_d,
        /// q_dim, kv_dim)`.
        ///
        /// Fired by [`super::split_qkv_bf16_devwin`], the host program, over
        /// the [`SPLIT_QKV_DEVWIN`] bind.
        ///
        /// **This doc said "and NOT by a bind" and gave two reasons.** The
        /// first was true and is now answered: `grid.y` is the FIRE's lane
        /// count and not the statement's rectangle — so under a peel a rule's
        /// `Dims::rows` would be the tail's length while the kernel compares
        /// an ABSOLUTE `blockIdx.y` against the device window, the rows past
        /// that length never visited, Q, K and V keeping the previous fire's
        /// bytes there. `Cx::rows().total` is that lane count, and
        /// `bind/facts.rs:319` names a `_devwin` launch in the field's doc.
        ///
        /// The second was *"`Cx::arg_in`/`Cx::arg_out` return pointers
        /// `resolve_arg_windowed` has already offset by the region's first
        /// row"* and it was **false here and true everywhere else**:
        /// `bind/mod.rs:3973` resolves a `_devwin` kernel's args at row 0,
        /// by suffix, because *"their contract is BASE pointers"*.
        ///
        /// [`SPLIT_QKV_DEVWIN`]: super::SPLIT_QKV_DEVWIN
        /// [`super::split_qkv_bf16_devwin`]: super::split_qkv_bf16_devwin
        ///
        /// # TWO SYMBOLS FOR ONE KERNEL, and the row is the device one
        ///
        /// The string here is `attn::split_qkv_devwin`. The string
        /// `table/attn.rs` states is `attn::split_qkv_bf16_devwin`, and they
        /// are deliberately different: the table symbol is what a DISPATCH
        /// names and the device symbol is what NVRTC resolves.
        /// `driver-cuda/src/fire/split_packed.rs` holds both as constants
        /// side by side — `SPLIT_DEVWIN_SYMBOL` at `:49` and
        /// `SPLIT_DEVWIN_DEVICE` at `:52` — and is the only thing that
        /// bridges them. A `unit!` row states the DEVICE symbol, so getting
        /// this wrong would resolve nothing and fail at the fire rather than
        /// at a type.
        fn split_qkv_devwin = "attn::device::split_qkv_devwin" <T> (
            src: *const T,
            q_out: *mut T,
            k_out: *mut T,
            v_out: *mut T,
            win: *const u32,
            q_dim: i32,
            kv_dim: i32,
        ) where *const T, *mut T {
            "attn::split_qkv_devwin" => where [T = bf16] "device::bf16",
        }
    }
}

/// `attn/attention_flashinfer.cuh` — the per-head → per-request score fold.
///
/// **One `__global__`, and the root holds only that one.** The other three of
/// `attention_flashinfer.cu` stayed in the `.cu` and the header says why in
/// one line: *"They move when something asks for them."* Nothing has.
///
/// # Unit AND contract, and the host program is `driver-cuda`'s
///
/// A third arrangement for this family, and the one `x/mod.rs`'s table calls a
/// contract over a driver-fired unit: the device text is here, the trace
/// symbol has a [`contract!`](crate::contract) so no row is needed for it, and
/// the launcher stays `driver-cuda/src/fire/attn_score.rs` — 1,548 lines with
/// live consumers in `fire/scratch.rs`, `fire/stage_hooks.rs`,
/// `fire/launch.rs` and `bind/mod.rs`, none of which is about this kernel.
/// Moving the fold out of that file would move a `Launch` and leave its
/// staging behind.
///
/// # §60.6's `_dev` split, applied here for the first time by this port
///
/// `families::attn`'s row spelled the device symbol
/// `attn::attn_score_fold_heads` — the SAME string as `table::attn`'s row and
/// the same string `dsl::cuda::attn_score_fold_heads` states. One string for
/// the trace's name and the device's name is exactly what §60.6 split, and it
/// is why a `contract!` could not be written here before: the invariant is
/// that a contract symbol is never a unit row's symbol, and
/// `migration_status`'s `refused_set()` and `execution`'s
/// `a_walk_is_only_a_walk` both read it.
///
/// So the device row is `attn::attn_score_fold_heads_dev` and
/// `fire/attn_score.rs`'s `FOLD_SYMBOL` says so. **That constant is the only
/// firer**, and it resolves through `unit::unit_of` rather than through a
/// table, which is what makes the rename a one-line change instead of a
/// migration.
///
/// # `LaunchRule::Unstated`, argued rather than left blank
///
/// The launcher is `const dim3 grid(num_requests, 64u)` at 256 threads. `64`
/// is in no `Dims`: not heads, not requests, not pages, not a head dimension
/// — an occupancy constant, guessed once about one GPU. The body strides
/// `i += blockDim.x * gridDim.y`, so EVERY value of `gridDim.y` produces the
/// same floats and any parity test would pass a rule that is wrong by 64x in
/// blocks alone.
///
/// The tempting repair is `PerRequestFanout(64)`, and the measurement that
/// kills it: there are exactly TWO literal grid axes in all of `csrc/src`,
/// both in this one file — `(num_requests, 64u)` and `(cache.num_requests,
/// 32u)`. Different literals, no shared rule to extract, and §10.5 forbids
/// vocabulary growth for a single literal.
///
/// In fn-world the argument is shorter, because there is no rule to decline:
/// a `Launch` is written by whoever fires it, and `fire/attn_score.rs` carries
/// the 64 and the 256 as named constants with the `.cu` line beside them. The
/// number is a citation, not a derivation.
pub mod attention_flashinfer {
    use core::ffi::c_void;

    unit! {
        /// One row, `DeviceKernel::PLAIN`.
        ///
        /// # Not a template, and the header argues it must not become one
        ///
        /// Every buffer is `float` or page-table metadata, the block width
        /// arrives as `blockDim.x` and the fanout as `gridDim.y`, so a
        /// `template <int BLOCK>` would name a parameter the body never
        /// mentions and an arm that cannot differ from its sibling.
        unit ATTENTION_FLASHINFER = "attn/attention_flashinfer",
            text = include_str!("../../csrc/src/attn/attention_flashinfer.cuh"),
            file = "attn/attention_flashinfer.cuh";

        /// The fold: per-head scores summed to one row per request.
        ///
        /// Seven parameters where the trace's symbol states nine. `stream` was
        /// never an operand and `num_requests` is `grid.x` — the same two the
        /// row world dropped between a table row and a device row, here
        /// dropped once because there is only one list.
        ///
        /// Fired by `driver-cuda/src/fire/attn_score.rs`'s `FOLD_SYMBOL`,
        /// which is this string.
        fn attn_score_fold_heads = "attn::device::attn_score_fold_heads" (
            scores: *const c_void,
            score_indptr: *const i32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            page_size: i32,
            num_q_heads: i32,
            folded: *mut c_void,
        ) {
            "attn::attn_score_fold_heads_dev" => crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `attn/pack_dense_mask.cuh` — the two custom-mask packers, both plain
/// `__global__`s, both unit-only.
///
/// # Unit and nothing else, and this is the *emptiest* of the five arrangements
///
/// `split_packed` crossed as a unit with its host programs left in place.
/// These two cross as a unit with **no host program anywhere**. Their
/// `table::driver_internal` rows are already deleted, and
/// `driver-cuda/tests/launch_abi.rs:651-654` holds the evidence and the
/// verdict: *"`pack_dense_mask` and `pack_structured_mask` stood here and are
/// GONE with `attn/pack_dense_mask.cu`, its `.hpp` and their two
/// `table::driver_internal` rows. Empty consumer set on all five channels;
/// not ported, per §60.1."* `driver-cuda/src/fire/page_mask.rs` — the module
/// whose name suggests otherwise — plans the sideband arena, carves the mask
/// slots and compacts the page CSR, and launches neither kernel.
///
/// So there is no `contract!` and no `bind!`, and a `none:` arm would be
/// wrong: §5.1's rule is that `none:` surfaces as `Route::Unbound` at model
/// LOAD, and no trace can state a symbol the DSL has no statement for.
///
/// **The device text is kept and that is a decision, not an oversight.**
/// `tests/specialise.rs:2743-3298` is its reading consumer and a substantial
/// one: it compiles this unit through NVRTC, loads both plain rows, fires
/// `attn::pack_dense_mask` with `pack_dense_mask.cu:94`'s literal geometry
/// and compares every byte of the packed bitmap against a hand
/// transliteration of `pack_dense_mask.cuh:127-160`. It finds the unit
/// through `unit::UNITS`, which concatenates `families::ALL`, which lists
/// `x::attn::UNITS` beside `families::attn::UNITS` — so the move is
/// transparent to it. It also asserts `instantiation()` is
/// `::pie_cuda_driver::kernels::attn::device::pack_dense_mask` and that the
/// mangled name carries no `I...E` template bracket, both of which the rows
/// below keep.
///
/// # The geometry, preserved
///
/// Both launched `<<<B, 128, 0, stream>>>` — one block per lane at a fixed
/// 128 threads, with a stride loop over that lane's output bytes.
/// `pack_dense_mask.cu:93-94` and `:109-110`, before that file's deletion;
/// `families/attn.rs` recorded it as [`LaunchRule::PerRowNarrow`] *"to the
/// digit"* and added the caveat that survives the move: the 128 *"is not a
/// preference here the way it is for the audio tower — nothing folds warp
/// partials, so the width is not a numerics contract — but it is still the
/// launcher's, and a rule that widened it to 256 would state a launch this
/// tree does not make."* No `fn` writes that `Launch` today because nothing
/// fires these; whoever writes one writes `Launch { grid: [b, 1, 1], block:
/// [128, 1, 1], smem: 0, .. }` and cites this paragraph.
///
/// # `b` is an operand and the grid does not recover it
///
/// Both kernels READ it — `if (b >= B) return;` and `if (request >= B)
/// return;` are the first lines of each — so a declaration that dropped it on
/// the grounds that the grid's `x` extent already carries it would hand the
/// kernel whatever the previous launch left in that slot.
/// `PAGE_COMPACT_SIGS` keeps `num_requests` for the same reason.
///
/// # `DeviceKernel::PLAIN` and not `""`
///
/// The constant is the row's STATEMENT that this `__global__` has no template
/// parameter list; the empty string is what an unfilled field looks like. The
/// distinction is checked by NVRTC in both directions, and
/// `examples/argform_probe.rs` holds the measurement: `plain<device::bf16>`
/// is *"type name is not allowed"*, and a bare template path is *"cannot
/// determine which instance of function template … is intended"*. So a row
/// that states the wrong one of the two fails `tests/units.rs`, with NVRTC's
/// own sentence. **No device text changed for either row**, then or now.
///
/// # THE ONE THING THE FLOOR STILL OWES THIS ROOT, and it is not blocking
///
/// `runtime::args::is_pointer` (`src/runtime/args.rs:396`) does not list
/// `Ty::StructuredMasks`, so `Args::bind` falls through to its catch-all and
/// answers `ArgError::Unsupported`. That is the same refusal `emit::crossing`
/// made in the row world, from the same predicate. **It means the typed stub
/// `raw::pack_structured_mask` would panic at the bind** — `x::fire::fire`
/// binds through `Args::bind` exactly as `fire::hand::fire` does.
///
/// Nothing hits it today: no host program fires this kernel, which is why the
/// crossing is unit-only. It is stated here so that whoever writes one finds
/// the answer rather than the panic. The patch is one token, and the row
/// world already wrote the argument for it — *"the descriptor array IS a
/// device pointer, and saying so is a change to the `Ty` vocabulary rather
/// than to this row"*:
///
/// ```text
/// src/runtime/args.rs:396, in `is_pointer`'s `matches!` list:
///     | Ty::I32Array
/// +   | Ty::StructuredMasks
/// ```
///
/// It is NOT taken in this pass — **and the premise it was declined on has
/// since moved, so here is the re-derivation.** `src/emit.rs` was retired
/// (§6 half A, `0a03f064c`) and its `one()` was a reader: it asked
/// `is_pointer` whether every operand marshalled before emitting a typed
/// `fn`. Swept again after the retirement:
///
/// ```text
/// crates/kernels-cuda-new/src/runtime/args.rs:405  const fn is_pointer   (the definition)
/// crates/kernels-cuda-new/src/runtime/args.rs:507  Args::bind            (the ONLY caller)
/// ```
///
/// **One caller.** Everything else that names the string is prose — the
/// mentions in `abi.rs:996`, `device.rs:2072` and this module — or a
/// DIFFERENT function: `driver-cuda/src/bind/device.rs:226` is a second,
/// independent `const fn is_pointer` with its own single caller at `:293`.
/// The two lists are byte-identical, twenty-four entries in the same order,
/// **and both omit `Ty::StructuredMasks`**, so the patch is one token in two
/// files and taking only one of them makes the crates disagree about a type.
///
/// My own argument therefore flips: *"`is_pointer` is read by more than
/// `Args::bind`"* is no longer true. What is still true is the second half —
/// no host program fires this kernel — so the patch remains reviewable rather
/// than urgent. `device.rs::scalar`'s doc bounds the risk exactly: it is the
/// complement list, written as the closed SCALAR set on purpose, and *"the
/// two lists drifting apart costs a refusal, never a launch."* A `Ty` on
/// neither list is refused before a launch can happen; moving one onto the
/// pointer list can only remove a refusal, never mis-marshal a cell.
///
/// [`LaunchRule::PerRowNarrow`]: kernels::LaunchRule::PerRowNarrow
pub mod pack_dense_mask {
    use super::params::StructuredMaskParams;

    unit! {
        /// Two `__global__`s and no host code at all.
        ///
        /// The unit `instantiation()` could not spell while it could only
        /// write `path<...>`: neither kernel has a type or a compile-time
        /// value to abstract over — every buffer is `u8`/`u32`/`i32` mask
        /// metadata and the block width reaches them as `blockDim.x` — and
        /// `pack_dense_mask.cuh` refused to invent one on `mxfp4_marlin.cuh`'s
        /// precedent: *"a width parameter would be a lie that compiles."*
        unit PACK_DENSE_MASK = "attn/pack_dense_mask",
            text = include_str!("../../csrc/src/attn/pack_dense_mask.cuh"),
            file = "attn/pack_dense_mask.cuh";

        /// `pack_dense_mask.cuh:149` — a dense byte-per-cell mask packed to
        /// FlashInfer's bitmap ABI.
        ///
        /// `kvm_dense` is `[TOTAL_Q, STRIDE]` with one byte per cell (0/1);
        /// `mask_indptr` is the per-lane BYTE offset into `packed`
        /// (`[LANES+1]`, prefix-summed on the host from `ceil(qo_len[l] *
        /// klen[l] / 8)`); `qo_indptr` (`[LANES+1]`) gives each lane's
        /// query-row range; `packed` is pre-zeroed.
        ///
        /// `p_page` is `STRIDE`, the dense mask's logical row stride, and
        /// `b` is the lane count — see the module doc for why `b` is not
        /// recovered from the grid.
        ///
        /// Every buffer is unsourced and stays that way: `mask_indptr` is a
        /// host-built prefix sum the driver owns, `packed` is a pre-zeroed
        /// driver allocation, and `p_page` is the dense mask's row stride. No
        /// `Source` spells any of the three, which is why the row world's two
        /// rows were `table::driver_internal`'s and not `table::attn`'s.
        fn pack_dense_mask = "attn::device::pack_dense_mask" (
            kvm_dense: *const u8,
            klen: *const u32,
            qo_indptr: *const u32,
            mask_indptr: *const i32,
            packed: *mut u8,
            b: i32,
            p_page: i32,
        ) {
            "attn::pack_dense_mask" => crate::device::DeviceKernel::PLAIN,
        }

        /// `pack_dense_mask.cuh:189` — the same bitmap ABI materialised
        /// straight out of a causal / sliding-window / sink descriptor, with
        /// no dense tensor in between.
        ///
        /// `masks` is one [`StructuredMaskParams`] per lane, read as
        /// `masks[request]` at `pack_dense_mask.cuh:204`. Its mirror,
        /// its measured layout and the `Abi` impl behind this parameter are
        /// [`super::params`]; that module also records the two numberings of
        /// `kind` and which one this kernel reads.
        ///
        /// **This is the parameter that blocked the crossing**, and the row
        /// world was blocked on the same operand from the other side:
        /// `Ty::StructuredMasks` is a `Ty` that `runtime::args`' `is_pointer`
        /// does not admit, so `emit::crossing` refused it and the row had no
        /// generated entry point. A `unit!` declaration has no such refusal —
        /// `Abi` is an open set of impls and `is_pointer` is not consulted —
        /// so the declaration below is the first statement of this kernel's
        /// full signature that anything checks.
        fn pack_structured_mask = "attn::device::pack_structured_mask" (
            positions: *const u32,
            klen: *const u32,
            qo_indptr: *const u32,
            mask_indptr: *const i32,
            masks: *const StructuredMaskParams,
            packed: *mut u8,
            b: i32,
        ) {
            "attn::pack_structured_mask" => crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `attn/dsa_indexer.cuh` — glm5's sparse-attention index network, three
/// `__global__` templates, unit-only.
///
/// # The third unit-only crossing, and the reason is `split_packed`'s
///
/// All three host programs are already Rust and already where they belong:
/// `driver-cuda/src/fire/dsa_indexer.rs` holds `knorm_rope`, `q_rope` and
/// `topk_mask`, each firing its DEVICE symbol through `fire::hand::fire`.
/// None of them is a bind and none can become one here — two of the three
/// rows are unsourced in `table::attn` and the third's three integers arrive
/// on `Source::Param`, which is the statement's parameter channel and not a
/// `Cx` query. So the device text moves and nothing else does.
///
/// # THE SYMBOL SPLIT IS LIVE ON ALL THREE, and it is §60.6's
///
/// | table symbol (`table::attn`) | device symbol (declared below) |
/// |---|---|
/// | `attn::dsa_index_knorm_rope_bf16` | `attn::dsa_index_knorm_rope_dev` |
/// | `attn::dsa_index_q_rope_bf16` | `attn::dsa_index_q_rope_dev` |
/// | `attn::dsa_index_topk_mask` | `attn::dsa_index_topk_mask_dev` |
///
/// `fire/dsa_indexer.rs:45-61` holds both halves of each pair as constants
/// side by side and is the only thing that bridges them. A `unit!` row states
/// the DEVICE symbol; getting it wrong resolves nothing and fails at the
/// fire, not at a type. Note also that `_bf16` is DROPPED and not merely
/// suffixed: these are `template <class T>` and the ROW picks `T`, so the
/// format lives in the binding group and not in the string.
///
/// # The geometry, preserved — all three, and two of them state a shape no
/// # rule states
///
/// From `dsa_indexer.cu` before its deletion, and now from
/// `driver-cuda/src/fire/dsa_indexer.rs`, which holds the same numbers:
///
/// * `index_knorm_rope` — `<<<tokens, kBlock = 256, 0, stream>>>`, one block
///   per token. `LaunchRule::PerRow` and **not `Rms`**: `Rms` requests
///   thirty-two bytes of dynamic shared memory that no launcher passes and no
///   kernel here reads — `block_sum`'s warp buffer, which this shape has no
///   reduction to need, because its reduction is a static `__shared__ float
///   red[256]`. Harmless in effect and wrong as a contract: a rule is meant to
///   REPRODUCE its launcher, and one that asks for memory the launcher did not
///   is a rule nobody can check against the `<<<>>>` it came from.
/// * `index_q_rope` — `<<<tokens, round_up(n_heads, 32), 0, stream>>>` with a
///   one-warp floor (`dsa_indexer.cu:34-35`, now
///   `fire::dsa_indexer::q_rope_block`). ONE THREAD PER HEAD. No rule states
///   it and none can: every rule that sizes a block on a row sizes it on the
///   row's WIDTH, and `idx_q`'s row is `n_heads * head_dim` — the two differ
///   by a factor of 64 or 128. `LaunchRule::RouteRows` would open 128× the
///   block. The block is a statement PARAMETER, not a rectangle. **In fn-world
///   that objection evaporates**: a `Launch` is a `fn`'s literal, and the
///   `fn` already exists in the driver.
/// * `index_topk_mask` — `<<<tokens, kBlock, tokens * sizeof(float),
///   stream>>>`, guarded by `if (tokens <= 0) return;`.
///   `LaunchRule::RowScores` was ported FROM this launcher and states grid,
///   block AND the dynamic allocation exactly: `rows * 4` bytes is `tokens *
///   sizeof(float)` written twice. **The shared allocation is why it is
///   neither `Rms` nor `PerRow`**: the kernel declares `extern __shared__
///   float logit[]` and fills `logit[0..nkeys)` where `nkeys = blockIdx.x +
///   1` — one float per KEY, and every key of this fire is a row of it. At
///   `Rms`' thirty-two bytes the last row of a 4,096-token prefill would
///   select its top-k from eight floats it wrote and 4,088 it did not; at
///   `PerRow`'s zero, from none. Neither faults. `dsa_indexer.cuh`'s own
///   words: *"a launch that under-sizes shared memory does not fail, it reads
///   another block's floats"* — a wrong mask, a wrong attention, and nothing
///   downstream checks it.
///
/// # The extents that stay operands
///
/// `tokens` is gone from `knorm_rope` — one block per row IS `tokens` and the
/// kernel never addresses with it. `head_dim` stays, because the kernel
/// strides over it. `N` stays on `topk_mask` although the grid opens over it,
/// because the kernel needs it a second time as the pitch of `mask` (`mrow =
/// mask + i * N`) and as the bound of its causal zero-fill. **An extent a rule
/// recovers is not an operand — an extent a kernel ADDRESSES with is.**
///
/// # `kMaxRopeDim` is a bound this declaration cannot state
///
/// Both RoPE kernels stage `rope_dim` floats in a per-thread `float buf[256]`
/// before rotating them. `rope_dim` is a run-time value, so the array cannot
/// be sized on it, and **a model with `rope_dim > 256` overruns it**. Nothing
/// in a `unit!`, a `Launch` or a `Refusal` can see that: it is a device-side
/// local. `dsa_indexer.cuh` states the bound in its own header and this
/// paragraph is the second place it is written, because a host program that
/// eventually binds these must refuse above 256 and no type will remind it.
pub mod dsa_indexer {
    use super::bf16;

    /// `dsa_indexer.cuh`'s `kBlock`, which `knorm_rope` and `topk_mask` both
    /// open — `fire/dsa_indexer.rs:64` held the same 256.
    ///
    /// It is a file-scope `constexpr int` the kernels stride by rather than a
    /// template argument, so there is no non-type argument to pin it with and
    /// a launcher's block width has to AGREE with the header instead. It
    /// does; that agreement is the whole of the contract.
    pub const K_BLOCK: u32 = 256;

    /// `dsa_indexer.cu:34-35` — `index_q_rope`'s block width.
    ///
    /// ```text
    /// :34   int block = ((n_heads + 31) / 32) * 32;
    /// :35   if (block < 32) block = 32;
    /// ```
    ///
    /// One thread per HEAD, rounded up to a whole warp, with a floor of one
    /// warp so `n_heads == 0` does not produce a zero-width block. This is
    /// why the launcher could never be a `LaunchRule`: a rule reads a
    /// rectangle, and the head count that sizes this block is a statement
    /// parameter.
    #[must_use]
    pub fn q_rope_block(n_heads: i32) -> u32 {
        let rounded = ((n_heads.max(0) + 31) / 32) * 32;
        #[allow(clippy::cast_sign_loss)]
        let block = rounded as u32;
        if block < 32 { 32 } else { block }
    }

    unit! {
        /// Three `__global__` templates and the RoPE helper they share. No
        /// host code.
        ///
        /// The rotation is INTERLEAVED and not split-half — pairs are `(2i,
        /// 2i+1)`, which is what glm5's index network trains against. A
        /// split-half rotation on the same buffer is a different function and
        /// the two agree only when `rope_dim` is 2, so getting it wrong is
        /// invisible in a unit test with tiny dims.
        unit DSA_INDEXER = "attn/dsa_indexer",
            text = include_str!("../../csrc/src/attn/dsa_indexer.cuh"),
            file = "attn/dsa_indexer.cuh";

        /// `dsa_indexer.cuh:106` — LayerNorm over `head_dim` then interleaved
        /// RoPE, in place on `idx_k`.
        ///
        /// **LayerNorm and not RMSNorm**: the mean is subtracted and there is
        /// a bias. That is why `w` and `b` are two operands where a `norm`
        /// kernel would have one, and why this file cannot borrow `norm`'s
        /// reduction — the prelude's `block_sum` folds in a different order,
        /// and the last bit of this LayerNorm feeds a top-k RANKING.
        ///
        /// Fired by `driver-cuda/src/fire/dsa_indexer.rs::knorm_rope` with
        /// `Launch { grid: [tokens, 1, 1], block: [256, 1, 1], smem: 0 }`.
        fn index_knorm_rope = "attn::device::index_knorm_rope" <T> (
            idx_k: *mut T,
            w: *const T,
            b: *const T,
            positions: *const i32,
            head_dim: i32,
            rope_dim: i32,
            theta: f32,
            eps: f32,
        ) where *const T, *mut T {
            "attn::dsa_index_knorm_rope_dev" => where [T = bf16] "device::bf16",
        }

        /// `dsa_indexer.cuh:151` — interleaved RoPE on the first `rope_dim`
        /// of each index head of `idx_q`.
        ///
        /// One block per token, one thread per head, `if (h >= n_heads)
        /// return;`. `n_heads` is passed AND sizes the block, which is
        /// `Control::Supplies` exactly — see the module doc for why no rule
        /// can state that shape.
        ///
        /// Fired by `driver-cuda/src/fire/dsa_indexer.rs::q_rope`.
        fn index_q_rope = "attn::device::index_q_rope" <T> (
            idx_q: *mut T,
            positions: *const i32,
            n_heads: i32,
            head_dim: i32,
            rope_dim: i32,
            theta: f32,
        ) where *mut T {
            "attn::dsa_index_q_rope_dev" => where [T = bf16] "device::bf16",
        }

        /// `dsa_indexer.cuh:187` — causal top-k mask over the index scores,
        /// one block per query token.
        ///
        /// ```text
        /// logit[i, j] = sum_h relu(q[i, h] . k[j]) * w[i, h]
        /// ```
        ///
        /// The softmax scale is monotonic and therefore irrelevant to a
        /// RANKING, so it is omitted rather than computed and divided out.
        /// The threshold is forty rounds of bisection on the logit range and
        /// not a sort: a sort of `nkeys` floats per block costs shared memory
        /// proportional to the sequence and a partial sort still has to be
        /// exact at the boundary. Forty halvings of an fp32 interval reach the
        /// representable neighbourhood of the true k-th value, and the tie
        /// behaviour (`>= thr` admits every equal logit) is the original's —
        /// so a row of equal scores admits more than `topk` keys, exactly as
        /// it did.
        ///
        /// `template <class T>` and nothing else: `kBlock` is a file-scope
        /// `constexpr int` the kernel strides by, not a template argument, so
        /// there is no non-type argument to cite and the 256 a launcher opens
        /// has to agree with `dsa_indexer.cuh`'s `kBlock` instead. It does.
        ///
        /// Fired by `driver-cuda/src/fire/dsa_indexer.rs::topk_mask`, whose
        /// `Launch` carries the `rows * 4` shared allocation the module doc
        /// argues for.
        fn index_topk_mask = "attn::device::index_topk_mask" <T> (
            idx_q: *const T,
            idx_k: *const T,
            idx_w: *const T,
            mask: *mut u8,
            n: i32,
            n_heads: i32,
            head_dim: i32,
            topk: i32,
        ) where *const T {
            "attn::dsa_index_topk_mask_dev" => where [T = bf16] "device::bf16",
        }
    }
}

// ---------------------------------------------------------------------------
// THE DSA INDEXER'S THREE HOST PROGRAMS.
//
// `driver-cuda/src/fire/dsa_indexer.rs` held these, and its only consumer
// was `bind::service`'s three entry points, so the whole module crosses.
// None of the three needs a driver resource -- a grid, a block and a stream
// is the entire host side -- so all three are MOVES rather than driver ops.
//
// `Indexer` / `IndexerDecline` do not cross with them. Both were one variant
// wide (`tokens <= 0`) and `Fired::Declined(Refusal::Empty { what: "tokens" })`
// says the same thing in the floor's own vocabulary, so a family-local enum
// that carried exactly one fact is one fact less to keep in step.
// ---------------------------------------------------------------------------

/// `attn/page_compact.cuh` — dropping the pages a keep-mask rejects and
/// rewriting the CSR so the survivors are contiguous.
///
/// Two `__global__`s that must run in order, and the dependency between them
/// is a device buffer: the first counts, the second scans its counts and
/// scatters. That is what `execution::COMPOSED` recorded, and a `fn` firing
/// two units on one stream is what it becomes.
pub mod page_compact {
    unit! {
        /// The page compactor, both halves.
        ///
        /// REPLACES `families::attn`'s `PAGE_COMPACT`, which declared the
        /// same two rows over the same root. A root is in exactly one of the
        /// two lists.
        unit PAGE_COMPACT = "attn/page_compact",
            text = include_str!("../../csrc/src/attn/page_compact.cuh"),
            file = "attn/page_compact.cuh";

        /// `page_compact.cuh:212` — one block per request: how many of its
        /// pages the mask keeps.
        ///
        /// `BLOCK` is a NON-TYPE parameter and a compile-time width, not a
        /// launch geometry: `__shared__ u32 tmp[BLOCK / 32]` and
        /// `block_sum_u32<BLOCK>` are both sized by it, so the 256 in the
        /// instantiation and the 256 in the `<<<>>>` are the same constant
        /// and the `.cuh` says so at `:114` — `constexpr int kBlock = 256`.
        fn count_kept = "attn::device::count_kept"(
            page_indptr_in: *const u32,
            keep: *const u8,
            keep_stride: u32,
            num_requests: i32,
            counts: *mut u32,
        ) {
            "attn::count_kept" => "device::i32(256)",
        }

        /// `page_compact.cuh:242` — scan and scatter, fused into one launch.
        ///
        /// The `.cuh`'s own note on why there is no third kernel: *"the only
        /// thing block `r` needed from the separate scan pass was its own
        /// output base ... so the block can just add them up itself.
        /// Recomputing an O(R) sum per block is far cheaper than the kernel
        /// launch it replaces, because this runs once per LAYER per fire."*
        ///
        /// **The last three parameters are not in the op's declared order.**
        /// The op declares `page_indices_out, page_indptr_out,
        /// last_page_lens_out`; the kernel takes them as below. All three are
        /// `u32` pointers, so a transposition type-checks in both languages —
        /// this declaration is transcribed from the `__global__` and the host
        /// program below passes them in this order.
        fn scan_and_scatter = "attn::device::scan_and_scatter"(
            page_indices_in: *const u32,
            page_indptr_in: *const u32,
            last_page_lens_in: *const u32,
            keep: *const u8,
            counts: *const u32,
            keep_stride: u32,
            num_requests: i32,
            page_indptr_out: *mut u32,
            last_page_lens_out: *mut u32,
            page_indices_out: *mut u32,
        ) {
            "attn::scan_and_scatter" => "device::i32(256)",
        }
    }

    /// `page_compact.cuh:114` — `constexpr int kBlock = 256`.
    ///
    /// The block width AND the shared reduction's template argument, which is
    /// why it is one constant. Both `<<<>>>` spelled it `device::kBlock`.
    pub const K_BLOCK: u32 = 256;
}

/// `attn::compact_page_csr` — the page compactor's host program.
///
/// ```text
/// :45   device::count_kept<device::kBlock>
/// :46       <<<num_requests, device::kBlock, 0, stream>>>(...);
/// :48   device::scan_and_scatter<device::kBlock>
/// :49       <<<num_requests, device::kBlock, 0, stream>>>(...);
/// ```
///
/// **BOTH REFUSALS ARE RESOLVED BEFORE THE FIRST LAUNCH**, which is §5.1's
/// rule for every multi-launch body: a `Declined` returned after something
/// has gone to the device says nothing ran, and something ran. Neither guard
/// depends on a device-side value, so neither has to be a device-side branch.
///
/// The second guard is the one a launch rule could never have answered.
/// `execution::COMPOSED` called the first half *"`Ungeometric::Empty` from
/// `Dims::rows`, which every rule already answers"* — true — and the null
/// scratch is not a geometry check at all: the buffer CARRIES the dependency
/// between the two launches, so a null one is a caller that has not allocated
/// the thing the composition is about.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across BOTH
/// launches — `scratch_counts` especially, which is written by the first and
/// read by the second — and `stream` is the caller's stream.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn compact_page_csr(
    page_indices_in: *const u32,
    page_indptr_in: *const u32,
    last_page_lens_in: *const u32,
    keep: *const u8,
    scratch_counts: *mut u32,
    keep_stride: u32,
    num_requests: i32,
    page_indices_out: *mut u32,
    page_indptr_out: *mut u32,
    last_page_lens_out: *mut u32,
    stream: *mut c_void,
) -> Fired {
    // `page_compact.cu:44`, split so the caller learns which half refused.
    if num_requests <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if scratch_counts.is_null() {
        return Fired::Declined(Refusal::Absent { what: "the compaction scratch buffer" });
    }
    let launch = Launch::per_row(num_requests.unsigned_abs(), page_compact::K_BLOCK);
    // SAFETY: the caller's obligation, above.
    unsafe {
        // Step one — `:45`. Fills `scratch_counts`.
        page_compact::raw::count_kept(
            "attn::count_kept",
            launch,
            page_indptr_in,
            keep,
            keep_stride,
            num_requests,
            scratch_counts,
            stream,
        );
        // Step two — `:48`. Reads what step one wrote. Same stream, so the
        // ordering is the stream's and needs no event. This is `Composed`
        // written as two statements in a `fn`, in order, which is what §5.1
        // says a composition is now that no floor combinator is coming.
        page_compact::raw::scan_and_scatter(
            "attn::scan_and_scatter",
            launch,
            page_indices_in,
            page_indptr_in,
            last_page_lens_in,
            keep,
            scratch_counts,
            keep_stride,
            num_requests,
            page_indptr_out,
            last_page_lens_out,
            page_indices_out,
            stream,
        );
    }
    Fired::Launched
}

/// `attn/attention_naive.cuh` — the MTP pair and the reference attention.
///
/// **The root holds FIVE `__global__` templates and this declares THREE.**
/// `families::attn`'s `ATTENTION_NAIVE` declared the same three and this
/// unit REPLACES it: a root is in exactly one of the two lists, because a
/// second `unit!` over the same text compiles it twice and `unit_of` answers
/// with whichever won.
///
/// The `.cuh` says why the other two have no row: *"NO ROW STATES THIS
/// KERNEL: per-head grid, extent-sized shared memory"* (`attn_mtp_history`)
/// and the same plus *"a launcher that chooses between this kernel and
/// `attn_mtp_history`"* (`attn_mtp_paged_history`). Neither has a host
/// program anywhere in this tree, so declaring them would be a `fn` nobody
/// can call. The `text` is the whole file regardless, which is the unit's
/// point: the device half is not partial, only the `fn`s are.
pub mod attention_naive {
    use super::bf16;

    unit! {
        /// Multi-token prediction's two hidden-state movers.
        unit ATTENTION_NAIVE = "attn/attention_naive",
            text = include_str!("../../csrc/src/attn/attention_naive.cuh"),
            file = "attn/attention_naive.cuh";

        /// `attention_naive.cuh:305` — the previous step's pending hidden
        /// state becomes this step's first token, per request.
        ///
        /// `slot_ids` is NULLABLE and the null branch means slot zero: a
        /// single-request fire does not carry a slot table.
        fn mtp_shift_hidden = "attn::device::mtp_shift_hidden" <T> (
            target_hidden: *const T,
            pending_hidden: *const T,
            qo_indptr: *const u32,
            slot_ids: *const i32,
            out: *mut T,
            num_requests: i32,
            hidden_size: i32,
        ) where *const T, *mut T {
            "attn::mtp_shift_hidden_dev" => where [T = bf16] "device::bf16",
        }

        /// `attention_naive.cuh:337` — the end-of-step refresh: each
        /// request's LAST target hidden state becomes its slot's pending
        /// state for the next step.
        ///
        /// The `.cuh`'s note is a measurement worth keeping: *"NO ROW STATED
        /// THIS KERNEL for as long as every ported rule opened its grid over
        /// rows: a fire of eight requests and ninety-three tokens would open
        /// ninety-three blocks — eighty-five of them writing a slot that is
        /// not theirs."* One block per REQUEST, and the grid is the whole
        /// difference between this and its twin.
        /// `attention_naive.cuh:104` — the reference attention, kept so a
        /// parity test has something to compare flashinfer against on a
        /// shape flashinfer does not cover.
        ///
        /// **NO STATEMENT LOWERS TO IT AND IT HAS NO HOST PROGRAM.** It is
        /// declared because it is a device row `families::attn` carried and
        /// this unit replaces that one; the launcher measurement travels
        /// with it, from the `LaunchRule::SdpaVector` doc that stated it:
        ///
        /// ```text
        /// dim3 grid(num_q_heads, num_tokens);  dim3 block(256);
        /// smem = sizeof(float) * (num_tokens + BLOCK)
        /// ```
        ///
        /// The shared allocation is the whole reason no other rule could
        /// stand in. `attn_naive` lays `scores[num_tokens]` and
        /// `reduce_buf[BLOCK]` in one `extern __shared__` block and takes the
        /// second as `smem + num_tokens`; launched with less, the reduction
        /// scratch overlaps the scores it is reducing, the softmax
        /// denominator is computed from bytes the same kernel is overwriting,
        /// and THE ANSWER IS FINITE. `scale` is the launcher's
        /// `1 / sqrtf(head_dim)`, a host computation rather than an operand.
        fn attn_naive = "attn::device::attn_naive" <T> (
            q: *const T,
            k: *const T,
            v: *const T,
            o: *mut T,
            num_tokens: i32,
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            scale: f32,
        ) where *const T, *mut T {
            "attn::attention_naive_bf16" => where [T = bf16] "device::bf16",
        }

        fn mtp_update_pending_hidden = "attn::device::mtp_update_pending_hidden" <T> (
            target_hidden: *const T,
            pending_hidden: *mut T,
            qo_indptr: *const u32,
            slot_ids: *const i32,
            num_requests: i32,
            hidden_size: i32,
        ) where *const T, *mut T {
            "attn::mtp_update_pending_hidden_dev" => where [T = bf16] "device::bf16",
        }
    }

    /// `attention_naive.cu:57` — `constexpr int BLOCK = device::BLOCK;`,
    /// which is `attention_naive.cuh:91`'s `256`.
    ///
    /// Both launchers spell it, and it is the only width either uses.
    pub const BLOCK: u32 = 256;
}

/// `attn::mtp_shift_hidden_bf16` — one block per TOKEN.
///
/// ```text
/// :64   device::mtp_shift_hidden<bf16><<<total_tokens, BLOCK, 0, stream>>>(
/// ```
///
/// `total_tokens` is the grid and does not reach the kernel; `num_requests`
/// does, because it bounds `find_request_u32`'s scan and a request count is
/// not a row count.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mtp_shift_hidden_bf16(
    target_hidden: *const bf16,
    pending_hidden: *const bf16,
    qo_indptr: *const u32,
    slot_ids: *const i32,
    out: *mut bf16,
    total_tokens: i32,
    num_requests: i32,
    hidden_size: i32,
    stream: *mut c_void,
) -> Fired {
    // `attention_naive.cu:60-63`, one `if` with four clauses, split so the
    // caller learns which one refused.
    if total_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if num_requests <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if hidden_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden width" });
    }
    if pending_hidden.is_null() {
        // NOT a geometry check: the pending buffer IS the MTP state, so a
        // fire without one is a fire this pair has nothing to do for.
        return Fired::Declined(Refusal::Absent { what: "the MTP pending-hidden state" });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        attention_naive::raw::mtp_shift_hidden(
            "attn::mtp_shift_hidden_dev",
            Launch::per_row(total_tokens.unsigned_abs(), attention_naive::BLOCK),
            target_hidden,
            pending_hidden,
            qo_indptr,
            slot_ids,
            out,
            num_requests,
            hidden_size,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::mtp_update_pending_hidden_bf16` — one block per REQUEST.
///
/// ```text
/// :85   device::mtp_update_pending_hidden<bf16><<<num_requests, BLOCK, 0, stream>>>(
/// ```
///
/// Stashes each request's LAST hidden state into the pending buffer, so the
/// next step's [`mtp_shift_hidden_bf16`] has something to shift in. The twin
/// is per-row and this is per-request, and the statement is the reason: it
/// records a `StateRef` and no result, so it names no rectangle of its own.
///
/// # Safety
///
/// [`mtp_shift_hidden_bf16`]'s.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mtp_update_pending_hidden_bf16(
    target_hidden: *const bf16,
    pending_hidden: *mut bf16,
    qo_indptr: *const u32,
    slot_ids: *const i32,
    num_requests: i32,
    hidden_size: i32,
    stream: *mut c_void,
) -> Fired {
    // `attention_naive.cu:84-86`.
    if num_requests <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if hidden_size <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden width" });
    }
    if pending_hidden.is_null() {
        return Fired::Declined(Refusal::Absent { what: "the MTP pending-hidden state" });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        attention_naive::raw::mtp_update_pending_hidden(
            "attn::mtp_update_pending_hidden_dev",
            Launch::per_row(num_requests.unsigned_abs(), attention_naive::BLOCK),
            target_hidden,
            pending_hidden,
            qo_indptr,
            slot_ids,
            num_requests,
            hidden_size,
            stream,
        );
    }
    Fired::Launched
}

// ---------------------------------------------------------------------------
// `attn/mla_paged.cu`'S TWO HOST PROGRAMS.
//
// `driver-cuda/src/fire/mla_paged.rs` held both and is deleted with this
// change. Neither needs a driver resource -- two `<<<>>>`, one grid each,
// no handle and no cache -- so both are MOVES.
//
// They cannot bind, and the reason is one query: [`Cx::mla_layer`], which
// refuses because NOTHING FILLS IT. That is a different refusal from the
// `dsv4` three's ratio, and the difference is worth keeping: the ratio has
// no producer ANYWHERE, whereas the MLA layer view has a producer --
// `pools::mla_cache::MlaCachePool::layer_view` -- that no `Fire` can reach.
// `Cx::mla_layer`'s own doc states the remedy and its cost: `AttnCtx`
// carrying the MLA layer list the way it carries the paged one.
//
// TWO TYPES DISSOLVE INTO THE FLOOR HERE, and both were the driver spelling
// something `Cx` already spells:
//
// * `MlaCacheLayerView`, taken BY VALUE and unpacked before both `<<<>>>`,
//   is [`MlaLayer`] field for field. `execution::Control::Supplies` named
//   `page_size`, `kv_lora_rank` and `qk_rope_head_dim` as *"three operands
//   the kernel declares and no `Source` can reach, because the view is one
//   dispatch argument and its fields are five"*. A `fn` taking the view
//   reaches all five, so that `Walk` dissolves rather than being satisfied.
// * `YarnOriginal` is [`Yarn`] with the fields in a different order. One
//   struct, not two, and the `Option` is the C++'s nullable pointer with
//   the null removed -- which the driver had already done.
//
// `MlaDecline` had ONE variant, `NoTokens`, for a guard both launchers open
// with. It is `Refusal::Empty { what: "tokens" }`.
// ---------------------------------------------------------------------------

/// `mla_paged.cu:52` — `constexpr int BS = 256;`, the prepare block.
///
/// A block width AND the comparison `half >= BS` that picks
/// [`mla_heads_per_block`], which is why it is named once and used twice.
pub const MLA_PREPARE_BLOCK: i32 = 256;

/// `mla_paged.cu:105` — `write_mla`'s block, one per token row.
pub const MLA_WRITE_BLOCK: u32 = 256;

/// `mla_paged.cu:64` — the query lane's head packing.
///
/// ```text
/// :63   const int half = rope / 2;
/// :64   const int heads_per_block = half >= BS ? 1 : (BS / half);
/// ```
///
/// The comment beside it in the C++ is a MEASUREMENT and travels with the
/// arithmetic rather than being consumed by the port:
///
/// > Match `kernels::rope::rope_bf16`'s head packing so the query lane has
/// > the same shape of work per block that the standalone kernel had.
///
/// `half` is `qk_rope_head_dim / 2`, one thread per rotated pair. It is
/// `pub` because `execution::Control::Supplies`'s doc names this exact value
/// as its worked example — *"passed to the kernel AND divides the head axis
/// of the grid"* — and a reader who follows that sentence should land on the
/// arithmetic rather than on a private copy of it.
#[must_use]
pub fn mla_heads_per_block(rope: i32) -> i32 {
    let half = rope / 2;
    if half >= MLA_PREPARE_BLOCK {
        1
    } else if half > 0 {
        MLA_PREPARE_BLOCK / half
    } else {
        // `half == 0` would divide by zero. The C++ could not reach it —
        // `qk_rope_head_dim` is a layer field and never 0 for an MLA layer —
        // and Rust does not get to say "could not reach it" in a division,
        // so it says 1 and the grid stays valid.
        1
    }
}

/// `mla_paged.cu:65` — the grid's second axis, less its KV lane.
///
/// ```text
/// :65   const int q_blocks = (heads + heads_per_block - 1) / heads_per_block;
/// ```
#[must_use]
pub fn mla_q_blocks(heads: i32, heads_per_block: i32) -> i32 {
    if heads_per_block <= 0 {
        return 0;
    }
    heads.saturating_add(heads_per_block - 1) / heads_per_block
}

/// `attn::mla_prepare_bf16` — the whole MLA prologue in one kernel.
///
/// The `kv_a` RMSNorm, the `k_pe` rotation, the paged write of both, and the
/// query-side nope/pe split: one grid lane for the KV work and `q_blocks`
/// lanes for the heads.
///
/// ```text
/// :67   dim3 grid(total_tokens, 1 + q_blocks);
/// :68   device::mla_prepare<BS><<<grid, BS, 0, stream>>>(
/// ```
///
/// **The `1 +` is not spare capacity.** `mla_paged.cuh:236` reads
/// `const int qb = blockIdx.y - 1;` and takes the KV path when `qb < 0`, so
/// lane `y = 0` does the `kv_a` RMSNorm, the `k_pe` rotation and the paged
/// write for its token, and lanes `1..=q_blocks` are the query heads.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// `layer`'s two page pointers included, and `stream` is the caller's stream.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub unsafe fn mla_prepare_bf16(
    layer: MlaLayer,
    kv_a: *const bf16,
    kv_a_norm_weight: *const bf16,
    q_b: *const bf16,
    kv_c: *mut bf16,
    k_pe: *mut bf16,
    q_nope: *mut bf16,
    q_pe: *mut bf16,
    positions: *const i32,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    row_valid: *const u8,
    total_tokens: i32,
    num_requests: i32,
    heads: i32,
    qk_nope_head_dim: i32,
    eps: f32,
    theta: f32,
    interleaved: bool,
    kv_a_row_stride: i32,
    yarn: Option<Yarn>,
    stream: *mut c_void,
) -> Fired {
    // `mla_paged.cu:47`. Both launchers open one grid lane per token, so an
    // empty batch is an empty grid, which CUDA rejects.
    if total_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    let kv_lora = layer.kv_lora_rank;
    let rope = layer.qk_rope_head_dim;
    // `:55-56` — a non-positive stride means "rows are packed", and the
    // packed width is the two planes side by side.
    let stride = if kv_a_row_stride > 0 { kv_a_row_stride } else { kv_lora + rope };
    let per_block = mla_heads_per_block(rope);
    let blocks = mla_q_blocks(heads, per_block);

    // `:61-66` — the ramp, on the host, before the launch.
    // `rope_device.cuh`'s `yarn_original_ramp_bounds` is
    // `__host__ __device__` and shared with the fused rope kernels; the Rust
    // transcription is shared the same way, so the two cannot disagree about
    // where the ramp starts. It is now one crate away rather than two.
    let (low_dim, high_dim) = match yarn {
        Some(y) => crate::x::rope::ramp_bounds(
            rope,
            theta,
            y.beta_fast,
            y.beta_slow,
            y.original_max_position,
        ),
        // `:60` — `float low_dim = 0.f, high_dim = 0.f;`, left untouched
        // when `yarn == nullptr`. The kernel reads them only when
        // `yarn_factor > 0`, and the sentinel below turns that off.
        None => (0.0, 0.0),
    };
    // `:81` and `:83` — the two sentinels. `-1.f` for the factor is what
    // "no YaRN" is spelled as on the device side; `1.f` is the identity
    // magnitude correction.
    let yarn_factor = yarn.map_or(-1.0_f32, |y| y.factor);
    let yarn_mscale = yarn.map_or(1.0_f32, |y| y.attention_factor);

    // SAFETY: the caller's obligation, above.
    unsafe {
        mla_paged::raw::mla_prepare(
            "attn::mla_prepare",
            Launch {
                grid: [total_tokens.unsigned_abs(), blocks.saturating_add(1).max(1).unsigned_abs(), 1],
                block: [MLA_PREPARE_BLOCK.unsigned_abs(), 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            kv_a,
            kv_a_norm_weight,
            q_b,
            kv_c,
            k_pe,
            q_nope,
            q_pe,
            layer.ckv_pages.cast(),
            layer.kpe_pages.cast(),
            positions,
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            row_valid,
            num_requests,
            layer.page_size,
            heads,
            kv_lora,
            qk_nope_head_dim,
            rope,
            stride,
            eps,
            theta,
            interleaved,
            per_block,
            yarn_factor,
            low_dim,
            high_dim,
            yarn_mscale,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::write_mla_to_pages` — appends one step's compressed latent and rope
/// plane to the paged MLA cache.
///
/// ```text
/// :105   device::write_mla<<<total_tokens, 256, 0, stream>>>(
/// ```
///
/// One block per token row, which is [`Launch::per_row`] to the digit:
/// `ckv_curr` is shaped `[Tokens, kv_lora_rank]` and this opens one block per
/// row of it.
///
/// The C++ had a two-line forwarder, `write_mla_to_pages_bf16`, which the
/// archive's Rust deleted as dead (§60.1: *"a port of a launcher with an
/// empty consumer set is a contract nobody signed"*) and folded here. The
/// `<<<>>>` has been in one place since.
///
/// # Safety
///
/// [`mla_prepare_bf16`]'s.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn write_mla_to_pages(
    layer: MlaLayer,
    ckv_curr: *const bf16,
    kpe_curr: *const bf16,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    row_valid: *const u8,
    total_tokens: i32,
    num_requests: i32,
    stream: *mut c_void,
) -> Fired {
    // `mla_paged.cu:104`, which the forwarder at `:116` reached through.
    if total_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        mla_paged::raw::write_mla(
            "attn::write_mla",
            Launch::per_row(total_tokens.unsigned_abs(), MLA_WRITE_BLOCK),
            ckv_curr,
            kpe_curr,
            layer.ckv_pages.cast(),
            layer.kpe_pages.cast(),
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            row_valid,
            num_requests,
            layer.page_size,
            layer.kv_lora_rank,
            layer.qk_rope_head_dim,
            stream,
        );
    }
    Fired::Launched
}

// ---------------------------------------------------------------------------
// `attn/dsv4_compress.cu`'S THREE SURVIVING HOST PROGRAMS.
//
// `driver-cuda/src/fire/dsv4_compress.rs` held these and
// `combine_attn_outputs_bf16`, which crossed earlier and BOUND. These three
// cross the same way and refuse, and the difference between the four is the
// whole content of this block: `combine` needed only what its statement
// said, and these need what deepseek_v4's statements do not carry.
//
// None of the three needs a driver resource. All three are MOVES.
//
// `Dsv4` / `Dsv4Decline` do not cross. Five variants, and every one is a
// `Refusal` the floor already spells: `NoElements`, `NoRatio`,
// `NoQueryTokens` and `NoHeads` are all `Refusal::Empty { what }` with the
// extent named, which is strictly more than an enum whose variant names had
// to be read against the C++ to know which `<=` they came from.
//
// `ATTN_BLOCK` is 128, and it is the block width AND half the shared
// allocation's second term, which is why it is one constant and not two.
// `META_BLOCK` is a plain elementwise 128, stated locally in both boundary
// launchers as `const int threads = 128;`.
// ---------------------------------------------------------------------------

/// `dsv4_compress.cu:37` — `constexpr int ATTN_BLOCK = 128;`.
#[cfg(feature = "_cuda")]
const DSV4_ATTN_BLOCK: u32 = 128;

/// `dsv4_compress.cu:139` and `:161` — the boundary-meta block.
#[cfg(feature = "_cuda")]
const DSV4_META_BLOCK: u32 = 128;

/// `attn::dsv4_boundary_meta_decode` — each decode row's compressed-block
/// boundary metadata: the position it lands on, the request it belongs to,
/// and the rope index.
///
/// ```text
/// :138   if (n <= 0 || ratio <= 0) return;
/// :139   const int threads = 128;
/// :140   const int blocks = (n + threads - 1) / threads;
/// :141   device::dsv4_boundary_meta_decode<<<blocks, threads, 0, stream>>>(
/// ```
///
/// `LaunchRule::Elementwise` to the digit, which [`Launch::flat`] is.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the launch,
/// and `stream` is the caller's stream.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_boundary_meta_decode(
    positions: *const i32,
    out_pos: *mut i32,
    out_req: *mut i32,
    out_rope: *mut i32,
    n: i32,
    ratio: i32,
    row_valid: *const u8,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "elements" });
    }
    if ratio <= 0 {
        // The compression ratio DIVIDES a position, so zero or negative is a
        // division the kernel would do and this program will not reach.
        return Fired::Declined(Refusal::Narrow { what: "ratio", at: ratio });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        dsv4_compress::raw::dsv4_boundary_meta_decode(
            "attn::dsv4_boundary_meta_decode_dev",
            Launch::flat(n.unsigned_abs(), DSV4_META_BLOCK),
            positions,
            out_pos,
            out_req,
            out_rope,
            n,
            ratio,
            row_valid,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::dsv4_boundary_meta_paged` — the prefill form of
/// [`dsv4_boundary_meta_decode`].
///
/// Same geometry, same launcher shape; it differs in one line, resolving the
/// request index by a binary search over `qo_indptr` instead of shortcutting
/// it to the token index.
///
/// # Safety
///
/// [`dsv4_boundary_meta_decode`]'s.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsv4_boundary_meta_paged(
    positions: *const i32,
    qo_indptr: *const u32,
    out_pos: *mut i32,
    out_req: *mut i32,
    out_rope: *mut i32,
    n: i32,
    num_requests: i32,
    ratio: i32,
    row_valid: *const u8,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "elements" });
    }
    if ratio <= 0 {
        return Fired::Declined(Refusal::Narrow { what: "ratio", at: ratio });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        dsv4_compress::raw::dsv4_boundary_meta_paged(
            "attn::dsv4_boundary_meta_paged_dev",
            Launch::flat(n.unsigned_abs(), DSV4_META_BLOCK),
            positions,
            qo_indptr,
            out_pos,
            out_req,
            out_rope,
            n,
            num_requests,
            ratio,
            row_valid,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::attention_compressed_paged_bf16` — attention against the COMPRESSED
/// KV pages, one block per (token, head).
///
/// ```text
/// :317   if (total_tokens <= 0 || num_q_heads <= 0) return;
/// :318   dim3 grid(static_cast<unsigned>(total_tokens),
/// :319             static_cast<unsigned>(num_q_heads));
/// :320   const std::size_t smem =
/// :321       (static_cast<std::size_t>(head_dim) + ATTN_BLOCK) * sizeof(float);
/// :322   device::compressed_attn_paged<<<grid, ATTN_BLOCK, smem, stream>>>(
/// ```
///
/// The scores tile plus the accumulator row, which is why the shared size is
/// `head_dim + ATTN_BLOCK` and not either alone.
///
/// **`qo_indptr` IS NOT A PARAMETER HERE, and its absence is the finding.**
/// The C++ spelled it `const device::u32* /*qo_indptr*/` at `:307` —
/// commented out in its own parameter list — so the ahead-of-time row
/// carried a cell the kernel has no parameter for. The archive's Rust port
/// kept it in the signature as `_qo_indptr` *"so callers do not have to
/// change"*. There are no such callers now, so it is dropped: a parameter
/// that exists only to be ignored is a row's shape surviving into a `fn`.
///
/// # Safety
///
/// [`dsv4_boundary_meta_decode`]'s.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_compressed_paged_bf16(
    q: *const bf16,
    comp_kv_pages: *const bf16,
    o: *mut bf16,
    lse_out: *mut f32,
    positions: *const i32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    req_of_token: *const i32,
    total_tokens: i32,
    num_q_heads: i32,
    head_dim: i32,
    ratio: i32,
    page_size: i32,
    sm_scale: f32,
    stream: *mut c_void,
) -> Fired {
    if total_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "query tokens" });
    }
    if num_q_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "q heads" });
    }
    // `:320` — `(head_dim + ATTN_BLOCK) * sizeof(float)`.
    let smem = head_dim
        .max(0)
        .unsigned_abs()
        .saturating_add(DSV4_ATTN_BLOCK)
        .saturating_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4));
    // SAFETY: the caller's obligation, above.
    unsafe {
        dsv4_compress::raw::compressed_attn_paged(
            "attn::compressed_attn_paged_dev",
            Launch {
                grid: [total_tokens.unsigned_abs(), num_q_heads.unsigned_abs(), 1],
                block: [DSV4_ATTN_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            }
            .smem(smem),
            q,
            comp_kv_pages,
            o,
            lse_out,
            positions,
            kv_page_indices,
            kv_page_indptr,
            req_of_token,
            num_q_heads,
            head_dim,
            ratio,
            page_size,
            sm_scale,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::dsa_index_knorm_rope_bf16` — LayerNorm then interleaved RoPE on the
/// indexer's KEY vectors, in place.
///
/// ```text
/// :20   device::index_knorm_rope<bf16><<<tokens, device::kBlock, 0, stream>>>(
/// :21       idx_k, k_norm_weight, k_norm_bias, positions, head_dim, rope_dim,
/// :22       theta, eps);
/// ```
///
/// One block per token, and NOT the `Rms` rule: the deleted device row's
/// comment records that `Rms` would request thirty-two bytes of dynamic
/// shared memory no launcher passes and no kernel reads — harmless in effect
/// and wrong as a contract. `tokens` is the grid and does not reach the
/// kernel; `head_dim` does, because the kernel strides over it.
///
/// # Safety
///
/// Every pointer is a device address the caller keeps live across the
/// launch, and `stream` is the caller's stream.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsa_index_knorm_rope_bf16(
    idx_k: *mut bf16,
    k_norm_weight: *const bf16,
    k_norm_bias: *const bf16,
    positions: *const i32,
    tokens: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        dsa_indexer::raw::index_knorm_rope(
            "attn::dsa_index_knorm_rope_dev",
            Launch::per_row(tokens.unsigned_abs(), dsa_indexer::K_BLOCK),
            idx_k,
            k_norm_weight,
            k_norm_bias,
            positions,
            head_dim,
            rope_dim,
            theta,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::dsa_index_q_rope_bf16` — interleaved RoPE on the indexer's QUERY
/// vectors, in place.
///
/// ```text
/// :36   device::index_q_rope<bf16><<<tokens, block, 0, stream>>>(
/// :37       idx_q, positions, n_heads, head_dim, rope_dim, theta);
/// ```
///
/// No norm, because the query side carries no weight or bias. The block is
/// [`dsa_indexer::q_rope_block`] and `n_heads` is passed AND sizes it, which
/// is `Control::Supplies` exactly.
///
/// # Safety
///
/// [`dsa_index_knorm_rope_bf16`]'s.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsa_index_q_rope_bf16(
    idx_q: *mut bf16,
    positions: *const i32,
    tokens: i32,
    n_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        dsa_indexer::raw::index_q_rope(
            "attn::dsa_index_q_rope_dev",
            Launch::per_row(tokens.unsigned_abs(), dsa_indexer::q_rope_block(n_heads)),
            idx_q,
            positions,
            n_heads,
            head_dim,
            rope_dim,
            theta,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::dsa_index_topk_mask` — score every causal (query, key) pair and
/// write a byte mask keeping the top `topk`.
///
/// ```text
/// :48   const std::size_t smem = static_cast<std::size_t>(tokens) * sizeof(float);
/// :49   device::index_topk_mask<bf16><<<tokens, device::kBlock, smem, stream>>>(
/// :50       idx_q, idx_k, idx_w, mask, tokens, n_heads, head_dim, topk);
/// ```
///
/// **`tokens` is the grid AND an operand, and that is not duplication.** The
/// grid gives each block its query; the kernel needs the number again as the
/// pitch of `mask` (`mrow = mask + i * N`) and as the bound of its causal
/// zero-fill. `families/attn.rs` states the rule: *an extent a rule recovers
/// is not an operand — an extent a kernel ADDRESSES with is.*
///
/// The shared allocation is one float per key, and under-sizing it does not
/// fault; `dsa_indexer.cuh`'s header carries that finding in full.
///
/// # Safety
///
/// [`dsa_index_knorm_rope_bf16`]'s.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dsa_index_topk_mask_bf16(
    idx_q: *const bf16,
    idx_k: *const bf16,
    idx_w: *const bf16,
    mask: *mut u8,
    tokens: i32,
    n_heads: i32,
    head_dim: i32,
    topk: i32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    // `:48` — `std::size_t` in the C++, narrowed to the `u32` a `Launch`
    // carries. One float per key, one key per token.
    let smem = tokens
        .unsigned_abs()
        .saturating_mul(u32::try_from(core::mem::size_of::<f32>()).unwrap_or(4));
    // SAFETY: the caller's obligation, above.
    unsafe {
        dsa_indexer::raw::index_topk_mask(
            "attn::dsa_index_topk_mask_dev",
            Launch::per_row(tokens.unsigned_abs(), dsa_indexer::K_BLOCK).smem(smem),
            idx_q,
            idx_k,
            idx_w,
            mask,
            tokens,
            n_heads,
            head_dim,
            topk,
            stream,
        );
    }
    Fired::Launched
}

/// `flashinfer::MLAParams` — measured, mirrored, and pinned with
/// [`by_value!`](crate::by_value)'s untagged arm.
///
/// # The measurement
///
/// `nvrtc-probes/attn_mla_params.py`, NVRTC 13.0, `compute_89`,
/// `-std=c++17 -default-device -diag-suppress=1105`, `-I csrc/{shim,vendor,src}`,
/// over `flashinfer/attention/mla_params.cuh:26` instantiated as
/// `MLAParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>` — the
/// instantiation `attention_mla.cu:264` uses. Every member is a pointer or a
/// scalar, so the four type arguments change no offset; they only have to
/// compile.
///
/// ```text
/// sizeof = 288   alignof = 8
///   0..176  the twenty-three pointers, eight apart
///     184   block_size          (uint_fastdiv)
///     208   num_heads           (uint_fastdiv)
///     232   q_nope_stride_n ... 268  o_stride_h   (ten uint32_t)
///     272   sm_scale     276  ckv_scale     280  kpe_scale
///     284   return_lse_base_on_e (bool, then three bytes of tail padding)
/// ```
///
/// # WHY TRANSCRIBING IT WOULD HAVE BEEN WRONG BY FORTY BYTES
///
/// This is the strongest case in the tree for §5.1's *measure, do not
/// transcribe*, and it fails on two counts a reader cannot see at the
/// declaration site.
///
/// **1. `uint_fastdiv` is twenty-four bytes, not four.** Measured directly:
/// `sizeof(::flashinfer::uint_fastdiv) = 24, alignof = 8`. `mla_params.cuh`
/// spells `uint_fastdiv block_size; uint_fastdiv num_heads;` and a reader
/// sees two divisors, which read as `uint32_t`s — that is what the NAME says.
/// `params_layout.py` caught this exact trap once before and it recurs here
/// with two instances instead of one. A transcription would put
/// `q_nope_stride_n` at 192; it is at **232**. Every field from `num_heads`
/// on is displaced, and `sizeof` would have been written as 248.
///
/// **2. `PROFILER_PARAMS_DECL` is a conditional field IN THE MIDDLE.**
/// `mla_params.cuh:56` expands to `uint64_t* profiler_buffer;` at
/// `profiler.cuh:87` and to NOTHING at `:139`, chosen by
/// `FLASHINFER_ENABLE_PROFILER` — defined in a different file, invisible
/// where the struct is declared. Both sides were measured:
///
/// ```text
/// JIT default (what a unit! text is compiled with)   sizeof = 288
/// -DFLASHINFER_ENABLE_PROFILER=1                     sizeof = 296   (+8)
/// ```
///
/// Every field from `block_size` on moves by eight. **This is `x/xqa.rs`'s
/// `ENABLE_4BIT_KV_CACHE` again** — *"the struct's shape depends on
/// [a macro], which inserts [a pointer] in the middle, and it is not visible
/// at the declaration site"* — in a second family, which is the second
/// instance and therefore the pattern rather than the anecdote. The mirror
/// below pins the JIT-default shape, and if a build ever defines the profiler
/// macro the `size_of` assertion fires and names the struct.
///
/// # THE FLOOR GAP THIS ROOT FOUND — CLOSED, as `by_value!`'s untagged arm
///
/// Recorded because the gap is the finding and the patch is its answer.
/// `by_value!`'s original grammar required a tag:
///
/// ```text
/// by_value! { $rust as $cpp, tag = $tag:ident, … }
///     const _: () = assert!(::kernels::Ty::$tag.needs_mirror(), …);
/// ```
///
/// and `Ty::needs_mirror()` (`kernels/src/lib.rs:1174`) is a CLOSED list of
/// six: `AttentionWorkspaceView`, `KvCacheLayerView`, `MlaCacheLayerView`,
/// `HopperPrefillPlan`, `YarnOriginalParams`, `StructuredMasks`. None is
/// `MLAParams`, and none may be borrowed for it — `runtime/args.rs:487`
/// already records why: *"the check would pass on a `MLAParams` bound where a
/// `HopperParams` is declared and catch nothing."*
///
/// **So an open set of `Abi` impls was gated behind a closed set of `Ty`
/// tags**, in a crate three portable backends share. `x/xqa.rs`'s
/// `KvCacheList` was the tree's only `by_value!` because
/// `Ty::KvCacheLayerView` happened to already exist and happened to mean
/// roughly the right thing; eleven families produced no second one.
///
/// The obvious patch — a seventh variant — was the one `x/abi.rs:415-417`
/// argues against in its own words: *"a `Ty` variant per aggregate would have
/// been the forty-variant `LaunchRule` mistake one level down."* So the tag
/// became optional instead. `Abi::TY` for an untagged aggregate is
/// `Ty::MlaPlanCache`, which is on **neither** `is_pointer`'s list nor
/// `bind::device::scalar`'s in either crate — a walker that consulted it gets
/// a named `ArgError::Unsupported`, never a silent accept of eight bytes
/// where two hundred and eighty-eight were meant. Every assertion the tagged
/// arm makes is kept; only the permission is dropped. `x/abi.rs`'s new arm
/// states it best: **the field was never carrying a fact, it was carrying a
/// permission.**
///
/// One thing fell out of the sibling patch and is worth carrying here:
/// `Ty::StructuredMasks` is the only entry on `is_pointer`'s list whose
/// `needs_mirror()` is ALSO true, and **the pair proves the two properties
/// are independent.** `needs_mirror` asks *is there a struct pair to keep in
/// sync*; `is_pointer` asks *does a launch marshal eight bytes of address*.
/// The old tag assertion conflated them.
///
/// # What there is NOT to compare against, and what that makes this
///
/// §5.1 hoped a `by_value!` pin might disagree with what the host packs
/// today. It cannot: swept `driver-cuda/src` and `kernels-cuda-new/src` for
/// `MLAParams`/`MlaParams`/`mla_params` and **every hit is prose.** The
/// struct is packed only in `attention_mla.cu`, which is C++ and
/// `xqa-finish`'s.
///
/// **So this is the reference, not a copy of one.** It is written to be the
/// thing the archive is checked against: every offset is a measurement with a
/// named probe, and the two fields a reader gets wrong carry the number they
/// would have got wrong in the assertion message.
pub mod mla_params {
    use super::bf16;
    use crate::by_value;
    use crate::x::{ByValue, Layout};

    /// `flashinfer::uint_fastdiv` — twenty-four bytes, and that is the whole
    /// point of it being a type here.
    ///
    /// Opaque on purpose. `fastdiv.cuh:26-48` makes `impl_` and `d_` PRIVATE,
    /// so the probe cannot reach their offsets — `nvrtcCompileProgram`
    /// answers *"member `flashinfer::uint_fastdiv::d_` is inaccessible"* —
    /// and a mirror that named them would be transcribing exactly what this
    /// module refuses to transcribe. What IS measurable is the size and the
    /// alignment, and those are what a by-value crossing needs.
    ///
    /// A host that eventually fills one must compute the magic-number pair
    /// the same way `fastdiv.cuh:36`'s `__host__` constructor does; that
    /// constructor is `#ifndef __CUDACC_RTC__` precisely because NVRTC
    /// refuses an explicitly `__host__` function, so **the device never
    /// constructs one and the Rust caller must**. That is a second host
    /// program's problem and it is recorded here because nothing else will
    /// say it.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct UintFastdiv {
        /// `impl_` and `d_` together, unreachable individually.
        pub opaque: [u64; 3],
    }

    impl UintFastdiv {
        /// Build the pair the device halves read, by the shim's algorithm.
        ///
        /// # This is a mirror of `csrc/shim/cuda/cmath`, and it says so
        ///
        /// The vendored `flashinfer/fastdiv.cuh` is **not** the classical
        /// magic-number `uint_fastdiv` its name suggests. It is a wrapper
        /// around `cuda::fast_mod_div<uint32_t>` with members `impl_` then
        /// `d_`, and `cuda::fast_mod_div` is **implemented by this repository**
        /// — the toolkit's CCCL 3.0.1 has no such class, measured, *"namespace
        /// cuda has no member fast_mod_div"*. So the algorithm below is read
        /// off a file this tree owns:
        ///
        /// ```text
        /// csrc/shim/cuda/cmath:196-206
        ///     all_ones = ~0ull;  q = all_ones / d;  r = all_ones % d
        ///     __magic_ = q + (r + 1 == d ? 1 : 0) + 1     // = floor(2^64 / d)
        ///     __divisor_ = d
        /// layout: { u32 __divisor_ @ 0, u64 __magic_ @ 8 }  size 16, align 8
        /// ```
        ///
        /// and `uint_fastdiv` wraps it as `{ fast_mod_div impl_ @ 0, u32 d_ @
        /// 16 }` — **24 bytes, align 8**, which is what the assertions at the
        /// foot of this module pin.
        ///
        /// # The rule the shim states, obeyed here
        ///
        /// `cmath:265-273` says it verbatim: *"a Rust mirror of anything
        /// containing a `uint_fastdiv` must be pinned against THIS layout, and
        /// must say so."* It also names the hazard it exists to prevent — *"a
        /// params block filled by the ahead-of-time path and fired by the JIT
        /// one"* — because **CCCL's `fast_mod_div` is `{divisor@0,
        /// multiplier@4, add@8, shift@12}`, the same `sizeof` and a different
        /// interior.** `paged_kv_t::num_heads` measured at +24 under the shim
        /// and +20 under CCCL, and an earlier check of `sizeof` alone reported
        /// agreement.
        ///
        /// **The consequence for this family, stated once and here:**
        /// `MlaParams` is pinned at 288 bytes against the SHIM, which is what
        /// NVRTC compiles. Under nvcc and CCCL each `uint_fastdiv` is 20 bytes
        /// and the struct is 40 short. `attention_mla.cu` therefore packs a
        /// differently-shaped `MLAParams` than this module does, and the two
        /// were never going to agree. That is fine while each side is
        /// internally consistent — it is exactly the shim's stated rule — but
        /// it means **nothing built here may fill a struct the ahead-of-time
        /// path launches with, in either direction.**
        ///
        /// # `d == 0`
        ///
        /// The shim's device path special-cases `d == 1`; it has no answer for
        /// zero and neither does this. A zero divisor yields a zero magic,
        /// which is a wrong answer rather than a fault, so **every host
        /// program in this module refuses a zero extent before it packs** and
        /// this is a `const fn` that cannot refuse for them.
        #[must_use]
        pub const fn new(d: u32) -> Self {
            let d64 = d as u64;
            let magic = if d == 0 {
                0
            } else {
                let q = u64::MAX / d64;
                let r = u64::MAX % d64;
                q + if r + 1 == d64 { 1 } else { 0 } + 1
            };
            // Word 0 is `impl_.__divisor_` in its low half and four bytes of
            // padding in its high half; word 1 is `impl_.__magic_`; word 2 is
            // `d_` and the struct's own tail padding. Written as three `u64`
            // because the members are unreachable individually — the pin
            // above is on the whole 24 bytes. Little-endian is assumed and is
            // not a portability gap: the struct exists to be memcpy'd to a
            // CUDA device, and there is no big-endian CUDA host.
            Self { opaque: [d64, magic, d64] }
        }
    }

    /// `flashinfer::MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>`, at
    /// `<bf16, bf16, bf16, i32>`.
    ///
    /// Field order and spelling are `mla_params.cuh:31-77`'s; every offset is
    /// the module doc's measurement, asserted below.
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct MlaParams {
        /// @0 — the non-positional half of Q.
        pub q_nope: *mut bf16,
        /// @8 — the rotary half of Q.
        pub q_pe: *mut bf16,
        /// @16 — the compressed KV cache.
        pub ckv: *mut bf16,
        /// @24 — the K positional cache.
        pub kpe: *mut bf16,
        /// @32 — split-K partial output.
        pub partial_o: *mut bf16,
        /// @40 — split-K partial log-sum-exp, always f32.
        pub partial_lse: *mut f32,
        /// @48 — the merged output.
        pub final_o: *mut bf16,
        /// @56 — the merged log-sum-exp, always f32.
        pub final_lse: *mut f32,
        /// @64
        pub q_indptr: *mut i32,
        /// @72
        pub kv_indptr: *mut i32,
        /// @80
        pub partial_indptr: *mut i32,
        /// @88
        pub merge_packed_offset_start: *mut i32,
        /// @96
        pub merge_packed_offset_end: *mut i32,
        /// @104
        pub merge_partial_packed_offset_start: *mut i32,
        /// @112
        pub merge_partial_packed_offset_end: *mut i32,
        /// @120
        pub merge_partial_stride: *mut i32,
        /// @128
        pub kv_indices: *mut i32,
        /// @136
        pub q_len: *mut i32,
        /// @144
        pub kv_len: *mut i32,
        /// @152
        pub q_start: *mut i32,
        /// @160
        pub kv_start: *mut i32,
        /// @168
        pub kv_end: *mut i32,
        /// @176 — the persistent-kernel work queue.
        pub work_indptr: *mut i32,
        /// @184 — **twenty-four bytes**, not four. See the module doc.
        pub block_size: UintFastdiv,
        /// @208 — twenty-four bytes.
        pub num_heads: UintFastdiv,
        /// @232
        pub q_nope_stride_n: u32,
        /// @236
        pub q_nope_stride_h: u32,
        /// @240
        pub q_pe_stride_n: u32,
        /// @244
        pub q_pe_stride_h: u32,
        /// @248
        pub ckv_stride_page: u32,
        /// @252
        pub ckv_stride_n: u32,
        /// @256
        pub kpe_stride_page: u32,
        /// @260
        pub kpe_stride_n: u32,
        /// @264
        pub o_stride_n: u32,
        /// @268
        pub o_stride_h: u32,
        /// @272
        pub sm_scale: f32,
        /// @276 — per-tensor symmetric dequant scale for an fp8 `ckv`.
        /// Defaults to `1.0` in C++ and has no effect on the bf16/f16 path;
        /// a Rust packer must write the 1.0 itself, because a zeroed struct
        /// would scale every value to zero and that is a silent wrong answer
        /// rather than a fault.
        pub ckv_scale: f32,
        /// @280 — the same for `kpe`, and the same warning.
        pub kpe_scale: f32,
        /// @284 — one byte, then three of tail padding to reach 288.
        pub return_lse_base_on_e: bool,
    }

    // THE PIN. Every offset is `nvrtc-probes/attn_mla_params.py`'s, and the
    // assertions are what make this a mirror rather than a transcription: a
    // field inserted, widened or reordered on either side fails the build
    // here and names the field, instead of launching over a struct whose tail
    // is forty bytes out of place.
    //
    // WHICH VARIANT IS PINNED: the JIT default, `sizeof = 288`, WITHOUT
    // `profiler_buffer`. That is the one NVRTC compiles from a `unit!`'s text
    // — no `-DFLASHINFER_ENABLE_PROFILER` is passed anywhere in this tree —
    // and it is stated here because the declaration site does not show it. If
    // a build ever defines the macro, `sizeof` becomes 296 and the first
    // assertion below fires with that number in its message.
    //
    // The named fields are the ones a reader would get WRONG, plus the ends:
    // the first pointer, the last pointer, both `uint_fastdiv`s, the first
    // and last `uint32_t`, and all four tail scalars. The twenty-one interior
    // pointers are eight apart with nothing between them and are checked by
    // `work_indptr @ 176` closing the run.
    by_value! {
        MlaParams as "::flashinfer::MLAParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>",
        untagged,
        probe = "nvrtc-probes/attn_mla_params.py",
        size = 288, align = 8,
        {
            q_nope               @ 0   as "q_nope",
            work_indptr          @ 176 as "work_indptr",
            block_size           @ 184 as "block_size",
            num_heads            @ 208 as "num_heads",
            q_nope_stride_n      @ 232 as "q_nope_stride_n",
            o_stride_h           @ 268 as "o_stride_h",
            sm_scale             @ 272 as "sm_scale",
            ckv_scale            @ 276 as "ckv_scale",
            kpe_scale            @ 280 as "kpe_scale",
            return_lse_base_on_e @ 284 as "return_lse_base_on_e",
        }
    }

    /// The layouts this module pins.
    ///
    /// `typecheck_tu` has no callers yet — `xqa::LAYOUTS` is unconsumed for
    /// the same reason — but when it gets one, this is the entry that turns
    /// the probe's numbers into a compiled `static_assert` against the real
    /// `mla_params.cuh`, which is the only check that survives a vendor bump.
    pub static LAYOUTS: &[Layout] = &[<MlaParams as ByValue>::LAYOUT];

    // `uint_fastdiv` is asserted separately because `by_value!` above cannot
    // reach it: the macro asserts the fields it is GIVEN, and the two
    // `uint_fastdiv` members are named there by offset, not by size. If this
    // type were 4 bytes as its name suggests, `block_size @ 184` would still
    // hold and `num_heads @ 208` would fail with no explanation of why. These
    // two assertions are that explanation, and they fire first.
    const _: () = assert!(
        ::core::mem::size_of::<UintFastdiv>() == 24,
        "UintFastdiv: sizeof disagrees with the measured ::flashinfer::uint_fastdiv \
         (24, NOT 4 — see nvrtc-probes/attn_mla_params.py)",
    );
    const _: () = assert!(
        ::core::mem::align_of::<UintFastdiv>() == 8,
        "UintFastdiv: alignof disagrees with the measured ::flashinfer::uint_fastdiv",
    );
}

/// `attn/attention_mla_naive.cuh` — the Blackwell MLA fallback pair,
/// unit-only, and the root that stands between the tree and nvcc-zero.
///
/// # Why this root and not `kv_paged`
///
/// §66: `kernels-cuda/native` is the only switch over the whole nvcc and
/// `.cpp` surface in the workspace, it is turned on by exactly one thing
/// (`driver-cuda/bridge`), and that is deletable when `ROW_TABLES` empties.
/// The last two `<<<>>>` in the tree are `mla_naive_paged_kernel` and
/// `mla_mma_paged_kernel` in `attention_mla.cu` — the device text below.
/// **Crossing this root's row is retiring that file**; they were never two
/// tasks.
///
/// The row is not crossed yet and this is the fourth unit-only crossing.
/// `driver-cuda/src/fire/mla_naive.rs` already holds both host programs in
/// Rust and fires both symbols by name through `hand::fire`, so by the rule
/// the last three roots established, the device text moves and nothing else
/// does. What remains for the ROW is stated in that module's own header and
/// is not this crossing's business: `attn::dispatch_attention_mla_bf16` has
/// TWO arms — this pair, and `flashinfer::mla::BatchMLAPagedAttention<MASK,
/// 512, 64>` for everything below sm_100 — and *"a row loses its shim entry
/// whole or not at all, so both arms must be Rust before either can be."*
/// The FA2 arm is the one that passes [`super::mla_params::MlaParams`] by
/// value, which is now unblocked.
///
/// # Why the pair exists — `attention_mla.cu:150-157`, the only place it is
/// # argued, and it must not be lost with the file
///
/// > FlashInfer's FA2 `BatchMLAPagedAttention` (a cooperative kernel)
/// > produces zero output on sm_100; the ecosystem (sglang/vllm) routes
/// > Blackwell MLA to trtllm/cutlass/ragged kernels instead. This is a
/// > correctness-first, arch-agnostic latent-space MLA: one block per (token,
/// > head), flash-style online softmax over the paged ckv/kpe cache. Output
/// > is in the kv_lora latent space (same as the FA2 path), so the rest of
/// > the MLA forward (latent_to_v, o_proj) is unchanged.
///
/// The selector is a device query — `cudaDevAttrComputeCapabilityMajor >= 10`
/// at `attention_mla.cu:56-62` — and it is not these rows' business: it
/// chooses between this pair and FlashInfer's MLA, a different symbol in a
/// different unit.
///
/// **The two kernels are ALTERNATIVES, not a sequence.** The C++ launcher
/// tries the tensor-core kernel first (`attention_mla_naive.cuh:218`) and
/// falls through to the scalar one; `fire::mla_naive` plans one or the other
/// and fires exactly once. Nothing composes them, so there is no `Composed`
/// here and no intermediate buffer.
///
/// # THE GEOMETRY, and why neither kernel can have a `LaunchRule`
///
/// ```text
/// attention_mla_naive.cuh:265   dim3 grid(total_tokens, num_heads / G);      block 256
/// attention_mla_naive.cuh:725   dim3 grid(num_heads / kBM, total_tokens);    block 256
/// ```
///
/// Same block, and the grids are **TRANSPOSES** of each other — tokens on `x`
/// for the scalar kernel, tokens on `y` for the tensor-core one. A rule
/// stating one would be actively wrong for the other while looking right:
/// `grid.y` is capped at 65 535 and `grid.x` is not, **so the transpose
/// decides which of tokens and head blocks may exceed 65 535.** That is a
/// correctness fact wearing the clothes of a layout preference.
///
/// The scalar kernel's `G` is `execution::Control::Supplies`' own example —
/// *"passed to the kernel AND divides the head axis of the grid"* — and it is
/// not merely unstated but **UNSTATEABLE by a formula**: `:241-249` SEARCHES
/// for it, halving from 8 until the grid reaches `kMlaWaveTarget = 296`
/// blocks. A rule computes; this looks. In fn-world both objections dissolve,
/// because a `Launch` is a `fn`'s literal and `fire::mla_naive::plan` is
/// already that `fn`.
///
/// # The shared-memory opt-in, and the measurement that closed it
///
/// The old refusal — *"`attention_mla_naive.cu` keeps its
/// `cudaFuncSetAttribute` opt-in to 200 KB of shared memory behind a
/// `std::call_once`, host state no `LaunchRule` can carry"* — was wrong on
/// both halves and both corrections must survive:
///
/// * **It is not host state a rule has to carry.** `runtime::module`'s
///   `raise_dynamic_smem_cap` performs the opt-in inside `KernelModule::fire`,
///   once per `(CUdevice, CUfunction)` above a 48 KiB high-water mark, driven
///   by `Launch::smem` and nothing else. `x::launch::Launch`'s `smem_opt_in`
///   is the author's side of the same fact.
/// * **The 200 KiB was never needed.** `attention_mla_naive.cuh:251`'s
///   allocation is `(8 * CKV + 16) * 4` and `:228` refuses `CKV > 512`, so
///   the largest request the SCALAR kernel can make is **16 448 bytes** — a
///   third of the 48 KiB default. The arithmetic is preserved in
///   `fire::mla_naive::NAIVE_OPT_IN_BYTES_UNREACHED`. The TENSOR-CORE
///   kernel's **100 032** is above the default and IS raised.
///
/// The real blocker was neither: the file was MIXED — two `__global__`s and
/// four host functions in one header, opening `<mutex>`, `<stdexcept>`,
/// `<string>` and `<cuda_runtime.h>`, so it could not be a unit root at all.
/// The host half now lives in `attention_mla.cu` and in Rust in
/// `fire/mla_naive.rs`, and what is left compiles.
///
/// # PROBED — NVRTC 13.0, `sm_89`, carried headers only
///
/// Under this crate's numerics contract (`--fmad=false --prec-div=true
/// --prec-sqrt=true`) and `-I csrc/{src,shim,vendor}` with **no toolkit
/// include path**:
///
/// ```text
///   rc = 0, 0 errors
///   117 621 bytes of PTX, 2 .entry
///     _ZN15pie_cuda_driver7kernels4attn9mla_naive22mla_naive_paged_kernelE...
///     _ZN15pie_cuda_driver7kernels4attn9mla_naive10mma_detail20mla_mma_paged_kernelE...
/// ```
///
/// It needed three new shim headers and they are **measured, not assumed**:
/// the same text compiled with `/usr/local/cuda/include` answering
/// `cuda_pipeline.h`, `math_constants.h` and `cstring` produced
/// **byte-identical PTX, register allocation included.** See
/// `csrc/shim/cuda_pipeline.h`, which carries the comparison and the one PTX
/// operand it turned on.
///
/// **A fourth finding from the same probe, and it is the shape this sweep
/// keeps meeting:** the file called `std::memcpy` and never included
/// `<cstring>`. Under nvcc `<cuda_runtime.h>` supplied it transitively; under
/// NVRTC it is an error no include path can fix, because the include was
/// never written. **The set nvcc accepted was not the set the file
/// declared.**
///
/// # No options, and the `#ifndef` defaults are why
///
/// `PIE_MLA_MMA_BK`, `_WARPS`, `_STAGES` and `_MINBLK` are all `#ifndef`
/// guarded with their defaults at `:302-322`, so the unit needs no `-D` to
/// compile at the shape everything currently runs. Putting them in
/// `Unit::options` would be the hook `unit.rs`'s own doc warns against: they
/// are tuning constants with one live value, and `Unit::cache_key` spanning
/// them would make a cubin cache key out of a number nobody varies. A second
/// tile is a second unit with a second root, the way `XQA_LATTICE` spells its
/// six.
pub mod mla_naive {
    use super::bf16;

    unit! {
        /// Two `__global__`s and the `mma_detail` helpers the second one
        /// needs. No host code — that left for `attention_mla.cu` and
        /// `fire/mla_naive.rs` before this crossing.
        unit MLA_NAIVE = "attn/attention_mla_naive",
            text = include_str!("../../csrc/src/attn/attention_mla_naive.cuh"),
            file = "attn/attention_mla_naive.cuh";

        /// `attention_mla_naive.cuh:92` — the scalar flash-softmax kernel,
        /// nineteen parameters ending in `G`.
        ///
        /// **`DeviceKernel::PLAIN` and no `_bf16` suffix**, both for the same
        /// reason: there is no template parameter list, so there is nothing
        /// for `elem` to pick and nothing a format suffix could claim a
        /// choice about. Every buffer is `__nv_bfloat16` in the kernel's own
        /// declaration. A suffix here would assert a specialisation that does
        /// not exist.
        ///
        /// The path is two levels deep and that is the header's own nesting:
        /// `pie_cuda_driver::kernels::attn::mla_naive`.
        ///
        /// **`index_mask` is nullable and null is not merely "no mask".**
        /// `attn/attention_mla.hpp:36-38`, which went with the file:
        ///
        /// > DSA top-k mask for the naive path: `[num_query_tokens,
        /// > mask_stride]` uint8 (1=attend). Applied to in-batch keys
        /// > (`j < mask_stride`). Null = dense. **Only valid for
        /// > single-request pure prefill (key `j` == batch token `j`).**
        ///
        /// That last sentence is a correctness precondition no type states
        /// and no refusal can check — the kernel indexes `mask + t *
        /// index_mask_stride` and a multi-request batch makes `j` mean two
        /// different things. It travels here and in
        /// `fire::mla_naive::NaivePtrs::index_mask`, and those are now the
        /// only two places it is written.
        ///
        /// Scores are reduced INSIDE A WARP, not across the block: a
        /// block-wide tree reduction per key costs seven `__syncthreads()`
        /// per KV entry, which at decode dwarfs the arithmetic. Each warp
        /// keeps its own running max/sum/accumulator in registers and the
        /// partial softmax states merge once at the end — flash-decoding's
        /// structure, and the reason `G` exists at all.
        fn mla_naive_paged_kernel = "attn::mla_naive::mla_naive_paged_kernel" (
            q_nope: *const bf16,
            q_pe: *const bf16,
            ckv_pages: *const bf16,
            kpe_pages: *const bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            o: *mut bf16,
            index_mask: *const u8,
            index_mask_stride: i32,
            r: i32,
            h: i32,
            ckv: i32,
            kpe: i32,
            page_size: i32,
            sm_scale: f32,
            causal: bool,
            g: i32,
        ) {
            "attn::mla_naive_paged" => crate::device::DeviceKernel::PLAIN,
        }

        /// `attention_mla_naive.cuh:371` — the tensor-core kernel, sixteen
        /// parameters.
        ///
        /// **THE THREE MISSING PARAMETERS ARE THE MOST INFORMATIVE THING
        /// ABOUT THIS DECLARATION.** There is no `ckv`, no `kpe` and no `G`.
        /// The kernel is compiled AGAINST `kCkv = 512` and `kKpe = 64`
        /// (`:281-282`) because the `mma.sync` fragment shapes are written
        /// for them, and its head group is fixed at `kBM = 16` (`:275`). That
        /// is exactly why `mla_mma_supported` (`:698`) COMPARES those three
        /// rather than forwarding them: the predicate is the only place the
        /// shape is checked, and passing them would imply a generality the
        /// `ld_b_v`/`ld_a` offsets do not have.
        ///
        /// So a host program that fires this kernel must check the three
        /// against 512, 64 and 16 BEFORE it launches — there is no operand to
        /// carry them and therefore no chance of the kernel disagreeing.
        /// `fire::mla_naive` does; that check is the refusal, and it belongs
        /// above the fire for §5.1's hoisting reason.
        ///
        /// Path is three levels deep — `attn::mla_naive::mma_detail` — which
        /// is where `ld_a`, `ld_b_v` and `mma_m16n8k16` live too.
        ///
        /// **100 032 bytes of dynamic shared memory**, which is above the
        /// 48 KiB default and IS raised by `raise_dynamic_smem_cap`. Its
        /// `__launch_bounds__(kThreads, PIE_MLA_MMA_MINBLK)` is in the device
        /// text and needs no host statement.
        fn mla_mma_paged_kernel = "attn::mla_naive::mma_detail::mla_mma_paged_kernel" (
            q_nope: *const bf16,
            q_pe: *const bf16,
            ckv_pages: *const bf16,
            kpe_pages: *const bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            o: *mut bf16,
            index_mask: *const u8,
            index_mask_stride: i32,
            r: i32,
            h: i32,
            page_size: i32,
            sm_scale: f32,
            causal: bool,
        ) {
            "attn::mla_mma_paged" => crate::device::DeviceKernel::PLAIN,
        }
    }
}

/// `attn/kimi_mla.cuh` — kimi_k3's two latent-attention preparation kernels.
///
/// **The only FULL crossing in this pass**, and the reason it was taken
/// before the larger roots: both of its table rows are sourced on every
/// operand, both are stated by `model-compiler/src/dsl.rs`
/// (`kimi_split_kv_a_norm` at `:5452`, `kimi_split_q_b` at `:5483`), and
/// `crates/model/src/kimi_k3/forward/mod.rs:150-156` states them in that
/// order. A unit, two contracts and two binds retire two of the family's
/// thirty-four rows, leaving thirty-two — thirty-one once [`mla_fa2`]'s
/// contract took `attn::dispatch_attention_mla_bf16`.
///
/// # The crossing CLOSES a measured wrong-answer defect
///
/// `device.rs:991-1006` records it in the past tense and the record is worth
/// repeating here, because the port is what ends it rather than what found
/// it: `kimi_split_q_b`'s row describes the LAUNCHER, whose `tokens`,
/// `heads`, `nope` and `rope` are `Rows` and `Param(0..2)` and which formed
/// the device kernel's `total` from them. **The JIT has no launcher**, so it
/// sizes the grid from `LaunchRule::Elementwise` instead — `rows *
/// width_of(b, n_in + 0)`, the FIRST OUTPUT's width. This kernel splits `q_b`
/// into `q_nope` and `q_pe`, so the extent it must cover is wider than out 0
/// *by construction*, and the generated launch under-covers by exactly
/// `nope / (nope + rope)`. Measured at 6 rows of 8 heads, nope 128, rope 64:
/// it wrote four of six rows and left **4,082 of 12,544 bytes** of `q_nope`
/// and **2,041 of 6,400** of `q_pe` holding the harness's poison fill.
///
/// The near-miss is the part to keep: the row's third harness shape — one
/// row, one head — agrees in every byte, because 200 elements round up into a
/// 256-thread block that covers all 255 the kernel wanted. **One shape would
/// have certified it.**
///
/// A `bind!` body computes its own extent, so the arm below reads
/// `rows * in_width(0)` — the kernel's own `total`, which is what
/// `kimi_mla.cuh:13` says the archive launcher computed — and there is no
/// rule left to disagree with it. That is the same shape as
/// [`per_head`]'s head-count defect, which `head_dim_pad`'s crossing closed
/// two passes ago, and it is the second time in this family that **a
/// launch-rule input and a kernel's own addressing were two different
/// numbers wearing one name**.
///
/// # No twin to check
///
/// §60.6's symbol split reached neither kernel: `families::attn`'s
/// `KIMI_MLA_SIGS` and `table::attn`'s two rows name the same two strings,
/// `attn::kimi_split_q_b_bf16` and `attn::kimi_split_kv_a_norm_bf16`, and
/// `driver-cuda/src/fire/` has no `kimi` module to bridge anything. The
/// device symbol IS the table symbol, so the `unit!` rows below are spelled
/// exactly as the deleted rows were.
pub mod kimi_mla {
    use super::bf16;

    unit! {
        /// Two `__global__` templates and nothing else, which is what the
        /// header says about itself in its first line.
        ///
        /// The `<<<>>>`s were in `kimi_mla.cu`, which `#include`d this
        /// header rather than defining what it launched — so the
        /// ahead-of-time build and the JIT compiled ONE text and the
        /// crossing costs no reconciliation. The header records why that
        /// split exists: `norm/altup_aux` once shipped **two definitions of
        /// six kernels**, each correct for whichever half of the tests
        /// exercised it, and no test could see the disagreement because no
        /// test ran both.
        unit KIMI_MLA = "attn/kimi_mla",
            text = include_str!("../../csrc/src/attn/kimi_mla.cuh"),
            file = "attn/kimi_mla.cuh";

        /// `kimi_mla.cuh:67` — split a fused `q_b` projection into its nope
        /// and rope halves.
        ///
        /// `q_b` is `[tokens, heads, nope + rope]`; the results are
        /// `[tokens, heads, nope]` and `[tokens, heads, rope]`. One thread
        /// per SOURCE element, which is why `total` is an argument and not a
        /// grid read: `Elementwise` rounds the grid up and `if (i >= total)`
        /// at `:77` is the only guard there has ever been.
        ///
        /// **`total` is the input's element count and nothing else.** The
        /// module header above measures what happens when something computes
        /// it from an output instead.
        ///
        /// The `long long` casts on the destination indices at `:84` and
        /// `:86` are not decoration — `tokens * heads * nope` overflows
        /// `int` on a long prefill at kimi_k3's head count, and the product
        /// is formed before it is used as an index.
        fn split_q_b = "attn::device::split_q_b" <T> (
            q_b: *const T,
            q_nope: *mut T,
            q_pe: *mut T,
            total: i32,
            heads: i32,
            nope: i32,
            rope: i32,
        ) where *const T, *mut T {
            "attn::kimi_split_q_b_bf16" => where [T = bf16] "device::bf16",
        }

        /// `kimi_mla.cuh:101` — split `kv_a` into a normalised latent and
        /// its rope-carrying companion.
        ///
        /// One kernel rather than a split followed by an RMSNorm, because
        /// the latent half is read twice by the norm and would otherwise
        /// make a round trip through global memory in between. The `k_pe`
        /// copy is **unnormalised on purpose**: rope is applied to it later
        /// and normalising a value that is about to be rotated changes the
        /// angle.
        ///
        /// `src_row_stride` is the SOURCE row width, which is
        /// `kv_lora + rope` unless a caller hands a wider buffer — the fused
        /// MLA prepare does, which is why the stride is an operand and not a
        /// sum.
        ///
        /// # `256` IS THIS ROW'S TO STATE
        ///
        /// `split_kv_a_norm` is `template <class T, int BLOCK_DIM = 256>`,
        /// and until the argument LIST was statable this row could only
        /// spell `<device::bf16>` and let the default supply the rest. That
        /// worked and was fragile in a way nothing here would have caught:
        /// the kernel declares `__shared__ float buf[BLOCK_DIM]` at `:107`
        /// and reduces by halving from `BLOCK_DIM / 2` at `:127`, so the
        /// width **sizes an array and fixes a tree**. Had someone moved the
        /// default to 512, `kimi_mla.cu` would have kept working — it spelled
        /// `<device::bf16, BS>` with `constexpr int BS = 256` — while this
        /// row started instantiating a 512-wide reduction under a 256-wide
        /// launch, where the upper half of `buf` is never written and the
        /// first halving step reads it. **That is a plausible number, not a
        /// crash.**
        ///
        /// Both halves are cited, as a non-type argument requires: the
        /// launcher was `<<<tokens, BS>>>` with `BS = 256`, the template's
        /// default is 256, they agree today, and this row no longer depends
        /// on their continuing to. [`super::rms`] is the other end of the
        /// same 256.
        fn split_kv_a_norm = "attn::device::split_kv_a_norm" <T> (
            kv_a: *const T,
            norm_weight: *const T,
            kv_c: *mut T,
            k_pe: *mut T,
            kv_lora: i32,
            rope: i32,
            src_row_stride: i32,
            eps: f32,
        ) where *const T, *mut T {
            "attn::kimi_split_kv_a_norm_bf16" => where [T = bf16] "device::bf16, 256",
        }
    }
}

/// `attn/attention_naive_paged.cuh` — the reference paged attention.
///
/// **The last root `families::attn` held that a statement could reach**, and
/// the crossing that empties `table::attn` of everything but
/// `qkv_decode_fused`. Two `__global__`s, both `template <int BLOCK>` at 128,
/// and a host program that had NO Rust and no C++ — it was the generated JIT
/// arm, which is why this one took a floor question rather than a move.
///
/// # What actually blocked it, and it was never the geometry
///
/// [`LaunchRule::PagedScores`](crate::LaunchRule) computed
/// `dim3(num_requests, total_tokens, num_q_heads)` at 128 with
/// `(head_dim + 128) * sizeof(float)` of dynamic shared memory, and that was
/// right. What blocked a `bind!` was two operands: both kernels take
/// `device::KvScheme scheme` and `device::KvDType storage_dtype` **by value**,
/// each an `enum class ... : ::std::uint8_t`, and `x::Abi` had no impl that
/// marshals a scalar byte. [`kv_scheme`] and [`kv_dtype`] are that impl, added
/// where `x/abi.rs:226` said to add it.
///
/// # `128` IS A SHARED-MEMORY CONTRACT
///
/// [`PAGED_BLOCK`] states it once for both kernels. The launcher asks for
/// `(head_dim + BLOCK) * sizeof(float)` and the kernel cuts the TAIL of that
/// allocation into exactly `BLOCK` reduction slots
/// (`attention_naive_paged.cuh:402-404`), so a launch at another block width
/// reads slots nothing wrote. It is also the `acc[]` divisor:
/// `acc[(kMaxHeadDim + BLOCK - 1) / BLOCK]` is 8 at 128 and would be 4 at 256,
/// which is a per-thread array too short for the columns the loop visits.
/// One number, three dependents, and a plausible wrong answer at any other
/// value.
///
/// # THE PREDICATE THE ROW WORLD LOST, restored here
///
/// `attention_naive_paged.cuh:220` says the deleted `.cu` read `kMaxHeadDim`
/// as `device::kMaxHeadDim` in **`check_head_dim_supported`** — *"so the array
/// and the predicate that keeps launches inside it are ONE constant, not
/// two."* The `.cu` went; the predicate went with it; the JIT arm never had
/// one, because a `LaunchRule` opens a grid and cannot refuse. So between the
/// `.cu`'s deletion and this `fn`, a `head_dim` above 1024 launched a kernel
/// that indexes `acc[8]` past its end — not a crash, a wrong answer.
///
/// [`attention_naive_paged`] refuses it as [`Refusal::Wide`], with the ceiling
/// carried, which is what that variant's `max` field is for.
///
/// # One host program, and the decode row has none
///
/// `naive_paged_decode` is `NoRow::KernelsInternal`
/// (`driver-cuda/tests/launch_abi.rs:491`) — called by kernels code and by no
/// statement — so its row states its contract and its geometry and nothing
/// fires it. That is the `attn_naive` arrangement in
/// [`attention_naive`](crate::x::attn::attention_naive), and it is
/// deliberate: firing the decode kernel where the prefill kernel is correct
/// would be a behaviour change wearing a port. The prefill kernel handles a
/// decode exactly — one token per request makes `qo_off == 0` and the
/// `if (qo_off >= qo_hi - qo_lo) return;` guard at `:373` covers the rest.
///
/// # `num_pages_in_batch` is an operand nothing reads
///
/// `table::attn`'s row stated it and the launcher **cast it to `void`**
/// (`attention_naive_paged.cu:193`). It is not in either `__global__`'s
/// parameter list and it is not in this `fn`'s. A row that carries an operand
/// its launcher discards is a row describing a C++ signature rather than a
/// launch, and there is nowhere in fn-world to put it.
#[cfg(feature = "_cuda")]
pub mod attention_naive_paged {
    use super::{bf16, kv_dtype, kv_scheme};
    use core::ffi::c_void;

    unit! {
        /// Two `__global__`s, both `template <int BLOCK>`, both at 128.
        ///
        /// Neither is a template over an ELEMENT type: `q` and `o` are
        /// `device::bf16*` in the text, and the pages are `const void*`
        /// because [`kv_scheme`] and [`kv_dtype`] decide what they hold at
        /// run time. That is the arrangement that makes ONE instantiation
        /// serve five quantisation schemes, and it is why the two byte
        /// operands exist at all.
        unit ATTENTION_NAIVE_PAGED = "attn/attention_naive_paged",
            text = include_str!("../../csrc/src/attn/attention_naive_paged.cuh"),
            file = "attn/attention_naive_paged.cuh";

        /// `attention_naive_paged.cuh:346` — the prefill, over
        /// `dim3(requests, tokens, q_heads)`.
        ///
        /// `attn::attention_naive_paged_dev`, and the `_dev` is §60.6: the
        /// device row's symbol WAS `attn::attention_naive_paged`, the same
        /// string as `table::attn`'s row and the same string
        /// `dsl::cuda::attention_naive_paged` states. A contract symbol is
        /// never a unit row's symbol, so the crossing renamed the device row
        /// — the second root in this family to need it and the second to have
        /// never had it applied.
        ///
        /// # `custom_mask` and the scale planes are both nullable and they
        /// are not the same absent
        ///
        /// The scale planes are null under
        /// [`KvScheme::Native`](crate::x::cx::KvScheme::Native) — absence
        /// means *"this bank is not quantised"*, which the `scheme` operand
        /// states in the same breath. The mask pair is null because THIS host
        /// program passes it null: the deleted `.cu` handed `nullptr` twice at
        /// `:208-209` where its `_custom` sibling handed a real mask at
        /// `:255-256`. Absence there means *"causal, not custom"*, and
        /// `:393`'s `use_custom_mask = custom_mask != nullptr` is what reads
        /// it.
        ///
        /// Both are real and the difference is written down exactly once,
        /// which is here.
        fn naive_paged_attn = "attn::device::naive_paged_attn" (
            q: *const bf16,
            k_pages: *const c_void,
            v_pages: *const c_void,
            k_scales: *const f32,
            v_scales: *const f32,
            o: *mut bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            custom_mask: *const u8,
            custom_mask_indptr: *const i32,
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            page_size: i32,
            scheme: kv_scheme,
            storage_dtype: kv_dtype,
            block_size: i32,
            window_left: i32,
            sm_scale: f32,
            logits_soft_cap: f32,
            lse_out: *mut f32,
        ) {
            "attn::attention_naive_paged_dev" => "device::i32(128)",
        }

        /// `attention_naive_paged.cuh:518` — the decode, over
        /// `dim3(requests, q_heads)`.
        ///
        /// **No host program, by measurement.** `launch_abi.rs:491` records
        /// `attention_naive_paged_decode` as `NoRow::KernelsInternal`, so no
        /// statement routes it and a `fn` here would be one nobody can call.
        /// The row exists because the TEXT exists and NVRTC must be told what
        /// to instantiate; that is what a `unit!` row is for.
        ///
        /// Three parameters shorter than the prefill: no `qo_indptr` (a
        /// decode has one row per request, so `blockIdx.x` addresses it) and
        /// no mask pair (a decode's `kv_lim` is the whole context —
        /// `:544`'s `const int kv_lim = kv_total;` with no causal subtraction
        /// and no custom branch at all).
        fn naive_paged_decode = "attn::device::naive_paged_decode" (
            q: *const bf16,
            k_pages: *const c_void,
            v_pages: *const c_void,
            k_scales: *const f32,
            v_scales: *const f32,
            o: *mut bf16,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            page_size: i32,
            scheme: kv_scheme,
            storage_dtype: kv_dtype,
            block_size: i32,
            window_left: i32,
            sm_scale: f32,
            logits_soft_cap: f32,
            lse_out: *mut f32,
        ) {
            "attn::naive_paged_decode" => "device::i32(128)",
        }
    }
}

/// `attn/mla_paged.cuh` — the MLA cache's append and its preparation pass.
///
/// **Unit-only, the fifth**, and by the rule the last four established: *a
/// root whose host programs are already Rust and already outside a `bind!`
/// crosses as a `unit!` and nothing else.* Both host programs are
/// `driver-cuda/src/fire/mla_paged.rs`, whole, with every grid and block
/// figure already cited to a line there. **Both table rows stay**, because a
/// `unit!` moves device text and only a `bind!` retires a row.
///
/// # Why they cannot be bound, which is not a floor gap
///
/// `table::attn`'s two rows are UNSOURCED on every operand and `whole =
/// true`. That is §60.7's case and it is legitimate: `crate::abi` skips a row
/// with any `Source::Unbound` operand whole, so no dispatch arm was ever
/// generated for either and neither was reachable before the crossing or
/// after it. What the rows buy is the shim entry. What a `bind!` would need
/// is a `MlaCacheLayerView` — one dispatch argument whose FIVE fields the
/// kernels take unpacked — and `Cx::kv_layer()` answers the KV cache's
/// layout, not the MLA cache's. That is a real gap and it is **not asked for
/// here**, because the consumer that would exercise it is
/// `dispatch_attention_mla_bf16`, whose own arm is blocked on three other
/// things (see [`mla_fa2`]) and which would state the view itself.
///
/// # The two symbols are BOTH twinned, and this is the family's worst pair
///
/// §60.6's split reached both kernels and `fire/mla_paged.rs:66-70` is the
/// only bridge:
///
/// ```text
/// device  attn::write_mla        table  attn::write_mla_to_pages
/// device  attn::mla_prepare      table  attn::mla_prepare_bf16
/// ```
///
/// The `unit!` below states the DEVICE names, as `families::attn`'s deleted
/// `MLA_PAGED_SIGS` did and for its stated reason: the ahead-of-time symbol
/// takes the `MlaCacheLayerView` by value and unpacks it, so a row claiming
/// the launcher's name would claim a view the `__global__` has never seen.
///
/// **`mla_prepare_bf16` carries a format suffix its kernel cannot justify and
/// `write_mla_to_pages` does not** — two twins of one root spelled by two
/// different conventions. Neither kernel has an element-type parameter; both
/// are `bf16` in their own declarations. The rows are what they are and this
/// pass does not rename them, but a reader deriving a device name from a
/// table name would get one of the two wrong.
pub mod mla_paged {
    use super::bf16;

    unit! {
        /// Two `__global__`s, one of them a template over its BLOCK WIDTH
        /// and neither over an element type.
        ///
        /// `mla_paged.cuh:87-95` argues that `write_mla` stays a non-template:
        /// it has no honest parameter, and §21.6's measurement — a plain
        /// `__global__` is nameable by its bare qualified path, which NVRTC
        /// lowers and `cuModuleGetFunction` resolves — is what lets it be a
        /// row at all.
        unit MLA_PAGED = "attn/mla_paged",
            text = include_str!("../../csrc/src/attn/mla_paged.cuh"),
            file = "attn/mla_paged.cuh";

        /// `mla_paged.cuh:174` — append one token's latent KV to its page.
        ///
        /// `<<<total_tokens, 256, 0, stream>>>`, one block per row of
        /// `ckv_curr`, which is `[Tokens, kv_lora_rank]`.
        ///
        /// `row_valid` is NULLABLE and the kernel says so at `:190`:
        /// `if (row_valid != nullptr && row_valid[t] == 0) return;`. A fire
        /// that published no validity mask hands a null pointer, which is
        /// why the parameter is `*const u8` and not an operand the caller may
        /// omit.
        ///
        /// `R` is `num_requests` — the CSR's request count, which
        /// `mla_resolve_dst` walks — and NOT the token count the grid opens
        /// over. Two extents, one launch, and only one of them is
        /// recoverable from a rule.
        fn write_mla = "attn::device::write_mla" (
            ckv_curr: *const bf16,
            kpe_curr: *const bf16,
            ckv_pages: *mut bf16,
            kpe_pages: *mut bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            row_valid: *const u8,
            r: i32,
            page_size: i32,
            kv_lora_rank: i32,
            qk_rope_head_dim: i32,
        ) {
            "attn::write_mla" => crate::device::DeviceKernel::PLAIN,
        }

        /// `mla_paged.cuh:223` — the fused MLA prepare, at `BLOCK_DIM = 256`.
        ///
        /// # The grid's leading lane is not a head
        ///
        /// ```text
        /// mla_paged.cu:67    dim3 grid(total_tokens, 1 + q_blocks);
        /// mla_paged.cu:68    device::mla_prepare<BS><<<grid, BS, 0, stream>>>(...)
        /// ```
        ///
        /// `mla_paged.cuh:236` reads `const int qb = blockIdx.y - 1;` and
        /// takes the KV path when `qb < 0`, so lane `y = 0` owns the `kv_a`
        /// norm, the `k_pe` rotation and the paged write for its token, and
        /// lanes `1..=q_blocks` are the query heads. A rule that folded the
        /// `1` into the head axis would open the right number of blocks,
        /// shift every head down by one, drop the last, and **never write the
        /// cache** — while `q_nope`/`q_pe` still filled, so the fire would
        /// produce a plausible query against an unwritten page.
        ///
        /// # `256` is a block width AND a comparison
        ///
        /// `mla_paged.cu:64` computes `heads_per_block = half >= BS ? 1 : BS
        /// / half` from the same `BS`, where `half` is `qk_rope_head_dim / 2`
        /// — so the block width and the second grid axis are **one number
        /// stated twice**, and the row states it because
        /// `__shared__ float buf[BLOCK_DIM]` is sized by it and reduced by
        /// halving over it. A row at `<512>` under a 256-wide launch would
        /// leave the upper half of `buf` unwritten, read it on the first
        /// halving step, AND compute half the query blocks: two wrong
        /// answers from one changed literal.
        ///
        /// `device::i32(256)` and not `256`: `DeviceKernel::instantiation`
        /// qualifies an `elem` that does not begin `::` with
        /// `::pie_cuda_driver::kernels::`, so the functional-cast spelling is
        /// what survives that prefix as a non-type argument.
        fn mla_prepare = "attn::device::mla_prepare" (
            kv_a: *const bf16,
            kv_a_norm_w: *const bf16,
            q_b: *const bf16,
            kv_c: *mut bf16,
            k_pe: *mut bf16,
            q_nope: *mut bf16,
            q_pe: *mut bf16,
            ckv_pages: *mut bf16,
            kpe_pages: *mut bf16,
            positions: *const i32,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            row_valid: *const u8,
            r: i32,
            page_size: i32,
            heads: i32,
            kv_lora: i32,
            nope: i32,
            rope: i32,
            src_row_stride: i32,
            eps: f32,
            theta: f32,
            interleaved: bool,
            heads_per_block: i32,
            yarn_factor: f32,
            yarn_low_dim: f32,
            yarn_high_dim: f32,
            yarn_mscale: f32,
        ) {
            "attn::mla_prepare" => "device::i32(256)",
        }
    }
}

/// The FlashInfer FA2 MLA host program — **compiled, and not yet fireable.**
///
/// This is the host half of `table::attn`'s `dispatch_attention_mla_bf16`
/// FA2 arm: `flashinfer::mla::BatchMLAPagedAttention<MASK, 512, 64>`, whose
/// device text is `attention_mla.cu`'s `mla_mma_paged_kernel` — **two of the
/// last `<<<>>>` nvcc compiles in the workspace.** §66 measured what that row
/// is worth: `kernels-cuda/native` is the only switch over the entire nvcc
/// and `.cpp` surface, it is turned on by `driver-cuda/bridge` alone, and
/// `bridge` is deletable when `ROW_TABLES` empties.
///
/// # State, in one paragraph
///
/// The root is on disk, the unit is enrolled with its one NVRTC option, all
/// six instantiations and all three shared-memory echoes lower, the launch is
/// declared `cooperative = true`, and [`arm_for`], [`pack`] and [`grid`]
/// produce everything [`raw::attention`] takes. The FA2 arm of
/// `attn::dispatch_attention_mla_bf16` is Rust end to end, the row is gone
/// from `table/attn.rs`, and `attention_mla.cu` is deleted.
///
/// **What has no caller is the ARM CHOICE**, not this arm. The contract
/// `crate::x::attn::ATTENTION_MLA` is `none:`, because a bind body would
/// have to pick between this and `driver-cuda/src/fire/mla_naive.rs` on
/// `cudaDevAttrComputeCapabilityMajor >= 10` and nothing in `Cx` or the
/// runtime states a compute capability — nor the MLA cache layer, the plan
/// handle, the workspace or `sm_scale`. Four `Cx` queries and one device
/// attribute; the exact patch is beside the `none:` arm.
///
/// # The `unit!` exists now, and the reason it did not is worth keeping
///
/// **The vendored `mla.cuh` did not compile under NVRTC and never had.**
/// `csrc/vendor/flashinfer/attention/mla.cuh:806` and `:847` write
///
/// ```text
/// o_smem.template store_128b(o_smem_offset_w, o_partial_ptr);
/// ```
///
/// `store_128b` IS a member template (`permuted_smem.cuh:184`,
/// `template <typename T>`), but `T` is deducible, so the `template`
/// disambiguator is followed by a name rather than a template-id. **NVRTC
/// 13.0 rejects this at both `-std=c++17` and `-std=c++20`** —
/// `error: argument list for template "S<N>::f [with N=N]" is missing` —
/// isolated away from FlashInfer entirely in `nvrtc-probes/mla_template_kw.py`
/// so that the finding is about the construct and not about the closure. Both
/// alternative spellings compile: dropping the keyword, which is what
/// `prefill.cuh:1922` writes for the identical call, and adding an explicit
/// `<DTypeO>`.
///
/// **This is the `<cstring>` shape a third time, and worse.** That one was a
/// dependency satisfied by accident; this is a header CARRIED but never
/// compiled. `source.rs` names no `mla.cuh` entry, but `carried.rs` generates
/// the set by WALKING `csrc/`, so the file has been shipped to every NVRTC
/// compile for months while nothing instantiated anything that reached those
/// two lines. `csrc/shim/cooperative_groups.h`'s own banner says as much:
/// `mla.cuh` is *"the third file to open this door and which nothing in the
/// tree includes"*. A `unit!` naming this root would have been a compilation
/// that fails at the JIT, which is worse than no unit at all — so the device
/// text stayed where it was until the one-token repair landed, and the repair
/// was a vendoring decision and therefore an ask.
///
/// # The three asks, all landed as `f622dcf8d`
///
/// 1. **`csrc/vendor/flashinfer/attention/mla.cuh:806,847`** — the `template`
///    keyword, now under `#ifndef __CUDACC_RTC__` with upstream's spelling
///    kept and ours in the `#else`. `MODIFICATIONS`' recovery transform
///    gained one clause — *discard the `#else` branch* — so FlashInfer
///    v0.6.15 is still recoverable byte for byte.
/// 2. **`csrc/shim/cooperative_groups.h`** — `grid_group` and `this_grid()`.
///    The shim had omitted them ON PURPOSE and its banner was right to: *"a
///    grid-wide barrier is a LAUNCH MODE"*, and a faked `sync()` either
///    deadlocks or lets `mla.cuh:1061`'s stage two read stage one's partials.
///    It was asked for only ALONGSIDE (3), which is what the banner's own
///    last sentence had said to do. `sync()` calls NVIDIA's own pair,
///    `cudaCGGetIntrinsicHandle(scopeGrid)` then `cudaCGSynchronize(handle, 0)`,
///    rather than anything invented.
/// 3. **`runtime/module.rs::fire_ex`** — a third `CUlaunchAttribute` slot for
///    `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` and a `cooperative: bool` parameter
///    in fifth position. `fire_ex` was already the precedent AND the
///    argument: `fimoe-rust` added it rather than growing `Launch`, because
///    *"`Launch` is `eval`'s return type and every family builds one, and a
///    cluster is a property of the kernel at instantiation."*
///
/// **One thing is UNMEASURED and the shim marks it `CG_SYNC_RESOLUTION_UNMEASURED`.**
/// The two `cudaCG*` intrinsics lower to `.extern .func`, which is a promise
/// and not a link, and whether `cuModuleLoadData` resolves them needs a CUDA
/// context no probe here takes. `libcudadevrt.a` carries no `cudaCG*` — and
/// no `cudaGraphSetConditional` either, which is §62's case, where the driver
/// resolved it anyway because NVRTC and nvcc share `cicc`. If MLA's second
/// stage fails at module load, that marker is the line.
///
/// # The two clauses this root asked `unit!` for, both landed as `a9a633d38`
///
/// **`options =` on the unit line.** `unit!` hard-coded `options: &[]` on the
/// reasoning that a compile-option list is a property of the recipe rather
/// than of a unit — true for eleven families and false for the twelfth. See
/// [`OPTIONS`] and the sixteen errors it was measured against; the default
/// stays `&[]`.
///
/// **`cooperative =` on the `fn` line.** On the DECLARATION and not at the
/// call site, because it is a property of the kernel: `mla.cuh:1061`'s two
/// stages are separated by a `this_grid().sync()` and every other kernel in
/// this tree synchronises no further than its own block. At a call site a
/// caller can forget it and get a hang; on the declaration a mismatch is a
/// compile error.
///
/// `x::fire::fire_ex(symbol, launch, cooperative, values, stream)` is what
/// `raw::attention` now reaches, and `fire` delegates to it with `false` — so
/// the resolution order is still stated once, which is what that file is for.
/// **Nothing here ever held a private copy of it**, and the four days that
/// choice cost were the right four days.
///
/// **With (1) applied the kernel LOWERS, and now does so from the tree.**
/// `nvrtc-probes/attn_mla_fa2_root.py` compiles the candidate root and
/// `attn_mla_fa2_ondisk.py` compiles `csrc/src/attn/attention_mla_fa2.cuh`
/// as written, against `csrc/{shim,vendor,src}` with nothing patched out of
/// tree: rc=0, 2.3 MB of PTX, six `.entry`, and all nine name expressions —
/// six kernels and three `&`-prefixed echoes — lowered.
///
/// # Three things that were believed to be blockers and are NOT
///
/// **The grid is not an occupancy query.** `flashinfer_decode.rs:1860-1885`
/// says *"the GRID must come from an occupancy query rather than from a
/// rectangle"*. It does not: `scheduler.cuh:1607-1608` sets
/// `num_blks_x = cluster_size` (1 or 2) and `num_blks_y = num_sm /
/// cluster_size`, so the grid is **exactly `num_sm` blocks and resident by
/// construction**, from `cudaDevAttrMultiProcessorCount`. That is why
/// [`grid`] below is two reads and no query. The planner is already Rust —
/// `crate::plan::mla`, `Schedule { cluster_size, num_clusters, … }` — and
/// [`MlaPlanInfo`] is already mirrored and offset-asserted.
///
/// **The shared-memory size needs no runtime measurement.** `mla.cuh:1128`
/// computes `smem_size = sizeof(KTraits::SharedStorage)` in C++, and
/// `nvrtc-probes/attn_mla_fa2_smem.py` measured all three
/// `DISPATCH_SMEM_CONFIG` arms: **`sizeof(SharedStorage)` is EXACTLY the
/// arm's own threshold literal**, 221 696 / 147 968 / 92 672, align 16, with
/// `causal` changing nothing. So the selection rule is literally *"the
/// biggest tile whose shared storage fits an SM"* and a host needs one number
/// per arm. [`ARMS`] is that table.
///
/// **The plan cache exists.** `driver-cuda/src/fire/flashinfer_fa2.rs:534`'s
/// `MlaPlanCache` is documented as *"plans; nothing in this crate yet
/// launches from it"*.
///
/// # The instantiation name is a §3.2 hazard in a new dress
///
/// There are TWO `KernelTraits` in the closure — `prefill.cuh:159`, whose
/// first parameter is a `MaskMode`, and `mla.cuh:81`, whose first parameter
/// is a `bool CAUSAL_`. `mla.cuh:1124` spells it unqualified and
/// enclosing-namespace lookup picks the right one; **a transcription that
/// writes the qualified name has to know which, and both are nameable.** The
/// root's `Traits` alias writes `::flashinfer::mla::KernelTraits` once, so
/// the six rows never spell it at all — which is the third reason the alias
/// exists, after the two `fa2.cuh` gives. Fifteen parameters against eleven
/// is a substitution failure, not a diagnostic that names the confusion.
pub mod mla_fa2 {
    use super::bf16;
    use super::mla_params::{MlaParams, UintFastdiv};
    use crate::plan::MlaPlanInfo;
    use crate::x::launch::Launch;

    /// One `DISPATCH_SMEM_CONFIG` arm of `mla.cuh:1079`.
    ///
    /// `sizeof(KTraits::SharedStorage)` and the threshold that selects the arm
    /// are the SAME NUMBER, which is a measurement and not a coincidence of
    /// this table's construction — see the module doc.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Arm {
        /// `KTraits::NUM_STAGES`.
        pub stages: u32,
        /// `KTraits::CTA_TILE_KV`.
        pub cta_tile_kv: u32,
        /// `KTraits::QK_SHARD`.
        pub qk_shard: bool,
        /// `sizeof(KTraits::SharedStorage)`, in bytes, and the smallest
        /// `smem_limit_per_sm` this arm may be chosen for.
        pub smem: u32,
    }

    /// The three arms, widest first, which is `DISPATCH_SMEM_CONFIG`'s order.
    ///
    /// `NUM_THREADS` is 256 on all three (`nthrs(32, 4, 2)`) and `CTA_TILE_Q`
    /// is 64; `NUM_MMA_KV` is `CTA_TILE_KV / 16`. `causal` selects nothing
    /// here — it is a template argument of the kernel, not of the storage.
    pub const ARMS: [Arm; 3] = [
        Arm { stages: 2, cta_tile_kv: 64, qk_shard: true, smem: 221_696 },
        Arm { stages: 2, cta_tile_kv: 32, qk_shard: true, smem: 147_968 },
        Arm { stages: 1, cta_tile_kv: 16, qk_shard: false, smem: 92_672 },
    ];

    /// The widest arm this device's shared-memory budget admits.
    ///
    /// `smem_limit_per_sm` is `cudaDevAttrMaxSharedMemoryPerMultiprocessor`.
    /// Returns `None` when even the narrowest arm does not fit, which is a
    /// device this kernel was never compiled for and is a refusal the caller
    /// must make **before the first launch**.
    #[must_use]
    pub const fn arm_for(smem_limit_per_sm: u32) -> Option<Arm> {
        let mut i = 0;
        while i < ARMS.len() {
            if smem_limit_per_sm >= ARMS[i].smem {
                return Some(ARMS[i]);
            }
            i += 1;
        }
        None
    }

    unit! {
        /// `flashinfer::mla::BatchMLAPagedAttentionKernel`, six ways.
        ///
        /// The root holds no `__global__` of ours: it is the `#include` list,
        /// four `using`s, one alias template and one `__device__` echo, in
        /// `csrc/src/attn/fa2.cuh`'s idiom and for its reasons. Read that
        /// file's header for the two `KernelTraits` hazard, the three
        /// residency facts and the `&`-prefix rule.
        unit MLA_FA2 = "attn/attention_mla_fa2",
            text = include_str!("../../csrc/src/attn/attention_mla_fa2.cuh"),
            file = "attn/attention_mla_fa2.cuh",
            options = OPTIONS;

        /// `mla.cuh:879` — the whole of paged MLA, in two stages separated by
        /// a grid-wide barrier.
        ///
        /// **ONE parameter, by value.** `mla.cuh:1130` is
        /// `void* args[] = {(void*)&params};` and the kernel takes
        /// `const __grid_constant__ Params params` — so there is no operand
        /// list to speak of and every pointer, extent and stride the kernel
        /// reads is a field of [`MlaParams`], packed by [`pack`]. That is
        /// what made `ArgValue::Bytes` this family's dependency, and it is
        /// the failure mode §5.1 named: a wrong bypass is a launch with a
        /// garbage struct, not a type error.
        ///
        /// # This launch MUST be cooperative, and the declaration says so
        ///
        /// `mla.cuh:1061` calls `grid.sync()` between the two stages and
        /// `:1132` launches through `cudaLaunchCooperativeKernel`. A
        /// non-cooperative launch is not an error at any layer — it is a
        /// deadlock in stage two — so `cooperative = true` is on the `fn`
        /// line and `raw::attention` reaches `x::fire::fire_ex`. **On the
        /// declaration and not at the call site**, because it is a property
        /// of this kernel and of no other in the tree: every other
        /// `__global__` here synchronises no further than its own block. A
        /// caller cannot forget it, and a caller that disagreed with it would
        /// not compile.
        ///
        /// # The six rows are three shared-memory arms times the mask
        ///
        /// [`ARMS`] is `DISPATCH_SMEM_CONFIG` (`mla.cuh:1100-1120`) and the
        /// mask is a `bool`: `MaskMode::kCustom` is refused with
        /// `cudaErrorNotSupported` at `:1123`, before a traits type is
        /// formed, so nothing here is three-valued.
        ///
        /// Each `elem` starts with `::` so `DeviceKernel::qualify` leaves it
        /// alone — it has to, because the kernel takes TWO template arguments
        /// and `qualify` prefixes a field once rather than per argument.
        fn attention = "::flashinfer::mla::BatchMLAPagedAttentionKernel" (
            params: MlaParams,
        ), cooperative = true {
            "attn::mla_fa2_kv64_causal" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 2u, true, 64u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv64_full" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<false, 2u, true, 64u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv32_causal" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 2u, true, 32u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv32_full" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<false, 2u, true, 32u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv16_causal" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 1u, false, 16u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
            "attn::mla_fa2_kv16_full" =>
                "::pie_cuda_driver::kernels::attn::mla_fa2::Traits<false, 1u, false, 16u>, \
                 ::pie_cuda_driver::kernels::attn::mla_fa2::Params",
        }
    }

    /// The one NVRTC option this root needs, and the sixteen errors that say
    /// so.
    ///
    /// **Measured, not assumed.** Without
    /// `--device-as-default-execution-space` the root is rejected sixteen
    /// times — `csrc/shim/type_traits:253`, seven sites in `cascade.cuh` and
    /// eight in `prefill.cuh` — all *"A function without execution space
    /// annotations ... is considered a host function"*. `mla.cuh:33` includes
    /// `prefill.cuh`, so this is the same closure `families::fa2` compiles
    /// and the same flag it passes (`families/fa2.rs:302`): a third instance
    /// of one entry rather than a new class.
    ///
    /// Per-unit and never global, for the reason `unit::Unit::options` gives
    /// at its own field: turning it on for everyone would silently compile
    /// OUR unannotated host helpers onto the device, and the shared options
    /// are a contract the cache key spans. It is the first `options =` clause
    /// in the tree and the reason the clause exists — the default stays
    /// `&[]`, because an option repeated per unit would be a recipe wearing a
    /// declaration.
    pub const OPTIONS: &[&str] = &["--device-as-default-execution-space"];

    /// The six row symbols, indexed by `[arm][causal]`, parallel to [`ARMS`].
    ///
    /// `HEAD_DIM_CKV = 512`, `HEAD_DIM_KPE = 64`, `CTA_TILE_Q = 64` and the
    /// four types are fixed in the root's `Traits` alias, so a row states the
    /// four numbers that vary and nothing else — the `<MASK, 512, 64>` the
    /// dispatch arm names, spread over the traits the kernel actually takes.
    ///
    /// **`__nv_bfloat16` mangles as `pie_cuda_driver::kernels::device::bf16`**
    /// in the lowered PTX, because the shim's `cuda_bf16.h` typedefs it into
    /// the prelude namespace. That is a property of this tree's shim and not
    /// of FlashInfer, and it is the reason a symbol lookup against a
    /// vendor-built cubin would not find these.
    pub const SYMBOLS: [[&str; 2]; 3] = [
        ["attn::mla_fa2_kv64_full", "attn::mla_fa2_kv64_causal"],
        ["attn::mla_fa2_kv32_full", "attn::mla_fa2_kv32_causal"],
        ["attn::mla_fa2_kv16_full", "attn::mla_fa2_kv16_causal"],
    ];

    /// The compiler's own `sizeof(KTraits::SharedStorage)` per arm, as name
    /// expressions, parallel to [`ARMS`].
    ///
    /// `fa2::PrefillGeometry::ECHO_TEMPLATE` is the precedent and the
    /// `&` prefix is not decoration: `nvrtcAddNameExpression` refuses
    /// `smem_bytes_mla<KT>` and accepts `&smem_bytes_mla<KT>`, because a
    /// function's name is its address and a variable's is not. All three
    /// lowered in `nvrtc-probes/attn_mla_fa2_ondisk.py`.
    ///
    /// **Nothing reads these yet**, and what they would catch is narrow and
    /// serious: `ARMS[i].smem` is a LITERAL copied out of upstream's own
    /// threshold comparison, and upstream changing `SharedStorage` without
    /// changing the threshold leaves the launch correctly sized by its
    /// `sizeof` and the ARM chosen wrong, silently, on a device whose shared
    /// memory falls between the old literal and the new size. Whoever wires
    /// `cuModuleGetGlobal` compares and refuses rather than trusting either
    /// side.
    pub const SMEM_ECHO: [&str; 3] = [
        "&::pie_cuda_driver::kernels::attn::mla_fa2::smem_bytes_mla<\
         ::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 2u, true, 64u>>",
        "&::pie_cuda_driver::kernels::attn::mla_fa2::smem_bytes_mla<\
         ::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 2u, true, 32u>>",
        "&::pie_cuda_driver::kernels::attn::mla_fa2::smem_bytes_mla<\
         ::pie_cuda_driver::kernels::attn::mla_fa2::Traits<true, 1u, false, 16u>>",
    ];

    /// The MLA cache's shape, as `dispatch_mla_512_64` reads it off
    /// `MlaPlanCache`.
    ///
    /// Nine numbers and a scale, and the reason they are a struct rather than
    /// ten parameters is that eight of the ten `*_stride_*` fields below are
    /// PRODUCTS of two of them — writing the products at the call site is how
    /// a packer gets one wrong.
    #[derive(Clone, Copy, Debug)]
    pub struct Shape {
        /// Tokens per page.
        pub page_size: u32,
        /// Query heads.
        pub num_heads: u32,
        /// The compressed KV width — `HEAD_DIM_CKV`, 512.
        pub kv_lora_rank: u32,
        /// The positional width — `HEAD_DIM_KPE`, 64.
        pub qk_rope_head_dim: u32,
        /// `1 / sqrt(head_dim)`, or whatever the deployment states.
        pub sm_scale: f32,
    }

    /// Where the two workspace arenas start, and the addresses the fire
    /// carries.
    #[derive(Clone, Copy, Debug)]
    pub struct Buffers {
        /// `AttentionWorkspaceView::int_buffer`.
        pub int_buffer: *mut u8,
        /// `AttentionWorkspaceView::float_buffer`.
        pub float_buffer: *mut u8,
        /// `[tokens, num_heads, kv_lora_rank]`.
        pub q_nope: *mut bf16,
        /// `[tokens, num_heads, qk_rope_head_dim]`.
        pub q_pe: *mut bf16,
        /// The layer's compressed KV pages.
        pub ckv_pages: *mut bf16,
        /// The layer's positional pages.
        pub kpe_pages: *mut bf16,
        /// The result, in the latent space.
        pub out: *mut bf16,
        /// The uploaded page-index array. NOT workspace-relative — it is a
        /// device pointer the fire already holds.
        pub kv_page_indices: *mut i32,
        /// The LSE, or null when the statement does not ask for one.
        pub lse: *mut f32,
    }

    /// `int_buf + offset`, in ELEMENTS of `T`.
    ///
    /// `attention_mla.cu`'s `offset_ptr<T>` to the byte: every `*_offset` in
    /// [`MlaPlanInfo`] is an index into the arena's element array and not a
    /// byte count, which is why the multiply is here and not at the call
    /// sites. Getting this wrong scales every plan pointer by four and is a
    /// fault rather than a wrong answer, which is the only reason it is safe
    /// to state once.
    unsafe fn offset_ptr<T>(base: *mut u8, offset: i64) -> *mut T {
        unsafe { base.cast::<T>().offset(offset as isize) }
    }

    /// Fill an [`MlaParams`] the way `attention_mla.cu:264-320` does.
    ///
    /// # This is the FIRST Rust packer of this struct and is the reference
    ///
    /// There is no second side to compare against: `MLAParams` is packed only
    /// in `attention_mla.cu`, which is the ahead-of-time path, and
    /// [`UintFastdiv::new`] records why the two shapes could never have
    /// agreed anyway — the shim's `fast_mod_div` is 16 bytes where CCCL's
    /// interior differs at the same `sizeof`, so `MLAParams` is 288 here and
    /// 248 there. **This is not a transcription of that file; it is the thing
    /// that file would have to be checked against.**
    ///
    /// # `ckv_scale` and `kpe_scale` are written HERE and are not in the C++
    ///
    /// `attention_mla.cu` never assigns them: it relies on
    /// `mla_params.cuh`'s default member initialiser of `1.f`. A Rust `struct`
    /// literal has no such thing, so a packer that omitted them would leave
    /// whatever the caller's memory held, and a ZEROED struct would scale
    /// every value to zero — **a silent wrong answer, not a fault.** The
    /// struct's own field doc says this and this function is where it is
    /// obeyed.
    ///
    /// # Safety
    ///
    /// Every pointer in `buffers` must be a device address valid for the
    /// fire, and `plan` must be the plan those arenas were uploaded from.
    /// Nothing is dereferenced here; the requirement is the kernel's.
    #[must_use]
    pub unsafe fn pack(
        plan: &MlaPlanInfo,
        shape: Shape,
        buffers: Buffers,
        want_lse: bool,
    ) -> MlaParams {
        let int_buf = buffers.int_buffer;
        let float_buf = buffers.float_buffer;
        MlaParams {
            q_nope: buffers.q_nope,
            q_pe: buffers.q_pe,
            ckv: buffers.ckv_pages,
            kpe: buffers.kpe_pages,
            partial_o: unsafe { offset_ptr(float_buf, plan.partial_o_offset) },
            partial_lse: unsafe { offset_ptr(float_buf, plan.partial_lse_offset) },
            final_o: buffers.out,
            // `want_lse` is the statement's, not the plan's: FlashInfer reads
            // a null `final_lse` as "do not write one", so the decision is a
            // pointer and not a flag. A caller that passed a live pointer for
            // a statement with one result would write past the end of the
            // fire's allocation.
            final_lse: if want_lse { buffers.lse } else { ::core::ptr::null_mut() },
            q_indptr: unsafe { offset_ptr(int_buf, plan.q_indptr_offset) },
            kv_indptr: unsafe { offset_ptr(int_buf, plan.kv_indptr_offset) },
            partial_indptr: unsafe { offset_ptr(int_buf, plan.partial_indptr_offset) },
            merge_packed_offset_start: unsafe {
                offset_ptr(int_buf, plan.merge_packed_offset_start_offset)
            },
            merge_packed_offset_end: unsafe {
                offset_ptr(int_buf, plan.merge_packed_offset_end_offset)
            },
            merge_partial_packed_offset_start: unsafe {
                offset_ptr(int_buf, plan.merge_partial_packed_offset_start_offset)
            },
            merge_partial_packed_offset_end: unsafe {
                offset_ptr(int_buf, plan.merge_partial_packed_offset_end_offset)
            },
            merge_partial_stride: unsafe {
                offset_ptr(int_buf, plan.merge_partial_stride_offset)
            },
            // NOT workspace-relative. The page indices are uploaded by the
            // caller and handed in as a device address, which is what
            // `attention_mla.cu:293` does with its `kv_page_indices_d`.
            kv_indices: buffers.kv_page_indices,
            q_len: unsafe { offset_ptr(int_buf, plan.q_len_offset) },
            kv_len: unsafe { offset_ptr(int_buf, plan.kv_len_offset) },
            q_start: unsafe { offset_ptr(int_buf, plan.q_start_offset) },
            kv_start: unsafe { offset_ptr(int_buf, plan.kv_start_offset) },
            kv_end: unsafe { offset_ptr(int_buf, plan.kv_end_offset) },
            work_indptr: unsafe { offset_ptr(int_buf, plan.work_indptr_offset) },
            block_size: UintFastdiv::new(shape.page_size),
            num_heads: UintFastdiv::new(shape.num_heads),
            q_nope_stride_n: shape.num_heads * shape.kv_lora_rank,
            q_nope_stride_h: shape.kv_lora_rank,
            q_pe_stride_n: shape.num_heads * shape.qk_rope_head_dim,
            q_pe_stride_h: shape.qk_rope_head_dim,
            ckv_stride_page: shape.page_size * shape.kv_lora_rank,
            ckv_stride_n: shape.kv_lora_rank,
            kpe_stride_page: shape.page_size * shape.qk_rope_head_dim,
            kpe_stride_n: shape.qk_rope_head_dim,
            // The output is in the LATENT space, so its strides are
            // `kv_lora_rank`'s and not the model's head dim. That is the same
            // reading `attention_mla_naive.cuh` takes and the reason the rest
            // of the MLA forward is unchanged between the two backends.
            o_stride_n: shape.num_heads * shape.kv_lora_rank,
            o_stride_h: shape.kv_lora_rank,
            sm_scale: shape.sm_scale,
            ckv_scale: 1.0,
            kpe_scale: 1.0,
            // `attention_mla.cu:320` writes `true` unconditionally: the
            // kernel emits its LSE in natural log rather than log2, which is
            // the opposite convention to the rest of FlashInfer and is why
            // `attn::lse_log2_to_ln` exists for the OTHER backends and must
            // NOT be applied to this one.
            return_lse_base_on_e: true,
        }
    }

    /// The grid, from the plan and nothing else.
    ///
    /// `[num_blks_x, num_blks_y, 1]` — `cluster_size` by `num_sm /
    /// cluster_size`, so the product is `num_sm` and **every block is
    /// resident by construction.** That is what makes a cooperative launch
    /// legal here without an occupancy query, and it is the correction to
    /// `flashinfer_decode.rs:1873`.
    ///
    /// The block is `[256, 1, 1]` on all three arms — `nthrs(32, 4, 2)` — and
    /// the lowered PTX agrees: `.maxntid 256, 1, 1`.
    #[must_use]
    pub const fn grid(plan: &MlaPlanInfo, arm: Arm) -> Launch {
        Launch {
            grid: [plan.num_blks_x as u32, plan.num_blks_y as u32, 1],
            block: [256, 1, 1],
            smem: arm.smem,
            // 221 696 and 147 968 are both far above the 48 KiB static cap,
            // so two of the three arms REQUIRE the opt-in.
            // `runtime::module::raise_dynamic_smem_cap` performs it inside
            // the fire, once per `(CUdevice, CUfunction)`, driven by
            // `Launch::smem` — which is why this is a flag and not a call.
            smem_opt_in: true,
        }
    }
}

/// The units `attn` compiles in fn-world.
///
/// Hand-written where a one-root family's is generated, for the reason the
/// block comment above gives. `families::ALL` reads this **beside**
/// `families::attn::UNITS`, which holds the TWO roots these passes did not
/// take — `ATTN_SCORE_POST` and `ATTN_XQA`, four device rows between them.
/// A root appears in exactly one of the two lists: a second `unit!`
/// naming the same text would be a second compilation of it under a second
/// unit name, and `unit_of` would answer with whichever won.
/// `attn/qkv_fused.cuh` — the fused QKV epilogues.
///
/// Three `__global__` templates and no host code in the header, which is the
/// whole reason this root crossed cleanly: every decision the deleted
/// `qkv_fused.cu` made was already Rust when the text arrived, and both of
/// its host programs are in this module now — the packed prefill epilogue and
/// [`qkv_decode_fused_dispatch`], which came across from
/// `driver-cuda/src/fire/qkv_fused.rs` once its `unit!` was here to fire.
///
/// # ELEVEN ROWS OVER NINE INSTANTIATIONS, and the `#` names are carried
/// # VERBATIM
///
/// `families::attn::QKV_FUSED` stated eleven rows, and two of them —
/// `attn::qkv_decode_qk_norm_rope_write_kv` and
/// `..._warp_d128` — are BASE rows that name the same instantiation as their
/// own `#norope` arm. That was a `Specialisation` mechanism: a base row plus
/// arms, with `flags_are_covered` proving the base unreachable.
///
/// **fn-world has no `Specialisation` and the rows are carried anyway.** The
/// reason is not symmetry, it is that the host program fires by NAME:
/// [`warp_symbol`] and [`block_symbol`] return exactly these strings, `#rope`
/// and `#norope` suffixes included. Renaming them here would be a
/// `NoLoweredName` at the first decode fire on a machine with a GPU, and the
/// suffix is legal in fn-world for a checkable reason: `x/abi.rs:824`'s
/// `mangle` already lists `'#'` among the characters it replaces when it
/// writes the typecheck TU. **The row world's arm spelling is a symbol here,
/// not a mechanism**, and that is the cheapest possible crossing of a
/// `Specialisation` — cheaper than `x/norm.rs:1033`'s *"`Specialisation`s
/// become `if`s"*, because the `if` already exists and is in the driver.
///
/// A duplicate name expression costs nothing: NVRTC accepts it, and
/// `QKV_FUSED_ROWS` said so first.
///
/// # `128` means two different things and the two are spelled the same
///
/// `qkv_decode_qk_norm_rope_write_kv<128, …>`'s first argument is a BLOCK
/// width — it sizes `__shared__ float buf[BLOCK]`.
/// `qkv_decode_qk_norm_rope_write_kv_warp<128, …>`'s is a HEAD width: it
/// fixes `ELEMS_PER_THREAD = HEAD_DIM / 32` and every `#pragma unroll` under
/// it, while the block width is the launcher's `WARP_BLOCK = 256`, read at
/// run time from `blockDim.x`. The warp form is compiled at 64, 128 and 256
/// because `head_dim` decides which, and the block form at 128 only because
/// its argument is not a head width at all.
///
/// # `win` and `row_valid` are both nullable and mean different things
///
/// `row_valid` is a validity mask a fire either published or did not, tested
/// `row_valid != nullptr && row_valid[row] == 0`: absent means *every row is
/// valid*. `win` is the Peel device window's PREFIX form, and the non-devwin
/// entry point hands it `nullptr` outright: absent means *the split is not
/// device-decided*. Both are `| null` and both are real.
///
/// # `rope_table` is a HOST NULL TEST, not a launch parameter
///
/// It is a `const float*` PARAMETER of both instantiations — the host passes
/// it to `USE_ROPE_TABLE = false` too, which reads it never. So every arm
/// forwards all twenty-two arguments and the base binds exactly what a
/// fall-through kernel declares; there is no cell to leave unread. That is
/// why the pair is two instantiations rather than a runtime branch: the
/// unrolled table read is compiled out, not skipped.
pub mod qkv_fused {
    use super::bf16;
    // Gated with the host `fn` below, because `raw` is and because
    // `Fired` and `Refusal` are `x::attn`'s own gated imports: a plain
    // `use super::Fired` here does not resolve in the toolkit-free build.
    #[cfg(feature = "_cuda")]
    use super::{Fired, Launch, Refusal};

    unit! {
        /// Three `__global__` templates, no host code.
        ///
        /// The root moved from `crates/kernels-cuda/csrc/src/attn/` long ago;
        /// what moves here is the DECLARATION, out of `families::attn`'s
        /// `DeviceKernel` rows and into the grammar that states a kernel's
        /// parameters beside its instantiations.
        unit QKV_FUSED = "attn/qkv_fused",
            text = include_str!("../../csrc/src/attn/qkv_fused.cuh"),
            file = "attn/qkv_fused.cuh";

        /// `qkv_fused.cuh:412` — the PACKED prefill epilogue.
        ///
        /// Six statements in one launch — q norm, q rope, k norm, k rope, v
        /// norm and the paged KV write — and the only value that survives to
        /// a result is q. Everything else lands in the cache, which is what
        /// the contract's `sink: Some("kv.pages")` says.
        ///
        /// `<<<dim3(num_rows, num_q_heads + num_kv_heads), 256>>>`: one block
        /// per (row, head), q heads first and kv heads after, which is what
        /// `head_idx < num_q_heads` reads. `BLOCK` is 256 and IS the block
        /// width here, unlike the warp form below.
        ///
        /// Eighteen parameters and no `rope_table`, no `w_page`, no `w_off`
        /// and no `win`: the prefill form derives its destination from the
        /// CSR and takes its angles from `theta`.
        fn packed = "attn::device::qkv_packed_qk_norm_rope_vnorm_write_kv" (
            packed: *const bf16,
            q_out: *mut bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            q_weight: *const bf16,
            k_weight: *const bf16,
            positions: *const i32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            row_valid: *const u8,
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            page_size: i32,
            hnd_layout: bool,
            theta: f32,
            eps: f32,
        ) {
            "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16" => "device::i32(256)",
        }

        /// `qkv_fused.cuh:115` — the BLOCK decode form.
        ///
        /// One block per request row, `BLOCK = 128` threads, a shared
        /// reduction per head. The fall-through when the warp form has no
        /// instantiation for the fire's head width.
        ///
        /// Three rows for two instantiations: the base and `#norope` are the
        /// same text. See the module doc.
        fn block = "attn::device::qkv_decode_qk_norm_rope_write_kv" (
            packed: *const bf16,
            q_out: *mut bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            q_weight: *const bf16,
            k_weight: *const bf16,
            positions: *const i32,
            rope_table: *const f32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            w_page: *const u32,
            w_off: *const u32,
            row_valid: *const u8,
            win: *const u32,
            num_q_heads: i32,
            num_kv_heads: i32,
            head_dim: i32,
            page_size: i32,
            hnd_layout: bool,
            theta: f32,
            eps: f32,
        ) {
            "attn::qkv_decode_qk_norm_rope_write_kv" => "device::i32(128), false",
            "attn::qkv_decode_qk_norm_rope_write_kv#rope" => "device::i32(128), true",
            "attn::qkv_decode_qk_norm_rope_write_kv#norope" => "device::i32(128), false",
        }

        /// `qkv_fused.cuh:252` — the WARP decode form.
        ///
        /// One warp per head instead of one block per row, which is why it
        /// takes `num_requests` where the block form takes `head_dim`: the
        /// head width is the TEMPLATE argument and the request count is what
        /// the grid stride bounds itself on.
        ///
        /// Seven rows for six instantiations, at three head widths. The two
        /// `d64` and two `d256` pairs are the arms
        /// `families::attn::QKV_FUSED_ROWS` called *"the two expansions this
        /// unit was missing"* — before them, `head_dim == 64` and
        /// `head_dim == 256` reached a launch no row named.
        fn warp = "attn::device::qkv_decode_qk_norm_rope_write_kv_warp" (
            packed: *const bf16,
            q_out: *mut bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            q_weight: *const bf16,
            k_weight: *const bf16,
            positions: *const i32,
            rope_table: *const f32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            w_page: *const u32,
            w_off: *const u32,
            row_valid: *const u8,
            win: *const u32,
            num_requests: i32,
            num_q_heads: i32,
            num_kv_heads: i32,
            page_size: i32,
            hnd_layout: bool,
            theta: f32,
            eps: f32,
        ) {
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128" => "device::i32(128), false",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#rope" => "device::i32(128), true",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#norope" => "device::i32(128), false",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#rope" => "device::i32(64), true",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#norope" => "device::i32(64), false",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#rope" => "device::i32(256), true",
            "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#norope" => "device::i32(256), false",
        }
    }

    /// `BLOCK` for the packed form, and it IS the block width.
    ///
    /// The template argument and `blockDim.x` are the same number here, which
    /// is not true of [`warp`] one `fn` up — so this constant may be read as
    /// either and the warp form's may not.
    ///
    /// The launcher this and [`qkv_packed_qk_norm_rope_vnorm_write_kv_bf16`]
    /// were transcribed from, verbatim, because the `.cu` is deleted and this
    /// Rust IS the citation now:
    ///
    /// ```text
    /// constexpr int BLOCK = 256;
    /// dim3 grid(num_rows, num_q_heads + num_kv_heads);
    /// ```
    ///
    /// `tests/launch_rules.rs`' `rows_packed_heads` pins both lines and had
    /// been failing on them: they were pinned against `fire/qkv_fused.rs`,
    /// which is the DECODE dispatch and never held the prefill launcher. The
    /// pin was right and pointed at the wrong file.
    pub const PACKED_BLOCK: u32 = 256;

    /// `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` — the fused
    /// prefill epilogue, over a packed `[N, q + 2·kv]` row.
    ///
    /// # The grid is two-dimensional and the second axis is the HEAD, not a
    /// # tile
    ///
    /// `dim3(num_rows, num_q_heads + num_kv_heads)`. The kernel reads
    /// `head_idx < num_q_heads` to decide whether it is a q head or a kv one,
    /// so the two banks are ONE axis and their order is load-bearing. A grid
    /// that put kv heads first would norm the wrong weights and write the
    /// wrong pages, with every pointer valid.
    ///
    /// # Every refusal is hoisted, and there is only one launch to hoist past
    ///
    /// One kernel, so the `layout` rule is satisfied trivially — but the
    /// refusals are still stated before it rather than folded into the grid,
    /// because a zero extent here is a silently empty launch and not a fault.
    ///
    /// # Safety
    ///
    /// Every pointer must be a device address valid for the fire, `packed`
    /// must hold `num_rows` rows of `(num_q_heads + 2·num_kv_heads)·head_dim`
    /// elements, and the page arrays must describe the layer the cache
    /// pointers came from.
    #[cfg(feature = "_cuda")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
        packed: *const bf16,
        q_out: *mut bf16,
        k_pages: *mut bf16,
        v_pages: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        row_valid: *const u8,
        num_rows: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        hnd_layout: bool,
        theta: f32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> Fired {
        if num_rows <= 0 {
            return Fired::Declined(Refusal::Empty { what: "rows" });
        }
        if head_dim <= 0 {
            return Fired::Declined(Refusal::Empty { what: "head_dim" });
        }
        if num_q_heads <= 0 {
            return Fired::Declined(Refusal::Empty { what: "q heads" });
        }
        if num_kv_heads <= 0 {
            return Fired::Declined(Refusal::Empty { what: "kv heads" });
        }
        if page_size <= 0 {
            return Fired::Declined(Refusal::Empty { what: "page size" });
        }
        let heads = num_q_heads.unsigned_abs() + num_kv_heads.unsigned_abs();
        unsafe {
            raw::packed(
                "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
                Launch {
                    grid: [num_rows.unsigned_abs(), heads, 1],
                    block: [PACKED_BLOCK, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                },
                packed,
                q_out,
                k_pages,
                v_pages,
                q_weight,
                k_weight,
                positions,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                row_valid,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                hnd_layout,
                theta,
                eps,
                stream,
            );
        }
        Fired::Launched
    }
    /// `qkv_fused.cu:51` — `constexpr int WARP_BLOCK = 256;`, and it is NOT
    /// the unit width.
    ///
    /// The warp form assigns one WARP per `(request, head)` unit, so 256
    /// threads carry eight units and not 256. See [`WARPS_PER_BLOCK`].
    pub const WARP_BLOCK: u32 = 256;

    /// `qkv_fused.cu:105` — `constexpr int BLOCK = 128;`, the DECODE block.
    ///
    /// Named apart from [`PACKED_BLOCK`] because the two forms in this unit
    /// are 128 and 256 wide and a shared name would make the wrong one a
    /// typo away.
    pub const DECODE_BLOCK: u32 = 128;

    /// Warps per block: `WARP_BLOCK / 32`.
    ///
    /// The divisor is what makes the warp grid a grid of UNITS rather than
    /// of threads. Getting it wrong by the factor 32 is a grid eight times
    /// too large, which does not fault — the kernel bounds itself on
    /// `total_units` — so it is named rather than inlined.
    const WARPS_PER_BLOCK: u32 = WARP_BLOCK / 32;

    /// The six warp instantiations, by `(head_dim, rope_table)`.
    ///
    /// `None` for a head width the warp form was not compiled for, which is
    /// the fallthrough to the block form. `head_dim` is the TEMPLATE
    /// argument here, which is why it does not appear in [`raw::warp`]'s
    /// argument list.
    #[cfg(feature = "_cuda")]
    fn warp_symbol(head_dim: i32, rope_table: bool) -> Option<&'static str> {
        Some(match (head_dim, rope_table) {
            (64, true) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#rope",
            (64, false) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#norope",
            (128, true) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#rope",
            (128, false) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#norope",
            (256, true) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#rope",
            (256, false) => "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#norope",
            _ => return None,
        })
    }

    /// The block form's two arms.
    ///
    /// The unit declares a third, unsuffixed row naming the same
    /// instantiation as `#norope`. It is the base of a `Specialisation` that
    /// fn-world does not have, and nothing chooses it — see the `fn block`
    /// doc above for why it is carried anyway.
    #[cfg(feature = "_cuda")]
    fn block_symbol(rope_table: bool) -> &'static str {
        if rope_table {
            "attn::qkv_decode_qk_norm_rope_write_kv#rope"
        } else {
            "attn::qkv_decode_qk_norm_rope_write_kv#norope"
        }
    }

    /// `attn/qkv_fused.cu:31` — `qkv_decode_fused_dispatch`, the `static` one.
    ///
    /// The whole host program: pick the form on `head_dim`, pick the arm on
    /// `rope_table`, compute the grid, fire ONCE.
    ///
    /// ```text
    /// :50    if (num_requests == 0) return;
    /// :51    constexpr int WARP_BLOCK = 256;
    /// :52    const int total_units = num_requests * (num_q_heads + num_kv_heads);
    /// :53    dim3 warp_grid((total_units + (WARP_BLOCK / 32) - 1) / (WARP_BLOCK / 32));
    /// :57-58 ...write_kv_warp<(HEAD_DIM_VALUE), true ><<<warp_grid, WARP_BLOCK, 0, stream>>>
    /// :70-71 ...write_kv_warp<(HEAD_DIM_VALUE), false><<<warp_grid, WARP_BLOCK, 0, stream>>>
    /// :105   constexpr int BLOCK = 128;
    /// :106   dim3 grid(num_requests, num_q_heads + num_kv_heads);
    /// :108   ...write_kv<BLOCK, true ><<<grid, BLOCK, 0, stream>>>
    /// :134   ...write_kv<BLOCK, false><<<grid, BLOCK, 0, stream>>>
    /// ```
    ///
    /// The C++ wrote the two rope arms once per head width through
    /// `LAUNCH_QKV_DECODE_POST_WARP`, a `do { } while (0)` macro expanded
    /// three times. This picks the symbol instead, which is the same
    /// three-by-two table with the duplication removed and the argument list
    /// written once.
    ///
    /// # `num_requests` is an operand of the warp form and not of the block
    /// # form
    ///
    /// Not an inconsistency — the grid's. The block form gets its request
    /// index from `blockIdx.x` and needs no count; the warp form flattens
    /// `(request, head)` onto one axis, recovers `r = unit / total_qk_heads`
    /// at `qkv_fused.cuh:267`, and has to be told where the units stop. The
    /// two argument lists differ in exactly that one cell and in `head_dim`,
    /// which the warp form carries as a TEMPLATE argument and so must not
    /// also be passed.
    ///
    /// # Every refusal is hoisted, and one of them is not a transcription
    ///
    /// Six, all before the single launch. Five are zero extents the C++
    /// either tested or could not reach. The sixth is `q_out`:
    ///
    /// ```text
    /// producer   driver-cuda bind/launch.rs:3248
    ///            q_pin.and_then(..).unwrap_or(core::ptr::null_mut())
    /// row        table/attn.rs   q_out: BufMut <- Source::Attn("q_out")
    /// device     qkv_fused.cuh:177  dst = q_out + (r * num_q_heads + h) * head_dim;
    /// ```
    ///
    /// The producer CAN write null; the row states a plain `Source::Attn`,
    /// which asserts presence and installs no test; and `q_out` is the one
    /// pointer in the list the kernels do NOT test. A fire with no query pin
    /// was an unconditional device store through a null base. It is
    /// [`Refusal::Absent`] here, before anything goes to the device.
    ///
    /// The five the kernels DO test — `rope_table`, `w_page`, `w_off`,
    /// `row_valid` and `win` — are passed through null and are not refusals.
    /// `w_page`/`w_off` null is the CSR path (`qkv_fused.cuh:182`), which is
    /// why a caller holding `Cx` must recover their null rather than
    /// propagate `Cx::w_page_d`'s in-query refusal with `?`.
    ///
    /// # Safety
    ///
    /// Every pointer is a device address the caller keeps live across the
    /// launch; the five named above may be null. `stream` is the caller's.
    #[cfg(feature = "_cuda")]
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    pub unsafe fn qkv_decode_fused_dispatch(
        packed: *const bf16,
        q_out: *mut bf16,
        k_pages: *mut bf16,
        v_pages: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        rope_table: *const f32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        w_page: *const u32,
        w_off: *const u32,
        row_valid: *const u8,
        win: *const u32,
        num_requests: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        hnd_layout: bool,
        theta: f32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> Fired {
        // `qkv_fused.cu:50`, widened from `== 0`: the C++ could not reach a
        // negative count, and a negative one here would underflow the grid.
        if num_requests <= 0 {
            return Fired::Declined(Refusal::Empty { what: "requests" });
        }
        // NOT A TRANSCRIPTION. See the section above.
        if q_out.is_null() {
            return Fired::Declined(Refusal::Absent { what: "q_out" });
        }
        if num_q_heads <= 0 {
            return Fired::Declined(Refusal::Empty { what: "q heads" });
        }
        if num_kv_heads <= 0 {
            return Fired::Declined(Refusal::Empty { what: "kv heads" });
        }
        if head_dim <= 0 {
            return Fired::Declined(Refusal::Empty { what: "head_dim" });
        }
        if page_size <= 0 {
            return Fired::Declined(Refusal::Empty { what: "page size" });
        }

        let use_rope_table = !rope_table.is_null();
        let heads = num_q_heads.unsigned_abs() + num_kv_heads.unsigned_abs();

        if let Some(symbol) = warp_symbol(head_dim, use_rope_table) {
            // `:52-53` — one WARP per `(request, head)` unit,
            // `WARPS_PER_BLOCK` units per block. `head_dim` does NOT appear
            // below: it is the template argument the symbol names.
            let units = num_requests.unsigned_abs().saturating_mul(heads);
            unsafe {
                raw::warp(
                    symbol,
                    Launch {
                        grid: [units.div_ceil(WARPS_PER_BLOCK), 1, 1],
                        block: [WARP_BLOCK, 1, 1],
                        smem: 0,
                        smem_opt_in: false,
                    },
                    packed,
                    q_out,
                    k_pages,
                    v_pages,
                    q_weight,
                    k_weight,
                    positions,
                    rope_table,
                    kv_page_indices,
                    kv_page_indptr,
                    kv_last_page_lens,
                    w_page,
                    w_off,
                    row_valid,
                    win,
                    num_requests,
                    num_q_heads,
                    num_kv_heads,
                    page_size,
                    hnd_layout,
                    theta,
                    eps,
                    stream,
                );
            }
            return Fired::Launched;
        }

        // `:105-106` — the fallthrough. One block per `(request, head)`, and
        // the kernel is TOLD `head_dim` because no template argument carries
        // it here.
        unsafe {
            raw::block(
                block_symbol(use_rope_table),
                Launch {
                    grid: [num_requests.unsigned_abs(), heads, 1],
                    block: [DECODE_BLOCK, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                },
                packed,
                q_out,
                k_pages,
                v_pages,
                q_weight,
                k_weight,
                positions,
                rope_table,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                w_page,
                w_off,
                row_valid,
                win,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                hnd_layout,
                theta,
                eps,
                stream,
            );
        }
        Fired::Launched
    }

    /// `attn/qkv_fused.cu:160` — `qkv_decode_qk_norm_rope_write_kv_bf16`.
    ///
    /// The public launcher: [`qkv_decode_fused_dispatch`] with `win = null`.
    ///
    /// ```text
    /// :183   qkv_decode_fused_dispatch(
    /// :189       w_page, w_off, row_valid, /*win=*/nullptr,
    /// ```
    ///
    /// The `_devwin` twin that passed a real window was deleted in an earlier
    /// pass, so `win` survives on the shared dispatch with exactly one caller
    /// passing exactly one value. It is kept on
    /// [`qkv_decode_fused_dispatch`] rather than folded away, because the
    /// kernels read it per row and a future peel-aware caller is a caller of
    /// that function and not a new kernel.
    ///
    /// # Safety
    ///
    /// [`qkv_decode_fused_dispatch`]'s.
    #[cfg(feature = "_cuda")]
    #[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
    pub unsafe fn qkv_decode_qk_norm_rope_write_kv_bf16(
        packed: *const bf16,
        q_out: *mut bf16,
        k_pages: *mut bf16,
        v_pages: *mut bf16,
        q_weight: *const bf16,
        k_weight: *const bf16,
        positions: *const i32,
        rope_table: *const f32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        w_page: *const u32,
        w_off: *const u32,
        row_valid: *const u8,
        num_requests: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        page_size: i32,
        hnd_layout: bool,
        theta: f32,
        eps: f32,
        stream: *mut core::ffi::c_void,
    ) -> Fired {
        // SAFETY: the caller's contract, forwarded; `win` is null here.
        unsafe {
            qkv_decode_fused_dispatch(
                packed,
                q_out,
                k_pages,
                v_pages,
                q_weight,
                k_weight,
                positions,
                rope_table,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                w_page,
                w_off,
                row_valid,
                core::ptr::null(),
                num_requests,
                num_q_heads,
                num_kv_heads,
                head_dim,
                page_size,
                hnd_layout,
                theta,
                eps,
                stream,
            )
        }
    }
}

/// `attn/dsv4_compress.cuh` — deepseek_v4's SECOND KV cache, and the eleven
/// `__global__`s that build, attend and merge it.
///
/// The model attends a fine-grained cache and a compressed one holding one
/// entry per `ratio` tokens, and merges the two outputs by their
/// log-sum-exps. That merge is exact algebra and not an approximation — the
/// same one FlashInfer's KV-split uses — which is why [`combine_attn_outputs`]
/// is a kernel here rather than a fixup.
///
/// # Ten rows, and the six that no longer need a `Specialisation` argument
///
/// Transcribed one for one from `families::attn::DSV4_COMPRESS`, which stated
/// all ten before this pass and stated them completely. Six carry `bf16`, two
/// carry `device::i32` — the DEFAULT the `.cuh` gives `T` in
/// `template <class T = device::i32>`, spelled out because
/// `DeviceKernel::instantiation()` always emits an argument list and naming
/// the default is what keeps the JIT's object and the archive's the same one
/// rather than two that happen to agree — and one is `PLAIN`.
///
/// **`compressed_attn_paged` is the `PLAIN` one and it is the only unadorned
/// `__global__` in this crate's fn-world.** It takes no template parameter at
/// all, so there is no `<T>` to write; `DeviceKernel::PLAIN` is the `elem`
/// that says so, and `qualify` leaves it alone.
///
/// # THE SYMBOL SPLIT, and why fn-world does not need it
///
/// Four of these rows are spelled `…_dev`: `dsv4_boundary_meta_decode_dev`,
/// `dsv4_boundary_meta_paged_dev`, `compressed_attn_paged_dev` and
/// `combine_attn_outputs_dev`. §60.6 split them because a symbol that was
/// both a table row and a unit row was unit-hosted and therefore unwalkable
/// (§52.11), and the launcher could not be taken over while the two names
/// were one.
///
/// **The suffixes are CARRIED VERBATIM and must be**, for the same reason
/// `qkv_fused`'s `#rope` is: `driver-cuda/src/fire/dsv4_compress.rs` fires
/// them by name — `COMBINE_DEVICE`, `META_DECODE_DEVICE`,
/// `META_PAGED_DEVICE` and `COMPRESSED_PAGED_DEVICE` at `:59`-`:68` — through
/// `hand::fire`, which resolves `unit_of(symbol)` and binds against that
/// row's own `sig`. Renaming one is a panic at the first deepseek_v4 fire on
/// a machine with a GPU, and nothing on a machine without one would say so.
///
/// That the split is no longer NEEDED is a separate statement from that it is
/// no longer THERE. Collapsing it is four edits in a file this pass does not
/// own, for no gain, and the four table rows it would collide with are still
/// in `table/attn.rs`.
///
/// # What crosses and what stays
///
/// Two of the six table rows this root serves cross here:
/// `attn::dsv4_compress_gather_paged_bf16` and
/// `attn::dsv4_store_comp_entries_bf16`, both UNBOUND — see their contracts
/// below for what a bind would need and why the trace cannot supply it.
///
/// The other four stay, and they stay for `qkv_decode_fused`'s reason
/// exactly: their host programs are written, in Rust, in
/// `driver-cuda/src/fire/dsv4_compress.rs`, and served through
/// `bind::service`. A row served by a driver host program is not a row
/// waiting to cross; it is a row whose fn-world half lives one crate up.
/// **This unit is what those four fire**, and moving it here changed nothing
/// for them: `families/mod.rs` concatenates `crate::x::attn::UNITS`, so
/// `unit_of` answers the same four symbols it always did.
///
/// # The two stale sentences in the root, still stale
///
/// `dsv4_compress.cuh:50-52` says *"No ported rule computes a shared-memory
/// size from an operand width"*; `PagedScores` and `PagedScoresDecode` both
/// do. `:76-78` says *"`compressed_attn` and `compressed_attn_paged` are
/// blocked by their HOST half"*; true of the first, whose launcher builds a
/// `CompressedAttnParams[R]` on the host and `cudaMallocAsync`s it, and false
/// of the second, whose host half is a null guard, a grid, a smem and one
/// launch. `families/attn.rs` corrected both and
/// `driver-cuda/src/fire/dsv4_compress.rs` restates them; this is the third
/// place, and the reason there are three is that none of them is the file.
///
/// **The header's whole "which launchers became rows, and which did not"
/// section is now answered by its own text being here.** Every reason it
/// gives is a reason a `LaunchRule` could not state a geometry, and a host
/// `fn` states geometries. The two it named as structurally blocked — the
/// boundary-meta pair, "blocked TWICE", once for having no element type and
/// once for a 128-wide block where `Elementwise` is 256 — are two of the four
/// the driver already launches with a `Launch` it wrote itself.
pub mod dsv4_compress {
    use super::bf16;
    #[cfg(feature = "_cuda")]
    use super::{Fired, Launch, Refusal};

    unit! {
        unit DSV4_COMPRESS = "attn/dsv4_compress",
            text = include_str!("../../csrc/src/attn/dsv4_compress.cuh"),
            file = "attn/dsv4_compress.cuh";

        /// `:105` — the mean over each window of `ratio` input tokens.
        ///
        /// `n` is the INPUT token count and the grid covers
        /// `n / ratio * dim`, so the extent a caller sizes the launch from
        /// and the extent the kernel is told differ by the ratio. Both
        /// survive: the launch is sized off the result, the kernel divides
        /// its own index by `dim`.
        fn average_pool = "attn::device::average_pool"(
            input: *const bf16,
            output: *mut bf16,
            n: i32,
            dim: i32,
            ratio: i32,
        ) {
            "attn::average_pool_bf16" => "device::bf16",
        }

        /// `:130` — the absolute position table, added in place.
        ///
        /// `_f32` in the symbol names the TABLE's format and not the data's:
        /// `ape` is fp32 and `data` is the row type's. The launcher was named
        /// for the table and the symbol keeps that name, because a symbol
        /// that changes spelling during a migration is a symbol two tables
        /// disagree about.
        fn add_ape = "attn::device::add_ape"(
            data: *mut bf16,
            ape: *const f32,
            n_compressed: i32,
            dim: i32,
            ratio: i32,
        ) {
            "attn::add_ape_f32" => "device::bf16",
        }

        /// `:154` — a per-dimension softmax over `ratio` gate scores, then
        /// the weighted sum of the values under it.
        fn gated_softmax_pool = "attn::device::gated_softmax_pool"(
            kv: *const bf16,
            score: *const bf16,
            output: *mut bf16,
            n: i32,
            dim: i32,
            ratio: i32,
        ) {
            "attn::gated_softmax_pool_bf16" => "device::bf16",
        }

        /// The unpaged gather — one block per compressed entry, striding its
        /// own row.
        ///
        /// Carried, and it has no live caller: `fire/dsv4_compress.rs`'
        /// header records the unpaged five as *"a closed cycle of dead
        /// callers"*. It stays because the family declared it and a
        /// transcription that drops a row is a transcription nobody can
        /// check against the thing it came from. Its cost is one NVRTC
        /// instantiation.
        fn dsv4_compress_gather = "attn::device::dsv4_compress_gather"(
            kv_proj: *const bf16,
            score_proj: *const bf16,
            ape: *const f32,
            boundary_tok: *const i32,
            boundary_pos: *const i32,
            window_lo: *const i32,
            out: *mut bf16,
            head_dim: i32,
            ratio: i32,
            coff: i32,
        ) {
            "attn::dsv4_compress_gather_bf16" => "device::bf16",
        }

        /// `:578` — the paged gather, and the first of the two the planner
        /// actually names.
        ///
        /// `ape` is nullable: the kernel tests `ape != nullptr` twice, once
        /// per pass over the window.
        fn dsv4_compress_gather_paged = "attn::device::dsv4_compress_gather_paged"(
            state_kv: *const bf16,
            state_score: *const bf16,
            ape: *const f32,
            boundary_pos: *const i32,
            boundary_req: *const i32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            out: *mut bf16,
            head_dim: i32,
            ratio: i32,
            coff: i32,
            page_size: i32,
        ) {
            "attn::dsv4_compress_gather_paged_bf16" => "device::bf16",
        }

        /// `:648` — commit those entries to the compressed cache, each at its
        /// own boundary token's slot.
        ///
        /// `boundary_pos[c] < 0` marks a padding row and the kernel returns
        /// on it, which is what makes a CUDA-graph-safe decode able to launch
        /// a fixed number of blocks whatever the batch does.
        fn dsv4_store_comp_entries = "attn::device::dsv4_store_comp_entries"(
            entries: *const bf16,
            comp_kv_pages: *mut bf16,
            boundary_pos: *const i32,
            boundary_req: *const i32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            head_dim: i32,
            page_size: i32,
        ) {
            "attn::dsv4_store_comp_entries_bf16" => "device::bf16",
        }

        /// `:530` — which decode rows close a compression window.
        ///
        /// `row_valid` is nullable and absence means *every row is valid*.
        /// Fired by `driver-cuda/src/fire/dsv4_compress.rs:220`.
        fn dsv4_boundary_meta_decode = "attn::device::dsv4_boundary_meta_decode"(
            positions: *const i32,
            out_pos: *mut i32,
            out_req: *mut i32,
            out_rope: *mut i32,
            n: i32,
            ratio: i32,
            row_valid: *const u8,
        ) {
            "attn::dsv4_boundary_meta_decode_dev" => "device::i32",
        }

        /// `:544` — the prefill form.
        ///
        /// One line different from its decode twin: the request index comes
        /// from a binary search over `qo_indptr` instead of being shortcut to
        /// the token index. A SECOND kernel rather than a wider first one,
        /// because the decode form is what a CUDA-graph capture calls and
        /// giving it two more parameters would make every capture carry a
        /// `qo_indptr` it does not read.
        fn dsv4_boundary_meta_paged = "attn::device::dsv4_boundary_meta_paged"(
            positions: *const i32,
            qo_indptr: *const u32,
            out_pos: *mut i32,
            out_req: *mut i32,
            out_rope: *mut i32,
            n: i32,
            num_requests: i32,
            ratio: i32,
            row_valid: *const u8,
        ) {
            "attn::dsv4_boundary_meta_paged_dev" => "device::i32",
        }

        /// `:666` — the attention itself, over the compressed cache.
        ///
        /// The `PLAIN` row. `grid(total_tokens, num_q_heads)` at 128 with
        /// `(head_dim + 128) * sizeof(float)` of dynamic shared memory —
        /// eleven lines of host half, which is why this one was never
        /// blocked by anything but the sentence its sibling earned.
        fn compressed_attn_paged = "attn::device::compressed_attn_paged"(
            q: *const bf16,
            comp_kv_pages: *const bf16,
            o: *mut bf16,
            lse_out: *mut f32,
            positions: *const i32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            req_of_token: *const i32,
            num_q_heads: i32,
            head_dim: i32,
            ratio: i32,
            page_size: i32,
            scale: f32,
        ) {
            "attn::compressed_attn_paged_dev" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:216` — the merge, by log-sum-exp.
        ///
        /// If `lse2` is `-inf` the compressed half had no entries and `o1`
        /// passes through unchanged, which is the empty case handled ON THE
        /// DEVICE rather than by a host refusal — §5.1's rule for a geometry
        /// that depends on a device-side output.
        fn combine_attn_outputs = "attn::device::combine_attn_outputs"(
            o1: *const bf16,
            lse1: *const f32,
            o2: *const bf16,
            lse2: *const f32,
            o_out: *mut bf16,
            lse_out: *mut f32,
            num_heads: i32,
            head_dim: i32,
        ) {
            "attn::combine_attn_outputs_dev" => "device::bf16",
        }
    }

    /// `route_rows`' warp rounding, and the clamp that makes it legal at any
    /// width.
    ///
    /// `runtime/launch.rs:1044`. One block per compressed entry, the block as
    /// wide as the row rounded up to a warp and capped at 1024; above the cap
    /// the kernel's `for (int d = threadIdx.x; d < head_dim; d += blockDim.x)`
    /// covers the row in several passes. **The cap is safe here only because
    /// of that stride** — before it, this rounding would have silently
    /// computed a prefix.
    #[cfg(feature = "_cuda")]
    #[expect(clippy::cast_sign_loss, reason = "both are guarded positive by every caller")]
    fn route_rows(rows: i32, width: i32) -> Launch {
        let (rows, width) = (rows as u32, width as u32);
        Launch::per_row(rows, width.div_ceil(32).max(1).saturating_mul(32).min(1024))
    }

    /// Build one compressed entry per boundary token.
    ///
    /// # Safety
    ///
    /// Every pointer addresses a live allocation of the extent the kernel
    /// reads, `ape` and nothing else may be null, and the stream outlives the
    /// launch.
    #[cfg(feature = "_cuda")]
    pub unsafe fn dsv4_compress_gather_paged_bf16(
        state_kv: *const bf16,
        state_score: *const bf16,
        ape: *const f32,
        boundary_pos: *const i32,
        boundary_req: *const i32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        out: *mut bf16,
        num_entries: i32,
        head_dim: i32,
        ratio: i32,
        coff: i32,
        page_size: i32,
        stream: *mut core::ffi::c_void,
    ) -> Fired {
        // Every refusal before the one launch, which for a single launch is
        // free and is written this way anyway: the day a second statement
        // joins this body, the order is already right.
        if num_entries <= 0 {
            return Fired::Declined(Refusal::Empty { what: "entries" });
        }
        if head_dim <= 0 {
            return Fired::Declined(Refusal::Empty { what: "head_dim" });
        }
        if ratio <= 0 {
            return Fired::Declined(Refusal::Empty { what: "ratio" });
        }
        if coff <= 0 {
            return Fired::Declined(Refusal::Empty { what: "coff" });
        }
        if page_size <= 0 {
            return Fired::Declined(Refusal::Empty { what: "page_size" });
        }
        // SAFETY: the caller's contract, forwarded unchanged.
        unsafe {
            raw::dsv4_compress_gather_paged(
                "attn::dsv4_compress_gather_paged_bf16",
                route_rows(num_entries, head_dim),
                state_kv,
                state_score,
                ape,
                boundary_pos,
                boundary_req,
                kv_page_indices,
                kv_page_indptr,
                out,
                head_dim,
                ratio,
                coff,
                page_size,
                stream,
            );
        }
        Fired::Launched
    }

    /// Commit those entries to the compressed cache.
    ///
    /// # Safety
    ///
    /// As above; no operand of this one is nullable.
    #[cfg(feature = "_cuda")]
    pub unsafe fn dsv4_store_comp_entries_bf16(
        entries: *const bf16,
        comp_kv_pages: *mut bf16,
        boundary_pos: *const i32,
        boundary_req: *const i32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        num_entries: i32,
        head_dim: i32,
        page_size: i32,
        stream: *mut core::ffi::c_void,
    ) -> Fired {
        if num_entries <= 0 {
            return Fired::Declined(Refusal::Empty { what: "entries" });
        }
        if head_dim <= 0 {
            return Fired::Declined(Refusal::Empty { what: "head_dim" });
        }
        if page_size <= 0 {
            return Fired::Declined(Refusal::Empty { what: "page_size" });
        }
        // SAFETY: the caller's contract, forwarded unchanged.
        unsafe {
            raw::dsv4_store_comp_entries(
                "attn::dsv4_store_comp_entries_bf16",
                route_rows(num_entries, head_dim),
                entries,
                comp_kv_pages,
                boundary_pos,
                boundary_req,
                kv_page_indices,
                kv_page_indptr,
                head_dim,
                page_size,
                stream,
            );
        }
        Fired::Launched
    }
}

/// `attn/kv_paged.cuh` — the paged KV cache's appenders, its quantised
/// writers and its dequantisers.
///
/// # The five `Specialisation`s are gone and this is why
///
/// `families::attn` carried fifteen rows for five `template <bool
/// HND_LAYOUT>` kernels — a BASE row per kernel and two arms — plus a
/// [`crate::device::Specialisation`] each, five `Take` prefix arrays, and an
/// entry in `families::attn::SPECIALISATIONS`. Twenty artefacts to express
/// `if (hnd_layout)`.
///
/// **`driver-cuda/src/fire/kv_paged.rs` already wrote that `if` in Rust**
/// and already fired the chosen arm BY NAME through `fire::hand::fire` —
/// `"attn::write_kv_bf16#hnd"`/`"…#nhd"`, and the same pair for
/// `copy_kv_cells_bf16`, `write_kv_explicit_bf16_dev` and
/// `write_kv_explicit_bf16_devwin_dev`. So `x/norm.rs:1033`'s
/// *"`Specialisation`s become `if`s"* is not work this crossing does;
/// **it is work the driver port had already done**, and the declaration was
/// the last thing still saying otherwise. Four of those five `if`s are in
/// this module now, unchanged, above `raw::` calls instead of above
/// `hand::fire`; the fifth is `copy_kv_cells_bf16`, which stayed in the
/// driver. This unit hosts the ten arms; `runtime::fire::selects` is asked
/// about none of them.
///
/// # The five base rows went with them, and could not have gone alone
///
/// `attn::write_kv_bf16` and its four siblings are the specialisations'
/// BASES: a sixteenth operand `hnd_layout: Bool` that no kernel takes, so
/// that a fire could hand the flag to `selects` and `TAKE_15` could drop it
/// again. Nothing else ever named them — the driver names arms, and a base
/// has no instantiation to lower to.
///
/// This is the second time in this family that two artefacts were each
/// other's only reason to exist (the first was `QKV_DECODE_BLOCK` and six
/// `quoted()` pins). Deleting the `Specialisation`s alone would have left
/// five rows with a bogus operand; deleting the base rows alone would have
/// broken `Specialisation::agrees`. **They are one edit, and the shape is
/// worth recognising: an artefact whose only citation is the artefact that
/// only exists to cite it.**
///
/// # `write_kv_at_positions` has no caller in the workspace
///
/// Its two arms are carried anyway, exactly as `dsv4_compress_gather` is:
/// they were being instantiated before this crossing and a transcription
/// that silently drops a kernel is a transcription nobody can check against
/// the thing it came from. Cost is two NVRTC instantiations.
///
/// # Half B: the bodies moved, and the four rows have now crossed
///
/// The seven host programs moved out of `driver-cuda/src/fire/kv_paged.rs`
/// and are the `pub unsafe fn`s below. The four `table::attn` rows over this
/// root — `attn::write_kv_to_pages`, `attn::write_kv_explicit_bf16`,
/// `attn::write_kv_explicit_bf16_devwin` and
/// `attn::dequant_kv_cache_layer_to_bf16_active` — **did not cross with
/// them**, because four facts the rows sourced from `AttnCtx` had no `Cx`
/// query. `666fbbeee` landed all four and they cross here.
///
/// ```text
/// row                        the query it was waiting for
/// write_kv_to_pages          cx.first_token()
/// dequant                    cx.num_pages_in_batch()
/// write_kv_explicit_bf16     cx.w_page_d()   cx.w_off_d()
/// write_kv_explicit_devwin   cx.w_page_d()   cx.w_off_d()   win_d — NO
/// ```
///
/// Everything else each row needed was already stated: the layer is
/// `cx.kv_layer()`, the four CSR arrays and `row_valid` and `num_requests`
/// are `cx.plan()`, `k_curr`/`v_curr` are `arg_in(0)`/`arg_in(1)`, and
/// `total_tokens`, `B` and `n_max` are all `cx.rows().count` — the last of
/// those because the devwin row is `whole = true`, so `n_max` is the fire's
/// full lane count and no windowed statement can reach it.
///
/// **The four had a PRODUCER before they had a query**, and that is the
/// distinction worth keeping: `first_token` at `bind/mod.rs:1401`,
/// `num_pages_in_batch` at `:1399`, `w_page_d` at `:1403`, `w_off_d` at
/// `:1405`, all carried by `AttnCtx` since before fn-world existed. So the
/// ask was four `query!` lines and four field reads rather than a feature —
/// which is what separates it from `Cx::mla_layer`, whose refusal is that
/// nothing fills it, and from `Cx::mla_layer`'s cache provisioning, which is
/// a feature and not a seam.
///
/// **`w_page_d` and `w_off_d` are null-checked in the query.** A fire that
/// appends no KV carries a null there, and a body that took the pointer
/// anyway would index it — so absence is `None` rather than a null that
/// looks like an address. That check IS the row's old
/// `Source::AttnNonZero`, moved from the emitter to the fact, which is where
/// it can be read by something other than a code generator.
///
/// **`Cx::plan()` is exercised here for the first time in the migration.**
/// §5.1 named `attn` and `ssm` as where `Facts::plan()` and `Facts::slab()`
/// would first be exercised and therefore where they were most likely to be
/// wrong. It came out right: `bind/facts.rs:505` fills all six fields by
/// direct copy from `AttnCtx`, and one query replaced six `Source::Attn`
/// cells because those six describe one thing and are read together.
///
/// # Three bind, one is a `none:` arm, and the difference was measured
///
/// `WRITE_KV_EXPLICIT_DEVWIN` crosses as a `contract!` with a `none:` arm.
/// **`x/gemm.rs:1145`'s overlap rule says that is normally the dangerous
/// move** — a `none:` arm mints an `Entry`, an `Entry` with no arm is
/// `Route::Unbound`, and `Route::Unbound` shadowing a working generated
/// dispatch refuses the model at load. It is safe here for two measured
/// reasons and would not be safe on the strength of either alone:
///
/// 1. The row stated `Source::Unbound` on **all nine** operands, and
///    `abi.rs:810` skips a row with any `Unbound` operand WHOLE. No dispatch
///    arm has ever been generated for this symbol. There is nothing to
///    shadow.
/// 2. `dsl::cuda::write_kv_explicit_devwin` has **zero callers** in the
///    workspace — `model-compiler/src/dsl.rs:3594` states the symbol and
///    nothing in `crates/model/src` reaches the builder.
///
/// `win_d` is **not** a fourth query of the same kind, and that is why this
/// one could not simply wait for a fifth `query!`: `AttnCtx` has no window
/// array at all. `win_d` is missing its FILL, not merely its query, and a
/// query over a field that does not exist is the `Cx::mla_layer` shape.
///
/// # Why the move, in its final form
///
/// **A driver op is a symbol whose body needs a driver RESOURCE** — a
/// cuBLAS handle, an NCCL communicator, a pool, an allocator. `x::gemm`'s
/// twelve are driver ops because `cublasLtMatmul` is on the far side of a
/// seam no `Cx` query can cross. These seven need no resource: they need a
/// KV layer's seventeen facts, and `Cx::kv_layer()` states all seventeen.
///
/// Half A said instead that they were blocked because `kernels-cuda-new`
/// cannot call `driver-cuda`. **That sentence is true and it is not the
/// reason** — the dependency runs the other way, and two of these bodies
/// were already calling `x::layout::envelope_*` from the middle of
/// themselves. The correction is kept in place rather than replaced,
/// because a true statement standing in for the reason, in a document
/// written to record the reason, is the failure worth naming.
///
/// # `WriteKvNative` needed no floor change, and the measurement is why
///
/// The four `Launched`/`Declined` enums did not move. All ten call sites —
/// four shim entries and four preludes in `bind/service.rs`, plus two
/// internal — consumed the return with `let _ =`. **No reader distinguished
/// `Launched` from `Declined`, and none inspected a payload.** `Fired` is
/// `#[must_use]` and says strictly more than anything read, so the rich
/// enums were a distinction with no consumer. No third `Fired` variant, no
/// floor edit.
///
/// The declines gained information rather than losing it. `kv_paged.cu:384`
/// folded two reasons into one `if (a || b) return;` because `void` could
/// carry neither; `Refusal` carries both, so the dequant now says
/// `Absent { "quantised pages on a bf16 layer" }` where it used to say the
/// same thing as `Empty { "active pages" }`. **A port that only preserves is
/// a transcription.**
///
/// # Two things the move surfaced
///
/// **`dequant_kv_cache_layer_to_bf16_active` is a subroutine before it is a
/// trace symbol.** Four *other* host programs call it as a prelude, at
/// `bind/service.rs`'s two FA2 decode entries and two prefill entries. A
/// `bind!` arm is reachable only from a trace, so the moved body is a
/// `pub fn` those four call and the arm is a call to it — which is
/// `x::layout::envelope_*`'s arrangement again. Those four preludes
/// SURVIVED the shim-entry deletion for exactly that reason.
///
/// **`execution.rs`'s `Walk` for `attn::write_kv_explicit_bf16_devwin`
/// stated that its row was fully sourced. Its row stated `Source::Unbound`
/// on all nine operands.** `abi.rs:810` skips such a row whole, so
/// `emit_rust_dispatch` never wrote a dispatch arm for it, so its
/// `RUST_SERVED` entry and its `bind::service` shim were never reachable.
/// The claim is true of the *sibling* `write_kv_explicit` row, which was
/// fully sourced, and was written for this one. That is the fifth artefact
/// in this family that nothing re-derives, after `device.rs:991`'s hold,
/// `DSV4_COMPRESS_SIGS[4]`'s sources, `assert_eq!(checked, 14)` and
/// `RUST_SERVED`'s "all four unsourced". It was corrected in place at the
/// `Walk` rather than deleted, and the `Walk` is now retracted with the
/// finding kept.
///
/// # §52.11, discharged
///
/// All four were `execution::WALKED` symbols, and
/// `execution::tests::a_walk_is_only_a_walk` requires `unit_of(sym)` to be
/// `None` for every one. **No symbol in this unit is any of those four** —
/// that is what §60.6's `_dev` suffix bought, and it is why this unit could
/// exist while they were still walks. **All four `Walk`s are retracted in
/// the same change as these contracts**, which is what §52.11 asks for; the
/// `_dev` suffix stays, because the device rows still need names their host
/// programs do not collide with.
pub mod kv_paged {
    use super::bf16;
    use crate::x::abi::MaybeConst;
    use crate::x::fp8_kind;
    // `core::ffi::c_void` and not `super::c_void`: `x::attn`'s import of it
    // is `#[cfg(feature = "_cuda")]` and a `use super::c_void` here would
    // inherit that obligation without saying so, which is exactly the defect
    // `qkv_fused` shipped and had to have fixed. A `unit!` is compiled in
    // every configuration.
    use core::ffi::c_void;

    // `Launch` is ungated at the top of `x::attn`; `Fired` and `Refusal` are
    // not. So these three name their gate EXPLICITLY and reach the canonical
    // path rather than `super::`, because `use super::X` inherits the
    // parent's cfg obligations and says nothing about it — the defect
    // `qkv_fused` shipped, invisible in any build with the feature on.
    #[cfg(feature = "_cuda")]
    use crate::x::contract::{Fired, Refusal};
    #[cfg(feature = "_cuda")]
    use crate::x::{KvDType, KvLayer, KvScheme};
    #[cfg(feature = "_cuda")]
    use super::Launch;

    // `fp8_kind` STOOD HERE — a local newtype and an `Abi` impl, carried
    // under `X_ABI_FP8_KIND_LOCAL` because `x/abi.rs` had none and is the
    // owner's file. **It is the floor now** (`x::fp8_kind`, `63d8aaebe`) and
    // the workaround is deleted; the two `fn` lines below name the real one.

    unit! {
        unit KV_PAGED = "attn/kv_paged",
            text = include_str!("../../csrc/src/attn/kv_paged.cuh"),
            file = "attn/kv_paged.cuh";

        /// `:153` — the batched append, one block per token.
        ///
        /// `first_token` is ADDED to `blockIdx.x` inside the kernel, so a
        /// launch covers `[first_token, first_token + rows)` and the grid is
        /// the count rather than the end. `row_valid` and `win` are both
        /// nullable and both tested per block.
        ///
        /// `r` and not `R`: the row spelled it `r` and the kernel spells it
        /// `R`, and a Rust parameter cannot be `R` without reading as a type
        /// parameter at every call site. The typecheck translation unit
        /// compares TYPES, not names.
        fn write_kv = "attn::device::write_kv"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            row_valid: MaybeConst<u8>,
            win: MaybeConst<u32>,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
            first_token: i32,
        ) {
            "attn::write_kv_bf16#hnd" => "device::true_type::value",
            "attn::write_kv_bf16#nhd" => "device::false_type::value",
        }

        /// `:223` — the same append with each token's absolute KV position
        /// taken as data rather than derived from the page tables.
        ///
        /// **Measured: no caller.** `write_kv_at_positions` appears in this
        /// workspace exactly four times — the `__global__` at
        /// `kv_paged.cuh:223`, the two arm names below, and this line. No
        /// `<<<>>>`, no `hand::fire`, no `bind::service` entry, no
        /// `model-compiler` builder, no `table::attn` row. It is a kernel
        /// with device text and no host program on either side of the
        /// migration, and it was one before this crossing too.
        ///
        /// Carried rather than dropped, for the reason the module header
        /// gives: a transcription that silently drops a kernel is a
        /// transcription nobody can check against the thing it came from.
        /// Cost is two NVRTC instantiations in a unit that already compiles
        /// eighteen. It goes when someone deletes the `__global__`, and that
        /// deletion is a `.cuh` edit rather than a row.
        fn write_kv_at_positions = "attn::device::write_kv_at_positions"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            positions: *const i32,
            position_delta: i32,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::write_kv_at_positions_bf16#hnd" => "device::true_type::value",
            "attn::write_kv_at_positions_bf16#nhd" => "device::false_type::value",
        }

        /// `:279` — the append that is told each lane's physical page and
        /// offset outright, one block per lane.
        ///
        /// **The `_dev` in the symbol is §60.6's split and not a decoration.**
        /// `table::attn`'s `attn::write_kv_explicit_bf16` is a WALK — a
        /// throw, an empty-extent decline, an instantiation choice and a
        /// conditional second launch — and §52.11 requires a walked symbol to
        /// be hosted by no unit. The kernel and the walk therefore spend two
        /// spellings, and this is the kernel's.
        fn write_kv_explicit = "attn::device::write_kv_explicit"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            w_page: *const u32,
            w_off: *const u32,
            row_valid: MaybeConst<u8>,
            b: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::write_kv_explicit_bf16_dev#hnd" => "device::true_type::value",
            "attn::write_kv_explicit_bf16_dev#nhd" => "device::false_type::value",
        }

        /// `:781` — the explicit append under a device-resident window.
        ///
        /// `win` is a two-word `{start, len}` the DEVICE wrote, and `n_max`
        /// is the launch's upper bound rather than the count: the kernel
        /// reads the real length out of `win` and returns on the blocks past
        /// it. That is the shape §5.1 names — a refusal that depends on a
        /// device-side value is a device-side branch, and this kernel is
        /// already written as one.
        fn write_kv_explicit_devwin = "attn::device::write_kv_explicit_devwin"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            w_page: *const u32,
            w_off: *const u32,
            row_valid: MaybeConst<u8>,
            win: *const u32,
            n_max: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::write_kv_explicit_bf16_devwin_dev#hnd" => "device::true_type::value",
            "attn::write_kv_explicit_bf16_devwin_dev#nhd" => "device::false_type::value",
        }

        /// `:326` — cell-to-cell copy inside the page arena, one block per
        /// cell. The only kernel here that reads and writes the same buffer.
        fn copy_kv_cells = "attn::device::copy_kv_cells"(
            k_pages: *mut bf16,
            v_pages: *mut bf16,
            dst_page: *const u32,
            dst_off: *const u32,
            src_page: *const u32,
            src_off: *const u32,
            n: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::copy_kv_cells_bf16#hnd" => "device::true_type::value",
            "attn::copy_kv_cells_bf16#nhd" => "device::false_type::value",
        }

        /// `:390` — the fp8 append with one scale for the whole tensor.
        ///
        /// `k_pages` is `*mut u8` and the kernel says `__nv_fp8_storage_t*`,
        /// which IS `unsigned char`; `static_assert(is_same_v<>)` resolves
        /// the typedef and the row already said `U8sMut`. The BYTE is not the
        /// format — the format is `fp8_kind`, and it is an operand rather
        /// than a template default because defaulting it would decode an
        /// E5M2 page as E4M3 and produce a numerically plausible wrong
        /// answer.
        fn write_kv_fp8_per_tensor = "attn::device::write_kv_fp8_per_tensor"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut u8,
            v_pages: *mut u8,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
            fp8_kind: fp8_kind,
        ) {
            "attn::write_kv_fp8_per_tensor" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:425` — the per-token-per-head quantised append, `template <bool
        /// UseFp8>`.
        ///
        /// **This is the one where an enum is not a flag.** The two
        /// instantiations are int8 and fp8, and the symbols say so; the
        /// pages are `void*` because the kernel casts to the storage type
        /// under the template. `false_type` is INT8 and `true_type` is FP8 —
        /// the opposite reading of the pair to the four `HND_LAYOUT`
        /// kernels above, where `true` is the layout named second.
        fn write_kv_per_token_head = "attn::device::write_kv_per_token_head"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages_raw: *mut c_void,
            v_pages_raw: *mut c_void,
            k_scales: *mut f32,
            v_scales: *mut f32,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::write_kv_int8_per_token_head" => "device::false_type::value",
            "attn::write_kv_fp8_per_token_head" => "device::true_type::value",
        }

        /// `:562` — the fp4 append, two values to the byte, blocked scales.
        fn write_kv_fp4_block = "attn::device::write_kv_fp4_block"(
            k_curr: *const bf16,
            v_curr: *const bf16,
            k_pages: *mut u8,
            v_pages: *mut u8,
            k_scales: *mut f32,
            v_scales: *mut f32,
            qo_indptr: *const u32,
            kv_page_indices: *const u32,
            kv_page_indptr: *const u32,
            kv_last_page_lens: *const u32,
            r: i32,
            page_size: i32,
            h_kv: i32,
            d: i32,
            block_size: i32,
        ) {
            "attn::write_kv_fp4_block" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:655` — the per-tensor fp8 dequantiser over the active pages.
        ///
        /// `n` is `i64` because it indexes a page arena that is multiple
        /// gigabytes at production page counts, and `page_elems` is the whole
        /// page's element count rather than a head dim: this form needs no
        /// per-head geometry because its scale is the tensor's.
        fn dequant_fp8_pages_active = "attn::device::dequant_fp8_pages_active"(
            k_pages: *const u8,
            v_pages: *const u8,
            k_out: *mut bf16,
            v_out: *mut bf16,
            page_indices: *const u32,
            n: i64,
            page_elems: i32,
            fp8_kind: fp8_kind,
        ) {
            "attn::dequant_fp8_pages_active_bf16" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:678` — the per-token-per-head fp8 dequantiser.
        fn dequant_fp8_per_token_head = "attn::device::dequant_fp8_per_token_head_pages_active"(
            k_pages: *const u8,
            v_pages: *const u8,
            k_scales: *const f32,
            v_scales: *const f32,
            k_out: *mut bf16,
            v_out: *mut bf16,
            page_indices: *const u32,
            n: i64,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::dequant_fp8_per_token_head_pages_active_bf16" => "device::bf16",
        }

        /// `:708` — the same, for int8 pages.
        fn dequant_int8_per_token_head = "attn::device::dequant_int8_per_token_head_pages_active"(
            k_pages: *const i8,
            v_pages: *const i8,
            k_scales: *const f32,
            v_scales: *const f32,
            k_out: *mut bf16,
            v_out: *mut bf16,
            page_indices: *const u32,
            n: i64,
            page_size: i32,
            h_kv: i32,
            d: i32,
        ) {
            "attn::dequant_int8_per_token_head_pages_active_bf16" => "device::bf16",
        }

        /// `:736` — the fp4 dequantiser.
        ///
        /// **`logical_n` and not `n`.** An fp4 page holds two values per
        /// byte, so the count the grid covers is the LOGICAL element count
        /// and every address inside the kernel halves it. The kernel's name
        /// is kept because an operand spelled `n` here would be the one
        /// number in the signature that means something else.
        fn dequant_fp4 = "attn::device::dequant_fp4_pages_active"(
            k_pages: *const u8,
            v_pages: *const u8,
            k_scales: *const f32,
            v_scales: *const f32,
            k_out: *mut bf16,
            v_out: *mut bf16,
            page_indices: *const u32,
            logical_n: i64,
            page_size: i32,
            h_kv: i32,
            d: i32,
            block_size: i32,
        ) {
            "attn::dequant_fp4_pages_active_bf16" => "device::bf16",
        }

        /// `:826` — the window page view, one thread walking every request.
        ///
        /// `LaunchRule::Single` in the family it left: one block, one
        /// thread. `fire/kv_paged.rs` writes that `Launch` by hand and this
        /// row states no rule, which is the same arrangement every unit row
        /// is in.
        fn build_window_page_view = "attn::device::build_window_page_view"(
            src_indices: *const u32,
            src_indptr: *const u32,
            keep_pages: i32,
            dst_indptr: *mut u32,
            dst_indices: *mut u32,
            r: i32,
        ) {
            "attn::build_window_page_view" => crate::device::DeviceKernel::PLAIN,
        }

        /// `:860` — the full split view, one warp.
        ///
        /// Note the parameter order: `src_indices` is LAST, after three
        /// outputs. That is the kernel's and it is kept, because the only
        /// thing a reordering would buy is a signature that reads better and
        /// binds wrong.
        fn build_full_split_view = "attn::device::build_full_split_view"(
            src_indptr: *const u32,
            src_last_page_len: *const u32,
            splits: i32,
            page_size: i32,
            dst_indptr: *mut u32,
            dst_indices: *mut u32,
            dst_last: *mut u32,
            src_indices: *const u32,
        ) {
            "attn::build_full_split_view" => crate::device::DeviceKernel::PLAIN,
        }
    }

    // =====================================================================
    // HALF B — the four host programs, MOVED (not made driver ops).
    //
    // The discriminator, which Half A got wrong and stated the correction
    // for: **a driver op is a symbol whose body needs a driver RESOURCE** —
    // a cuBLAS handle, an NCCL comm, a pool, an allocator. `x::gemm`'s
    // twelve are driver ops because `cublasLtMatmul` lives across a seam no
    // `Cx` can reach. These four need no resource: they need a KV layer's
    // seventeen facts, and `Cx::kv_layer()` states all seventeen.
    //
    // What each body needed and did NOT have before `d391f583c`: eleven of
    // those seventeen. `Cx::KvLayer` carries them now, including the two
    // predicates ANSWERED — `has_envelopes` and `is_native_bf16` — so no
    // body below re-derives either.
    // =====================================================================

    /// `kv_paged.cu`'s `constexpr int BLOCK = 256`, which every launch in
    /// that file used and only one of which was ever written down.
    #[cfg(feature = "_cuda")]
    const BLOCK: u32 = 256;

    /// `::__nv_fp8_interpretation_t`'s two values, by the names the header
    /// gives them.
    ///
    /// Spelled as constants and not as literals at the two call sites
    /// because they are the same two values in both, and the C++ wrote the
    /// ternary twice (`kv_paged.cu:394-396` is the second copy).
    #[cfg(feature = "_cuda")]
    const NV_E4M3: u32 = 0;
    #[cfg(feature = "_cuda")]
    const NV_E5M2: u32 = 1;

    /// The interpretation an fp8 page is written and read under.
    ///
    /// **E4M3 is the fallthrough and not a default.** The C++ tested for
    /// E5M2 and took E4M3 for everything else, including dtypes that are
    /// not fp8 at all — which is unreachable, because only an fp8 scheme
    /// reaches either caller. Reproduced rather than tightened: tightening
    /// it would be a behaviour change dressed as a cleanup.
    #[cfg(feature = "_cuda")]
    fn fp8_kind_of(storage_dtype: KvDType) -> fp8_kind {
        fp8_kind(if matches!(storage_dtype, KvDType::Fp8E5M2) { NV_E5M2 } else { NV_E4M3 })
    }

    /// NVFP4's block, when the layer states none.
    ///
    /// `kv_paged.cu:242-244`. 16 is the arena's layout and not a tuning
    /// knob — the writer and the reader both spelled it, and a cache
    /// written at one block and read at another is silently wrong rather
    /// than slow. One `fn` so the two cannot drift.
    #[cfg(feature = "_cuda")]
    fn fp4_block_size(layer: &KvLayer) -> i32 {
        if layer.block_size > 0 { layer.block_size } else { 16 }
    }

    /// An upper bound on the pages an append can touch.
    ///
    /// `pub` because it is not this module's number: `envelope_update_appended`
    /// takes a `max_touched` it cannot derive — it holds the page CSR but not
    /// the token count that will be written into it — so the caller that
    /// knows both supplies it.
    ///
    /// The bound is the token span rounded up to whole pages, plus one page
    /// per request for the partially-filled tail each request appends into.
    /// Returns `0` for a non-positive page size rather than dividing by it.
    #[must_use]
    pub fn max_touched_pages(total_tokens: i32, num_requests: i32, page_size: i32) -> i32 {
        if page_size <= 0 {
            return 0;
        }
        (total_tokens + page_size - 1) / page_size + num_requests
    }

    // ---------------------------------------------------------------------
    // 1. The explicit-slot writes — a (page, offset) pair per row, already
    //    resolved on the device, so neither of these two reads a CSR.
    // ---------------------------------------------------------------------

    /// `attn::write_kv_explicit_bf16` — write B rows to B explicit slots.
    ///
    /// The parameter order is the driver's, unchanged, including `stream`
    /// arriving before `row_valid`. Reordering it would read better and is
    /// exactly the edit that turns a mechanical move into a silent
    /// mis-binding, which is the class this port keeps finding.
    ///
    /// # Panics
    ///
    /// If the layer is not native bf16. `kv_paged.cu:314` threw, and a
    /// throw is not a refusal: a quantised cache reaching an unquantised
    /// writer is a caller that computed the wrong thing, not a shape this
    /// kernel declines to handle.
    ///
    /// # Safety
    ///
    /// Every pointer must be a device allocation of the stated extent, and
    /// `stream` a live stream.
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_explicit_bf16(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        w_page: *const u32,
        w_off: *const u32,
        b: i32,
        stream: *mut c_void,
        row_valid: *const u8,
    ) -> Fired {
        assert!(
            layer.is_native_bf16,
            "attn::write_kv_explicit_bf16 requires native bf16 KV cache"
        );
        if b <= 0 {
            return Fired::Declined(Refusal::Empty { what: "rows" });
        }

        let symbol = if layer.hnd {
            "attn::write_kv_explicit_bf16_dev#hnd"
        } else {
            "attn::write_kv_explicit_bf16_dev#nhd"
        };
        unsafe {
            raw::write_kv_explicit(
                symbol,
                Launch::per_row(b.unsigned_abs(), BLOCK),
                k_curr,
                v_curr,
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                w_page,
                w_off,
                MaybeConst::new(row_valid),
                b,
                layer.page_size,
                layer.num_kv_heads,
                layer.head_dim,
                stream,
            );
        }

        // `k_curr` and NOT `k_pages`: the merge reads the ROWS just written,
        // which are contiguous and are the only thing whose envelope changed.
        // Reading them back through the page indirection would be the same
        // values at a worse stride, and would need the write to have landed.
        if layer.has_envelopes && !layer.hnd {
            let _ = unsafe {
                crate::x::layout::envelope_merge_written(
                    k_curr,
                    w_page,
                    w_off,
                    MaybeConst::new(row_valid),
                    layer.k_env_min.cast(),
                    layer.k_env_max.cast(),
                    b,
                    layer.num_kv_heads,
                    layer.head_dim,
                    stream,
                )
            };
        }
        Fired::Launched
    }

    /// `attn::write_kv_explicit_bf16_devwin` — the same write with a
    /// device-side window, so the row count is a ceiling rather than a
    /// count and the kernel reads `win[]` to find the real one.
    ///
    /// # Panics
    ///
    /// If the layer is not native bf16 (`kv_paged.cu:252`), or if it
    /// carries envelopes (`:262`) — envelope maintenance was never
    /// windowed, and merging a window's rows against a full-row envelope
    /// would widen it with rows the window excluded.
    ///
    /// # Safety
    ///
    /// As [`write_kv_explicit_bf16`].
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_explicit_bf16_devwin(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        w_page: *const u32,
        w_off: *const u32,
        win_d: *const u32,
        n_max: i32,
        stream: *mut c_void,
        row_valid: *const u8,
    ) -> Fired {
        assert!(
            layer.is_native_bf16,
            "attn::write_kv_explicit_bf16_devwin requires native bf16 KV cache"
        );
        if n_max <= 0 {
            return Fired::Declined(Refusal::Empty { what: "lanes" });
        }
        // `:262`, and it stays a throw for the reason the C++ gave it one:
        // a windowed write into an enveloped layer would leave the envelope
        // describing rows the window never wrote.
        assert!(
            !layer.has_envelopes,
            "attn::write_kv_explicit_bf16_devwin: envelope maintenance not yet \
             windowed — use the host-window form"
        );

        let symbol = if layer.hnd {
            "attn::write_kv_explicit_bf16_devwin_dev#hnd"
        } else {
            "attn::write_kv_explicit_bf16_devwin_dev#nhd"
        };
        unsafe {
            raw::write_kv_explicit_devwin(
                symbol,
                Launch::per_row(n_max.unsigned_abs(), BLOCK),
                k_curr,
                v_curr,
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                w_page,
                w_off,
                MaybeConst::new(row_valid),
                win_d,
                n_max,
                layer.page_size,
                layer.num_kv_heads,
                layer.head_dim,
                stream,
            );
        }
        Fired::Launched
    }

    // ---------------------------------------------------------------------
    // 2. The CSR append — one entry point, two halves, four quantised arms.
    // ---------------------------------------------------------------------

    /// The native-bf16 append, `kv_paged.cu:60-120`.
    ///
    /// # Safety
    ///
    /// As [`write_kv_explicit_bf16`]; the four CSR arrays must describe
    /// `num_requests` requests over `total_tokens` tokens.
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_to_pages_bf16(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        qo_indptr: *const u32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        total_tokens: i32,
        num_requests: i32,
        stream: *mut c_void,
        row_valid: *const u8,
        first_token: i32,
    ) -> Fired {
        let launch_tokens = total_tokens - first_token;
        if launch_tokens <= 0 {
            return Fired::Declined(Refusal::Empty { what: "tokens after first_token" });
        }

        let symbol =
            if layer.hnd { "attn::write_kv_bf16#hnd" } else { "attn::write_kv_bf16#nhd" };
        unsafe {
            raw::write_kv(
                symbol,
                Launch::per_row(launch_tokens.unsigned_abs(), BLOCK),
                k_curr,
                v_curr,
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                qo_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                MaybeConst::new(row_valid),
                // The window this kernel can take and this caller never has:
                // the CSR append writes every token it was given.
                MaybeConst::none(),
                num_requests,
                layer.page_size,
                layer.num_kv_heads,
                layer.head_dim,
                first_token,
                stream,
            );
        }

        // Two statements in a `fn`, in order, on one stream — `Composed`'s
        // shape without `Composed`, because the second reads what the first
        // wrote and nothing between them is a decision.
        if layer.has_envelopes && !layer.hnd && total_tokens > 0 {
            let _ = unsafe {
                crate::x::layout::envelope_update_appended(
                    layer.k_pages.cast(),
                    qo_indptr,
                    kv_page_indices,
                    kv_page_indptr,
                    kv_last_page_lens,
                    layer.k_env_min.cast(),
                    layer.k_env_max.cast(),
                    num_requests,
                    max_touched_pages(total_tokens, num_requests, layer.page_size),
                    layer.page_size,
                    layer.num_kv_heads,
                    layer.head_dim,
                    stream,
                )
            };
        }
        Fired::Launched
    }

    /// The quantised append, `kv_paged.cu:130-190` — four schemes, three
    /// kernels, and a fifth arm that declines.
    ///
    /// The per-token-head case fires **two symbols and not a
    /// `Specialisation`**: `UseFp8` is read off the cache's scheme and
    /// appears in no parameter of either kernel, so the choice is the host's
    /// and the registry had no way to make it. That was true of all five of
    /// this file's `Specialisation`s and is why `device::SPECIALISED` is
    /// empty.
    ///
    /// # Safety
    ///
    /// As [`write_kv_to_pages_bf16`]; the layer's scale planes must be
    /// sized for its scheme.
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_to_pages_quantised(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        qo_indptr: *const u32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        total_tokens: i32,
        num_requests: i32,
        stream: *mut c_void,
    ) -> Fired {
        if total_tokens <= 0 {
            return Fired::Declined(Refusal::Empty { what: "tokens" });
        }
        let page_size = layer.page_size;
        let h_kv = layer.num_kv_heads;
        let d = layer.head_dim;
        let tokens = total_tokens.unsigned_abs();
        let heads = h_kv.unsigned_abs();

        match layer.scheme {
            KvScheme::Fp8PerTensor => unsafe {
                raw::write_kv_fp8_per_tensor(
                    "attn::write_kv_fp8_per_tensor",
                    Launch::per_row(tokens, BLOCK),
                    k_curr,
                    v_curr,
                    layer.k_pages.cast::<u8>(),
                    layer.v_pages.cast::<u8>(),
                    qo_indptr,
                    kv_page_indices,
                    kv_page_indptr,
                    kv_last_page_lens,
                    num_requests,
                    page_size,
                    h_kv,
                    d,
                    fp8_kind_of(layer.storage_dtype),
                    stream,
                );
            },

            KvScheme::Int8PerTokenHead | KvScheme::Fp8PerTokenHead => {
                let symbol = if matches!(layer.scheme, KvScheme::Fp8PerTokenHead) {
                    "attn::write_kv_fp8_per_token_head"
                } else {
                    "attn::write_kv_int8_per_token_head"
                };
                // Two `float`s per warp: the block reduces an absmax for K
                // and one for V, and a warp contributes one of each.
                let smem = 2 * (BLOCK / 32) * (core::mem::size_of::<f32>() as u32);
                let launch = Launch {
                    grid: [tokens, heads, 1],
                    block: [BLOCK, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                }
                .smem(smem);
                unsafe {
                    raw::write_kv_per_token_head(
                        symbol,
                        launch,
                        k_curr,
                        v_curr,
                        layer.k_pages,
                        layer.v_pages,
                        layer.k_scales.cast::<f32>(),
                        layer.v_scales.cast::<f32>(),
                        qo_indptr,
                        kv_page_indices,
                        kv_page_indptr,
                        kv_last_page_lens,
                        num_requests,
                        page_size,
                        h_kv,
                        d,
                        stream,
                    );
                }
            }

            KvScheme::Fp4Block => {
                let block_size = fp4_block_size(layer);
                let blocks = d.div_euclid(block_size) + i32::from(d.rem_euclid(block_size) != 0);
                // A warp per block and not a block per token: an fp4 block is
                // 16 values, and 32 lanes cover one with room for the pair
                // packing.
                let launch = Launch {
                    grid: [tokens, heads, blocks.unsigned_abs()],
                    block: [32, 1, 1],
                    smem: 0,
                    smem_opt_in: false,
                };
                unsafe {
                    raw::write_kv_fp4_block(
                        "attn::write_kv_fp4_block",
                        launch,
                        k_curr,
                        v_curr,
                        layer.k_pages.cast::<u8>(),
                        layer.v_pages.cast::<u8>(),
                        layer.k_scales.cast::<f32>(),
                        layer.v_scales.cast::<f32>(),
                        qo_indptr,
                        kv_page_indices,
                        kv_page_indptr,
                        kv_last_page_lens,
                        num_requests,
                        page_size,
                        h_kv,
                        d,
                        block_size,
                        stream,
                    );
                }
            }

            KvScheme::Native => {
                return Fired::Declined(Refusal::Absent {
                    what: "a quantised writer for Native storage",
                });
            }
        }
        Fired::Launched
    }

    /// `attn::write_kv_to_pages` — the entry point, which chooses.
    ///
    /// # Panics
    ///
    /// If `first_token != 0` on a cache that is not native bf16
    /// (`kv_paged.cu:130-134`). A partial write resumes into a page the
    /// quantised writers cannot address mid-block, and the C++ threw rather
    /// than write a wrong scale.
    ///
    /// # Safety
    ///
    /// As [`write_kv_to_pages_bf16`].
    #[cfg(feature = "_cuda")]
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn write_kv_to_pages(
        layer: &KvLayer,
        k_curr: *const bf16,
        v_curr: *const bf16,
        qo_indptr: *const u32,
        kv_page_indices: *const u32,
        kv_page_indptr: *const u32,
        kv_last_page_lens: *const u32,
        total_tokens: i32,
        num_requests: i32,
        stream: *mut c_void,
        row_valid: *const u8,
        first_token: i32,
    ) -> Fired {
        assert!(
            first_token == 0 || layer.is_native_bf16,
            "attn::write_kv_to_pages: partial (first_token) writes require the \
             native bf16 cache"
        );
        if layer.is_native_bf16 {
            return unsafe {
                write_kv_to_pages_bf16(
                    layer,
                    k_curr,
                    v_curr,
                    qo_indptr,
                    kv_page_indices,
                    kv_page_indptr,
                    kv_last_page_lens,
                    total_tokens,
                    num_requests,
                    stream,
                    row_valid,
                    first_token,
                )
            };
        }
        // The driver translated the quantised decline into a native one by
        // hand, through two enums that agreed on nothing. One `Fired` needs
        // no translation, and the `Refusal` the callee stated is the one the
        // caller returns — which is the reason `Fired` says more here than
        // the pair it replaced, not less.
        unsafe {
            write_kv_to_pages_quantised(
                layer,
                k_curr,
                v_curr,
                qo_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_last_page_lens,
                total_tokens,
                num_requests,
                stream,
            )
        }
    }

    // ---------------------------------------------------------------------
    // 3. The active-page dequant — the beam-repair cell move,
    //    `kv_paged.cu:352-378`.
    // ---------------------------------------------------------------------

    /// The fp8-per-tensor arm, called by name from
    /// [`dequant_kv_cache_layer_to_bf16_active`] and directly by the four
    /// host programs that only ever hold that scheme.
    ///
    /// # Safety
    ///
    /// `kv_page_indices` must list `num_pages_in_batch` valid page indices,
    /// and the layer's bf16 mirror planes must be sized for them.
    #[cfg(feature = "_cuda")]
    #[must_use]
    pub unsafe fn dequant_fp8_per_tensor_pages_active(
        layer: &KvLayer,
        kv_page_indices: *const u32,
        num_pages_in_batch: i32,
        stream: *mut c_void,
    ) -> Fired {
        // `kv_paged.cu:384` folded these two into one `if (a || b) return;`
        // because `void` could carry neither reason. `Refusal` can carry
        // both, so it does — a native layer has nothing to dequantise and an
        // empty batch has nothing to dequantise it from, and those are not
        // the same answer to the caller that asked.
        if layer.is_native_bf16 {
            return Fired::Declined(Refusal::Absent { what: "quantised pages on a bf16 layer" });
        }
        if num_pages_in_batch <= 0 {
            return Fired::Declined(Refusal::Empty { what: "active pages" });
        }
        if !matches!(layer.scheme, KvScheme::Fp8PerTensor) {
            return Fired::Declined(Refusal::Absent { what: "an fp8-per-tensor layer" });
        }

        let (logical_n, page_elems, launch) = active_geometry(layer, num_pages_in_batch);
        unsafe {
            raw::dequant_fp8_pages_active(
                "attn::dequant_fp8_pages_active_bf16",
                launch,
                layer.k_pages.cast::<u8>().cast_const(),
                layer.v_pages.cast::<u8>().cast_const(),
                layer.k_bf16_pages.cast::<bf16>(),
                layer.v_bf16_pages.cast::<bf16>(),
                kv_page_indices,
                logical_n,
                page_elems,
                fp8_kind_of(layer.storage_dtype),
                stream,
            );
        }
        Fired::Launched
    }

    /// The element count an active-page pass covers, and the grid that
    /// covers it.
    ///
    /// **The widening is load-bearing** and is why it is a `fn`:
    /// `page_elems` is an `int`, and `num_pages_in_batch * page_elems`
    /// overflows 32 bits at production page counts — which is the whole
    /// reason every one of these kernels takes `n` as a `long long`. The C++
    /// widened before multiplying (`kv_paged.cu:385-388`); so does this, in
    /// one place, so the four arms cannot each get it right separately.
    #[cfg(feature = "_cuda")]
    fn active_geometry(layer: &KvLayer, num_pages_in_batch: i32) -> (i64, i32, Launch) {
        let page_elems = layer.page_size * layer.num_kv_heads * layer.head_dim;
        let logical_n = i64::from(num_pages_in_batch) * i64::from(page_elems);
        let blocks = (logical_n + i64::from(BLOCK) - 1) / i64::from(BLOCK);
        let launch = Launch {
            grid: [blocks as u32, 1, 1],
            block: [BLOCK, 1, 1],
            smem: 0,
            smem_opt_in: false,
        };
        (logical_n, page_elems, launch)
    }

    /// `attn::dequant_kv_cache_layer_to_bf16_active` — dequantise the pages
    /// this batch touches into the layer's bf16 mirror.
    ///
    /// `pub` and called from four other host programs as well as from its
    /// own arm: it is a prelude, not a leaf. Those four called it through
    /// the shim and now call it here.
    ///
    /// There is no `cudaGetLastError` after the launch, deliberately: the
    /// C++ checked because a shim entry that fired one of four names could
    /// not say which had failed, and `hand::fire` names the symbol it fired.
    ///
    /// # Safety
    ///
    /// As [`dequant_fp8_per_tensor_pages_active`].
    #[cfg(feature = "_cuda")]
    #[must_use]
    pub unsafe fn dequant_kv_cache_layer_to_bf16_active(
        layer: &KvLayer,
        kv_page_indices: *const u32,
        num_pages_in_batch: i32,
        stream: *mut c_void,
    ) -> Fired {
        if layer.is_native_bf16 {
            return Fired::Declined(Refusal::Absent { what: "quantised pages on a bf16 layer" });
        }
        if num_pages_in_batch <= 0 {
            return Fired::Declined(Refusal::Empty { what: "active pages" });
        }
        let (logical_n, _page_elems, launch) = active_geometry(layer, num_pages_in_batch);

        match layer.scheme {
            // Called rather than repeated. Its own guards re-run and pass —
            // three predicates over facts nothing between the two calls can
            // change.
            KvScheme::Fp8PerTensor => unsafe {
                dequant_fp8_per_tensor_pages_active(
                    layer,
                    kv_page_indices,
                    num_pages_in_batch,
                    stream,
                )
            },

            // A scale plane per (token, head) rather than one per tensor, so
            // the kernel needs the page geometry to find a scale and takes
            // `page_size`, `h_kv` and `d` where the per-tensor arm took a
            // flat `page_elems`.
            KvScheme::Fp8PerTokenHead => {
                unsafe {
                    raw::dequant_fp8_per_token_head(
                        "attn::dequant_fp8_per_token_head_pages_active_bf16",
                        launch,
                        layer.k_pages.cast::<u8>().cast_const(),
                        layer.v_pages.cast::<u8>().cast_const(),
                        layer.k_scales.cast::<f32>().cast_const(),
                        layer.v_scales.cast::<f32>().cast_const(),
                        layer.k_bf16_pages.cast::<bf16>(),
                        layer.v_bf16_pages.cast::<bf16>(),
                        kv_page_indices,
                        logical_n,
                        layer.page_size,
                        layer.num_kv_heads,
                        layer.head_dim,
                        stream,
                    );
                }
                Fired::Launched
            }

            // Byte-for-byte the arm above with a different element type on
            // the page planes. Two symbols and not one template because the
            // pages are `i8` in one and `u8` in the other, and a single
            // declaration could not say which.
            KvScheme::Int8PerTokenHead => {
                unsafe {
                    raw::dequant_int8_per_token_head(
                        "attn::dequant_int8_per_token_head_pages_active_bf16",
                        launch,
                        layer.k_pages.cast::<i8>().cast_const(),
                        layer.v_pages.cast::<i8>().cast_const(),
                        layer.k_scales.cast::<f32>().cast_const(),
                        layer.v_scales.cast::<f32>().cast_const(),
                        layer.k_bf16_pages.cast::<bf16>(),
                        layer.v_bf16_pages.cast::<bf16>(),
                        kv_page_indices,
                        logical_n,
                        layer.page_size,
                        layer.num_kv_heads,
                        layer.head_dim,
                        stream,
                    );
                }
                Fired::Launched
            }

            // The only arm with a twelfth operand, and the only one whose
            // `n` is LOGICAL rather than physical: an fp4 page holds two
            // values per byte, so the grid covers twice the bytes it reads
            // and every address inside the kernel is derived by halving.
            KvScheme::Fp4Block => {
                unsafe {
                    raw::dequant_fp4(
                        "attn::dequant_fp4_pages_active_bf16",
                        launch,
                        layer.k_pages.cast::<u8>().cast_const(),
                        layer.v_pages.cast::<u8>().cast_const(),
                        layer.k_scales.cast::<f32>().cast_const(),
                        layer.v_scales.cast::<f32>().cast_const(),
                        layer.k_bf16_pages.cast::<bf16>(),
                        layer.v_bf16_pages.cast::<bf16>(),
                        kv_page_indices,
                        logical_n,
                        layer.page_size,
                        layer.num_kv_heads,
                        layer.head_dim,
                        fp4_block_size(layer),
                        stream,
                    );
                }
                Fired::Launched
            }

            // `case KvCacheScheme::Native: break;`. Unreachable in the C++,
            // which returned on `is_native_bf16()` first, and reachable here
            // only for a cache declaring `Native` storage in a dtype that is
            // not bf16.
            KvScheme::Native => {
                Fired::Declined(Refusal::Absent { what: "a quantised dequant for Native storage" })
            }
        }
    }
}

pub static UNITS: &[Unit] = &[
    attention_flashinfer::ATTENTION_FLASHINFER,
    attention_naive::ATTENTION_NAIVE,
    attention_naive_paged::ATTENTION_NAIVE_PAGED,
    attn_res::ATTN_RES,
    attn_sink::ATTN_SINK,
    dsa_indexer::DSA_INDEXER,
    dsv4_compress::DSV4_COMPRESS,
    head_dim_pad::HEAD_DIM_PAD,
    kimi_mla::KIMI_MLA,
    kv_paged::KV_PAGED,
    mla_fa2::MLA_FA2,
    mla_naive::MLA_NAIVE,
    mla_paged::MLA_PAGED,
    pack_dense_mask::PACK_DENSE_MASK,
    page_compact::PAGE_COMPACT,
    qkv_fused::QKV_FUSED,
    softcap::SOFTCAP,
    split_packed::SPLIT_PACKED,
];

// ---------------------------------------------------------------------------
// The numbers, once each.
// ---------------------------------------------------------------------------

/// `runtime/launch.rs:578` — `const BLOCK: u32 = 256;`.
///
/// The block every pointwise rule in this tree uses, and the block
/// [`elementwise`] and [`attn_res_blend_bf16`] take.
const BLOCK: u32 = 256;

/// `runtime/launch.rs:599` — `const PAD_BLOCK: u32 = 128;`.
///
/// **A kernel requirement and not a tuning number.** Both head-dim kernels
/// stride `d += kPadBlock`, the compile-time constant at
/// `head_dim_pad.cuh:65`, so a narrower block never visits the columns above
/// it — which for `pad_head_dim` is padding that was never zeroed, and for
/// `strip_head_dim` a head whose tail keeps whatever the destination held.
/// Neither fails; both answer.
const PAD_BLOCK: u32 = 128;

/// `runtime/launch.rs:608` — `const SINK_BLOCK_MIN: u32 = WARP;`.
const SINK_BLOCK_MIN: u32 = 32;

/// `runtime/launch.rs:610` — `const SINK_BLOCK_MAX: u32 = 128;`.
const SINK_BLOCK_MAX: u32 = 128;

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
///
/// `runtime/launch.rs:828-834` and its `eval` arm — `n = dims.rows *
/// dims.width`, then `grid [ceil(n / 256), 1, 1]`, `block [256, 1, 1]`, no
/// shared memory. The grid rounds UP, which is why every kernel fired through
/// it keeps its own element count as an operand.
#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

/// `LaunchRule::Rms`, as the expression it evaluates to.
///
/// `runtime/launch.rs:737-746` — grid `[rows, 1, 1]`, block `[256, 1, 1]`,
/// shared `(256 / 32) * 4` bytes: one `float` per warp, which is what a
/// two-stage block reduction needs and all it needs.
///
/// One block per ROW and not per element, because the reduction is over the
/// row: `kimi_mla.cuh:127` halves from `BLOCK_DIM / 2`, so the block width is
/// the tree's width and the same 256 the row's `elem` states. **They are one
/// number and only their agreement makes the shared array the right size.**
///
/// `smem` is 32 bytes at 256 threads. Stated as an expression rather than a
/// literal so that a changed [`BLOCK`] changes both ends together — the thing
/// that could not be said while the width lived in a template default.
#[must_use]
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / 32) * 4)
}

/// `LaunchRule::PerHeadElementwise`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1417-1423`, and **this launcher is the one the rule was
/// derived from** — `attn/attn_sink.cu`, `attention_sink_rescale_bf16`:
///
/// ```text
/// const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(num_q_heads));
/// const int block = (head_dim < 32) ? 32 : (head_dim > 128 ? 128 : head_dim);
/// device::attn_sink_rescale<bf16><<<grid, block, 0, stream>>>(...);
/// ```
///
/// which is `[rows, q_heads, 1]` and `clamp(head_dim, 32, 128)` to the digit.
/// **The ROW is `grid.x` and the head is `grid.y` here, the transpose of
/// [`per_head`]'s** — the two axis orders are the kernels' and not a
/// convention, and a rule read off the wrong one runs the same block count
/// over the wrong cells.
///
/// `q_heads` and not `kv_heads`, because the tensor this rescales is the
/// attention OUTPUT: one row per query head. A grouped-query fire has two
/// head counts to pick the wrong one from.
#[must_use]
const fn per_head_elementwise(rows: u32, heads: u32, head_dim: u32) -> Launch {
    Launch {
        grid: [rows, heads, 1],
        block: [head_dim_block(head_dim), 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `u32::clamp` is not `const`, and the rule's expression is transcribed
/// rather than rearranged.
#[must_use]
const fn head_dim_block(head_dim: u32) -> u32 {
    if head_dim < SINK_BLOCK_MIN {
        SINK_BLOCK_MIN
    } else if head_dim > SINK_BLOCK_MAX {
        SINK_BLOCK_MAX
    } else {
        head_dim
    }
}

/// `LaunchRule::PerHead`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1381-1383` — `grid [heads, rows, 1]`, `block [128, 1,
/// 1]`, no shared memory — and `head_dim_pad.cu` is the launcher it cites:
/// `dim3 grid(num_heads, num_tokens)`, `dim3 block(kPadBlock)`. **The head is
/// `grid.x` and the row is `grid.y`**, the transpose of every other
/// head-shaped rule in the vocabulary, because that is the axis order these
/// two kernels read: `blockIdx.y` is the token at `head_dim_pad.cuh:78` and
/// `:97`, `blockIdx.x` is the head at `:79` and `:98`.
///
/// # THE DEFECT THIS FUNCTION CLOSES, measured
///
/// The rule evaluates `per_head(dims.rows, dims.kv_heads)` — it reads
/// `ctx.num_kv_heads`, **a field no part of either row mentions**. The head
/// count these kernels address with is the packed side's width over
/// `head_dim`, which is a QUERY head count wherever a q-projection is what
/// was padded. At Phi-3-mini's 12 heads of 64 with `num_kv_heads = 6` the two
/// arms differ in **6,100 of 12,544 bytes for the pad** and **4,588 of 9,472
/// for the strip**: the JIT writes half the rectangle and reports success.
///
/// Both symbols are in `device::JIT_DISPATCHED` all the same, which is how
/// the defect was reachable. **The port closes it by construction**: the `fn`
/// below is handed `num_heads`, uses that same number for `grid.x` and for
/// the kernel's addressing, and has no way to reach a KV head count. That is
/// the whole of the argument for taking this root early.
#[must_use]
const fn per_head(rows: u32, heads: u32) -> Launch {
    Launch { grid: [heads, rows, 1], block: [PAD_BLOCK, 1, 1], smem: 0, smem_opt_in: false }
}

/// The merge's geometry — the grid of [`per_head_elementwise`] and a
/// **different block**, which is why no `LaunchRule` states it.
///
/// `dsv4_compress.cu:65` and `:87`:
///
/// ```text
/// dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(num_heads));
/// const int block = (head_dim < 32) ? 32 : ((head_dim > 256) ? 256 : head_dim);
/// ```
///
/// The grid is `PerHeadElementwise` to the digit. The block is not: this
/// clamps into `[32, 256]` and the rule clamps into `[32, 128]`, so on a head
/// wider than 128 the rule answers with half these threads. The kernel
/// strides `d += blockDim.x` and reduces nothing, so the narrow block
/// computes the same bytes in two passes — **a slower kernel and never a
/// wrong answer**, which is precisely why rowing it was refused: the row
/// would agree at deepseek_v4's 128-wide heads, stop agreeing at the first
/// config that widened one, and nothing would fail and nothing would report.
///
/// `driver-cuda/src/fire/dsv4_compress.rs` carried that argument at length
/// and closed it *"Reconciling it is a decision about `SINK_BLOCK_MAX` in
/// `runtime/launch.rs`, which is not this file's to make."* **It is still not
/// this file's to make, and fn-world is why it never has to be**: a `fn`
/// states its own geometry, so the two clamps can differ in the open instead
/// of one of them being wrong. What the crossing changes is that the
/// divergence is now four lines from the launch it belongs to rather than in
/// another crate.
#[must_use]
const fn combine_attn(rows: u32, heads: u32, head_dim: u32) -> Launch {
    Launch {
        grid: [rows, heads, 1],
        block: [combine_block(head_dim), 1, 1],
        smem: 0,
        smem_opt_in: false,
    }
}

/// `[32, 256]`, transcribed rather than rearranged — `u32::clamp` is not
/// `const`, and [`head_dim_block`]'s twin is deliberately not reused: these
/// are two clamps that agree on the floor and disagree on the ceiling, and a
/// shared helper would be the place someone later unifies them.
#[must_use]
const fn combine_block(head_dim: u32) -> u32 {
    if head_dim < COMBINE_BLOCK_MIN {
        COMBINE_BLOCK_MIN
    } else if head_dim > COMBINE_BLOCK_MAX {
        COMBINE_BLOCK_MAX
    } else {
        head_dim
    }
}

/// A warp. `dsv4_compress.cu:87`'s `(head_dim < 32) ? 32`.
const COMBINE_BLOCK_MIN: u32 = 32;

/// `dsv4_compress.cu:87`'s `(head_dim > 256) ? 256`. **Not
/// `SINK_BLOCK_MAX`** — see [`combine_attn`].
const COMBINE_BLOCK_MAX: u32 = 256;

// ---------------------------------------------------------------------------
// Truth two: the host programs. One `fn` per launcher, each returning
// `Fired` so that "it declined" cannot be spelled like "it ran".
// ---------------------------------------------------------------------------

/// `attn::lse_log2_to_ln` — rebase flashinfer's LSE from log2 to ln, in place.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// The row was `LaunchRule::Elementwise` in `device::JIT_DISPATCHED`, so
/// `attn/attn_sink.cu` went with the file and there was nothing to port.
/// [`elementwise`] is that rule, and `n` is the element count the deleted row
/// sourced `Source::OutElements(0)` — `rows * out_width(0)`, which the bind
/// spells with the same two queries.
///
/// The rebase is in place on the value it names: the statement's result and
/// its first operand are the same buffer, so the element count is the
/// result's own extent.
///
/// # Safety
///
/// `lse` must address `n` live, writable `f32`s, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn lse_log2_to_ln(lse: *mut f32, n: usize, stream: *mut c_void) -> Fired {
    if n == 0 {
        return Fired::Declined(Refusal::Empty { what: "lse elements" });
    }
    let Ok(elems) = u32::try_from(n) else {
        // `Elementwise`'s grid is `ceil(n / 256)` in `u32`. An LSE with more
        // than 4.29e9 entries is not a fire this tree can make, and a silent
        // truncation here would launch a grid over a prefix of it.
        //
        // WIDE AND NOT NARROW, which this arm had backwards until
        // `Refusal::Wide` existed to say it: the count is ABOVE what the
        // grid can express, not below the kernel's smallest unit of work.
        // Both fields saturate — the ceiling is `u32::MAX` and the value is
        // larger still, and neither fits the `i32` the refusal carries — so
        // the sentence reads with `at` equal to `max`. What it gets right is
        // the direction, which is the whole reason the variant exists.
        return Fired::Declined(Refusal::Wide {
            what: "lse elements",
            at: i32::MAX,
            max: i32::MAX,
        });
    };
    unsafe {
        attn_sink::raw::lse_log2_to_ln(
            "attn::lse_log2_to_ln",
            elementwise(elems),
            lse,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::attention_sink_rescale_bf16` — gpt-oss's per-head sink correction,
/// in place on the attention output.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// [`per_head_elementwise`] transcribes `attn/attn_sink.cu`'s own `<<<>>>`,
/// which is the launcher `runtime::launch::per_head_elementwise` was derived
/// from. Row on `grid.x`, head on `grid.y`, block `clamp(head_dim, 32, 128)`.
///
/// In place on the output it corrects, which is what lets the o_proj GEMM and
/// the residual add downstream read rescaled activations without a copy.
/// `lse` is the dispatch's SECOND result — a value only a sink layer declares
/// — and the sinks are the layer's learned weight.
///
/// # Safety
///
/// `o` addresses `n * num_q_heads * head_dim` live, writable bf16 elements;
/// `lse` addresses `n * num_q_heads` live `f32`s; `sinks` addresses
/// `num_q_heads` live bf16 elements. All three live on `stream`, which must
/// outlive the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn attention_sink_rescale_bf16(
    o: *mut bf16,
    lse: *const f32,
    sinks: *const bf16,
    n: i32,
    num_q_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if num_q_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_q_heads" });
    }
    // `runtime/launch.rs`' `Ungeometric::Empty`: a head of no channels makes
    // the loop execute zero times, so the launch would report success having
    // written nothing.
    if head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        attn_sink::raw::attn_sink_rescale(
            "attn::attention_sink_rescale_bf16",
            per_head_elementwise(n.unsigned_abs(), num_q_heads.unsigned_abs(), head_dim.unsigned_abs()),
            o,
            lse,
            sinks,
            n,
            num_q_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `split_packed.cu:30` — `constexpr int BLOCK = 256;`.
///
/// Both of the root's launchers used it and neither took it as a parameter.
#[cfg(feature = "_cuda")]
pub const SPLIT_BLOCK: u32 = 256;

/// `attn::split_qkv_bf16_devwin` — the packed activation cut into Q, K and V,
/// over a DEVICE-RESIDENT row window.
///
/// # This crossing retracts two recorded reasons, and both were false
///
/// The unit's `split_qkv_devwin` doc, `families/attn.rs`' `SPLIT_PACKED_SIGS`
/// and this symbol's `execution::Walk` all said the same two things, at
/// length, and both had stopped being true before I read them:
///
/// 1. *"`grid.y` is the FIRE's lane count (`Ctx("rows_total")`) and not the
///    statement's rectangle."* True, and no longer a blocker:
///    [`crate::x::Rows::total`] IS `DispatchCtx::rows_total`, and
///    `bind/facts.rs:319` says so in the field's own doc — *"the whole fire's
///    row count, which a `_devwin` launch spans regardless of how many rows
///    its own region serves"*. It was written for this launch.
/// 2. *"`Cx::arg_in` / `Cx::arg_out` return pointers `resolve_arg_windowed`
///    has already offset by the region's first row, which this kernel would
///    window a second time."* **FALSE, and falsifiable in one line.**
///    `bind/mod.rs:3973` reads `let row = if kernel.ends_with("_devwin") { 0 }
///    else { launch.rows.start };` and hands that to every
///    `resolve_arg_windowed` for the launch — *"The `_devwin` forms are the
///    stated exception. Their contract is BASE pointers."* `Fire::arg_in` then
///    returns `bound.args[i].ptr` unchanged. The pointers a bind sees for
///    THIS symbol are the same pointers the generated arm passed.
///
/// So the sentence that made this the second edge of the driver-op
/// discriminator — *"needs no driver resource and still cannot bind"* — has
/// one true half. It needs no resource, and `Cx` states every one of its nine
/// operands. It is a MOVE.
///
/// The general fact (`arg_in` is pre-windowed) was derived once and stored
/// beside a symbol the exception names by suffix. §75's shape, sixth
/// instance, and the one that cost the most: it is the reason
/// [`crate::x::Cx::window_left`] was asked for. That query is not wasted —
/// FA2's sliding window is a real consumer and `window_of` is genuinely a
/// three-tier decision — but its doc names the wrong beneficiary, because
/// this kernel never wanted a span. It wants `win_d`, a device POINTER, and
/// that is [`crate::x::Cx::peel_window`], which has existed all along.
///
/// # The geometry, quoted
///
/// ```text
/// :43   const int max_dim = q_dim > kv_dim ? q_dim : kv_dim;
/// :44   const int xblocks = (max_dim + BLOCK - 1) / BLOCK;
/// :45   dim3 grid(xblocks, n_max);
/// :46   device::split_qkv_devwin<bf16><<<grid, BLOCK, 0, stream>>>(
/// ```
///
/// `grid.x` covers the WIDER of the two outputs and not the packed width:
/// `split_packed.cuh` licenses the difference — *"every loop below strides by
/// `blockDim.x * gridDim.x` and bounds itself on its own output width, so
/// extra blocks contribute nothing but a shorter loop"* — and the direction
/// matters only one way. A grid narrower than an output leaves the tail of
/// every row unwritten, so `max` and not `min`, transcribed rather than
/// re-derived.
///
/// `n_max` is `grid.y` and reaches the kernel as nothing else; `win` is the
/// seventh kernel parameter and the only thing that says which rows to touch.
///
/// # Safety
///
/// `packed`, the three outputs and `win` are device addresses live across the
/// launch, and `stream` is the caller's. The four buffer pointers must be
/// BASE pointers — the kernel windows them itself from `win`, so a
/// pre-windowed pointer is windowed twice. The binder guarantees it by the
/// `_devwin` suffix; a hand caller must not.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn split_qkv_bf16_devwin(
    packed: *const bf16,
    q_out: *mut bf16,
    k_out: *mut bf16,
    v_out: *mut bf16,
    win: *const u32,
    n_max: i32,
    q_dim: i32,
    kv_dim: i32,
    stream: *mut c_void,
) -> Fired {
    // `split_packed.cu:42` — `n_max` IS `grid.y`, so an empty lane count is
    // an empty grid. Hoisted, like every refusal here, ahead of the one
    // launch; there is nothing to hoist it past.
    if n_max <= 0 {
        return Fired::Declined(Refusal::Empty { what: "lanes" });
    }
    let max_dim = if q_dim > kv_dim { q_dim } else { kv_dim };
    if max_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "output width" });
    }
    let xblocks = max_dim.unsigned_abs().div_ceil(SPLIT_BLOCK);
    unsafe {
        split_packed::raw::split_qkv_devwin(
            "attn::split_qkv_devwin",
            Launch {
                grid: [xblocks.max(1), n_max.unsigned_abs(), 1],
                block: [SPLIT_BLOCK, 1, 1],
                smem: 0,
                smem_opt_in: false,
            },
            packed,
            q_out,
            k_out,
            v_out,
            win,
            q_dim,
            kv_dim,
            stream,
        );
    }
    Fired::Launched
}

/// `attention_naive_paged.cuh:33` — `constexpr int BLOCK = 128`.
///
/// **A shared-memory contract and an array divisor, not a tuning number.**
/// [`attention_naive_paged`](crate::x::attn::attention_naive_paged)'s module
/// doc argues all three dependents; this is the one place the digit is
/// written on the launch side. `crate::runtime::launch`'s `PAGED_BLOCK` is
/// the row world's copy of it, and it goes with `LaunchRule::PagedScores`.
#[cfg(feature = "_cuda")]
pub const PAGED_BLOCK: u32 = 128;

/// `attention_naive_paged.cuh:223` — `constexpr int kMaxHeadDim = 1024`.
///
/// The bound `acc[(kMaxHeadDim + BLOCK - 1) / BLOCK]` is sized against, and
/// therefore the largest head dim either kernel accepts. The `.cuh` states
/// the invariant: *"the array and the predicate that keeps launches inside it
/// are ONE constant, not two."* The predicate was `check_head_dim_supported`
/// in a `.cu` that is deleted; this is where it lives now.
#[cfg(feature = "_cuda")]
pub const PAGED_MAX_HEAD_DIM: i32 = 1024;

/// `attn::attention_naive_paged` — the reference paged attention.
///
/// *"Head dims flashinfer's prefill template rejects (gemma-4's 512) take a
/// naive paged kernel instead. No plan at all; fire-shaped."* —
/// `table::attn`'s row, which this `fn` retires.
///
/// # The geometry, quoted
///
/// ```text
/// attention_naive_paged.cu:195-221 --
///
///     dim3 grid(num_requests, total_tokens, num_q_heads);
///     dim3 block(BLOCK);
///     const std::size_t smem = (kv_layer.head_dim + BLOCK) * sizeof(float);
///     device::naive_paged_attn<BLOCK><<<grid, block, smem, stream>>>(
/// ```
///
/// Three grid axes and only two of them are extents this fire states: `grid.z`
/// is the head COUNT, which nothing carries and which the row derived as the
/// query's width over the cache's head dim. That division is done here, on the
/// two `Cx` facts, and it is the reason `head_dim <= 0` has to refuse before
/// anything else — a zero denominator is a panic in a `fn` a `bind!` reaches,
/// which §0 forbids.
///
/// # Every refusal is hoisted, and one of them is restored rather than ported
///
/// Six refusals, all ahead of the single launch, which is trivially satisfied
/// here because there IS one launch. Five are extents. The sixth is
/// [`Refusal::Wide`] on `head_dim > PAGED_MAX_HEAD_DIM`, and it is the
/// interesting one: the deleted `.cu` made it in `check_head_dim_supported`
/// and NOTHING has made it since, because the generated JIT arm that replaced
/// the `.cu` opens a grid through a `LaunchRule` and a `LaunchRule` cannot
/// refuse. A 2048-wide head has been reaching a kernel that indexes `acc[8]`
/// past its end. **A port that only preserved would have preserved the gap.**
///
/// # Safety
///
/// Every pointer must address live device memory of the extent the kernel
/// reads or writes, and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn attention_naive_paged(
    layer: &crate::x::cx::KvLayer,
    q: *const bf16,
    o: *mut bf16,
    qo_indptr: *const u32,
    kv_page_indices: *const u32,
    kv_page_indptr: *const u32,
    kv_last_page_lens: *const u32,
    total_tokens: i32,
    num_requests: i32,
    q_width: i32,
    window_left: i32,
    sm_scale: f32,
    logits_soft_cap: f32,
    lse_out: *mut f32,
    stream: *mut c_void,
) -> Fired {
    if num_requests <= 0 {
        return Fired::Declined(Refusal::Empty { what: "requests" });
    }
    if total_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if layer.head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the cache's head dim" });
    }
    if layer.head_dim > PAGED_MAX_HEAD_DIM {
        return Fired::Declined(Refusal::Wide {
            what: "head_dim",
            at: layer.head_dim,
            max: PAGED_MAX_HEAD_DIM,
        });
    }
    // The head COUNT, which no fact carries: the query's width over the
    // cache's head dim. `table::attn`'s row spelled it
    // `Source::Div(Width(In(0)), KvLayerField("head_dim"))` and the device row
    // spelled it the same way; one division, once, is what a `fn` buys.
    let num_q_heads = q_width / layer.head_dim;
    if num_q_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "q heads" });
    }
    if layer.num_kv_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "kv heads" });
    }
    // `(kv_layer.head_dim + BLOCK) * sizeof(float)` -- the query row in
    // fp32 followed by exactly `BLOCK` reduction slots. Both halves are
    // sized by constants this module states, so neither can drift from the
    // kernel that cuts them apart.
    let smem = (layer.head_dim.unsigned_abs() + PAGED_BLOCK) * 4;
    unsafe {
        attention_naive_paged::raw::naive_paged_attn(
            "attn::attention_naive_paged_dev",
            Launch {
                grid: [
                    num_requests.unsigned_abs(),
                    total_tokens.unsigned_abs(),
                    num_q_heads.unsigned_abs(),
                ],
                block: [PAGED_BLOCK, 1, 1],
                smem,
                smem_opt_in: false,
            },
            q,
            layer.k_pages.cast_const(),
            layer.v_pages.cast_const(),
            layer.k_scales.cast::<f32>().cast_const(),
            layer.v_scales.cast::<f32>().cast_const(),
            o,
            qo_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_last_page_lens,
            // `nullptr, nullptr` -- `attention_naive_paged.cu:208-209`. This
            // host program is the CAUSAL one; its `_custom` sibling passed a
            // real mask and had its own symbol.
            core::ptr::null(),
            core::ptr::null(),
            num_q_heads,
            layer.num_kv_heads,
            layer.head_dim,
            layer.page_size,
            kv_scheme::of(layer.scheme),
            kv_dtype::of(layer.storage_dtype),
            layer.block_size,
            window_left,
            sm_scale,
            logits_soft_cap,
            lse_out,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::attn_res_blend_bf16` — K3's residual-block blend.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// `LaunchRule::PerRow` — `runtime/launch.rs:1103`, `grid [rows, 1, 1]`,
/// `block [256, 1, 1]`, no shared memory — and the launcher it reproduces is
/// `<<<T, kThreads, 0>>>` in `attn/attn_res.cu` with `constexpr int kThreads
/// = 256` at `attn_res.cuh:69`.
///
/// **`PerRow`, not `Rms`.** `Rms` requests thirty-two bytes of dynamic shared
/// memory that no launcher here passes and no kernel here reads —
/// `block_sum`'s warp buffer, which this shape has no reduction to need: its
/// reduction is the static `__shared__ float scratch[kThreads / 32]` at
/// `attn_res.cuh:109`. Harmless in effect and wrong as a contract: a rule is
/// meant to REPRODUCE its launcher, and one that asks for memory the launcher
/// did not is a rule nobody can check against the `<<<>>>` it came from.
///
/// # `b` is an operand over an operand
///
/// How many candidate blocks the packed input holds is the BLOCKS operand's
/// row width over the RESULT's — an operand-over-operand ratio, where every
/// `*WidthOver` variant the row grammar had divides by a CONTEXT field. A
/// caller that guessed a param would launch the right kernel over the wrong
/// rectangle. The bind reads `in_width(1) / out_width(0)`, which is the two
/// widths of one statement.
///
/// # Safety
///
/// `prefix` and `out` address `t * h` live bf16 elements, `blocks` addresses
/// `t * b * h`, and `norm_weight` and `proj_weight` address `h` each. `out`
/// is writable and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn attn_res_blend_bf16(
    prefix: *const bf16,
    blocks: *const bf16,
    norm_weight: *const bf16,
    proj_weight: *const bf16,
    out: *mut bf16,
    t: i32,
    b: i32,
    h: i32,
    block_rows: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if t <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if b <= 0 {
        return Fired::Declined(Refusal::Empty { what: "blocks" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    unsafe {
        attn_res::raw::attn_res_blend(
            "attn::attn_res_blend_bf16",
            Launch::per_row(t.unsigned_abs(), BLOCK),
            prefix,
            blocks,
            norm_weight,
            proj_weight,
            out,
            b,
            h,
            block_rows,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::pad_head_dim_bf16` — pad every head out to a width flashinfer
/// compiles for.
///
/// # This launcher is NEW, and its geometry is quoted rather than invented
///
/// [`per_head`], which is `head_dim_pad.cu`'s own `dim3 grid(num_heads,
/// num_tokens)` / `dim3 block(kPadBlock)` — and see that function's doc for
/// the `dims.kv_heads` defect this signature closes.
///
/// # Which side is PACKED
///
/// Whichever end is `head_dim` wide — the input on the way in, the output on
/// the way out. So the head count divides out of the packed side and the
/// padded width is the other side over that count. Both readings are the
/// ahead-of-time rows', kept verbatim; the bind below is where they are
/// spelled.
///
/// # Safety
///
/// `packed` addresses `num_tokens * num_heads * head_dim` live bf16 elements
/// and `padded` addresses `num_tokens * num_heads * head_dim_padded`
/// writable ones. `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn pad_head_dim_bf16(
    packed: *const bf16,
    padded: *mut bf16,
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
    stream: *mut c_void,
) -> Fired {
    if let Some(why) = head_dim_refusal(num_tokens, num_heads, head_dim, head_dim_padded) {
        return Fired::Declined(why);
    }
    unsafe {
        head_dim_pad::raw::pad_head_dim(
            "attn::pad_head_dim_bf16",
            per_head(num_tokens.unsigned_abs(), num_heads.unsigned_abs()),
            packed,
            padded,
            num_heads,
            head_dim,
            head_dim_padded,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::strip_head_dim_bf16` — the inverse of [`pad_head_dim_bf16`].
///
/// Same geometry, same refusals, the two buffers swapped: `padded` is read
/// and `packed` is written, so the head count comes off the PACKED side,
/// which is the output here.
///
/// # Safety
///
/// `padded` addresses `num_tokens * num_heads * head_dim_padded` live bf16
/// elements and `packed` addresses `num_tokens * num_heads * head_dim`
/// writable ones. `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn strip_head_dim_bf16(
    padded: *const bf16,
    packed: *mut bf16,
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
    stream: *mut c_void,
) -> Fired {
    if let Some(why) = head_dim_refusal(num_tokens, num_heads, head_dim, head_dim_padded) {
        return Fired::Declined(why);
    }
    unsafe {
        head_dim_pad::raw::strip_head_dim(
            "attn::strip_head_dim_bf16",
            per_head(num_tokens.unsigned_abs(), num_heads.unsigned_abs()),
            padded,
            packed,
            num_heads,
            head_dim,
            head_dim_padded,
            stream,
        );
    }
    Fired::Launched
}

/// The four preconditions both head-dim launchers share, resolved BEFORE
/// either of them launches anything.
///
/// One function rather than two copies, because the pad and the strip are the
/// same rectangle read from opposite ends and a copy that drifted would
/// refuse one direction and not the other.
///
/// The last is not a geometry check: `head_dim_padded < head_dim` makes
/// `pad_head_dim` copy `head_dim` values into a `head_dim_padded`-wide row,
/// which is a write past the destination's head and not an empty launch.
/// `head_dim_pad.cuh`'s loop bounds itself on the PADDED extent, so nothing
/// on the device stops it.
#[cfg(feature = "_cuda")]
#[must_use]
fn head_dim_refusal(
    num_tokens: i32,
    num_heads: i32,
    head_dim: i32,
    head_dim_padded: i32,
) -> Option<Refusal> {
    if num_tokens <= 0 {
        return Some(Refusal::Empty { what: "rows" });
    }
    if num_heads <= 0 {
        return Some(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Some(Refusal::Empty { what: "head_dim" });
    }
    if head_dim_padded < head_dim {
        return Some(Refusal::Narrow { what: "head_dim_padded", at: head_dim_padded });
    }
    None
}

// ---------------------------------------------------------------------------
/// The guard `attn_softcap.cu`'s launcher opened with, as a refusal.
///
/// `if (n == 0 || !(cap > 0.f)) return;` — a bare `return` inside a `<<<>>>`
/// wrapper, which the caller could not tell apart from a launch. Here it has
/// two names, because the two halves are different facts: an empty extent is
/// [`Refusal::Empty`], and an absent cap is [`Refusal::Unstated`], which is
/// the same sentence `Source::CtxNonZero("final_logit_softcap")` was making
/// in the row.
///
/// **`cap.is_nan() || cap <= 0.0` is `!(cap > 0.f)` written without a negated
/// comparison**, and the equality is exact: for NaN the original's `>` is
/// false so it returned, and for every other value `cap <= 0.0` is the
/// complement of `cap > 0.0`. Positive infinity passes both, as it did in the
/// archive — `1/inf` is 0, `tanh(0)` is 0 and `inf * 0` is NaN, so an
/// infinite cap poisons the logits. That is the launcher's behaviour and it
/// is kept: a port is not the place to fix a case no `Facts` can produce,
/// and inventing a refusal here would make this `fn` and the row it replaces
/// answer differently.
#[cfg(feature = "_cuda")]
fn softcap_launch(cap: f32, n: usize) -> Result<Launch, Refusal> {
    if cap.is_nan() || cap <= 0.0 {
        return Err(Refusal::Unstated { what: "a logit soft cap" });
    }
    if n == 0 {
        return Err(Refusal::Empty { what: "logit elements" });
    }
    let Ok(elems) = u32::try_from(n) else {
        // As in [`lse_log2_to_ln`], including the saturation: `Elementwise`'s
        // grid is `ceil(n / 256)` in `u32`, a silent truncation would launch
        // over a prefix, and the count is ABOVE the ceiling rather than below
        // a floor.
        return Err(Refusal::Wide {
            what: "logit elements",
            at: i32::MAX,
            max: i32::MAX,
        });
    };
    Ok(elementwise(elems))
}

/// `attn::logit_softcap_bf16` — gemma's final logit cap, in place.
///
/// # Geometry
///
/// `attn_softcap.cu`, quoted whole by `softcap.cuh`'s header: `const auto
/// blocks = (n + 255) / 256; logit_softcap_bf16_kernel<<<blocks, 256, 0,
/// stream>>>(x, 1.f / cap, cap, n);`. That is [`elementwise`], which is
/// `LaunchRule::Elementwise`, which is what the deleted row said — three
/// spellings of one grid and they agree.
///
/// # The pairing this line is responsible for
///
/// `T` is inferred from `x`, and the symbol is the literal beside it. Nothing
/// in Rust ties the two together; the header says so at length. This is one
/// of the two places in the family where that pairing is written, the other
/// being [`logit_softcap_f16`] directly below, and they are adjacent so that
/// a reader checking one checks both.
///
/// # Safety
///
/// `x` must address `n` live, writable `bf16`s, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn logit_softcap_bf16(
    x: *mut bf16,
    cap: f32,
    n: usize,
    stream: *mut c_void,
) -> Fired {
    let launch = match softcap_launch(cap, n) {
        Ok(launch) => launch,
        Err(refusal) => return Fired::Declined(refusal),
    };
    unsafe {
        softcap::raw::logit_softcap("attn::logit_softcap_bf16", launch, x, cap, n, stream);
    }
    Fired::Launched
}

/// `attn::logit_softcap_f16` — the same cap over an fp16 buffer.
///
/// **This program has no contract and no caller**, and that is the state
/// `families/attn.rs` left the row in: *"the row the ahead-of-time build
/// never had"*, a second instantiation of a template that was already there.
/// It exists because the device row exists and truth two is a `fn` — a row
/// that can be fired needs a program, whether or not a trace says it yet.
///
/// Everything else is [`logit_softcap_bf16`]'s, including the geometry: the
/// element count is elements and not bytes, so the same `(n + 255) / 256`
/// covers a buffer of the same length in either format.
///
/// # Safety
///
/// `x` must address `n` live, writable `f16`s, and `stream` must be live
/// across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn logit_softcap_f16(x: *mut f16, cap: f32, n: usize, stream: *mut c_void) -> Fired {
    let launch = match softcap_launch(cap, n) {
        Ok(launch) => launch,
        Err(refusal) => return Fired::Declined(refusal),
    };
    unsafe {
        softcap::raw::logit_softcap("attn::logit_softcap_f16", launch, x, cap, n, stream);
    }
    Fired::Launched
}

/// `attn::kimi_split_q_b_bf16` — split a fused query projection into its
/// nope and rope halves.
///
/// # The extent is computed HERE, and that is the point of the crossing
///
/// `total` is `tokens * heads * (nope + rope)` — the INPUT's element count,
/// which is what [`kimi_mla`]'s kernel guards on and what the archive
/// launcher passed. The row this replaces stated the same thing
/// (`total <- InElements(0)`) and then had its grid sized by
/// `LaunchRule::Elementwise` from the first OUTPUT's width, which for a
/// kernel whose job is to make two narrower tensors out of one wider one is
/// short by exactly the split ratio. Measured at 6 rows of 8 heads, nope 128,
/// rope 64: **4,082 of 12,544 bytes of `q_nope` and 2,041 of 6,400 of `q_pe`
/// still held the poison fill.**
///
/// The parameters below are the four EXTENTS and not the product, so no
/// caller can hand a `total` that disagrees with the shape it also hands. A
/// launcher that takes a count someone else computed can be given a wrong
/// one; this one cannot.
///
/// # The product is formed in `i64` because the kernel's is not
///
/// `total` is an `int` on the device and `kimi_mla.cuh:84` casts to
/// `long long` only for the destination INDEX. At kimi_k3's head count a long
/// prefill reaches 2^31 elements before it reaches anything else, so the
/// product is formed wide here and the row count is refused when it will not
/// fit. That refusal is hoisted above the single launch, as every refusal in
/// this file is.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn kimi_split_q_b_bf16(
    q_b: *const bf16,
    q_nope: *mut bf16,
    q_pe: *mut bf16,
    tokens: i32,
    heads: i32,
    nope: i32,
    rope: i32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_heads" });
    }
    if nope <= 0 {
        return Fired::Declined(Refusal::Empty { what: "qk_nope_head_dim" });
    }
    if rope <= 0 {
        return Fired::Declined(Refusal::Empty { what: "qk_rope_head_dim" });
    }
    let width = i64::from(heads) * (i64::from(nope) + i64::from(rope));
    let total = i64::from(tokens) * width;
    if total > i64::from(i32::MAX) {
        return Fired::Declined(Refusal::Wide {
            what: "rows",
            at: tokens,
            max: i32::try_from(i64::from(i32::MAX) / width).unwrap_or(i32::MAX),
        });
    }
    let total = total as i32;
    unsafe {
        kimi_mla::raw::split_q_b(
            "attn::kimi_split_q_b_bf16",
            elementwise(total.unsigned_abs()),
            q_b,
            q_nope,
            q_pe,
            total,
            heads,
            nope,
            rope,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::kimi_split_kv_a_norm_bf16` — split `kv_a`, RMS-normalise the latent
/// half, copy the rope half unnormalised.
///
/// `<<<tokens, 256>>>` with `(256 / 32) * 4` bytes of shared memory — one
/// block per row, which is what [`rms`] says and why the row's `elem` states
/// `256` rather than letting the template default supply it. The block width
/// sizes `__shared__ float buf[BLOCK_DIM]` and fixes the halving tree; the
/// launch width and the instantiation width are one number and this file is
/// where they meet.
///
/// # `src_row_stride` is checked, which the row world could not do
///
/// The source row is `kv_lora + rope` wide unless a caller hands a wider
/// buffer — the fused prepare does. A stride NARROWER than the two halves it
/// is being asked to read out of is a read past the row into the next one,
/// which produces a plausible normalised vector built from the wrong token.
/// It is refused here because a `fn` can see all three numbers at once; a row
/// carrying three independent `Source`s cannot compare them.
#[cfg(feature = "_cuda")]
#[must_use]
pub unsafe fn kimi_split_kv_a_norm_bf16(
    kv_a: *const bf16,
    norm_weight: *const bf16,
    kv_c: *mut bf16,
    k_pe: *mut bf16,
    tokens: i32,
    kv_lora: i32,
    rope: i32,
    src_row_stride: i32,
    eps: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    if kv_lora <= 0 {
        return Fired::Declined(Refusal::Empty { what: "kv_lora_rank" });
    }
    if rope <= 0 {
        return Fired::Declined(Refusal::Empty { what: "qk_rope_head_dim" });
    }
    if src_row_stride < kv_lora + rope {
        return Fired::Declined(Refusal::Narrow { what: "src_row_stride", at: src_row_stride });
    }
    unsafe {
        kimi_mla::raw::split_kv_a_norm(
            "attn::kimi_split_kv_a_norm_bf16",
            rms(tokens.unsigned_abs()),
            kv_a,
            norm_weight,
            kv_c,
            k_pe,
            kv_lora,
            rope,
            src_row_stride,
            eps,
            stream,
        );
    }
    Fired::Launched
}

/// `attn::combine_attn_outputs_bf16` — merge two attention halves and their
/// log-sum-exps into one.
///
/// `dsv4_compress.cu:57-96`. How a sparse branch and a dense branch are
/// recombined: deepseek_v4 attends over a compressed cache and over the
/// selected fine blocks, and this is the single launch that folds the two
/// results into the one the rest of the layer reads.
///
/// The geometry is [`combine_attn`] and the paragraph there is the reason
/// this launcher was a `Walk` rather than a row for as long as the row world
/// lasted.
///
/// # The empty case is on the device, and that is not a gap
///
/// `lse2 == -inf` means the second half had no entries, and the kernel
/// passes `o1` through unchanged. That is a device-side branch on a
/// device-side value, which is §5.1's rule: no host can read it without a
/// synchronise, and a fire is a straight line. The three refusals below are
/// the ones a host CAN see, and every one of them is resolved before the
/// launch.
///
/// # `num_heads` and `head_dim` are the statement's, not the fire's
///
/// They are `Param(0)` and `Param(1)` on the row this replaces — not widths
/// and not `Cx::head_dim()`. The merged tensor is `[N, num_heads, head_dim]`,
/// so `out_width(0)` is their PRODUCT and no division of it recovers two
/// numbers. The statement carries both because it has to.
///
/// # Safety
///
/// Every pointer must address the extents these three numbers describe, and
/// `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub unsafe fn combine_attn_outputs_bf16(
    o1: *const bf16,
    lse1: *const f32,
    o2: *const bf16,
    lse2: *const f32,
    o_out: *mut bf16,
    lse_out: *mut f32,
    n: i32,
    num_heads: i32,
    head_dim: i32,
    stream: *mut c_void,
) -> Fired {
    // `dsv4_compress.cu:64` — `if (N <= 0) return;`, and it is `grid.x`.
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "rows" });
    }
    // Neither of these was tested by the C++, which formed `grid.y` from
    // `num_heads` and would have launched a zero-extent grid. A `fn` can see
    // all three numbers at once and a launch of no blocks that reports
    // success is the thing `Fired` exists to distinguish.
    if num_heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_heads" });
    }
    if head_dim <= 0 {
        return Fired::Declined(Refusal::Empty { what: "head_dim" });
    }
    unsafe {
        dsv4_compress::raw::combine_attn_outputs(
            "attn::combine_attn_outputs_dev",
            combine_attn(n.unsigned_abs(), num_heads.unsigned_abs(), head_dim.unsigned_abs()),
            o1,
            lse1,
            o2,
            lse2,
            o_out,
            lse_out,
            num_heads,
            head_dim,
            stream,
        );
    }
    Fired::Launched
}

// ---------------------------------------------------------------------------
// MLA'S ABSORB PAIR — the two cuBLAS host programs of this family.
//
// `gemm::mla_absorb_q_to_latent_bf16` and `gemm::mla_absorb_latent_to_v_bf16`.
// Two `gemm::` symbols in `table/attn.rs`, which is not a filing error: they
// are MLA's, they are stated by MLA's lowering, and `attn` is where they were
// measured.
//
// **The discriminator answers in one line for both, and it is the same line:
// `ctx.cublas`.** A `cublasHandle_t` created once per shell, carrying
// `cublasSetMathMode`, with a stream rebound per fire — §3.3's forbidden
// surface, which is why `x::gemm`'s twelve took this shape and why these two
// take it too. So the contracts below state NO `Entry`, and
// `driver-cuda/src/bind`'s driver-op `match` fires them with the handle in
// hand. Their `execution::SERVED` entries were `Service::Cublas` and are
// `Service::DriverOp` now: the FINDING has not changed -- one
// `cublasGemmStridedBatchedEx` each, and extracting a kernel from that
// extracts nothing -- only the question the list is asked, which since
// `x::route` is also *"does something the driver owns already fire this?"*
//
// The bodies were `bind::service::gemm_mla_absorb_*`. They are here for the
// reason `x::gemm`'s twelve are in `x::gemm`: a host program belongs beside
// the truth it is one of, and `handle: *mut c_void` as a first parameter is
// how fn-world spells a resource it cannot own.
//
// `COMPUTE`, `ALGO` and `check` below are a THIRD copy -- `x::gemm::dense`
// has all three and `bind::service` had all three. They are private in
// `x::gemm::dense`, so this is a copy the module boundary forces rather than
// one anybody chose; naming it here is cheaper than a reader re-deriving why.
// ---------------------------------------------------------------------------

/// `CUBLAS_COMPUTE_32F` — see `x::gemm::dense`'s `COMPUTE` for the tp > 1
/// boot failure that pinned it.
#[cfg(feature = "_cuda")]
const ABSORB_COMPUTE: cublasComputeType_t = cublasComputeType_t::CUBLAS_COMPUTE_32F;

/// `CUBLAS_GEMM_DEFAULT_TENSOR_OP`, which the archive pinned on both calls.
#[cfg(feature = "_cuda")]
const ABSORB_ALGO: cublasGemmAlgo_t = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;

/// The archive's `check(status, api)` — `gemm.cpp:47`.
///
/// Panics rather than returning, and that is the port being faithful: the C++
/// threw, the generated shim caught, printed and aborted. A `Refusal` here
/// would say the launch declined, and it did not — it failed.
#[cfg(feature = "_cuda")]
fn absorb_check(status: cublasStatus_t, what: &str) {
    assert!(
        status == cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        "cuBLAS error ({}): {what}",
        status as i32
    );
}

/// The absorb pair's shared call — `cublasGemmStridedBatchedEx` over the head
/// axis, `batchCount = heads`.
///
/// Both absorptions are the same strided-batched GEMM with a different slice
/// of `kv_b_proj` and a different transpose, so the argument assembly is
/// written once. `stride_a` is `(qk_nope_dim + v_head_dim) * kv_lora_rank`
/// for both — the FULL bank stride, because both read a slice of a bank
/// whose per-head pitch includes the other half.
///
/// # Safety
///
/// The caller's, per entry point below.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
unsafe fn absorb(
    handle: *mut c_void,
    op_a: cublasOperation_t,
    a: *const c_void,
    b: *const c_void,
    c: *mut c_void,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    stride_a: i64,
    ldb: i32,
    stride_b: i64,
    ldc: i32,
    stride_c: i64,
    heads: i32,
    what: &str,
) {
    let alpha = 1.0f32;
    let beta = 0.0f32;
    // SAFETY: the caller's obligation.
    let status = unsafe {
        cublasGemmStridedBatchedEx(
            handle.cast::<cublasContext>(),
            op_a,
            cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            core::ptr::from_ref(&alpha).cast(),
            a,
            cudaDataType::CUDA_R_16BF,
            lda,
            stride_a,
            b,
            cudaDataType::CUDA_R_16BF,
            ldb,
            stride_b,
            core::ptr::from_ref(&beta).cast(),
            c,
            cudaDataType::CUDA_R_16BF,
            ldc,
            stride_c,
            heads,
            ABSORB_COMPUTE,
            ABSORB_ALGO,
        )
    };
    absorb_check(status, what);
}

/// `gemm::mla_absorb_q_to_latent_bf16` — `gemm.cpp:2419-2442`.
///
/// Row-major `C[T, kv_lora] = A[T, nope] @ B[nope, kv_lora]` per head,
/// written column-major as `C^T = B^T @ A^T` — which is why both operands
/// are `OP_N` and `kv_b_proj` is the *first*.
///
/// **The archive's `tokens <= 0 || heads <= 0` early return is a
/// [`Refusal::Empty`] here and not a bare `return`.** It is a HOST decision
/// made before any launch, so it is exactly what `Fired` was added to be able
/// to say: under `void` the caller could not tell it apart from a launch, and
/// this pair is fired from a driver-op arm whose whole job is to know.
///
/// # Safety
///
/// `q_nope` must address `tokens * heads * qk_nope_dim` bf16 elements,
/// `kv_b_proj` the whole `heads * (qk_nope_dim + v_head_dim) * kv_lora_rank`
/// bank, and `q_latent` `tokens * heads * kv_lora_rank` writable elements.
/// `handle` must be a live `cublasHandle_t` with this fire's stream bound.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mla_absorb_q_to_latent_bf16(
    handle: *mut c_void,
    q_nope: *const c_void,
    kv_b_proj: *const c_void,
    q_latent: *mut c_void,
    tokens: i32,
    heads: i32,
    qk_nope_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "heads" });
    }
    // SAFETY: the caller's obligation, above.
    unsafe {
        absorb(
            handle,
            cublasOperation_t::CUBLAS_OP_N,
            kv_b_proj,
            q_nope,
            q_latent,
            kv_lora_rank,
            tokens,
            qk_nope_dim,
            kv_lora_rank,
            i64::from(qk_nope_dim + v_head_dim) * i64::from(kv_lora_rank),
            heads * qk_nope_dim,
            i64::from(qk_nope_dim),
            heads * kv_lora_rank,
            i64::from(kv_lora_rank),
            heads,
            "mla_absorb_q_to_latent_bf16",
        );
    }
    Fired::Launched
}

/// `gemm::mla_absorb_latent_to_v_bf16` — `gemm.cpp:2444-2468`.
///
/// The mirror: row-major `C[T, v_dim] = A[T, kv_lora] @ W[v_dim, kv_lora]^T`
/// per head, so `OP_T` on the weight, and the weight is the SECOND half of
/// each head's bank — `kv_b_proj + qk_nope_dim * kv_lora_rank`, in bf16
/// elements, which is the one pointer arithmetic step this port must not get
/// wrong.
///
/// # Safety
///
/// As [`mla_absorb_q_to_latent_bf16`], with `attn_latent` in place of
/// `q_nope` and `attn_v` (`tokens * heads * v_head_dim`) as the output.
#[cfg(feature = "_cuda")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn mla_absorb_latent_to_v_bf16(
    handle: *mut c_void,
    attn_latent: *const c_void,
    kv_b_proj: *const c_void,
    attn_v: *mut c_void,
    tokens: i32,
    heads: i32,
    qk_nope_dim: i32,
    v_head_dim: i32,
    kv_lora_rank: i32,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if heads <= 0 {
        return Fired::Declined(Refusal::Empty { what: "heads" });
    }
    // The `__nv_bfloat16*` arithmetic of `gemm.cpp:2452`, in bytes: two per
    // element, and the element count is `qk_nope_dim * kv_lora_rank`.
    // SAFETY: the offset lands inside the same bank the caller guaranteed —
    // `qk_nope_dim * kv_lora_rank` elements into a head pitch of
    // `(qk_nope_dim + v_head_dim) * kv_lora_rank`.
    let wv = unsafe {
        kv_b_proj
            .cast::<u8>()
            .add(2 * (qk_nope_dim as usize) * (kv_lora_rank as usize))
            .cast::<c_void>()
    };
    // SAFETY: the caller's obligation, above.
    unsafe {
        absorb(
            handle,
            cublasOperation_t::CUBLAS_OP_T,
            wv,
            attn_latent,
            attn_v,
            v_head_dim,
            tokens,
            kv_lora_rank,
            kv_lora_rank,
            i64::from(qk_nope_dim + v_head_dim) * i64::from(kv_lora_rank),
            heads * kv_lora_rank,
            i64::from(kv_lora_rank),
            heads * v_head_dim,
            i64::from(v_head_dim),
            heads,
            "mla_absorb_latent_to_v_bf16",
        );
    }
    Fired::Launched
}

// Truth three, declared: what a trace may say.
//
// Twenty-eight contracts, carrying twenty-nine of `table/attn.rs`' forty-one
// rows minus everything that described a launcher — seventeen and eighteen
// before the FlashInfer six and MLA's absorb pair, which are declared at the
// end of the block and are the eight here that state a contract and DECLINE
// to bind, because something the driver owns already fires them. (Seventeen and eighteen because
// [`head_dim_pad`]'s two rows are two contracts and [`kv_paged`]'s four are
// four; the counts diverge only where a contract retires more than the row
// that named it, which is why both are written and neither is derived from
// the other.) `softcap`
// declares two device rows and only one of them is a thing a trace may say —
// and [`qkv_fused`] declares ELEVEN and states one, which is the same fact at
// eleven times the scale: ten of its rows are a launcher's arms.
//
// [`dsv4_compress`] declares TEN and states three, and its other seven are
// not arms — three of them are fired by
// `driver-cuda/src/fire/dsv4_compress.rs` through `hand::fire`, which is a
// host program in another crate rather than a contract here. **A unit row
// does not need a contract to have a caller.** The fourth of those was
// `combine_attn_outputs_dev`, and it now has both: a contract here and a
// host `fn` above, which is what a row becomes when nothing outside the
// statement is needed to fire it.
// `Contract::DEFAULT` supplies the other fields of each. `lacks`,
// `publishes_aux` and `lowered_as` are stated by nothing here, as they were
// by nothing in the rows these replace; `needs`, `sink`, `whole` and
// `depth_prefix_plan` were in the same position until the FlashInfer six
// arrived, and all four are load-bearing for those: `needs` is what raises
// the plan cache their driver arm then reads, so a wrong one is a
// `Decline::Unplanned` at run time and not a compile error.
//
// The seventeen rows these passes did not take keep their contracts in
// `table/attn.rs`. `table::TABLES` concatenates both lists, so
// `model-compiler` reads one vocabulary and cannot tell which world serves a
// symbol — which is the property the split is allowed to have.
// ---------------------------------------------------------------------------

contract! {
    /// FlashInfer publishes its LSE in log2 and the sink correction works in
    /// ln. A unit conversion, stated so a reader never has to guess which
    /// base an LSE is in.
    ///
    /// In place on the value it names: the statement's result and its first
    /// operand are the same buffer, which is what `in_place` says.
    /// `model-compiler/src/dsl.rs` states it through
    /// `dsl::cuda::lse_log2_to_ln`.
    LSE_LOG2_TO_LN = "attn::lse_log2_to_ln" as lse_log2_to_ln {
        in_place: &[(0, 0)],
    }

    /// Rescales the attention output IN PLACE against the per-head sink
    /// logit; the LSE is read-only.
    ///
    /// gpt-oss's sink layers state it right after the dispatch, so
    /// `attn.out` observes the RESCALED result. The LSE is the dispatch's
    /// second RESULT, which only a sink layer declares — so it is operand 1
    /// and traced, not a scratch the executor remembers handing the
    /// dispatch.
    ATTENTION_SINK_RESCALE = "attn::attention_sink_rescale_bf16" as attention_sink_rescale {
        in_place: &[(0, 0)],
    }

    /// K3's residual-block blend: a prefix and `B` candidate blocks, scored
    /// by an RMS-normalised projection and combined.
    ATTN_RES_BLEND = "attn::attn_res_blend_bf16" as attn_res_blend

    /// The pad half of what `head_dim_padded` COSTS.
    ///
    /// Stating the pair turns `if (c.head_dim_padded)` in the model body
    /// into a fact the trace carries. Row-shaped — each token's heads pad
    /// independently.
    PAD_HEAD_DIM = "attn::pad_head_dim_bf16" as pad_head_dim

    /// The strip half. See [`PAD_HEAD_DIM`].
    STRIP_HEAD_DIM = "attn::strip_head_dim_bf16" as strip_head_dim

    /// Gemma's final logit cap — `cap * tanh(x / cap)` over the logits,
    /// where they lie.
    ///
    /// One buffer and no destination, which `Buffers::assign` was already
    /// relying on while the row said nothing: it widens a seam's pin over an
    /// alias set, the set had one member, and the widening reached nothing.
    /// The head wrote the logits into the arena, the cap ran over
    /// `ws.logits`, and the sampler read an uncapped previous fire. `in_place`
    /// is that alias, stated.
    ///
    /// A dispatch parameter and not a launch parameter: gemma-2's
    /// `attn_logit_softcapping` is a different fact and is not this. Only
    /// Gemma-2, Gemma-3 and Gemma-3n state a final cap, so the bind's
    /// [`Cx::final_logit_softcap`] refuses on every other deployment — which
    /// is a per-fire refusal and NOT a `none:` arm, because a `none:` arm
    /// refuses at model load and would take Gemma with it.
    ///
    /// `model-compiler/src/dsl.rs:6379` states it through
    /// `dsl::cuda::logit_softcap`, gated on `final_logit_softcapping` being
    /// present in the config — so its presence is a trace-time decision and
    /// the trace either carries the statement or does not.
    ///
    /// [`Cx::final_logit_softcap`]: crate::x::Cx::final_logit_softcap
    LOGIT_SOFTCAP = "attn::logit_softcap_bf16" as logit_softcap {
        in_place: &[(0, 0)],
    }

    /// kimi_k3's fused query projection, split into the halves attention
    /// wants: `[tokens, heads, nope]` and `[tokens, heads, rope]` out of
    /// `[tokens, heads, nope + rope]`.
    ///
    /// `model-compiler/src/dsl.rs:5483` states it; `crates/model`'s
    /// `kimi_k3/forward/mod.rs:156` is the caller. Two results and no alias,
    /// so no `in_place`: the source stays live, because the rope half is
    /// rotated afterwards and the nope half is not.
    ///
    /// Row-shaped — each token's heads split independently — which is why
    /// the deleted row was not `whole` and this contract states nothing
    /// about windows.
    KIMI_SPLIT_Q_B = "attn::kimi_split_q_b_bf16" as kimi_split_q_b

    /// The key/value half of the same split, with an RMS norm fused into it.
    ///
    /// `model-compiler/src/dsl.rs:5452` states it; `kimi_k3/forward/mod.rs:150`
    /// is the caller, one line before [`KIMI_SPLIT_Q_B`]'s.
    ///
    /// One kernel and not a split followed by a norm, because the latent
    /// half is read twice by the reduction and would otherwise make a round
    /// trip through global memory in between. The rope half is copied
    /// **unnormalised**, which is a property of the statement and not of the
    /// launch: normalising a value that is about to be rotated changes the
    /// angle.
    KIMI_SPLIT_KV_A_NORM = "attn::kimi_split_kv_a_norm_bf16" as kimi_split_kv_a_norm

    /// deepseek_v4's compressed-cache gather — one entry per boundary token,
    /// by a per-dimension softmax over the gate scores of the window ending
    /// there.
    ///
    /// `model-compiler/src/dsl.rs:4684` states it as
    /// `dsv4_compress_gather_paged`, and `crates/model/src/deepseek_v4` is
    /// its only caller. `whole` on the row it replaces; that is not a
    /// geometry fact and does not survive into a contract.
    DSV4_COMPRESS_GATHER_PAGED = "attn::dsv4_compress_gather_paged_bf16"
        as dsv4_compress_gather_paged {
        sink: Some("kv.compressed"),
    }

    /// The commit half — those entries into the compressed cache, each at its
    /// own boundary token's slot.
    ///
    /// `dsl.rs:4702`. **It produces no value at all** — `record` is called
    /// with `None` for the result — so its whole effect is a store the
    /// contract vocabulary cannot name, and `whole: true` with no `sink` is
    /// what the row said about that and what this says.
    DSV4_STORE_COMP_ENTRIES = "attn::dsv4_store_comp_entries_bf16"
        as dsv4_store_comp_entries {
        whole: true,
    }

    /// The merge that puts deepseek_v4's two attention branches back
    /// together — two outputs and two log-sum-exps in, one of each out.
    ///
    /// `model-compiler/src/dsl.rs:4749` states it and
    /// `model/src/deepseek_v4/forward/mod.rs:146` is its only caller: the
    /// compressed branch and the fine-block branch each attend over their own
    /// cache, and neither result means anything until the LSEs have weighted
    /// them against each other.
    ///
    /// TWO RESULTS, and both are traced. `lse_out` is `Out(1)` and not a
    /// scratch the executor remembers handing the launcher — the caller
    /// destructures `(o, lse)` and the LSE is read again by the layer above.
    ///
    /// `num_heads` and `head_dim` are `Param`s. See the `fn`: the merged
    /// tensor's width is their product and no division of a width recovers
    /// two numbers.
    COMBINE_ATTN_OUTPUTS = "attn::combine_attn_outputs_bf16" as combine_attn_outputs

    /// The fused QKV prefill epilogue — six statements in one launch, and
    /// the only value that survives is q.
    ///
    /// q norm, q rope, k norm, k rope, v norm and the paged KV write. The
    /// `sink` is what says the rest of it lands in the cache: everything but
    /// q is written to `k_pages`/`v_pages` and observed by a later dispatch
    /// rather than by a result.
    ///
    /// `model-compiler/src/dsl.rs` states it as `qkv_packed_post`. Its decode
    /// sibling is [`QKV_DECODE_FUSED`], immediately below; the two share this
    /// module's unit and its eleven device rows, and crossed one commit apart
    /// because the decode form needed `Cx::q_out` and this one did not.
    QKV_PACKED_POST = "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16" as qkv_packed_post {
        sink: Some("kv.pages"),
    }

    /// The decode form of the same fusion, and the last row `ROW_TABLES` had.
    ///
    /// One packed row per request instead of a prefill's token block: q norm,
    /// q rope, k norm, k rope and the paged KV write, over `num_requests`
    /// rows. The `sink` is the same store as its prefill sibling's, and for
    /// the same reason — everything but q is observed through `k_pages` and
    /// `v_pages` by a later dispatch rather than through a result.
    ///
    /// # `sink` is stated here and the row defaulted it to `None`
    ///
    /// Not a transcription error and worth the line, because it is the one
    /// place this contract does not match the row it retires. `kernel!`
    /// (`kernels/src/lib.rs:2109`) defaults `sink: None` and this row never
    /// overrode it, while `QKV_PACKED_POST` — the same fusion, the same
    /// store, the same pages — states `Some("kv.pages")`. Two rows over one
    /// state store disagreeing about whether they write it is §75's shape:
    /// a fact derived from one member of a set and stored next to a different
    /// member. The kernel's own text settles it — `k_pages` and `v_pages`
    /// are `BufMut` in the row and `*mut bf16` in the unit, and
    /// `qkv_fused.cuh:230` stores through both — so the sibling was right and
    /// the default was never a decision. Adding a `sink` can only add an
    /// ordering edge, never remove one.
    ///
    /// `whole` stays `false`, which the row also defaulted and which is
    /// correct rather than merely inherited: `num_requests` is
    /// `Source::Rows`, so the count and the `packed` pointer must describe
    /// the SAME region, and `Cx::arg_in` is pre-windowed by
    /// `resolve_arg_windowed`. A `whole: true` here would pair a whole-buffer
    /// pointer with a windowed count.
    ///
    /// `model-compiler/src/dsl.rs:3553` states it as
    /// `qkv_decode_qk_norm_rope_write_kv_region`.
    QKV_DECODE_FUSED = "attn::qkv_decode_qk_norm_rope_write_kv_bf16" as qkv_decode_fused {
        sink: Some("kv.pages"),
    }

    /// Attention over the latent cache — DeepSeek/Kimi MLA, the row that
    /// retires `attention_mla.cu` and with it the last nvcc-compiled
    /// `<<<>>>` in the workspace.
    ///
    /// # ONE SYMBOL, TWO HOST PROGRAMS, AND THAT IS WHY IT IS NOT BOUND
    ///
    /// `attention_mla.cu:150` chooses between them at run time on the
    /// compute capability, because FA2 MLA **produces zero output on
    /// sm_100** — a wrong answer and not a failure. Below sm_100 it is
    /// FlashInfer's cooperative `BatchMLAPagedAttentionKernel`, which is
    /// [`mla_fa2`] in this file, unit and all; on sm_100 it is the scalar
    /// and mma pair in [`mla_naive`], whose host program is
    /// `driver-cuda/src/fire/mla_naive.rs`, one crate up.
    ///
    /// A row loses its ahead-of-time shim entry whole or not at all, so both
    /// arms had to be Rust before either could be. **They both now are.**
    /// What is left is the seam between them, and the seam is `Cx` — see the
    /// `none:` arm below, which names the four queries and nothing else.
    ///
    /// # `needs: Prepare::MlaPlan`
    ///
    /// Its own kind of plan, built from the latent geometry (`kv_lora_rank`,
    /// `qk_rope_head_dim`) that no other prepare has a field for and cached
    /// in an `MlaPlanCache` rather than in the shared attention workspace.
    /// `crate::plan::mla` is already the Rust half of it.
    ///
    /// # `lacks: &[Cap::Scores]`
    ///
    /// There is no capture variant of this dispatch, so a program whose
    /// `attn.out` seam wants the score matrix cannot be served over the rows
    /// it covers. It publishes an LSE, which is a different thing and not
    /// what the capability names — and, per [`mla_fa2`]'s `pack`, an LSE in
    /// **natural log**, so [`LSE_LOG2_TO_LN`] must never be applied to it.
    ATTENTION_MLA = "attn::dispatch_attention_mla_bf16" as attention_mla {
        needs: Prepare::MlaPlan,
        lacks: &[Cap::Scores],
    }
    /// The paged KV append — **the most-fired symbol in this family**, once
    /// per layer of every fire.
    ///
    /// One contract over two host programs and four quantised arms:
    /// `x::attn::kv_paged::write_kv_to_pages` reads `layer.is_native_bf16`
    /// and calls the bf16 appender or the quantised switch. That choice was
    /// never a `Specialisation` and never could be — the schemes take
    /// different argument lists, not different instantiations of one.
    ///
    /// States no `sink` and no `whole`, transcribed from the row it retires.
    /// The sibling [`WRITE_KV_EXPLICIT_DEVWIN`] states both, for the same
    /// cache; that asymmetry pre-dates this crossing and is reproduced rather
    /// than resolved, because resolving it is a change to what a trace means
    /// and this is a change to where a program lives.
    WRITE_KV_TO_PAGES = "attn::write_kv_to_pages" as write_kv_to_pages

    /// The explicit-slot append: the fire states each row's destination page
    /// and offset instead of deriving them from the CSR.
    ///
    /// Only a fire that computed those carries them, which is what the row's
    /// `Source::AttnNonZero` tested and what [`Cx::w_page_d`]'s null check
    /// now tests — **the same predicate, moved from the emitter to the
    /// query.** A fire that appends no KV carries a null there, and a body
    /// that took the pointer anyway would index it.
    WRITE_KV_EXPLICIT = "attn::write_kv_explicit_bf16" as write_kv_explicit

    /// The same write over a device-carried window.
    ///
    /// `whole` and `sink` transcribed from the row. `whole` is what makes
    /// `n_max` recoverable: the grid spans the fire's FULL lane count and
    /// out-of-window rows early out on `win[0]`/`win[1]`, which is what lets
    /// a captured launch replay across row splits — so `Rows::total` is the
    /// number, and no windowed statement can reach the symbol to disagree.
    ///
    /// Bound by nothing; the `none:` arm states the one missing fact.
    WRITE_KV_EXPLICIT_DEVWIN = "attn::write_kv_explicit_bf16_devwin" as write_kv_explicit_devwin {
        whole: true,
        sink: Some("kv.pages"),
    }

    /// Dequantise the pages this batch touches into the layer's bf16 mirror.
    ///
    /// Reads and writes the cache and states no buffer operand at all: every
    /// argument is the layer's view or the fire's page table, which is why
    /// its bind is three `Cx` queries and a call.
    ///
    /// **It is a subroutine before it is a trace symbol.** Four other host
    /// programs — FA2's two decode entries and two prefill entries — call it
    /// as a prelude, because `families::fa2`'s rows carry one KV width and an
    /// fp8 cache must be bf16 before they can read it. So the moved body is a
    /// `pub fn` and this arm is a call to it, which is
    /// `x::layout::envelope_*`'s arrangement again.
    DEQUANT_KV_ACTIVE = "attn::dequant_kv_cache_layer_to_bf16_active" as dequant

    // ── FLASHINFER'S SIX, AND THE THIRD REGISTRATION SHAPE ───────────────
    //
    // These six state a contract and get NO `bind!` arm — not even a `none:`
    // one. That is deliberate and it is the shape `x::gemm`'s twelve and
    // `x::moe`'s two already take: the symbol is `Service::DriverOp` in
    // `crate::execution::SERVED`, so `x::route` answers `Route::Driver` and
    // `driver-cuda/src/bind/mod.rs`'s driver-op `match` fires it. An `Entry`
    // here — including a `none:` arm — would shadow that and mint
    // `Route::Unbound`, which refuses a live model at load; `x/gemm.rs:1145`
    // is what that costs when it happens to a working dispatch, and all six
    // of these ARE working dispatches on every model that decodes.
    //
    // WHAT THE CONTRACT IS FOR, given it names no operands and no arm.
    // Exactly the readers that cannot call: `Contract::sig` puts the symbol
    // in `x::SIGS` and therefore in `table::KERNELS`, so `table::sig`
    // resolves, `bind::abi`'s `stated()` finds nothing to state, and
    // `emit_c_shim` emits nothing. That is what RETIRES the row. A `bind!`
    // would not have done it — an arm without a contract leaves the row
    // standing and the count wrong, which is the rule this family has now
    // tested six more times.
    //
    // The `needs`/`sink`/`whole` below are transcribed from the deleted rows
    // and they are not decoration: `needs` is what makes the planner raise
    // the plan cache the arm then reads, and `sink` is what tells the
    // liveness pass the KV pages are written. Getting `needs` wrong here is
    // a `Decline::Unplanned` at run time, not a compile error.

    /// The plain paged decode.
    ///
    /// Three arities behind one symbol — `[q]`, `[q, o]`, `[q, o, lse]` —
    /// which the row spelled with `Or(Out(i), Attn(f))` and the arm spells
    /// as two `if`s over `spec.n_out`. `depth_prefix_plan` because the
    /// decode plan is built over the fire's whole depth prefix and a row
    /// window would describe a different batch than the one planned.
    FA2_DECODE = "attn::dispatch_attention_flashinfer_decode" as fa2_decode {
        needs: Prepare::DecodePlan,
        sink: Some("kv.pages"),
        depth_prefix_plan: true,
    }

    /// The same decode with the attention scores captured.
    ///
    /// `Cap::Scores` is not `lacks`ed here: the capture IS the point. What
    /// the arm refuses instead is a null sink, because a dispatch that ran
    /// anyway would be the non-capturing kernel wearing this symbol's name.
    /// AND IT DOES NOT SET `depth_prefix_plan`, WHERE ITS SIBLING DOES.
    /// Transcribed rather than tidied: the two decode rows differed on this
    /// flag and the difference is either a real fact about the capture path
    /// or a row-world omission. Deciding which is not a transcription's job
    /// and guessing it would be exactly the class this port keeps finding —
    /// so it is recorded here, where the next reader of the capture path
    /// will be standing.
    FA2_DECODE_CAPTURE = "attn::dispatch_attention_flashinfer_decode_capture" as fa2_decode_capture {
        needs: Prepare::DecodePlan,
        sink: Some("kv.pages"),
    }

    /// The plain paged prefill.
    FA2_PREFILL = "attn::dispatch_attention_flashinfer_prefill_bf16" as fa2_prefill {
        needs: Prepare::PrefillPlan,
        sink: Some("kv.pages"),
    }

    /// The prefill sibling of the capturing decode — SnapKV's half.
    ///
    /// It takes one more sink than the decode form: a prefill's raw scores
    /// and their per-request fold are different extents, so `folded_out`
    /// rides beside `score_out`.
    FA2_PREFILL_CAPTURE = "attn::dispatch_attention_flashinfer_prefill_capture_bf16" as fa2_prefill_capture {
        needs: Prepare::PrefillPlan,
        sink: Some("kv.pages"),
    }

    /// The custom-mask prefill.
    ///
    /// No causal axis anywhere below this symbol: the mask IS the
    /// causality, so a dispatch that also set `CAUSAL` would mask twice.
    /// `Prepare::CustomPlan` and NOT `PrefillPlan`, which is the row's own
    /// word: the plan is a prefill plan, but the prepare that raises it also
    /// stages the mask and its CSR, and those are what the arm reads off
    /// `AttnCtx::mask_d` / `mask_indptr_d`.
    FA2_PREFILL_CUSTOM = "attn::dispatch_attention_flashinfer_prefill_custom" as fa2_prefill_custom {
        needs: Prepare::CustomPlan,
        sink: Some("kv.pages"),
    }

    /// The PLANLESS prefill: it plans and fires in one call.
    ///
    /// `whole` and `FireWide` for the reason XQA is — it builds an R-shaped
    /// plan on the way in, so it owes its caller nothing and cannot be
    /// handed a row window.
    ///
    /// **This is the one of the six that names no driver resource**, and it
    /// is a driver op anyway: its body walks `qo_indptr_h` and
    /// `kv_page_indptr_h`, the HOST mirrors of the CSR, and no `Cx` query
    /// answers a host pointer. Reading the device copy host-side would be a
    /// synchronise and §0 says a fire is a straight line. That makes it
    /// `split_qkv_bf16_devwin`'s shape rather than `gemm`'s, and the pair is
    /// the evidence that *name the resource* is necessary and not
    /// sufficient — the second condition is whether a `Cx` can STATE what
    /// the body reads, and here it cannot.
    FA2_PREFILL_PLANLESS = "attn::attention_flashinfer_prefill" as fa2_prefill_planless {
        whole: true,
        needs: Prepare::FireWide,
        sink: Some("kv.pages"),
    }

    // ── MLA'S ABSORB PAIR — driver ops for `ctx.cublas` ──────────────────
    //
    // The second group to take the third registration shape here, and the
    // discriminator answered identically for both members, which is worth
    // saying only because the FA2 six did not. Run as a set, `ctx.cublas`
    // is the whole answer twice. Bodies above; arms in
    // `driver-cuda/src/bind`; `Service::DriverOp` in `execution::SERVED`,
    // reclassified from `Service::Cublas` on `x::gemm`'s precedent — the
    // finding about the body is unchanged and is kept in the sentence.

    /// MLA's first absorption: `q_nope` into the latent basis.
    ///
    /// Both absorbs take the WHOLE `kv_b_proj` bank and slice it themselves,
    /// which is why the row stated four widths through `Source::Param` and
    /// not through the result's extents: `heads` and `kv_lora_rank` are the
    /// result's trailing extents, `qk_nope_dim` is the operand's, and
    /// `v_head_dim` rides the param channel because this statement's result
    /// does not carry it. The arm reads all four off `LaunchSpec::params`.
    MLA_ABSORB_Q_TO_LATENT = "gemm::mla_absorb_q_to_latent_bf16" as mla_absorb_q_to_latent

    /// MLA's second absorption: the attention latent back out to `v`.
    MLA_ABSORB_LATENT_TO_V = "gemm::mla_absorb_latent_to_v_bf16" as mla_absorb_latent_to_v

    // ── THE DSA INDEXER'S THREE, and they split 1-2 ─────────────────────
    //
    // Neither `q_rope` nor `knorm_rope` needs a driver resource, so neither
    // is a driver op; all three are moves. What splits them is the SECOND
    // condition -- whether `Cx` can state what the body reads -- and here
    // the answer is not about `Cx` at all.
    //
    // `dsl::cuda::dsa_index_q_rope` records ONE input and NO parameters, and
    // puts `heads` and `head_dim` into the RESULT SHAPE only, so
    // `out_width(0)` is their product and nothing splits it. `rope_dim`
    // appears in no statement, no shape and no context. `dsa_index_knorm_\
    // rope` is worse: it names no weight bank at all, and the kernel reads a
    // LayerNorm weight AND a bias.
    //
    // That is not a missing query. `Cx` cannot answer what the trace did not
    // say, and there is no producer to ask for -- this is
    // `DSV4_COMPRESS_GATHER_PAGED`'s shape, *"what it is waiting for is a
    // statement rather than a query"*, and not `Cx::mla_layer`'s.
    //
    // A `none:` arm is safe for both by the two measured conditions
    // `x/gemm.rs:1145` requires: both rows stated `Source::Unbound` on every
    // operand, so `abi.rs:810` skipped them whole and no dispatch arm has
    // ever existed to shadow; and no model in the workspace calls either
    // wrapper. `topk_mask` is the one that binds, and it binds for the
    // reason the other two do not: its statement is COMPLETE -- three
    // inputs, three params, one result, every operand sourced.

    /// LayerNorm then interleaved RoPE on the indexer's keys.
    DSA_INDEX_KNORM_ROPE = "attn::dsa_index_knorm_rope_bf16" as dsa_index_knorm_rope {
        whole: true,
    }

    /// Interleaved RoPE on the indexer's queries.
    DSA_INDEX_Q_ROPE = "attn::dsa_index_q_rope_bf16" as dsa_index_q_rope {
        whole: true,
    }

    /// The causal top-k mask over the index scores. The one of the three
    /// whose statement carries everything its kernel reads.
    DSA_INDEX_TOPK_MASK = "attn::dsa_index_topk_mask" as dsa_index_topk_mask {
        whole: true,
    }

    /// The decode row's compressed-block boundary metadata.
    ///
    /// Three outputs from one input and a ratio. Everything but the ratio is
    /// statable, which is why the refusal is one word long.
    DSV4_BOUNDARY_META_DECODE = "attn::dsv4_boundary_meta_decode" as dsv4_boundary_meta_decode {
        whole: true,
    }

    /// The prefill form, resolving the request by binary search.
    DSV4_BOUNDARY_META_PAGED = "attn::dsv4_boundary_meta_paged" as dsv4_boundary_meta_paged {
        whole: true,
    }

    /// Attention over the COMPRESSED KV pages.
    DSV4_ATTENTION_COMPRESSED_PAGED =
        "attn::attention_compressed_paged_bf16" as attention_compressed_paged {
        whole: true,
    }

    /// The whole MLA prologue in one kernel.
    ///
    /// Its `Walk` named `heads_per_block` as `Control::Supplies`'s worked
    /// example. A `fn` supplies it — `x::attn::mla_heads_per_block`, four
    /// lines above the launch it divides.
    MLA_PREPARE = "attn::mla_prepare_bf16" as mla_prepare {
        whole: true,
    }

    /// One step's latent and rope plane, appended to the paged MLA cache.
    WRITE_MLA_TO_PAGES = "attn::write_mla_to_pages" as write_mla_to_pages {
        whole: true,
    }

    /// The page compactor — TWO launches, one stream, in order.
    ///
    /// It keeps its `execution::COMPOSED` entry: that entry is a finding
    /// about the BODY, and the body is still two ops. What went is the row.
    COMPACT_PAGE_CSR = "attn::compact_page_csr" as compact_page_csr {
        whole: true,
    }

    /// MTP's shift: the previous step's pending hidden becomes this step's
    /// first token, per request.
    MTP_SHIFT_HIDDEN = "attn::mtp_shift_hidden_bf16" as mtp_shift_hidden {
        whole: true,
    }

    /// MTP's refresh: each request's last hidden becomes its slot's pending.
    MTP_UPDATE_PENDING_HIDDEN =
        "attn::mtp_update_pending_hidden_bf16" as mtp_update_pending_hidden {
        whole: true,
    }

    /// The packed QKV split over a DEVICE-RESIDENT row window.
    ///
    /// The peel's tail states it and nothing else does — `lower.rs:1546`,
    /// `SplitQkv { .. }` under `peel_tail`. No field block, because the row
    /// carried none: it was nine `operands` and nothing else.
    ///
    /// **`as split_qkv_devwin` is the ROW's name and not the DSL's.** The DSL
    /// function is `dsl::split_qkv` for both halves — the lowering picks the
    /// symbol from the region, so one wrapper reaches two symbols — and the
    /// row this replaces was `kernel!(split_qkv_devwin ...)`. `Contract::sig`
    /// puts this string in `KernelSig::name`, so it has to be the row's or
    /// the sig changes name on the way across.
    SPLIT_QKV_DEVWIN = "attn::split_qkv_bf16_devwin" as split_qkv_devwin

    /// The per-head → per-request fold of captured attention scores.
    ///
    /// `whole` because the row was: the fold reads a CSR over the whole
    /// fire's rows and a row window would give it a prefix of one request.
    ///
    /// **This contract does not move the launcher.** The host program is
    /// `driver-cuda/src/fire/attn_score.rs`, which fires
    /// [`attention_flashinfer`]'s `attn::attn_score_fold_heads_dev` through
    /// `unit::unit_of`, and it stays there — see that module's doc for the
    /// `_dev` split this crossing had to perform first.
    ATTN_SCORE_FOLD_HEADS = "attn::attn_score_fold_heads" as attn_score_fold_heads {
        whole: true,
    }

    /// The reference paged attention — `table::attn`'s next-to-last row.
    ///
    /// `whole` and `sink: Some("kv.pages")`, both carried verbatim from the
    /// row. `whole` because `qo_indptr` is a CSR over the FIRE's rows and a
    /// row window would hand the kernel a prefix of one request; the sink
    /// because the launch reads the pages a preceding `write_kv` filled, and
    /// a sink is what orders the two without a floor combinator.
    ///
    /// **This is the only contract in this file whose row was in
    /// `device::JIT_DISPATCHED`**, and that entry goes in the same index. A
    /// row in that list gets a generated JIT arm instead of a shim call; the
    /// arm resolved `families::attn`'s device sig, which is why the two lists
    /// had to be measured together and why the crossing is one commit.
    ATTENTION_NAIVE_PAGED = "attn::attention_naive_paged" as attention_naive_paged {
        whole: true,
        sink: Some("kv.pages"),
    }
}

// ---------------------------------------------------------------------------
// What happens when a trace says it.
//
// Nine binds and two `none:` arms. Every operand `table/attn.rs` sourced for
// those nine rows is a `Cx` query that exists — some because they always
// did, `softcap`'s cap because `Facts::final_logit_softcap` was asked for and
// landed, and [`kv_paged`]'s four (`first_token`, `num_pages_in_batch`,
// `w_page_d`, `w_off_d`) because they were asked for and landed too. That is
// not true of `attention_mla`, whose four unsourced operands are four
// queries that do not exist, nor of `page_compact`, which wants buffers no
// `Source` ever spelled.
//
// **The two `none:` arms are not the same kind of gap and the difference is
// the one worth keeping.** `ATTENTION_MLA` and `WRITE_KV_EXPLICIT_DEVWIN`
// both mint an `Entry` with no arm, which is `Route::Unbound` at model load;
// `x/gemm.rs:1145` records what that costs when it shadows a working
// generated dispatch. Neither does, and neither escapes on the same
// argument: `attention_mla`'s row was never armed
// (`executor_bind.rs:1519`), and the devwin row stated `Source::Unbound` on
// all nine operands so `abi.rs:810` skipped it whole AND
// `dsl::cuda::write_kv_explicit_devwin` has no caller. **Two measurements
// per arm, made at the arm.** A `none:` arm written on the strength of "it
// probably does not fire" is the shape that refuses a model at load.
//
// `//` and never `///` inside this invocation: the arms are array elements
// and an attribute cannot precede one.
// ---------------------------------------------------------------------------

#[cfg(feature = "_cuda")]
bind! {
    LSE_LOG2_TO_LN => { cx, stream => {
        // `Source::OutElements(0)` is `rows * out_width(0)`, and both halves
        // are `Cx` queries. The result and the operand are the same buffer,
        // so reading the extent off the RESULT is reading it off the thing
        // that will be written.
        let elems = cx.rows().count.saturating_mul(cx.out_width(0)?);
        let Ok(n) = usize::try_from(elems) else {
            return Err(Refusal::Empty { what: "lse elements" });
        };
        unsafe { lse_log2_to_ln(cx.arg_out(0)?.cast::<f32>(), n, stream) }.ok()
    }},

    ATTENTION_SINK_RESCALE => { cx, stream => {
        // The deleted row's six sources, in order: `Out(0)`, `In(1)`,
        // `Weight(0)`, `Rows`, `Ctx("num_q_heads")`, `Ctx("head_dim")`.
        //
        // `In(1)` and not `In(0)`: the LSE is the dispatch's SECOND result,
        // and the statement that declares it is the only one that can.
        unsafe {
            attention_sink_rescale_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.rows().count,
                cx.num_q_heads()?,
                cx.head_dim()?,
                stream,
            )
        }
        .ok()
    }},

    ATTN_RES_BLEND => { cx, stream => {
        // The deleted row's nine sources, in order: `In(0..3)`, `Out(0)`,
        // `Rows`, `Div(Width(In(1)), Width(Out(0)))`, `OutWidth(0)`, `Rows`
        // again for `block_rows`, and `Ctx("eps")`.
        //
        // `Cx::rms_eps` IS `Ctx("eps")`: `bind/facts.rs:284` reads
        // `self.ctx.eps`, the same field the row named. The method's name
        // says which fire-wide epsilon it is, not which kernel may read it.
        //
        // `norm_weight` and `proj_weight` are `In(2)` and `In(3)` and not
        // weights: K3 states them as operands, so they are the statement's
        // and the binder resolves them like any other.
        let h = cx.out_width(0)?;
        let b = cx.in_width(1)? / h;
        unsafe {
            attn_res_blend_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_in(2)?.cast_const().cast::<bf16>(),
                cx.arg_in(3)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                b,
                h,
                cx.rows().count,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    PAD_HEAD_DIM => { cx, stream => {
        // THE PACKED SIDE IS THE INPUT ON THE WAY IN. The head count is the
        // packed side's width over `head_dim`, and the padded width is the
        // other side over that count — `PACKED_HEADS_IN` and the `Div` the
        // deleted row wrapped around it, as two divisions of numbers this
        // statement already carries.
        //
        // The head count is NOT `cx.num_kv_heads()`. See `per_head`'s doc:
        // the rule read one and the kernels address with the other, and at
        // 12 heads of 64 against 6 KV heads the difference is 6,100 of
        // 12,544 bytes.
        let head_dim = cx.head_dim()?;
        let packed_width = cx.in_width(0)?;
        let num_heads = packed_width / head_dim;
        if num_heads <= 0 {
            return Err(Refusal::Narrow { what: "in_width(0)", at: packed_width });
        }
        unsafe {
            pad_head_dim_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                num_heads,
                head_dim,
                cx.out_width(0)? / num_heads,
                stream,
            )
        }
        .ok()
    }},

    STRIP_HEAD_DIM => { cx, stream => {
        // THE PACKED SIDE IS THE OUTPUT ON THE WAY OUT, which is the whole
        // difference from the arm above and the reason the deleted rows kept
        // two constants rather than one expression written twice: a copy
        // that drifted would count heads on the PADDED side, where the
        // divisor is `head_dim_padded`, so the count comes out short and the
        // launch covers a prefix of the heads.
        let head_dim = cx.head_dim()?;
        let packed_width = cx.out_width(0)?;
        let num_heads = packed_width / head_dim;
        if num_heads <= 0 {
            return Err(Refusal::Narrow { what: "out_width(0)", at: packed_width });
        }
        unsafe {
            strip_head_dim_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                num_heads,
                head_dim,
                cx.in_width(0)? / num_heads,
                stream,
            )
        }
        .ok()
    }},

    LOGIT_SOFTCAP => { cx, stream => {
        // The deleted row's three sources, in order: `Out(0)`,
        // `CtxNonZero("final_logit_softcap")`, `OutElements(0)`. The stream
        // was a fourth and is a parameter now.
        //
        // `final_logit_softcap()` IS the `NonZero` half: the driver's impl
        // returns `None` for a cap of zero, so a deployment that states none
        // declines here with `nothing states a logit soft cap` rather than
        // scaling every logit to nothing. The `fn` re-checks it, because a
        // `fire/` caller can reach the `fn` without passing through a `Cx`.
        let cap = cx.final_logit_softcap()?;
        // In place on the value it names, so the extent is read off the
        // RESULT — the same buffer the first operand is.
        let elems = cx.rows().count.saturating_mul(cx.out_width(0)?);
        let Ok(n) = usize::try_from(elems) else {
            return Err(Refusal::Narrow { what: "logit elements", at: elems });
        };
        unsafe { logit_softcap_bf16(cx.arg_out(0)?.cast::<bf16>(), cap, n, stream) }.ok()
    }},

    KIMI_SPLIT_Q_B => { cx, stream => {
        // The deleted row's seven sources, in order: `In(0)`, `Out(0)`,
        // `Out(1)`, `Rows`, `Param(0)`, `Param(1)`, `Param(2)`. The stream
        // was an eighth and is a parameter now.
        //
        // THE EXTENT IS NOT AMONG THEM, and that is the whole of it. The
        // row's four numbers describe the LAUNCHER, which formed the device
        // kernel's `total` from them; the JIT has no launcher and so formed
        // its grid from `LaunchRule::Elementwise` instead, which is `rows *
        // out_width(0)` and is short by `rope / (nope + rope)`. The `fn`
        // takes the same four numbers the row states and forms the product
        // itself, which is the arrangement the launcher had.
        //
        // `heads`, `nope` and `rope` are `Param`s and not widths: `q_b` is
        // ONE operand of width `heads * (nope + rope)`, and no division of
        // its width recovers three numbers. The statement carries them.
        //
        // `unwrap_or(0)` on the narrowing is `x::moe`'s and `x::quant`'s
        // idiom and is safe for the same reason: a `u32` above `i32::MAX`
        // becomes zero, and the `fn` refuses zero with the extent's own
        // name. A silent narrowing would be the alternative.
        let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
        let heads = param(0)?;
        let nope = param(1)?;
        let rope = param(2)?;
        unsafe {
            kimi_split_q_b_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.rows().count,
                heads,
                nope,
                rope,
                stream,
            )
        }
        .ok()
    }},

    KIMI_SPLIT_KV_A_NORM => { cx, stream => {
        // The deleted row's nine sources, in order: `In(0)`, `Weight(0)`,
        // `Out(0)`, `Out(1)`, `Rows`, `OutWidth(0)`, `OutWidth(1)`,
        // `Ctx("eps")`, `InWidth(0)`. The stream was a tenth.
        //
        // Every extent here IS a width of the statement's own tensors, which
        // is why this half never had `kimi_split_q_b`'s problem: the two
        // results' widths ARE `kv_lora_rank` and `qk_rope_head_dim`, and the
        // operand's width IS the source stride. `Rows` is the grid.
        //
        // `norm_weight` is `Weight(0)` and not `In(1)`: kimi states the
        // latent norm's scale as a weight, unlike `attn_res_blend` one
        // contract up, which states both of its as operands. The two are
        // different statements and each is read as it was written.
        unsafe {
            kimi_split_kv_a_norm_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.rows().count,
                cx.out_width(0)?,
                cx.out_width(1)?,
                cx.in_width(0)?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    // NOT BINDS, AND THE MEASUREMENT IS IN THE STATEMENT RATHER THAN IN `Cx`.
    //
    // Both host programs are written and both are twelve lines: the geometry
    // is `route_rows(num_entries, head_dim)` and every refusal is hoisted.
    // What is missing is not a launch and not a query — it is that **the
    // trace does not name most of what these kernels read.**
    //
    // `dsl.rs:4684` records `dsv4_compress_gather_paged` with
    // `vec![boundary_pos.id]` for its inputs and `vec![]` for its params, and
    // `:4702` records `dsv4_store_comp_entries` with `vec![entries.id,
    // boundary_pos.id]` and `vec![]`. The kernels take twelve and eight
    // operands. So of the gather's five buffers a statement names ONE, and of
    // its three integers it names NONE.
    //
    // **This is the same defect `families::attn` has, one level up, and it is
    // worth naming because that file looks like it disagrees.**
    // `DSV4_COMPRESS_SIGS[4]` annotates the gather `state_kv <- Source::In(0)`
    // through `boundary_req <- Source::In(4)`, `ratio <- Source::Param(0)`,
    // `coff <- Source::Param(1)`. Those `Source`s describe a statement that
    // does not exist: `In(1)`..`In(4)` and both `Param`s have nothing behind
    // them, because `record` was handed one input and no parameters. The
    // TABLE row states no `Source` at all and is right to — and the two rows
    // have disagreed for as long as both have existed, with nothing
    // comparing them, because a device row's sources are read by nobody.
    // `driver-cuda/tests/executor_bind.rs`' UNARMED list holds both symbols,
    // which is the only place the truth was recorded.
    //
    // WHAT A BIND WOULD NEED, and it is a `dsl` change before it is a `Cx`
    // one: `state_kv` and `state_score` are deepseek_v4's compression state,
    // a slab no `StateStore` names; `ape` is a weight; `boundary_req` is
    // `dsv4_boundary_meta_*`'s SECOND output, which the statement drops on
    // the floor at `dsl.rs:4676`; `ratio` and `coff` are model config that
    // `record`'s empty parameter list refuses to carry. Every one of those is
    // a fact the STATEMENT would have to state before a query could answer
    // it, which is why neither of these is *"a fact `Cx` does not carry"* —
    // it is a statement that names one of its own operands.
    //
    // Crossing them UNBOUND costs nothing measurable and is the same trade
    // `ATTENTION_MLA` took: both symbols are on `executor_bind.rs`' UNARMED
    // list, both are in `device::JIT_DISPATCHED` so no shim entry was ever
    // emitted for either, and both table rows were unsourced so
    // `abi::emit_rust_dispatch` skipped them WHOLE and never generated an arm.
    // A row that no path fires, while holding a row table open, is worth less
    // than the contract that replaces it.
    DSA_INDEX_TOPK_MASK => { cx, stream => {
        // The deleted row's nine sources, in order: `In(0..2)`, `Out(0)`,
        // `Rows`, `Param(0..2)`, `Ctx("stream")`. Every one is a query, and
        // `Rows` twice over is deliberate -- `tokens` is the grid AND the
        // pitch of `mask`, which the host program's doc argues for.
        //
        // `top_k` rides the PARAM channel because it is a load-time number
        // no operand shape carries; `dsl::cuda::dsa_index_topk_mask`'s own
        // doc says so and `moe_align` reads it the same way.
        let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
        unsafe {
            dsa_index_topk_mask_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_in(2)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<u8>(),
                cx.rows().count,
                param(0)?,
                param(1)?,
                param(2)?,
                stream,
            )
        }
        .ok()
    }},

    DSA_INDEX_Q_ROPE => { none:
        "the indexer's query rotation is not a shape this trace states: \
         `dsl::cuda::dsa_index_q_rope` records ONE input and NO parameters, \
         and puts `heads` and `head_dim` into the RESULT SHAPE only -- so \
         `out_width(0)` is their product and nothing splits it -- while \
         `rope_dim` appears in no statement, no shape and no context at all. \
         The host program is written, in `x::attn::dsa_index_q_rope_bf16`, \
         and what it is waiting for is a statement rather than a query"
    },

    DSA_INDEX_KNORM_ROPE => { none:
        "the key half is blocked by the same statement and by one more: \
         `dsl::cuda::dsa_index_knorm_rope` names NO weight bank, and the \
         kernel reads a LayerNorm weight AND a bias -- two operands with \
         nothing to come from, on top of the `rope_dim` its sibling also \
         lacks. `head_dim` alone is statable, as `out_width(0)`. The host \
         program is `x::attn::dsa_index_knorm_rope_bf16` and is complete"
    },

    ATTN_SCORE_FOLD_HEADS => { none: "`score_indptr_d` -- the score-capture CSR, \
        which says where each request's rows begin in the folded sink. Eight of \
        the nine operands are queries that exist: `scores` is `arg_in(0)`, \
        `folded` is `arg_out(0)`, three come off `plan()`, `num_q_heads` and \
        `page_size` off `num_q_heads()` and `kv_layer()`. The ninth is an \
        `AttnCtx` field with a real producer -- `attn_score::DecodeScoreCapturePlan` \
        publishes it as an arena-stable device base -- and no `Cx` query reaches \
        it. Same shape as `first_token` and `w_page_d` before they landed, and \
        NOT `Cx::mla_layer`'s shape, which refuses because nothing fills it" },

    SPLIT_QKV_DEVWIN => { cx, stream => {
        // The deleted row's nine sources, in order: `In(0)`, `Out(0..3)`,
        // `CtxNonZero("peel_window")`, `Ctx("rows_total")`, `OutWidth(0)`,
        // `OutWidth(1)`, `Ctx("stream")`.
        //
        // `rows().total` and NOT `rows().count`. `Ctx("rows_total")` is the
        // FIRE's lane count and `grid.y` is compared against an ABSOLUTE
        // `blockIdx.y`, so under a peel the tail's own length would leave
        // every row past it unvisited -- Q, K and V keeping the previous
        // fire's bytes there. `bind/facts.rs:319` fills `total` from
        // `DispatchCtx::rows_total` and its doc names this launch.
        //
        // The four buffers are BASE pointers and `cx` hands them over as
        // such: `bind/mod.rs:3973` resolves every arg of a `_devwin` kernel
        // at row 0. Nothing here re-derives that; it is the binder's, stated
        // in the binder, and the `fn`'s safety section records the
        // precondition for a caller that is not the binder.
        //
        // `peel_window` is `Source::CtxNonZero`, a row that TESTS absence, so
        // `Cx` hands back an `Option` and `?` is the test.
        let win = cx.peel_window()?;
        unsafe {
            split_qkv_bf16_devwin(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<bf16>(),
                cx.arg_out(2)?.cast::<bf16>(),
                win.as_ptr().cast_const(),
                cx.rows().total,
                cx.out_width(0)?,
                cx.out_width(1)?,
                stream,
            )
        }
        .ok()
    }},

    COMPACT_PAGE_CSR => { none:
        "the statement declares ONE result and the kernel writes THREE CSR \
         arrays plus a scratch: `dsl::cuda::compact_page_csr` records one \
         input, no parameters, a `StateRef` and a single `[Requests] I32` \
         result, so `arg_out(0)` answers one of `page_indptr_out`, \
         `last_page_lens_out` and `page_indices_out` and there is no way to \
         say WHICH -- while `scratch_counts`, the buffer that carries the \
         dependency BETWEEN the two launches, and `keep_stride` have nothing \
         at all. Six of eleven ARE answered: `keep` is `arg_in(0)`, the three \
         CSR inputs and `num_requests` come off `plan()`. The host program is \
         `x::attn::compact_page_csr`, both launches in order with both \
         refusals hoisted ahead of the first, and it is complete"
    },

    MTP_SHIFT_HIDDEN => { none:
        "ONE operand of nine, and it is `slot_ids`: the only query that \
         reaches a request->slot map is `Cx::gdn()`, whose `slot_ids_d` is \
         exactly this pointer, and `Facts::gdn` answers `None` unless the \
         fire has a RECURRENT shape. An MTP head on a dense transformer has \
         none, so the query refuses for the fire that needs it. Everything \
         else is answered: `target_hidden` and `pending_hidden` are \
         `arg_in(0)` and `arg_in(1)` -- the statement hands the pending slab \
         over as an INPUT, so no `Slab` variant is wanted here -- `out` is \
         `arg_out(0)`, `qo_indptr` and `num_requests` come off `plan()`, \
         `total_tokens` is `rows()` and `hidden_size` is `out_width(0)`. The \
         host program is `x::attn::mtp_shift_hidden_bf16` and is complete"
    },

    MTP_UPDATE_PENDING_HIDDEN => { none:
        "its twin's `slot_ids`, and one more of a different kind: this \
         statement records NO result and a `StateRef { store: \
         RecurrentState }`, so `pending_hidden` -- which this kernel WRITES \
         -- is a slab reference rather than an argument, and `Slab` has two \
         variants, `Conv` and `Recurrent`, neither of which is the MTP \
         pending-hidden row. `RecurrentStateCache` carries it as a third \
         half, `Buffer::MtpHidden`, addressed by SLOT and not by layer, so it \
         is a slab kind rather than a stride on an existing one -- which is \
         the change `Slab`'s own doc asks for: `the next person to add a slab \
         kind adds a stride to Gdn in the same change`. `target_hidden` is \
         `arg_in(0)`, `hidden_size` is `in_width(0)`, `qo_indptr` and \
         `num_requests` come off `plan()`. The host program is \
         `x::attn::mtp_update_pending_hidden_bf16` and is complete"
    },

    MLA_PREPARE => { none:
        "`Cx::mla_layer` refuses, and it is the whole blocker: the two page \
         arrays, `page_size`, `kv_lora_rank` and `qk_rope_head_dim` all come \
         out of one view, so five of this kernel's thirty operands go \
         together or not at all. That query's refusal is STRUCTURAL and its \
         own doc says so -- `AttnCtx` carries `layers: Vec<KvCacheLayerView>` \
         and no MLA equivalent, and the views come from \
         `pools::mla_cache::MlaCachePool::layer_view`, which no `Fire` can \
         reach. This is `ATTENTION_MLA`'s refusal, one kernel earlier in the \
         same lane, and it is a DIFFERENT SHAPE from the `dsv4` three's \
         ratio: the ratio has no producer anywhere, and this has a producer \
         no fire reaches. Everything else is answered -- the four query \
         outputs and two KV outputs are `arg_out(0..5)`, `kv_a`/`q_b` are \
         `arg_in`, the norm weight is `weight(0)`, the four CSR arrays and \
         `row_valid` come off `plan()`, `eps` is `rms_eps()`, `theta` is \
         `rope_theta()`, `interleaved` is `rope_interleaved()` and `yarn` is \
         `yarn()`. The host program is `x::attn::mla_prepare_bf16` and is \
         complete"
    },

    WRITE_MLA_TO_PAGES => { none:
        "the same view, and nothing else missing: this kernel's thirteen \
         operands are two inputs, the four CSR arrays, `row_valid`, \
         `num_requests` -- all answered -- and the five that ARE the layer \
         view. `serve/load.rs` refuses every MLA checkpoint at load today, so \
         the refusal this states is the one a model would meet anyway, one \
         layer lower and in a sentence. The host program is \
         `x::attn::write_mla_to_pages` and is complete"
    },

    DSV4_BOUNDARY_META_DECODE => { none:
        "the compression RATIO is not a value this trace carries: \
         `dsl::cuda::dsv4_boundary_meta` records its inputs with \
         `record_many` and NO parameters, so the one integer the kernel \
         DIVIDES BY has no operand — and it appears in no `AttnCtx` field, \
         no `DispatchCtx` field and no `Facts` query either, so there is \
         nothing to answer it with. Everything ELSE is statable: `positions` \
         is `arg_in(0)`, the three outputs are `arg_out(0..2)`, `row_valid` \
         and `requests` come off `plan()`, and `n` is `rows()`. The host \
         program is `x::attn::dsv4_boundary_meta_decode` and is complete"
    },

    DSV4_BOUNDARY_META_PAGED => { none:
        "its twin's ratio, and nothing else: `qo_indptr` and `num_requests` \
         BOTH come off `plan()`, so the prefill form's two extra operands \
         are the two that are already answered. One statement carries both \
         rows -- `dsl::cuda::dsv4_boundary_meta` -- so a parameter added \
         there lands on both at once. The host program is \
         `x::attn::dsv4_boundary_meta_paged` and is complete"
    },

    DSV4_ATTENTION_COMPRESSED_PAGED => { none:
        "the same ratio and two buffers with no producer anywhere: \
         `comp_kv_pages` is deepseek_v4's COMPRESSED cache, which no pool \
         allocates and no context carries, and `req_of_token` is a \
         per-token request map that nothing in `driver-cuda` builds. \
         `sm_scale` is the one blocker of a different kind -- it HAS a \
         producer, `AttnCtx::sm_scale` at `bind/mod.rs:1489`, and six \
         generated arms read it -- so it is a query that could exist and \
         does not, where the other three are values that do not exist. \
         `q`, `o`, `lse_out`, `positions`, the two page arrays, \
         `total_tokens`, `num_q_heads`, `head_dim` and `page_size` are all \
         answered today. The host program is \
         `x::attn::attention_compressed_paged_bf16` and is complete"
    },

    DSV4_COMPRESS_GATHER_PAGED => { none:
        "deepseek_v4's compression state is not a value this trace names: \
         `dsl::cuda::dsv4_compress_gather_paged` records ONE input \
         (`boundary_pos`) and NO parameters for a kernel that reads five \
         buffers and three integers, so `state_kv`, `state_score`, `ape`, \
         `boundary_req`, `ratio` and `coff` have no operand to come from — \
         the host program is written, in `x::attn::dsv4_compress::\
         dsv4_compress_gather_paged_bf16`, and what it is waiting for is a \
         statement rather than a query"
    },

    DSV4_STORE_COMP_ENTRIES => { none:
        "the commit half is blocked by the same statement as the gather: \
         `dsl::cuda::dsv4_store_comp_entries` names `entries` and \
         `boundary_pos`, and the kernel also reads `boundary_req` — \
         `dsv4_boundary_meta_*`'s second output, which the trace discards — \
         and needs `head_dim` and `page_size` besides; the host program is \
         `x::attn::dsv4_compress::dsv4_store_comp_entries_bf16`"
    },

    COMBINE_ATTN_OUTPUTS => { cx, stream => {
        // The deleted row's ten sources, in order: `In(0)`, `In(1)`, `In(2)`,
        // `In(3)`, `Out(0)`, `Out(1)`, `Rows`, `Param(0)`, `Param(1)`. The
        // stream was a tenth and is a parameter now.
        //
        // EVERY ONE OF THEM IS SOURCED FROM THE STATEMENT, which is what
        // makes this the cheapest bind in the family: no layer, no
        // workspace, no plan, no device attribute. It was a `Walk` for one
        // reason and one only — `execution.rs`' `Control::Supplies`, the
        // BLOCK width — and a `fn` supplies its own geometry, so the
        // classification had nothing left to describe.
        let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
        let num_heads = param(0)?;
        let head_dim = param(1)?;
        unsafe {
            combine_attn_outputs_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.arg_in(2)?.cast_const().cast::<bf16>(),
                cx.arg_in(3)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                num_heads,
                head_dim,
                stream,
            )
        }
        .ok()
    }},

    QKV_PACKED_POST => { cx, stream => {
        // The deleted row's twenty-one sources, in order: `In(0)`, `Out(0)`,
        // `KvLayerField(k_pages)`, `KvLayerField(v_pages)`, `Weight(0)`,
        // `Weight(1)`, `Positions`, `Attn(kv_page_indices_d)`,
        // `Attn(kv_page_indptr_d)`, `Attn(kv_last_page_lens_d)`,
        // `Attn(row_valid_d)`, `Rows`, `Div(Width(Out(0)),
        // KvLayerField(head_dim))`, `KvLayerField(num_kv_heads)`,
        // `KvLayerField(head_dim)`, `KvLayerField(page_size)`,
        // `KvLayerField(hnd_layout)`, `CtxByLayer(theta)`, `Ctx(eps)`, and
        // the stream.
        //
        // SIX OF THEM ARE ONE QUERY. `Cx::kv_layer` returns the same five
        // fields `Source::KvLayerField` spelled one string at a time, plus
        // `hnd` — so a family that wanted six lookups pays for one, which is
        // the argument `Cx::gdn` makes for `ssm`'s eleven.
        //
        // FOUR MORE ARE ANOTHER. `Cx::plan` carries the three CSR arrays and
        // `row_valid` together, and they ARE together: the mask indexes the
        // same rows the CSR describes, and a fire that published one
        // published all four.
        //
        // `num_q_heads` off the RESULT and not the operand, exactly as the
        // row had it: `packed` is `[N, q + 2·kv]` and cannot say where the
        // cut falls; `q_out` is `[N, q]` and can. Dividing by the LAYER's
        // head_dim rather than the context's is the row's choice too — the
        // pages decide the width, because the pages are what is written.
        let layer = cx.kv_layer()?;
        let plan = cx.plan()?;
        if layer.head_dim <= 0 {
            return Err(Refusal::Empty { what: "head_dim" });
        }
        let num_q_heads = cx.out_width(0)? / layer.head_dim;
        unsafe {
            qkv_fused::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.weight(1)?.cast_const().cast::<bf16>(),
                cx.positions()?,
                plan.kv_page_indices,
                plan.kv_page_indptr,
                plan.kv_last_page_lens,
                plan.row_valid,
                cx.rows().count,
                num_q_heads,
                layer.num_kv_heads,
                layer.head_dim,
                layer.page_size,
                layer.hnd,
                cx.theta()?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    // NOT A BIND, AND THE MISSING THING IS VOCABULARY RATHER THAN A HOST
    // PROGRAM. Both arms of this symbol are now Rust and both are reachable
    // from a bind body's crate or one above it:
    //
    //   FA2, everything below sm_100 — `mla_fa2` in THIS FILE. Unit, six
    //   instantiations, `MlaParams` packer, `arm_for`, `grid`, and a
    //   cooperative `fire_ex`. Nothing about it is pending.
    //
    //   NAIVE, sm_100 only — `driver-cuda/src/fire/mla_naive.rs`, a
    //   driver-side host program in the crate ABOVE this one, the same
    //   shape as `fire/xqa.rs`. `attention_mla.cu:150` picks it on
    //   `cudaDevAttrComputeCapabilityMajor >= 10`, because FA2 MLA writes
    //   ZERO OUTPUT on sm_100 — a wrong answer, not a fault.
    //
    // THE FOUR `Cx` QUERIES THIS ARM ASKED FOR ALL LANDED, and two of them
    // answer. `Cx::attn_workspace` and `Cx::sm_scale` are implemented from
    // `AttnCtx::workspace` and `AttnCtx::sm_scale` — and they were `x/xqa.rs`'
    // ask as much as this one, two families naming the same two lines because
    // both rows carried `Source::Attn(..)`.
    //
    // `Cx::mla_layer` and `Cx::mla_plan` exist and REFUSE, and the reason is
    // one absence rather than two gaps. MEASURED:
    //
    //   * `fire/launch.rs:1521`'s `kv_pools_for` matches on `dep.kv` and
    //     returns `PIE_STATUS_UNSUPPORTED` for `KvStyle::Mla` before any pool
    //     is built. Its own comment is why the match is there: a missing MLA
    //     branch was *"not a `todo!()`, not a refusal, just an absence, which
    //     is how the MLA lineage loaded, reported itself healthy and would
    //     have died at its first fire."*
    //   * `pools::mla_cache::MlaCachePool` has ZERO callers in `driver-cuda`.
    //   * `serve/load.rs:397` refuses every MLA checkpoint AT MODEL LOAD:
    //     *"this checkpoint attends through a latent ckv/kpe pair, which this
    //     driver does not build — `pools::mla_cache` is ported and has no
    //     forward path to serve."*
    //
    // So `AttnCtx` carrying an `mla_layers: Vec<MlaCacheLayerView>` beside its
    // `layers` is three edits and an afternoon — and there would be NOTHING
    // TO FILL IT WITH. `crate::plan::mla` is in the same position one level
    // up: it builds a plan against a cache that is never materialised, which
    // is why `mla_plan` refuses too rather than being the easier of the two.
    // The task is provisioning the MLA cache — a `KvState` shape, a growth
    // path, capture-stable bases, a `views()` equivalent — and that is the
    // MLA LINEAGE, not this seam.
    //
    // WHICH IS WHY THIS ARM COSTS NOTHING TODAY, and the check is one line:
    // `load.rs:397` refuses the checkpoint before any trace exists to name
    // this symbol. Same shape as `x/xqa.rs`' decode arm, which refuses
    // nothing because every deployment states `xqa_decode: false`.
    //
    // The FIFTH fact is the arm predicate and it is NOT a `Cx` query: the
    // compute capability is a property of the device, not of the fire, and no
    // other fn-world body has needed one. It belongs beside `num_sm` in
    // whatever the runtime grows for device attributes. It is not on the
    // critical path — with no MLA cache there is no fire to choose an arm for
    // — but it is the one of the three that is genuinely small, and stating
    // it here keeps it from being rediscovered.
    //
    // ONE THING TO SETTLE WHEN THE CACHE LANDS: `Cx::attn_workspace` hands
    // back `AttnCtx::workspace`, the DECODE plan's, and deliberately does not
    // guess between that and `prefill_workspace`. `AttnCtx::prefill_workspace`
    // states the rule it is protecting — *"a launcher must take the workspace
    // its own plan was raised in"* — and MLA raises its own plan through
    // `MlaPlanCache`, so which of the two it must take is a question for
    // whoever wires `mla_plan`, and the answer may be a second query rather
    // than a smarter first one.
    ATTENTION_MLA => { none: "attention over the latent cache cannot be bound         because this driver does not build one: `fire/launch.rs`' `kv_pools_for`         refuses `KvStyle::Mla` and `serve/load.rs` refuses the checkpoint at         model load, so `Cx::mla_layer` and `Cx::mla_plan` have nothing to         answer with — both host programs are written, in `x::attn::mla_fa2` and         `driver-cuda/src/fire/mla_naive.rs`, and choosing between them needs a         device compute capability besides" },

    // The paged append. `Cx::plan()`'s first caller in this family and its
    // first anywhere — six fields that were six `Source::Attn(..)` operands
    // on the row, read together because they describe one thing.
    //
    // `first_token` is deliberately NOT a seventh field of `Plan`: a plan is
    // what the fire's geometry is; `first_token` is where a partial write
    // resumes from, which is a property of this append and of nothing else.
    WRITE_KV_TO_PAGES => { cx, stream => {
        let layer = cx.kv_layer()?;
        let plan = cx.plan()?;
        unsafe {
            kv_paged::write_kv_to_pages(
                &layer,
                cx.arg_in(0)?.cast::<bf16>().cast_const(),
                cx.arg_in(1)?.cast::<bf16>().cast_const(),
                plan.qo_indptr,
                plan.kv_page_indices,
                plan.kv_page_indptr,
                plan.kv_last_page_lens,
                cx.rows().count,
                plan.requests,
                stream,
                plan.row_valid,
                cx.first_token()?,
            )
        }
        .ok()
    }},

    // The explicit append. `w_page_d` and `w_off_d` are null-checked in the
    // query, so `?` here is the row's `Source::AttnNonZero` — a fire that
    // appends no KV refuses before the launch rather than indexing a null.
    // That is the whole of what `AttnNonZero` did, and it is now a property
    // of the fact rather than of the row that asked for it.
    WRITE_KV_EXPLICIT => { cx, stream => {
        let layer = cx.kv_layer()?;
        let plan = cx.plan()?;
        unsafe {
            kv_paged::write_kv_explicit_bf16(
                &layer,
                cx.arg_in(0)?.cast::<bf16>().cast_const(),
                cx.arg_in(1)?.cast::<bf16>().cast_const(),
                cx.w_page_d()?,
                cx.w_off_d()?,
                cx.rows().count,
                stream,
                plan.row_valid,
            )
        }
        .ok()
    }},

    // The reference paged attention. Fourteen `Source`s on the row; fourteen
    // `Cx` queries here, and the two that landed last -- `logits_soft_cap`
    // and `lse_out` (`247e78a99`) -- are the ones that took it from
    // "measured" to "written".
    //
    // `q_width` rather than `num_q_heads`: the head count is `in_width(0) /
    // head_dim` and the DIVISION belongs in the body, beside the refusal that
    // makes it safe. Handing the arm a pre-divided count would put the zero
    // check somewhere a reader of the `fn` cannot see it.
    ATTENTION_NAIVE_PAGED => { cx, stream => {
        let layer = cx.kv_layer()?;
        let plan = cx.plan()?;
        unsafe {
            attention_naive_paged(
                &layer,
                cx.arg_in(0)?.cast::<bf16>().cast_const(),
                cx.arg_out(0)?.cast::<bf16>(),
                plan.qo_indptr,
                plan.kv_page_indices,
                plan.kv_page_indptr,
                plan.kv_last_page_lens,
                cx.rows().count,
                plan.requests,
                cx.in_width(0)?,
                cx.window_left()?,
                cx.sm_scale()?,
                cx.logits_soft_cap()?,
                cx.lse_out()?,
                stream,
            )
        }
        .ok()
    }},

    QKV_DECODE_FUSED => { cx, stream => {
        let layer = cx.kv_layer()?;
        let plan = cx.plan()?;
        // RECOVER THE NULL. DO NOT `?` THESE TWO.
        //
        // `Cx::w_page_d`/`w_off_d` null-check INSIDE the query and answer
        // `Refusal::Absent` — correct for the `kv_paged` rows they were
        // landed for, whose operands are `Source::AttnNonZero`, and a FALSE
        // REFUSAL here, whose row states plain `Source::Attn`. The device
        // text is the authority and it is explicit:
        // `qkv_fused.cuh:182` reads
        // `if (w_page != nullptr && w_off != nullptr)` and takes the CSR
        // path when they are null. A `?` would refuse a fire the kernel
        // handles, so the query's `Err` is folded back to the null it was
        // derived from.
        let w_page = cx.w_page_d().unwrap_or(core::ptr::null());
        let w_off = cx.w_off_d().unwrap_or(core::ptr::null());
        let head_dim = layer.head_dim;
        let num_kv_heads = layer.num_kv_heads;
        // `Source::Div(&Source::Sub(&PACKED_W, &KV_BANKS), &HEAD_DIM)`, and
        // NOT `cx.num_q_heads()`. That query exists and answers the FIRE's
        // head count; this operand is what is left of the packed row after
        // the two kv banks come off, which is the same number only while the
        // fire and the layer agree. The row derived it and so does this.
        let num_q_heads = (cx.in_width(0)? - 2 * num_kv_heads * head_dim) / head_dim;
        unsafe {
            qkv_fused::qkv_decode_qk_norm_rope_write_kv_bf16(
                cx.arg_in(0)?.cast::<bf16>().cast_const(),
                cx.q_out()?.cast::<bf16>(),
                layer.k_pages.cast::<bf16>(),
                layer.v_pages.cast::<bf16>(),
                cx.weight(0)?.cast::<bf16>().cast_const(),
                cx.weight(1)?.cast::<bf16>().cast_const(),
                cx.positions()?,
                cx.arg_in(1)?.cast::<f32>().cast_const(),
                plan.kv_page_indices,
                plan.kv_page_indptr,
                plan.kv_last_page_lens,
                w_page,
                w_off,
                plan.row_valid,
                cx.rows().count,
                num_q_heads,
                num_kv_heads,
                head_dim,
                layer.page_size,
                layer.hnd,
                cx.theta()?,
                cx.rms_eps()?,
                stream,
            )
        }
        .ok()
    }},

    // NOT A BIND, AND THE MISSING FACT IS ONE POINTER.
    //
    // Seven of this symbol's nine operands are `Cx` queries that answer:
    // `kv_layer`, `arg_in(0)`, `arg_in(1)`, `w_page_d`, `w_off_d`,
    // `plan().row_valid`, and `n_max` as `rows().total` because the contract
    // is `whole`. The eighth is `stream`. **The ninth is `win_d`, and
    // `AttnCtx` has no window array of any kind** — which is exactly what
    // separates it from the four queries that landed in `666fbbeee`: those
    // had a producer waiting and were missing only the query; this is
    // missing the producer.
    //
    // AND A `none:` ARM IS SAFE HERE, WHICH IS NOT NORMALLY TRUE. An `Entry`
    // with no bind is `Route::Unbound` and refuses the model at load, which
    // is why `x/gemm.rs:1145` forbids one over a working dispatch. There is
    // no dispatch here to shadow, measured twice:
    //
    //   * `table/attn.rs:292` stated `Source::Unbound` on ALL NINE operands,
    //     and `abi.rs:810` skips a row with any `Unbound` operand WHOLE — so
    //     `emit_rust_dispatch` never wrote an arm, the `RUST_SERVED` entry
    //     was never reached and the `bind::service` shim never had a caller.
    //     `execution.rs`' `Walk` said the row was "fully sourced"; that was
    //     true of the SIBLING `write_kv_explicit` row and written for this
    //     one, and it is corrected in place there.
    //   * `dsl::cuda::write_kv_explicit_devwin` (`model-compiler`) has ZERO
    //     callers, so no model emits the statement at all.
    //
    // The host program is written and complete —
    // `x::attn::kv_paged::write_kv_explicit_bf16_devwin`, both instantiation
    // arms, both throws and the one decline. It needs one fact and one
    // caller, in that order.
 WRITE_KV_EXPLICIT_DEVWIN => { none: "the device-carried window has no producer: `AttnCtx` states `w_page_d` and `w_off_d` but no window array, so `win_d` is missing its FILL and not merely its query — unlike `first_token`, `num_pages_in_batch`, `w_page_d` and `w_off_d`, which `AttnCtx` had carried since before fn-world existed. The host program is `x::attn::kv_paged::write_kv_explicit_bf16_devwin` and is complete" },

    // The active-page dequant. Three queries and a call, because the row
    // stated no buffer operand: everything it reads is the layer's view or
    // the fire's page table.
    DEQUANT_KV_ACTIVE => { cx, stream => {
        let layer = cx.kv_layer()?;
        unsafe {
            kv_paged::dequant_kv_cache_layer_to_bf16_active(
                &layer,
                cx.plan()?.kv_page_indices,
                cx.num_pages_in_batch()?,
                stream,
            )
        }
        .ok()
    }},
}
