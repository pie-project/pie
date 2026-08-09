//! Attention: the paged dispatches, the KV writes, MLA, DSA and the sinks.
//!
//! **EMPTY, AND THAT IS THE POINT.** Forty-one rows when §5 step 5 reached
//! this family; zero now. [`KERNELS`] is `&[]`, `table::ROW_TABLES` is `&[]`,
//! and [`crate::table::KERNELS`] is [`crate::x::SIGS`] alone — which is what
//! north star step 5 finishing means, stated by a list rather than by prose.
//!
//! # Why the module is still here
//!
//! Deleting it is one line in `table/mod.rs` and one `git rm`, and it is
//! deliberately NOT this commit's. Two reasons, both from the code:
//!
//! * **`driver-cuda/tests/launch_abi.rs` reads `table::attn::KERNELS` by
//!   path**, at `:540` and `:809`. Both tests are the ROW WORLD's — one
//!   compiles the generated C shim for this family's rows, the other checks
//!   every launcher declared in `csrc/src/attn/*.hpp` has a row or a stated
//!   reason. Retiring them is step 6's work, and `csrc/**` is being
//!   internalized by another pass as this lands, so the second test's INPUT
//!   is moving too. An empty list keeps both compiling and makes the first
//!   vacuous rather than wrong.
//!
//! * **The tombstones below are the record of forty-one crossings** — what
//!   each `Source` became, which refusals were hoisted, which `.cuh` line
//!   settled a disagreement. Git history is not a reader. When a future pass
//!   asks why `attn::pad_head_dim_bf16` divides on the padded side, the
//!   answer is in this file and not in a diff.
//!
//! The three `Source` constants this file carried — `HEAD_DIM`
//! (`KvLayerField("head_dim")`), `PACKED_W` (`Width(In(0))`) and `KV_BANKS`
//! (`2 * num_kv_heads * head_dim`) — went with the last row. They are now
//! three `let` bindings in `x::attn`'s `QKV_DECODE_FUSED` bind body, where
//! `num_q_heads` is still `(packed_width - 2 * kv_heads * head_dim) /
//! head_dim` and still derived rather than read off `Cx::num_q_heads`.
//!
//! One row per launcher symbol, while there were rows. The words a row was
//! written in — [`KernelSig`], `whole`, `needs`, `lacks`, `sink` — are
//! `kernels`'.

use kernels::KernelSig;

#[rustfmt::skip]
pub static KERNELS: &[KernelSig] = &[
    // ── `attn/dsa_indexer.cu`'S THREE ROWS ARE DELETED ──────────────────
    //
    // `attn::dsa_index_knorm_rope_bf16`, `attn::dsa_index_q_rope_bf16` and
    // `attn::dsa_index_topk_mask`. All three are `x::attn` contracts now,
    // with the host programs beside `attn/dsa_indexer.cuh`'s unit.
    //
    // WHERE THE `Source`s WENT, and the three split 1-2. `topk_mask` stated
    // all nine -- `In(0..2)`, `Out(0)`, `Rows`, `Param(0..2)`, stream -- and
    // every one is a `Cx` query, so it binds. The other two stated
    // `Source::Unbound` on everything and their arms are `none:`, because
    // what they lack is a STATEMENT: `dsl::cuda::dsa_index_q_rope` records
    // one input and no params and puts `heads` and `head_dim` into the
    // result SHAPE, where `out_width(0)` is their product and nothing splits
    // it; `rope_dim` is in no shape, no param and no context; and
    // `dsa_index_knorm_rope` names no weight bank for a kernel that reads a
    // LayerNorm weight AND a bias.
    //
    // The rows recorded that split without being able to act on it, which is
    // what an unsourced operand IS. Three rows, one bind, two honest
    // refusals, and no model in the workspace calls any of the three
    // wrappers today.
    // ── MLA'S ABSORB PAIR IS DELETED — TWO cuBLAS DRIVER OPS ────────────
    //
    // `gemm::mla_absorb_q_to_latent_bf16` and `gemm::mla_absorb_latent_to_v_\
    // bf16` stood here, eight operands each and every one sourced, so both
    // had live generated arms. They are `x::attn`'s contracts now and
    // `Service::DriverOp` in `execution::SERVED`; the arms are hand-written
    // in `driver-cuda/src/bind`.
    //
    // WHERE THE `Source`s WENT. `In(0)`, `Weight(0)` and `Out(0)` off
    // `BoundLaunch::args`; `Rows` off the bound row count; and all four
    // widths off `LaunchSpec::params`, which is where `Source::Param(i)`
    // always read them. That last one is why these two were easy where the
    // FA2 six were long: `Param` is already a positional read of a vector
    // the arm holds, so the hand resolution IS the generated one.
    //
    // The row carried a note worth keeping, because it is about the ROW
    // grammar and not about the kernel: *"NO `handle: CublasHandle` -- §45.
    // `execution::RUST_SERVED` names both absorbs, so their bodies are
    // `driver-cuda`'s and the handle comes off the dispatch context rather
    // than out of the row."* Both halves of that are now false in the same
    // direction: the bodies are `x::attn`'s, and the handle is the driver-op
    // arm's first argument. A row could never have stated it either way --
    // which is the whole content of §3.3.
    // ── FLASHINFER'S SIX ROWS ARE DELETED — THEY CROSSED AS DRIVER OPS ──
    //
    // `attn::dispatch_attention_flashinfer_decode`, `..._decode_capture`,
    // `..._prefill_bf16`, `..._prefill_capture_bf16`, `..._prefill_custom`
    // and `attn::attention_flashinfer_prefill` stood here — the four longest
    // rows in the table, 13 to 19 operands each, and between them the whole
    // vocabulary `Source` was grown for: `Or(Out(i), Attn(f))` for the three
    // arities, `AttnPlan`, `AttnWindow`, `AttnNonZero`, `KvLayerField`, and
    // `Div(Width(In(0)), KvLayerField("head_dim"))` for a head count nobody
    // carries.
    //
    // They are `crate::x::attn`'s contracts now and `Service::DriverOp` in
    // `crate::execution::SERVED`, which is the third registration shape:
    // no `Entry`, not even a `none:` arm, because an `Entry` here would
    // shadow the driver arm and refuse a live model at load. The RESOURCE
    // is `DecodePlanCache` / `PrefillPlanCache` for five of the six, and
    // for `attention_flashinfer_prefill` it is nothing — that one is a
    // driver op because it walks the CSR's HOST mirrors, which no `Cx`
    // query can answer. See `execution::SERVED` for the full finding.
    //
    // WHERE EVERY `Source` WENT, because this is the part a deletion loses:
    // each is resolved by hand in `driver-cuda/src/bind/mod.rs`'s six
    // `fa2_*` arms, off the same three places the generated arm read —
    // `BoundLaunch::args` for `In`/`Out`/`Width`, `LaunchSpec` for `n_in`,
    // `n_out` and the join facts, and `AttnCtx` for everything else. Two
    // things the generated arm did that the hand arms had to be told:
    // `AttnPlan` is `bind::attn_plan`, hoisted out of `dispatch_generated`
    // because it is a DECISION (gemma-4's two plan kinds) and six copies of
    // a decision is the shape this port keeps finding; and every row also
    // carried `spec.aux.is_empty() && spec.per_head_dim.is_none()`, join
    // facts no `Source` can name, which are now a `ShapeDeclined` carrying
    // a sentence.
    //
    // The rows had to go in the same index as the arms. A row still here
    // routes `Route::Rows`, and `Route::Rows` against a symbol whose only
    // dispatch is now a driver-op arm is a `NoArm` on every decode fire.
    // `attn::attention_naive_paged`'s ROW STOOD HERE -- *"head dims
    // flashinfer's prefill template rejects (gemma-4's 512) take a naive
    // paged kernel instead. No plan at all; fire-shaped."*
    // `crate::x::attn::ATTENTION_NAIVE_PAGED` is a `contract!` with a `bind!`
    // and `crate::x::attn::attention_naive_paged` is the host program.
    //
    // THIS ROW WAS IN `device::JIT_DISPATCHED` AND NO OTHER ROW THIS FILE
    // CROSSED WAS, which is why its deletion is three edits rather than one:
    // the row, the `JIT_DISPATCHED` line, and `families::attn`'s unit. A row
    // in that list gets a generated JIT arm instead of a shim entry, so the
    // arm resolved the DEVICE sig -- twenty-three operands -- while the row
    // above states fourteen. `driver-cuda/build.rs`'s `armless` check reads
    // the pair, and leaving either behind is a build failure naming the other.
    //
    // Fourteen `Source`s, and every one is a `Cx` query. The last two landed
    // in `247e78a99` (`logits_soft_cap`, `lse_out`); the four before them
    // landed for `kv_paged`; `sm_scale` and `window_left` were already there.
    // That is the whole of what "blocked" meant here, and it was never the
    // geometry.
    //
    // TWO OPERANDS THIS ROW STATED DO NOT CROSS A LAUNCH AT ALL. `stream` is
    // `cuLaunchKernel`'s sixth parameter rather than an argument, and
    // `num_pages_in_batch` is CAST TO `void` by the launcher
    // (`attention_naive_paged.cu:193`) -- stated, sourced, and read by
    // nothing. A row that carries an operand its launcher discards is
    // describing a C++ signature rather than a launch, and a `fn`'s parameter
    // list has nowhere to put one.
    // XQA's row IS DELETED and its symbol has crossed to fn-world:
    // `crate::x::xqa::XQA_DECODE_BF16_PREPARED`, declared beside the five
    // `Unit`s that compile `attn/attention_xqa_mha.cuh` and the
    // `KVCacheList<true>` mirror that made their parameter list
    // expressible. What the row said, kept because the contract cannot
    // carry it: *"its prepare is fire-wide (R-shaped), so the kernel cannot
    // be given a row window — `whole`. And no capture variant of it exists,
    // so it cannot publish scores — `lacks Scores`. Both are hand-written
    // rules today: the first is the model body's `window_one &&
    // c.xqa_decode` test, the second a C++ throw."* `whole`, `needs` and
    // `lacks` are `Contract` fields and survive verbatim; the C++ throw does
    // not, because the file that threw it is deleted.
    //
    // THIS DELETION IS WHAT PERMITTED THE SIX `.cu` TO GO. The symbol
    // stated `operands`, was in neither `device::JIT_DISPATCHED` nor
    // `execution::RUST_SERVED`, and therefore kept a `pie_k_xqa_decode`
    // entry out of `abi::emit_c_shim` — of which
    // `src/attn/attention_xqa.cu` was the definition. Deleting the file
    // without deleting this row is a link error; deleting this row without
    // adding the contract is `Route::Unknown` at model load. Three atomic
    // edits, and this is the third.
    // THE LAST ROW IN `ROW_TABLES` CROSSED, AND `ROW_TABLES` IS NOW `&[]`.
    //
    // `attn::qkv_decode_qk_norm_rope_write_kv_bf16`, twenty-three operands,
    // is `crate::x::attn`'s `QKV_DECODE_FUSED` — a `contract!` and a `bind!`
    // over the `attn/qkv_fused` unit's eleven device rows, with its host
    // program in `x::attn::qkv_fused` since `f998bdaba`.
    //
    // IT WAS A MOVE, NOT A DRIVER OP, AND BOTH CONDITIONS WERE MEASURED.
    // `bind::service`'s entry point took `_ctx: &DispatchCtx` and never read
    // it; the body needs no cuBLAS handle, no communicator, no pool, no
    // allocator and no arena, so there is no resource to name. And the second
    // condition — whether `Cx` can STATE what the body reads — was the whole
    // of the wait: twenty-two operands had queries and `q_out` had none.
    //
    // WHAT THE LAST OPERAND SETTLED, because it outlives the row. The
    // producer (`fire/launch.rs:3248`) writes `null_mut()` when a fire pins
    // no query buffer; this row said plain `Source::Attn("q_out")`, which
    // ASSERTS presence; and `qkv_fused.cuh:177` stores through the pointer
    // with no null test while `:182`, two arguments along, DOES test
    // `w_page`/`w_off`. Three sources, and the row was the one that was
    // wrong. **When the row and the producer disagree, the producer is the
    // fact and the row is a claim** — a row cannot make a pointer non-null,
    // it can only be believed or checked, and the device text says which.
    //
    // AND THE BIND RECOVERS TWO NULLS RATHER THAN REFUSING THEM.
    // `Cx::w_page_d`/`w_off_d` null-check inside the query, which is right
    // for the `kv_paged` rows they were landed for (`Source::AttnNonZero`)
    // and a false refusal for this one (plain `Source::Attn`, and the kernel
    // takes the CSR path when they are null). The arm folds the `Err` back
    // to the null it was derived from, with the `.cuh` line beside it.
    //
    // WHAT WENT IN THE SAME INDEX, derived from every list that can name a
    // symbol rather than from this file alone:
    //
    //   table/attn.rs           this row                        (1 of 1)
    //   execution::WALKED       this symbol                     (1 of 1)
    //   execution::RUST_SERVED  this symbol                     (1 of 3)
    //   bind/service.rs:884     `attn_qkv_decode_qk_norm_rope_write_kv_bf16`
    //   fire/qkv_fused.rs       the file, and `fire/mod.rs`'s `pub mod`
    //
    // `WALKED` and `RUST_SERVED` had to go WITH the row and not after it:
    // `every_taken_over_row_is_stated` panics on a `RUST_SERVED` symbol with
    // no row, and `every_rust_served_symbol_is_spelled_here` reads
    // `include_str!("service.rs")`. The symbol was in NEITHER `SERVED`,
    // `COMPOSED` nor `device::JIT_DISPATCHED` — checked, not assumed.
    //
    // ONE FIELD DID NOT SURVIVE VERBATIM AND THE CONTRACT SAYS SO: this row
    // defaulted `sink: None` while its prefill sibling
    // `qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` stated
    // `Some("kv.pages")`, over the same fusion and the same pages. The
    // contract states the sink. §75's shape, in a default nobody set.
    // THE THREE APPEND ROWS CROSSED INTO FN-WORLD — `crate::x::attn`'s
    // `WRITE_KV_EXPLICIT`, `WRITE_KV_TO_PAGES` and
    // `WRITE_KV_EXPLICIT_DEVWIN`, over the `attn/kv_paged` unit's twenty
    // device rows, with their host programs in `x::attn::kv_paged`.
    //
    // What each `Source` became, because the mapping is the point:
    //
    //   Source::KvLayerView               -> Cx::kv_layer(), seventeen fields
    //   Source::In(0) / In(1)             -> Cx::arg_in(0) / arg_in(1)
    //   Source::Rows                      -> Cx::rows().count
    //   Source::Attn("qo_indptr_d") and   -> Cx::plan(), ONE query for six,
    //     the three other CSR arrays,        because they describe one thing
    //     "row_valid_d", "num_requests"      and are read together
    //   Source::Attn("first_token")       -> Cx::first_token()
    //   Source::AttnNonZero("w_page_d")   -> Cx::w_page_d(), null-checked IN
    //   Source::AttnNonZero("w_off_d")       THE QUERY — the predicate moved
    //                                        from the emitter to the fact
    //
    // `write_kv_explicit_devwin` is a `contract!` with a `none:` arm and no
    // bind: its `win_d` has no producer in `AttnCtx` at all, and this row
    // stated `Source::Unbound` on all nine operands, so `abi.rs:810` had
    // been skipping it whole and no dispatch arm ever existed for the
    // `Entry` to shadow. That is measured at the arm, not assumed.
    // `attn::pad_head_dim_bf16` and `attn::strip_head_dim_bf16` CROSSED INTO
    // FN-WORLD — `crate::x::attn`'s `PAD_HEAD_DIM` and `STRIP_HEAD_DIM`.
    // Stating the pair still turns `if (c.head_dim_padded)` in the model body
    // into a fact the trace carries; what the contracts no longer state is
    // the binding instruction, because the `fn` binds its own arguments.
    // `attn::merge_attention_states_bf16` WAS HERE — the KV-split's other
    // half. Deleted by `new-horizon.md` §38: its whole consumer set was
    // `dsl::cuda::merge_attention_states`, which nothing called.
    //
    // THE TABLE ROW STAYS DELETED AND THE DEVICE TEXT CAME BACK. Those are
    // two different things and this block is the record of both, because for
    // one pass the tree behaved as though they were one.
    //
    // `dsl::cuda::merge_attention_states` (`model-compiler/src/dsl.rs:3532`)
    // still exists and is still called by nothing — `tests/consumer.rs:63`
    // and `examples/migration_status.rs:926` both say so, in the same words:
    // zero callers, zero goldens, zero `pie_k_*`, zero `lower.rs` arms, no
    // peel stem, no fact gate. §38's argument about the CONSUMER SET was
    // correct and nothing here revises it. A table row is a thing
    // `model-compiler` can name in a statement; nothing names this one, so
    // there is no row.
    //
    // What §38 could not see is that the FA2 lattice's split path calls this
    // fold from INSIDE upstream's dispatch (`prefill.cuh:4350-4352`,
    // `decode.cuh:822-824`) rather than through the DSL. The C++ that ran was
    // compiled into `driver-cuda/csrc/attn/attention_flashinfer.cu`, and
    // closing the FA2 seams deleted that file — so
    // `fire/flashinfer_fa2.rs` had to set `disable_split_kv: true` and
    // split-KV prefill was off for a pass. That was a real performance
    // regression on short prompts and small batches.
    //
    // IT IS BACK, as `crate::families::cascade` — one unit,
    // `csrc/src/cascade/merge_states.cuh`, ten rows over the VENDORED
    // `cascade.cuh` — and `driver-cuda/src/fire/merge_states.rs`, the Rust
    // host program. `unit.rs`'s `DEMANDS` names it `Headers::LibraryAndVendor`,
    // the second of two entries.
    //
    // NOT the vendored `cascade.cuh`, which this comment used to claim, AND
    // THAT IS STILL A LIVE DISTINCTION. This crate carries
    // `csrc/vendor/flashinfer/attention/cascade.cuh`, but no `-I` anywhere in
    // the repository puts it in front of a C++ compiler:
    // `kernels-cuda/csrc/CMakeLists.txt`'s include list names
    // `${flashinfer_SOURCE_DIR}` — the CPM checkout — and never `csrc/vendor`.
    // The deleted launcher read the fetched copy; the vendored copy is
    // NVRTC's alone, reachable only through `Headers::LibraryAndVendor`. The
    // two copies being byte-for-byte the same upstream text is what has kept
    // the distinction invisible, and it is the distinction that decides
    // whether deleting `kernels-cuda` frees `csrc/vendor` — it does not. The
    // new unit points at the VENDORED copy, which is the whole point of the
    // return trip: no include path, no CPM, `carried.rs` hands NVRTC the
    // bytes.
    //
    // TWO CORRECTIONS TO THE SPECIFICATION THIS BLOCK USED TO BE.
    //
    // FIRST, IT NAMED THE WRONG LAUNCHER. The spec described `MergeStates`
    // (`cascade.cuh:637-668`) and its `num_index_sets >= seq_len` arm. That
    // launcher is real, it is ported (`fire/merge_states.rs::merge_states`),
    // and it is NOT the one the FA2 split path calls. Both batched dispatches
    // call `VariableLengthMergeStates` (`cascade.cuh:686-736`);
    // `MergeStates` is reached only from the SINGLE-request paths
    // (`prefill.cuh:2559`, `decode.cuh:739`), where every row was split into
    // the same number of chunks. The difference is correctness, not speed:
    // `MergeStatesKernel` folds one `num_index_sets` for every row
    // (`:221`), while `PersistentVariableLengthMergeStatesKernel` reads each
    // row's own count as `indptr[pos + 1] - indptr[pos]` (`:401`). A batch of
    // unequal KV lengths folded with a uniform count reads another row's
    // partials. Implementing only what this block specified and flipping
    // `disable_split_kv` would have been silent corruption.
    //
    // SECOND, IT WAS RIGHT ABOUT THE MISSING VOCABULARY AND THE MEASUREMENTS,
    // AND ALL OF THAT SURVIVES:
    //
    //   * Exactly one kernel fires, never both and never in sequence, so
    //     there is no intermediate buffer. The host decides an empty-work
    //     guard and one arm — `num_index_sets >= seq_len` (`:644`) picks the
    //     large-index-set kernel.
    //   * Shared memory is 8,704 B at head dim 64/128/256 and 16,896 B at
    //     512, all under 48 KB, so the `cudaFuncSetAttribute` at `:656` and
    //     `:715` is a no-op nothing has to express.
    //     `families::cascade::smem_bytes` re-derives both figures and a test
    //     pins them.
    //   * MISSING VOCABULARY: none, and that was a retraction of two
    //     entries. The arm had been written down as unstateable because it
    //     compares TWO operands while every `Term` is unary and `Source`'s
    //     combinators stop at `Ne`; the geometry had been written down as
    //     unstateable because both arms take a computed 2-D block
    //     `(HEAD_DIM / vec_size, bdy)`. Neither survived the rule that host
    //     composition is Rust. Both are now written: the comparison is an
    //     `if` in `fire/merge_states.rs` and the block is a `Launch` literal.
    //     A `LaunchRule` is for a table-driven row, not for a Rust walk.
    //   * Nothing crosses by value. `MergeStatesKernel` takes four pointers
    //     and three `uint32_t` (`:213-216`), the large one four and two
    //     (`:275-281`), the persistent one five and two plus a nullable
    //     device `uint32_t*` (`:366-371`) — every one of which `ArgValue`
    //     binds today. That made it, as this block predicted, the cheapest
    //     available proof that the whole shape works, and it needed no
    //     `params_layout.py` probe.
    //   * The header gate was the thing that ordered the work, and it was
    //     already clear. NVRTC sees only the vendored tree; the CPM checkout
    //     is on no NVRTC path. `csrc/vendor/flashinfer/attention/cascade.cuh`
    //     IS vendored — unlike the sm90 prefill and
    //     `comm/custom_all_reduce.cu`, whose headers are CPM-only and for
    //     which `csrc/vendor` has no `attention/hopper/` and no `comm/`
    //     directory at all.
    //   * `examples/vendor_probe.rs`' `MERGE` candidate compiled this header
    //     to 96,176 B with 8 of 8 symbols resolving, and that measurement is
    //     what made the return trip cheap. §31.4's precedent exactly: the
    //     probe is how you get there, and the row was never how.
    //
    // The one claim in the old block that has decayed: it cited
    // `attn/attention_merge_states.cu:31` as a surviving launcher. That file
    // is gone with the rest of `kernels-cuda/csrc/src/attn/`.
    // `examples/vendor_probe.rs:200` cites it too and is stale for the same
    // reason; the probe still runs, because it reads the vendored header and
    // never that file.
    // Rewrites `[R+1]` indptr arrays, so a row window would compact the wrong
    // requests' page lists.
    // `attn::split_qkv_bf16_devwin` CROSSED INTO FN-WORLD as
    // `crate::x::attn::SPLIT_QKV_DEVWIN`, BOUND. Its host program is
    // `x::attn::split_qkv_bf16_devwin`, `driver-cuda/src/fire/split_packed.rs`
    // is DELETED with `bind::service::attn_split_qkv_bf16_devwin`, and the
    // `execution::RUST_SERVED` and `execution::WALKED` entries are gone --
    // the walk retracted rather than lapsing, because what it measured is
    // still true and only its conclusion was wrong.
    //
    // THE CROSSING RETRACTED TWO REASONS THIS ROW EXISTED FOR, and one of
    // them was false rather than stale. Three places said the same pair at
    // length -- the `Walk`'s `because`, `families/attn.rs`' `SPLIT_PACKED_
    // SIGS`, and the unit's own `split_qkv_devwin` doc:
    //
    //   1. *"`grid.y` is the FIRE's lane count and not the statement's
    //      rectangle."* True, and answered: `Cx::rows().total` IS
    //      `DispatchCtx::rows_total`, and `bind/facts.rs:319` says in the
    //      field's own doc that it exists for a `_devwin` launch.
    //   2. *"`Cx::arg_in`/`arg_out` return pointers `resolve_arg_windowed`
    //      has already offset."* FALSE. `bind/mod.rs:3973` resolves every arg
    //      of a kernel whose name ends `_devwin` at row 0 -- *"The `_devwin`
    //      forms are the stated exception. Their contract is BASE pointers."*
    //      `Fire::arg_in` returns `bound.args[i].ptr` unchanged, so a bind
    //      sees exactly what the generated arm passed.
    //
    // A general fact about `arg_in`, derived once, stored beside the one
    // symbol the binder exempts BY NAME. That is Sec 75's shape, and this
    // instance is expensive rather than merely untidy: it is why
    // `Cx::window_left` was asked for. The query is good and FA2 uses it;
    // its doc names the wrong beneficiary, because this kernel wants
    // `win_d`, a device POINTER, which is `Cx::peel_window` and always was.
    //
    // Nine operands, nine queries: `In(0)`, `Out(0..3)`,
    // `CtxNonZero("peel_window")`, `Ctx("rows_total")` and two `OutWidth`s.

    // `attn::compact_page_csr`, `attn::mtp_shift_hidden_bf16` AND
    // `attn::mtp_update_pending_hidden_bf16` CROSSED INTO FN-WORLD as
    // `crate::x::attn`'s `COMPACT_PAGE_CSR`, `MTP_SHIFT_HIDDEN` and
    // `MTP_UPDATE_PENDING_HIDDEN`, all three UNBOUND. Their host programs
    // moved to `x::attn` and `driver-cuda/src/fire/page_compact.rs` and
    // `fire/attention_naive.rs` were DELETED.
    //
    // THESE THREE NEEDED THEIR UNITS WRITTEN, which the earlier crossings did
    // not: `attn/page_compact.cuh` and `attn/attention_naive.cuh` had no
    // `unit!` at all, so the device half was declared here for the first
    // time. `attention_naive.cuh`'s unit declares TWO of the root's five
    // `__global__`s, and the `.cuh` says why for the other three -- *"NO ROW
    // STATES THIS KERNEL: per-head grid, extent-sized shared memory"* -- so
    // they have no host program anywhere and declaring them would be a `fn`
    // nobody can call.
    //
    // `compact_page_csr` keeps its `execution::COMPOSED` entry. That entry is
    // a finding about the BODY -- two launches, one stream, the second
    // reading the first's buffer -- and the body is still two ops. What went
    // is the row. Both of its refusals are hoisted ahead of the first launch,
    // §5.1's rule, and neither could be: neither reads a device value.
    // `attn::attn_score_fold_heads`'s ROW STOOD HERE. `x::attn`'s
    // `ATTN_SCORE_FOLD_HEADS` is a `contract!`, and `contract!` retires a row.
    //
    // The row stated NINE operands and `Source::Unbound` on all nine, so
    // `abi.rs:810` skipped it whole and no arm was ever generated -- which is
    // what made a `none:` arm safe here and what makes the deletion visible
    // to nothing. The live firer was never this row: it is
    // `fire/attn_score.rs`, resolving `FOLD_SYMBOL` through `unit_of`, and it
    // still is. What the row cost was a NAME COLLISION -- one string was the
    // device row's symbol, this row's symbol and `dsl::cuda`'s stated symbol
    // at once -- and §60.6 forbids exactly that. The device row is
    // `attn::attn_score_fold_heads_dev` now.
    //
    // Note the operand count the row states and what the kernel takes. The
    // row says nine; the `__global__` takes SEVEN plus a stream. `stream` is
    // the row grammar's, not a parameter, and `num_requests` is the GRID's
    // x-extent, which the kernel never receives -- `dim3(num_requests, 64u)`.
    // A row that lists a grid extent among the operands is a row describing
    // the launch and the call in one vocabulary; a `fn` writes its own
    // `Launch` and the question does not arise.
    // MTP drafts several tokens per step and repairs on rejection, which
    // needs an attention that sees a HISTORY buffer beside the pages (the
    // drafted tokens are not committed -- committing them before acceptance
    // is the thing MTP must not do) and a per-slot pending-hidden shuffle.
    // All three address through `slot_ids` or `qo_indptr`.
    //
    // `attn::attention_mtp_paged_history_bf16` WAS the fourth. Deleted by
    // `new-horizon.md` §38: its whole consumer set was
    // `dsl::cuda::attention_mtp_paged_history`, which nothing called. The
    // launcher stays, and the reason is arithmetic rather than caution --
    // `attention_naive.cu:80`'s three-way host choice is the ONLY caller of
    // `attention_mtp_history_bf16` (`:52`), so deleting it would orphan two
    // launchers and two `<<<>>>`, not one, and move `EXPECTED` off 401.
    // Both are `NoRow` entries in `driver-cuda/tests/launch_abi.rs`.
    // Both walk `src_indptr[R+1]`. The window view is how sliding-window
    // attention is expressed without a second cache -- the window is a VIEW
    // over the same pages.
    // ── `attn::build_window_page_view` AND `attn::build_full_split_view` ─────
    //
    // BOTH ROWS ARE DELETED, with their two `dsl::cuda` wrappers and the two
    // launchers in `attn/kv_paged.cu`. The Rust is
    // `driver-cuda/src/fire/kv_paged.rs::build_window_page_view` and
    // `::build_full_split_view`.
    //
    // WHY DELETION AND NOT `RUST_SERVED`. Every operand of both rows was
    // `Source::Unbound` — `src_indptr` is the page table's CSR, `keep_pages`
    // is a model window divided by a page size, `splits` is a driver plan's
    // piece count, and no model text names any of them — so `crate::abi`
    // skipped each row WHOLE and neither ever generated a dispatch. §60.7
    // establishes that `RUST_SERVED` on an unsourced row is legitimate, and
    // it would work here; the reason it is not used is that it needs a
    // classification first (`every_taken_over_row_was_classified_first`) and
    // §58 says a single launch with no choice and no loop should carry none.
    // A row nothing binds, with a wrapper nothing calls, is §54's case: the
    // row and the wrapper go together.
    //
    // THE SWEEP. `crates/model/src`: no hit for either symbol string OR
    // either wrapper name — the two tokens were swept separately, because a
    // sweep for one has reported a live symbol as uncalled before.
    // `model-compiler/src/dsl.rs`: the two wrappers, deleted in the same
    // change. `lower.rs::semantic()`: no mapping. Hand `ffi::pie_k_*` arms:
    // none. C++: the only hit is
    // `kernels-cuda-new/csrc/src/attn/kv_paged.cuh`, which is the device text.
    //
    // The DEVICE rows stay — `LaunchRule::Single` and `SingleWarp`,
    // `families/attn.rs` — because a family row is a claim about what a
    // kernel IS. The warp-width argument for `SingleWarp` is carried into the
    // Rust's doc comment rather than left behind here.

      // A SECOND KV cache beside the fine-grained one, holding one entry per
    // `ratio` tokens. Every query attends both and the outputs are merged by
    // their log-sum-exps -- exact, not an approximation: the same algebra
    // flashinfer's own KV-split merge uses.
    // The prefill form. A SECOND row rather than a wider first one: the decode
    // launcher is what a CUDA-graph-captured decode calls, and giving it two
    // more operands would make every capture carry a `qo_indptr` it does not
    // read. The kernels differ in one line -- the request index -- and the
    // tables say so by naming both.
    // `attn::dsv4_boundary_meta_decode`, `attn::dsv4_boundary_meta_paged` AND
    // `attn::attention_compressed_paged_bf16` CROSSED INTO FN-WORLD as
    // `crate::x::attn`'s `DSV4_BOUNDARY_META_DECODE`,
    // `DSV4_BOUNDARY_META_PAGED` and `DSV4_ATTENTION_COMPRESSED_PAGED`, all
    // three UNBOUND, joining the two `dsv4_compress` contracts already there.
    // Their host programs moved to `x::attn` and
    // `driver-cuda/src/fire/dsv4_compress.rs` was DELETED -- the file's
    // fourth member, `combine_attn_outputs_bf16`, had already crossed and
    // BOUND, and `bind::service` was the only consumer of the other three.
    //
    // These three rows had operands and no `Source`, and that combination is
    // what the crossing records: THE RATIO. `dsl::cuda::dsv4_boundary_meta`
    // records its inputs with `record_many` and no parameters, so the one
    // integer the kernel divides by has no `Source::Param` to name -- and it
    // is in no `AttnCtx`, no `DispatchCtx` and no `Facts` query either, so
    // there is nothing to answer it with on any side. `abi.rs:810` skipped
    // all three whole and no arm was ever generated.
    //
    // What the compressed row adds is a SECOND KIND of blocker beside the
    // first, and the two should not be confused: `comp_kv_pages` and
    // `req_of_token` are values that do not exist -- no pool allocates the
    // compressed cache and nothing builds the per-token request map --
    // whereas `sm_scale` exists at `bind/mod.rs:1489` and is read by six
    // generated arms, and is missing only a `Cx` query. One of those is a
    // gap and three are absences.
    // `attn::dsv4_compress_gather_paged_bf16` AND
    // `attn::dsv4_store_comp_entries_bf16` CROSSED INTO FN-WORLD as
    // `crate::x::attn`'s `DSV4_COMPRESS_GATHER_PAGED` and
    // `DSV4_STORE_COMP_ENTRIES`, both UNBOUND, with the whole of
    // `attn/dsv4_compress.cuh` as `x::attn::dsv4_compress`'s unit.
    //
    // `whole = true` and no `sink` travelled unchanged. Nine paged rows above
    // state `sink = Some("kv.pages")` and these two never did, because they
    // write the COMPRESSED cache and this vocabulary has one name for the
    // fine-grained one. That absence is a statement and it is kept as one.
    //
    // WHY UNBOUND, and it is not a `Cx` gap. `dsl::cuda`'s two wrappers
    // (`dsl.rs:4684`, `:4702`) record ONE and TWO inputs and no parameters
    // for kernels that read twelve and eight operands, so `state_kv`,
    // `state_score`, `ape`, `boundary_req`, `ratio` and `coff` have no
    // operand to be bound from. Both symbols are on
    // `driver-cuda/tests/executor_bind.rs`' UNARMED list and both were in
    // `device::JIT_DISPATCHED`, so neither had a shim entry, and both rows
    // were unsourced, so `abi` skipped them WHOLE and generated no dispatch.
    // Nothing fired them here and nothing fires them there; what moved is the
    // device text and a written host program waiting on a statement.
    //
    // THREE OTHER ROWS OF THIS ROOT STAY, immediately below and above:
    // `dsv4_boundary_meta_decode`, `dsv4_boundary_meta_paged` and
    // `attention_compressed_paged` are served by
    // `driver-cuda/src/fire/dsv4_compress.rs` through `bind::service`, which
    // is `qkv_decode_fused`'s arrangement exactly. They fire the `_dev`
    // symbols of the unit that just moved, by name, through `hand::fire` —
    // so the §60.6 symbol split is carried VERBATIM into fn-world and
    // renaming any of the three is a panic at the first deepseek_v4 fire.
    //
    // THERE WERE FOUR. `combine_attn_outputs` has since crossed, and the
    // difference between it and these three is worth one sentence because it
    // is the whole of what makes a bind possible: its row sourced every
    // operand from the statement, and each of theirs needs a value the
    // statement does not carry.
    // `qo_indptr` + `kv_page_indptr`, like every other paged attention here.
    // No capture variant, so it cannot publish scores; it does publish an LSE,
    // which is what the combine below consumes.
    // `attn::mla_prepare_bf16` AND `attn::write_mla_to_pages` CROSSED INTO
    // FN-WORLD as `crate::x::attn`'s `MLA_PREPARE` and `WRITE_MLA_TO_PAGES`,
    // both UNBOUND, and `driver-cuda/src/fire/mla_paged.rs` was DELETED --
    // `bind::service` was its only consumer and the whole of
    // `attn/mla_paged.cu` is now `x::attn::mla_paged`'s unit.
    //
    // BOTH ROWS LED WITH `layer: MlaCacheLayerView` AND THAT IS THE FINDING
    // THE CROSSING KEEPS. One operand cell, five kernel parameters. `abi.rs`
    // could not source it, `execution::Control::Supplies` named three of the
    // five as values *"no `Source` can reach, because the view is one
    // dispatch argument and its fields are five"*, and a driver-side
    // launcher had to unpack it before every `<<<>>>`. `Cx::mla_layer`
    // returns the same five as `x::MlaLayer`, so the unpacking is the host
    // program's first three lines and the `Walk` has nothing left to supply.
    //
    // What blocks the bind is that query, not the operand list: `AttnCtx`
    // carries no MLA layer list, so `Cx::mla_layer` refuses. That refusal
    // has a PRODUCER it cannot reach -- `MlaCachePool::layer_view` -- which
    // is a different thing from the `dsv4` ratio above, whose value exists
    // nowhere at all. Both rows were unsourced on every operand, so
    // `abi.rs:810` skipped each whole and no arm was ever generated.
    // deepseek_v4, glm5 and kimi_k3 attend through a compressed KV: a
    // `kv_lora_rank`-wide latent row plus a small rope-carrying companion,
    // with the heads reconstructed on the way in. A different attention
    // algebra, not a different head count.
    //
    // The two paged statements are `whole` because they address through
    // `qo_indptr` / `kv_page_indptr` / `kv_last_page_lens`, which are
    // R-shaped: a row window would leave that arithmetic pointing at the
    // wrong request. The dispatch is not -- like the flashinfer dispatches,
    // it reads a plan built over the whole fire and still covers a row range.
    // `attn::dispatch_attention_mla_bf16` CROSSED INTO FN-WORLD as
    // `crate::x::attn::ATTENTION_MLA`, carrying `needs = Prepare::MlaPlan` and
    // `lacks = &[Cap::Scores]` verbatim. Its two arms are now BOTH Rust —
    // `x::attn::mla_fa2` is FlashInfer's cooperative kernel with its own unit
    // and a `fire_ex`, and `driver-cuda/src/fire/mla_naive.rs` is the sm_100
    // pair — which is what let the row go, since a row loses its shim entry
    // whole or not at all. `csrc/src/attn/attention_mla.cu` went with it, and
    // with that file the last nvcc-compiled `<<<>>>` in the workspace.
    //
    // It crossed UNBOUND, and the reason is in the contract's `none:` arm:
    // four of its fourteen operands are facts `Cx` does not state — the MLA
    // cache layer, the plan handle, the attention workspace and `sm_scale` —
    // which is the same sentence `executor_bind.rs:1519` was already using to
    // explain why nothing arms it. The row never fired; a row that never fires
    // and holds an nvcc translation unit hostage is worth strictly less than
    // the contract that replaces it.
    // `attn::logit_softcap_bf16` CROSSED INTO FN-WORLD as
    // `crate::x::attn::LOGIT_SOFTCAP`, once `Facts::final_logit_softcap()`
    // landed to source its cap. The row's argument travelled with it: it caps
    // the logits WHERE THEY LIE — one buffer, no destination, which
    // `Buffers::assign` was already relying on ("the logit softcap
    // accumulates into the logits it was handed", where it widens a seam's
    // pin over an alias set) while this row said nothing, so the set had one
    // member and the widening reached nothing. The head wrote the logits into
    // the arena and the cap ran over `ws.logits`, which is where the sampler
    // then read an uncapped previous fire. `in_place` is on the contract.
    // `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` CROSSED INTO
    // FN-WORLD as `crate::x::attn::QKV_PACKED_POST`, with a REAL bind: every
    // one of its twenty-one sources is a `Cx` query that already existed, and
    // ten of them are two — `Cx::kv_layer` carries the five `KvLayerField`
    // spellings plus `hnd`, and `Cx::plan` carries the three CSR arrays with
    // `row_valid`. Six statements in one launch and the only value that
    // survives is q; the rest is the `sink`, which travelled to the contract.
    //
    // `num_q_heads` still comes off the RESULT: `packed` is `[N, q + 2·kv]`
    // and cannot say where the cut falls, `q_out` is `[N, q]` and can.
    //
    // Its DECODE sibling `attn::qkv_decode_qk_norm_rope_write_kv_bf16` stays
    // right here, and NOT for the `attn::split_qkv_bf16_devwin` reason any
    // more: its host program crossed too and is
    // `crate::x::attn::qkv_fused::qkv_decode_fused_dispatch`. What holds the
    // row is one missing query — see the note above the row itself. Both
    // kernels' text is `crate::x::attn::qkv_fused`'s unit, `#rope` and
    // `#norope` suffixes carried verbatim because that program fires by name.
    // `attn::attention_sink_rescale_bf16` CROSSED INTO FN-WORLD —
    // `crate::x::attn`'s `ATTENTION_SINK_RESCALE`. gpt-oss's sink layers
    // still state it right after the dispatch, so `attn.out` observes the
    // RESCALED result, and the LSE is still the dispatch's SECOND result —
    // operand 1, a value only a sink layer declares, and not a scratch the
    // executor remembers handing the dispatch.
    // `attn::dequant_kv_cache_layer_to_bf16_active` CROSSED INTO FN-WORLD —
    // `crate::x::attn`'s `DEQUANT_KV_ACTIVE`. Three `Cx` queries and a call:
    // it stated no buffer operand, so there was nothing else to map.
    //
    // The moved body is a `pub fn` rather than an arm's private one, because
    // FA2's two decode and two prefill entries call it as a PRELUDE — an fp8
    // cache must be bf16 before rows that carry one KV width can read it.
    // A `bind!` arm is reachable only from a trace; those four are not
    // traces.
];
