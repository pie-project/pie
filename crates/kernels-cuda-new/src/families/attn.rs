//! `attn`'s two remaining JIT units, and the record of the seventeen that left.
//!
//! # What this module holds, MEASURED rather than described
//!
//! Two [`Unit`]s and four [`DeviceKernel`] rows:
//!
//! ```text
//! ATTN_SCORE_POST   3 rows   attn/attention_score_post.cuh
//! ATTN_XQA          1 row    attn/attention_xqa_mha.cuh
//! ```
//!
//! [`UNITS_SMALL`] IS EMPTY and [`UNITS_HEAVY`] holds those two, so
//! [`UNITS`] — the concatenation — is the same two. Every other `attn` root
//! is [`crate::x::attn`]'s: eighteen units there, declared beside their host
//! programs, against two here.
//!
//! Each sig is still its ahead-of-time twin minus the stream — a stream is
//! `cuLaunchKernel`'s sixth PARAMETER, outside the `void**`, so it was never
//! an operand — and minus whatever extent the launch rule recovers.
//!
//! # THE COUNTS IN EVERY SECTION BELOW ARE HISTORY, and are kept deliberately
//!
//! This header said *"Twenty-one kernels, thirteen rows"* and *"the heavy
//! half has nine"* through six passes that deleted roots out from under it,
//! and both had been false for some time when they were re-derived against
//! the file rather than read. The sections below are the RECORD of how this
//! module emptied — every one of them names the `crate::x::attn` unit its
//! subject became — and their findings are still true of the kernels they
//! are about. Their arithmetic is not true of this file. Read a number below
//! as a fact about the pass that wrote it, and read the block above as the
//! fact about now.
//!
//! One derived consequence, because it decides how the two longest sections
//! are read: **all four surviving rows are [`DeviceKernel::PLAIN`]**. There
//! is no template argument left anywhere in this module, so the
//! multi-argument ceiling below and the [`crate::device::Specialisation`]
//! argument after it are both about kernels that are elsewhere now. Neither
//! reaches anything this file still declares.
//!
//! The kernels still without rows are migrated as TEXT and unmigrated as
//! ROWS, and the reason is no longer uniform. [`crate::runtime::launch::eval`]
//! evaluates twelve of the sixteen rules now, so what is left is not a
//! backend behind its vocabulary — it is four kernels whose geometry no rule
//! in the vocabulary states, and each `.cuh` says which:
//!
//! * `attn/attention_naive`'s two remaining history kernels size DYNAMIC
//!   SHARED MEMORY on an extent `Dims` does not carry. `Rule::SdpaVector`
//!   sizes it on `rows`, which is `attn_naive`'s KV extent and nothing
//!   else's: `attn_mtp_history` asks for `history_steps + BLOCK` floats and
//!   `attn_mtp_paged_history` for a page window. Launch either at
//!   `rows + BLOCK` and the reduction scratch lands inside the scores it
//!   reduces — a wrong softmax, not a fault.
//!
//!   **`attn/dsa_indexer`'s `index_topk_mask` was named here and is a row
//!   now.** Its smem is `tokens * sizeof(float)` — one float per KEY, and
//!   every key of that fire is a ROW of it — so the extent `Dims` did not
//!   carry turned out to be `Dims::rows` read a second way, and
//!   `LaunchRule::RowScores` landed stating exactly that.
//!   `runtime::launch::row_scores` cites this launcher. The refusal was right
//!   about the shared memory and wrong about whose extent it was.
//! * `attn/attention_naive`'s `attn_mtp_paged_history` has a launcher that
//!   CHOOSES BETWEEN TWO KERNELS on a shared-memory budget. A `LaunchRule`
//!   selects a rectangle, not a kernel.
//! * `attn/attention_naive`'s `mtp_update_pending_hidden` opened one block per
//!   REQUEST, and every rule opened its grid over rows. **Retired.**
//!   [`LaunchRule::PerRequest`] opens its grid over [`Dims::requests`], which
//!   `jit_dims` fills from the attention context rather than from `rows`, and
//!   `crate::x::attn::attention_naive`'s `mtp_update_pending_hidden` is the
//!   row.
//! * **`attn/attention_flashinfer`'s `attn_score_fold_heads` LEFT this list
//!   by dissolving it, which is the outcome `combine_attn_outputs` had and
//!   the one this whole section argues for.** The root is
//!   [`crate::x::attn::attention_flashinfer`] now and the refusal below is
//!   retained as the ARGUMENT, not as a live verdict: in fn-world a `Launch`
//!   is written by whoever fires it, so a geometry no `LaunchRule` can state
//!   stops being a reason to refuse and starts being three lines of Rust.
//!   Nothing was satisfied; the question stopped existing.
//!
//!   The text had moved first — `attn/attention_flashinfer.cuh`, a PARTIAL
//!   split of a file that keeps its FlashInfer dispatch; its three private
//!   score-normalisation kernels went too, to
//!   `attn/attention_score_post.cuh`, and are [`ATTN_SCORE_POST`]'s rows as
//!   of §53.8. The launcher was
//!   `driver-cuda/csrc/attn/attention_flashinfer.cu:618-619`,
//!   `dim3(requests, 64)` at 256 threads
//!   with nothing shared, and the `64` is a LITERAL: a grid-stride fanout,
//!   not a dimension of anything. **[`LaunchRule::PerRequest`] is one number
//!   away** — `dim3(requests)` at 256 with nothing shared, the same `grid.x`,
//!   the same block, `grid.y` of one — and the body strides
//!   `i += blockDim.x * gridDim.y`, so it computes THE SAME FLOATS in 64x
//!   fewer blocks. Nothing in this tree would fail if that row were written,
//!   which is the `combine_attn_outputs` argument above and the reason not to
//!   write it. [`LaunchRule::PerRow`] is the same shape off the wrong axis (a
//!   request count is not a row count) and [`LaunchRule::PagedScores`] is
//!   `dim3(requests, rows, q_heads)` at 128 with dynamic shared memory.
//!   **The rule it needs is `PerRequest` with a fixed y-fanout —
//!   `dim3(requests, 64)` at 256, nothing shared** — and no rule here carries
//!   a literal grid axis. No unit is declared for that header, because
//!   `tests/units.rs::verdict` hard-fails a unit with no rows and a cubin
//!   nothing can fire satisfies nobody.
//!   `tests/launch_rules.rs::mod transcribed::pins` pins both launcher lines
//!   so the refusal's citation cannot rot — and the pin now reads
//!   `crate::x::attn`, because the host program is `fire/attn_score.rs`'s
//!   hand-built `Launch` and always was. That is the other half of why this
//!   entry dissolved: the row never had a host program to lose.
//! * `attn/split_packed`'s `split_qkv_devwin` shares its sibling's grid
//!   arithmetic and not its INPUTS — see
//!   [`crate::x::attn::split_packed`], which is the
//!   one place in this file where a ported rule computes the right shape from
//!   the wrong numbers.
//! * `attn/dsv4_compress`'s `combine_attn_outputs` had TWO blockers and has
//!   NONE — it crossed into `crate::x::attn` as `COMBINE_ATTN_OUTPUTS`, a
//!   contract with a bind, and the second blocker below is now four lines
//!   from its launch as `x::attn::combine_attn`'s doc. Read the rest as the
//!   history of why it took two passes: The `__global__` was not a template — concrete `device::bf16`
//!   parameters against an `instantiation()` that always emits `path<elem>`
//!   — and it is one now, retyped mechanically onto `Elem<T>::to_f32` /
//!   `from_f32` with `dsv4_compress.cu` launching `<device::bf16>` so the
//!   ahead-of-time build emits the same instructions. What survives is a
//!   BLOCK-WIDTH disagreement, and it is a finding rather than an obstacle:
//!   the launcher clamps `head_dim` into `[32, 256]` and
//!   `PerHeadElementwise` clamps into `[32, 128]`, so above a 128-wide head
//!   the rule answers with half the launcher's threads. The grids are
//!   identical — `grid(rows, q_heads)` on both sides — and the kernel
//!   strides `d += blockDim.x` through every loop, reduces nothing, and
//!   synchronises nowhere, so the narrower block computes BIT-IDENTICAL
//!   bytes in two passes. That is exactly what makes the row unsafe to
//!   write: the mismatch can only ever show up as latency, never as a wrong
//!   answer, so nothing in this tree would fail if it were wrong. deepseek
//!   v4 runs 128-wide heads today, where the two agree; a config that
//!   widens one silently halves the launch. Reconciling it is a decision
//!   about `SINK_BLOCK_MAX` in `runtime::launch` — whose own
//!   `per_head_elementwise` doc names this launcher as a second client of
//!   its clamp — and that file is not this one's to change. The row costs
//!   one line the day that number is settled.
//!
//! Both files this module used to record as *"not split at all"* are split,
//! and the two reasons it gave are now two different facts.
//!
//! * `attn/page_compact.cu` reached into `<cub/cub.cuh>` for
//!   `BlockReduce`/`BlockScan`, and CCCL is deliberately not carried — 13.7 MB
//!   in 1,691 files, and NVRTC answers no external include. The two
//!   collectives are written out in `page_compact.cuh` against
//!   `__shfl_down_sync` / `__shfl_up_sync` instead. Both fold `u32` under `+`,
//!   which is exact and associative modulo 2^32, so any correct fold order
//!   produces the same bits — not "close to CUB", the same integer. The file
//!   is `crate::x::attn::page_compact` now, with a row for each of its two
//!   kernels.
//! * `attn/pack_dense_mask.cu` took a `StructuredMaskParams` defined in its
//!   host `.hpp`, which NVRTC cannot include. The header carries a device
//!   MIRROR of that struct, and the duplication WAS checked rather than
//!   trusted: FIVE `static_assert`s in `pack_dense_mask.cu` pinned `sizeof`,
//!   `alignof` and the offset of all three fields (`kind`, `window`, `sink`)
//!   against the host type. A negative control transposing two fields fired
//!   exactly the two offset assertions and nothing else, which is what made
//!   this a check and not a comment.
//!
//!   **Both the `.cu` and the `.hpp` are deleted, so those five assertions
//!   are gone, and the root has CROSSED INTO FN-WORLD as
//!   [`crate::x::attn::pack_dense_mask`] — unit only, no contract and no bind,
//!   because nothing in the tree launches either kernel any more.** The one
//!   surviving mirror is [`crate::x::attn::params::StructuredMaskParams`],
//!   whose layout is MEASURED out of NVRTC's PTX (`sizeof=12 alignof=4`,
//!   `kind@0 window@4 sink@8`, `nvrtc-probes/attn_structured_mask.py`) rather
//!   than transcribed, and which `driver-cuda/src/bind/abi.rs` now re-exports
//!   so that two definitions cannot drift again. The measurement agreed with
//!   both transcriptions field for field.
//! * The obstacle that remains is neither of the two above — see the next
//!   section.
//!
//! # The limit that stopped three split files from having rows, and is gone
//!
//! [`crate::device::DeviceKernel::instantiation`] used to emit `path<...>`
//! and nothing else, so **a plain `__global__` could not be named by a row at
//! all**, however simple its launch. Three of the newly split kernels were
//! refused on that alone -- and the refusal was a report on a Rust `format!`
//! read as a fact about NVRTC. It is not one. `examples/argform_probe.rs`'s
//! twelfth case hands `nvrtcAddNameExpression` a bare qualified path with no
//! `<>`, NVRTC accepts it, `nvrtcGetLoweredName` answers a mangled symbol and
//! `cuModuleGetFunction` RESOLVES it on this L40S.
//! [`crate::device::DeviceKernel::PLAIN`] is what a row states to ask for
//! that spelling, and all three are rows now:
//!
//! * `attn/pack_dense_mask.cuh`'s `pack_dense_mask` and `pack_structured_mask`
//!   -- `pack_dense_mask.cu:94` and `:110`, both `<<<B, 128, 0, stream>>>`,
//!   which is [`LaunchRule::PerRowNarrow`] to the digit. The geometry was
//!   never the blocker; the spelling was. Both rows have since CROSSED, and
//!   `DeviceKernel::PLAIN` crossed with them —
//!   [`crate::x::attn::pack_dense_mask`] states it on both, for the reason
//!   this bullet gives.
//! * `attn/mla_paged.cuh`'s `write_mla` -- `mla_paged.cu:111`,
//!   `<<<total_tokens, 256, 0, stream>>>`, which is [`LaunchRule::PerRow`] to
//!   the digit, and whose fire's `Dims::rows` is the token count because
//!   `dsl::cuda::write_mla_to_pages` takes `kv_c` shaped `[Tokens, kv_lora_rank]`.
//!   See [`crate::x::attn::mla_paged`], which is where its text is now.
//!
//! **No device text changed for any of the three.** The fix this section used
//! to propose -- a DEFAULTED template parameter on each `__global__`, exactly
//! what `attn/dsv4_compress.cuh` did for its two boundary-metadata kernels --
//! would also have worked, and both headers still carry the argument for it.
//! It is a body change on every kernel it touches and `new-horizon.md` §8 then
//! demands parity evidence for each; naming costs nothing at the call sites
//! and changes no device text at all, so naming is what landed.
//!
//! The LINKAGE fact those headers state is untouched, and it is what a
//! defaulted parameter would still buy. §21.6's measurement holds: a
//! non-template `__global__` in a `.cuh` takes external linkage, so
//! `pack_dense_mask.cuh` and `mla_paged.cuh` may each be included by exactly
//! one translation unit -- `pack_dense_mask.cu` and `mla_paged.cu` -- and a
//! second includer is a hard `multiple definition` at link even when it never
//! launches anything. A row does not `#include`; NVRTC is handed the root and
//! compiles it alone. So the two facts have come apart, and only one of them
//! is closed -- a defaulted parameter is now a linkage decision to be made
//! per header for its own reasons, not a naming workaround.
//!
//! # The multi-argument ceiling, checked against this half
//!
//! `DeviceKernel::elem` turned out to carry a template argument LIST and not
//! only a type — measured against NVRTC, recorded in [`crate::device::args`]
//! — which took thirty-seven kernels off the tree's blocked list. **None of
//! them are in this half.** Every `__global__` in the seven headers above is
//! `template <class T>` with exactly one parameter, so there was never a
//! second argument to elide: the four unrowed kernels here are unrowed for
//! geometry, and re-checking each against the finding moved none of them.
//! The note is here so the next reader does not re-derive it.
//!
//! It does bear on `combine_attn_outputs`, and did: `instantiation()` could
//! not name a plain `__global__` AT ALL, which is a different limit than the
//! argument count and the one that kernel actually hit. Templating it fixed
//! that, and would no longer be needed for NAMING — the limit closed above —
//! though the kernel is templated now and stays so, because the edit also
//! bought it `Elem<T>` widening and a second numeric format for one row. What
//! the finding cannot reach is a `__global__` that is not a
//! template — `attn/dsv4_compress.cuh`'s two boundary-metadata kernels have
//! no type in them to abstract over, so there is nothing to put between the
//! brackets and no list makes one.
//!
//! # The heavy half has nine, and the list reaches exactly one of them
//!
//! [`UNITS_HEAVY`]'s three template-bearing headers hold every
//! multi-argument template in the
//! family, and auditing them against the finding produced one row change and
//! six refusals with a shared cause worth naming once. **Five of those six
//! refusals have since been overturned** — not by the argument-list finding
//! but by [`crate::device::Specialisation`], which arrived after it.
//!
//! ([`ATTN_SCORE_POST`] takes no part in this: its
//! three rows are `DeviceKernel::PLAIN`, non-templates whose block width is a
//! `constexpr int` inside the body. A unit with no template argument cannot
//! have an argument-list finding.)
//!
//! **`attn/kimi_mla.cuh`'s `split_kv_a_norm<class T, int BLOCK_DIM = 256>` is
//! the one it reaches**, and its row now states `elem: "device::bf16, 256"`
//! instead of leaning on the default. The comment on that row says what the
//! default was hiding; the short form is that `BLOCK_DIM` sizes a
//! `__shared__` array and fixes a halving reduction, so it is a value the row
//! must pin rather than inherit.
//!
//! **The other six are `attn/kv_paged.cuh`'s, and five of them are rows now.**
//! `write_kv`, `write_kv_at_positions`, `write_kv_explicit`,
//! `write_kv_explicit_devwin` and `copy_kv_cells` are
//! `template <bool HND_LAYOUT>`; `write_kv_per_token_head` is
//! `template <bool UseFp8>`.
//!
//! The GRAMMAR blocker this file reported is VOID, and the correction is
//! measured — `examples/argform_probe.rs`, this L40S, NVRTC 13.0.
//! `instantiation()` glues `::pie_cuda_driver::kernels::` to the FRONT of the
//! whole `elem` string, which reaches its first TOKEN and nothing after it,
//! so slot 1 must RESOLVE under the kernels root and need not be a TYPE: a
//! `constexpr` variable, a `static constexpr` member and a functional cast
//! all instantiate. `elem: "true"` does spell
//! `write_kv<::pie_cuda_driver::kernels::true>` and does come back `expected
//! an identifier` — the earlier report read that failure as a property of
//! non-type arguments when it is a property of BARE ones, and a refusal that
//! cites the wrong reason is a refusal nobody can overturn.
//! `pie_device.cuh:485` ships the spelling that works:
//! `elem: "device::true_type::value"` and its `false` twin name
//! `write_kv<true>` and `write_kv<false>`, the probe's sixth case. All six
//! kernels below are spellable today. Slots 2+ are the mirror image and
//! matter for `<device::bf16, 256>` above — they are NOT prefixed, so a bare
//! literal is the correct spelling there and a NAME would have to be written
//! out from `::`.
//!
//! What settled the six was the blocker the finding itself flagged: **the bool
//! is chosen at RUN TIME and every launcher spells BOTH arms.** *"Two rows
//! would not help: the table has no way to say 'this one when that operand is
//! true'."* That sentence was true and is not: [`crate::device::Fact::Bool`]
//! and [`crate::device::Term::Is`] landed, and **five of the six are rows
//! now** — and they are `crate::x::attn::kv_paged`'s ten `#hnd`/`#nhd`
//! rows, chosen by a host `if` in `driver-cuda/src/fire/kv_paged.rs`. The
//! `Specialisation` block that stood at the end of this file is gone; the
//! paragraph is kept because the argument it makes about the table is still
//! the reason those rows exist.
//!
//! Each is three rows and two arms: a CONTRACT row carrying the kernel's
//! parameters plus `hnd_layout: Bool`, and `#hnd` / `#nhd` variants carrying
//! the kernel's parameters alone, chosen by `Term::Is` over the flag.
//! `kv_paged.cu:84` opens `if (hnd_layout)` over `write_kv<true>` at `:85` and
//! `write_kv<false>` at `:95`; `:236` over `write_kv_at_positions<true|false>`
//! at `:237`/`:246`; `:283` over `write_kv_explicit_devwin` at `:284`/`:292`;
//! `:371` over `write_kv_explicit` at `:372`/`:380`; `:418` over
//! `copy_kv_cells` at `:419`/`:425`. (`:331`/`:332`/`:342` is a SECOND
//! launcher over `write_kv`, which is why that row is named for the kernel and
//! not for either host function.) All five launch `<<<n, 256, 0, stream>>>`
//! over a token, lane or cell count, so all five state
//! [`LaunchRule::PerRow`].
//!
//! Measured on an L40S sm_89 through the shipped fire path: both
//! instantiations resolve and, over five shapes × two layouts, 0 of 220,800
//! bf16 cells differ. The negative control is the reason to read the arms
//! carefully rather than the reason to relax — firing `write_kv<false>` where
//! the flag says `true` moved 34,273 of 55,200 cells **while writing the same
//! number of non-zero values**. A permutation, not a truncation: no count, no
//! norm and no tolerance check would flag it.
//!
//! **The refusal that survives inside this, and that a row must not defeat.**
//! `write_kv<HND_LAYOUT>` takes the same fifteen parameters either way, so a
//! fire whose flag matched no arm would fall through to the sixteen-operand
//! base row and bind sixteen cells for a fifteen-parameter kernel —
//! `cuLaunchKernel` reads the count from the cubin, never reads the sixteenth,
//! and SUCCEEDS. [`crate::device::Specialisation::agrees`] therefore requires
//! a flag no arm forwards to be covered on BOTH values.
//!
//! **`write_kv_per_token_head` is the sixth and stays refused.** It is
//! `template <bool UseFp8>` and its arms are `switch (layer.scheme)` at
//! `kv_paged.cu:155` — `<false>` at `:172`, `<true>` at `:185`. `Term::Is`
//! reads a `Ty::Bool` operand; a scheme is an enum, `Fact` has no reading for
//! one, and spelling it as a bool would state that two of the four schemes are
//! the same case. It also launches `<<<grid, BLOCK, shmem, stream>>>` with a
//! computed `shmem`, so the geometry is a second refusal behind the first.
//!
//! # The row the ahead-of-time build never had
//!
//! `attn::logit_softcap_f16`. The template was already there; a second
//! numeric format cost one line rather than a translation unit's worth of
//! `cicc` — which is the measurement `norm/elementwise.cuh` made first and
//! the reason this design was worth the migration. **Both softcap rows have
//! since crossed** to [`crate::x::attn::softcap`], where the f16 one is still
//! a device row with no contract, for the same reason it was a row with no
//! consumer here: no trace says it yet.
//!
//! # The sm90 prefill: no row, no unit, and the specification it needs
//!
//! `kernels-cuda/csrc/src/attn/attention_flashinfer_hopper.cu` has neither a
//! row nor a unit, so this header is the only place its specification can
//! hang. It holds zero `__global__`, zero `__device__` and zero `<<<>>>`:
//! every kernel is a template inside
//! `flashinfer::BatchPrefillWithPagedKVCacheDispatched`, so there is no device
//! text to migrate and the whole file is host program. Four of its six
//! FlashInfer includes are CPM-only; the other two (`attention/cascade.cuh`,
//! `attention/scheduler.cuh`, plus `layout.cuh`) have vendored twins that
//! **no `-I` in this repository reaches**. `new-horizon.md` §50.5 is the long
//! form.
//!
//!  1. FIRES nothing from `hopper_prefill_supported`; whatever
//!     `flashinfer::PrefillSM90Plan` fires from the plan entry; and one
//!     `BatchPrefillWithPagedKVCacheDispatched<HeadDim, HeadDim, Mask, Window,
//!     SameSchedule, Variant, HopperParams>` from the dispatch. The template
//!     cross-product over head dim × mask × window × schedule IS the
//!     instantiation set — each member is one row. `Specialisation` WAS the
//!     mechanism that picked among them, the same one that chose between five
//!     `kv_paged` appenders on a flag; both are gone, and in fn-world the
//!     choice is a host `if` over `raw::` calls (`x/norm.rs:1033`).
//!  2. INTERMEDIATE: the most demanding plan protocol in the tree.
//!     `PrefillSM90Plan` writes a `PrefillPlanSM90Info` across the float
//!     buffer, the int buffer AT AN OFFSET, and a PAGE-LOCKED HOST MIRROR at
//!     the same offset; the dispatch reads it back and throws on an empty
//!     plan. Two device regions plus a pinned host companion, with an
//!     `int_base_bytes >= workspace.int_bytes` end-of-buffer refusal, held
//!     across two calls.
//!  3. HOST DECIDES the shape gate `hopper_prefill_supported(head_dim, …)`, a
//!     head-layout refusal, the empty-plan refusal, and the mask/window/
//!     softcap template arms. All refusals, never fallbacks — a `throw` here
//!     means the caller knowingly picks another path.
//!  4. MISSING: the by-value aggregate (`HopperParams` is
//!     `BatchPrefillPagedParams<…>` and crosses by value — see the note on
//!     `ArgValue` in `runtime/args.rs`); a SECOND plan `Source`, because
//!     `Source::AttnPlan` names the FA2 plan and `PrefillPlanSM90Info` is a
//!     different struct with different fields; and there is no vocabulary at
//!     all for a host-pinned companion to a device region.
//!
//! `PrefillSM90Plan` is itself upstream HOST code, so NVRTC does not carry it.
//! Whether the Rust reimplements its index arithmetic or keeps calling into it
//! is the first decision this file forces and the one that sets the cost.
//!
//! **But none of that is the first blocker, and points 1-4 are not reachable
//! yet.** Before a unit can exist, NVRTC has to be able to SEE the device
//! text, and it cannot. NVRTC reads the vendored tree through
//! [`crate::unit::Headers::LibraryAndVendor`]; the CPM checkout
//! (`${flashinfer_SOURCE_DIR}`) is a *C++ compiler* include path and is not on
//! any NVRTC path. `kernels-cuda-new/csrc/vendor/flashinfer/` has **no
//! `attention/hopper/` directory at all**, and no `cutlass_utils.cuh`. Four of
//! this file's six includes are therefore unreachable, so there is no unit to
//! write, no row to name it, and no Rust in `driver-cuda/src/` that could fire
//! one. **The vendored tree must gain `attention/hopper/` first**, and that
//! tree is `vendor-role`'s.
//!
//! That is what the vendored-versus-CPM split actually MEANS, and it is worth
//! stating because the table on its own reads as an accident: **the vendored
//! set is precisely the set that has been prepared for NVRTC, and the CPM-only
//! set is precisely the set that has not.** `attention/cascade.cuh`,
//! `attention/mla.cuh`, `attention/scheduler.cuh`, `layout.cuh` and
//! `fastdiv.cuh` are vendored, which is why `merge_attention_states` and
//! `attention_mla` are carryable today; `attention/hopper/*`, `cutlass_utils`
//! and every `comm/*` are not, which is why this file and
//! `comm/custom_all_reduce.cu` are not. The dependency order for all four is
//! header, then unit, then row, then the by-value aggregate, then the Rust —
//! and this file is blocked at the first step, not the fourth.
//!
//! ## Its stub, and what a stub's Rust form is
//!
//! `attention_flashinfer_hopper_stub.cpp` is what actually compiles on this
//! box, and under the no-`.cpp` principle it is not a special case. §48.3:
//! *a stub goes when the thing it stands in for goes, and a stub with nothing
//! to stub is not a smaller problem, it is a deleted file.* So the correct
//! move for it is never "relocate it" — it is to go when this file goes.
//!
//! Its Rust form is not a file at all. An architecture stub exists because a
//! real implementation cannot compile for the target; in Rust that is `#[cfg]`
//! or a runtime capability check, and the driver already does both. Concretely
//! the stub IS the refusal arm: on `sm_89` the sm90 prefill is unsupported, so
//! what replaces it is a device fact read once at load and an eligibility
//! check that refuses — the same refusal `hopper_prefill_supported` already
//! spells, moved to the host language. Nothing needs a second artifact,
//! because a JIT unit that is never selected costs nothing to not compile,
//! which is the whole difference from an AOT archive that must contain a
//! symbol for every target it might run on. **The stub is an artefact of
//! ahead-of-time linking and it does not survive the move to NVRTC.**
//!
//! Stated once more so a future reader does not have to reconstruct it: this
//! box is an L40S, `sm_89`. The stub is what compiles; the real file has never
//! run here. §44.7 — every `sm_90` conclusion in this tree is argued from the
//! call graph and *"none of them is argued from a run"*. That the file has
//! been moved and moved back changes nothing about that standing.
//!
//! **None of this has been run.** This box is an L40S, `sm_89`, so
//! `attention_flashinfer_hopper_stub.cpp` is what compiles and the real file
//! is untested here. §44.7 recorded that every `sm_90` conclusion in this tree
//! is argued from the call graph and *"none of them is argued from a run"*.
//! The four points above are read off the source and inherit exactly that
//! standing — a specification to build against, not a claim that the Hopper
//! path works.

use kernels::KernelSig;
use kernels::Lit;
use kernels::LaunchRule;
use kernels::Source;
use kernels::kernel;
use kernels::operands;

use crate::device::DeviceKernel;
use crate::unit::Unit;

// ===========================================================================
// SMALL HALF — owned by the `attn` small-files migration.
//
// `mig-attn-heavy` APPENDS BELOW THE MARKER AT THE END OF THIS FILE. Nothing
// above it moves. When the heavy half lands it declares its own
// `UNITS_HEAVY`, and `UNITS` becomes the concatenation of the two — the one
// line in this file that both halves touch, and the comment beside it says
// how.
// ===========================================================================

// `SOFTCAP` CROSSED INTO FN-WORLD as [`crate::x::attn::softcap::SOFTCAP`],
// once `Facts::final_logit_softcap()` existed to source its cap. Both of its
// rows went — the bf16 one that has a contract and the f16 one that does
// not — because a unit is the whole of a root's device text and half a root
// is a second compilation of the same file.

/// The attention-sink pair and K3's residual blend CROSSED INTO FN-WORLD.
///
/// `ATTN_SINK` and `ATTN_RES` are [`crate::x::attn::attn_sink::ATTN_SINK`]
/// and [`crate::x::attn::attn_res::ATTN_RES`], declared beside the `fn`s that
/// fire them. Their measurements travelled with them — the 0.693 the LSE
/// rebase prevents, `PerHeadElementwise`'s axis order, and `PerRow`-not-`Rms`
/// with the thirty-two unused bytes that argument turns on.
// `DSA_INDEXER` CROSSED INTO FN-WORLD as
// [`crate::x::attn::dsa_indexer::DSA_INDEXER`], unit-only. All three host
// programs were already Rust in `driver-cuda/src/fire/dsa_indexer.rs` and
// stayed there; none of the three rows crossed, because a row is a claim
// about a trace and these three are fired by a host `fn` that already exists.
// See the deleted `DSA_INDEXER_ROWS`' crossing note below for the whole of
// what travelled.

// `ATTENTION_NAIVE` AND `PAGE_COMPACT` CROSSED INTO FN-WORLD as
// [`crate::x::attn::attention_naive::ATTENTION_NAIVE`] and
// [`crate::x::attn::page_compact::PAGE_COMPACT`], units and rows together.
// Both roots left this list because a root is in exactly one of the two:
// a second `unit!` over the same text compiles it twice and `unit_of` would
// answer with whichever won.
//
// All five device rows travelled -- `mtp_shift_hidden`, `attn_naive`,
// `mtp_update_pending_hidden`, `count_kept`, `scan_and_scatter` -- and so
// did every measurement that stood beside them:
//
// * `attn_naive`'s SdpaVector reading, `dim3 grid(num_q_heads, num_tokens)`
//   at 256 with `sizeof(float) * (num_tokens + BLOCK)`, and WHY no other
//   rule could stand in: the kernel lays `scores[num_tokens]` and
//   `reduce_buf[BLOCK]` in one `extern __shared__` block, so launched with
//   less the reduction scratch overlaps the scores it is reducing and THE
//   ANSWER IS FINITE. It has no host program and no statement lowers to it;
//   it is declared because it is a row, not because anything fires it.
// * `mtp_update_pending_hidden` was the row that MADE `LaunchRule::PerRequest`
//   and the only row on it. Its `fn` opens `Launch::per_row(num_requests, 256)`
//   directly, so the rule has no members left in this family.
// * `PAGE_COMPACT_SIGS`' two kept `LaunchRule::PerRow` against the same
//   `<<<num_requests, kBlock>>>`, which was this file's sharpest
//   demonstration that a launcher's SHAPE and a fire's rectangle are two
//   different questions -- `dsl::cuda::compact_page_csr` records
//   `Shape(vec![Dim::Requests])` and `mtp_update_pending_hidden` records no
//   result at all. Both `fn`s now state their own grid, so the
//   demonstration is a paragraph rather than a disagreement between rules.
// * `elem: "attn::device::kBlock"` was chosen over `device::i32(256)` so the
//   row could not drift from the launcher. The `fn` reads
//   `x::attn::page_compact::K_BLOCK`, one constant for both `<<<>>>`, which
//   is the same argument one crate over.
//
// `attn::compact_page_csr` was never claimed by either row here -- one
// launcher over two kernels -- and it is a `contract!` in `x::attn` now,
// which is the name those two rows would have been claiming half of.


/// flashinfer's supported head widths CROSSED INTO FN-WORLD, and closed a
/// measured defect on the way.
///
/// `HEAD_DIM_PAD` is [`crate::x::attn::head_dim_pad::HEAD_DIM_PAD`]. The
/// argument this doc carried — that a rowless unit is refused rather than
/// compiled, `every_unit_compiles_and_every_row_resolves` asserting
/// `!unit.rows.is_empty()`, because a cubin nothing can fire is one cached
/// per architecture for nobody — is unchanged and now lives there.
///
/// What changed is the head count. `LaunchRule::PerHead` evaluated
/// `per_head(dims.rows, dims.kv_heads)`, reading a context field neither row
/// mentioned; the `fn` is handed `num_heads` and has no way to reach a KV
/// head count. `crate::x::attn::per_head`'s doc carries the measurement.
// `SPLIT_PACKED` CROSSED INTO FN-WORLD as
// [`crate::x::attn::split_packed::SPLIT_PACKED`], and it crossed as a UNIT
// AND NOTHING ELSE — no contract and no bind, because both of its kernels
// already had host programs and neither host is a bind. `split_qkv_bf16` is
// [`crate::x::driver_internal::split_qkv_bf16`], the fourth arrangement;
// `split_qkv_bf16_devwin` keeps its table row and
// `driver-cuda/src/fire/split_packed.rs`.




// `SOFTCAP_ROWS`/`SOFTCAP_SIGS` CROSSED INTO FN-WORLD, both rows, with
// every measurement they carried:
//
// * `Elementwise` IS the launcher this replaced — `(n + 255) / 256` blocks
//   of 256, with an empty `n` refused rather than launched;
// * the `!(cap > 0.f)` half of that launcher's guard, which was the row's
//   `Source::CtxNonZero("final_logit_softcap")` and is now
//   `Cx::final_logit_softcap` plus a `fn`-level re-check for callers that
//   arrive without a `Cx`;
// * the reciprocal that moved INTO the kernel and is the same bits, because
//   `--prec-div=true` makes `1.f / cap` the correctly-rounded fp32 quotient
//   on the device exactly as it was on the host and `--fmad=false` keeps the
//   multiply from being contracted;
// * `in_place = &[(0, 0)]` and what `Buffers::assign` was relying on while
//   the row said nothing — the alias set with one member, the widening that
//   reached nothing, and the sampler reading an uncapped previous fire;
// * **the f16 row the ahead-of-time build never had**, which is the module
//   header's own example of what a JIT buys: a second numeric format costs a
//   line rather than a `cicc` invocation. It crossed as a device row with no
//   contract, because no trace says it yet.
//
// They are `crate::x::attn::softcap`'s now, and the two-formats-one-width
// hazard §3.2 names — bf16 and f16 are the same width and the same C spelling,
// so these two rows differed only in a symbol string — is answered there by
// `*mut bf16` and `*mut f16` being unrelated Rust types.

// `ATTN_SINK_ROWS`/`ATTN_SINK_SIGS` and `ATTN_RES_ROWS`/`ATTN_RES_SIGS`
// CROSSED INTO FN-WORLD, whole, with every measurement they carried:
//
// * the LSE rebase's 0.693 — *"without it the sigmoid argument was off by
//   0.693, which matched HF's top-1 on most prompts and then degenerated
//   greedy decoding after a few steps"*;
// * `elem = "attn::device::f32"` and why it is not `device::f32` — the
//   prelude has no such alias and `Elem` has no `float` specialisation to
//   hang one on;
// * `n: Usize` and not `I32`, because the kernel's parameter is
//   `device::usize` and the twin's `int` was the launcher's signature;
// * `PerHeadElementwise`'s axis order — the ROW is `grid.x` and the head is
//   `grid.y`, the transpose of `PerHead`'s, and `q_heads` and not `kv_heads`
//   because the tensor being rescaled is the attention OUTPUT;
// * `N` and `num_q_heads` staying operands though the geometry recovers
//   both, because they are the kernel's own bounds check and its row stride
//   and an operand list shorter than the `__global__`'s is a `void**` the
//   driver reads past;
// * `PerRow` and not `Rms` for the blend, with the thirty-two bytes of
//   dynamic shared memory `Rms` asks for that no launcher passes and no
//   kernel reads;
// * `B` as an operand over an operand — `Width(In(1)) / Width(Out(0))`, not
//   a plan dimension and not a param.
//
// They are `crate::x::attn`'s now, in the `unit!` beside the `fn`s that fire
// them and in the doc comments of those `fn`s.

// `DSA_INDEXER_ROWS` and `DSA_INDEXER_SIGS` DELETED, unit-only crossing —
// the device text is `crate::x::attn::dsa_indexer`'s.
//
// **AND ALL THREE TABLE ROWS HAVE NOW CROSSED TOO**, so the two sentences
// this comment used to end on are both retracted. It said the host programs
// *"never left `driver-cuda/src/fire/dsa_indexer.rs`"* and that
// `dsa_index_topk_mask`'s *"row stays there"*. Neither is true: the three
// host programs are `x::attn::dsa_index_{knorm_rope,q_rope,topk_mask}_bf16`,
// `fire/dsa_indexer.rs` is deleted, and `table/attn.rs` carries none of the
// three. They are three `contract!`s, one bind and two `none:` arms.
//
// **What was right is WHY the two are still refused**, and this comment had
// it first. It said two of the three rows were unsourced because `n_heads`,
// `head_dim`, `rope_dim` and `topk` arrive on the statement's parameter
// channel, and *"a JIT row that guessed which statement parameter carried
// which would bind three integers in an order nothing reports"*. The
// crossing measured the statements and found the sharper version: for
// `q_rope` and `knorm_rope` the parameters are not merely unlabelled, they
// are ABSENT -- `dsl::cuda::dsa_index_q_rope` records no params at all and
// `rope_dim` is in no shape, no param and no context anywhere. `topk_mask`
// records three, which is exactly why it is the one that binds.
//
// Everything these rows argued travelled to the new home, in the module doc
// and in the three `fn` docs beside the declarations:
//
// * `LaunchRule::PerRow` and NOT `Rms` for `knorm_rope`. The launcher is
//   `<<<tokens, kBlock = 256, 0>>>`; `Rms` requests thirty-two bytes of
//   dynamic shared memory that no launcher here passes and no kernel here
//   reads — `block_sum`'s warp buffer, which this shape has no reduction to
//   need, because its reduction is a static `__shared__ float red[256]`.
//   Harmless in effect and wrong as a contract: *"a rule is meant to
//   REPRODUCE its launcher, and one that asks for memory the launcher did not
//   is a rule nobody can check against the `<<<>>>` it came from."*
// * `LaunchRule::Unstated` for `q_rope`, and WHY no rule can state it:
//   `dsa_indexer.cu:34-35` is `block = round_up(n_heads, 32)` with a one-warp
//   floor. ONE THREAD PER HEAD. Every rule that sizes a block on a row sizes
//   it on the row's WIDTH, and `idx_q`'s row is `n_heads * head_dim` — the
//   two differ by 64 or 128, so `RouteRows` would open 128× the block. The
//   block is a statement PARAMETER, not a rectangle. In fn-world the
//   objection evaporates: a `Launch` is a `fn`'s literal.
// * `LaunchRule::RowScores` for `topk_mask`, and that
//   `runtime::launch::row_scores` was ported FROM this launcher — grid, block
//   and the dynamic allocation all agree, `rows * 4` being
//   `tokens * sizeof(float)` written twice. **The shared allocation is why it
//   is neither `Rms` nor `PerRow`**: the kernel declares
//   `extern __shared__ float logit[]` and fills `logit[0..nkeys)` where
//   `nkeys = blockIdx.x + 1` — one float per KEY. At `Rms`' thirty-two bytes
//   the last row of a 4 096-token prefill would select its top-k from eight
//   floats it wrote and 4 088 it did not; at `PerRow`'s zero, from none.
//   Neither faults. *"A launch that under-sizes shared memory does not fail,
//   it reads another block's floats"* — a wrong mask, a wrong attention, and
//   nothing downstream checks it.
// * `N` STAYS AN OPERAND although the grid opens over it, because the kernel
//   needs it a second time as the pitch of `mask` (`mrow = mask + i * N`) and
//   as the bound of its causal zero-fill. **An extent a rule recovers is not
//   an operand — an extent a kernel ADDRESSES with is.** Conversely `tokens`
//   is absent from `knorm_rope`, because one block per row IS `tokens` and
//   the kernel never addresses with it.
// * §60.6's SYMBOL SPLIT on all three. The device symbols are
//   `attn::dsa_index_{knorm_rope,q_rope,topk_mask}_dev` and the trace
//   symbols are `attn::dsa_index_knorm_rope_bf16`,
//   `attn::dsa_index_q_rope_bf16` and `attn::dsa_index_topk_mask`. `_bf16`
//   is DROPPED as well as `_dev` added, because these are
//   `template <class T>` and the ROW picks `T`. The two halves used to sit
//   side by side as constants in `fire/dsa_indexer.rs:45-61`; the crossing
//   put them one line apart instead -- each `contract!` names the trace
//   symbol and each host `fn` passes the `_dev` symbol to `raw::`, which is
//   the same bridge with the string in the call that uses it.
// * `template <class T>` and nothing else on `topk_mask` — `kBlock` is a
//   file-scope `constexpr int` the kernel strides by, not a template
//   argument, so there is no non-type argument to cite and the 256 a launcher
//   opens has to agree with `dsa_indexer.cuh`'s `kBlock` instead. It does.
//
// `DSA_INDEXER_ROWS` was `pub static`; swept for both consumer sets — the
// symbol strings and the identifier — across `crates/model/src`,
// `model-compiler/src/{dsl.rs,lower.rs}`, the hand-written `ffi::pie_k_*`
// arms and every `include_str!`/`#include` of `dsa_indexer.cuh`. The only
// reader was `DSA_INDEXER` itself, and the only `include_str!` is now
// `x::attn::dsa_indexer`'s.



// `HEAD_DIM_PAD_ROWS`/`HEAD_DIM_PAD_SIGS` CROSSED INTO FN-WORLD with their
// two constants, and the readings they carried are restated where the `fn`s
// are: `head_dim_pad.cu`'s `dim3 grid(num_heads, num_tokens)` with the head
// on `grid.x` and the row on `grid.y`; the 128 as a KERNEL REQUIREMENT and
// not a tuning number, because both kernels stride `d += kPadBlock`, so a
// narrower block leaves padding unzeroed on the way in and a head's tail
// unread on the way out and NEITHER FAILS, BOTH ANSWER; `num_tokens` and the
// stream leaving the operand list while `num_heads` stays; and WHICH SIDE IS
// PACKED being whichever end is `head_dim` wide, which is what
// `PACKED_HEADS_IN` and `PACKED_HEADS_OUT` were two constants for rather than
// one expression written twice.
//
// `table::attn`'s copies of those two constants went with the two table rows
// in the same change. `crate::x::attn`'s `PAD_HEAD_DIM` and
// `STRIP_HEAD_DIM` binds are where the two divisions are now written.

// `SPLIT_PACKED_ROWS`/`SPLIT_PACKED_SIGS` CROSSED INTO FN-WORLD, both rows,
// with every measurement they carried:
//
// * `SplitPacked`'s grid over the INPUT width (`q_dim + 2 * kv_dim`) being
//   WIDER than the launcher's over `max(q_dim, kv_dim)`, and why that is
//   safe in this direction and only this one — *"every loop strides by
//   `blockDim.x * gridDim.x` and bounds itself on its own output width, so
//   extra blocks contribute nothing but a shorter loop"*, while a grid
//   narrower than an output leaves the tail of every row unwritten;
// * six operands and not eight for the host-window form: `n_tokens` is the
//   rule's `grid.y` and the stream was never an operand, and the two widths
//   come off what is WRITTEN because a `[N, q + 2 * kv]` row cannot say where
//   the cut falls;
// * seven and not eight for the device-window form, where `n_max` is the
//   grid's second axis and never reaches the kernel;
// * **both halves of why `split_qkv_devwin` gets no rule and no `Source`** —
//   `grid.y` is the FIRE's lane count (`Ctx("rows_total")`) and not the
//   statement's rectangle, so under a peel the rule's `Dims::rows` would be
//   the tail's length while the kernel compares an ABSOLUTE `blockIdx.y`
//   against the device window and the rows past that length keep the
//   previous fire's bytes; and its buffers are BASE pointers by contract
//   while the binder resolves `In`/`Out` THROUGH the statement's window,
//   which is the double-window the `.cuh` refuses. Neither is an objection
//   to a DEVICE row, which is why both rows crossed and neither became a
//   bind.
//
// BOTH OF THOSE WERE OVERTAKEN and `attn::split_qkv_bf16_devwin` IS a bind
// now -- `x::attn`'s `SPLIT_QKV_DEVWIN`, over `x::attn::split_qkv_bf16_
// devwin`. The first was answered by `Cx::rows().total`, which is
// `DispatchCtx::rows_total` and whose own doc at `bind/facts.rs:319` names a
// `_devwin` launch. The second was never true of this symbol:
// `bind/mod.rs:3973` resolves every arg of a kernel whose name ends
// `_devwin` at row 0, BY SUFFIX, because *"their contract is BASE
// pointers"*. The double-window the `.cuh` refuses is refused in the binder,
// three hundred lines before a `Cx` exists.
//
// The two-symbol arrangement crossed with them: the device row is
// `attn::split_qkv_devwin` and the trace symbol is
// `attn::split_qkv_bf16_devwin`. `SPLIT_DEVWIN_SYMBOL` and
// `SPLIT_DEVWIN_DEVICE` bridged them in
// `driver-cuda/src/fire/split_packed.rs`, which is DELETED; the bridge is
// now the contract's `symbol` beside the unit row's string, which is the
// ordinary fn-world arrangement and needs no constants.

// `PACK_DENSE_MASK` CROSSED INTO FN-WORLD -- `crate::x::attn::pack_dense_mask`,
// as a UNIT AND NOTHING ELSE. Its two `DeviceKernel::PLAIN` rows went with it;
// there is no `contract!` and no `bind!`, because nothing in the tree launches
// either kernel. `driver-cuda/tests/launch_abi.rs:651-654` is the evidence and
// the verdict: *"`pack_dense_mask` and `pack_structured_mask` stood here and
// are GONE with `attn/pack_dense_mask.cu`, its `.hpp` and their two
// `table::driver_internal` rows. Empty consumer set on all five channels; not
// ported, per §60.1."* `driver-cuda/src/fire/page_mask.rs` plans the sideband
// arena and compacts the page CSR; it launches neither.
//
// EVERY MEASUREMENT THAT STOOD HERE, and where it is now:
//
// * The GEOMETRY. `pack_dense_mask.cu:93-94` -- `constexpr int BLOCK = 128`
//   then `device::pack_dense_mask<<<B, BLOCK, 0, stream>>>` -- and `:109-110`,
//   `constexpr int block = 128` then `device::pack_structured_mask<<<B, block,
//   0, stream>>>`. One block per lane at a fixed 128 threads with a stride
//   loop over that lane's output bytes, which is `LaunchRule::PerRowNarrow` to
//   the digit. And the caveat: the 128 is not a preference here the way it is
//   for the audio tower -- nothing folds warp partials, so the width is not a
//   numerics contract -- but it is still the launcher's, and a rule that
//   widened it to 256 would state a launch this tree does not make. Carried
//   into the `pack_dense_mask` module doc; no `fn` writes that `Launch` today
//   because nothing fires these.
//
// * `DeviceKernel::PLAIN` and not `""`. The constant is the row's STATEMENT
//   that this `__global__` has no template parameter list; the empty string is
//   what an unfilled field looks like. Checked by NVRTC in both directions and
//   `examples/argform_probe.rs` holds it: `plain<device::bf16>` is *"type name
//   is not allowed"*, a bare template path is *"cannot determine which
//   instance of function template ... is intended"*. Both rows still state it,
//   so `tests/units.rs` still checks it. NO DEVICE TEXT CHANGED, then or now.
//
// * `b` STAYS AN OPERAND even though the grid opens over it. Both kernels READ
//   it -- `if (b >= B) return;` and `if (request >= B) return;` are the first
//   lines of each -- so a declaration that dropped it on the grounds that the
//   rule recovers it would hand the kernel whatever the previous launch left
//   in that slot. `page_compact`'s two rows kept `num_requests` for the same
//   reason, and that sentence is why this one is repeated there --
//   `crate::x::attn::page_compact` carries it now.
//
// * EVERY OPERAND UNSOURCED, for the reason `page_compact`'s were:
//   `mask_indptr` is a host-built prefix sum the driver owns, `packed` is a
//   pre-zeroed driver allocation, and `p_page` is the dense mask's row stride.
//   No `Source` spells any of the three. That is why the two rows were
//   `table::driver_internal`'s and not `crate::table`'s, and why they did not
//   move `examples/migration_status`: a driver-internal row is one no trace
//   can state -- `FirePageMask` picked the packer, the DSL surface has no
//   statement for it.
//
// * `masks` WAS THE BLOCKER, from both sides. `Ty::StructuredMasks` is a `Ty`
//   that `runtime::args`' `is_pointer` does not admit, so `emit::crossing`
//   refused it and the row had NO generated entry point -- the emitter
//   recorded that as a comment naming the operand. The honest state was: the
//   descriptor array IS a device pointer, and saying so is a change to the
//   `Ty` vocabulary rather than to the row. Fn-world does not need that
//   change: `Abi` is an open set of impls and `is_pointer` is not consulted,
//   so `crate::x::attn::params` writes one impl and the declaration under it
//   is the first statement of this kernel's full signature that anything
//   checks.
//
// Nothing here may name the same root: a second `unit!` over the same text
// compiles it twice and `unit_of` answers with whichever won.

/// The units the small half of `attn` compiles.
///
/// Separate from [`UNITS`] so the heavy half can be appended without either
/// list being rewritten — see the marker at the end of this file.
pub const UNITS_SMALL: &[Unit] = &[
    // `SOFTCAP`, `ATTN_SINK`, `ATTN_RES`, `HEAD_DIM_PAD`, `PACK_DENSE_MASK`
    // and `DSA_INDEXER` CROSSED INTO FN-WORLD — `crate::x::attn`, whose
    // `UNITS` `families::ALL` lists beside this one. Their device text, their
    // rows and their host programs are all there; nothing here may name the
    // same roots, because a second `unit!` over the same text compiles it
    // twice and `unit_of` would answer with whichever won.
    // `ATTENTION_NAIVE` and `PAGE_COMPACT` left this list with their
    // crossing. IT IS NOW EMPTY, and that is the small half of `attn`
    // finished: every root it held is `crate::x::attn`'s.
];

/// The units `attn` compiles: the small half's, then the heavy half's.
///
/// **The one line both halves of this migration touch**, and now the
/// concatenation the comment below asked for. Built the way
/// [`crate::unit::UNITS`] builds its own — a `const fn` that fills a fixed
/// array, because `Unit` is `Copy` and neither `concat` nor iterator chaining
/// is const. Order is not semantic; a unit's position is its slot in the
/// module cache. It is stable, which is what keeps a diff readable.
pub static UNITS: &[Unit] = &concat_halves();

const fn concat_halves() -> [Unit; UNITS_SMALL.len() + UNITS_HEAVY.len()] {
    let mut out = [EMPTY; UNITS_SMALL.len() + UNITS_HEAVY.len()];
    let mut w = 0;
    let mut i = 0;
    while i < UNITS_SMALL.len() {
        out[w] = UNITS_SMALL[i];
        w += 1;
        i += 1;
    }
    let mut j = 0;
    while j < UNITS_HEAVY.len() {
        out[w] = UNITS_HEAVY[j];
        w += 1;
        j += 1;
    }
    out
}

/// A slot to fill and never a unit anything fires: it names no source and
/// holds no rows, so `unit_of` cannot return it. [`crate::unit`] keeps one
/// for the same reason and says so at greater length.
const EMPTY: Unit = Unit { name: "", root: "", rows: &[], options: &[] };

// ===========================================================================
// HEAVY HALF APPENDS BELOW THIS LINE — `attn/kv_paged`, `dsv4_compress`,
// `qkv_fused`, `attention_naive_paged`, `attention_flashinfer*`.
//
// `mla_paged`, `kimi_mla`, `attention_mla_naive` and `attention_xqa*` were
// on this list and are not any more: the first three are `crate::x::attn`'s
// and the fourth is `crate::x::xqa`'s.
//
// Declare `pub const UNITS_HEAVY: &[Unit] = &[...]` here, then change the ONE
// line above to the concatenation:
//
//     pub static UNITS: &[Unit] = &concat_halves();
//
//     const fn concat_halves() -> [Unit; UNITS_SMALL.len() + UNITS_HEAVY.len()] { ... }
//
// Nothing above the first marker needs to move, and nothing above it should:
// the small half's rows, sigs and units are complete and gated.
// ===========================================================================

// ===========================================================================
// HEAVY HALF — owned by the `attn` heavy-files migration.
//
// `kv_paged`, `dsv4_compress`, `kimi_mla` and `mla_paged`, in that file
// order. Nothing above the marker moved; the only shared line is `UNITS`,
// which is now the concatenation the comment above asked for.
//
// `qkv_fused` IS a unit now, and no longer THIS file's — it crossed whole to
// `crate::x::attn::qkv_fused`, and the note where it stood says what its
// eleven `#`-suffixed symbols cost to carry. `attention_naive_paged` is
// split — a `.cuh` of device text plus a `.cu` that keeps only its `<<<>>>`,
// probed against NVRTC 13.0 for `compute_89` and producing PTX — and is still
// not a unit, because a unit with no rows is refused and it has no row.
// `attention_mla_naive` is not split at all. `mla_paged` WAS in that list and
// is a unit now — see `crate::x::attn::mla_paged`.
//
// Kernel by kernel, with the launcher each refusal was measured against, and
// what closed it:
//
//  * `mla_paged.cuh`'s `write_mla` is a ROW — `mla_paged.cu:111`,
//    `<<<total_tokens, 256, 0, stream>>>`, which IS `LaunchRule::PerRow`. The
//    blocker was the spelling and not the shape, and the spelling is fixed:
//    `DeviceKernel::PLAIN` names a `__global__` with no template parameter
//    list by its bare qualified path, which NVRTC lowers and
//    `cuModuleGetFunction` resolves. No device text changed.
//  * **CLOSED.** `mla_paged.cuh`'s `mla_prepare<256>` — `mla_paged.cu:74`,
//    `<<<dim3(total_tokens, 1 + q_blocks), 256, 0, stream>>>`, where
//    `q_blocks = ceil(heads / heads_per_block)` and `heads_per_block` is
//    itself computed on the host from `qk_rope_head_dim / 2`. The refusal was
//    that NO RULE COMPUTES `1 + ceil(...)`, and `LaunchRule::MlaPrepare` now
//    does, off `Dims::rotary_dims` rather than `Dims::head_dim` — an MLA head
//    is `kv_lora_rank + qk_rope_head_dim` and only the rope tail turns. The
//    `1 +` is the KV lane (`blockIdx.y == 0` normalises the latent, rotates
//    `k_pe` and writes the page); see the rule's doc. The row is in
//    `crate::x::attn::mla_paged`'s `unit!` and does NOT claim the
//    ahead-of-time symbol, which that module argues.
//  * **CLOSED.** `qkv_fused.cuh`'s
//    `qkv_decode_qk_norm_rope_write_kv_warp<HEAD_DIM, USE_ROPE_TABLE>`
//    (`qkv_fused.cu:58`, `:71`) and `qkv_decode_qk_norm_rope_write_kv<128, USE_ROPE_TABLE>`
//    (`:102`, `:127`). `USE_ROPE_TABLE` is selected by `rope_table != nullptr`
//    — a POINTER-NULL test, which no `Term` could spell. `Term::Aligned`
//    holds of address 0, so an alignment clause would choose the table arm
//    for a fire that published no table and the kernel would dereference
//    null. `Term::Present { operand, value }` is the term that was added for
//    it: it reads a `Fact::Address`, faults on every other kind, and
//    `Specialisation::agrees` refuses it over a scalar operand and over a
//    pointer the row does not declare nullable. `QKV_DECODE_BLOCK` and
//    `QKV_DECODE_WARP` were the two pairs; both went with `qkv_fused`, where
//    the null test is a host `if` and `Term::Present` has nothing left to
//    select. The term itself stays — `WRITE_KV`'s arms still use it.
//
//    **`HEAD_DIM` is a SECOND selector on the same kernels and is NOT
//    reproduced.** `qkv_fused.cu:81`, `:85` and `:89` expand the macro at 64,
//    128 and 256 under `if (head_dim == …)`. `Term::Multiple { of: 64 }`
//    holds of 192 as well, so an ordered arm list would send a 192-wide head
//    to the 64 expansion — `ELEMS_PER_THREAD = 2` where 6 is needed, which is
//    §21.14's permutation and not a fault. A `Term::Equals { value: i64 }`
//    was refused instead of added, because it would make `Equals { ptr, 0 }`
//    well-formed: `Term::Present` spelled as its own negation, over an
//    operand where the two are not the same question. So the four decode rows
//    PIN 128 in `elem` and in their symbols, carry no `Source`, and are not
//    dispatchable.
//  * **CLOSED.** `qkv_fused.cuh`'s `qkv_packed_qk_norm_rope_vnorm_write_kv<256>`
//    — `qkv_fused.cu:248`, `<<<dim3(num_rows, num_q_heads + num_kv_heads),
//    256, 0, stream>>>`. The refusal was that no rule opens a grid axis over
//    the SUM of the two head counts, and that `Rule::GatedRms` — `[rows,
//    kv_heads, 1]` at 256 with `smem = 0` — is the same block, the same smem,
//    and a grid.y short by every query head. `LaunchRule::RowsPackedHeads`
//    states the sum. Fired on an L40S against a raw `cuLaunchKernel` at the
//    launcher's own geometry, over four shapes and both `row_valid` arms:
//    1 878 016 bytes compared, 185 856 values written, **0 differing**. The
//    `GatedRms` near-miss at the same shape wrote 6 144 of 24 576 query
//    values and 0 of 6 144 page values — the truncation the refusal
//    predicted. `tests/launch_rules.rs::fires` holds both numbers.
//  * **STILL REFUSED.** `attention_naive_paged.cuh`'s `naive_paged_attn<128>`
//    (`:111`, `:198`, `:248`) and `naive_paged_decode<128>` (`:150`). The
//    SHAPE is now stated — `LaunchRule::PagedScores` and
//    `PagedScoresDecode` compute the three-axis grid and the DYNAMIC
//    `(head_dim + 128) * sizeof(float)`, which no rule did before and which
//    `SdpaVector`'s `(rows + 256) * 4` gets wrong by adding the block to the
//    wrong extent. What still blocks the ROWS is the OPERANDS: both kernels
//    take `device::KvScheme` and `device::KvDType` BY VALUE
//    (`attention_naive_paged.cuh:187` and `:198`, `enum class … : u8`), and
//    `kernels::Ty` has no variant for an enum class — `runtime::args`' whole
//    bindable set is the pointer kinds plus `I32 | U32 | F32 | Usize | I64 |
//    Bool | Stream`. Adding one is a change to `crates/kernels/src/lib.rs`
//    beyond `LaunchRule` and to `runtime/args.rs`, which is where the
//    type-check and the binding live. Until then the rules are stated, pinned
//    against the launcher in `tests/launch_rules.rs::transcribed`, and reach
//    no kernel.
//  * `attention_mla_naive.cu` keeps its `cudaFuncSetAttribute` opt-in to
//    200 KB of shared memory behind a `std::call_once` — host state no
//    `LaunchRule` can carry — so it is not split either.
//
//    **THIS REFUSAL IS CLOSED AND IT WAS WRONG TWICE OVER.** The file is
//    split; the unit is `crate::x::attn::mla_naive::MLA_NAIVE` — it has since
//    CROSSED INTO FN-WORLD, see the note at the end of this file — and its
//    two rows are `attn::mla_naive_paged` and `attn::mla_mma_paged`. Both
//    halves of
//    the objection failed: the opt-in is not host state a rule has to carry
//    (`runtime::module::raise_dynamic_smem_cap` performs it inside
//    `KernelModule::fire`, once per `(CUdevice, CUfunction)` above a 48 KiB
//    high-water mark, driven by `Launch::smem` alone), and the 200 KB was
//    never reachable — the scalar kernel's allocation is `(8 * CKV + 16) * 4`
//    against a `CKV <= 512` refusal, so 16 448 bytes is its ceiling, a third
//    of the default it was said to exceed.
//
//    Worth recording WHY it read as a blocker for so long, because the shape
//    recurs: the sentence names a real mechanism (`std::call_once`) and a
//    real number (200 KB) and is refutable on both, but nobody refuted it,
//    because a refusal that cites C++ looks like a measurement of the C++.
//    It was a measurement of what the RULE could carry, made before the
//    runtime grew the thing that carries it — and it was never re-asked. The
//    actual blocker was in neither half and is not mentioned: the file was
//    MIXED, two `__global__`s and four host functions with `<mutex>`,
//    `<stdexcept>` and `<cuda_runtime.h>` in one header, so it could not be a
//    unit root at all no matter what any rule could state.
//
// `attention_flashinfer*.cu` and `attention_xqa*.cu` are out of scope by
// construction: FlashInfer migrates by vendoring its own headers (§14, and
// `source::VENDOR` already carries them), and XQA is a self-contained island
// with its own generated sources.
// ===========================================================================

// `KIMI_MLA` STOOD HERE — kimi_k3's two latent-attention preparation
// kernels, and its doc said *"the cleanest split in the family: both
// launchers were already exactly a ported rule, so the device half came out
// whole and cost nothing but the move."*
//
// **CROSSED INTO FN-WORLD** — `crate::x::attn::kimi_mla`, and the first FULL
// crossing of this family since `softcap`: unit, two contracts and two
// binds, so `table::attn`'s two rows went with it.
//
// That sentence is still true and is now the reason the crossing was cheap.
// What it did not say is what the move fixed: the launchers were a ported
// rule and the ROWS were not, because `kimi_split_q_b`'s row described the
// launcher — `Rows` and three `Param`s — and the JIT, having no launcher,
// formed its grid from `LaunchRule::Elementwise` instead. That is
// `rows * out_width(0)`, and out 0 is the NOPE half of a tensor the kernel
// splits in two, so the launch was short by `rope / (nope + rope)`. A `fn`
// forms the product the launcher formed. `crate::x::attn::kimi_mla` carries
// the measured bytes.
//
// `KIMI_MLA_ROWS[1]`'s `<device::bf16, 256>` argument survives the move
// verbatim, and its ARGUMENT survives with it: `BLOCK_DIM` sizes
// `__shared__ float buf[BLOCK_DIM]` and fixes the halving tree, so the
// instantiation width and the launch width are one number, and a row that
// let the template default supply it was one edit away from a 512-wide
// reduction under a 256-wide launch.

// `DSV4_COMPRESS` STOOD HERE. The root CROSSED INTO FN-WORLD as
// `crate::x::attn::dsv4_compress` — all ten rows, transcribed one for one,
// because this file had already stated every one of them completely.
//
// The doc that stood here argued the two things the crossing did not change
// and one it did. Unchanged: `template <class T = device::i32>` on the
// boundary-meta pair is what made them NAMEABLE while every call site in
// `kernels-cuda` kept compiling, and `DeviceKernel::instantiation()` still
// spells the default out because an unwritten default and a written one are
// two objects that happen to agree. Also unchanged: the §60.6 SYMBOL SPLIT,
// carried verbatim, because `driver-cuda/src/fire/dsv4_compress.rs` fires the
// four `_dev` names by string.
//
// CHANGED: the header's whole "which launchers became rows and which did not"
// section, and `dsv4_compress.cuh:44-64` with it. Every reason it gives is a
// reason a `LaunchRule` could not state a geometry — a shared-memory size off
// an operand width, a `[32,256]` clamp where the rule clamps `[32,128]`, a
// 128-wide block where `Elementwise` is 256, a kernel with no element type at
// all. A host `fn` states geometries, so none of those is a blocker any more,
// and four of the launchers named there have been Rust since the pass that
// deleted the `.cu`.
//
// Two of its six table rows went with it — `dsv4_compress_gather_paged_bf16`
// and `dsv4_store_comp_entries_bf16`, both UNBOUND, both for a reason that
// belongs on the record HERE because this file states the opposite: the
// device rows below annotated the gather `state_kv <- Source::In(0)` through
// `boundary_req <- Source::In(4)` with `ratio` and `coff` as `Param`s, and
// **`dsl.rs:4684` records that statement with one input and no parameters.**
// Those `Source`s described a statement that does not exist. Nothing compared
// them, because a device row's sources are read by nobody.

// `KV_PAGED` STOOD HERE — `attn/kv_paged.cuh`, the paged KV cache's
// appenders, quantised writers and dequantisers, and the largest root left
// in this family. **CROSSED INTO FN-WORLD, BOTH HALVES** —
// `crate::x::attn::kv_paged`, and `table::attn`'s four rows over this root
// are GONE with it.
//
// A row's survival is decided by whether a `contract!` names its symbol, and
// all four are named now: `attn::write_kv_to_pages`,
// `attn::write_kv_explicit_bf16`, `attn::write_kv_explicit_bf16_devwin` and
// `attn::dequant_kv_cache_layer_to_bf16_active`. Half A left them standing
// because their host programs were in `driver-cuda/src/fire/kv_paged.rs`;
// Half B moved the seven programs here, and four `Cx` queries
// (`first_token`, `num_pages_in_batch`, `w_page_d`, `w_off_d`) closed the
// last gap. Three bind; devwin is a `none:` arm because `win_d` has no
// producer in `AttnCtx` at all.
//
// Half A also said the block was that `kernels-cuda-new` cannot call
// `driver-cuda`. **True, and not the reason** — the dependency runs the
// other way. The reason was a driver RESOURCE, and there was none: the test
// is answerable in one line, name the resource, and if you cannot it is a
// move.
//
// §52.11 pointed the same way and is now discharged:
// `execution::tests::a_walk_is_only_a_walk` requires `unit_of(sym)` to be
// `None` for every walked symbol, and **no symbol in the new unit is any of
// those four**. §60.6's `_dev` suffix bought exactly that, which is why the
// unit could exist while the walks stood; the four `Walk`s are retracted in
// the same change as the contracts, and the suffix stays because the device
// rows still need names their host programs do not collide with.

// `MLA_PAGED` STOOD HERE — the MLA cache's append and its preparation pass,
// both header kernels, as two rows. **CROSSED INTO FN-WORLD** —
// `crate::x::attn::mla_paged`, UNIT-ONLY, and `table::attn`'s two rows STAY.
//
// A `unit!` moves device text; only a `bind!` retires a row. Both host
// programs have been `driver-cuda/src/fire/mla_paged.rs` since §60.6 and
// neither is reached through a `bind!`, so the crossing moves the text and
// touches nothing else — the fifth root in this family to cross that way,
// after `split_packed`, `pack_dense_mask`, `dsa_indexer` and `mla_naive`.
//
// `LaunchRule::MlaPrepare` is what the argument above bought and what the
// crossing gives back: `crate::x::attn::mla_paged`'s `fn mla_prepare` states
// `device::i32(256)` and its own doc carries the `blockIdx.y - 1` reading
// verbatim, because a fn-world launch is written where it is fired and the
// rule has one less consumer. The rule itself is NOT retired here — the two
// rows still name it and `tests/launch_rules.rs::transcribed` still measures
// it — but the reason it had to exist is now stated in two places, which is
// one more than a derived thing should need and is worth revisiting when the
// rows go.
//
// The `Dims::rotary_dims` measurement is the part to keep whichever world
// wins: an MLA head is `kv_lora_rank + qk_rope_head_dim` = 576, so a rule
// reading `head_dim` computes `heads_per_block = 1` where the launcher
// computes 8, and opens 129 lanes where the launcher opens 17.

// `ATTN_SCORE_FOLD`, `ATTN_SCORE_FOLD_ROWS` and `ATTN_SCORE_FOLD_SIGS` STOOD
// HERE, and the root CROSSED INTO FN-WORLD as
// `crate::x::attn::attention_flashinfer` -- one unit, one row, one symbol.
//
// `attn::attn_score_fold_heads` is `crate::x::attn::ATTN_SCORE_FOLD_HEADS`,
// a contract with a `none:` arm, and `table::attn`'s row is GONE. The host
// program does NOT move: `driver-cuda/src/fire/attn_score.rs` is 1,548 lines
// whose live consumers -- `fire/scratch.rs`, `fire/stage_hooks.rs`,
// `fire/launch.rs`, `bind/mod.rs` -- are about score STAGING and not about
// this launch. Moving the fold would move a `Launch` and leave its staging
// behind.
//
// §60.6's `_dev` SPLIT WAS NEVER APPLIED HERE and the crossing had to do it
// first. The device row's symbol was `attn::attn_score_fold_heads`, the same
// string as the table row and the same string `dsl::cuda` states, so a
// `contract!` on it would have made a contract symbol a unit row's symbol --
// which `execution::a_walk_is_only_a_walk` and `migration_status`'
// `refused_set()` both read. The device row is
// `attn::attn_score_fold_heads_dev` now and `fire/attn_score.rs`'s
// `FOLD_SYMBOL` is the ONE firer, resolving through `unit::unit_of` rather
// than a table, which is what made the rename one line.
//
// The `LaunchRule::Unstated` argument crossed verbatim into the unit doc,
// because none of it is retracted: `dim3(num_requests, 64u)` at 256 threads,
// `64` in no `Dims`, the body striding `i += blockDim.x * gridDim.y` so every
// fanout produces the same floats and any parity test would pass a rule wrong
// by 64x in blocks; and the two literal grid axes in all of `csrc/src` being
// DIFFERENT literals in one file, so there is no rule to extract. In fn-world
// the conclusion needs no rule to decline -- a `Launch` is written by whoever
// fires it.

/// The one `__global__` `attn/attention_xqa.cuh` holds — and the LAST one the
/// `kernels-cuda` archive held.
///
/// Not a template, for the same reason `attn/attention_flashinfer.cuh`'s one
/// row is not (it is `crate::x::attn::attention_flashinfer`'s now): every
/// buffer is a page-table integer width fixed by the KV cache's own layout,
/// there is no element type to vary, and a `template <int BLOCK>` would name a
/// parameter the body never mentions.
pub static ATTN_XQA_ROWS: &[DeviceKernel] = &[DeviceKernel {
    sig: &ATTN_XQA_SIGS[0],
    template_path: "attn::device::build_xqa_metadata",
    elem: DeviceKernel::PLAIN,
}];

#[rustfmt::skip]
static ATTN_XQA_SIGS: [KernelSig; 1] = [
    // `LaunchRule::Unstated`, and it is the SECOND refusal in this family with
    // the same mechanism behind it as `attn_score_fold_heads`' — which is what
    // makes it evidence rather than a coincidence.
    //
    // The launcher was `attention_xqa.cu:313`:
    //
    //     build_xqa_metadata_kernel<<<num_requests, 128, 0, stream>>>(
    //
    // `dim3(num_requests)` at 128 threads, nothing shared. The grid axis IS
    // statable — [`LaunchRule::PerRequest`] opens exactly `[requests, 1, 1]` —
    // and the BLOCK is not: `per_request` is 256 wide (`runtime::launch`'s
    // `BLOCK`) and this launcher is 128.
    //
    // A rule cannot be chosen by measuring bytes here either. The page loop is
    // `for (p = threadIdx.x; p < max_pages_per_seq; p += blockDim.x)` and the
    // sequence length is written under `if (threadIdx.x == 0)`, so the block
    // width is a pure STRIDE: 256 computes the same page table and the same
    // sequence lengths as 128, in half the iterations, with twice the threads.
    // `PerRequest` would pass any parity test ever written for this kernel and
    // would silently double the launch's occupancy cost on the one kernel in
    // the tree that runs once per fire rather than once per layer.
    //
    // So the honest reading is that this row's geometry is one axis short of
    // `PerRequest` and the missing axis is a literal — the same shape
    // `attn_score_fold_heads` refused, off the other side of the rectangle.
    // A parameterised `PerRequest(block)` would be vocabulary growth for two
    // literals in one family, which `new-horizon.md` §10.5 forbids at exactly
    // this size. `driver-cuda`'s `fire/xqa.rs` builds the `Launch` by hand and
    // carries the 128 as a named constant with this line cited beside it.
    //
    // `whole` because the prepare is fire-wide by construction: it writes one
    // dense page-table row per REQUEST of the whole fire, which is what
    // `Prepare::FireWide` means on the row that reads it back
    // (`table::attn`'s `attn::attention_xqa_decode_bf16_prepared`).
    //
    // UNSOURCED, whole, on purpose. Six of the eight operands have honest
    // `Source`s waiting for them — `KvPageIndices`, `KvPageIndptr`,
    // `Attn("kv_last_page_lens_d")`, `Rows`, `KvPageSize` — but `page_table`
    // and `seq_lens` are sub-buffers the DRIVER carves out of
    // `AttentionWorkspaceView::float_buffer` at offsets no `Source` spells,
    // and `max_pages_per_seq` is the bucketed stride, not the operand the
    // caller passed. Half a row sourced is worse than none (`families/rope.rs`
    // is the worked example), and this row is fired by hand regardless, so
    // there is nothing a partial binding would buy.
    kernel!(build_xqa_metadata "attn::build_xqa_metadata",
        file = Some("attn/attention_xqa.cuh"),
        launch = LaunchRule::Unstated,
        whole = true,
        operands = operands![
            kv_page_indices: U32s,
            kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            page_table: I32sMut,
            seq_lens: U32sMut,
            num_requests: I32,
            max_pages_per_seq: I32,
            page_size: I32,
        ]),
];

/// XQA's fire-wide prepare, as a JIT unit.
///
/// The last `__global__` to leave `kernels-cuda`, and the only one whose `.cu`
/// half could be deleted outright rather than kept beside it: the launcher had
/// no shim entry (`attn::prepare_attention_xqa_decode_bf16` is in no
/// [`crate::table`]) and no C++ caller, so its entire consumer set was the
/// obligation `Prepare::FireWide` states on a DIFFERENT row. That obligation
/// is discharged in Rust now — `driver-cuda`'s `fire/xqa.rs` — and the C++ is
/// gone.
///
/// **This unit does not make the XQA decode a JIT path.**
/// `attn::attention_xqa_decode_bf16_prepared` is still ahead-of-time and still
/// the shim's one XQA entry: its body ends in FlashInfer's
/// `launchMHAFlashInfer_xqa_gqa*` entry points, which are upstream HOST
/// functions that do their own launching, and `new-horizon.md` §50.1's
/// measurement applies to them unchanged — there is no device text of ours to
/// carry. What moved is the half that was ours.
pub const ATTN_XQA: Unit = Unit {
    name: "attn/attention_xqa",
    root: include_str!("../../csrc/src/attn/attention_xqa.cuh"),
    rows: ATTN_XQA_ROWS,
    options: &[],
};

/// The three `__global__`s of `attn/attention_score_post.cuh`.
///
/// # The refusal that named its own expiry, and the thing that asked
///
/// The header carries a section titled *"No unit, and that is deliberate
/// rather than pending"*, and its argument was sound at the time:
///
/// > a table row is a thing a model text can STATE — these are internal
/// > steps of a dispatch that has its own row already.
///
/// The premise is what changed, not the reasoning. "A dispatch that has its
/// own row already" meant `attn::attention_flashinfer_{decode,prefill}_capture_bf16`,
/// and both of those are [`crate::execution::Control::Switch`] walks whose
/// bodies are **host C++ that is going to be Rust**. A Rust body cannot fire
/// a kernel it has no row for: [`crate::unit::unit_of`] resolves a SYMBOL, so
/// the composing driver needs one row per launch it makes, whether or not a
/// model text ever spells it. That is the same inversion
/// `families::ssm`'s `causal_conv1d_prefill_noact_bf16` records — an EMPTY
/// consumer set was the whole objection, and the consumer is the port.
///
/// So these rows are not for `model-compiler`. They exist so that
/// `driver-cuda`'s Rust — which is where all host code lives — can fire the
/// three post-kernels by name. **There is already a function in the right
/// place to do it**: `driver-cuda/src/fire/attn_score.rs::publish` runs on
/// the fire's stream immediately after the capture dispatch and already
/// fires `attn_score_fold_heads` this way. Its module doc carries the move
/// and its one cost (a golden-hash regeneration in
/// `tests/attn_score_parity.rs`); `new-horizon.md` §53.10 is the same
/// finding at length.
///
/// # Why all three are [`LaunchRule::Unstated`]
///
/// Every grid is quoted verbatim in the row comment beside it, and not one of
/// them is an extent of an operand:
///
/// * `dim3(num_requests, num_q_heads)` — statable in principle, but there is
///   no `PerRequestPerHead` rule and adding one for a single launcher is what
///   §10.5 forbids.
/// * `dim3(num_requests, num_q_heads, window)` — THREE extents, the third a
///   host policy constant that arrives as a dispatch argument and appears
///   in no [`crate::runtime::Dims`] field.
/// * `dim3(num_requests, 32)` — the second literal grid axis this file pair
///   holds, and the sibling row above already measured why a
///   `PerRequestFanout(N)` covering both `64u` and `32u` would be vocabulary
///   growth for two constants that share a file.
///
/// The driver builds all three `Launch`es by hand, the way
/// `driver-cuda/src/fire/attn_score.rs` already builds the fold's.
///
/// # No `in_place`, and that is not an omission
///
/// Two of the three read and write `scores` through one pointer, so the
/// field looks like it wants `&[(0, 0)]`. It does not. `in_place` is read by
/// `lower::Buffers` to give a LOWERED op's output its operand's offset, and
/// nothing lowers these: they have no `table` symbol, no model text spells
/// them, and the aliasing is a fact the composing Rust already knows because
/// it allocated the buffer and passes the same pointer twice. The sibling
/// `attn_score_fold_heads` states none either. A pair here would be an index
/// into an output list that does not exist.
///
/// # `abi::emit_device_typecheck` refuses all three, by design
///
/// Every row here is [`DeviceKernel::PLAIN`], and that emitter spells each
/// buffer operand as a pointer to the head of `elem` — which for a PLAIN row
/// is the sentinel `"(no template arguments)"`, refused by name at the
/// `contains('(')` guard. `families::sample` records the same outcome for its
/// consumer row and `attn_score_fold_heads` above is a third. The offline
/// arity/constness check is simply not available for a non-template, and
/// saying so is the honest result rather than a gap: NVRTC still names the
/// instantiation at run time through `nvrtcAddNameExpression`, and a drifted
/// operand list surfaces there.
pub static ATTN_SCORE_POST_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &ATTN_SCORE_POST_SIGS[0],
        template_path: "attn::device::attn_score_normalize",
        // Not a template. `kThreads` is a `constexpr int` inside the body and
        // the block width is fixed at 256 by the shared array it sizes, so
        // there is no argument to give and no arm that could differ.
        elem: DeviceKernel::PLAIN,
    },
    DeviceKernel {
        sig: &ATTN_SCORE_POST_SIGS[1],
        template_path: "attn::device::attn_prefill_score_normalize",
        elem: DeviceKernel::PLAIN,
    },
    DeviceKernel {
        sig: &ATTN_SCORE_POST_SIGS[2],
        template_path: "attn::device::attn_prefill_score_fold",
        elem: DeviceKernel::PLAIN,
    },
];

#[rustfmt::skip]
static ATTN_SCORE_POST_SIGS: [KernelSig; 3] = [
    // `driver-cuda/csrc/attn/attention_flashinfer.cu`, at the foot of
    // `dispatch_attention_flashinfer_decode_capture_bf16`:
    //
    //     const dim3 grid(static_cast<unsigned>(cache.num_requests),
    //                     static_cast<unsigned>(cache.num_q_heads));
    //     device::attn_score_normalize<<<grid, 256, 0, stream>>>(
    //         score_out, score_indptr_d, kv_page_indptr_d, kv_last_page_lens_d,
    //         cache.page_size);
    //
    // In place: `scores` is read and written by the same block, which is why
    // it is `BufMut` and why no second buffer appears in the operand list.
    // `kv_len` is derived from the page CSR inside the body rather than
    // passed — the header argues that at length and the row must not
    // "helpfully" add a length operand the kernel would ignore.
    kernel!(attn_score_normalize "attn::attn_score_normalize",
        file = Some("attn/attention_score_post.cuh"),
        launch = LaunchRule::Unstated,
        whole = true,
        operands = operands![
            scores: BufMut,
            score_indptr: I32s,
            kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            page_size: I32,
        ]),

    // Same file, in `dispatch_attention_flashinfer_prefill_capture_bf16`:
    //
    //     const dim3 norm_grid(static_cast<unsigned>(cache.num_requests),
    //                          static_cast<unsigned>(cache.num_q_heads),
    //                          static_cast<unsigned>(window));
    //     device::attn_prefill_score_normalize<<<norm_grid, 256, 0, stream>>>(
    //         score_out, score_indptr_d, qo_indptr_d, kv_page_indptr_d,
    //         kv_last_page_lens_d, cache.page_size, window);
    //
    // `window` is BOTH the third grid extent and the last operand, and that
    // is the honest reading of the launcher rather than a redundancy to
    // clean up: `blockIdx.z` selects the window row and the operand bounds
    // `rows = min(window, qo_len)` inside the body.
    kernel!(attn_prefill_score_normalize "attn::attn_prefill_score_normalize",
        file = Some("attn/attention_score_post.cuh"),
        launch = LaunchRule::Unstated,
        whole = true,
        operands = operands![
            scores: BufMut,
            score_indptr: I32s,
            qo_indptr: U32s,
            kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            page_size: I32,
            window: I32,
        ]),

    // Same file, immediately after the normalize above:
    //
    //     const dim3 fold_grid(static_cast<unsigned>(cache.num_requests), 32u);
    //     device::attn_prefill_score_fold<<<fold_grid, 256, 0, stream>>>(
    //         score_out, folded_out, score_indptr_d, qo_indptr_d,
    //         kv_page_indptr_d, kv_last_page_lens_d, cache.page_size,
    //         cache.num_q_heads, window);
    //
    // `32u` is an occupancy constant exactly as the `64u` one unit up is, and
    // the body strides over `gridDim.y`, so — the sibling's measurement,
    // which applies here word for word — any value produces the same floats
    // and no parity test could ever choose the rule. The driver carries the
    // 32 and the 256 as named constants citing this line.
    //
    // NOT in place: unlike its two siblings this one reads `scores` and
    // writes a separate `folded`, the same split `attn_score_fold_heads`
    // makes.
    kernel!(attn_prefill_score_fold "attn::attn_prefill_score_fold",
        file = Some("attn/attention_score_post.cuh"),
        launch = LaunchRule::Unstated,
        whole = true,
        operands = operands![
            scores: Buf,
            folded: BufMut,
            score_indptr: I32s,
            qo_indptr: U32s,
            kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            page_size: I32,
            num_q_heads: I32,
            window: I32,
        ]),
];

/// The capture post-kernels, as a JIT unit.
///
/// Three `__global__`s, no host code, one `#include "pie_device.cuh"` — the
/// cheapest unit in the family to compile and the one with the longest
/// argument behind it. See [`ATTN_SCORE_POST_ROWS`].
pub const ATTN_SCORE_POST: Unit = Unit {
    name: "attn/attention_score_post",
    root: include_str!("../../csrc/src/attn/attention_score_post.cuh"),
    rows: ATTN_SCORE_POST_ROWS,
    options: &[],
};

/// The units the heavy half of `attn` compiles.
pub const UNITS_HEAVY: &[Unit] = &[
    ATTN_SCORE_POST,
    ATTN_XQA,
];

// `ATTENTION_NAIVE_PAGED`, `ATTENTION_NAIVE_PAGED_ROWS` and
// `ATTENTION_NAIVE_PAGED_SIGS` STOOD HERE, and the root CROSSED INTO
// FN-WORLD as `crate::x::attn::attention_naive_paged` -- one unit, two rows,
// a host `fn`, a `contract!` and a `bind!`. `table::attn`'s row is GONE and
// so is its `device::JIT_DISPATCHED` line.
//
// WHAT THIS BLOCK SAID WAS BLOCKING IT WAS TRUE AND IS CLOSED. Both kernels
// take `device::KvScheme` and `device::KvDType` BY VALUE, adjacently, each an
// `enum class ... : ::std::uint8_t`, and this file's own doc recorded that
// `kernels::Ty` had no variant for one. It gained two -- not one, because the
// operands are adjacent and the same width, so a shared kind would make the
// SWAP type-check on every side this crate can check. `x::Abi` then had the
// same gap one level down, and `x/abi.rs:226` had already predicted its own
// first caller: *"an open set adds the impl with its first kernel."*
// `x::attn::kv_scheme` and `x::attn::kv_dtype` are that impl, beside the one
// kernel that needs them.
//
// THE NARROWING IS THE PART WORTH KEEPING. `x::cx`'s mirrors are
// `#[repr(i32)]` and the device's are one byte, so the crossing is `as u8` --
// four bytes on one side and one on the other, which is §3.2's hazard turned
// ninety degrees. It is written once, in the `Abi` impl, so no host program
// can spell it a second way.
//
// AND THE CROSSING RESTORED A PREDICATE THE ROW WORLD LOST.
// `attention_naive_paged.cuh:220` records that the deleted `.cu` read
// `kMaxHeadDim` in `check_head_dim_supported` -- *"the array and the
// predicate that keeps launches inside it are ONE constant, not two."* The
// `.cu` went; the predicate went with it; the generated JIT arm that replaced
// it opens a grid through a `LaunchRule` and A `LaunchRule` CANNOT REFUSE.
// So a head dim above 1024 has been reaching a kernel that indexes `acc[8]`
// past its end -- a wrong answer, not a crash. `x::attn`'s `fn` refuses it as
// `Refusal::Wide { what: "head_dim", at, max: 1024 }`.
//
// Both arguments this block made about the SIGS survive where the rows are:
// the two kinds of nullability (`k_scales`/`v_scales` null under
// `KvScheme::Native` means "not quantised"; `custom_mask` null means "causal,
// not custom") are in the prefill row's doc, and §21.14's test -- *does the
// new spelling make a wrong predicate well-formed?* -- is answered by
// `ArgValue::U8` becoming `Fact::Opaque` rather than `Fact::Int`, so
// `Term::Multiple { operand: scheme, of: 2 }` is a `Fact::Kind` fault rather
// than a sentence about the parity of a name.
//
// The decode row kept its shape and gained nothing: `naive_paged_decode` is
// `NoRow::KernelsInternal`, so it is a `unit!` row with no host program, and
// a `unit!` row has nowhere to put a `Source` -- the absence is the shape
// rather than a decision a later editor can quietly reverse.

// `QKV_FUSED`, `QKV_FUSED_ROWS` and `QKV_FUSED_SIGS` STOOD HERE, and the
// root CROSSED INTO FN-WORLD as `crate::x::attn::qkv_fused` — three `fn`s,
// eleven rows, the same eleven symbols.
//
// **The `#rope`/`#norope` names are carried VERBATIM and that is the finding.**
// They were a `Specialisation` spelling here: a base row plus arms, with
// `flags_are_covered` proving the base unreachable. fn-world has no
// `Specialisation`, so the obvious crossing renames them — and that would be
// a `NoLoweredName` at the first decode fire, because the host program fires
// by NAME: `warp_symbol(head_dim, rope_table)` and `block_symbol(rope_table)`
// return exactly these strings. Those two are
// `crate::x::attn::qkv_fused`'s now, moved out of
// `driver-cuda/src/fire/qkv_fused.rs` with the dispatch they serve.
// The suffix is legal there for a checkable reason —
// `x/abi.rs:824`'s `mangle` already lists `'#'` among the characters it
// replaces when it writes the typecheck TU. **The row world's arm spelling
// survives as a plain symbol**, which is the cheapest crossing a
// `Specialisation` can have: `x/norm.rs:1033` says they become `if`s, and
// here the `if` was already written and already in the driver.
//
// The long argument that stood here — that a `Specialisation` may NOT change
// a `LaunchRule`, because `:50-53`'s `WarpPackedHeads` (1-D at 256) and
// `:97-99`'s `RowsPackedHeadsNarrow` (2-D at 128) are two GEOMETRIES and a
// base row states one `launch` — is retired by the crossing rather than
// answered: in fn-world a `Launch` is written by whoever fires it, so two
// geometries are two `Launch` literals in one host `fn` and nothing has to
// agree with anything. `x::attn::qkv_fused::qkv_decode_fused_dispatch`
// already writes both — and it is `x::attn`'s now rather than
// `driver-cuda`'s, so the two `Launch` literals sit beside the `unit!` that
// declares the eleven rows they fire.
//
// BOTH table rows went with it, one commit apart.
// `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` is
// `crate::x::attn::QKV_PACKED_POST` and
// `attn::qkv_decode_qk_norm_rope_write_kv_bf16` is
// `crate::x::attn::QKV_DECODE_FUSED`, each a contract and a real bind. The
// decode form waited on `Cx::q_out` and nothing else; the sentence that stood
// here said it *"STAYS in `table::attn`, because its host program is
// `bind::service`-served"*, which was true for as long as the query did not
// exist. It was the LAST ROW IN `ROW_TABLES`.

// `MLA_PAGED_ROWS`, `MLA_PAGED_SIGS`, `KIMI_MLA_ROWS` and `KIMI_MLA_SIGS`
// STOOD HERE, 224 lines of them. Both roots crossed into fn-world —
// `crate::x::attn::mla_paged` (unit-only) and `crate::x::attn::kimi_mla`
// (unit, contracts and binds) — and every argument these four carried went
// with the text rather than being summarised here, which is the whole of
// what "nothing is written twice" buys:
//
// * `DeviceKernel::PLAIN` on `attn::device::write_mla` — the row's statement
//   that the kernel has NO template parameter list, as against the empty
//   string, which is what an unfilled field looks like. The `unit!` says it
//   in the same words. `crate::x::attn::pack_dense_mask` holds the two
//   refusals that make the distinction checkable rather than conventional.
//
// * `row_valid` is NULLABLE and `r` is `num_requests`, not the token count
//   the grid opens over — two extents in one launch, only one of which a
//   rule can recover. `crate::x::attn::mla_paged`'s `fn write_mla` doc.
//
// * `256` IS THE ROW'S TO STATE, twice over: `split_kv_a_norm<T, BLOCK_DIM =
//   256>` and `mla_prepare<BLOCK_DIM>` both declare `__shared__ float
//   buf[BLOCK_DIM]` and reduce by halving over it, so the instantiation
//   width SIZES AN ARRAY and fixes a tree. A row spelling `<device::bf16>`
//   and letting the default supply the rest was one edit away from a
//   512-wide reduction under a 256-wide launch — the upper half of `buf`
//   never written and the first halving step reading it, which is a
//   plausible number and not a crash. Both `unit!`s state the full argument
//   list and `crate::x::attn::rms` is the launch end of the same 256.
//
// * `attn::device::mla_prepare` keeps `device::i32(256)` rather than `256`,
//   because `DeviceKernel::instantiation` prefixes an `elem` that does not
//   begin `::` with `::pie_cuda_driver::kernels::` and only the functional
//   cast survives that.
//
// * `KIMI_MLA_SIGS[0]`'s `total` is the SOURCE's element count. That is the
//   one sentence these rows got right and the generated dispatch did not —
//   see `crate::x::attn::kimi_mla` for the measured shortfall, and
//   `table::attn` for why the crossing closed it.

// `DSV4_COMPRESS_ROWS` and `DSV4_COMPRESS_SIGS` STOOD HERE, five hundred
// lines of them, and every line's content is now either a `unit!` row in
// `crate::x::attn::dsv4_compress` or a sentence beside one.
//
// What the doc here said, and what is now checkable instead of stated: four
// of the six named a launcher `launch_abi.rs` classifies
// `NoRow::KernelsInternal` — called by kernels code and by no statement — so
// their operands carried no `Source`, on the argument that inventing one
// would put a guess in the table where an absence belongs. **In fn-world
// there is no place to put the guess:** a `unit!` row's operands are its
// `fn`'s parameter list and every one of them is `Source::Unbound` by
// construction, so the absence is the shape rather than a decision a later
// editor can quietly reverse.
//
// Three findings went with the rows and all three are carried:
//
//  * `PagedScoresDecode` got its third launcher a row where its first two
//    could not, because `compressed_attn_paged` takes no `KvScheme` and no
//    `KvDType` — thirteen operands, every one of them a pointer, an `int` or
//    a `float`. In fn-world that constraint is gone entirely; a `fn` takes
//    what the kernel takes.
//  * `combine_attn_outputs` stated `LaunchRule::Unstated` as a FINDING: the
//    grid is `PerHeadElementwise` to the digit and the block is not, because
//    the launcher clamps `head_dim` into `[32,256]` and the rule into
//    `[32,128]`, so on a head wider than 128 the rule answers with half the
//    threads and the kernel's stride loop makes that a slower kernel and
//    never a wrong answer — invisible, which is why it was declined rather
//    than claimed. `driver-cuda/src/fire/dsv4_compress.rs` writes that
//    `Launch` by hand and reproduces the clamp, and `SINK_BLOCK_MAX` in
//    `runtime/launch.rs` is still left alone.
//  * The gather's `RouteRows` rounds a head dim up to a warp and caps at
//    1024, which is legal ONLY because the kernel strides `d +=
//    blockDim.x`. `x::attn::dsv4_compress::route_rows` reproduces both and
//    says so.

// `KV_PAGED_ROWS` and `KV_PAGED_SIGS` STOOD HERE — twenty-five rows over
// fourteen kernels, and the family's largest single declaration. **CROSSED
// INTO FN-WORLD** — `crate::x::attn::kv_paged`, as twenty rows.
//
// Twenty and not twenty-five: the five `Specialisation` BASE rows are gone
// with the `Specialisation`s themselves, for the reason set out where they
// stood. The remaining twenty are the ten `HND_LAYOUT`/`UseFp8` arms, the
// four dequantisers, the four quantised writers and the two view builders —
// every kernel `kv_paged.cuh` defines, each named by exactly one row.
//
// Kept, because they are facts about the kernels:
//
//  * Every operand is a field the ahead-of-time launcher unpacked out of a
//    `KvCacheLayerView` — the k half of a scale array, the packed page base,
//    the active page list — and no `Source` spells any of them, so these
//    rows carried none and the unit's carry none either. The dequantiser the
//    driver DOES name is `attn::dequant_kv_cache_layer_to_bf16_active`,
//    which is a launcher over all four schemes and not a kernel; it is one
//    of the four `table::attn` rows that stay.
//  * `n` is `I64` because the kernel's parameter is `long long`, and it is a
//    `long long` because it indexes a page arena that is multiple gigabytes
//    at production page counts. Restating it as `Usize` would have bought a
//    bindable row by describing a signed parameter as unsigned.
//  * The fp8 pages are `U8s`/`*const u8` and not a format of their own: on
//    the device they are `__nv_fp8_storage_t`, which IS one byte, and the
//    format is the kernel's to interpret through the `fp8_kind` operand.
//    `kernels::Ty::Fp8Kind` is what let that operand be stated at all —
//    defaulting it would decode an E5M2 page as E4M3 and give a numerically
//    plausible wrong answer. `x::attn::kv_paged::fp8_kind` is the
//    declaration side of the same argument and carries the `x/abi.rs` patch
//    that retires it.
//  * `logical_n` and not `n` for `dequant_fp4_pages_active`: an fp4 page
//    holds two values per byte, so the count the grid covers is the LOGICAL
//    element count and every address inside the kernel halves it.
//  * `write_kv_at_positions` has no caller anywhere in the workspace. Its
//    two arms are carried anyway, exactly as `dsv4_compress_gather` is: they
//    were being instantiated before the crossing, and a transcription that
//    silently drops a kernel is a transcription nobody can check against the
//    thing it came from.

// ── `WRITE_KV`, `WRITE_KV_AT_POSITIONS`, `WRITE_KV_EXPLICIT`,
//    `WRITE_KV_EXPLICIT_DEVWIN` and `COPY_KV_CELLS` STOOD HERE ───────────────
//
// With `prefix()`, `TAKE_15`/`13`/`12`/`11`/`10`, `SPECIALISATIONS` and the
// `each_arm_names_the_instantiation_its_name_claims` test. **CROSSED INTO
// FN-WORLD** — `crate::x::attn::kv_paged`, and `table::attn`'s four rows
// over this root have since gone with Half B.
//
// **The five `Specialisation`s did not become `if`s here; they had already
// become `if`s and nothing had noticed.** `driver-cuda/src/fire/kv_paged.rs`
// picks the arm in Rust and fires it BY NAME through `fire::hand::fire` —
// `"attn::write_kv_bf16#hnd"` at `:878` against `"…#nhd"` at `:880`, and the
// same pair at `:718`/`:720`, `:1082`/`:1084` and `:217`/`:219`. So
// `runtime::fire::selects` was asked about none of these five symbols by the
// time this crossing began. What went is a second copy of a decision, not
// the decision.
//
// The five BASE rows went with them, and this is the load-bearing half of
// the edit. `attn::write_kv_bf16` and its four siblings carry a sixteenth
// operand `hnd_layout: Bool` **that no kernel takes** — it exists so a fire
// can hand the flag to `selects` and `TAKE_15` can drop it again. Deleting
// the `Specialisation`s alone leaves five rows with a bogus operand and no
// reader; deleting the base rows alone breaks `Specialisation::agrees`.
// **Neither is removable without the other, which is the same cycle
// `QKV_DECODE_BLOCK` and its six `quoted()` pins formed** — an artefact
// whose only citation is the artefact that exists only to cite it. Twice in
// one family is a pattern; it is worth looking for a third.
//
// Kept, because they are facts about the KERNELS and not about the data
// that described them:
//
//  * `write_kv<HND_LAYOUT>` takes the SAME fifteen parameters either way, so
//    a fire that fell through to the sixteen-operand base would bind sixteen
//    cells for a fifteen-parameter kernel and **succeed** —
//    `cuLaunchKernel` reads the count from the cubin and never reads the
//    sixteenth cell. No fault, no error, wrong instantiation. In fn-world
//    the base has no symbol to fall through to, which is a stronger
//    guarantee than `flags_are_covered` was.
//  * The negative control, measured on an L40S sm_89 through the shipped
//    fire path over five shapes and both layouts: agreeing arms give 0 of
//    220,800 bf16 cells differing; firing `write_kv<false>` where the flag
//    says `true` moves 34,273 of 55,200 cells **while writing the same
//    number of non-zero values**. A permutation, not a truncation — no
//    count, no norm and no tolerance check sees it. That is why the
//    `#hnd`/`#nhd` to `true_type`/`false_type` correspondence is written out
//    per line in `x::attn::kv_paged` rather than left to a reader's column
//    scan.
//
// **And the test that went was already failing.** It closed with
// `assert_eq!(checked, 14, "seven specialised kernels, two arms each")`, and
// `SPECIALISATIONS` has held five entries — ten arms — since
// `QKV_DECODE_BLOCK` and `QKV_DECODE_WARP` crossed with `qkv_fused`. The
// count was a literal beside the list rather than a property of it, so
// removing two entries left it stating a number nothing produced. It is the
// third artefact this family has found that a join or a count could not
// re-derive, after `device.rs:991` and `DSV4_COMPRESS_SIGS[4]`. **A literal
// that names a length is a length that is written twice.**

// ===========================================================================
// The XQA lattice
// ===========================================================================

/// One member of the XQA lattice: a `-D` set, and what it is for.
///
/// `attn/attention_xqa.cu` and its five siblings in `kernels-cuda` are each a
/// `#define` block, one `#include`, and a host launcher. The launcher is Rust
/// now (`driver-cuda/src/fire/xqa.rs`), the include is
/// `csrc/src/attn/attention_xqa_mha.cuh`, and the `#define` block is this —
/// [`crate::unit::Unit::options`], which `unit.rs:95` describes as *"NVRTC
/// options this unit needs and the others must not have"* and which is the
/// right hook for exactly this: sixteen object-like macros that configure one
/// upstream template body six ways.
///
/// # `options` is the right hook, and two nearby things are not
///
/// `unit.rs:95-125` warns that `options` is the WRONG hook for two cases, and
/// XQA hits both of them separately, so it is worth being explicit about
/// which field each belongs in:
///
/// * **A toolchain floor** is [`crate::unit::Demands::floor`], not an option.
///   The lattice has no floor to state at all: the five non-Hopper members
///   compile under NVRTC 13.0, measured. The Hopper member is not blocked on
///   a NVRTC version either — see [`XQA_LATTICE`]'s last entry.
/// * **A header set** is [`crate::unit::Demands::headers`], not an option.
///   Every member needs [`crate::unit::Headers::LibraryAndVendor`], and that
///   is the one entry the currently-empty `DEMANDS` table would gain. Buying
///   the same thing with `-I` in `options` is precisely what `source.rs`'s
///   module header refuses: *"No include path on disk."*
///
/// What is left over after those two are moved out is a pure `-D` set, which
/// is what `options` was built to carry. XQA does not need the hook `unit.rs`
/// warns against.
pub struct XqaVariant {
    /// The `Unit::name` this member gets.
    pub unit: &'static str,
    /// The `-D` set, verbatim, as `Unit::options` would carry it.
    ///
    /// [`XQA_COMMON_OPTIONS`] is not repeated here; the full option array is
    /// the concatenation of that and this.
    pub options: &'static [&'static str],
    /// The `extern "C"` device entry point this member exports, after the
    /// rename in [`Self::options`].
    ///
    /// One name per member, and they must differ:
    /// [`crate::unit::unit_of`] resolves a symbol across the whole table, so
    /// six units exporting `kernel_mha` are six rows that cannot be told
    /// apart.
    pub entry: &'static str,
    /// The archive file this member's `#define` block came from, with the
    /// line the varying defines are on.
    ///
    /// **All six citations name DELETED files.** The six `attention_xqa*.cu`
    /// were retired when [`crate::x::xqa`] enrolled this lattice; the lines
    /// are correct against `0dc8e9e9b`, the last commit that contained them,
    /// and `kernels-cuda/csrc/CMakeLists.txt` carries the account. Nothing
    /// reads this field — it is a quotation from history kept so the `-D` set
    /// below can be checked against what it was transcribed from, and it is
    /// the reason the transcription is auditable at all now that the source
    /// is gone.
    pub from: &'static str,
    /// Why this member exists — the measurement its `.cu` carried.
    ///
    /// These are per-model justifications, not decoration. A lattice member
    /// with no reason to exist is a compile nobody asked for; a lattice
    /// member whose reason was dropped in the port is a regression that
    /// compiles.
    pub because: &'static str,
}

/// The twelve `-D`s every member of the lattice shares, plus the one that is
/// ours.
///
/// Cited to `attn/attention_xqa_gqa2.cu:17-30`, and byte-identical in the
/// other five (`attention_xqa.cu:58-71`,
/// `attention_xqa_gqa2_p16.cu:17-30`, `attention_xqa_gqa4.cu:17-30`,
/// `attention_xqa_gqa8.cu:17-30`, `attention_xqa_gqa8_sm90.cu:19-32`).
///
/// # `GENERATE_CUBIN` is the only one that is not the archive's
///
/// It is upstream's define, not ours — `xqa/mha.cu:2820` guards `launchMHA`
/// behind `#ifndef GENERATE_CUBIN` and `xqa/mha_stdheaders.cuh` swaps the
/// host standard library under the same name. The archive never set it
/// because the archive wanted the host launcher. A JIT does not, so this is
/// the one line the port adds rather than moves.
///
/// # `DTYPE` is spelled in the prelude's types, and that is not an accident
///
/// `device::bf16`, verbatim from `attention_xqa_gqa2.cu:22` — not
/// `__nv_bfloat16`. `csrc/shim/cuda_bf16.h:248` aliases `__nv_bfloat16` TO
/// `device::bf16`, so under the production header set the two spellings are
/// one type and the archive's is the one that does not depend on the shim
/// being reached first.
///
/// (The probes behind this section substituted `-DDTYPE=__nv_bfloat16`,
/// because they swapped the toolkit's dtype headers in to isolate the shim
/// gap enumerated below. That substitution is a property of the probe, not of
/// the unit.)
pub const XQA_COMMON_OPTIONS: &[&str] = &[
    "-DGENERATE_CUBIN=1",
    "-DNDEBUG=1",
    "-DBEAM_WIDTH=1",
    "-DUSE_INPUT_KV=0",
    "-DUSE_CUSTOM_BARRIER=1",
    "-DINPUT_FP16=0",
    "-DDTYPE=device::bf16",
    "-DCACHE_ELEM_ENUM=0",
    "-DHEAD_ELEMS=128",
    "-DSLIDING_WINDOW=0",
    "-DLOW_PREC_OUTPUT=0",
    "-DSPEC_DEC=0",
    "-DMLA_WRAPPER=0",
];

/// The root all six members compile, carried so a moved file is a compile
/// error here rather than a missing include at run time.
pub const XQA_ROOT: &str = include_str!("../../csrc/src/attn/attention_xqa_mha.cuh");

/// `sizeof(SharedMem)` (`xqa/mha.cu:409`), measured out of the PTX.
///
/// NVRTC 13.0, `compute_89`, [`XQA_ROOT`] verbatim: **79,488 bytes, the same
/// for every member measured** — HEAD_GRP_SIZE 2, 4, 5 and 8 at
/// TOKENS_PER_PAGE 32, and HEAD_GRP_SIZE 2 at TOKENS_PER_PAGE 16 all emit
/// `.global .align 4 .u32 pie_xqa_smem_size = 79488;`. Neither the head group
/// nor the page size moves it.
///
/// It is over the 48 KiB `runtime::module::DEFAULT_DYNAMIC_SMEM` default, so
/// the opt-in path is mandatory, and under sm_89's 99 KiB per-block opt-in
/// maximum, so the opt-in succeeds. `driver-cuda/src/fire/xqa.rs` puts it in
/// `Launch::smem`, which is what makes `KernelModule::fire` raise the cap.
pub const XQA_SMEM_BYTES: u32 = 79_488;

/// The six units, as option sets.
///
/// # Why this is a table and not six `Unit`s
///
/// Not because the include does not resolve — it does. `csrc/vendor/xqa/`
/// holds the fifteen-file closure now, and `carried.rs` walks the directory,
/// so it is carried as `xqa/mha.cuh` and answers the one directive that names
/// it, in [`XQA_ROOT`].
///
/// Two things stop the enrolment, and both are about the ROW rather than the
/// source:
///
/// * **`tests/units.rs:436` fails a unit that declares no rows** — *"a unit
///   with no rows would compile to a cubin nothing can fire"*. So `rows:
///   &[]` is not a way to enrol these early.
/// * **A row needs a `KernelSig`, and `kernel_mha`'s cannot be written yet.**
///   It takes `KVCacheList<usePagedKVCache> const cacheList` **by value**
///   (`xqa/mha.cu:2757`), and with `ENABLE_4BIT_KV_CACHE` off
///   (`xqa/mhaUtils.cuh:242-253`) that aggregate is four pointers plus a
///   `uint32_t`: **40 bytes, 8-aligned**. `runtime::args::ArgValue` has
///   `Ptr/I32/U32/F32/Usize/I64/Bool/U8` and no byte-buffer variant, which is
///   the gap `new-horizon.md` §3.2 names — *"a borrowed byte buffer, so
///   by-value aggregates over 8 bytes … can cross the JIT path"*. A
///   `KernelSig` written today would be inventing a spelling for a parameter
///   the runtime cannot pass.
///
/// A third thing is mechanical but is a claim about the whole table:
/// `unit.rs:929` asserts every unit in [`UNITS`] has `Demands::DEFAULT` and
/// that `DEMANDS` is empty, *"the table and the units above are two
/// spellings of one fact"*. Six `Headers::LibraryAndVendor` rows edit that
/// test — which its own message invites (*"update it with the reason"*) —
/// and the reason is this section.
///
/// So the honest artifact is the option sets, which are measured and exact.
/// Enrolling them afterwards is a `UNITS_HEAVY` entry per member with
/// `root: XQA_ROOT`, `options:` the concatenation of
/// [`XQA_COMMON_OPTIONS`] and [`XqaVariant::options`], and one `DEMANDS` row
/// each stating [`crate::unit::Headers::LibraryAndVendor`].
///
/// # The gate that does NOT cover this
///
/// `tests/layers.rs::every_include_reachable_from_a_unit_resolves` walks
/// `source::quoted_includes`, which reads `#include "..."` and nothing else.
/// [`XQA_ROOT`]'s includes are all ANGLE-bracketed — `<cuda_bf16.h>`,
/// `<xqa/mha.cu>` — so they pass that test whether or not the set carries
/// them, and would fail at first fire on a GPU box with a diagnostic naming
/// the include rather than the omission. `carried.rs`'s header calls this out
/// as the failure mode the generated set exists to prevent. **The directory
/// is the guarantee; the test is not.**
///
/// # Vendored, not CPM — and the reasoning, because it is easy to get wrong
///
/// `xqa/` **is vendored, at `csrc/vendor/xqa/`.** The CPM checkout cannot be
/// published to the JIT, for reasons that are all one reason:
///
/// * `runtime::nvrtc::options` (`nvrtc.rs:861`) passes
///   `--gpu-architecture=sm_XY -std=c++17 --fmad=false --prec-div=true
///   --prec-sqrt=true` and the unit's own `Unit::options`. **There is no
///   `-I` on that list and there is no mechanism for one.** Includes resolve
///   against `includeNames[]`, an in-memory set carried in the binary. There
///   is nothing for `${flashinfer_SOURCE_DIR}` to be spelled into.
/// * `${flashinfer_SOURCE_DIR}` is a **C++ compiler** include path.
///   `kernels-cuda/csrc/CMakeLists.txt` names it and never names
///   `csrc/vendor`; the two sets are populated by different mechanisms and
///   the archive's is not one a Rust binary can read at run time.
/// * The same header set is the **cache key**. `source.rs`'s
///   `the_digest_moves_when_any_header_does` exists because a compile keyed on
///   text that the key cannot see serves a stale cubin. A CPM checkout is a
///   host fact — a path, a git tag, a `_deps` directory that may or may not
///   have been populated — and none of it is in the digest.
///
/// (A probe rooted at `-I csrc/{src,shim,vendor}` resolves the same literal
/// names the carried set resolves, which is why probe results transfer. It is
/// a faithful simulation of the mechanism and not the mechanism, and `-I`
/// must never appear in a `Unit::options`.)
///
/// **The `<pie/…>` spelling split did not bite XQA**, which is the part worth
/// stating because it is the part that would have cost a rewrite. Every one
/// of the closure's internal includes is sibling-relative and quoted:
/// `"barriers.cuh"`, `"mha.h"`, `"utils.cuh"`, `"mhaUtils.cuh"`,
/// `"defines.h"` and ten more, all resolving within one directory —
/// `carried.rs`'s `collect` even builds the beside-the-includer aliases for
/// them. **Not one upstream byte was respelled**, so `PIE_INCLUDES`
/// (`tests/vendor_manifest.rs:276`) stays empty, which is the state that
/// makes the first `<pie/…>` fail loudly instead of passing silently.
///
/// # One name did change, and it is ours
///
/// Upstream's `mha.cu` is carried as **`csrc/vendor/xqa/mha.cuh`**, because
/// `kernels-cuda-new` holds no translation units: 120 `.cuh` and no `.cu`,
/// which is the device/host line this crate exists to draw. A `.cu` is
/// something nvcc compiles ahead of time; a `.cuh` is device text carried
/// into NVRTC.
///
/// The rename is free precisely because it is not an impersonation.
/// `carried.rs`'s *"the name is the path, because the name is what
/// resolves"* exists so a header we do not own can be answered under the
/// spelling ITS includer writes (§13.4) — and the only thing that ever wrote
/// `<xqa/mha.cu>` is `attn/attention_xqa_gqa2.cu:35` and its five siblings,
/// which this port replaces. Checked before renaming: **no file under
/// `csrc/vendor/xqa/` includes a `.cu` by name**, across all fifteen. So one
/// directive moved, in [`XQA_ROOT`], which is ours. Re-measured after it:
/// rc = 0, same entries, same 79,488.
///
/// If `mha_sm90` is vendored later the same applies — and the same check
/// applies first. `attn/attention_xqa_gqa8_sm90.cu:37` names
/// `<xqa/tensorMap.cpp>`, so if an upstream header there reaches a `.cu` or
/// `.cpp` by name, the spelling is upstream's and the decision is a
/// different one.
///
/// **Citation convention, because the rename makes it matter.** Every
/// `xqa/mha.cu:N` in this crate and in `driver-cuda/src/fire/xqa.rs` is a line
/// in UPSTREAM's file, which is the anchor that does not move. The carried
/// copy is offset by two insertions and its own `// PIE:` header states both
/// offsets and converts the three citations that are load-bearing.
///
/// # What was vendored, and what was left behind
///
/// Fifteen files, 272 KB: the transitive closure of upstream `xqa/mha.cu`'s
/// quoted includes and nothing else. `NVRTC` copies every byte of `VENDOR`
/// into every unit that asks for it, so the closure was computed rather than
/// the directory copied — upstream's `xqa/` is 25 files and 800 KB, and
/// `gmma_impl.cuh` alone is 323 KB of Hopper GMMA that `mha.cu` never
/// reaches.
///
/// Deliberately NOT vendored: `mha_sm90.cu`, `tensorMap.{h,cpp}` and their
/// closure. The Hopper member does not compile (see its entry below), so
/// carrying it would be 400 KB of text every vendored unit pays for and
/// nothing can use.
///
/// `csrc/vendor/xqa/` is outside `tests/vendor_manifest.rs`'s reach: that
/// test walks `csrc/vendor/flashinfer` specifically, so `MODIFICATIONS`
/// continues to describe the tree it describes. The one patch below is
/// therefore recorded HERE and in the file, and is not in that table.
///
/// One vendor patch was needed, and it is the difference between 2 errors and
/// 0: `xqa/mha.cu:2955-3050` — `configureKernel`, `hostSmemSize` and
/// `launchMHAFlashInfer` — is wrapped in `#ifndef GENERATE_CUBIN` under a
/// `// PIE:` marker. Upstream already guards `launchMHA` that way at
/// `:2820`; the tail is FlashInfer's own downstream addition and they never
/// guarded it. A preprocessor-depth walk confirms the trailing `#endif` at
/// EOF closes `#if !(IS_MLA)` from `:20`, so the tail really was unguarded
/// rather than merely appearing so.
///
/// # The shim gap, enumerated
///
/// With `csrc/shim`'s dtype headers rather than the toolkit's, and everything
/// else identical, the compile is **7 errors and not 0**. None of them is an
/// NVRTC limitation, which is the whole reason to write them down: they read
/// like one.
///
/// ```text
/// xqa/mhaUtils.cuh:368   no conversion float2 -> __nv_fp8x2_storage_t
/// xqa/utils.cuh:209      no __half2 constructor matches
/// xqa/mha.cu:1211        no operator* for the operands
/// xqa/mha.cu:1429  (x2)  no constructor converts int -> device::bf16
/// xqa/mha.cu:1442        no instance of __hadd2_rn matches
/// ```
///
/// What is actually missing, as opposed to what the error says:
///
/// 1. `__nv_fp8x2_e4m3(float2)` — a **constructor**. The error names
///    `__nv_fp8x2_storage_t` only because brace-init fell through to the
///    aggregate member. `csrc/shim/cuda_fp8.h:460-473` refuses this one in
///    writing: *"No constructor is spelled anywhere, so none is written here
///    — a `float2` constructor would be a second, untested spelling of
///    `__nv_cvt_float2_to_fp8x2`."*
/// 2. `__half2::__half2(__nv_fp8x2_e4m3)` — a constructor **from fp8**, not
///    (as it is easy to read from the line number alone) from two `__half`.
///    `utils.cuh:209` is
///    `half2(reinterpret_cast<__nv_fp8x2_e4m3 const&>(src[i]))`.
/// 3. `operator*(__nv_bfloat162, __nv_bfloat162)`.
///    `csrc/shim/cuda_bf16.h` defines **one** arithmetic intrinsic in total —
///    `__hmul2`, at `:460` — and no operators at all.
/// 4-5. `__nv_bfloat162{0, 0}` at `mha.cu:1429`, which needs `device::bf16`
///    constructible from an integer literal.
/// 6. `__hadd2_rn` for `__nv_bfloat162`. The shim has no `__hadd2` and no
///    `_rn` form of anything; `cuda_fp16.h` has `__hadd2` but no `_rn`.
///
/// And a second layer underneath, found by probing with those six patched
/// over: `float2(__nv_fp8x2_e4m3)` (`utils.cuh:186`),
/// `__nv_fp8x2_e4m3{__half2}` (`:217`), `__nv_fp8x2_e4m3{__nv_bfloat162}`
/// (`:229`), and the scalar `Dst{src}` forms beside them.
///
/// ## Why this was NOT fixed here, which is the actionable half
///
/// Because it is not a list of omissions. It is three design decisions
/// meeting a consumer they were not measured against:
///
/// * **The bf16 arithmetic set is a written refusal.**
///   `csrc/shim/cuda_bf16.h:489-496` has a section titled *"what is
///   deliberately not here"* naming `__hmul`, `__hadd`, `__hsub`, `__hfma`,
///   `__hmax`, `__hmin`, `__habs` *"and their packed forms"*, the comparison
///   set, and the operator set — each justified by a use count of zero across
///   two trees. **XQA is a third tree, and it was not in the count.** The
///   absence is still a measurement; the measurement's denominator changed.
///   Adding to it is right, and it is a deliberate revision of a recorded
///   decision, not a fill-in.
/// * **Half the gap is not in `csrc/shim` at all.**
///   `csrc/shim/cuda_bf16.h:248,262` alias `__nv_bfloat16` and
///   `__nv_bfloat162` to `device::bf16` and `device::bf16x2`, so items 3-6
///   land on `csrc/src/pie_device.cuh` — where `bf16`'s constructors are
///   `explicit` **on purpose** (`:68-108`: *"an implicit constructor from
///   `float` would make `bf16 x = 1.0f;` compile — silently narrowing"*) and
///   `bf16x2` (`:371`) is a plain aggregate. Item 4-5's `{0, 0}` is asking
///   for the exact implicit conversion that comment refuses.
/// * **The fp8 constructors cannot be checked here.** Bit-exactness against
///   `__nv_cvt_float2_to_fp8x2` is the property that matters and it needs a
///   device to compare on.
///
/// So the enumeration is the deliverable and the patch is not. The one item
/// that WAS obviously right has been added: `csrc/shim/cassert`, which
/// declares nothing — `xqa/barriers.cuh:19` and `xqa/utils.h:5` include
/// `<cassert>` unguarded, NVRTC's preamble already supplies `assert`, and the
/// missing FILE was a *catastrophic* error that ended the compile and hid
/// every other question behind it.
///
/// `.wiki/driver/new-horizon.md` §13.6 priced FA2's move as *"a FlashInfer
/// patch set plus ~39 bit-exact device intrinsics"*, and §62 records the
/// patch set as paid. This gap is very likely what the second half of that
/// quote was pointing at, and it is shared: anything reaching bf16 packed
/// arithmetic or fp8x2 conversion hits it, not just XQA.
/// **`const` and not `static`, so [`crate::x::xqa`] can read it in a `const`
/// initialiser.** The five enrolled `Unit`s take their `name` and their
/// per-member `-D` set straight out of this table rather than restating
/// either, which is the only way two spellings of one option set cannot
/// drift; a `static` may not be read by a `const`, and that restriction is
/// the whole of the change. Nothing here is mutable, nothing takes its
/// address, and there is no code consumer outside this file and that module.
pub const XQA_LATTICE: [XqaVariant; 6] = [
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa2_p32",
        options: &[
            "-DHEAD_GRP_SIZE=2",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa2_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa2_bf16_p32_h128",
        from: "attn/attention_xqa_gqa2.cu:24-33",
        because: "head_group_size=2, used by small Qwen GQA models such as \
                  Qwen3-0.6B and Qwen3-1.7B",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa2_p16",
        options: &[
            "-DHEAD_GRP_SIZE=2",
            "-DTOKENS_PER_PAGE=16",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa2_bf16_p16_h128",
        ],
        entry: "kernel_mha_xqa_gqa2_bf16_p16_h128",
        from: "attn/attention_xqa_gqa2_p16.cu:24-33",
        because: "the same head group at a 16-token page. Dead code TODAY and \
                  kept anyway: `xqa_decode_page_bucket` never returns 16 \
                  because `xqa_gqa2_page16_enabled()` returns false, so the \
                  only way to reach this member is to flip that — which is \
                  what it is for. Deleting it would make flipping the flag a \
                  port rather than a flag.",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa4_p32",
        options: &[
            "-DHEAD_GRP_SIZE=4",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa4_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa4_bf16_p32_h128",
        from: "attn/attention_xqa_gqa4.cu:24-33",
        because: "head_group_size=4, used by medium Qwen GQA models such as \
                  Qwen3-4B and Qwen3-8B",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa5_p32",
        options: &[
            "-DHEAD_GRP_SIZE=5",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa5_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa5_bf16_p32_h128",
        from: "attn/attention_xqa.cu:65-74",
        because: "head_group_size=5, the ratio Llama-3.1-8B-shaped models use \
                  (32 query heads over 8 KV heads is 4; 40 over 8 is 5). It \
                  lives in the family's dispatch head rather than a sibling \
                  file because that file is also where the host program was.",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa8_p32",
        options: &[
            "-DHEAD_GRP_SIZE=8",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=0",
            "-Dkernel_mha=kernel_mha_xqa_gqa8_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa8_bf16_p32_h128",
        from: "attn/attention_xqa_gqa8.cu:24-33",
        because: "head_group_size=8, used by common large GQA models such as \
                  Qwen3-32B and Llama-70B-style shapes. Its launcher \
                  FORWARDS to the sm90 member when `current_device_major() \
                  >= 9`, which is why the two exist as a pair rather than as \
                  alternatives.",
    },
    XqaVariant {
        unit: "attn/attention_xqa_mha_gqa8_p32_sm90",
        options: &[
            "-DHEAD_GRP_SIZE=8",
            "-DTOKENS_PER_PAGE=32",
            "-DUSE_SM90_MHA=1",
            "-Dkernel_mha=kernel_mha_xqa_gqa8_sm90_bf16_p32_h128",
        ],
        entry: "kernel_mha_xqa_gqa8_sm90_bf16_p32_h128",
        from: "attn/attention_xqa_gqa8_sm90.cu:26-35",
        because: "Hopper GMMA/TMA, kept in a separate translation unit \
                  because FlashInfer's `xqa/mha.cu` and `xqa/mha_sm90.cu` \
                  intentionally define the same static kernel symbols. It \
                  also passes `enable_pdl = true` unconditionally where the \
                  other five pass `current_device_major() >= 9` — which is \
                  the same predicate, already known true here. NOT READY: \
                  measured at compute_90a it stops on `std::pair` in DEVICE \
                  code (`xqa/mha_sm90.cu:1980`, 12 diagnostics cascading \
                  from that one line), the header set has no `<utility>`, \
                  `csrc/shim/cuda.h` has no `CUtensorMap` or \
                  `CUtensorMapDataType_enum` for `xqa/tensorMap.h` to \
                  declare against, and the archive unit compiles \
                  `<xqa/tensorMap.cpp>` first — HOST code building tensor \
                  maps through `cuTensorMapEncodeTiled`, which is a second \
                  and larger host-to-Rust port than `launchMHA` was.",
    },
];

// ===========================================================================
// `attn/attention_mla_naive` CROSSED INTO FN-WORLD, unit-only, as
// [`crate::x::attn::mla_naive::MLA_NAIVE`].
//
// **This was the root that stood between the tree and nvcc-zero, and the tree
// is there.** §66: the last two `<<<>>>` in the workspace were
// `mla_naive_paged_kernel` and `mla_mma_paged_kernel`; this root's TABLE row,
// `attn::dispatch_attention_mla_bf16`, has crossed as
// [`crate::x::attn::ATTENTION_MLA`], and `attention_mla.cu` is DELETED with
// `attention_mla.hpp`. They were never two tasks.
//
// The row's obstacle was that it has TWO arms — this pair, and
// `flashinfer::mla::BatchMLAPagedAttention<MASK, 512, 64>` for everything
// below sm_100 — and a row loses its shim entry whole or not at all. Both are
// Rust now: THIS arm in `driver-cuda/src/fire/mla_naive.rs`, firing both
// symbols by name through `hand::fire`, and the FA2 arm — the one that passes
// `MLAParams` by value, whose measured mirror is `crate::x::attn::mla_params`
// — as [`crate::x::attn::mla_fa2`], with its own root, unit and cooperative
// launch. The contract is UNBOUND: choosing between the arms needs a compute
// capability, and four of the statement's operands are `Cx` queries that do
// not exist. That is a seam, and it does not hold a translation unit.
//
// Every measurement travelled to `crate::x::attn::mla_naive`'s module doc and
// its two `fn` docs. Named here so a reader of THIS file knows what to expect
// to find there:
//
//  * `attention_mla.cu:150-157`, the only argument for why the pair exists —
//    FA2's cooperative `BatchMLAPagedAttention` produces ZERO OUTPUT on
//    sm_100, and the ecosystem routes Blackwell MLA elsewhere.
//  * THE TRANSPOSED GRIDS. `:265` is `grid(total_tokens, num_heads / G)` and
//    `:725` is `grid(num_heads / kBM, total_tokens)` — same block, axes
//    swapped. `grid.y` is capped at 65 535 and `grid.x` is not, so the
//    transpose decides which of tokens and head blocks may exceed 65 535. A
//    rule stating one would be actively wrong for the other while looking
//    right.
//  * `G` is `Control::Supplies` and UNSTATEABLE by a formula: `:241-249`
//    SEARCHES for it, halving from 8 until the grid reaches
//    `kMlaWaveTarget = 296` blocks. A rule computes; this looks.
//  * THE SHARED-MEMORY OPT-IN, and both halves of the correction: the opt-in
//    is `runtime::module::raise_dynamic_smem_cap`'s, driven by `Launch::smem`
//    alone; and the 200 KiB was never needed — `:251`'s allocation is
//    `(8 * CKV + 16) * 4` with `:228` refusing `CKV > 512`, so the scalar
//    kernel's largest request is 16 448 bytes, a third of the 48 KiB default.
//    The tensor-core kernel's 100 032 IS above it and IS raised.
//  * THE PROBE. NVRTC 13.0, sm_89, `--fmad=false --prec-div=true
//    --prec-sqrt=true`, carried headers only: rc = 0, 117 621 bytes of PTX,
//    2 `.entry`. Three new shim headers, MEASURED not assumed — the same text
//    with `/usr/local/cuda/include` answering `cuda_pipeline.h`,
//    `math_constants.h` and `cstring` produced BYTE-IDENTICAL PTX, register
//    allocation included.
//  * THE FOURTH FINDING, which is this sweep's recurring shape: the file
//    called `std::memcpy` and never included `<cstring>`. nvcc supplied it
//    transitively through `<cuda_runtime.h>`; NVRTC cannot, because the
//    include was never written. **The set nvcc accepted was not the set the
//    file declared.**
//  * `index_mask`'s precondition, which no type states and no refusal can
//    check — *"only valid for single-request pure prefill (key j == batch
//    token j)"*.
//  * THE THREE MISSING PARAMETERS on the tensor-core kernel — no `ckv`, no
//    `kpe`, no `G` — because it is compiled against `kCkv = 512`,
//    `kKpe = 64`, `kBM = 16` and `mla_mma_supported` (`:698`) COMPARES them
//    rather than forwarding them.
//  * `DeviceKernel::PLAIN` and no `_bf16` suffix on either, for the same
//    reason: no template parameter list, so nothing for `elem` to pick and
//    nothing a suffix could claim a choice about.
//  * NO `Unit::options`: `PIE_MLA_MMA_BK`/`_WARPS`/`_STAGES`/`_MINBLK` are
//    `#ifndef`-guarded with their defaults at `:302-322`, and a cubin cache
//    key over a number nobody varies is the hook `unit.rs` warns against.
//
// `MLA_NAIVE_ROWS` and `MLA_NAIVE_SIGS` were both PRIVATE (`static`, no
// `pub`), so their consumer set was this file. `MLA_NAIVE` was `pub`; swept
// both consumer sets across `crates/`, and the only external mention is a
// PROSE citation in `crate::x::xqa`'s doc (`x/xqa.rs:537`, backticked text
// arguing the same `DeviceKernel::PLAIN` point). It is `xqa-finish`'s file and
// is left alone.
