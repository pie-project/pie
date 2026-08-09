//! `attn`'s JIT units — the small half of the family.
//!
//! # What this module holds
//!
//! One [`Unit`] per migrated `.cuh`, the [`DeviceKernel`] rows those units
//! instantiate, and the [`KernelSig`]s behind them. Each sig is its
//! ahead-of-time twin minus the stream — a stream is `cuLaunchKernel`'s sixth
//! PARAMETER, outside the `void**`, so it was never an operand — and minus
//! whatever extent the launch rule recovers.
//!
//! # Twenty-one kernels, thirteen rows, and why the gap is the point
//!
//! Nine `.cu` files in the small half held twenty-one `__global__`s. All nine
//! are split now into a `.cuh` of templates plus a `.cu` of launchers, so the
//! tree has exactly ONE definition of each — the property that matters most,
//! because two copies that agree today drift tomorrow and each stays right for
//! whichever half its tests exercise.
//!
//! Six rows came out of that over five units, and five more landed the day
//! [`crate::runtime::launch`] grew a head axis and a KV-sized shared
//! allocation: `attn_sink_rescale` states `PerHeadElementwise`, `pad_head_dim`
//! and `strip_head_dim` state `PerHead`, `split_qkv` states `SplitPacked`, and
//! `attn_naive` states `SdpaVector`. Two of those units — `head_dim_pad`
//! and `split_packed` — did not exist before, because a unit with no rows is
//! refused rather than compiled: a cubin nothing can fire is cached under an
//! architecture and satisfies nobody. [`PAGE_COMPACT`] is the third, and its
//! two rows are the newest: see that unit's doc for the statement that
//! distinguishes its `<<<num_requests, 256>>>` from the one `per_row` refuses.
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
//!   `ATTENTION_NAIVE_SIGS[2]` is the row.
//! * **`attn/attention_flashinfer`'s `attn_score_fold_heads` is the newest
//!   member, and the last kernel in the tree whose blocker was that its text
//!   had not moved.** The text has moved — `attn/attention_flashinfer.cuh`,
//!   a PARTIAL split of a file that keeps its FlashInfer dispatch; its three
//!   private score-normalisation kernels went too, to
//!   `attn/attention_score_post.cuh`, and are [`ATTN_SCORE_POST`]'s rows as
//!   of §53.8 — so the row is now refused
//!   for GEOMETRY like the rest of this list. The launcher is
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
//!   so the refusal's citation cannot rot.
//! * `attn/split_packed`'s `split_qkv_devwin` shares its sibling's grid
//!   arithmetic and not its INPUTS — see
//!   [`crate::x::attn::split_packed`], which is the
//!   one place in this file where a ported rule computes the right shape from
//!   the wrong numbers.
//! * `attn/dsv4_compress`'s `combine_attn_outputs` had TWO blockers and has
//!   one left. The `__global__` was not a template — concrete `device::bf16`
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
//!   is [`PAGE_COMPACT`] now, with a row for each of its two kernels.
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
//!   See [`MLA_PAGED`].
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
//! now** — see [`SPECIALISATIONS`] and the block at the end of this file.
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
//!     instantiation set — each member is one row, and [`SPECIALISATIONS`]
//!     above is the mechanism that picks among them, the same one already
//!     choosing between five `kv_paged` appenders on a flag.
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
use crate::device::{Arm, Specialisation, Take, Term};
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

/// The reference attention kernels and MTP's hidden-state plumbing. Two rows
/// of five; the `.cuh` names the obstacle for each of the other three.
pub const ATTENTION_NAIVE: Unit = Unit {
    name: "attn/attention_naive",
    root: include_str!("../../csrc/src/attn/attention_naive.cuh"),
    rows: ATTENTION_NAIVE_ROWS,
    options: &[],
};

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

/// The paged-KV CSR compactor: quest's page-eviction gather.
///
/// Two rows over the header's two `__global__`s, and the unit exists because
/// both of the things that kept it out are gone. `<cub/cub.cuh>` was the first
/// — CCCL is 13.7 MB in 1,691 files and NVRTC answers no external include, so
/// the two collectives this file used are written out against `__shfl_down_sync`
/// / `__shfl_up_sync`, exactly, in `u32` under `+`, which is associative modulo
/// 2^32 and therefore the same integer rather than a close one.
///
/// The second was the grid, and the header still records the refusal:
/// *"one block per REQUEST, not per row of anything ... No ported rule opens a
/// grid over requests"*. That reading was one statement short.
/// [`LaunchRule::PerRow`]'s grid is `Dims::rows`, which `driver-cuda`'s
/// `jit_dims` fills from `BoundLaunch::rows` — documented at
/// `driver-cuda/src/bind/mod.rs:93` as *"the rectangle, in the op's own row
/// space"* — and `model-compiler`'s `dsl::cuda::compact_page_csr` records this
/// op's result as `Shape(vec![Dim::Requests])`: rank one, `Dim::Requests`,
/// which `lower.rs:716` resolves to `n_requests`. The statement is `whole`, so
/// the rectangle is all of it. For this op and no other reading, `Dims::rows`
/// IS the request count.
///
/// That is the whole of the distinction from the launcher `per_row`'s own doc
/// refuses by name. `attn/attention_naive`'s `mtp_update_pending_hidden` also
/// opens `<<<num_requests, 256>>>`, and `dsl::cuda::mtp_update_pending_hidden`
/// records NO result at all — its rectangle is its input's, `[Tokens, hidden]`
/// — so its fire's `rows` is the token count and `PerRow` would run one block
/// per token against a buffer with one slot per request. Same launcher shape,
/// opposite verdict, and the statement is what tells them apart.
pub const PAGE_COMPACT: Unit = Unit {
    name: "attn/page_compact",
    root: include_str!("../../csrc/src/attn/page_compact.cuh"),
    rows: PAGE_COMPACT_ROWS,
    options: &[],
};

/// `attn/page_compact.cuh`'s two instantiations.
///
/// `elem` is `attn::device::kBlock` and not `256`: `instantiation()` prefixes
/// slot 1 with `::pie_cuda_driver::kernels::`, so a bare literal comes back
/// `expected an identifier` — and the constant it names is the SAME one
/// `page_compact.cu:45` and `:48` spell in both `<<<>>>`, so the row cannot
/// drift from the launcher by construction. `device::i32(256)` would also
/// resolve (`quant`, `layout` and `rope` all record the measurement) and would
/// be a second copy of a number this header already owns.
///
/// `BLOCK` is not a decoration. It sizes `__shared__ u32 tmp[BLOCK / 32]` and
/// fixes how many warp partials the two collectives fold, so a row that named
/// 128 would fold four partials that were never written — a plausible page
/// list, not a fault.
static PAGE_COMPACT_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &PAGE_COMPACT_SIGS[0],
        template_path: "attn::device::count_kept",
        elem: "attn::device::kBlock",
    },
    DeviceKernel {
        sig: &PAGE_COMPACT_SIGS[1],
        template_path: "attn::device::scan_and_scatter",
        elem: "attn::device::kBlock",
    },
];

/// Two kernels, two rows, and the ORDER between them that no row states.
///
/// `scan_and_scatter` reads the `counts` buffer `count_kept` fills, on the same
/// stream. Two rows state two geometries and no dependency, so a caller firing
/// these must fire them in this order on one stream — which is what
/// `page_compact.cu:45`/`:48` do and what the ahead-of-time entry point
/// `attn::compact_page_csr` wraps.
///
/// **Neither row claims `attn::compact_page_csr`.** That symbol is ONE launcher
/// over TWO kernels, and a row that took its name would be claiming half a
/// launcher — the same honesty `KV_PAGED_ROWS` keeps for
/// `attn::dequant_kv_cache_layer_to_bf16_active`. The consequence is visible
/// and intended: these two do not move `examples/migration_status`, they appear
/// in its "hosted but not stated" list.
///
/// Every operand is the launcher's, unsourced for the same reason
/// [`KV_PAGED_ROWS`]'s are: `scratch_counts` is a driver-owned scratch buffer
/// and `keep_stride` comes off a host CSR, and no `Source` spells either.
#[rustfmt::skip]
static PAGE_COMPACT_SIGS: [KernelSig; 2] = [
    // `page_compact.cu:45` -- `device::count_kept<device::kBlock>
    // <<<num_requests, device::kBlock, 0, stream>>>`.
    kernel!(count_kept "attn::count_kept",
        file = Some("attn/page_compact.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            page_indptr_in: U32s, keep: U8s, keep_stride: U32,
            num_requests: I32, counts: U32sMut,
        ]),
    // `page_compact.cu:48` -- `device::scan_and_scatter<device::kBlock>
    // <<<num_requests, device::kBlock, 0, stream>>>`.
    kernel!(scan_and_scatter "attn::scan_and_scatter",
        file = Some("attn/page_compact.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            page_indices_in: U32s, page_indptr_in: U32s, last_page_lens_in: U32s,
            keep: U8s, counts: U32s, keep_stride: U32, num_requests: I32,
            page_indptr_out: U32sMut, last_page_lens_out: U32sMut,
            page_indices_out: U32sMut,
        ]),
];

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
// the device text is `crate::x::attn::dsa_indexer`'s and the three host
// programs never left `driver-cuda/src/fire/dsa_indexer.rs`.
//
// **No row crossed and none could.** A `table::` row is a claim about what a
// TRACE may say; these three kernels are fired by Rust that already exists
// and is already correct. Two of the three rows were unsourced — `n_heads`,
// `head_dim`, `rope_dim` and `topk` arrive on `Source::Param`, which is the
// statement's parameter channel, and *"a JIT row that guessed which statement
// parameter carried which would bind three integers in an order nothing
// reports"*. The third, `dsa_index_topk_mask`, IS fully sourced in
// `table::attn` and its row stays there; only the device text moved.
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
//   `attn::dsa_index_{knorm_rope,q_rope,topk_mask}_dev`; `table::attn` still
//   carries `attn::dsa_index_knorm_rope_bf16`, `attn::dsa_index_q_rope_bf16`
//   and `attn::dsa_index_topk_mask`. `_bf16` is DROPPED as well as `_dev`
//   added, because these are `template <class T>` and the ROW picks `T`.
//   `fire/dsa_indexer.rs:45-61` holds both halves of each pair as constants
//   side by side and is the only thing that bridges them.
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

/// MTP's input shift, the reference attention, and MTP's pending-state stash.
pub static ATTENTION_NAIVE_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &ATTENTION_NAIVE_SIGS[0],
        template_path: "attn::device::mtp_shift_hidden",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &ATTENTION_NAIVE_SIGS[1],
        template_path: "attn::device::attn_naive",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &ATTENTION_NAIVE_SIGS[2],
        template_path: "attn::device::mtp_update_pending_hidden",
        elem: "device::bf16",
    },
];

#[rustfmt::skip]
static ATTENTION_NAIVE_SIGS: [KernelSig; 3] = [
    // `whole`, as the twin is: the kernel reads `qo_indptr` to find which
    // request a token belongs to, and a row window starting anywhere but zero
    // would index that table with the wrong token number.
    //
    // `total_tokens` is gone — `Rms` opens one block per row. `num_requests`
    // is NOT: it bounds `find_request_u32`'s scan, and a request count is not
    // a row count.
    // §60.6's SYMBOL SPLIT. This row was `attn::mtp_shift_hidden_bf16`, the
    // same string the `table::attn` row carries, and that made the table
    // symbol unit-hosted -- which §52.11 forbids for a `Walk` (*a walk may
    // drive a JIT'd kernel; it may not be one*), enforced by
    // `execution::tests::a_walk_is_only_a_walk` through `unit_of`. The
    // launcher in `attention_naive.cu` could not be taken over while the two
    // names were one. The TABLE symbol does not move: it is what a trace
    // records. The `_bf16` suffix is dropped here as well as `_dev` added,
    // for `MLA_PAGED_SIGS`' reason about `attn::write_mla` -- this is
    // `template <typename T>` and the ROW picks `T`, so a format suffix on
    // the row's own name advertises a choice at a level that does not make
    // it. The launcher is `driver-cuda/src/fire/attention_naive.rs`.
    kernel!(mtp_shift_hidden "attn::mtp_shift_hidden_dev",
        file = Some("attn/attention_naive.cuh"),
        launch = LaunchRule::PerRow,
        // `PerRow`, not `Rms`. The launcher is `<<<total_tokens, BLOCK=256, 0>>>` in
        // `attn/attention_naive.cu:154`, and `Rms` requests thirty-two bytes of dynamic
        // shared memory that no launcher here passes and no kernel here
        // reads -- `block_sum`'s warp buffer, which this shape has no
        // reduction to need. Harmless in effect and wrong as a contract:
        // a rule is meant to REPRODUCE its launcher, and one that asks
        // for memory the launcher did not is a rule nobody can check
        // against the `<<<>>>` it came from.
        whole = true,
        operands = operands![
            target_hidden: Buf, pending_hidden: Buf, qo_indptr: U32s,
            slot_ids: I32s, out: BufMut, num_requests: I32, hidden_size: I32,
        ]),
    // The reference attention, and the row the `.cuh` said would arrive as a
    // diff of one line when a rule did. Its two obstacles were the same
    // rule's: a head count on `grid.x`, and a dynamic shared allocation sized
    // on a KV extent. `SdpaVector` is BOTH, and it is this launcher's
    // arithmetic rather than a shape that resembles it --
    // `dim3 grid(num_q_heads, num_tokens)`, `dim3 block(256)`,
    // `sizeof(float) * (num_tokens + BLOCK)` -- which `eval` returns as
    // `[q_heads, rows, 1]`, `[256, 1, 1]` and `(rows + 256) * 4`.
    //
    // The smem is the whole reason no other rule could stand in. `attn_naive`
    // lays `scores[num_tokens]` and `reduce_buf[BLOCK]` in one
    // `extern __shared__` block and takes the second as `smem + num_tokens`;
    // launched with less, the reduction scratch overlaps the scores it is
    // reducing, the softmax denominator is computed from bytes the same
    // kernel is overwriting, and the answer is finite. A rule that defaulted
    // `smem` to zero would do that on every fire.
    //
    // UNSOURCED, as its paged twin's ahead-of-time row is: this kernel exists
    // so a parity test has something to compare flashinfer against on a shape
    // flashinfer does not cover, and no statement lowers to it. `scale` is
    // the launcher's `1 / sqrtf(head_dim)`, which is a host computation and
    // not a `Source` -- inventing one so the row LOOKED bindable would put a
    // guess where an absence belongs.
    //
    // `num_tokens` stays an operand. The rule recovers the row count for the
    // GRID; the kernel reads it as its KV extent and as the bound on the
    // score loop, and those are the same number only because this is the
    // unpaged form.
    kernel!(attention_naive "attn::attention_naive_bf16",
        file = Some("attn/attention_naive.cuh"),
        launch = LaunchRule::SdpaVector,
        whole = true,
        operands = operands![
            q: Buf, k: Buf, v: Buf, o: BufMut,
            num_tokens: I32, num_q_heads: I32, num_kv_heads: I32,
            head_dim: I32, scale: F32,
        ]),
    // The kernel whose `.cuh` doc says *"NO ROW STATES THIS KERNEL: one block
    // per REQUEST"* and spells out what a row over rows would cost: *"a fire
    // of eight requests and ninety-three tokens would open ninety-three
    // blocks — eighty-five of them writing a slot that is not theirs."*
    // That doc is now half true and is corrected in place.
    //
    // `attn/attention_naive.cu:174`, and `BLOCK` is `device::BLOCK = 256` at
    // `attention_naive.cuh:91`:
    //
    // ```text
    // :174   device::mtp_update_pending_hidden<bf16><<<num_requests, BLOCK, 0, stream>>>(
    // :175       static_cast<const bf16*>(target_hidden),
    // :176       static_cast<bf16*>(pending_hidden),
    // :177       qo_indptr, slot_ids, num_requests, hidden_size);
    // ```
    //
    // `LaunchRule::PerRequest` is that grid: `[Dims::requests, 1, 1]` at 256,
    // no shared memory. **This is the row that made the rule**, and it is the
    // only row on it today (§10.5, stated rather than inferred).
    // `attn/page_compact.cu:45` and `:48` open the same
    // `<<<num_requests, kBlock>>>` and KEEP [`LaunchRule::PerRow`] — see
    // [`PAGE_COMPACT`] — which is the sharpest demonstration in this file
    // that a launcher's SHAPE and a fire's rectangle are two different
    // questions, and the reason this variant is an axis rather than a
    // one-kernel convenience.
    //
    // `num_requests` stays an operand where `mtp_shift_hidden`'s
    // `total_tokens` went: the rule recovers the request count for the GRID,
    // and the kernel reads the operand as the bound on `r >= num_requests`.
    // Dropping it would leave the guard reading a register nothing set.
    //
    // UNSOURCED, and `table/attn.rs:745` is too: `pending_hidden` is a
    // recurrent-state store the driver owns, `slot_ids` is the batch's slot
    // map, and `qo_indptr` is the fire's CSR. `dsl::cuda::mtp_update_pending_hidden`
    // records a `StateRef` and NO result, so the statement names no rectangle
    // of its own — which is the second half of why `PerRow` is wrong here and
    // right for `compact_page_csr`, whose result IS `Shape(vec![Dim::Requests])`.
    // §60.6's symbol split, for the twin's reason above.
    kernel!(mtp_update_pending_hidden "attn::mtp_update_pending_hidden_dev",
        file = Some("attn/attention_naive.cuh"),
        launch = LaunchRule::PerRequest,
        whole = true,
        operands = operands![
            target_hidden: Buf, pending_hidden: BufMut, qo_indptr: U32s,
            slot_ids: I32s, num_requests: I32, hidden_size: I32,
        ]),
];

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
// The two-symbol arrangement crossed with them: the device row is
// `attn::split_qkv_devwin` and the table row is
// `attn::split_qkv_bf16_devwin`, bridged by `SPLIT_DEVWIN_SYMBOL` and
// `SPLIT_DEVWIN_DEVICE` in `driver-cuda/src/fire/split_packed.rs`.

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
//   in that slot. `PAGE_COMPACT_SIGS` keeps `num_requests` for the same
//   reason, and that sentence is why this one is repeated there.
//
// * EVERY OPERAND UNSOURCED, for the reason `PAGE_COMPACT_SIGS`' are:
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
    ATTENTION_NAIVE,
    PAGE_COMPACT,
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
// `qkv_fused`, `mla_paged`, `kimi_mla`, `attention_mla_naive`,
// `attention_xqa*`, `attention_flashinfer*`.
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
// `qkv_fused` IS a unit now — see `QKV_FUSED`. `attention_naive_paged` is
// split — a `.cuh` of device text plus a `.cu` that keeps only its `<<<>>>`,
// probed against NVRTC 13.0 for `compute_89` and producing PTX — and is still
// not a unit, because a unit with no rows is refused and it has no row.
// `attention_mla_naive` is not split at all. `mla_paged` WAS in that list and
// is a unit now — see `MLA_PAGED`.
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
//    `MLA_PAGED_ROWS` and does NOT claim the ahead-of-time symbol, which
//    `MLA_PAGED_SIGS` argues.
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
//    `QKV_DECODE_WARP` are the two pairs.
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
//    split; the unit is `MLA_NAIVE` at the end of this file and its two rows
//    are `attn::mla_naive_paged` and `attn::mla_mma_paged`. Both halves of
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

/// kimi_k3's two latent-attention preparation kernels.
///
/// The cleanest split in the family: both launchers were already exactly a
/// ported rule, so the device half came out whole and cost nothing but the
/// move.
pub const KIMI_MLA: Unit = Unit {
    name: "attn/kimi_mla",
    root: include_str!("../../csrc/src/attn/kimi_mla.cuh"),
    rows: KIMI_MLA_ROWS,
    options: &[],
};

/// deepseek_v4's compressed-KV builders, gathers and stores.
///
/// Eight rows over eleven kernels. The three without rows are named in
/// `dsv4_compress.cuh` with the geometry each launches: two meaningful grid
/// axes, a dynamic shared-memory size computed from a head dim, and a
/// `cudaMallocAsync`'d parameter block the launcher fills on the host.
///
/// The two boundary-metadata rows are new, and they overturn a refusal that
/// read "the kernel is a plain `__global__` and
/// [`DeviceKernel::instantiation`] can only spell `path<...>`". That was a
/// report on the SPELLING, not on the kernel: a `template <class T =
/// device::i32>` on a `__global__` makes it nameable while leaving every
/// existing call site — which lives in `kernels-cuda`, a tree this migration
/// may not edit — compiling unchanged. Measured under nvcc 13 for `sm_89`:
/// `device::dsv4_boundary_meta_decode<<<blocks, threads, 0, stream>>>(...)`
/// with no `<>` still resolves against the defaulted template, and the
/// archive's `attn/dsv4_compress.cu.o` rebuilt clean. `combine_attn_outputs`
/// had to template its kernel AND edit the launcher; a default argument does
/// the same job across a file boundary.
pub const DSV4_COMPRESS: Unit = Unit {
    name: "attn/dsv4_compress",
    root: include_str!("../../csrc/src/attn/dsv4_compress.cuh"),
    rows: DSV4_COMPRESS_ROWS,
    options: &[],
};

/// The paged KV cache's dequantisers and its five specialised appenders.
///
/// Eighteen rows over fourteen kernels — more rows than kernels, because five
/// of them are `template <bool HND_LAYOUT>` and a specialised kernel costs
/// THREE rows: a contract and two variants. Eight kernels are covered.
///
/// The first three rows are the dequantisers, `<<<(n + 255) / 256, 256>>>` to
/// the digit. They compile, and as of `Args::bind`'s `I64` arm they also FIRE.
/// Each takes its element count as a `long long`, so the operand is `Ty::I64`,
/// and the binder refused that type by name for as long as these rows existed
/// — `ArgError::Unsupported`, at every fire. Restating it as `Usize` would
/// have bought a bindable row by describing a signed parameter as unsigned,
/// which is the kind of agreement that holds until a count is negative for a
/// reason nobody predicted. The row said what the kernel says and the gap was
/// the binder's; the binder closed it, for the batched SSM kernels' `long
/// long` slot stride, and these three came along with it.
///
/// The fifteen after them are `write_kv`, `write_kv_at_positions`,
/// `write_kv_explicit`, `write_kv_explicit_devwin` and `copy_kv_cells`, all
/// [`LaunchRule::PerRow`] and all chosen by [`SPECIALISATIONS`]. The six
/// kernels still without rows are named in `kv_paged.cuh` with the geometry
/// each launches; `write_kv_per_token_head` is the interesting one, and the
/// module header above says why an enum is not a flag.
pub const KV_PAGED: Unit = Unit {
    name: "attn/kv_paged",
    root: include_str!("../../csrc/src/attn/kv_paged.cuh"),
    rows: KV_PAGED_ROWS,
    options: &[],
};

/// The MLA cache's append and its preparation pass — both header kernels, as
/// two rows.
///
/// **The two were blocked on different things and both are unblocked.**
/// `write_mla` was blocked on the SPELLING — a plain `__global__` against an
/// `instantiation()` that could only emit `path<...>` — and that limit is
/// gone: [`DeviceKernel::PLAIN`] names it by its bare qualified path, which
/// NVRTC lowers and `cuModuleGetFunction` resolves, measured on this L40S in
/// `examples/argform_probe.rs`. No device text changed, no launcher changed,
/// and `mla_paged.cuh`'s single-includer constraint is untouched — a row does
/// not `#include` anything, so naming a plain kernel and lifting its linkage
/// are now two separate decisions.
///
/// **`mla_prepare<256>` was blocked on a GEOMETRY, and
/// [`LaunchRule::MlaPrepare`] is that geometry.** `mla_paged.cu:74` launches
/// `dim3 grid(total_tokens, 1 + q_blocks)` where
/// `q_blocks = ceil(heads / heads_per_block)` and `heads_per_block` is itself
/// computed on the host at `:64` from `half >= BS ? 1 : BS / half`, `half`
/// being `qk_rope_head_dim / 2`. Nothing in [`crate::runtime::launch`]
/// computed `1 + ceil(heads / f(rope, block))`; [`LaunchRule::Rope`] had the
/// closest `grid.y` — a head count over a head-group factor — and it neither
/// added the leading KV lane nor launched with `smem = 0`.
///
/// The leading `1` is not a head, which is why it could not be folded into
/// the head axis and why the rule adds it rather than rounding up: it is the
/// lane that owns the `kv_a` norm, the `k_pe` rotation and the paged write,
/// and every head lane is `blockIdx.y - 1`. A rule that dropped it would open
/// the right number of head blocks, shift every head down by one, drop the
/// last, and never write the cache — while `q_nope`/`q_pe` still filled, so
/// the fire would produce a plausible query against an unwritten page.
///
/// The rule reads [`crate::runtime::Dims::rotary_dims`] and not `head_dim`,
/// which the rule's own doc argues and
/// `tests/launch_rules.rs::transcribed` measures: an MLA head is
/// `kv_lora_rank + qk_rope_head_dim` = 576, giving `heads_per_block = 1`
/// where the launcher computes 8, and 129 lanes where the launcher opens 17.
pub const MLA_PAGED: Unit = Unit {
    name: "attn/mla_paged",
    root: include_str!("../../csrc/src/attn/mla_paged.cuh"),
    rows: MLA_PAGED_ROWS,
    options: &[],
};

/// The one `__global__` `attn/attention_flashinfer.cuh` holds.
///
/// The other three `__global__`s of `attention_flashinfer.cu` stay in the
/// `.cu`, and the header says why in one line: *"They move when something
/// asks for them."*
pub static ATTN_SCORE_FOLD_ROWS: &[DeviceKernel] = &[DeviceKernel {
    sig: &ATTN_SCORE_FOLD_SIGS[0],
    template_path: "attn::device::attn_score_fold_heads",
    // Not a template, and the header argues at length that it must not
    // become one: every buffer is `float` or page-table metadata, the block
    // width arrives as `blockDim.x` and the fanout as `gridDim.y`, so a
    // `template <int BLOCK>` would be a parameter the body never mentions
    // and an arm that cannot differ from its sibling.
    elem: DeviceKernel::PLAIN,
}];

#[rustfmt::skip]
static ATTN_SCORE_FOLD_SIGS: [KernelSig; 1] = [
    // `LaunchRule::Unstated`, and this is a refusal that was argued rather
    // than a gap nobody filled.
    //
    // The launcher is `attention_flashinfer.cu`:
    //
    //     const dim3 grid(static_cast<unsigned>(num_requests), 64u);
    //     device::attn_score_fold_heads<<<grid, 256, 0, stream>>>(
    //
    // `dim3(requests, 64)` at 256 threads, no shared memory. `64` is not in
    // `Dims`: not heads, not requests, not pages, not a head dimension. It
    // is an occupancy constant — a guess about one GPU, made once.
    //
    // The rule CANNOT be chosen by measuring bytes. The body strides
    // `i += blockDim.x * gridDim.y`, so every value of `gridDim.y` produces
    // the same floats; `LaunchRule::PerRequest` would pass any parity test
    // ever written for this kernel and be wrong by 64x in blocks alone.
    //
    // The tempting repair is a parameterised `PerRequestFanout(64)`, and the
    // measurement that kills it: there are exactly TWO literal grid axes in
    // all of `csrc/src`, both in this one file — `(num_requests, 64u)` and
    // `(cache.num_requests, 32u)`. DIFFERENT literals. There is no shared
    // rule waiting to be extracted, only two constants that share a file, and
    // a rule covering both would be vocabulary growth for a single literal —
    // which is what §10.5 exists to forbid.
    //
    // So the vocabulary declines to lie and the driver builds the `Launch` by
    // hand. `KernelModule::fire`'s own doc anticipates it — *"reaching here
    // with one means a caller built a `Launch` by hand"* — and
    // `runtime::fire::fire` still refuses `Unstated` through `launch::eval`,
    // so the hand-built path is the ONLY path and it is visible at the one
    // site that takes it. `driver-cuda`'s `fire/attn_score.rs` carries the 64
    // and the 256 as named constants with the `.cu` line cited beside them:
    // the number is a citation, not a derivation.
    kernel!(attn_score_fold_heads "attn::attn_score_fold_heads",
        file = Some("attn/attention_flashinfer.cuh"),
        launch = LaunchRule::Unstated,
        whole = true,
        operands = operands![
            scores: Buf,
            score_indptr: I32s,
            kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            page_size: I32,
            num_q_heads: I32,
            folded: BufMut,
        ]),
];

/// The per-head → per-request score fold, as a JIT unit.
///
/// This is the migration in one launcher: the `__global__` is ours, the
/// geometry is a literal, and nothing about either needs the archive. The
/// device text is `attention_flashinfer.cuh`; the launch is
/// `driver-cuda`'s `fire/attn_score.rs`. `model-compiler` cannot tell.
pub const ATTN_SCORE_FOLD: Unit = Unit {
    name: "attn/attention_flashinfer",
    root: include_str!("../../csrc/src/attn/attention_flashinfer.cuh"),
    rows: ATTN_SCORE_FOLD_ROWS,
    options: &[],
};

/// The one `__global__` `attn/attention_xqa.cuh` holds — and the LAST one the
/// `kernels-cuda` archive held.
///
/// Not a template, for the same reason [`ATTN_SCORE_FOLD_ROWS`] is not: every
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
    KV_PAGED,
    DSV4_COMPRESS,
    KIMI_MLA,
    MLA_PAGED,
    QKV_FUSED,
    ATTENTION_NAIVE_PAGED,
    ATTN_SCORE_FOLD,
    ATTN_SCORE_POST,
    ATTN_XQA,
    MLA_NAIVE,
];

/// The reference paged attention — `attention_naive_paged.cuh`'s two rows.
///
/// # What was blocking them, and what closed it
///
/// The SHAPE was stated a round ago: [`LaunchRule::PagedScores`] computes
/// `dim3(num_requests, total_tokens, num_q_heads)` with the dynamic
/// `(head_dim + 128) * sizeof(float)`, and [`LaunchRule::PagedScoresDecode`]
/// its decode twin. What blocked the ROWS was the OPERANDS: both kernels take
/// `device::KvScheme scheme` and `device::KvDType storage_dtype` **by value**,
/// adjacently, and `kernels::Ty` had no variant for an `enum class`.
///
/// [`kernels::Ty::KvScheme`] and [`kernels::Ty::KvDType`] are that variant —
/// two of them, not one. The refusal that mattered is
/// [`kernels::Ty::KvScheme`]'s own: the two operands are ADJACENT in both
/// parameter lists and the same width, so one shared kind would make the swap
/// type-check on every side this crate can check. Two kinds put the check
/// where the C++ can make it, in `abi::emit_device_typecheck`'s
/// function-pointer initialisation, which admits no conversions and which an
/// `enum class` admits none to begin with.
///
/// # §21.14's test, applied
///
/// *Does the new spelling make a wrong predicate well-formed?* The value
/// arrives as [`crate::runtime::ArgValue::U8`] and becomes
/// [`crate::device::Fact::Opaque`] — deliberately not a
/// [`Fact::Int`](crate::device::Fact::Int). An enumerator read as an integer
/// would make `Term::Multiple { operand: scheme, of: 2 }` a well-formed
/// clause meaning *"the bank is `Native` or `Int8PerTokenHead`"*, which is a
/// sentence nobody means and which selects an arm on the parity of a name.
/// With `Opaque` it is a [`Fact::Kind`](crate::device::Fault::Kind) fault,
/// and `Specialisation::agrees` refuses the clause before a fire, because
/// `Term::Multiple` requires `Ty::I32`. A NAME is not a NUMBER; the
/// vocabulary now says so.
///
/// # Why one row is sourced and one is not
///
/// `attn::attention_naive_paged` is DISPATCHED — `model::gemma_4`'s forward
/// reaches it through `dsl::cuda::attention_naive_paged` when a head width
/// FlashInfer's prefill template refuses (gemma-4's 512) needs a fallback —
/// so its row states where every argument comes from, and the operands are
/// `table::attn`'s own row expanded: that row hands the launcher a whole
/// `KvCacheLayerView` and the launcher takes it apart, so the fields it takes
/// apart INTO are what this row names.
///
/// `naive_paged_decode` has no dispatched launcher.
/// `driver-cuda/tests/launch_abi.rs:491` records
/// `attention_naive_paged_decode` as `NoRow::KernelsInternal` — it is called
/// by kernels code and by no statement — so a row claiming a symbol would be
/// claiming one nothing routes. It states its contract and its geometry and
/// carries no `Source`, exactly as [`QKV_FUSED_ROWS`]' decode triple does and
/// for the same reason.
pub const ATTENTION_NAIVE_PAGED: Unit = Unit {
    name: "attn/attention_naive_paged",
    root: include_str!("../../csrc/src/attn/attention_naive_paged.cuh"),
    rows: ATTENTION_NAIVE_PAGED_ROWS,
    options: &[],
};

/// `attn/attention_naive_paged.cuh`'s two rows.
///
/// `128` is `attention_naive_paged.cu:35`'s `constexpr int BLOCK = 128` and
/// is a SHARED-MEMORY contract, not a tuning constant: the launcher asks for
/// `(head_dim + BLOCK) * sizeof(float)` and the kernel cuts the tail of that
/// allocation into exactly `BLOCK` reduction slots
/// (`attention_naive_paged.cuh:402-404`). A row at another width would read
/// slots nothing wrote. [`crate::runtime::launch`]'s `PAGED_BLOCK` states the
/// same number on the geometry side and says so at greater length — including
/// that the `.cu` named above is DELETED, that the 128 was read off it while
/// it existed, and that the surviving oracle is the `.cuh`'s `template <int
/// BLOCK>` plus the instantiation string NVRTC checks.
#[rustfmt::skip]
static ATTENTION_NAIVE_PAGED_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &ATTENTION_NAIVE_PAGED_SIGS[0],
        template_path: "attn::device::naive_paged_attn",
        elem: "device::i32(128)",
    },
    DeviceKernel {
        sig: &ATTENTION_NAIVE_PAGED_SIGS[1],
        template_path: "attn::device::naive_paged_decode",
        elem: "device::i32(128)",
    },
];

/// The two contracts, in [`ATTENTION_NAIVE_PAGED_ROWS`]' order.
///
/// # `k_scales` and `v_scales` are nullable and `custom_mask` is nullable and
/// they are not the same kind of absent
///
/// The scale planes are null under `KvCacheScheme::Native`
/// (`bind::abi::KvCacheLayerView::k_scales`, *"null under
/// `KvCacheScheme::Native`"*) — absence means *"this bank is not quantised"*,
/// which is a fact the `scheme` operand states in the same breath. The mask
/// pair is null because THIS LAUNCHER passes it null:
/// `attention_naive_paged.cu:208-209` hands `nullptr` twice where
/// `attention_naive_paged_custom` at `:255-256` hands a real mask. Absence
/// there means *"causal, not custom"*, and the kernel's own
/// `use_custom_mask = custom_mask != nullptr` at `attention_naive_paged.cuh:393`
/// is what reads it.
///
/// Both are `| null` and both are real; naming the difference here is the
/// only place it is written down.
///
/// # The prefill row does NOT claim the ahead-of-time symbol's argument list
///
/// It claims the SYMBOL — `attn::attention_naive_paged` is what a statement
/// dispatches — and the operands are the `__global__`'s, which is the whole
/// point of a device row: the launcher's `KvCacheLayerView` and its
/// `num_pages_in_batch` (which the launcher casts to `void` at `:193`) do not
/// cross a `cuLaunchKernel`, and its `stream` is that call's sixth parameter
/// rather than an argument. `MLA_PAGED_SIGS` argues the same split.
#[rustfmt::skip]
static ATTENTION_NAIVE_PAGED_SIGS: [KernelSig; 2] = [
    // `attention_naive_paged.cu:195-221` --
    //
    //     dim3 grid(num_requests, total_tokens, num_q_heads);
    //     dim3 block(BLOCK);
    //     const std::size_t smem = (kv_layer.head_dim + BLOCK) * sizeof(float);
    //     device::naive_paged_attn<BLOCK><<<grid, block, smem, stream>>>(
    //         static_cast<const device::bf16*>(q),
    //         kv_layer.k_pages, kv_layer.v_pages,
    //         static_cast<const float*>(kv_layer.k_scales),
    //         static_cast<const float*>(kv_layer.v_scales),
    //         static_cast<device::bf16*>(o),
    //         qo_indptr_d, kv_page_indices_d, kv_page_indptr_d,
    //         kv_last_page_lens_d,
    //         nullptr, nullptr,
    //         num_q_heads, kv_layer.num_kv_heads, kv_layer.head_dim,
    //         kv_layer.page_size,
    //         static_cast<device::KvScheme>(kv_layer.scheme),
    //         static_cast<device::KvDType>(kv_layer.storage_dtype),
    //         kv_layer.block_size,
    //         window_left, sm_scale, logits_soft_cap, lse_out);
    //
    // The two `static_cast`s are the mirror correspondence this row's two new
    // `Ty`s reproduce: the host enum cannot cross NVRTC (its header pulls
    // `<cstdint>`), so `attention_naive_paged.cuh:187` and `:198` declare
    // device mirrors and `driver-cuda/tests/enum_mirrors.rs` asserts every
    // enumerator of both. A row naming the mirror is naming a checked type.
    //
    // THAT CHECK USED TO BE `attention_naive_paged.cu`'s `static_assert`s and
    // the file is deleted; the conclusion above is unchanged and the pair
    // being checked is now the LIVE one. Those asserts compared host C++
    // against the device mirror, and under NVRTC no host enum reaches a launch
    // -- the operand this row builds is Rust's `KvCacheScheme`/`DType`. The
    // replacement compares Rust against the `.cuh` directly and found the
    // drift the old pair could not see: two `DType` enumerators, `MXFP4_PACKED`
    // and `E8M0`, were never mirrored at all.
    kernel!(attention_naive_paged "attn::attention_naive_paged",
        file = Some("attn/attention_naive_paged.cuh"),
        launch = LaunchRule::PagedScores,
        operands = operands![
            q: Bf16s <- Source::In(0),
            k_pages: Buf <- Source::KvLayerField("k_pages"),
            v_pages: Buf <- Source::KvLayerField("v_pages"),
            k_scales: F32s | null <- Source::KvLayerField("k_scales"),
            v_scales: F32s | null <- Source::KvLayerField("v_scales"),
            o: BufMut <- Source::Out(0),
            qo_indptr: U32s <- Source::Attn("qo_indptr_d"),
            kv_page_indices: U32s <- Source::Attn("kv_page_indices_d"),
            kv_page_indptr: U32s <- Source::Attn("kv_page_indptr_d"),
            kv_last_page_lens: U32s <- Source::Attn("kv_last_page_lens_d"),
            custom_mask: U8s | null <- Source::Lit(Lit::Null),
            custom_mask_indptr: I32s | null <- Source::Lit(Lit::Null),
            // The head COUNT, which nobody carries: the query's width over
            // the cache's head dim. `table::attn`'s own row spells it the
            // same way and for the same reason.
            num_q_heads: I32 <- Source::Div(
                &Source::Width(&Source::In(0)),
                &Source::KvLayerField("head_dim"),
            ),
            num_kv_heads: I32 <- Source::KvLayerField("num_kv_heads"),
            head_dim: I32 <- Source::KvLayerField("head_dim"),
            page_size: I32 <- Source::KvLayerField("page_size"),
            scheme: KvScheme <- Source::KvLayerField("scheme"),
            storage_dtype: KvDType <- Source::KvLayerField("storage_dtype"),
            block_size: I32 <- Source::KvLayerField("block_size"),
            window_left: I32 <- Source::AttnWindow,
            sm_scale: F32 <- Source::Attn("sm_scale"),
            logits_soft_cap: F32 <- Source::Attn("logits_soft_cap"),
            lse_out: F32sMut <- Source::Attn("lse_out_d"),
        ]),
    // `attention_naive_paged.cu:147-171` --
    //
    //     dim3 grid(num_requests, num_q_heads);
    //     dim3 block(BLOCK);
    //     const std::size_t smem = (kv_layer.head_dim + BLOCK) * sizeof(float);
    //     device::naive_paged_decode<BLOCK><<<grid, block, smem, stream>>>(
    //
    // **No `Source`s**, and `ATTENTION_NAIVE_PAGED`'s doc gives the reason:
    // `attention_naive_paged_decode` is `NoRow::KernelsInternal`, so there is
    // no statement whose operands this row could be sourced from.
    //
    // `grid.x` is `num_requests` and [`LaunchRule::PagedScoresDecode`] reads
    // `Dims::rows`, which is the identification a decode's contract licenses
    // and a prefill's does not: one token per request makes `total_tokens ==
    // num_requests`. The rule's own doc argues it against `PagedScores`,
    // which cannot make it because a prefill spells both numbers in one
    // `dim3`.
    kernel!(naive_paged_decode "attn::naive_paged_decode",
        file = Some("attn/attention_naive_paged.cuh"),
        launch = LaunchRule::PagedScoresDecode,
        operands = operands![
            q: Bf16s, k_pages: Buf, v_pages: Buf,
            k_scales: F32s | null, v_scales: F32s | null,
            o: BufMut,
            kv_page_indices: U32s, kv_page_indptr: U32s, kv_last_page_lens: U32s,
            num_q_heads: I32, num_kv_heads: I32, head_dim: I32, page_size: I32,
            scheme: KvScheme, storage_dtype: KvDType, block_size: I32,
            window_left: I32, sm_scale: F32, logits_soft_cap: F32,
            lse_out: F32sMut,
        ]),
];

/// The three fused QKV epilogues — `qkv_fused.cu`'s five `<<<>>>`, as seven
/// rows.
///
/// # What changed, and what did not
///
/// The header's own prose says *"None of the five [becomes a row], and for one
/// reason each"*, and both reasons were about the VOCABULARY rather than about
/// this text. Both are gone:
///
///  * The warp launches size the grid in units of WARPS —
///    `ceil(num_requests * (num_q_heads + num_kv_heads) / (256/32))` — and
///    [`LaunchRule::WarpPackedHeads`] is now that arithmetic, cited at
///    `qkv_fused.cu:51-53`.
///  * The block and packed launches open `dim3(rows, num_q_heads +
///    num_kv_heads)`, and [`LaunchRule::RowsPackedHeads`] /
///    [`LaunchRule::RowsPackedHeadsNarrow`] are that grid at 256 and at 128
///    threads, cited at `:245-248` and `:98-102`.
///
/// The `USE_ROPE_TABLE` selector was the third blocker and is gone too:
/// [`Term::Present`] reads a `Fact::Address` and answers on whether the
/// pointer is null, which is what `qkv_fused.cu:56` and `:100` test.
/// [`Term::Aligned`] could not stand in for it — `0 % 16 == 0`, so an
/// alignment clause selects the TABLE arm for a fire with no table, and that
/// arm dereferences it.
///
/// # `HEAD_DIM` is a second selector and these rows do NOT reproduce it
///
/// `qkv_fused.cu:81`, `:85` and `:89` are a host chain — `if (head_dim == 64)
/// … == 128 … == 256` — choosing among THREE instantiations of the warp form,
/// and falling through to the block form for everything else. A row cannot
/// state that chain, and the reason is worth being exact about because the
/// near-misses are all spellable:
///
///  * [`Term::Multiple`] is the only clause that reads an integer, and
///    `Multiple { of: 64 }` holds of 64, 128, 192 and 256 alike. An arm list
///    ordered `256, 128, 64` would answer correctly for exactly the three
///    values the host tests and would send `head_dim = 192` to
///    `warp<64>` — `ELEMS_PER_THREAD = 2` where the head needs 6, so each
///    warp would norm and rotate the first 64 channels of a 192-wide head and
///    write them. That is the shape §21.14 measured: 34,273 of 55,200 cells
///    moved **while writing the same number of non-zero values**. A
///    permutation, not a truncation, and no count or norm flags it.
///  * A `Term::Equals { operand, value: i64 }` would spell the chain. It is
///    not added here, because adding a term to serve one launcher is how a
///    vocabulary stops being checkable — and because §21.14's test applies:
///    an integer equality makes `Equals { operand: a_pointer, value: 0 }`
///    well-formed, which is [`Term::Present`] spelled as its own negation and
///    is exactly the clause that term exists to make unspellable.
///
/// **So each warp row PINS its `HEAD_DIM` rather than inheriting it**, states
/// it in `elem` and in its symbol, and carries no [`Source`] — it is not
/// dispatchable, it is an instantiation `tests/units.rs` compiles and resolves.
/// One value is stated, 128, because that is the value the fires in this tree
/// use; 64 and 256 are two more `DeviceKernel`s of three lines each whenever a
/// fire needs them, and nothing about adding them is a decision.
///
/// # Why only ONE of the seven rows claims an ahead-of-time symbol
///
/// `attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16` is one launcher over
/// one kernel: `qkv_fused.cu:247` is the only `<<<>>>` it holds, its twenty
/// arguments are the kernel's eighteen plus the grid extent and the stream,
/// and the row states all eighteen with the same [`Source`]s
/// `table::attn`'s row states. It claims the symbol.
///
/// `attn::qkv_decode_qk_norm_rope_write_kv_bf16` is one launcher over FOUR
/// kernels, and a row taking its name would be claiming a quarter of a
/// launcher — the refusal [`PAGE_COMPACT_SIGS`] records for
/// `attn::compact_page_csr`, in a sharper form. Sharper because the four are
/// not interchangeable at the bit level: the block form reduces the norm
/// through `__shared__ float buf[BLOCK]` by halving and the warp form through
/// `__shfl_xor_sync`, so they sum the same `head_dim` floats in different
/// ORDERS. A row that claimed the launcher's symbol and always fired the block
/// form would compute the right answer to a different rounding — which passes
/// every tolerance and fails the byte-identity bar this crate is gated on. The
/// six decode rows are named for their kernels.
///
/// # THE refusal, restated after the `LaunchRule` audit
///
/// The two paragraphs above are both true and neither is the wall any more,
/// because the vocabulary moved under them. `WarpPackedHeads` and
/// `RowsPackedHeadsNarrow` were BOTH ported from this launcher — `:51-53` and
/// `:98-102` — and `Term::Present` now reads `rope_table != nullptr`, so of
/// this launcher's two selectors one is spellable and both of its geometries
/// are rules. That is enough to make the row look landable and it is not.
///
/// **What refuses it is that a [`Specialisation`] may not change a
/// [`LaunchRule`].** The four arms do not merely pick four instantiations;
/// they pick two GEOMETRIES:
///
/// ```text
/// :50-53   WARP_BLOCK = 256, total = num_requests * (num_q_heads + num_kv_heads),
///          warp_grid((total + 7) / 8)      -> WarpPackedHeads, a 1-D grid at 256
/// :97-99   BLOCK = 128, dim3(num_requests, num_q_heads + num_kv_heads)
///                                          -> RowsPackedHeadsNarrow, 2-D at 128
/// ```
///
/// A base row states ONE `launch`, and `Specialisation::agrees` requires every
/// arm's row to state the same one — `device.rs:1159-1163`, *"a specialisation
/// chooses an instantiation, not a geometry"*. This is a real invariant and
/// this audit kept it, for four reasons that do not depend on each other:
///
///  1. `runtime::fire` evaluates the geometry from the BASE row and only then
///     consults the specialisation (`runtime/fire.rs:176-186`). An arm that
///     changed the rule would be read after the grid it wanted was already
///     computed — so lifting the invariant means reordering `fire`, not
///     relaxing a check.
///  2. A row's `launch` would become a DEFAULT rather than a contract, and
///     `abi`, `emit`, `table` and `examples/migration_status` all read
///     `KernelSig::launch` as the row's answer. Four readers would start
///     reporting a geometry that some fires do not use.
///  3. The confusion is measured, not hypothetical. [`LaunchRule::
///     WarpPackedHeads`]' own doc records what the two rules give for the same
///     shape: *"eight times the blocks covering an eighth of the pairs"*.
///  4. It would not land this row anyway, and that is the part worth writing
///     down. The arm that would select between them is `head_dim == 64 | 128 |
///     256` (`:81`, `:85`, `:89`), and the section above already establishes
///     that no [`Term`] spells integer equality — `Multiple { of: 64 }` holds
///     of 192 — and that §21.14's test refuses adding one. So lifting a real
///     invariant here buys a row that is still refused on its other selector.
///     *Do not lift an invariant merely because lifting it would land a row*,
///     and doubly not when it would not.
///
/// The six kernel-named rows below are the whole of what this launcher can
/// give a JIT that keeps the invariant, and they are already landed.
/// `tests/specialise.rs::agrees_refuses_an_arm_that_changes_the_launch_rule`
/// is this paragraph as a test.
pub const QKV_FUSED: Unit = Unit {
    name: "attn/qkv_fused",
    root: include_str!("../../csrc/src/attn/qkv_fused.cuh"),
    rows: QKV_FUSED_ROWS,
    options: &[],
};

/// `attn/qkv_fused.cuh`'s seven rows.
///
/// The order is the file's: the packed prefill epilogue, then the block decode
/// triple, then the warp decode triple. Each triple is a CONTRACT row carrying
/// the kernel's twenty-two parameters and the two instantiations under
/// `#rope` / `#norope`, which is [`crate::device`]'s worked shape — the same
/// one [`KV_PAGED_ROWS`]' five `template <bool>` appenders take.
///
/// **The difference from those five, and it is the whole reason
/// [`Term::Present`] is not [`Term::Is`]**: `write_kv`'s flag is an operand of
/// the CONTRACT that no INSTANTIATION takes, so an arm forwards fifteen of
/// sixteen and `flags_are_covered` has to prove the base unreachable. Here
/// `rope_table` is a `const float*` PARAMETER of both instantiations — the
/// host passes it to `USE_ROPE_TABLE = false` too, which reads it never — so
/// every arm forwards all twenty-two and the base binds exactly what a
/// fall-through kernel declares. There is no cell to leave unread, and
/// `flags_are_covered` correctly finds nothing to check: it collects
/// [`Term::Is`] operands, and a null clause is not one.
#[rustfmt::skip]
static QKV_FUSED_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &QKV_FUSED_SIGS[0],
        template_path: "attn::device::qkv_packed_qk_norm_rope_vnorm_write_kv",
        elem: "device::i32(256)",
    },
    // ── the block form, `template <int BLOCK, bool USE_ROPE_TABLE>` ────────
    //
    // The base and the `#norope` row name the SAME instantiation, for the
    // reason `KV_PAGED_ROWS` gives: the base is unreachable once the arms are
    // proved total, and NVRTC accepts the repeated name expression.
    //
    // `flags_are_covered` enumerates `Term::Present` operands as well as
    // `Term::Is` ones, and finds nothing to prove here — both arms FORWARD
    // `rope_table` to their instantiation, so the retain drops it. That is the
    // right answer for the right reason: the hazard that check exists for is a
    // base row binding one cell more than the instantiation reads, and it
    // cannot arise when nothing is dropped.
    DeviceKernel { sig: &QKV_FUSED_SIGS[1], template_path: "attn::device::qkv_decode_qk_norm_rope_write_kv", elem: "device::i32(128), false" },
    DeviceKernel { sig: &QKV_FUSED_SIGS[2], template_path: "attn::device::qkv_decode_qk_norm_rope_write_kv", elem: "device::i32(128), true"  },
    DeviceKernel { sig: &QKV_FUSED_SIGS[3], template_path: "attn::device::qkv_decode_qk_norm_rope_write_kv", elem: "device::i32(128), false" },
    // ── the warp form, `template <int HEAD_DIM, bool USE_ROPE_TABLE>` ──────
    //
    // `128` here is a HEAD width and not a block width, and the two are
    // spelled the same. `BLOCK` above sizes `__shared__ float buf[BLOCK]`;
    // `HEAD_DIM` here fixes `ELEMS_PER_THREAD = HEAD_DIM / 32` and every
    // `#pragma unroll` under it, while the block width is the launcher's
    // `WARP_BLOCK = 256`, which the kernel reads at run time from
    // `blockDim.x`. See `QKV_FUSED_SIGS` for why the row states one head
    // width instead of dispatching on it.
    DeviceKernel { sig: &QKV_FUSED_SIGS[4], template_path: "attn::device::qkv_decode_qk_norm_rope_write_kv_warp", elem: "device::i32(128), false" },
    DeviceKernel { sig: &QKV_FUSED_SIGS[5], template_path: "attn::device::qkv_decode_qk_norm_rope_write_kv_warp", elem: "device::i32(128), true"  },
    DeviceKernel { sig: &QKV_FUSED_SIGS[6], template_path: "attn::device::qkv_decode_qk_norm_rope_write_kv_warp", elem: "device::i32(128), false" },
    // ── THE TWO EXPANSIONS THIS UNIT WAS MISSING ────────────────────────
    //
    // `QKV_FUSED`'s doc says the warp form is stated at ONE expansion, and
    // `qkv_fused.cu`'s header names the consequence: *"`head_dim == 64` and
    // `head_dim == 256` reach a `<<<>>>` below that no row names"*. The
    // launcher is a three-armed host `if` on `head_dim` and it cannot be
    // ported while two of its three arms have no instantiation to fire.
    //
    // These four are those arms. The `#rope`/`#norope` pair per width is the
    // `USE_ROPE_TABLE` template argument, chosen from
    // `rope_table != nullptr`, which is a host null test and not a `Term`.
    // There is no base row for either width: the base at `d128` exists for a
    // `Specialisation` that names it, and nothing names these.
    DeviceKernel { sig: &QKV_FUSED_SIGS[7],  template_path: "attn::device::qkv_decode_qk_norm_rope_write_kv_warp", elem: "device::i32(64), true"   },
    DeviceKernel { sig: &QKV_FUSED_SIGS[8],  template_path: "attn::device::qkv_decode_qk_norm_rope_write_kv_warp", elem: "device::i32(64), false"  },
    DeviceKernel { sig: &QKV_FUSED_SIGS[9],  template_path: "attn::device::qkv_decode_qk_norm_rope_write_kv_warp", elem: "device::i32(256), true"  },
    DeviceKernel { sig: &QKV_FUSED_SIGS[10], template_path: "attn::device::qkv_decode_qk_norm_rope_write_kv_warp", elem: "device::i32(256), false" },
];

/// The seven contracts, in [`QKV_FUSED_ROWS`]' order.
///
/// # `win` is nullable and `row_valid` is nullable and they are not the same
/// kind of absent
///
/// `row_valid` is a validity mask a fire either published or did not, and the
/// kernel's test is `row_valid != nullptr && row_valid[row] == 0` — absence
/// means *"every row is valid"*. `win` is the Peel device window's prefix
/// form, and `qkv_fused.cu:180` hands it `nullptr` from the non-devwin entry
/// point outright: absence means *"the split is not device-decided"*. Both
/// are `| null` and both are real; naming them together here is the only
/// place the difference is written down.
///
/// # `rope_table` is nullable, and that is what makes the arms statable
///
/// [`Specialisation::agrees`] refuses a [`Term::Present`] over an operand the
/// row does not declare nullable, and the refusal is not a formality: if the
/// binder cannot produce a null there, the clause is true for every fire that
/// reaches it, the `#rope` arm always wins, and `#norope` is an instantiation
/// that compiles and never runs. An arm that can never be taken is worse than
/// no arm — it reads as a covered case.
#[rustfmt::skip]
static QKV_FUSED_SIGS: [KernelSig; 11] = [
    // `qkv_fused.cu:245-248` --
    //
    //     constexpr int BLOCK = 256;
    //     dim3 grid(num_rows, num_q_heads + num_kv_heads);
    //     device::qkv_packed_qk_norm_rope_vnorm_write_kv<BLOCK>
    //         <<<grid, BLOCK, 0, stream>>>(...);
    //
    // `Source`s copied from `table::attn`'s `qkv_packed_post` row minus its
    // `num_rows` and its `stream`: the extent is `LaunchRule::RowsPackedHeads`'
    // `grid.x` and the stream is not a kernel parameter. Every other operand
    // is the same expression, deliberately -- the JIT row and the
    // ahead-of-time row bind ONE kernel and a fire that disagreed with itself
    // across the two paths would be the §21.7 defect in a new place.
    kernel!(qkv_packed_post "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
        file = Some("attn/qkv_fused.cuh"),
        launch = LaunchRule::RowsPackedHeads,
        operands = operands![
            packed: Buf <- Source::In(0),
            q_out: BufMut <- Source::Out(0),
            k_pages: BufMut <- Source::KvLayerField("k_pages"),
            v_pages: BufMut <- Source::KvLayerField("v_pages"),
            q_weight: Buf <- Source::Weight(0),
            k_weight: Buf <- Source::Weight(1),
            positions: I32s <- Source::Positions,
            kv_page_indices: U32s <- Source::Attn("kv_page_indices_d"),
            kv_page_indptr: U32s <- Source::Attn("kv_page_indptr_d"),
            kv_last_page_lens: U32s <- Source::Attn("kv_last_page_lens_d"),
            row_valid: U8s | null <- Source::Attn("row_valid_d"),
            num_q_heads: I32 <- Source::Div(
                &Source::Width(&Source::Out(0)),
                &Source::KvLayerField("head_dim"),
            ),
            num_kv_heads: I32 <- Source::KvLayerField("num_kv_heads"),
            head_dim: I32 <- Source::KvLayerField("head_dim"),
            page_size: I32 <- Source::KvLayerField("page_size"),
            hnd_layout: Bool <- Source::KvLayerField("hnd_layout"),
            theta: F32 <- Source::CtxByLayer("theta"),
            eps: F32 <- Source::Ctx("eps"),
        ]),
    // `qkv_fused.cu:98-102` and `:126-127` --
    //
    //     constexpr int BLOCK = 128;
    //     dim3 grid(num_requests, num_q_heads + num_kv_heads);
    //     if (rope_table != nullptr) {
    //         device::qkv_decode_qk_norm_rope_write_kv<BLOCK, true>
    //             <<<grid, BLOCK, 0, stream>>>(...);   // :101-102
    //     } else {
    //         device::qkv_decode_qk_norm_rope_write_kv<BLOCK, false>
    //             <<<grid, BLOCK, 0, stream>>>(...);   // :126-127
    //     }
    //
    // **No `_bf16` suffix and no `Source`s, and the two absences are one
    // decision.** `MLA_PAGED_SIGS` argues the suffix: a format suffix claims
    // this row picked bf16 out of the formats the template could take, and
    // this template has no type parameter to pick with -- every buffer is
    // `device::bf16` in its own declaration. The `Source`s are absent because
    // the symbol a statement dispatches is
    // `attn::qkv_decode_qk_norm_rope_write_kv_bf16`, which is the LAUNCHER
    // over four kernels, and this row is one of them.
    //
    // This row states `RowsPackedHeadsNarrow` and the warp triple below states
    // `WarpPackedHeads`, which is the whole refusal in two lines: the launcher
    // chooses between them on `head_dim`, and a `Specialisation` over the
    // launcher's symbol would have to change the `LaunchRule` between its
    // arms, which `Specialisation::agrees` forbids and this audit decided to
    // keep forbidding. See `QKV_FUSED`'s doc for the four legs of that
    // decision, including the one that matters most -- lifting it would not
    // land the row, because `head_dim == 64 | 128 | 256` is still unspellable.
    kernel!(qkv_decode_block "attn::qkv_decode_qk_norm_rope_write_kv",
        file = Some("attn/qkv_fused.cuh"),
        launch = LaunchRule::RowsPackedHeadsNarrow,
        operands = operands![
            packed: Buf, q_out: BufMut, k_pages: BufMut, v_pages: BufMut,
            q_weight: Buf, k_weight: Buf, positions: I32s,
            rope_table: F32s | null,
            kv_page_indices: U32s, kv_page_indptr: U32s, kv_last_page_lens: U32s,
            w_page: U32s | null, w_off: U32s | null, row_valid: U8s | null,
            win: U32s | null,
            num_q_heads: I32, num_kv_heads: I32, head_dim: I32, page_size: I32,
            hnd_layout: Bool, theta: F32, eps: F32,
        ]),
    kernel!(qkv_decode_block_rope "attn::qkv_decode_qk_norm_rope_write_kv#rope",
        file = Some("attn/qkv_fused.cuh"),
        launch = LaunchRule::RowsPackedHeadsNarrow,
        operands = operands![
            packed: Buf, q_out: BufMut, k_pages: BufMut, v_pages: BufMut,
            q_weight: Buf, k_weight: Buf, positions: I32s,
            rope_table: F32s | null,
            kv_page_indices: U32s, kv_page_indptr: U32s, kv_last_page_lens: U32s,
            w_page: U32s | null, w_off: U32s | null, row_valid: U8s | null,
            win: U32s | null,
            num_q_heads: I32, num_kv_heads: I32, head_dim: I32, page_size: I32,
            hnd_layout: Bool, theta: F32, eps: F32,
        ]),
    kernel!(qkv_decode_block_norope "attn::qkv_decode_qk_norm_rope_write_kv#norope",
        file = Some("attn/qkv_fused.cuh"),
        launch = LaunchRule::RowsPackedHeadsNarrow,
        operands = operands![
            packed: Buf, q_out: BufMut, k_pages: BufMut, v_pages: BufMut,
            q_weight: Buf, k_weight: Buf, positions: I32s,
            rope_table: F32s | null,
            kv_page_indices: U32s, kv_page_indptr: U32s, kv_last_page_lens: U32s,
            w_page: U32s | null, w_off: U32s | null, row_valid: U8s | null,
            win: U32s | null,
            num_q_heads: I32, num_kv_heads: I32, head_dim: I32, page_size: I32,
            hnd_layout: Bool, theta: F32, eps: F32,
        ]),
    // `qkv_fused.cu:51-53`, `:57-58` and `:70-71` --
    //
    //     constexpr int WARP_BLOCK = 256;
    //     const int total_units = num_requests * (num_q_heads + num_kv_heads);
    //     dim3 warp_grid((total_units + (WARP_BLOCK / 32) - 1) / (WARP_BLOCK / 32));
    //     if (rope_table != nullptr) {
    //         device::qkv_decode_qk_norm_rope_write_kv_warp<(HEAD_DIM_VALUE), true>
    //             <<<warp_grid, WARP_BLOCK, 0, stream>>>(...);   // :57-58
    //     } else {
    //         device::qkv_decode_qk_norm_rope_write_kv_warp<(HEAD_DIM_VALUE), false>
    //             <<<warp_grid, WARP_BLOCK, 0, stream>>>(...);   // :70-71
    //     }
    //
    // The symbol carries `_d128` because the row does: `HEAD_DIM` is a
    // template argument here and the launcher chooses it from a host `if`
    // chain no `Term` reproduces. See `QKV_FUSED`'s doc.
    //
    // **`num_requests` is an operand here and is not one above**, which reads
    // like an inconsistency and is the grid's. The block form gets the request
    // index from `blockIdx.x` and needs no count; the warp form flattens
    // `(request, head)` into one axis, recovers `r = unit / total_qk_heads` at
    // `qkv_fused.cuh:267`, and has to be told where the units stop. Both are
    // `Dims::rows` on the rule side -- `WarpPackedHeads` multiplies it in and
    // `RowsPackedHeadsNarrow` opens an axis over it -- and the kernel that
    // needs it as a bound takes it as a bound. `MLA_PAGED_SIGS`' `r` is the
    // same split.
    //
    // **And `head_dim` is NOT an operand here**, because `HEAD_DIM` is the
    // template argument. A row that carried both would let a fire state a
    // width the instantiation was not compiled for.
    kernel!(qkv_decode_warp "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128",
        file = Some("attn/qkv_fused.cuh"),
        launch = LaunchRule::WarpPackedHeads,
        operands = operands![
            packed: Buf, q_out: BufMut, k_pages: BufMut, v_pages: BufMut,
            q_weight: Buf, k_weight: Buf, positions: I32s,
            rope_table: F32s | null,
            kv_page_indices: U32s, kv_page_indptr: U32s, kv_last_page_lens: U32s,
            w_page: U32s | null, w_off: U32s | null, row_valid: U8s | null,
            win: U32s | null,
            num_requests: I32, num_q_heads: I32, num_kv_heads: I32,
            page_size: I32, hnd_layout: Bool, theta: F32, eps: F32,
        ]),
    kernel!(qkv_decode_warp_rope "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#rope",
        file = Some("attn/qkv_fused.cuh"),
        launch = LaunchRule::WarpPackedHeads,
        operands = operands![
            packed: Buf, q_out: BufMut, k_pages: BufMut, v_pages: BufMut,
            q_weight: Buf, k_weight: Buf, positions: I32s,
            rope_table: F32s | null,
            kv_page_indices: U32s, kv_page_indptr: U32s, kv_last_page_lens: U32s,
            w_page: U32s | null, w_off: U32s | null, row_valid: U8s | null,
            win: U32s | null,
            num_requests: I32, num_q_heads: I32, num_kv_heads: I32,
            page_size: I32, hnd_layout: Bool, theta: F32, eps: F32,
        ]),
    kernel!(qkv_decode_warp_norope "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128#norope",
        file = Some("attn/qkv_fused.cuh"),
        launch = LaunchRule::WarpPackedHeads,
        operands = operands![
            packed: Buf, q_out: BufMut, k_pages: BufMut, v_pages: BufMut,
            q_weight: Buf, k_weight: Buf, positions: I32s,
            rope_table: F32s | null,
            kv_page_indices: U32s, kv_page_indptr: U32s, kv_last_page_lens: U32s,
            w_page: U32s | null, w_off: U32s | null, row_valid: U8s | null,
            win: U32s | null,
            num_requests: I32, num_q_heads: I32, num_kv_heads: I32,
            page_size: I32, hnd_layout: Bool, theta: F32, eps: F32,
        ]),
    // ── THE OTHER TWO HEAD WIDTHS ───────────────────────────────────────
    //
    // Same contract as `_d128` above, four times over: the width is a
    // TEMPLATE argument, so `head_dim` is not an operand and each row names
    // exactly one instantiation. `qkv_fused.cu:92-104` is the host `if` chain
    // that picks between them --
    //
    // ```text
    // :92    if (head_dim == 64)  { LAUNCH_QKV_DECODE_POST_WARP(64);  return; }
    // :96    if (head_dim == 128) { LAUNCH_QKV_DECODE_POST_WARP(128); return; }
    // :100   if (head_dim == 256) { LAUNCH_QKV_DECODE_POST_WARP(256); return; }
    // ```
    //
    // -- and it falls THROUGH to the block form for every other width, which
    // is why the chain cannot be a `Specialisation`: the fallthrough changes
    // the `LaunchRule` from `WarpPackedHeads` to `RowsPackedHeadsNarrow`, and
    // `Specialisation::agrees` forbids an arm that changes the rule.
    // `driver-cuda/src/fire/qkv_fused.rs` is the chain now.
    kernel!(qkv_decode_warp_d64_rope "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#rope",
        file = Some("attn/qkv_fused.cuh"),
        launch = LaunchRule::WarpPackedHeads,
        operands = operands![
            packed: Buf, q_out: BufMut, k_pages: BufMut, v_pages: BufMut,
            q_weight: Buf, k_weight: Buf, positions: I32s,
            rope_table: F32s | null,
            kv_page_indices: U32s, kv_page_indptr: U32s, kv_last_page_lens: U32s,
            w_page: U32s | null, w_off: U32s | null, row_valid: U8s | null,
            win: U32s | null,
            num_requests: I32, num_q_heads: I32, num_kv_heads: I32,
            page_size: I32, hnd_layout: Bool, theta: F32, eps: F32,
        ]),
    kernel!(qkv_decode_warp_d64_norope "attn::qkv_decode_qk_norm_rope_write_kv_warp_d64#norope",
        file = Some("attn/qkv_fused.cuh"),
        launch = LaunchRule::WarpPackedHeads,
        operands = operands![
            packed: Buf, q_out: BufMut, k_pages: BufMut, v_pages: BufMut,
            q_weight: Buf, k_weight: Buf, positions: I32s,
            rope_table: F32s | null,
            kv_page_indices: U32s, kv_page_indptr: U32s, kv_last_page_lens: U32s,
            w_page: U32s | null, w_off: U32s | null, row_valid: U8s | null,
            win: U32s | null,
            num_requests: I32, num_q_heads: I32, num_kv_heads: I32,
            page_size: I32, hnd_layout: Bool, theta: F32, eps: F32,
        ]),
    kernel!(qkv_decode_warp_d256_rope "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#rope",
        file = Some("attn/qkv_fused.cuh"),
        launch = LaunchRule::WarpPackedHeads,
        operands = operands![
            packed: Buf, q_out: BufMut, k_pages: BufMut, v_pages: BufMut,
            q_weight: Buf, k_weight: Buf, positions: I32s,
            rope_table: F32s | null,
            kv_page_indices: U32s, kv_page_indptr: U32s, kv_last_page_lens: U32s,
            w_page: U32s | null, w_off: U32s | null, row_valid: U8s | null,
            win: U32s | null,
            num_requests: I32, num_q_heads: I32, num_kv_heads: I32,
            page_size: I32, hnd_layout: Bool, theta: F32, eps: F32,
        ]),
    kernel!(qkv_decode_warp_d256_norope "attn::qkv_decode_qk_norm_rope_write_kv_warp_d256#norope",
        file = Some("attn/qkv_fused.cuh"),
        launch = LaunchRule::WarpPackedHeads,
        operands = operands![
            packed: Buf, q_out: BufMut, k_pages: BufMut, v_pages: BufMut,
            q_weight: Buf, k_weight: Buf, positions: I32s,
            rope_table: F32s | null,
            kv_page_indices: U32s, kv_page_indptr: U32s, kv_last_page_lens: U32s,
            w_page: U32s | null, w_off: U32s | null, row_valid: U8s | null,
            win: U32s | null,
            num_requests: I32, num_q_heads: I32, num_kv_heads: I32,
            page_size: I32, hnd_layout: Bool, theta: F32, eps: F32,
        ]),
];

/// `attn/mla_paged.cuh`'s one row.
///
/// `elem` is [`DeviceKernel::PLAIN`] — the row's statement that
/// `attn::device::write_mla` has no template parameter list, as against the
/// empty string, which is what an unfilled field looks like. See
/// [`crate::x::attn::pack_dense_mask`] for the two refusals that make the
/// distinction checkable rather than conventional; that module took the
/// measurement with it when `PACK_DENSE_MASK` crossed.
static MLA_PAGED_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &MLA_PAGED_SIGS[0],
        template_path: "attn::device::write_mla",
        elem: DeviceKernel::PLAIN,
    },
    // `256` is `mla_paged.cu:56`'s `constexpr int BS = 256`, and the row
    // states it for `KIMI_MLA_ROWS[1]`'s reason: `mla_prepare<BLOCK_DIM>`
    // declares `__shared__ float buf[BLOCK_DIM]` and reduces by halving over
    // it, so the width sizes an array and fixes a tree rather than tuning
    // anything. `mla_paged.cuh:77` says the same in its own words — *"a value
    // the kernel is compiled AGAINST, not a hint"*.
    //
    // It also reaches the GRID. `LaunchRule::MlaPrepare` computes
    // `heads_per_block = half >= 256 ? 1 : 256 / half` from the same 256, so
    // the block width and the second grid axis are one number stated twice
    // and the rule's doc cites the line both readings come from. A row at
    // `<512>` under a 256-wide launch would leave the upper half of `buf`
    // unwritten and read it on the first halving step, AND compute half the
    // query blocks — two wrong answers from one changed literal.
    DeviceKernel {
        sig: &MLA_PAGED_SIGS[1],
        template_path: "attn::device::mla_prepare",
        elem: "device::i32(256)",
    },
];

/// The contract, which is the kernel's thirteen parameters and not the
/// launcher's eleven.
///
/// **The symbol is `attn::write_mla` and not `attn::write_mla_to_pages`**,
/// which is [`KV_PAGED_SIGS`]' rule applied to the same shape. The
/// ahead-of-time symbol takes a `MlaCacheLayerView` BY VALUE and unpacks it —
/// `mla_paged.cu:122` reads `layer.ckv_pages`, `layer.kpe_pages`,
/// `layer.page_size`, `layer.kv_lora_rank` and `layer.qk_rope_head_dim` out
/// of it and forwards to `write_mla_to_pages_bf16`, which holds the `<<<>>>`
/// at `:111`. The kernel takes those five unpacked, so a row claiming the
/// launcher's symbol would have to claim a view the `__global__` has never
/// seen. The row states what the kernel states, and the consequence is
/// visible and intended: this row does not move `examples/migration_status`,
/// it appears in the "no ahead-of-time twin" list beside
/// `attn::write_kv_bf16`, which is there for the same reason.
///
/// **No `_bf16` suffix**, and the absence is the point rather than an
/// oversight. A format suffix on a row means *"this row picked bf16 out of
/// the formats the template could have been instantiated at"* —
/// `attn::logit_softcap_f16` is the shape of that claim. This row picks
/// nothing: there is no template parameter to pick with, every buffer is
/// `bf16` in the kernel's own declaration, and a suffix would advertise a
/// choice that does not exist. `attn::count_kept` and `attn::scan_and_scatter`
/// are spelled the same way for the same reason.
///
/// `row_valid` is nullable and declared so: `mla_paged.cuh:190` is
/// `if (row_valid != nullptr && row_valid[t] == 0) return;`, and a fire that
/// published no validity mask hands a null. `r` is `num_requests` — the CSR's
/// request count, which `mla_resolve_dst` walks — and NOT the token count the
/// grid opens over, so it stays an operand: [`LaunchRule::PerRow`] recovers
/// `total_tokens` from `Dims::rows` and there is nothing in a rule that could
/// recover the other.
#[rustfmt::skip]
static MLA_PAGED_SIGS: [KernelSig; 2] = [
    // `mla_paged.cu:111` -- `device::write_mla<<<total_tokens, 256, 0, stream>>>`.
    //
    // `Dims::rows` is the token count for this op because `write_mla_to_pages`
    // is handed `ckv_curr` shaped `[Tokens, kv_lora_rank]` and opens one block
    // per row of it -- the same reading `PAGE_COMPACT` had to argue for its
    // request axis, and the easy direction of it.
    kernel!(write_mla "attn::write_mla",
        file = Some("attn/mla_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            ckv_curr: Buf, kpe_curr: Buf, ckv_pages: BufMut, kpe_pages: BufMut,
            qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            kv_last_page_lens: U32s, row_valid: U8s | null,
            r: I32, page_size: I32, kv_lora_rank: I32, qk_rope_head_dim: I32,
        ]),
    // `mla_paged.cu:73-74` --
    //
    //     dim3 grid(total_tokens, 1 + q_blocks);
    //     device::mla_prepare<BS><<<grid, BS, 0, stream>>>(...);
    //
    // with `:64-65` supplying the second axis:
    //
    //     const int heads_per_block = half >= BS ? 1 : (BS / half);
    //     const int q_blocks = (heads + heads_per_block - 1) / heads_per_block;
    //
    // and `:59` supplying `half = rope / 2`, `rope = layer.qk_rope_head_dim`.
    //
    // **The symbol is `attn::mla_prepare` and not `attn::mla_prepare_bf16`**,
    // and both halves of that are `write_mla`'s argument above applied
    // unchanged. The ahead-of-time symbol takes a `MlaCacheLayerView` BY VALUE
    // and unpacks `layer.ckv_pages`, `layer.kpe_pages`, `layer.page_size`,
    // `layer.kv_lora_rank` and `layer.qk_rope_head_dim` out of it before the
    // `<<<>>>`; the kernel takes those five unpacked, so a row claiming the
    // launcher's symbol would claim a view the `__global__` has never seen.
    // And there is no type template parameter to pick bf16 out of, so a format
    // suffix would advertise a choice that does not exist. The consequence is
    // the same and is intended: this row does not move
    // `examples/migration_status`.
    //
    // # The `1 +` is an operand nobody passes and a lane the kernel branches
    // on
    //
    // `mla_paged.cuh:236` reads `const int qb = blockIdx.y - 1;` and takes the
    // KV path when `qb < 0`. So `grid.y = 0` is one lane doing the `kv_a`
    // RMSNorm, the `k_pe` rotation and the paged write for its token, and
    // lanes `1..=q_blocks` are the query heads. Nothing in the argument list
    // says which is which -- the arithmetic is the rule's, entirely, which is
    // why `LaunchRule::MlaPrepare` had to be written rather than approximated.
    //
    // # `heads` is an operand AND reaches the rule, and the two readings
    // differ
    //
    // The rule computes `q_blocks` from `Dims::q_heads`; the kernel is told
    // `heads` so it can bound `h < heads` inside a block that covers
    // `heads_per_block` of them. Same number, two jobs -- the split
    // `KIMI_MLA_SIGS`' `total` keeps and the one this crate never collapses.
    //
    // `heads_per_block` is passed too, and it is the one operand that is pure
    // duplication of the rule: the host computes it at `:64`, the rule
    // recomputes it from `Dims::rotary_dims`, and the kernel is handed it
    // because it strides by it. A row cannot omit a parameter the kernel
    // declares, so the check that they agree is `tests/launch_rules.rs`'
    // transcription pin rather than anything at fire time.
    kernel!(mla_prepare "attn::mla_prepare",
        file = Some("attn/mla_paged.cuh"),
        launch = LaunchRule::MlaPrepare,
        operands = operands![
            kv_a: Buf, kv_a_norm_w: Buf, q_b: Buf,
            kv_c: BufMut, k_pe: BufMut, q_nope: BufMut, q_pe: BufMut,
            ckv_pages: BufMut, kpe_pages: BufMut,
            positions: I32s, qo_indptr: U32s, kv_page_indices: U32s,
            kv_page_indptr: U32s, kv_last_page_lens: U32s,
            row_valid: U8s | null,
            r: I32, page_size: I32, heads: I32, kv_lora: I32, nope: I32,
            rope: I32, src_row_stride: I32,
            eps: F32, theta: F32, interleaved: Bool, heads_per_block: I32,
            yarn_factor: F32, yarn_low_dim: F32, yarn_high_dim: F32,
            yarn_mscale: F32,
        ]),
];

/// `attn/kimi_mla.cuh`'s instantiations.
static KIMI_MLA_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &KIMI_MLA_SIGS[0],
        template_path: "attn::device::split_q_b",
        elem: "device::bf16",
    },
    // `256` IS THE ROW'S TO STATE, and stating it is the point of this
    // comment.
    //
    // `split_kv_a_norm` is `template <class T, int BLOCK_DIM = 256>`, and
    // until the argument LIST was shown to be statable this row could only
    // spell `<device::bf16>` and let the default supply the rest. That worked
    // and was fragile in a way nothing here would have caught: the kernel
    // declares `__shared__ float buf[BLOCK_DIM]` and reduces by halving from
    // `BLOCK_DIM / 2`, so the width is not a tuning knob — it SIZES AN ARRAY
    // and fixes a tree. `Rms` launches 256 threads. Had someone moved the
    // default to 512, `kimi_mla.cu` would have kept working, because it
    // spells `<device::bf16, BS>` with `constexpr int BS = 256` at line 57;
    // this row would have started instantiating a 512-wide reduction under a
    // 256-wide launch, where the upper half of `buf` is never written and the
    // first halving step reads it. That is a plausible number, not a crash.
    //
    // Both halves are cited, as a non-type argument requires: the launcher is
    // `attn/kimi_mla.cu:60`, `split_kv_a_norm<device::bf16, BS><<<tokens, BS,
    // 0, stream>>>` with `BS = 256`, and the template's default is 256. They
    // agree today, and the row no longer depends on their continuing to.
    DeviceKernel {
        sig: &KIMI_MLA_SIGS[1],
        template_path: "attn::device::split_kv_a_norm",
        elem: "device::bf16, 256",
    },
];

#[rustfmt::skip]
static KIMI_MLA_SIGS: [KernelSig; 2] = [
    // `total` is the source's element count and `Elementwise` covers exactly
    // that many threads, so the operand and the grid state the same number
    // for the reason `norm::tanh_bf16`'s `numel` does: the rule sizes the
    // launch, the argument bounds the guard, and a kernel cannot read a grid.
    // The twin passed `tokens` too, which is the extent the rule recovers,
    // and a stream: eight operands become seven.
    kernel!(kimi_split_q_b "attn::kimi_split_q_b_bf16",
        file = Some("attn/kimi_mla.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            q_b: Buf <- Source::In(0),
            q_nope: BufMut <- Source::Out(0),
            q_pe: BufMut <- Source::Out(1),
            total: I32 <- Source::InElements(0),
            heads: I32 <- Source::Param(0),
            nope: I32 <- Source::Param(1),
            rope: I32 <- Source::Param(2),
        ]),
    // `Rms` because the GEOMETRY is `Rms` — one block per token row, 256
    // wide, the row width read by a stride loop, the sum reduced in shared
    // memory. That the algebra is a split with an RMSNorm inside it is not
    // the rule's business: a rule names how many threads land where. The 32
    // bytes of dynamic shared memory the rule requests go unused, because the
    // reduction buffer is static and sized by the kernel's `BLOCK_DIM`.
    kernel!(kimi_split_kv_a_norm "attn::kimi_split_kv_a_norm_bf16",
        file = Some("attn/kimi_mla.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            kv_a: Buf <- Source::In(0),
            norm_weight: Buf <- Source::Weight(0),
            kv_c: BufMut <- Source::Out(0),
            k_pe: BufMut <- Source::Out(1),
            kv_lora_rank: I32 <- Source::OutWidth(0),
            qk_rope_dim: I32 <- Source::OutWidth(1),
            src_row_stride: I32 <- Source::InWidth(0),
            eps: F32 <- Source::Ctx("eps"),
        ]),
];

/// `attn/dsv4_compress.cuh`'s instantiations.
///
/// Four of the six name a launcher `driver-cuda/tests/launch_abi.rs`
/// classifies `NoRow::KernelsInternal` — `attention_compressed_bf16` calls
/// them and no statement does — so their operands carry no [`Source`]. That
/// is how this workspace's tables already spell "the binding is not decided
/// yet"; [`crate::table::attn`] does it for a dozen rows. A row is what tells
/// the compile which template to instantiate and what geometry to launch, and
/// both are facts about the kernel whether or not a statement reaches it.
/// Inventing a `Source` so a row LOOKED bindable would put a guess in the
/// table where an absence belongs.
static DSV4_COMPRESS_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &DSV4_COMPRESS_SIGS[0],
        template_path: "attn::device::average_pool",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DSV4_COMPRESS_SIGS[1],
        template_path: "attn::device::add_ape",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DSV4_COMPRESS_SIGS[2],
        template_path: "attn::device::gated_softmax_pool",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DSV4_COMPRESS_SIGS[3],
        template_path: "attn::device::dsv4_compress_gather",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DSV4_COMPRESS_SIGS[4],
        template_path: "attn::device::dsv4_compress_gather_paged",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DSV4_COMPRESS_SIGS[5],
        template_path: "attn::device::dsv4_store_comp_entries",
        elem: "device::bf16",
    },
    // `device::i32` is the DEFAULT the `.cuh` gives `T`, spelled out because
    // `instantiation()` always emits an argument list. `T` is unread by both
    // kernels; naming the default is what keeps the JIT's instantiation and
    // the archive's the same object rather than a second one that happens to
    // agree.
    DeviceKernel {
        sig: &DSV4_COMPRESS_SIGS[6],
        template_path: "attn::device::dsv4_boundary_meta_decode",
        elem: "device::i32",
    },
    DeviceKernel {
        sig: &DSV4_COMPRESS_SIGS[7],
        template_path: "attn::device::dsv4_boundary_meta_paged",
        elem: "device::i32",
    },
    // The attention itself, and a PLAIN `__global__` — no template parameter
    // to instantiate, so `elem` is the constant and not a type. See the row's
    // contract below for what the geometry cost.
    DeviceKernel {
        sig: &DSV4_COMPRESS_SIGS[8],
        template_path: "attn::device::compressed_attn_paged",
        elem: DeviceKernel::PLAIN,
    },
    // THE COMBINE HAS A ROW NOW, and `LaunchRule::Unstated` is the whole
    // point of it -- see the sig below, which carries the measurement.
    DeviceKernel {
        sig: &DSV4_COMPRESS_SIGS[9],
        template_path: "attn::device::combine_attn_outputs",
        elem: "device::bf16",
    },
];

#[rustfmt::skip]
static DSV4_COMPRESS_SIGS: [KernelSig; 10] = [
    // `n` is the INPUT token count and the grid covers `n / ratio * dim`, so
    // the extent the rule recovers and the extent the kernel is told differ
    // by the compression ratio. Both survive: the rule sizes the launch off
    // the result, the kernel divides its own index by `dim`.
    kernel!(dsv4_average_pool "attn::average_pool_bf16",
        file = Some("attn/dsv4_compress.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            input: Buf, output: BufMut, n: I32, dim: I32, ratio: I32,
        ]),
    // `_f32` names the ABSOLUTE POSITION TABLE's format and not the data's:
    // the table is fp32 and the rows it is added to are the row type's. The
    // launcher was named for the table and the row keeps that name, because a
    // symbol that changes spelling during a migration is a symbol two tables
    // disagree about.
    kernel!(dsv4_add_ape "attn::add_ape_f32",
        file = Some("attn/dsv4_compress.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            data: BufMut, ape: F32s, n_compressed: I32, dim: I32, ratio: I32,
        ]),
    kernel!(dsv4_gated_softmax_pool "attn::gated_softmax_pool_bf16",
        file = Some("attn/dsv4_compress.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            kv: Buf, score: Buf, output: BufMut, n: I32, dim: I32, ratio: I32,
        ]),
    // `RouteRows` — one block per compressed entry, the block as wide as the
    // row rounded up to a warp and clamped at 1024. That is the launcher's
    // `head_dim < 256 ? round32(head_dim) : 256` for every head dim this
    // family runs at; above 256 the kernel's stride loop covers the row
    // whatever the block width is, which is why the clamp is the rule's
    // business and not the kernel's.
    kernel!(dsv4_compress_gather "attn::dsv4_compress_gather_bf16",
        file = Some("attn/dsv4_compress.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            kv_proj: Buf, score_proj: Buf, ape: F32s,
            boundary_tok: I32s, boundary_pos: I32s, window_lo: I32s,
            out: BufMut, head_dim: I32, ratio: I32, coff: I32,
        ]),
    // The paged form, and the first of the two the planner actually names.
    // The twin's `num_entries` is gone — `RouteRows` recovers it as the row
    // count — and so is the stream: fourteen operands become twelve.
    kernel!(dsv4_compress_gather_paged "attn::dsv4_compress_gather_paged_bf16",
        file = Some("attn/dsv4_compress.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            state_kv: Buf <- Source::In(0),
            state_score: Buf <- Source::In(1),
            ape: F32s <- Source::In(2),
            boundary_pos: I32s <- Source::In(3),
            boundary_req: I32s <- Source::In(4),
            kv_page_indices: U32s <- Source::KvPageIndices,
            kv_page_indptr: U32s <- Source::KvPageIndptr,
            out: BufMut <- Source::Out(0),
            head_dim: I32 <- Source::OutWidth(0),
            ratio: I32 <- Source::Param(0),
            coff: I32 <- Source::Param(1),
            page_size: I32 <- Source::KvPageSize,
        ]),
    kernel!(dsv4_store_comp_entries "attn::dsv4_store_comp_entries_bf16",
        file = Some("attn/dsv4_compress.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            entries: Buf <- Source::In(0),
            comp_kv_pages: BufMut <- Source::Out(0),
            boundary_pos: I32s <- Source::In(1),
            boundary_req: I32s <- Source::In(2),
            kv_page_indices: U32s <- Source::KvPageIndices,
            kv_page_indptr: U32s <- Source::KvPageIndptr,
            head_dim: I32 <- Source::InWidth(0),
            page_size: I32 <- Source::KvPageSize,
        ]),
    // `Elementwise` — one thread per token, `ceil(rows * width / 256)` blocks
    // of 256.
    //
    // The launcher is `attn/dsv4_compress.cu`'s
    // `<<<(n + 127) / 128, 128, 0, stream>>>`, so the BLOCK differs and the
    // grid differs with it. That is in bounds here and nowhere else in this
    // family, because block width is not observable by these two kernels:
    // they hold no `__shared__`, take no `__syncthreads`, run no warp
    // primitive, and read `blockDim` only inside the flat index
    // `blockIdx.x * blockDim.x + threadIdx.x`. Every thread past `n` returns
    // on the kernel's own `t >= n`. Both shapes therefore visit `[0, n)` once
    // and nothing else — which is exactly the argument `dsv4_compress_gather`
    // above makes for `RouteRows` rounding a head dim up to a warp, and
    // exactly the argument `l2norm_scale` CANNOT make, because its
    // `__shared__ float buf[BLOCK]` puts the width in the algebra.
    //
    // The cover is an equality and not an inequality, which is what makes it
    // safe to state at all: `model-compiler`'s `dsl::cuda::dsv4_boundary_meta`
    // records all three outputs as `Shape(vec![Dim::Tokens])`, rank one, so a
    // fire's `Dims::rows` is the token count and its `width` is 1. The `n` the
    // kernel is told is that same token count. A row whose grid merely COVERS
    // an extent it cannot tie to `Dims` would drop trailing tokens the day a
    // caller passed a larger `n`; this one cannot, because `rows * width` IS
    // `n` by the statement that produces it.
    //
    // No `Source`s, mirroring the ahead-of-time twins, which carry none
    // either: `record_many` passes an empty parameter list, so `ratio` has no
    // `Source::Param` to name and inventing one would put a guess in the table
    // where an absence belongs. The stream is gone, as it is from every row.
    // §60.6's SYMBOL SPLIT: this row carried the `table::attn` symbol, which
    // made that symbol unit-hosted and so unwalkable (§52.11), and the
    // launcher in `dsv4_compress.cu` could not be taken over while the two
    // names were one. The TABLE symbol does not move -- it is what a trace
    // records. Launcher: `driver-cuda/src/fire/dsv4_compress.rs`.
    kernel!(dsv4_boundary_meta_decode "attn::dsv4_boundary_meta_decode_dev",
        file = Some("attn/dsv4_compress.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            positions: I32s, out_pos: I32sMut, out_req: I32sMut, out_rope: I32sMut,
            n: I32, ratio: I32, row_valid: U8s,
        ]),
    // The prefill form, same geometry and same launcher shape; it differs
    // only in resolving the request index by a binary search over `qo_indptr`
    // instead of shortcutting it to the token index.
    // §60.6's SYMBOL SPLIT: this row carried the `table::attn` symbol, which
    // made that symbol unit-hosted and so unwalkable (§52.11), and the
    // launcher in `dsv4_compress.cu` could not be taken over while the two
    // names were one. The TABLE symbol does not move -- it is what a trace
    // records. Launcher: `driver-cuda/src/fire/dsv4_compress.rs`.
    kernel!(dsv4_boundary_meta_paged "attn::dsv4_boundary_meta_paged_dev",
        file = Some("attn/dsv4_compress.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            positions: I32s, qo_indptr: U32s,
            out_pos: I32sMut, out_req: I32sMut, out_rope: I32sMut,
            n: I32, num_requests: I32, ratio: I32, row_valid: U8s,
        ]),
    // THE ROW THIS FILE REFUSED FOR A REASON THAT HAS SINCE GONE STALE, and
    // the second stale sentence about it is in the `.cuh` rather than here.
    //
    // `attn/dsv4_compress.cu:318-323`:
    //
    //     if (total_tokens <= 0 || num_q_heads <= 0) return;
    //     dim3 grid(static_cast<unsigned>(total_tokens),
    //               static_cast<unsigned>(num_q_heads));
    //     const std::size_t smem =
    //         (static_cast<std::size_t>(head_dim) + ATTN_BLOCK) * sizeof(float);
    //     device::compressed_attn_paged<<<grid, ATTN_BLOCK, smem, stream>>>(
    //
    // with `constexpr int ATTN_BLOCK = 128;` at `:37`.
    // `LaunchRule::PagedScoresDecode` is `[rows, q_heads, 1]` at
    // `[PAGED_BLOCK=128, 1, 1]` with `(head_dim + 128) * FLOAT` shared. Every
    // field, including the one this family's `.cuh` says no rule can state:
    // *"No ported rule computes a shared-memory size from an operand width"*
    // (`csrc/src/attn/dsv4_compress.cuh:50-52`). `PagedScores` and
    // `PagedScoresDecode` both do now, and that sentence is stale. It sits in
    // `csrc/**`, which this pass may not edit, so it is CORRECTED HERE and
    // reported.
    //
    // A SECOND stale sentence, `:76-78`: *"`compressed_attn` and
    // `compressed_attn_paged` are blocked by their HOST half."* True of
    // `compressed_attn`, whose launcher builds a `CompressedAttnParams[R]` on
    // the host and `cudaMallocAsync`s it. NOT true of this one: its host half
    // is a null guard, a grid, a smem, and one `<<<>>>`. Over-generalising the
    // sibling's blocker is how a refusal outlives its reason, and it is the
    // reason this row went unnoticed twice.
    //
    // `PagedScoresDecode` is ROWLESS in `runtime::launch` because the two
    // kernels it was ported for take `KvScheme`/`KvDType` by value and no
    // `Ty` names an enum operand. **This kernel takes neither** — thirteen
    // parameters, every one of them `Buf`/`BufMut`/`F32sMut`/`I32s`/`U32s`/
    // `I32`/`F32` — which is the whole of why the rule's third launcher gets
    // a row where its first two could not.
    //
    // Two of the twin's sixteen operands go, both for reasons this file has
    // already stated: `stream` is `cuLaunchKernel`'s sixth parameter and
    // `total_tokens` is the grid's first axis, which the rule opens and the
    // kernel reads as `blockIdx.x`. A third goes that is neither: `qo_indptr`
    // is `/*qo_indptr*/` in the launcher's own parameter list — commented out
    // at `dsv4_compress.cu:307`, never forwarded — so the twin carries a cell
    // the kernel has no parameter for. Thirteen operands, and the kernel's.
    //
    // No `Source`s, mirroring the twin (`table/attn.rs:480`), which carries
    // none. It is `whole = true` and `lacks = &[Cap::Scores]` there; neither
    // is a geometry fact and neither survives into a row that states its
    // rectangle.
    // §60.6's SYMBOL SPLIT: this row carried the `table::attn` symbol, which
    // made that symbol unit-hosted and so unwalkable (§52.11), and the
    // launcher in `dsv4_compress.cu` could not be taken over while the two
    // names were one. The TABLE symbol does not move -- it is what a trace
    // records. Launcher: `driver-cuda/src/fire/dsv4_compress.rs`.
    kernel!(attention_compressed_paged "attn::compressed_attn_paged_dev",
        file = Some("attn/dsv4_compress.cuh"),
        launch = LaunchRule::PagedScoresDecode,
        operands = operands![
            q: Buf, comp_kv_pages: Buf, o: BufMut, lse_out: F32sMut,
            positions: I32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            req_of_token: I32s,
            num_q_heads: I32, head_dim: I32, ratio: I32, page_size: I32,
            scale: F32,
        ]),
    // THE COMBINE, AND `Unstated` IS A FINDING RATHER THAN AN ABSENCE.
    //
    // `dsv4_compress.cu:65` and `:87-88`:
    //
    // ```text
    // :65   dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(num_heads));
    // :87   const int block = (head_dim < 32) ? 32 : ((head_dim > 256) ? 256 : head_dim);
    // :88   device::combine_attn_outputs<device::bf16><<<grid, block, 0, stream>>>(
    // ```
    //
    // The GRID is `LaunchRule::PerHeadElementwise` to the digit -- token on
    // `grid.x`, head on `grid.y`. The BLOCK is not: the launcher clamps
    // `head_dim` into `[32, 256]` and the rule clamps into `[32, 128]`, so on
    // a head wider than 128 the rule answers with half these threads. The
    // kernel strides `d += blockDim.x` and reduces nothing, so the narrower
    // block computes the same bytes in two passes -- a slower kernel, never a
    // wrong answer.
    //
    // That invisibility is why the rule is NOT claimed here. The launcher's
    // own comment put it best and it survives in
    // `driver-cuda/src/fire/dsv4_compress.rs`' doc comment in full: a row
    // stating `PerHeadElementwise` would agree with the `<<<>>>` at
    // deepseek_v4's 128-wide heads and stop agreeing at the first config that
    // widens one, with nothing failing and nothing reporting. Reconciling it
    // is a decision about `SINK_BLOCK_MAX` in `runtime/launch.rs`, and
    // `Unstated` is how a row declines to prejudge it.
    //
    // Unsourced, as every row in this unit is: the twin in `table::attn`
    // carries no `Source` either.
    kernel!(combine_attn_outputs "attn::combine_attn_outputs_dev",
        file = Some("attn/dsv4_compress.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            o1: Buf, lse1: F32s, o2: Buf, lse2: F32s,
            o_out: BufMut, lse_out: F32sMut,
            num_heads: I32, head_dim: I32,
        ]),
];

/// `attn/kv_paged.cuh`'s instantiations.
///
/// Every operand here is a field the ahead-of-time launcher unpacks out of a
/// `KvCacheLayerView` — the k half of a scale array, the packed page base,
/// the active page list — and no `Source` spells any of them, so the rows
/// carry none. The dequantiser the driver DOES name is
/// `attn::dequant_kv_cache_layer_to_bf16_active`, which is a launcher over
/// all four schemes and not a kernel.
///
/// `#[rustfmt::skip]`: the five specialised triples below differ in three
/// columns, and fifteen rows read as a table where seventy-five lines do not.
/// A reader checking that `#hnd` is `true` and `#nhd` is `false` for every one
/// of the five is doing a column scan.
#[rustfmt::skip]
static KV_PAGED_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &KV_PAGED_SIGS[0],
        template_path: "attn::device::dequant_fp8_per_token_head_pages_active",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &KV_PAGED_SIGS[1],
        template_path: "attn::device::dequant_int8_per_token_head_pages_active",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &KV_PAGED_SIGS[2],
        template_path: "attn::device::dequant_fp4_pages_active",
        elem: "device::bf16",
    },
    // ── the five `template <bool HND_LAYOUT>` appenders ────────────────────
    //
    // Three rows each, and the shape is [`crate::device`]'s worked example
    // rather than an invention of this file: a CONTRACT row carrying the
    // kernel's parameters plus `hnd_layout: Bool`, and the two variants under
    // `#hnd` / `#nhd` carrying the kernel's parameters and nothing else.
    //
    // `elem` is `device::true_type::value` and `device::false_type::value`.
    // These kernels take ONE template parameter and it is the flag, so the
    // flag lands in the slot [`DeviceKernel::instantiation`] prefixes with
    // `::pie_cuda_driver::kernels::` — and `::pie_cuda_driver::kernels::true`
    // is `expected an identifier` under NVRTC 13.0, measured. `pie_device.cuh`
    // ships the two tag types for exactly this.
    //
    // The base row and the `#nhd` row name the SAME instantiation. That is
    // deliberate and measured: the base is unreachable once
    // [`crate::device::Specialisation::agrees`] has proved the arms total, and
    // NVRTC accepts the repeated name expression rather than rejecting it as a
    // duplicate.
    DeviceKernel { sig: &KV_PAGED_SIGS[3],  template_path: "attn::device::write_kv",                 elem: "device::false_type::value" },
    DeviceKernel { sig: &KV_PAGED_SIGS[4],  template_path: "attn::device::write_kv",                 elem: "device::true_type::value"  },
    DeviceKernel { sig: &KV_PAGED_SIGS[5],  template_path: "attn::device::write_kv",                 elem: "device::false_type::value" },
    DeviceKernel { sig: &KV_PAGED_SIGS[6],  template_path: "attn::device::write_kv_at_positions",    elem: "device::false_type::value" },
    DeviceKernel { sig: &KV_PAGED_SIGS[7],  template_path: "attn::device::write_kv_at_positions",    elem: "device::true_type::value"  },
    DeviceKernel { sig: &KV_PAGED_SIGS[8],  template_path: "attn::device::write_kv_at_positions",    elem: "device::false_type::value" },
    DeviceKernel { sig: &KV_PAGED_SIGS[9],  template_path: "attn::device::write_kv_explicit",        elem: "device::false_type::value" },
    DeviceKernel { sig: &KV_PAGED_SIGS[10], template_path: "attn::device::write_kv_explicit",        elem: "device::true_type::value"  },
    DeviceKernel { sig: &KV_PAGED_SIGS[11], template_path: "attn::device::write_kv_explicit",        elem: "device::false_type::value" },
    DeviceKernel { sig: &KV_PAGED_SIGS[12], template_path: "attn::device::write_kv_explicit_devwin", elem: "device::false_type::value" },
    DeviceKernel { sig: &KV_PAGED_SIGS[13], template_path: "attn::device::write_kv_explicit_devwin", elem: "device::true_type::value"  },
    DeviceKernel { sig: &KV_PAGED_SIGS[14], template_path: "attn::device::write_kv_explicit_devwin", elem: "device::false_type::value" },
    DeviceKernel { sig: &KV_PAGED_SIGS[15], template_path: "attn::device::copy_kv_cells",            elem: "device::false_type::value" },
    DeviceKernel { sig: &KV_PAGED_SIGS[16], template_path: "attn::device::copy_kv_cells",            elem: "device::true_type::value"  },
    DeviceKernel { sig: &KV_PAGED_SIGS[17], template_path: "attn::device::copy_kv_cells",            elem: "device::false_type::value" },
    // ── the two view builders ─────────────────────────────────────────────
    //
    // Both are plain `__global__`s over `u32` indices, so both are
    // `DeviceKernel::PLAIN`, and both are one block by construction rather
    // than by budget: the CSR they build is a running sum, and the gather
    // that reads it has to see it.
    DeviceKernel { sig: &KV_PAGED_SIGS[18], template_path: "attn::device::build_window_page_view",   elem: DeviceKernel::PLAIN         },
    DeviceKernel { sig: &KV_PAGED_SIGS[19], template_path: "attn::device::build_full_split_view",    elem: DeviceKernel::PLAIN         },
    // ── the quantised appenders and the per-tensor dequant ────────────────
    //
    // Three plain `__global__`s and one `template <bool UseFp8>`. The two
    // `write_kv_per_token_head` rows are TWO SYMBOLS off one template and not
    // a `Specialisation`; the sigs below argue it where the geometry is.
    DeviceKernel { sig: &KV_PAGED_SIGS[20], template_path: "attn::device::write_kv_fp8_per_tensor",  elem: DeviceKernel::PLAIN         },
    DeviceKernel { sig: &KV_PAGED_SIGS[21], template_path: "attn::device::write_kv_per_token_head",  elem: "device::false_type::value" },
    DeviceKernel { sig: &KV_PAGED_SIGS[22], template_path: "attn::device::write_kv_per_token_head",  elem: "device::true_type::value"  },
    DeviceKernel { sig: &KV_PAGED_SIGS[23], template_path: "attn::device::write_kv_fp4_block",       elem: DeviceKernel::PLAIN         },
    DeviceKernel { sig: &KV_PAGED_SIGS[24], template_path: "attn::device::dequant_fp8_pages_active", elem: DeviceKernel::PLAIN         },
];

#[rustfmt::skip]
static KV_PAGED_SIGS: [KernelSig; 25] = [
    // `n` is `I64` because the kernel's parameter is `long long`, and it is a
    // `long long` because it indexes a page arena that is multiple gigabytes
    // at production page counts — `Ty::I64` says exactly that and the row
    // says nothing else.
    //
    // The fp8 pages are `U8s` and not a format of their own: on the device
    // they are `__nv_fp8_storage_t`, which IS one byte, and the format is the
    // kernel's to interpret. `attn::device::dequant_fp8_pages_active` — the
    // per-TENSOR form — is `KV_PAGED_SIGS[24]` at the end of this array, and
    // for most of this migration it was NOT: it takes the interpretation as
    // an `__nv_fp8_interpretation_t` argument and the `Ty` vocabulary had no
    // enum, so the row could not be spelled and defaulting the value to
    // `__NV_E4M3` would have decoded an E5M2 page to a numerically plausible
    // wrong answer. `kernels::Ty::Fp8Kind` closed that; the row is stated
    // now, and this paragraph is kept because the reasoning is why the value
    // is an operand rather than a template default.
    kernel!(dequant_fp8_per_token_head "attn::dequant_fp8_per_token_head_pages_active_bf16",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            k_pages: U8s, v_pages: U8s, k_scales: F32s, v_scales: F32s,
            k_out: BufMut, v_out: BufMut, page_indices: U32s,
            n: I64, page_size: I32, h_kv: I32, d: I32,
        ]),
    kernel!(dequant_int8_per_token_head "attn::dequant_int8_per_token_head_pages_active_bf16",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            k_pages: I8s, v_pages: I8s, k_scales: F32s, v_scales: F32s,
            k_out: BufMut, v_out: BufMut, page_indices: U32s,
            n: I64, page_size: I32, h_kv: I32, d: I32,
        ]),
    // `logical_n` and not `n`: an fp4 page holds two values per byte, so the
    // count the grid covers is the LOGICAL element count and every address
    // inside the kernel is derived from it by halving. The name is the
    // kernel's and the row keeps it, because an operand spelled `n` here
    // would be the one number in the signature that means something else.
    kernel!(dequant_fp4 "attn::dequant_fp4_pages_active_bf16",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            k_pages: U8s, v_pages: U8s, k_scales: F32s, v_scales: F32s,
            k_out: BufMut, v_out: BufMut, page_indices: U32s,
            logical_n: I64, page_size: I32, h_kv: I32, d: I32, block_size: I32,
        ]),

    // ── `write_kv`, `kv_paged.cu:84` ──────────────────────────────────────
    //
    // `if (hnd_layout)` over `write_kv<true>` at `:85` and `write_kv<false>`
    // at `:95`, both `<<<launch_tokens, 256, 0, stream>>>` — `LaunchRule::PerRow`
    // to the digit, where the launch's rows are the tokens it covers and the
    // kernel adds `first_token` to `blockIdx.x` itself.
    //
    // The symbol is the KERNEL'S and not `attn::write_kv_to_pages_bf16`'s,
    // because two host functions launch this one `__global__` —
    // `write_kv_to_pages_bf16` at `:85`/`:95` and `write_kv_to_pages_bf16_devwin`
    // at `kv_paged.cu:332`/`:342` — and a row that claimed either would be
    // claiming a launcher it only half is.
    //
    // `hnd_layout` is the SIXTEENTH operand and no kernel's parameter: it is
    // the launcher's argument, threaded down from the layer's KV-cache layout.
    // A fire has to be able to hand it and no instantiation can be handed it,
    // which is exactly why it belongs to the base and to nothing else.
    kernel!(write_kv "attn::write_kv_bf16",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            kv_last_page_lens: U32s, row_valid: U8s | null, win: U32s | null,
            r: I32, page_size: I32, h_kv: I32, d: I32, first_token: I32,
            hnd_layout: Bool,
        ]),
    kernel!(write_kv_hnd "attn::write_kv_bf16#hnd",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            kv_last_page_lens: U32s, row_valid: U8s | null, win: U32s | null,
            r: I32, page_size: I32, h_kv: I32, d: I32, first_token: I32,
        ]),
    kernel!(write_kv_nhd "attn::write_kv_bf16#nhd",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            kv_last_page_lens: U32s, row_valid: U8s | null, win: U32s | null,
            r: I32, page_size: I32, h_kv: I32, d: I32, first_token: I32,
        ]),

    // ── `write_kv_at_positions`, `kv_paged.cu:236` ────────────────────────
    //
    // `if (layer.hnd_layout)` over `<true>` at `:237` and `<false>` at `:246`,
    // both `<<<total_tokens, 256, 0, stream>>>`. `PerRow`, with no `first_token`
    // to offset: this form takes each token's absolute KV position as data.
    kernel!(write_kv_at_positions "attn::write_kv_at_positions_bf16",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            positions: I32s, position_delta: I32, qo_indptr: U32s,
            kv_page_indices: U32s, kv_page_indptr: U32s,
            r: I32, page_size: I32, h_kv: I32, d: I32,
            hnd_layout: Bool,
        ]),
    kernel!(write_kv_at_positions_hnd "attn::write_kv_at_positions_bf16#hnd",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            positions: I32s, position_delta: I32, qo_indptr: U32s,
            kv_page_indices: U32s, kv_page_indptr: U32s,
            r: I32, page_size: I32, h_kv: I32, d: I32,
        ]),
    kernel!(write_kv_at_positions_nhd "attn::write_kv_at_positions_bf16#nhd",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            positions: I32s, position_delta: I32, qo_indptr: U32s,
            kv_page_indices: U32s, kv_page_indptr: U32s,
            r: I32, page_size: I32, h_kv: I32, d: I32,
        ]),

    // ── `write_kv_explicit`, `kv_paged.cu:371` ────────────────────────────
    //
    // `if (layer.hnd_layout)` over `<true>` at `:372` and `<false>` at `:380`,
    // both `<<<B, 256, 0, stream>>>`.
    //
    // ── THE SYMBOL SPLIT, §60.6, AND WHY THESE THREE ROWS ARE `_dev` ─────
    //
    // These rows read `attn::write_kv_explicit_bf16` until this change, and
    // the paragraph below — kept verbatim, because it is still true about
    // the KERNEL — celebrated the join with `table::attn`'s row of the same
    // name. **That join is what had to go**, and the reason is §52.11's law
    // rather than a preference: `execution::tests::a_walk_is_only_a_walk`
    // requires every `WALKED` symbol to satisfy `unit_of(sym).is_none()` —
    // *a walk may drive a JIT'd kernel; it may not be one* — and
    // `kv_paged.cu:304`'s launcher IS a walk: a throw, an empty-extent
    // decline, an instantiation choice, and a CONDITIONAL second launch into
    // the envelope tier. §58 says the same thing from the other side: a
    // symbol is `Specialisation`-selected **or** `Walk`-driven, never both,
    // and a host program that needs a walk AND an instantiation choice
    // spends two symbols on it.
    //
    // So the walk keeps the trace's spelling — `table::attn` and
    // `model-compiler/src/dsl.rs:7393` still say `attn::write_kv_explicit_bf16`,
    // and no model text moved — and the DEVICE rows take the `_dev` suffix.
    // That is the direction §60.6 fixes: the ahead-of-time row's name is the
    // one a trace records, so it is the one that may not move.
    // `driver-cuda/src/fire/kv_paged.rs::write_kv_explicit_bf16` fires
    // `#hnd`/`#nhd` below directly, which is why [`WRITE_KV_EXPLICIT`]'s
    // arms resolve for a reader and no longer for a dispatcher.
    //
    // **The one symbol here that is also an ahead-of-time row's.**
    // `attn::write_kv_explicit_bf16` in [`crate::table::attn`] is the host
    // function at `kv_paged.cu:355`, and that function holds this `__global__`
    // and no other — a null guard, `if (layer.hnd_layout)`, two `<<<B, 256>>>`.
    // Its `B` is `Source::Rows` there and `PerRow`'s grid is `Dims::rows` here,
    // so the two state the same rectangle from the same number rather than
    // agreeing by coincidence. Sharing the string is what `examples/migration_status`
    // means by a join, and this row earns it.
    kernel!(write_kv_explicit "attn::write_kv_explicit_bf16_dev",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            w_page: U32s, w_off: U32s, row_valid: U8s | null,
            b: I32, page_size: I32, h_kv: I32, d: I32,
            hnd_layout: Bool,
        ]),
    kernel!(write_kv_explicit_hnd "attn::write_kv_explicit_bf16_dev#hnd",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            w_page: U32s, w_off: U32s, row_valid: U8s | null,
            b: I32, page_size: I32, h_kv: I32, d: I32,
        ]),
    kernel!(write_kv_explicit_nhd "attn::write_kv_explicit_bf16_dev#nhd",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            w_page: U32s, w_off: U32s, row_valid: U8s | null,
            b: I32, page_size: I32, h_kv: I32, d: I32,
        ]),

    // ── `write_kv_explicit_devwin`, `kv_paged.cu:283` ─────────────────────
    //
    // `if (layer.hnd_layout)` over `<true>` at `:284` and `<false>` at `:292`,
    // both `<<<n_max, 256, 0, stream>>>`. `win` is NOT nullable here — the
    // kernel reads `win[0]` and `win[1]` before any guard — which is the whole
    // difference from `write_kv_explicit` and the reason it is a second kernel.
    //
    // **THE SYMBOL WAS WRONG AND THE JOIN WAS THE THING THAT NOTICED.** These
    // three rows read `attn::write_kv_explicit_devwin_bf16` — the word order
    // swapped — where `model-compiler` records
    // `attn::write_kv_explicit_bf16_devwin` (`dsl.rs:3415`, emitted at
    // `model/src/shared/llama_like/forward/emit.rs`). A JIT symbol that is
    // not the trace's spelling is a kernel no model text can state, which is
    // exactly what `examples/migration_status`' join exists to catch: the
    // symbol is the same string in both DELIBERATELY. So this is a defect
    // repaired, not bookkeeping, and the row it lands is the one the audit
    // predicted from `Fact::Bool`/`Term::Is` — which had indeed landed, and
    // which the sibling `write_kv_explicit` rows above already take.
    //
    // WHAT THE AUDIT'S LINE DID NOT SAY, AND IS THE LOAD-BEARING PART: the
    // **AND THEN THE SYMBOL MOVED AGAIN, FOR §60.6's REASON.** These three
    // rows now read `attn::write_kv_explicit_bf16_devwin_dev`, which reads
    // like the defect above being reintroduced and is its opposite. The
    // paragraph above is about a device row whose spelling no model text
    // could state; this is about a device row that shares its spelling with
    // an ahead-of-time row the DRIVER now executes. `execution::tests::
    // a_walk_is_only_a_walk` asserts a `WALKED` symbol has no unit -- §52.11,
    // *a walk may drive a JIT'd kernel; it may not be one* -- so as long as
    // this unit hosted the string `attn::write_kv_explicit_bf16_devwin`, the
    // table row of that name could not be walked and `kv_paged.cu`'s C++
    // launcher could not be deleted. The sibling `write_kv_explicit` rows
    // twenty lines up took the same `_dev` suffix for the same reason and
    // their comment says so.
    //
    // The trace's spelling has NOT moved: `table::attn`'s row and
    // `dsl.rs:3468` are both still `attn::write_kv_explicit_bf16_devwin`, and
    // the join `examples/migration_status` performs is between THOSE. What
    // this rename separates is the two ends of the join, which had been one
    // string by accident of both being right.
    //
    // [`WRITE_KV_EXPLICIT_DEVWIN`]'s `base` moved with them, and the driver
    // fires `_dev#hnd` / `_dev#nhd` directly
    // (`driver-cuda/src/fire/kv_paged.rs`), so the `Specialisation` states
    // the selection for a reader and no longer performs it -- exactly the
    // arrangement [`WRITE_KV_EXPLICIT`] is already in.
    //
    // WHAT THE AUDIT'S LINE DID NOT SAY, AND IS THE LOAD-BEARING PART: the
    // grid. `PerRow` is `Dims::rows`; the launcher opens `n_max`, and `n_max`
    // is NOT this launch's region. It is the fire's FULL lane count —
    // `DispatchCtx::rows_total`, *"the fire's FULL row count, which a
    // `_devwin` launch spans regardless of how many rows its own region
    // serves"* (`driver-cuda/src/bind/mod.rs:884`) — because the grid spans
    // every lane and out-of-window rows early out on `win[0]`/`win[1]`, which
    // is what makes a captured launch replay across splits. `Dims::rows` is
    // `bound.rows.end - bound.rows.start` (`bind/mod.rs:1761`), the REGION,
    // and the `_devwin` special case just above it (`:1766`) zeroes the
    // pointer window and not the row count.
    //
    // The two are the same number anyway, and `whole` is why: the twin is
    // `whole = true` (`table/attn.rs:298`), and a `whole` statement is
    // refused any window but the whole fire — statically against `Peel`
    // regions (`model-compiler/src/kernels.rs:112`) and dynamically against
    // an arm that happens to select a subset (`lower.rs:1064-1073`,
    // `Uncovered::WholeKernelSplit`). So `bound.rows` is `[0, rows_total)`
    // for every launch that reaches here, and `PerRow` reproduces
    // `<<<n_max, 256>>>` by construction rather than by coincidence. Stated
    // here because a reader who checked only the unpeeled shape would have
    // found the grids byte-identical and learned nothing: that is hazard 1,
    // and `whole` is the thing that actually holds it.
    kernel!(write_kv_explicit_devwin "attn::write_kv_explicit_bf16_devwin_dev",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            w_page: U32s, w_off: U32s, row_valid: U8s | null, win: U32s,
            n_max: I32, page_size: I32, h_kv: I32, d: I32,
            hnd_layout: Bool,
        ]),
    kernel!(write_kv_explicit_devwin_hnd "attn::write_kv_explicit_bf16_devwin_dev#hnd",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            w_page: U32s, w_off: U32s, row_valid: U8s | null, win: U32s,
            n_max: I32, page_size: I32, h_kv: I32, d: I32,
        ]),
    kernel!(write_kv_explicit_devwin_nhd "attn::write_kv_explicit_bf16_devwin_dev#nhd",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            w_page: U32s, w_off: U32s, row_valid: U8s | null, win: U32s,
            n_max: I32, page_size: I32, h_kv: I32, d: I32,
        ]),

    // ── `copy_kv_cells`, `kv_paged.cu:418` ────────────────────────────────
    //
    // `if (layer.hnd_layout)` over `<true>` at `:419` and `<false>` at `:425`,
    // both `<<<N, 256, 0, stream>>>`. The beam-repair cell move: one block per
    // CELL, disjoint spans by contract, and the only one of the five whose
    // source and destination are both the page arena.
    //
    // The symbol matches [`crate::table::driver_internal`]'s
    // `attn::copy_kv_cells_bf16` for the same reason `write_kv_explicit` above
    // matches its own — one launcher, this kernel, `<<<N, 256>>>` on both
    // sides. `driver_internal` is out of `table::KERNELS` by construction, so
    // the match moves no migration number; it is here because a different
    // string would say these were different kernels.
    kernel!(copy_kv_cells "attn::copy_kv_cells_bf16",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_pages: BufMut, v_pages: BufMut,
            dst_page: U32s, dst_off: U32s, src_page: U32s, src_off: U32s,
            n: I32, page_size: I32, h_kv: I32, d: I32,
            hnd_layout: Bool,
        ]),
    kernel!(copy_kv_cells_hnd "attn::copy_kv_cells_bf16#hnd",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_pages: BufMut, v_pages: BufMut,
            dst_page: U32s, dst_off: U32s, src_page: U32s, src_off: U32s,
            n: I32, page_size: I32, h_kv: I32, d: I32,
        ]),
    kernel!(copy_kv_cells_nhd "attn::copy_kv_cells_bf16#nhd",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_pages: BufMut, v_pages: BufMut,
            dst_page: U32s, dst_off: U32s, src_page: U32s, src_off: U32s,
            n: I32, page_size: I32, h_kv: I32, d: I32,
        ]),
    // ── the two view builders, which are the two halves of one refusal ────
    //
    // This module's header refused both under *"a `<<<1, N>>>` no rule
    // spells"*, and named `RowsFlat` as the near miss it had checked and
    // rejected: `RowsFlat` answers `ceil(rows / 256)`, which is 1 for every
    // rectangle up to 256 rows and 2 at 257 — a second block walking the
    // same CSR from `threadIdx.x == 0`, writing the same running sum twice
    // with no ordering between them. `LaunchRule::Single` and
    // `LaunchRule::SingleWarp` are that literal `1`, and the two are two
    // rules because their BLOCKS differ and a block is the launcher's, not
    // the fire's.
    //
    // `attn/kv_paged.cu:515-517`:
    //
    // ```text
    // :515   if (R <= 0 || keep_pages <= 0) return;
    // :516   device::build_window_page_view<<<1, 256, 0, stream>>>(
    // :517       src_indices, src_indptr, keep_pages, dst_indptr, dst_indices, R);
    // ```
    //
    // Unsourced, and the ahead-of-time twin at `table/attn.rs:423` is too.
    // Every operand is a CSR the DRIVER builds while planning a windowed
    // read — `src_indptr` is the page table's, `keep_pages` is the model's
    // window divided by the page size, `R` is the batch — and no model text
    // names any of them. `crate::abi` skips a row with any
    // `Source::Unbound` operand whole, so this row states geometry and
    // generates no dispatch, which is the established shape here
    // (`qk_rmsnorm_mrope`, `naive_paged_decode`, the `_devwin` trio).
    kernel!(build_window_page_view "attn::build_window_page_view", whole = true,
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::Single,
        operands = operands![
            src_indices: U32s, src_indptr: U32s, keep_pages: I32,
            dst_indptr: U32sMut, dst_indices: U32sMut, R: I32,
        ]),
    // `attn/kv_paged.cu:533-535`:
    //
    // ```text
    // :533   device::build_full_split_view<<<1, 32, 0, stream>>>(
    // :534       src_indptr, src_last_page_len, splits, page_size,
    // :535       dst_indptr, dst_indices, dst_last, src_indices);
    // ```
    //
    // **32 and not 256, and the kernel says why**: `kv_paged.cuh:842` is
    // `if (threadIdx.x != 0) return;` and the whole body is a serial walk
    // over `splits`. Every thread but one exits immediately, so the launch
    // is one warp because a warp is the smallest thing the hardware
    // schedules — a fact about the DEVICE, which is why `SingleWarp` fixes
    // 32 rather than taking it from a `Dims` field.
    kernel!(build_full_split_view "attn::build_full_split_view", whole = true,
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::SingleWarp,
        operands = operands![
            src_indptr: U32s, src_last_page_len: U32s, splits: I32, page_size: I32,
            dst_indptr: U32sMut, dst_indices: U32sMut, dst_last: U32sMut,
            src_indices: U32s,
        ]),

    // ═══ THE QUANTISED APPENDERS AND THE PER-TENSOR DEQUANT ═══════════════
    //
    // Five rows added when `kernels::Ty::Fp8Kind` closed the gap this file's
    // header and `kv_paged.cuh:63-78` both name. Before it, two of these
    // kernels could not be spelled at all — they take an
    // `__nv_fp8_interpretation_t` and the vocabulary had no enum — and the
    // other three were held back with them because they are the same host
    // switch's other arms and a row set covering three arms of four is a
    // dispatch that silently writes a page in the wrong format.
    //
    // The switch is `kv_paged.cu:156-215`, on `layer.scheme`:
    //
    // ```text
    // :158   case KvCacheScheme::Fp8PerTensor:
    // :162       device::write_kv_fp8_per_tensor<<<total_tokens, BLOCK, 0, stream>>>
    // :170   case KvCacheScheme::Int8PerTokenHead:
    // :174       device::write_kv_per_token_head<false><<<grid, BLOCK, shmem, stream>>>
    // :183   case KvCacheScheme::Fp8PerTokenHead:
    // :187       device::write_kv_per_token_head<true><<<grid, BLOCK, shmem, stream>>>
    // :196   case KvCacheScheme::Fp4Block:
    // :203       device::write_kv_fp4_block<<<grid, 32, 0, stream>>>
    // :212   case KvCacheScheme::Native:
    // :213       break;                       // handled above, before the switch
    // ```
    //
    // `BLOCK` is `constexpr int BLOCK = 256` at `:155`.
    //
    // # Why `write_kv_per_token_head` is TWO symbols and not a specialisation
    //
    // Its template parameter is `bool UseFp8`, and the five appenders above
    // are `template <bool HND_LAYOUT>` under [`crate::device::Specialisation`]
    // — so the shape looks identical and is not. A `Specialisation` resolves
    // a flag that the CONTRACT row carries as an operand: `hnd_layout: Bool`
    // is in the base signature, the model states it, and `choose` reads it
    // back off the fire. `UseFp8` is nowhere in this kernel's parameter list.
    // It is read off `layer.scheme`, which is a property of the CACHE and not
    // of the call, and a base row inventing an operand to carry it would put
    // a cell in `cuLaunchKernel`'s array that the instantiation does not
    // read — exactly what `Specialisation::flags_are_covered` exists to
    // forbid in the other direction.
    //
    // The precedent is three rows up in this same array:
    // `dequant_fp8_per_token_head_pages_active_bf16` and
    // `dequant_int8_per_token_head_pages_active_bf16` are two symbols for two
    // storage formats off two templates, chosen by the same `layer.scheme`
    // switch at `kv_paged.cu:404`/`:416`. These two are that pair's write
    // halves and are named the same way.
    //
    // `attn/kv_paged.cu:162`:
    //
    // ```text
    // :162   device::write_kv_fp8_per_tensor<<<total_tokens, BLOCK, 0, stream>>>(
    // ```
    //
    // One block per token at 256 threads — `LaunchRule::PerRow` to the digit,
    // the same rule the bf16 `write_kv` rows carry, because it is the same
    // grid over the same tokens with a different destination format.
    //
    // The pages are `U8sMut` for the reason the dequant rows give above:
    // `__nv_fp8_storage_t` IS `unsigned char`, so the width is a byte and the
    // FORMAT is the `fp8_kind` operand's to say. That operand is why this row
    // exists: `kv_paged.cu:159-161` computes it on the host as
    // `layer.storage_dtype == DType::FP8_E5M2 ? __NV_E5M2 : __NV_E4M3`, and
    // that ternary is host logic, so it becomes Rust and arrives here as an
    // argument rather than as a template default. `kv_paged.cuh:66-68`
    // records what the default would have cost: *"`__NV_E5M2` pages would
    // silently decode as `__NV_E4M3`"*.
    kernel!(write_kv_fp8_per_tensor "attn::write_kv_fp8_per_tensor",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: U8sMut, v_pages: U8sMut,
            qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            R: I32, page_size: I32, h_kv: I32, d: I32, fp8_kind: Fp8Kind,
        ]),
    // `attn/kv_paged.cu:172-174` and `:185-187` — the two arms differ only in
    // the template argument and the scheme that selects them:
    //
    // ```text
    // :172   const dim3 grid(total_tokens, num_kv_heads);
    // :173   const std::size_t shmem = 2 * (BLOCK / 32) * sizeof(float);
    // :174   device::write_kv_per_token_head<false><<<grid, BLOCK, shmem, stream>>>(
    // ```
    //
    // **`LaunchRule::Unstated`, and the reason is the grid and the shared
    // memory both.** Two meaningful axes — tokens on `x`, KV heads on `y`,
    // and the kernel reads `blockIdx.y` at `kv_paged.cuh:427` — which no rule
    // here states; `RowsPerHead` is the nearest and is a FLATTENED product,
    // `rows · (width / head_dim)` blocks on one axis, so a kernel reading
    // `blockIdx.y` under it would find zero. On top of that the launch
    // carries dynamic shared memory, and no `LaunchRule` states smem at all.
    //
    // So the driver owns the `Launch`, which is `fire/attn_score.rs`' shape
    // and `fire/moe.rs`' precedent for this family of row. §10.5 forbids
    // growing the rule vocabulary for one kernel and this is two.
    //
    // `shmem` is `2 * (256 / 32) * sizeof(float)` = **64 bytes** — two floats
    // per warp, which `kv_paged.cuh:428` spends on the K and V absmax
    // reductions. It is a function of the BLOCK and not of `head_dim`, so it
    // is the same 64 bytes at every geometry; the driver states it as a
    // constant with that derivation beside it.
    kernel!(write_kv_int8_per_token_head "attn::write_kv_int8_per_token_head",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            k_scales: F32sMut, v_scales: F32sMut,
            qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            R: I32, page_size: I32, h_kv: I32, d: I32,
        ]),
    kernel!(write_kv_fp8_per_token_head "attn::write_kv_fp8_per_token_head",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            k_scales: F32sMut, v_scales: F32sMut,
            qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            R: I32, page_size: I32, h_kv: I32, d: I32,
        ]),
    // `attn/kv_paged.cu:199-203`:
    //
    // ```text
    // :199   const int block_size = layer.block_size > 0 ? layer.block_size : 16;
    // :201   const int blocks = (head_dim + block_size - 1) / block_size;
    // :202   const dim3 grid(total_tokens, num_kv_heads, blocks);
    // :203   device::write_kv_fp4_block<<<grid, 32, 0, stream>>>(
    // ```
    //
    // **Three meaningful axes at 32 threads**, and the kernel reads all three
    // (`kv_paged.cuh:563-565` is `blockIdx.x`, `.y`, `.z`). `Unstated` for
    // the reason above, one axis further out: `SingleWarp` fixes 32 threads
    // and `dim3(1, heads, rows)`, which is two axes and a fixed `x`.
    //
    // **The `block_size` default is a MEASUREMENT and it is carried.** The
    // host reads `layer.block_size` and substitutes 16 when it is not
    // positive — 16 because an fp4 block scale covers 16 values, which is the
    // arena's own layout and not a tuning knob. It appears twice in the C++,
    // here and at `kv_paged.cu:429-431` in the dequant, and the two must
    // agree or a page is written in blocks of one width and read in blocks
    // of another. The Rust states it once, in the view, so they cannot drift.
    kernel!(write_kv_fp4_block "attn::write_kv_fp4_block",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: U8sMut, v_pages: U8sMut,
            k_scales: F32sMut, v_scales: F32sMut,
            qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            kv_last_page_lens: U32s,
            R: I32, page_size: I32, h_kv: I32, d: I32, block_size: I32,
        ]),
    // `attn/kv_paged.cu:397`:
    //
    // ```text
    // :394   const auto fp8_kind = layer.storage_dtype == DType::FP8_E5M2
    // :395       ? __NV_E5M2 : __NV_E4M3;
    // :397   device::dequant_fp8_pages_active<<<blocks, BLOCK, 0, stream>>>(
    // ```
    //
    // where `blocks` is `(logical_n + 255) / 256` at `:388` — the same
    // `LaunchRule::Elementwise` its three siblings above carry, over the same
    // `logical_n`.
    //
    // **This is the row the array's own comment said could not exist.** The
    // note above `dequant_fp8_per_token_head` reads *"`dequant_fp8_pages_active`
    // — the per-TENSOR form — has no row for the other half of that sentence:
    // it takes the interpretation as an `__nv_fp8_interpretation_t` argument
    // and the `Ty` vocabulary has no enum."* The vocabulary now has one, so
    // the row is here and that sentence is answered rather than deleted.
    //
    // `page_elems` and not `page_size, h_kv, d`: the per-tensor form takes
    // the product pre-multiplied (`layer.page_size * num_kv_heads * head_dim`
    // at `kv_paged.cu:385`) because a per-tensor scale needs no per-head
    // addressing, which is exactly the difference from its three siblings.
    kernel!(dequant_fp8_per_tensor "attn::dequant_fp8_pages_active_bf16",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            k_pages: U8s, v_pages: U8s, k_out: BufMut, v_out: BufMut,
            page_indices: U32s, n: I64, page_elems: I32, fp8_kind: Fp8Kind,
        ]),
];

// ===========================================================================
// SPECIALISATION — the five `template <bool HND_LAYOUT>` appenders.
// ===========================================================================

/// The base's first `n` operands, in the kernel's order.
///
/// A `const fn` and not five hand-written arrays, because the five takes are
/// the same list at five lengths and the one thing that could go wrong with
/// hand-writing them is an index that skips. Every arm here forwards a PREFIX
/// — the flag is always the last operand of the base and is forwarded by no
/// arm, which is what makes `Specialisation::flags_are_covered` apply.
const fn prefix<const N: usize>() -> [Take; N] {
    let mut out = [Take::From(0); N];
    let mut i = 0;
    while i < N {
        out[i] = Take::From(i);
        i += 1;
    }
    out
}

static TAKE_15: [Take; 15] = prefix();
static TAKE_13: [Take; 13] = prefix();
static TAKE_12: [Take; 12] = prefix();
static TAKE_11: [Take; 11] = prefix();
static TAKE_10: [Take; 10] = prefix();
/// The identity over twenty-two operands, for `qkv_fused`'s four arms.
///
/// **`prefix()` here is the WHOLE list and not a prefix of it**, and that is
/// the structural difference between these arms and the five above.
/// `write_kv`'s arms take fifteen of sixteen because the sixteenth is the flag
/// no instantiation declares; these take twenty-two of twenty-two because
/// `rope_table` is a parameter of both instantiations — `qkv_fused.cu:64` and
/// `:77` pass it to `<..., true>` and `<..., false>` alike, and the `false`
/// arm simply never reads it. Nothing is dropped, so nothing has to be covered.
static TAKE_22: [Take; 22] = prefix();

/// This family's specialised rows, which is how [`crate::device::SPECIALISED`]
/// finds them.
///
/// The family owns the list so that specialising a sixth `attn` row is an edit
/// here and nowhere else — `device.rs` names this slice once and never again.
pub static SPECIALISATIONS: &[&Specialisation] = &[
    &WRITE_KV,
    &WRITE_KV_AT_POSITIONS,
    &WRITE_KV_EXPLICIT,
    &WRITE_KV_EXPLICIT_DEVWIN,
    &COPY_KV_CELLS,
    &QKV_DECODE_BLOCK,
    &QKV_DECODE_WARP,
];

/// `qkv_fused.cu:100`, as data — and the first selection in this tree that is
/// not a flag.
///
/// # What the C++ says
///
/// ```text
/// dim3 grid(num_requests, num_q_heads + num_kv_heads);          // :99
/// if (rope_table != nullptr) {
///     qkv_decode_qk_norm_rope_write_kv<BLOCK, true ><<<...>>>(...);  // :101
/// } else {
///     qkv_decode_qk_norm_rope_write_kv<BLOCK, false><<<...>>>(...);  // :126
/// }
/// ```
///
/// `rope_table` is operand 7 of twenty-two, and it is `F32s | null` — which is
/// what makes the `#norope` arm reachable. [`Specialisation::agrees`] refuses
/// a [`Term::Present`] over an operand the row does not declare nullable,
/// because if the binder cannot produce a null there the clause is true for
/// every fire and the second arm is an instantiation that compiles and never
/// runs.
///
/// # Why the two arms are not `Term::Is { value: true / false }`
///
/// There is no `bool` to read. The host tests a POINTER, and the nearest
/// spellable clause — `Term::Aligned { operand: 7, bytes: 16 }` — **holds of
/// address zero**: `0 % 16 == 0`. An alignment clause here would select
/// `USE_ROPE_TABLE = true` for a fire that published no table, and
/// `qkv_fused.cuh:311` would read `rope_table[pos * head_dim + ...]` off a
/// null base. That is the measured hazard `Term::Present`'s doc records, and
/// it is why `Fact::Bool` — which unblocked the five flag arms above — did
/// nothing for these two.
///
/// # Why `flags_are_covered` finds nothing here, correctly
///
/// It collects the operands named by [`Term::Is`] clauses, and a null clause
/// is not one, so this pair is not enumerated. It does not need to be: both
/// instantiations declare the SAME twenty-two parameters as the base, so a
/// fire whose pointer somehow matched neither arm would fall through to a base
/// row that binds exactly what `<BLOCK, false>` declares. The hazard that
/// check exists for — a base binding one cell more than the instantiation
/// reads, which `cuLaunchKernel` accepts and never reports — cannot arise when
/// nothing is dropped. And the two clauses are exhaustive anyway: a pointer is
/// null or it is not.
pub static QKV_DECODE_BLOCK: Specialisation = Specialisation {
    base: "attn::qkv_decode_qk_norm_rope_write_kv",
    arms: &[
        Arm {
            name: "rope",
            when: &[Term::Present { operand: 7, value: true }],
            row: &QKV_FUSED_ROWS[2],
            take: &TAKE_22,
            because: "qkv_fused.cu:100 `if (rope_table != nullptr)` -> \
                      qkv_decode_qk_norm_rope_write_kv<BLOCK, true> at :101",
        },
        Arm {
            name: "norope",
            when: &[Term::Present { operand: 7, value: false }],
            row: &QKV_FUSED_ROWS[3],
            take: &TAKE_22,
            because: "qkv_fused.cu:100 `else` -> \
                      qkv_decode_qk_norm_rope_write_kv<BLOCK, false> at :126",
        },
    ],
};

/// `qkv_fused.cu:56`, as data — the same test, one macro expansion up.
///
/// ```text
/// dim3 warp_grid((total_units + (WARP_BLOCK / 32) - 1) / (WARP_BLOCK / 32));  // :53
/// if (rope_table != nullptr) {
///     qkv_decode_qk_norm_rope_write_kv_warp<(HEAD_DIM_VALUE), true ><<<...>>>(...);  // :57
/// } else {
///     qkv_decode_qk_norm_rope_write_kv_warp<(HEAD_DIM_VALUE), false><<<...>>>(...);  // :70
/// }
/// ```
///
/// `rope_table` is operand 7 here too, and it is operand 7 by coincidence of
/// two argument lists that agree for their first fifteen and part company at
/// the sixteenth — the warp form takes `num_requests` where the block form
/// goes straight to `num_q_heads`. The index is written twice rather than
/// shared for that reason.
///
/// **`HEAD_DIM_VALUE` is not a clause and cannot be one.** The macro is
/// expanded three times, at 64, 128 and 256, under `if (head_dim == …)` at
/// `:81`, `:85` and `:89`. These arms name the 128 expansion; see [`QKV_FUSED`]
/// for why an integer-equality `Term` was refused rather than added, and what
/// the `Term::Multiple` near-miss would cost.
pub static QKV_DECODE_WARP: Specialisation = Specialisation {
    base: "attn::qkv_decode_qk_norm_rope_write_kv_warp_d128",
    arms: &[
        Arm {
            name: "rope",
            when: &[Term::Present { operand: 7, value: true }],
            row: &QKV_FUSED_ROWS[5],
            take: &TAKE_22,
            because: "qkv_fused.cu:56 `if (rope_table != nullptr)` -> \
                      qkv_decode_qk_norm_rope_write_kv_warp<128, true> at :57",
        },
        Arm {
            name: "norope",
            when: &[Term::Present { operand: 7, value: false }],
            row: &QKV_FUSED_ROWS[6],
            take: &TAKE_22,
            because: "qkv_fused.cu:56 `else` -> \
                      qkv_decode_qk_norm_rope_write_kv_warp<128, false> at :70",
        },
    ],
};

/// `kv_paged.cu:84`, as data.
///
/// # Why both arms are mandatory, and what happens if one is missing
///
/// `write_kv<HND_LAYOUT>` takes the SAME fifteen parameters either way. So a
/// fire whose flag matched no arm would fall through to the sixteen-operand
/// base row and bind sixteen cells for a fifteen-parameter kernel —
/// `cuLaunchKernel` reads the parameter count from the cubin, never reads the
/// sixteenth cell, and **succeeds**. No fault, no error, and the wrong
/// instantiation ran. `Specialisation::flags_are_covered` is the check that
/// forbids it: a flag no arm forwards must be covered on BOTH values.
///
/// # Why the negative control is the reason to read this carefully
///
/// Measured on an L40S sm_89 through the shipped fire path, over five shapes
/// and both layouts: 0 of 220,800 bf16 cells differ. Firing `write_kv<false>`
/// where the flag says `true` moved 34,273 of 55,200 cells **while writing the
/// same number of non-zero values** — a permutation, not a truncation. No
/// count, no norm and no tolerance check would flag it, which is why the
/// agreement between these terms and the C++ is a citation rather than an
/// argument.
pub static WRITE_KV: Specialisation = Specialisation {
    base: "attn::write_kv_bf16",
    arms: &[
        Arm {
            name: "hnd",
            when: &[Term::Is { operand: 15, value: true }],
            row: &KV_PAGED_ROWS[4],
            take: &TAKE_15,
            because: "kv_paged.cu:84 `if (hnd_layout)` -> write_kv<true> at :85",
        },
        Arm {
            name: "nhd",
            when: &[Term::Is { operand: 15, value: false }],
            row: &KV_PAGED_ROWS[5],
            take: &TAKE_15,
            because: "kv_paged.cu:84 `else` -> write_kv<false> at :95",
        },
    ],
};

/// `kv_paged.cu:236`, as data. The flag is operand 13 of fourteen.
pub static WRITE_KV_AT_POSITIONS: Specialisation = Specialisation {
    base: "attn::write_kv_at_positions_bf16",
    arms: &[
        Arm {
            name: "hnd",
            when: &[Term::Is { operand: 13, value: true }],
            row: &KV_PAGED_ROWS[7],
            take: &TAKE_13,
            because: "kv_paged.cu:236 `if (layer.hnd_layout)` -> \
                      write_kv_at_positions<true> at :237",
        },
        Arm {
            name: "nhd",
            when: &[Term::Is { operand: 13, value: false }],
            row: &KV_PAGED_ROWS[8],
            take: &TAKE_13,
            because: "kv_paged.cu:236 `else` -> write_kv_at_positions<false> at :246",
        },
    ],
};

/// `kv_paged.cu:371`, as data. The flag is operand 11 of twelve.
pub static WRITE_KV_EXPLICIT: Specialisation = Specialisation {
    base: "attn::write_kv_explicit_bf16_dev",
    arms: &[
        Arm {
            name: "hnd",
            when: &[Term::Is { operand: 11, value: true }],
            row: &KV_PAGED_ROWS[10],
            take: &TAKE_11,
            because: "kv_paged.cu:371 `if (layer.hnd_layout)` -> \
                      write_kv_explicit<true> at :372",
        },
        Arm {
            name: "nhd",
            when: &[Term::Is { operand: 11, value: false }],
            row: &KV_PAGED_ROWS[11],
            take: &TAKE_11,
            because: "kv_paged.cu:371 `else` -> write_kv_explicit<false> at :380",
        },
    ],
};

/// `kv_paged.cu:283`, as data. The flag is operand 12 of thirteen.
pub static WRITE_KV_EXPLICIT_DEVWIN: Specialisation = Specialisation {
    base: "attn::write_kv_explicit_bf16_devwin_dev",
    arms: &[
        Arm {
            name: "hnd",
            when: &[Term::Is { operand: 12, value: true }],
            row: &KV_PAGED_ROWS[13],
            take: &TAKE_12,
            because: "kv_paged.cu:283 `if (layer.hnd_layout)` -> \
                      write_kv_explicit_devwin<true> at :284",
        },
        Arm {
            name: "nhd",
            when: &[Term::Is { operand: 12, value: false }],
            row: &KV_PAGED_ROWS[14],
            take: &TAKE_12,
            because: "kv_paged.cu:283 `else` -> write_kv_explicit_devwin<false> at :292",
        },
    ],
};

/// `kv_paged.cu:418`, as data. The flag is operand 10 of eleven.
pub static COPY_KV_CELLS: Specialisation = Specialisation {
    base: "attn::copy_kv_cells_bf16",
    arms: &[
        Arm {
            name: "hnd",
            when: &[Term::Is { operand: 10, value: true }],
            row: &KV_PAGED_ROWS[16],
            take: &TAKE_10,
            because: "kv_paged.cu:418 `if (layer.hnd_layout)` -> \
                      copy_kv_cells<true> at :419",
        },
        Arm {
            name: "nhd",
            when: &[Term::Is { operand: 10, value: false }],
            row: &KV_PAGED_ROWS[17],
            take: &TAKE_10,
            because: "kv_paged.cu:418 `else` -> copy_kv_cells<false> at :425",
        },
    ],
};

#[cfg(test)]
mod tests {
    use super::SPECIALISATIONS;

    /// Every `#hnd` arm names `write_kv<true>` and every `#nhd` arm names
    /// `write_kv<false>` — and every `#rope` arm names
    /// `qkv_decode_…<…, true>`.
    ///
    /// **The one thing `Specialisation::agrees` cannot check.** It proves the
    /// arms are structurally sound — same rule, same unit, same `Ty` through
    /// the reshape, both flag values covered — and every one of those checks
    /// passes just as well if the two `elem` strings are SWAPPED. What that
    /// costs is measured: firing `write_kv<false>` where the flag says `true`
    /// moved 34,273 of 55,200 cells while writing the same number of non-zero
    /// values. A permutation, not a truncation, so no count and no norm sees
    /// it — which is precisely why the correspondence is asserted here rather
    /// than read off the table by eye.
    ///
    /// The `#rope` pair is the same assertion over a pointer clause, and the
    /// swap costs strictly more there: `USE_ROPE_TABLE = true` with no table
    /// dereferences null, and `= false` with a table recomputes the angle in
    /// `powf`/`__sincosf` — *"different numbers, close, not equal"*, which is
    /// §18's 99.83% shape.
    ///
    /// **`elem` is matched on its SUFFIX for the two `qkv_decode` pairs**,
    /// because their template argument lists carry the head width first:
    /// `device::i32(128), true`. The flag is still the last argument and still
    /// the whole difference between the two arms, so the check is the same one
    /// — it just cannot be an equality against a bare tag type.
    #[test]
    fn each_arm_names_the_instantiation_its_name_claims() {
        let mut checked = 0;
        for spec in SPECIALISATIONS {
            assert_eq!(spec.arms.len(), 2, "{}: two selections, two arms", spec.base);
            for arm in spec.arms {
                // (symbol suffix, `elem` tail, the template argument cited)
                let want = match arm.name {
                    "hnd" => ("#hnd", "device::true_type::value", "true"),
                    "nhd" => ("#nhd", "device::false_type::value", "false"),
                    "rope" => ("#rope", "true", "true"),
                    "norope" => ("#norope", "false", "false"),
                    other => panic!("{}: unknown arm `{other}`", spec.base),
                };
                assert!(
                    arm.row.elem.ends_with(want.1),
                    "{} arm `{}` instantiates <{}>",
                    spec.base,
                    arm.name,
                    arm.row.elem,
                );
                // `false` is a suffix of nothing else, but `true` IS a suffix
                // of `device::true_type::value` — so the flag arms are pinned
                // to the exact string as well, and only the two `qkv_decode`
                // pairs get the looser test.
                if want.1.starts_with("device::") {
                    assert_eq!(arm.row.elem, want.1, "{} arm `{}`", spec.base, arm.name);
                }
                assert!(
                    arm.row.sig.symbol.ends_with(want.0),
                    "{} arm `{}` fires {}",
                    spec.base,
                    arm.name,
                    arm.row.sig.symbol,
                );
                assert!(
                    arm.because.contains(&format!("{}>", want.2)),
                    "{} arm `{}` cites {}",
                    spec.base,
                    arm.name,
                    arm.because,
                );
                assert!(
                    arm.because.starts_with("kv_paged.cu:")
                        || arm.because.starts_with("qkv_fused.cu:"),
                    "a rule with no cited launcher is a guess: {}",
                    arm.because,
                );
                checked += 1;
            }
        }
        assert_eq!(checked, 14, "seven specialised kernels, two arms each");
    }
}

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
// `attn/attention_mla_naive` — the Blackwell MLA fallback, now a unit.
//
// The refusal recorded above ("`attention_mla_naive.cu` keeps its
// `cudaFuncSetAttribute` opt-in to 200 KB of shared memory behind a
// `std::call_once` -- host state no `LaunchRule` can carry -- so it is not
// split either") is CLOSED, and it was wrong on both halves:
//
//  * The opt-in is not host state a rule has to carry. `runtime::module`'s
//    `raise_dynamic_smem_cap` performs it inside `KernelModule::fire`, once
//    per `(CUdevice, CUfunction)` above a 48 KiB high-water mark, driven by
//    `Launch::smem` and nothing else. The north star §5 step 1 `smem_opt_in`
//    field is the author's side of that same fact.
//  * The 200 KiB was never needed. `attention_mla_naive.cuh:251`'s allocation
//    is `(8 * CKV + 16) * 4` and `:228` refuses `CKV > 512`, so the largest
//    request the scalar kernel can make is 16 448 bytes — a third of the
//    48 KiB default. The measurement is preserved in
//    `driver-cuda/src/fire/mla_naive.rs::NAIVE_OPT_IN_BYTES_UNREACHED` with
//    that arithmetic; the tensor-core kernel's 100 032 IS above the default
//    and IS raised.
//
// The real blocker was neither: the file was MIXED. Two `__global__`s and
// four host functions in one header, opening `<mutex>`, `<stdexcept>`,
// `<string>` and `<cuda_runtime.h>`, so it could not be a unit root at all.
// The host half now lives in `kernels-cuda/csrc/src/attn/attention_mla.cu`
// (and in Rust at `driver-cuda/src/fire/mla_naive.rs`), and what is left
// compiles.
//
// PROBED, NVRTC 13.0, `sm_89`, under this crate's own numerics contract
// (`--fmad=false --prec-div=true --prec-sqrt=true`) and the CARRIED header
// set only — `csrc/{src,shim,vendor}`, no toolkit include path:
//
// ```text
//   rc = 0, 0 errors
//   117 621 bytes of PTX, 2 .entry
//     _ZN15pie_cuda_driver7kernels4attn9mla_naive22mla_naive_paged_kernelE...
//     _ZN15pie_cuda_driver7kernels4attn9mla_naive10mma_detail20mla_mma_paged_kernelE...
// ```
//
// It needed three new shim headers and they are measured, not assumed: the
// same text compiled with `/usr/local/cuda/include` answering
// `cuda_pipeline.h`, `math_constants.h` and `cstring` produced **byte-identical
// PTX**, register allocation included. See `csrc/shim/cuda_pipeline.h`, which
// carries the comparison and the one PTX operand it turned on.
//
// A FOURTH FINDING, from the same probe: the file called `std::memcpy` and
// never included `<cstring>`. Under nvcc `<cuda_runtime.h>` supplied it
// transitively; under NVRTC it is an error no include path can fix, because
// the include was never written. The set nvcc accepted was not the set the
// file declared.
// ===========================================================================

/// The naive/Blackwell paged MLA pair: a scalar kernel and a tensor-core one.
///
/// **Two rows, one root, and they are ALTERNATIVES rather than a sequence.**
/// The C++ launcher tries the tensor-core kernel first
/// (`attention_mla_naive.cuh:218`) and falls through to the scalar one; the
/// host program that replaces it, [`driver-cuda`'s `fire::mla_naive`], plans
/// one or the other and fires exactly once. Nothing composes them, so there
/// is no `execution::Step` and no intermediate buffer.
///
/// # Why this pair exists at all
///
/// `attention_mla.cu:150-157`, which is the only place it is argued and must
/// not be lost with the file:
///
/// > FlashInfer's FA2 `BatchMLAPagedAttention` (a cooperative kernel) produces
/// > zero output on sm_100; the ecosystem (sglang/vllm) routes Blackwell MLA
/// > to trtllm/cutlass/ragged kernels instead. This is a correctness-first,
/// > arch-agnostic latent-space MLA: one block per (token, head), flash-style
/// > online softmax over the paged ckv/kpe cache. Output is in the kv_lora
/// > latent space (same as the FA2 path), so the rest of the MLA forward
/// > (latent_to_v, o_proj) is unchanged.
///
/// The selector is a device query — `cudaDevAttrComputeCapabilityMajor >= 10`
/// at `attention_mla.cu:56-62` — and it is NOT one of these rows' business:
/// it chooses between this pair and FlashInfer's MLA, which is a different
/// symbol in a different unit that does not exist yet.
///
/// # Options: none, and the `#ifndef` defaults are why
///
/// `PIE_MLA_MMA_BK`, `_WARPS`, `_STAGES` and `_MINBLK` are all `#ifndef`
/// guarded with their defaults in the header (`:302-322`), so the unit needs
/// no `-D` to compile at the shape everything currently runs. Putting them in
/// `Unit::options` would be the hook that file's own doc warns against: they
/// are not options this unit needs and the others must not have, they are
/// tuning constants with one live value, and `Unit::cache_key` spanning them
/// would make a cubin cache key out of a number nobody varies. If a second
/// tile is ever wanted it is a second unit with a second root, the way
/// `XQA_LATTICE` spells its six.
pub const MLA_NAIVE: Unit = Unit {
    name: "attn/attention_mla_naive",
    root: include_str!("../../csrc/src/attn/attention_mla_naive.cuh"),
    rows: MLA_NAIVE_ROWS,
    options: &[],
};

/// The two `__global__`s, by their qualified paths.
///
/// Both are `DeviceKernel::PLAIN`: neither has a template parameter list, so
/// there is nothing for `elem` to pick and the bare qualified path is what
/// NVRTC lowers and `cuModuleGetFunction` resolves. `MLA_PAGED_ROWS[0]` is the
/// same case and makes the argument at length.
///
/// The paths are two levels deeper than most of this file's, and that is the
/// header's own nesting rather than a convention: the pair lives in
/// `pie_cuda_driver::kernels::attn::mla_naive`, and the tensor-core kernel is
/// inside a further `mma_detail` that also holds its `ld_a`/`ld_b_v`/`mma_m16n8k16`
/// helpers.
static MLA_NAIVE_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &MLA_NAIVE_SIGS[0],
        template_path: "attn::mla_naive::mla_naive_paged_kernel",
        elem: DeviceKernel::PLAIN,
    },
    DeviceKernel {
        sig: &MLA_NAIVE_SIGS[1],
        template_path: "attn::mla_naive::mma_detail::mla_mma_paged_kernel",
        elem: DeviceKernel::PLAIN,
    },
];

/// The two contracts, which are the kernels' parameters and not the launcher's.
///
/// **Both are `LaunchRule::Unstated`, and the two rectangles are why.**
///
/// ```text
/// attention_mla_naive.cuh:265   dim3 grid(total_tokens, num_heads / G);      block 256
/// attention_mla_naive.cuh:725   dim3 grid(num_heads / kBM, total_tokens);    block 256
/// ```
///
/// Same block, and the grids are TRANSPOSES of each other — tokens on x for
/// the scalar kernel, tokens on y for the tensor-core one. No rule states
/// either, and a rule that stated one would be actively wrong for the other
/// while looking right: `grid.y` is capped at 65 535 and `grid.x` is not, so
/// the transpose decides which of tokens and head blocks may exceed 65 535.
/// Both are built by the driver in `fire::mla_naive::plan`, which is
/// `MLA_PAGED`'s `mla_prepare` arrangement for the same reason.
///
/// The scalar kernel's `G` is the case `execution::Control::Supplies`' own doc
/// names — *"passed to the kernel AND divides the head axis of the grid"* —
/// and it is not merely unstated but UNSTATEABLE by a formula: `:241-249`
/// SEARCHES for it, halving from 8 until the grid reaches `kMlaWaveTarget =
/// 296` blocks. A rule computes; this looks.
///
/// **Unsourced on every operand**, which is `MLA_PAGED_SIGS`' §60.7 case: the
/// rows exist so the unit can be enrolled and the symbols resolved, and
/// `crate::abi` skips a row with any `Source::Unbound` operand whole, so no
/// dispatch arm is generated and nothing reaches them except
/// `fire::mla_naive` by name through `hand::fire`.
///
/// **No `_bf16` suffix on either**, for `attn::write_mla`'s reason: a format
/// suffix claims a choice, and there is no template parameter here to have
/// chosen with — every buffer is `__nv_bfloat16` in the kernels' own
/// declarations.
static MLA_NAIVE_SIGS: [KernelSig; 2] = [
    // `attention_mla_naive.cuh:66-78` — nineteen parameters, ending in `G`.
    //
    // `index_mask` is nullable and `attn/attention_mla.hpp:36-38` says what
    // null means and what the non-null case is restricted to:
    //
    // > DSA top-k mask for the naive path: [num_query_tokens, mask_stride]
    // > uint8 (1=attend). Applied to in-batch keys (j < mask_stride). Null =
    // > dense. Only valid for single-request pure prefill (key j == batch
    // > token j).
    //
    // That last sentence is a correctness precondition no type states, so it
    // travels here and in `fire::mla_naive::NaivePtrs::index_mask`.
    kernel!(mla_naive_paged "attn::mla_naive_paged",
        file = Some("attn/attention_mla_naive.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            q_nope: Buf, q_pe: Buf, ckv_pages: Buf, kpe_pages: Buf,
            qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            kv_last_page_lens: U32s, o: BufMut,
            index_mask: U8s | null, index_mask_stride: I32,
            r: I32, h: I32, ckv: I32, kpe: I32, page_size: I32,
            sm_scale: F32, causal: Bool, g: I32,
        ]),
    // `attention_mla_naive.cuh:420-431` — sixteen parameters, and the three
    // MISSING ones are the contract's most informative feature: there is no
    // `ckv`, no `kpe` and no `G`. The tensor-core kernel is compiled AGAINST
    // `kCkv = 512` and `kKpe = 64` (`:330-331`) because the `mma.sync`
    // fragment shapes are written for them, and its head group is fixed at
    // `kBM = 16` (`:324`). That is exactly why `mla_mma_supported` (`:698`)
    // COMPARES those three rather than forwarding them: the predicate is the
    // only place the shape is checked, and passing them would imply a
    // generality the `ld_b_v`/`ld_a` offsets do not have.
    kernel!(mla_mma_paged "attn::mla_mma_paged",
        file = Some("attn/attention_mla_naive.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            q_nope: Buf, q_pe: Buf, ckv_pages: Buf, kpe_pages: Buf,
            qo_indptr: U32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            kv_last_page_lens: U32s, o: BufMut,
            index_mask: U8s | null, index_mask_stride: I32,
            r: I32, h: I32, page_size: I32, sm_scale: F32, causal: Bool,
        ]),
];
