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
//!   a PARTIAL split of a file that keeps its FlashInfer dispatch and its
//!   three private score-normalisation kernels — so the row is now refused
//!   for GEOMETRY like the rest of this list. The launcher is
//!   `attention_flashinfer.cu:828-829`, `dim3(requests, 64)` at 256 threads
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
//!   arithmetic and not its INPUTS — see [`SPLIT_PACKED_SIGS`], which is the
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
//! * `attn/pack_dense_mask.cu` takes a `StructuredMaskParams` defined in its
//!   host `.hpp`, which NVRTC cannot include. The header carries a device
//!   MIRROR of that struct, and the duplication is checked rather than
//!   trusted: FIVE `static_assert`s in `pack_dense_mask.cu` pin `sizeof`,
//!   `alignof` and the offset of all three fields (`kind`, `window`, `sink`)
//!   against the host type. A negative control transposing two fields fired
//!   exactly the two offset assertions and nothing else, which is what makes
//!   this a check and not a comment. The obstacle that remains is neither of
//!   the two above — see the next section.
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
//!   never the blocker; the spelling was. See [`PACK_DENSE_MASK`].
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
//! # The heavy half has seven, and the list reaches exactly one of them
//!
//! [`UNITS_HEAVY`]'s three headers hold every multi-argument template in the
//! family, and auditing them against the finding produced one row change and
//! six refusals with a shared cause worth naming once. **Five of those six
//! refusals have since been overturned** — not by the argument-list finding
//! but by [`crate::device::Specialisation`], which arrived after it.
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
//! numeric format cost the line below rather than a translation unit's worth
//! of `cicc` — which is the measurement `norm/elementwise.cuh` made first and
//! the reason this design was worth the migration.

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

/// `attn`'s logit cap, at both numeric formats.
pub const SOFTCAP: Unit = Unit {
    name: "attn/softcap",
    root: include_str!("../../csrc/src/attn/softcap.cuh"),
    rows: SOFTCAP_ROWS,
    options: &[],
};

/// The attention-sink pair, both rows: the log2→ln rebase and the per-head
/// rescale that reads it.
pub const ATTN_SINK: Unit = Unit {
    name: "attn/attn_sink",
    root: include_str!("../../csrc/src/attn/attn_sink.cuh"),
    rows: ATTN_SINK_ROWS,
    options: &[],
};

/// K3's residual-block blend.
pub const ATTN_RES: Unit = Unit {
    name: "attn/attn_res",
    root: include_str!("../../csrc/src/attn/attn_res.cuh"),
    rows: ATTN_RES_ROWS,
    options: &[],
};

/// glm5's sparse-attention index network. One row of three: the other two
/// want a block sized on a head count and a shared-memory budget sized on the
/// sequence.
pub const DSA_INDEXER: Unit = Unit {
    name: "attn/dsa_indexer",
    root: include_str!("../../csrc/src/attn/dsa_indexer.cuh"),
    rows: DSA_INDEXER_ROWS,
    options: &[],
};

/// The reference attention kernels and MTP's hidden-state plumbing. Two rows
/// of five; the `.cuh` names the obstacle for each of the other three.
pub const ATTENTION_NAIVE: Unit = Unit {
    name: "attn/attention_naive",
    root: include_str!("../../csrc/src/attn/attention_naive.cuh"),
    rows: ATTENTION_NAIVE_ROWS,
    options: &[],
};

/// flashinfer's supported head widths, reached by padding — and reached
/// back out of by stripping.
///
/// A unit that could not exist until [`LaunchRule::PerHead`] did. Both
/// kernels were extracted, proved NVRTC-clean and left rowless, and a rowless
/// unit is refused rather than compiled: `every_unit_compiles_and_every_row_
/// resolves` asserts `!unit.rows.is_empty()`, because a cubin nothing can
/// fire is one cached per architecture for nobody.
pub const HEAD_DIM_PAD: Unit = Unit {
    name: "attn/head_dim_pad",
    root: include_str!("../../csrc/src/attn/head_dim_pad.cuh"),
    rows: HEAD_DIM_PAD_ROWS,
    options: &[],
};

/// The fused QKV product, taken apart into three packed operands.
///
/// One row of the header's two, and the missing one is not a missing rule —
/// see [`SPLIT_PACKED_SIGS`]. Rowless until [`LaunchRule::SplitPacked`]
/// landed, and a unit for the same reason [`HEAD_DIM_PAD`] is.
pub const SPLIT_PACKED: Unit = Unit {
    name: "attn/split_packed",
    root: include_str!("../../csrc/src/attn/split_packed.cuh"),
    rows: SPLIT_PACKED_ROWS,
    options: &[],
};

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

/// Two formats of one template.
pub static SOFTCAP_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &SOFTCAP_SIGS[0],
        template_path: "attn::device::logit_softcap",
        elem: "device::bf16",
    },
    // The row the ahead-of-time build never had. See the module header.
    DeviceKernel {
        sig: &SOFTCAP_SIGS[1],
        template_path: "attn::device::logit_softcap",
        elem: "device::f16",
    },
];

#[rustfmt::skip]
static SOFTCAP_SIGS: [KernelSig; 2] = [
    // Caps the logits WHERE THEY LIE — one buffer, no destination, which is
    // what `in_place` states and what `Buffers::assign` was already relying
    // on. `Elementwise` IS the launcher this replaces: `(n + 255) / 256`
    // blocks of 256, and an empty `n` refused rather than launched.
    //
    // `cap` is `CtxNonZero`, so a model that states no cap does not bind
    // this kernel at all — the `!(cap > 0.f)` half of the launcher's guard,
    // moved to the only place that can answer it before a launch exists.
    // The reciprocal the launcher used to compute moved INTO the kernel and
    // is the same bits; `attn/softcap.cuh` records why.
    kernel!(logit_softcap "attn::logit_softcap_bf16",
        file = Some("attn/softcap.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            cap: F32 <- Source::CtxNonZero("final_logit_softcap"),
            n: Usize <- Source::OutElements(0),
        ]),
    // A DIFFERENT symbol for a different numeric format, because a symbol is
    // what a text states and a text that says `softcap` must not get to
    // choose its own precision. Every other word is the row above's.
    kernel!(logit_softcap_f16 "attn::logit_softcap_f16",
        file = Some("attn/softcap.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            cap: F32 <- Source::CtxNonZero("final_logit_softcap"),
            n: Usize <- Source::OutElements(0),
        ]),
];

/// Both of the templates `attn/attn_sink.cuh` holds.
pub static ATTN_SINK_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &ATTN_SINK_SIGS[0],
        template_path: "attn::device::lse_log2_to_ln",
        // NOT `device::f32` — the prelude has no such alias, and `Elem` has no
        // `float` specialisation to hang one on. `attn::device::f32` is declared
        // in the `.cuh` beside the kernel that is the only thing asking for it.
        elem: "attn::device::f32",
    },
    DeviceKernel {
        sig: &ATTN_SINK_SIGS[1],
        template_path: "attn::device::attn_sink_rescale",
        elem: "device::bf16",
    },
];

#[rustfmt::skip]
static ATTN_SINK_SIGS: [KernelSig; 2] = [
    // FlashInfer publishes its LSE in log2 and the sink correction works in
    // ln. A unit conversion, stated so a reader never has to guess which base
    // an LSE is in — and the drift it prevents is measured: without it the
    // sigmoid argument was off by 0.693, which matched HF's top-1 on most
    // prompts and then degenerated greedy decoding after a few steps.
    //
    // The rebase is in place on the value it names: `Out(0)` is the
    // statement's result and `In(0)` is the same buffer, so the element count
    // is the result's own extent. `n` is `Usize` where the twin said `I32`,
    // because the kernel's parameter is `device::usize` — the twin's `int`
    // was the launcher's signature, not the kernel's.
    kernel!(lse_log2_to_ln "attn::lse_log2_to_ln",
        file = Some("attn/attn_sink.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            lse: F32sMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
        ]),
    // `PerHeadElementwise`, and this launcher is the one the rule was derived
    // from: `dim3 grid(N, num_q_heads)` with
    // `block = (head_dim < 32) ? 32 : (head_dim > 128 ? 128 : head_dim)`,
    // which is `eval`'s `[rows, q_heads, 1]` and `clamp(head_dim, 32, 128)`
    // to the digit. The ROW is `grid.x` and the head is `grid.y` here, which
    // is the transpose of `PerHead`'s — the two axis orders are the kernels'
    // and not a convention, and a rule read off the wrong one runs the same
    // block count over the wrong cells.
    //
    // `q_heads` and not `kv_heads`, because the tensor this rescales is the
    // attention OUTPUT: one row per query head. The rule reads
    // `Dims::q_heads` for exactly that reason and a grouped-query fire has
    // two head counts to pick the wrong one from.
    //
    // `N` and `num_q_heads` stay operands though the rule recovers both.
    // They are the kernel's own `if (t >= N || h >= num_q_heads) return;`
    // and its row stride `num_q_heads * head_dim`; an operand list shorter
    // than the `__global__`'s parameter list is a `void**` the driver reads
    // past. What left is the stream, which was never one.
    //
    // In place on the output it corrects, which is what lets the o_proj GEMM
    // and the residual add downstream read rescaled activations without a
    // copy. `lse` is the dispatch's SECOND result — `In(1)`, a value only a
    // sink layer declares — and the sinks are the layer's learned weight.
    kernel!(attention_sink_rescale "attn::attention_sink_rescale_bf16",
        file = Some("attn/attn_sink.cuh"),
        launch = LaunchRule::PerHeadElementwise,
        in_place = &[(0, 0)],
        operands = operands![
            o: BufMut <- Source::Out(0),
            lse: F32s <- Source::In(1),
            sinks: Buf <- Source::Weight(0),
            N: I32 <- Source::Rows,
            num_q_heads: I32 <- Source::Ctx("num_q_heads"),
            head_dim: I32 <- Source::Ctx("head_dim"),
        ]),
];

/// K3's blend, at bf16.
pub static ATTN_RES_ROWS: &[DeviceKernel] = &[DeviceKernel {
    sig: &ATTN_RES_SIGS[0],
    template_path: "attn::device::attn_res_blend",
    elem: "device::bf16",
}];

#[rustfmt::skip]
static ATTN_RES_SIGS: [KernelSig; 1] = [
    // `Rms` — one block per token, 256 wide, the row width read by a stride
    // loop. The rule also hands the kernel 32 bytes of dynamic shared memory
    // for `block_sum`'s warp scratch, which this kernel does not use: its
    // reduction is a static `__shared__`. A rule per unused allocation is not
    // a trade worth making.
    //
    // `T` is gone from the operand list where the twin states it. It did two
    // jobs: a bound check, which is now the rule's grid promise, and a block
    // stride, which survives as `block_rows`. The launcher's
    // `block_rows > 0 ? block_rows : T` default is `Source::Rows`, which is
    // the value that ternary produced on every call site that existed.
    kernel!(attn_res_blend "attn::attn_res_blend_bf16",
        file = Some("attn/attn_res.cuh"),
        launch = LaunchRule::PerRow,
        // `PerRow`, not `Rms`. The launcher is `<<<T, kThreads=256, 0>>>` in
        // `attn/attn_res.cu`, and `Rms` requests thirty-two bytes of dynamic
        // shared memory that no launcher here passes and no kernel here
        // reads -- `block_sum`'s warp buffer, which this shape has no
        // reduction to need. Harmless in effect and wrong as a contract:
        // a rule is meant to REPRODUCE its launcher, and one that asks
        // for memory the launcher did not is a rule nobody can check
        // against the `<<<>>>` it came from.
        operands = operands![
            prefix: Buf <- Source::In(0),
            blocks: Buf <- Source::In(1),
            norm_weight: Buf <- Source::In(2),
            proj_weight: Buf <- Source::In(3),
            out: BufMut <- Source::Out(0),
            // AN OPERAND OVER AN OPERAND, not a plan dimension. `B` is how
            // many blocks the packed input holds, which is its width divided
            // by the output's — the two are in the same statement, so the row
            // can say it. A row that guessed a param would launch the right
            // kernel over the wrong rectangle.
            B: I32 <- Source::Div(&Source::Width(&Source::In(1)), &Source::Width(&Source::Out(0))),
            H: I32 <- Source::OutWidth(0),
            block_rows: I32 <- Source::Rows,
            eps: F32 <- Source::Ctx("eps"),
        ]),
];

/// The index network's LayerNorm-plus-RoPE and its top-k mask, at bf16.
pub static DSA_INDEXER_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &DSA_INDEXER_SIGS[0],
        template_path: "attn::device::index_knorm_rope",
        elem: "device::bf16",
    },
    // `template <class T>` and nothing else — `kBlock` is a file-scope
    // `constexpr int` the kernel strides by, not a template argument, so
    // there is no non-type argument to cite and the 256 the rule launches
    // has to agree with `dsa_indexer.cuh`'s `kBlock` instead. It does.
    DeviceKernel {
        sig: &DSA_INDEXER_SIGS[1],
        template_path: "attn::device::index_topk_mask",
        elem: "device::bf16",
    },
];

#[rustfmt::skip]
static DSA_INDEXER_SIGS: [KernelSig; 2] = [
    // UNSOURCED, exactly as the twin is. A source is a claim about where a
    // value comes from at fire time, and a guessed one binds the wrong buffer
    // with nothing to report it — so the sources arrive when the statement
    // that fires this kernel does, not before.
    //
    // `tokens` is gone: `Rms` opens one block per row and that IS `tokens`.
    // `head_dim` stays, because the kernel strides over it and no rule the
    // grid states recovers a row's width for a kernel whose row is one head.
    kernel!(dsa_index_knorm_rope "attn::dsa_index_knorm_rope_bf16",
        file = Some("attn/dsa_indexer.cuh"),
        launch = LaunchRule::PerRow,
        // `PerRow`, not `Rms`. The launcher is `<<<tokens, kBlock=256, 0>>>` in
        // `attn/dsa_indexer.cu`, and `Rms` requests thirty-two bytes of dynamic
        // shared memory that no launcher here passes and no kernel here
        // reads -- `block_sum`'s warp buffer, which this shape has no
        // reduction to need. Harmless in effect and wrong as a contract:
        // a rule is meant to REPRODUCE its launcher, and one that asks
        // for memory the launcher did not is a rule nobody can check
        // against the `<<<>>>` it came from.
        operands = operands![
            idx_k: BufMut, k_norm_weight: Buf, k_norm_bias: Buf, positions: I32s,
            head_dim: I32, rope_dim: I32, theta: F32, eps: F32,
        ]),
    // THE CAUSAL TOP-K MASK, and the row `dsa_indexer.cuh` said no rule
    // stated. `LaunchRule::RowScores` is that rule, and
    // `runtime::launch::row_scores` was ported FROM this launcher:
    //
    // ```text
    // if (tokens <= 0) return;
    // const std::size_t smem = static_cast<std::size_t>(tokens) * sizeof(float);
    // device::index_topk_mask<bf16><<<tokens, device::kBlock, smem, stream>>>(...);
    // ```
    //
    // `kBlock` is 256 and `RowScores` launches 256, so grid, block AND the
    // shared allocation agree for every rectangle: `rows * 4` bytes is
    // `tokens * sizeof(float)` written twice.
    //
    // **The shared allocation is why this is not `Rms` and not `PerRow`.**
    // The kernel declares `extern __shared__ float logit[]` and fills
    // `logit[0..nkeys)` where `nkeys = blockIdx.x + 1` — one float per KEY,
    // and every key of this fire is a row of it. At `Rms`' thirty-two bytes
    // the last row of a 4 096-token prefill would select its top-k from eight
    // floats it wrote and 4 088 it did not; at `PerRow`'s zero, from none.
    // Neither faults. `dsa_indexer.cuh` states the consequence in its own
    // words — *"a launch that under-sizes shared memory does not fail, it
    // reads another block's floats"* — and that is a wrong mask, which is a
    // wrong attention, which nothing downstream checks.
    //
    // `N` STAYS AN OPERAND although `RowScores` opens the grid over it. The
    // rule recovers the GRID; the kernel needs the number a second time, as
    // the pitch of `mask` (`mrow = mask + i * N`) and as the bound of its
    // causal zero-fill. An extent a rule recovers is not an operand — an
    // extent a kernel ADDRESSES with is.
    //
    // Unsourced, exactly as the twin is: `n_heads`, `head_dim` and `topk`
    // reach the ahead-of-time row from `Source::Param`, and a JIT row that
    // guessed which statement parameter carried which would bind three
    // integers in an order nothing reports.
    kernel!(dsa_index_topk_mask "attn::dsa_index_topk_mask",
        file = Some("attn/dsa_indexer.cuh"),
        launch = LaunchRule::RowScores,
        whole = true,
        operands = operands![
            idx_q: Buf,
            idx_k: Buf,
            idx_w: Buf,
            mask: U8sMut,
            n: I32,
            n_heads: I32,
            head_dim: I32,
            topk: I32,
        ]),
];

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
    kernel!(mtp_shift_hidden "attn::mtp_shift_hidden_bf16",
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
    kernel!(mtp_update_pending_hidden "attn::mtp_update_pending_hidden_bf16",
        file = Some("attn/attention_naive.cuh"),
        launch = LaunchRule::PerRequest,
        whole = true,
        operands = operands![
            target_hidden: Buf, pending_hidden: BufMut, qo_indptr: U32s,
            slot_ids: I32s, num_requests: I32, hidden_size: I32,
        ]),
];

/// `attn/head_dim_pad.cuh`'s two instantiations.
static HEAD_DIM_PAD_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &HEAD_DIM_PAD_SIGS[0],
        template_path: "attn::device::pad_head_dim",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &HEAD_DIM_PAD_SIGS[1],
        template_path: "attn::device::strip_head_dim",
        elem: "device::bf16",
    },
];

/// The contracts, in [`HEAD_DIM_PAD_ROWS`]' order.
///
/// Both are `LaunchRule::PerHead`, and `head_dim_pad.cu` is the launcher
/// `runtime::launch::per_head` cites — `dim3 grid(num_heads, num_tokens)`,
/// `dim3 block(kPadBlock)`, which `eval` returns as `[kv_heads, rows, 1]` and
/// `[128, 1, 1]`. **The head is `grid.x` and the row is `grid.y`**, the
/// transpose of every other head-shaped rule in the vocabulary, because that
/// is the axis order these two kernels read.
///
/// The 128 is the kernels' requirement and not a tuning number: both stride
/// `d += kPadBlock`, the compile-time constant, so a narrower block never
/// visits the columns above it — which for `pad_head_dim` is padding that was
/// never zeroed, and for `strip_head_dim` a head whose tail keeps whatever the
/// destination held. Neither fails; both answer.
///
/// `num_tokens` and the stream leave both rows: the first is `grid.y` and the
/// second was never an operand. Everything the `__global__` declares stays,
/// including `num_heads` — the rule puts the count on an axis the kernels do
/// not read it back from, so a row without it is a `void**` one entry short.
///
/// Which side is PACKED is whichever end is `head_dim` wide — the input on
/// the way in, the output on the way out — so the head count divides out of
/// the packed side and the padded width is the other side over that count.
/// Both readings are the ahead-of-time rows', kept verbatim.
#[rustfmt::skip]
static HEAD_DIM_PAD_SIGS: [KernelSig; 2] = [
    kernel!(pad_head_dim "attn::pad_head_dim_bf16",
        file = Some("attn/head_dim_pad.cuh"),
        launch = LaunchRule::PerHead,
        operands = operands![
            packed: Buf <- Source::In(0),
            padded: BufMut <- Source::Out(0),
            num_heads: I32 <- PACKED_HEADS_IN,
            head_dim: I32 <- Source::CtxNonZero("head_dim"),
            head_dim_padded: I32 <- Source::Div(
                &Source::Width(&Source::Out(0)),
                &PACKED_HEADS_IN,
            ),
        ]),
    kernel!(strip_head_dim "attn::strip_head_dim_bf16",
        file = Some("attn/head_dim_pad.cuh"),
        launch = LaunchRule::PerHead,
        operands = operands![
            padded: Buf <- Source::In(0),
            packed: BufMut <- Source::Out(0),
            num_heads: I32 <- PACKED_HEADS_OUT,
            head_dim: I32 <- Source::CtxNonZero("head_dim"),
            head_dim_padded: I32 <- Source::Div(
                &Source::Width(&Source::In(0)),
                &PACKED_HEADS_OUT,
            ),
        ]),
];

/// The heads of whichever side of a pad is `head_dim` wide.
///
/// Two constants rather than one expression written twice, because the pad
/// and the strip read them off OPPOSITE ends and a copy that drifted would
/// count heads on the padded side — where the divisor is `head_dim_padded`,
/// so the count comes out short and the launch covers a prefix of the heads.
const PACKED_HEADS_IN: Source =
    Source::Div(&Source::Width(&Source::In(0)), &Source::CtxNonZero("head_dim"));
/// See [`PACKED_HEADS_IN`].
const PACKED_HEADS_OUT: Source =
    Source::Div(&Source::Width(&Source::Out(0)), &Source::CtxNonZero("head_dim"));

/// `attn/split_packed.cuh`'s one instantiation.
static SPLIT_PACKED_ROWS: &[DeviceKernel] = &[DeviceKernel {
    sig: &SPLIT_PACKED_SIGS[0],
    template_path: "attn::device::split_qkv",
    elem: "device::bf16",
}];

/// The contract, and the twin that does not get one.
///
/// `SplitPacked`, whose arithmetic `runtime::launch` derived from this very
/// launcher: `dim3 grid(xblocks, n_tokens)` at 256 threads. The rule's
/// `grid.x` is deliberately WIDER — `ceil(in_width / 256)` over the packed
/// `q_dim + 2 * kv_dim` where the launcher used `ceil(max(q_dim, kv_dim) /
/// 256)` — and `split_packed.cuh` licensed that before the port existed:
/// *"every loop below strides by `blockDim.x * gridDim.x` and bounds itself
/// on its own output width, so extra blocks contribute nothing but a shorter
/// loop."* Wider is safe in this direction and only this one; a grid narrower
/// than an output leaves the tail of every row unwritten.
///
/// **`split_qkv_devwin` has no row, and its geometry is not why.** The rule
/// computes the same `dim3(xblocks, rows)` for it — the launcher is cited
/// beside this one — but it computes it from the wrong numbers, twice over:
///
/// * `grid.y` is `n_max`, which the ahead-of-time row sources from
///   `Ctx("rows_total")` — the FIRE's lane count — while a rule reads
///   `Dims::rows`, the statement's rectangle. The two are equal until a peel
///   splits, and a peel's tail region is the only place this symbol is ever
///   stated. Under the split, `grid.y` would be the tail's length while the
///   kernel compares an ABSOLUTE `blockIdx.y` against the device window, so
///   the rows past that length are never visited and Q, K and V keep the
///   previous fire's bytes there.
/// * Its buffers are BASE pointers by contract — the window lives in device
///   memory so a captured graph can replay across row splits without
///   re-recording — and the JIT binder resolves `In`/`Out` THROUGH the
///   statement's window. The kernel would window pointers the binder had
///   already offset, which is the double-window the `.cuh` names.
///
/// Neither is something a `LaunchRule` can carry, so the row is absent rather
/// than approximate and the launcher in `split_packed.cu` stays.
#[rustfmt::skip]
static SPLIT_PACKED_SIGS: [KernelSig; 1] = [
    // `n_tokens` is the rule's `grid.y` and the stream was never an operand:
    // eight arguments become six. The two widths come off what is WRITTEN and
    // not off the packed operand — a `[N, q + 2 * kv]` row cannot say where
    // the cut falls, and both results can.
    kernel!(split_qkv "attn::split_qkv_bf16",
        file = Some("attn/split_packed.cuh"),
        launch = LaunchRule::SplitPacked,
        operands = operands![
            packed: Buf <- Source::In(0),
            q_out: BufMut <- Source::Out(0),
            k_out: BufMut <- Source::Out(1),
            v_out: BufMut <- Source::Out(2),
            q_dim: I32 <- Source::OutWidth(0),
            kv_dim: I32 <- Source::OutWidth(1),
        ]),
];

/// The custom-mask packers: a dense byte-per-cell mask packed to
/// FlashInfer's bitmap ABI, and the same ABI materialised straight out of a
/// causal / sliding-window / sink descriptor.
///
/// **The unit that could not exist while `instantiation()` could only spell
/// `path<...>`.** Both kernels are plain `__global__`s — every buffer is
/// `u8`/`u32`/`i32` mask metadata and the block width reaches them as
/// `blockDim.x` — so neither has a type or a compile-time value to abstract
/// over, and `pack_dense_mask.cuh` refused to invent one on
/// `mxfp4_marlin.cuh`'s precedent: *"a width parameter would be a lie that
/// compiles."* That header named the other way out in the same sentence —
/// *"the rows wait on a real argument or on an `instantiation()` that needs
/// none"* — and [`DeviceKernel::PLAIN`] is that, measured rather than
/// assumed. **No device text changed for these two rows.**
///
/// The geometry was never the blocker and the header says so: both launch
/// `<<<B, 128, 0, stream>>>`, one block per lane at a fixed 128 threads with
/// a stride loop over that lane's output bytes, which is
/// [`LaunchRule::PerRowNarrow`] to the digit. The 128 is not a preference
/// here the way it is for the audio tower — nothing folds warp partials, so
/// the width is not a numerics contract — but it is still the launcher's, and
/// a rule that widened it to 256 would state a launch this tree does not
/// make.
pub const PACK_DENSE_MASK: Unit = Unit {
    name: "attn/pack_dense_mask",
    root: include_str!("../../csrc/src/attn/pack_dense_mask.cuh"),
    rows: PACK_DENSE_MASK_ROWS,
    options: &[],
};

/// `attn/pack_dense_mask.cuh`'s two kernels, both named by their bare
/// qualified path.
///
/// `elem` is [`DeviceKernel::PLAIN`] and not `""`: the constant is the row's
/// STATEMENT that this `__global__` has no template parameter list, and the
/// empty string is what an unfilled field looks like. The distinction is not
/// decoration — it is checked by the compiler on this box, in both
/// directions, and `examples/argform_probe.rs` holds the measurement:
/// `plain<device::bf16>` is `type name is not allowed` and a bare template
/// path is `cannot determine which instance of function template ... is
/// intended`. So a row that states the wrong one of the two fails
/// `tests/units.rs`, with NVRTC's own sentence.
static PACK_DENSE_MASK_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &PACK_DENSE_MASK_SIGS[0],
        template_path: "attn::device::pack_dense_mask",
        elem: DeviceKernel::PLAIN,
    },
    DeviceKernel {
        sig: &PACK_DENSE_MASK_SIGS[1],
        template_path: "attn::device::pack_structured_mask",
        elem: DeviceKernel::PLAIN,
    },
];

/// The two contracts, each its launcher's minus the stream.
///
/// **These two symbols are `table::driver_internal`'s and not
/// [`crate::table`]'s**, which is why they do not move
/// `examples/migration_status`: a driver-internal row is one no trace can
/// state — `FirePageMask` picks the packer, the DSL surface has no statement
/// for it — and `driver_internal` is deliberately outside `KERNELS` for that
/// reason. The symbol is still the launcher's own, spelled the same string in
/// both tables, because `driver-cuda`'s dispatch is what reads it.
///
/// Every operand is unsourced, for the reason [`PAGE_COMPACT_SIGS`]'s are:
/// `mask_indptr` is a host-built prefix sum the driver owns, `packed` is a
/// pre-zeroed driver allocation, and `p_page` is the dense mask's row stride.
/// No [`Source`] spells any of the three, and a fire hands them directly.
///
/// `b` stays an operand even though [`LaunchRule::PerRowNarrow`] opens the
/// grid over it. The kernel READS it — `if (b >= B) return;` and
/// `if (request >= B) return;` are the first lines of both — so a row that
/// dropped it on the grounds that the rule recovers it would hand the kernel
/// whatever the previous launch left in that slot. `PAGE_COMPACT_SIGS` keeps
/// `num_requests` for the same reason.
#[rustfmt::skip]
static PACK_DENSE_MASK_SIGS: [KernelSig; 2] = [
    // `pack_dense_mask.cu:94` -- `device::pack_dense_mask<<<B, BLOCK, 0, stream>>>`
    // with `constexpr int BLOCK = 128` at `:93`.
    kernel!(pack_dense_mask "attn::pack_dense_mask",
        file = Some("attn/pack_dense_mask.cuh"),
        launch = LaunchRule::PerRowNarrow,
        operands = operands![
            kvm_dense: U8s, klen: U32s, qo_indptr: U32s, mask_indptr: I32s,
            packed: U8sMut, b: I32, p_page: I32,
        ]),
    // `pack_dense_mask.cu:110` -- `device::pack_structured_mask<<<B, block, 0,
    // stream>>>` with `constexpr int block = 128` at `:109`.
    //
    // `masks` is `Ty::StructuredMasks`, which `runtime::args`' `is_pointer`
    // does not admit and `emit::crossing` therefore refuses — so this row
    // compiles and resolves and has NO generated entry point yet, which the
    // emitter records as a comment naming the operand. That is the honest
    // state: the descriptor array IS a device pointer, and saying so is a
    // change to the `Ty` vocabulary rather than to this row.
    kernel!(pack_structured_mask "attn::pack_structured_mask",
        file = Some("attn/pack_dense_mask.cuh"),
        launch = LaunchRule::PerRowNarrow,
        operands = operands![
            positions: U32s, klen: U32s, qo_indptr: U32s, mask_indptr: I32s,
            masks: StructuredMasks, packed: U8sMut, b: I32,
        ]),
];

/// The units the small half of `attn` compiles.
///
/// Separate from [`UNITS`] so the heavy half can be appended without either
/// list being rewritten — see the marker at the end of this file.
pub const UNITS_SMALL: &[Unit] = &[
    SOFTCAP,
    ATTN_SINK,
    ATTN_RES,
    DSA_INDEXER,
    ATTENTION_NAIVE,
    HEAD_DIM_PAD,
    SPLIT_PACKED,
    PAGE_COMPACT,
    PACK_DENSE_MASK,
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
//    (`attention_naive_paged.cuh:141` and `:152`, `enum class … : u8`), and
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

/// The units the heavy half of `attn` compiles.
pub const UNITS_HEAVY: &[Unit] =
    &[KV_PAGED, DSV4_COMPRESS, KIMI_MLA, MLA_PAGED, QKV_FUSED, ATTENTION_NAIVE_PAGED];

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
/// (`attention_naive_paged.cuh:334-336`). A row at another width would read
/// slots nothing wrote. [`crate::runtime::launch`]'s `PAGED_BLOCK` states the
/// same number on the geometry side and says so at greater length.
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
/// `use_custom_mask = custom_mask != nullptr` at `attention_naive_paged.cuh:339`
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
    // `<cstdint>`), so `attention_naive_paged.cuh:140` and `:151` declare
    // device mirrors and `attention_naive_paged.cu` `static_assert`s every
    // enumerator of both. A row naming the mirror is naming a checked type.
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
static QKV_FUSED_SIGS: [KernelSig; 7] = [
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
];

/// `attn/mla_paged.cuh`'s one row.
///
/// `elem` is [`DeviceKernel::PLAIN`] — the row's statement that
/// `attn::device::write_mla` has no template parameter list, as against the
/// empty string, which is what an unfilled field looks like. See
/// [`PACK_DENSE_MASK_ROWS`] for the two refusals that make the distinction
/// checkable rather than conventional.
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
];

#[rustfmt::skip]
static DSV4_COMPRESS_SIGS: [KernelSig; 9] = [
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
    kernel!(dsv4_boundary_meta_decode "attn::dsv4_boundary_meta_decode",
        file = Some("attn/dsv4_compress.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            positions: I32s, out_pos: I32sMut, out_req: I32sMut, out_rope: I32sMut,
            n: I32, ratio: I32, row_valid: U8s,
        ]),
    // The prefill form, same geometry and same launcher shape; it differs
    // only in resolving the request index by a binary search over `qo_indptr`
    // instead of shortcutting it to the token index.
    kernel!(dsv4_boundary_meta_paged "attn::dsv4_boundary_meta_paged",
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
    kernel!(attention_compressed_paged "attn::attention_compressed_paged_bf16",
        file = Some("attn/dsv4_compress.cuh"),
        launch = LaunchRule::PagedScoresDecode,
        operands = operands![
            q: Buf, comp_kv_pages: Buf, o: BufMut, lse_out: F32sMut,
            positions: I32s, kv_page_indices: U32s, kv_page_indptr: U32s,
            req_of_token: I32s,
            num_q_heads: I32, head_dim: I32, ratio: I32, page_size: I32,
            scale: F32,
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
];

#[rustfmt::skip]
static KV_PAGED_SIGS: [KernelSig; 20] = [
    // `n` is `I64` because the kernel's parameter is `long long`, and it is a
    // `long long` because it indexes a page arena that is multiple gigabytes
    // at production page counts — `Ty::I64` says exactly that and the row
    // says nothing else.
    //
    // The fp8 pages are `U8s` and not a format of their own: on the device
    // they are `__nv_fp8_storage_t`, which IS one byte, and the format is the
    // kernel's to interpret. `attn::device::dequant_fp8_pages_active` — the
    // per-TENSOR form — has no row for the other half of that sentence: it
    // takes the interpretation as an `__nv_fp8_interpretation_t` argument and
    // the `Ty` vocabulary has no enum. Defaulting it to `__NV_E4M3` would
    // decode an E5M2 page to a numerically plausible wrong answer.
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
    // **The one symbol here that is also an ahead-of-time row's.**
    // `attn::write_kv_explicit_bf16` in [`crate::table::attn`] is the host
    // function at `kv_paged.cu:355`, and that function holds this `__global__`
    // and no other — a null guard, `if (layer.hnd_layout)`, two `<<<B, 256>>>`.
    // Its `B` is `Source::Rows` there and `PerRow`'s grid is `Dims::rows` here,
    // so the two state the same rectangle from the same number rather than
    // agreeing by coincidence. Sharing the string is what `examples/migration_status`
    // means by a join, and this row earns it.
    kernel!(write_kv_explicit "attn::write_kv_explicit_bf16",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            w_page: U32s, w_off: U32s, row_valid: U8s | null,
            b: I32, page_size: I32, h_kv: I32, d: I32,
            hnd_layout: Bool,
        ]),
    kernel!(write_kv_explicit_hnd "attn::write_kv_explicit_bf16#hnd",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            w_page: U32s, w_off: U32s, row_valid: U8s | null,
            b: I32, page_size: I32, h_kv: I32, d: I32,
        ]),
    kernel!(write_kv_explicit_nhd "attn::write_kv_explicit_bf16#nhd",
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
    kernel!(write_kv_explicit_devwin "attn::write_kv_explicit_bf16_devwin",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            w_page: U32s, w_off: U32s, row_valid: U8s | null, win: U32s,
            n_max: I32, page_size: I32, h_kv: I32, d: I32,
            hnd_layout: Bool,
        ]),
    kernel!(write_kv_explicit_devwin_hnd "attn::write_kv_explicit_bf16_devwin#hnd",
        file = Some("attn/kv_paged.cuh"),
        launch = LaunchRule::PerRow,
        operands = operands![
            k_curr: Buf, v_curr: Buf, k_pages: BufMut, v_pages: BufMut,
            w_page: U32s, w_off: U32s, row_valid: U8s | null, win: U32s,
            n_max: I32, page_size: I32, h_kv: I32, d: I32,
        ]),
    kernel!(write_kv_explicit_devwin_nhd "attn::write_kv_explicit_bf16_devwin#nhd",
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
    base: "attn::write_kv_explicit_bf16",
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
    base: "attn::write_kv_explicit_bf16_devwin",
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
