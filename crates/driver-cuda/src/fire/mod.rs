//! One forward pass: its scratch, its tables, its recordings.
//!
//! A fire is the unit this shell exists to serve — a batch of rows
//! through a lowered program — and everything here has that lifetime or
//! is pooled across it. [`launch`] is the pass itself; the rest is what
//! it needs standing before it can run.
//!
//! The pooling is not an optimisation. A recorded graph BAKES an
//! address, so a buffer that moved between fires would be replayed
//! against memory that is no longer there; [`scratch::Scratch`] exists
//! so that a fire's addresses are the same as the last fire's.

/// `moe/flashinfer_moe.cu`'s HOST PROGRAM, in Rust — 817 lines with **zero**
/// `__global__`, `<<<>>>` and `__device__`, which is what made it a host
/// program that happened to have a `.cu` extension. The workspace query, the
/// arch probe, the coordinate-descent autotuner, the per-device tactic memo
/// and the on-disk tactic cache all came here; what stayed C++ is a five-
/// function `extern "C"` seam over `CutlassMoeFCRunner`, a class template no
/// Rust and no NVRTC can name.
///
/// `bridge` and not just `_cuda`: the seam is compiled into
/// `libpie_kernels_cuda.a` beside the CUTLASS instantiations, and `bridge` is
/// exactly the feature that links that archive — the same reason [`tower`] is
/// gated.
///
/// [`tower`]: crate::tower
#[cfg(feature = "bridge")]
pub mod flashinfer_moe;
/// `attn/attention_flashinfer.cu`'s PLAN HALF, in Rust — the sixth file this
/// migration found wearing a `.cu` extension for linkage rather than content:
/// 1,258 lines with `__global__` 0, `__device__` 0 and one launch, which is
/// `device::attn_score_fold_heads` and already a row.
///
/// The three plan caches, the two real planner factories over
/// `kernels_cuda_new::plan`, the static non-split short-circuit and its two
/// environment gates. **The four `switch (cache.head_dim)` dispatches are NOT
/// here yet** and the C++ file is still the one that runs; see the module's
/// header for why that ordering is deliberate.
pub mod flashinfer_fa2;
// `comm/custom_all_reduce.cu`'s HOST PROGRAM, in Rust — 664 lines with zero
// `__global__` and zero `<<<>>>`, the fifth file this migration found wearing
// a `.cu` extension for linkage rather than content. The whole lifecycle came
// here: peer-access enablement, the IPC handle exchange, the `RankData` slab,
// the fusion plane's four allocations and its Lamport initialisation, the
// NCCL crossover query, and the 240-point template cross product that
// `kernels.def`'s `PIE_AR_FUSION_PATTERN` axis existed to prune. What did not
// cross is two launches into CPM-fetched flashinfer headers this tree does
// not vendor, and those are refusals naming the exact template point.
pub mod all_reduce;
pub mod attention_workspace;
pub mod attn_score;
// `ssm/causal_conv1d.cu`'s prefill launcher, in Rust — one symbol, two
// `__global__`s, and a host `if` on the request count.
pub mod causal_conv1d;
// The three families ported out of `crates/kernels-cuda/csrc/src` — `norm/`
// and (until §5 step 3 moved it) `rope/`'s host launchers, in Rust, firing
// NVRTC-compiled device text out of `kernels-cuda-new/csrc/src/norm/*.cuh`.
// [`hand`] is the launch they share: a `Launch` this driver states, for the
// geometries no `LaunchRule` does.
pub mod dsv4_hc;
/// The loader's two dtype casts — `quant/dtype_cast.cu`'s whole surviving
/// surface, fired through the JIT. Named by `fire::lora`, which called them
/// through `ffi::pie_k_quant_*` until the rows were routed.
pub mod dtype_cast;
/// `layout/envelope.cu`'s three launchers, in Rust — quest per-page key
/// envelopes, seeded, appended to and merged into. **That file is deleted.**
///
/// It is the first of the two `layout` files to go and it went first because
/// it was the wall: `families/layout.rs` had refused a row for every envelope
/// kernel, and eight launches in `attn/kv_paged.cu` were behind those
/// refusals. The finding survives — no `LaunchRule` states a
/// `(token, kv_head)` grid at `min(head_dim, 256)`, and none learns to — but
/// [`kernels::LaunchRule::Unstated`] plus a driver-owned `Launch` was always
/// the answer to *"no rule states this"*, and *"one symbol, two launches"* is
/// a Rust `if`.
/// `layout/embed.cu`'s one launcher, in Rust — and with it the whole of
/// `kernels-cuda/csrc/src/layout/`, which is now EMPTY. The `VEC` choice is a
/// 16-byte alignment test on two pointers plus `hidden % 8`, and the extent
/// it launches over depends on the answer; that is the host program the C++
/// kept the file for. `layout/embed.cuh` is a unit again (`families/layout.rs`
/// `EMBED`, two instantiations of `embed<bool VEC>`), and the row moved from
/// `table::driver_internal` to `table::layout` so `RUST_SERVED` could take it.
/// `attn/attention_naive.cu`'s two surviving launchers, in Rust — the MTP
/// state pair, and with them the file. Three of that file's five went in an
/// earlier pass on an empty consumer set; these two sit behind live
/// `dsl::cuda` wrappers. Both needed §60.6's symbol split first: the device
/// rows carried the same strings as the table rows, and §52.11 forbids
/// walking a symbol a unit hosts.
pub mod attention_naive;
/// `attn/dsa_indexer.cu`'s three launchers, in Rust — the whole file.
/// DeepSeek sparse attention's indexer. One needed only §60.6's symbol split,
/// one needed a device row that did not exist (its block is
/// `round_up(n_heads, 32)`, a statement parameter no `Dims` carries), and the
/// third — the causal top-k mask, whose row is FULLY SOURCED — needed the
/// split so a live dispatch could move off the C shim.
pub mod dsa_indexer;
/// `attn/dsv4_compress.cu`'s four surviving launchers, in Rust — the whole
/// file. Nine went in earlier passes. Three needed only §60.6's symbol split;
/// `combine_attn_outputs_bf16` needed a device row that had been deliberately
/// withheld, and the reason it was withheld — the block clamps to 256 where
/// `PerHeadElementwise` clamps to 128, invisibly — is why the new row is
/// `LaunchRule::Unstated` and the geometry is the driver's.
pub mod dsv4_compress;
pub mod embed;
pub mod envelope;
/// `gemm/gemm.cpp`'s HOST PROGRAM, in Rust — **zero `__global__`, zero
/// `<<<>>>`, 138 cuBLAS/cuBLASLt calls**. Not a kernel file and never was: a
/// shape ladder, a cuBLASLt plan cache, a private-stream autotuner and an
/// on-disk tactic memo. Every measurement in it — five per-checkpoint
/// heuristic indices, the `min_n` FAULT guard, the split-K determinism
/// restriction, the 0.98 tie margin, the `NOT_SUPPORTED` TP hang — is carried
/// onto the item that made it.
///
/// It went last because `gemm_bf16_impl` called `gemv_bf16`, whose `bool`
/// meant *"I did not launch"*, and a row cannot decline. [`gemv`] answered
/// that by not being a row: its refusal is a type, so the tuner's `Gemv`
/// candidate is now `matches!(.., Gemv::Launched)`.
pub mod gemm;
pub mod gemv;
// `ssm/gated_delta_net.cu`'s four surviving launchers, in Rust — Qwen3.5's
// decode recurrence and its two prefill scans. **Eight of that file's
// seventeen `<<<>>>` were unreachable** behind `constexpr false` selectors
// and are not ported; the module header carries the audit.
pub mod gated_delta_net;
pub mod hand;
// `ssm/kda.cu`'s two recurrence launchers, in Rust. Both rows are
// deliberately UNSOURCED — `state_base` is a driver-owned slab and `Source`
// has no `Scratch` (`new-horizon.md` §52.3, §56, §57.3) — so neither is reachable
// from a model trace yet. The geometry is captured now, while the C++ that
// states it is still readable.
pub mod kda;
// `ssm/nemotron_h.cu`'s two multi-armed launchers, in Rust: the fused
// in-projection split and the Mamba-2 selective scan. **Two of that file's
// eleven `<<<>>>` were dead** — an `if constexpr (false)` and a block after
// an unconditional `return` — and are not ported.
pub mod nemotron_h;
// `quant/quant_bf16_to_fp8.cu`'s four launchers, in Rust — three ported and
// one deleted for having no consumer in any language. It is the file the
// three hand-written `ffi::pie_k_quant_*` arms in `bind/quant_gemm.rs` held
// alive; those arms now call this module and `csrc/src/quant/` is gone.
pub mod quant_int8;
pub mod rmsnorm;
// `rope` IS NOT HERE, and its absence is the migration.
//
// Its nine host programs are `kernels-cuda-new/src/x/rope.rs`, beside the
// `rope.cuh` whose `__global__`s they fire — `.wiki/kernel-x/northstar.md`
// §5 step 3, the first family to cross. Nothing was lost in the move: the
// measurements came with the fns that carry them, and the family gained
// four host programs the row world could declare and not reach.
//
// What a reader looking for a launcher here should know is that this
// directory is now the UNPORTED half. A family in `fire/` has its device
// text in one crate and its host program in another; a family in
// `kernels-cuda-new/src/x/` has both in one place and a `contract!` that
// `model-compiler` reads without knowing a GPU exists.
// GATED ON `abi`, and that is a finding rather than a tidy-up. The
// forward pass takes `driver_api::PieFrameDesc` and reads
// `serve::state::Shell`, so `fire/` — which the tree calls "one forward
// pass: its scratch, its tables, its recordings, its retirement" —
// cannot be built without the door. §6's middle build spelling
// (`--features cuda-13`, cudarc only, no toolkit) did not work before
// this line, and it is the build a CI without a card would run.
//
// The right fix is that a fire is described by a value this crate owns
// rather than by the ABI's struct, which is §3.2's move applied one
// layer further in. Until then the gate says where the seam actually
// is, instead of a build error saying it three ways.
#[cfg(feature = "abi")]
pub mod launch;
/// `moe/dsv4_routing.cu`'s one launcher, in Rust — the whole file, which is
/// DELETED. DeepSeek-V4's hash router: expert INDICES gathered from a
/// `[vocab, K]` checkpoint table keyed by token id, expert WEIGHTS still
/// `sqrt(softplus(logits))` read at those indices. Classified
/// `Execution::Walk` with `Control::Supplies` — the table and its first
/// extent are what no `Source` names — and in `execution::RUST_SERVED`, which
/// is what drops its shim entry. The row itself has no generated arm, because
/// all eleven of its `table::moe` operands are unsourced.
pub mod dsv4_routing;
/// `attn/kv_paged.cu`'s device-window explicit KV append and its quantised
/// appenders — refusals that threw, a `Term::Is` over the page layout, a grid
/// over the fire's full lane count, and the four-armed `layer.scheme` switch
/// that `kernels::Ty::Fp8Kind` unblocked.
///
/// **FOUR of the five launchers here are now REACHED**, and §58 is the
/// reason the fourth took two passes. §58 read `a_walk_is_only_a_walk`
/// against `Specialisation::agrees` and concluded a specialised symbol wants
/// no `Walk`, no `RUST_SERVED` entry and no `bind::service` shim — it wants
/// the device row and the specialisation it already has. True, and it leaves
/// the Rust unreachable: the only thing that CAN call it is a generated
/// dispatch arm, and the emitter writes one only for `JIT_DISPATCHED` or
/// `RUST_SERVED`. "It needs a caller" and "it needs a classification" were
/// the same sentence.
///
/// §60.6 dissolves it rather than choosing a side. The DEVICE rows for
/// `write_kv_explicit_bf16_devwin` are `..._devwin_dev` and
/// `WRITE_KV_EXPLICIT_DEVWIN`'s `base` moved with them, so the ahead-of-time
/// symbol is unit-free and walkable while the `Specialisation` still
/// resolves — exactly the arrangement the sibling `write_kv_explicit_bf16`
/// was already in. `write_kv_to_pages_quantised` stays unreached and is the
/// genuine §58 case: it is a staging function with a live Rust caller in
/// this same module, not a row anyone dispatches.
///
/// `write_kv_to_pages` and `write_kv_explicit_bf16` took the other door:
/// classified `Execution::Walk`, named in `execution::RUST_SERVED`, reached
/// through `bind::service`. Both fire the quantised port and
/// [`envelope`] underneath them, so the two staging functions are live code
/// with a caller even though their own symbols are not routed.
pub mod kv_paged;
/// The fused LM-head GEMV + argmax — `sample/argmax.cu`'s last launcher, and
/// the whole of `csrc/src/sample/`. Two JIT'd kernels with a growable device
/// scratch between them, on grids read off an occupancy query; reached
/// through `bind::service`, so the model text sees one symbol.
pub mod lm_head_argmax;
pub mod lora;
/// `attn/mla_paged.cu`'s two launchers, in Rust — the whole file, which is
/// DELETED. The MLA prologue (`kv_a` RMSNorm, `k_pe` rotation, paged write,
/// query nope/pe split) on a `dim3(total_tokens, 1 + q_blocks)` grid whose
/// second axis the host computes, and the paged append. Both classified
/// `Execution::Walk` with `Control::Supplies`, both in
/// `execution::RUST_SERVED`, both reached through `bind::service`. The third
/// function that file held, `write_mla_to_pages_bf16`, had an empty consumer
/// set on all five channels and is deleted rather than ported.
pub mod mla_paged;
pub mod moe;
/// `moe/moe_dispatch.cu`'s launchers, in Rust — ALL EIGHT, and the file is
/// DELETED. The two decode GEMVs, the MXFP4 group-scale relayout, the aligned
/// pointer builder, the route-order scatter, the exact counting sort, the
/// per-route expert-bias add and the weighted scatter-accumulate. The last
/// three landed after the module header had spent a section arguing they
/// could not: they are unit-hosted AND their `table::moe` rows are unsourced,
/// which bars `JIT_DISPATCHED` and forced a `_dev` symbol split, but the
/// unsourcedness is precisely what makes each a host program supplying a
/// value — `Control::Supplies`, not a closed door. That header now carries
/// the correction, since the same shape will come up again.
///
/// `csrc/src/moe/` is left holding `flashinfer_moe.cu`, which is now an
/// `extern "C"` INSTANTIATION SEAM and nothing else — five functions over
/// `CutlassMoeFCRunner` and two standard headers, down from fourteen. Its
/// 817-line host program (workspace arithmetic, arch probe, autotuner,
/// tactic caches, dispatch) is `fire::flashinfer_moe`, which had zero
/// `__global__` to move because it never had any.
pub mod moe_dispatch;
/// `attn/page_compact.cu`'s one launcher, in Rust — the whole file. Two
/// launches on one stream, the second reading the `scratch_counts` buffer the
/// first fills; `execution::COMPOSED` has stated that pair since the split
/// and what was missing was the host. `RUST_SERVED` takes the row.
pub mod page_compact;
/// `attn/qkv_fused.cu`'s fused decode epilogue, in Rust — the whole file,
/// which is DELETED, and the LAST `attn` dispatch to fall. One launcher over
/// four kernels and eight instantiations: `head_dim` picks the warp form at
/// 64, 128 or 256 and falls THROUGH to the block form otherwise, and
/// `rope_table != nullptr` picks the `USE_ROPE_TABLE` arm inside whichever it
/// picked. No `Specialisation` can state that, because the fallthrough
/// changes the `LaunchRule` from `WarpPackedHeads` to `RowsPackedHeadsNarrow`
/// and `Specialisation::agrees` forbids an arm that changes the rule;
/// `families/attn.rs` had written the refusal and added that lifting it would
/// not help, *because `head_dim == 64 | 128 | 256` is still unspellable*. A
/// host program spells it. Four device rows were written here for the
/// `_d64`/`_d256` warp expansions the unit never had — the absence
/// `qkv_fused.cu`'s own header named as what blocked its port.
/// `Execution::Walk` with `Control::Switch`, in `execution::RUST_SERVED`, and
/// its row is FULLY SOURCED, so this one moves a LIVE dispatch off the shim.
pub mod qkv_fused;
pub mod page_mask;
pub mod recordings;
/// `attn/split_packed.cu`'s one surviving launcher, in Rust — the whole file.
/// `families/attn.rs` argued at length that no `LaunchRule` may state this
/// geometry; every clause of that argument is answered by a host rather than
/// by a rule, which is what a driver-owned `Launch` is.
pub mod split_packed;
pub mod scratch;
pub mod sideband_arena;
pub mod stage_hooks;
/// `csrc/supergraph.cu`'s two launchers, in Rust — **that file is deleted**,
/// and with it the second of this crate's three nvcc builds. The device text
/// is a JIT unit now (`kernels-cuda-new/csrc/src/graph/supergraph.cuh`); the
/// claim that stood in the way — *"this needs nvcc"* — was measured and is
/// false, and the measurement is in this module's header.
pub mod supergraph;
/// XQA's fire-wide prepare — the last `__global__` the `kernels-cuda` archive
/// held, as a JIT'd kernel with its workspace carve in Rust. Discharges the
/// `Prepare::FireWide` that `attn::attention_xqa_decode_bf16_prepared`'s row
/// states and that no code implemented on any reachable channel.
pub mod xqa;
