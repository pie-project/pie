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
/// gated. Both the `.a` and the ARCHIVE crate whose nvcc build produced it
/// are gone, deleted at `85c6c674b`.
///
/// [`tower`]: crate::tower
// `pub mod flashinfer_moe;` was here, under `#[cfg(feature = "bridge")]`. It
// was the host half of the fused CUTLASS MoE — the workspace query, the arch
// probe, `CutlassMoeFCRunner<bf16,bf16>` through five `extern "C"` seams, and
// latterly a Rust `to_underlying_arguments` filling 1,408 measured bytes of
// `GemmKernel::Params`.
//
// It went with the leg it served. **What it leaves behind is one bind**:
// `moe::build_moe_ptrs_aligned_bf16` declares `gu_stage`/`act_stage`/
// `out_stage`, the destinations every op in the aligned leg writes into, and
// it has never had an arm in either world — the fused leg had been covering
// for it. Until it binds, qwen3.5's MoE has no leg that starts.

// `flashinfer_fa2` AND `flashinfer_fa2_dispatch` ARE GONE FROM THIS
// DIRECTORY, and they are the largest thing §6.3 asked for.
//
// `attn/attention_flashinfer.cu`'s plan half and its dispatch half were the
// last of FA2's host program still living above the crate boundary. They are
// `kernels_cuda::attn::fa2::{plan, dispatch}` now, beside the 56-root
// lattice they fire and the `#[repr(C)]` mirrors they fill.
//
// **The mirrors are why it had to be this direction.** The params filling is
// checked against MEASURED struct layouts (`nvrtc-probes/params_layout.py`),
// and those assertions live with the mirrors; a filler on this side of the
// boundary could not be reached by them. The same is true of the plan half
// for a different reason -- it computes a geometry out of four device
// attributes and the lattice's own occupancy arithmetic, and half of that
// arithmetic was already `attn::fa2::geometry`'s.
//
// What stayed here is the part that is genuinely the driver's: `bind/arms/
// fa2.rs` joins a trace statement to one of the six routines, `bind/mod.rs`
// owns the two plan caches' lifetimes behind raw-pointer handles, and
// `tower/qwen3_vl/attn.rs` drives the prepare-then-fire pair by path. All
// three reach down for what they need and none of them holds FA2 arithmetic
// any more.

// `merge_states` WENT DOWN WITH THEM, and it had been ready to for longer.
//
// The whole file imported `kernels_cuda` and `std::ffi::c_void` and
// nothing else: `VarLen` is `cascade.cuh:687-690`'s operand list in
// `cascade.cuh`'s own field names, and `variable_length` is one call to
// `kernels_cuda::cascade::merge_states_varlen`. It was a host program for
// a kernel one crate down that happened to be compiled one crate up.
//
// What made it visible is that `attn::fa2::dispatch` builds the job. Once
// that file descended, its `merge_job` was a function in `kernels-cuda`
// whose RETURN TYPE lived here — and `merge_job`'s own doc had recorded the
// boundary as the reason it was a free function rather than a method on
// `Partials`. It is `impl Partials` now, which is what that comment was
// waiting for.
//
// `kernels_cuda::cascade::merge_states` is the address.
// `comm/custom_all_reduce.cu`'s LIFECYCLE, in Rust — peer-access enablement,
// the IPC handle exchange, the `RankData` slab, the fusion plane's four
// allocations and its Lamport initialisation, the NCCL crossover query, and
// the destructor that closes every mapping the constructor opened. The whole
// 664-line host program came here first (the fifth file this migration found
// wearing a `.cu` extension for linkage rather than content); §6.3 then split
// it, and the half that left is the half a LAUNCH reads — the 240-point cross
// product `kernels.def`'s `PIE_AR_FUSION_PATTERN` axis existed to prune, the
// `AllReduceFusionParams` mirror, both refusals and both bodies. That half is
// `kernels_cuda::comm`, and `comm::all_reduce_bf16` and
// `comm::all_reduce_residual_rmsnorm_bf16` are derived rows off its `fn`s.
//
// **The line is `crate::error::Error`.** Every function that stayed reports
// through it, carrying a `cudaError` and a `String`; `kernels-cuda` can
// name neither, and `Refusal` deliberately carries no driver code. What is
// left here therefore owns something — three other processes hold the far end
// of the IPC handles — and what left owns nothing. `comm::Plane` is the five
// facts a launch reads off this side, filled per call -- and it grew a
// `PeerPlane` when the plain reduction's launcher landed, because that
// kernel takes peer addresses the fused one derives on the device.
//
// Both launches cross now. They named two flashinfer headers that were
// CPM-fetched and unvendored, and were refusals naming the exact template
// point; `csrc/src/attn/flashinfer/comm/` holds both and
// `kernels_cuda::comm::CAN_LAUNCH` is `true`. What has not been done is
// FIRING one: a collective needs a peer and this box has one GPU.
pub mod all_reduce;
pub mod attention_workspace;
pub mod attn_score;
// `fire::causal_conv1d` IS GONE — §5 step 5 took `ssm` into fn-world and its
// two host programs are `kernels_cuda::ssm::causal_conv1d`'s. The
// two-`__global__` `if` on the request count went with them, unchanged.
// The three families ported out of the ARCHIVE crate's `csrc/src` — `norm/`
// and (until §5 step 3 moved it) `rope/`'s host launchers, in Rust, firing
// NVRTC-compiled device text out of `kernels-cuda/csrc/src/norm/*.cuh`.
// [`hand`] is the launch they share: a `Launch` this driver states, for the
// geometries no `LaunchRule` does.
//
// `dsv4_hc` IS NOT HERE EITHER, and for `rmsnorm`'s reason — see below. Its
// four launchers are `kernels-cuda/src/x/norm.rs`'s `hc_*` fns, beside
// the `dsv4_hc.cuh` whose `__global__`s they fire, and the three §43.9 had
// already deleted for having no launcher (`hc_expand_bf16`,
// `attn_sink_correction_bf16`, `per_head_rmsnorm_bf16`) are `fn`s there now
// as well, so all seven of HC's kernels are written in one place for the
// first time.
// `fire::dtype_cast` IS GONE — §5 step 5 took its two host programs into
// `kernels_cuda::quant`. Its doc read: *"The loader's two dtype casts
// — `quant/dtype_cast.cu`'s whole surviving surface, fired through the JIT.
// Named by `fire::lora`, which called them through `ffi::pie_k_quant_*` until
// the rows were routed."* Both sentences are still true of
// `x::quant::{cast_fp32_to_bf16, scale_rows_bf16}`, one crate over, beside
// the `dtype_cast.cuh` whose `__global__`s they fire.
//
// The move is the third station of a three-station journey and the file
// recorded the first two: `ffi::pie_k_quant_*`, then `bind::jit::fire` on a
// routed row, now a host `fn`. What each step removed is one place the
// geometry could be written — the C launcher, then the `LaunchRule` — and
// what is left is `<<<(n + 255) / 256, 256>>>` written once, in
// `x::quant::elementwise`, next to the `.cuh` it came from.
/// `layout/envelope.cu`'s three launchers, IN FN-WORLD — quest per-page key
/// envelopes, seeded, appended to and merged into. **That file is deleted.**
///
/// It was the first of the two `layout` files to go and it went first because
/// it was the wall: `families/layout.rs` had refused a row for every envelope
/// kernel, and eight launches in `attn/kv_paged.cu` were behind those
/// refusals. The finding survives — no `LaunchRule` states a
/// `(token, kv_head)` grid at `min(head_dim, 256)`, and none learns to — but
/// [`kernels::LaunchRule::Unstated`] plus a driver-owned `Launch` was always
/// the answer to *"no rule states this"*, and *"one symbol, two launches"* is
/// a Rust `if`.
///
/// **`fire::envelope` IS GONE.** §5 step 5 took `layout` into fn-world and
/// the three programs are `kernels_cuda::layout::envelope_*`, beside
/// the declarations that name the `__global__`s they fire. `unit!` states
/// `LaunchRule::Unstated` for every row it generates, so the finding above
/// is no longer a choice a row makes — it is the only thing a device
/// declaration can say, which is what "no rule states this" always meant.
/// The one caller in `bind::abi` calls the new path directly, and the two
/// that were in [`kv_paged`] moved with their bodies into
/// `x::attn::kv_paged` — where the call is now a sibling module's, which is
/// the shape the two `envelope_*` calls always wanted.
/// `layout/embed.cu`'s one launcher, IN FN-WORLD — and with it the last file
/// in the ARCHIVE crate's `kernels/layout/`, which that left EMPTY. The
/// `VEC` choice is a 16-byte alignment test on two pointers plus
/// `hidden % 8`, and the extent it launches over depends on the answer; that
/// is the host program the C++ kept the file for.
///
/// **`fire::embed` IS GONE**, with `fire::envelope` and in the same change:
/// the program is `kernels_cuda::layout::embed_bf16` and the
/// predicate is `x::layout::vectorisable`, public because a caller staging
/// its own buffers may ask the same question. `layout/embed.cuh` is one of
/// `x::layout`'s five units (`EMBED`, two instantiations of
/// `embed<bool VEC>`), and the note that the row *"moved from
/// `table::driver_internal` to `table::layout` so `RUST_SERVED` could take
/// it"* is history: there is no row, and an empty operand list drops the
/// shim entry without a list to keep in step.
/// `attn/attention_naive.cu`'s two surviving launchers, in Rust — the MTP
/// state pair, and with them the file. Three of that file's five went in an
/// earlier pass on an empty consumer set; these two sit behind live
/// `dsl::cuda` wrappers. Both needed §60.6's symbol split first: the device
/// rows carried the same strings as the table rows, and §52.11 forbids
/// walking a symbol a unit hosts.
// `attention_naive` DELETED -- the MTP pair crossed into
// `kernels_cuda::attn` and refuses on `slot_ids`, which only
// `Cx::gdn` reaches. `attn/attention_naive.cuh` is
// `x::attn::attention_naive`'s unit, declaring the two of its five
// `__global__`s that have a host program.
// `dsa_indexer` CROSSED WHOLE into `kernels_cuda::attn`. It held
// `attn/dsa_indexer.cu`'s three launchers and `bind::service`'s three entry
// points were its only consumer, so the module went with them rather than
// being left as a body nothing calls. The block width `round_up(n_heads, 32)`
// -- the reason one of the three never had a device row -- is
// `x::attn::dsa_indexer::q_rope_block` now, next to the unit whose kernel
// reads it.
/// `attn/dsv4_compress.cu`'s three surviving launchers, in Rust — the whole
/// file. Nine went in earlier passes and a tenth,
/// `combine_attn_outputs_bf16`, has crossed into fn-world as
/// `kernels_cuda::attn`'s `COMBINE_ATTN_OUTPUTS`; it is the only one
/// of the four whose every value came out of the statement. The three that
/// remain needed §60.6's symbol split and nothing else.
// `dsv4_compress` DELETED -- all four host programs crossed into
// `kernels_cuda::attn`, `combine_attn_outputs_bf16` bound and the
// other three `none:` on the compression ratio. `attn/dsv4_compress.cuh` is
// `x::attn::dsv4_compress`'s unit and holds all five kernels.
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
///
/// # BOTH FILES HAVE MOVED, AND THESE TWO NAMES ARE NOW RE-EXPORTS
///
/// §5 step 5 took `gemm` into fn-world. A host program belongs beside its
/// device text, so `fire/gemm.rs` is `kernels_cuda::gemm::dense` and
/// `fire/gemv.rs` is `kernels_cuda::gemm::gemv`; `x::gemm` itself
/// holds the `unit!` for the GEMV rows, the twelve `contract!`s and the
/// `bind!`.
///
/// The two names stay HERE as re-exports rather than being deleted, and that
/// is not softness about a deletion. `crate::fire::gemm::act_x_wt_bf16` is
/// spelled by `tower::gemma4_vision`, `tower::qwen3_vl` and `fire::lora`,
/// and by a dozen doc links besides; a re-export makes
/// every one of them keep resolving to the one definition, which is exactly
/// what a re-root is supposed to cost. The alternative — editing four
/// unrelated modules to say a longer path — would be a rewrite of files this
/// change has no business in.
pub use kernels_cuda::gemm::dense as gemm;
pub use kernels_cuda::gemm::gemv;
// `fire::gated_delta_net` IS GONE — §5 step 5 took it into
// `kernels_cuda::ssm::gated_delta_net`, where its four launchers
// became ten host programs (six of the ten were rule-driven rows with no
// `.cu` launcher to move). **The eight-of-seventeen dead-`<<<>>>` audit and
// the 34 % / nine-fold measurements moved with them**; nothing in it was
// carried by this file.
pub mod hand;
// The step's descriptor resolution: the geometry a `DecodeEnvelope` member
// leaves off the wire, read off the channel rings before the forward, and the
// working-set→physical page translation every member needs.
#[cfg(feature = "abi")]
pub(crate) mod envelope;
// `fire::kda` IS GONE — §5 step 5 took it into
// `kernels_cuda::ssm::kda`. Its two deliberately UNSOURCED rows are
// now two `bind!` `none:` arms, which is the fn-world spelling of the same
// fact: `state_base` is a driver-owned slab and `Source` has no `Scratch`
// (`new-horizon.md` §52.3, §56, §57.3), so a load-time refusal prints the
// sentence the row was carrying as prose.
// `fire::nemotron_h` IS GONE — §5 step 5 took it into
// `kernels_cuda::ssm::nemotron_h`, four launchers becoming seven host
// programs. **The two-of-eleven dead-`<<<>>>` audit moved with them.**
// `fire::quant_int8` IS GONE — §5 step 5 took its three host programs into
// `kernels_cuda::quant`. Its doc read: *"`quant/quant_bf16_to_fp8.cu`'s
// four launchers, in Rust — three ported and one deleted for having no
// consumer in any language. It is the file the three hand-written
// `ffi::pie_k_quant_*` arms in `bind/quant_gemm.rs` held alive; those arms
// now call this module and `kernels/quant/` is gone."*
//
// `gemm::quant` calls `x::quant` directly now, and the two geometries
// this module carried BECAUSE no rule could state them — `<<<(ceil(k /
// group_size), m), 128>>>` and `<<<(ceil(N / 32), ceil(M / 8)), (32, 8)>>>` —
// are literal `Launch` values there, which is §5.1's rule that a kernel
// fitting neither `flat` nor `per_row` writes the literal. The fourth
// launcher is still deleted and still has no consumer in any language.
// `rmsnorm` IS NOT HERE, and its absence is the migration.
//
// Its five host programs are `kernels-cuda/src/x/norm.rs`, beside the
// `rmsnorm.cuh` whose `__global__`s they fire — `.wiki/kernel-x/northstar.md`
// §5 step 5, the fifth family to cross and the one §5.1 named as the first
// proof of `Composed`/`Walk`. Nothing was lost in the move: the
// `RMSNORM_STRIDED_VEC8` sweep, the RASR sweep, the bit-identity measurement
// and the `EMIT_FP16` defect all came with the fns that carry them, and
// `norm::rmsnorm_bf16_with_fp16`'s three arms are now one `fn` body rather
// than a `Choose` the row world could not write.
// `rope` IS NOT HERE, and its absence is the migration.
//
// Its nine host programs are `kernels-cuda/src/x/rope.rs`, beside the
// `rope.cuh` whose `__global__`s they fire — `.wiki/kernel-x/northstar.md`
// §5 step 3, the first family to cross. Nothing was lost in the move: the
// measurements came with the fns that carry them, and the family gained
// four host programs the row world could declare and not reach.
//
// What a reader looking for a launcher here should know is that this
// directory is now the UNPORTED half. A family in `fire/` has its device
// text in one crate and its host program in another; a family in
// `kernels-cuda/src/x/` has both in one place and a `contract!` that
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
// `fire::dsv4_routing` IS GONE — §5 step 5 took `moe` into fn-world as
// `x::moe`. It was `moe/dsv4_routing.cu`'s one launcher in Rust:
// DeepSeek-V4's hash router, expert INDICES gathered from a `[vocab, K]`
// checkpoint table keyed by token id, expert WEIGHTS still
// `sqrt(softplus(logits))` read at those indices. Classified
// `Execution::Walk` with `Control::Supplies` — the table and its first
// extent are what no `Source` names — and its row had no generated arm,
// because all eleven of its `table::moe` operands were unsourced.
//
// `x::moe::hash_route_lookup` is the host program and the symbol is a
// `none:` arm: `Cx` cannot be asked whether the deployment renormalises its
// top-k or by what factor it scales, which is the one thing between it and a
// bind. `x/moe.rs`'s header states the two-line patch.
/// What is LEFT of `attn/kv_paged.cu`'s host side: the cell move
/// `serve::transfer` fires, and the two page-view builders the driver's own
/// plan building consumes.
///
/// **The seven appenders and dequanters MOVED** to
/// `kernels_cuda::attn::kv_paged`, with `fp4_block_size`,
/// `max_touched_pages`, and `Fp8Kind` as `fp8_kind_of` over the floor's own
/// `x::fp8_kind`. The four `Launched`/`Declined` enums did NOT go with them. The head of
/// [`kv_paged`] states the discriminator that decided it — a driver op is a
/// symbol whose body needs a driver RESOURCE, and none of the seven does —
/// and the measurement that let the enums go: all ten call sites consumed
/// their return with `let _ =`, so `Fired` says strictly more than anything
/// read.
///
/// What remains here is the `TryFrom<&KvCacheLayerView> for KvLayer` those
/// call sites now go through, and the three survivors above.
///
/// **The `Specialisation`s that this doc used to argue about are gone.**
/// §58 read `a_walk_is_only_a_walk` against `Specialisation::agrees` and
/// concluded a specialised symbol wants no `Walk`, no `RUST_SERVED` entry
/// and no `bind::service` shim; §60.6 dissolved that by moving the device
/// rows to `..._dev` names. Both were reasoning about a mechanism no reader
/// consulted: this module had been picking `#hnd`/`#nhd` in Rust and firing
/// by name through `hand::fire` the whole time, so `selects()` was asked
/// about none of the five. `pie::SPECIALISED` is empty and terminal.
pub mod kv_paged;
/// The fused LM-head GEMV + argmax IS GONE — `sample/argmax.cu`'s last
/// launcher, and the whole of `kernels/sample/`. Two JIT'd kernels with a
/// growable device scratch between them, on grids read off an occupancy
/// query.
///
/// §5 step 5 took `sample` into fn-world: the program is
/// `kernels_cuda::sample::lm_head_gemv_argmax_int8`, beside the two
/// declarations it fires. It is no longer reached through `bind::service`
/// and no model text sees the symbol — the contract carries a written
/// refusal instead, because the int8 head and its dequant scale are named
/// weights no statement names. The `fn` is public and complete.
pub mod lora;
/// `attn/mla_paged.cu`'s two launchers, in Rust — the whole file, which is
/// DELETED. The MLA prologue (`kv_a` RMSNorm, `k_pe` rotation, paged write,
/// query nope/pe split) on a `dim3(total_tokens, 1 + q_blocks)` grid whose
/// second axis the host computes, and the paged append. Both classified
/// `Execution::Walk` with `Control::Supplies`, both in
/// `execution::RUST_SERVED`, both reached through `bind::service`. The third
/// function that file held, `write_mla_to_pages_bf16`, had an empty consumer
/// set on all five channels and is deleted rather than ported.
// `mla_paged` DELETED -- both host programs crossed into
// `kernels_cuda::attn` and refuse on `Cx::mla_layer`.
// `attn/mla_paged.cuh` is `x::attn::mla_paged`'s unit and holds both kernels;
// `YarnOriginal` dissolved into `x::Yarn` and `MlaDecline` into
// `Refusal::Empty`.
// `mla_naive` DELETED -- `attention_mla_naive.cuh`'s two launchers crossed
// into `kernels_cuda::attn::mla_naive`, which already held the root
// and both template-ids. `NaiveShape`, `NaivePtrs`, `NaivePlan`, the
// wave-target head-group search, `mma_supported` and both shared-memory
// figures went verbatim; `launch` became `x::attn::mla_naive::fire` over
// `Ctx` instead of `hand::fire` over a stream, so a compile failure is a
// `Refusal` and no longer a panic. `MlaNaive` still says which of the pair
// ran, and a shape neither serves is still `Declined` rather than an error.
//
// Nothing in this crate called it before the move and nothing can call it
// after: `fire/launch.rs`' `kv_pools_for` refuses `KvStyle::Mla`, so the
// arm choice this module was waiting on -- `cudaDevAttrComputeCapability\
// Major >= 10`, now answerable as `Ctx::compute_capability_major` -- has no
// caller to make it.
// `fire::moe` IS GONE — §5 step 5 took `moe` into fn-world as `x::moe`. It
// held the routing/gemv/finalize launchers with no doc block of their own;
// every one of them is a host `fn` in `x/moe.rs` now, under the `csrc` root
// that owns its `__global__`.
// `fire::moe_dispatch` IS GONE — §5 step 5. It was `moe/moe_dispatch.cu`'s
// launchers in Rust, ALL EIGHT: the two decode GEMVs, the MXFP4 group-scale
// relayout, the aligned pointer builder, the route-order scatter, the exact
// counting sort, the per-route expert-bias add and the weighted
// scatter-accumulate. The last three landed after this module's header had
// spent a section arguing they could not: they are unit-hosted AND their
// `table::moe` rows are unsourced, which barred `JIT_DISPATCHED` and forced
// a `_dev` symbol split. Both halves of that survive the port and they no
// longer cost the same thing. The `_dev` names STAY, because seven `moe`
// symbols are still `execution::Walk` classifications and
// `a_walk_is_only_a_walk` means a walked symbol may not also be a unit row
// — but the split is now invisible to every caller: the device name is a
// `unit!` instantiation string, the stated name is a `contract!`, and the
// host `fn` between them is the only thing either reader calls. The
// unsourcedness is what it always meant: `none:` arms, `Route::Unbound` at
// model load with a sentence, and `Control::Supplies` still the reason.
//
// `kernels/moe/` is left holding `flashinfer_moe.cu`, which is an
// `extern "C"` INSTANTIATION SEAM and nothing else — five functions over
// `CutlassMoeFCRunner` and two standard headers, down from fourteen. Its
// 817-line host program (workspace arithmetic, arch probe, autotuner,
// tactic caches, dispatch) is `fire::flashinfer_moe`, WHICH STAYS: it had
// zero `__global__` to move because it never had any, and
// `moe::flashinfer_cutlass_moe_bf16` is `x::moe`'s one driver op — a
// `contract!` with no `Entry`, the third registration shape.

/// `moe::build_moe_ptrs_aligned_bf16` — the aligned MoE leg's pointer build,
/// and `x::moe`'s SECOND driver op. It became one because the six device
/// pointer arrays it fills have no stated consumer: the batched-cuBLAS
/// fallback that reads them is a lowering of `moe::moe_grouped_gemm_bf16`,
/// not a statement, so declaring them as trace results would hand the plan
/// six values liveness frees at the next op. The file's header carries the
/// measurement; the short version is that this is the gate on retiring the
/// fused CUTLASS leg, because the aligned leg is the only leg left and it
/// cannot start without this call.
pub mod moe_ptrs;

/// `moe::moe_grouped_gemm_bf16` — the aligned MoE leg's two grouped GEMMs,
/// and `x::moe`'s THIRD driver op. One symbol with two implementations: the
/// WMMA kernel inside `x::moe::supported`'s rectangle, and the batched cuBLAS
/// call outside it, over the arrays `moe_ptrs` built. It is a driver op
/// because the second implementation needs the cuBLAS handle and those
/// arrays, and because `bind/mod.rs`'s "a refusal is not a fallthrough" makes
/// a bind that declines `K = 2048` the final answer rather than the first
/// half of a choice.
///
/// Gated on `_cuda` rather than carrying per-item gates: every item in it
/// calls the device, so the module is the unit.
#[cfg(feature = "_cuda")]
pub mod moe_grouped;

/// `attn/page_compact.cu`'s one launcher, in Rust — the whole file. Two
/// launches on one stream, the second reading the `scratch_counts` buffer the
/// first fills; `execution::COMPOSED` has stated that pair since the split
/// and what was missing was the host. `RUST_SERVED` takes the row.
// `page_compact` DELETED -- `compact_page_csr` crossed into
// `kernels_cuda::attn`, both launches in order with both refusals
// hoisted ahead of the first. `attn/page_compact.cuh` is
// `x::attn::page_compact`'s unit, written for that crossing.
// `qkv_fused` DELETED -- `attn::qkv_decode_qk_norm_rope_write_kv_bf16` crossed
// into `kernels_cuda::attn` as `QKV_DECODE_FUSED`, and it was THE LAST
// ROW IN `ROW_TABLES`.
//
// It was the whole launcher here, and the last `attn` dispatch to fall: one
// program over four kernels and eight instantiations, `head_dim` picking the
// warp form at 64, 128 or 256 and falling THROUGH to the block form
// otherwise, `rope_table != nullptr` picking the `USE_ROPE_TABLE` arm inside
// whichever it picked. No `Specialisation` can state that, because the
// fallthrough changes the `LaunchRule` from `WarpPackedHeads` to
// `RowsPackedHeadsNarrow` and `Specialisation::agrees` forbids an arm that
// changes the rule; `families/attn.rs` had written the refusal and added that
// lifting it would not help, *because `head_dim == 64 | 128 | 256` is still
// unspellable*. **A host program spells it**, and that is the whole of what
// §5 step 5 is.
//
// This module survived the body's move for exactly one commit as the cast
// from the generated dispatch's `*const c_void` to the types the `unit!`
// declares. The `bind!` arm resolves those from `Cx` directly, so the
// generated dispatch is gone, `bind::service`'s entry point is gone, and a
// cast with no caller is not a seam. **Name the driver resource or it is a
// move** — there was none to name, and `_ctx: &DispatchCtx` went unread from
// the day it was written.
pub mod page_mask;
// `predicate_of` LIVED IN `recordings` and does not any more, and the gate
// below is the whole reason. It is the one thing in the union machinery that
// touches no device object — `lower::select`'s body, evaluated against a
// fire's rows — and its own doc comment insists on that: *"the equivalence
// between the eager leg and the captured leg is a HOST fact, so it must be
// provable without a GPU."* The proof is `tests/union_lower.rs`, which gates
// on `_cuda` because that is all it needs. Behind `abi` the function was
// still host-only and no longer reachable from the target that proves it, so
// the gate had quietly made the doc false.
pub mod predicate;
// `recordings` and `scratch` are BEHIND THE SAME GATE AS `launch`, because
// every reader of either is: `fire::launch` and `serve` are both
// `feature = "abi"`, and nothing outside them constructs a `Scratch`, reads a
// `slot::` name or holds a `Recordings`. Ungated they compiled into a build
// that could not reach them, and rustc said so correctly — sixteen dead-code
// warnings naming a live subsystem, which is the shape of a warning nobody
// can act on and everybody learns to skim.
#[cfg(feature = "abi")]
pub mod recordings;
// `split_packed` DELETED. `attn::split_qkv_bf16_devwin` crossed into
// fn-world with a real bind, so its host program is
// `kernels_cuda::attn::split_qkv_bf16_devwin` and the module that
// held it had nothing else in it.
#[cfg(feature = "abi")]
pub mod scratch;
pub mod sideband_arena;
pub mod stage_hooks;
/// `csrc/supergraph.cu`'s two launchers, in Rust — **that file is deleted**,
/// and with it the second of this crate's three nvcc builds. The device text
/// is a JIT unit now (`kernels-cuda/kernels/graph/supergraph.cuh`); the
/// claim that stood in the way — *"this needs nvcc"* — was measured and is
/// false, and the measurement is in this module's header.
pub mod supergraph;
// `xqa` DELETED. It was two host programs and the attention-workspace carve
// between them, kept here on the argument that a workspace is the driver's
// vocabulary. It is not: a routine takes the workspace's two halves as four
// arguments, and then the carve is arithmetic over a pointer like any other.
// Both launches are `kernels_cuda::attn::xqa`'s
// `attention_xqa_decode_bf16_prepared`, in that order on one stream, which is
// what retired the `Prepare::FireWide` obligation nothing here discharged.
// `bind/arms/xqa.rs` is the whole of what is left on this side.
