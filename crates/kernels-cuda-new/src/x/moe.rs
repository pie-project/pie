//! `moe` in fn-world — nine device roots, six units, twenty host programs.
//!
//! # What this file is
//!
//! A kernel has exactly two truths: the device text (`.cuh`) and the host
//! program (a Rust `fn`). This file is the second, beside the declaration
//! that carries the first. `families::moe`'s 1,623 lines of rows and
//! `table::moe`'s 520 lines of contracts collapse into one `unit!` per root,
//! one `fn` per launcher, one `contract!` and one `bind!`; the three
//! `driver-cuda/src/fire/` modules that hosted the walked half —
//! `fire/moe.rs`, `fire/moe_dispatch.rs`, `fire/dsv4_routing.rs` — move here
//! whole, because a host program that fires a JIT kernel by symbol has no
//! reason to live in the driver.
//!
//! **Both of those files are DELETED**, and `table/moe.rs` outlived this one
//! by a round for a reason worth keeping: after its twenty `moe::` rows left,
//! it still held four `quant::` ones — the routed MoE decode GEMVs — because
//! **`table/` was organised by who DISPATCHES and `x/` by who owns the
//! code**. It survived as `quant`'s tenant, not as `moe`'s remainder. Those
//! four are `x::quant` contracts now, `table::moe` is gone, `pub mod moe`
//! with it, and `table::ROW_TABLES` is `attn` alone.
//!
//! `fire/flashinfer_moe.rs` does NOT move, and the "one driver op" section
//! below is why.
//!
//! # Nine roots, six units, and the three that carry no unit
//!
//! `csrc/src/moe/` holds nine `.cuh` files. Six of them compile under this
//! crate's NVRTC today and each gets an inline `pub mod` with one `unit!` —
//! **`unit!` emits `UNITS`, `ROWS`, `PARAMS` and `mod raw` at module scope,
//! so two invocations cannot share one**, and the module is also what gives
//! `raw::` a qualifier worth reading. `x/layout.rs` found this idiom with
//! five roots; the note is repeated per multi-root family because it is the
//! first thing a reader of one hits.
//!
//! ```text
//!   topk_sigmoid.cuh            unit, 1 row     mod topk_sigmoid
//!   dsv4_routing.cuh            unit, 2 rows    mod dsv4_routing
//!   topk_softmax.cuh            unit, 3 rows    mod topk_softmax
//!   moe_dispatch.cuh            unit, 14 rows   mod moe_dispatch
//!   moe_grouped_gemm.cuh        unit, 1 row     mod moe_grouped_gemm
//!   expert_offsets.cuh          unit, 4 rows    mod expert_offsets
//!   moe_grouped_gemm_tile.cuh   TEXT ONLY       — compiler floor
//!   moe_fused_tile.cuh          TEXT ONLY       — negative result
//!   topk_softmax_tile.cuh       TEXT ONLY       — compiler floor
//! ```
//!
//! The last three are `cuda::tiles` kernels. They are carried on disk, they
//! are measured, they are gated by `tests/upstream_manifest.rs`, and they are
//! not units because **a unit is a claim that this crate's compiler can
//! compile it** and that claim is false at the system `libnvrtc`. Their
//! measurements are below, in full, because a port that consumes a
//! measurement is a regression even if it compiles.
//!
//! # `csrc/src/moe/` has no launchers left, and ten symbols are `_dev` for it
//!
//! `moe/moe_dispatch.cu` and `moe/dsv4_routing.cu` are deleted with their
//! headers. Everything below that reads as *"the launcher does X"* is history
//! about a file, kept in the present tense of the `<<<>>>` it transcribes,
//! because a transcription that starts hedging stops being checkable.
//!
//! Several device rows carry `_dev` symbols — `scatter_add_weighted`,
//! `moe_bucket_exact`, `add_moe_route_bias`, `transpose_expert_scales`,
//! `build_moe_ptrs_aligned`, `hash_route_lookup` and the four
//! `expert_offsets` phases — because their ABI symbols WERE
//! `Execution::Walk` classifications in `execution.rs` and
//! `a_walk_is_only_a_walk` refuses a walked symbol that a unit also hosts.
//!
//! **That law has gone and the names stay.** `moe`'s ten `WALKED` entries
//! are deleted: §5 step 5 moved every one of their host programs into this
//! file, and a description of a control flow the reader can now read has no
//! reading consumer left. Nothing keeps these symbols out of a unit any
//! more.
//!
//! The names stay because renaming buys nothing and costs a sweep. The
//! device name is a `unit!` instantiation string, the stated name is a
//! `contract!`, the host `fn` between them is the only thing either reader
//! calls, and `unit_of`, `Unit::row`, `x::fire::fire` and
//! `examples/unit_probe_moe.rs` all resolve by string.
//!
//! WORTH KNOWING WHY THE SPLIT OUTLIVED ITS REASON, because it is the sharp
//! end of that deletion and `execution.rs` argues it where the entries
//! stood: eight of the ten walks were stale for two edits and passed every
//! mechanical check, and this suffix is why. It existed to let a walk sit
//! beside a unit, and it went on doing exactly that after the walk stopped
//! being true. **A name chosen to keep two things legal beside each other
//! keeps them legal after one of them stops being true.**
//!
//! # THREE driver ops, and each one arrived by a different road
//!
//! The first was always one, the second arrived by re-reading a refusal, and
//! the third was a working bind that gave the shape up because half of its
//! shapes have no kernel. Three roads to one registration shape, and the
//! third is the only one in the tree that ran BACKWARDS — see
//! `MOE_GROUPED_GEMM`'s contract.
//!
//! ## `moe::flashinfer_cutlass_moe_bf16`
//!
//! `csrc/src/moe/flashinfer_moe.cu` is still C++ and is the last
//! ahead-of-time compile in the tree. Its host program reaches
//! `CutlassMoeFCRunner<bf16, bf16>` through five `extern "C"` seams because
//! that class template has no Rust spelling — *"Rust cannot name it, cudarc
//! cannot reach it, NVRTC cannot compile it"* — and the finding is about the
//! **runner**, not about the kernels.
//!
//! So this symbol is a DRIVER OP and not a bind: its body needs a device API
//! surface `Cx` does not have and must not grow. It gets a `contract!` entry
//! and **no `bind!` arm** — the third registration shape in `x/mod.rs`'s
//! table, the one `x/adapter.rs` is the worked example of. A `none:` arm here
//! would be wrong in the one way §5.1 names: something else already fires the
//! symbol, and `Route::Unbound` would refuse a live model at load.
//! `dsl::moe_fused_cutlass` is stated at
//! `crates/model/src/qwen_3_5/forward/mod.rs:362`.
//!
//! `driver-cuda/src/fire/flashinfer_moe.rs` stays exactly where it is and
//! becomes that op's body. `moe/expert_offsets.cuh` — the routing front-end
//! it drives — IS this file's, and its four rows are below.
//!
//! ## `moe::build_moe_ptrs_aligned_bf16`
//!
//! The aligned leg's pointer build, and it reached this shape from the other
//! direction: it was a `none:` arm, and before that an unsourced row, so it
//! has **never had an arm in either world**. Its sentence asked for a dtype —
//! *"an array of device addresses has no dtype in this vocabulary"* — and
//! that ask was wrong in a way worth keeping.
//!
//! The six pointer arrays have **no stated consumer**. What reads them is the
//! batched-cuBLAS fallback inside `moe::moe_grouped_gemm_bf16`, which is a
//! LOWERING of that statement and not a statement of its own; the grouped
//! GEMM's parameter list names no pointer array. Six trace results nothing
//! reads are freed by `lower.rs:1911`'s liveness at the first op past the
//! build, so the dtype would have bought a declaration that is a wrong answer
//! rather than a refusal. Stating them properly means stating an operand only
//! ONE of two lowerings reads — the thing this migration exists to retire.
//!
//! So the arrays stay the driver's arena and the symbol takes the same shape
//! as the fused call above it. Body: `driver-cuda/src/fire/moe_ptrs.rs`. Its
//! call site is `bind/mod.rs`'s driver-op table, beside
//! `pie_lora_qkv_correction`.
//!
//! **This is the gate on retiring the fused CUTLASS leg**, and the reason is
//! one sentence: every condition in `forward/mod.rs:341-349` that turns the
//! fused leg off already returns `moe_mlp_body_aligned_cuda`, so deleting the
//! fused leg does not add a case — it makes the aligned leg the ONLY leg, and
//! the aligned leg cannot start without this call. The pointer build DECLARES
//! `gu_stage`/`act_stage`/`out_stage`, the three destinations every op below
//! it writes into. It is not an optimisation, it is step 3 of 8.
//!
//! ## `moe::moe_grouped_gemm_bf16`
//!
//! Steps 4 and 5, and the only consumer the six arrays have. It became a
//! driver op **from a working bind**, which no other symbol in the tree has
//! done, and the reason is that the bind was only ever half the symbol:
//!
//! ```text
//!   gate_up   M=16  N=2*I=1024  K=H=2048   K > SHORT_K  ->  batched cuBLAS
//!   down      M=16  N=H=2048    K=I=512    supported    ->  WMMA
//! ```
//!
//! `supported`'s `K > 512` refusal names its own replacement — *"above which
//! cuBLAS wins"* — and qwen3.5 decode fires one statement on each side of it.
//! The bind served `down` and refused `gate_up` with `Refusal::Wide`, which
//! `bind::DispatchRefusal::ShapeDeclined` recorded in as many words: *"the C++
//! driver reads the same predicate and takes a batched-cuBLAS fallback; until
//! that fallback exists here, saying so is the only honest answer."* So step 3
//! built six arrays whose only reader was unwritten, and the leg stopped at 4.
//!
//! It could not be finished in the bind, because the batched form needs the
//! cuBLAS handle and the arrays; and it could not be finished by the caller,
//! because *"a refusal is not a fallthrough"* makes the bind's `Wide` the
//! final answer rather than the first half of a choice. Both implementations
//! therefore live behind one host program in
//! `driver-cuda/src/fire/moe_grouped.rs`, which asks `supported` and picks.
//!
//! **The fallback was already in the tree.**
//! `x::gemm::dense::batched_act_x_wt_bf16` is `gemm.cpp:1145-1241` verbatim,
//! grouped-batched falling back to plain batched, with the stream-capture
//! latch that makes the first form safe — ported under §45.2 with its row
//! struck and its doc saying *"its whole consumer set was one unreachable
//! inline"*. What was missing was never the arithmetic. **A body with no
//! caller and a caller with no body sat in two crates for the whole of it**,
//! and the thing that connected them was reading one refusal's own sentence.
//!
//! # What this port needed from the floor, asked for, and got
//!
//! Six lines in three files this family may not edit, and **every one of them
//! was a fact or a type the row world already carried** and `Cx`/`Abi` had not
//! caught up with. **All six landed as `a41a1df0a`**, with the three matching
//! `impl Facts for Fire` methods. Nothing below was written here, and the ask
//! is kept rather than deleted because the ask is the record.
//!
//! **THE TWO HALVES COST DIFFERENT THINGS AND THE READER MUST NOT AVERAGE
//! THEM.** The three `x/cx.rs` lines were LOAD-TIME: without them six live
//! symbols answered `Route::Unbound` with the sentence their `none:` arm
//! carried, and everything else in this file worked. The three `x/abi.rs`
//! lines were COMPILE-TIME: `unit!` derives each row's operand list from the
//! host `fn`'s parameter types through `Abi::TY`, so a parameter whose type
//! has no `impl Abi` is a missing-trait error and **THIS FILE DID NOT COMPILE
//! UNTIL THEY LANDED**. That was stated here, at the top, rather than
//! discovered: the alternative was to spell those five parameters
//! `*mut c_void` and `*mut *const c_void`, which compiles today, produces the
//! very `Ty`s the deleted rows carried — and puts `void*` in the typecheck
//! translation unit where the kernel says `int64_t*` and `const bf16**`. The
//! bypass with no type error anywhere is the thing this migration exists to
//! remove, so the ask was made instead of taken:
//!
//! ```text
//!   x/cx.rs    query!(moe_norm_topk -> bool,     "whether the router renormalises its top-k");
//!   x/cx.rs    query!(moe_routed_scaling -> f32, "the router's routed scaling factor");
//!   x/cx.rs    query!(in_rows(i: usize) -> i32,  "an input's row count");
//!   x/abi.rs   ptr_abi!(i64, "const ::std::int64_t*", I64s, "::std::int64_t*", BufMut);
//!   x/abi.rs   ptr_abi!(*const bf16, "const …device::bf16* const*", BufArrayOut,
//!                                    "const …device::bf16**",       BufArrayOut);
//!   x/abi.rs   ptr_abi!(*mut bf16,   "…device::bf16* const*", BufArrayOutMut,
//!                                    "…device::bf16**",       BufArrayOutMut);
//! ```
//!
//! and the three matching methods on `impl Facts for Fire` landed with them
//! in `driver-cuda/src/bind/facts.rs`, which is where every `Cx` query bottoms
//! out.
//!
//! **The two router constants** are `Source::Ctx("moe_norm_topk")` and
//! `Source::Ctx("moe_routed_scaling")`, which four rows carried and which
//! `emit_rust_dispatch` rendered as `ctx.moe_norm_topk` and
//! `ctx.moe_routed_scaling`. `DispatchCtx` already holds both
//! (`driver-cuda/src/bind/mod.rs:1179`, filled from
//! `model.deployment.norm_topk_prob` at `driver-cuda/src/fire/launch.rs:3435`),
//! so the `Facts` methods are `Some(self.ctx.moe_norm_topk)` and
//! `Some(self.ctx.moe_routed_scaling)` — **always `Some`**, because they are
//! deployment constants with defaults rather than optional facts, and a
//! refusal would invent an absence. With them `topk_sigmoid`,
//! `topk_sqrtsoftplus`, `topk_sigmoid_bias` and `hash_route_lookup` BIND;
//! three of the four fire today. `x/layout.rs` got four `Cx` queries this way
//! and the process is the same one.
//!
//! **`in_rows`** is `Source::InRows(1)`, which the aligned path's gather and
//! reorder both read for `aligned_rows` — the padded rectangle's height, an
//! operand's own extent that no width, param or context field carries. The
//! driver computed it for every generated arm already (`rows_of(b, i, rows)`
//! in `bind/mod.rs`), so the `Facts` method is that expression with the index
//! bound-checked — `(i < self.spec.n_in).then_some(self.rows)`, because
//! `Cx::in_rows` refuses with `Refusal::Absent` and an out-of-range index IS
//! an absent operand. **Reproducing the row exactly is how the two binds
//! surfaced that `rows_of` ignores its index**, so `Source::InRows(1)` and
//! `Source::Rows` have always rendered the same number; the gather's arm
//! states it where two adjacent arguments make it legible.
//!
//! **The `i64` pointer** is `hash_route_lookup`'s `tid2eid`, a `[vocab, K]`
//! `const int64_t*` table. `kernels::Ty::I64s` exists and `I64sMut` does not,
//! so the mut spelling reuses `BufMut` exactly as `bf16` and `f16` do.
//!
//! **The two pointer-array impls** are `build_moe_ptrs_aligned`'s six
//! operands, `const T**` and `T**` at `moe_dispatch.cuh:1046-1051`. The
//! pointee is the *pointer*, which is why these are `ptr_abi!(*const bf16,
//! …)` and not an impl on `*mut *const c_void`: the declaration's `CPP` is
//! the DEVICE parameter's spelling and the device parameter is `const
//! bf16**`, while `Ty::BufArrayOut::cpp()` is `const void**` because that was
//! the deleted C launcher's. `ptr_abi!(bf16, …, Bf16s, …, BufMut)` already
//! carries exactly that split, so this is the established shape and not a new
//! one. Spelling any of the six as `*const c_void` instead would compile, put
//! `Ty::Buf` where `Args::bind` checks, and put `const void*` in the typecheck
//! translation unit where the kernel says `const int64_t*` — the bypass with
//! no type error anywhere that `x/xqa.rs`'s header names.
//!
//! # The wmma call sites, which is why this family waited
//!
//! `moe_dispatch.cuh` and `moe_grouped_gemm.cuh` are the two `wmma` users in
//! the tree. NVRTC 13.0 refuses `mma.h` outright — *"could not open source
//! file 'mma.h'"* — so until `pie_mma.cuh` existed and was proved
//! bit-identical to `nvcuda::wmma` on an L40S, no unit carrying either file
//! could compile at all. Both call sites want the one shape it implements:
//! 16x16x16, bf16 x bf16 -> f32, A `row_major`, B `col_major`, store
//! `mem_row_major`. `examples/unit_probe_moe.rs` instantiates all three wmma
//! kernels by hand, which is the only way the shim's coverage of this family
//! is a measurement rather than a reading.
//!
//! # `moe/moe_grouped_gemm_tile.cuh` — text, and the reason is the floor
//!
//! The same GEMM written in `cuda::tiles` — one `__tile_global__`, no
//! launcher, no `wmma`, no CUTLASS. It is correct (worst relative error
//! **0** against an fp64 reference at every shape and tiling swept, with the
//! padding blocks' poison bytes untouched, so the early exit is measured
//! rather than claimed) and it is **faster than either kernel pie fires for
//! MoE decode**.
//!
//! At `kTileM = 16`, which is what `moe_align_decode` emits and the only
//! height `supported` accepts (`M == kFrag`), on the decode census of 318
//! aligned blocks with 106 live:
//!
//! ```text
//!                       gate_up            down
//!                       N=512 K=2048       N=2048 K=256
//!    cutile (best)      0.324 ms           0.149 ms
//!    wmma twin          0.858              0.214
//!    cuBLAS, captured   0.972              0.449
//!    cuBLAS, ideal      0.327              0.177
//! ```
//!
//! `cuBLAS, ideal` batches the 106 LIVE blocks and is unattainable: the batch
//! count is a host argument baked into a captured graph and must be the worst
//! case, which is the 318-block row. So against what pie can actually run the
//! tile kernel wins by 3.0x at both shapes — and it now beats the
//! unattainable ideal too, which says the early exit is not merely cheap but
//! free.
//!
//! Those figures moved once already and by a lot. Until §23.20 they read
//! 0.349 and 0.185, measured on a kernel whose extents and loop bounds were
//! run-time values; making them compile-time constants, as NVIDIA's own
//! `matmul.cuh` does, is the whole difference. N and K are model constants
//! and a JIT instantiates per shape, so this costs nothing but saying so.
//!
//! **An earlier version of this comment said the opposite**, reporting the
//! tile kernel 1.93x slow at gate_up and 1.21-1.31x slow at down, and
//! locating the limit at 214 registers against the twin's 40. Those numbers
//! were real but they measured a workaround, not the kernel: bf16 was being
//! carried through as `unsigned short` and widened to fp32 inside the tile,
//! which doubled the operand register footprint and bypassed the bf16
//! tensor-core path. The widening existed because 16-bit float tiles appeared
//! not to compile, and they appeared not to compile because the runtime
//! headers were CUDA 13.0 under a 13.3 tile compiler.
//! `.wiki/driver/cutile-16bit-header-trap.cu` is that story; the kernel's own
//! header carries the full before-and-after.
//!
//! Re-measured since the rewrite, and it does not all go one way. The CUTLASS
//! island (`flashinfer_cutlass_moe_bf16`, which fuses permute + fc1 +
//! activation + fc2 + finalise) was timed directly rather than quoted, at 318
//! tokens / hidden 2048 / inter 256 / 256 experts / top-k 8:
//!
//! ```text
//!                                    island       two tile GEMMs
//!                                    (5 stages)   (2 stages)
//!   256 experts, 16 rows each        1.241 ms     1.328 ms   kTileM=16
//!   106 experts, 24 rows each        0.581        0.654      kTileM=32
//! ```
//!
//! The GEMM gap closed — a previously recorded 1.9x is now 7% at the first
//! census and **12%** at the second, where the island does five stages
//! against two. Fusing the three stages into one kernel measures 0.984 ms at
//! this census, worse than not fusing — but that is a statement about the
//! GRID and not about fusion: 106 blocks on a 142-SM part is under one block
//! per SM. Past 212 blocks the fused kernel wins, and at 1,696 it is 2.3x
//! ahead of the unfused pair. The island stays ahead of the best tile option
//! at both ends, 1.13x here and 1.41x at 54,272 routed rows, because fewer
//! experts over the same rows means more rows per expert, and this kernel
//! reads `W[e]` once per block and round-trips the intermediate through HBM
//! where the island reads each expert once and keeps fc1's output resident.
//!
//! So the CUTLASS dependency is not removable on these numbers, which was the
//! original question. What changed is that it is now a question about one
//! fused kernel rather than a class of them.
//!
//! # `moe/moe_fused_tile.cuh` — text, and a negative result
//!
//! That file answers the question the section above leaves open. It writes
//! fc1 + swiglu + fc2 as one `__tile_global__`, one expert block per CUDA
//! block owning the whole fc2 output panel, intermediate never stored — the
//! island's access pattern stated directly. It compiles, and it is correct to
//! 0.42% worst relative error on positive data, which is 2^-8 and therefore
//! the bf16 rounding floor.
//!
//! ```text
//!   island   permute+fc1+act+fc2+finalise   0.581 ms   573 GB/s
//!   two unfused tile GEMMs                  0.933      ~358
//!   the fused kernel, best of the sweep     1.778      187
//! ```
//!
//! Fusing made it twice as slow as not fusing. The cause is shared memory
//! rather than registers: the tile compiler stages `partition_view` loads
//! through it, and the fused working set takes 92-99 KB of a 100 KB budget
//! against the unfused grouped GEMM's 16 KB. That is one block per SM, and
//! 106 to 318 blocks each alone on an SM cannot hide HBM latency.
//!
//! It is pinned there — `FNK`, `FN2` and `FM` sweeps never drop below 92 KB —
//! and `cuda::tiles` exposes no shared scratch, no `insert` and no occupancy
//! control, so there is no third version to reach for. The file is carried,
//! with a banner that says all of this, because it looks exactly like a
//! kernel someone should finish and it is not.
//! `tests/upstream_manifest.rs` gates the banner.
//!
//! # `moe/topk_softmax_tile.cuh` — text, and it beats the warp ladder
//!
//! Measured against `topk_softmax`'s block form at 256 experts: **4.52 us
//! against 6.23 at one row and 4.65 against 6.52 at 128**, with identical
//! expert indices, crossing over at about a thousand rows. A tile does not
//! care how wide the router is, which is the whole reason the warp ladder
//! exists. It does not retire the ladder: the alternative needs a toolchain
//! this crate does not load, so the block-form row and its five unrowed rungs
//! stay.
//!
//! # The compiler floor, which is the actual refusal for all three
//!
//! A unit is a claim that **this crate's compiler** can compile it, and that
//! claim is false today:
//!
//! * the crate loads the system `libnvrtc`, 13.0.88 here, and CUDA 13.0 ships
//!   the `__tile_global__` macro with no tile API behind it (`cuda_tile.h` is
//!   60 lines and declares `print`). `tests/units.rs` compiles every declared
//!   unit with that compiler, so a unit added today would fail the gate for
//!   the whole crate;
//! * the runtime headers must be 13.3 or newer for any 16-bit float tile to
//!   compile at all, per the trap above;
//! * `tileiras` must be on `PATH` with `CUDA_ROOT` set, to assemble what
//!   NVRTC returns.
//!
//! All three are pip wheels. None is a wait on NVIDIA or on a driver.
//!
//! **The JIT path itself works on this box's 13.0 driver**, which two earlier
//! versions of this comment denied. Measured end to end with a bf16 tile
//! `mma`: nvrtc compiles, `nvrtcGetTileIR` yields 6,314 bytes, `tileiras`
//! assembles it, `cuModuleLoadData` and `cuModuleGetFunction` and
//! `cuLaunchKernel` all succeed, and the result is exact. NVRTC returns pure
//! Tile IR — `.note.nv.tkinfo`, no `.text` — and the driver loads that image
//! without assembling it, so stopping there gives `NOT_FOUND`; running it
//! through `tileiras` yourself closes the gap. Cold latency is 0.62-0.71 s,
//! of which `tileiras` is 0.18 s, which is what `PIE_HOME/cache` is for.
//!
//! `tileiras` requires `CUDA_ROOT` and does not say so — without it every
//! input fails with a bare `failed to compile Tile IR program`, including
//! nvcc's own `.tilebc`. See `.wiki/driver/new-horizon.md` §23.18.
//!
//! When those wheels are packaged the unit is three lines and its
//! `Unit::opts` are already known and each load-bearing: `-std=c++20
//! -enable-tile -default-device`. `kTileN`/`kTileK` want to be set per row —
//! 32x128 at gate_up, 32x32 at down — which is what `opts` is for.
//!
//! # FOUR `none:` arms, and NOT ONE OF THEM FIRES
//!
//! A `none:` says the symbol is declared, callable as a `fn`, and unsourceable
//! from a statement — it surfaces as `Route::Unbound` at model load with the
//! sentence written beside it. Four of this family's twenty contracts have
//! one, thirteen bind, and the other THREE are driver ops. **This is the
//! number to read the port by**, and it was ELEVEN for one round:
//!
//! ```text
//!   FOUR that never fired, and still do not -- no `Source` on at least one
//!   operand, so `emit_rust_dispatch` skipped the row whole and wrote no arm,
//!   and no hand arm in `driver-cuda/src/bind/mod.rs` names any of them:
//!
//!     add_moe_route_bias_bf16    `out_stride` is a PITCH, not an extent
//!     transpose_expert_scales_u8 the statement has NO inputs at all
//!     moe_bucket_exact           a caller -- the third result is declared now
//!     scatter_add_weighted_bf16  the launch count is a device readback
//!
//!   A FIFTH stood here and left the list UPWARD, not by a floor patch:
//!   `build_moe_ptrs_aligned_bf16` said "an array of device addresses has no
//!   dtype", and the answer turned out not to be a dtype at all. The six
//!   arrays have no stated CONSUMER -- the batched-cuBLAS fallback that reads
//!   them is a lowering of `moe_grouped_gemm`, not a statement -- so
//!   declaring them would hand the plan six values `lower.rs:1911`'s liveness
//!   frees at the next op, and the GEMM would dereference bytes the next
//!   allocation took. It is a driver op now, body in `fire/moe_ptrs.rs`. Of
//!   the eleven it is the only one that left by being RE-READ, and it is the
//!   one that mattered most: it is the gate on retiring the fused CUTLASS
//!   leg, because that leg is the only one qwen3.5 decode takes and every
//!   condition that turns it off already falls to the aligned leg -- which
//!   cannot start without this call.
//!
//!   Each arm's sentence now ends in WHAT WOULD MAKE IT FIRE, and the four
//!   answers are four different things -- a pitch, a caller from the weight
//!   loader, a caller from anywhere, and a device-side bound. Two
//!   of the four said the wrong thing until the binds next door made the
//!   reachable numbers obvious: `add_moe_route_bias` and `moe_bucket_exact`
//!   both blamed the ROUTE COUNT, which is `rows * in_width` for each of
//!   them exactly as it is for `moe_align_decode`.
//!
//!   AND ONE OF THE FOUR MOVED WITHIN THE DAY. `moe_bucket_exact` read
//!   "two results declared, three buffers written" until `dsl.rs:5121` was
//!   given its third, in the kernel's own parameter order; what is left is
//!   that nothing calls it. The gap it named was real and is closed, which
//!   is the difference between a `none:` arm that records a defect and one
//!   that records a wait.
//!
//!   SIX MORE were `none:` when this file first landed, blocked on three `Cx`
//!   queries that did not exist. `a41a1df0a` landed all three and they are
//!   binds below -- kept here as the record of what a `none:` costs while it
//!   stands, because five of the six were LIVE:
//!
//!     topk_sigmoid_bf16          FIRES (kimi_k2, kimi_k3, glm_5)
//!     topk_sigmoid_bias_fp32     FIRES (nemotron_h)
//!     topk_sqrtsoftplus_bf16     FIRES (deepseek_v4)      } moe_norm_topk +
//!     hash_route_lookup          no caller: see below     } moe_routed_scaling
//!     gather_moe_aligned_inputs_bf16   FIRES (qwen_3_5)   } in_rows(1)
//!     reorder_moe_aligned_output_bf16  FIRES (qwen_3_5)   }
//! ```
//!
//! **`hash_route_lookup` DOES NOT FIRE, and the first version of this list
//! said it did.** `dsl::cuda::hash_route_lookup` exists
//! (`model-compiler/src/dsl.rs:4826`) and `crates/model/src` calls it from
//! nowhere; the only thing that would is `deepseek_v4`'s `hash_routed`, which
//! is `false` at both construction sites (`spec.rs:202`, `mod.rs:134`) and
//! read at neither. It is bound anyway — every operand it takes is now
//! sourceable and a bind is the truth about it — but it is bound UNEXERCISED,
//! which is a weaker claim than the other five carry and is said here rather
//! than left to look the same.
//!
//! **FIVE LIVE SYMBOLS REFUSED AT LOAD FOR ONE ROUND**, between this file
//! landing and `a41a1df0a`. That was a regression, and it was stated rather
//! than worked around because the alternative was to invent a fact `Cx` did
//! not have. The precedent is exact and it is the largest one in the tree:
//! `norm` shipped `norm::rmsnorm_bf16`, the most-fired kernel there is, as a
//! `none:` arm naming `Facts::per_head_dim()` (`x/norm.rs:3258`). Each
//! sentence named its own missing query, which is what made the patch six
//! lines rather than an investigation.
//!
//! The four that never fired are the older shape: a `none:` is what they
//! already do, said out loud — the `sample` precedent at
//! `driver-cuda/src/bind/service.rs:815`.
//!
//! # Every refusal in every multi-launch body hoists
//!
//! §5.1: a multi-launch body must resolve every refusal condition BEFORE its
//! first launch, and a refusal that cannot be hoisted because a later
//! kernel's geometry depends on an earlier one's device-side output is not a
//! refusal at all — it is a device-side branch. This family has no such
//! branch. Every guard below is host arithmetic over parameters
//! (`routes <= 0`, `h % 8 != 0`, `num_experts <= 0`) or a pointer-nullity
//! test the host can make without reading device memory. The one predicate
//! that looks like an exception —
//! `reorder_moe_aligned_output_bf16`'s vectorisability fork — chooses BETWEEN
//! two symbols before a SINGLE launch, so there is no "something already went
//! to the device" state for it to strand the caller in.
//!
//! # Sources
//!
//! Every geometry below is quoted from the `<<<>>>` it came from or from the
//! `driver-cuda/src/fire/` function that already stated it, and every quote
//! carries its file and line. Nothing here is inferred from a kernel body.
//! Where the row world's `LaunchRule` was the only statement of a geometry —
//! six of the twenty — the rule's own expression is transcribed from
//! `runtime/launch.rs` and cited by line, because `Launch`'s conveniences are
//! conveniences: a kernel fitting neither `flat` nor `per_row` writes the
//! literal.

#![allow(clippy::too_many_arguments)]

use crate::unit::Unit;
use crate::x::abi::bf16;
use crate::x::launch::Launch;

#[cfg(feature = "_cuda")]
use crate::x::contract::{Fired, Refusal};
#[cfg(feature = "_cuda")]
use crate::x::fire::aligned16;
#[cfg(feature = "_cuda")]
use core::ffi::c_void;

// ---------------------------------------------------------------------------
// Truth one, declared: the device text and its instantiations.
//
// SIX `unit!` INVOCATIONS CANNOT SHARE A SCOPE — each emits `UNITS`, `ROWS`,
// `PARAMS` and `mod raw` at module scope. Each root gets a module and the
// family re-exports the six below.
// ---------------------------------------------------------------------------

/// `moe/topk_sigmoid.cuh` — the sigmoid router, one block per token.
pub mod topk_sigmoid {
    use super::bf16;

    unit! {
        /// One kernel, one instantiation: the router a checkpoint with a
        /// sigmoid gate and an optional correction bias uses.
        unit TOPK_SIGMOID = "moe/topk_sigmoid",
            text = include_str!("../../csrc/src/moe/topk_sigmoid.cuh"),
            file = "moe/topk_sigmoid.cuh";

        /// `topk_sigmoid.cuh:` the block form — a token per block, the block
        /// striding the expert axis.
        ///
        /// # The stride constant became `blockDim.x`, and that is why this
        ///   kernel can be launched at all
        ///
        /// It staged its experts with loops stepping a `constexpr` block
        /// width, which pinned each launch to the width its `.cu` happened to
        /// pass — 128 here, 256 for the sibling in `dsv4_routing.cuh` — and
        /// 128 is not a rule. Both now step by `blockDim.x`, which is the
        /// same arithmetic per element at the width the ahead-of-time path
        /// fired and correct at any other. Its static `__shared__` slab is
        /// sized by an expert bound the host refuses to exceed, which is why
        /// widening the BLOCK is safe and widening the ROUTER is not.
        ///
        /// `correction_bias` is nullable and the kernel reads a null as
        /// *"there is none"* — a family without a correction bias states no
        /// fourth operand, and `super::topk_sigmoid_bf16` passes the null
        /// through rather than branching on it.
        fn topk_sigmoid = "moe::device::topk_sigmoid" <T> (
            logits: *const T,
            topk_idx: *mut i32,
            topk_w: *mut f32,
            correction_bias: *const f32,
            e: i32,
            k: i32,
            renormalize: bool,
            routed_scaling_factor: f32,
        ) where *const T, *mut T {
            "moe::topk_sigmoid_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `moe/dsv4_routing.cuh` — DeepSeek-V4's two routers, and they are not
/// variants of each other.
pub mod dsv4_routing {
    use super::bf16;

    unit! {
        /// Two kernels: a sqrt-softplus router that gives a token a BLOCK,
        /// and a hash-table lookup that gives a token a THREAD.
        unit DSV4_ROUTING = "moe/dsv4_routing",
            text = include_str!("../../csrc/src/moe/dsv4_routing.cuh"),
            file = "moe/dsv4_routing.cuh";

        /// The sqrt-softplus router — a token per block, `num_experts` logits
        /// staged in shared memory and reduced.
        ///
        /// `kDsv4MaxExperts = 512` IS a precondition of this kernel and of
        /// this kernel only: it stages logits in a `[kDsv4MaxExperts]` array,
        /// and *"a wider router would overrun the kernel's static shared
        /// arrays"* — the deleted `moe/dsv4_routing.cu`'s own words.
        /// `super::topk_sqrtsoftplus_bf16` carries that refusal.
        fn topk_sqrtsoftplus = "moe::device::topk_sqrtsoftplus" <T> (
            logits: *const T,
            topk_idx: *mut i32,
            topk_w: *mut f32,
            correction_bias: *const f32,
            e: i32,
            k: i32,
            renormalize: bool,
            routed_scaling_factor: f32,
        ) where *const T, *mut T {
            "moe::topk_sqrtsoftplus_bf16" => where [T = bf16] "device::bf16",
        }

        /// The hash-table lookup — one THREAD per token, because the whole
        /// body is a table read and a K-long gather.
        ///
        /// # The indices come from the table; the WEIGHTS still come from the
        ///   logits
        ///
        /// This is the half of the kernel that a name like "hash routing"
        /// hides, and it is the part a reimplementation gets wrong.
        /// DeepSeek-V4 fixes each token's expert SET by a hash of its id, so
        /// `topk_idx` is a gather out of `tid2eid` and the router logits never
        /// choose anything. But `topk_w` is still `sqrt(softplus(logits))`
        /// READ AT THOSE HASHED INDICES, renormalised across `K` when
        /// `renormalize` is set and scaled by `routed_scaling_factor`.
        /// Substituting a uniform `1 / K` — which looks harmless once the
        /// indices are fixed — is a different model, not a faster path to the
        /// same one.
        ///
        /// **`tid2eid` is `*const i64` and needs the `ptr_abi!(i64, ...)` this
        /// file's header asks the floor for.** The row world spelled it
        /// `Ty::I64s`; `x/abi.rs` has no impl yet. Spelling it `*const c_void`
        /// to compile today would put `const void*` in the typecheck
        /// translation unit against the kernel's `const int64_t*`, which is a
        /// silent bypass, and this row would rather not build than lie.
        fn hash_route_lookup = "moe::device::hash_route_lookup" <T> (
            token_ids: *const i32,
            tid2eid: *const i64,
            logits: *const T,
            topk_idx: *mut i32,
            topk_w: *mut f32,
            tokens: i32,
            vocab_size: i32,
            e: i32,
            k: i32,
            renormalize: bool,
            routed_scaling_factor: f32,
        ) where *const T, *mut T {
            "moe::hash_route_lookup_dev" => where [T = bf16] "device::bf16",
        }
    }
}

/// `moe/topk_softmax.cuh` — the softmax routers, three of the file's nine
/// templates.
pub mod topk_softmax {
    use super::bf16;

    unit! {
        /// # The six routers still carried as TEXT, and what blocks each
        ///
        /// The five `topk_softmax_warp_x*` rungs fire ONE warp and reduce with
        /// `__shfl_xor_sync`; the host chose between them at run time on
        /// `num_experts`, which is what a declaration cannot state and what
        /// fn-world moves to a `fn` — not to this unit. `router_topk_softmax`
        /// is the fused-GEMV form and no statement names it. The file's own
        /// header says which and why.
        ///
        /// # THE WARP LADDER, AND WHAT THESE ROWS ARE NOT
        ///
        /// Until `new-horizon.md` §52 the launcher `moe/topk_softmax.cu` chose
        /// between six instantiations at fire time:
        ///
        /// ```text
        /// if (use_warp && K <= 8 && num_experts <= kSoftmaxMaxExperts) {
        ///     if      (num_experts <=  32) topk_softmax_warp_x1 <<<N, 32>>>
        ///     else if (num_experts <=  64) topk_softmax_warp_x2 <<<N, 32>>>
        ///     else if (num_experts <= 128) topk_softmax_warp_x4 <<<N, 32>>>
        ///     else if (num_experts <= 256) topk_softmax_warp_x8 <<<N, 32>>>
        ///     else                         topk_softmax_warp_x16<<<N, 32>>>
        ///     return;
        /// }
        /// topk_softmax<T><<<N, kSoftmaxBlock>>>(...)
        /// ```
        ///
        /// The measurement that motivated the ladder, in the launcher's words:
        /// *"The warp form keeps the experts in registers, so it applies while
        /// they fit (<= 512, which is `kSoftmaxMaxExperts`) and while the K
        /// winners fit the small result array (<= 8). **Qwen3.6-35B-A3B routes
        /// through more than 128 and was falling back to the block form at
        /// 7.56 us/call, 4.9% of its step.**"*
        ///
        /// Two host facts went with the launcher and neither is stated
        /// anywhere yet. **`num_experts > kSoftmaxMaxExperts` THREW**
        /// (`std::runtime_error("topk_softmax_bf16: num_experts exceeds
        /// MAX_EXPERTS")`) rather than returning quietly — the kernel's result
        /// array is sized by that constant and a wider router would overrun
        /// it. And `PIE_TOPK_WARP=0` forced the block form for A/B
        /// measurement; the env read is deleted with the file, which is §30's
        /// rule applied again (a `getenv` may not pick a kernel), but the A/B
        /// it enabled is how the 7.56 us was obtained.
        ///
        /// Re-landing the ladder in fn-world is five rows over the five `_xN`
        /// templates and an `if` chain in `super::topk_softmax_bf16` — which
        /// fn-world CAN say and a row could not. §52 carries the
        /// specification; nothing here is blocked on the floor.
        unit TOPK_SOFTMAX = "moe/topk_softmax",
            text = include_str!("../../csrc/src/moe/topk_softmax.cuh"),
            file = "moe/topk_softmax.cuh";

        /// The per-expert scale fold — `topk_w[i] *= scale[topk_idx[i]]`, over
        /// the flat `[tokens, top_k]` rectangle of routes.
        ///
        /// `n` and `k` were two operands of the ahead-of-time twin and are one
        /// here, because the kernel only ever used their product — and the
        /// product is the weight vector's own element count.
        fn apply_per_expert_scale = "moe::device::apply_per_expert_scale" <T> (
            topk_idx: *const i32,
            topk_w: *mut f32,
            per_expert_scale: *const T,
            total: i32,
        ) where *const T, *mut T {
            "moe::apply_per_expert_scale_bf16" => where [T = bf16] "device::bf16",
        }

        /// The BLOCK form of the softmax router, 64 threads wide.
        ///
        /// The width is the algorithm: `topk_softmax.cuh` carries
        /// `static_assert(kSoftmaxBlock == 64, "block_argmax folds exactly one
        /// upper warp")`, and a launch that widened it would compile, run, and
        /// fold a warp that was never written.
        ///
        /// `act`, `bias` and `hidden` are the FUSED form's operands and the
        /// launch passes two nulls and a zero. They stay because the
        /// `__global__` declares them: a row's operand list is the kernel's
        /// parameter list, and `cuLaunchKernel` reads `sizeof(param)` per cell
        /// off an array whose length nothing else checks.
        /// `router_topk_softmax` is the same body with `FusedGemv` true and is
        /// what reads them.
        fn topk_softmax = "moe::device::topk_softmax" <T> (
            logits: *const T,
            act: *const T,
            bias: *const T,
            topk_idx: *mut i32,
            topk_w: *mut f32,
            num_experts: i32,
            k: i32,
            hidden: i32,
        ) where *const T, *mut T {
            "moe::topk_softmax_bf16" => where [T = bf16] "device::bf16",
        }

        /// DeepSeek's sigmoid routing with the correction bias entering the
        /// RANKING and not the published weight — the fp32 instantiation.
        ///
        /// fp32 and not bf16 because the fp32 entry point is the one a trace
        /// can state: `topk_sigmoid_bias_bf16` existed in the `.cu` and in no
        /// table, so declaring it would name a symbol no model text reaches.
        /// One template, two element types, and the declaration picks the one
        /// with a caller.
        ///
        /// `moe::device::f32` and NOT `device::f32` — the prelude names no
        /// fp32 alias, and `topk_softmax.cuh` declares its own beside the
        /// `Load` specialisation that makes an fp32 router one kernel with the
        /// bf16 one. The unqualified spelling compiles to `namespace ... has
        /// no member "f32"` at the name-map pragma, before any launch.
        ///
        /// `normalize` is `int` and not `bool`, and the difference is not
        /// drift: the deleted C++ HOST function took a `bool` and narrowed it
        /// with `normalize ? 1 : 0`, and the `__global__` declares `int
        /// normalize`. A declaration describes the cubin's parameter list, so
        /// the ternary that lived in the launcher becomes this row's type. The
        /// cell would carry the right value either way — which is exactly why
        /// the wrong spelling here would never be caught by a fire.
        ///
        /// `correction_bias` is read UNCONDITIONALLY by this kernel, unlike
        /// `topk_sigmoid`'s optional one, so there is no null path: this entry
        /// point is the one a checkpoint WITH a bias uses, and a null here is
        /// a fault rather than an absence.
        fn topk_sigmoid_bias = "moe::device::topk_sigmoid_bias" <T> (
            logits: *const T,
            correction_bias: *const f32,
            topk_idx: *mut i32,
            topk_w: *mut f32,
            num_experts: i32,
            k: i32,
            normalize: i32,
            routed_scaling_factor: f32,
        ) where *const T, *mut T {
            "moe::topk_sigmoid_bias_fp32" => where [T = f32] "moe::device::f32",
        }
    }
}

/// `moe/moe_dispatch.cuh` — fourteen of the file's twenty-four templates.
///
/// Six are bf16, two are `i32` and one is `u8`. The `i32` pair is the counting
/// sort: its element type is the routing INDEX rather than an activation, and
/// `moe_dispatch.cuh` says so in a `static_assert(is_same<T, i32>::value, "the
/// routing indices are i32")` that a declaration naming any other type would
/// trip at compile rather than at fire. The `u8` one is the MXFP4 group-scale
/// relayout, where the element is an E8M0 exponent byte.
///
/// **The one `printf` in this family, and it is covered.**
/// `moe_dispatch.cuh:857-860` guards a `printf` with
/// `#if defined(PIE_MOE_ALIGN_REPORT)`, which nothing in the tree defines.
/// It is recorded here because it is the near-miss of the sweep that found
/// `attn`'s `std::memcpy` with no `<cstring>` — nvcc supplied that header
/// transitively and NVRTC does not — and of all 89 carried `.cuh` this was
/// the only other file with a host-library call in its text. NVRTC's implicit
/// preamble declares `printf`, and the guard is never on, so it costs
/// nothing; a future edit that turns the guard into an unconditional call
/// would be relying on the preamble rather than on an include, which is the
/// shape that broke `attn`.
///
/// The file's other ten templates are carried as device text with no row:
/// `build_dual_gemm_ptrs` (`<<<1, 1>>>`), `build_moe_ptrs_decode`
/// (`<<<1, top_k>>>`), `build_moe_ptrs_decode_batched`, the two
/// `moe_decode_wmma_by_*` forms, and the `_vec` twins the host selects by
/// pointer alignment. Nothing declares them because nothing calls them; the
/// `.cuh`'s own header repeats the list beside the kernels it is about.
pub mod moe_dispatch {
    use super::bf16;

    unit! {
        /// The MoE dispatch kernels: the combine, the gather, the counting
        /// sorts, the decode GEMVs and the aligned-path plumbing.
        unit MOE_DISPATCH = "moe/moe_dispatch",
            text = include_str!("../../csrc/src/moe/moe_dispatch.cuh"),
            file = "moe/moe_dispatch.cuh";

        /// `out += weight * src`, one scalar weight over the whole rectangle
        /// — the shared expert's contribution.
        ///
        /// **Declared, compiled, instantiated, and reachable from no
        /// statement.** `weight` is a scalar the deleted arm computed per
        /// fire and nothing in the vocabulary names it; there is no
        /// `contract!` entry below and therefore no `none:` either, because a
        /// `none:` is about a DECLARED symbol a trace can state and no trace
        /// states this one. The device text is what it always was.
        fn scalar_weighted_add = "moe::device::scalar_weighted_add" <T> (
            out: *mut T,
            src: *const T,
            weight: f32,
            n: i32,
        ) where *const T, *mut T {
            "moe::scalar_weighted_add_bf16" => where [T = bf16] "device::bf16",
        }

        /// The combine: `out[n, h] = sum_k weights[n, k] * src[n, k, h]`.
        fn token_batched_weighted_sum = "moe::device::token_batched_weighted_sum" <T> (
            out: *mut T,
            src: *const T,
            weights: *const f32,
            top_k: i32,
            hidden: i32,
        ) where *const T, *mut T {
            "moe::token_batched_weighted_sum_bf16" => where [T = bf16] "device::bf16",
        }

        /// The same combine ACCUMULATING onto `out` — the residual add folded
        /// into the epilogue, which is why its contract states `in_place`.
        fn token_batched_weighted_sum_add = "moe::device::token_batched_weighted_sum_add" <T> (
            out: *mut T,
            src: *const T,
            weights: *const f32,
            top_k: i32,
            hidden: i32,
        ) where *const T, *mut T {
            "moe::token_batched_weighted_sum_add_bf16" => where [T = bf16] "device::bf16",
        }

        /// Gathers token rows into the expert-sorted, block-padded rectangle
        /// the grouped GEMM reads.
        ///
        /// It WRITES the padded rectangle, which is why its launch opens over
        /// the padded rows and its mirror `reorder_moe_aligned_output` opens
        /// over them too — the reorder READS the padded rectangle and writes
        /// route rows, so both grids are `aligned_rows` deep and neither is
        /// the statement's output rows.
        fn gather_moe_aligned_inputs = "moe::device::gather_moe_aligned_inputs" <T> (
            norm_x: *const T,
            sorted_route_ids: *const i32,
            aligned_in: *mut T,
            num_routes: i32,
            aligned_rows: i32,
            top_k: i32,
            hidden: i32,
            shared_row_begin: i32,
            num_tokens: i32,
        ) where *const T, *mut T {
            "moe::gather_moe_aligned_inputs_bf16" => where [T = bf16] "device::bf16",
        }

        /// Adds each route's expert bias onto that route's row, in place.
        ///
        /// One block per ROUTE. It looks like `scatter_add_weighted`'s launch
        /// and differs in the one way that decides it: the value THIS one
        /// writes is the route-major staging, so its rectangle's rows ARE the
        /// launch's routes.
        fn add_moe_route_bias = "moe::device::add_moe_route_bias" <T> (
            out: *mut T,
            bias: *const T,
            topk_idx: *const i32,
            num_routes: i32,
            cols: i32,
            out_stride: i32,
        ) where *const T, *mut T {
            "moe::add_moe_route_bias_dev_bf16" => where [T = bf16] "device::bf16",
        }

        /// The block-padded counting sort: routes to expert-sorted, padded
        /// rows, with the per-block expert id and the inverse permutation.
        ///
        /// ONE BLOCK, whatever the routing — the scan is block-wide and the
        /// counters are in shared memory. *"A grid over rows would run N
        /// copies of the sort, each clearing what the others are reading."*
        /// N copies of a sort do not fail. They produce a permutation, the
        /// GEMMs consume it, and the mixture answers with tokens routed to
        /// experts the router did not pick.
        ///
        /// `T = i32`: `moe_dispatch.cuh`'s `static_assert(is_same<T,
        /// i32>::value, "the routing indices are i32")` is what a declaration
        /// naming any other element would trip, at compile rather than at
        /// fire.
        fn moe_align_decode = "moe::device::moe_align_decode" <T> (
            topk_idx: *const T,
            sorted_route_ids: *mut T,
            expert_ids: *mut T,
            route_to_aligned_row: *mut T,
            num_routes: i32,
            num_experts: i32,
            block_size: i32,
            max_blocks: i32,
            num_tokens_past_padded: *mut T,
        ) where *const T, *mut T {
            "moe::moe_align_decode" => where [T = i32] "device::i32",
        }

        /// The DENSE counting sort — the same block-wide scan without the
        /// padding, producing a compact permutation and the per-expert counts.
        fn moe_bucket_exact = "moe::device::moe_bucket_exact" <T> (
            topk_idx: *const T,
            sorted_route_ids: *mut T,
            route_to_sorted_row: *mut T,
            counts_out: *mut T,
            num_routes: i32,
            num_experts: i32,
        ) where *const T, *mut T {
            "moe::moe_bucket_exact_dev" => where [T = i32] "device::i32",
        }

        /// Folds routed rows back onto the residual stream, each scaled by its
        /// router weight — one block per ROUTED ROW.
        ///
        /// `num_routed` is the GRID and is not a parameter of the
        /// `__global__` at all; the kernel reads its row from `blockIdx.x`.
        /// That is why the contract below is a `none:` and why a rows-shaped
        /// launch would be wrong: the value it accumulates into is
        /// `[tokens, hidden]`, so a grid over the output's rows launches
        /// `top_k` times too few blocks and scatters a prefix of the routes.
        fn scatter_add_weighted = "moe::device::scatter_add_weighted" <T> (
            out: *mut T,
            src: *const T,
            dst_idx: *const i32,
            row_weights: *const f32,
            hidden: i32,
        ) where *const T, *mut T {
            "moe::scatter_add_weighted_dev_bf16" => where [T = bf16] "device::bf16",
        }

        /// The decode gate/up projection: one warp per output tile, one grid
        /// row per route, activation indexed BY TOKEN.
        ///
        /// `expert_stride` is `long long` and the deleted launcher cast the
        /// FIRST factor, so the product is computed in 64 bits — an expert's
        /// gate/up plane is `2 * I_moe * H` elements and at Qwen-3.5's widths
        /// that passes 2^31 well before the expert count does.
        fn moe_decode_gemv_by_token = "moe::device::moe_decode_gemv_by_token" <T> (
            topk_idx: *const i32,
            act: *const T,
            weight_base: *const T,
            out: *mut T,
            top_k: i32,
            k: i32,
            n: i32,
            expert_stride: i64,
        ) where *const T, *mut T {
            "moe::moe_decode_gemv_by_token_bf16" => where [T = bf16] "device::bf16",
        }

        /// The decode down projection — the same body with the activation
        /// indexed BY ROUTE, which is what the `_by_route` in the name is.
        fn moe_decode_gemv_by_route = "moe::device::moe_decode_gemv_by_route" <T> (
            topk_idx: *const i32,
            act: *const T,
            weight_base: *const T,
            out: *mut T,
            top_k: i32,
            k: i32,
            n: i32,
            expert_stride: i64,
        ) where *const T, *mut T {
            "moe::moe_decode_gemv_by_route_bf16" => where [T = bf16] "device::bf16",
        }

        /// The MXFP4 group-scale relayout: `[e][n][kg] -> [e][kg][n]`, one
        /// E8M0 byte per group.
        ///
        /// `num_experts` is NOT a parameter: the kernel reads it as
        /// `blockIdx.z`. Transcribed rather than "completed" — an extra
        /// operand would not bind.
        fn transpose_expert_scales = "moe::device::transpose_expert_scales" <T> (
            src: *const T,
            dst: *mut T,
            n: i32,
            k_groups: i32,
        ) where *const T, *mut T {
            "moe::transpose_expert_scales_dev_u8" => where [T = u8] "device::u8",
        }

        /// Fills the six pointer arrays a pair of batched GEMMs reads, one
        /// thread per padded block.
        ///
        /// **The six arrays need the two pointer-array `Abi` impls this
        /// file's header asks the floor for.** The row world typed them
        /// `Ty::BufArrayOut` and `Ty::BufArrayOutMut`, whose `cpp()` is
        /// `const void**` and `void**`; the `__global__` declares `const T**`
        /// and `T**`, so declaring them at `*mut *const T` / `*mut *mut T`
        /// keeps the `Ty` the binder checks and improves the typecheck line
        /// from `void` to the element type. The Rust spellings are the ones
        /// `driver-cuda/src/fire/moe_dispatch.rs` already passes.
        fn build_moe_ptrs_aligned = "moe::device::build_moe_ptrs_aligned" <T> (
            expert_ids: *const i32,
            gate_up_base: *const T,
            down_base: *const T,
            aligned_in: *const T,
            aligned_gate_up: *mut T,
            aligned_act: *mut T,
            aligned_out: *mut T,
            a_gu_ptrs: *mut *const T,
            b_gu_ptrs: *mut *const T,
            c_gu_ptrs: *mut *mut T,
            a_dn_ptrs: *mut *const T,
            b_dn_ptrs: *mut *const T,
            c_dn_ptrs: *mut *mut T,
            max_blocks: i32,
            block_size: i32,
            h: i32,
            i_moe: i32,
            routed_blocks: i32,
            shared_gate_up_base: *const T,
            shared_down_base: *const T,
        ) where *const T, *mut T, *mut *const T, *mut *mut T {
            "moe::build_moe_ptrs_aligned_dev_bf16" => where [T = bf16] "device::bf16",
        }

        /// Scatters an aligned GEMM's output rows back to route order,
        /// optionally folding a shared-expert row on the way — the SCALAR
        /// arm.
        fn reorder_moe_aligned_output = "moe::device::reorder_moe_aligned_output" <T> (
            aligned_out: *const T,
            sorted_route_ids: *const i32,
            route_out: *mut T,
            num_routes: i32,
            aligned_rows: i32,
            hidden: i32,
            shared_row_begin: i32,
            num_tokens: i32,
            shared_out: *mut T,
        ) where *const T, *mut T {
            "moe::reorder_moe_aligned_output_scalar_bf16" => where [T = bf16] "device::bf16",
        }

        /// The same scatter over eight-wide vector loads — the arm the host
        /// selects when `hidden % 8 == 0` and all three pointers are
        /// 16-byte aligned. `hidden_vec` is `hidden / 8` and sizes the grid
        /// as well as bounding the kernel.
        fn reorder_moe_aligned_output_vec = "moe::device::reorder_moe_aligned_output_vec" <T> (
            aligned_out: *const T,
            sorted_route_ids: *const i32,
            route_out: *mut T,
            num_routes: i32,
            aligned_rows: i32,
            hidden_vec: i32,
            shared_row_begin: i32,
            num_tokens: i32,
            shared_out: *mut T,
        ) where *const T, *mut T {
            "moe::reorder_moe_aligned_output_vec_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `moe/moe_grouped_gemm.cuh` — the short-K grouped GEMM, one instantiation.
pub mod moe_grouped_gemm {
    use super::bf16;

    unit! {
        /// The file's only template, and the second of the tree's two `wmma`
        /// users.
        ///
        /// A unit is a list of instantiations; it is NOT a list of
        /// instantiations a *rule* can state. This file's grid is
        /// `dim3(N / kNTile, max_blocks)` — two axes, neither in any
        /// rectangle — and `super::moe_grouped_gemm_bf16` builds it by hand.
        unit MOE_GROUPED_GEMM = "moe/moe_grouped_gemm",
            text = include_str!("../../csrc/src/moe/moe_grouped_gemm.cuh"),
            file = "moe/moe_grouped_gemm.cuh";

        /// One launch over a padded, expert-sorted batch.
        ///
        /// `expert_ids[b]` names the expert of padded block `b` — negative
        /// for a padding block, which the kernel exits on immediately
        /// (`moe_grouped_gemm.cuh:129`, *"padding block: the whole point of
        /// this kernel"*).
        ///
        /// The parameters are the `__global__`'s six
        /// (`moe_grouped_gemm.cuh:116`), NOT the deleted launcher's nine:
        /// `max_blocks` and `M` never reached the kernel — the first was
        /// `grid.y`, the second only a predicate input.
        ///
        /// One arm is not a shortage. `static_assert(is_same<T,
        /// bf16>::value)` at `moe_grouped_gemm.cuh:124` — `pie_mma.cuh`
        /// implements bf16 fragments only, and its own comment forbids
        /// extending it without a parity run.
        fn moe_grouped_gemm = "moe::device::moe_grouped_gemm" <T> (
            a: *const T,
            weight_base: *const T,
            c: *mut T,
            expert_ids: *const i32,
            n: i32,
            k: i32,
        ) where *const T, *mut T {
            "moe::moe_grouped_gemm_wmma_bf16" => where [T = bf16] "device::bf16",
        }
    }
}

/// `moe/expert_offsets.cuh` — the CUTLASS fused MoE's routing front-end.
///
/// Four `__global__`s lifted out of FlashInfer's CPM-fetched
/// `cutlass_fused_moe_kernels.cuh`, which is the last ahead-of-time CUDA
/// compile in this tree. They are the tractable end of that file: no CUTLASS
/// types in their signatures, no `Params`, no `CUtensorMap` — three phases of
/// a segmented integer count that produce the `expert_first_token_offset`
/// array everything downstream is indexed by.
///
/// Measured through NVRTC on this L40S (13.0, `compute_89`, the recipe in
/// `csrc/shim/README.md`): **rc=0, 28,503 B of PTX, exactly 4 `.entry`**, all
/// four lowered names returned by `nvrtcGetLoweredName`. One unit, four rows.
///
/// # None of these has a contract, and that is the point
///
/// They are internal steps of `moe::flashinfer_cutlass_moe_bf16`, which is a
/// DRIVER OP — `driver-cuda/src/fire/flashinfer_moe.rs` drives them through
/// its own `Launch`es, exactly as it drives the CUTLASS runner behind the five
/// `extern "C"` seams. A declared kernel with no contract has no statement to
/// lose; the same arrangement `moe::moe_grouped_gemm_wmma_bf16` was in before
/// this port gave it one.
///
/// # `DeviceKernel::PLAIN` on all four, and fourteen rows became four
///
/// Upstream templated three of these on their block width because
/// `cub::BlockScan` needs a compile-time width, and laddered six
/// instantiations of each behind a host `if` chain over function pointers.
/// Replacing cub with `block_exclusive_sum_i32` — 4.2 MB of carried CCCL
/// against twenty-six lines of `__shfl_up_sync`, and an exact-integer argument
/// for why the rewrite needs no tolerance — makes the width a run-time value.
/// **Fourteen rows became four**, and the width moved from the symbol to the
/// `Launch`, which is where a launch geometry belongs.
///
/// # `expert_first_token_offset` is `int64_t*`, and the row world could not
///   say so
///
/// `kernels::Ty` has `I64s` for a read-only `int64_t*` — added, its own doc
/// says, because *"only the DECLARED width makes the mismatch a compile error
/// instead of a stride bug"* — and has **no `I64sMut`**. Adding one is not a
/// one-line change: `Ty` is matched exhaustively in fourteen places across five
/// crates, including `kernels-vulkan`, `kernels-wgpu`, `driver-vulkan` and
/// `driver-wgpu`, none of which has anything to do with a CUDA MoE. So the
/// deleted rows spelled the widest thing that existed, `BufMut`, and recorded
/// what it cost: a caller that handed these kernels an `i32` array would be
/// caught by neither the row nor the compile.
///
/// **The `ptr_abi!(i64, ...)` this file asks for closes half of that** without
/// touching `Ty`: `*mut i64` takes `Ty::BufMut` — the same cell the row world
/// bound — while its `CPP` becomes `::std::int64_t*`, so the typecheck
/// translation unit sees the width even though the binder still cannot.
pub mod expert_offsets {
    use crate::device::DeviceKernel;

    unit! {
        /// The routing front-end: three phases, four kernels, one compile.
        unit EXPERT_OFFSETS = "moe/expert_offsets",
            text = include_str!("../../csrc/src/moe/expert_offsets.cuh"),
            file = "moe/expert_offsets.cuh";

        /// Phase one, the per-block count. `dim3(num_experts_per_node,
        /// num_blocks_per_seq)` blocks of `num_tokens_per_block` threads,
        /// upstream `:646-679`.
        ///
        /// `blocked_row_to_unpermuted_row` is
        /// `[num_experts_per_node, num_tokens]` and is written SPARSELY —
        /// only the first `count` slots of each block's slice are live — so
        /// its extent is not a row count.
        fn expert_offsets_block = "moe::device::block_expert_prefix_sum" (
            token_selected_experts: *const i32,
            blocked_expert_counts: *mut i32,
            blocked_row_to_unpermuted_row: *mut i32,
            num_tokens: i64,
            num_experts_per_token: i64,
            start_expert_id: i32,
        ) {
            "moe::expert_offsets_block_dev" => DeviceKernel::PLAIN,
        }

        /// Phase two, the global scan — ONE block, the block width carrying
        /// the whole `num_experts_per_node * num_blocks_per_seq` array.
        fn expert_offsets_scan = "moe::device::global_expert_prefix_sum" (
            blocked_expert_counts: *const i32,
            blocked_expert_counts_cumsum: *mut i32,
            expert_first_token_offset: *mut i64,
            num_experts_per_node: i64,
            num_blocks_per_seq: i64,
        ) {
            "moe::expert_offsets_scan_dev" => DeviceKernel::PLAIN,
        }

        /// Phase two at the large size — one block at a fixed 1024, each
        /// thread folding `num_elem_per_thread` counters before the scan.
        fn expert_offsets_scan_large = "moe::device::global_expert_prefix_sum_large" (
            blocked_expert_counts: *const i32,
            blocked_expert_counts_cumsum: *mut i32,
            expert_first_token_offset: *mut i64,
            num_experts_per_node: i64,
            num_blocks_per_seq: i64,
            num_elem_per_thread: i64,
        ) {
            "moe::expert_offsets_scan_large_dev" => DeviceKernel::PLAIN,
        }

        /// Phase three, the scatter. Phase one's grid at phase one's width,
        /// upstream `:843-868`.
        ///
        /// `num_tokens` is `i32` here and `i64` in phase one, which is
        /// upstream's inconsistency carried across rather than tidied: it is a
        /// STRIDE into `blocked_row_to_unpermuted_row` in this kernel and a
        /// bound in that one, and quietly widening it would be a body change
        /// wearing a type change's clothes.
        fn expert_offsets_merge = "moe::device::merge_expert_prefix_sum" (
            blocked_expert_counts: *const i32,
            blocked_expert_counts_cumsum: *const i32,
            blocked_row_to_unpermuted_row: *const i32,
            permuted_token_selected_experts: *mut i32,
            permuted_row_to_unpermuted_row: *mut i32,
            unpermuted_row_to_permuted_row: *mut i32,
            num_tokens: i32,
        ) {
            "moe::expert_offsets_merge_dev" => DeviceKernel::PLAIN,
        }
    }
}

/// The units `moe` compiles.
///
/// Hand-written where a one-root family's is generated, for the reason the
/// block comment above the modules gives. `families::ALL` reads this.
pub static UNITS: &[Unit] = &[
    topk_sigmoid::TOPK_SIGMOID,
    dsv4_routing::DSV4_ROUTING,
    topk_softmax::TOPK_SOFTMAX,
    moe_dispatch::MOE_DISPATCH,
    moe_grouped_gemm::MOE_GROUPED_GEMM,
    expert_offsets::EXPERT_OFFSETS,
];

// ---------------------------------------------------------------------------
// The numbers, once each. Every one carries the file it was read from —
// either a `.cuh` constant, or a `runtime/launch.rs` rule this port has to
// transcribe because the rule was the only statement of a geometry.
// ---------------------------------------------------------------------------

/// `runtime/launch.rs:578` — `const BLOCK: u32 = 256;`.
///
/// The block every pointwise rule in this tree uses, and the block the six
/// launches below that came from `Rms`, `Elementwise`, `ElementwiseRows`,
/// `RowsFlat` and `PerRow` take.
const BLOCK: u32 = 256;

/// `runtime/launch.rs:584` — `const WARP: u32 = 32;`.
///
/// A warp. Also the `x` extent of both decode GEMVs' blocks, because the
/// reduction they perform is a warp shuffle
/// (`driver-cuda/src/fire/moe_dispatch.rs:155`).
const WARP: u32 = 32;

/// `runtime/launch.rs:589` — `const FLOAT: u32 = 4;`, `sizeof(int)` and
/// `sizeof(float)`, which the two counting sorts size their counters in.
const FLOAT: u32 = 4;

/// `runtime/launch.rs:622` — `const ROUTER_BLOCK: u32 = 64;`.
///
/// `LaunchRule::RouterLane`'s width, and it is the algorithm rather than a
/// tuning: `topk_softmax.cuh` carries `static_assert(kSoftmaxBlock == 64,
/// "block_argmax folds exactly one upper warp")`.
const ROUTER_BLOCK: u32 = 64;

/// `runtime/launch.rs:614` — `const SORT_BLOCK: u32 = MAX_BLOCK;`, and
/// `driver-cuda/src/fire/moe_dispatch.rs:636` restates it as `1024`.
///
/// Not [`DISPATCH_BLOCK`], and not free to become it. The scan in
/// `moe_bucket_exact` is block-wide, so this number is the whole of the
/// parallelism the sort gets; the file's other launches are 256 because they
/// stride a row and more threads would only idle.
const SORT_BLOCK: u32 = 1024;

/// `moe/dsv4_routing.cu:19` — `kDsv4Block = 256`, quoted at
/// `driver-cuda/src/fire/dsv4_routing.rs:58`.
///
/// The same number as [`BLOCK`] and kept separate from it, because
/// `LaunchRule::RowsFlat` — `runtime/launch.rs:949-955`, `grid [ceil(rows /
/// 256), 1, 1]`, `block [256, 1, 1]`, `smem 0` — was ported FROM this
/// `<<<>>>` and not the other way round. `hash_route_lookup` was the only row
/// that rule ever had; the launch below states the `.cuh`'s constant, which is
/// the one that would move if anything moved.
const DSV4_BLOCK: u32 = 256;

/// `moe_dispatch.cuh`'s `device::kDispatchBlock`, restated at
/// `driver-cuda/src/fire/moe_dispatch.rs:131`.
///
/// The width every flat dispatch kernel in the file is launched at, and the
/// C++ read it from the header rather than restating it. It is restated here
/// because a Rust launcher cannot `#include`, and it carries this paragraph so
/// that the next person to change it knows it is four launches wide and not
/// one.
const DISPATCH_BLOCK: u32 = 256;

/// `moe_dispatch.cuh`'s `device::kMoeVecWidth` — eight bf16, one `uint4`
/// (`driver-cuda/src/fire/moe_dispatch.rs:140`).
///
/// Read TWICE for different purposes and the difference matters. In the two
/// decode GEMVs it is a DIVISIBILITY REQUIREMENT on the reduction extent and
/// failing it is a refusal; in [`reorder_moe_aligned_output_bf16`] it is half
/// of a vectorisability TEST and failing it selects the other kernel. Same
/// constant, one refusal and one fork.
const MOE_VEC_WIDTH: i32 = 8;

/// `moe_dispatch.cuh`'s `device::kGemvWarps` — four warps per block, and the
/// `y` extent of both decode GEMVs' blocks
/// (`driver-cuda/src/fire/moe_dispatch.rs:151`).
///
/// It is simultaneously `blockDim.y` and the number of OUTPUT COLUMNS a block
/// covers, which is why it divides the grid's `x` as well: one warp reduces
/// one output column, four warps to a block, `ceil(N / 4)` blocks across. That
/// coupling is the whole reason no `LaunchRule` states this rectangle — `Qmv`
/// is one warp per output row at a fixed 256-wide block, a different shape —
/// and it is why the constant is not free to change on its own.
const GEMV_WARPS: i32 = 4;

/// `moe_grouped_gemm.cuh`'s `constexpr int kFrag = 16`.
///
/// Load-bearing twice, which is why it is one constant: the support test
/// requires `M == kFrag` exactly (the kernel computes one fragment of rows per
/// block and has no tail path) and requires `K % kFrag == 0` (the mainloop
/// steps K by a fragment and never checks a remainder).
const FRAG: i32 = 16;

/// Warps per block — `moe_grouped_gemm.cuh:76`'s `constexpr int kGemmWarps =
/// 4`. The launch was `<<<grid, device::kGemmWarps * 32, ...>>>`, so the block
/// is 128 threads, and the header's `__launch_bounds__(kGemmWarps * 32)`
/// states the same number to the compiler.
const GEMM_WARPS: u32 = 4;

/// The N-axis tile — `moe_grouped_gemm.cuh`'s `kNTile`.
///
/// Spelled as the product rather than as `64` because that is how the header
/// spells it: each of the four warps owns one 16-wide fragment of the N axis.
/// `grid.x` is `N / kNTile` with no rounding, which is why the support test
/// demands `N % kNTile == 0` — a rounded-up grid would have a fourth warp
/// writing past the row and a rounded-down one would leave the tail unwritten,
/// and the kernel bounds-checks neither.
#[allow(clippy::cast_possible_wrap)]
const N_TILE: i32 = FRAG * GEMM_WARPS as i32;

/// The reduction bound past which the grouped GEMM stops paying.
///
/// A HOST constant because the decision is the launcher's: the kernel is
/// correct at any K and this is the bound at which firing it stops paying.
/// Measured on Qwen3.6-35B-A3B tp2 decode against cuBLAS —
/// `down K=256 7.94 -> 5.91 ms` taken, `gate_up K=2048 11.08 -> 11.98` left on
/// cuBLAS (`moe_grouped_gemm.cu:19-21`).
const SHORT_K: i32 = 512;

/// The smallest block the aligned MoE path is ever padded to.
///
/// `moe_dispatch.hpp`'s `kMoeAlignedBlockMin`, which is deleted with the rest
/// of that header.
pub const MOE_ALIGNED_BLOCK_MIN: i32 = 16;

/// The largest, and the cap is a measurement rather than a limit.
///
/// `moe_dispatch.hpp`'s `kMoeAlignedBlockMax`. See [`moe_aligned_block`] for
/// the numbers behind it.
pub const MOE_ALIGNED_BLOCK_MAX: i32 = 64;

// ---------------------------------------------------------------------------
// The launch rules this family's rows were fired through, as the expressions
// they evaluate to. `Launch`'s conveniences are conveniences: `Rms`,
// `ElementwiseRows` and `RouterSort` fit neither `flat` nor `per_row`, so they
// write the literal.
// ---------------------------------------------------------------------------

/// `LaunchRule::Rms`, as the expression it evaluates to.
///
/// `runtime/launch.rs:737-746`: `grid [rows, 1, 1]`, `block [256, 1, 1]`,
/// `smem (BLOCK / WARP) * 4` — thirty-two bytes, one float per warp, sized on
/// the block width because *"`block_sum` writes one float per warp and reads
/// them back from lane 0 of the first. Sizing this on anything but the block
/// width is a race the hardware does not report."*
///
/// **The two routers fired through this rule do not read those thirty-two
/// bytes**, and they get them anyway. That is reproduction rather than
/// improvement: dropping the allocation would be a change to what the device
/// sees, made by a port whose duty is that the device sees what it saw. The
/// direction of the error is the safe one — a dynamic allocation larger than
/// the kernel's use of it — and it is the same 132-byte over-allocation
/// `RouterSort` hands `moe_bucket_exact` for the same reason.
#[must_use]
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * FLOAT)
}

/// `LaunchRule::Elementwise`, as the expression it evaluates to.
///
/// `runtime/launch.rs:828-834` — `grid [ceil(n / 256), 1, 1]`,
/// `block [256, 1, 1]`, no shared memory. The grid rounds UP, which is why
/// every kernel fired through it keeps its own element count as an operand.
#[must_use]
const fn elementwise(n: u32) -> Launch {
    Launch::flat(n, BLOCK)
}

/// `LaunchRule::ElementwiseRows`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1014-1020` — `grid [rows, ceil(width / 256), 1]`,
/// `block [256, 1, 1]`, `smem 0`.
///
/// **The row is on `x` and the deleted `<<<>>>` had it on `y`.** Four kernels
/// launched `dim3(ceil(width / 256), rows)`; the two index lines in each moved
/// with the rule and the `dim3` in `moe_dispatch.cu` moved with them, so both
/// compilers launch the transposed grid and every thread computes the element
/// it computed before. The guard is `h >= hidden` either way.
#[must_use]
const fn elementwise_rows(rows: u32, width: u32) -> Launch {
    Launch { grid: [rows, width.div_ceil(BLOCK), 1], block: [BLOCK, 1, 1], smem: 0, smem_opt_in: false }
}

/// `LaunchRule::RouterLane`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1758-1760` — `grid [rows, 1, 1]`, `block [64, 1, 1]`,
/// `smem 0`. The width does not scale with the expert count:
/// `topk_softmax.cuh` gives each expert several lanes and folds them, and the
/// expert bound is a value precondition the host carries, not a geometry.
#[must_use]
const fn router_lane(rows: u32) -> Launch {
    Launch::per_row(rows, ROUTER_BLOCK)
}

/// `LaunchRule::RouterSort`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1801-1808` — `grid [1, 1, 1]`, `block [1024, 1, 1]`,
/// `smem (3 * n_experts + 34) * 4`, which is the deleted launcher's own
/// arithmetic: *"counts + offsets(+1) + fill, then 32 warp partials and one
/// running base for the block-wide scan"*, `moe_dispatch.cu:129-133`.
///
/// The smem is stated from the expert count and is the reason this could not
/// have been faked with a constant: the sort's counters, offsets and fill are
/// each `n_experts` long, and a mixture with 256 experts wants four times what
/// one with 64 does. Under-allocate and the scan's warp partials land inside
/// the offsets it is scanning.
///
/// [`moe_bucket_exact`] does NOT use this: it is the same launch without the
/// scan's 33 extra words, and its own host program states `(3E + 1) * 4`
/// exactly. The rule handed it 132 bytes it would not read — legal, and the
/// direction of the error that is not a silent one — and that gap is gone with
/// the rule.
#[must_use]
const fn router_sort(n_experts: u32) -> Launch {
    Launch::per_row(1, SORT_BLOCK).smem((3 * n_experts + 34) * FLOAT)
}

/// `LaunchRule::PerRow`, as the expression it evaluates to.
///
/// `runtime/launch.rs:1103-1105` — `grid [rows, 1, 1]`, `block [256, 1, 1]`,
/// `smem 0`. For [`scatter_add_weighted_bf16`] the "rows" are ROUTES, which is
/// the whole reason that symbol's contract is a `none:`.
#[must_use]
const fn per_row(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK)
}

/// The expert ceiling all four routers share, and it is a shared-memory bound.
///
/// `topk_sigmoid.cuh:75` `kSigmoidMaxExperts`, `dsv4_routing.cuh:71`
/// `kDsv4MaxExperts`, `topk_softmax.cuh:125` `kSoftmaxMaxExperts` — three
/// names, one number, and each backs a static `__shared__ float[512]` the
/// kernel indexes by expert. `families::moe` put it in one sentence: *"their
/// static `__shared__` slabs are sized by an expert bound the launcher REFUSES
/// to exceed, which is why widening the BLOCK is safe and widening the ROUTER
/// is not."*
///
/// `moe/topk_softmax.cu` did not merely return on it — it THREW,
/// `std::runtime_error("topk_softmax_bf16: num_experts exceeds MAX_EXPERTS")`.
/// A throw is not a spelling fn-world has: the four `fn`s below answer
/// `Refusal::Wide { what, at, max: MAX_EXPERTS }` and the fire declines, which
/// is the same fact reported where §0 wants it. `Wide` is the right variant
/// because 512 is exactly what `Wide` describes — a ceiling the compiled unit
/// cannot exceed — and its sentence, *"is {at}, above the {max} this unit was
/// compiled for"*, is the throw's message with the number in it.
const MAX_EXPERTS: i32 = 512;

// ---------------------------------------------------------------------------
// Truth two: the host programs. One `fn` per launcher, each returning `Fired`
// so that "it declined" cannot be spelled like "it ran".
//
// `Fired` and `Result<(), Refusal>` are two spellings of one outcome and stay
// two: a `fn` returns `Fired`, a `bind!` body returns the `Result`, and
// `Fired::ok()` is the only bridge.
// ---------------------------------------------------------------------------

/// `moe::topk_sigmoid_bf16` — the sigmoid router, one block per token.
///
/// # Geometry
///
/// [`rms`], which is the rule the deleted row was fired through
/// (`families::moe`'s `topk_sigmoid` row, `LaunchRule::Rms`): `grid [tokens,
/// 1, 1]`, `block [256, 1, 1]`, 32 bytes of shared memory this kernel does not
/// read. The ahead-of-time launcher fired 128; the staging loops step by
/// `blockDim.x`, so the rule's wider block reaches the same experts in half
/// the iterations and the arithmetic per element is unchanged.
///
/// `tokens` is not an operand — the grid IS the tokens.
///
/// # `correction_bias` may be null
///
/// A family without a correction bias states no fourth operand, and the kernel
/// reads a null as *"there is none"*. The deleted row spelled that
/// `Source::Or(Weight(0), Lit::Null)`; the bind spells it
/// `cx.weight(0).unwrap_or(null)`, and this `fn` does not branch on it at all.
///
/// # Safety
///
/// `logits` addresses `tokens * e` live elements, `topk_idx` and `topk_w`
/// `tokens * k` writable ones, `correction_bias` either null or `e` floats,
/// and `stream` must be live across the launch.
#[cfg(feature = "_cuda")]
pub unsafe fn topk_sigmoid_bf16(
    logits: *const bf16,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    correction_bias: *const f32,
    tokens: i32,
    e: i32,
    k: i32,
    renormalize: bool,
    routed_scaling_factor: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if e > MAX_EXPERTS {
        return Fired::Declined(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: e,
            max: MAX_EXPERTS,
        });
    }
    unsafe {
        topk_sigmoid::raw::topk_sigmoid::<bf16>(
            "moe::topk_sigmoid_bf16",
            rms(tokens.unsigned_abs()),
            logits,
            topk_idx,
            topk_w,
            correction_bias,
            e,
            k,
            renormalize,
            routed_scaling_factor,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::topk_sqrtsoftplus_bf16` — DeepSeek-V4's sqrt-softplus router.
///
/// # Geometry
///
/// [`rms`], the rule its deleted row carried, at the same 32 unread bytes as
/// [`topk_sigmoid_bf16`]. The ahead-of-time launcher fired 256, which is what
/// the rule fires.
///
/// # Safety
///
/// As [`topk_sigmoid_bf16`].
#[cfg(feature = "_cuda")]
pub unsafe fn topk_sqrtsoftplus_bf16(
    logits: *const bf16,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    correction_bias: *const f32,
    tokens: i32,
    e: i32,
    k: i32,
    renormalize: bool,
    routed_scaling_factor: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    // The one host precondition `moe/dsv4_routing.cu` carried, in its own
    // words: *"`num_experts > device::kDsv4MaxExperts` returns without
    // launching, because a wider router would overrun the kernel's static
    // shared arrays."* It belongs to THIS router and not to its `.cuh`
    // sibling — `hash_route_lookup` stages nothing, so the ceiling does not
    // apply to it and reproducing the check there would refuse a launch that
    // is fine.
    if e > MAX_EXPERTS {
        return Fired::Declined(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: e,
            max: MAX_EXPERTS,
        });
    }
    unsafe {
        dsv4_routing::raw::topk_sqrtsoftplus::<bf16>(
            "moe::topk_sqrtsoftplus_bf16",
            rms(tokens.unsigned_abs()),
            logits,
            topk_idx,
            topk_w,
            correction_bias,
            e,
            k,
            renormalize,
            routed_scaling_factor,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::hash_route_lookup` — DeepSeek-V4's hashed expert sets.
///
/// # Geometry
///
/// `Launch::flat` at [`DSV4_BLOCK`], which is what `LaunchRule::RowsFlat`
/// evaluates to and was ported from.
/// `moe/dsv4_routing.cu:56-66`, verbatim:
///
/// ```text
/// if (tokens <= 0 || top_k <= 0) return;
/// // One thread per token, not one block: the kernel's whole body is a table
/// // read and a K-long gather.
/// const int grid = (tokens + kDsv4Block - 1) / kDsv4Block;
/// device::hash_route_lookup<device::bf16><<<grid, kDsv4Block, 0, stream>>>(
///     token_ids, tid2eid,
///     static_cast<const device::bf16*>(logits),
///     topk_idx, topk_w,
///     tokens, vocab_size, num_experts, top_k,
///     renormalize, routed_scaling_factor);
/// ```
///
/// with `kDsv4Block = 256` at `:19` — the same number [`DSV4_BLOCK`] and
/// [`BLOCK`] both are, kept as its own constant because it is the `.cuh`'s and
/// not the rule's.
///
/// `tokens` is spent on the grid AND passed as an operand, because the last
/// block is partial and the guard is the kernel's own.
///
/// # Safety
///
/// `token_ids` is `[tokens]` i32, each entry in `[0, vocab_size)`; `tid2eid`
/// is `[vocab_size, top_k]` i64; `logits` is `[tokens, num_experts]` bf16;
/// `topk_idx` is writable for `[tokens, top_k]` i32 and `topk_w` for
/// `[tokens, top_k]` f32. A token id past `vocab_size` reads the table out of
/// bounds — the kernel bounds `n` against `tokens` and nothing else.
#[cfg(feature = "_cuda")]
pub unsafe fn hash_route_lookup(
    token_ids: *const i32,
    tid2eid: *const i64,
    logits: *const bf16,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    tokens: i32,
    vocab_size: i32,
    num_experts: i32,
    top_k: i32,
    renormalize: bool,
    routed_scaling_factor: f32,
    stream: *mut c_void,
) -> Fired {
    // `:36`, both terms, kept apart so the caller learns which it hit.
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if top_k <= 0 {
        return Fired::Declined(Refusal::Empty { what: "top_k" });
    }
    unsafe {
        dsv4_routing::raw::hash_route_lookup::<bf16>(
            "moe::hash_route_lookup_dev",
            // `:39-40` — `grid = (tokens + kDsv4Block - 1) / kDsv4Block`, then
            // `<<<grid, kDsv4Block, 0, stream>>>`.
            Launch::flat(tokens.unsigned_abs(), DSV4_BLOCK),
            token_ids,
            tid2eid,
            logits,
            topk_idx,
            topk_w,
            tokens,
            vocab_size,
            num_experts,
            top_k,
            renormalize,
            routed_scaling_factor,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::topk_softmax_bf16` — the softmax router's BLOCK form.
///
/// # Geometry
///
/// [`router_lane`]: `grid [tokens, 1, 1]`, `block [64, 1, 1]`, no shared
/// memory. `moe/topk_softmax.cu`'s launch, which the rule was written from:
///
/// ```text
/// topk_softmax<T><<<N, kSoftmaxBlock>>>(...)
/// ```
///
/// # This is the block form and only the block form
///
/// The launcher chose between five warp rungs and this kernel at fire time;
/// the module's `unit!` doc carries the ladder, the 7.56 us/call measurement
/// behind it, and what re-landing it costs. fn-world can express that choice —
/// it is an `if` in this `fn` — and this port does not make it, because
/// declaring five more instantiations is a change to what NVRTC compiles and
/// belongs to the session that measures it.
///
/// `act`, `bias` and `hidden` are the fused form's operands: two nulls and a
/// zero, exactly as the launcher passed them.
///
/// # Safety
///
/// `logits` addresses `tokens * num_experts` live elements and `topk_idx` /
/// `topk_w` `tokens * k` writable ones.
#[cfg(feature = "_cuda")]
pub unsafe fn topk_softmax_bf16(
    logits: *const bf16,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    tokens: i32,
    num_experts: i32,
    k: i32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    // What the deleted launcher threw on. See [`MAX_EXPERTS`].
    if num_experts > MAX_EXPERTS {
        return Fired::Declined(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: num_experts,
            max: MAX_EXPERTS,
        });
    }
    unsafe {
        topk_softmax::raw::topk_softmax::<bf16>(
            "moe::topk_softmax_bf16",
            router_lane(tokens.unsigned_abs()),
            logits,
            core::ptr::null(),
            core::ptr::null(),
            topk_idx,
            topk_w,
            num_experts,
            k,
            0,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::topk_sigmoid_bias_fp32` — sigmoid routing with the correction bias in
/// the ranking, over fp32 logits.
///
/// # Geometry
///
/// [`router_lane`] again, and the same `<<<N, kSoftmaxBlock>>>`.
///
/// `normalize` is an `int` because the `__global__` declares one: the deleted
/// C++ host took a `bool` and narrowed it with `normalize ? 1 : 0`. The
/// narrowing lives here now, at the one place that knows both types.
///
/// # Safety
///
/// `logits` addresses `tokens * num_experts` live floats, `correction_bias`
/// `num_experts` live floats and NOT null — this entry point is the one a
/// checkpoint with a bias uses, and a null is a fault rather than an absence.
#[cfg(feature = "_cuda")]
pub unsafe fn topk_sigmoid_bias_fp32(
    logits: *const f32,
    correction_bias: *const f32,
    topk_idx: *mut i32,
    topk_w: *mut f32,
    tokens: i32,
    num_experts: i32,
    k: i32,
    normalize: bool,
    routed_scaling_factor: f32,
    stream: *mut c_void,
) -> Fired {
    if tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "tokens" });
    }
    if num_experts > MAX_EXPERTS {
        return Fired::Declined(Refusal::Wide {
            what: "num_experts, which the router stages in shared memory",
            at: num_experts,
            max: MAX_EXPERTS,
        });
    }
    unsafe {
        topk_softmax::raw::topk_sigmoid_bias::<f32>(
            "moe::topk_sigmoid_bias_fp32",
            router_lane(tokens.unsigned_abs()),
            logits,
            correction_bias,
            topk_idx,
            topk_w,
            num_experts,
            k,
            i32::from(normalize),
            routed_scaling_factor,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::apply_per_expert_scale_bf16` — fold a per-expert scale into the
/// router weights, in place.
///
/// # Geometry
///
/// [`elementwise`] over the flat `[tokens, top_k]` rectangle of routes:
/// `ceil(total / 256)` blocks of 256, the rule the deleted row carried.
///
/// `n` and `k` were two operands of the ahead-of-time twin and are one here,
/// because the kernel only ever used their product — and the product is the
/// weight vector's own element count, which is why it can be sourced at all.
///
/// # Safety
///
/// `topk_idx` and `topk_w` each address `total` live elements, and
/// `per_expert_scale` one per expert named by any of them.
#[cfg(feature = "_cuda")]
pub unsafe fn apply_per_expert_scale_bf16(
    topk_idx: *const i32,
    topk_w: *mut f32,
    per_expert_scale: *const bf16,
    total: i32,
    stream: *mut c_void,
) -> Fired {
    if total <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the route count" });
    }
    unsafe {
        topk_softmax::raw::apply_per_expert_scale::<bf16>(
            "moe::apply_per_expert_scale_bf16",
            elementwise(total.unsigned_abs()),
            topk_idx,
            topk_w,
            per_expert_scale,
            total,
            stream,
        );
    }
    Fired::Launched
}

/// Whether the short-K grouped GEMM can compute this rectangle at all, and
/// whether firing it is worth doing.
///
/// `moe/moe_grouped_gemm.cu:18-24`, one conjunct at a time and in the C++'s
/// order — `M == kFrag && N > 0 && K > 0 && K <= kShortK && (N % kNTile) == 0
/// && (K % kFrag) == 0` — so that the answer says WHICH shape it was.
///
/// # A refusal is never a fallback, and one of these is not a shape
///
/// Five of the six conjuncts are correctness: this kernel cannot compute
/// those rectangles. The sixth, `K <= kShortK`, is a MEASUREMENT — the kernel
/// is correct at any K and 512 is where cuBLAS's tuned mainloop starts
/// winning ([`SHORT_K`]). Both answer the same way here, because a caller
/// that has to run the general path does not care which of the two reasons
/// sent it there, and the reason is in the `what`.
///
/// # Which refusal, and which direction
///
/// `Refusal::Narrow`'s sentence ends *"below the kernel's smallest unit of
/// work"*, and `Refusal::Wide`'s *"above the {max} this unit was compiled
/// for"*. The four conjuncts here fail in three different directions and are
/// spelled accordingly:
///
/// - `K > 512` is a CEILING — above it cuBLAS wins — so it is
///   [`Refusal::Wide`] with `max: SHORT_K`.
/// - `M != 16` is TWO refusals sharing one line of C++: `M > 16` is `Wide`
///   with `max: FRAG`, `M < 16` is `Narrow`. One `!=` collapsed them, and
///   collapsing them again in Rust would put a direction in the `what` that
///   the variant contradicts.
/// - `N % 64` and `K % 16` are DIVISIBILITY, which has no direction at all —
///   `N = 100` is neither too wide nor too narrow for a 64-wide tile, it is
///   off the grid. These stay [`Refusal::Narrow`], whose `what` names the
///   requirement (*"in whole 64-wide tiles"*) rather than an extent, and
///   `Wide` would be a worse fit because there is no `max` to report.
///
/// The upper bounds used to be `Narrow` with the requirement in the `what` —
/// the phrasing `norm` (`x/norm.rs:1194`) and `attn` (`x/attn.rs:470`) had
/// to invent before the floor had a variant for it. It has one now.
///
/// # Errors
///
/// The conjunct that failed, with the extent that failed it.
#[cfg(feature = "_cuda")]
pub const fn supported(m: i32, n: i32, k: i32) -> Result<(), Refusal> {
    // `m != kFrag` in one line of C++ is TWO refusals in two directions, and
    // the floor now spells both. A taller rectangle is `Wide` -- the kernel
    // computes exactly one 16-row fragment per block and has no tail path --
    // and a shorter one is `Narrow`, below that fragment.
    if m > FRAG {
        return Err(Refusal::Wide {
            what: "M, which must be exactly one 16-row fragment",
            at: m,
            max: FRAG,
        });
    }
    if m < FRAG {
        return Err(Refusal::Narrow {
            what: "M, which must be exactly one 16-row fragment",
            at: m,
        });
    }
    if n <= 0 || k <= 0 {
        return Err(Refusal::Empty { what: "the N by K rectangle" });
    }
    if k > SHORT_K {
        return Err(Refusal::Wide { what: "K, above which cuBLAS wins", at: k, max: SHORT_K });
    }
    if n % N_TILE != 0 {
        return Err(Refusal::Narrow { what: "N, in whole 64-wide tiles", at: n });
    }
    if k % FRAG != 0 {
        return Err(Refusal::Narrow { what: "K, in whole 16-deep fragments", at: k });
    }
    Ok(())
}

/// `moe::moe_grouped_gemm_bf16` — the short-K grouped GEMM, one launch over a
/// padded, expert-sorted batch.
///
/// `a` is `[aligned_rows, K]`, `weight_base` the per-expert weight bank, `c`
/// is `[aligned_rows, N]`, and `expert_ids[b]` names the expert of padded
/// block `b` — negative for a padding block, which the kernel exits on
/// immediately (`moe_grouped_gemm.cuh:129`, *"padding block: the whole point
/// of this kernel"*).
///
/// # The grid, which no rule states
///
/// `moe/moe_grouped_gemm.cu:40-41`, verbatim:
///
/// ```text
/// const dim3 grid(N / device::kNTile, max_blocks);
/// device::moe_grouped_gemm<device::bf16><<<grid, device::kGemmWarps * 32, 0, stream>>>(
/// ```
///
/// `max_blocks` bounds the PADDED batch and is not an extent of any operand,
/// which is why the deleted row was `LaunchRule::Unstated` and why this is a
/// `Launch` literal rather than a convenience. `N / kNTile` divides exactly:
/// [`supported`] rejected every `N` for which it does not.
///
/// # What used to be dropped here
///
/// `bind::service::moe_moe_grouped_gemm_bf16` spelled the call `let _ =
/// unsafe { ... }` and said why: the generated arm returns `bool`, its `true`
/// means *"a branch ran"* rather than *"the kernel launched"*, so a refusal
/// had nowhere to go. In fn-world the refusal is returned and the fire
/// reports it with this symbol named. Nothing about the launch changed; what
/// changed is that the decline is no longer discarded one frame above the
/// only code that could act on it.
///
/// **And one frame above is now where the decline is ACTED ON.** This `fn`
/// is called by `fire::moe_grouped`, not by a `bind!` body: the symbol is a
/// driver op, because [`supported`] refuses half of qwen3.5's shapes and the
/// implementation that serves them is a batched cuBLAS call over the pointer
/// arrays `build_moe_ptrs_aligned_bf16` fills. That caller asks [`supported`]
/// first and comes here only when the answer is yes, so the predicate below
/// is now checked twice on the WMMA path and once on the other. Keeping it
/// here anyway is deliberate: it is this host program's own precondition, and
/// a `fn` that is correct only when its caller has already checked something
/// is a `fn` with an unstated argument.
///
/// # Safety
///
/// The four pointers must be device allocations of the shapes above, live on
/// `stream` until the launch completes.
#[cfg(feature = "_cuda")]
pub unsafe fn moe_grouped_gemm_bf16(
    a: *const bf16,
    weight_base: *const bf16,
    c: *mut bf16,
    expert_ids: *const i32,
    max_blocks: i32,
    m: i32,
    n: i32,
    k: i32,
    stream: *mut c_void,
) -> Fired {
    // `moe_grouped_gemm.cu:37`, before the shape test: an empty padded batch
    // has nothing to launch OVER, which is a different fact from a rectangle
    // this kernel cannot compute.
    if max_blocks <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the padded block count" });
    }
    if let Err(why) = supported(m, n, k) {
        return Fired::Declined(why);
    }
    unsafe {
        moe_grouped_gemm::raw::moe_grouped_gemm::<bf16>(
            "moe::moe_grouped_gemm_wmma_bf16",
            Launch {
                // `:40` — `dim3 grid(N / kNTile, max_blocks)`.
                grid: [(n / N_TILE).unsigned_abs(), max_blocks.unsigned_abs(), 1],
                // `:41` — `device::kGemmWarps * 32`, which the header's
                // `__launch_bounds__(kGemmWarps * 32)` states again.
                block: [GEMM_WARPS * 32, 1, 1],
                // None: the mma fragments are registers and the staging tile
                // is a static `__shared__` array inside the kernel.
                smem: 0,
            },
            a,
            weight_base,
            c,
            expert_ids,
            n,
            k,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::moe_gate_up_decode_gemv_bf16` — the decode gate/up leg, one fused
/// GEMV per route over the expert's `[2 * I_moe, H]` projection.
///
/// # Geometry
///
/// `moe_dispatch.cu:85-110`, verbatim:
///
/// ```text
/// const int routes = num_tokens * top_k;
/// const int N = 2 * I_moe;
/// if (routes <= 0 || H <= 0 || N <= 0 || (H % device::kMoeVecWidth) != 0) return;
/// constexpr int kWarps = device::kGemvWarps;
/// const dim3 grid((N + kWarps - 1) / kWarps, routes);
/// const dim3 block(32, kWarps);
/// device::moe_decode_gemv_by_token<device::bf16><<<grid, block, 0, stream>>>(
///     topk_idx, norm_x, gate_up_base, expert_gate_up,
///     top_k, H, N, static_cast<long long>(N) * H);
/// ```
///
/// `N = 2 * I_moe` because gate and up are one allocation, and it crosses
/// three times: as the grid's `x` before the warp fold, as the kernel's
/// output width, and as the first factor of the per-expert stride. One
/// binding, three readers — the C++'s arrangement, kept, because a second
/// derivation of `2 * I_moe` is how a grid and a stride come to disagree.
///
/// # The `H % 8` term is not an optimisation gate
///
/// `moe_dispatch.cu:97-98`: *"`float4` loads need every row to start 16-byte
/// aligned, which holds iff the reduction extent is a multiple of 8 bf16."*
/// A refusal is never a fallback, so an `H` that is not a multiple of eight
/// leaves `expert_gate_up` exactly as the arena had it — which is what the
/// C++ did silently, and what this does with the reason written down.
///
/// # Safety
///
/// `topk_idx` is `[num_tokens, top_k]` i32, `norm_x` `[num_tokens, H]` bf16,
/// `gate_up_base` the expert-major `[experts, 2 * I_moe, H]` weight,
/// `expert_gate_up` writable for `[num_tokens * top_k, 2 * I_moe]` bf16.
#[cfg(feature = "_cuda")]
pub unsafe fn moe_gate_up_decode_gemv_bf16(
    topk_idx: *const i32,
    norm_x: *const bf16,
    gate_up_base: *const bf16,
    expert_gate_up: *mut bf16,
    num_tokens: i32,
    top_k: i32,
    h: i32,
    i_moe: i32,
    stream: *mut c_void,
) -> Fired {
    // `:95-96` — both products in the C++'s order and both in i32, so an
    // overflow lands where it landed before rather than somewhere new.
    let routes = num_tokens * top_k;
    let n = 2 * i_moe;
    // `:99`, all four terms, each answering for itself.
    if routes <= 0 {
        return Fired::Declined(Refusal::Empty { what: "routes" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "H" });
    }
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "2 * I_moe" });
    }
    if h % MOE_VEC_WIDTH != 0 {
        return Fired::Declined(Refusal::Narrow { what: "H, in whole float4 loads of 8", at: h });
    }
    unsafe {
        moe_dispatch::raw::moe_decode_gemv_by_token::<bf16>(
            "moe::moe_decode_gemv_by_token_bf16",
            Launch {
                // `:101` — `(N + kWarps - 1) / kWarps` by `routes`.
                grid: [
                    n.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()),
                    routes.unsigned_abs(),
                    1,
                ],
                // `:102` — `dim3(32, kWarps)`, the two-dimensional block no
                // `LaunchRule` states and that §10.5 forbids adding one for.
                block: [WARP, GEMV_WARPS.unsigned_abs(), 1],
                smem: 0,
            },
            topk_idx,
            norm_x,
            gate_up_base,
            expert_gate_up,
            top_k,
            // The kernel's `k` is the reduction extent, which for this leg is
            // `H`, and its `n` is the output width.
            h,
            n,
            // `:109` — `static_cast<long long>(N) * H`, the cast on the FIRST
            // factor so the product is computed in 64 bits.
            i64::from(n) * i64::from(h),
            stream,
        );
    }
    Fired::Launched
}

/// `moe::moe_down_decode_gemv_bf16` — the decode down leg, reading the
/// activation BY ROUTE rather than by token.
///
/// # Geometry
///
/// `moe_dispatch.cu:112-137`, verbatim:
///
/// ```text
/// const int routes = num_tokens * top_k;
/// if (routes <= 0 || H <= 0 || I_moe <= 0 ||
///     (I_moe % device::kMoeVecWidth) != 0) {
///     return;
/// }
/// constexpr int kWarps = device::kGemvWarps;
/// const dim3 grid((H + kWarps - 1) / kWarps, routes);
/// const dim3 block(32, kWarps);
/// device::moe_decode_gemv_by_route<device::bf16><<<grid, block, 0, stream>>>(
///     topk_idx, expert_act, down_base, expert_out,
///     top_k, I_moe, H, static_cast<long long>(H) * I_moe);
/// ```
///
/// The mirror of [`moe_gate_up_decode_gemv_bf16`] and worth reading as one:
/// the divisibility test moved from `H` to `I_moe` and the grid from `N` to
/// `H`, because the reduction extent and the output width swapped. Same
/// kernel body, the `ActByToken = false` instantiation — the activation this
/// reads is already route-major, since the gate/up leg wrote it that way.
///
/// # Safety
///
/// `expert_act` is `[num_tokens * top_k, I_moe]` bf16 (the SwiGLU of the leg
/// above's output), `down_base` the `[experts, H, I_moe]` weight, `expert_out`
/// writable for `[num_tokens * top_k, H]` bf16.
#[cfg(feature = "_cuda")]
pub unsafe fn moe_down_decode_gemv_bf16(
    topk_idx: *const i32,
    expert_act: *const bf16,
    down_base: *const bf16,
    expert_out: *mut bf16,
    num_tokens: i32,
    top_k: i32,
    h: i32,
    i_moe: i32,
    stream: *mut c_void,
) -> Fired {
    // `:122`
    let routes = num_tokens * top_k;
    // `:123-127`
    if routes <= 0 {
        return Fired::Declined(Refusal::Empty { what: "routes" });
    }
    if h <= 0 {
        return Fired::Declined(Refusal::Empty { what: "H" });
    }
    if i_moe <= 0 {
        return Fired::Declined(Refusal::Empty { what: "I_moe" });
    }
    if i_moe % MOE_VEC_WIDTH != 0 {
        return Fired::Declined(Refusal::Narrow {
            what: "I_moe, in whole float4 loads of 8",
            at: i_moe,
        });
    }
    unsafe {
        moe_dispatch::raw::moe_decode_gemv_by_route::<bf16>(
            "moe::moe_decode_gemv_by_route_bf16",
            Launch {
                // `:129` — `(H + kWarps - 1) / kWarps` by `routes`.
                grid: [
                    h.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()),
                    routes.unsigned_abs(),
                    1,
                ],
                block: [WARP, GEMV_WARPS.unsigned_abs(), 1],
                smem: 0,
            },
            topk_idx,
            expert_act,
            down_base,
            expert_out,
            top_k,
            // `I_moe` is `k` here and `H` is `n` — the swap described above.
            i_moe,
            h,
            // `:136` — `static_cast<long long>(H) * I_moe`.
            i64::from(h) * i64::from(i_moe),
            stream,
        );
    }
    Fired::Launched
}

/// `moe::transpose_expert_scales_u8` — the MXFP4 group-scale relayout,
/// `[e][n][kg] -> [e][kg][n]`, one E8M0 byte per scale.
///
/// # Geometry
///
/// `moe_dispatch.cu:187-199`, verbatim:
///
/// ```text
/// if (num_experts <= 0 || n <= 0 || k_groups <= 0) return;
/// const dim3 block(32, 8);
/// const dim3 grid((k_groups + block.x - 1) / block.x,
///                 (n + block.y - 1) / block.y,
///                 num_experts);
/// device::transpose_expert_scales<device::u8><<<grid, block, 0, stream>>>(
///     src, dst, n, k_groups);
/// ```
///
/// **Three grid axes and two block axes**, which is two axes past anything
/// the `LaunchRule` vocabulary could state, and `families::moe`'s header said
/// so from the split: *"`transpose_expert_scales` wants `dim3(32, 8)` on a 3D
/// grid. Every ported rule produces `[BLOCK, 1, 1]`."* The row was
/// `LaunchRule::Unstated`; here the rectangle is simply written down, which
/// is what §5.1 means by a kernel that fits neither convenience.
///
/// `u8` and not an activation type: the kernel only MOVES bytes — one indexed
/// load, one indexed store — so the instantiation names the storage width and
/// nothing else.
///
/// # Nothing binds this today
///
/// `moe::transpose_expert_scales_u8` gets a `none:` arm: `num_experts`, `n`
/// and `k_groups` describe a WEIGHT's group-scale layout, and no fire states
/// the shape of a weight it did not produce. The host program is here anyway,
/// because the geometry above is the thing that would otherwise be lost, and
/// because the day a fact arrives this is the body that binds.
///
/// # Safety
///
/// `src` and `dst` are both `num_experts * n * k_groups` bytes of device
/// memory and must not overlap: the kernel writes `dst[e][j][i]` from
/// `src[e][i][j]`, and in place is not a transpose.
#[cfg(feature = "_cuda")]
pub unsafe fn transpose_expert_scales_u8(
    src: *const u8,
    dst: *mut u8,
    num_experts: i32,
    n: i32,
    k_groups: i32,
    stream: *mut c_void,
) -> Fired {
    // `:191`, one term at a time.
    if num_experts <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_experts" });
    }
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "n" });
    }
    if k_groups <= 0 {
        return Fired::Declined(Refusal::Empty { what: "k_groups" });
    }
    // `:192` — `dim3(32, 8)`. Named here rather than at module scope because
    // this is the only launch in the family with this block, and hoisting it
    // would suggest otherwise.
    const BX: u32 = 32;
    const BY: u32 = 8;
    unsafe {
        moe_dispatch::raw::transpose_expert_scales::<u8>(
            "moe::transpose_expert_scales_dev_u8",
            Launch {
                // `:193-195`, in the C++'s axis order: `k_groups` on `x`
                // (contiguous in the SOURCE), `n` on `y`, the expert on `z`.
                grid: [
                    k_groups.unsigned_abs().div_ceil(BX),
                    n.unsigned_abs().div_ceil(BY),
                    num_experts.unsigned_abs(),
                ],
                block: [BX, BY, 1],
                smem: 0,
            },
            src,
            dst,
            // `num_experts` is NOT an operand: the kernel reads it as
            // `blockIdx.z`. Transcribed rather than "completed" — an extra
            // operand would not bind.
            n,
            k_groups,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::build_moe_ptrs_aligned_bf16` — fills the six pointer arrays a pair of
/// batched GEMMs reads, one thread per padded block of the aligned layout.
///
/// # Geometry
///
/// `moe_dispatch.cu:204-250`:
///
/// ```text
/// if (max_blocks <= 0) return;
/// if (shared_gate_up_base == nullptr || shared_down_base == nullptr) {
///     routed_blocks = max_blocks;
/// }
/// constexpr int BS = device::kDispatchBlock;
/// const int grid = (max_blocks + BS - 1) / BS;
/// device::build_moe_ptrs_aligned<device::bf16><<<grid, BS, 0, stream>>>(...);
/// ```
///
/// # The `if` that is not a geometry
///
/// `:246-248` is why this was a `Walk` and never a rule. If EITHER
/// shared-expert base is null the launcher OVERWRITES the `routed_blocks`
/// operand with `max_blocks`, which makes the kernel's `is_shared = (b >=
/// routed_blocks)` false for every block, so the shared tail is never
/// addressed and the null pointers are never dereferenced. That is a host
/// decision about an OPERAND taken from a POINTER'S NULLITY. No `Source`
/// could read an address and §10.5 forbade adding one for a single kernel;
/// `Source::Lit(Lit::Null)` could only STATE a null, not branch on one.
///
/// §30's question — do the arms differ? — does not apply: this is not a fork
/// between two kernels but a single launch with one operand rewritten, and
/// the two values of that operand produce different work. Deleting it would
/// dereference null.
///
/// `max_blocks` opens the grid AND is an operand, because the kernel bounds
/// `b < max_blocks` itself; it is a HOST SCALAR — the padded block count the
/// counting sort produced — and not an extent of any value a fire names.
///
/// # Safety
///
/// The six pointer arrays are device arrays of at least `max_blocks` pointers
/// each. `shared_gate_up_base` and `shared_down_base` may be null, and the
/// rewrite above is what makes that safe. Everything else is a device
/// allocation of the aligned layout's shape.
#[cfg(feature = "_cuda")]
pub unsafe fn build_moe_ptrs_aligned_bf16(
    expert_ids: *const i32,
    gate_up_base: *const bf16,
    down_base: *const bf16,
    aligned_in: *const bf16,
    aligned_gate_up: *mut bf16,
    aligned_act: *mut bf16,
    aligned_out: *mut bf16,
    a_gu_ptrs: *mut *const bf16,
    b_gu_ptrs: *mut *const bf16,
    c_gu_ptrs: *mut *mut bf16,
    a_dn_ptrs: *mut *const bf16,
    b_dn_ptrs: *mut *const bf16,
    c_dn_ptrs: *mut *mut bf16,
    max_blocks: i32,
    block_size: i32,
    h: i32,
    i_moe: i32,
    routed_blocks: i32,
    shared_gate_up_base: *const bf16,
    shared_down_base: *const bf16,
    stream: *mut c_void,
) -> Fired {
    // `:245`
    if max_blocks <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the padded block count" });
    }
    // `:246-248`, and the C++ mutated its own by-value parameter. Rust
    // rebinds instead, which is the same thing said without a `mut`.
    let routed_blocks = if shared_gate_up_base.is_null() || shared_down_base.is_null() {
        max_blocks
    } else {
        routed_blocks
    };
    unsafe {
        moe_dispatch::raw::build_moe_ptrs_aligned::<bf16>(
            "moe::build_moe_ptrs_aligned_dev_bf16",
            // `:250` — `(max_blocks + BS - 1) / BS` blocks of
            // `kDispatchBlock`.
            Launch::flat(max_blocks.unsigned_abs(), DISPATCH_BLOCK),
            expert_ids,
            gate_up_base,
            down_base,
            aligned_in,
            aligned_gate_up,
            aligned_act,
            aligned_out,
            a_gu_ptrs,
            b_gu_ptrs,
            c_gu_ptrs,
            a_dn_ptrs,
            b_dn_ptrs,
            c_dn_ptrs,
            max_blocks,
            block_size,
            h,
            i_moe,
            routed_blocks,
            // Null crosses as a null pointer; the kernel tests it and the
            // rewrite above guarantees nothing reads through it when it is.
            shared_gate_up_base,
            shared_down_base,
            stream,
        );
    }
    Fired::Launched
}

/// `moe_dispatch.cu:56-60`, the anonymous-namespace helper, verbatim.
///
/// ```text
/// bool moe_vectorizable(const void* a, const void* b, int hidden) {
///     return (hidden % device::kMoeVecWidth) == 0 &&
///            (reinterpret_cast<std::uintptr_t>(a) % 16) == 0 &&
///            (reinterpret_cast<std::uintptr_t>(b) % 16) == 0;
/// }
/// ```
///
/// It survived the port because the thing it tests survived: eight bf16 make
/// a `uint4` only if the row divides by eight AND both allocations start
/// 16-byte aligned, and the second half is a fact about an ARENA rather than
/// about a shape — which is why no `Source` and no `LaunchRule` could ever
/// carry it, and why every launcher that made this test had to be ported.
///
/// It had four callers in the C++ and has one here. The other three
/// (`token_batched_weighted_sum`, `..._add` and `gather_moe_aligned_inputs`)
/// lost their forks to §43.9, which deleted the launchers and left the
/// decision recorded in comments; those three symbols still have exactly one
/// declared kernel each, so there is nothing here to fork TO. fn-world could
/// express the fork the day a second instantiation is declared and measured —
/// this port does not declare one, for the same reason it does not re-land
/// `topk_softmax`'s warp ladder.
///
/// A `const fn` it is not: [`aligned16`] reads an address.
#[cfg(feature = "_cuda")]
#[must_use]
fn moe_vectorizable(a: *const c_void, b: *const c_void, hidden: i32) -> bool {
    hidden % MOE_VEC_WIDTH == 0 && aligned16(a) && aligned16(b)
}

/// `moe::reorder_moe_aligned_output_bf16` — scatters an aligned GEMM's output
/// rows back to route order, optionally folding a shared-expert row on the
/// way.
///
/// # Geometry
///
/// `moe_dispatch.cu:252-286`, the whole body:
///
/// ```text
/// if (aligned_rows <= 0 || hidden <= 0) return;
/// if (shared_out == nullptr) shared_row_begin = -1;
/// constexpr int BS = device::kDispatchBlock;
/// const bool vectorizable =
///     moe_vectorizable(src, dst, hidden) &&
///     (reinterpret_cast<std::uintptr_t>(sdst) % 16) == 0;
/// if (vectorizable) {
///     const int hidden_vec = hidden / device::kMoeVecWidth;
///     const dim3 grid(aligned_rows, (hidden_vec + BS - 1) / BS);
///     device::reorder_moe_aligned_output_vec<device::bf16>
///         <<<grid, BS, 0, stream>>>(
///             src, sorted_route_ids, dst, num_routes, aligned_rows, hidden_vec,
///             shared_row_begin, num_tokens, sdst);
///     return;
/// }
/// const dim3 grid(aligned_rows, (hidden + BS - 1) / BS);
/// device::reorder_moe_aligned_output<device::bf16><<<grid, BS, 0, stream>>>(
///     src, sorted_route_ids, dst, num_routes, aligned_rows, hidden,
///     shared_row_begin, num_tokens, sdst);
/// ```
///
/// # Two host `if`s, and §30 answers them differently
///
/// **The fork was measured and the arms DIFFER**, so it is a port and not a
/// deletion. Not by timing: structurally, and in a way no timing could
/// soften. `reorder_moe_aligned_output_vec` `static_assert`s `sizeof(T) == 2`
/// and `reinterpret_cast`s three pointers to `uint4`, so on a base that is
/// not 16-byte aligned it does not run *slower*, it **faults**; and its width
/// operand is `hidden / 8` where the scalar's is `hidden`, so the two grids
/// are not even the same rectangle. There is no shape at which running the
/// wrong one is merely a worse choice. That is the opposite of
/// `PIE_QWEN35_GDN_SMEM_STEP`, whose two arms differed by zero everywhere.
///
/// **This is `Composed`'s shape without `Composed`'s rule.** Two kernels
/// behind one symbol, chosen by an `if` in a `fn` — `layout::merge_written` in
/// the ordered-pair form, this one in the fork form — and the hoisting rule
/// costs nothing here because the fork precedes a SINGLE launch. Nothing has
/// gone to the device when either arm is chosen.
///
/// **The `shared_out == nullptr` line is not a fork at all**, and reading it
/// as one is the mistake worth naming: it rewrites an OPERAND, not a
/// geometry. `shared_row_begin = -1` is how the kernel is told there is no
/// fold, and the deleted row already stated `Source::Lit(Lit::I32(-1))`
/// beside `shared_out <- Source::Lit(Lit::Null)` — the generated arm passed
/// null and −1 together and always had. The line is kept because a HAND
/// caller does not go through that arm and may pass a real `shared_out`.
///
/// # Safety
///
/// `aligned_out` is `[aligned_rows, hidden]` bf16, `sorted_route_ids`
/// `[aligned_rows]` i32, `route_out` writable for `[num_routes, hidden]`
/// bf16. `shared_out` may be null; when it is not it is `[num_tokens, hidden]`
/// bf16 and `shared_row_begin` indexes into the aligned rectangle.
#[cfg(feature = "_cuda")]
pub unsafe fn reorder_moe_aligned_output_bf16(
    aligned_out: *const bf16,
    sorted_route_ids: *const i32,
    route_out: *mut bf16,
    num_routes: i32,
    aligned_rows: i32,
    hidden: i32,
    shared_row_begin: i32,
    num_tokens: i32,
    shared_out: *mut bf16,
    stream: *mut c_void,
) -> Fired {
    // `:263`, one term at a time.
    if aligned_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "aligned_rows" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    // `:264` — an operand rewrite, see the paragraph above.
    let shared_row_begin = if shared_out.is_null() { -1 } else { shared_row_begin };
    // `:271-273`. The third term is separate in the C++ too, because
    // `moe_vectorizable` takes two pointers and there are three here.
    //
    // A NULL `shared_out` PASSES the alignment test, in both languages: zero
    // is a multiple of sixteen. That is correct rather than lucky — a null
    // `shared_out` means the kernel never dereferences it, since
    // `shared_row_begin` is now −1 — but it is worth a sentence, because it
    // is the one input for which "aligned" is true of a pointer that cannot
    // be read.
    let vectorizable = moe_vectorizable(aligned_out.cast(), route_out.cast_const().cast(), hidden)
        && aligned16(shared_out.cast_const().cast());
    // `:275` — `hidden / kMoeVecWidth`, which crosses as the kernel's width
    // operand as well as sizing the grid.
    let width = if vectorizable { hidden / MOE_VEC_WIDTH } else { hidden };
    // `:276` and `:283` — the SAME expression over the two widths:
    // `dim3(aligned_rows, ceil(width / 256))`. Written once because the two
    // arms differ in their width and in nothing else about the grid.
    let launch = Launch {
        grid: [aligned_rows.unsigned_abs(), width.unsigned_abs().div_ceil(DISPATCH_BLOCK), 1],
        block: [DISPATCH_BLOCK, 1, 1],
        smem: 0,
    };
    unsafe {
        if vectorizable {
            moe_dispatch::raw::reorder_moe_aligned_output_vec::<bf16>(
                "moe::reorder_moe_aligned_output_vec_bf16",
                launch,
                aligned_out,
                sorted_route_ids,
                route_out,
                num_routes,
                aligned_rows,
                width,
                shared_row_begin,
                num_tokens,
                shared_out,
                stream,
            );
        } else {
            moe_dispatch::raw::reorder_moe_aligned_output::<bf16>(
                "moe::reorder_moe_aligned_output_scalar_bf16",
                launch,
                aligned_out,
                sorted_route_ids,
                route_out,
                num_routes,
                aligned_rows,
                width,
                shared_row_begin,
                num_tokens,
                shared_out,
                stream,
            );
        }
    }
    Fired::Launched
}

/// `moe::moe_align_decode` — the block-padded counting sort: routes to
/// expert-sorted padded rows, plus the per-block expert id and the inverse
/// permutation.
///
/// # Geometry
///
/// [`router_sort`], which is `<<<1, 1024, (3E + 34) * 4>>>`. **One block, and
/// that is the rule rather than a coincidence of this fire's routing**: the
/// exclusive scan over per-expert padded counts is block-wide and the
/// counters live in the shared slab, so a grid with a row axis would launch N
/// copies of the sort, each zeroing the counters the others were accumulating
/// into. Nothing about that fails — it returns, the permutation is a
/// permutation, the batched GEMM consumes it, and the mixture answers with
/// tokens delivered to experts the router did not choose.
///
/// The slab's five regions — `counts`, `offsets` (+1), `fill`, 32 warp
/// partials, one running base — add to exactly `3E + 34` words.
///
/// # `num_tokens_past_padded` is a nullable output
///
/// `Source::Lit(Lit::Null)` in the deleted row, and null here by default at
/// the bind: the Marlin and Triton grouped GEMMs read it, cuBLAS's does not,
/// and the kernel guards the store.
///
/// # Safety
///
/// `topk_idx` is `[num_routes]` i32 with every entry in `[0, num_experts)`;
/// `sorted_route_ids` and `route_to_aligned_row` are writable for
/// `[num_routes]`, `expert_ids` for `[max_blocks]`;
/// `num_tokens_past_padded` is null or one writable i32. `block_size *
/// max_blocks` is the padded rectangle's row count — the two ride the param
/// channel because no `Source` divides.
#[cfg(feature = "_cuda")]
pub unsafe fn moe_align_decode(
    topk_idx: *const i32,
    sorted_route_ids: *mut i32,
    expert_ids: *mut i32,
    route_to_aligned_row: *mut i32,
    num_routes: i32,
    num_experts: i32,
    block_size: i32,
    max_blocks: i32,
    num_tokens_past_padded: *mut i32,
    stream: *mut c_void,
) -> Fired {
    if num_routes <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_routes" });
    }
    // The slab is sized from this, so a zero would ask the driver for four
    // bytes and let thread 0 scan past the end of them.
    if num_experts <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_experts" });
    }
    unsafe {
        moe_dispatch::raw::moe_align_decode::<i32>(
            "moe::moe_align_decode",
            router_sort(num_experts.unsigned_abs()),
            topk_idx,
            sorted_route_ids,
            expert_ids,
            route_to_aligned_row,
            num_routes,
            num_experts,
            block_size,
            max_blocks,
            num_tokens_past_padded,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::moe_bucket_exact` — the UNPADDED sort: exact per-expert counts, for a
/// host to build cuBLAS grouped shapes from.
///
/// # Geometry, and the 132 bytes
///
/// `moe_dispatch.cu:119-135`, the whole body:
///
/// ```text
/// if (num_routes <= 0 || num_experts <= 0) return;
/// constexpr int BS = 1024;
/// const std::size_t smem =
///     static_cast<std::size_t>(3 * num_experts + 1) * sizeof(std::int32_t);
/// device::moe_bucket_exact<device::i32><<<1, BS, smem, stream>>>(
///     topk_idx, sorted_route_ids, route_to_sorted_row, counts_out,
///     num_routes, num_experts);
/// ```
///
/// `(3E + 1) · 4` and NOT [`router_sort`]'s `(3E + 34) · 4`. Both are
/// correct and they are about different sorts: the padded
/// [`moe_align_decode`] next door runs a warp-partial scan and wants 32 words
/// of partial sums plus a running base, thirty-three words this one does not,
/// because its own scan is serial on thread 0. The rule over-allocated 132
/// bytes for this kernel, which `runtime/launch.rs` blessed in these words —
/// one slab size for both sorts is a rule, two would be two rules that differ
/// by a constant, and over-allocating dynamic shared memory is legal while
/// under-allocating is a launch failure or, worse, a silent overlap. 132
/// bytes against an L40S's 100 KB per block is not a number that changes an
/// occupancy.
///
/// **This is the whole reason the symbol split.** What no rule could do is
/// state the launcher's OWN number, and a dynamic shared allocation sized
/// from an OPERAND is exactly what `execution::Control::Supplies` named. In
/// fn-world it is one expression in a `Launch` literal, and the split it
/// forced is over: the row's `_dev` suffix outlives it only as a symbol
/// string.
///
/// # ONE BLOCK — see [`moe_align_decode`], same argument, same kernel shape.
///
/// # Instantiated at `device::i32` because the indices ARE the data
///
/// `moe_dispatch.cuh` carries `static_assert(is_same<T, i32>::value, "the
/// routing indices are i32")`, so a declaration naming any other element type
/// trips at compile rather than at fire.
///
/// # Safety
///
/// `topk_idx` is `[num_routes]` i32 with every entry in `[0, num_experts)`;
/// `sorted_route_ids` and `route_to_sorted_row` are writable for
/// `[num_routes]` i32; `counts_out` for `[num_experts]` i32. An out-of-range
/// expert id indexes past the shared slab.
#[cfg(feature = "_cuda")]
pub unsafe fn moe_bucket_exact(
    topk_idx: *const i32,
    sorted_route_ids: *mut i32,
    route_to_sorted_row: *mut i32,
    counts_out: *mut i32,
    num_routes: i32,
    num_experts: i32,
    stream: *mut c_void,
) -> Fired {
    // `:128`, both terms, kept apart so the caller learns which it hit.
    if num_routes <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_routes" });
    }
    // Load-bearing twice: the extent the sort buckets over AND the operand
    // the shared slab is sized from.
    if num_experts <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_experts" });
    }
    unsafe {
        moe_dispatch::raw::moe_bucket_exact::<i32>(
            "moe::moe_bucket_exact_dev",
            Launch {
                // `:132` — `<<<1, BS, smem, stream>>>`, `BS = 1024` at `:129`.
                grid: [1, 1, 1],
                block: [SORT_BLOCK, 1, 1],
                // `:130-131` — `(3 * num_experts + 1) * sizeof(int32)`. The
                // guard above is what makes the multiply safe in `u32`.
                smem: (3 * num_experts.unsigned_abs() + 1) * FLOAT,
            },
            topk_idx,
            sorted_route_ids,
            route_to_sorted_row,
            counts_out,
            num_routes,
            num_experts,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::gather_moe_aligned_inputs_bf16` — gathers token rows into the
/// expert-sorted, block-padded rectangle the routed GEMMs read.
///
/// # Geometry
///
/// [`elementwise_rows`] over the PADDED rectangle: one block row per
/// `aligned_rows`, `ceil(hidden / 256)` tiles wide. The rule opens its grid
/// over the OUTPUT's rows and the output here IS that rectangle — which is
/// the whole of why this one is rowed and [`reorder_moe_aligned_output_bf16`]
/// is not.
///
/// `aligned_rows` stays an operand even though the grid is exactly that deep:
/// the kernel guards on it, and dropping an operand because a rule happens to
/// make its guard unreachable is how a kernel breaks when the rule changes.
///
/// `num_tokens` is the FIRE's rows and not the grid's — the two differ here,
/// which is exactly why `Source::Rows` and `Dims::rows` were two things.
///
/// `shared_row_begin` is `-1` at every call site in the deleted tree, and the
/// contract states it once rather than each arm restating it.
///
/// # Safety
///
/// `norm_x` is `[num_tokens, hidden]` bf16, `sorted_route_ids`
/// `[aligned_rows]` i32, `aligned_in` writable for `[aligned_rows, hidden]`
/// bf16.
#[cfg(feature = "_cuda")]
pub unsafe fn gather_moe_aligned_inputs_bf16(
    norm_x: *const bf16,
    sorted_route_ids: *const i32,
    aligned_in: *mut bf16,
    num_routes: i32,
    aligned_rows: i32,
    top_k: i32,
    hidden: i32,
    shared_row_begin: i32,
    num_tokens: i32,
    stream: *mut c_void,
) -> Fired {
    if aligned_rows <= 0 {
        return Fired::Declined(Refusal::Empty { what: "aligned_rows" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    unsafe {
        moe_dispatch::raw::gather_moe_aligned_inputs::<bf16>(
            "moe::gather_moe_aligned_inputs_bf16",
            elementwise_rows(aligned_rows.unsigned_abs(), hidden.unsigned_abs()),
            norm_x,
            sorted_route_ids,
            aligned_in,
            num_routes,
            aligned_rows,
            top_k,
            hidden,
            shared_row_begin,
            num_tokens,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::token_batched_weighted_sum_bf16` — the combine,
/// `out[n, h] = sum_k weights[n, k] * src[n, k, h]`.
///
/// # Geometry
///
/// [`elementwise_rows`] — one block ROW per token, `ceil(hidden / 256)` tiles
/// wide. The C++ launched the TRANSPOSE of this rectangle; the kernel's two
/// index lines and its `dim3` moved together, so the coverage is identical
/// and the guard is `h >= hidden` either way.
///
/// `num_tokens` leaves the argument list because the grid IS the tokens.
///
/// # Safety
///
/// `src` is `[num_tokens, top_k, hidden]` bf16, `weights` `[num_tokens,
/// top_k]` f32, `out` writable for `[num_tokens, hidden]` bf16.
#[cfg(feature = "_cuda")]
pub unsafe fn token_batched_weighted_sum_bf16(
    out: *mut bf16,
    src: *const bf16,
    weights: *const f32,
    num_tokens: i32,
    top_k: i32,
    hidden: i32,
    stream: *mut c_void,
) -> Fired {
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    unsafe {
        moe_dispatch::raw::token_batched_weighted_sum::<bf16>(
            "moe::token_batched_weighted_sum_bf16",
            elementwise_rows(num_tokens.unsigned_abs(), hidden.unsigned_abs()),
            out,
            src,
            weights,
            top_k,
            hidden,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::token_batched_weighted_sum_add_bf16` — the same combine, accumulating
/// onto `out`.
///
/// A separate kernel and not a flag, because a read-modify-write that a branch
/// skips still costs the read. Its contract states `in_place: &[(0, 2)]` — the
/// residual is the statement's THIRD operand and the output aliases it.
///
/// # Geometry
///
/// [`elementwise_rows`], exactly as [`token_batched_weighted_sum_bf16`].
///
/// # Safety
///
/// As [`token_batched_weighted_sum_bf16`], and `out` is read as well as
/// written.
#[cfg(feature = "_cuda")]
pub unsafe fn token_batched_weighted_sum_add_bf16(
    out: *mut bf16,
    src: *const bf16,
    weights: *const f32,
    num_tokens: i32,
    top_k: i32,
    hidden: i32,
    stream: *mut c_void,
) -> Fired {
    if num_tokens <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_tokens" });
    }
    if hidden <= 0 {
        return Fired::Declined(Refusal::Empty { what: "hidden" });
    }
    unsafe {
        moe_dispatch::raw::token_batched_weighted_sum_add::<bf16>(
            "moe::token_batched_weighted_sum_add_bf16",
            elementwise_rows(num_tokens.unsigned_abs(), hidden.unsigned_abs()),
            out,
            src,
            weights,
            top_k,
            hidden,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::scalar_weighted_add_bf16` — `out += weight * src` over a flat run.
///
/// # Geometry
///
/// [`elementwise`], the rule the device row carried.
///
/// # No contract, and therefore no `none:` either
///
/// `weight` is a scalar the deleted arm computed per fire; nothing in the
/// vocabulary names it, and no `table::moe` row was ever written for this
/// symbol. A `none:` is about a DECLARED symbol a trace can state, and no
/// trace states this one — so the declaration compiles the kernel, this `fn`
/// is its host program, and `x::route` never sees the string.
///
/// # Safety
///
/// `out` and `src` each address `n` live elements; `out` is read as well as
/// written and the two may alias exactly (`in_place: &[(0, 0)]` on the device
/// row).
#[cfg(feature = "_cuda")]
pub unsafe fn scalar_weighted_add_bf16(
    out: *mut bf16,
    src: *const bf16,
    weight: f32,
    n: i32,
    stream: *mut c_void,
) -> Fired {
    if n <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the element count" });
    }
    unsafe {
        moe_dispatch::raw::scalar_weighted_add::<bf16>(
            "moe::scalar_weighted_add_bf16",
            elementwise(n.unsigned_abs()),
            out,
            src,
            weight,
            n,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::scatter_add_weighted_bf16` — folds routed rows back onto their
/// destination rows, each scaled by its route's weight.
///
/// # Geometry
///
/// `moe_dispatch.cu:29-41`, the whole body:
///
/// ```text
/// if (num_routed <= 0) return;
/// device::scatter_add_weighted<device::bf16>
///     <<<num_routed, device::kDispatchBlock, 0, stream>>>(
///         static_cast<device::bf16*>(out),
///         static_cast<const device::bf16*>(src),
///         dst_idx, row_weights,
///         hidden);
/// ```
///
/// which is [`per_row`] over the ROUTED rows.
///
/// # The guard has ONE term and the port keeps it at one
///
/// `families::moe`'s row quotes this launcher as `if (num_routed <= 0 ||
/// hidden <= 0) return;`. **That quote is wrong**, and tidying this `fn` to
/// match it would be a change in behaviour dressed as a transcription. There
/// is no `hidden` term: a zero-width row makes the kernel's stride loop `for
/// (h = threadIdx.x; h < hidden; h += kDispatchBlock)` execute zero times, so
/// the launch writes nothing and costs one empty grid. That is a fire with
/// nothing in it, which is not the same event as a refusal, and the C++ was
/// right not to conflate them.
///
/// # `num_routed` is the grid and is NOT an operand
///
/// The `__global__` takes five arguments and `num_routed` is not among them.
/// It reads `blockIdx.x` and has no bound to test it against — the launch
/// geometry IS the bound. `LaunchRule::PerRow` stated the same rectangle from
/// `Dims::rows` and the row wrote down the precondition that made the two
/// agree: the fire's rows must have counted ROUTED SLOTS and not tokens.
/// Nothing checked it. Here it is a named parameter, so a caller has to say
/// which it meant — and the contract below is a `none:` for exactly this
/// reason.
///
/// # The block width is CONTRACT
///
/// 256 is not a tuning number in this kernel. The stride loop advances by the
/// FILE-SCOPE `kDispatchBlock`, not by `blockDim.x`, so a launch at any other
/// width is silently wrong in both directions at once: at a narrower block
/// every row has a slice no thread ever computes, and at a wider one the
/// threads past 256 re-run elements the first 256 already did — on a
/// read-modify-write, which double-adds. Neither faults. [`per_row`] is 256
/// and this is the kernel that makes that number load-bearing.
///
/// # Safety
///
/// `dst` is writable bf16 addressable at every `dst_idx[r] * hidden`; `src`
/// is `[num_routed, hidden]` bf16; `dst_idx` and `row_weights` are
/// `[num_routed]`. The accumulate is `atomicAdd`-free and rows may collide,
/// which is the point — two routes landing on one token is what makes this a
/// sum — so `dst` must not alias `src`.
#[cfg(feature = "_cuda")]
pub unsafe fn scatter_add_weighted_bf16(
    dst: *mut bf16,
    src: *const bf16,
    dst_idx: *const i32,
    row_weights: *const f32,
    num_routed: i32,
    hidden: i32,
    stream: *mut c_void,
) -> Fired {
    // `:34`, and it is the whole guard.
    if num_routed <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_routed" });
    }
    unsafe {
        moe_dispatch::raw::scatter_add_weighted::<bf16>(
            "moe::scatter_add_weighted_dev_bf16",
            // `:36` — `<<<num_routed, device::kDispatchBlock, 0, stream>>>`.
            per_row(num_routed.unsigned_abs()),
            dst,
            src,
            dst_idx,
            row_weights,
            hidden,
            stream,
        );
    }
    Fired::Launched
}

/// `moe::add_moe_route_bias_bf16` — adds each route's expert bias onto that
/// route's row, in place.
///
/// # Geometry
///
/// `moe_dispatch.cu:137-147`, the whole body:
///
/// ```text
/// if (num_routes <= 0 || cols <= 0) return;
/// device::add_moe_route_bias<device::bf16>
///     <<<num_routes, device::kDispatchBlock, 0, stream>>>(
///         static_cast<device::bf16*>(out),
///         static_cast<const device::bf16*>(bias),
///         topk_idx, num_routes, cols, out_stride);
/// ```
///
/// which is [`rms`] over the routes, digit for digit.
///
/// # This one IS width-agnostic, unlike its neighbour
///
/// `add_moe_route_bias` strides by `blockDim.x`, so 256 here is a tuning
/// choice and not a contract. The contrast with [`scatter_add_weighted_bf16`]
/// one function up is worth holding: same header constant, same file, one of
/// them free to change and one of them not.
///
/// # Why the kernel exists at all
///
/// Marlin's own bias epilogue would do this for free, and cannot be used.
/// GPT-OSS publishes its expert biases at the UNPADDED intermediate width
/// while the packed weights are padded to a multiple of 128, so the epilogue
/// — which indexes `[num_experts, prob_n]` with a single stride — reads the
/// wrong column for every row past the first. Two strides, two operands, and
/// a separate kernel is the cheapest way to say so.
///
/// # `cols` and `out_stride` are why the contract is a `none:`
///
/// Those two numbers are the BIAS's row width and the route-major staging's
/// PITCH, and they differ for the reason just given. A fire that splits a
/// fused bias holds neither as an extent of a value it named, so sourcing
/// them would be inventing an edge in the trace. The deleted row stated four
/// `Source`s and left these two blank — exactly the half-bound row
/// `families/rope.rs` warned about, *"a row whose unbound cells look like an
/// oversight rather than a fact"* — and here it is a fact.
///
/// # Safety
///
/// `out` is writable bf16 for `[num_routes, out_stride]` and is read as well
/// as written; `bias` is `[num_experts, cols]` bf16; `topk_idx` is
/// `[num_routes]` i32 with every entry a valid expert. `cols <= out_stride`
/// or the add runs off each row's end.
#[cfg(feature = "_cuda")]
pub unsafe fn add_moe_route_bias_bf16(
    out: *mut bf16,
    bias: *const bf16,
    topk_idx: *const i32,
    num_routes: i32,
    cols: i32,
    out_stride: i32,
    stream: *mut c_void,
) -> Fired {
    // `:141`, both terms, kept apart: an empty rectangle and an absent bias
    // are different facts and the C++ could not tell them apart.
    if num_routes <= 0 {
        return Fired::Declined(Refusal::Empty { what: "num_routes" });
    }
    // `cols` is the BIAS's row width, so a zero means there is no bias to add
    // rather than an empty output rectangle.
    if cols <= 0 {
        return Fired::Declined(Refusal::Empty { what: "the bias width" });
    }
    unsafe {
        moe_dispatch::raw::add_moe_route_bias::<bf16>(
            "moe::add_moe_route_bias_dev_bf16",
            // `:143` — `<<<num_routes, device::kDispatchBlock, 0, stream>>>`.
            rms(num_routes.unsigned_abs()),
            out,
            bias,
            topk_idx,
            num_routes,
            cols,
            out_stride,
            stream,
        );
    }
    Fired::Launched
}

/// The aligned MoE path's block size for one forward, from that batch's route
/// count.
///
/// `moe_dispatch.hpp:113-148`. **The header comment is the measurement and it
/// survives the port verbatim:**
///
/// > Block size for the aligned MoE path above. Every expert's routes are
/// > padded up to a multiple of this, so the useful value tracks how many
/// > rows an expert actually receives (routes / experts). A full 384-expert
/// > checkpoint gets ~3 rows per expert at batch 128 and needs a small block
/// > or it pads 3 rows up to 64; a reduced expert bank gets ~128 and wants
/// > fat blocks so the batched GEMM has a usable M dimension. **Measured on
/// > kimi26-mini at batch 128, moe_prefill: 16 -> 1.184 ms, 32 -> 0.811,
/// > 64 -> 0.746, 128 -> 0.796** -- it turns back up at 128 because eight
/// > blocks no longer fill the GPU, hence the cap.
/// >
/// > Callers pick this per forward from that batch's route count, so scratch
/// > must be sized for both extremes: `kMoeAlignedBlockMin` yields the most
/// > blocks, the value returned here for the largest batch yields the most
/// > padded rows.
///
/// `crates/model/src/glm_5/spec.rs:63` and
/// `crates/model/src/qwen_3_5/forward/mod.rs:12` both name this function in
/// prose, and `model-compiler/src/trace.rs:95` says the driver computes the
/// same number from it. It had no C++ CALLER — the numbers reach the kernels
/// through the plan — so nothing but those references moves with it.
///
/// # It is not a launch and it never was
///
/// No `<<<>>>`, no `Fired`: this is host arithmetic that decides a PADDING,
/// and its answer reaches the kernels as the `block_size` operand of
/// [`moe_align_decode`]. It lives beside them because the numbers above are
/// about these kernels and about nothing else.
///
/// # The `forced` override is DELETED, not ported, and its arms never differed
///
/// The C++ opened with a static lambda that read as an environment knob and
/// was not one:
///
/// ```text
/// static const int forced = [] {
///     return 0;
///     const int parsed = 0;
///     return (parsed >= 8 && parsed <= 256 && (parsed & (parsed - 1)) == 0)
///                ? parsed : 0;
/// }();
/// if (forced != 0) return forced;
/// ```
///
/// The lambda's FIRST statement is `return 0`. Everything after it is
/// unreachable, `forced` is a compile-time zero, and `if (forced != 0)` is a
/// branch whose taken arm cannot be entered. That is §30's reading of
/// `PIE_QWEN35_GDN_SMEM_STEP` arrived at without measuring anything: a host
/// `if` selecting between two behaviours that cannot differ is a deletion and
/// not a port. Reproducing it in Rust would resurrect a knob the C++ had
/// already switched off.
#[must_use]
pub fn moe_aligned_block(routes: i32, num_experts: i32) -> i32 {
    if num_experts <= 0 {
        return MOE_ALIGNED_BLOCK_MIN;
    }
    let per_expert = routes / num_experts;
    let mut block = MOE_ALIGNED_BLOCK_MIN;
    while block * 2 <= MOE_ALIGNED_BLOCK_MAX && block * 2 <= per_expert {
        block *= 2;
    }
    block
}

// ---------------------------------------------------------------------------
// What a trace may say
// ---------------------------------------------------------------------------

contract! {
    /// The short-K grouped GEMM over the padded, expert-sorted batch.
    ///
    /// `c` is an OPERAND as well as the result: the aligned staging's
    /// addresses are baked into the pointer arrays `build_moe_ptrs_aligned`
    /// fills, so this GEMM's destination is the buffer that build named and
    /// not one the arena may pick freshly. That is what `in_place` claims,
    /// and in fn-world it is the whole claim — the deleted arm carried a
    /// `stage_d2d` device-to-device copy as a fallback for a planner that
    /// ignored the request, and a bind body has no device API to copy with.
    /// `rope::rope_bf16`, the pilot, crossed with the same fallback dropped.
    ///
    /// The two block numbers come off the param channel because the operands
    /// carry only their PRODUCT — the aligned rectangle's leading extent.
    /// `n` and `k` need no help: they are the result's and the operand's own
    /// row widths.
    ///
    /// # The THIRD contract here with no [`Entry`](crate::x::Entry)
    ///
    /// A driver op, and the only one of the three that reached the shape from
    /// a WORKING bind rather than from a gap. The predicate is
    /// [`supported`](super::supported): inside it the WMMA kernel is 3.0x the
    /// library at both of this model's shapes, outside it there is no kernel
    /// at all, and qwen3.5 decode fires one statement on each side —
    /// `gate_up` at `K = 2048` against a `SHORT_K` of 512, `down` at
    /// `K = 512`. One symbol, two implementations, and the choice cannot be
    /// made in a `bind!` body: the outside-the-predicate one is a batched
    /// cuBLAS call over the six pointer arrays `BUILD_MOE_PTRS_ALIGNED` fills,
    /// which needs the driver's cuBLAS handle (§3.3 forbids `Cx` to hand one
    /// over) and arrays that are nobody's operand.
    ///
    /// Nor can the choice be left to the caller. `bind/mod.rs` states *"a
    /// refusal is not a fallthrough"*: a `Refusal::Wide` for `K = 2048`
    /// returned from a bind is the answer the driver reports, not the first
    /// half of a decision it finishes. So the whole choice moves to the one
    /// place that holds both implementations' inputs —
    /// `driver-cuda/src/fire/moe_grouped.rs`, called from `bind/mod.rs`'s
    /// driver-op table.
    ///
    /// `in_place` is what makes the batched leg's destination this
    /// statement's: the arrays were baked with the staging's base addresses
    /// and the library writes there, so a planner that handed result 0 a
    /// fresh buffer would leave the WMMA leg right and this one writing bytes
    /// the swiglu never reads. The claim below is the load-bearing one on
    /// this leg, not a hint.
    MOE_GROUPED_GEMM = "moe::moe_grouped_gemm_bf16" as moe_grouped_gemm {
        in_place: &[(0, 2)],
    }

    /// Fold a per-expert scale into the router weights.
    ///
    /// `in_place` is stated HERE and was not on the deleted row, because the
    /// deleted row stated no `Source` at all and so never had to say where
    /// the result lands. The kernel read-modify-writes `topk_w`, which
    /// `dsl.rs:4272` passes as input 1 and re-declares as result 0; without
    /// the pair the planner may hand result 0 a fresh buffer that nothing
    /// ever writes. The claim is the `.cuh`'s: `topk_w[i] *= scale[e]`.
    ///
    /// **THE CLAIM IS AHEAD OF ITS USE AND THAT IS SAID RATHER THAN GUESSED.**
    /// `dsl::cuda::apply_per_expert_scale` exists and no model calls it:
    /// nothing under `crates/model/src` names the wrapper and nothing names
    /// the symbol, so no planner has ever been handed this pair and no fire
    /// has ever tested whether it honours it. The pair is stated because the
    /// kernel read-modify-writes and the shape of that fact does not depend
    /// on a caller arriving; what it does mean is that the FIRST caller is
    /// also the first check, and a planner that ignores `in_place` would
    /// surface there and not here.
    APPLY_PER_EXPERT_SCALE = "moe::apply_per_expert_scale_bf16" as apply_per_expert_scale {
        in_place: &[(0, 1)],
    }

    /// Add each route's expert bias. `topk_idx` is route-global, so a row
    /// window would pick the wrong experts' biases.
    ADD_MOE_ROUTE_BIAS = "moe::add_moe_route_bias_bf16" as add_moe_route_bias {
        whole: true,
    }

    /// The per-expert group-scale plane transpose.
    TRANSPOSE_EXPERT_SCALES = "moe::transpose_expert_scales_u8" as transpose_expert_scales

    /// deepseek_v4's router: `sqrt(softplus(x))` over the logits.
    TOPK_SQRTSOFTPLUS = "moe::topk_sqrtsoftplus_bf16" as topk_sqrtsoftplus

    /// Expert INDICES from a table keyed by token id — a route that is a pure
    /// function of the token rather than of its activations. The WEIGHTS still
    /// come from the router logits, so the logits GEMM above it does not go
    /// away.
    HASH_ROUTE_LOOKUP = "moe::hash_route_lookup" as hash_route_lookup

    /// The fp32-logits sigmoid router with a correction bias.
    TOPK_SIGMOID_BIAS = "moe::topk_sigmoid_bias_fp32" as topk_sigmoid_bias

    /// The UNPADDED counterpart of `moe_align`: exact per-expert counts the
    /// host reads to build cuBLAS grouped shapes. `whole` for the same reason
    /// — the sort is over all routes.
    MOE_BUCKET_EXACT = "moe::moe_bucket_exact" as moe_bucket_exact {
        whole: true,
    }

    /// Bucket routes by expert and pad each bucket to whole blocks.
    ///
    /// `whole` for the reason every member of the aligned path is: the
    /// permutation is computed over ALL routes in the fire, so a statement
    /// addressed through `sorted_route_ids` cannot take a row window — the
    /// window would name different routes than the sort did.
    ///
    /// The deleted row's third result carried a claim worth keeping:
    /// *"`route_to_aligned_row` is BOUND, where the arm passed null. The
    /// statement declares three results and the arena places all three; the
    /// inverse map is the one this leg's combine does not read, and 'declared
    /// but not written' is a claim the declaration does not make."* In
    /// fn-world it is simply the fourth parameter and the caller decides,
    /// which is the same answer reached by not having the question.
    MOE_ALIGN = "moe::moe_align_decode" as moe_align {
        whole: true,
    }

    /// Gather the aligned rectangle's rows from the token-ordered input.
    ///
    /// # What the row world could not say here, and what is left
    ///
    /// The deleted row headed a note titled *"THE THREE THAT REMAIN
    /// UNSTATED"*, and its blocker was arithmetic: *"they take the ROUTE
    /// count and `top_k` as separate arguments, and neither is reachable
    /// here … Both numbers ARE in the aligned dim's packed word. What is
    /// missing is a way to say 'the fire's rows times a number packed in a
    /// dim', and the table deliberately has no arithmetic: an expression
    /// language here is one more place a binding can be wrong, checked by
    /// nothing."*
    ///
    /// **A `bind!` body IS that expression language, and it is checked by the
    /// compiler**, so two thirds of the blocker went at once: `num_routes` is
    /// `cx.rows().count * top_k`, which is what `Source::RoutesOfParam(0)`
    /// meant, and `top_k` rides the param channel exactly as the row said.
    /// The last third was one number no arithmetic over the fire's rows can
    /// reach — `aligned_rows`, the PADDED rectangle's height, which is
    /// `Source::InRows(1)` on the row and an operand's own extent. That was
    /// the single missing `Cx` query this arm's `none:` named, it is
    /// `Cx::in_rows` since `a41a1df0a`, and it is why three unstated rows
    /// became one unstated fact and then one bind.
    GATHER_MOE_ALIGNED_INPUTS = "moe::gather_moe_aligned_inputs_bf16" as gather_moe_aligned_inputs {
        whole: true,
    }

    /// Fill the six pointer arrays the batched GEMMs read — and DECLARE the
    /// three staging buffers everything below the build writes into.
    ///
    /// # The second contract here with no [`Entry`](crate::x::Entry)
    ///
    /// A driver op, for the same reason `MOE_FUSED_CUTLASS` above is one and
    /// by a different road. This symbol was a `none:` arm and had been
    /// unsourced in the deleted row before that, so it has never had an arm
    /// in either world; what changed is not the gap but the reading of it.
    ///
    /// **The six arrays have no stated consumer.** They are read by the
    /// batched-cuBLAS fallback INSIDE `moe::moe_grouped_gemm_bf16`, which is
    /// a lowering of that statement and not a statement of its own — the
    /// grouped GEMM's parameter list is `(a, weight_base, c, expert_ids)` and
    /// names no pointer array. So the dtype the old sentence asked for would
    /// not have been enough and would not even have been safe: six trace
    /// results nothing reads are freed by `lower.rs:1911`'s liveness at the
    /// first op past the build, and the batched GEMM would dereference
    /// pointer arrays whose bytes the next allocation took. A wrong answer,
    /// not a refusal, and the same failure `lower.rs:1949` records for the
    /// rotated `k`. Stating them properly means stating an operand only ONE
    /// of two lowerings reads, which is what this migration is retiring.
    ///
    /// So: `contract!` yes, `Entry` no, and **no `none:` arm** — a `none:`
    /// here would put an `Entry` in `x::moe::ENTRIES`, and `x/mod.rs`'s "THE
    /// ONE OVERLAP" says an `Entry` shadows the `DriverOp` arm and turns a
    /// symbol that fires into `Route::Unbound` at model load. The body is
    /// `driver-cuda/src/fire/moe_ptrs.rs`; the arm that calls it belongs in
    /// `bind/mod.rs`'s driver-op table, beside `pie_lora_qkv_correction`.
    ///
    /// `whole: true` stays and is now load-bearing twice: the build fixes
    /// where the aligned staging LIVES, so a row window over it would name
    /// addresses the two GEMMs below do not have.
    BUILD_MOE_PTRS_ALIGNED = "moe::build_moe_ptrs_aligned_bf16" as build_moe_ptrs_aligned {
        whole: true,
    }

    /// The gather's other half: undo the permutation.
    REORDER_MOE_ALIGNED_OUTPUT = "moe::reorder_moe_aligned_output_bf16" as reorder_moe_aligned_output {
        whole: true,
    }

    /// `out[dst_idx[i]] += src[i]·w[i]`, and `dst_idx` is route-global: a
    /// window over output ROWS is not a window over routes.
    SCATTER_ADD_WEIGHTED = "moe::scatter_add_weighted_bf16" as scatter_add_weighted {
        whole: true,
    }

    /// The sigmoid router. The exception among the aligned path's neighbours,
    /// and it is the router: a token's top-k reads only its own logits row, so
    /// this one splits like any elementwise statement.
    TOPK_SIGMOID = "moe::topk_sigmoid_bf16" as topk_sigmoid

    /// The softmax router, which takes no deployment constants and is the only
    /// router in this family that binds.
    TOPK_SOFTMAX = "moe::topk_softmax_bf16" as topk_softmax

    /// The whole routed block as one call — permute, both grouped GEMMs, the
    /// activation and the weighted finalize.
    ///
    /// # The one contract here with no [`Entry`](crate::x::Entry)
    ///
    /// The third registration shape, and the reason is a boundary and not a
    /// difficulty: this symbol's body is `driver-cuda/src/fire/flashinfer_moe.rs`,
    /// which reaches CUTLASS through five `extern "C"` seams into
    /// `csrc/src/moe/flashinfer_moe.cu`. It needs a workspace query, a
    /// workspace allocation and a device API surface [`Cx`](crate::x::Cx) does
    /// not have and must not grow, so it is a driver op: `contract!` yes,
    /// `Entry` no.
    ///
    /// It left `execution::RUST_SERVED` with the rest of the family even so,
    /// and the reason is mechanical rather than a change of mind: that list
    /// exists to drop a symbol's ahead-of-time C shim entry, and
    /// `every_taken_over_row_is_stated` asks each name on it for a
    /// `table::sig` with a NON-EMPTY operand list. `table::sig` answers this
    /// symbol from `x::moe::SIGS` now, and a `Contract::sig` states no
    /// operands — so the entry is already dropped and naming it would only
    /// make the test right to refuse it. `bind::service::
    /// moe_flashinfer_cutlass_moe_bf16` and `fire::flashinfer_moe` both STAY:
    /// the executor did not move, only the declaration did.
    ///
    /// The contract is still stated because `model-compiler` must be able to
    /// answer "is this a symbol?" — [`crate::x::SIGS`] is where that answer
    /// lives once the row is gone.
    MOE_FUSED_CUTLASS = "moe::flashinfer_cutlass_moe_bf16" as moe_fused_cutlass

    /// The decode GEMV's gate/up leg. The expert axis rides INSIDE the value,
    /// so this leg is a list of rectangles and every number is an operand's
    /// own extent.
    MOE_GATE_UP_GEMV = "moe::moe_gate_up_decode_gemv_bf16" as moe_gate_up_gemv

    /// The down leg: `h` is what it WRITES per route and `i_moe` what it
    /// reads, which is the mirror of the gate/up contract above.
    MOE_DOWN_GEMV = "moe::moe_down_decode_gemv_bf16" as moe_down_gemv

    /// The combine.
    MOE_WEIGHTED_SUM = "moe::token_batched_weighted_sum_bf16" as moe_weighted_sum

    /// The `_add` spelling accumulates into the residual, which the statement
    /// carries as its THIRD operand (`weighted_sum_add(x, weights, residual)`);
    /// the plain spelling above writes a fresh value and aliases nothing. One
    /// launch where the semantic text has a `WeightedSum` and a `ResidualAdd`.
    MOE_WEIGHTED_SUM_ADD = "moe::token_batched_weighted_sum_add_bf16" as moe_weighted_sum_add {
        in_place: &[(0, 2)],
    }
}

// ---------------------------------------------------------------------------
// What happens when a trace says it
// ---------------------------------------------------------------------------

// SEVENTEEN arms for twenty contracts. `MOE_FUSED_CUTLASS`,
// `BUILD_MOE_PTRS_ALIGNED` and `MOE_GROUPED_GEMM` have none, on purpose and
// by the third registration shape: an `Entry` here — a bind OR a `none:` —
// would shadow the driver op that fires the symbol, and `Route::Unbound`
// refuses a live model at load.
//
// THIRTEEN bind, FOUR refuse, and the four are one group rather than three:
// their rows stated no `Source` on at least one operand, so there was
// nothing to derive a binding from, no arm was ever generated, and nothing
// fires them today either. A `none:` here is what they already did, said out
// loud.
//
// It was EIGHT and ELEVEN for one round. The other six were blocked on three
// `Cx` queries the row world had and the floor did not — `moe_norm_topk` and
// `moe_routed_scaling` for the four routers, `in_rows(i)` for the aligned
// path's gather and reorder — and five of the six were live, so the port
// shipped a stated regression rather than a bind that guessed a deployment
// constant. `a41a1df0a` landed the three queries and the three `Facts`
// methods under them; these six arms are the other half of that change, and
// the header's floor block is kept as the record of the ask.
//
// It was FOURTEEN binds until the aligned leg was finished. `MOE_GROUPED_GEMM`
// left DOWNWARD — the only symbol in this family that was bound, worked, and
// became a driver op anyway — because half of its shapes have no kernel and
// the implementation that serves them needs the driver's cuBLAS handle. See
// the tombstone where its arm stood.
#[cfg(feature = "_cuda")]
bind! {
    // `MOE_GROUPED_GEMM` STOOD HERE and is GONE. It did not become a
    // refusal: it became `x::moe`'s THIRD DRIVER OP, so it must have no
    // `Entry` at all — a bind OR a `none:` shadows `route()`'s `DriverOp`
    // arm, and the shadowed answer would refuse at load a symbol the
    // driver-op table fires.
    //
    // IT IS THE ONLY ONE OF THE THREE THAT WAS BOUND AND WORKING. The other
    // two never had an arm in either world. This one fired the WMMA kernel
    // for every shape `supported` accepts, and still does — through
    // `fire::moe_grouped`, which asks the same predicate and then chooses.
    // What the bind could not do is the OTHER side of the predicate:
    //
    //   gate_up   M=16  N=1024  K=2048   K > SHORT_K  ->  batched cuBLAS
    //   down      M=16  N=2048  K=512    supported    ->  WMMA, as before
    //
    // and qwen3.5 decode takes both. A `bind!` body cannot make that call:
    // the batched form needs the cuBLAS handle, which §3.3 forbids `Cx` to
    // hand over, and the six device pointer arrays, which are not operands
    // of anything. And it cannot be left to the caller either, because
    // `bind/mod.rs` states *"a refusal is not a fallthrough"* — a `Wide` for
    // `K = 2048` returned from here is the final answer, not the first half
    // of a choice. So the choice moves to where both resources are.
    //
    // The mapping this arm made, kept because the driver-op arm re-derives
    // every one of these from `bound`/`spec` and nothing connects the two:
    //
    //   a           `cx.arg_in(0)`     the aligned rectangle
    //   weight_base `cx.weight(0)`     the `[E, N, K]` bank, `spec.weight`
    //   c           `cx.arg_out(0)`    the staging, in place over `arg_in(2)`
    //   expert_ids  `cx.arg_in(1)`     what the kernel indexes the bank by
    //   max_blocks  `cx.param(1)`      the two block numbers the operands
    //   m           `cx.param(0)`        carry only the PRODUCT of
    //   n           `cx.out_width(0)`  the result's row width
    //   k           `cx.in_width(0)`   the operand's
    //
    // Body: `driver-cuda/src/fire/moe_grouped.rs`. The batched leg it
    // reaches, `x::gemm::dense::batched_act_x_wt_bf16`, was already in the
    // tree with no caller — §38 struck its row and §45.2 ported it anyway.

    APPLY_PER_EXPERT_SCALE => { cx, stream => {
        // `n` and `k` were two operands of the ahead-of-time twin and are
        // one here: the kernel only ever used their product, and the
        // product is the route count.
        let top_k = cx.in_width(1)?;
        unsafe {
            apply_per_expert_scale_bf16(
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.arg_in(1)?.cast::<f32>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.rows().count.saturating_mul(top_k),
                stream,
            )
        }
        .ok()
    }},

    ADD_MOE_ROUTE_BIAS => { none: "Nothing states the destination's row \
        PITCH, and that is the whole of it. Two of this kernel's three \
        numbers are reachable and an earlier version of this sentence said \
        otherwise: dsl.rs:4300 passes topk_idx as input 1, so num_routes is \
        the fire's rows times that operand's own width, exactly as \
        moe::moe_align_decode's is, and cols is the result's own width. \
        out_stride is not reachable -- the kernel writes a slice of a wider \
        rectangle, and a stride is the caller's arithmetic rather than an \
        operand's extent, so no Source spelled one and no Cx query answers \
        one. WHAT WOULD MAKE IT FIRE: a lowering that states the \
        destination pitch, and then a model that calls the wrapper, \
        because today nothing under crates/model/src does" },

    TRANSPOSE_EXPERT_SCALES => { none: "Weight preparation is not a trace \
        statement, and this one is the proof: dsl.rs:4418 records it with \
        inputs vec![], THE ONLY STATEMENT IN THIS FAMILY WITH NO INPUTS AT \
        ALL. It rewrites a checkpoint's per-expert group-scale planes from \
        [experts, k_groups, n] to [experts, n, k_groups] once, over \
        weights, before any fire exists; its row stated no Source on any of \
        its five operands because there is no statement to read one from, \
        and its three numbers are the RESULT's three dims where Cx answers \
        only a width. WHAT WOULD MAKE IT FIRE: nothing from the trace, ever \
        -- it wants the driver-op shape, a call from driver-cuda's weight \
        loader with the host fn above as its body, which is where \
        moe::flashinfer_cutlass_moe_bf16 already sits. A none: here is \
        permanent unless that call is written" },

    TOPK_SQRTSOFTPLUS => { cx, stream => {
        // `Source::Or(&Source::Weight(0), &Source::Lit(Lit::Null))` was the
        // deleted row's fourth operand, and its comment is the claim: A
        // FAMILY WITHOUT A CORRECTION BIAS STATES NO FOURTH OPERAND, and
        // the kernel reads a null as "there is none". `Cx::weight` refuses
        // where the row branched, so the `Or` is spelled here. This is the
        // ONE place in this file a `?` is deliberately not written, and it
        // is not the escape `Cx::weight_bias`'s doc warns about: the row
        // stated the alternative, the alternative is a literal null, and
        // three of the four checkpoints that fire this kernel have no bias.
        let correction_bias =
            cx.weight(0).map_or(core::ptr::null(), |w| w.cast_const().cast::<f32>());
        // The two deployment constants, `Cx` queries since `a41a1df0a` --
        // `Source::Ctx("moe_norm_topk")` and `Source::Ctx("moe_routed_scaling")`
        // on the deleted row, `ctx.moe_norm_topk` and `ctx.moe_routed_scaling`
        // in the arm it generated. Both always answer `Some`.
        unsafe {
            topk_sqrtsoftplus_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<i32>(),
                cx.arg_out(1)?.cast::<f32>(),
                correction_bias,
                cx.rows().count,
                cx.in_width(0)?,
                cx.out_width(0)?,
                cx.moe_norm_topk()?,
                cx.moe_routed_scaling()?,
                stream,
            )
        }
        .ok()
    }},

    HASH_ROUTE_LOOKUP => { cx, stream => {
        // THE OPERAND ORDER HERE IS THE STATEMENT'S, NOT THE KERNEL'S, and
        // the two differ. The deleted row listed the kernel's parameters in
        // the kernel's order and stated NO `Source` on any of them, so it
        // never generated an arm and nothing recorded which input was
        // which. `dsl.rs:4826` does: `vec![token_ids.id, logits.id]`, so
        // TOKEN IDS ARE INPUT 0 AND THE LOGITS ARE INPUT 1 -- which makes
        // `num_experts` the width of input 1 and not of input 0, the one
        // place a reader porting from the parameter list would get it
        // backwards. The table is the statement's single weight
        // (`vec![table.to_string()]`).
        unsafe {
            hash_route_lookup(
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.weight(0)?.cast_const().cast::<i64>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<i32>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                cx.vocab()?,
                cx.in_width(1)?,
                cx.out_width(0)?,
                cx.moe_norm_topk()?,
                cx.moe_routed_scaling()?,
                stream,
            )
        }
        .ok()
    }},

    TOPK_SIGMOID_BIAS => { cx, stream => {
        // The bias is `Source::Weight(0)` FLAT, with no `Or` and no null
        // alternative -- nemotron_h is the one family whose router cannot
        // run without one -- so this is the one router arm where the weight
        // carries a `?` and a checkpoint that lacks it refuses.
        unsafe {
            topk_sigmoid_bias_fp32(
                cx.arg_in(0)?.cast_const().cast::<f32>(),
                cx.weight(0)?.cast_const().cast::<f32>(),
                cx.arg_out(0)?.cast::<i32>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                cx.in_width(0)?,
                cx.out_width(0)?,
                cx.moe_norm_topk()?,
                cx.moe_routed_scaling()?,
                stream,
            )
        }
        .ok()
    }},

    MOE_BUCKET_EXACT => { none: "AHEAD OF A CALLER, AND NO LONGER AHEAD OF \
        A DECLARATION. The statement declared TWO results while the kernel \
        writes THREE buffers: moe_dispatch.cuh:907 takes topk_idx, \
        sorted_route_ids, route_to_sorted_row, counts_out, and the inverse \
        map was named nowhere in this crate. Passing a null for it would \
        not be a wrong answer but a write to null -- the store at :952 has \
        no null guard, which is the one place this kernel differs from its \
        padded twin, whose route_to_aligned_row IS guarded and IS therefore \
        optional. dsl.rs:5121 declares three now, in the kernel's own \
        parameter order, so a binding reads straight down: sorted_route_ids, \
        route_to_sorted_row, counts. The route count was NOT the gap and an \
        earlier version of this sentence said it was: topk_idx IS input 0 \
        and IS [Tokens, top_k], so num_routes reads exactly as \
        moe::moe_align_decode's does, and num_experts is the THIRD result's \
        own extent now that counts has moved behind the inverse map. WHAT \
        WOULD MAKE IT FIRE: a caller. Nothing under crates/model/src names \
        this symbol, and a bind nothing exercises is a claim nothing checks" },

    MOE_ALIGN => { cx, stream => {
        // `num_routes` is `topk_idx`'s element count: the operand IS
        // `[Tokens, top_k]`, so the fire's rows times its own width is
        // exactly what it holds. `whole`, so these rows are the fire's.
        let param = |i: usize| cx.param(i).map(|v| i32::try_from(v).unwrap_or(0));
        let num_routes = cx.rows().count.saturating_mul(cx.in_width(0)?);
        unsafe {
            moe_align_decode(
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.arg_out(0)?.cast::<i32>(),
                cx.arg_out(1)?.cast::<i32>(),
                cx.arg_out(2)?.cast::<i32>(),
                num_routes,
                param(0)?,
                param(1)?,
                param(2)?,
                // The statement declares three results and the arena
                // places all three; this leg's combine does not read the
                // padded-token count, and the deleted row passed the same
                // null.
                core::ptr::null_mut(),
                stream,
            )
        }
        .ok()
    }},

    GATHER_MOE_ALIGNED_INPUTS => { cx, stream => {
        // `Source::RoutesOfParam(0)`, which `abi.rs:1136` rendered
        // `rows.saturating_mul(spec.params[0])`: the one product that is
        // neither an operand's extent nor a load-time number, and the
        // arithmetic a `bind!` body may write where the table could not.
        let top_k = i32::try_from(cx.param(0)?).unwrap_or(0);
        //
        // WHAT THE BIND MAKES VISIBLE AND THE ROW HID. `aligned_rows` was
        // `Source::InRows(1)` and `num_tokens` was `Source::Rows`, two
        // spellings the doc on the `fn` above says name different numbers:
        // *"`num_tokens` is the FIRE's rows and not the grid's -- the two
        // differ here"*. They render to the SAME value: `bind/mod.rs:1596`
        // is `fn rows_of(b, i, rows) { let _ = (b, i); rows }`, and
        // `Facts::in_rows` is that function, bound-checked. So the padded
        // rectangle's height has been the token count since the row
        // shipped. This port reproduces it exactly -- it changes no number
        // that reaches the device -- and states it here because two
        // adjacent arguments are where it is legible and two `operands![]`
        // lines are where it was not. The height lives in
        // `Dim::MoeAlignedRoutes`, which is the compiler's and not `Cx`'s;
        // fixing it is `rows_of`'s job, and the index it already ignores is
        // where that fix goes.
        unsafe {
            gather_moe_aligned_inputs_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<i32>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count.saturating_mul(top_k),
                cx.in_rows(1)?,
                top_k,
                cx.out_width(0)?,
                // `Source::Lit(Lit::I32(-1))`: the -1 every call site in the
                // deleted tree passed, stated once here.
                -1,
                cx.rows().count,
                stream,
            )
        }
        .ok()
    }},

    // `BUILD_MOE_PTRS_ALIGNED` STOOD HERE as a `none:` arm and is GONE, and
    // the sentence it held is the contract's now. It did not become a bind:
    // it became `x::moe`'s SECOND DRIVER OP, so it must have no `Entry` at
    // all. An `Entry` — a bind OR a `none:` — shadows `route()`'s `DriverOp`
    // arm (`x/mod.rs`, "THE ONE OVERLAP"), and for this symbol that is not a
    // theoretical cost: the shadowed answer is `Route::Unbound`, which
    // refuses the model at load for a symbol the driver-op table fires.
    //
    // The five arms this file shipped with are four: `add_moe_route_bias`,
    // `transpose_expert_scales`, `moe_bucket_exact` and
    // `scatter_add_weighted`. Twenty symbols, thirteen binds, three driver
    // ops, four refusals.
    //
    // The dtype the old sentence asked for was the wrong ask, and finding
    // out why is the only part worth keeping: the six arrays have no stated
    // CONSUMER, so a dtype buys a declaration that `lower.rs:1911`'s
    // liveness frees at the next op. `fire/moe_ptrs.rs` is the body and the
    // contract above carries the argument.

    REORDER_MOE_ALIGNED_OUTPUT => { cx, stream => {
        let top_k = i32::try_from(cx.param(0)?).unwrap_or(0);
        // `hidden` reads off the INPUT where the gather reads off its
        // OUTPUT, and the deleted row's comment is why: *"The RESULT is
        // `[Tokens, top_k, hidden]`, so its row width is `top_k * hidden`
        // and not this. The OPERAND is `[aligned, hidden]` -- one row of
        // the aligned rectangle IS the hidden width."* Taking
        // `cx.out_width(0)` here launches `top_k` times too wide.
        //
        // `aligned_rows` carries the same finding the gather states.
        unsafe {
            reorder_moe_aligned_output_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<i32>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count.saturating_mul(top_k),
                cx.in_rows(1)?,
                cx.in_width(0)?,
                -1,
                cx.rows().count,
                // `Source::Lit(Lit::Null)`: this leg writes no shared
                // output, and the row passed the same null.
                core::ptr::null_mut(),
                stream,
            )
        }
        .ok()
    }},

    SCATTER_ADD_WEIGHTED => { none: "THE LAUNCH COUNT IS A DEVICE READBACK, \
        so this is not one statement and no lowering makes it one. \
        dsl.rs:6868, at the combine that IS stated: `the per-expert \
        scatter_add_weighted_bf16 loop is the OTHER combine, and it is not \
        stated here: it runs once per expert with a row count the host \
        learned from a device readback, which is a launch count no \
        declaration fixes`. num_routed is that count and it is the GRID -- \
        the kernel reads its row from blockIdx.x and does not take it as a \
        parameter -- and dst_idx is route-global besides, so a row window \
        is not a route window. A wrapper exists (dsl.rs:5795) and nothing \
        calls it. WHAT WOULD MAKE IT FIRE: not a Source and not a Cx query. \
        Either a kernel that takes its own bound and handles the empty case \
        on the device -- which is what §5.1 says a refusal that cannot be \
        hoisted actually is -- or a driver op that owns the loop and the \
        synchronise it needs" },

    TOPK_SIGMOID => { cx, stream => {
        // The same row as `TOPK_SQRTSOFTPLUS` operand for operand, and the
        // `Or` on the bias is read there. This is the most-fired member of
        // the family -- kimi_k2, kimi_k3 and glm_5 all route through it --
        // and it was the largest of the eleven `none:` arms this port
        // shipped with.
        let correction_bias =
            cx.weight(0).map_or(core::ptr::null(), |w| w.cast_const().cast::<f32>());
        unsafe {
            topk_sigmoid_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<i32>(),
                cx.arg_out(1)?.cast::<f32>(),
                correction_bias,
                cx.rows().count,
                cx.in_width(0)?,
                cx.out_width(0)?,
                cx.moe_norm_topk()?,
                cx.moe_routed_scaling()?,
                stream,
            )
        }
        .ok()
    }},

    TOPK_SOFTMAX => { cx, stream => {
        unsafe {
            topk_softmax_bf16(
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<i32>(),
                cx.arg_out(1)?.cast::<f32>(),
                cx.rows().count,
                cx.in_width(0)?,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    MOE_GATE_UP_GEMV => { cx, stream => {
        // The result is `[Tokens, top_k * i_moe]`, so the intermediate is
        // what is left of a row once the routes are divided out -- and the
        // routes are `topk_idx`'s own width.
        let top_k = cx.in_width(0)?;
        if top_k <= 0 {
            return Err(Refusal::Empty { what: "the route width" });
        }
        unsafe {
            moe_gate_up_decode_gemv_bf16(
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                top_k,
                cx.in_width(1)?,
                cx.out_width(0)? / top_k,
                stream,
            )
        }
        .ok()
    }},

    MOE_DOWN_GEMV => { cx, stream => {
        // The mirror: `h` is what it writes per route, `i_moe` what it
        // reads.
        let top_k = cx.in_width(0)?;
        if top_k <= 0 {
            return Err(Refusal::Empty { what: "the route width" });
        }
        unsafe {
            moe_down_decode_gemv_bf16(
                cx.arg_in(0)?.cast_const().cast::<i32>(),
                cx.arg_in(1)?.cast_const().cast::<bf16>(),
                cx.weight(0)?.cast_const().cast::<bf16>(),
                cx.arg_out(0)?.cast::<bf16>(),
                cx.rows().count,
                top_k,
                cx.out_width(0)? / top_k,
                cx.in_width(1)?,
                stream,
            )
        }
        .ok()
    }},

    MOE_WEIGHTED_SUM => { cx, stream => {
        // `weights` IS `[Tokens, top_k]`, so its row width is the route
        // count, and the result IS `[Tokens, hidden]`.
        unsafe {
            token_batched_weighted_sum_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.rows().count,
                cx.in_width(1)?,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},

    MOE_WEIGHTED_SUM_ADD => { cx, stream => {
        // The residual is input 2 and the result aliases it; the contract
        // states the pair and the planner gives them one address.
        unsafe {
            token_batched_weighted_sum_add_bf16(
                cx.arg_out(0)?.cast::<bf16>(),
                cx.arg_in(0)?.cast_const().cast::<bf16>(),
                cx.arg_in(1)?.cast_const().cast::<f32>(),
                cx.rows().count,
                cx.in_width(1)?,
                cx.out_width(0)?,
                stream,
            )
        }
        .ok()
    }},
}

#[cfg(test)]
mod tests {
    //! What can be checked without a device: that the support predicate says
    //! what `moe/moe_grouped_gemm.cu:18-24` said, and that the block ladder
    //! still lands on the sizes the kimi26-mini table was measured at.
    //!
    //! Ported from `driver-cuda/src/fire/moe.rs`, whose `Decline` enum these
    //! were written against. The values are [`Refusal`]'s now and the shapes
    //! are unchanged: the two rows of the launcher's own measurement table
    //! are still the two cases, and each conjunct still refuses on its own.

    //! The two `supported` tests are `_cuda`-gated because the predicate is:
    //! it returns a [`Refusal`], `Refusal` lives beside [`Fired`], and layer 3
    //! is what `default = []` leaves out. The ladder is pure host arithmetic
    //! and runs in every profile.

    use super::{MOE_ALIGNED_BLOCK_MAX, MOE_ALIGNED_BLOCK_MIN, moe_aligned_block};
    #[cfg(feature = "_cuda")]
    use super::{FRAG, Refusal, SHORT_K, supported};

    /// The shipping shapes from the launcher's own measurement table.
    #[cfg(feature = "_cuda")]
    #[test]
    fn the_measured_shapes_answer_as_they_were_measured() {
        // `down K=256`, taken: M is one fragment, N a multiple of 64.
        // 7.94 ms -> 5.91 ms against cuBLAS, which is why the bound is
        // where it is.
        assert_eq!(supported(16, 2048, 256), Ok(()));
        // `gate_up K=2048`, left on cuBLAS by the K bound and nothing
        // else -- 11.08 ms -> 11.98 ms when it was not.
        assert_eq!(
            supported(16, 2048, 2048),
            Err(Refusal::Wide { what: "K, above which cuBLAS wins", at: 2048, max: SHORT_K })
        );
    }

    /// Each conjunct refuses on its own, and says which.
    #[cfg(feature = "_cuda")]
    #[test]
    fn every_conjunct_is_its_own_decline() {
        // The two directions of `m != kFrag` are two refusals, and a test
        // that only ever passed the taller one is how a backwards `Narrow`
        // survives review.
        assert_eq!(
            supported(32, 2048, 256),
            Err(Refusal::Wide {
                what: "M, which must be exactly one 16-row fragment",
                at: 32,
                max: FRAG,
            })
        );
        assert_eq!(
            supported(8, 2048, 256),
            Err(Refusal::Narrow {
                what: "M, which must be exactly one 16-row fragment",
                at: 8,
            })
        );
        assert_eq!(supported(16, 0, 256), Err(Refusal::Empty { what: "the N by K rectangle" }));
        assert_eq!(supported(16, 2048, 0), Err(Refusal::Empty { what: "the N by K rectangle" }));
        assert_eq!(
            supported(16, 100, 256),
            Err(Refusal::Narrow { what: "N, in whole 64-wide tiles", at: 100 })
        );
        assert_eq!(
            supported(16, 2048, 24),
            Err(Refusal::Narrow { what: "K, in whole 16-deep fragments", at: 24 })
        );
    }

    /// The ladder the block-size table was measured on: double while a
    /// doubling still fits both the ceiling and the per-expert occupancy.
    #[test]
    fn the_block_ladder_stays_between_its_two_bounds() {
        // No experts is not a division; the floor is the answer.
        assert_eq!(moe_aligned_block(0, 0), MOE_ALIGNED_BLOCK_MIN);
        // One route per expert cannot pay for a bigger block.
        assert_eq!(moe_aligned_block(8, 8), MOE_ALIGNED_BLOCK_MIN);
        // A deep fire saturates at the ceiling and never above it.
        assert_eq!(moe_aligned_block(1 << 20, 8), MOE_ALIGNED_BLOCK_MAX);
        // And every answer in between is a power of two on the ladder.
        for experts in [4_i32, 8, 32, 128] {
            for routes in [16_i32, 64, 256, 1024, 4096] {
                let block = moe_aligned_block(routes, experts);
                assert!(block >= MOE_ALIGNED_BLOCK_MIN && block <= MOE_ALIGNED_BLOCK_MAX);
                assert_eq!(block.count_ones(), 1);
            }
        }
    }
}
