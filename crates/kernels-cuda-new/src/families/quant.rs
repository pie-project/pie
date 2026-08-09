//! `quant`'s JIT units — seven headers, thirty-one rows, ten ahead-of-time
//! symbols restated verbatim, and the fourteen kernels that stayed behind.
//!
//! The family's device text now lives in `csrc/src/quant/*.cuh`; the `.cu`
//! files include those headers and hold nothing but launchers. That is the
//! whole of what "split, do not copy" means here: there is exactly ONE
//! definition of every `__global__` in the tree, and the ahead-of-time build
//! compiles the same text NVRTC does. `norm/altup_aux` shipped a release with
//! two copies that agreed on the day they were written and disagreed by the
//! time anyone looked; this family will not repeat it.
//!
//! # The symbols were renamed, and that cost the family everything
//!
//! Measured before this commit: `quant jit=28 aot=11 overlap=0`. Twenty-eight
//! rows that compiled, resolved a lowered name on an L40S, and could be fired
//! by nothing — because every one of them had been given a NEW name.
//! `quant::cast_f32_to_bf16` where the table says `quant::cast_fp32_to_bf16`;
//! `quant::dequant_fp8_e4m3_bf16` where it says `..._e4m3_to_bf16`;
//! `..._per_channel_bf16` where it says `..._to_bf16_per_channel`;
//! `quant::scale_bf16` standing in for `quant::scale_rows_bf16`, which is not
//! even the same kernel. Every other migrated family had kept its symbols —
//! `layout` 8/8, `mlp` 15/16, `norm` 8/9, `moe` 8/9 — and
//! `examples/migration_status`, which joins the two tables ON the symbol,
//! reported `quant 0%`.
//!
//! A [`KernelSig::symbol`] is not a label. It is the string `model-compiler`
//! writes into a trace, the key `runtime::fire` looks a row up by, and the
//! name `model-loader` calls at load time — `pie_k_quant_cast_fp32_to_bf16`,
//! `scale_rows_bf16`, `quantize_bf16_to_mxfp4_e2m1_per_block`,
//! `quantize_bf16_to_fp8_e4m3_per_channel`. A renamed row is therefore a
//! kernel no existing model text can state and no existing caller can reach:
//! it is not a migration, it is a second family with no callers, sitting
//! beside the one that has them. `norm_device.rs` states the invariant in one
//! line — a symbol `model-compiler` can state must have exactly one contract
//! — and two spellings of one kernel is that invariant read from the other
//! end. Ten of the eleven ahead-of-time `quant::` symbols are now stated here
//! VERBATIM; the eleventh is blocked in C++ and named below.
//!
//! # The operand diff, which is the same defect one level down
//!
//! A row that takes the right symbol and changes the operand order or a kind
//! is worse than a renamed row, because it compiles, resolves, and fires. So
//! every row below was diffed against `kernels-cuda-new/src/table/quant.rs`
//! one row at
//! a time — arity, order, and kind — and a shorter list was accepted for
//! exactly two reasons, both of which are structural and neither of which is
//! a judgement call:
//!
//! * **A stream is not an operand.** It is `cuLaunchKernel`'s sixth
//!   parameter, outside the `void**`; `abi::device_cpp_ty` refuses a row that
//!   names one.
//! * **An extent the rule recovers is not an operand.** `rows` drops from the
//!   five `RouteRows`/`Rms` rows because the rule IS `grid.x = rows`, and
//!   `out_dim` drops from `dequant_mxfp4_to_bf16` and
//!   `dequant_wna16_int4b8_to_bf16` for the same reason. In every case the
//!   `__global__` itself never had the parameter — the launcher spent it on
//!   the grid — so the row mirrors the KERNEL exactly and the table minus one
//!   number.
//!
//! Three differences in KIND survive that diff, and all three are the row
//! being sharper than the launcher rather than different from it:
//! `cast_fp32_to_bf16` takes `F32s` where the table takes `Buf`,
//! `mxfp4_scales_to_marlin_e8m0` takes `U8s`/`U8sMut` where it takes
//! `Buf`/`BufMut`, and `row_select` is an `I32` where it is an
//! `Mxfp4RowSelect`. The launcher `static_cast`s each of them on the first
//! line of its body; the row states the same type in the type system instead,
//! and `emit_device_typecheck` then makes a mismatch a compile error rather
//! than a stride bug. The C++ the two spell is identical — `device::u8` IS
//! `std::uint8_t`, and `enum class Mxfp4RowSelect : int` makes the narrowing
//! the identity — which is the test the difference had to pass.
//!
//! # Seven rows the offline typecheck could not cover, and the three variants that closed them
//!
//! `emit_device_typecheck` was run over all of `unit::UNITS` — one translation
//! unit per unit, `void (*const chk)(<operand C++ types>) = &<instantiation>;`
//! compiled with `nvcc -fatbin`, which admits no parameter conversions and so
//! makes arity, order, constness and width drift a hard compile error. Two of
//! this family's seven units failed it, on ONE operand each, and the refusal
//! stood as long as `Ty` had no word for what the operand was. It has three
//! now — `Ty::Bf16s`, `Ty::F16s`, `Ty::I8sMut` — and all seven rows are
//! inside the net.
//!
//! The root cause was that `Ty::Buf` is read two ways. To the ahead-of-time
//! ABI it means `void*` — what a launcher takes, and what the table's
//! `quant::quantize_bf16_to_fp8_e4m3_per_channel` states. To
//! `abi::device_cpp_ty` it means `const {elem}*`, resolved against the ROW's
//! element. Those readings agree for every kernel whose buffers are all the
//! templated type and part company for a kernel with a FIXED-element buffer
//! beside a templated one:
//!
//! * `quant_flat<Fmt>`, `quant_per_channel<Fmt>` and `cast_per_channel<Fmt>`
//!   take `const bf16* W` whatever `Fmt` is. Five rows stated `w: Buf`, which
//!   the typecheck read as `const fp8_e4m3*` and `const int8_sym*`. They say
//!   `w: Bf16s` now. The fp8 OUTPUT was already spelled correctly and
//!   deliberately — `out: U8sMut`, not `BufMut`, which emits
//!   `::std::uint8_t*` and matches `fp8_e4m3::store = u8` exactly. Measured:
//!   the checker for `quant_flat<fp8_e4m3>` as the row USED to state it is
//!   rejected, and the SAME checker with `w` spelled `const bf16*` and `out`
//!   left alone compiles. One defective operand per row, not two.
//! * `cast_f16_to<T>` takes `const f16* src`, fixed, while `T` is the
//!   DESTINATION. `quant::cast_f16_to_bf16` stated `src: Buf`, read as
//!   `const bf16*` for an operand the kernel reads as `const f16*`; it says
//!   `src: F16s`.
//! * `bf16_to_narrow<T>` is the same shape read the other way and is the
//!   SEVENTH, added with `quant::bf16_to_fp16`: `const bf16* in` is fixed
//!   while `T` is the destination, the row's `elem` is `device::f16`, so
//!   `in_bf16: Buf` emitted `const f16*` for a `const bf16*`. Its OTHER two
//!   operands were exact — `out_fp16: BufMut` is `f16*` against `T* out`, and
//!   `n: I64` is `long long` against `long long n` — so it was one defective
//!   operand and it was the same `Bf16s` the five tag rows wanted.
//!
//! **The family already showed what the fix had to look like.**
//! `cast_e8m0_to<T>` has exactly this shape — a fixed-element input, a
//! templated output — and its row spells the input `U8s`, which
//! `device_cpp_ty` emits as `const ::std::uint8_t*` and `nvcc` accepts
//! against `const u8*`. That row always passed. The pattern was right and the
//! vocabulary ran out two variants short: there was no `Bf16s` and no `F16s`.
//! `U16s` could not stand in, because `pie_device.cuh` made `bf16` and `f16`
//! STRUCTS precisely so they would not collapse — *"as typedefs they would be
//! ONE type: `tanh_inplace<bf16>` and `tanh_inplace<f16>` would be one
//! instantiation"* — so a row saying `U16s` would have traded one wrong
//! spelling for another and given up the element identity as well. `Bf16s`
//! and `F16s` spell the prelude's structs by name, which is why the swap is
//! still a pointer conversion C++ refuses. That property has a negative
//! control: `tests/device_typecheck_types.rs` compiles `bf16_to_fp16`'s
//! checker as stated and again with `in_bf16` mis-typed as `F16s`, and nvcc
//! accepts the first and rejects the second.
//!
//! The int8 rows carried a second defective operand of the same kind.
//! `int8_sym::store` is `i8`, `Ty` had `I8s` but no `I8sMut`, and the two
//! int8 rows therefore stated `out: U8sMut` — unsigned bytes for a signed
//! store, and measured as rejected: with `w` corrected, the fp8 arm's checker
//! compiled and the int8 arm's did not. The alternative considered at the
//! time was to spell the fp8 and int8 arms of ONE template with different
//! operand kinds, which is exactly the divergence the tag-struct collapse
//! exists to prevent — and it is not what happened: `I8sMut` means both arms
//! still state the same KIND of thing, a destination whose element the format
//! tag fixes, and each states its own correctly.
//!
//! What was at stake was the CHECK, not the fire: every buffer marshals as a
//! pointer whatever its `Ty`, so these rows always launched correctly and the
//! measured cost was that seven of this family's thirty-two rows sat outside
//! the net that would catch the next one. `norm` and `mlp` each have a row in
//! the same position — a `bf16* y` beside an `f16* y_fp16` — and `mlp`'s is
//! the live warning: `gpt_oss_glu`'s fp16 output is `Source::Lit(Lit::Null)`
//! today, so nothing writes through the mis-stated pointer, and the day
//! someone sources it the kernel writes half-width data to legal addresses
//! and raises no fault. Those two are the same shape as these seven and the
//! vocabulary now has the word for them; they are in `norm` and `mlp`, which
//! this file does not own.
//!
//! A `Bf16sMut`/`F16sMut` pair is deliberately NOT here. No row in this
//! family writes through a FIXED sixteen-bit destination — every writable
//! half-width buffer in `quant` is the templated end, which `BufMut` already
//! spells from `elem` — and a variant added for symmetry rather than for a
//! parameter is a variant with no kernel to be checked against.
//!
//! # Which rows have an ahead-of-time twin
//!
//! Eleven do, and each one carries the table's symbol character for
//! character: `bf16_to_fp16`, `cast_fp32_to_bf16`, `scale_rows_bf16`,
//! `dequant_fp8_e4m3_to_bf16`,
//! `dequant_fp8_e4m3_to_bf16_per_channel`,
//! `dequant_fp8_e4m3_to_bf16_per_group`, `dequant_mxfp4_to_bf16`,
//! `dequant_wna16_int4b8_to_bf16`, `mxfp4_scales_to_marlin_e8m0`,
//! `quantize_bf16_to_mxfp4_e2m1_per_block` and
//! `quantize_bf16_to_fp8_e4m3_per_channel`. Four of them are rows this
//! commit ADDS — `scale_rows_bf16`, `dequant_fp8_e4m3_to_bf16_per_group`,
//! `dequant_wna16_int4b8_to_bf16` and `bf16_to_fp16` — and the other seven
//! are renames of rows that were already here and already right about
//! everything except the one thing a caller uses.
//!
//! The remaining twenty-one rows have no twin. They are the fp16 siblings of
//! bf16 kernels, the INT8 siblings of the FP8 quantisers, and the flat
//! `scale` the loader drives from C++ — every one of them a template that was
//! already in the header, instantiated at a type the ahead-of-time build
//! never asked for. Their names are derived from the ahead-of-time
//! convention, not invented: `quant::cast_fp32_to_bf16` exists, so its fp16
//! sibling is `quant::cast_fp32_to_f16`; `dequant_fp8_e4m3_to_bf16_per_channel`
//! exists, so the INT8 form is `dequant_int8_to_bf16_per_channel` — which is
//! also, exactly, what its launcher is called. Where a launcher already names
//! the kernel the row is its symbol minus `launch_`. Half is spelled `f16`
//! and 32-bit float `fp32`, matching the table's own `cast_fp32_to_bf16` and
//! the JIT tree's `ssm::fp32_to_f16`; the `.cu` entry points spell half
//! `fp16` in four launcher names, and that is the one place the two
//! conventions disagree.
//!
//! # The twin that could not be rowed, and is one now
//!
//! `quant::bf16_to_fp16` is the eleventh ahead-of-time symbol. It was
//! believed to have two independent blockers; it had one, finding that out
//! cost a compile, and [`kernels::LaunchRule::Slab`] has since closed the
//! one that was real. See [`DEQUANT_WNA16_SIGS`]`[1]`. What follows is the
//! audit that decided it, kept because a blocker retracted without its
//! reasoning is a blocker that comes back.
//!
//! **The naming blocker does not exist.** The reading was that
//! `dequant_wna16.cuh`'s `bf16_to_narrow<T>` is templated but `Narrow2<T>` is
//! specialised for `__half` and `bf16` only, and `__half` has no name under
//! `::pie_cuda_driver::kernels::` — the namespace
//! `DeviceKernel::instantiation` prefixes BOTH halves of an instantiation
//! with. That is true of nvcc and false of the compiler this crate actually
//! uses: `csrc/shim/cuda_fp16.h` opens the shim with `using __half =
//! ::pie_cuda_driver::kernels::device::f16;`, so under NVRTC the two are ONE
//! TYPE and `bf16_to_narrow<__half>` already is `bf16_to_narrow<device::f16>`.
//! Adding a `Narrow2<device::f16>` specialisation to make it nameable made it
//! a redefinition instead, and `units` said so in one line. The specialisation
//! survives behind `#ifndef __CUDACC_RTC__`, where it keeps the same text
//! instantiable at the same names under nvcc; that is a parity guard, not a
//! row enabler.
//!
//! **The geometry blocker did exist, and `Slab` is it.** The launcher opens
//! `min(ceil((n / 8) / 256), 1024)` blocks of 256 — a capped grid-stride over
//! `float4` units — and `Elementwise` states `ceil(n / 256)` over elements.
//! That is eight times the blocks and no cap. Both of the kernel's loops are
//! guarded, so the rule's grid would have computed the right answer, which is
//! precisely why the row was not written on it: it would have been a launch
//! nothing could falsify, differing from the launcher it claims to mirror by
//! an order of magnitude in occupancy. *"Until a rule states a cap and a
//! vector width"* was the condition the old refusal set, and
//! `runtime::launch::slab` states both — `min(ceil(units / 256), 1024)` with
//! `units = n >= 8 ? n / 8 : n` — against this launcher by name, at
//! `quant/dequant_wna16.cu:63-75`. The row is [`DEQUANT_WNA16_SIGS`]`[1]`.
//!
//! That audit also repaired the ahead-of-time build, which was broken at
//! `HEAD` before this commit. The split had rewritten `bf16_to_narrow`'s
//! scalar tail onto `Elem<T>::from_f32`, and `Elem` is specialised on the
//! prelude's `bf16` and `f16` structs only — `Elem<__half>` exists nowhere in
//! the tree — so `quant/dequant_wna16.cu`'s single instantiation, `T =
//! __half`, failed with *"incomplete type `Elem<__half>` is not allowed"* and
//! this was the one translation unit in `csrc` that no longer compiled.
//! `Narrow2<T>` now carries a `narrow(float)` hook beside its
//! `pack(float, float)` one, restoring the original `__float2half` for the
//! tail and keeping a destination format described in a single place.
//!
//! # The multi-argument finding, checked against every kernel here
//!
//! `DeviceKernel::elem` carries a template argument LIST and not only a type,
//! which took thirty-seven kernels off the tree's blocked list. **It moves
//! nothing in `quant`, and the four candidates fail for three different
//! reasons — worth recording, because each is a different lesson.**
//!
//! `transcode_rowmajor_kernel<int GROUP, typename Decode, typename Encode>`
//! is the family's only multi-argument template, and it was already refused
//! on operands: `Decode` and `Encode` arrive BY VALUE as aggregates of
//! pointers and extents, and `runtime::args` marshals pointers, `I32`, `U32`,
//! `F32` and `Usize` and nothing else. The argument list makes its NAME
//! spellable and leaves it unbindable, which is the useful shape of the
//! result — arity was never this one's blocker, only its most visible.
//!
//! `dequant_fp4.cuh`'s `mxfp4_moe_gate_up_decode<int kPairsT>`,
//! `mxfp4_moe_down_decode<int kRowsT>` and
//! `mxfp4_moe_gate_up_decode_grouped<int kTok>` are single non-type
//! templates, spellable as `device::i32(...)` and refused below on geometry:
//! all three build a warp-slab second grid axis sized on the template
//! argument, which is not any of the twelve ported rules. `_grouped` is
//! refused a second time and for the reason the finding itself flags — its
//! `kTok` is read from `PIE_MXFP4_MOE_KTOK` and dispatched through a
//! `switch` over `std::integral_constant`, `case 1 / 2 / 8 / 16`, so a row
//! would freeze one arm of a five-way choice the host makes per process.
//! The other two take their argument from a `constexpr` and have no such
//! arm; their grid is the whole of what stops them.
//!
//! They also expose a limit in the mechanism — a narrower one than this file
//! first recorded, and the difference is the whole of what the corrected
//! finding buys. `instantiation()` pastes `::pie_cuda_driver::kernels::`
//! before `elem` and leaves the rest of the list alone, so the FIRST argument
//! must RESOLVE under that root; it does not have to be a TYPE.
//! `<device::bf16, 256>` is fine and `elem: "8"` spells
//! `<::pie_cuda_driver::kernels::8>`, which is not C++ — a fact about BARE
//! tokens, not about non-type parameters. A qualified constant expression
//! survives the prefix, and the refusal list below carries the measurement:
//! `"device::i32(128)"` came back as `...ILi128EEE...` under NVRTC 13.0 on
//! this L40S. All four of this family's leading-non-type templates are
//! therefore nameable, and all four are refused by something else — a
//! run-time `switch`, a warp-slab grid, a pointer-to-pointer operand, a
//! struct passed by value. **Nothing in `quant` was ever blocked by the
//! spelling alone**, which is why the count moved and the rows did not.
//!
//! **What the finding does un-park is narrower and worth being exact about.**
//! Three kernels here were refused partly because no single element type was
//! honest — `wna16_gate_up_decode` and `wna16_down_decode` take an fp16
//! activation and write bf16, and `mxfp4_weight_to_gptq_w4` reads `u8` and
//! writes `u32`. A two-TYPE template states that honestly now, and the
//! objection is withdrawn. It changes no row, because each has a second
//! blocker that a type list does not touch: the two `wna16` decodes launch
//! 2-D grids over (route, output-row slab) with a warp per row, which no
//! ported rule states, and templating them would mean rewriting inner loops
//! built on `__hfma2` — a body change, which needs its own parity evidence
//! and is not a retype. `mxfp4_weight_to_gptq_w4` is `Elementwise`-shaped
//! (`ceil(total / 256)` blocks of 256) but hard-codes eight nibbles per
//! 32-bit word throughout, so `<u8, u32>` would be its only instantiation
//! ever — a parameter added to satisfy the table's grammar, which is what
//! the table exists to avoid.
//!
//! # Why no launcher was deleted
//!
//! `quant` is the one family `model-loader` calls DIRECTLY, by name, from
//! Rust — `pie_k_quant_cast_fp32_to_bf16`, `scale_rows_bf16`,
//! `quantize_bf16_to_mxfp4_e2m1_per_block`, `quantize_bf16_to_fp8_*`. Those
//! callers are nowhere near the JIT and run before a plan exists. Every
//! launcher therefore stays exactly where it was; what changed is that it now
//! calls a kernel it does not also define. Rows are ADDED here, not moved out
//! of there — and now that the symbols agree, a caller that moves from the
//! entry point to a fire is changing WHERE the launch is decided and nothing
//! about what it is called or what it takes.
//!
//! # The fourteen with no row, and why each one
//!
//! **Re-audited at `LaunchRule` 21 → 28.** One entry moved — `bf16_to_narrow`,
//! which `Slab` was ported from and which is [`DEQUANT_WNA16_SIGS`]`[1]` now.
//! One rule landed against this family's text and still moves nothing:
//! `RoutedQmv`, below, for three reasons that are `device.rs`' and
//! `driver-cuda`'s rather than the rule's. The remaining six new rules were
//! checked launcher by launcher: `RowsFlat` is `quant_bf16_to_fp8.cu:94` and
//! `:149`, which are ALREADY rows under `Elementwise` and stay there — see
//! the note at the end of this list — and `PerRowNarrow`, `RowsPerHead`,
//! `Tile16`, `AxialRope` and `WarpTiledScan` have no launcher of their shape
//! in this family at all.
//!
//! Every omission below is a missing RULE or a missing row GRAMMAR, never a
//! missing kernel. `runtime::launch` evaluates twelve rules now; the four
//! this family uses are `Rms`, `Elementwise`, `ElementwiseRows` and
//! `RouteRows`, and a row that computed the wrong extent would be worse than
//! the launcher it replaced — a real fire, with a real result, and every row
//! past the first untouched.
//!
//! * **Two-dimensional grids over something that is not a rectangle.**
//!   `dequant_fp4`'s three MoE decode GEMVs and `dequant_wna16`'s
//!   `wna16_gate_up_decode` and `wna16_down_decode` build a
//!   `dim3(routes, row_slabs)` grid with one WARP per output row;
//!   `quant_bf16_to_fp8`'s `quant_act_fp8_per_group` builds
//!   `dim3(n_groups, m)` at 128 threads. `ElementwiseRows` is the only
//!   PORTED rule with a second grid axis and it puts `ceil(width / 256)`
//!   there, which is neither an expert count nor a warp slab.
//!
//!   [`kernels::LaunchRule::RoutedQmv`] landed against the then-live
//!   `quant/dequant_wna16.cu:70-75` and states that first grid exactly:
//!   `grid [rows * experts_per_token, ceil(intermediate / 8), 1]` at 256.
//!   That launcher has since been deleted as unreached (§43.9) — a routed row
//!   gets no shim entry — and the grid is now witnessed by the kernel itself
//!   at `quant/dequant_wna16.cuh:295` and `:298`.
//!   It unblocked a geometry and no row, for three stated reasons. **Two of
//!   the three are now void and the third stands**, so the two
//!   `wna16_*_decode` rows are written and the three `dequant_fp4` siblings
//!   are not:
//!
//!   1. **VOID.** The claim was that a plain `__global__` cannot be named,
//!      because `instantiation()` always emits `template_path<elem>`. Both
//!      kernels are in fact `template <int Tu = 0>`
//!      (`dequant_wna16.cuh:265` and `:279`) — a LINKAGE parameter and not
//!      an element type, added so the definitions have no external linkage
//!      until something instantiates one. So the rows spell
//!      `elem: "device::i32(0)"` and are the ORDINARY shape, not
//!      [`crate::device::DeviceKernel::PLAIN`]'s.
//!   2. **STANDS, and it is why neither row is reachable in production.**
//!      `driver-cuda/src/bind/mod.rs`' `jit_dims` hard-codes
//!      `experts_per_token: 0`, and `DispatchCtx` carries no fire-wide
//!      expert count to fill it from — `bind/mod.rs:1277-1290` argues that
//!      at length. At every generated call site `eval` therefore answers
//!      `Ungeometric::Empty`, which is the honest answer and not a fire.
//!      The rows exist, compile, and are pinned against their launchers in
//!      `tests/launch_rules.rs::transcribed`; they are not dispatched.
//!   3. **VOID.** The claim was that `wna16_down_decode` at `:101-104` is
//!      `dim3(ceil(hidden / WARPS), routes)` — the same two axes
//!      TRANSPOSED — so one rule could not state both.
//!      [`kernels::LaunchRule::RoutedQmvTransposed`] states the second, and
//!      the two are separate variants rather than one variant with a flag
//!      because a rule is a launcher's arithmetic and these are two
//!      launchers.
//!
//!   **The three `mxfp4_moe_*` siblings stay refused, and for a fourth
//!   reason neither rule fixes.** They tile at a different warp-tile
//!   divisor: `kMxfp4DecodeBlock = 128` and `kMxfp4GateUpPairs = 4` give 16
//!   output rows a block against these two's 8, so `RoutedQmv`'s
//!   `ceil(intermediate / 8)` is twice their grid. And one of the three
//!   selects between two kernels on `getenv` — **an environment variable is
//!   not geometry**, and a rule that read one would be a `LaunchRule` whose
//!   value depends on the process environment, which nothing in
//!   `runtime::launch` can express and nothing should.
//! * **Two-dimensional blocks.** `dtype_cast`'s `awq_dequant_to_bf16` and
//!   `gptq_dequant_to_bf16` and `quant_bf16_to_fp8`'s `w8a8_dequant` all
//!   launch `dim3(32, 8)` over a `dim3(ceil(N/32), ceil(M/8))` grid. Every
//!   ported rule fixes `blockDim.y == 1`.
//! * **A 64-wide block.** `marlin_permute_scales_per_group` is one block per
//!   64-scale group with a `__shared__ bf16[64]` staging buffer sized by the
//!   block. `Elementwise` fixes 256 threads and `RouteRows` rounds to a warp
//!   multiple of the row width; either would run threads the buffer has no
//!   slot for.
//! * **A capped grid-stride the divisor decides.** `absmax_bf16` caps its
//!   grid at 1024 blocks deliberately — the `atomicMax` traffic is then
//!   bounded by the cap and not by `n` — and it stays refused.
//!   [`kernels::LaunchRule::Slab`] is the cap and it is NOT this launcher's:
//!   `slab` divides by the eight-wide vector FIRST, because the kernel it was
//!   ported from loads `float4`s, and
//!   `absmax_bf16` strides UNVECTORISED elements at
//!   `<<<min((n + 255) / 256, 1024), 256>>>`. Same shape, different divisor,
//!   and stating `Slab` here would launch an eighth of this kernel's grid —
//!   correct only while its own stride loop is, which is a property of a
//!   kernel and not of the rule that launched it. `runtime::launch::slab`'s
//!   own doc names it as the launcher it does not serve.
//!   `bf16_to_narrow` WAS the other half of this bullet and is a row now:
//!   `Slab` was ported from its launcher, digit for digit.
//! * **A rectangle no rule produces, and one arm of a run-time choice.**
//!   `mxfp4_moe_gate_up_decode<kPairs>`, `mxfp4_moe_down_decode<kRows>` and
//!   `mxfp4_moe_gate_up_decode_grouped<kTok>` were listed here as unnameable
//!   because they are templated on an `int`. That reason is void — `elem` is
//!   pasted whole between the angle brackets and carries an argument LIST,
//!   and a non-type argument spelled as a qualified constant expression
//!   resolves: `"device::i32(128)"` came back as `...ILi128EEE...` under
//!   NVRTC 13.0 on this L40S, while the bare `"128"` fails the name-map
//!   pragma with `expected an identifier`, because the
//!   `::pie_cuda_driver::kernels::` prefix lands on the string's first token.
//!   What actually refuses all three is the launch. `dequant_fp4.cu` builds
//!   `dim3 grid(num_tokens * top_k, ceil(intermediate / (warps * kPairs)))`
//!   and `dim3 grid(num_experts, ceil(intermediate / (warps * kPairs)))` —
//!   a WARP-SLAB second axis sized on the template argument, which is not any
//!   of the twelve ported rules and is not `ElementwiseRows` at any width.
//!   `kTok` refuses a row a second time: it is read from
//!   `PIE_MXFP4_MOE_KTOK` at run time and dispatched through a switch over
//!   five instantiations, so a row would freeze one arm of a decision the
//!   host makes per process — the case the arity note itself leaves blocked.
//!   Their activations are `const __half*` and their weights
//!   `const u8* const*`, a pointer-to-pointer for which no `Ty` exists,
//!   which would refuse them again.
//! * **An operand kind that does not exist.**
//!   `transcode_rowmajor_kernel<GROUP, Decode, Encode>` is likewise
//!   nameable now: `elem` may hold three arguments, the second and third
//!   spelled in full — `quant::transcode::DecodeBf16` and
//!   `quant::transcode::EncodeMxfp4` qualified from `::pie_cuda_driver::\
//!   kernels` by hand — because the prefix reaches the first token only. And
//!   `transcode.cu` launches it `<<<rows, 256>>>`, which is
//!   `Rule::RouteRows` to the digit. Neither arity nor geometry refuses this
//!   kernel. Its OPERANDS do: `Decode` and `Encode` arrive BY VALUE as
//!   functor aggregates, `kernels::Ty` has no kind for a struct passed by
//!   value, and `runtime::args` marshals pointers, `I32`, `U32`, `F32` and
//!   `Usize` and nothing else. Passing the struct's FIELDS instead is
//!   `new-horizon.md` §4.3's struct-layout inversion, which is not
//!   implemented. That blocker is structural and stands; the two this file
//!   used to give beside it did not survive the measurement.
//! * **A kernel with no honest type parameter.** `mxfp4_weight_to_gptq_w4` is
//!   `Elementwise`-shaped, but its `k_pack` arithmetic hard-codes eight
//!   nibbles per 32-bit word. Templating it would be a lie about what varies.
//!
//! And one entry that is not a refusal but a rule that was checked and NOT
//! restated. [`kernels::LaunchRule::RowsFlat`] cites
//! `quant/quant_bf16_to_fp8.cu:94` and `:149` — the two
//! `absmax_to_scale_inv` launchers — among the `<<<>>>`s it reproduces, and
//! both are already rows, under `Elementwise`, at
//! [`QUANT_BF16_TO_FP8_SIGS`]`[1]` and `[2]`. They stay there. `Elementwise`
//! is `ceil(rows * width / 256)`, and each row's guard operand is
//! `n: I32 <- Source::OutElements(0)` — the SAME number the rule divides —
//! so grid and guard agree by construction whichever way the absmax vector's
//! rectangle is oriented. `rows_flat` reads `Dims::rows` alone, and would
//! agree only if that rectangle is `[R, 1]`; on `[1, R]` it launches one
//! block for R rows and the tail of the scale vector keeps the absmax values
//! it was supposed to reciprocate. Both rows are JIT-only with no external
//! contract fixing the orientation, so restating them would trade a
//! guarantee for an assumption and gain no row. A rule citing a launcher is
//! the rule's evidence that its arithmetic is real; it is not an instruction
//! to move a row that is already correct.
//!
//! That leaves `quant/transcode.cuh` alone with zero rows and therefore no
//! `Unit` — a unit is a compile, and a compile with nothing to look up proves
//! nothing. It was still split and still converted off `<cstdint>`, because
//! `build.rs` carries every `.cuh` under `csrc/src` into the header set and a
//! standard-library include in a carried header is a compile error waiting
//! for whichever unit first includes it. `quant/dequant_wna16.cuh` was in
//! that position until this commit and is now a `Unit` with one row; its own
//! file header still opens by saying it carries none, which was true when the
//! grid axes were `(word_cols, rows)` and stopped being true when they were
//! swapped — the paragraph two hundred lines further down, which works out
//! that the swap is what makes `ElementwiseRows` "the rule this geometry
//! already is", is the current one.
//!
//! # What the rows are allowed to assume
//!
//! `runtime::args` marshals pointers and `I32`/`U32`/`F32`/`Usize`, and
//! nothing else. That is why no row here takes a stream — a stream is
//! `cuLaunchKernel`'s sixth parameter, outside the `void**` — and why
//! `Mxfp4RowSelect` crosses into device code as a plain `int`: the enum is
//! declared in a host-only `.hpp` NVRTC never sees, and `enum class ... : int`
//! makes the narrowing the identity.

use kernels::KernelSig;
use kernels::Lit;
use kernels::LaunchRule;
use kernels::Source;
use kernels::kernel;
use kernels::operands;

use crate::device::DeviceKernel;
use crate::unit::Unit;

/// The six type-cast templates `dtype_cast.cu` used to hold inline.
///
/// Ten rows over six templates. `cast_f32_to`, `cast_to_f32` and `scale`
/// were `_bf16`, `_f16` and `_f32` families of near-identical kernels in the
/// ahead-of-time build because a `.cu` has to pick its instantiations; here
/// the template is written once and the element type is a row. Two of the ten
/// carry ahead-of-time symbols — `quant::cast_fp32_to_bf16` and
/// `quant::scale_rows_bf16`, both of which `model-loader` calls by name from
/// Rust — and the rest are instantiations that build never asked for.
pub const DTYPE_CAST: Unit = Unit {
    name: "quant/dtype_cast",
    root: include_str!("../../csrc/src/quant/dtype_cast.cuh"),
    rows: DTYPE_CAST_ROWS,
    options: &[],
};

/// The four FP8 E4M3 dequantisers: flat, per-channel, per-tile, per-group.
///
/// Three of the four scale shapes are ahead-of-time symbols, because which
/// one a checkpoint ships is a fact the declaration reads and not a fact a
/// driver may guess — a guess dequantizes correctly on one checkpoint and
/// silently wrongly on the next.
pub const DEQUANT_FP8: Unit = Unit {
    name: "quant/dequant_fp8",
    root: include_str!("../../csrc/src/quant/dequant_fp8.cuh"),
    rows: DEQUANT_FP8_ROWS,
    options: &[],
};

/// The MXFP4 encoder — one block per row, 32 values per E8M0 scale.
pub const QUANT_BF16_TO_MXFP4: Unit = Unit {
    name: "quant/quant_bf16_to_mxfp4",
    root: include_str!("../../csrc/src/quant/quant_bf16_to_mxfp4.cuh"),
    rows: QUANT_BF16_TO_MXFP4_ROWS,
    options: &[],
};

/// The narrow-format quantisers, unified over a format tag.
///
/// The ahead-of-time file held FP8 and INT8 twins of four kernels — same
/// body, different `max_abs` and different store type. They are one template
/// per shape here, parameterised by a tag struct, and the row picks the
/// format. That is where twelve ahead-of-time `__global__`s became nine.
pub const QUANT_BF16_TO_FP8: Unit = Unit {
    name: "quant/quant_bf16_to_fp8",
    root: include_str!("../../csrc/src/quant/quant_bf16_to_fp8.cuh"),
    rows: QUANT_BF16_TO_FP8_ROWS,
    options: &[],
};

/// The two Marlin repackers a row selector can drive.
pub const MXFP4_MARLIN: Unit = Unit {
    name: "quant/mxfp4_marlin",
    root: include_str!("../../csrc/src/quant/mxfp4_marlin.cuh"),
    rows: MXFP4_MARLIN_ROWS,
    options: &[],
};

/// The MXFP4 decoder, and the three MoE GEMVs beside it that no rule fits.
pub const DEQUANT_FP4: Unit = Unit {
    name: "quant/dequant_fp4",
    root: include_str!("../../csrc/src/quant/dequant_fp4.cuh"),
    rows: DEQUANT_FP4_ROWS,
    options: &[],
};

/// The W4A16 decoder, and the three kernels beside it that no rule fits.
///
/// This header carried no unit until the axis swap in
/// `dequant_wna16_int4b8` landed: it launched `dim3(ceil(words / 128),
/// min(out_dim, 65535))` and rode a `gridDim.y` stride loop, and no rule
/// states a hardware clamp. With rows on x and word-columns on y the launcher
/// is `ElementwiseRows` outright, which is what the `.cu` beside it now says
/// in as many words — so the header gets a unit, one row, and the compile
/// that `examples/unit_probe_quant` had been giving it as a rootless text.
///
/// # THE LAUNCHER'S TWO GUARDS, WHICH THIS ROW DOES NOT MAKE
///
/// §54 deleted `dequant_wna16_int4b8_to_bf16` from
/// `kernels-cuda/csrc/src/quant/dequant_wna16.cu` — routed, no C++ caller,
/// no hand arm. It returned WITHOUT LAUNCHING on two conditions, and a
/// rule-driven fire launches on both:
///
///   * `out_dim <= 0 || in_dim <= 0 || group_size <= 0`. The first two are
///     the fire's empty rectangle and are harmless; `group_size <= 0` is
///     not, because the kernel divides by it.
///   * `in_dim % 8 != 0 || in_dim % group_size != 0`. The packing is eight
///     4-bit weights per `int32`, so a row whose width is not a multiple of
///     8 has a partial final word the kernel reads WHOLE — it dequantizes
///     the padding lanes into real output columns. A `group_size` that does
///     not divide `in_dim` puts a scale boundary inside a word, so the last
///     group of each row is scaled by its neighbour's exponent.
///
/// Neither is a rectangle, so neither is statable as a `LaunchRule`. Every
/// weight this driver has loaded satisfies both — compressed-tensors emits
/// `group_size = 32` over `in_dim` that is always a multiple of 128 — which
/// is why nothing has caught it, and is the same sentence §43.9 wrote when
/// it took the two decode GEMVs and their identical guards out of that file.
pub const DEQUANT_WNA16: Unit = Unit {
    name: "quant/dequant_wna16",
    root: include_str!("../../csrc/src/quant/dequant_wna16.cuh"),
    rows: DEQUANT_WNA16_ROWS,
    options: &[],
};

/// The units `quant` compiles.
pub static UNITS: &[Unit] = &[
    DTYPE_CAST,
    DEQUANT_FP8,
    QUANT_BF16_TO_MXFP4,
    QUANT_BF16_TO_FP8,
    MXFP4_MARLIN,
    DEQUANT_FP4,
    DEQUANT_WNA16,
];

// ---------------------------------------------------------------------------
// quant/dtype_cast
// ---------------------------------------------------------------------------

/// The instantiations `quant/dtype_cast.cuh` is compiled for.
///
/// `cast_f16_to<f16>` is absent on purpose: it would be a memcpy with a
/// launch configuration. `cast_e8m0_to` is rowed at `f32` alone because an
/// E8M0 byte decodes to a power of two that a bf16 destination would round —
/// the kernel exists to hand a scale to arithmetic, not to store one.
///
/// `scale_rows` is last and is the one row here with an ahead-of-time twin
/// besides `cast_fp32_to_bf16`. It was missing entirely while
/// `quant::scale_bf16` — the FLAT scalar multiply three rows above it — sat
/// under a name one character's worth of reading away from
/// `quant::scale_rows_bf16`, which is a per-column vector multiply over a
/// row-major matrix. Two kernels, two contracts, and the near-collision is
/// why both rows say which is which.
pub static DTYPE_CAST_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &DTYPE_CAST_SIGS[0],
        template_path: "quant::device::cast_f32_to",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DTYPE_CAST_SIGS[1],
        template_path: "quant::device::cast_f32_to",
        elem: "device::f16",
    },
    DeviceKernel {
        sig: &DTYPE_CAST_SIGS[2],
        template_path: "quant::device::cast_to_f32",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DTYPE_CAST_SIGS[3],
        template_path: "quant::device::cast_to_f32",
        elem: "device::f16",
    },
    DeviceKernel {
        sig: &DTYPE_CAST_SIGS[4],
        template_path: "quant::device::cast_f16_to",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DTYPE_CAST_SIGS[5],
        template_path: "quant::device::cast_e8m0_to",
        elem: "quant::device::f32",
    },
    DeviceKernel {
        sig: &DTYPE_CAST_SIGS[6],
        template_path: "quant::device::scale",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DTYPE_CAST_SIGS[7],
        template_path: "quant::device::scale",
        elem: "device::f16",
    },
    DeviceKernel {
        sig: &DTYPE_CAST_SIGS[8],
        template_path: "quant::device::scale",
        elem: "quant::device::f32",
    },
    DeviceKernel {
        sig: &DTYPE_CAST_SIGS[9],
        template_path: "quant::device::scale_rows",
        elem: "device::bf16",
    },
];

/// The contracts, in [`DTYPE_CAST_ROWS`]' order.
///
/// The first nine are flat: rows stack, the guard is the kernel's own, and
/// `n` is `rows * width` — the same number `Elementwise` computes to size the
/// grid. It stays an operand because the kernel needs it to test its own
/// index, which is the distinction §10.5 draws between an extent a rule
/// RECOVERS and an extent a kernel READS.
///
/// One of the nine, `quant::cast_fp32_to_bf16`, is an ahead-of-time symbol
/// and is spelled the way the table spells it. It was `quant::cast_f32_to_bf16`
/// here until this commit — a name `model-loader`'s
/// `pie_k_quant_cast_fp32_to_bf16` call site could not reach and no trace
/// could state.
#[rustfmt::skip]
static DTYPE_CAST_SIGS: [KernelSig; 10] = [
    // The ahead-of-time table's `quant::cast_fp32_to_bf16`, character for
    // character. Operands diffed against it: `src_fp32, dst_bf16, n` in that
    // order, minus the stream. `F32s` is the kind, where the table erases the
    // source to `Buf` and the launcher casts it back on the first line — the
    // `__global__` takes `const float*`, so the sharper kind is the one
    // `emit_device_typecheck` can actually check.
    kernel!(cast_fp32_to_bf16 "quant::cast_fp32_to_bf16",
        file = Some("quant/dtype_cast.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            src_fp32: F32s <- Source::In(0),
            dst_bf16: BufMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
        ]),
    // The fp16 twin the ahead-of-time build never instantiated. No new C++:
    // `Cast<f16>` was already there, waiting for someone to ask. The name is
    // the twin's with the destination changed, because that is the only thing
    // that changed.
    kernel!(cast_fp32_to_f16 "quant::cast_fp32_to_f16",
        file = Some("quant/dtype_cast.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            src: F32s <- Source::In(0),
            dst: BufMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
        ]),
    kernel!(cast_bf16_to_fp32 "quant::cast_bf16_to_fp32",
        file = Some("quant/dtype_cast.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            src: Buf <- Source::In(0),
            dst: F32sMut <- Source::Out(0),
            n: Usize <- Source::InElements(0),
        ]),
    kernel!(cast_f16_to_fp32 "quant::cast_f16_to_fp32",
        file = Some("quant/dtype_cast.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            src: Buf <- Source::In(0),
            dst: F32sMut <- Source::Out(0),
            n: Usize <- Source::InElements(0),
        ]),
    // `src` is `Buf` and the parameter is `const f16*`: `cast_f16_to<T>`
    // FIXES the source and templates the destination, so the one element
    // type `elem` carries belongs to the wrong end of this cast. The two
    // neighbours above escape by having `F32s`/`F32sMut` to name their fixed
    // end with; half precision has no such `Ty`. Measured under nvcc 13.0
    // `-arch=sm_89` — the kernel is `(const f16*, bf16*, usize)`, and
    // `(const bf16*, bf16*, usize)` is a different type that a
    // function-pointer initialisation refuses.
    //
    // Live, unlike the two `y_fp16` misses elsewhere: `Source::In(0)` binds a
    // real tensor here. It FIRED correctly even while mis-stated — a `Buf`
    // crosses as `*const c_void` — so what was wrong was the row's WORD about
    // the buffer, which is what a reader and any dtype check that ever
    // arrives consult.
    //
    // `Ty::F16s` is that word, and it is the fix the paragraph above asked
    // for rather than a rename that would have hidden the miss: it spells
    // `const ::pie_cuda_driver::kernels::device::f16*`, which is the
    // parameter, and `const bf16*` is a pointer conversion the checker's
    // initialisation refuses.
    kernel!(cast_f16_to_bf16 "quant::cast_f16_to_bf16",
        file = Some("quant/dtype_cast.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            src: F16s <- Source::In(0),
            dst: BufMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
        ]),
    // `src` is `U8s` and not `Buf` because an E8M0 scale IS a byte — the
    // whole format is a biased exponent — and a `void*` row would let a bf16
    // scale tensor through the same hole.
    kernel!(cast_e8m0_to_fp32 "quant::cast_e8m0_to_fp32",
        file = Some("quant/dtype_cast.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            src: U8s <- Source::In(0),
            dst: F32sMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
        ]),
    // `factor` is a HOST float. It reaches the kernel by value in the
    // `void**`, which is the only reason a scalar multiply needs no second
    // buffer and no second launch.
    //
    // NOT `quant::scale_rows_bf16`: this multiplies every element by ONE
    // number and that one takes a vector. The launcher names are
    // `scale_bf16` and `scale_rows_bf16` and the rows are spelled the same
    // way, so the two cannot be confused by anything that reads either table.
    kernel!(scale_bf16 "quant::scale_bf16",
        file = Some("quant/dtype_cast.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            src: Buf <- Source::In(0),
            dst: BufMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
            factor: F32 <- Source::Param(0),
        ]),
    kernel!(scale_f16 "quant::scale_f16",
        file = Some("quant/dtype_cast.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            src: Buf <- Source::In(0),
            dst: BufMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
            factor: F32 <- Source::Param(0),
        ]),
    kernel!(scale_fp32 "quant::scale_fp32",
        file = Some("quant/dtype_cast.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            src: F32s <- Source::In(0),
            dst: F32sMut <- Source::Out(0),
            n: Usize <- Source::OutElements(0),
            factor: F32 <- Source::Param(0),
        ]),
    // The ahead-of-time table's `quant::scale_rows_bf16`. Added, not renamed:
    // nothing here stated it, and `model-loader` calls it by that name to
    // fold a per-column vector into a weight after a merge.
    //
    // `RouteRows` against the launcher it replaces —
    // `device::scale_rows<device::bf16><<<rows, 256>>>` — grid `[rows, 1, 1]`
    // both ways, and the block widths differ on purpose: the rule passes
    // `ceil_warp(width)` capped at 1024 where the launcher passes 256, and
    // the kernel's `for (c = threadIdx.x; c < width; c += blockDim.x)` makes
    // both exact. The `.cu` says so beside the `<<<>>>`.
    //
    // Operands diffed against the table: `buf_bf16, l_bf16, rows, width`
    // becomes `buf, l, width` — `rows` IS the grid the rule computes, and the
    // `__global__` never took it. `buf` is in place and declared so.
    kernel!(scale_rows_bf16 "quant::scale_rows_bf16",
        file = Some("quant/dtype_cast.cuh"),
        launch = LaunchRule::RouteRows,
        in_place = &[(0, 0)],
        operands = operands![
            buf_bf16: BufMut <- Source::Out(0),
            l_bf16: Buf <- Source::In(1),
            width: I32 <- Source::OutWidth(0),
        ]),
];

// ---------------------------------------------------------------------------
// quant/dequant_fp8
// ---------------------------------------------------------------------------

/// The instantiations `quant/dequant_fp8.cuh` is compiled for.
///
/// The three row-shaped dequantisers are rowed at bf16 alone because that is
/// what the checkpoints they read are dequantised INTO; the flat one gets an
/// fp16 row as well, because it costs a row and no C++.
///
/// Four of the five are ahead-of-time symbols and three of those four were
/// spelled differently here until this commit — `dequant_fp8_e4m3_bf16` for
/// `dequant_fp8_e4m3_to_bf16`, `..._per_channel_bf16` for
/// `..._to_bf16_per_channel`. The scale's SHAPE is the only thing that
/// distinguishes these forms, a property of the checkpoint rather than of the
/// fire, so a declaration states which — and it can only state one it can
/// name.
pub static DEQUANT_FP8_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &DEQUANT_FP8_SIGS[0],
        template_path: "quant::device::dequant_fp8_e4m3",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DEQUANT_FP8_SIGS[1],
        template_path: "quant::device::dequant_fp8_e4m3",
        elem: "device::f16",
    },
    DeviceKernel {
        sig: &DEQUANT_FP8_SIGS[2],
        template_path: "quant::device::dequant_fp8_e4m3_per_channel",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DEQUANT_FP8_SIGS[3],
        template_path: "quant::device::dequant_fp8_e4m3_blocked",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DEQUANT_FP8_SIGS[4],
        template_path: "quant::device::dequant_fp8_e4m3_per_group",
        elem: "device::bf16",
    },
];

/// The contracts, in [`DEQUANT_FP8_ROWS`]' order.
///
/// The last three are `RouteRows` — one block per row, as wide as the row —
/// and that is only sound because all three kernels stride their column loop
/// by `blockDim.x` rather than by the 256 the launcher happens to pass. The
/// ahead-of-time launcher still passes 256, so the change is inert there and
/// load-bearing here.
#[rustfmt::skip]
static DEQUANT_FP8_SIGS: [KernelSig; 5] = [
    // The table's `quant::dequant_fp8_e4m3_to_bf16`. Operands diffed against
    // it: `fp8_in, bf16_out, scale, n` in that order and those kinds, minus
    // the stream. Nothing else moved.
    kernel!(dequant_fp8_e4m3_to_bf16 "quant::dequant_fp8_e4m3_to_bf16",
        file = Some("quant/dequant_fp8.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            fp8_in: U8s <- Source::In(0),
            bf16_out: BufMut <- Source::Out(0),
            scale: F32 <- Source::Param(0),
            n: Usize <- Source::OutElements(0),
        ]),
    kernel!(dequant_fp8_e4m3_to_f16 "quant::dequant_fp8_e4m3_to_f16",
        file = Some("quant/dequant_fp8.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            src: U8s <- Source::In(0),
            dst: BufMut <- Source::Out(0),
            scale: F32 <- Source::Param(0),
            n: Usize <- Source::OutElements(0),
        ]),
    // The table's `quant::dequant_fp8_e4m3_to_bf16_per_channel`. Diffed:
    // `fp8_in, bf16_out, scale_inv_dev, rows, cols` becomes the same list
    // minus `rows`, which is `RouteRows`' grid and which the `__global__`
    // never took — the launcher spent it on `<<<rows, 256>>>` and read
    // `blockIdx.x` for it.
    kernel!(dequant_fp8_e4m3_to_bf16_per_channel "quant::dequant_fp8_e4m3_to_bf16_per_channel",
        file = Some("quant/dequant_fp8.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            fp8_in: U8s <- Source::In(0),
            bf16_out: BufMut <- Source::Out(0),
            scale_inv_dev: F32s <- Source::In(1),
            cols: I32 <- Source::OutWidth(0),
        ]),
    // `scale_cols` is the caller's statement about the SCALE tensor's shape,
    // not this tensor's, so no extent source reaches it: it is the second
    // dimension of a different buffer. It stays a parameter, which is what
    // `Source::Param` is for.
    //
    // No ahead-of-time twin — the `.cu` exposes the tile form only as
    // `per_group`, below — so the name is the launcher's shape with the
    // table's `_to_bf16` word order.
    kernel!(dequant_fp8_e4m3_to_bf16_blocked "quant::dequant_fp8_e4m3_to_bf16_blocked",
        file = Some("quant/dequant_fp8.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            src: U8s <- Source::In(0),
            dst: BufMut <- Source::Out(0),
            scales: F32s <- Source::In(1),
            cols: I32 <- Source::OutWidth(0),
            row_block: I32 <- Source::Param(0),
            col_block: I32 <- Source::Param(1),
            scale_cols: I32 <- Source::Param(2),
        ]),
    // The table's `quant::dequant_fp8_e4m3_to_bf16_per_group`, added: the
    // third fp8 scale shape had a launcher, a `__global__` and a row in the
    // ahead-of-time table, and nothing here.
    //
    // `RouteRows` against `device::dequant_fp8_e4m3_per_group<device::bf16>
    // <<<rows, 256>>>` — grid `[rows, 1, 1]` both ways; the rule's
    // `ceil_warp(cols)` block and the launcher's 256 agree because the shared
    // tile body strides `j += blockDim.x`.
    //
    // Diffed against the table: `fp8_in, bf16_out, scale_dev, rows, cols,
    // group_size` minus `rows` (the grid) and minus the stream. `scale_cols`
    // does NOT appear — this form derives it as `ceil(cols / group_size)`,
    // which is the whole reason it is its own `__global__` rather than the
    // blocked one with two arguments repeated.
    kernel!(dequant_fp8_e4m3_to_bf16_per_group "quant::dequant_fp8_e4m3_to_bf16_per_group",
        file = Some("quant/dequant_fp8.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            fp8_in: U8s <- Source::In(0),
            bf16_out: BufMut <- Source::Out(0),
            scale_dev: F32s <- Source::In(1),
            cols: I32 <- Source::OutWidth(0),
            group_size: I32 <- Source::Param(0),
        ]),
];

// ---------------------------------------------------------------------------
// quant/quant_bf16_to_mxfp4
// ---------------------------------------------------------------------------

/// The one instantiation `quant/quant_bf16_to_mxfp4.cuh` is compiled for.
pub static QUANT_BF16_TO_MXFP4_ROWS: &[DeviceKernel] = &[DeviceKernel {
    sig: &QUANT_BF16_TO_MXFP4_SIGS[0],
    template_path: "quant::device::quant_bf16_to_mxfp4_row",
    elem: "device::bf16",
}];

/// The contract.
///
/// The ahead-of-time table's `quant::quantize_bf16_to_mxfp4_e2m1_per_block`,
/// which `model-loader` calls by that name from the Encode path — it was
/// `quant::quantize_bf16_to_mxfp4_row_bf16` here, a spelling taken from the
/// TEMPLATE rather than from the contract, and therefore reachable by nothing
/// that already existed.
///
/// Two outputs — the packed nibbles and the E8M0 scale bytes — and the row
/// says so, because a kernel that writes two tensors and declares one is a
/// kernel whose second tensor nobody knows is live. Operands diffed against
/// the table: `w_bf16, w_packed, w_scale_e8m0, rows, cols` minus `rows`,
/// which is `RouteRows`' grid and which the `__global__` never took.
#[rustfmt::skip]
static QUANT_BF16_TO_MXFP4_SIGS: [KernelSig; 1] = [
    kernel!(quantize_bf16_to_mxfp4_e2m1_per_block "quant::quantize_bf16_to_mxfp4_e2m1_per_block",
        file = Some("quant/quant_bf16_to_mxfp4.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            w_bf16: Buf <- Source::In(0),
            w_packed: U8sMut <- Source::Out(0),
            w_scale_e8m0: U8sMut <- Source::Out(1),
            cols: I32 <- Source::InWidth(0),
        ]),
];

// ---------------------------------------------------------------------------
// quant/quant_bf16_to_fp8
// ---------------------------------------------------------------------------

/// The instantiations `quant/quant_bf16_to_fp8.cuh` is compiled for.
///
/// `fp8_e4m3` and `int8_sym` are TAG structs, not element types: each names a
/// storage type, a `max_abs()` and a `narrow()`. Rowing one shape at both
/// tags is what collapsed the file's twelve `__global__`s into nine
/// templates, and it is the only reason the INT8 and FP8 paths cannot drift
/// — there is one body.
pub static QUANT_BF16_TO_FP8_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &QUANT_BF16_TO_FP8_SIGS[0],
        template_path: "quant::device::quant_flat",
        elem: "quant::device::fp8_e4m3",
    },
    DeviceKernel {
        sig: &QUANT_BF16_TO_FP8_SIGS[1],
        template_path: "quant::device::absmax_to_scale_inv",
        elem: "quant::device::fp8_e4m3",
    },
    DeviceKernel {
        sig: &QUANT_BF16_TO_FP8_SIGS[2],
        template_path: "quant::device::absmax_to_scale_inv",
        elem: "quant::device::int8_sym",
    },
    DeviceKernel {
        sig: &QUANT_BF16_TO_FP8_SIGS[3],
        template_path: "quant::device::dequant_int8_per_channel",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &QUANT_BF16_TO_FP8_SIGS[4],
        template_path: "quant::device::absmax_per_row",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &QUANT_BF16_TO_FP8_SIGS[5],
        template_path: "quant::device::quant_per_channel",
        elem: "quant::device::fp8_e4m3",
    },
    DeviceKernel {
        sig: &QUANT_BF16_TO_FP8_SIGS[6],
        template_path: "quant::device::quant_per_channel",
        elem: "quant::device::int8_sym",
    },
    DeviceKernel {
        sig: &QUANT_BF16_TO_FP8_SIGS[7],
        template_path: "quant::device::cast_per_channel",
        elem: "quant::device::fp8_e4m3",
    },
    DeviceKernel {
        sig: &QUANT_BF16_TO_FP8_SIGS[8],
        template_path: "quant::device::cast_per_channel",
        elem: "quant::device::int8_sym",
    },
    // The two the header called "no rule and therefore no row". They have no
    // rule STILL -- both are `LaunchRule::Unstated` -- and they have rows now
    // because a row is what `runtime::cache` resolves a name expression
    // through, and the geometry a rule cannot state is stated by
    // `driver-cuda/src/fire/quant_int8.rs` instead. See the two `Unstated`
    // sigs below for the argument, which is `fire/attn_score.rs`'s.
    DeviceKernel {
        sig: &QUANT_BF16_TO_FP8_SIGS[9],
        template_path: "quant::device::w8a8_dequant",
        elem: DeviceKernel::PLAIN,
    },
    DeviceKernel {
        sig: &QUANT_BF16_TO_FP8_SIGS[10],
        template_path: "quant::device::quant_act_fp8_per_group",
        elem: DeviceKernel::PLAIN,
    },
];

/// The contracts, in [`QUANT_BF16_TO_FP8_ROWS`]' order.
///
/// `absmax_per_row` and `quant_per_channel` are `Rms`: one block per row, 256
/// threads, `(256 / 32) * 4` bytes of dynamic shared memory. Those two keep a
/// hard-coded `kBlock = 256` where the `RouteRows` kernels below were freed
/// to stride by `blockDim.x`, and the reason is the shared array — it is
/// sized `kBlock / 32` by the LAUNCH and the final fold reads `tid < kBlock /
/// 32`. A block width the kernel did not agree to would read past it.
///
/// One row here is an ahead-of-time symbol —
/// `quant::quantize_bf16_to_fp8_e4m3_per_channel`, which `model-loader` calls
/// by name — and it was `quant::quant_per_channel_fp8` until this commit. The
/// other eight have no twin: they are the INT8 halves of tag-struct templates
/// the ahead-of-time build only ever instantiated at one tag, plus the two
/// staged halves of the per-channel quantiser a tensor-parallel all-reduce
/// has to run between. Each is named after the launcher that fires it, minus
/// `launch_`, which is how a caller moving off the entry point finds it.
///
/// **Eleven now, and the two appended are the file's last two launchers.**
/// `[9]` and `[10]` are `LaunchRule::Unstated` and carry their own argument
/// where they sit; they arrived when `quant/quant_bf16_to_fp8.cu` was deleted
/// and its host program became `driver-cuda/src/fire/quant_int8.rs`.
///
/// # Five rows called a `const bf16*` by their tag, and two called an `i8*`
/// unsigned
///
/// `Buf` takes its element from the row's `elem`, and `elem` on the tag rows
/// is `fp8_e4m3` or `int8_sym` — which are not element types, as the
/// paragraph above says — while the weight parameter is `const bf16*` in
/// every one of them. Measured under nvcc 13.0 `-arch=sm_89` with a
/// function-pointer initialisation, which admits no conversions:
/// `quant_flat<fp8_e4m3>` is `(const bf16*, u8*, float, usize)` and
/// `cast_per_channel<int8_sym>` is `(const bf16*, i8*, const float*, i32)`,
/// and the tag-derived reading of the first parameter matched neither.
///
/// The five weights say `Bf16s` now, which names the format instead of
/// deriving it from an `elem` that describes the other end of the
/// conversion. `Bf16s` is the prelude's `device::bf16` by name, so a row that
/// swapped it for `F16s` is still a pointer conversion nvcc refuses — the
/// property is extended here, not spent.
///
/// The FP8 outputs were always exact, and not by luck: `fp8_e4m3::store` IS
/// `u8`, which `U8sMut` spells character for character. The INT8 outputs were
/// the second miss. `int8_sym::store` is `i8` and both INT8 rows said
/// `U8sMut`, because `Ty` carried `I8s` and no `I8sMut`. Same width, same
/// addresses, so nothing miscomputed — but the row told a reader to allocate
/// unsigned where the kernel narrows to signed. `I8sMut` is the repair and
/// it is `crates/kernels`', which is where the note said it belonged.
#[rustfmt::skip]
static QUANT_BF16_TO_FP8_SIGS: [KernelSig; 11] = [
    // `launch_quant_bf16_to_fp8_e4m3`, the one-shot cast with the scale
    // already known. NOT `quantize_bf16_to_fp8_e4m3_per_tensor`: that entry
    // point is absmax, a device-to-host copy, a stream sync and THEN this
    // kernel, so a row claiming the name would claim a launch it is one of
    // four steps in.
    kernel!(quant_bf16_to_fp8_e4m3 "quant::quant_bf16_to_fp8_e4m3",
        file = Some("quant/quant_bf16_to_fp8.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            w: Bf16s <- Source::In(0),
            out: U8sMut <- Source::Out(0),
            scale_inv: F32 <- Source::Param(0),
            n: Usize <- Source::InElements(0),
        ]),
    // In place, and stated as such: this rewrites the absmax vector into the
    // scale vector rather than allocating a second one. `in_place` is the
    // claim the planner checks, so the aliasing is declared and not assumed.
    kernel!(absmax_to_scale_inv_fp8 "quant::absmax_to_scale_inv_fp8",
        file = Some("quant/quant_bf16_to_fp8.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: F32sMut <- Source::Out(0),
            n: I32 <- Source::OutElements(0),
        ]),
    kernel!(absmax_to_scale_inv_int8 "quant::absmax_to_scale_inv_int8",
        file = Some("quant/quant_bf16_to_fp8.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: F32sMut <- Source::Out(0),
            n: I32 <- Source::OutElements(0),
        ]),
    // `cols` AND `n` both cross, and they are not the same statement: `n`
    // bounds the guard, `cols` is the divisor that recovers which row a
    // linear index fell in. `Elementwise` recovers `n`; nothing recovers a
    // divisor.
    //
    // Named after `launch_dequant_int8_to_bf16_per_channel` — which is also
    // the word order the table uses for the fp8 form, so the INT8 and FP8
    // dequantisers read as the pair they are.
    kernel!(dequant_int8_to_bf16_per_channel "quant::dequant_int8_to_bf16_per_channel",
        file = Some("quant/quant_bf16_to_fp8.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            w: I8s <- Source::In(0),
            out: BufMut <- Source::Out(0),
            scale_inv: F32s <- Source::In(1),
            cols: I32 <- Source::OutWidth(0),
            n: Usize <- Source::OutElements(0),
        ]),
    kernel!(absmax_per_row_bf16 "quant::absmax_per_row_bf16",
        file = Some("quant/quant_bf16_to_fp8.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            w: Buf <- Source::In(0),
            absmax_out: F32sMut <- Source::Out(0),
            cols: I32 <- Source::InWidth(0),
        ]),
    // The table's `quant::quantize_bf16_to_fp8_e4m3_per_channel`. Diffed
    // against it: `w_bf16, w_fp8, scale_inv_dev, rows, cols` minus `rows` —
    // `Rms` is `<<<rows, 256, (256/32)*4>>>`, which is the launcher digit for
    // digit — and minus the stream.
    //
    // Two outputs from one pass: the narrow row and the MULTIPLICATIVE
    // `weight_scale_inv` the GEMM dispatcher hands cuBLASLt. Emitting the
    // multiplicative form here is what stops the dispatcher computing a
    // reciprocal at fire time.
    kernel!(quantize_bf16_to_fp8_e4m3_per_channel "quant::quantize_bf16_to_fp8_e4m3_per_channel",
        file = Some("quant/quant_bf16_to_fp8.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            w_bf16: Bf16s <- Source::In(0),
            w_fp8: U8sMut <- Source::Out(0),
            scale_inv_dev: F32sMut <- Source::Out(1),
            cols: I32 <- Source::InWidth(0),
        ]),
    kernel!(quantize_bf16_to_int8_per_channel "quant::quantize_bf16_to_int8_per_channel",
        file = Some("quant/quant_bf16_to_fp8.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            w: Bf16s <- Source::In(0),
            out: I8sMut <- Source::Out(0),
            scale_inv: F32sMut <- Source::Out(1),
            cols: I32 <- Source::InWidth(0),
        ]),
    kernel!(cast_bf16_to_fp8_e4m3_per_channel "quant::cast_bf16_to_fp8_e4m3_per_channel",
        file = Some("quant/quant_bf16_to_fp8.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            w: Bf16s <- Source::In(0),
            out: U8sMut <- Source::Out(0),
            scale_inv: F32s <- Source::In(1),
            cols: I32 <- Source::InWidth(0),
        ]),
    kernel!(cast_bf16_to_int8_per_channel "quant::cast_bf16_to_int8_per_channel",
        file = Some("quant/quant_bf16_to_fp8.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            w: Bf16s <- Source::In(0),
            out: I8sMut <- Source::Out(0),
            scale_inv: F32s <- Source::In(1),
            cols: I32 <- Source::InWidth(0),
        ]),
    // ── the two `Unstated` rows, and why they are rows at all ─────────────
    //
    // `table/driver_internal.rs` carried `quant::dequant_int32_w8a8_to_bf16`
    // and `quant::quantize_bf16_to_fp8_e4m3_per_token_group` as AOT rows and
    // wrote down what it would take to retire them:
    //
    //   *"Give those three grids rules and all three rows leave this table
    //   for `families::quant`, `bind::quant_gemm` fires them through
    //   `bind::jit::fire` instead of through `ffi::pie_k_*`, and
    //   `quant/quant_bf16_to_fp8.cu` loses its last consumer."*
    //
    // Two of the three moved, and NOT by gaining rules. `new-horizon.md`
    // §10.5 refuses vocabulary grown for one kernel, and each of these grids
    // is one kernel:
    //
    //   * `w8a8_dequant` launches a 2-D BLOCK, `(32, 8)`. No rule in
    //     `kernels::LaunchRule` states a two-dimensional block, and the only
    //     kernel in either archive that wants one is this.
    //   * `quant_act_fp8_per_group` launches `grid(ceil(k / group_size), m)`
    //     -- a 2-D grid whose x axis is a COUNT OF GROUPS, which is `k`
    //     divided by an operand. No `Term` divides by an operand, and adding
    //     one would state this launcher and nothing else.
    //
    // So the escape hatch is the one `fire/attn_score.rs` uses and argues:
    // the DRIVER states the rectangle, citing the `<<<>>>` it came from, and
    // the row states only the contract. A row with `LaunchRule::Unstated`
    // makes `runtime::launch` answer `Ungeometric::Unstated` if anything ever
    // tries to derive a grid from it, which is the refusal that keeps an
    // unstated geometry from being silently invented downstream.
    //
    // The third of the three, `quant::quantize_bf16_to_int8_per_token`, needed
    // no row at all: it was a C++ forwarder onto
    // `quantize_bf16_to_int8_per_channel`, which is `[7]` above and has had a
    // `LaunchRule::Rms` row since this family landed.
    //
    // Both are `DeviceKernel::PLAIN`: neither `__global__` has a template
    // parameter list, so its name IS its qualified path.
    kernel!(dequant_int32_w8a8_to_bf16 "quant::dequant_int32_w8a8_to_bf16",
        file = Some("quant/quant_bf16_to_fp8.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            acc: I32s,
            act_scale_inv: F32s,
            w_scale_inv: F32s,
            out: BufMut,
            m: I32,
            n: I32,
        ]),
    // `n_groups` is `ceil(k / gs)` and crosses as an OPERAND as well as
    // sizing `grid.x`: the kernel bounds `blockIdx.x` against it at
    // `quant_bf16_to_fp8.cuh:340`. The launcher computed it once and used it
    // twice, and so does the port -- two derivations of one quotient is how
    // a grid and a guard come to disagree.
    kernel!(quantize_bf16_to_fp8_e4m3_per_token_group
        "quant::quantize_bf16_to_fp8_e4m3_per_token_group",
        file = Some("quant/quant_bf16_to_fp8.cuh"),
        launch = LaunchRule::Unstated,
        operands = operands![
            act: Bf16s,
            out: U8sMut,
            scale_out: F32sMut,
            m: I32,
            k: I32,
            gs: I32,
            n_groups: I32,
        ]),
];

// ---------------------------------------------------------------------------
// quant/mxfp4_marlin
// ---------------------------------------------------------------------------

/// The instantiations `quant/mxfp4_marlin.cuh` is compiled for.
///
/// `mxfp4_scales_to_marlin_e8m0` is rowed at `u8` — an E8M0 scale is a byte
/// and the kernel only moves it — and `row_map_to_dense` at bf16 and fp16,
/// because it moves whole elements and both widths are live.
pub static MXFP4_MARLIN_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &MXFP4_MARLIN_SIGS[0],
        template_path: "quant::device::mxfp4_scales_to_marlin_e8m0",
        elem: "device::u8",
    },
    DeviceKernel {
        sig: &MXFP4_MARLIN_SIGS[1],
        template_path: "quant::device::row_map_to_dense",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &MXFP4_MARLIN_SIGS[2],
        template_path: "quant::device::row_map_to_dense",
        elem: "device::f16",
    },
];

/// The contracts, in [`MXFP4_MARLIN_ROWS`]' order.
///
/// Both shapes are flat over `selected_rows * (target_groups | batch)`, which
/// is exactly the output's element count, so `Elementwise` sizes the grid the
/// launcher sized it. Everything else describes a SLICE of a source tensor
/// the kernel never has the shape of — the offsets, the stride, the
/// valid-row count, the selector — and none of it is recoverable geometry.
///
/// The first is an ahead-of-time symbol and was `..._e8m0_u8` here, the `u8`
/// being the row's own ELEMENT type appended to a contract that already
/// named the format. The two `row_map_to_dense` rows take their names from
/// the `bf16_row_map_to_dense` launcher beside them.
/// # THE LAUNCHER'S FOUR REFUSALS, WHICH THIS ROW DOES NOT MAKE
///
/// `quant/mxfp4_marlin.cu` is deleted (§54) and its host half was not one
/// `<<<>>>`: it `throw`ed `std::runtime_error` on four conditions before
/// launching, and a rule-driven fire makes none of them. They are recorded
/// here because a measurement a port drops is a measurement nobody can find
/// again, and in `new-horizon.md` §54 with the same words:
///
///   1. `validate_row_select` — `row_select` outside `{Identity, Even, Odd}`
///      threw. Here it is an `I32` from `Source::Param(6)` and the kernel's
///      `select_row` `switch` has a `default` that falls through to the
///      identity, so a bad value SILENTLY reads the wrong half of an
///      interleaved gate/up bank. The three legal values are
///      `quant/mxfp4_marlin.cuh:70-72`'s `kRowSelect*` and
///      `driver-cuda/tests/launch_abi.rs` pins them against the Rust mirror.
///   2. `source_row_offset + selected_rows * stride > source_rows` for the
///      chosen parity — a slice that runs off the end of the source bank.
///   3. `source_group_offset + target_groups > source_groups` — the same
///      check on the group axis, which is where a tensor-parallel shard of
///      the scale table is taken.
///   4. `total % 64 != 0` — Marlin's E8M0 scale tile is 64 bytes wide and
///      the kernel writes whole tiles, so a total that is not a multiple of
///      64 leaves a partial tile of uninitialised scales that the GEMM then
///      reads as exponents.
///
/// All four are HOST checks on operands, which is exactly the shape a
/// `LaunchRule` cannot carry: a rule states a rectangle, not a predicate.
/// Whoever wants them back writes them where the fire is composed — the
/// caller that knows the shard — and not in a new rule.
#[rustfmt::skip]
static MXFP4_MARLIN_SIGS: [KernelSig; 3] = [
    // The table's `quant::mxfp4_scales_to_marlin_e8m0`. Operands diffed
    // against it: eleven, in the table's order, minus the stream. Two kinds
    // are sharper — `U8s`/`U8sMut` where the table erases to `Buf`/`BufMut` —
    // and at `elem = device::u8` both spell `const std::uint8_t*`, which is
    // what the `__global__` takes; the launcher writes the same `static_cast`
    // on its first line. `row_select` is `Mxfp4RowSelect` in the launcher and
    // an `int` here: the enum is declared in `mxfp4_marlin.hpp`, which NVRTC
    // never sees, and `runtime::args` marshals no enum kind — so the device
    // side takes the underlying type and `enum class ... : int` makes the
    // cast exact.
    kernel!(mxfp4_scales_to_marlin_e8m0 "quant::mxfp4_scales_to_marlin_e8m0",
        file = Some("quant/mxfp4_marlin.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            raw_e8m0: U8s <- Source::In(0),
            marlin_e8m0: U8sMut <- Source::Out(0),
            source_rows: I32 <- Source::Param(0),
            source_row_offset: I32 <- Source::Param(1),
            selected_rows: I32 <- Source::OutRows(0),
            valid_rows: I32 <- Source::Param(2),
            source_stride_groups: I32 <- Source::Param(3),
            source_group_offset: I32 <- Source::Param(4),
            source_groups: I32 <- Source::Param(5),
            target_groups: I32 <- Source::OutWidth(0),
            row_select: I32 <- Source::Param(6),
        ]),
    kernel!(bf16_row_map_to_dense "quant::bf16_row_map_to_dense",
        file = Some("quant/mxfp4_marlin.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            raw: Buf <- Source::In(0),
            out: BufMut <- Source::Out(0),
            batch: I32 <- Source::OutWidth(0),
            source_rows: I32 <- Source::Param(0),
            source_row_offset: I32 <- Source::Param(1),
            selected_rows: I32 <- Source::OutRows(0),
            valid_rows: I32 <- Source::Param(2),
            row_select: I32 <- Source::Param(3),
        ]),
    kernel!(f16_row_map_to_dense "quant::f16_row_map_to_dense",
        file = Some("quant/mxfp4_marlin.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            raw: Buf <- Source::In(0),
            out: BufMut <- Source::Out(0),
            batch: I32 <- Source::OutWidth(0),
            source_rows: I32 <- Source::Param(0),
            source_row_offset: I32 <- Source::Param(1),
            selected_rows: I32 <- Source::OutRows(0),
            valid_rows: I32 <- Source::Param(2),
            row_select: I32 <- Source::Param(3),
        ]),
];

// ---------------------------------------------------------------------------
// quant/dequant_fp4
// ---------------------------------------------------------------------------

/// The instantiations `quant/dequant_fp4.cuh` is compiled for.
///
/// One template of the four `__global__`s the header holds — plus, now, two
/// of the three MoE decode GEMVs.
///
/// The reason the GEMVs had none was the GEOMETRY: `dim3 grid(num_tokens *
/// top_k, ceil(width / (warps * rows_per_warp)))` needs a fire-wide expert
/// fanout to open at all, and `Dims::experts_per_token` was zero.
/// [`LaunchRule::RoutedQmvQuad`] is that arithmetic and
/// `DispatchCtx::experts_per_token` is that count.
///
/// The THIRD, `mxfp4_moe_gate_up_decode_grouped`, stays without a row and for
/// two independent reasons, either sufficient — see
/// [`DEQUANT_FP4_SIGS`]' own doc.
pub static DEQUANT_FP4_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &DEQUANT_FP4_SIGS[0],
        template_path: "quant::device::dequant_mxfp4",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DEQUANT_FP4_SIGS[1],
        template_path: "quant::device::dequant_mxfp4",
        elem: "device::f16",
    },
    // `4` is `dequant_fp4.cu:42`'s `kMxfp4GateUpPairs` and `:44`'s
    // `kMxfp4DownRows`, and the two numbers are the SAME NUMBER by
    // coincidence rather than by contract — one counts gate/up PAIRS (so the
    // warp owns `2 * kPairs` packed rows) and the other counts output ROWS.
    // They are spelled separately here because they are spelled separately
    // there, and because a sweep that retuned one would retune it alone; the
    // shared 16-row block tile that `RoutedQmvQuad` computes is what happens
    // to fall out of both at 4.
    DeviceKernel {
        sig: &DEQUANT_FP4_SIGS[2],
        template_path: "quant::device::mxfp4_moe_gate_up_decode",
        elem: "device::i32(4)",
    },
    DeviceKernel {
        sig: &DEQUANT_FP4_SIGS[3],
        template_path: "quant::device::mxfp4_moe_down_decode",
        elem: "device::i32(4)",
    },
];

/// The contracts, in [`DEQUANT_FP4_ROWS`]' order.
///
/// `RouteRows` where the launcher passed a fixed 128, and the kernel strides
/// its block loop by `blockDim.x`, so a wider block is fewer iterations and
/// never a different answer. `in_dim` is the OUTPUT width: the packed input
/// is half as wide in bytes and the scale tensor a thirty-second, so neither
/// input's extent is the one the kernel means.
///
/// # The two MoE decode GEMVs, and the third that is still refused
///
/// `mxfp4_moe_gate_up_decode<4>` and `mxfp4_moe_down_decode<4>` are
/// [`LaunchRule::RoutedQmvQuad`], which is `dim3(rows * k, ceil(intermediate
/// / 16))` at 128 threads and is a different rule from
/// [`LaunchRule::RoutedQmv`] in both numbers — half the block, twice the
/// tile. The near miss is measured in that variant's doc.
///
/// `mxfp4_moe_gate_up_decode_grouped<kTok>` stays without a row, and either
/// reason alone would be sufficient:
///
///  1. **Its `grid.x` is an EXPERT count.** The deleted `dequant_fp4.cu:102-103`
///     opened
///     `dim3 grid(num_experts, ceil(intermediate / pairs_per_block))` where
///     both siblings opened `num_tokens * top_k`, and the kernel still reads
///     it that way at `quant/dequant_fp4.cuh:471`. `Dims::n_experts` is the
///     field that would serve it and `driver-cuda`'s `jit_dims` fills it with
///     zero — a fire-wide expert COUNT is not on `model::deployment::Geometry`
///     and is not on the wire the way the FANOUT is.
///  2. **Its template argument came from the ENVIRONMENT.**
///     `dequant_fp4.cu:108-111` read `std::getenv("PIE_MXFP4_MOE_KTOK")` and
///     `:122-135` switched four cases with a default of 4 — that host text is
///     gone with the launcher, and this reason is why no row inherited it. An
///     environment
///     variable is not an extent of the rectangle: a row that named
///     `mxfp4_moe_gate_up_decode_grouped<4>` would be right on the machines
///     that do not set it and would silently name a different cubin entry
///     from the one the shim launches on the machines that do, which is a
///     wrong answer selected by a shell.
///
/// Its own tile is `warps * kMxfp4GroupedPairs`, and at the grouped path's
/// `kMxfp4DecodeBlock` that is **8** against these two's 16 — so even if the
/// expert count arrived, [`LaunchRule::RoutedQmvQuad`] would state exactly
/// twice its `grid.y` and every output row would be claimed by two blocks.
#[rustfmt::skip]
static DEQUANT_FP4_SIGS: [KernelSig; 4] = [
    // The table's `quant::dequant_mxfp4_to_bf16`, which was
    // `quant::dequant_mxfp4_bf16` here. Operands diffed against it: `packed,
    // block_scale, out, out_dim, in_dim` minus `out_dim` — `RouteRows`' grid,
    // and `dim3 grid(out_dim)` is what the launcher spends it on — and minus
    // the stream.
    kernel!(dequant_mxfp4_to_bf16 "quant::dequant_mxfp4_to_bf16",
        file = Some("quant/dequant_fp4.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            packed: U8s <- Source::In(0),
            block_scale: U8s <- Source::In(1),
            out: BufMut <- Source::Out(0),
            in_dim: I32 <- Source::OutWidth(0),
        ]),
    kernel!(dequant_mxfp4_to_f16 "quant::dequant_mxfp4_to_f16",
        file = Some("quant/dequant_fp4.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            packed: U8s <- Source::In(0),
            block_scale: U8s <- Source::In(1),
            out: BufMut <- Source::Out(0),
            in_dim: I32 <- Source::OutWidth(0),
        ]),
    // The deleted `quant/dequant_fp4.cu:67-77` --
    //
    //     dim3 grid(num_tokens * top_k,
    //               (intermediate + pairs_per_block - 1) / pairs_per_block);
    //     device::mxfp4_moe_gate_up_decode<kMxfp4GateUpPairs>
    //         <<<grid, kMxfp4DecodeBlock, 0, stream>>>(
    //         static_cast<const __half*>(act_fp16), topk_idx,
    //         gate_up_packed, gate_up_scales, gate_bias, up_bias,
    //         static_cast<device::bf16*>(gate_out_bf16),
    //         static_cast<device::bf16*>(up_out_bf16),
    //         static_cast<__half*>(act_out_fp16), glu_limit, glu_alpha,
    //         top_k, hidden, intermediate);
    //
    // `Source`s diffed against `table::moe`'s `mxfp4_moe_gate_up` row: minus
    // `num_tokens`, which is `grid.x`'s first factor, and minus the stream,
    // which is `cuLaunchKernel`'s sixth parameter and not an argument. The
    // ORDER differs from the shim's — the `__global__` takes `act_out_fp16`,
    // `glu_limit` and `glu_alpha` BEFORE the three extents, where the shim
    // takes them after the stream — which is exactly the kind of divergence a
    // row exists to state rather than to inherit.
    //
    // `top_k` STAYS an operand though the rule also reads the fanout: the
    // kernel divides by it to recover `token = route / top_k` and cannot read
    // a grid. `hidden` and `intermediate` are `Div`s of two widths for
    // `table::moe`'s reason — the statement's outputs carry the ROUTED extent
    // `[Tokens, k, intermediate]`, so a bare `OutWidth(0)` would be `k`
    // times too wide, which is the same stacking `LaunchRule::RoutedQmvQuad`
    // divides out on the geometry side.
    kernel!(mxfp4_moe_gate_up_decode "quant::mxfp4_moe_gate_up_decode_bf16",
        file = Some("quant/dequant_fp4.cuh"),
        launch = LaunchRule::RoutedQmvQuad,
        operands = operands![
            act: F16s <- Source::In(1),
            topk_idx: I32s <- Source::In(0),
            packed_ptrs: U8Array <- Source::Weight(0),
            scale_ptrs: U8Array <- Source::WeightSuffix("_scales"),
            gate_bias_ptrs: BufArray <- Source::WeightSuffix("_gate_bias"),
            up_bias_ptrs: BufArray <- Source::WeightSuffix("_up_bias"),
            gate_out: BufMut <- Source::Out(0),
            up_out: BufMut <- Source::Out(1),
            act_out_fp16: BufMut <- Source::Lit(Lit::Null),
            glu_limit: F32 <- Source::Ctx("glu_limit"),
            glu_alpha: F32 <- Source::Ctx("glu_alpha"),
            top_k: I32 <- Source::InWidth(0),
            hidden: I32 <- Source::InWidth(1),
            intermediate: I32 <- Source::Div(&Source::Width(&Source::Out(0)), &Source::Width(&Source::In(0))),
        ]),
    // The deleted `quant/dequant_fp4.cu:152-162` -- the down leg, five lines
    // of grid arithmetic that differed from the gate/up's only in which
    // extent is slabbed:
    //
    //     dim3 grid(num_tokens * top_k,
    //               (hidden + rows_per_block - 1) / rows_per_block);
    //     device::mxfp4_moe_down_decode<kMxfp4DownRows>
    //         <<<grid, kMxfp4DecodeBlock, 0, stream>>>(
    //         static_cast<const __half*>(act_fp16), topk_idx,
    //         down_packed, down_scales, down_bias,
    //         static_cast<device::bf16*>(out_bf16),
    //         hidden, intermediate);
    //
    // **The same rule and NOT the transposed one**, which is the pair
    // `dequant_wna16.cu` has and this file does not: `dequant_fp4.cuh:357`
    // takes `route = blockIdx.x` in BOTH kernels, where `dequant_wna16.cuh`
    // swaps them between its two. So `RoutedQmvQuad` serves both legs here
    // and `RoutedQmv`/`RoutedQmvTransposed` are two rules there — a
    // difference between two C++ files, stated rather than smoothed.
    //
    // The kernel takes `hidden` and `intermediate` and NOT `top_k`: it reads
    // its expert straight out of `topk_idx[route]` and never needs the token,
    // where the gate/up leg's fused activation epilogue does.
    kernel!(mxfp4_moe_down_decode "quant::mxfp4_moe_down_decode_bf16",
        file = Some("quant/dequant_fp4.cuh"),
        launch = LaunchRule::RoutedQmvQuad,
        operands = operands![
            act: F16s <- Source::In(1),
            topk_idx: I32s <- Source::In(0),
            packed_ptrs: U8Array <- Source::Weight(0),
            scale_ptrs: U8Array <- Source::WeightSuffix("_scales"),
            bias_ptrs: BufArray <- Source::WeightSuffix("_bias"),
            out: BufMut <- Source::Out(0),
            hidden: I32 <- Source::Div(&Source::Width(&Source::Out(0)), &Source::Width(&Source::In(0))),
            intermediate: I32 <- Source::Div(&Source::Width(&Source::In(1)), &Source::Width(&Source::In(0))),
        ]),
];

// ---------------------------------------------------------------------------
// quant/dequant_wna16
// ---------------------------------------------------------------------------

/// The two instantiations `quant/dequant_wna16.cuh` is compiled for.
///
/// Two templates of the four `__global__`s the header holds. The other two
/// are the W4A16 decode GEMVs, which put one WARP on an output row and slabs
/// of rows on `gridDim.y`; that reason is recorded at the top of this file,
/// and so is the one that used to keep `bf16_to_narrow` out — a capped
/// grid-stride no rule stated, which
/// [`kernels::LaunchRule::Slab`] was ported from this very launcher to state.
pub static DEQUANT_WNA16_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &DEQUANT_WNA16_SIGS[0],
        template_path: "quant::device::dequant_wna16_int4b8",
        elem: "device::bf16",
    },
    // `device::f16` and not `__half`, and they are the SAME TYPE here.
    // `csrc/shim/cuda_fp16.h` opens the NVRTC shim with `using __half =
    // ::pie_cuda_driver::kernels::device::f16;`, so the launcher's
    // `bf16_to_narrow<__half>` and this row's
    // `bf16_to_narrow<::pie_cuda_driver::kernels::device::f16>` name one
    // instantiation and resolve to one `Narrow2<__half>` specialisation --
    // the `__half2` one at `dequant_wna16.cuh:451`, which is the whole of
    // what makes the cast fp16 rather than something else. Under nvcc the
    // two are distinct and the header carries a `Narrow2<f16_or_inert>`
    // behind `half_key::PickF16` so both compilers instantiate the same text
    // at the same names; that is a parity guard and this row does not depend
    // on it.
    DeviceKernel {
        sig: &DEQUANT_WNA16_SIGS[1],
        template_path: "quant::device::bf16_to_narrow",
        elem: "device::f16",
    },
    // `device::i32(0)` is `Tu`, and it is NOT an element type — the header
    // says so at `dequant_wna16.cuh:265-279` in its own words. `Tu` is a
    // LINKAGE parameter: nvcc 13.0 gives a non-template `__global__` in a
    // header external linkage for the function AND its `__device_stub__`, so
    // a second includer is a hard "multiple definition" at link even when it
    // launches nothing — measured, four collisions across two TUs that only
    // `#include`. A defaulted non-type parameter drops each instantiation to
    // internal linkage (`nm` says `t`) and every un-edited `<<<>>>` selects
    // `Tu = 0`.
    //
    // So these rows must state `0` and not the default's absence, for
    // `KIMI_MLA_ROWS[1]`'s reason in a sharper form: `DeviceKernel::PLAIN`
    // would be a LIE here — it says "this `__global__` has no template
    // parameter list", and this one does — and an empty `elem` is what an
    // unfilled field looks like. The mangled name NVRTC answers with carries
    // the argument either way; the row states which one it asked for.
    //
    // `device::i32(0)` rather than a bare `0` because
    // `DeviceKernel::instantiation` prefixes the FIRST token with
    // `::pie_cuda_driver::kernels::`, and `::pie_cuda_driver::kernels::0` is
    // `expected an identifier` under NVRTC 13.0 — the same measurement
    // `KV_PAGED_ROWS` records for `true`. `families::norm`'s
    // `rmsnorm_strided_vec8` row spells the identical workaround.
    DeviceKernel {
        sig: &DEQUANT_WNA16_SIGS[2],
        template_path: "quant::device::wna16_gate_up_decode",
        elem: "device::i32(0)",
    },
    DeviceKernel {
        sig: &DEQUANT_WNA16_SIGS[3],
        template_path: "quant::device::wna16_down_decode",
        elem: "device::i32(0)",
    },
];

/// The contracts, in [`DEQUANT_WNA16_ROWS`]' order.
///
/// The ahead-of-time table's `quant::dequant_wna16_int4b8_to_bf16` — the
/// symbol a declaration states when a checkpoint ships INT4B8 weights with a
/// bf16 scale per group along K, which is a different quantization from MXFP4
/// and from fp8 and is stated as one.
///
/// `ElementwiseRows` against the launcher it mirrors:
///
/// ```text
/// dim3 grid(out_dim, (words_per_row + 255) / 256);   // words_per_row = in_dim / 8
/// device::dequant_wna16_int4b8<device::bf16><<<grid, 256, 0, stream>>>(...)
/// ```
///
/// and `elementwise_rows(rows, width)` is `grid [rows, ceil(width / 256), 1]`,
/// `block [256, 1, 1]` — same axis order, same block, and a y extent of
/// `ceil(in_dim / 256)` where the launcher computes `ceil(in_dim / 8 / 256)`.
/// The rule is therefore eight times oversubscribed on that axis and the
/// kernel's `if (word_col >= words_per_row) return;` discards the excess:
/// coverage is what a rule has to get right and coverage holds. The waste is
/// one predicated exit on a kernel that runs once per weight at load, and the
/// header says so where the guard is.
///
/// The row exists at all because the axes were SWAPPED in `csrc`: the old
/// order put word-columns on x and rode `for (row = blockIdx.y; row < out_dim;
/// row += gridDim.y)`, because `gridDim.y` caps at 65535 and an
/// expert-stacked weight has more rows than that. No rule states a hardware
/// clamp, so there was no row. `gridDim.x` is 2^31-1, so a row per block is
/// exact and nothing iterates.
///
/// Operands diffed against the table: `packed, scale_bf16, out_bf16, out_dim,
/// in_dim, group_size` minus `out_dim` — the rule's `grid.x`, which the
/// `__global__` never took — and minus the stream. `packed` is `I32s` in both
/// tables and in the kernel, because an INT4B8 word is eight nibbles in a
/// 32-bit int and reading it as anything else is a stride bug a `Buf` would
/// have allowed.
#[rustfmt::skip]
static DEQUANT_WNA16_SIGS: [KernelSig; 4] = [
    kernel!(dequant_wna16_int4b8_to_bf16 "quant::dequant_wna16_int4b8_to_bf16",
        file = Some("quant/dequant_wna16.cuh"),
        launch = LaunchRule::ElementwiseRows,
        operands = operands![
            packed: I32s <- Source::In(0),
            scale_bf16: Buf <- Source::In(1),
            out_bf16: BufMut <- Source::Out(0),
            in_dim: I32 <- Source::OutWidth(0),
            group_size: I32 <- Source::Param(0),
        ]),
    // THE CAPPED GRID-STRIDE, and the row `Slab` was written for.
    //
    // `quant/dequant_wna16.cu:63-75`:
    //
    // ```text
    // constexpr int BS = 256;
    // const long long n = static_cast<long long>(count);
    // const long long n_vec8 = n / 8;
    // const long long units = n_vec8 > 0 ? n_vec8 : n;
    // const int blocks = static_cast<int>(
    //     std::min<long long>((units + BS - 1) / BS, 1024));
    // device::bf16_to_narrow<__half><<<std::max(blocks, 1), BS, 0, stream>>>(
    //     ..., n);
    // ```
    //
    // `runtime::launch::slab(n)` is `units = n >= 8 ? n / 8 : n` and
    // `grid [clamp(ceil(units / 256), 1, 1024), 1, 1]`, `block [256, 1, 1]`,
    // `smem 0` — the launcher's five lines with the `max(blocks, 1)` folded
    // into the clamp's lower bound and the `n < 8` arm folded into the
    // conditional the launcher spells as `n_vec8 > 0 ? n_vec8 : n`. `eval`
    // supplies `n` as `dims.rows * dims.width`, which is the rectangle this
    // statement's result covers and is the same number `OutElements(0)`
    // hands the kernel below.
    //
    // **The grid does not cover the extent, and that is the contract.**
    // `bf16_to_narrow` walks `for (i = tid; i < n_vec8; i += gridDim.x *
    // blockDim.x)` with a scalar tail after it, so 1024 blocks cast a
    // hundred-million-element tensor in as many passes as they need. A rule
    // that dropped the cap would launch 1025 blocks where the launcher
    // launches 1024 and would be a DIFFERENT kernel with the same body:
    // occupancy is what this launcher chose and the cap is where it wrote the
    // choice down.
    //
    // **`n` stays an operand and it is `I64`.** The rule recovers the same
    // number for the grid; the kernel still needs the bound, because with a
    // capped grid no block can infer the extent from `gridDim`. It is `I64`
    // and not `Usize` because the `__global__` declares `long long n` --
    // the LAUNCHER takes `std::size_t count` and narrows on the line above
    // the `<<<>>>`, and a row states the kernel's parameter list rather than
    // its caller's. `abi::elem_count` spells `Source::OutElements` at that
    // width with `i64::try_from(..).unwrap_or(0)`, so the binder produces an
    // `ArgValue::I64` and `Ty::I64` accepts it.
    //
    // `in_bf16` IS THE OPERAND THAT NAMED `Ty::Bf16s`. It was `Buf`, exactly
    // as the ahead-of-time twin spells it, and `abi::device_cpp_ty` reads
    // `Buf` as `const {elem}*` -- `const f16*` for a parameter the kernel
    // declares `const bf16*`. Every buffer marshals as a pointer, so nothing
    // miscomputed at fire time; what was lost was the offline check, on the
    // one row where losing it means the two sixteen-bit formats are
    // interchangeable in the only place that distinguishes them.
    //
    // `Bf16s` states the format rather than deriving it from `elem`, which is
    // the wrong end of this cast: `bf16_to_narrow<T>` FIXES the source at
    // bf16 and templates the destination. The checker is now
    // `(const bf16*, f16*, long long)` against
    // `bf16_to_narrow<device::f16>`, and the `Buf` spelling --
    // `(const f16*, f16*, long long)` -- is measured as rejected by nvcc 13.0
    // `-arch=sm_89`: *"no instance of function template … matches the
    // required type"*. `tests/device_typecheck_types.rs` compiles both.
    kernel!(bf16_to_fp16 "quant::bf16_to_fp16",
        file = Some("quant/dequant_wna16.cuh"),
        launch = LaunchRule::Slab,
        operands = operands![
            in_bf16: Bf16s <- Source::In(0),
            out_fp16: BufMut <- Source::Out(0),
            n: I64 <- Source::OutElements(0),
        ]),
    // `quant/dequant_wna16.cu:73-75`, before §43.9 deleted this launcher as
    // unreached — `quant/dequant_wna16.cuh:295`/`:298` is the live witness --
    //
    //     constexpr int GU_WARPS = DECODE_BLOCK / 32;                    // 8
    //     const dim3 grid(routes, (intermediate + GU_WARPS - 1) / GU_WARPS);
    //     device::wna16_gate_up_decode<<<grid, DECODE_BLOCK, 0, stream>>>(...);
    //
    // with `:70` supplying `routes = num_tokens * top_k`, which is exactly
    // `LaunchRule::RoutedQmv`'s `Dims::rows * Dims::experts_per_token`.
    //
    // `Source`s copied from `table::moe`'s row minus its `num_tokens` and its
    // `stream`: `num_tokens` is `grid.x`'s first factor and the stream is not
    // a kernel parameter. `top_k` STAYS, because the kernel divides by it to
    // recover `token = route / top_k` and cannot read it off a grid.
    //
    // # This row now fires, and what closed it
    //
    // `RoutedQmv` reads `Dims::experts_per_token`, and `driver-cuda`'s
    // `jit_dims` used to fill it with 0 — *absent, not zero-as-a-value* — so
    // `eval` answered `Ungeometric::Empty` at every generated call site. That
    // was the honest answer rather than a defect, and the fix named here has
    // landed: `DispatchCtx::experts_per_token` now carries a FIRE-WIDE count,
    // derived once in `fire::launch::fire_experts_per_token` from the lowered
    // plan's own routed launches.
    //
    // It is keyed on the KERNEL SYMBOL and not on a param index, which is the
    // part that matters: the wire's `params[1]` is `window_left` on an
    // attention dispatch and `w.width` on an unrouted `qmv`, so an
    // index-keyed reading would have been spellable over the wrong statement.
    // The derivation cannot spell that, because it has no index in its
    // interface — a symbol's layout comes from the one `dsl` constructor that
    // emits it. A fire whose routed statements disagree still answers 0, and
    // 0 still refuses: a guess would open `rows * 1` routes for a top-4 fire
    // and dequantise a quarter of the banks.
    kernel!(wna16_gate_up_decode "quant::wna16_gate_up_decode_bf16",
        file = Some("quant/dequant_wna16.cuh"),
        launch = LaunchRule::RoutedQmv,
        operands = operands![
            act_fp16: F16s <- Source::In(0),
            topk_idx: I32s <- Source::In(1),
            gate_packed: I32Array <- Source::Weight(0),
            gate_scale: BufArray <- Source::Weight(1),
            up_packed: I32Array <- Source::Weight(2),
            up_scale: BufArray <- Source::Weight(3),
            gate_out_bf16: BufMut <- Source::Out(0),
            up_out_bf16: BufMut <- Source::Out(1),
            top_k: I32 <- Source::InWidth(1),
            hidden: I32 <- Source::InWidth(0),
            intermediate: I32 <- Source::OutWidth(0),
            group_size: I32 <- Source::Ctx("wna16_group_size"),
        ]),
    // The deleted `quant/dequant_wna16.cu:101-104` -- THE TRANSPOSE, and the
    // reason `RoutedQmvTransposed` exists rather than a second reading of the
    // rule above. `quant/dequant_wna16.cuh:371`/`:374` still reads the two
    // axes back the swapped way round.
    //
    //     constexpr int BS = 256;
    //     constexpr int WARPS = BS / 32;                                  // 8
    //     const dim3 grid((hidden + WARPS - 1) / WARPS, routes);
    //     device::wna16_down_decode<<<grid, BS, 0, stream>>>(...);
    //
    // Same divisor, same block, same `routes = num_tokens * top_k` at `:98` --
    // and the axes swapped. `wna16_down_decode` reads `blockIdx.y` for its
    // route and `blockIdx.x` for its output column, which is the mirror of
    // the gate/up kernel, so the two rules are not one rule with an argument.
    // Firing this row under `RoutedQmv` would launch `routes` columns and
    // `ceil(hidden/8)` routes: at Kimi K2.6's decode shapes (routes 8,
    // hidden 7168) that is 8 columns of a 7168-wide row and 896 routes over
    // 8 -- a grid that is neither a subset nor a superset of the right one,
    // which is exactly the class of wrong the header's *"byte-identical
    // inside the rectangle"* measurement is about.
    //
    // The extents each rule reads are the OUTPUT width in both cases:
    // `intermediate` is `OutWidth(0)` above and `hidden` is `OutWidth(0)`
    // here. `Dims::width` serves both without either meaning something else.
    kernel!(wna16_down_decode "quant::wna16_down_decode_bf16",
        file = Some("quant/dequant_wna16.cuh"),
        launch = LaunchRule::RoutedQmvTransposed,
        operands = operands![
            act_fp16: F16s <- Source::In(0),
            topk_idx: I32s <- Source::In(1),
            down_packed: I32Array <- Source::Weight(0),
            down_scale: BufArray <- Source::Weight(1),
            out_bf16: BufMut <- Source::Out(0),
            top_k: I32 <- Source::InWidth(1),
            hidden: I32 <- Source::OutWidth(0),
            intermediate: I32 <- Source::InWidth(0),
            group_size: I32 <- Source::Ctx("wna16_group_size"),
        ]),
];
